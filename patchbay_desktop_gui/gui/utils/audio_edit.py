"""Audio editing utilities for output post-processing.

This module intentionally contains *pure* numpy-based functions so it can be
used both from the DearPyGui frontend and (later) by other frontends.

All functions are designed for PCM audio in floating point representation
(float32 preferred). The typical value range is ``[-1.0, 1.0]``.

The GUI implements non-destructive editing by keeping snapshots for undo/redo.
This module therefore only provides stateless transformations.

Implemented effects
-------------------

Normalize
    Peak normalize the signal to a target peak (default: 0.999).

Compressor
    An envelope-based (peak detector) hard-knee compressor with configurable
    threshold ("knee point") in dBFS and ratio. In contrast to a sample-by-sample
    waveshaper, this compressor uses attack/release smoothing to avoid distortion.

Low-pass filter
    Linear-phase FIR low-pass filter using a windowed-sinc kernel and fast FFT
    convolution (overlap-add). This is used by the built-in "Low-pass filter"
    audio FX plugin.

Notes
-----
- The compressor is intentionally dependency-free and deterministic.
- Threshold is interpreted as dBFS relative to full-scale 1.0.
- The gain computer follows the standard hard-knee curve in dB:
    * below threshold: output = input
    * above threshold: output = threshold + (input-threshold)/ratio
- **No post-normalization** is applied after compression. If you want extra
  headroom, use an explicit normalize step (or an explicit output gain).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def ensure_float32(audio: np.ndarray) -> np.ndarray:
    """Return a contiguous float32 array.

    Parameters
    ----------
    audio:
        Audio array of shape (N,) or (N, C).

    Returns
    -------
    np.ndarray
        Contiguous float32 array with shape (N, C).
    """
    if audio is None:
        raise ValueError("audio is None")
    if audio.ndim == 1:
        audio2 = audio[:, None]
    elif audio.ndim == 2:
        audio2 = audio
    else:
        raise ValueError(f"audio must be 1D or 2D, got ndim={audio.ndim}")

    if audio2.dtype != np.float32:
        audio2 = audio2.astype(np.float32, copy=False)
    return np.ascontiguousarray(audio2)


def peak_normalize(audio: np.ndarray, *, target_peak: float = 0.999) -> np.ndarray:
    """Peak-normalize an audio buffer.

    Parameters
    ----------
    audio:
        Float PCM audio, shape (N,) or (N,C).
    target_peak:
        Target peak magnitude in linear full-scale units.
        Use < 1.0 to avoid potential inter-sample clipping.

    Returns
    -------
    np.ndarray
        Normalized audio (float32, shape (N,C)).
    """
    x = ensure_float32(audio)
    if x.size == 0:
        return x

    peak = float(np.max(np.abs(x)))
    if peak <= 1e-12:
        return x

    tp = float(target_peak)
    if tp <= 0.0:
        tp = 0.999

    gain = tp / peak
    return (x * np.float32(gain)).astype(np.float32, copy=False)


def compressor_hard_knee(
    audio: np.ndarray,
    *,
    threshold_db: float,
    ratio: float,
    eps: float = 1e-8,
) -> np.ndarray:
    """Legacy: sample-by-sample hard-knee static compressor.

    This function is kept for reference, but it behaves like a waveshaper
    (instantaneous gain). For musical compression prefer
    :func:`compressor_peak_ar`.

    Parameters
    ----------
    audio:
        Float PCM audio, shape (N,) or (N,C).
    threshold_db:
        Threshold (knee point) in dBFS. Typical values: -30..0.
    ratio:
        Compression ratio >= 1.0.
        Example: 2.0 means 2:1.
    eps:
        Small number to avoid log(0).

    Returns
    -------
    np.ndarray
        Compressed audio (float32, shape (N,C)).
    """
    x = ensure_float32(audio)
    if x.size == 0:
        return x

    r = float(ratio)
    if r < 1.0:
        r = 1.0

    thr = float(threshold_db)

    # Convert magnitude to dBFS
    mag = np.maximum(np.abs(x), np.float32(eps))
    db = 20.0 * np.log10(mag)

    # For samples above threshold: output_db = thr + (db - thr) / ratio
    # Gain in dB: output_db - db
    over = db - np.float32(thr)
    gain_db = np.where(over > 0.0, (np.float32(thr) + over / np.float32(r)) - db, 0.0)

    gain = np.power(np.float32(10.0), gain_db / np.float32(20.0)).astype(np.float32, copy=False)
    y = (x * gain).astype(np.float32, copy=False)

    # Important: Do NOT normalize or apply any post-gain after compression.
    #
    # The user expects the compressor to strictly follow the static curve:
    # - Below the threshold: gain = 1.0 (signal unchanged)
    # - Above the threshold: attenuation according to the chosen ratio
    #
    # Any additional global scaling ("safety peak" normalization) would alter
    # the resulting loudness and is therefore intentionally omitted.
    #
    # We still guard against NaN/Inf values to keep subsequent processing stable.
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def compressor_peak_ar(
    audio: np.ndarray,
    *,
    sample_rate: int,
    threshold_db: float,
    ratio: float,
    attack_ms: float = 10.0,
    release_ms: float = 100.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Envelope-based (attack/release) hard-knee compressor (peak detector).

    Why this exists
    ---------------
    The earlier implementation :func:`compressor_hard_knee` computes the gain
    *per sample* from the instantaneous sample magnitude. While the static curve
    is correct on paper, applying it sample-by-sample behaves like a non-linear
    waveshaper and introduces harmonic distortion, especially on loud passages.

    A traditional audio compressor therefore uses a *level detector* and applies
    the gain based on a smoothed envelope (attack/release). This keeps the gain
    changes slow compared to the waveform itself and avoids audible distortion.

    Algorithm
    ---------
    1) Detector: peak magnitude across channels (preserves stereo image)
    2) Smooth the detector with attack/release in the *linear* domain
    3) Convert the envelope to dBFS and apply the standard hard-knee curve:
       - below threshold: output = input
       - above threshold: output = thr + (input-thr)/ratio
    4) Convert the resulting gain back to linear and apply it to all channels

    Parameters
    ----------
    audio:
        Float PCM audio, shape (N,) or (N,C).
    sample_rate:
        Sampling rate in Hz.
    threshold_db:
        Threshold in dBFS.
    ratio:
        Compression ratio >= 1.0.
    attack_ms, release_ms:
        Attack/release time constants in milliseconds.
        Typical starting values: attack 5..20 ms, release 50..250 ms.
    eps:
        Small number to avoid log(0).

    Returns
    -------
    np.ndarray
        Compressed audio (float32, shape (N,C)).

    Notes
    -----
    - No post-normalization is applied.
    - This is still a "static" compressor (no lookahead, no soft-knee), but it
      behaves like an actual compressor due to the envelope smoothing.
    """
    x = ensure_float32(audio)
    if x.size == 0:
        return x

    sr = int(sample_rate)
    if sr <= 0:
        raise ValueError(f"sample_rate must be > 0, got {sample_rate}")

    r = float(ratio)
    if r < 1.0:
        r = 1.0

    thr = float(threshold_db)

    # ---- detector (peak across channels) ----
    det = np.max(np.abs(x), axis=1).astype(np.float32, copy=False)  # (N,)

    # ---- envelope smoothing (linear domain) ----
    atk_s = max(float(attack_ms), 0.0) / 1000.0
    rel_s = max(float(release_ms), 0.0) / 1000.0

    # If attack/release are 0, we want immediate tracking.
    if atk_s <= 0.0:
        a_coeff = 0.0
    else:
        a_coeff = float(np.exp(-1.0 / (sr * atk_s)))
    if rel_s <= 0.0:
        r_coeff = 0.0
    else:
        r_coeff = float(np.exp(-1.0 / (sr * rel_s)))

    env = np.empty_like(det)
    env_prev = float(det[0])
    env[0] = env_prev
    for i in range(1, det.shape[0]):
        d = float(det[i])
        if d > env_prev:
            coeff = a_coeff
        else:
            coeff = r_coeff
        env_prev = coeff * env_prev + (1.0 - coeff) * d
        env[i] = env_prev

    # ---- gain computer in dBFS ----
    env = np.maximum(env, np.float32(eps))
    env_db = 20.0 * np.log10(env)
    over = env_db - np.float32(thr)

    # Standard hard-knee curve (in dB)
    out_db = np.where(over > 0.0, np.float32(thr) + over / np.float32(r), env_db)
    gain_db = out_db - env_db
    gain = np.power(np.float32(10.0), gain_db / np.float32(20.0)).astype(np.float32, copy=False)

    y = (x * gain[:, None]).astype(np.float32, copy=False)
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def to_mono_mix(audio: np.ndarray) -> np.ndarray:
    """Utility: mix a (N,C) buffer down to mono (N,).

    This is used by the GUI for waveform display only.
    """
    x = ensure_float32(audio)
    if x.shape[1] == 1:
        return x[:, 0]
    return np.mean(x, axis=1).astype(np.float32, copy=False)


def describe_peak_dbfs(audio: np.ndarray, eps: float = 1e-12) -> float:
    """Return peak level in dBFS."""
    x = ensure_float32(audio)
    if x.size == 0:
        return -120.0
    peak = float(np.max(np.abs(x)))
    peak = max(peak, float(eps))
    return float(20.0 * np.log10(peak))


# ---------------------------------------------------------------------------
# FIR low-pass filter (windowed-sinc + FFT convolution)
# ---------------------------------------------------------------------------

_STEEPNESS_TAPS = {
    # The mapping is a pragmatic trade-off between a visibly steeper transition
    # band and compute cost. All values are odd to keep the kernel symmetric
    # around the center sample.
    6: 101,
    12: 201,
    18: 401,
    24: 801,
    36: 1201,
    48: 1601,
}


def _next_pow2(n: int) -> int:
    """Return the next power-of-two >= n (for FFT sizes)."""
    n = int(max(1, n))
    return 1 << (n - 1).bit_length()


def _design_lowpass_kernel(*, sample_rate: int, cutoff_hz: float, taps: int) -> np.ndarray:
    """Design a symmetric windowed-sinc low-pass kernel.

    The kernel is normalized to sum to 1.0 so that DC gain is unity.

    Returns
    -------
    np.ndarray
        1D float64 array of length `taps`.
    """
    sr = int(sample_rate)
    if sr <= 0:
        raise ValueError(f"sample_rate must be > 0, got {sample_rate}")

    # Clamp cutoff to a safe range below Nyquist.
    nyq = 0.5 * sr
    fc = float(cutoff_hz)
    fc = max(1.0, min(nyq * 0.98, fc))

    M = int(taps)
    if M < 3:
        M = 3
    if M % 2 == 0:
        M += 1

    # Normalized cutoff (0..0.5)
    f = fc / sr

    # Symmetric index around center
    n = np.arange(M, dtype=np.float64) - (M - 1) / 2.0

    # Ideal low-pass (sinc) impulse response.
    # np.sinc(x) = sin(pi*x) / (pi*x)
    h = 2.0 * f * np.sinc(2.0 * f * n)

    # Window to reduce sidelobes.
    w = np.hamming(M).astype(np.float64, copy=False)
    h *= w

    # Normalize to unity DC gain.
    s = float(np.sum(h))
    if abs(s) > 1e-12:
        h /= s

    return h


def _fft_convolve_same_ola(
    x: np.ndarray,
    h: np.ndarray,
    *,
    block_len: int = 262_144,
) -> np.ndarray:
    """FFT-based overlap-add convolution returning 'same' length as x.

    Parameters
    ----------
    x:
        1D float array (N,).
    h:
        1D float array (M,), FIR kernel.
    block_len:
        Processing block length for overlap-add.

    Returns
    -------
    np.ndarray
        1D float32 array of length N.
    """
    if x.ndim != 1:
        raise ValueError("_fft_convolve_same_ola expects 1D input")
    N = int(x.shape[0])
    M = int(h.shape[0])
    if N == 0:
        return x.astype(np.float32, copy=True)
    if M <= 1:
        return x.astype(np.float32, copy=True)

    # Ensure odd M for clean centering.
    if M % 2 == 0:
        raise ValueError("kernel length M must be odd")

    blk = int(max(1024, min(block_len, N)))
    nfft = _next_pow2(blk + M - 1)
    H = np.fft.rfft(h, nfft)

    y = np.zeros(N + M - 1, dtype=np.float32)

    i = 0
    while i < N:
        x_blk = x[i : i + blk].astype(np.float64, copy=False)
        X = np.fft.rfft(x_blk, nfft)
        tmp = np.fft.irfft(X * H, nfft)[: x_blk.size + M - 1]
        y[i : i + tmp.size] += tmp.astype(np.float32, copy=False)
        i += blk

    # "same" slice: remove the group delay of a symmetric kernel.
    start = (M - 1) // 2
    return y[start : start + N]


def lowpass_fir(
    audio: np.ndarray,
    *,
    sample_rate: int,
    cutoff_hz: float,
    steepness_db_oct: int = 24,
) -> np.ndarray:
    """Apply a linear-phase FIR low-pass filter.

    Parameters
    ----------
    audio:
        Audio buffer, shape (N,) or (N,C). Float PCM.
    sample_rate:
        Sample rate in Hz.
    cutoff_hz:
        Cutoff frequency in Hz.
    steepness_db_oct:
        Approximate slope setting in dB/oct.
        Supported values: 6, 12, 18, 24, 36, 48.

    Returns
    -------
    np.ndarray
        Filtered audio (float32, shape (N,C)).

    Notes
    -----
    - The filter is applied in a *non-causal* manner (centered convolution),
      which keeps the output aligned with the input in offline processing.
    - Unity gain in the low-frequency passband is achieved by kernel
      normalization (DC gain = 1.0).
    """
    x = ensure_float32(audio)
    if x.size == 0:
        return x

    slope = int(steepness_db_oct)
    taps = int(_STEEPNESS_TAPS.get(slope, _STEEPNESS_TAPS[24]))

    h = _design_lowpass_kernel(sample_rate=int(sample_rate), cutoff_hz=float(cutoff_hz), taps=taps)

    y = np.empty_like(x, dtype=np.float32)
    for c in range(x.shape[1]):
        y[:, c] = _fft_convolve_same_ola(x[:, c], h)
    return y
