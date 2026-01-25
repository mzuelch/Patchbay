"""Spectral Noise Reduction Audio FX plugin for PATCHBAY.

This plugin performs *spectral noise reduction* (a common "spectral gate" / spectral attenuation approach)
using a noise profile estimated from the audio signal.

Key features
------------
- Pure NumPy implementation (no SciPy).
- STFT-based spectral attenuation with:
  - configurable FFT size and overlap
  - noise profile estimation (initial segment, global median, global percentile)
  - threshold + soft knee (in dB)
  - maximum attenuation (in dB)
  - optional frequency and time smoothing (attack/release)

Notes
-----
This is a pragmatic denoiser aimed at post-processing separated stems.
It will not match state-of-the-art ML denoisers, but is often good enough for
hum/hiss/steady background noise and is fast & offline.

The full algorithm is contained in this plugin file.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from patchbay_desktop_gui.audiofx.base import AudioEffectPluginBase, ParamSpec


# ----------------------------
# Utilities
# ----------------------------

def _ensure_float32(audio: np.ndarray) -> np.ndarray:
    """Return a contiguous float32 array with shape (N, C)."""
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

    if not audio2.flags["C_CONTIGUOUS"]:
        audio2 = np.ascontiguousarray(audio2)
    return audio2


def _hann(n: int) -> np.ndarray:
    # Hann window (periodic=False variant)
    if n <= 1:
        return np.ones((n,), dtype=np.float32)
    t = np.arange(n, dtype=np.float32)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * t / (n - 1))


def _stft(x: np.ndarray, n_fft: int, hop: int, *, center: bool = True) -> tuple[np.ndarray, int, int]:
    """STFT for 1D float32 signal.

    Returns (S, pad_left, pad_right) where:
      - S: complex64 array shape (T, F)
      - pad_left/right: number of samples padded on each side
    """
    if n_fft <= 0 or hop <= 0:
        raise ValueError("n_fft and hop must be positive")
    x = np.asarray(x, dtype=np.float32)
    n = int(x.shape[0])

    pad_left = n_fft // 2 if center else 0
    # pad right so that (n + pad_left + pad_right - n_fft) is divisible by hop
    n_total_no_right = n + pad_left
    if n_total_no_right < n_fft:
        # ensure at least one frame
        n_total_no_right = n_fft
    remainder = (n_total_no_right - n_fft) % hop
    pad_right = (hop - remainder) % hop
    # also ensure we cover original samples fully during inverse OLA
    # (this formula ensures frames span the padded signal)
    n_frames = 1 + (n_total_no_right + pad_right - n_fft) // hop
    needed_total = (n_frames - 1) * hop + n_fft
    pad_right = int(needed_total - (n + pad_left))

    if pad_left > 0 or pad_right > 0:
        mode = "reflect" if n > 1 else "constant"
        x_pad = np.pad(x, (pad_left, pad_right), mode=mode)
    else:
        x_pad = x

    # Frame using stride tricks (view)
    x_pad = np.ascontiguousarray(x_pad, dtype=np.float32)
    total = int(x_pad.shape[0])
    if total < n_fft:
        x_pad = np.pad(x_pad, (0, n_fft - total), mode="constant")
        total = n_fft

    n_frames = 1 + (total - n_fft) // hop
    shape = (n_frames, n_fft)
    strides = (x_pad.strides[0] * hop, x_pad.strides[0])
    frames = np.lib.stride_tricks.as_strided(x_pad, shape=shape, strides=strides)

    win = _hann(n_fft).astype(np.float32)
    frames_w = frames * win[None, :]
    S = np.fft.rfft(frames_w, n=n_fft, axis=1).astype(np.complex64, copy=False)
    return S, pad_left, pad_right


def _istft(S: np.ndarray, n_fft: int, hop: int, pad_left: int, pad_right: int, *, length: int) -> np.ndarray:
    """Inverse STFT returning 1D float32 signal of given `length`."""
    S = np.asarray(S)
    n_frames = int(S.shape[0])
    win = _hann(n_fft).astype(np.float32)
    out_len = (n_frames - 1) * hop + n_fft
    y = np.zeros((out_len,), dtype=np.float32)
    wsum = np.zeros((out_len,), dtype=np.float32)

    for t in range(n_frames):
        frame = np.fft.irfft(S[t], n=n_fft).astype(np.float32, copy=False)
        start = t * hop
        y[start:start + n_fft] += frame * win
        wsum[start:start + n_fft] += win * win

    # Normalize overlap-add
    eps = 1e-8
    y = y / np.maximum(wsum, eps)

    # Remove padding
    if pad_left or pad_right:
        y = y[pad_left: y.shape[0] - pad_right if pad_right > 0 else None]

    # Ensure requested length
    if y.shape[0] < length:
        y = np.pad(y, (0, length - y.shape[0]), mode="constant")
    elif y.shape[0] > length:
        y = y[:length]
    return y


def _moving_average_freq(x: np.ndarray, k: int) -> np.ndarray:
    """Frequency-axis moving average (over last dimension), keeping the same shape."""
    if k <= 1:
        return x
    k = int(k)
    if k % 2 == 0:
        k += 1  # prefer odd so padding is symmetric
    pad = k // 2
    # Pad with edge values so output length matches input.
    xpad = np.pad(x, ((0, 0), (pad, pad)), mode="edge")
    # Cumsum with a leading zero column for correct window sums.
    c = np.cumsum(np.pad(xpad, ((0, 0), (1, 0)), mode="constant"), axis=1, dtype=np.float32)
    # Window sum over length k for each original position (shape preserved).
    y = (c[:, k:] - c[:, :-k]) / float(k)
    return y


# ----------------------------
# Noise reduction core
# ----------------------------

def _estimate_noise_profile(mag: np.ndarray, mode: str, noise_frames: int, percentile: float) -> np.ndarray:
    """Estimate noise magnitude spectrum from |STFT|.

    mag: (T, F)
    returns: (F,)
    """
    T = mag.shape[0]
    if T == 0:
        return np.ones((mag.shape[1],), dtype=np.float32) * 1e-4

    mode = (mode or "initial").lower().strip()
    if mode == "initial":
        n = max(1, min(int(noise_frames), T))
        prof = np.median(mag[:n, :], axis=0)
    elif mode == "global_median":
        prof = np.median(mag, axis=0)
    elif mode == "global_percentile":
        q = float(np.clip(percentile, 1.0, 99.0))
        prof = np.percentile(mag, q, axis=0)
    else:
        # fallback
        n = max(1, min(int(noise_frames), T))
        prof = np.median(mag[:n, :], axis=0)

    prof = np.asarray(prof, dtype=np.float32)
    return np.maximum(prof, 1e-8)


def _spectral_gate(
    S: np.ndarray,
    sample_rate: int,
    *,
    n_fft: int,
    hop: int,
    noise_profile_mode: str,
    noise_seconds: float,
    noise_percentile: float,
    threshold_db: float,
    knee_db: float,
    reduction_db: float,
    freq_smooth_bins: int,
    attack_ms: float,
    release_ms: float,
    mix: float,
) -> np.ndarray:
    """Apply spectral attenuation to STFT matrix S (T,F)."""
    eps = 1e-8
    mag = np.abs(S).astype(np.float32, copy=False)

    # Noise profile estimation
    noise_frames = int(max(1, round(float(noise_seconds) * float(sample_rate) / float(hop))))
    noise_mag = _estimate_noise_profile(mag, noise_profile_mode, noise_frames, noise_percentile)  # (F,)

    # dB domain comparison with soft knee
    noise_db = 20.0 * np.log10(noise_mag + eps)  # (F,)
    mag_db = 20.0 * np.log10(mag + eps)          # (T,F)
    margin_db = mag_db - noise_db[None, :]

    th = float(threshold_db)
    knee = float(max(knee_db, 1e-6))
    lower = th - knee

    # Max attenuation (floor gain)
    gain_floor = float(10.0 ** (-float(reduction_db) / 20.0))
    gain_floor = float(np.clip(gain_floor, 0.0, 1.0))

    # Piecewise-linear soft knee in [lower, th]
    t = (margin_db - lower) / knee
    t = np.clip(t, 0.0, 1.0).astype(np.float32, copy=False)
    gain = (gain_floor + t * (1.0 - gain_floor)).astype(np.float32, copy=False)

    # Frequency smoothing
    fsb = int(max(1, freq_smooth_bins))
    if fsb > 1:
        gain = _moving_average_freq(gain, fsb).astype(np.float32, copy=False)

    # Time smoothing (attack/release on gain)
    atk = max(float(attack_ms), 0.1) / 1000.0
    rel = max(float(release_ms), 0.1) / 1000.0
    # Coeff per frame (based on hop length)
    frame_dt = float(hop) / float(sample_rate)
    a_atk = float(np.exp(-frame_dt / atk))
    a_rel = float(np.exp(-frame_dt / rel))

    # Iterate frames; vectorized across frequency bins
    for i in range(1, gain.shape[0]):
        prev = gain[i - 1]
        cur = gain[i]
        coeff = np.where(cur < prev, a_atk, a_rel).astype(np.float32, copy=False)
        gain[i] = coeff * prev + (1.0 - coeff) * cur

    # Apply gain, preserve phase
    mix = float(np.clip(mix, 0.0, 1.0))
    if mix >= 0.999:
        Sout = S * gain.astype(np.float32)
    elif mix <= 0.001:
        Sout = S
    else:
        Sout = S * (mix * gain + (1.0 - mix)).astype(np.float32)

    return Sout.astype(np.complex64, copy=False)


def _noise_reduce(
    audio: np.ndarray,
    sample_rate: int,
    *,
    fft_size: int,
    overlap: float,
    noise_profile_mode: str,
    noise_seconds: float,
    noise_percentile: float,
    threshold_db: float,
    knee_db: float,
    reduction_db: float,
    freq_smooth_bins: int,
    attack_ms: float,
    release_ms: float,
    mix: float,
) -> np.ndarray:
    """Spectral noise reduction for (N,C) audio."""
    a = _ensure_float32(audio)
    sr = int(sample_rate) if sample_rate else 48000

    n_fft = int(max(256, fft_size))
    overlap = float(np.clip(overlap, 0.0, 0.95))
    hop = int(max(64, round(n_fft * (1.0 - overlap))))
    hop = max(1, hop)

    out = np.zeros_like(a, dtype=np.float32)
    for ch in range(a.shape[1]):
        x = a[:, ch]
        S, pad_l, pad_r = _stft(x, n_fft=n_fft, hop=hop, center=True)
        S2 = _spectral_gate(
            S, sr,
            n_fft=n_fft,
            hop=hop,
            noise_profile_mode=noise_profile_mode,
            noise_seconds=noise_seconds,
            noise_percentile=noise_percentile,
            threshold_db=threshold_db,
            knee_db=knee_db,
            reduction_db=reduction_db,
            freq_smooth_bins=freq_smooth_bins,
            attack_ms=attack_ms,
            release_ms=release_ms,
            mix=mix,
        )
        y = _istft(S2, n_fft=n_fft, hop=hop, pad_left=pad_l, pad_right=pad_r, length=a.shape[0])
        out[:, ch] = y

    return out


# ----------------------------
# Plugin class
# ----------------------------

class NoiseReductionPlugin(AudioEffectPluginBase):
    plugin_id = "noise_reduction"
    display_name = "Spectral noise reduction"
    params = [
        ParamSpec.choice(
            "fft_size",
            "FFT size",
            default="2048",
            choices=("512", "1024", "2048", "4096"),
        ),
        ParamSpec.float(
            "overlap",
            "Overlap (0..0.95)",
            default=0.75,
            min_value=0.0,
            max_value=0.95,
            step=0.01,
            fmt="%.2f",
        ),
        ParamSpec.choice(
            "noise_profile_mode",
            "Noise profile",
            default="initial",
            choices=("initial", "global_median", "global_percentile"),
        ),
        ParamSpec.float(
            "noise_seconds",
            "Noise seconds (for 'initial')",
            default=0.50,
            min_value=0.05,
            max_value=10.0,
            step=0.05,
            fmt="%.2f",
        ),
        ParamSpec.float(
            "noise_percentile",
            "Percentile (for 'global_percentile')",
            default=20.0,
            min_value=1.0,
            max_value=80.0,
            step=1.0,
            fmt="%.0f",
        ),
        ParamSpec.float(
            "threshold_db",
            "Threshold above noise (dB)",
            default=6.0,
            min_value=0.0,
            max_value=24.0,
            step=0.5,
            fmt="%.1f",
        ),
        ParamSpec.float(
            "knee_db",
            "Soft knee (dB)",
            default=6.0,
            min_value=0.5,
            max_value=24.0,
            step=0.5,
            fmt="%.1f",
        ),
        ParamSpec.float(
            "reduction_db",
            "Max reduction (dB)",
            default=12.0,
            min_value=0.0,
            max_value=40.0,
            step=0.5,
            fmt="%.1f",
        ),
        ParamSpec.integer(
            "freq_smooth_bins",
            "Freq smoothing (bins)",
            default=3,
            min_value=1,
            max_value=31,
            step=2,
        ),
        ParamSpec.float(
            "attack_ms",
            "Attack (ms)",
            default=5.0,
            min_value=0.1,
            max_value=100.0,
            step=0.5,
            fmt="%.1f",
        ),
        ParamSpec.float(
            "release_ms",
            "Release (ms)",
            default=60.0,
            min_value=0.1,
            max_value=500.0,
            step=1.0,
            fmt="%.1f",
        ),
        ParamSpec.float(
            "mix",
            "Wet mix (0..1)",
            default=1.0,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            fmt="%.2f",
        ),
    ]

    def apply(self, audio: np.ndarray, sample_rate: int, params: Dict[str, Any]) -> np.ndarray:
        fft_size = int(params.get("fft_size", "2048"))
        overlap = float(params.get("overlap", 0.75))
        noise_profile_mode = str(params.get("noise_profile_mode", "initial"))
        noise_seconds = float(params.get("noise_seconds", 0.50))
        noise_percentile = float(params.get("noise_percentile", 20.0))
        threshold_db = float(params.get("threshold_db", 6.0))
        knee_db = float(params.get("knee_db", 6.0))
        reduction_db = float(params.get("reduction_db", 12.0))
        freq_smooth_bins = int(params.get("freq_smooth_bins", 3))
        attack_ms = float(params.get("attack_ms", 5.0))
        release_ms = float(params.get("release_ms", 60.0))
        mix = float(params.get("mix", 1.0))

        return _noise_reduce(
            audio,
            sample_rate,
            fft_size=fft_size,
            overlap=overlap,
            noise_profile_mode=noise_profile_mode,
            noise_seconds=noise_seconds,
            noise_percentile=noise_percentile,
            threshold_db=threshold_db,
            knee_db=knee_db,
            reduction_db=reduction_db,
            freq_smooth_bins=freq_smooth_bins,
            attack_ms=attack_ms,
            release_ms=release_ms,
            mix=mix,
        )


PLUGIN = NoiseReductionPlugin()
