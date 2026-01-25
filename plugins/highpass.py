"""High-pass filter Audio FX plugin for PATCHBAY.

Linear-phase FIR high-pass filter (windowed-sinc, spectral inversion) with FFT
overlap-add convolution.

Implementation
--------------
The full FIR design + FFT overlap-add convolution is contained in this plugin
file (NumPy-only). The filter is applied *non-causally* (centered convolution)
so the output stays aligned with the input for offline processing.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from patchbay_desktop_gui.audiofx.base import AudioEffectPluginBase, ParamSpec


def _ensure_float32(audio: np.ndarray) -> np.ndarray:
    """Return a contiguous float32 array with shape (N, C)."""
    if audio is None:
        raise ValueError("audio is None")
    if audio.ndim == 1:
        audio2 = audio[:, None]
    elif audio.ndim == 2:
        audio2 = audio
    else:
        raise ValueError(f"audio must be 1D or 2D (N or N,C), got shape={audio.shape}")
    if audio2.dtype != np.float32:
        audio2 = audio2.astype(np.float32, copy=False)
    return np.ascontiguousarray(audio2)


def _next_pow2(n: int) -> int:
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


# Mapping of slope choice to FIR length (odd tap count).
# Larger tap counts => narrower transition band (steeper slope) at the cost of
# higher CPU and stronger ringing near the cutoff.
_STEEPNESS_TAPS = {
    6: 101,
    12: 201,
    18: 401,
    24: 801,
    36: 1201,
    48: 1601,
}


def _design_lowpass_kernel(*, sample_rate: int, cutoff_hz: float, taps: int) -> np.ndarray:
    """Design a symmetric windowed-sinc low-pass kernel (normalized DC gain)."""
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


def _design_highpass_kernel(*, sample_rate: int, cutoff_hz: float, taps: int) -> np.ndarray:
    """Design a symmetric windowed-sinc high-pass kernel (spectral inversion).

    We design a low-pass prototype and then perform spectral inversion:
        h_hp[n] = delta[n] - h_lp[n]
    This yields ~0 DC gain and ~1 gain in the pass band above cutoff.
    """
    h_lp = _design_lowpass_kernel(sample_rate=sample_rate, cutoff_hz=cutoff_hz, taps=taps).astype(np.float64, copy=False)
    M = int(h_lp.shape[0])
    if M % 2 == 0:
        raise ValueError("kernel length M must be odd")
    h_hp = -h_lp
    h_hp[(M - 1) // 2] += 1.0
    return h_hp


def _fft_convolve_same_ola(
    x: np.ndarray,
    h: np.ndarray,
    *,
    block_len: int = 262_144,
) -> np.ndarray:
    """FFT-based overlap-add convolution returning 'same' length as x."""
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


def _highpass_fir(
    audio: np.ndarray,
    *,
    sample_rate: int,
    cutoff_hz: float,
    steepness_db_oct: int = 24,
) -> np.ndarray:
    """Apply a linear-phase FIR high-pass filter (float32, shape (N, C))."""
    x = _ensure_float32(audio)
    if x.size == 0:
        return x

    slope = int(steepness_db_oct)
    taps = int(_STEEPNESS_TAPS.get(slope, _STEEPNESS_TAPS[24]))

    h = _design_highpass_kernel(sample_rate=int(sample_rate), cutoff_hz=float(cutoff_hz), taps=taps)

    y = np.empty_like(x, dtype=np.float32)
    for c in range(x.shape[1]):
        y[:, c] = _fft_convolve_same_ola(x[:, c], h)
    return y


class HighpassPlugin(AudioEffectPluginBase):
    plugin_id = "highpass"
    display_name = "High-pass filter"

    params = [
        ParamSpec.float(
            "cutoff_hz",
            "Cutoff frequency (Hz)",
            default=120.0,
            min_value=20.0,
            max_value=20000.0,
            step=10.0,
            fmt="%.0f",
        ),
        ParamSpec.choice(
            "steepness",
            "Slope",
            default="24 dB/oct",
            choices=(
                "6 dB/oct",
                "12 dB/oct",
                "18 dB/oct",
                "24 dB/oct",
                "36 dB/oct",
                "48 dB/oct",
            ),
        ),
    ]

    def apply(self, audio: np.ndarray, sample_rate: int, params: Dict[str, Any]) -> np.ndarray:
        cutoff = float(params.get("cutoff_hz", 120.0))
        slope_s = str(params.get("steepness", "24 dB/oct"))
        try:
            slope = int(slope_s.strip().split()[0])
        except Exception:
            slope = 24
        return _highpass_fir(audio, sample_rate=int(sample_rate), cutoff_hz=cutoff, steepness_db_oct=slope)


PLUGIN = HighpassPlugin()
