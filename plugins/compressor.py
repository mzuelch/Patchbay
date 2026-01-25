"""Compressor Audio FX plugin for PATCHBAY.

Envelope-based (peak detector) hard-knee compressor with attack/release
smoothing.

Implementation
--------------
The full algorithm is contained in this plugin file (NumPy-only). It works for
mono or multi-channel input; for multi-channel audio the detector uses the peak
across channels to preserve stereo balance.
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
        raise ValueError(f"audio must be 1D or 2D, got ndim={audio.ndim}")

    if audio2.dtype != np.float32:
        audio2 = audio2.astype(np.float32, copy=False)
    return np.ascontiguousarray(audio2)


def _compressor_peak_ar(
    audio: np.ndarray,
    *,
    sample_rate: int,
    threshold_db: float,
    ratio: float,
    attack_ms: float = 10.0,
    release_ms: float = 100.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Envelope-based hard-knee compressor (peak detector, attack/release)."""
    x = _ensure_float32(audio)
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
        coeff = a_coeff if d > env_prev else r_coeff
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


class CompressorPlugin(AudioEffectPluginBase):
    plugin_id = "compressor"
    display_name = "Compressor"
    params = [
        ParamSpec.float("threshold_db", "Threshold (dBFS)", default=-18.0, step=1.0, fmt="%.1f"),
        ParamSpec.float("ratio", "Ratio (>= 1.0)", default=2.0, min_value=1.0, step=0.1, fmt="%.2f"),
        ParamSpec.float("attack_ms", "Attack (ms)", default=10.0, min_value=0.0, step=1.0, fmt="%.0f"),
        ParamSpec.float("release_ms", "Release (ms)", default=100.0, min_value=0.0, step=1.0, fmt="%.0f"),
    ]

    def apply(self, audio: np.ndarray, sample_rate: int, params: Dict[str, Any]) -> np.ndarray:
        thr = float(params.get("threshold_db", -18.0))
        ratio = float(params.get("ratio", 2.0))
        att = float(params.get("attack_ms", 10.0))
        rel = float(params.get("release_ms", 100.0))
        return _compressor_peak_ar(
            audio,
            sample_rate=int(sample_rate),
            threshold_db=thr,
            ratio=ratio,
            attack_ms=att,
            release_ms=rel,
        )


PLUGIN = CompressorPlugin()
