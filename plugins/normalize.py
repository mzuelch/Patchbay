"""Normalize Audio FX plugin for PATCHBAY.

This plugin performs *peak normalization* to a target peak level.

Implementation
--------------
The full algorithm is contained in this plugin file (NumPy-only).
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


def _peak_normalize(audio: np.ndarray, *, target_peak: float = 0.999) -> np.ndarray:
    """Peak-normalize an audio buffer (float32, shape (N, C))."""
    x = _ensure_float32(audio)
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


class NormalizePlugin(AudioEffectPluginBase):
    plugin_id = "normalize"
    display_name = "Normalization"
    params = [
        ParamSpec.float(
            "target_peak",
            "Target peak (0..1)",
            default=0.999,
            min_value=0.001,
            max_value=1.0,
            step=0.01,
            fmt="%.3f",
        )
    ]

    def apply(self, audio: np.ndarray, sample_rate: int, params: Dict[str, Any]) -> np.ndarray:
        tp = float(params.get("target_peak", 0.999))
        return _peak_normalize(audio, target_peak=tp)


PLUGIN = NormalizePlugin()
