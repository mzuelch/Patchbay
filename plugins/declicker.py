"""De-Clicker Audio FX plugin for PATCHBAY.

This plugin removes short impulse-like clicks/pops using a pragmatic approach:
1) Detect click candidates via a high-frequency proxy (highpass residual or derivative).
2) Group detections into short regions (merge small gaps).
3) Repair each region by inpainting (linear interpolation between boundary samples).

Design goals
------------
- Pure NumPy (no SciPy).
- Fast enough for offline post-processing.
- Typical, useful parameters for de-clicking separated stems.

Limitations
-----------
- Very strong transients may be detected as clicks if sensitivity is too high.
- Long disturbances (> max_click_ms) are skipped (raise max_click_ms if needed).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

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
        a = audio[:, None]
    elif audio.ndim == 2:
        a = audio
    else:
        raise ValueError(f"audio must be 1D or 2D, got ndim={audio.ndim}")

    if a.dtype != np.float32:
        a = a.astype(np.float32, copy=False)
    if not a.flags["C_CONTIGUOUS"]:
        a = np.ascontiguousarray(a)
    return a


def _moving_average_1d(x: np.ndarray, win: int) -> np.ndarray:
    """Centered moving average (reflect padding), pure NumPy, returns float32."""
    x = np.asarray(x, dtype=np.float32)
    n = int(x.shape[0])
    win = int(max(1, win))
    if win <= 1 or n == 0:
        return x.copy()

    # Prefer odd window for symmetric centering
    if win % 2 == 0:
        win += 1
    pad = win // 2

    mode = "reflect" if n > 1 else "edge"
    xpad = np.pad(x, (pad, pad), mode=mode)
    # cumsum with leading zero for easy window sums
    c = np.cumsum(np.pad(xpad, (1, 0), mode="constant"), dtype=np.float64)
    y = (c[win:] - c[:-win]) / float(win)  # length == n
    return y.astype(np.float32, copy=False)


def _mad_sigma(x: np.ndarray) -> float:
    """Robust sigma estimate from MAD (median absolute deviation)."""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return 0.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return 1.4826 * mad  # consistent for normal distribution


def _linear_inpaint(y: np.ndarray, s: int, e: int) -> None:
    """Inpaint y[s:e+1] in-place via linear interpolation between boundary samples."""
    n = int(y.shape[0])
    if n == 0 or s > e:
        return
    s = int(max(0, s))
    e = int(min(n - 1, e))
    if s > e:
        return

    left_i = s - 1
    right_i = e + 1

    if left_i < 0 and right_i >= n:
        # whole signal selected; do nothing
        return
    if left_i < 0:
        y[s:e + 1] = y[right_i]
        return
    if right_i >= n:
        y[s:e + 1] = y[left_i]
        return

    left = float(y[left_i])
    right = float(y[right_i])
    L = e - s + 1
    # +2 so endpoints are included, then drop them
    fill = np.linspace(left, right, L + 2, dtype=np.float32)[1:-1]
    y[s:e + 1] = fill


def _segments_from_mask(mask: np.ndarray, merge_gap: int) -> List[Tuple[int, int]]:
    """Return list of (start,end) segments from boolean mask, merging gaps <= merge_gap."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    merge_gap = int(max(0, merge_gap))

    segs: List[Tuple[int, int]] = []
    s = int(idx[0])
    p = int(idx[0])
    for k in idx[1:]:
        k = int(k)
        if k - p <= merge_gap + 1:
            p = k
        else:
            segs.append((s, p))
            s = k
            p = k
    segs.append((s, p))
    return segs


# ----------------------------
# De-click core
# ----------------------------

def _detect_click_mask(
    x: np.ndarray,
    sr: int,
    *,
    mode: str,
    smooth_ms: float,
    sensitivity: float,
    min_level_db: float,
    merge_ms: float,
) -> Tuple[np.ndarray, int]:
    """Return (mask, merge_gap_samples) for a single channel signal x."""
    n = int(x.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=bool), 0

    sr = int(sr) if sr else 48000
    mode = (mode or "highpass").lower().strip()

    eps = 1e-12
    abs_min = float(10.0 ** (float(min_level_db) / 20.0))  # dBFS -> linear

    if mode == "derivative":
        # First difference as HF proxy
        d = np.diff(x, prepend=x[0]).astype(np.float32, copy=False)
        proxy = np.abs(d)
        sigma = _mad_sigma(proxy)
        thr = max(float(np.median(proxy)) + float(sensitivity) * sigma, abs_min)
        base_mask = proxy > thr

        # Derivative spikes indicate a discontinuity between i-1 and i;
        # expand to cover neighbor samples as well.
        mask = base_mask.copy()
        if n > 1:
            mask[:-1] |= base_mask[1:]
            mask[1:] |= base_mask[:-1]
    else:
        # Highpass proxy via subtracting a moving average (simple lowpass)
        smooth_ms = float(max(0.05, smooth_ms))
        win = int(max(1, round(sr * smooth_ms / 1000.0)))
        lp = _moving_average_1d(x, win)
        hp = (x - lp).astype(np.float32, copy=False)
        proxy = np.abs(hp)

        sigma = _mad_sigma(proxy)
        thr = max(float(np.median(proxy)) + float(sensitivity) * sigma, abs_min)
        mask = proxy > thr

    merge_gap = int(max(0, round(float(merge_ms) * sr / 1000.0)))
    return mask.astype(bool, copy=False), merge_gap


def _declick_channel(
    x: np.ndarray,
    sr: int,
    *,
    mode: str,
    smooth_ms: float,
    sensitivity: float,
    min_level_db: float,
    max_click_ms: float,
    pad_ms: float,
    merge_ms: float,
    mix: float,
) -> np.ndarray:
    """De-click for a single channel (1D float32)."""
    x = np.asarray(x, dtype=np.float32)
    n = int(x.shape[0])
    if n == 0:
        return x.copy()

    sr = int(sr) if sr else 48000

    mask, merge_gap = _detect_click_mask(
        x, sr,
        mode=mode,
        smooth_ms=smooth_ms,
        sensitivity=sensitivity,
        min_level_db=min_level_db,
        merge_ms=merge_ms,
    )
    if not np.any(mask):
        return x.copy()

    segs = _segments_from_mask(mask, merge_gap=merge_gap)
    if not segs:
        return x.copy()

    max_click_samp = int(max(1, round(float(max_click_ms) * sr / 1000.0)))
    pad_samp = int(max(0, round(float(pad_ms) * sr / 1000.0)))

    y = x.copy()
    for s, e in segs:
        s2 = max(0, s - pad_samp)
        e2 = min(n - 1, e + pad_samp)
        if (e2 - s2 + 1) > max_click_samp:
            continue
        _linear_inpaint(y, s2, e2)

    mix = float(np.clip(mix, 0.0, 1.0))
    if mix <= 0.001:
        return x.copy()
    if mix >= 0.999:
        return y
    return (x * (1.0 - mix) + y * mix).astype(np.float32, copy=False)


def _declick(audio: np.ndarray, sample_rate: int, params: Dict[str, Any]) -> np.ndarray:
    a = _ensure_float32(audio)
    sr = int(sample_rate) if sample_rate else 48000

    mode = str(params.get("mode", "highpass"))
    smooth_ms = float(params.get("smooth_ms", 1.0))
    sensitivity = float(params.get("sensitivity", 8.0))
    min_level_db = float(params.get("min_level_db", -45.0))
    max_click_ms = float(params.get("max_click_ms", 5.0))
    pad_ms = float(params.get("pad_ms", 0.20))
    merge_ms = float(params.get("merge_ms", 0.50))
    mix = float(params.get("mix", 1.0))

    out = np.zeros_like(a, dtype=np.float32)
    for ch in range(a.shape[1]):
        out[:, ch] = _declick_channel(
            a[:, ch],
            sr,
            mode=mode,
            smooth_ms=smooth_ms,
            sensitivity=sensitivity,
            min_level_db=min_level_db,
            max_click_ms=max_click_ms,
            pad_ms=pad_ms,
            merge_ms=merge_ms,
            mix=mix,
        )
    return out


# ----------------------------
# Plugin class
# ----------------------------

class DeClickerPlugin(AudioEffectPluginBase):
    plugin_id = "declicker"
    display_name = "De-clicker (impulse/click removal)"
    params = [
        ParamSpec.choice(
            "mode",
            "Detection mode",
            default="highpass",
            choices=("highpass", "derivative"),
        ),
        ParamSpec.float(
            "smooth_ms",
            "HP proxy smoothing (ms) (highpass mode)",
            default=1.0,
            min_value=0.05,
            max_value=10.0,
            step=0.05,
            fmt="%.2f",
        ),
        ParamSpec.float(
            "sensitivity",
            "Sensitivity (higher = more clicks)",
            default=8.0,
            min_value=1.0,
            max_value=25.0,
            step=0.5,
            fmt="%.1f",
        ),
        ParamSpec.float(
            "min_level_db",
            "Min click level (dBFS)",
            default=-45.0,
            min_value=-120.0,
            max_value=0.0,
            step=1.0,
            fmt="%.0f",
        ),
        ParamSpec.float(
            "max_click_ms",
            "Max click length (ms)",
            default=5.0,
            min_value=0.2,
            max_value=50.0,
            step=0.2,
            fmt="%.1f",
        ),
        ParamSpec.float(
            "pad_ms",
            "Padding around detected click (ms)",
            default=0.20,
            min_value=0.0,
            max_value=5.0,
            step=0.05,
            fmt="%.2f",
        ),
        ParamSpec.float(
            "merge_ms",
            "Merge gap (ms)",
            default=0.50,
            min_value=0.0,
            max_value=10.0,
            step=0.05,
            fmt="%.2f",
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
        return _declick(audio, sample_rate, params)


PLUGIN = DeClickerPlugin()
