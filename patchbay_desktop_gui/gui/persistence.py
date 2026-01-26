"""Persistent GUI settings (Qt-free).

The original Qt6 version used QSettings. To avoid any Qt dependency, this
DearPyGui version stores settings as JSON under the user's roaming profile.

Windows location
----------------
We store settings in::

    %APPDATA%\CatSynth\PATCHBAY\settings.json

Rationale:
- %APPDATA% is writable without admin rights.
- Roaming profile is a reasonable default for personal workstation tools.

The settings are intentionally kept simple (plain dict). The GUI reads them
on startup and writes them on changes or on exit.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple


RGBA = Tuple[int, int, int, int]


def _appdata_dir() -> Path:
    base = os.environ.get("APPDATA") or os.environ.get("LOCALAPPDATA") or str(Path.home())
    # Keep vendor prefix stable, but use the application name for the leaf.
    return Path(base) / "CatSynth" / "PATCHBAY"


def settings_path() -> Path:
    return _appdata_dir() / "settings.json"


def _default_settings() -> Dict[str, Any]:
    return {
        "ui": {
            # Anchor overlays
            "anchor_plus_color": [0, 180, 0, 80],
            "anchor_minus_color": [200, 60, 60, 80],
            # Marker selection
            "marker_color": [120, 120, 120, 80],
            # Playhead line
            "playhead_color": [255, 200, 0, 220],
            # Number of points drawn in the waveform canvas per view.
            "waveform_points": 6000,
        },
        "backend": {
            "model": "facebook/sam-audio-small",
            "device": "auto",
            "fp16": True,
            "predict_spans": False,
            "reranking_candidates": 1,
            "anchor_mode": "strict",
            "no_resample": False,
        },
        "logging": {
            "enabled": False,
            "level": "Complete",
            "file": "",
            "append": True,
        },
        "chunking_global": {
            "use_chunking": False,
            "max_len_s": 15.0,
            "overlap_s": 2.0,
        },
        "session": {
            "last_audio": "",
            "last_description": "",
        },
        "audiofx": {
            "input": {},
            "target": {},
            "residual": {},
        },
        "output_processing": {
            # Defaults for output post-processing tools
            # (used by the DearPyGui output page).
            "compressor_threshold_db": -12.0,
            "compressor_ratio": 2.0,
            "compressor_attack_ms": 10.0,
            "compressor_release_ms": 100.0,
            # Peak normalization target (linear full scale).
            # 0.999 leaves a tiny headroom to reduce risk of clipping.
            "normalize_target_peak": 0.999,
            # Maximum number of undo steps kept per output buffer.
            "max_undo_steps": 10,
        },
    }


@dataclass
class Settings:
    """Simple JSON-backed settings helper."""

    data: Dict[str, Any]

    @classmethod
    def load(cls) -> "Settings":
        p = settings_path()
        if not p.exists():
            return cls(_default_settings())
        try:
            with open(p, "r", encoding="utf-8") as f:
                loaded = json.load(f)
        except Exception:
            # Corrupted file -> reset to defaults.
            loaded = _default_settings()

        # Merge with defaults so new keys appear after upgrades.
        merged = _default_settings()
        _deep_update(merged, loaded)
        return cls(merged)

    def save(self) -> None:
        p = settings_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    # Convenience getters -------------------------------------------------

    def get_rgba(self, key: str, default: RGBA) -> RGBA:
        v = self.data.get("ui", {}).get(key, list(default))
        if isinstance(v, (list, tuple)) and len(v) == 4:
            try:
                return (int(v[0]), int(v[1]), int(v[2]), int(v[3]))
            except Exception:
                return default
        return default

    def set_rgba(self, key: str, rgba: RGBA) -> None:
        self.data.setdefault("ui", {})[key] = [int(x) for x in rgba]


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    """Recursively update *dst* with *src* without losing unknown keys."""
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
