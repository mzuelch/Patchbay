"""Anchor sidecar file I/O.

The GUI allows users to create *anchors* (temporal regions) interactively.
To make this workflow practical, anchors should persist alongside the input
audio file.

This module implements a small, stable JSON sidecar format:

    <audio_stem>.anchors.json

Example
-------
If the input file is::

    G:\\audio\\mix.wav

the anchor file will be stored as::

    G:\\audio\\mix.anchors.json

Format
------
The JSON file contains:

* ``version``: integer format version (currently 1)
* ``anchors``: list of objects with ``sign`` ('+' or '-') and ``start_s`` / ``end_s``

We intentionally keep the schema minimal and forward-compatible.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional


# NOTE:
# AnchorItem is the small data container used by the waveform widget to
# represent anchors. Importing it here keeps the persistence layer
# compatible with the interactive editor.
from ..widgets.waveform_widget import AnchorItem


SIDE_CAR_SUFFIX = ".anchors.json"


def anchor_sidecar_path(audio_path: str | Path) -> Path:
    """Return the sidecar path for *audio_path*."""
    p = Path(audio_path)
    return p.parent / f"{p.stem}{SIDE_CAR_SUFFIX}"


def save_anchors_for_audio(audio_path: str | Path, anchors: List[AnchorItem]) -> Path:
    """Save anchors as a JSON sidecar next to *audio_path*.

    If *anchors* is empty, an existing sidecar will be removed.

    Returns
    -------
    Path
        The sidecar path.
    """
    side = anchor_sidecar_path(audio_path)
    if not anchors:
        try:
            if side.exists():
                side.unlink()
        except Exception:
            # Best effort: empty anchors should not crash the GUI.
            pass
        return side

    payload = {
        "version": 1,
        "anchors": [asdict(a) for a in anchors],
    }

    side.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return side


def load_anchors_for_audio(audio_path: str | Path) -> Optional[List[AnchorItem]]:
    """Load anchors from the sidecar next to *audio_path*.

    Returns ``None`` if no sidecar exists.
    Returns an empty list if the file exists but contains no anchors.
    """
    side = anchor_sidecar_path(audio_path)
    if not side.exists():
        return None

    txt = side.read_text(encoding="utf-8")
    obj = json.loads(txt)

    anchors_raw = obj.get("anchors", []) if isinstance(obj, dict) else []
    anchors: List[AnchorItem] = []
    if isinstance(anchors_raw, list):
        for it in anchors_raw:
            if not isinstance(it, dict):
                continue
            sign = str(it.get("sign", "+"))
            if sign not in ("+", "-"):
                continue
            try:
                s = float(it.get("start_s"))
                e = float(it.get("end_s"))
            except Exception:
                continue
            if e <= s:
                continue
            anchors.append(AnchorItem(sign=sign, start_s=s, end_s=e))
    return anchors
