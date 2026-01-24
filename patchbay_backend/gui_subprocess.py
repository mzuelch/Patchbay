"""GUI subprocess entrypoint.

This module exists for the DearPyGui frontend: it runs the heavy ML backend in a
*separate process* so that CUDA contexts and GPU memory are reliably released
when the job is finished (process exit).

Protocol
--------
- Reads a JSON config file produced by the GUI (see patchbay_desktop_gui.worker).
- Emits progress as JSON Lines (one JSON object per line) on stdout.
  The GUI reads these lines and updates the progress bar.

Each line has at least:
  {"event": "<header|update|done|fail|result>", "message": "...", "percent": <float|null>}

The final "result" line additionally contains:
  {"out_target": "...", "out_residual": "..."}.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

from .config import Config, Anchor
from .file_logging import FileLogger
from .pipeline import run_pipeline
from .progress import CallbackProgress
from .warnings_ctl import suppress_warnings


def _emit(obj: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)

    # Anchors come from JSON as lists; convert to list[tuple[str,float,float]].
    anchors_raw = d.get("anchors", []) or []
    anchors: List[Anchor] = []
    for a in anchors_raw:
        if not isinstance(a, (list, tuple)) or len(a) != 3:
            continue
        sign = str(a[0])
        try:
            start_s = float(a[1])
            end_s = float(a[2])
        except Exception:
            continue
        anchors.append((sign, start_s, end_s))

    # Build Config (ignore unknown keys gracefully).
    allowed = {
        "model", "audio", "description", "out_target", "out_residual",
        "predict_spans", "reranking_candidates",
        "device", "fp16",
        "max_len_s", "overlap_s",
        "anchor_mode",
        "no_resample",
        "log_file", "debug",
    }
    cfg_kwargs = {k: d[k] for k in d.keys() if k in allowed}
    cfg_kwargs["anchors"] = anchors
    cfg = Config(**cfg_kwargs)  # type: ignore[arg-type]
    cfg.validate_basic()
    return cfg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="patchbay-gui-worker",
        description="Run patchbay_backend in a separate process for GUI usage (JSON progress on stdout).",
    )
    p.add_argument("--config", required=True, help="Path to JSON config file.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    suppress_warnings()
    faulthandler.enable(all_threads=True)

    args = build_parser().parse_args(argv)
    cfg = _load_config(args.config)

    # Progress callback: always JSON lines on stdout.
    def cb(event: str, message: str, percent: Optional[float]) -> None:
        _emit({"event": str(event), "message": str(message), "percent": percent})

    progress = CallbackProgress(cb)
    logger = FileLogger(cfg.log_file)

    try:
        out_t, out_r = run_pipeline(cfg, progress=progress, logger=logger)
        # Ensure a final machine-readable result for the GUI.
        _emit({"event": "result", "message": "OK", "percent": 100.0, "out_target": out_t, "out_residual": out_r})
        return 0
    except Exception as e:
        tb = traceback.format_exc()
        _emit({"event": "fail", "message": f"{type(e).__name__}: {e}", "percent": None, "traceback": tb})
        return 1
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
