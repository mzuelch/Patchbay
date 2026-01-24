"""Command-line interface (CLI).

This module provides a stable CLI entry point for the backend package.

Key properties
--------------
- By default, the CLI is quiet: it suppresses warnings and does not emit
  verbose logs.
- It shows a live progress indicator with percent and current work unit.
- If the user specifies `--log-file`, detailed diagnostics are written to that file.

The CLI is deliberately thin: it only parses arguments, builds a `Config`,
creates the logger and progress reporter, then calls `run_pipeline`.

For GUI integration, import and call `patchbay_backend.pipeline.run_pipeline` directly.
"""

from __future__ import annotations

import argparse
import sys
import faulthandler

from .config import Config
from .progress import ConsoleProgress
from .file_logging import FileLogger
from .warnings_ctl import suppress_warnings
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser.

    We use argparse.BooleanOptionalAction for flags like --fp16/--no-fp16
    where supported (Python 3.9+).
    """

    p = argparse.ArgumentParser(
        prog="patchbay-anchor",
        description="PATCHBAY / facebook/SAM-Audio separation with chunking and temporal anchors (backend + CLI)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model + task
    p.add_argument("--model", default="facebook/sam-audio-small",
                   help="HF model id or local directory (from_pretrained).")
    p.add_argument("--audio", required=True, help="Path to the input WAV file.")
    p.add_argument("--description", required=True, help="Text description of the desired sound.")

    # Model parameters
    p.add_argument(
        "--predict-spans",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable span prediction (can improve results but increases compute).",
    )
    p.add_argument(
        "--reranking-candidates",
        type=int,
        default=1,
        help="Number of reranking candidates (>=1). Higher uses more memory/compute.",
    )

    # Compute
    p.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Force compute device or select automatically.",
    )
    p.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use FP16 autocast/weights on CUDA (reduces VRAM).",
    )

    # Chunking
    p.add_argument(
        "--max-len-s",
        type=float,
        default=None,
        help="Maximum chunk length in seconds. Setting this forces chunking.",
    )
    p.add_argument(
        "--overlap-s",
        type=float,
        default=None,
        help="Chunk overlap in seconds. Ignored unless --max-len-s is set.",
    )

    # Temporal anchors
    p.add_argument(
        "--anchor",
        action="append",
        nargs=3,
        metavar=("TYPE", "START_S", "END_S"),
        default=None,
        help="Temporal anchor: TYPE '+' or '-', START/END in seconds. Repeatable.",
    )
    p.add_argument(
        "--anchor-mode",
        choices=list(ANCHOR_MODE_CHOICES),
        default="strict",
        help="How to apply anchors when chunking. See ANCHOR_MODE_CHOICES in config.py for details.",
    )

    # Outputs
    p.add_argument(
        "--out-target",
        default=None,
        help="Target output WAV. Default: <audio>_target.wav next to input.",
    )
    p.add_argument(
        "--out-residual",
        default=None,
        help="Residual output WAV. Default: <audio>_residual.wav next to input.",
    )

    # Audio processing
    p.add_argument(
        "--no-resample",
        action="store_true",
        help="Do not resample to the model sample rate (advanced use only).",
    )

    # Diagnostics
    p.add_argument(
        "--log-file",
        default=None,
        help="Write detailed diagnostics to this file (default: no logs).",
    )
    p.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable extra debug logging (only relevant together with --log-file).",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns a process exit code.
    """

    suppress_warnings()

    # Enable faulthandler for post-mortem debugging of hard crashes.
    faulthandler.enable(all_threads=True)

    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = Config.from_cli_args(args)

    # Set up logging (file only) and progress reporting.
    logger = FileLogger(cfg.log_file)
    progress = ConsoleProgress()

    # Print a short header; afterwards we use a single-line live status.
    progress.header([
        "PATCHBAY / facebook/SAM-Audio Separation (backend CLI) — progress",
        f"Command: {' '.join(sys.argv)}",
        f"Config: {cfg.summary()}",
        (f"Log file: {cfg.log_file}" if cfg.log_file else "Log file: (none)"),
        "",
    ])

    try:
        run_pipeline(cfg, progress=progress, logger=logger)
        return 0
    except Exception as e:
        progress.fail(f"Aborted: {type(e).__name__}: {e}")
        # Write full traceback to log file if enabled.
        if logger.enabled:
            import traceback

            logger.log("EXCEPTION: " + repr(e))
            logger.log(traceback.format_exc())
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
