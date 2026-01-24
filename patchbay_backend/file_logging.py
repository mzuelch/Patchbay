"""File-only logging utilities.

This backend intentionally keeps the console output clean by default.
Detailed logs (debug traces, memory stats, etc.) are written only when the
user explicitly requests a log file (CLI: --log-file), or when a GUI
integrator chooses to capture verbose diagnostic output.

We avoid the standard-library name `logging.py` to prevent shadowing.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Any


def timestamp() -> str:
    """Return a short human-readable timestamp (HH:MM:SS)."""
    return time.strftime("%H:%M:%S")


class FileLogger:
    """A minimal file logger.

    - If `path` is None, logging is disabled and calls are no-ops.
    - Logs are written immediately (flush) to survive crashes.

    The logger is intentionally simple; it is designed to be embedded into
    ML pipelines where reliability matters more than advanced formatting.
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self._fh: Optional[Any] = None

        if path:
            p = Path(path)
            if p.parent and not p.parent.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
            # Overwrite by default: reproducible runs, no accidental growth.
            self._fh = open(p, "w", encoding="utf-8")

    @property
    def enabled(self) -> bool:
        return self._fh is not None

    def log(self, msg: str) -> None:
        """Write a single log line with timestamp."""
        if not self._fh:
            return
        self._fh.write(f"[{timestamp()}] {msg}\n")
        self._fh.flush()

    def close(self) -> None:
        """Close the file handle (safe to call multiple times)."""
        try:
            if self._fh:
                self._fh.flush()
                self._fh.close()
        finally:
            self._fh = None


def fmt_bytes(n: int) -> str:
    """Format bytes into a human-friendly string."""
    x = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if x < 1024.0:
            return f"{x:.1f}{unit}"
        x /= 1024.0
    return f"{x:.1f}EB"


def log_ram(logger: FileLogger, prefix: str = "RAM") -> None:
    """Log system memory stats (requires `psutil`)."""
    if not logger.enabled:
        return
    try:
        import psutil

        vm = psutil.virtual_memory()
        logger.log(f"{prefix}: used={fmt_bytes(vm.used)} avail={fmt_bytes(vm.available)} {vm.percent}%")
    except Exception as e:
        logger.log(f"{prefix}: psutil not available/failed ({e})")


def log_cuda_mem(logger: FileLogger, prefix: str = "CUDA") -> None:
    """Log CUDA memory stats (requires torch and a CUDA device)."""
    if not logger.enabled:
        return
    try:
        import torch

        if not torch.cuda.is_available():
            logger.log(f"{prefix}: cuda not available")
            return

        a = torch.cuda.memory_allocated(0)
        r = torch.cuda.memory_reserved(0)
        mx_a = torch.cuda.max_memory_allocated(0)
        mx_r = torch.cuda.max_memory_reserved(0)
        logger.log(
            f"{prefix}: alloc={fmt_bytes(a)} reserved={fmt_bytes(r)} "
            f"max_alloc={fmt_bytes(mx_a)} max_reserved={fmt_bytes(mx_r)}"
        )
    except Exception as e:
        logger.log(f"{prefix}: failed to query cuda memory ({e})")
