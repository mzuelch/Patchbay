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


LOG_LEVELS = {
    "error": 1,
    "warning": 2,
    "error+warning": 2,
    "info": 3,
    "complete": 3,
}

LOG_LEVEL_LABELS = {
    "Error": "error",
    "Error+Warning": "error+warning",
    "Complete": "complete",
}


def normalize_log_level(level: Optional[str]) -> str:
    """Normalize a log level label to an internal key."""
    if not level:
        return "complete"
    if level in LOG_LEVEL_LABELS:
        return LOG_LEVEL_LABELS[level]
    key = str(level).strip().lower()
    if key in LOG_LEVELS:
        return key
    key_no_space = key.replace(" ", "")
    if key_no_space in LOG_LEVELS:
        return key_no_space
    return "complete"


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

    def __init__(self, path: Optional[str] = None, *, level: str = "Complete", append: bool = True):
        self.path = path
        self._fh: Optional[Any] = None
        self.level = normalize_log_level(level)
        self.append = bool(append)

        if path:
            p = Path(path)
            if p.parent and not p.parent.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if self.append else "w"
            self._fh = open(p, mode, encoding="utf-8")

    @property
    def enabled(self) -> bool:
        return self._fh is not None

    def _should_log(self, level: str) -> bool:
        if not self._fh:
            return False
        key = normalize_log_level(level)
        return LOG_LEVELS.get(key, 3) <= LOG_LEVELS.get(self.level, 3)

    def log(self, msg: str, *, level: str = "info") -> None:
        """Write a single log line with timestamp."""
        if not self._should_log(level):
            return
        self._fh.write(f"[{timestamp()}] {msg}\n")
        self._fh.flush()

    def log_info(self, msg: str) -> None:
        self.log(msg, level="info")

    def log_warning(self, msg: str) -> None:
        self.log(msg, level="warning")

    def log_error(self, msg: str) -> None:
        self.log(msg, level="error")

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
        logger.log_info(f"{prefix}: used={fmt_bytes(vm.used)} avail={fmt_bytes(vm.available)} {vm.percent}%")
    except Exception as e:
        logger.log_warning(f"{prefix}: psutil not available/failed ({e})")


def log_cuda_mem(logger: FileLogger, prefix: str = "CUDA") -> None:
    """Log CUDA memory stats (requires torch and a CUDA device)."""
    if not logger.enabled:
        return
    try:
        import torch

        if not torch.cuda.is_available():
            logger.log_info(f"{prefix}: cuda not available")
            return

        a = torch.cuda.memory_allocated(0)
        r = torch.cuda.memory_reserved(0)
        mx_a = torch.cuda.max_memory_allocated(0)
        mx_r = torch.cuda.max_memory_reserved(0)
        logger.log_info(
            f"{prefix}: alloc={fmt_bytes(a)} reserved={fmt_bytes(r)} "
            f"max_alloc={fmt_bytes(mx_a)} max_reserved={fmt_bytes(mx_r)}"
        )
    except Exception as e:
        logger.log_warning(f"{prefix}: failed to query cuda memory ({e})")
