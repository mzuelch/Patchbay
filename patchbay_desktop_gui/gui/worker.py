"""Background worker for PATCHBAY (running the SAM-Audio backend).

This module provides a tiny threading wrapper used by the DearPyGui frontend.

Why a worker?
-------------
Deep learning inference and audio IO can take seconds to minutes. Running the
backend on the GUI thread would freeze the UI. The :class:`BackendWorker`
executes :func:`patchbay_backend.pipeline.run_pipeline` in a background thread
and forwards progress updates and terminal states (done/cancelled/error) via a
thread-safe :class:`queue.Queue`.

The GUI is expected to poll :attr:`BackendWorker.queue` periodically.
"""

from __future__ import annotations

import queue
import threading
import traceback as _tb
from dataclasses import dataclass
from typing import Optional

from patchbay_backend.cancel_token import CancellationToken, CancelledError
from patchbay_backend.file_logging import FileLogger
from patchbay_backend.pipeline import run_pipeline
from patchbay_backend.progress import CallbackProgress
from patchbay_backend.config import Config


@dataclass
class WorkerEvent:
    """Event emitted by :class:`BackendWorker`.

    Attributes
    ----------
    kind:
        One of ``"progress"``, ``"done"``, ``"cancelled"``, ``"error"``.
    message:
        Human-readable message suitable for a status label.
    percent:
        Progress percent in the range 0..100 (only for ``kind="progress"``).
    out_target / out_residual:
        Output paths (only for ``kind="done"``).
    traceback_text:
        Full traceback text (only for ``kind="error"``).
    """

    kind: str
    message: str = ""
    percent: Optional[float] = None
    out_target: Optional[str] = None
    out_residual: Optional[str] = None
    traceback_text: Optional[str] = None


class BackendWorker:
    """Run the backend in a background thread and emit events to a queue."""

    def __init__(self) -> None:
        self.queue: "queue.Queue[WorkerEvent]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._cancel: Optional[CancellationToken] = None
        self._lock = threading.Lock()

    def is_running(self) -> bool:
        t = self._thread
        return bool(t is not None and t.is_alive())

    def start(self, cfg: Config, *, logger: Optional[FileLogger] = None) -> None:
        """Start processing if not already running."""
        with self._lock:
            if self.is_running():
                return

            self._cancel = CancellationToken()

            def _on_progress(event: str, message: str, percent: Optional[float]) -> None:
                # Keep payload small; GUI combines percent+message.
                self.queue.put(WorkerEvent(kind="progress", message=message or event, percent=percent))

            progress = CallbackProgress(_on_progress)

            def _run() -> None:
                try:
                    out_target, out_residual = run_pipeline(
                        cfg,
                        progress=progress,
                        logger=logger,
                        cancel_token=self._cancel,
                    )
                    self.queue.put(
                        WorkerEvent(
                            kind="done",
                            message="Done",
                            out_target=str(out_target) if out_target else "",
                            out_residual=str(out_residual) if out_residual else "",
                        )
                    )
                except CancelledError:
                    self.queue.put(WorkerEvent(kind="cancelled", message="Cancelled"))
                except Exception:
                    if logger and logger.enabled:
                        logger.log_error("Backend worker exception:")
                        logger.log_error(_tb.format_exc())
                    self.queue.put(
                        WorkerEvent(
                            kind="error",
                            message="Error",
                            traceback_text=_tb.format_exc(),
                        )
                    )

            self._thread = threading.Thread(target=_run, name="PatchbayBackendWorker", daemon=True)
            self._thread.start()

    def cancel(self) -> None:
        """Request cancellation (best effort)."""
        with self._lock:
            if self._cancel is not None:
                try:
                    self._cancel.cancel()
                except Exception:
                    pass
