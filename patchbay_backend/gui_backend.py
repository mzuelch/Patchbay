"""GUI-oriented helper wrappers.

This package is intentionally GUI-framework agnostic. Different GUIs (Qt,
Tk, web, ...) have different threading and event-loop requirements.

To make GUI integration easier, we provide small convenience wrappers that
use the same core pipeline but:
- accept a simple progress callback
- return output paths

You can ignore this module and call `run_pipeline` directly if you prefer.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

from .config import Config, Anchor
from .progress import CallbackProgress, ProgressReporter, NullProgress
from .file_logging import FileLogger
from .pipeline import run_pipeline
from .cancel_token import CancellationToken


ProgressCallback = Callable[[str, str, Optional[float]], None]


def run_with_callback(
    cfg: Config,
    *,
    callback: Optional[ProgressCallback] = None,
    log_file: Optional[str] = None,
    cancel_token: Optional[CancellationToken] = None,
) -> Tuple[str, str]:
    """Run the pipeline and forward progress events to a callback.

    Parameters
    ----------
    cfg:
        A pre-built Config.
    callback:
        Function called as: callback(event, message, percent)
        where event is one of: header/update/done/fail.
    log_file:
        Optional explicit log file override. If provided, it takes precedence
        over cfg.log_file.

    Returns
    -------
    (target_path, residual_path)

    Notes
    -----
    - Many GUI frameworks require that UI updates happen on the main thread.
      If you run this function in a background thread, your callback must
      marshal updates back to the UI thread.
    """

    progress: ProgressReporter
    if callback is None:
        progress = NullProgress()
    else:
        progress = CallbackProgress(callback)

    logger = FileLogger(log_file if log_file is not None else cfg.log_file)
    try:
        return run_pipeline(cfg, progress=progress, logger=logger, cancel_token=cancel_token)
    finally:
        logger.close()
