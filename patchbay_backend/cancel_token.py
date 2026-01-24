"""Cancellation primitives for long-running work.

Why this exists
---------------
When the backend is used from a desktop GUI, the user expects to be able to
cancel processing.  In a CLI context, cancellation is typically handled by
Ctrl+C which triggers a KeyboardInterrupt.

Deep learning inference cannot be interrupted safely at arbitrary instruction
boundaries (e.g. mid-kernel on a GPU).  A robust and simple approach is to
support *cooperative cancellation*:

* The GUI sets a cancellation flag.
* The pipeline checks that flag at *safe points* (before starting the next
  chunk, before/after heavy steps) and aborts.

This module provides:

* ``CancellationToken``: thread-safe flag backed by a ``threading.Event``.
* ``CancelledError``: raised when cancellation is detected.

The GUI worker thread can catch ``CancelledError`` and update the UI state
cleanly.
"""

from __future__ import annotations

import threading


class CancelledError(RuntimeError):
    """Raised when a cooperative cancellation request is observed."""


class CancellationToken:
    """A thread-safe cancellation flag.

    The token is intentionally tiny so it can be passed around easily.
    """

    def __init__(self):
        self._event = threading.Event()

    def cancel(self) -> None:
        """Request cancellation."""

        self._event.set()

    def is_cancelled(self) -> bool:
        """Return True if cancellation has been requested."""

        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        """Raise ``CancelledError`` if cancellation has been requested."""

        if self.is_cancelled():
            raise CancelledError("Processing was cancelled by the user")
