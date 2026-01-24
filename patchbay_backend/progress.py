"""Progress reporting for both CLI and GUI scenarios.

A CLI typically wants a single-line live status (TTY) or periodic lines
(non-TTY) that include an elapsed time and a percent indicator.

A GUI backend typically wants to receive progress events via a callback so
it can update widgets (progress bar, current chunk label, ...).

To support both, this module defines a small interface `ProgressReporter` and
provides:
- `ConsoleProgress`: live console display
- `CallbackProgress`: forwards events to a Python callback
- `NullProgress`: does nothing

The rest of the backend depends only on the interface.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol

import inspect


class ProgressReporter(Protocol):
    """Interface implemented by all progress reporters."""

    def header(self, lines: List[str]) -> None:
        """Print or emit initial header lines."""

    def update(self, message: str, percent: Optional[float] = None) -> None:
        """Update the current progress state."""

    def done(self, message: str = "DONE") -> None:
        """Mark completion."""

    def fail(self, message: str = "FAILED") -> None:
        """Mark failure."""


@dataclass
class _ConsoleState:
    start_t: float
    last_len: int


class ConsoleProgress:
    """Progress reporter that prints to stderr.

    Features
    --------
    - Shows elapsed time and percent.
    - On TTY: uses carriage return to update a single line.
    - On non-TTY: prints each update as a full line (useful for logs).

    Notes
    -----
    We intentionally write to stderr so stdout can remain reserved for
    program output in future extensions.
    """

    def __init__(self, stream=None):
        self.stream = stream or sys.stderr
        self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self._state = _ConsoleState(start_t=time.time(), last_len=0)

    def _elapsed(self) -> str:
        s = int(time.time() - self._state.start_t)
        mm = s // 60
        ss = s % 60
        return f"{mm:02d}:{ss:02d}"

    @staticmethod
    def _pct_str(p: Optional[float]) -> str:
        if p is None:
            return "   ---%"
        p = max(0.0, min(100.0, float(p)))
        return f"{p:7.2f}%"

    def header(self, lines: List[str]) -> None:
        for line in lines:
            self.stream.write(line.rstrip() + "\n")
        self.stream.flush()

    def update(self, message: str, percent: Optional[float] = None) -> None:
        line = f"[{self._elapsed()}] {self._pct_str(percent)} | {message}"

        if self.is_tty:
            pad = max(0, self._state.last_len - len(line))
            self.stream.write("\r" + line + (" " * pad))
            self.stream.flush()
            self._state.last_len = len(line)
        else:
            self.stream.write(line + "\n")
            self.stream.flush()
            self._state.last_len = 0

    def _newline(self) -> None:
        if self.is_tty:
            self.stream.write("\n")
            self.stream.flush()
            self._state.last_len = 0

    def done(self, message: str = "DONE") -> None:
        self.update(message, percent=100.0)
        self._newline()

    def fail(self, message: str = "FAILED") -> None:
        self.update(message, percent=None)
        self._newline()


class CallbackProgress:
    """Progress reporter that forwards events to a callback.

    Preferred callback signature
    ----------------------------
    The preferred callback signature is::

        callback(event: str, message: str, percent: Optional[float]) -> None

    Backwards compatible signatures
    -------------------------------
    To make the backend easy to embed into various GUIs, we also support
    older / simpler callback forms that appear frequently in small apps:

    - ``callback(message: str, percent: Optional[float])``
    - ``callback(message: str)``

    In those cases, the *event* information is not forwarded.

    Where `event` is one of: "header", "update", "done", "fail".

    This is intentionally minimal so GUI frameworks can wrap it easily.
    """

    def __init__(self, callback: Callable[..., None]):
        self._cb = callback

        # Determine how many positional arguments the callback can accept.
        #
        # We avoid brittle try/except TypeError-on-every-call because that
        # can accidentally hide genuine callback bugs.
        #
        # Rules:
        # - If the callback has *args (VAR_POSITIONAL), we assume it can
        #   accept the full (event, message, percent) triplet.
        # - Otherwise, we count positional parameters and use 3/2/1/0.
        self._arity = 3
        try:
            sig = inspect.signature(callback)
            params = list(sig.parameters.values())
            if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
                self._arity = 3
            else:
                positional = [
                    p for p in params
                    if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                ]
                self._arity = len(positional)
        except Exception:
            # If signature introspection fails (builtins, callables with
            # unusual __signature__), fall back to the safest option.
            self._arity = 3

    def _emit(self, event: str, message: str, percent: Optional[float]) -> None:
        """Emit a progress event using the best-matching callback signature."""
        if self._arity >= 3:
            self._cb(event, message, percent)
        elif self._arity == 2:
            self._cb(message, percent)
        elif self._arity == 1:
            self._cb(message)
        else:
            self._cb()

    def header(self, lines: List[str]) -> None:
        for line in lines:
            self._emit("header", line, None)

    def update(self, message: str, percent: Optional[float] = None) -> None:
        self._emit("update", message, percent)

    def done(self, message: str = "DONE") -> None:
        self._emit("done", message, 100.0)

    def fail(self, message: str = "FAILED") -> None:
        self._emit("fail", message, None)


class NullProgress:
    """Progress reporter that does nothing (useful for silent batch processing)."""

    def header(self, lines: List[str]) -> None:
        return

    def update(self, message: str, percent: Optional[float] = None) -> None:
        return

    def done(self, message: str = "DONE") -> None:
        return

    def fail(self, message: str = "FAILED") -> None:
        return
