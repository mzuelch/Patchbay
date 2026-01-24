"""Backward-compatible entry point.

The GUI implementation was modularized into :mod:`patchbay_desktop_gui.gui`.
This module remains as a stable import target for launchers such as
``patchbay_gui.py``.
"""

from __future__ import annotations

from .gui.app import main

__all__ = ["main"]
