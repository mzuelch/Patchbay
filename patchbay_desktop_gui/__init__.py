"""PATCHBAY desktop GUI package.

This repository contains a DearPyGui-based desktop frontend for the
``patchbay_backend`` pipeline.

Stable import targets
---------------------
The project is often executed *from source* (without installation). To keep
launchers and external scripts stable, the following imports are considered
part of the public API:

- :func:`patchbay_desktop_gui.app.main` – GUI entry point
- :class:`patchbay_desktop_gui.main_window.MainWindow` – main controller
- :class:`patchbay_desktop_gui.settings.Settings` – settings persistence

Everything else should be treated as implementation detail.
"""

from __future__ import annotations

from .app import main
from .main_window import MainWindow
from .settings import Settings

__all__ = ["main", "MainWindow", "Settings"]
