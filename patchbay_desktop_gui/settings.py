"""Backward-compatible settings module.

Settings persistence lives in :mod:`patchbay_desktop_gui.gui.persistence`.
"""

from __future__ import annotations

from .gui.persistence import Settings

__all__ = ["Settings"]
