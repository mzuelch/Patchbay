"""Audio FX plugin discovery and management."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .base import AudioEffectPluginBase


@dataclass
class PluginLoadError:
    path: str
    message: str


def _unique_module_name(prefix: str = "patchbay_audiofx_plugin") -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _iter_plugin_files(dirs: Sequence[Path]) -> Iterable[Path]:
    for d in dirs:
        try:
            if not d.exists() or not d.is_dir():
                continue
            for p in sorted(d.glob("*.py")):
                if p.name.startswith("_"):
                    continue
                yield p
        except Exception:
            continue


def _load_plugin_from_file(path: Path) -> Tuple[Optional[AudioEffectPluginBase], Optional[str]]:
    """Load a plugin from a single .py file.

    Returns (plugin_instance, error_message).
    """
    try:
        mod_name = _unique_module_name()
        spec = importlib.util.spec_from_file_location(mod_name, str(path))
        if spec is None or spec.loader is None:
            return None, "import spec loader not available"
        module = importlib.util.module_from_spec(spec)
        # Ensure the module name is unique and does not shadow existing packages.
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        plugin = getattr(module, "PLUGIN", None)
        if plugin is None:
            return None, "module has no PLUGIN variable"
        if not isinstance(plugin, AudioEffectPluginBase):
            return None, "PLUGIN is not an AudioEffectPluginBase instance"
        return plugin, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


class AudioFxManager:
    """Discover and hold audio effect plugins."""

    def __init__(self) -> None:
        self.plugins: Dict[str, AudioEffectPluginBase] = {}
        self.load_errors: List[PluginLoadError] = []

    def register(self, plugin: AudioEffectPluginBase, *, source: str = "") -> None:
        pid = getattr(plugin, "plugin_id", None)
        if not pid or not isinstance(pid, str):
            raise ValueError("plugin has no valid plugin_id")
        if pid in self.plugins:
            # Keep the first one; record error for duplicates.
            self.load_errors.append(PluginLoadError(source or pid, f"duplicate plugin_id '{pid}'"))
            return
        self.plugins[pid] = plugin

    def discover(self, *, user_dirs: Sequence[Path]) -> None:
        """Discover Audio FX plugins from the configured plugin directories.

        Notes
        -----
        Effects shipped with the project are regular file-based plugins located
        in the `plugins/` folder (beside the project/package root). They are
        discovered the same way as user-provided plugins.
        """
        self.plugins.clear()
        self.load_errors.clear()

        # Discover plugins from directories (files starting with "_" are ignored)
        for p in _iter_plugin_files(user_dirs):
            plugin, err = _load_plugin_from_file(p)
            if plugin is None:
                self.load_errors.append(PluginLoadError(str(p), err or "unknown error"))
                continue
            self.register(plugin, source=str(p))


    def list_plugins(self) -> List[AudioEffectPluginBase]:
        """Return plugins ordered by display name (stable)."""
        return sorted(self.plugins.values(), key=lambda pl: (pl.display_name.lower(), pl.plugin_id))
