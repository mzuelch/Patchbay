"""Template Audio FX plugin for PATCHBAY.

How to use
----------
1) Copy this file to your plugin directory and rename it to e.g. `gain.py`.
   Supported plugin directories:
   - %APPDATA%\CatSynth\PATCHBAY\plugins
   - ./plugins (current working directory)
   - additional dirs in env var PATCHBAY_AUDIOFX_PLUGIN_DIRS (semicolon separated)
   (legacy: SAMAUDIO_AUDIOFX_PLUGIN_DIRS is also accepted)

2) Remove the leading underscore from the filename if you keep it in ./plugins.
   Files starting with "_" are ignored by the auto-discovery.

3) Restart the GUI. The plugin will appear in the Output processing panel for
   both Target and Residual, with parameters stored separately per output.

Notes
-----
- Plugins must export a module-level variable named `PLUGIN`.
- `plugin_id` must be unique across all loaded plugins.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from patchbay_desktop_gui.audiofx.base import AudioEffectPluginBase, ParamSpec


class GainPlugin(AudioEffectPluginBase):
    plugin_id = "gain"
    display_name = "Gain (Template)"
    params = [
        ParamSpec.float("gain_db", "Gain (dB)", default=0.0, min_value=-24.0, max_value=24.0, step=0.5, fmt="%.1f"),
    ]

    def apply(self, audio: np.ndarray, sample_rate: int, params: Dict[str, Any]) -> np.ndarray:
        g_db = float(params.get("gain_db", 0.0))
        g = 10.0 ** (g_db / 20.0)
        # Keep float32, do not clip automatically.
        return (audio.astype(np.float32, copy=False) * np.float32(g)).astype(np.float32, copy=False)


PLUGIN = GainPlugin()
