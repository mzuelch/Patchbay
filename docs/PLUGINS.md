# AudioFX Plugins

PATCHBAY loads AudioFX plugins from the `plugins/` directory (and optionally from
additional user plugin directories).

A plugin is a Python module that exposes a plugin class with:

- a stable `plugin_id`
- human-readable `name`
- a list of parameter specifications
- an `apply(audio, sample_rate, params)` method

## Where to place plugins

- Shipped plugins: `PATCHBAY/plugins/`
- Optional user plugins: set `PATCHBAY_AUDIOFX_PLUGIN_DIRS` (semicolon-separated on Windows)

## Minimal skeleton

```python
from patchbay_desktop_gui.plugins_api.base import AudioFXPlugin

class MyFx(AudioFXPlugin):
    plugin_id = "myfx"
    name = "My FX"

    def parameters(self):
        return []

    def apply(self, audio, sample_rate, params):
        return audio
```
