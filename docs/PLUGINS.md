# AudioFX plugins

PATCHBAY loads AudioFX dynamically from:

1. `./plugins` (shipped plugins)
2. Optional user plugin directories (if configured)

Plugins are regular Python modules that expose a class implementing the
`AudioFXPlugin` interface.

## Plugin API

The runtime API lives in:

- `patchbay_desktop_gui.audiofx.base`
- `patchbay_desktop_gui.audiofx.manager`

A plugin typically defines:

- An `id` (stable identifier)
- A short `name`
- A parameter schema (for UI + persistence)
- A `process(audio, sample_rate, **params)` method

## Writing a plugin

Copy `plugins/_template_gain.py` and adjust:

- `PLUGIN_ID`, `PLUGIN_NAME`
- parameters
- processing function

Restart PATCHBAY and the plugin will appear automatically.
