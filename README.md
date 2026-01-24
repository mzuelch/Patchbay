# PATCHBAY

PATCHBAY is a **desktop GUI** (DearPyGui) with a small Python backend for running the
**facebook/SAM-Audio** family of models on long audio.

Core features:

- **Chunking** (overlap + reconstruction)
- **Temporal anchors** (span prompting)
- **Run workflow UI** (model selection, runtime settings, progress)
- **AudioFX plugins** applied to **input** and **outputs** with per-stream persistent parameters

## Quickstart

1. Create the conda environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate patchbay
```

2. Start the GUI from the repository root:

```bash
python patchbay_gui.py
```

## Project layout

```
PATCHBAY/
├─ patchbay_backend/          # backend API (chunking, anchors, model run)
├─ patchbay_desktop_gui/      # DearPyGui frontend + plugin runtime
├─ plugins/                   # shipped AudioFX plugins (loaded at runtime)
├─ docs/                      # installation / usage / models / plugins
├─ patchbay_gui.py            # GUI launcher
└─ patchbay_anchor.py         # CLI entrypoint
```

## Documentation

- `docs/INSTALLATION.md` – environment setup and first run
- `docs/USAGE.md` – GUI workflow (Input → Run → Output)
- `docs/MODELS.md` – model discovery, local/online detection
- `docs/PLUGINS.md` – AudioFX plugin API and how to write plugins
- `docs/TROUBLESHOOTING.md` – common issues (CUDA, audio I/O, DearPyGui)

## License

MIT (see `pyproject.toml`).
