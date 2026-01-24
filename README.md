# PATCHBAY

PATCHBAY is a **desktop GUI** (DearPyGui) and a small **Python backend** for running
the **facebook/SAM-Audio** family of models on long audio via:

- **Chunking** (overlap + crossfade reconstruction)
- **Temporal anchors** (span prompting / example snippets)
- A **Run** workflow UI (models, runtime settings, progress)
- **AudioFX plugins** (normalize, compressor, filters, declicker, …) applied to
  input/outputs with persistent per-stream parameters

PATCHBAY is designed to run well from source inside a pinned conda environment.

## Quickstart (from source)

1. Create/activate your known-good conda environment (see `docs/conda_list_paketliste.txt`).
2. From the repository root:

```bash
python patchbay_gui.py
```

> Legacy launchers `patchbay_gui.py` / `patchbay_anchor.py` are kept as thin
> wrappers for compatibility.

## Key concepts

### Chunking
Long audio is processed in overlapping chunks (`max_len_s`, `overlap_s`) and then
reconstructed using a linear crossfade.

### Anchors
Anchors are short time spans in the original audio that provide examples
(“span prompting”) to guide separation. PATCHBAY supports multiple *anchor modes*
(e.g. *strict* vs. *append/prepend* variants) depending on your workflow.

### Models
PATCHBAY discovers available models from the HuggingFace collection:

- `https://huggingface.co/collections/facebook/sam-audio`

The model selector shows whether each model is already available **(local)** or
needs to be fetched **(online)**.

### AudioFX plugins
AudioFX are loaded dynamically from the `plugins/` folder (and optionally from a
user plugin directory). There is **no “builtin” AudioFX folder** anymore—every
effect is a plugin.

## Project layout

```
PATCHBAY/
├─ patchbay_backend/          # backend API + CLI (chunking, anchors, model run)
├─ patchbay_desktop_gui/      # DearPyGui frontend + plugin runtime API
├─ plugins/                   # shipped AudioFX plugins (loaded at runtime)
├─ docs/                      # documentation (install/usage/plugins/models)
├─ tests/                     # unit tests for core algorithms
├─ patchbay_gui.py            # GUI launcher (from source)
└─ patchbay_anchor.py         # CLI entrypoint (from source)
```

## Documentation

- `docs/INSTALLATION.md` – environment setup and first run
- `docs/USAGE.md` – GUI workflow (Input → Run → Output)
- `docs/MODELS.md` – model discovery, local/online detection
- `docs/PLUGINS.md` – AudioFX plugin API and how to write plugins
- `docs/TROUBLESHOOTING.md` – common issues (CUDA, audio I/O, DearPyGui)

## License
MIT (see `pyproject.toml`).
