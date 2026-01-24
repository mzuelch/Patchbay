# Installation

PATCHBAY is intended to run reliably inside a pinned conda environment.

## Known-good environment

The file `conda_list_paketliste.txt` is a snapshot of a working environment
(created via `conda list`). Use it as a reference for what is known to work on
your machine.

> Tip: If you want, we can generate an `environment.yml` from that snapshot in a
> follow-up step.

## From source

1. Activate your conda environment.
2. Install any missing Python dependencies (only if needed).
3. Run the GUI:

```bash
python patchbay_gui.py
```

## Optional: install as a package

If you prefer console scripts:

```bash
pip install -e .
patchbay-gui
```

Console scripts defined in `pyproject.toml`:

- `patchbay-gui` – start the GUI
- `patchbay-anchor` – run the backend CLI
