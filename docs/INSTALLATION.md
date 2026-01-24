# Installation

## Prerequisites

- Windows (tested in a conda environment)
- An NVIDIA GPU is optional but recommended for faster inference.

## Create the conda environment

From the repository root:

```bash
conda env create -f environment.yml
conda activate patchbay
```

## Run PATCHBAY

Start the GUI:

```bash
python patchbay_gui.py
```

Optional: run the CLI entrypoint (anchors/automation tooling):

```bash
python patchbay_anchor.py --help
```
