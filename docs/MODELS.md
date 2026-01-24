# Models

PATCHBAY populates the model drop-down from the HuggingFace collection:

- `facebook/sam-audio`

Each entry is annotated as:

- **(local)** – present in the local HuggingFace cache
- **(online)** – will be downloaded on first use

## Offline behaviour

If PATCHBAY cannot reach HuggingFace, it falls back to a small built-in list of
commonly used SAM-Audio models.
