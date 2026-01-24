# Models

PATCHBAY aligns the model dropdown with the HuggingFace collection:

- `facebook/sam-audio` collection on HuggingFace

## Online vs. local

The dropdown shows:

- **(local)** if the model is already available in the HuggingFace cache
- **(online)** if the model needs to be downloaded

Local detection is implemented as a best-effort check against the default
HuggingFace cache directories.

## Offline usage

If the collection cannot be fetched (offline), PATCHBAY uses a small built-in
fallback list so the GUI remains usable.
