# Usage

PATCHBAY has three main tabs:

- **Input & Anchors** – load audio, inspect waveform, create anchors and apply AudioFX to the input.
- **Description & Run** – configure description, chunking and runtime settings; run/abort inference.
- **Output** – inspect target/residual outputs, apply AudioFX and export results.

## Typical workflow

1. **Load input audio** in *Input & Anchors*.
2. (Optional) Add **markers** and define **anchors** to guide separation.
3. (Optional) Apply **AudioFX** to the input (plugins).
4. Switch to *Description & Run*:
   - enter a **description** (what to separate)
   - choose **chunking** settings
   - pick a **model** and runtime settings
5. Click **Run**. Use **Abort** if needed.
6. Inspect results in *Output* and export.

## AudioFX

AudioFX are loaded from the `plugins/` directory. Parameters are persisted per stream:

- `input`
- `target`
- `residual`
