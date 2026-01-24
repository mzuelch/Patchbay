# Usage

PATCHBAY has three main tabs:

- **Input & Anchors** – load audio, define anchors, apply AudioFX to input
- **Description & Run** – describe the target sound, choose model/settings, run
- **Output** – review outputs, apply AudioFX to outputs, export audio

## Typical workflow

1. **Load input audio** in *Input & Anchors*.
2. Add **anchors** (optional) to guide the model.
3. Define your **description** (comma-separated tags work well).
4. Pick a **model** and **runtime settings**.
5. Click **Run**.
6. Inspect and optionally post-process results in *Output*.

## Abort / cancellation

Abort stops the current run as soon as the backend reaches a safe cancellation
point (chunk boundaries). The GUI stays responsive.
