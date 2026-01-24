# Troubleshooting

## GUI opens but audio does not play

- Ensure your Windows audio device is available.
- Try loading a short WAV first to validate basic playback.

## Model download fails

- Check that HuggingFace is reachable from your network.
- If you are offline, PATCHBAY can still run using any models already in the local cache.

## CUDA is not used

- Confirm that your PyTorch build supports CUDA.
- In *Description & Run* set **Device** to `cuda` (or `auto`).

## DearPyGui rendering issues

- Update GPU drivers.
- Try running with a smaller viewport size.
