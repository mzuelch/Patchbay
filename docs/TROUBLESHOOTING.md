# Troubleshooting

## CUDA / device selection
- `auto` selects CUDA if available, otherwise CPU.
- If CUDA is installed but not detected, verify your PyTorch build.

## Audio I/O
If playback fails, check:
- `sounddevice` availability
- correct output device selection at OS level
- sample rate mismatches (PATCHBAY can resample if enabled)

## Model downloads
If model download fails:
- verify internet / proxy settings
- check HuggingFace cache permissions
