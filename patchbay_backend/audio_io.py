"""Audio I/O and lightweight preprocessing.

This module is responsible for:
- loading audio from disk
- ensuring a canonical tensor format
- resampling to the model's expected sample rate (optional)
- saving output WAV files in a deterministic PCM16 format

All functions operate on CPU tensors. The model inference code can move
inputs to GPU as needed.
"""

from __future__ import annotations

from typing import Tuple


def load_audio_mono_2d(audio_path: str):
    """Load audio via torchaudio and return a mono tensor of shape (1, T).

    Returns
    -------
    wav_2d:
        torch.Tensor on CPU, dtype float32, shape (1, T)
    sr:
        Source sample rate

    Notes
    -----
    - Multi-channel audio is downmixed by simple average.
    - We keep the (1, T) shape to match common processor expectations.
    """
    import torchaudio

    wav, sr = torchaudio.load(audio_path)
    if wav is None:
        raise RuntimeError("torchaudio.load returned None")

    # Normalize to 2D (C, T)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)

    if wav.ndim != 2:
        raise RuntimeError(f"Unexpected wav.ndim={wav.ndim} (expected 1 or 2)")

    # Downmix to mono (keep 2D)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    wav = wav.contiguous().float().cpu()
    return wav, int(sr)


def resample_2d(wav_2d, sr_in: int, sr_out: int):
    """Resample a (1, T) tensor from sr_in to sr_out using torchaudio."""
    import torchaudio

    if sr_in == sr_out:
        return wav_2d, sr_in

    wav_2d = torchaudio.functional.resample(wav_2d, sr_in, sr_out)
    return wav_2d, sr_out


def save_pcm16_wav(path: str, wav_1d_float, sr: int) -> None:
    """Save a 1D float waveform as PCM 16-bit WAV.

    Parameters
    ----------
    path:
        Output file path.
    wav_1d_float:
        torch.Tensor 1D float32 on CPU. Values are clamped to [-1, 1].
    sr:
        Sample rate.

    Why PCM16?
    ----------
    PCM16 is widely supported by audio tools and avoids floating point WAV
    quirks across different toolchains.
    """
    import torch
    import torchaudio

    wav_1d_float = wav_1d_float.detach().cpu().to(torch.float32).clamp(-1, 1)
    wav_i16 = (wav_1d_float * 32767.0).round().to(torch.int16).unsqueeze(0)  # (1, T)

    torchaudio.save(path, wav_i16, int(sr), encoding="PCM_S", bits_per_sample=16)


def audio_duration_seconds(total_samples: int, sr: int) -> float:
    """Return duration in seconds for convenience."""
    if sr <= 0:
        return 0.0
    return float(total_samples) / float(sr)
