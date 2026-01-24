"""Chunk planning and overlap-add reconstruction.

Chunking is useful for long audio inputs that exceed a model's comfortable
context window or to reduce memory usage.

This module contains pure, testable logic:
- deciding whether to chunk and which parameters to use
- converting seconds to samples
- iterating over chunk windows
- overlap-add reconstruction with a simple linear crossfade

No model-specific code lives here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class ChunkPlan:
    """A precomputed chunk plan.

    Attributes
    ----------
    enable:
        Whether chunking is enabled.
    chunk_len:
        Chunk length in samples.
    overlap:
        Overlap length in samples.
    step:
        Step size between consecutive chunk starts: chunk_len - overlap.
    num_chunks:
        Expected number of chunks.
    max_len_s:
        Effective max length in seconds (after applying defaults).
    overlap_s:
        Effective overlap in seconds (after applying defaults).
    """

    enable: bool
    chunk_len: int
    overlap: int
    step: int
    num_chunks: int
    max_len_s: float
    overlap_s: float


def decide_chunking(
    *,
    duration_s: float,
    max_len_s: Optional[float],
    overlap_s: Optional[float],
    auto_threshold_s: float = 30.0,
    auto_max_len_s: float = 15.0,
    auto_overlap_s: float = 2.0,
) -> Tuple[bool, float, float]:
    """Decide whether chunking should be enabled and return effective parameters.

    **Current default behavior (requested change):**

    - If the user does *not* provide ``max_len_s`` (CLI: ``--max-len-s``), the full
      audio file is processed in a single pass, regardless of its duration.
    - If the user provides ``max_len_s``, chunking is enabled.
      ``overlap_s`` is optional; if omitted we apply a conservative default
      (2 seconds) to enable overlap-add reconstruction.

    Notes
    -----
    The arguments ``auto_threshold_s`` and ``auto_*`` are retained for backward
    compatibility with older versions of this backend, but *are not used* to
    auto-enable chunking anymore.

    Returns
    -------
    enable, eff_max_len_s, eff_overlap_s

    - When ``enable`` is False, the returned max length is the input duration and
      overlap is 0.
    - When ``enable`` is True, the returned parameters represent the effective
      chunking settings.
    """

    # Chunking is enabled *only* when max_len_s is explicitly provided.
    # If callers provide overlap_s without max_len_s (CLI: --overlap-s without
    # --max-len-s), we silently ignore overlap_s as requested.
    if max_len_s is None:
        return False, max(duration_s, 0.0), 0.0

    eff_max = float(max_len_s)
    eff_ov = float(overlap_s) if overlap_s is not None else float(auto_overlap_s)
    return True, eff_max, eff_ov


def make_chunk_plan(
    *,
    total_samples: int,
    sr: int,
    duration_s: float,
    max_len_s: Optional[float],
    overlap_s: Optional[float],
) -> ChunkPlan:
    """Create a ChunkPlan in sample units.

    Full validation happens here once sr and total_samples are known.
    """

    if sr <= 0:
        raise ValueError("Sample rate must be > 0")
    if total_samples < 0:
        raise ValueError("total_samples must be >= 0")

    enable, eff_max_len_s, eff_overlap_s = decide_chunking(
        duration_s=duration_s,
        max_len_s=max_len_s,
        overlap_s=overlap_s,
    )

    if not enable:
        # Dummy plan: single chunk
        return ChunkPlan(
            enable=False,
            chunk_len=max(0, int(total_samples)),
            overlap=0,
            step=max(0, int(total_samples)),
            num_chunks=1,
            max_len_s=float(eff_max_len_s),
            overlap_s=0.0,
        )

    if eff_max_len_s <= 0:
        raise ValueError("max_len_s must be > 0 when chunking is enabled")
    if eff_overlap_s < 0:
        raise ValueError("overlap_s must be >= 0")
    if eff_overlap_s >= eff_max_len_s:
        raise ValueError("overlap_s must be smaller than max_len_s")

    chunk_len = int(round(eff_max_len_s * sr))
    overlap = int(round(eff_overlap_s * sr))
    step = chunk_len - overlap

    if chunk_len <= 0:
        raise ValueError("Chunk length in samples must be > 0")
    if step <= 0:
        raise ValueError("Invalid chunk parameters: step <= 0")

    if total_samples == 0:
        num_chunks = 1
    else:
        # Expected chunk count (same formula used in your monolithic script).
        num_chunks = 1 + max(0, (max(0, total_samples - chunk_len) + step - 1) // step)

    return ChunkPlan(
        enable=True,
        chunk_len=chunk_len,
        overlap=overlap,
        step=step,
        num_chunks=int(num_chunks),
        max_len_s=float(eff_max_len_s),
        overlap_s=float(eff_overlap_s),
    )


def iter_chunks(total_samples: int, plan: ChunkPlan) -> Iterable[Tuple[int, int, int]]:
    """Iterate over chunk windows.

    Yields
    ------
    (chunk_index_1based, start_sample, end_sample)
    """

    if total_samples <= 0:
        yield 1, 0, 0
        return

    if not plan.enable:
        yield 1, 0, total_samples
        return

    idx = 0
    start = 0
    while start < total_samples:
        end = min(start + plan.chunk_len, total_samples)
        if end <= start:
            break
        idx += 1
        yield idx, start, end
        if end >= total_samples:
            break
        start += plan.step


def _linear_fade(n: int, fade_in: bool):
    """Create a linear fade ramp (torch tensor) on CPU."""
    import torch

    if n <= 0:
        return None
    ramp = torch.linspace(0.0, 1.0, steps=n, dtype=torch.float32)
    return ramp if fade_in else (1.0 - ramp)


def overlap_add(parts: List, overlap_samples: int):
    """Overlap-add reconstruction with a linear crossfade.

    Parameters
    ----------
    parts:
        List of 1D float tensors (CPU) in time order.
    overlap_samples:
        Overlap length in samples.

    Returns
    -------
    1D float tensor (CPU)

    Notes
    -----
    This is deliberately simple and deterministic. If you later want a
    higher-quality windowing strategy, swap this implementation without
    touching the rest of the pipeline.
    """

    import torch

    if not parts:
        return torch.empty(0, dtype=torch.float32)

    if overlap_samples <= 0 or len(parts) == 1:
        return torch.cat(parts, dim=0)

    out_parts = [parts[0].clone()]

    for i in range(1, len(parts)):
        prev = out_parts[-1]
        cur = parts[i]

        ov = min(overlap_samples, prev.numel(), cur.numel())
        if ov > 0:
            fi = _linear_fade(ov, fade_in=True)
            fo = _linear_fade(ov, fade_in=False)
            prev_tail = prev[-ov:] * fo + cur[:ov] * fi
            out_parts[-1] = torch.cat([prev[:-ov], prev_tail], dim=0)
            out_parts.append(cur[ov:])
        else:
            out_parts.append(cur)

    return torch.cat(out_parts, dim=0)
