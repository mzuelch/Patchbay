"""Temporal anchor handling (Span Prompting).

SAM-Audio supports temporal anchors (sometimes referred to as span prompting)
that guide the model about *when* the described sound should (or should not)
occur.

In this backend we support two strategies:

1) strict
   - Anchors are applied only to chunks that overlap the anchor time window
   - For each chunk we pass the intersection of the global anchors with the
     chunk window, expressed relative to the chunk start.

2) append
   - For each chunk, we append the raw anchor audio segments to the end of
     the chunk (as an example prompt)
   - Anchors are remapped so they reference the appended tail segments
   - After model inference, the appended samples are removed from the output
     before reconstruction.

The algorithms are intentionally separated from the pipeline orchestrator, so
it is easy to test and easy to extend.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from .config import Anchor


def parse_anchors_cli(anchor_args: Optional[List[List[Any]]]) -> List[Anchor]:
    """Parse `--anchor` CLI arguments into a list of Anchor tuples.

    The argparse configuration uses:
        --anchor TYPE START_S END_S
    with `action='append'` so anchor_args becomes e.g.
        [['+', '1.0', '2.5'], ['-', '5', '6.2']]

    We parse and validate basic shape and numeric conversions here.
    Clipping to audio duration happens later.
    """

    anchors: List[Anchor] = []
    if not anchor_args:
        return anchors

    for idx, triplet in enumerate(anchor_args, start=1):
        if len(triplet) != 3:
            raise ValueError(f"--anchor expects 3 values, got {triplet}")

        sign = str(triplet[0]).strip()
        if sign not in ("+", "-"):
            raise ValueError(f"--anchor[{idx}] type must be '+' or '-', got: {sign}")

        try:
            s = float(triplet[1])
            e = float(triplet[2])
        except Exception:
            raise ValueError(f"--anchor[{idx}] start/end must be floats (seconds): {triplet}")

        if e <= s:
            raise ValueError(f"--anchor[{idx}] end must be > start: {triplet}")

        anchors.append((sign, s, e))

    return anchors


def validate_and_clip_anchors(anchors: List[Anchor], total_dur_s: float) -> List[Anchor]:
    """Clip anchors to the audio duration and validate again.

    Parameters
    ----------
    anchors:
        Global anchors in seconds.
    total_dur_s:
        Total audio duration in seconds.

    Returns
    -------
    List[Anchor]
        Anchors clipped into [0, total_dur_s].

    Raises
    ------
    ValueError
        If an anchor becomes invalid after clipping.
    """

    if total_dur_s <= 0:
        # No valid anchors can exist.
        return []

    out: List[Anchor] = []
    for sign, s, e in anchors:
        s2 = max(0.0, min(float(s), float(total_dur_s)))
        e2 = max(0.0, min(float(e), float(total_dur_s)))
        if e2 <= s2:
            raise ValueError(
                f"Invalid anchor after clipping to [0,{total_dur_s:.3f}]: {(sign, s, e)} -> {(sign, s2, e2)}"
            )
        out.append((sign, s2, e2))
    return out


def strict_chunk_anchors(global_anchors: List[Anchor], chunk_start_s: float, chunk_end_s: float) -> List[Anchor]:
    """Return the intersection of anchors with a chunk, relative to chunk start."""

    chunk: List[Anchor] = []
    for sign, s, e in global_anchors:
        inter_s = max(float(s), float(chunk_start_s))
        inter_e = min(float(e), float(chunk_end_s))
        if inter_e > inter_s:
            chunk.append((sign, inter_s - chunk_start_s, inter_e - chunk_start_s))
    return chunk


def anchors_to_sample_spans(anchors: List[Anchor], sr: int, total_samples: int) -> List[Tuple[str, int, int]]:
    """Convert anchors in seconds to integer sample spans."""

    spans: List[Tuple[str, int, int]] = []
    for sign, s, e in anchors:
        s_idx = int(round(float(s) * sr))
        e_idx = int(round(float(e) * sr))
        s_idx = max(0, min(s_idx, total_samples))
        e_idx = max(0, min(e_idx, total_samples))
        if e_idx > s_idx:
            spans.append((sign, s_idx, e_idx))
    return spans


def prepare_append_payload(wav_2d, sr: int, anchors: List[Anchor]):
    """Create the appended anchor audio and metadata for append-mode.

    Parameters
    ----------
    wav_2d:
        Input audio tensor, shape (1, T) on CPU.
    sr:
        Sample rate.
    anchors:
        Global anchors clipped to the audio duration.

    Returns
    -------
    appended_audio_2d, appended_meta

    appended_audio_2d:
        torch.Tensor shape (1, Ta), concatenation of all anchor segments
        in the user-specified order.
    appended_meta:
        List of (sign, duration_s) for each appended segment.

    If no valid spans exist, returns (None, None).
    """

    import torch

    total_samples = int(wav_2d.shape[-1])
    spans = anchors_to_sample_spans(anchors, sr=sr, total_samples=total_samples)
    if not spans:
        return None, None

    segs = []
    meta: List[Tuple[str, float]] = []

    for sign, s_idx, e_idx in spans:
        seg = wav_2d[:, s_idx:e_idx].contiguous()
        if seg.numel() == 0:
            continue
        segs.append(seg)
        meta.append((sign, float(e_idx - s_idx) / float(sr)))

    if not segs:
        return None, None

    appended = torch.cat(segs, dim=-1).contiguous()
    return appended, meta


def build_append_chunk_anchors(chunk_samples: int, sr: int, appended_meta: List[Tuple[str, float]]) -> List[Anchor]:
    """Build anchors for a chunk-with-appended-tail.

    For append-mode we create anchors that point to the appended segments:
    - the chunk is [0, chunk_samples/sr)
    - appended tail starts at base = chunk_samples/sr

    We return anchors in seconds relative to chunk start.
    """

    base = float(chunk_samples) / float(sr)
    off = 0.0
    out: List[Anchor] = []
    for sign, dur_s in appended_meta:
        out.append((sign, base + off, base + off + float(dur_s)))
        off += float(dur_s)
    return out


# ---------------------------------------------------------------------------
# Extended anchor strategies for chunking
# ---------------------------------------------------------------------------

from dataclasses import dataclass


@dataclass(frozen=True)
class AnchorSegment:
    """Pre-sliced anchor segment information.

    We store both time-domain metadata (seconds) and sample indices so that
    different chunking strategies can quickly select appropriate segments.

    Notes
    -----
    - The actual audio tensor for each segment is kept separately (see
      :func:`build_anchor_segment_pool`), because tensors are not hashable
      and should not be stored in frozen dataclasses by default.
    """

    sign: str
    start_s: float
    end_s: float
    center_s: float
    dur_s: float
    start_idx: int
    end_idx: int


def build_anchor_segment_pool(wav_2d, sr: int, anchors: List[Anchor]) -> Tuple[List[Any], List[AnchorSegment]]:
    """Pre-slice all anchor segments once and return a pool for per-chunk selection.

    Parameters
    ----------
    wav_2d:
        torch.Tensor with shape (1, T) on CPU (same convention used throughout the backend).
    sr:
        Sample rate in Hz.
    anchors:
        Global anchors clipped to the audio duration.

    Returns
    -------
    segments, meta:
        segments:
            List[torch.Tensor], each tensor has shape (1, S_i) and is contiguous.
        meta:
            List[AnchorSegment] with matching indices.
    """
    import torch  # local import to keep module import cheap when anchors are unused

    total_samples = int(wav_2d.shape[-1])
    spans = anchors_to_sample_spans(anchors, sr=sr, total_samples=total_samples)
    segments: List[torch.Tensor] = []
    meta: List[AnchorSegment] = []
    for sign, s_idx, e_idx in spans:
        if e_idx <= s_idx:
            continue
        seg = wav_2d[:, s_idx:e_idx].contiguous()
        if seg.numel() == 0:
            continue
        s_s = float(s_idx) / float(sr)
        e_s = float(e_idx) / float(sr)
        c_s = 0.5 * (s_s + e_s)
        meta.append(
            AnchorSegment(
                sign=str(sign),
                start_s=s_s,
                end_s=e_s,
                center_s=c_s,
                dur_s=float(e_idx - s_idx) / float(sr),
                start_idx=int(s_idx),
                end_idx=int(e_idx),
            )
        )
        segments.append(seg)
    return segments, meta


def _select_anchor_indices_nearest(meta: List[AnchorSegment], chunk_center_s: float, k: int) -> List[int]:
    if not meta or k <= 0:
        return []
    # sort by distance, then by time for stability
    ranked = sorted(
        range(len(meta)),
        key=lambda i: (abs(meta[i].center_s - chunk_center_s), meta[i].start_s),
    )
    sel = ranked[: min(k, len(ranked))]
    # keep chronological order for concatenation
    sel.sort(key=lambda i: meta[i].start_s)
    return sel


def _select_anchor_indices_previous(meta: List[AnchorSegment], chunk_center_s: float, k: int) -> List[int]:
    """Select the most recent anchors before the chunk center (fallback: next anchors)."""
    if not meta or k <= 0:
        return []

    prev = [i for i, m in enumerate(meta) if m.center_s <= chunk_center_s]
    if prev:
        # take up to k most recent ones
        prev.sort(key=lambda i: meta[i].center_s, reverse=True)
        sel = prev[: min(k, len(prev))]
        sel.sort(key=lambda i: meta[i].start_s)
        return sel

    # no previous -> take next anchors
    nxt = list(range(len(meta)))
    nxt.sort(key=lambda i: meta[i].center_s)  # earliest first
    sel = nxt[: min(k, len(nxt))]
    sel.sort(key=lambda i: meta[i].start_s)
    return sel


def select_anchor_indices_for_chunk(
    meta: List[AnchorSegment],
    *,
    chunk_start_s: float,
    chunk_end_s: float,
    strategy: str,
    k: int = 2,
) -> List[int]:
    """Select anchor segments for a specific chunk.

    This is the key building block for strategies that want to provide the
    model with an *example* of the target sound even when the chunk itself
    does not contain the anchor time range.

    Strategies
    ----------
    "nearest":
        Pick the `k` anchors whose center time is closest to the chunk's center time.

    "previous":
        Pick the `k` anchors that occur most recently before the chunk's center time.
        If no anchor exists before the chunk, fall back to the earliest anchors after.

    Returns
    -------
    List[int]
        Indices into the segment pool created by :func:`build_anchor_segment_pool`.
    """
    chunk_center_s = 0.5 * (float(chunk_start_s) + float(chunk_end_s))
    strategy = str(strategy).strip().lower()

    if strategy == "nearest":
        return _select_anchor_indices_nearest(meta, chunk_center_s, int(k))
    if strategy == "previous":
        return _select_anchor_indices_previous(meta, chunk_center_s, int(k))
    raise ValueError(f"Unsupported segment selection strategy: {strategy}")


def build_prompt_from_pool(segments: List[Any], meta: List[AnchorSegment], indices: List[int]):
    """Build a concatenated prompt waveform and its (sign, duration_s) meta.

    Returns (prompt_audio_2d, prompt_meta). If indices is empty, returns (None, None).
    """
    if not indices:
        return None, None
    import torch

    segs = []
    m2: List[Tuple[str, float]] = []
    for i in indices:
        if i < 0 or i >= len(segments):
            continue
        segs.append(segments[i])
        m2.append((meta[i].sign, float(meta[i].dur_s)))
    if not segs:
        return None, None
    prompt = torch.cat(segs, dim=-1).contiguous()
    return prompt, m2


def build_prepend_chunk_anchors(sr: int, prompt_meta: List[Tuple[str, float]]) -> List[Anchor]:
    """Build anchors that reference the *prepended* prompt audio at the chunk start."""
    off = 0.0
    out: List[Anchor] = []
    for sign, dur_s in prompt_meta:
        out.append((sign, off, off + float(dur_s)))
        off += float(dur_s)
    return out


def shift_anchors(anchors: List[Anchor], offset_s: float) -> List[Anchor]:
    """Shift anchors by a constant time offset (seconds)."""
    if not anchors:
        return []
    off = float(offset_s)
    return [(sign, float(s) + off, float(e) + off) for (sign, s, e) in anchors]
