"""Configuration model.

The backend is used in two ways:

1) CLI usage: `patchbay_backend.cli` parses argparse arguments and turns them
   into a `Config` instance.
2) GUI usage: the GUI can build a `Config` directly (or via `from_parameters`)
   and pass it into `run_pipeline`.

We keep the `Config` as a plain dataclass (no heavy dependencies), making it
serializable and easy to display in a GUI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Anchor: (type, start_s, end_s)
# type is '+' or '-' according to the SAM-Audio temporal anchors convention.
Anchor = Tuple[str, float, float]

# Supported anchor application strategies for chunking.
#
# 'strict' : apply only anchors intersecting the current chunk (time-accurate, but chunks without
#            overlap receive no temporal guidance).
# 'append' : append ALL anchor audio spans to the *end* of every chunk (example prompting) and merge
#            with strict (time-accurate) anchors for chunks that overlap.
# The following additional strategies are implemented in `patchbay_backend.anchors` and handled by
# the pipeline:
# 'strict_or_append' : strict on overlapping chunks, append (example prompting) only on non-overlapping chunks.
# 'append_nearest'   : append only the K nearest anchors for each chunk (reduces prompt length, improves locality).
# 'append_previous'  : append the most recent anchor before the chunk (fallback: next anchor if none).
# 'prepend_nearest'  : prepend the K nearest anchors before each chunk (then remove the prepended part after inference).
ANCHOR_MODE_CHOICES = (
    "strict",
    "append",
    "strict_or_append",
    "append_nearest",
    "append_previous",
    "prepend_nearest",
)


@dataclass(frozen=True)
class Config:
    """Run configuration for the separation pipeline."""

    # Model + task
    model: str
    audio: str
    description: str

    # Outputs
    out_target: str
    out_residual: str

    # Model behavior
    predict_spans: bool = False
    reranking_candidates: int = 1

    # Compute settings
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    fp16: bool = True

    # Chunking
    max_len_s: Optional[float] = None
    overlap_s: Optional[float] = None

    # Temporal anchors
    anchor_mode: str = "strict"  # see ANCHOR_MODE_CHOICES
    anchors: List[Anchor] = None  # type: ignore[assignment]

    # Audio processing
    no_resample: bool = False

    # Diagnostics
    log_file: Optional[str] = None
    debug: bool = False

    def __post_init__(self):
        # Ensure anchors is always a list.
        if self.anchors is None:
            object.__setattr__(self, "anchors", [])

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_default_outputs(audio_path: str, out_target: Optional[str], out_residual: Optional[str]) -> Tuple[str, str]:
        """Derive default output paths from the input audio file."""
        ap = Path(audio_path)
        stem = ap.stem
        parent = ap.parent if str(ap.parent) else Path(".")
        if out_target is None:
            out_target = str(parent / f"{stem}_target.wav")
        if out_residual is None:
            out_residual = str(parent / f"{stem}_residual.wav")
        return out_target, out_residual

    @classmethod
    def from_cli_args(cls, args) -> "Config":
        """Build a Config from an argparse Namespace.

        This is used by the CLI entrypoint.
        """
        out_target, out_residual = cls._derive_default_outputs(args.audio, args.out_target, args.out_residual)

        # Parse anchors from the CLI representation.
        # We do *not* clip to audio duration here because duration is only
        # known after loading the file.
        from .anchors import parse_anchors_cli

        anchors = parse_anchors_cli(args.anchor)

        # Requested UX:
        # - If the user does not set max_len_s, we process the full file.
        # - If overlap_s is provided without max_len_s, it is ignored.
        max_len_s = args.max_len_s
        overlap_s = args.overlap_s if max_len_s is not None else None

        cfg = cls(
            model=args.model,
            audio=args.audio,
            description=args.description,
            out_target=out_target,
            out_residual=out_residual,
            predict_spans=bool(args.predict_spans),
            reranking_candidates=int(args.reranking_candidates),
            device=str(args.device),
            fp16=bool(args.fp16),
            max_len_s=max_len_s,
            overlap_s=overlap_s,
            anchor_mode=str(args.anchor_mode),
            anchors=anchors,
            no_resample=bool(args.no_resample),
            log_file=args.log_file,
            debug=bool(args.debug),
        )
        cfg.validate_basic()
        return cfg

    @classmethod
    def from_parameters(
        cls,
        *,
        model: str = "facebook/sam-audio-small",
        audio: str,
        description: str,
        out_target: Optional[str] = None,
        out_residual: Optional[str] = None,
        predict_spans: bool = False,
        reranking_candidates: int = 1,
        device: str = "auto",
        fp16: bool = True,
        max_len_s: Optional[float] = None,
        overlap_s: Optional[float] = None,
        anchor_mode: str = "strict",
        anchors: Optional[List[Anchor]] = None,
        no_resample: bool = False,
        log_file: Optional[str] = None,
        debug: bool = False,
    ) -> "Config":
        """Convenience constructor for GUI/backends."""
        out_target2, out_residual2 = cls._derive_default_outputs(audio, out_target, out_residual)

        # Keep behavior consistent with the CLI:
        # - Chunking is enabled only if max_len_s is provided.
        # - overlap_s without max_len_s is ignored.
        if max_len_s is None:
            overlap_s = None
        cfg = cls(
            model=model,
            audio=audio,
            description=description,
            out_target=out_target2,
            out_residual=out_residual2,
            predict_spans=predict_spans,
            reranking_candidates=reranking_candidates,
            device=device,
            fp16=fp16,
            max_len_s=max_len_s,
            overlap_s=overlap_s,
            anchor_mode=anchor_mode,
            anchors=anchors or [],
            no_resample=no_resample,
            log_file=log_file,
            debug=debug,
        )
        cfg.validate_basic()
        return cfg

    # ------------------------------------------------------------------
    # Validation + presentation helpers
    # ------------------------------------------------------------------

    def validate_basic(self) -> None:
        """Validate config values that do not require audio metadata."""
        if self.reranking_candidates < 1:
            raise ValueError("reranking_candidates must be >= 1")

        if self.device not in ("auto", "cuda", "cpu"):
            raise ValueError("device must be one of: auto, cuda, cpu")

        if self.anchor_mode not in ANCHOR_MODE_CHOICES:
            raise ValueError(f"anchor_mode must be one of: {', '.join(ANCHOR_MODE_CHOICES)}")

        # Chunking constraints (partial; full checks happen once sr is known).
        if self.max_len_s is not None and self.max_len_s <= 0:
            raise ValueError("max_len_s must be > 0")
        if self.overlap_s is not None and self.overlap_s < 0:
            raise ValueError("overlap_s must be >= 0")

        # Chunking is enabled only when max_len_s is provided.
        # If overlap_s is set without max_len_s, it is ignored by the pipeline.
        # We therefore do not treat this as an error.

        # Basic anchor shape checks
        for a in self.anchors:
            if len(a) != 3:
                raise ValueError(f"invalid anchor tuple: {a}")
            sign, s, e = a
            if sign not in ("+", "-"):
                raise ValueError(f"anchor type must be '+' or '-': {a}")
            if float(e) <= float(s):
                raise ValueError(f"anchor end must be > start: {a}")

    def summary(self, max_desc: int = 60) -> str:
        """Return a short one-line summary (useful for headers and GUI displays)."""
        desc = self.description.replace("\n", " ").strip()
        if len(desc) > max_desc:
            desc = desc[: max_desc - 3] + "..."
        return (
            f"model={self.model} | audio={self.audio} | desc='{desc}' | device={self.device} | "
            f"fp16={self.fp16} | predict_spans={self.predict_spans} | rerank={self.reranking_candidates} | "
            f"max_len_s={self.max_len_s} | overlap_s={self.overlap_s} | "
            f"anchor_mode={self.anchor_mode} | anchors={len(self.anchors)} | "
            f"out_target={self.out_target} | out_residual={self.out_residual}"
        )