"""High-level orchestration of the separation pipeline.

This module is the main backend entry point:
    run_pipeline(config, progress=..., logger=...)

It combines:
- warning suppression (best effort)
- audio loading and optional resampling
- anchor validation/clipping
- chunk planning and iteration
- model inference via the adapter in `separator.py`
- overlap-add reconstruction
- deterministic output writing

The pipeline is intentionally structured so that a GUI can:
- pass its own `ProgressReporter` implementation (e.g. progressbar updates)
- choose whether and where to write logs

The pipeline itself does not parse CLI arguments. That is done in `cli.py`.
"""

from __future__ import annotations

import sys
from typing import Optional, Tuple, List

from .config import Config, Anchor
from .progress import ProgressReporter, NullProgress
from .file_logging import FileLogger, log_ram, log_cuda_mem
from .warnings_ctl import suppress_warnings
from .audio_io import load_audio_mono_2d, resample_2d, save_pcm16_wav, audio_duration_seconds
from .chunking import make_chunk_plan, iter_chunks, overlap_add
from .anchors import (
    validate_and_clip_anchors,
    strict_chunk_anchors,
    prepare_append_payload,
    build_append_chunk_anchors,
    build_anchor_segment_pool,
    select_anchor_indices_for_chunk,
    build_prompt_from_pool,
    build_prepend_chunk_anchors,
    shift_anchors,
)
from .separator import SamAudioSeparator
from .cancel_token import CancellationToken


def _log(logger: FileLogger, msg: str) -> None:
    if logger.enabled:
        logger.log_info(msg)


def _log_tensor(logger: FileLogger, name: str, x) -> None:
    """Log tensor shapes at info level when logging is enabled."""
    if not logger.enabled:
        return
    try:
        import torch

        if isinstance(x, torch.Tensor):
            logger.log_info(
                f"{name}: ndim={x.ndim} shape={tuple(x.shape)} numel={x.numel()} dtype={x.dtype} device={x.device}"
            )
        else:
            logger.log_info(f"{name}: type={type(x)}")
    except Exception as e:
        logger.log_warning(f"{name}: (failed to log tensor) {e}")


def run_pipeline(
    cfg: Config,
    *,
    progress: Optional[ProgressReporter] = None,
    logger: Optional[FileLogger] = None,
    cancel_token: Optional[CancellationToken] = None,
) -> Tuple[str, str]:
    """Run the separation pipeline.

    Parameters
    ----------
    cfg:
        Pipeline configuration.
    progress:
        Progress reporter. If None, a NullProgress is used.
        A CLI should pass ConsoleProgress; a GUI typically passes CallbackProgress
        or a custom reporter.
    logger:
        File logger. If None, the logger is constructed from cfg.log_file.
    cancel_token:
        Optional cooperative cancellation token.  If provided, the pipeline
        checks this token at safe points (between chunks, between major steps)
        and aborts by raising ``CancelledError``.

    Returns
    -------
    (out_target_path, out_residual_path)

    Raises
    ------
    Any exception raised by torchaudio / torch / sam_audio is propagated.
    The CLI wrapper catches and formats exceptions for the user.
    """

    # Ensure global warnings are suppressed for both CLI and GUI usage.
    suppress_warnings()

    # Cooperative cancellation is optional.  If not provided, we use a token
    # that is never cancelled.
    cancel_token = cancel_token or CancellationToken()

    progress = progress or NullProgress()
    logger = logger or FileLogger(cfg.log_file, level=cfg.log_level, append=cfg.log_append)

    # Basic config sanity checks (does not depend on audio metadata).
    cfg.validate_basic()

    # First safe point.
    cancel_token.raise_if_cancelled()

    _log(logger, "BOOT: pipeline started")
    _log(logger, f"python: {sys.version.replace(chr(10), ' ')}")
    _log(logger, f"config: {cfg.summary()}")

    # ------------------------------------------------------------------
    # Load processor (to discover model sample rate)
    # ------------------------------------------------------------------
    progress.update("Loading SAM-Audio processor (to read model sample rate)...", percent=5.0)
    cancel_token.raise_if_cancelled()
    separator = SamAudioSeparator(
        model_id=cfg.model,
        device_choice=cfg.device,
        fp16=cfg.fp16,
        logger=logger,
    )
    separator.load_processor()
    cancel_token.raise_if_cancelled()
    model_sr = separator.sample_rate
    _log(logger, f"model_sr={model_sr}")

    # ------------------------------------------------------------------
    # Load audio (CPU) and resample if required
    # ------------------------------------------------------------------
    progress.update("Loading audio file...", percent=10.0)
    cancel_token.raise_if_cancelled()
    wav_2d, sr = load_audio_mono_2d(cfg.audio)
    total_samples = int(wav_2d.shape[-1])
    dur_s = audio_duration_seconds(total_samples, sr)

    _log_tensor(logger, "wav_2d@loaded", wav_2d)
    _log(logger, f"audio: sr={sr} samples={total_samples} duration_s={dur_s:.3f}")

    if not cfg.no_resample and sr != model_sr:
        progress.update(f"Resampling {sr}Hz -> {model_sr}Hz ...", percent=12.0)
        cancel_token.raise_if_cancelled()
        wav_2d, sr = resample_2d(wav_2d, sr, model_sr)
        total_samples = int(wav_2d.shape[-1])
        dur_s = audio_duration_seconds(total_samples, sr)
        _log(logger, f"resampled: sr={sr} samples={total_samples} duration_s={dur_s:.3f}")

    # ------------------------------------------------------------------
    # Anchors: validate + clip to duration
    # ------------------------------------------------------------------
    progress.update("Validating temporal anchors...", percent=15.0)
    cancel_token.raise_if_cancelled()
    global_anchors: List[Anchor] = []
    if cfg.anchors:
        global_anchors = validate_and_clip_anchors(list(cfg.anchors), total_dur_s=dur_s)
        _log(logger, f"anchors: {len(global_anchors)} (mode={cfg.anchor_mode})")
        for i, a in enumerate(global_anchors, start=1):
            _log(logger, f"anchor[{i}]: {a[0]} {a[1]:.3f}s..{a[2]:.3f}s")
    else:
        _log(logger, "anchors: none")

    # ------------------------------------------------------------------
    # Chunk plan
    # ------------------------------------------------------------------
    progress.update("Planning chunking and overlap...", percent=18.0)
    cancel_token.raise_if_cancelled()

    # Requested behavior:
    # - If max_len_s is not set, we always process the entire file in one pass.
    # - If overlap_s is provided without max_len_s, it is silently ignored.
    # We keep this note in the optional log file to help with diagnostics.
    if cfg.max_len_s is None and cfg.overlap_s is not None:
        _log(logger, "note: overlap_s was provided without max_len_s -> ignoring overlap_s and processing full file")

    plan = make_chunk_plan(
        total_samples=total_samples,
        sr=sr,
        duration_s=dur_s,
        max_len_s=cfg.max_len_s,
        overlap_s=cfg.overlap_s,
    )
    _log(logger, f"chunking: enable={plan.enable} max_len_s={plan.max_len_s} overlap_s={plan.overlap_s}"
                 f" chunk_len={plan.chunk_len} overlap={plan.overlap} step={plan.step} num_chunks={plan.num_chunks}")

    # ------------------------------------------------------------------
    # Anchor prompting payload preparation
    # ------------------------------------------------------------------
    #
    # We support multiple anchor strategies (cfg.anchor_mode). Some strategies
    # need additional audio payload prepared upfront:
    #
    # - 'append' and 'strict_or_append' use ONE global tail built from ALL anchors.
    # - 'append_nearest' / 'append_previous' / 'prepend_nearest' select a *subset*
    #   of anchor segments per chunk, so we build a reusable segment pool here.
    #
    append_audio_2d = None
    append_meta = None
    segment_pool = None
    segment_meta = None

    if global_anchors and cfg.anchor_mode in ("append", "strict_or_append"):
        progress.update("Preparing anchor payload (append-based mode)...", percent=22.0)
        cancel_token.raise_if_cancelled()
        append_audio_2d, append_meta = prepare_append_payload(wav_2d, sr=sr, anchors=global_anchors)
        if append_audio_2d is None:
            _log(logger, f"{cfg.anchor_mode}: no valid anchor spans -> append payload disabled")
        else:
            _log(
                logger,
                f"{cfg.anchor_mode}: appended_tail_samples={int(append_audio_2d.shape[-1])}"
                f" tail_duration_s={float(append_audio_2d.shape[-1]) / float(sr):.3f}s"
                f" segments={len(append_meta) if append_meta else 0}",
            )

    if global_anchors and cfg.anchor_mode in ("append_nearest", "append_previous", "prepend_nearest"):
        progress.update("Preparing anchor segment pool (per-chunk selection)...", percent=23.0)
        cancel_token.raise_if_cancelled()
        segment_pool, segment_meta = build_anchor_segment_pool(wav_2d, sr=sr, anchors=global_anchors)
        if not segment_pool:
            _log(logger, f"{cfg.anchor_mode}: no valid anchor spans -> segment pool disabled")
        else:
            _log(logger, f"{cfg.anchor_mode}: segment_pool_size={len(segment_pool)}")
    # ------------------------------------------------------------------
    # Load model (weights)
    # ------------------------------------------------------------------
    # NOTE:
    # This backend is used in two different ways:
    #   1) via the GUI, where the heavy separation work is typically executed
    #      in a *separate subprocess* so GPU memory is released when the worker
    #      exits.
    #   2) as a library, where users may call `run_pipeline(cfg)` directly.
    #
    # In both cases, the pipeline must ensure the model is loaded before
    # processing the first chunk. (Older refactors moved model loading into
    # the subprocess entrypoint which broke direct library usage.)
    progress.update("Loading SAM-Audio model weights...", percent=30.0)
    cancel_token.raise_if_cancelled()
    separator.load_model()
    cancel_token.raise_if_cancelled()

    # ------------------------------------------------------------------
    # Process chunks
    # ------------------------------------------------------------------
    progress.update(f"Starting inference (chunks: {plan.num_chunks})...", percent=40.0)
    cancel_token.raise_if_cancelled()

    target_parts = []
    residual_parts = []

    def chunk_percent(end_sample: int) -> float:
        # Map chunking work to 40..95%.
        if total_samples <= 0:
            return 40.0
        frac = max(0.0, min(1.0, float(end_sample) / float(total_samples)))
        return 40.0 + frac * 55.0

    for chunk_idx, start, end in iter_chunks(total_samples, plan):
        # Safe point: allow cancellation between chunks.
        cancel_token.raise_if_cancelled()
        chunk = wav_2d[:, start:end]
        chunk_samples = int(chunk.shape[-1])
        chunk_dur_s = float(chunk_samples) / float(sr) if sr > 0 else 0.0

        progress.update(
            f"Chunk {chunk_idx}/{plan.num_chunks} | {start}/{total_samples} samples | ~{chunk_dur_s:.2f}s",
            percent=chunk_percent(end),
        )

        _log(logger, f"chunk[{chunk_idx}]: start={start} end={end} samples={chunk_samples}")
        _log_tensor(logger, f"chunk[{chunk_idx}]", chunk)

        # Decide anchors for this chunk.
        #
        # Important background (SAM-Audio "span prompting")
        # -----------------------------------------------
        # SAM-Audio temporal anchors are specified as time ranges where the target
        # sound *is* present ("+") or *is not* present ("-"). This acts like an
        # example signal that tells the model what to isolate and when.
        #
        # When we split the original file into chunks, a strict intersection of
        # anchors with the chunk often results in *no anchors* for chunks that
        # don't overlap the user-provided anchor ranges.
        #
        # The additional anchor modes implemented in this backend therefore add
        # a small "prompt" audio payload (built from anchor spans) either to the
        # end (append-*) or the beginning (prepend-*) of each chunk. After model
        # inference, this extra payload is removed again before overlap-add
        # reconstruction, so the output length matches the original chunk.
        per_chunk_anchors: Optional[List[Anchor]] = None

        # Track per-chunk prompt removal.
        prepend_samples = 0
        append_samples = 0
        effective_chunk_samples_for_output = chunk_samples

        if global_anchors:
            c_start_s = float(start) / float(sr)
            c_end_s = float(end) / float(sr)

            strict_for_chunk = strict_chunk_anchors(global_anchors, c_start_s, c_end_s)
            has_strict = bool(strict_for_chunk)

            mode = str(cfg.anchor_mode)

            if mode == "strict":
                per_chunk_anchors = strict_for_chunk if strict_for_chunk else None

            elif mode == "append":
                # Append ALL anchor spans to the end of every chunk and merge with strict anchors.
                if append_audio_2d is not None and append_meta is not None:
                    import torch

                    chunk = torch.cat([chunk, append_audio_2d], dim=-1)
                    append_samples = int(append_audio_2d.shape[-1])
                    effective_chunk_samples_for_output = int(chunk.shape[-1])

                    appended_for_chunk = build_append_chunk_anchors(
                        chunk_samples,
                        sr=sr,
                        appended_meta=append_meta,
                    )

                    merged: List[Anchor] = []
                    if strict_for_chunk:
                        merged.extend(strict_for_chunk)
                    if appended_for_chunk:
                        merged.extend(appended_for_chunk)

                    per_chunk_anchors = merged if merged else None
                else:
                    per_chunk_anchors = strict_for_chunk if strict_for_chunk else None

            elif mode == "strict_or_append":
                # Strict anchors are time-accurate but only exist for overlapping chunks.
                # This mode keeps overlapping chunks clean (strict only), and only
                # appends examples for chunks that would otherwise have no anchors.
                if has_strict:
                    per_chunk_anchors = strict_for_chunk
                else:
                    if append_audio_2d is not None and append_meta is not None:
                        import torch

                        chunk = torch.cat([chunk, append_audio_2d], dim=-1)
                        append_samples = int(append_audio_2d.shape[-1])
                        effective_chunk_samples_for_output = int(chunk.shape[-1])

                        appended_for_chunk = build_append_chunk_anchors(
                            chunk_samples,
                            sr=sr,
                            appended_meta=append_meta,
                        )
                        per_chunk_anchors = appended_for_chunk if appended_for_chunk else None
                    else:
                        per_chunk_anchors = None

            elif mode in ("append_nearest", "append_previous"):
                # Per-chunk selection: choose only a few relevant anchors to reduce prompt length.
                if segment_pool and segment_meta:
                    sel_strategy = "nearest" if mode == "append_nearest" else "previous"
                    k = 2 if mode == "append_nearest" else 1
                    indices = select_anchor_indices_for_chunk(
                        segment_meta,
                        chunk_start_s=c_start_s,
                        chunk_end_s=c_end_s,
                        strategy=sel_strategy,
                        k=k,
                    )
                    prompt_audio_2d, prompt_meta = build_prompt_from_pool(segment_pool, segment_meta, indices)
                else:
                    prompt_audio_2d, prompt_meta = None, None

                if prompt_audio_2d is not None and prompt_meta is not None:
                    import torch

                    chunk = torch.cat([chunk, prompt_audio_2d], dim=-1)
                    append_samples = int(prompt_audio_2d.shape[-1])
                    effective_chunk_samples_for_output = int(chunk.shape[-1])

                    appended_for_chunk = build_append_chunk_anchors(
                        chunk_samples,
                        sr=sr,
                        appended_meta=prompt_meta,
                    )

                    merged: List[Anchor] = []
                    if strict_for_chunk:
                        merged.extend(strict_for_chunk)
                    if appended_for_chunk:
                        merged.extend(appended_for_chunk)

                    per_chunk_anchors = merged if merged else None
                else:
                    per_chunk_anchors = strict_for_chunk if strict_for_chunk else None

            elif mode == "prepend_nearest":
                # Like append_nearest, but we prepend the prompt before the chunk.
                # This can help when the model is more sensitive to early context.
                if segment_pool and segment_meta:
                    indices = select_anchor_indices_for_chunk(
                        segment_meta,
                        chunk_start_s=c_start_s,
                        chunk_end_s=c_end_s,
                        strategy="nearest",
                        k=2,
                    )
                    prompt_audio_2d, prompt_meta = build_prompt_from_pool(segment_pool, segment_meta, indices)
                else:
                    prompt_audio_2d, prompt_meta = None, None

                if prompt_audio_2d is not None and prompt_meta is not None:
                    import torch

                    prepend_samples = int(prompt_audio_2d.shape[-1])
                    prompt_dur_s = float(prepend_samples) / float(sr)

                    chunk = torch.cat([prompt_audio_2d, chunk], dim=-1)
                    effective_chunk_samples_for_output = int(chunk.shape[-1])

                    prompt_anchors = build_prepend_chunk_anchors(sr=sr, prompt_meta=prompt_meta)
                    strict_shifted = shift_anchors(strict_for_chunk, offset_s=prompt_dur_s) if strict_for_chunk else []

                    merged: List[Anchor] = []
                    if prompt_anchors:
                        merged.extend(prompt_anchors)
                    if strict_shifted:
                        merged.extend(strict_shifted)

                    per_chunk_anchors = merged if merged else None
                else:
                    per_chunk_anchors = strict_for_chunk if strict_for_chunk else None

            else:
                raise ValueError(f"Unsupported anchor_mode: {cfg.anchor_mode}")

        # Run model inference.cancelled()
        t, r = separator.separate_chunk(
            chunk,
            cfg.description,
            per_chunk_anchors,
            predict_spans=cfg.predict_spans,
            reranking_candidates=cfg.reranking_candidates,
        )

        # Cut to effective length (in case the model returns padded output).
        if t.numel() > effective_chunk_samples_for_output:
            t = t[:effective_chunk_samples_for_output]
        if r.numel() > effective_chunk_samples_for_output:
            r = r[:effective_chunk_samples_for_output]

        # Remove per-chunk prompt payload again before reconstruction.
        #
        # - append_* modes: remove tail
        # - prepend_* modes: remove head
        if prepend_samples > 0:
            t = t[prepend_samples: prepend_samples + chunk_samples]
            r = r[prepend_samples: prepend_samples + chunk_samples]
        elif append_samples > 0:
            t = t[:chunk_samples]
            r = r[:chunk_samples]

        target_parts.append(t)
        residual_parts.append(r)

        # Another safe point: allow cancellation after finishing a chunk.
        cancel_token.raise_if_cancelled()

        _log_tensor(logger, f"chunk[{chunk_idx}].target", t)
        _log_tensor(logger, f"chunk[{chunk_idx}].residual", r)

    # ------------------------------------------------------------------
    # Reconstruction
    # ------------------------------------------------------------------
    progress.update("Reconstructing full-length outputs (overlap-add)...", percent=96.0)
    cancel_token.raise_if_cancelled()

    if plan.enable and plan.overlap > 0:
        target_full = overlap_add(target_parts, plan.overlap)
        residual_full = overlap_add(residual_parts, plan.overlap)
    else:
        import torch

        target_full = torch.cat(target_parts, dim=0) if len(target_parts) > 1 else target_parts[0]
        residual_full = torch.cat(residual_parts, dim=0) if len(residual_parts) > 1 else residual_parts[0]

    # Ensure exact length.
    target_full = target_full[:total_samples]
    residual_full = residual_full[:total_samples]

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    progress.update("Writing output WAV files (PCM16)...", percent=99.0)
    cancel_token.raise_if_cancelled()
    save_pcm16_wav(cfg.out_target, target_full, sr)
    save_pcm16_wav(cfg.out_residual, residual_full, sr)

    _log(logger, f"saved: target={cfg.out_target}")
    _log(logger, f"saved: residual={cfg.out_residual}")

    progress.done(f"Done. target='{cfg.out_target}' | residual='{cfg.out_residual}'")

    log_ram(logger, "RAM@done")
    log_cuda_mem(logger, "CUDA@done")

    return cfg.out_target, cfg.out_residual
