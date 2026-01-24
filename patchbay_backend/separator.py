"""Model adapter (SAM-Audio) used by the pipeline.

This module isolates all model-specific details:
- loading the processor and model
- moving tensors to the desired device
- enabling mixed precision (FP16) on CUDA
- calling the model's `separate(...)` method
- converting outputs back to CPU float32

By keeping this logic encapsulated, the rest of the backend (chunking,
anchors, reconstruction, ...) remains independent and easier to maintain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

from .config import Anchor
from .file_logging import FileLogger


@dataclass(frozen=True)
class DeviceSpec:
    """Resolved compute device.

    Attributes
    ----------
    device:
        torch.device instance.
    use_fp16:
        Whether FP16 autocast / model weights should be used.
    """

    device: "object"  # torch.device
    use_fp16: bool


def resolve_device(device_choice: str, fp16: bool, logger: Optional[FileLogger] = None) -> DeviceSpec:
    """Resolve the requested device string to a torch.device.

    Parameters
    ----------
    device_choice:
        'auto' | 'cuda' | 'cpu'
    fp16:
        Whether FP16 is allowed. Only used when CUDA is available.
    """

    import torch

    if device_choice == "cpu":
        dev = torch.device("cpu")
    elif device_choice == "cuda":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dev.type != "cuda" and logger and logger.enabled:
            logger.log("WARN: device='cuda' requested but CUDA not available -> falling back to CPU")
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_fp16 = bool(fp16 and dev.type == "cuda")
    return DeviceSpec(device=dev, use_fp16=use_fp16)


class SamAudioSeparator:
    """Thin adapter around SAM-Audio.

    Lifecycle
    ---------
    1) Instantiate the adapter.
    2) Call `load_processor()` to obtain the model sample rate.
    3) Call `load_model()` once you are ready to run inference.
    4) Call `separate_chunk(...)` for each chunk.

    The separation method returns CPU float32 tensors.
    """

    def __init__(
        self,
        model_id: str,
        device_choice: str = "auto",
        fp16: bool = True,
        logger: Optional[FileLogger] = None,
    ):
        self.model_id = model_id
        self.logger = logger or FileLogger(None)

        self.device_spec = resolve_device(device_choice, fp16, logger=self.logger)

        self.processor = None
        self.model = None

    def load_processor(self) -> None:
        """Load the SAM-Audio processor."""
        from sam_audio import SAMAudioProcessor

        self.processor = SAMAudioProcessor.from_pretrained(self.model_id)
        if self.logger.enabled:
            self.logger.log(f"processor loaded | audio_sampling_rate={int(self.processor.audio_sampling_rate)}")

    @property
    def sample_rate(self) -> int:
        """Return the processor's audio sampling rate."""
        if self.processor is None:
            raise RuntimeError("Processor not loaded. Call load_processor() first.")
        return int(self.processor.audio_sampling_rate)

    def load_model(self) -> None:
        """Load and place the SAM-Audio model on the resolved device."""
        from sam_audio import SAMAudio

        import torch

        self.model = SAMAudio.from_pretrained(self.model_id)

        if self.device_spec.use_fp16:
            self.model = self.model.to(device=self.device_spec.device, dtype=torch.float16).eval()
        else:
            self.model = self.model.to(device=self.device_spec.device).eval()

        if self.logger.enabled:
            self.logger.log(
                f"model loaded | device={self.device_spec.device} | fp16={self.device_spec.use_fp16}"
            )

    def separate_chunk(
        self,
        audio_2d,
        description: str,
        anchors: Optional[List[Anchor]],
        *,
        predict_spans: bool,
        reranking_candidates: int,
    ) -> Tuple["object", "object"]:
        """Run separation on a single chunk.

        Parameters
        ----------
        audio_2d:
            torch.Tensor shape (1, T) on CPU.
        description:
            Text prompt.
        anchors:
            Optional list of Anchor tuples in seconds relative to chunk start.
            If None or empty, no anchors are passed.
        predict_spans, reranking_candidates:
            Model parameters.

        Returns
        -------
        target_1d_cpu, residual_1d_cpu
            torch.Tensor 1D float32 on CPU.
        """

        if self.processor is None:
            raise RuntimeError("Processor not loaded. Call load_processor() first.")
        # Robustness: the high-level pipeline loads the model once before
        # processing chunks. However, the separator can also be used
        # standalone (or older code paths might call `separate_chunk` without
        # an explicit `load_model()`), so we lazily load here as a safe fallback.
        if self.model is None:
            if self.logger.enabled:
                self.logger.log("WARN: model not loaded at first inference -> calling load_model() lazily")
            self.load_model()

        import torch

        # Build processor inputs. The processor expects a batch.
        kwargs = {
            "audios": [audio_2d],
            "descriptions": [description],
        }
        if anchors:
            kwargs["anchors"] = [anchors]

        inputs = self.processor(**kwargs).to(self.device_spec.device)

        # Run inference.
        with torch.inference_mode():
            if self.device_spec.use_fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    result = self.model.separate(
                        inputs,
                        predict_spans=bool(predict_spans),
                        reranking_candidates=int(reranking_candidates),
                    )
            else:
                result = self.model.separate(
                    inputs,
                    predict_spans=bool(predict_spans),
                    reranking_candidates=int(reranking_candidates),
                )

        # Move outputs to CPU float32.
        t = result.target[0].detach().cpu().to(torch.float32).squeeze()
        r = result.residual[0].detach().cpu().to(torch.float32).squeeze()

        if t.ndim != 1 or r.ndim != 1:
            raise RuntimeError(f"Unexpected output dims: target.ndim={t.ndim}, residual.ndim={r.ndim}")

        # Cleanup (important for long runs).
        del inputs, result
        if self.device_spec.device.type == "cuda":
            torch.cuda.empty_cache()

        return t, r
