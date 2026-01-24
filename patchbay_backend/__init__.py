"""patchbay_backend

A small, modular backend package around the `sam_audio` model family
(e.g. `facebook/sam-audio-small`) that supports:

1) Stand-alone CLI operation (chunking, overlap-add reconstruction, optional
   temporal anchors / span prompting).
2) Programmatic use as a backend for a GUI (progress callbacks, clean API).

Design goals
------------
- Keep algorithmic pieces (chunk planning, overlap-add, anchor mapping)
  independent and unit-testable.
- Encapsulate model loading / inference behind a thin adapter.
- Provide a progress reporting interface that works for both console and GUI.
- Avoid module names that shadow common third-party packages.

Typical programmatic usage
--------------------------

>>> from patchbay_backend import Config, run_pipeline
>>> cfg = Config.from_parameters(audio="input.wav", description="speech")
>>> target_path, residual_path = run_pipeline(cfg)

See `README.md` for more examples.
"""

from .config import Config, Anchor
from .pipeline import run_pipeline
from .progress import ConsoleProgress, CallbackProgress, NullProgress, ProgressReporter
from .warnings_ctl import suppress_warnings
from .cancel_token import CancellationToken, CancelledError

__all__ = [
    "Anchor",
    "Config",
    "run_pipeline",
    "ProgressReporter",
    "ConsoleProgress",
    "CallbackProgress",
    "NullProgress",
    "suppress_warnings",
    "CancellationToken",
    "CancelledError",
]
