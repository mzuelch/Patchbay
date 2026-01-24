"""Audio FX plugin interface for PATCHBAY.

This module defines a small, stable interface so that new audio post-processing
effects can be added later without touching the GUI code.

Design goals
------------
- **Pure numpy**: effects operate on numpy arrays and do not depend on DearPyGui.
- **Self-describing parameters**: each plugin provides a parameter schema so
  the GUI can build an editor automatically.
- **Safe discovery**: plugins can live in user folders and are imported with
  unique module names to avoid clashes with third-party packages.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


class ParamKind(str, Enum):
    """Supported parameter kinds for auto-generated UIs."""
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    CHOICE = "choice"


@dataclass(frozen=True)
class ParamSpec:
    """Describe one editable parameter.

    Notes
    -----
    - `param_id` must be stable; it is used for persistence (settings JSON).
    - GUI uses the `kind` to decide which input widget to render.
    """
    param_id: str
    label: str
    kind: ParamKind
    default: Any

    # Optional numeric metadata
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    fmt: Optional[str] = None

    # For choice parameters
    choices: Optional[Sequence[str]] = None

    @staticmethod
    def float(
        param_id: str,
        label: str,
        *,
        default: float,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        step: Optional[float] = None,
        fmt: str = "%.3f",
    ) -> "ParamSpec":
        return ParamSpec(param_id, label, ParamKind.FLOAT, float(default), min_value, max_value, step, fmt, None)

    @staticmethod
    def integer(
        param_id: str,
        label: str,
        *,
        default: int,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
        step: Optional[int] = 1,
    ) -> "ParamSpec":
        return ParamSpec(param_id, label, ParamKind.INT, int(default), float(min_value) if min_value is not None else None,
                        float(max_value) if max_value is not None else None, float(step) if step is not None else None, None, None)

    @staticmethod
    def boolean(param_id: str, label: str, *, default: bool) -> "ParamSpec":
        return ParamSpec(param_id, label, ParamKind.BOOL, bool(default))

    @staticmethod
    def choice(param_id: str, label: str, *, default: str, choices: Sequence[str]) -> "ParamSpec":
        if default not in choices and len(choices) > 0:
            default = choices[0]
        return ParamSpec(param_id, label, ParamKind.CHOICE, str(default), choices=tuple(choices))


class AudioEffectPluginBase:
    """Base class for audio effects.

    Plugins must provide a module-level `PLUGIN` variable containing an instance
    of a subclass of this base class.

    The GUI will:
    - display `display_name` in the editor
    - render inputs from `params`
    - store parameter values separately for target and residual outputs
    - call `apply(...)` to transform audio

    Important
    ---------
    `apply()` must be deterministic and must NOT modify its input array in place.
    """

    # Stable identifier used for settings persistence; must be unique.
    plugin_id: str = "base"

    # Human-readable label shown in the GUI.
    display_name: str = "Base Effect"

    # Parameter schema for auto-UI.
    params: List[ParamSpec] = []

    def apply(self, audio, sample_rate: int, params: Dict[str, Any]):
        """Apply the effect and return a new audio array.

        Parameters
        ----------
        audio:
            np.ndarray float PCM, shape (N, C).
        sample_rate:
            Sample rate in Hz.
        params:
            Dict of parameter values keyed by ParamSpec.param_id.

        Returns
        -------
        np.ndarray
            New audio buffer (float32 preferred), shape (N, C).
        """
        raise NotImplementedError
