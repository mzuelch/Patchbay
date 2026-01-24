"""Warning and verbosity suppression utilities.

Why this module exists
----------------------
When running heavy ML pipelines, many libraries emit warnings that are not
actionable for end-users (FutureWarnings, deprecation notices, etc.). In a
CLI these warnings clutter the output; in a GUI they often surface as noisy
popups/log panels.

This backend therefore suppresses warnings by default and tries to reduce
verbosity of the Hugging Face / transformers logging system.

Notes
-----
- The suppression is *best-effort*. Some libraries print directly to stderr.
- The function is safe to call multiple times.
"""

from __future__ import annotations

import os
import warnings


def suppress_warnings() -> None:
    """Suppress common Python warnings and reduce HF/transformers verbosity.

    This function should be called as early as possible in both CLI and GUI
    entrypoints.
    """

    # 1) Python warnings
    # ---------------------------------------------------------------------
    # Suppress all warnings by default as requested.
    warnings.filterwarnings("ignore")
    warnings.simplefilter("ignore")

    # 2) Environment knobs used by Hugging Face tooling
    # ---------------------------------------------------------------------
    # Avoid noisy tokenizers warnings about parallelism.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Prevent HF hub progress bars.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    # Ask transformers to use error-only verbosity.
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    # 3) transformers logging (best-effort)
    # ---------------------------------------------------------------------
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        try:
            hf_logging.disable_progress_bar()
        except Exception:
            # Older transformers versions may not provide this.
            pass
    except Exception:
        # transformers might not be installed in minimal environments.
        pass
