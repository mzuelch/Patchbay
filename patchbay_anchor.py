"""PATCHBAY backend CLI launcher (run-from-source).

Usage:
    python patchbay_anchor.py --audio input.wav --description "..."

This forwards to :func:`patchbay_backend.cli.main`.

If you installed PATCHBAY, you can also use:
    patchbay-anchor
"""

from __future__ import annotations

from patchbay_backend.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
