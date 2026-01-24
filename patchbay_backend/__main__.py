"""Module entrypoint.

Allows running the CLI via:

    python -m patchbay_backend --audio input.wav --description "..."

This is a convenience; the recommended options are:
- `python patchbay_anchor.py.py ...` (stand-alone script)
- `patchbay-anchor ...` after installation
"""

from __future__ import annotations

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
