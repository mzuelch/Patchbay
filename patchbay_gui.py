"""PATCHBAY GUI launcher (run-from-source).

Usage:
    python patchbay_gui.py

This script exists to make running PATCHBAY from a cloned folder easy, without
installing the package. It forwards to :func:`patchbay_desktop_gui.app.main`.

If you installed PATCHBAY, you can also use the console script:
    patchbay-gui
"""

from __future__ import annotations

from patchbay_desktop_gui.app import main


if __name__ == "__main__":
    raise SystemExit(main())
