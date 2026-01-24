"""Application entry point for the DearPyGui GUI.

This module provides the ``main()`` function used by:

- ``python patchbay_gui.py`` (repo-style launcher)
- the optional console script entry point configured in pyproject.toml

The GUI uses an explicit DearPyGui render loop so we can:
- poll the backend worker queue each frame
- update playback playheads smoothly while audio is playing
"""

from __future__ import annotations

from typing import Optional


def main(argv: Optional[list] = None) -> int:
    """Start the desktop GUI.

    Parameters
    ----------
    argv:
        Optional argument list. Currently unused; kept for CLI-style entry points.

    Returns
    -------
    int
        Exit code (0 on normal shutdown).
    """
    import dearpygui.dearpygui as dpg

    from .persistence import Settings
    from .main_window import MainWindow

    dpg.create_context()

    # IMPORTANT (Win11 usability + modal reliability)
    # ----------------------------------------------
    # For a "single-window" desktop experience we want the *viewport* (the
    # native OS window) to be the one and only container the user resizes.
    # The internal DearPyGui main window is then pinned to the viewport and
    # made non-movable/non-collapsible.
    #
    # Additionally, creating the viewport *before* building the UI avoids a
    # class of issues where modal windows ("Settings") do not appear on some
    # DearPyGui builds if they were created while no viewport existed yet.
    # Application name (branding)
    dpg.create_viewport(title="PATCHBAY", width=1200, height=820)

    settings = Settings.load()
    mw = MainWindow(settings)

    # Build UI
    mw.build()

    # Finalize & show
    dpg.setup_dearpygui()
    dpg.show_viewport()

    # Keep the main DPG window synced to viewport size.
    def _on_vp_resize(sender, app_data):
        try:
            mw.on_viewport_resize()
        except Exception:
            pass

    try:
        dpg.set_viewport_resize_callback(_on_vp_resize)
    except Exception:
        pass

    # Apply initial sizing.
    try:
        mw.on_viewport_resize()
    except Exception:
        pass

    # Main loop
    while dpg.is_dearpygui_running():
        try:
            mw.on_frame()
        except Exception:
            # Never let the render loop crash. If something goes wrong, it will
            # be surfaced through the GUI's own error dialog in most cases.
            pass
        dpg.render_dearpygui_frame()

    # Shutdown
    try:
        mw.shutdown()
    except Exception:
        pass

    dpg.destroy_context()
    return 0
