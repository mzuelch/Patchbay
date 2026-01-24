"""Tab builder for PATCHBAY (DearPyGui frontend).

This module contains the UI construction code for the "output" tab.
The functions operate on the MainWindow instance and are intentionally
kept free of application logic beyond UI composition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..main_window import MainWindow

def build_tab_output(self: "MainWindow") -> None:
    """Compose the "Output" tab.
    
    Parameters
    ----------
    self:
        MainWindow instance that owns the shared GUI state.
    """
    dpg = self.dpg
    # ------------------------------------------------------------------
    # Target output
    # ------------------------------------------------------------------
    dpg.add_text("Target")

    # Two-column layout:
    #   - Left: file path + history + transport + waveform
    #   - Right: collapsible processing panel (Normalize / Compressor)
    with dpg.table(
        header_row=False,
        resizable=False,
        policy=dpg.mvTable_SizingStretchProp,
        borders_innerV=True,
        borders_outerV=False,
        borders_innerH=False,
        borders_outerH=False,
    ):
        # Use proportional sizing so the processing panel adapts to
        # the currently available viewport width (no hard-coded px).
        # In StretchProp mode, `init_width_or_weight` is interpreted
        # as a *weight*.
        dpg.add_table_column(init_width_or_weight=0.72)
        dpg.add_table_column(init_width_or_weight=0.28)

        with dpg.table_row():
            # Left column (main output display)
            with dpg.group():
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="Save Target As...",
                        callback=lambda s, a, ud=None: self._show_dialog("dlg_save_target"),
                    )
                    self.tags["target_path"] = dpg.add_text("")

                # Transport bar + waveform
                with dpg.group() as tg:
                    # Waveform (fixed 0 dBFS full-scale), then transport controls underneath.
                    self.wave_target.build(parent=tg, height=240)
                    self._build_waveform_controls_generic(kind="target")

                # Cache tags used by other parts of the UI (seek / hit tests).
                self.tags["target_canvas"] = self.wave_target.get_canvas_tag()
                self.tags["target_draw"] = self.wave_target.get_draw_tag()

            # Right column (processing panel)
            with dpg.group():
                dpg.add_text("AudioFX")
                with dpg.child_window(border=True, height=250, width=-1):
                    self._build_output_processing_panel(kind="target")
                dpg.add_separator()
                self._build_output_history_controls(kind="target")

    dpg.add_separator()

    # ------------------------------------------------------------------
    # Residual output
    # ------------------------------------------------------------------
    dpg.add_text("Residual")
    with dpg.table(
        header_row=False,
        resizable=False,
        policy=dpg.mvTable_SizingStretchProp,
        borders_innerV=True,
        borders_outerV=False,
        borders_innerH=False,
        borders_outerH=False,
    ):
        dpg.add_table_column(init_width_or_weight=0.72)
        dpg.add_table_column(init_width_or_weight=0.28)

        with dpg.table_row():
            with dpg.group():
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="Save Residual As...",
                        callback=lambda s, a, ud=None: self._show_dialog("dlg_save_residual"),
                    )
                    self.tags["residual_path"] = dpg.add_text("")

                with dpg.group() as rg:
                    # Waveform (fixed 0 dBFS full-scale), then transport controls underneath.
                    self.wave_residual.build(parent=rg, height=240)
                    self._build_waveform_controls_generic(kind="residual")

                self.tags["residual_canvas"] = self.wave_residual.get_canvas_tag()
                self.tags["residual_draw"] = self.wave_residual.get_draw_tag()

            with dpg.group():
                dpg.add_text("AudioFX")
                with dpg.child_window(border=True, height=250, width=-1):
                    self._build_output_processing_panel(kind="residual")
                dpg.add_separator()
                self._build_output_history_controls(kind="residual")

