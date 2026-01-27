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
    with dpg.child_window(
        tag="output_target_panel",
        border=False,
        width=-1,
        height=300,
        no_scrollbar=True,
        no_scroll_with_mouse=True,
    ):
        self.tags["output_target_panel"] = "output_target_panel"
        self.tags["target_title"] = dpg.add_text("Target")

        with dpg.group(horizontal=True):
            # Left column (main output display)
            with dpg.child_window(
                tag="target_left_panel",
                border=False,
                width=-1,
                height=-1,
                no_scrollbar=True,
                no_scroll_with_mouse=True,
            ):
                self.tags["target_left_panel"] = "target_left_panel"
                self.tags["target_save_group"] = dpg.add_group(horizontal=True, tag="target_save_group")
                dpg.add_button(
                    label="Save Target As...",
                    callback=lambda s, a, ud=None: self._show_dialog("dlg_save_target"),
                    parent=self.tags["target_save_group"],
                )
                self.tags["target_path"] = dpg.add_text("", parent=self.tags["target_save_group"])

                # Transport bar + waveform
                with dpg.group() as tg:
                    # Waveform (fixed 0 dBFS full-scale), then transport controls underneath.
                    self.wave_target.build(parent=tg, height=240)
                    self.tags["target_controls_group"] = "target_controls_group"
                    self._build_waveform_controls_generic(kind="target", tag="target_controls_group")

                # Cache tags used by other parts of the UI (seek / hit tests).
                self.tags["target_canvas"] = self.wave_target.get_canvas_tag()
                self.tags["target_draw"] = self.wave_target.get_draw_tag()

            # Right column (processing panel)
            with dpg.child_window(
                tag="target_right_panel",
                border=False,
                width=-1,
                height=-1,
                no_scrollbar=True,
                no_scroll_with_mouse=True,
            ):
                self.tags["target_right_panel"] = "target_right_panel"
                self.tags["target_audiofx_label"] = dpg.add_text("AudioFX")
                with dpg.child_window(tag="target_audiofx_panel", border=True, height=250, width=-1):
                    self.tags["target_audiofx_panel"] = "target_audiofx_panel"
                    self._build_output_processing_panel(kind="target")
                self.tags["target_audiofx_sep"] = dpg.add_separator()
                self._build_output_history_controls(kind="target")

    self.tags["output_split_sep"] = dpg.add_separator()

    # ------------------------------------------------------------------
    # Residual output
    # ------------------------------------------------------------------
    with dpg.child_window(
        tag="output_residual_panel",
        border=False,
        width=-1,
        height=300,
        no_scrollbar=True,
        no_scroll_with_mouse=True,
    ):
        self.tags["output_residual_panel"] = "output_residual_panel"
        self.tags["residual_title"] = dpg.add_text("Residual")

        with dpg.group(horizontal=True):
            with dpg.child_window(
                tag="residual_left_panel",
                border=False,
                width=-1,
                height=-1,
                no_scrollbar=True,
                no_scroll_with_mouse=True,
            ):
                self.tags["residual_left_panel"] = "residual_left_panel"
                self.tags["residual_save_group"] = dpg.add_group(horizontal=True, tag="residual_save_group")
                dpg.add_button(
                    label="Save Residual As...",
                    callback=lambda s, a, ud=None: self._show_dialog("dlg_save_residual"),
                    parent=self.tags["residual_save_group"],
                )
                self.tags["residual_path"] = dpg.add_text("", parent=self.tags["residual_save_group"])

                with dpg.group() as rg:
                    # Waveform (fixed 0 dBFS full-scale), then transport controls underneath.
                    self.wave_residual.build(parent=rg, height=240)
                    self.tags["residual_controls_group"] = "residual_controls_group"
                    self._build_waveform_controls_generic(kind="residual", tag="residual_controls_group")

                self.tags["residual_canvas"] = self.wave_residual.get_canvas_tag()
                self.tags["residual_draw"] = self.wave_residual.get_draw_tag()

            with dpg.child_window(
                tag="residual_right_panel",
                border=False,
                width=-1,
                height=-1,
                no_scrollbar=True,
                no_scroll_with_mouse=True,
            ):
                self.tags["residual_right_panel"] = "residual_right_panel"
                self.tags["residual_audiofx_label"] = dpg.add_text("AudioFX")
                with dpg.child_window(tag="residual_audiofx_panel", border=True, height=250, width=-1):
                    self.tags["residual_audiofx_panel"] = "residual_audiofx_panel"
                    self._build_output_processing_panel(kind="residual")
                self.tags["residual_audiofx_sep"] = dpg.add_separator()
                self._build_output_history_controls(kind="residual")
