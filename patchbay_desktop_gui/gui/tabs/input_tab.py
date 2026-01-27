"""Tab builder for PATCHBAY (DearPyGui frontend).

This module contains the UI construction code for the "input" tab.
The functions operate on the MainWindow instance and are intentionally
kept free of application logic beyond UI composition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..main_window import MainWindow

def build_tab_input(self: "MainWindow") -> None:
    """Compose the "Input & Anchors" tab.
    
    Parameters
    ----------
    self:
        MainWindow instance that owns the shared GUI state.
    """
    dpg = self.dpg
    # Row 1: input load
    with dpg.group(horizontal=True):
        dpg.add_text("Input file:")
        self.tags["input_path"] = dpg.add_input_text(
            default_value=self.state.input_path,
            width=700,
            callback=self._on_input_path_changed,
        )
        dpg.add_button(label="Browse...", callback=lambda s, a, u=None: self._show_dialog("dlg_open_audio"))
        dpg.add_button(label="Reload", callback=lambda s, a, u=None: self._load_input_file(self.state.input_path))

    dpg.add_separator()

    # Row 2: two-column layout (left: waveform/anchors, right: AudioFX panel)
    with dpg.table(
        header_row=False,
        resizable=True,
        policy=dpg.mvTable_SizingStretchProp,
        borders_innerV=True,
        borders_outerV=False,
        borders_innerH=False,
        borders_outerH=False,
    ):
        dpg.add_table_column(init_width_or_weight=0.72)
        dpg.add_table_column(init_width_or_weight=0.28)

        with dpg.table_row():
            with dpg.group() as g_left:
                # Waveform (fixed 0 dBFS full-scale), then transport controls underneath.
                self.wave_input.build(parent=g_left, height=320)
                self._build_waveform_controls_input(tag="input_controls_group")

                # Marker tools in their own panel between waveform and anchors.
                dpg.add_separator()
                with dpg.child_window(border=True, height=90, width=-1):
                    dpg.add_text("Markers")
                    self._build_marker_controls(show_label=False)

                dpg.add_separator()
                # The Anchors panel itself should not scroll.
                # Only the anchors *table* becomes scrollable (inside _build_anchor_table).
                with dpg.child_window(border=True, height=-1, width=-1, no_scrollbar=True, no_scroll_with_mouse=True):
                    dpg.add_text("Anchors")
                    self._build_anchor_table(width=-1, height=-1)

            with dpg.child_window(
                tag="input_right_panel",
                border=False,
                height=-1,
                width=-1,
                no_scrollbar=True,
                no_scroll_with_mouse=True,
            ):
                self.tags["input_right_panel"] = "input_right_panel"
                self.tags["input_audiofx_label"] = dpg.add_text("Input processing (AudioFX)")
                with dpg.child_window(tag="input_audiofx_panel", border=True, height=520, width=-1):
                    self.tags["input_audiofx_panel"] = "input_audiofx_panel"
                    self._build_output_processing_panel(kind="input")
                self.tags["input_audiofx_sep"] = dpg.add_separator()
                # Edit history controls (Undo/Redo/Restore + peak) beneath AudioFX.
                self._build_output_history_controls(kind="input")
