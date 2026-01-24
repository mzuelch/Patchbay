"""Tab builder for PATCHBAY (DearPyGui frontend).

This module contains the UI construction code for the "run" tab.
The functions operate on the MainWindow instance and are intentionally
kept free of application logic beyond UI composition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..main_window import MainWindow

def build_tab_run(self: "MainWindow") -> None:
    """Compose the "Description & Run" tab.
    
    Parameters
    ----------
    self:
        MainWindow instance that owns the shared GUI state.
    """
    dpg = self.dpg

    # DearPyGui version compatibility:
    # Some installs do not provide draw_triangle_filled().
    # Use draw_triangle(..., fill=...) as a portable fallback.
    def _draw_filled_triangle(p1, p2, p3, *, color):
        if hasattr(dpg, "draw_triangle_filled"):
            dpg.draw_triangle_filled(p1, p2, p3, color=color)
            return
        try:
            dpg.draw_triangle(p1, p2, p3, color=color, fill=color, thickness=2)
        except TypeError:
            # Older API without fill support: fall back to outline only
            dpg.draw_triangle(p1, p2, p3, color=color, thickness=2)

    # ------------------------------------------------------------------
    # Top: Description
    # ------------------------------------------------------------------
    dpg.add_text("Description")
    with dpg.group(horizontal=True):
        self.tags["description"] = dpg.add_input_text(
            default_value=self.state.description,
            multiline=True,
            width=900,
            height=120,
            callback=self._on_description_changed,
        )
        with dpg.group():
            dpg.add_button(
                label="Load .txt...",
                callback=lambda s, a, u=None: self._show_dialog("dlg_open_text"),
            )
            dpg.add_button(
                label="Save .txt...",
                callback=lambda s, a, u=None: self._show_dialog("dlg_save_text"),
            )

    dpg.add_separator()

    # ------------------------------------------------------------------
    # Workflow row: Chunking -> Settings -> Run -> Parameters
    # ------------------------------------------------------------------
    with dpg.child_window(border=False, width=-1, height=-1, no_scrollbar=True, no_scroll_with_mouse=True) as _cw_workflow:
        self.tags["cw_workflow"] = _cw_workflow
        with dpg.table(
            header_row=False,
            resizable=True,
            policy=dpg.mvTable_SizingStretchProp,
            borders_innerV=True,
            borders_outerV=False,
            borders_innerH=False,
            borders_outerH=False,
            pad_outerX=False,
        ):
            # Chunking | → | Settings | → | Run | → | Parameters
            dpg.add_table_column(init_width_or_weight=0.24)
            dpg.add_table_column(init_width_or_weight=0.03)
            dpg.add_table_column(init_width_or_weight=0.24)
            dpg.add_table_column(init_width_or_weight=0.03)
            dpg.add_table_column(init_width_or_weight=0.20)
            dpg.add_table_column(init_width_or_weight=0.03)
            dpg.add_table_column(init_width_or_weight=0.33)

            with dpg.table_row():
                # ----------------------------
                # Chunking
                # ----------------------------
                with dpg.child_window(border=True, height=-1, no_scrollbar=True, no_scroll_with_mouse=True) as _cw_chunk:
                    self.tags["cw_chunk"] = _cw_chunk
                    self.tags["cw_chunk_title"] = dpg.add_text("Chunking")
                    self.tags["cw_chunk_sp_top"] = dpg.add_spacer(height=0)
                    with dpg.group() as _chunk_content:
                        self.tags["cw_chunk_content"] = _chunk_content

                        self.tags["use_chunking"] = dpg.add_checkbox(
                            label="Enable",
                            default_value=self.state.use_chunking,
                            callback=self._on_chunking_toggle,
                        )

                        with dpg.table(
                            header_row=False,
                            resizable=False,
                            policy=dpg.mvTable_SizingStretchProp,
                            borders_innerH=False,
                            borders_outerH=False,
                            borders_innerV=False,
                            borders_outerV=False,
                            pad_outerX=False,
                        ):
                            dpg.add_table_column(init_width_or_weight=0.55)
                            dpg.add_table_column(init_width_or_weight=0.45)

                            with dpg.table_row():
                                dpg.add_text("Max length (s)")
                                self.tags["max_len_s"] = dpg.add_input_float(
                                    default_value=float(self.state.max_len_s),
                                    width=-1,
                                    callback=self._on_chunking_params_changed,
                                )
                            with dpg.table_row():
                                dpg.add_text("Overlap (s)")
                                self.tags["overlap_s"] = dpg.add_input_float(
                                    default_value=float(self.state.overlap_s),
                                    width=-1,
                                    callback=self._on_chunking_params_changed,
                                )

                        dpg.add_spacer(height=6)
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="Save global", callback=lambda s, a, u=None: self._save_chunking_global())
                            dpg.add_button(label="Load global", callback=lambda s, a, u=None: self._load_chunking_global())
                    self.tags["cw_chunk_sp_bot"] = dpg.add_spacer(height=0)

                # Arrow
                with dpg.child_window(border=False, height=-1, no_scrollbar=True, no_scroll_with_mouse=True) as _cw_a1:
                    self.tags["cw_a1"] = _cw_a1
                    self.tags["cw_a1_sp_top"] = dpg.add_spacer(height=0)
                    with dpg.group() as _a1_content:
                        self.tags["cw_a1_content"] = _a1_content
                        with dpg.drawlist(width=34, height=60) as _dl_a1:
                            self.tags["dl_a1"] = _dl_a1
                            dpg.draw_line((2, 30), (26, 30), color=(220, 220, 220, 255), thickness=2)
                            _draw_filled_triangle((26, 30), (18, 24), (18, 36), color=(220, 220, 220, 255))
                    self.tags["cw_a1_sp_bot"] = dpg.add_spacer(height=0)
                # ----------------------------
                # Runtime settings
                # ----------------------------
                with dpg.child_window(border=True, height=-1, no_scrollbar=True, no_scroll_with_mouse=True) as _cw_set:
                    self.tags["cw_set"] = _cw_set
                    self.tags["cw_set_title"] = dpg.add_text("Settings")
                    self.tags["cw_set_sp_top"] = dpg.add_spacer(height=0)
                    with dpg.group() as _set_content:
                        self.tags["cw_set_content"] = _set_content

                        with dpg.table(
                            header_row=False,
                            resizable=False,
                            policy=dpg.mvTable_SizingStretchProp,
                            borders_innerH=False,
                            borders_outerH=False,
                            borders_innerV=False,
                            borders_outerV=False,
                            pad_outerX=False,
                        ):
                            dpg.add_table_column(init_width_or_weight=0.42)
                            dpg.add_table_column(init_width_or_weight=0.58)

                            with dpg.table_row():
                                dpg.add_text("Model")
                                _model_items = self._run_model_labels if getattr(self, "_run_model_labels", None) else [str(self.state.model)]
                                _model_default = self._run_model_default_label if getattr(self, "_run_model_default_label", None) else str(self.state.model)
                                self.tags["run_model"] = dpg.add_combo(
                                    items=_model_items,
                                    default_value=_model_default,
                                    width=-1,
                                    callback=lambda s, a, u=None: self._on_run_model_changed(a),
                                )

                            with dpg.table_row():
                                dpg.add_text("Compute")
                                self.tags["run_device"] = dpg.add_combo(
                                    items=["auto", "cuda", "cpu"],
                                    default_value=str(self.state.device),
                                    width=-1,
                                    callback=self._on_run_device_changed,
                                )

                            with dpg.table_row():
                                dpg.add_text("FP16")
                                self.tags["run_fp16"] = dpg.add_checkbox(
                                    label="",
                                    default_value=bool(self.state.fp16),
                                    callback=self._on_run_fp16_changed,
                                )

                            with dpg.table_row():
                                dpg.add_text("Time spans")
                                self.tags["run_predict"] = dpg.add_checkbox(
                                    label="",
                                    default_value=bool(self.state.predict_spans),
                                    callback=self._on_run_predict_changed,
                                )

                            with dpg.table_row():
                                dpg.add_text("Re-rank K")
                                self.tags["run_rerank"] = dpg.add_input_int(
                                    default_value=int(self.state.reranking_candidates),
                                    min_value=1,
                                    max_value=10,
                                    step=1,
                                    width=-1,
                                    callback=self._on_run_rerank_changed,
                                )

                            with dpg.table_row():
                                dpg.add_text("No resample")
                                self.tags["run_no_resample"] = dpg.add_checkbox(
                                    label="",
                                    default_value=bool(self.state.no_resample),
                                    callback=self._on_run_no_resample_changed,
                                )
                    self.tags["cw_set_sp_bot"] = dpg.add_spacer(height=0)

                # Arrow
                with dpg.child_window(border=False, height=-1, no_scrollbar=True, no_scroll_with_mouse=True) as _cw_a2:
                    self.tags["cw_a2"] = _cw_a2
                    self.tags["cw_a2_sp_top"] = dpg.add_spacer(height=0)
                    with dpg.group() as _a2_content:
                        self.tags["cw_a2_content"] = _a2_content
                        with dpg.drawlist(width=34, height=60) as _dl_a2:
                            self.tags["dl_a2"] = _dl_a2
                            dpg.draw_line((2, 30), (26, 30), color=(220, 220, 220, 255), thickness=2)
                            _draw_filled_triangle((26, 30), (18, 24), (18, 36), color=(220, 220, 220, 255))
                    self.tags["cw_a2_sp_bot"] = dpg.add_spacer(height=0)
                # ----------------------------
                # Run
                # ----------------------------
                with dpg.child_window(border=True, height=-1, no_scrollbar=True, no_scroll_with_mouse=True) as _cw_run:
                    self.tags["cw_run"] = _cw_run
                    self.tags["cw_run_title"] = dpg.add_text("Run")
                    self.tags["cw_run_sp_top"] = dpg.add_spacer(height=0)
                    with dpg.group() as _run_content:
                        self.tags["cw_run_content"] = _run_content

                        # Layout: [Run] [Progress] [Abort]
                        with dpg.table(
                            header_row=False,
                            resizable=False,
                            policy=dpg.mvTable_SizingStretchProp,
                            borders_innerH=False,
                            borders_outerH=False,
                            borders_innerV=False,
                            borders_outerV=False,
                            pad_outerX=False,
                        ):
                            dpg.add_table_column(init_width_or_weight=0.28)
                            dpg.add_table_column(init_width_or_weight=0.44)
                            dpg.add_table_column(init_width_or_weight=0.28)

                            with dpg.table_row():
                                self.tags["run_btn"] = dpg.add_button(
                                    label="Run",
                                    width=-1,
                                    height=48,
                                    callback=lambda s, a, u=None: self._on_run(),
                                )
                                self.tags["progress"] = dpg.add_progress_bar(default_value=0.0, width=-1)
                                self.tags["abort_btn"] = dpg.add_button(
                                    label="Abort",
                                    width=-1,
                                    height=48,
                                    callback=lambda s, a, u=None: self._on_abort(),
                                    enabled=False,
                                )

                            # Labels under progress bar (percent + current phase)
                            #
                            # NOTE: drawlists inside tables have shown to be unreliable on some
                            # DearPyGui builds (rect size may stay 0 and draw-children may not
                            # render). We therefore use normal text items inside a small
                            # child_window and position them manually for true centering.
                            with dpg.table_row():
                                dpg.add_text("")
                                with dpg.child_window(
                                    border=False,
                                    width=-1,
                                    height=140,
                                    no_scrollbar=True,
                                    no_scroll_with_mouse=True,
                                ) as _cw_prog_lbl:
                                    self.tags["progress_labels_cw"] = _cw_prog_lbl
                                    # Percent (single line) – positioned/centered by controller.
                                    self.tags["progress_pct_text"] = dpg.add_text("", pos=(0, 2))
                                    # Phase lines – controller will wrap and center per line.
                                    self.tags["progress_phase_lines"] = []
                                    for _i in range(0, 12):
                                        _tag = f"progress_phase_{_i}"
                                        self.tags["progress_phase_lines"].append(_tag)
                                        dpg.add_text("", tag=_tag, pos=(0, 0), show=False)
                                dpg.add_text("")

                        dpg.add_spacer(height=8)
                    self.tags["cw_run_sp_bot"] = dpg.add_spacer(height=0)

                # Arrow
                with dpg.child_window(border=False, height=-1, no_scrollbar=True, no_scroll_with_mouse=True) as _cw_a3:
                    self.tags["cw_a3"] = _cw_a3
                    self.tags["cw_a3_sp_top"] = dpg.add_spacer(height=0)
                    with dpg.group() as _a3_content:
                        self.tags["cw_a3_content"] = _a3_content
                        with dpg.drawlist(width=34, height=60) as _dl_a3:
                            self.tags["dl_a3"] = _dl_a3
                            dpg.draw_line((2, 30), (26, 30), color=(220, 220, 220, 255), thickness=2)
                            _draw_filled_triangle((26, 30), (18, 24), (18, 36), color=(220, 220, 220, 255))
                    self.tags["cw_a3_sp_bot"] = dpg.add_spacer(height=0)
                # ----------------------------
                # Backend call parameters + snippet
                # ----------------------------
                with dpg.child_window(border=True, height=-1, no_scrollbar=True, no_scroll_with_mouse=True) as _cw_par:
                    self.tags["cw_par"] = _cw_par
                    self.tags["cw_par_title"] = dpg.add_text("Parameters")
                    self.tags["cw_par_sp_top"] = dpg.add_spacer(height=0)
                    with dpg.group() as _par_content:
                        self.tags["cw_par_content"] = _par_content

                        # Parameter table (stable rows, values updated live)
                        with dpg.child_window(border=False, height=-72, no_scrollbar=True, no_scroll_with_mouse=True):
                            with dpg.table(
                                header_row=True,
                                resizable=True,
                                policy=dpg.mvTable_SizingStretchProp,
                                borders_innerH=True,
                                borders_outerH=False,
                                borders_innerV=True,
                                borders_outerV=False,
                                row_background=True,
                            ):
                                dpg.add_table_column(label="Parameter", init_width_or_weight=0.35)
                                dpg.add_table_column(label="Value", init_width_or_weight=0.65)

                                self.tags["param_model"] = None
                                self.tags["param_audio"] = None
                                self.tags["param_out_target"] = None
                                self.tags["param_out_residual"] = None
                                self.tags["param_description"] = None
                                self.tags["param_predict_spans"] = None
                                self.tags["param_reranking_candidates"] = None
                                self.tags["param_device"] = None
                                self.tags["param_fp16"] = None
                                self.tags["param_max_len_s"] = None
                                self.tags["param_overlap_s"] = None
                                self.tags["param_anchor_mode"] = None
                                self.tags["param_anchors"] = None
                                self.tags["param_no_resample"] = None

                                def _row(label: str, key: str):
                                    with dpg.table_row():
                                        dpg.add_text(label)
                                        self.tags[key] = dpg.add_text("")

                                _row("model", "param_model")
                                _row("audio", "param_audio")
                                _row("out_target", "param_out_target")
                                _row("out_residual", "param_out_residual")
                                _row("description", "param_description")
                                _row("predict_spans", "param_predict_spans")
                                _row("reranking_candidates", "param_reranking_candidates")
                                _row("device", "param_device")
                                _row("fp16", "param_fp16")
                                _row("max_len_s", "param_max_len_s")
                                _row("overlap_s", "param_overlap_s")
                                _row("anchor_mode", "param_anchor_mode")
                                _row("anchors", "param_anchors")
                                _row("no_resample", "param_no_resample")

                        dpg.add_spacer(height=6)
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="Save snippet .py...", callback=lambda s, a, u=None: self._show_dialog("dlg_save_snippet"))
                            dpg.add_button(label="Copy snippet", callback=lambda s, a, u=None: self._copy_backend_snippet_to_clipboard())

                        # NOTE: The snippet text display was intentionally removed.
                    self.tags["cw_par_sp_bot"] = dpg.add_spacer(height=0)

    self._refresh_backend_call_preview()
    self._update_chunking_enable_state()


