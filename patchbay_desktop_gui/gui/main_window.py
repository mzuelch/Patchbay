"""Main window implementation for the DearPyGui frontend.

The previous GUI was Qt6/PySide6 based. This version re-implements the full
GUI with DearPyGui while keeping the overall layout and workflow.

Layout
------
We keep the 3-page layout requested by the user:

Page 1 (Input / Anchors)
  Row 1: Input file loading
  Row 2: Waveform + transport + zoom + marker tools + anchor list

Page 2 (Description / Chunking / Run)
  Row 1: Description editor + load/save
  Row 2: Chunking controls + save/load global defaults
  Row 3: Run/Abort + progress bar + backend call preview

Page 3 (Output)
  Row 1: Target waveform + transport + zoom + save-as
  Row 2: Residual waveform + transport + zoom + save-as

Threading
---------
Backend processing runs in a background thread (see :mod:`.worker`). Progress
events are pushed to a queue and applied in the GUI render callback.

Audio Playback
--------------
Playback is implemented via sounddevice/soundfile (see :mod:`.widgets.audio_player`).
"""

from __future__ import annotations

import os
import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from patchbay_backend import Anchor, Config
from patchbay_backend.config import ANCHOR_MODE_CHOICES

from .widgets.audio_player import AudioPlayer, load_audio_file, mono_mix
from ..audiofx import AudioFxManager, ParamKind, ParamSpec
from .utils.audio_edit import compressor_peak_ar, describe_peak_dbfs, peak_normalize
from .utils.anchor_io import load_anchors_for_audio, save_anchors_for_audio, anchor_sidecar_path
from .persistence import Settings
from .widgets.waveform_widget import AnchorItem, WaveformWidget
from .worker import BackendWorker, WorkerEvent
from .utils.sam_models import build_model_dropdown


# Short, user-facing descriptions for anchor modes (shown next to the dropdown).
ANCHOR_MODE_SHORTDESC = {
    "strict": "Only anchors overlapping the chunk",
    "append": "Append all anchors to every chunk",
    "strict_or_append": "Strict; otherwise append all",
    "append_nearest": "Append nearest anchors (K)",
    "append_previous": "Append previous anchor",
    "prepend_nearest": "Prepend nearest anchors (K)",
}


def anchor_mode_shortdesc(mode: str) -> str:
    m = str(mode or "").strip()
    return ANCHOR_MODE_SHORTDESC.get(m, m)


def _norm_path(p: str) -> str:
    try:
        return str(Path(p))
    except Exception:
        return str(p)


def _safe_float(x: object, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


@dataclass
class AppState:
    """In-memory GUI state."""

    input_path: str = ""
    description: str = ""

    # loaded audio
    input_audio: Optional[np.ndarray] = None  # (N,C) float32
    input_sr: int = 48000

    # output files
    out_target_path: str = ""
    out_residual_path: str = ""

    # anchors (GUI list)
    anchors: List[AnchorItem] = None  # type: ignore

    # chunking controls
    use_chunking: bool = False
    max_len_s: float = 15.0
    overlap_s: float = 2.0

    # backend settings
    model: str = "facebook/sam-audio-small"
    device: str = "auto"
    fp16: bool = True
    predict_spans: bool = False
    reranking_candidates: int = 1
    anchor_mode: str = "strict"
    no_resample: bool = False

    # UI navigation
    ui_selected_tab: str = "input"

    def __post_init__(self) -> None:
        if self.anchors is None:
            self.anchors = []


@dataclass
class OutputEditState:
    """Non-destructive editing state for an output buffer.

    The GUI keeps the current working buffer in memory and maintains undo/redo
    stacks as full snapshots. This is memory-heavier than a diff-based
    approach, but it is robust and easy to reason about.

    Notes
    -----
    - Buffers are stored as float32 numpy arrays of shape (N, C).
    - Undo/redo stacks are truncated to a maximum length configured via
      settings.
    """

    audio: Optional[np.ndarray] = None
    sr: int = 48000

    original_audio: Optional[np.ndarray] = None
    original_sr: int = 48000

    undo_stack: List[np.ndarray] = None  # type: ignore
    redo_stack: List[np.ndarray] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.undo_stack is None:
            self.undo_stack = []
        if self.redo_stack is None:
            self.redo_stack = []

    def clear_history(self) -> None:
        self.undo_stack.clear()
        self.redo_stack.clear()


class MainWindow:
    """DearPyGui main window controller."""

    def __init__(self, settings: Settings) -> None:
        import dearpygui.dearpygui as dpg

        self.dpg = dpg
        self.settings = settings
        self.state = AppState()

        # Restore last session
        self.state.input_path = str(self.settings.data.get("session", {}).get("last_audio", ""))
        self.state.description = str(self.settings.data.get("session", {}).get("last_description", ""))

        # Restore backend defaults
        b = self.settings.data.get("backend", {})
        self.state.model = str(b.get("model", self.state.model))
        self.state.device = str(b.get("device", self.state.device))
        self.state.fp16 = bool(b.get("fp16", self.state.fp16))
        self.state.predict_spans = bool(b.get("predict_spans", self.state.predict_spans))
        self.state.reranking_candidates = int(b.get("reranking_candidates", self.state.reranking_candidates))
        # Anchor mode (chunking strategy)
        # ------------------------------
        # The selected mode is persisted under settings["backend"]["anchor_mode"].
        # When upgrading, older settings files may contain an unknown value.
        # We clamp such values to the default ("strict") to keep the GUI
        # startable and to migrate the user's settings automatically.
        _loaded_anchor_mode = str(b.get("anchor_mode", self.state.anchor_mode))
        if _loaded_anchor_mode not in ANCHOR_MODE_CHOICES:
            _loaded_anchor_mode = "strict"
            try:
                b["anchor_mode"] = _loaded_anchor_mode
                self.settings.data["backend"] = b
                self.settings.save()
            except Exception:
                # If we cannot save (e.g. read-only filesystem), we still
                # continue with a safe in-memory default.
                pass
        self.state.anchor_mode = _loaded_anchor_mode
        self.state.no_resample = bool(b.get("no_resample", self.state.no_resample))

        # Build model dropdown items (live from HF collection if possible)
        try:
            self._run_model_labels, self._run_model_label_to_id, self._run_model_default_label = build_model_dropdown(
                str(self.state.model), timeout_s=2.0
            )
        except Exception:
            self._run_model_labels, self._run_model_label_to_id, self._run_model_default_label = ([], {}, str(self.state.model))

        # Restore global chunking
        cg = self.settings.data.get("chunking_global", {})
        self.state.use_chunking = bool(cg.get("use_chunking", False))
        self.state.max_len_s = float(cg.get("max_len_s", 15.0))
        self.state.overlap_s = float(cg.get("overlap_s", 2.0))

        # Worker
        self.worker = BackendWorker()

        # Audio players
        self.player_input = AudioPlayer()
        self.player_target = AudioPlayer()
        self.player_residual = AudioPlayer()

        # Output editing state (undo/redo + restore original)
        self._out_edit = {
            "input": OutputEditState(),
            "target": OutputEditState(),
            "residual": OutputEditState(),
        }

        # Input editing state (processed input temp file for backend runs)
        self._input_audio_dirty: bool = False
        self._input_edit_rev: int = 0
        self._input_temp_written_rev: int = -1
        self._input_temp_path: Optional[str] = None

        # Output processing defaults (persisted)
        op = self.settings.data.get("output_processing", {})
        self._comp_threshold_db_default = float(op.get("compressor_threshold_db", -12.0))
        self._comp_ratio_default = float(op.get("compressor_ratio", 2.0))
        # Envelope parameters for the compressor. These were added after the
        # initial "ratio + threshold" implementation to avoid audible
        # distortion caused by sample-by-sample gain changes.
        self._comp_attack_ms_default = float(op.get("compressor_attack_ms", 10.0))
        self._comp_release_ms_default = float(op.get("compressor_release_ms", 100.0))
        self._normalize_target_peak_default = float(op.get("normalize_target_peak", 0.999))
        self._max_undo_steps = int(op.get("max_undo_steps", 10))

        # ------------------------------------------------------------------
        # Audio FX plugins
        # ------------------------------------------------------------------
        # Plugins are discovered on startup. Parameter values are persisted
        # *separately per stream* (input/target/residual) under settings["audiofx"].
        self.audiofx = AudioFxManager()

        # Default plugin search locations:
        #  - %APPDATA%\CatSynth\PATCHBAY\plugins
        #  - ./plugins (project working directory)
        #  - optional extra directories via env var PATCHBAY_AUDIOFX_PLUGIN_DIRS
        try:
            from .persistence import settings_path as _settings_json_path  # type: ignore
            app_plugins = _settings_json_path().parent / "plugins"
        except Exception:
            app_plugins = Path.cwd() / "plugins"
        cwd_plugins = Path.cwd() / "plugins"

        # Plugins shipped with the source tree / installed package (if present)
        try:
            repo_plugins = Path(__file__).resolve().parents[2] / "plugins"
        except Exception:
            repo_plugins = cwd_plugins


        extra_dirs = []
        env_dirs = os.environ.get("PATCHBAY_AUDIOFX_PLUGIN_DIRS", "").strip()
        if env_dirs:
            for part in env_dirs.split(";"):
                part = part.strip()
                if part:
                    extra_dirs.append(Path(part))

        self._audiofx_user_dirs = [app_plugins, cwd_plugins, repo_plugins] + extra_dirs

        # Deduplicate while preserving order
        _seen = set()
        _uniq_dirs = []
        for _d in self._audiofx_user_dirs:
            try:
                _d = Path(_d)
            except Exception:
                continue
            if str(_d) in _seen:
                continue
            _seen.add(str(_d))
            _uniq_dirs.append(_d)
        self._audiofx_user_dirs = _uniq_dirs

        self.audiofx.discover(user_dirs=self._audiofx_user_dirs)
        self._audiofx_plugins = self.audiofx.list_plugins()

        # Ensure settings structure for per-stream AudioFX parameter persistence
        afx = self.settings.data.setdefault("audiofx", {})
        for _k in ("input", "target", "residual"):
            afx.setdefault(_k, {})

        # Seed defaults for commonly bundled file-based plugins if missing.
        # We also use legacy values from settings["output_processing"] (if present)
        # as a migration base for first-run defaults.
        try:
            base_op = self.settings.data.get("output_processing", {})
            base_norm = float(base_op.get("normalize_target_peak", 0.999))
            base_thr = float(base_op.get("compressor_threshold_db", -12.0))
            base_ratio = float(base_op.get("compressor_ratio", 2.0))
            base_att = float(base_op.get("compressor_attack_ms", 10.0))
            base_rel = float(base_op.get("compressor_release_ms", 100.0))
        except Exception:
            base_norm, base_thr, base_ratio, base_att, base_rel = 0.999, -12.0, 2.0, 10.0, 100.0

        for _k in ("input", "target", "residual"):
            afx[_k].setdefault("normalize", {"target_peak": base_norm})
            afx[_k].setdefault("compressor", {
                "threshold_db": base_thr,
                "ratio": base_ratio,
                "attack_ms": base_att,
                "release_ms": base_rel,
            })
            afx[_k].setdefault("lowpass", {
                "cutoff_hz": 8000.0,
                "steepness": "24 dB/oct",
            })

        # Save once if we added missing keys (best effort).
        try:
            self.settings.save()
        except Exception:
            pass

        # Trigger a re-center of the run-tab workflow panels.
        try:
            self._run_centering_dirty = True
        except Exception:
            pass


        # Waveform widgets
        pts = int(self.settings.data.get("ui", {}).get("waveform_points", 6000))
        self.wave_input = WaveformWidget(label="Input", points=pts, on_seek=self._on_input_seek)

        # Output waveforms: attach a right-click callback so the output editing
        # context menu opens reliably on all DearPyGui builds, independent of
        # global mouse coordinate semantics.
        self.wave_target = WaveformWidget(
            label="Target",
            points=pts,
            on_seek=self._on_target_seek,
        )
        self.wave_residual = WaveformWidget(
            label="Residual",
            points=pts,
            on_seek=self._on_residual_seek,
        )

        # Apply colors from settings
        ui = self.settings.data.get("ui", {})
        def rgba(key, default):
            v = ui.get(key, list(default))
            return (int(v[0]), int(v[1]), int(v[2]), int(v[3])) if isinstance(v, (list, tuple)) and len(v) == 4 else default

        self._apply_colors(
            playhead=rgba("playhead_color", (255, 200, 0, 220)),
            marker=rgba("marker_color", (120, 120, 120, 80)),
            anchor_plus=rgba("anchor_plus_color", (0, 180, 0, 80)),
            anchor_minus=rgba("anchor_minus_color", (200, 60, 60, 80)),
        )

        # DearPyGui item tags (filled in build())
        self.tags = {}

        # Deferred UI actions
        # ------------------
        # DearPyGui (and the underlying Dear ImGui) can be sensitive when
        # showing *modal* windows directly inside a callback (especially if
        # the callback originated from another popup/context menu).
        #
        # We therefore provide a tiny "run next frame" queue. Settings dialogs
        # are scheduled to open on the next frame to ensure they always appear.
        self._deferred_ui_tasks: List[callable] = []
        self._active_modal_dialog: Optional[str] = None

        # Cached backend snippet text.
        # The UI no longer shows the snippet, but the user can still copy/save it.
        self._backend_snippet_text: str = ""

        # Run-tab workflow vertical centering: updated each frame.
        self._run_centering_dirty: bool = True


        # Output context menu state
        # ------------------------
        # We implement output editing actions (Normalize/Compressor) via a
        # custom right-click menu. We intentionally do not rely on
        # ``dpg.popup(item, mousebutton=...)`` bound to a single item, because
        # popup binding can be inconsistent across DearPyGui versions and item
        # types (and users naturally right-click on the waveform area rather
        # than on a small text label).
        self._output_ctx_kind: Optional[str] = None  # "target" or "residual"
        self._output_ctx_tag: str = "__ctx_output_menu"
        self._output_ctx_visible: bool = False

        # Playback UI update throttling (avoid excessive redraw work).
        self._last_playhead_seconds = {"input": -1.0, "target": -1.0, "residual": -1.0}
        self._playhead_update_min_dt = 1.0 / 30.0  # ~30 Hz UI update

        # File dialogs: we keep them alive as hidden windows.
        self._build_file_dialogs()

    def on_frame(self) -> None:
        """Per-frame hook called from the application's render loop.

        We:
        - poll backend worker events (progress/done/error)
        - update playback playheads smoothly while audio is playing
        - allow waveform widgets to redraw once a real canvas size is available
          (DearPyGui may report 0x0 sizes before the viewport is shown).
        """
        # NOTE:
        # DearPyGui callbacks can occasionally trigger unexpected exceptions in
        # unrelated UI update code (e.g., when an item was deleted/hidden and a
        # playhead sync still tries to touch it).
        #
        # If `on_frame()` raises, the app's render loop will swallow the exception
        # but deferred UI tasks (like showing settings dialogs) would never run.
        #
        # Therefore we guard *each* step independently and always execute the
        # deferred UI queue at the end.
        try:
            self.poll_worker_events()
        except Exception:
            pass
        try:
            self._sync_playheads()
        except Exception:
            pass

        # Ensure waveform drawlists resize/redraw when the viewport appears or
        # when the window is resized.
        try:
            self.wave_input.tick()
            self.wave_target.tick()
            self.wave_residual.tick()
        except Exception:
            pass

        # Keep the run-tab workflow panels vertically centered.
        try:
            self._update_run_workflow_centering()
        except Exception:
            pass

        # Execute any deferred UI tasks (best effort).
        if self._deferred_ui_tasks:
            tasks = self._deferred_ui_tasks[:]
            self._deferred_ui_tasks.clear()
            for fn in tasks:
                try:
                    fn()
                except Exception:
                    # Never allow a deferred UI action to kill the render loop.
                    pass

    def _update_run_workflow_centering(self) -> None:
        """Vertically center workflow panel contents (excluding the section titles).

        DearPyGui aligns content to the top of child windows by default.
        The user requested that each section keeps its title at the top while
        the remaining widgets are vertically centered within the available
        panel height.

        We implement this via top/bottom spacer items whose heights are
        adjusted every frame (best-effort; if sizes are not known yet, we skip).
        """
        dpg = self.dpg

        # Required tags are only available after the Run tab is built.
        required = [
            "cw_workflow",
            "cw_chunk",
            "cw_chunk_title",
            "cw_chunk_sp_top",
            "cw_chunk_content",
            "cw_chunk_sp_bot",
            "cw_set",
            "cw_set_title",
            "cw_set_sp_top",
            "cw_set_content",
            "cw_set_sp_bot",
            "cw_run",
            "cw_run_title",
            "cw_run_sp_top",
            "cw_run_content",
            "cw_run_sp_bot",
            "cw_par",
            "cw_par_title",
            "cw_par_sp_top",
            "cw_par_content",
            "cw_par_sp_bot",
            "cw_a1",
            "cw_a1_sp_top",
            "cw_a1_content",
            "cw_a1_sp_bot",
            "cw_a2",
            "cw_a2_sp_top",
            "cw_a2_content",
            "cw_a2_sp_bot",
            "cw_a3",
            "cw_a3_sp_top",
            "cw_a3_content",
            "cw_a3_sp_bot",
        ]
        for k in required:
            if k not in self.tags:
                return


        # Ensure the workflow container and its panels consume the full remaining
        # height of the main window/tab area (no nested scrolling).
        try:
            if not dpg.is_item_shown(self.tags["cw_workflow"]):
                return
            vh = int(dpg.get_viewport_client_height())
            y = float(dpg.get_item_rect_min(self.tags["cw_workflow"])[1] or 0)
            avail = max(120, vh - y - 12)
            dpg.configure_item(self.tags["cw_workflow"], height=int(avail))
            for key in ("chunk", "set", "run", "par", "a1", "a2", "a3"):
                it = self.tags.get(f"cw_{key}")
                if it:
                    dpg.configure_item(it, height=int(avail))
        except Exception:
            pass

        def _center(panel_key: str) -> None:
            cw = self.tags.get(f"cw_{panel_key}")
            title = self.tags.get(f"cw_{panel_key}_title")
            sp_top = self.tags.get(f"cw_{panel_key}_sp_top")
            content = self.tags.get(f"cw_{panel_key}_content")
            sp_bot = self.tags.get(f"cw_{panel_key}_sp_bot")
            if not cw or not title or not sp_top or not content or not sp_bot:
                return

            cw_h = float(dpg.get_item_rect_size(cw)[1] or 0)
            title_h = float(dpg.get_item_rect_size(title)[1] or 0)
            content_h = float(dpg.get_item_rect_size(content)[1] or 0)

            # If sizes are not known yet (0x0 before first frame), skip.
            if cw_h <= 2 or content_h <= 0:
                return

            # Small padding between title and content.
            avail = max(0.0, cw_h - title_h - 8.0)
            pad = max(0.0, (avail - content_h) / 2.0)
            # DearPyGui expects integers.
            pad_i = int(pad)
            dpg.configure_item(sp_top, height=pad_i)
            dpg.configure_item(sp_bot, height=pad_i)

        def _center_arrow(key: str) -> None:
            cw = self.tags.get(f"cw_{key}")
            sp_top = self.tags.get(f"cw_{key}_sp_top")
            content = self.tags.get(f"cw_{key}_content")
            sp_bot = self.tags.get(f"cw_{key}_sp_bot")
            if not cw or not sp_top or not content or not sp_bot:
                return

            cw_h = float(dpg.get_item_rect_size(cw)[1] or 0)
            content_h = float(dpg.get_item_rect_size(content)[1] or 0)

            if cw_h <= 2 or content_h <= 0:
                return

            avail = max(0.0, cw_h - 8.0)
            pad = max(0.0, (avail - content_h) / 2.0)
            pad_i = int(pad)
            dpg.configure_item(sp_top, height=pad_i)
            dpg.configure_item(sp_bot, height=pad_i)


        _center("chunk")
        _center("set")
        _center("run")
        _center("par")
        _center_arrow("a1")
        _center_arrow("a2")
        _center_arrow("a3")

    def on_viewport_resize(self) -> None:
        """Resize/pin the main window to the viewport.

        The user expects the *OS window* (DearPyGui's viewport) to be the
        only thing that can be moved/resized/minimized. The internal main
        DearPyGui window should behave like a fixed root container that fills
        the viewport.

        DearPyGui does not automatically resize a window to the viewport on
        all builds; we therefore synchronize sizes explicitly on startup and
        on every viewport resize callback.
        """
        dpg = self.dpg
        if not dpg.does_item_exist("__main_window"):
            return

        # Use client size when available (excludes window frame/title bar).
        try:
            vw = int(dpg.get_viewport_client_width())
            vh = int(dpg.get_viewport_client_height())
        except Exception:
            vw = int(dpg.get_viewport_width())
            vh = int(dpg.get_viewport_height())

        # Defensive minimums (avoid 0x0 during early initialization).
        vw = max(400, vw)
        vh = max(300, vh)

        try:
            dpg.set_item_pos("__main_window", [0, 0])
        except Exception:
            pass
        try:
            dpg.configure_item("__main_window", width=vw, height=vh)
        except Exception:
            pass

        # Keep modal blocker in sync with viewport if it is visible.
        try:
            if dpg.does_item_exist("__modal_blocker") and dpg.is_item_shown("__modal_blocker"):
                dpg.set_item_pos("__modal_blocker", [0, 0])
                dpg.configure_item("__modal_blocker", width=vw, height=vh)
                if "modal_block_btn" in self.tags:
                    dpg.configure_item(self.tags["modal_block_btn"], width=vw, height=vh)
        except Exception:
            pass

        # Keep the active dialog centered if one is open.
        try:
            dlg = getattr(self, "_active_modal_dialog", None)
            if dlg and dpg.does_item_exist(dlg) and dpg.is_item_shown(dlg):
                w = int(dpg.get_item_width(dlg)) or 400
                h = int(dpg.get_item_height(dlg)) or 200
                x = max(0, vw // 2 - w // 2)
                y = max(0, vh // 2 - h // 2)
                dpg.set_item_pos(dlg, [x, y])
        except Exception:
            pass

        
        # Resize the tabs container so the footer remains visible.
        try:
            if dpg.does_item_exist("__tabs_container"):
                # Footer is a separator + one horizontal group. Keep a stable margin.
                footer_h = 44
                tabs_h = max(120, int(vh - footer_h))
                dpg.configure_item("__tabs_container", width=vw, height=tabs_h)
        except Exception:
            pass

        # Re-center progress labels (percent/phase) after resize.
        try:
            if hasattr(self, "_ui_progress_percent") and hasattr(self, "_ui_progress_phase"):
                self._set_progress_labels(float(self._ui_progress_percent), str(self._ui_progress_phase))
        except Exception:
            pass

# Mark run-tab workflow layout as dirty so its content gets re-centered
        # in the next frames.
        try:
            self._run_centering_dirty = True
        except Exception:
            pass

    def _sync_playheads(self) -> None:
        """Synchronize playhead indicators with the underlying players."""
        # Update playheads based on current playback positions.
        self._sync_one("input", self.player_input, self.wave_input)
        self._sync_one("target", self.player_target, self.wave_target)
        self._sync_one("residual", self.player_residual, self.wave_residual)

    def _sync_one(self, kind: str, player: AudioPlayer, wave: WaveformWidget) -> None:
        st = player.get_state()
        if st.total_samples <= 0 or st.sample_rate <= 0:
            return
        t = float(st.position_samples) / float(st.sample_rate)
        # Throttle updates to avoid redrawing too often.
        prev = float(self._last_playhead_seconds.get(kind, -1.0))
        if abs(t - prev) < 0.02 and not st.playing:
            return
        self._last_playhead_seconds[kind] = t
        wave.set_playhead_seconds(t)

    # ------------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------------

    def build(self) -> None:
        dpg = self.dpg

        # NOTE ABOUT TOP-LEVEL WINDOWS (IMPORTANT FOR DEARPYGUI)
        # -----------------------------------------------------
        # DearPyGui window items (created via ``dpg.window``) are conceptually
        # *top-level* windows. Creating them while another window is the active
        # container can lead to inconsistent behavior on some DPG builds,
        # especially when those auxiliary windows are initially hidden (show=False)
        # and later shown via ``show_item``.
        #
        # In particular, context-menu windows and modal settings dialogs have
        # been observed to not appear if they were created while a different
        # window is the current container.
        #
        # Therefore:
        #   - We build the main application window first.
        #   - Then we create auxiliary windows (context menu, settings dialogs)
        #     as true top-level windows outside of the main window scope.

        # Main window: pinned to the viewport
        # ---------------------------------
        # This is the root container of the GUI. It should *not* be draggable,
        # resizable or collapsible. Resizing should happen only at the OS
        # window level (the DearPyGui viewport).
        with dpg.window(
            tag="__main_window",
            label="",
            width=1200,
            height=800,
            no_title_bar=True,
            no_move=True,
            no_resize=True,
            no_collapse=True,
            no_close=True,
            no_scrollbar=True,
            no_scroll_with_mouse=True,
        ):
            # Main content (tabs) – resized to fill the viewport minus footer.
            with dpg.child_window(
                tag="__tabs_container",
                border=False,
                width=-1,
                height=600,  # updated in on_viewport_resize()
                no_scrollbar=True,
                no_scroll_with_mouse=True,
            ):
                                # Main navigation (compact, left-aligned)
                #
                # DearPyGui's built-in tab_bar does not offer stable cross-version control over
                # label fitting/width. To keep the tab selector compact and avoid full-width
                # distribution on some builds, PATCHBAY uses a button-based tab header and
                # shows/hides dedicated content panels below.
                with dpg.group(horizontal=True):
                    self.tags['tab_btn_input'] = dpg.add_button(
                        label='Input & Anchors',
                        callback=lambda s, a, u=None: self._select_main_tab('input'),
                    )
                    self.tags['tab_btn_run'] = dpg.add_button(
                        label='Description & Run',
                        callback=lambda s, a, u=None: self._select_main_tab('run'),
                    )
                    self.tags['tab_btn_output'] = dpg.add_button(
                        label='Output',
                        callback=lambda s, a, u=None: self._select_main_tab('output'),
                    )
                    dpg.add_spacer(width=-1)

                dpg.add_separator()

                with dpg.child_window(
                    tag='__tab_panels',
                    border=False,
                    width=-1,
                    height=-1,
                    no_scrollbar=True,
                    no_scroll_with_mouse=True,
                ):
                    with dpg.child_window(
                        tag='__panel_input',
                        border=False,
                        width=-1,
                        height=-1,
                        show=True,
                        no_scrollbar=True,
                        no_scroll_with_mouse=True,
                    ):
                        self._build_tab_input()
                    with dpg.child_window(
                        tag='__panel_run',
                        border=False,
                        width=-1,
                        height=-1,
                        show=False,
                        no_scrollbar=True,
                        no_scroll_with_mouse=True,
                    ):
                        self._build_tab_run()
                    with dpg.child_window(
                        tag='__panel_output',
                        border=False,
                        width=-1,
                        height=-1,
                        show=False,
                        no_scrollbar=True,
                        no_scroll_with_mouse=True,
                    ):
                        self._build_tab_output()

                # Ensure the initial selection is applied (and button state updated).
                self._select_main_tab('input')

            # Footer: Options + status line (always at the bottom)
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_button(label="Options...", callback=lambda s, a, u=None: self._open_options())
                dpg.add_spacer(width=10)
                self.tags["status_line"] = dpg.add_text("Ready")

        # Mark the main window as the primary/root window (best effort) *once*.
        # Doing this repeatedly has been observed to interfere with popups
        # and modal windows on some DearPyGui builds.
        try:
            dpg.set_primary_window("__main_window", True)
        except Exception:
            pass

        # (No auxiliary context-menu windows/handlers in this build)

        # If a last session audio exists, try loading it (best effort)
        if self.state.input_path and Path(self.state.input_path).exists():
            self._load_input_file(self.state.input_path, quiet=True)

    def _build_tab_input(self) -> None:
        """Build the input tab UI."""
        from .tabs.input_tab import build_tab_input

        build_tab_input(self)

    def _build_tab_run(self) -> None:
        """Build the run tab UI."""
        from .tabs.run_tab import build_tab_run

        build_tab_run(self)

    def _build_tab_output(self) -> None:
        """Build the output tab UI."""
        from .tabs.output_tab import build_tab_output

        build_tab_output(self)


    def _select_main_tab(self, key: str) -> None:
        """Show one of the main panels (Input / Description & Run / Output).

        PATCHBAY uses a compact button header instead of DearPyGui's built-in
        tab_bar to avoid cross-version tab fitting/width differences.

        Parameters
        ----------
        key:
            One of ``"input"``, ``"run"`` or ``"output"``.
        """
        dpg = self.dpg
        panels = {
            "input": "__panel_input",
            "run": "__panel_run",
            "output": "__panel_output",
        }
        if key not in panels:
            return

        # Show selected panel, hide the others.
        for k, tag in panels.items():
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, show=(k == key))

        # Visual feedback: disable the active tab button.
        btns = {
            "input": self.tags.get("tab_btn_input"),
            "run": self.tags.get("tab_btn_run"),
            "output": self.tags.get("tab_btn_output"),
        }
        for k, btn in btns.items():
            if not btn:
                continue
            try:
                dpg.configure_item(btn, enabled=(k != key))
            except Exception:
                pass

        self.state.ui_selected_tab = key

    def _build_waveform_controls_input(self) -> None:
        dpg = self.dpg
        with dpg.group(horizontal=True):
            # Transport
            dpg.add_button(label="Play", callback=lambda s, a, u=None: self._play("input"))
            dpg.add_button(label="Stop", callback=lambda s, a, u=None: self._stop("input"))
            dpg.add_button(label="Rewind", callback=lambda s, a, u=None: self._rewind("input"))
            dpg.add_spacer(width=10)
            dpg.add_button(label="-10s", callback=lambda s, a, u=None: self._step("input", -10.0))
            dpg.add_button(label="-1s", callback=lambda s, a, u=None: self._step("input", -1.0))
            dpg.add_button(label="+1s", callback=lambda s, a, u=None: self._step("input", +1.0))
            dpg.add_button(label="+10s", callback=lambda s, a, u=None: self._step("input", +10.0))
            dpg.add_spacer(width=10)
            # Zoom
            dpg.add_button(label="Zoom In", callback=lambda s, a, u=None: self.wave_input.zoom_in())
            dpg.add_button(label="Zoom Out", callback=lambda s, a, u=None: self.wave_input.zoom_out())
            dpg.add_button(label="Zoom to fit", callback=lambda s, a, u=None: self.wave_input.zoom_to_fit())

    # ------------------------------------------------------------------
    # Output processing panels (Normalize / Compressor)
    # ------------------------------------------------------------------
    def _build_output_processing_panel(self, *, kind: str) -> None:
        """Build the right-side processing panel for the given output.

        The panel is generated dynamically from Audio FX plugins. This allows
        new post-processing methods to be added later as drop-in Python files.

        Parameter persistence
        ---------------------
        Parameters are stored *separately per stream*:
            settings["audiofx"]["input"][plugin_id][param_id]
            settings["audiofx"]["target"][plugin_id][param_id]
            settings["audiofx"]["residual"][plugin_id][param_id]
        """
        dpg = self.dpg

        # Show plugin load errors (if any) so users can fix their plugin files.
        if getattr(self.audiofx, "load_errors", None):
            errs = self.audiofx.load_errors
            if errs:
                dpg.add_text(f"⚠ Plugin load errors: {len(errs)}")
                with dpg.tooltip(dpg.last_item()):
                    for e in errs[:20]:
                        dpg.add_text(f"{e.path}: {e.message}")
                    if len(errs) > 20:
                        dpg.add_text(f"... and {len(errs) - 20} more")

        # Helper: 2-column table with labels left, inputs right.
        def _begin_param_table() -> None:
            dpg.push_container_stack(
                dpg.add_table(
                    header_row=False,
                    resizable=False,
                    policy=dpg.mvTable_SizingStretchProp,
                    borders_innerH=False,
                    borders_outerH=False,
                    borders_innerV=False,
                    borders_outerV=False,
                    pad_outerX=False,
                )
            )
            dpg.add_table_column(init_width_or_weight=0.48)  # label column
            dpg.add_table_column(init_width_or_weight=0.52)  # input column

        def _end_param_table() -> None:
            dpg.pop_container_stack()

        def _get_saved_param_value(plugin_id: str, spec: ParamSpec):
            afx = self.settings.data.get("audiofx", {})
            kmap = afx.get(kind, {}) if isinstance(afx, dict) else {}
            pmap = kmap.get(plugin_id, {}) if isinstance(kmap, dict) else {}
            if isinstance(pmap, dict) and spec.param_id in pmap:
                return pmap.get(spec.param_id)
            return spec.default

        def _make_input_widget(plugin_id: str, spec: ParamSpec):
            """Create an input widget for one ParamSpec and return its item tag."""
            tag_key = f"audiofx_{kind}_{plugin_id}_{spec.param_id}"
            saved = _get_saved_param_value(plugin_id, spec)

            # Ensure correct type for default_value
            if spec.kind == ParamKind.BOOL:
                dv = bool(saved)
            elif spec.kind == ParamKind.INT:
                try:
                    dv = int(saved)
                except Exception:
                    dv = int(spec.default)
            else:
                dv = saved

            kwargs = {"label": "", "width": -1}
            if spec.kind == ParamKind.FLOAT:
                try:
                    dv_f = float(dv)
                except Exception:
                    dv_f = float(spec.default)
                kwargs["default_value"] = dv_f
                if spec.min_value is not None:
                    kwargs["min_value"] = float(spec.min_value)
                if spec.max_value is not None:
                    kwargs["max_value"] = float(spec.max_value)
                if spec.step is not None:
                    kwargs["step"] = float(spec.step)
                if spec.fmt is not None:
                    kwargs["format"] = str(spec.fmt)
                item = dpg.add_input_float(**kwargs)

            elif spec.kind == ParamKind.INT:
                kwargs2 = {"label": "", "width": -1, "default_value": int(dv)}
                if spec.min_value is not None:
                    kwargs2["min_value"] = int(spec.min_value)
                if spec.max_value is not None:
                    kwargs2["max_value"] = int(spec.max_value)
                if spec.step is not None:
                    kwargs2["step"] = int(spec.step)
                item = dpg.add_input_int(**kwargs2)

            elif spec.kind == ParamKind.BOOL:
                item = dpg.add_checkbox(label="", default_value=bool(dv))

            elif spec.kind == ParamKind.CHOICE:
                items = list(spec.choices or [])
                if not items:
                    items = [str(spec.default)]
                dv_s = str(dv) if str(dv) in items else items[0]
                item = dpg.add_combo(items=items, default_value=dv_s, label="", width=-1)

            else:
                # Fallback: treat as string
                item = dpg.add_input_text(label="", default_value=str(dv), width=-1)

            self.tags[tag_key] = item
            return item

        # Build one collapsible section per plugin.
        plugins = getattr(self, "_audiofx_plugins", None) or []
        if not plugins:
            dpg.add_text("No audio effect plugins found.")
            return

        for plugin in plugins:
            pid = plugin.plugin_id
            title = plugin.display_name or pid

            with dpg.collapsing_header(label=title, default_open=True):
                _begin_param_table()
                for spec in list(getattr(plugin, "params", []) or []):
                    with dpg.table_row():
                        dpg.add_text(spec.label)
                        _make_input_widget(pid, spec)
                _end_param_table()

                # Apply button (targets current output kind)
                btn_tag = f"audiofx_apply_{kind}_{pid}"
                self.tags[btn_tag] = dpg.add_button(
                    label="Anwenden",
                    width=-1,
                    callback=lambda *args, k=kind, p=pid: self._on_output_apply_audiofx(k, p),
                )

    def _on_output_apply_audiofx(self, kind: str, plugin_id: str) -> None:
        """Apply an Audio FX plugin to the current buffer (input/target/residual)."""
        dpg = self.dpg
        try:
            plugin = self.audiofx.plugins.get(plugin_id)
        except Exception:
            plugin = None

        if plugin is None:
            self._show_error(f"Audio FX plugin not available: {plugin_id}")
            return

        try:
            st = self._get_output_state(kind)
        except Exception as e:
            self._show_error(str(e))
            return

        if st.audio is None or st.audio.size == 0:
            self._set_status(f"No {kind} audio loaded.")
            return

        # Collect parameters from UI
        params = {}
        for spec in list(getattr(plugin, "params", []) or []):
            tag_key = f"audiofx_{kind}_{plugin_id}_{spec.param_id}"
            item = self.tags.get(tag_key)
            if item is None:
                params[spec.param_id] = spec.default
                continue
            try:
                val = dpg.get_value(item)
            except Exception:
                val = spec.default

            # Normalize types
            if spec.kind == ParamKind.FLOAT:
                try:
                    val = float(val)
                except Exception:
                    val = float(spec.default)
            elif spec.kind == ParamKind.INT:
                try:
                    val = int(val)
                except Exception:
                    val = int(spec.default)
            elif spec.kind == ParamKind.BOOL:
                val = bool(val)
            elif spec.kind == ParamKind.CHOICE:
                val = str(val)
            else:
                val = val
            params[spec.param_id] = val

        # Persist parameters separately per stream
        afx = self.settings.data.setdefault("audiofx", {})
        if kind not in afx or not isinstance(afx.get(kind), dict):
            afx[kind] = {}
        if plugin_id not in afx[kind] or not isinstance(afx[kind].get(plugin_id), dict):
            afx[kind][plugin_id] = {}
        afx[kind][plugin_id].update(params)
        try:
            self.settings.save()
        except Exception:
            pass

        # Apply transformation (non-destructive editing: push undo snapshot)
        try:
            self._push_undo_snapshot(kind)
            st.redo_stack.clear()
            new_audio = plugin.apply(st.audio, st.sr, params)
            # Ensure float32 contiguous, 2D
            from .utils.audio_edit import ensure_float32  # local import to avoid cycles
            new_audio = ensure_float32(new_audio)
            # Replace buffer + refresh playback/waveform
            self._set_output_audio(kind, new_audio, st.sr, mark_input_dirty=(kind == "input"))

            # Editing the input affects the backend run configuration.
            if kind == "input":
                try:
                    self._refresh_backend_call_preview()
                except Exception:
                    pass

            self._set_status(f"Applied {plugin.display_name} to {kind}.")
        except Exception as e:
            self._show_error(f"Failed to apply {plugin_id}: {e}")

    def _on_output_normalize_from_ui(self, kind: str) -> None:
        """Read normalize parameters from the panel and apply to the output."""
        dpg = self.dpg
        tag = self.tags.get(f"{kind}_norm_peak")
        try:
            peak = float(dpg.get_value(tag)) if tag else float(self._normalize_target_peak_default)
        except Exception:
            peak = float(self._normalize_target_peak_default)

        # Clamp to a sane range; do NOT auto-normalize beyond what the user set.
        peak = max(0.001, min(1.0, float(peak)))
        self._normalize_target_peak_default = float(peak)

        op = self.settings.data.setdefault("output_processing", {})
        op["normalize_target_peak"] = float(peak)
        try:
            self.settings.save()
        except Exception:
            pass

        self._on_output_normalize(kind)

    def _on_output_compress_from_ui(self, kind: str) -> None:
        """Read compressor parameters from the panel and apply to the output."""
        dpg = self.dpg

        def _getf(key: str, default: float) -> float:
            tag = self.tags.get(f"{kind}_{key}")
            if tag is None:
                return float(default)
            try:
                return float(dpg.get_value(tag))
            except Exception:
                return float(default)

        thr = _getf("comp_thr_db", float(self._comp_threshold_db_default))
        ratio = _getf("comp_ratio", float(self._comp_ratio_default))
        attack_ms = _getf("comp_attack_ms", float(self._comp_attack_ms_default))
        release_ms = _getf("comp_release_ms", float(self._comp_release_ms_default))

        # Update shared defaults (used by the actual processing function).
        self._comp_threshold_db_default = float(thr)
        self._comp_ratio_default = float(max(1.0, ratio))
        self._comp_attack_ms_default = float(max(0.0, attack_ms))
        self._comp_release_ms_default = float(max(0.0, release_ms))

        op = self.settings.data.setdefault("output_processing", {})
        op["compressor_threshold_db"] = float(self._comp_threshold_db_default)
        op["compressor_ratio"] = float(self._comp_ratio_default)
        op["compressor_attack_ms"] = float(self._comp_attack_ms_default)
        op["compressor_release_ms"] = float(self._comp_release_ms_default)
        try:
            self.settings.save()
        except Exception:
            pass

        self._on_output_compress(kind)

    def _build_waveform_controls_generic(self, *, kind: str) -> None:
        """Controls for target/residual (no markers)."""
        dpg = self.dpg
        with dpg.group(horizontal=True):
            dpg.add_button(label="Play", callback=lambda s, a, u=None: self._play(kind))
            dpg.add_button(label="Stop", callback=lambda s, a, u=None: self._stop(kind))
            dpg.add_button(label="Rewind", callback=lambda s, a, u=None: self._rewind(kind))
            dpg.add_spacer(width=10)
            dpg.add_button(label="-10s", callback=lambda s, a, u=None: self._step(kind, -10.0))
            dpg.add_button(label="-1s", callback=lambda s, a, u=None: self._step(kind, -1.0))
            dpg.add_button(label="+1s", callback=lambda s, a, u=None: self._step(kind, +1.0))
            dpg.add_button(label="+10s", callback=lambda s, a, u=None: self._step(kind, +10.0))
            dpg.add_spacer(width=10)
            if kind == "target":
                dpg.add_button(label="Zoom In", callback=lambda s, a, u=None: self.wave_target.zoom_in())
                dpg.add_button(label="Zoom Out", callback=lambda s, a, u=None: self.wave_target.zoom_out())
                dpg.add_button(label="Zoom to fit", callback=lambda s, a, u=None: self.wave_target.zoom_to_fit())
            else:
                dpg.add_button(label="Zoom In", callback=lambda s, a, u=None: self.wave_residual.zoom_in())
                dpg.add_button(label="Zoom Out", callback=lambda s, a, u=None: self.wave_residual.zoom_out())
                dpg.add_button(label="Zoom to fit", callback=lambda s, a, u=None: self.wave_residual.zoom_to_fit())


    # ------------------------------------------------------------------
    # Output processing context menu (right-click)
    # ------------------------------------------------------------------

    def _build_output_processing_context_menu_window(self) -> None:
        """Create the custom right-click menu for output processing.

        Why a custom menu?
        ------------------
        DearPyGui supports context menus via ``dpg.popup(item, mousebutton=...)``.
        In practice, however, popup binding can be unreliable across item types
        and container hierarchies (notably tabs + nested groups). Users also
        tend to right-click on the waveform itself, not on a small file-path
        label.

        We therefore implement a simple custom menu as a small floating window
        that we show/hide from a global right-click handler.

        The menu keeps the requested hierarchy:

        Normalisieren
          - Anwenden
          - Einstellungen
        Kompressor
          - Anwenden
          - Einstellungen
        """
        dpg = self.dpg

        if dpg.does_item_exist(self._output_ctx_tag):
            return

        with dpg.window(
            tag=self._output_ctx_tag,
            label="",
            show=False,
            no_title_bar=True,
            no_resize=True,
            no_move=False,
            no_collapse=True,
            autosize=True,
            no_scrollbar=True,
        ):
            # Updated dynamically when the menu opens (Target vs. Residual).
            self.tags["ctx_output_title"] = dpg.add_text("Audio-Bearbeitung")
            dpg.add_separator()

            # The context menu must not contain collapsible elements.
            # In earlier iterations we used collapsing headers which can start
            # in a collapsed state (e.g. "Kompressor"), confusing users.
            # We therefore show the submenu content *always expanded*.

            dpg.add_text("Normalisieren")
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Anwenden",
                    width=120,
                    callback=lambda s, a, u=None: self._on_output_normalize_ctx(),
                )
                dpg.add_button(
                    label="Einstellungen",
                    width=120,
                    callback=lambda s, a, u=None: self._open_normalize_settings(),
                )

            dpg.add_spacer(height=4)

            dpg.add_text("Kompressor")
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Anwenden",
                    width=120,
                    callback=lambda s, a, u=None: self._on_output_compress_ctx(),
                )
                dpg.add_button(
                    label="Einstellungen",
                    width=120,
                    callback=lambda s, a, u=None: self._open_compressor_settings(),
                )

            dpg.add_separator()
            dpg.add_text("(Right-click on Target/Residual waveform or path to open)")

    def _build_global_context_handlers(self) -> None:
        """Register global mouse handlers for the output context menu."""
        dpg = self.dpg
        if dpg.does_item_exist("__ctx_handlers"):
            return
        with dpg.handler_registry(tag="__ctx_handlers"):
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Right, callback=self._on_global_right_click)
            # Left click closes the menu if it is open and the click happens
            # outside the menu window.
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left, callback=self._on_global_left_click)

    def _mouse_pos_viewport(self) -> Tuple[float, float]:
        """Return the current mouse position in viewport coordinates."""
        dpg = self.dpg
        try:
            return tuple(dpg.get_mouse_pos(local=False))  # type: ignore[arg-type]
        except TypeError:
            return tuple(dpg.get_mouse_pos())

    def _mouse_pos_candidates(self) -> List[Tuple[float, float]]:
        """Return a small set of candidate mouse positions.

        DearPyGui has changed the semantics/signature of ``get_mouse_pos``
        across releases and builds.

        - Some builds support ``local=...`` and return viewport coordinates for
          ``local=False``.
        - Some builds ignore the kwarg and always return local coordinates.
        - Older builds don't accept the kwarg at all.

        For hit-testing we therefore try multiple variants and accept a match
        from any of them.
        """
        dpg = self.dpg
        out: List[Tuple[float, float]] = []
        # Best effort: explicit viewport coordinates.
        try:
            out.append(tuple(dpg.get_mouse_pos(local=False)))  # type: ignore[arg-type]
        except Exception:
            pass
        # Some builds interpret local=True as window-local; include it.
        try:
            out.append(tuple(dpg.get_mouse_pos(local=True)))  # type: ignore[arg-type]
        except Exception:
            pass
        # Parameterless call.
        try:
            out.append(tuple(dpg.get_mouse_pos()))
        except Exception:
            pass
        # De-duplicate
        uniq: List[Tuple[float, float]] = []
        for p in out:
            if p and p not in uniq:
                uniq.append(p)
        return uniq

    def _item_rect(self, item: str) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Return (min,max) rectangle for an item in viewport coordinates."""
        dpg = self.dpg
        try:
            if not item or not dpg.does_item_exist(item):
                return None
            mn = tuple(dpg.get_item_rect_min(item))
            mx = tuple(dpg.get_item_rect_max(item))
            return (mn, mx)
        except Exception:
            return None

    def _hit_test_item(self, item: Optional[str], pos: Tuple[float, float]) -> bool:
        if not item:
            return False
        rect = self._item_rect(item)
        if not rect:
            return False
        (x0, y0), (x1, y1) = rect
        x, y = pos
        return (x0 <= x <= x1) and (y0 <= y <= y1)

    def _hovered(self, item: Optional[str]) -> bool:
        """Return True if *item* is currently hovered (best effort)."""
        dpg = self.dpg
        if not item:
            return False
        try:
            if not dpg.does_item_exist(item):
                return False
            return bool(dpg.is_item_hovered(item))
        except Exception:
            return False

    def _on_global_right_click(self, sender, app_data) -> None:
        """Open the output context menu when right-clicking an output region."""
        dpg = self.dpg
        # If a settings dialog is active, ignore context-menu interaction.
        # The main window is input-disabled during settings, but handler
        # registries can still fire globally.
        if getattr(self, "_active_modal_dialog", None):
            return
        # We compute multiple mouse position candidates because DearPyGui's
        # coordinate systems vary across versions/builds.
        pos_candidates = self._mouse_pos_candidates()
        pos = pos_candidates[0] if pos_candidates else self._mouse_pos_viewport()

        # Determine whether the click happened over Target or Residual.
        target_canvas = self.tags.get("target_canvas")
        target_draw = self.tags.get("target_draw")
        residual_canvas = self.tags.get("residual_canvas")
        residual_draw = self.tags.get("residual_draw")

        def hit_any(item: Optional[str]) -> bool:
            for p in pos_candidates:
                if self._hit_test_item(item, p):
                    return True
            return False

        # Prefer hovered checks (most robust), fall back to rectangle hit-test.
        if (
            self._hovered(target_canvas)
            or self._hovered(target_draw)
            or self._hovered(self.tags.get("target_path"))
            or hit_any(target_canvas)
            or hit_any(target_draw)
            or hit_any(self.tags.get("target_path"))
        ):
            self._show_output_ctx_menu(kind="target", pos=pos)
            return
        if (
            self._hovered(residual_canvas)
            or self._hovered(residual_draw)
            or self._hovered(self.tags.get("residual_path"))
            or hit_any(residual_canvas)
            or hit_any(residual_draw)
            or hit_any(self.tags.get("residual_path"))
        ):
            self._show_output_ctx_menu(kind="residual", pos=pos)
            return

        # We intentionally do NOT close the menu on right-click elsewhere.
        #
        # Rationale: On some DearPyGui builds, global mouse coordinate
        # semantics can make hit-testing unreliable in nested layouts.
        # If an item-specific right-click handler (bound to the waveform
        # drawlist) has just opened the menu, a global right-click handler
        # should not immediately close it.
        #
        # The menu is closed on:
        # - left-click outside the menu, or
        # - explicit actions inside the menu.
        return

    def _on_global_left_click(self, sender, app_data) -> None:
        """Close the output context menu on *left click outside* the menu.

        DearPyGui calls global mouse handlers for *every* click, including
        clicks on buttons inside our custom floating context menu window.

        If we hide the menu *immediately* in this global handler, the click
        intended for a menu button can be lost (the button callback will not
        run). This manifests exactly as reported by the user:

        - "Apply" appears to do nothing
        - "Settings" does not open a dialog

        To avoid swallowing button clicks we:
        - treat a click as "inside the menu" if ANY of our mouse position
          candidates hits the menu's rect (coordinate semantics differ across
          DearPyGui builds)
        - and defer closing the menu to the next frame when the click is
          outside, so item callbacks on the current frame still execute.
        """
        if getattr(self, "_active_modal_dialog", None):
            return
        if not self._output_ctx_visible:
            return

        # If the menu itself is hovered (best effort), do not close it.
        if self._hovered(self._output_ctx_tag):
            return

        # Robust hit test using multiple candidate coordinate systems.
        inside = False
        for p in self._mouse_pos_candidates():
            if self._hit_test_item(self._output_ctx_tag, p):
                inside = True
                break
        if inside:
            return

        # Defer closing to the next frame so the current click can be
        # delivered to any item that may have been clicked.
        self._deferred_ui_tasks.append(lambda: self._hide_output_ctx_menu())

    def _show_output_ctx_menu(self, *, kind: str, pos: Tuple[float, float]) -> None:
        dpg = self.dpg
        self._output_ctx_kind = kind

        # Update menu header to show which output will be affected.
        try:
            if "ctx_output_title" in self.tags:
                dpg.set_value(self.tags["ctx_output_title"], f"Audio-Bearbeitung ({kind})")
        except Exception:
            pass

        # Clamp menu position to viewport
        try:
            vw = int(dpg.get_viewport_width())
            vh = int(dpg.get_viewport_height())
        except Exception:
            vw = 1200
            vh = 800

        x, y = pos
        # Offset so the cursor doesn't sit on top of the menu
        x += 2
        y += 2

        # A rough clamp (autosize window).
        x = max(0, min(float(vw - 50), float(x)))
        y = max(0, min(float(vh - 50), float(y)))

        try:
            dpg.set_item_pos(self._output_ctx_tag, [x, y])
            dpg.show_item(self._output_ctx_tag)
            dpg.focus_item(self._output_ctx_tag)
            # Ensure the menu is not hidden behind other windows.
            if hasattr(dpg, "bring_item_to_front"):
                try:
                    dpg.bring_item_to_front(self._output_ctx_tag)
                except Exception:
                    pass
            self._output_ctx_visible = True
        except Exception:
            return

    def _hide_output_ctx_menu(self) -> None:
        dpg = self.dpg
        try:
            dpg.hide_item(self._output_ctx_tag)
        except Exception:
            pass
        self._output_ctx_visible = False

    # ------------------------------------------------------------------
    # Output processing settings dialogs
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Modal settings dialogs (DearPyGui 2.1.x robust)
    # ------------------------------------------------------------------
    def _show_modal_dialog(self, dialog_tag: str) -> bool:
        """Show a settings dialog in a *robust modal* way.

        DearPyGui's built-in ``modal=True`` window feature has shown
        inconsistent behavior on some 2.1.x builds when dialogs are opened
        from transient UI elements (custom context menus, popups, tabs).

        Earlier versions of this GUI used an invisible full-viewport blocker
        window to emulate modality. That approach can leave the app in a
        "blocked" state if a show/hide request is dropped (user sees buttons
        that do nothing).

        The most reliable approach for DearPyGui 2.1.x is:
        - Disable input on the main root window via ``no_inputs=True``.
        - Show the dialog as a top-level window, centered, brought to front.

        This keeps the dialog fully interactive while preventing interaction
        with the main UI, without relying on any fragile overlay widget.
        """
        dpg = self.dpg
        if not dpg.does_item_exist(dialog_tag):
            return False

        # Determine viewport size for centering.
        try:
            vw = int(getattr(dpg, "get_viewport_client_width", dpg.get_viewport_width)())
            vh = int(getattr(dpg, "get_viewport_client_height", dpg.get_viewport_height)())
        except Exception:
            vw, vh = 1200, 800

        # Center dialog (use configured width/height; fall back to defaults).
        try:
            w = int(dpg.get_item_width(dialog_tag)) or 420
            h = int(dpg.get_item_height(dialog_tag)) or 220
        except Exception:
            w, h = 420, 220

        try:
            x = max(0, vw // 2 - w // 2)
            y = max(0, vh // 2 - h // 2)
            dpg.set_item_pos(dialog_tag, [x, y])
        except Exception:
            pass

        # Show and bring to front.
        shown_ok = False
        try:
            dpg.configure_item(dialog_tag, show=True)
            shown_ok = True
        except Exception:
            try:
                dpg.show_item(dialog_tag)
                shown_ok = True
            except Exception:
                shown_ok = False

        # Best-effort visibility check. If the dialog could not be shown,
        # do NOT disable the main window inputs (otherwise the UI appears dead).
        try:
            if hasattr(dpg, "is_item_shown") and not dpg.is_item_shown(dialog_tag):
                shown_ok = False
        except Exception:
            pass

        if shown_ok:
            if hasattr(dpg, "bring_item_to_front"):
                try:
                    dpg.bring_item_to_front(dialog_tag)
                except Exception:
                    pass
            try:
                dpg.focus_item(dialog_tag)
            except Exception:
                pass

            # Disable input on the main root window to emulate modality.
            try:
                if dpg.does_item_exist("__main_window"):
                    dpg.configure_item("__main_window", no_inputs=True)
            except Exception:
                pass

            self._active_modal_dialog = dialog_tag
            return True
        else:
            # Ensure main window remains interactive if the dialog could not
            # be shown for any reason.
            try:
                if dpg.does_item_exist("__main_window"):
                    dpg.configure_item("__main_window", no_inputs=False)
            except Exception:
                pass

        return False


    def _defer_show_modal_dialog(self, dialog_tag: str) -> None:
        """Show a modal dialog *reliably* by deferring to the render loop.

        Some DearPyGui builds (and some UI interaction sequences) can drop a
        window show/hide request when called from within callbacks of transient
        UI elements (context menus, popups).

        We therefore schedule the show operation to run in `on_frame()` and
        (if needed) perform a second attempt on the next frame.
        """
        def attempt() -> None:
            try:
                ok = self._show_modal_dialog(dialog_tag)
            except Exception:
                ok = False

            # Second-chance attempt if the window still isn't visible.
            if not ok:
                self._deferred_ui_tasks.append(lambda: self._show_modal_dialog(dialog_tag))

        self._deferred_ui_tasks.append(attempt)

    def _hide_modal_layer(self) -> None:
        """Close the active settings dialog and re-enable main window input."""
        dpg = self.dpg

        # Hide dialog first (if any).
        try:
            if getattr(self, "_active_modal_dialog", None) and dpg.does_item_exist(self._active_modal_dialog):
                dpg.hide_item(self._active_modal_dialog)
        except Exception:
            pass

        # Re-enable inputs on the main window.
        try:
            if dpg.does_item_exist("__main_window"):
                dpg.configure_item("__main_window", no_inputs=False)
        except Exception:
            pass

        # Best-effort: hide legacy blocker if it exists from older versions.
        try:
            if dpg.does_item_exist("__modal_blocker"):
                dpg.hide_item("__modal_blocker")
        except Exception:
            pass

        self._active_modal_dialog = None

    def _build_output_processing_settings_windows(self) -> None:
        """Create hidden settings dialogs for Normalize and Compressor.

        DearPyGui's built-in ``modal=True`` windows can behave inconsistently
        across versions and complex container hierarchies (tabs + nested groups),
        especially when dialogs are opened from callbacks of other transient UI
        elements (context menus, popups).

        To guarantee reliability on DearPyGui 2.1.x, we *do not* use
        DearPyGui's internal ``modal=True`` feature.

        Instead we emulate modality by temporarily disabling input on the
        application's root window (``__main_window``) via ``no_inputs=True``
        while a settings dialog is shown.

        This avoids the class of failures where an invisible blocker overlay
        remains active and makes the UI appear "dead" (buttons stop responding).
        """
        dpg = self.dpg

        # Note: Legacy overlay-based modal blocker windows are no longer built.

        # Normalize settings dialog
        if not dpg.does_item_exist("__dlg_norm_settings"):
            with dpg.window(
                tag="__dlg_norm_settings",
                label="Normalize settings",
                # The user explicitly requested modal settings dialogs.
                # To keep them reliable we schedule the actual showing on
                # the next frame (see _open_*_settings).
                modal=False,
                show=False,
                no_move=True,
                no_resize=True,
                # Prevent the user from closing the dialog via the window's
                # [X] button. If they could close the window that way, our
                # custom modal layer (blocker + active dialog state) would not
                # be reset, which can lead to "works once" behaviour.
                no_close=True,
                width=360,
                height=150,
            ):
                dpg.add_text("Target peak (linear full-scale, 0..1)")
                self.tags["norm_target_peak"] = dpg.add_input_float(
                    default_value=float(self._normalize_target_peak_default),
                    width=160,
                    min_value=0.01,
                    max_value=1.0,
                    min_clamped=True,
                    max_clamped=True,
                    step=0.01,
                    step_fast=0.05,
                )
                dpg.add_spacer(height=8)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="OK", callback=lambda s, a, u=None: self._save_normalize_settings())
                    dpg.add_button(label="Cancel", callback=lambda s, a, u=None: self._hide_modal_layer())

        # Compressor settings dialog
        if not dpg.does_item_exist("__dlg_comp_settings"):
            with dpg.window(
                tag="__dlg_comp_settings",
                label="Compressor settings",
                modal=False,
                show=False,
                no_move=True,
                no_resize=True,
                # Same rationale as for the Normalize settings dialog.
                no_close=True,
                width=520,
                height=240,
            ):
                dpg.add_text("Hard-knee compressor (peak detector, attack/release envelope)")
                with dpg.group(horizontal=True):
                    dpg.add_text("Threshold (dBFS)")
                    self.tags["comp_thr_setting"] = dpg.add_input_float(
                        default_value=float(self._comp_threshold_db_default),
                        width=120,
                        step=1.0,
                        step_fast=3.0,
                    )
                    dpg.add_spacer(width=10)
                    dpg.add_text("Ratio")
                    self.tags["comp_ratio_setting"] = dpg.add_input_float(
                        default_value=float(self._comp_ratio_default),
                        width=100,
                        step=0.1,
                        step_fast=0.5,
                        min_value=1.0,
                        min_clamped=True,
                    )
                with dpg.group(horizontal=True):
                    dpg.add_text("Attack (ms)")
                    self.tags["comp_attack_setting"] = dpg.add_input_float(
                        default_value=float(self._comp_attack_ms_default),
                        width=120,
                        step=1.0,
                        step_fast=5.0,
                        min_value=0.0,
                        min_clamped=True,
                    )
                    dpg.add_spacer(width=10)
                    dpg.add_text("Release (ms)")
                    self.tags["comp_release_setting"] = dpg.add_input_float(
                        default_value=float(self._comp_release_ms_default),
                        width=120,
                        step=5.0,
                        step_fast=20.0,
                        min_value=0.0,
                        min_clamped=True,
                    )

                dpg.add_spacer(height=8)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="OK", callback=lambda s, a, u=None: self._save_compressor_settings())
                    dpg.add_button(label="Cancel", callback=lambda s, a, u=None: self._hide_modal_layer())

    def _open_normalize_settings(self) -> None:
        """Open the Normalize settings dialog as a custom modal."""
        dpg = self.dpg
        if not dpg.does_item_exist("__dlg_norm_settings"):
            return
        # Make this operation idempotent and robust: close any leftover modal
        # state and hide the context menu first.
        self._hide_modal_layer()
        self._hide_output_ctx_menu()

        # Refresh inputs from current defaults
        if "norm_target_peak" in self.tags:
            try:
                dpg.set_value(self.tags["norm_target_peak"], float(self._normalize_target_peak_default))
            except Exception:
                pass

        # Prefer showing immediately. If DearPyGui drops the request (rare but
        # observed when called from transient UI elements), fall back to a
        # deferred show on the next frame.
        ok = False
        try:
            ok = bool(self._show_modal_dialog("__dlg_norm_settings"))
        except Exception:
            ok = False
        if not ok:
            self._defer_show_modal_dialog("__dlg_norm_settings")


    def _save_normalize_settings(self) -> None:
        dpg = self.dpg
        tp = float(self._normalize_target_peak_default)
        if "norm_target_peak" in self.tags:
            tp = _safe_float(dpg.get_value(self.tags["norm_target_peak"]), tp)
        tp = max(0.01, min(1.0, tp))
        self._normalize_target_peak_default = float(tp)
        op = self.settings.data.setdefault("output_processing", {})
        op["normalize_target_peak"] = float(tp)
        try:
            self.settings.save()
        except Exception:
            pass
        dpg.hide_item("__dlg_norm_settings")
        self._set_status(f"Normalize settings saved (target_peak={tp:.3f})")
        # Close the modal dialog after saving.
        self._hide_modal_layer()



    def _open_compressor_settings(self) -> None:
        """Open the Compressor settings dialog as a custom modal."""
        dpg = self.dpg
        if not dpg.does_item_exist("__dlg_comp_settings"):
            return
        # Close any leftover modal state and hide the context menu first.
        self._hide_modal_layer()
        self._hide_output_ctx_menu()

        # Refresh inputs from current defaults
        try:
            if "comp_thr_setting" in self.tags:
                dpg.set_value(self.tags["comp_thr_setting"], float(self._comp_threshold_db_default))
            if "comp_ratio_setting" in self.tags:
                dpg.set_value(self.tags["comp_ratio_setting"], float(self._comp_ratio_default))
            if "comp_attack_setting" in self.tags:
                dpg.set_value(self.tags["comp_attack_setting"], float(self._comp_attack_ms_default))
            if "comp_release_setting" in self.tags:
                dpg.set_value(self.tags["comp_release_setting"], float(self._comp_release_ms_default))
        except Exception:
            pass

        # Prefer showing immediately. Fall back to deferred show if needed.
        ok = False
        try:
            ok = bool(self._show_modal_dialog("__dlg_comp_settings"))
        except Exception:
            ok = False
        if not ok:
            self._defer_show_modal_dialog("__dlg_comp_settings")


    def _save_compressor_settings(self) -> None:
        dpg = self.dpg
        thr = float(self._comp_threshold_db_default)
        ratio = float(self._comp_ratio_default)
        atk = float(self._comp_attack_ms_default)
        rel = float(self._comp_release_ms_default)

        if "comp_thr_setting" in self.tags:
            thr = _safe_float(dpg.get_value(self.tags["comp_thr_setting"]), thr)
        if "comp_ratio_setting" in self.tags:
            ratio = _safe_float(dpg.get_value(self.tags["comp_ratio_setting"]), ratio)
        if "comp_attack_setting" in self.tags:
            atk = _safe_float(dpg.get_value(self.tags["comp_attack_setting"]), atk)
        if "comp_release_setting" in self.tags:
            rel = _safe_float(dpg.get_value(self.tags["comp_release_setting"]), rel)

        if ratio < 1.0:
            ratio = 1.0
        atk = max(0.0, atk)
        rel = max(0.0, rel)

        self._comp_threshold_db_default = float(thr)
        self._comp_ratio_default = float(ratio)
        self._comp_attack_ms_default = float(atk)
        self._comp_release_ms_default = float(rel)

        op = self.settings.data.setdefault("output_processing", {})
        op["compressor_threshold_db"] = float(thr)
        op["compressor_ratio"] = float(ratio)
        op["compressor_attack_ms"] = float(atk)
        op["compressor_release_ms"] = float(rel)
        try:
            self.settings.save()
        except Exception:
            pass

        dpg.hide_item("__dlg_comp_settings")
        self._set_status(f"Compressor settings saved (thr={thr:.1f} dB, ratio={ratio:.2f}, atk={atk:.0f}ms, rel={rel:.0f}ms)")

        # Close the custom modal layer (dialog + blocker). This MUST happen
        # after saving, otherwise the invisible blocker continues to intercept
        # mouse input and subsequent dialogs will not open.
        self._hide_modal_layer()
    def _build_output_history_controls(self, *, kind: str) -> None:
        """Create the output history control bar (Undo/Redo/Restore + peak).

        The actual audio processing operations (Normalize, Compressor) were
        moved into a hierarchical right-click context menu on the output file
        path, as requested.

        This bar therefore only contains:
        - Undo / Redo
        - Restore original output
        - Peak display (informational)
        """
        dpg = self.dpg
        prefix = f"out_{kind}_"

        with dpg.group(horizontal=True):
            self.tags[prefix + "undo_btn"] = dpg.add_button(label="Undo", callback=lambda s, a, u=None: self._on_output_undo(kind))
            self.tags[prefix + "redo_btn"] = dpg.add_button(label="Redo", callback=lambda s, a, u=None: self._on_output_redo(kind))
            dpg.add_spacer(width=10)
            self.tags[prefix + "restore_btn"] = dpg.add_button(label="Restore original", callback=lambda s, a, u=None: self._on_output_restore_original(kind))
            dpg.add_spacer(width=20)
            self.tags[prefix + "peak_text"] = dpg.add_text("")

        self._update_output_edit_ui(kind)



    def _build_marker_controls(self, show_label: bool = True) -> None:
        dpg = self.dpg
        with dpg.group(horizontal=True):
            if show_label:
                dpg.add_text("Markers:")
            dpg.add_button(label="Set Start", callback=lambda s, a, u=None: self._on_set_marker_start())
            dpg.add_button(label="Set Stop", callback=lambda s, a, u=None: self._on_set_marker_stop())
            dpg.add_button(label="Clear", callback=lambda s, a, u=None: self._on_clear_markers())
            dpg.add_spacer(width=10)
            dpg.add_button(label="Send to Anchor +", callback=lambda s, a, u=None: self._send_marker_to_anchor("+"))
            dpg.add_button(label="Send to Anchor -", callback=lambda s, a, u=None: self._send_marker_to_anchor("-"))
            dpg.add_spacer(width=10)
            dpg.add_button(label="Remove Anchor @ Playhead", callback=lambda s, a, u=None: self._delete_anchor_at_playhead())

    def _build_anchor_table(self, *, width: int = 360, height: int = 360) -> None:
        """Build the anchors table.

        Parameters
        ----------
        width, height:
            Size of the scrolling table container. Use ``width=-1`` to fill
            the available horizontal space.
        """
        dpg = self.dpg
        self.tags["anchor_table"] = dpg.add_child_window(width=width, height=height, border=True)
        with dpg.group(parent=self.tags["anchor_table"]):
            # Anchor strategy selection (persistent).
            #
            # This control used to live in the modal "Options" window. It is now
            # located here next to the anchor list, because it directly affects
            # how anchors are applied during chunking.
            with dpg.group(horizontal=True):
                dpg.add_text("Anchor mode")
                self.tags["anchor_mode_combo"] = dpg.add_combo(
                    items=list(ANCHOR_MODE_CHOICES),
                    default_value=str(self.state.anchor_mode),
                    width=220,
                    callback=self._on_anchor_mode_changed,
                )
                self.tags["anchor_mode_desc"] = dpg.add_text(anchor_mode_shortdesc(str(self.state.anchor_mode)))
            dpg.add_spacer(height=4)
            with dpg.table(header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp):
                dpg.add_table_column(label="Type", width_fixed=True, init_width_or_weight=40)
                dpg.add_table_column(label="Start (s)")
                dpg.add_table_column(label="Stop (s)")
                self.tags["anchor_rows"] = dpg.add_table_row()  # placeholder; we rebuild table content
        with dpg.group(horizontal=True):
            dpg.add_button(label="Delete selected", callback=lambda s, a, u=None: self._delete_selected_anchor())
            dpg.add_button(label="Clear all", callback=lambda s, a, u=None: self._clear_all_anchors())
        self.tags["anchor_selected_index"] = None
        self._refresh_anchor_table()

    # ------------------------------------------------------------------
    # File dialogs
    # ------------------------------------------------------------------

    def _build_file_dialogs(self) -> None:
        dpg = self.dpg

        def make_open_dialog(tag, extensions, callback):
            with dpg.file_dialog(
                tag=tag,
                show=False,
                directory_selector=False,
                modal=True,
                callback=callback,
                cancel_callback=lambda s, a, u=None: None,
                width=700,
                height=400,
            ):
                for ext, label in extensions:
                    dpg.add_file_extension(ext, color=(150, 255, 150, 255), custom_text=label)

        def make_save_dialog(tag, extensions, callback):
            with dpg.file_dialog(
                tag=tag,
                show=False,
                directory_selector=False,
                modal=True,
                callback=callback,
                cancel_callback=lambda s, a, u=None: None,
                width=700,
                height=400,
            ):
                for ext, label in extensions:
                    dpg.add_file_extension(ext, color=(150, 255, 150, 255), custom_text=label)

        make_open_dialog(
            "dlg_open_audio",
            [(".wav", "WAV"), (".flac", "FLAC"), (".mp3", "MP3"), (".*", "All")],
            lambda s, a: self._on_open_audio_selected(s, a),
        )
        make_open_dialog(
            "dlg_open_text",
            [(".txt", "Text"), (".*", "All")],
            lambda s, a: self._on_open_text_selected(s, a),
        )
        make_save_dialog(
            "dlg_save_text",
            [(".txt", "Text")],
            lambda s, a: self._on_save_text_selected(s, a),
        )
        make_save_dialog(
            "dlg_save_snippet",
            [(".py", "Python")],
            lambda s, a: self._on_save_snippet_selected(s, a),
        )
        make_save_dialog(
            "dlg_save_target",
            [(".wav", "WAV")],
            lambda s, a: self._on_save_target_selected(s, a),
        )
        make_save_dialog(
            "dlg_save_residual",
            [(".wav", "WAV")],
            lambda s, a: self._on_save_residual_selected(s, a),
        )

    def _show_dialog(self, tag: str) -> None:
        """Show a pre-created file dialog."""
        self.dpg.show_item(tag)

    # ------------------------------------------------------------------
    # Callbacks: Input file
    # ------------------------------------------------------------------

    def _on_input_path_changed(self, sender, app_data) -> None:
        self.state.input_path = str(app_data)
        self._refresh_backend_call_preview()

    def _on_open_audio_selected(self, sender, app_data) -> None:
        # app_data has keys: file_path_name, file_name, current_path
        path = app_data.get("file_path_name") or ""
        if path:
            self.dpg.set_value(self.tags["input_path"], path)
            self._load_input_file(path)

    def _load_input_file(self, path: str, quiet: bool = False) -> None:
        path = _norm_path(path)
        if not path or not Path(path).exists():
            if not quiet:
                self._set_status("Input file not found")
            return
        try:
            audio, sr = load_audio_file(path)
            self.state.input_audio = audio
            self.state.input_sr = int(sr)
            self.state.input_path = path

            # Initialize non-destructive edit state for the input buffer.
            # This enables AudioFX plugins (and undo/redo) on the input stream.
            self._set_output_state_from_file("input", audio, sr)

            # Reset "processed input" temp file bookkeeping.
            self._input_audio_dirty = False
            self._input_edit_rev = 0
            self._input_temp_written_rev = -1
            self._input_temp_path = None

            # Auto-load anchors sidecar if present
            # ----------------------------------
            # Anchors are persisted as a JSON sidecar next to the input file.
            # When a matching file exists, we replace the current in-memory
            # anchors with the stored anchors.
            try:
                loaded = load_anchors_for_audio(path)
            except Exception:
                loaded = None
            if loaded is not None:
                self.state.anchors = list(loaded)
                # Reset selection state to avoid referencing stale indices
                self.tags["anchor_selected_index"] = None


            # Waveform + player
            self.player_input.load_buffer(audio, sr)
            self.wave_input.set_waveform(mono_mix(audio), sr)
            self.wave_input.set_anchors(self.state.anchors)
            self.wave_input.clear_markers()

            # Refresh anchor table view after potential auto-load
            self._refresh_anchor_table()

            # Update input edit UI (Undo/Redo/Peak)
            self._update_output_edit_ui("input")

            self._set_status(f"Loaded input: {path}")
            self._refresh_backend_call_preview()

            # Auto-load matching output files, if they exist.
            # ------------------------------------------------
            # Many users iterate on a single input file and expect the latest
            # generated outputs to be shown immediately when they reopen the
            # input. We therefore look for sibling files following the common
            # naming pattern:
            #   <stem>_target.wav
            #   <stem>_residual.wav
            # and load them into the Output tab.
            self._auto_load_outputs_for_input(path)

            # Inform about anchor file loading if present
            if loaded is not None:
                side = anchor_sidecar_path(path)
                if len(loaded) > 0:
                    self._set_status(f"Loaded input: {path}  |  Anchors loaded: {side.name}")
                else:
                    self._set_status(f"Loaded input: {path}  |  Anchor file present (empty): {side.name}")

            # Persist session
            self.settings.data.setdefault("session", {})["last_audio"] = path
            self.settings.save()
        except Exception as e:
            self._show_error(f"Failed to load input audio: {e}")

    def _auto_load_outputs_for_input(self, input_path: str) -> None:
        """Auto-load sibling output files for a given input.

        The GUI uses the same naming convention as the CLI/backend defaults:
        for an input file ``foo.wav`` the outputs are typically
        ``foo_target.wav`` and ``foo_residual.wav``.

        This method is intentionally conservative:
        - It only updates a given output if the sibling file exists.
        - It does not clear an already loaded output if the sibling file is
          missing.
        """
        try:
            p = Path(input_path)
            if not p.exists():
                return

            t = p.with_name(p.stem + "_target.wav")
            r = p.with_name(p.stem + "_residual.wav")

            changed = False
            if t.exists():
                self.state.out_target_path = str(t)
                changed = True
            if r.exists():
                self.state.out_residual_path = str(r)
                changed = True

            if changed:
                self._load_outputs()
                parts = []
                if t.exists():
                    parts.append(t.name)
                if r.exists():
                    parts.append(r.name)
                self._set_status("Auto-loaded outputs: " + ", ".join(parts))
        except Exception:
            # Never fail input loading because of output auto-load.
            return

    # ------------------------------------------------------------------
    # Callbacks: Description
    # ------------------------------------------------------------------

    def _on_description_changed(self, sender, app_data) -> None:
        self.state.description = str(app_data)
        self._refresh_backend_call_preview()
        self.settings.data.setdefault("session", {})["last_description"] = self.state.description
        self.settings.save()

    def _on_open_text_selected(self, sender, app_data) -> None:
        path = app_data.get("file_path_name") or ""
        if not path:
            return
        try:
            txt = Path(path).read_text(encoding="utf-8")
            self.dpg.set_value(self.tags["description"], txt)
            self._set_status(f"Loaded description: {path}")
        except Exception as e:
            self._show_error(f"Failed to load description file: {e}")

    def _on_save_text_selected(self, sender, app_data) -> None:
        path = app_data.get("file_path_name") or ""
        if not path:
            return
        try:
            Path(path).write_text(self.state.description, encoding="utf-8")
            self._set_status(f"Saved description: {path}")
        except Exception as e:
            self._show_error(f"Failed to save description file: {e}")


    def _on_save_snippet_selected(self, sender, app_data) -> None:
        path = app_data.get("file_path_name") or ""
        if not path:
            return
        try:
            p = Path(path)
            if p.suffix.lower() != ".py":
                p = p.with_suffix(".py")

            txt = str(getattr(self, "_backend_snippet_text", "") or "")

            if not txt:
                # Best-effort fallback
                try:
                    cfg = self._build_config()
                    txt = self._format_backend_call(cfg, original_audio=str(self.state.input_path))
                except Exception:
                    txt = ""

            if not txt:
                self._set_status("Nothing to save (missing input/description?)")
                return

            p.write_text(txt, encoding="utf-8")
            self._set_status(f"Saved snippet: {str(p)}")
        except Exception as e:
            self._show_error(f"Failed to save snippet: {e}")


    # ------------------------------------------------------------------
    # Callbacks: Chunking
    # ------------------------------------------------------------------

    def _on_chunking_toggle(self, sender, app_data) -> None:
        self.state.use_chunking = bool(app_data)
        self._update_chunking_enable_state()
        self._refresh_backend_call_preview()

    def _on_chunking_params_changed(self, sender, app_data) -> None:
        self.state.max_len_s = _safe_float(self.dpg.get_value(self.tags["max_len_s"]), self.state.max_len_s)
        self.state.overlap_s = _safe_float(self.dpg.get_value(self.tags["overlap_s"]), self.state.overlap_s)
        self._refresh_backend_call_preview()

    def _update_chunking_enable_state(self) -> None:
        # When chunking is disabled, we still keep the numeric values but they
        # are ignored when building the Config.
        en = bool(self.state.use_chunking)
        self.dpg.configure_item(self.tags["max_len_s"], enabled=en)
        self.dpg.configure_item(self.tags["overlap_s"], enabled=en)

    def _save_chunking_global(self) -> None:
        self.settings.data.setdefault("chunking_global", {})["use_chunking"] = bool(self.state.use_chunking)
        self.settings.data.setdefault("chunking_global", {})["max_len_s"] = float(self.state.max_len_s)
        self.settings.data.setdefault("chunking_global", {})["overlap_s"] = float(self.state.overlap_s)
        self.settings.save()
        self._set_status("Saved global chunking defaults")

    def _load_chunking_global(self) -> None:
        cg = self.settings.data.get("chunking_global", {})
        self.state.use_chunking = bool(cg.get("use_chunking", False))
        self.state.max_len_s = float(cg.get("max_len_s", 15.0))
        self.state.overlap_s = float(cg.get("overlap_s", 2.0))
        self.dpg.set_value(self.tags["use_chunking"], self.state.use_chunking)
        self.dpg.set_value(self.tags["max_len_s"], float(self.state.max_len_s))
        self.dpg.set_value(self.tags["overlap_s"], float(self.state.overlap_s))
        self._update_chunking_enable_state()
        self._refresh_backend_call_preview()
        self._set_status("Loaded global chunking defaults")

    # ------------------------------------------------------------------
    # Markers + Anchors
    # ------------------------------------------------------------------

    def _on_set_marker_start(self) -> None:
        # Prefer the *player* position as source of truth.
        #
        # Rationale:
        # The waveform playhead is usually synced from the AudioPlayer in the
        # per-frame hook. In edge cases (e.g. immediately after a click/seek),
        # the widget and player can be out-of-sync for a single frame.
        # Using the player state ensures "Set Start/Stop" always uses the
        # position that the transport backend will actually play from.
        st = self.player_input.get_state()
        if st.total_samples > 0 and st.sample_rate > 0:
            t = float(st.position_samples) / float(st.sample_rate)
        else:
            t = self.wave_input.get_playhead_seconds()
        self.wave_input.set_marker_start_seconds(t)
        self._set_status(f"Marker start set to {t:.2f}s")

    def _on_set_marker_stop(self) -> None:
        st = self.player_input.get_state()
        if st.total_samples > 0 and st.sample_rate > 0:
            t = float(st.position_samples) / float(st.sample_rate)
        else:
            t = self.wave_input.get_playhead_seconds()
        self.wave_input.set_marker_stop_seconds(t)
        self._set_status(f"Marker stop set to {t:.2f}s")

    def _delete_anchor_at_playhead(self) -> None:
        """Delete the anchor that currently contains the input playhead.

        If multiple anchors overlap the playhead, the first match is removed.
        This operation is intended as a quick "fix" tool while stepping
        through audio.
        """
        st = self.player_input.get_state()
        if st.total_samples > 0 and st.sample_rate > 0:
            t = float(st.position_samples) / float(st.sample_rate)
        else:
            t = self.wave_input.get_playhead_seconds()

        idx = None
        for i, a in enumerate(self.state.anchors):
            if float(a.start_s) <= t <= float(a.end_s):
                idx = i
                break
        if idx is None:
            self._set_status("Playhead is not inside an anchor")
            return

        try:
            removed = self.state.anchors.pop(int(idx))
            self.wave_input.set_anchors(self.state.anchors)
            self._refresh_anchor_table()
            self._refresh_backend_call_preview()
            self._autosave_anchors_sidecar()
            self._set_status(f"Removed anchor {removed.sign}: {removed.start_s:.2f}s..{removed.end_s:.2f}s")
        except Exception as e:
            self._show_error(f"Failed to remove anchor: {e}")

    def _on_clear_markers(self) -> None:
        self.wave_input.clear_markers()
        self._set_status("Markers cleared")

    # ------------------------------------------------------------------
    # Anchor persistence (sidecar file)
    # ------------------------------------------------------------------

    def _autosave_anchors_sidecar(self) -> None:
        """Persist anchors next to the current input audio.

        This is a best-effort operation: failures are reported to the user
        but must not crash the GUI.
        """
        try:
            if not self.state.input_path:
                return
            if not Path(self.state.input_path).exists():
                return
            save_anchors_for_audio(self.state.input_path, list(self.state.anchors))
        except Exception as e:
            # Keep errors visible but non-fatal.
            self._set_status(f"Warning: failed to save anchors sidecar: {e}")

    def _send_marker_to_anchor(self, sign: str) -> None:
        mr = self.wave_input.get_marker_range()
        if mr is None:
            self._set_status("Please set Start and Stop markers first")
            return
        s, e = mr
        if sign not in ("+", "-"):
            return
        self.state.anchors.append(AnchorItem(sign=sign, start_s=float(s), end_s=float(e)))
        self.wave_input.set_anchors(self.state.anchors)
        self._refresh_anchor_table()
        self._refresh_backend_call_preview()
        self._autosave_anchors_sidecar()
        self._set_status(f"Added anchor {sign}: {s:.2f}s..{e:.2f}s")

    def _refresh_anchor_table(self) -> None:
        """Refresh the anchors table *and* the anchor-mode selector.

        Important
        ---------
        The anchor table lives inside a scrollable child window. Earlier
        iterations refreshed the table by clearing the entire child window and
        rebuilding only the table rows.

        Since the *anchor mode* selector is located **next to the anchor list**
        (and is stored inside the same child window), clearing the child window
        would also delete the selector, making it disappear after the first
        refresh (e.g. after loading an input file).

        This method therefore rebuilds the complete content of the anchors
        child window: selector + table.
        """
        dpg = self.dpg
        if "anchor_table" not in self.tags:
            return

        # Clear only the children of the anchor table container.
        dpg.delete_item(self.tags["anchor_table"], children_only=True)

        with dpg.group(parent=self.tags["anchor_table"]):
            # Anchor strategy selection (persistent).
            with dpg.group(horizontal=True):
                dpg.add_text("Anchor mode")
                self.tags["anchor_mode_combo"] = dpg.add_combo(
                    items=list(ANCHOR_MODE_CHOICES),
                    default_value=str(self.state.anchor_mode),
                    width=220,
                    callback=self._on_anchor_mode_changed,
                )
                self.tags["anchor_mode_desc"] = dpg.add_text(anchor_mode_shortdesc(str(self.state.anchor_mode)))

            dpg.add_spacer(height=4)

            with dpg.table(header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp):
                dpg.add_table_column(label="#", width_fixed=True, init_width_or_weight=30)
                dpg.add_table_column(label="Type", width_fixed=True, init_width_or_weight=40)
                dpg.add_table_column(label="Start (s)")
                dpg.add_table_column(label="Stop (s)")

                for idx, a in enumerate(self.state.anchors):
                    with dpg.table_row():
                        # Use a small selectable in the first column to
                        # implement selection without hiding the other
                        # columns. (Using span_columns=True would cover the
                        # entire row and make the remaining cells invisible.)
                        dpg.add_selectable(
                            label=str(idx + 1),
                            # DearPyGui usually calls item callbacks with
                            # (sender, app_data, user_data). If a callback
                            # only accepts two parameters, the optional
                            # user_data parameter is still passed as the
                            # third positional argument (often `None`).
                            #
                            # In earlier builds we used a 3-arg lambda with a
                            # default parameter capturing idx. Unfortunately,
                            # DearPyGui's third argument overwrote this default
                            # with `None`, leading to:
                            #   TypeError: int() argument ... not 'NoneType'
                            #
                            # We therefore accept the 3rd argument explicitly
                            # and capture `idx` in a 4th default parameter.
                            callback=lambda s, v, ud, u=idx: self._on_select_anchor(u),
                        )
                        dpg.add_text(a.sign)
                        dpg.add_text(f"{a.start_s:.3f}")
                        dpg.add_text(f"{a.end_s:.3f}")

        # Reset selection after rebuild.
        self.tags["anchor_selected_index"] = None




    def _on_select_anchor(self, idx: int) -> None:
        # Defensive: in case a callback accidentally routes `None` here
        # (e.g. due to a widget or API change), do not crash the GUI.
        if idx is None:
            self.tags["anchor_selected_index"] = None
            return

        self.tags["anchor_selected_index"] = int(idx)
        # Optionally: jump playhead to anchor start
        try:
            a = self.state.anchors[idx]
            self._seek("input", float(a.start_s))
            self._set_status(f"Selected anchor #{idx + 1}")
        except Exception:
            return


    def _on_anchor_mode_changed(self, sender, app_data, user_data=None) -> None:
        """Handle changes of the anchor strategy (aka *anchor mode*).

        Why this exists
        ---------------
        SAM-Audio can use **temporal anchors** (span prompting) to guide the model
        towards or away from specific time regions. When the input audio is
        processed in chunks, the backend must decide *how to apply* anchors for
        each chunk. This choice is called **anchor mode**.

        The GUI stores the selected mode persistently in ``settings.json`` under
        ``backend.anchor_mode`` and also updates the in-memory :class:`AppState`.
        The next backend run will therefore use the same mode, even after the GUI
        is restarted.

        DearPyGui callback signature
        ----------------------------
        ``callback(sender, app_data, user_data)`` where ``app_data`` holds the
        new combo value as a string.

        Parameters
        ----------
        sender:
            DearPyGui item id (unused).
        app_data:
            The selected anchor mode string.
        user_data:
            Optional user data (unused).
        """
        _ = sender, user_data  # unused

        new_mode = str(app_data) if app_data is not None else ""

        # Defensive: only accept known modes.
        if new_mode not in ANCHOR_MODE_CHOICES:
            # Revert selection to the last known value.
            try:
                tag = self.tags.get("anchor_mode_combo")
                if tag:
                    self.dpg.set_value(tag, str(self.state.anchor_mode))
            except Exception:
                pass
            self._set_status("Invalid anchor mode ignored")
            return

        # Update runtime state.
        self.state.anchor_mode = new_mode

        # Update the short description next to the dropdown.
        try:
            tag = self.tags.get("anchor_mode_desc")
            if tag:
                self.dpg.set_value(tag, anchor_mode_shortdesc(new_mode))
        except Exception:
            pass

        # Persist for the next session.
        b = self.settings.data.setdefault("backend", {})
        b["anchor_mode"] = new_mode
        try:
            self.settings.save()
        except Exception:
            pass

        # Keep the backend-call preview in sync (if visible).
        try:
            self._refresh_backend_call_preview()
        except Exception:
            pass

        self._set_status(f"Anchor mode set: {new_mode}")


    def _delete_selected_anchor(self) -> None:
        idx = self.tags.get("anchor_selected_index")
        if idx is None:
            self._set_status("No anchor selected")
            return
        try:
            self.state.anchors.pop(int(idx))
            self.wave_input.set_anchors(self.state.anchors)
            self._refresh_anchor_table()
            self._refresh_backend_call_preview()
            self._autosave_anchors_sidecar()
            self._set_status("Anchor deleted")
        except Exception:
            self._show_error("Failed to delete anchor")

    def _clear_all_anchors(self) -> None:
        self.state.anchors.clear()
        self.wave_input.set_anchors(self.state.anchors)
        self._refresh_anchor_table()
        self._refresh_backend_call_preview()
        self._autosave_anchors_sidecar()
        self._set_status("All anchors cleared")

    # ------------------------------------------------------------------
    # Transport + seek plumbing
    # ------------------------------------------------------------------

    def _on_input_seek(self, t: float) -> None:
        self._seek("input", t)

    def _on_target_seek(self, t: float) -> None:
        self._seek("target", t)

    def _on_residual_seek(self, t: float) -> None:
        self._seek("residual", t)

    def _player_and_wave(self, kind: str) -> Tuple[AudioPlayer, WaveformWidget]:
        if kind == "input":
            return self.player_input, self.wave_input
        if kind == "target":
            return self.player_target, self.wave_target
        return self.player_residual, self.wave_residual

    def _play(self, kind: str) -> None:
        """Start playback for the given stream (input/target/residual).

        ``AudioPlayer.play()`` is intentionally best-effort and does not raise on
        PortAudio device/format errors. Instead it stores an error string.
       
        To make failures visible to the user we surface that error here.
        """
        p, _w = self._player_and_wave(kind)
        try:
            p.play()
        except Exception as e:
            self._show_error(f"Audio playback error: {e}")
            return

        # Surface silent failures (e.g., device busy, unsupported sample rate).
        err = p.get_last_error()
        if err:
            self._set_status(f"Playback error ({kind}): {err}")
            # Also show as a dialog (best effort).
            try:
                self._show_error(f"Playback error ({kind}):\n\n{err}")
            except Exception:
                pass

    def _stop(self, kind: str) -> None:
        p, _w = self._player_and_wave(kind)
        try:
            p.stop()
        except Exception:
            pass

    def _rewind(self, kind: str) -> None:
        p, w = self._player_and_wave(kind)
        try:
            p.rewind()
            w.set_playhead_seconds(0.0)
        except Exception as e:
            self._show_error(f"Audio control error: {e}")

    def _step(self, kind: str, delta: float) -> None:
        p, w = self._player_and_wave(kind)
        try:
            p.step_seconds(delta)
            st = p.get_state()
            t = float(st.position_samples) / float(max(1, st.sample_rate))
            w.set_playhead_seconds(t)
        except Exception as e:
            self._show_error(f"Audio control error: {e}")

    def _seek(self, kind: str, t: float) -> None:
        p, w = self._player_and_wave(kind)
        try:
            p.seek_seconds(t)
            w.set_playhead_seconds(t)
        except Exception as e:
            self._show_error(f"Audio control error: {e}")

    # ------------------------------------------------------------------
    # Run / Abort
    # ------------------------------------------------------------------

    def _on_run(self) -> None:
        if self.worker.is_running():
            return
        if not self.state.input_path or not Path(self.state.input_path).exists():
            self._set_status("Please select an input file")
            return
        if not self.state.description.strip():
            self._set_status("Please enter a description")
            return

        cfg = self._build_config()

        self.dpg.configure_item(self.tags["run_btn"], enabled=False)
        self.dpg.configure_item(self.tags["abort_btn"], enabled=True)
        self.dpg.set_value(self.tags["progress"], 0.0)
        self._set_progress_labels(0.0, "Starting…")
        self._set_status("Running backend...")

        self.worker.start(cfg)

    def _on_abort(self) -> None:
        if not self.worker.is_running():
            return
        self.worker.cancel()
        self._set_status("Abort requested")

    def poll_worker_events(self) -> None:
        """Poll events from the worker queue and update GUI.

        Call this from the application's render callback.
        """
        q = self.worker.queue
        updated = False
        while True:
            try:
                ev = q.get_nowait()
            except Exception:
                break
            self._handle_worker_event(ev)
            updated = True
        if updated:
            self._refresh_backend_call_preview()

    def _handle_worker_event(self, ev: WorkerEvent) -> None:
        dpg = self.dpg
        if ev.kind == "progress":
            p = max(0.0, min(100.0, float(ev.percent)))
            dpg.set_value(self.tags["progress"], p / 100.0)
            self._set_progress_labels(p, ev.message)
            return

        # Terminal states
        dpg.configure_item(self.tags["run_btn"], enabled=True)
        dpg.configure_item(self.tags["abort_btn"], enabled=False)

        try:
            self._refresh_run_model_dropdown(keep_selection=True)
        except Exception:
            pass

        if ev.kind == "done":
            self.state.out_target_path = ev.out_target or ""
            self.state.out_residual_path = ev.out_residual or ""
            dpg.set_value(self.tags["progress"], 1.0)
            self._set_progress_labels(100.0, "Done")
            self._set_status("Done")
            self._load_outputs()
        elif ev.kind == "cancelled":
            self._set_progress_labels(0.0, "Cancelled")
            self._set_status("Cancelled")
        elif ev.kind == "error":
            self._set_progress_labels(0.0, "Error")
            self._set_status("Error")
            self._show_error(ev.traceback_text or "Unknown error")


    def _set_progress_labels(self, percent: float, phase: str) -> None:
        """Update the Run-tab progress labels.

        UI requirement (PATCHBAY):
        - show the percentage *alone* centered under the progress bar
        - show the current phase name centered *below* the percentage and allow
          line wrapping if needed.

        Implementation notes:
        - We use regular text widgets inside a small child_window and position
          them manually for reliable rendering/centering across DearPyGui builds.
        - The previously used drawlist approach remains as a backward compatible
          fallback if the new widgets are not present.
        """
        dpg = self.dpg

        # Clamp and normalize.
        p = max(0.0, min(100.0, float(percent)))
        pct_text = f"{p:0.1f}%"
        phase = (phase or "").strip()

        # Cache last values so we can re-center on viewport resize.
        self._ui_progress_percent = p
        self._ui_progress_phase = phase

        # --- Preferred implementation: manual-centering text widgets ---
        pct_tag = self.tags.get("progress_pct_text")
        cw_tag = self.tags.get("progress_labels_cw")
        phase_tags = self.tags.get("progress_phase_lines")  # list[str] or None

        if pct_tag and dpg.does_item_exist(pct_tag) and cw_tag and dpg.does_item_exist(cw_tag) and isinstance(phase_tags, list):
            # Determine available width. Prefer progress bar rect size (same column).
            w = 0
            try:
                pb = self.tags.get("progress")
                if pb and dpg.does_item_exist(pb):
                    w, _ = dpg.get_item_rect_size(pb)
            except Exception:
                w = 0
            if not w or w <= 4:
                try:
                    w, _ = dpg.get_item_rect_size(cw_tag)
                except Exception:
                    w = 0
            if not w or w <= 4:
                try:
                    w = int(dpg.get_viewport_client_width())
                except Exception:
                    w = 400

            # Keep the label child window aligned to the progress bar width.
            try:
                dpg.configure_item(cw_tag, width=int(max(80, w)))
            except Exception:
                pass

            # Prefer the *actual* child-window width once DearPyGui has laid out the row.
            # Some builds return 0 for get_item_rect_size(progress) at this point.
            try:
                _cw_w, _cw_h = dpg.get_item_rect_size(cw_tag)
                if _cw_w and _cw_w > 10:
                    w = int(_cw_w)
            except Exception:
                pass

            # Wrap helper (pixel-based).
            def _wrap_to_width(text0: str, max_px: float) -> list[str]:
                if not text0:
                    return []
                parts = text0.splitlines() or [text0]
                out: list[str] = []
                for part in parts:
                    words = part.split(" ")
                    line = ""
                    for w0 in words:
                        if not w0:
                            continue
                        cand = w0 if not line else f"{line} {w0}"
                        try:
                            tw, _ = dpg.get_text_size(cand)
                        except Exception:
                            tw = len(cand) * 7
                        if tw <= max_px or not line:
                            line = cand
                        else:
                            out.append(line)
                            line = w0
                    if line:
                        out.append(line)

                # Break very long tokens if needed.
                final: list[str] = []
                for line in out:
                    try:
                        tw, _ = dpg.get_text_size(line)
                    except Exception:
                        tw = len(line) * 7
                    if tw <= max_px:
                        final.append(line)
                        continue
                    buf = ""
                    for ch in line:
                        cand = buf + ch
                        try:
                            cw, _ = dpg.get_text_size(cand)
                        except Exception:
                            cw = len(cand) * 7
                        if cw <= max_px or not buf:
                            buf = cand
                        else:
                            final.append(buf)
                            buf = ch
                    if buf:
                        final.append(buf)
                return final

            margin_x = 6.0
            max_w = max(40.0, float(w) - 2.0 * margin_x)

            # Percent centered (single line).
            try:
                pct_w, pct_h = dpg.get_text_size(pct_text)
            except Exception:
                pct_w, pct_h = (len(pct_text) * 7, 14)
            pct_x = max(0.0, (float(w) - float(pct_w)) / 2.0)
            # Clamp into visible range in case the width estimate overshoots.
            pct_x = max(0.0, min(pct_x, max(0.0, float(w) - float(pct_w))))
            pct_y = 2.0
            try:
                dpg.set_value(pct_tag, pct_text)
            except Exception:
                pass
            try:
                dpg.set_item_pos(pct_tag, [pct_x, pct_y])
                dpg.configure_item(pct_tag, show=True)
            except Exception:
                pass

            # Phase lines (wrapped) centered below.
            phase_lines = _wrap_to_width(phase, max_w)
            y = pct_y + float(pct_h) + 6.0
            shown = 0
            for i, ln in enumerate(phase_lines):
                if i >= len(phase_tags):
                    break
                tag = phase_tags[i]
                if not dpg.does_item_exist(tag):
                    continue
                try:
                    lw, lh = dpg.get_text_size(ln)
                except Exception:
                    lw, lh = (len(ln) * 7, 14)
                x = max(0.0, (float(w) - float(lw)) / 2.0)
                x = max(0.0, min(x, max(0.0, float(w) - float(lw))))
                try:
                    dpg.set_value(tag, ln)
                    dpg.set_item_pos(tag, [x, y])
                    dpg.configure_item(tag, show=True)
                except Exception:
                    pass
                y += float(lh) + 2.0
                shown += 1

            # Hide remaining placeholder lines.
            for j in range(shown, len(phase_tags)):
                tag = phase_tags[j]
                if tag and dpg.does_item_exist(tag):
                    try:
                        dpg.set_value(tag, "")
                        dpg.configure_item(tag, show=False)
                    except Exception:
                        pass
            return

        # --- Backward-compatible fallback: drawlist-based labels (older builds) ---
        dl = self.tags.get("progress_labels_dl")
        if dl and dpg.does_item_exist(dl):
            try:
                w, h = dpg.get_item_rect_size(dl)
            except Exception:
                w, h = 0, 0
            if w <= 4:
                pb = self.tags.get("progress")
                if pb and dpg.does_item_exist(pb):
                    try:
                        w, _ = dpg.get_item_rect_size(pb)
                        if w > 4:
                            dpg.configure_item(dl, width=int(w))
                    except Exception:
                        pass
            if h <= 4:
                try:
                    dpg.configure_item(dl, height=64)
                except Exception:
                    pass
                h = 64

            # Wrap helper.
            def _wrap_to_width(text0: str, max_px: float) -> list[str]:
                if not text0:
                    return []
                parts = text0.splitlines() or [text0]
                out: list[str] = []
                for part in parts:
                    words = part.split(" ")
                    line = ""
                    for w0 in words:
                        if not w0:
                            continue
                        cand = w0 if not line else f"{line} {w0}"
                        try:
                            tw, _ = dpg.get_text_size(cand)
                        except Exception:
                            tw = len(cand) * 7
                        if tw <= max_px or not line:
                            line = cand
                        else:
                            out.append(line)
                            line = w0
                    if line:
                        out.append(line)
                return out

            try:
                dpg.delete_item(dl, children_only=True)
            except Exception:
                pass

            margin_x = 6.0
            max_w = max(40.0, float(w) - 2.0 * margin_x)

            try:
                pct_w, pct_h = dpg.get_text_size(pct_text)
            except Exception:
                pct_w, pct_h = (len(pct_text) * 7, 14)
            pct_x = max(0.0, (float(w) - float(pct_w)) / 2.0)
            pct_y = 2.0

            # Ensure parent is provided.
            dpg.draw_text((pct_x, pct_y), pct_text, color=(255, 255, 255, 255), parent=dl)
            phase_lines = _wrap_to_width(phase, max_w)
            y = pct_y + float(pct_h) + 6.0
            for ln in phase_lines:
                try:
                    lw, lh = dpg.get_text_size(ln)
                except Exception:
                    lw, lh = (len(ln) * 7, 14)
                x = max(0.0, (float(w) - float(lw)) / 2.0)
                dpg.draw_text((x, y), ln, color=(255, 255, 255, 255), parent=dl)
                y += float(lh) + 2.0
            return

        # Final fallback: legacy one-line text widget.
        t = self.tags.get("progress_text")
        if t and dpg.does_item_exist(t):
            try:
                dpg.set_value(t, f"{p:0.1f}%  {phase}")
            except Exception:
                pass

    def _refresh_run_model_dropdown(self, *, keep_selection: bool = True) -> None:
        """Rebuild model dropdown items and (lokal)/(online) markers."""
        try:
            labels, mapping, default_label = build_model_dropdown(str(self.state.model), timeout_s=2.0)
        except Exception:
            return

        self._run_model_labels = labels
        self._run_model_label_to_id = mapping
        self._run_model_default_label = default_label

        if "run_model" not in self.tags:
            return

        dpg = self.dpg
        try:
            # Avoid re-entrant callbacks if DearPyGui triggers on set_value
            self._updating_model_combo = True
            dpg.configure_item(self.tags["run_model"], items=list(labels))
            if keep_selection:
                # Keep current selection if possible
                cur_label = None
                for lab, rid in mapping.items():
                    if rid == str(self.state.model):
                        cur_label = lab
                        break
                if cur_label is None:
                    cur_label = default_label
                dpg.set_value(self.tags["run_model"], cur_label)
            else:
                dpg.set_value(self.tags["run_model"], default_label)
        finally:
            try:
                self._updating_model_combo = False
            except Exception:
                pass

    def _on_run_model_changed(self, selected_label: str) -> None:
        """Handle model dropdown changes in the Run area."""
        if getattr(self, "_updating_model_combo", False):
            return

        lab = str(selected_label or "").strip()
        if not lab:
            return
        repo_id = self._run_model_label_to_id.get(lab)
        if not repo_id:
            # Fallback: strip the "(lokal)/(online)" suffix
            repo_id = lab.split(" (", 1)[0].strip()
        if not repo_id:
            return

        self.state.model = str(repo_id)

        # Persist immediately (this replaces the old Options->Model entry)
        try:
            b = self.settings.data.setdefault("backend", {})
            b["model"] = str(repo_id)
            self.settings.save()
        except Exception:
            pass

        # Update suffix markers and backend call preview
        self._refresh_run_model_dropdown(keep_selection=True)
        self._refresh_backend_call_preview()


    def _persist_backend_setting(self, key: str, value) -> None:
        """Persist a backend-related setting to settings.json (best-effort)."""
        try:
            b = self.settings.data.setdefault("backend", {})
            b[key] = value
            self.settings.save()
        except Exception:
            pass

    def _on_run_device_changed(self, sender, app_data) -> None:
        self.state.device = str(app_data)
        self._persist_backend_setting("device", str(self.state.device))
        self._refresh_backend_call_preview()

    def _on_run_fp16_changed(self, sender, app_data) -> None:
        self.state.fp16 = bool(app_data)
        self._persist_backend_setting("fp16", bool(self.state.fp16))
        self._refresh_backend_call_preview()

    def _on_run_predict_changed(self, sender, app_data) -> None:
        self.state.predict_spans = bool(app_data)
        self._persist_backend_setting("predict_spans", bool(self.state.predict_spans))
        self._refresh_backend_call_preview()

    def _on_run_rerank_changed(self, sender, app_data) -> None:
        try:
            v = int(app_data)
        except Exception:
            try:
                v = int(self.dpg.get_value(self.tags.get("run_rerank")))
            except Exception:
                v = int(self.state.reranking_candidates)
        v = max(1, int(v))
        self.state.reranking_candidates = v
        self._persist_backend_setting("reranking_candidates", int(self.state.reranking_candidates))
        self._refresh_backend_call_preview()

    def _on_run_no_resample_changed(self, sender, app_data) -> None:
        self.state.no_resample = bool(app_data)
        self._persist_backend_setting("no_resample", bool(self.state.no_resample))
        self._refresh_backend_call_preview()


    def _build_config(self) -> Config:
        anchors: List[Anchor] = [(a.sign, float(a.start_s), float(a.end_s)) for a in self.state.anchors]

        max_len = float(self.state.max_len_s) if self.state.use_chunking else None
        overlap = float(self.state.overlap_s) if self.state.use_chunking else None

        # Backend audio path:
        # If the user edited the input stream via AudioFX, we run the backend on a
        # temp WAV while keeping output naming anchored to the *original* input file.
        original_audio_path = str(self.state.input_path)
        audio_for_backend = original_audio_path
        try:
            if getattr(self, "_input_audio_dirty", False):
                try:
                    self._sync_input_temp_file()
                except Exception:
                    pass
                if getattr(self, "_input_temp_path", None):
                    audio_for_backend = str(self._input_temp_path)
        except Exception:
            pass

        # Keep default outputs next to the original input file so the GUI's
        # auto-load pattern (<stem>_target.wav / <stem>_residual.wav) continues
        # to work even when we run the backend on a temp audio path.
        out_target = None
        out_residual = None
        try:
            ap = Path(original_audio_path)
            if ap.name:
                out_target = str(ap.parent / f"{ap.stem}_target.wav")
                out_residual = str(ap.parent / f"{ap.stem}_residual.wav")
        except Exception:
            out_target = None
            out_residual = None

        cfg = Config.from_parameters(
            model=str(self.state.model),
            audio=str(audio_for_backend),
            description=str(self.state.description),
            out_target=out_target,
            out_residual=out_residual,
            predict_spans=bool(self.state.predict_spans),
            reranking_candidates=int(self.state.reranking_candidates),
            device=str(self.state.device),
            fp16=bool(self.state.fp16),
            max_len_s=max_len,
            overlap_s=overlap,
            anchor_mode=str(self.state.anchor_mode),
            anchors=anchors,
            no_resample=bool(self.state.no_resample),
            log_file=None,
            debug=False,
        )
        return cfg

    def _load_outputs(self) -> None:
        dpg = self.dpg
        if self.state.out_target_path:
            dpg.set_value(self.tags["target_path"], self.state.out_target_path)
            try:
                audio, sr = load_audio_file(self.state.out_target_path)
                self.player_target.load_buffer(audio, sr)
                self.wave_target.set_waveform(mono_mix(audio), sr)
                self._set_output_state_from_file("target", audio, sr)
            except Exception as e:
                self._show_error(f"Failed to load target: {e}")
        if self.state.out_residual_path:
            dpg.set_value(self.tags["residual_path"], self.state.out_residual_path)
            try:
                audio, sr = load_audio_file(self.state.out_residual_path)
                self.player_residual.load_buffer(audio, sr)
                self.wave_residual.set_waveform(mono_mix(audio), sr)
                self._set_output_state_from_file("residual", audio, sr)
            except Exception as e:
                self._show_error(f"Failed to load residual: {e}")

        # Update edit UI enable states and peak display
        self._update_output_edit_ui("target")
        self._update_output_edit_ui("residual")

    def _set_output_state_from_file(self, kind: str, audio: np.ndarray, sr: int) -> None:
        """Initialize output edit state from a freshly produced output file."""
        st = self._out_edit.get(kind)
        if st is None:
            return
        # Keep both original and current buffer (float32, contiguous)
        a = np.ascontiguousarray(audio.astype(np.float32, copy=False))
        st.audio = a
        st.sr = int(sr)
        st.original_audio = np.array(a, copy=True)
        st.original_sr = int(sr)
        st.clear_history()

    def _get_output_state(self, kind: str) -> OutputEditState:
        st = self._out_edit.get(kind)
        if st is None:
            raise KeyError(f"Unknown output kind: {kind}")
        return st

    def _push_undo_snapshot(self, kind: str) -> None:
        """Push a snapshot of the current buffer to the undo stack."""
        st = self._get_output_state(kind)
        if st.audio is None:
            return
        st.undo_stack.append(np.array(st.audio, copy=True))
        # Truncate history to keep memory bounded
        if self._max_undo_steps > 0 and len(st.undo_stack) > self._max_undo_steps:
            st.undo_stack = st.undo_stack[-self._max_undo_steps :]
        st.redo_stack.clear()

    def _set_output_audio(self, kind: str, audio: np.ndarray, sr: int, *, mark_input_dirty: bool = True) -> None:
        """Replace the current buffer and refresh player + waveform.

        This function is used for all three streams:
          - input
          - target
          - residual
        """
        st = self._get_output_state(kind)

        # Preserve current playhead position (best effort) so edits feel
        # non-disruptive.
        try:
            if kind == "input":
                cur_t = float(self.wave_input.get_playhead_seconds())
            elif kind == "target":
                cur_t = float(self.wave_target.get_playhead_seconds())
            else:
                cur_t = float(self.wave_residual.get_playhead_seconds())
        except Exception:
            cur_t = 0.0

        a = np.ascontiguousarray(audio.astype(np.float32, copy=False))
        st.audio = a
        st.sr = int(sr)

        # Update playback + waveform
        if kind == "input":
            # Keep state mirrors for convenience (not heavily used elsewhere)
            self.state.input_audio = a
            self.state.input_sr = int(sr)

            self.player_input.load_buffer(a, st.sr)
            self.wave_input.set_waveform(mono_mix(a), st.sr)
            try:
                self.wave_input.set_anchors(self.state.anchors)
            except Exception:
                pass
            try:
                self.player_input.seek_seconds(cur_t)
                self.wave_input.set_playhead_seconds(cur_t)
            except Exception:
                pass

            # Input edits may be used by the backend via a temp WAV.
            self._input_edit_rev += 1
            if mark_input_dirty:
                self._input_audio_dirty = True
                self._sync_input_temp_file()
            else:
                self._input_audio_dirty = False
                self._input_temp_path = None
                self._input_temp_written_rev = -1

        elif kind == "target":
            self.player_target.load_buffer(a, st.sr)
            self.wave_target.set_waveform(mono_mix(a), st.sr)
            try:
                self.player_target.seek_seconds(cur_t)
                self.wave_target.set_playhead_seconds(cur_t)
            except Exception:
                pass

        else:
            self.player_residual.load_buffer(a, st.sr)
            self.wave_residual.set_waveform(mono_mix(a), st.sr)
            try:
                self.player_residual.seek_seconds(cur_t)
                self.wave_residual.set_playhead_seconds(cur_t)
            except Exception:
                pass

        self._update_output_edit_ui(kind)

    def _sync_input_temp_file(self) -> None:
        """Write the current *edited* input buffer to a temp WAV (best effort).

        The backend expects an audio file path. When the user applies AudioFX
        plugins to the input stream, we keep the original input path in the UI
        but run the backend on this temp WAV.
        """
        if not getattr(self, "_input_audio_dirty", False):
            return
        try:
            if self._input_temp_written_rev == self._input_edit_rev and self._input_temp_path:
                return
        except Exception:
            pass

        st = self._out_edit.get("input")
        if st is None or st.audio is None or st.audio.size == 0:
            return

        try:
            from .persistence import settings_path as _settings_json_path  # type: ignore
            base_dir = _settings_json_path().parent
        except Exception:
            base_dir = Path.cwd()

        try:
            tmp_dir = Path(base_dir) / "temp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return

        stem = None
        try:
            stem = Path(self.state.input_path).stem
        except Exception:
            stem = None
        if not stem:
            stem = "input"

        tmp_path = str(Path(tmp_dir) / f"{stem}_processed_input.wav")

        try:
            import soundfile as sf
            # Store float WAV to avoid unintended clipping.
            sf.write(tmp_path, st.audio, st.sr, subtype="FLOAT")
            self._input_temp_path = str(tmp_path)
            self._input_temp_written_rev = int(self._input_edit_rev)
        except Exception:
            # Keep backend usable even if temp writing fails.
            pass

    def _update_output_edit_ui(self, kind: str) -> None:
        """Enable/disable output editing buttons and update peak display."""
        dpg = self.dpg
        st = self._out_edit.get(kind)
        if st is None:
            return
        prefix = f"out_{kind}_"

        have_audio = st.audio is not None and st.audio.size > 0
        can_undo = have_audio and len(st.undo_stack) > 0
        can_redo = have_audio and len(st.redo_stack) > 0

        # Configure buttons if they exist (they are created during build())
        for key, enabled in [
            (prefix + "undo_btn", can_undo),
            (prefix + "redo_btn", can_redo),
            (prefix + "restore_btn", have_audio),
            (prefix + "norm_btn", have_audio),
            (prefix + "comp_apply_btn", have_audio),
            (prefix + "comp_thr", have_audio),
            (prefix + "comp_ratio", have_audio),
            (prefix + "comp_attack", have_audio),
            (prefix + "comp_release", have_audio),
        ]:
            if key in self.tags:
                try:
                    dpg.configure_item(self.tags[key], enabled=bool(enabled))
                except Exception:
                    pass

        # Peak display
        if prefix + "peak_text" in self.tags:
            try:
                if have_audio:
                    peak_db = describe_peak_dbfs(st.audio)
                    dpg.set_value(self.tags[prefix + "peak_text"], f"Peak: {peak_db:5.1f} dBFS")
                else:
                    dpg.set_value(self.tags[prefix + "peak_text"], "")
            except Exception:
                pass

    def _on_output_undo(self, kind: str) -> None:
        st = self._get_output_state(kind)
        if st.audio is None or not st.undo_stack:
            return
        st.redo_stack.append(np.array(st.audio, copy=True))
        prev = st.undo_stack.pop()
        self._set_output_audio(kind, prev, st.sr)

    def _on_output_redo(self, kind: str) -> None:
        st = self._get_output_state(kind)
        if st.audio is None or not st.redo_stack:
            return
        st.undo_stack.append(np.array(st.audio, copy=True))
        nxt = st.redo_stack.pop()
        self._set_output_audio(kind, nxt, st.sr)

    def _on_output_restore_original(self, kind: str) -> None:
        st = self._get_output_state(kind)
        if st.original_audio is None:
            return
        # Restore does not add to undo history; it is an explicit reset.
        st.clear_history()
        self._set_output_audio(kind, np.array(st.original_audio, copy=True), st.original_sr, mark_input_dirty=(kind != "input"))
        if kind == "input":
            self._set_status("input: restored original input")
        else:
            self._set_status(f"{kind}: restored original output")

    # Context-menu wrappers
    # ---------------------
    def _ctx_output_kind(self) -> Optional[str]:
        """Return the output kind the context menu was opened for.

        We intentionally do *not* silently default to "target" because that
        can make it look like "Apply" has no effect when the user right-clicks
        Residual but the kind could not be detected (hit-test edge cases).
        """
        if self._output_ctx_kind in ("target", "residual"):
            return self._output_ctx_kind
        self._set_status("Output context menu: no output region selected. Right-click on Target or Residual.")
        return None

    def _on_output_normalize_ctx(self) -> None:
        kind = self._ctx_output_kind()
        if kind:
            self._on_output_normalize(kind)

    def _on_output_compress_ctx(self) -> None:
        kind = self._ctx_output_kind()
        if kind:
            self._on_output_compress(kind)

    def _on_output_normalize(self, kind: str) -> None:
        st = self._get_output_state(kind)
        if st.audio is None:
            self._set_status(f"{kind}: no audio loaded to normalize")
            return
        self._push_undo_snapshot(kind)
        y = peak_normalize(st.audio, target_peak=float(self._normalize_target_peak_default))
        self._set_output_audio(kind, y, st.sr)
        self._set_status(f"{kind}: normalized")
        # Close the context menu after applying an action.
        if self._output_ctx_visible:
            self._hide_output_ctx_menu()

    def _on_output_compress(self, kind: str) -> None:
        st = self._get_output_state(kind)
        if st.audio is None:
            self._set_status(f"{kind}: no audio loaded to compress")
            return

        # Parameters are shared defaults and are edited via the settings
        # dialogs (opened from the right-click context menu).
        thr = float(self._comp_threshold_db_default)
        ratio = float(self._comp_ratio_default)
        attack_ms = float(self._comp_attack_ms_default)
        release_ms = float(self._comp_release_ms_default)

        if ratio < 1.0:
            ratio = 1.0

        # Persist the last used settings (shared between target/residual).
        self._comp_threshold_db_default = float(thr)
        self._comp_ratio_default = float(ratio)
        self._comp_attack_ms_default = float(max(0.0, attack_ms))
        self._comp_release_ms_default = float(max(0.0, release_ms))
        op = self.settings.data.setdefault("output_processing", {})
        op["compressor_threshold_db"] = float(thr)
        op["compressor_ratio"] = float(ratio)
        op["compressor_attack_ms"] = float(self._comp_attack_ms_default)
        op["compressor_release_ms"] = float(self._comp_release_ms_default)
        try:
            self.settings.save()
        except Exception:
            pass

        self._push_undo_snapshot(kind)
        y = compressor_peak_ar(
            st.audio,
            sample_rate=int(st.sr),
            threshold_db=float(thr),
            ratio=float(ratio),
            attack_ms=float(self._comp_attack_ms_default),
            release_ms=float(self._comp_release_ms_default),
        )
        self._set_output_audio(kind, y, st.sr)
        self._set_status(
            f"{kind}: compressor applied (thr={thr:.1f} dB, ratio={ratio:.2f}, "
            f"atk={self._comp_attack_ms_default:.0f}ms, rel={self._comp_release_ms_default:.0f}ms)"
        )
        # Close the context menu after applying an action.
        if self._output_ctx_visible:
            self._hide_output_ctx_menu()

    # ------------------------------------------------------------------
    # Save-as for outputs
    # ------------------------------------------------------------------

    def _on_save_target_selected(self, sender, app_data) -> None:
        dst = app_data.get("file_path_name") or ""
        if not dst:
            return
        self._save_output_buffer("target", dst)

    def _on_save_residual_selected(self, sender, app_data) -> None:
        dst = app_data.get("file_path_name") or ""
        if not dst:
            return
        self._save_output_buffer("residual", dst)

    def _save_output_buffer(self, kind: str, dst: str) -> None:
        """Save the *current* (potentially edited) output buffer to disk."""
        st = self._out_edit.get(kind)
        if st is None or st.audio is None or st.audio.size == 0:
            self._set_status("No output audio to save")
            return
        try:
            # Import soundfile lazily so the GUI can still start without it.
            import soundfile as sf

            Path(dst).parent.mkdir(parents=True, exist_ok=True)

            # soundfile expects (N,) or (N,C)
            data = st.audio
            if data.shape[1] == 1:
                data_to_write = data[:, 0]
            else:
                data_to_write = data

            # Choose a sane default encoding for WAV.
            # For other formats soundfile will pick a default.
            subtype = "PCM_16" if str(dst).lower().endswith(".wav") else None
            if subtype:
                sf.write(dst, data_to_write, st.sr, subtype=subtype)
            else:
                sf.write(dst, data_to_write, st.sr)

            self._set_status(f"Saved: {dst}")
        except Exception as e:
            self._show_error(f"Save failed: {e}")

    # ------------------------------------------------------------------
    # Options window (persistent)
    # ------------------------------------------------------------------



    def _estimate_tabbar_width(self, labels: list[str]) -> int:
        """Estimate a compact width for the main tab bar.

        Some DearPyGui/ImGui builds distribute tabs across the full available width.
        We avoid version-specific fitting-policy enums by constraining the tab_bar widget
        width to roughly the sum of its tab label widths plus padding.
        """
        dpg = self.dpg

        # Conservative per-tab padding estimate (left+right, incl. internal spacing).
        pad_per_tab = 48
        extra = 24

        try:
            widths = [dpg.get_text_size(lbl)[0] for lbl in labels]
        except Exception:
            # Fallback: assume ~8 px per character (reasonable for default fonts).
            widths = [len(lbl) * 8 for lbl in labels]

        return int(sum(widths) + pad_per_tab * len(labels) + extra)

    def _open_options(self) -> None:
        dpg = self.dpg
        if dpg.does_item_exist("__options_window"):
            dpg.show_item("__options_window")
            return

        ui = self.settings.data.setdefault("ui", {})

        with dpg.window(tag="__options_window", label="Options", modal=True, show=True, width=520, height=420):
            dpg.add_text("UI Colors")
            self.tags["opt_anchor_plus"] = dpg.add_color_edit(default_value=ui.get("anchor_plus_color", [0, 180, 0, 80]), alpha_bar=True)
            self.tags["opt_anchor_minus"] = dpg.add_color_edit(default_value=ui.get("anchor_minus_color", [200, 60, 60, 80]), alpha_bar=True)
            self.tags["opt_marker"] = dpg.add_color_edit(default_value=ui.get("marker_color", [120, 120, 120, 80]), alpha_bar=True)
            self.tags["opt_playhead"] = dpg.add_color_edit(default_value=ui.get("playhead_color", [255, 200, 0, 220]), alpha_bar=True)
            dpg.add_separator()
            dpg.add_text("Waveform")
            self.tags["opt_points"] = dpg.add_input_int(default_value=int(ui.get("waveform_points", 6000)), min_value=500, max_value=40000)

            dpg.add_spacer(height=8)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Apply", callback=lambda s, a, u=None: self._apply_options())
                dpg.add_button(label="Close", callback=lambda s, a, u=None: dpg.hide_item("__options_window"))



    def _apply_options(self) -> None:
        dpg = self.dpg
        ui = self.settings.data.setdefault("ui", {})

        def v(tag):
            return dpg.get_value(tag)

        ui["anchor_plus_color"] = [int(x) for x in v(self.tags["opt_anchor_plus"])[:4]]
        ui["anchor_minus_color"] = [int(x) for x in v(self.tags["opt_anchor_minus"])[:4]]
        ui["marker_color"] = [int(x) for x in v(self.tags["opt_marker"])[:4]]
        ui["playhead_color"] = [int(x) for x in v(self.tags["opt_playhead"])[:4]]
        ui["waveform_points"] = int(v(self.tags["opt_points"]))

        self.settings.save()

        # Apply UI changes immediately
        self._apply_colors(
            playhead=tuple(ui["playhead_color"]),
            marker=tuple(ui["marker_color"]),
            anchor_plus=tuple(ui["anchor_plus_color"]),
            anchor_minus=tuple(ui["anchor_minus_color"]),
        )
        self.wave_input.points = int(ui["waveform_points"])
        self.wave_target.points = int(ui["waveform_points"])
        self.wave_residual.points = int(ui["waveform_points"])
        self.wave_input.zoom_to_fit()
        self.wave_target.zoom_to_fit()
        self.wave_residual.zoom_to_fit()

        self._set_status("Options applied")


    def _apply_colors(self, *, playhead, marker, anchor_plus, anchor_minus) -> None:
        self.wave_input.set_colors(playhead=playhead, marker=marker, anchor_plus=anchor_plus, anchor_minus=anchor_minus)
        self.wave_target.set_colors(playhead=playhead, marker=marker, anchor_plus=anchor_plus, anchor_minus=anchor_minus)
        self.wave_residual.set_colors(playhead=playhead, marker=marker, anchor_plus=anchor_plus, anchor_minus=anchor_minus)

    # ------------------------------------------------------------------
    # Backend call preview
    # ------------------------------------------------------------------


    def _refresh_backend_call_preview(self) -> None:
        """Refresh the backend parameter summary + reproducible snippet."""
        dpg = self.dpg

        # If the Run tab isn't built yet, nothing to update.
        if "param_model" not in self.tags:
            return

        cfg = None
        try:
            if self.state.input_path and Path(self.state.input_path).exists() and self.state.description.strip():
                cfg = self._build_config()
        except Exception:
            cfg = None

        def _set(tag_key: str, value: str) -> None:
            try:
                tag = self.tags.get(tag_key)
                if tag:
                    dpg.set_value(tag, value)
            except Exception:
                pass

        if cfg is None:
            # Clear UI elements
            for k in (
                "param_model",
                "param_audio",
                "param_out_target",
                "param_out_residual",
                "param_description",
                "param_predict_spans",
                "param_reranking_candidates",
                "param_device",
                "param_fp16",
                "param_max_len_s",
                "param_overlap_s",
                "param_anchor_mode",
                "param_anchors",
                "param_no_resample",
            ):
                _set(k, "")
            self._backend_snippet_text = ""
            return

        # Update parameter table
        _set("param_model", str(cfg.model))
        _set("param_audio", str(cfg.audio))
        _set("param_out_target", str(cfg.out_target))
        _set("param_out_residual", str(cfg.out_residual))
        _set("param_description", str(cfg.description))
        _set("param_predict_spans", str(bool(cfg.predict_spans)))
        _set("param_reranking_candidates", str(int(cfg.reranking_candidates)))
        _set("param_device", str(cfg.device))
        _set("param_fp16", str(bool(cfg.fp16)))
        _set("param_max_len_s", "" if cfg.max_len_s is None else str(float(cfg.max_len_s)))
        _set("param_overlap_s", "" if cfg.overlap_s is None else str(float(cfg.overlap_s)))
        _set("param_anchor_mode", str(cfg.anchor_mode))

        if cfg.anchors:
            anchors_lines = []
            for sign, s, e in cfg.anchors:
                anchors_lines.append(f"{sign}  {float(s):.3f}  {float(e):.3f}")
            _set("param_anchors", "\n".join(anchors_lines))
        else:
            _set("param_anchors", "(none)")

        _set("param_no_resample", str(bool(cfg.no_resample)))

        # Snippet (cached; no longer displayed in the UI)
        self._backend_snippet_text = self._format_backend_call(cfg, original_audio=str(self.state.input_path))


    def _format_backend_call(self, cfg: Config, *, original_audio: str) -> str:
        """Format a self-contained backend call as a reproducible Python snippet."""
        anchor_items = []
        for (sign, a, b) in (cfg.anchors or []):
            anchor_items.append(f"({sign!r}, {float(a):.3f}, {float(b):.3f})")
        if anchor_items:
            anchors_repr = "[\n        " + ",\n        ".join(anchor_items) + "\n    ]"
        else:
            anchors_repr = "[]"

        max_len = "None" if cfg.max_len_s is None else f"{float(cfg.max_len_s):.3f}"
        overlap = "None" if cfg.overlap_s is None else f"{float(cfg.overlap_s):.3f}"

        # If the GUI ran the backend on a processed-input temp file, we show both.
        audio_for_backend = str(cfg.audio)
        orig = str(original_audio or cfg.audio)

        lines = [
            "from patchbay_backend import Config, run_pipeline",
            "",
            f"original_audio = {orig!r}",
            f"audio_for_backend = {audio_for_backend!r}",
            "",
            "cfg = Config.from_parameters(",
            f"    model={cfg.model!r},",
            "    audio=audio_for_backend,",
            f"    description={cfg.description!r},",
            f"    out_target={cfg.out_target!r},",
            f"    out_residual={cfg.out_residual!r},",
            f"    predict_spans={cfg.predict_spans!r},",
            f"    reranking_candidates={int(cfg.reranking_candidates)!r},",
            f"    device={cfg.device!r},",
            f"    fp16={cfg.fp16!r},",
            f"    max_len_s={max_len},",
            f"    overlap_s={overlap},",
            f"    anchor_mode={cfg.anchor_mode!r},",
            f"    anchors={anchors_repr},",
            f"    no_resample={cfg.no_resample!r},",
            ")",
            "",
            "out_target, out_residual = run_pipeline(cfg)",
        ]
        return "\n".join(lines)

    def _copy_backend_snippet_to_clipboard(self) -> None:
        """Copy current snippet to clipboard (best-effort)."""
        try:
            txt = str(getattr(self, "_backend_snippet_text", "") or "")
            if not txt:
                # Best-effort fallback
                try:
                    cfg = self._build_config()
                    txt = self._format_backend_call(cfg, original_audio=str(self.state.input_path))
                except Exception:
                    txt = ""

            if not txt:
                self._set_status("Nothing to copy (missing input/description?)")
                return

            self.dpg.set_clipboard_text(str(txt))
            self._set_status("Snippet copied to clipboard")
        except Exception:
            pass


    # ------------------------------------------------------------------
    # Status + Error dialogs
    # ------------------------------------------------------------------

    def _set_status(self, msg: str) -> None:
        if "status_line" in self.tags:
            self.dpg.set_value(self.tags["status_line"], str(msg))

    def _show_error(self, text: str) -> None:
        dpg = self.dpg
        if dpg.does_item_exist("__error_window"):
            dpg.delete_item("__error_window")
        with dpg.window(tag="__error_window", label="Error", modal=True, show=True, width=800, height=400):
            dpg.add_text("An error occurred:")
            dpg.add_input_text(multiline=True, readonly=True, default_value=str(text), width=-1, height=300)
            dpg.add_button(label="Close", callback=lambda s, a, u=None: dpg.delete_item("__error_window"))

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Called when the app is closing."""
        try:
            self.worker.cancel()
        except Exception:
            pass
        try:
            self.player_input.close()
            self.player_target.close()
            self.player_residual.close()
        except Exception:
            pass