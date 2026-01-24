"""Waveform canvas widget for DearPyGui.

This module replaces the previous pyqtgraph-based widget.

DearPyGui provides both "plot" widgets and low-level drawlists. For this
project we use a drawlist to keep the implementation framework-agnostic and
to avoid relying on plot-specific APIs that sometimes vary between DearPyGui
versions.

The widget supports:
- Waveform display (downsampled)
- Click-to-seek (reports time position)
- Zoom in/out / zoom-to-fit
- Playhead line
- Marker selection region (start/stop)
- Anchor overlays (+ and - anchors)

The widget **does not** own playback logic. It only draws and reports user
interactions. Playback is handled by :class:`patchbay_desktop_gui.audio_player.AudioPlayer`.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, List, Optional, Tuple

import numpy as np


RGBA = Tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# Display scaling policy
# ---------------------------------------------------------------------------
#
# The waveform display uses a *fixed* vertical scale of 0 dBFS full scale:
#
# - The top/bottom of the display correspond to +/- 1.0 (0 dBFS).
# - Values outside [-1.0, 1.0] are clipped *for display only* (no highlight).
#
# This keeps the visual scale consistent across files and avoids auto-scaling.
DISPLAY_Y_MAX_AMP: float = 1.0


@dataclass
class AnchorItem:
    sign: str  # '+' or '-'
    start_s: float
    end_s: float


class WaveformWidget:
    """A drawlist-based waveform widget.

    Parameters
    ----------
    label:
        Human readable label (used in the panel header).
    points:
        Maximum number of waveform points rendered for the *current view*.
        Larger values look nicer but cost CPU during zoom operations.
    on_seek:
        Callback invoked when the user clicks the canvas. Signature:
            on_seek(seconds: float) -> None
    """

    def __init__(
        self,
        *,
        label: str,
        points: int = 6000,
        on_seek: Optional[Callable[[float], None]] = None,
        on_right_click: Optional[Callable[[Tuple[float, float]], None]] = None,
    ) -> None:
        self.label = label
        self.points = int(points)
        self.on_seek = on_seek
        # Optional right-click callback (used e.g. for context menus).
        # Signature: on_right_click((x_viewport, y_viewport)) -> None
        self.on_right_click = on_right_click

        # Audio data for drawing (mono only)
        self._mono: Optional[np.ndarray] = None  # shape (N,)
        self._sr: int = 48000

        # View state
        self._view_start_s: float = 0.0
        self._view_end_s: float = 1.0
        self._duration_s: float = 0.0
        # Display scale is fixed to 0 dBFS full scale.
        self._y_max: float = float(DISPLAY_Y_MAX_AMP)
        # Peak of the *loaded* signal (for diagnostics / status display).
        self._signal_peak: float = 0.0

        # Playhead / markers
        self._playhead_s: float = 0.0
        self._marker_start_s: Optional[float] = None
        self._marker_stop_s: Optional[float] = None

        # Anchors
        self._anchors: List[AnchorItem] = []

        # Colors
        self.playhead_color: RGBA = (255, 200, 0, 220)
        self.marker_color: RGBA = (120, 120, 120, 80)
        self.anchor_plus_color: RGBA = (0, 180, 0, 80)
        self.anchor_minus_color: RGBA = (200, 60, 60, 80)
        self.axis_color: RGBA = (200, 200, 200, 180)
        self.wave_color: RGBA = (80, 180, 255, 200)

        # DearPyGui tags (set in build())
        self._root_tag = None
        self._canvas_tag = None
        self._draw_tag = None
        self._handler_tag = None
        self._item_handler_tag = None
        # Prefer item-specific click handlers (bound to the drawlist) when
        # available. Some DearPyGui versions restrict item-click handlers for
        # certain item types (notably child windows), but drawlists typically
        # support them. Using an item handler removes the need for unreliable
        # "is hovered" checks or geometry hit-tests.
        self._use_item_clicked_handler: bool = False
        self._use_item_right_click_handler: bool = False

        # Cached display data for current view
        self._disp_x: Optional[np.ndarray] = None
        self._disp_y: Optional[np.ndarray] = None

        # Canvas sizing / deferred drawing
        self._last_canvas_size = (0, 0)  # (w,h)
        self._pending_redraw: bool = True  # draw once we have a real size

        # Mouse interaction state
        # -----------------------
        # DearPyGui's mouse handlers are typically global (handler_registry),
        # meaning every widget receives the same mouse events.
        #
        # We therefore maintain a small state machine to implement:
        # - Drag-to-scrub: left-mouse press inside the canvas starts scrubbing;
        #   mouse move updates the playhead continuously until release.
        # - Modifier clicks: Shift+Click sets marker start, Ctrl+Click sets
        #   marker stop.
        #
        # All events are filtered by explicit rectangle hit-testing against the
        # canvas, so multiple waveform widgets can coexist without interfering
        # with each other.
        self._scrub_active: bool = False
        self._scrub_last_t: float = 0.0
        self._scrub_last_emit_wall: float = 0.0

        # Timestamp of the last item-specific click callback.
        #
        # Why do we track this?
        # ---------------------
        # DearPyGui's click handling can vary across versions/builds:
        # - On some builds, binding an item handler registry with an
        #   item-click handler to a drawlist works perfectly.
        # - On others, the bind succeeds, but the callback may not fire in
        #   certain nested container hierarchies (e.g. tabs + groups).
        #
        # To make click-to-seek reliable for *all* waveform widgets (Input,
        # Target, Residual), we always register a global click handler as a
        # fallback. When the item-specific handler does fire, we suppress the
        # fallback click for a short time window to avoid double-processing.
        self._last_item_click_wall: float = 0.0

    # ------------------------------------------------------------------
    # DearPyGui compatibility helpers
    # ------------------------------------------------------------------

    def _get_mouse_pos_viewport(self, dpg) -> Optional[Tuple[float, float]]:
        """Return mouse position in **viewport coordinates**.

        DearPyGui's ``get_mouse_pos`` changed signatures across releases:

        - Newer builds provide ``get_mouse_pos(local=...)``.
        - Older builds provide ``get_mouse_pos()`` without parameters.

        Unfortunately, some builds default to returning *local* coordinates
        when called without ``local=False``. Local coordinates break hit
        testing when the widget is not near the origin, which is exactly the
        case for the Target/Residual panels.

        We therefore try ``local=False`` first and only fall back to the
        parameterless call if needed.
        """
        try:
            # Prefer explicit viewport coordinates.
            return tuple(dpg.get_mouse_pos(local=False))  # type: ignore[arg-type]
        except TypeError:
            # Older builds without the local kwarg.
            try:
                return tuple(dpg.get_mouse_pos())
            except Exception:
                return None
        except Exception:
            try:
                return tuple(dpg.get_mouse_pos())
            except Exception:
                return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, parent: str, *, height: int = 240) -> str:
        """Create the widget under *parent* and return the root tag."""
        import dearpygui.dearpygui as dpg

        with dpg.group(parent=parent) as root:
            self._root_tag = root

            # The drawing area is a child window so we can reliably query its
            # content region size.
            #
            # IMPORTANT:
            # ---------
            # We disable scrollbars and mouse-wheel scrolling. Without this,
            # DearPyGui may show scrollbars for large drawlists in nested
            # layouts and the mouse wheel would scroll the child window
            # instead of zooming the waveform.
            with dpg.child_window(
                width=-1,
                height=height,
                border=True,
                no_scrollbar=True,
                no_scroll_with_mouse=True,
                horizontal_scrollbar=False,
            ) as canvas:
                self._canvas_tag = canvas
                # A drawlist covers the full child window.
                self._draw_tag = dpg.add_drawlist(parent=canvas, width=-1, height=-1)

        # Try to bind an item-specific click handler to the drawlist.
        # ----------------------------------------------------------
        # This is the most reliable way to implement click-to-seek across
        # different container hierarchies (tabs/groups/child windows), because
        # DearPyGui itself decides when the drawlist was clicked.
        #
        # If this fails (older/stricter builds), we fall back to the global
        # mouse click handler with explicit rectangle hit-testing.
        self._use_item_clicked_handler = False
        self._item_handler_tag = None
        try:
            with dpg.item_handler_registry() as ireg:
                # Left click (seek)
                try:
                    dpg.add_item_clicked_handler(button=dpg.mvMouseButton_Left, callback=self._on_item_clicked)
                except TypeError:
                    # Older builds may not expose the button kwarg and always
                    # treat this as a left click.
                    dpg.add_item_clicked_handler(callback=self._on_item_clicked)

                # Right click (context menu)
                self._use_item_right_click_handler = False
                try:
                    dpg.add_item_clicked_handler(button=dpg.mvMouseButton_Right, callback=self._on_item_right_clicked)
                    self._use_item_right_click_handler = True
                except Exception:
                    # Not supported on some builds; we will fall back to a
                    # global right-click handler in the main window.
                    self._use_item_right_click_handler = False

            dpg.bind_item_handler_registry(self._draw_tag, ireg)
            self._item_handler_tag = ireg
            self._use_item_clicked_handler = True
        except Exception:
            # Fall back to global handlers (see below).
            self._use_item_clicked_handler = False
            self._item_handler_tag = None
            self._use_item_right_click_handler = False

        # IMPORTANT DearPyGui compatibility note
        # -------------------------------------
        # Some DearPyGui versions do NOT allow binding an item handler registry
        # containing an mvClickedHandler ("item clicked") to a child window
        # (mvChildWindow). Doing so raises:
        #   "Item Handler Registry includes inapplicable handler: mvClickedHandler"
        #
        # To keep this widget compatible across DearPyGui releases, we use a
        # *global* mouse click handler and only react when the mouse is currently
        # hovering over *our* canvas. This avoids any item-type restrictions.
        #
        # See also: https://github.com/hoffstadt/DearPyGui/issues (various handler
        # applicability changes across versions).
        with dpg.handler_registry() as hreg:
            # Global mouse handlers
            # ---------------------
            # In DearPyGui, these handlers fire for *every* mouse event in the
            # viewport. We therefore filter events using explicit rectangle hit
            # testing (see _canvas_hit_test()).

            # Click-to-seek and modifier clicks (Shift/Ctrl)
            # ---------------------------------------------
            # We *always* register a global click handler as a fallback.
            #
            # Rationale:
            # Some DearPyGui builds accept binding an item-click handler to a
            # drawlist, but the callback may not fire reliably for drawlists
            # inside certain container hierarchies (notably tabs). This was
            # observed for the Target/Residual output panels.
            #
            # To guarantee click-to-seek everywhere, we register a global click
            # handler and suppress it when we have just received an item-click
            # callback (see _last_item_click_wall).
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left, callback=self._on_mouse_click)

            # Drag-to-scrub uses a very stable pattern: mouse down -> move
            # updates while pressed -> release.
            #
            # We intentionally do NOT rely on add_mouse_drag_handler here,
            # because its signature and behavior has changed across DearPyGui
            # versions/builds. The down/move/release approach works everywhere.
            dpg.add_mouse_down_handler(button=dpg.mvMouseButton_Left, callback=self._on_mouse_down)
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=self._on_mouse_release)
            dpg.add_mouse_move_handler(callback=self._on_mouse_move)

            # Mouse wheel zoom (zoom around mouse position)
            if hasattr(dpg, "add_mouse_wheel_handler"):
                dpg.add_mouse_wheel_handler(callback=self._on_mouse_wheel)
        self._handler_tag = hreg

        # Draw initial empty canvas
        self.redraw()
        return str(root)

    # ------------------------------------------------------------------
    # Minimal accessors (used by the main window)
    # ------------------------------------------------------------------

    def get_canvas_tag(self) -> Optional[str]:
        """Return the internal canvas (child_window) tag.

        This is primarily used by the main window to perform rectangle hit
        testing for global context menus (right-click actions).
        """
        return str(self._canvas_tag) if self._canvas_tag is not None else None

    def get_draw_tag(self) -> Optional[str]:
        """Return the internal drawlist tag."""
        return str(self._draw_tag) if self._draw_tag is not None else None

    def set_colors(
        self,
        *,
        playhead: RGBA,
        marker: RGBA,
        anchor_plus: RGBA,
        anchor_minus: RGBA,
    ) -> None:
        self.playhead_color = playhead
        self.marker_color = marker
        self.anchor_plus_color = anchor_plus
        self.anchor_minus_color = anchor_minus

    def set_waveform(self, mono: np.ndarray, sr: int) -> None:
        """Set the audio waveform used for display.

        ``mono`` must be a 1D numpy array.
        """
        if mono.ndim != 1:
            raise ValueError("mono must be 1D")
        self._mono = np.asarray(mono, dtype=np.float32)
        self._sr = int(sr)
        self._duration_s = float(len(self._mono)) / float(self._sr) if len(self._mono) else 0.0
        # Keep the display scale fixed; only compute signal peak for optional
        # diagnostics.
        self._signal_peak = float(np.max(np.abs(self._mono))) if len(self._mono) else 0.0
        self._y_max = float(DISPLAY_Y_MAX_AMP)
        self._pending_redraw = True
        self.zoom_to_fit()

    def clear_waveform(self) -> None:
        self._mono = None
        self._sr = 48000
        self._duration_s = 0.0
        self._signal_peak = 0.0
        self._y_max = float(DISPLAY_Y_MAX_AMP)
        self._view_start_s = 0.0
        self._view_end_s = 1.0
        self._disp_x = None
        self._disp_y = None
        self._playhead_s = 0.0
        self._marker_start_s = None
        self._marker_stop_s = None
        self._anchors = []
        self.redraw()

    def request_redraw(self) -> None:
        """Request a redraw on the next GUI frame.

        DearPyGui items often report a (0,0) size before the viewport is shown.
        In that case, drawing into the canvas would collapse to a 1-pixel line
        and appear as if nothing was drawn.

        We therefore allow drawing to be *deferred* until a reasonable canvas
        size is available.
        """
        self._pending_redraw = True

    def tick(self) -> None:
        """Per-frame hook.

        Call this from the application's render loop. The widget will redraw
        itself when:
        - the canvas size changes (window resize), or
        - a redraw has been requested while the canvas size was still 0.
        """
        try:
            import dearpygui.dearpygui as dpg
            if not self._canvas_tag or not self._draw_tag:
                return
            if not dpg.does_item_exist(self._canvas_tag) or not dpg.does_item_exist(self._draw_tag):
                return
            w, h = dpg.get_item_rect_size(self._canvas_tag)
            w = int(w)
            h = int(h)
            # If the viewport is not shown yet, sizes can be 0. Defer drawing.
            if w < 20 or h < 20:
                return
            if (w, h) != tuple(self._last_canvas_size) or self._pending_redraw:
                self._pending_redraw = False
                self._last_canvas_size = (w, h)
                self.redraw()
        except Exception:
            return

    def set_playhead_seconds(self, t: float) -> None:
        self._playhead_s = self._clip_time(float(t))
        self._pending_redraw = True
        self.redraw()

    def get_playhead_seconds(self) -> float:
        return float(self._playhead_s)

    def zoom_to_fit(self) -> None:
        self._view_start_s = 0.0
        self._view_end_s = max(1e-6, float(self._duration_s))
        self._recompute_display()
        self._pending_redraw = True
        self.redraw()

    def zoom_in(self, factor: float = 0.5) -> None:
        """Zoom in around playhead. factor<1 shrinks the window."""
        self._zoom_around_playhead(factor=float(factor))

    def zoom_out(self, factor: float = 2.0) -> None:
        """Zoom out around playhead. factor>1 expands the window."""
        self._zoom_around_playhead(factor=float(factor))

    def set_marker_start_seconds(self, t: float) -> None:
        self._marker_start_s = self._clip_time(float(t))
        self._pending_redraw = True
        self.redraw()

    def set_marker_stop_seconds(self, t: float) -> None:
        self._marker_stop_s = self._clip_time(float(t))
        self._pending_redraw = True
        self.redraw()

    def clear_markers(self) -> None:
        self._marker_start_s = None
        self._marker_stop_s = None
        self._pending_redraw = True
        self.redraw()

    def get_marker_range(self) -> Optional[Tuple[float, float]]:
        if self._marker_start_s is None or self._marker_stop_s is None:
            return None
        a = float(self._marker_start_s)
        b = float(self._marker_stop_s)
        if b <= a:
            return None
        return a, b

    def set_anchors(self, anchors: List[AnchorItem]) -> None:
        self._anchors = list(anchors)
        self._pending_redraw = True
        self.redraw()

    def get_duration_seconds(self) -> float:
        return float(self._duration_s)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def redraw(self) -> None:
        """Redraw the canvas. Safe to call frequently."""
        import dearpygui.dearpygui as dpg

        if not self._draw_tag or not dpg.does_item_exist(self._draw_tag):
            return

        # Query drawing size (child window content region).
        w, h = dpg.get_item_rect_size(self._canvas_tag)
        w = int(w)
        h = int(h)

        # DearPyGui may report (0,0) sizes before the viewport is shown or
        # while a layout pass is pending. Drawing with a 1px canvas would
        # make the waveform appear invisible. In that case we defer drawing
        # to the next frame (see tick()).
        if w < 20 or h < 20:
            self._pending_redraw = True
            return

        # Ensure the drawlist matches the canvas size. Not all DearPyGui
        # versions auto-resize drawlists inside child windows.
        try:
            dpg.configure_item(self._draw_tag, width=w, height=h)
        except Exception:
            pass

        dpg.delete_item(self._draw_tag, children_only=True)

        # Background
        dpg.draw_rectangle((0, 0), (w, h), fill=(20, 20, 20, 255), color=(0, 0, 0, 0), parent=self._draw_tag)

        # Midline (0 amplitude)
        mid_y = h * 0.5
        dpg.draw_line((0, mid_y), (w, mid_y), color=self.axis_color, thickness=1, parent=self._draw_tag)

        # Marker region shading (draw BEFORE waveform)
        mr = self.get_marker_range()
        if mr is not None:
            x1 = self._time_to_x(mr[0], w)
            x2 = self._time_to_x(mr[1], w)
            if x2 > x1:
                dpg.draw_rectangle((x1, 0), (x2, h), fill=self.marker_color, color=(0, 0, 0, 0), parent=self._draw_tag)

        # Anchor overlays (draw BEFORE waveform)
        for a in self._anchors:
            x1 = self._time_to_x(a.start_s, w)
            x2 = self._time_to_x(a.end_s, w)
            if x2 <= x1:
                continue
            col = self.anchor_plus_color if a.sign == "+" else self.anchor_minus_color
            dpg.draw_rectangle((x1, 0), (x2, h), fill=col, color=(0, 0, 0, 0), parent=self._draw_tag)

        # Waveform
        if self._disp_x is None or self._disp_y is None:
            self._recompute_display()

        if self._disp_x is not None and self._disp_y is not None and len(self._disp_x) >= 2:
            pts: List[Tuple[float, float]] = []
            yscale = (h * 0.45) / float(self._y_max)

            # Precompute points (with hard clipping at the display headroom)
            # so we never draw outside of the canvas.
            for t, y in zip(self._disp_x, self._disp_y):
                x = self._time_to_x(float(t), w)
                yv = float(y)
                if yv > self._y_max:
                    yv = self._y_max
                elif yv < -self._y_max:
                    yv = -self._y_max
                yy = mid_y - yv * yscale
                pts.append((x, yy))

            # Draw base waveform
            dpg.draw_polyline(pts, color=self.wave_color, thickness=1, parent=self._draw_tag)

        # Marker boundary lines
        if self._marker_start_s is not None:
            x = self._time_to_x(float(self._marker_start_s), w)
            dpg.draw_line((x, 0), (x, h), color=(180, 180, 180, 200), thickness=1, parent=self._draw_tag)
        if self._marker_stop_s is not None:
            x = self._time_to_x(float(self._marker_stop_s), w)
            dpg.draw_line((x, 0), (x, h), color=(180, 180, 180, 200), thickness=1, parent=self._draw_tag)

        # Playhead line (draw last)
        xph = self._time_to_x(float(self._playhead_s), w)
        dpg.draw_line((xph, 0), (xph, h), color=self.playhead_color, thickness=2, parent=self._draw_tag)

        # Status text (top-left)
        st = self._status_text()
        dpg.draw_text((6, 6), st, color=(220, 220, 220, 220), parent=self._draw_tag)

    # ------------------------------------------------------------------
    # Mouse handling
    # ------------------------------------------------------------------

    def _on_item_clicked(self, sender, app_data, user_data=None) -> None:
        """Item-specific click handler bound to the drawlist.

        This is the preferred path for click-to-seek because DearPyGui itself
        determines whether the click happened on *our* waveform drawlist.

        We still compute the local x-coordinate using the **canvas** geometry
        (child window) because drawlist rect geometry can be inconsistent in
        nested layouts on some DearPyGui builds.
        """
        try:
            import dearpygui.dearpygui as dpg

            if not self._canvas_tag or not dpg.does_item_exist(self._canvas_tag):
                return

            # Mouse position in viewport space
            mp = self._get_mouse_pos_viewport(dpg)
            if mp is None:
                return
            mx, my = mp

            # Canvas rectangle in viewport space
            try:
                cx, cy = dpg.get_item_rect_min(self._canvas_tag)
                w, h = dpg.get_item_rect_size(self._canvas_tag)
            except Exception:
                return

            w = float(w)
            h = float(h)
            if w < 5 or h < 5:
                return

            local_x = float(mx) - float(cx)
            local_y = float(my) - float(cy)
            if not (0.0 <= local_x <= w and 0.0 <= local_y <= h):
                # Should not happen for an item-click handler, but keep it safe.
                return

            t = float(self._x_to_time(local_x, int(w)))

            shift = self._is_shift_down(dpg)
            ctrl = self._is_ctrl_down(dpg)

            if shift and not ctrl:
                self._last_item_click_wall = time.time()
                self.set_marker_start_seconds(t)
                self.set_playhead_seconds(t)
                if self.on_seek:
                    self.on_seek(t)
                return

            if ctrl and not shift:
                self._last_item_click_wall = time.time()
                self.set_marker_stop_seconds(t)
                self.set_playhead_seconds(t)
                if self.on_seek:
                    self.on_seek(t)
                return

            # Default: click-to-seek
            self._last_item_click_wall = time.time()
            self.set_playhead_seconds(t)
            if self.on_seek:
                self.on_seek(t)

        except Exception:
            return

    def _on_item_right_clicked(self, sender, app_data, user_data=None) -> None:
        """Item-specific right-click handler bound to the drawlist.

        This is used for context menus in the Output panels. We only forward
        the click to the GUI layer (MainWindow) and do not modify any waveform
        state here.
        """
        try:
            import dearpygui.dearpygui as dpg

            if not self.on_right_click:
                return

            mp = self._get_mouse_pos_viewport(dpg)
            if mp is None:
                return

            # We still validate that the click is inside the canvas to avoid
            # spurious right-clicks in nested layouts.
            if self._canvas_tag and dpg.does_item_exist(self._canvas_tag):
                try:
                    cx, cy = dpg.get_item_rect_min(self._canvas_tag)
                    w, h = dpg.get_item_rect_size(self._canvas_tag)
                    mx, my = mp
                    if not (cx <= mx <= cx + w and cy <= my <= cy + h):
                        return
                except Exception:
                    pass

            self.on_right_click((float(mp[0]), float(mp[1])))
        except Exception:
            return

    def _on_mouse_click(self, sender, app_data, user_data=None) -> None:
        """Global mouse click handler (left button).

        The handler is global; we only act if the mouse is over our canvas.
        """
        try:
            import dearpygui.dearpygui as dpg

            # If an item-specific click handler fired very recently, suppress
            # the global click handling to avoid duplicate seek operations.
            #
            # We keep the time window deliberately short to still allow the
            # fallback path when item-click callbacks do *not* fire reliably.
            if self._use_item_clicked_handler:
                if (time.time() - float(self._last_item_click_wall)) < 0.05:
                    return

            hit = self._canvas_hit_test(dpg)
            if hit is None:
                return
            local_x, _local_y, w, _h = hit

            # Convert x to time (seconds)
            t = float(self._x_to_time(local_x, int(w)))

            # Modifier clicks:
            # - Shift + Click  -> set marker start
            # - Ctrl  + Click  -> set marker stop
            # Otherwise behave like normal click-to-seek.
            shift = self._is_shift_down(dpg)
            ctrl = self._is_ctrl_down(dpg)

            if shift and not ctrl:
                self._last_item_click_wall = time.time()
                self.set_marker_start_seconds(t)
                self.set_playhead_seconds(t)
                if self.on_seek:
                    self.on_seek(t)
                return

            if ctrl and not shift:
                self._last_item_click_wall = time.time()
                self.set_marker_stop_seconds(t)
                self.set_playhead_seconds(t)
                if self.on_seek:
                    self.on_seek(t)
                return

            # Default: click-to-seek
            self._last_item_click_wall = time.time()
            self.set_playhead_seconds(t)
            if self.on_seek:
                self.on_seek(t)

        except Exception:
            # Do not crash GUI on handler issues
            return

    def _on_mouse_down(self, sender, app_data, user_data=None) -> None:
        """Start drag-to-scrub if the press begins inside the canvas."""
        try:
            import dearpygui.dearpygui as dpg
            hit = self._canvas_hit_test(dpg)
            if hit is None:
                self._scrub_active = False
                return

            # Do not start scrubbing when the user performs modifier-click actions.
            if self._is_shift_down(dpg) or self._is_ctrl_down(dpg):
                self._scrub_active = False
                return

            local_x, _local_y, w, _h = hit
            t = float(self._x_to_time(local_x, int(w)))
            self._scrub_active = True
            self._scrub_last_t = t
            self.set_playhead_seconds(t)
            if self.on_seek:
                self.on_seek(t)
        except Exception:
            self._scrub_active = False

    def _on_mouse_release(self, sender, app_data, user_data=None) -> None:
        """Stop drag-to-scrub."""
        self._scrub_active = False

    def _on_mouse_move(self, sender, app_data, user_data=None) -> None:
        """Fallback path for drag-to-scrub (down+move+release)."""
        if not self._scrub_active:
            return
        try:
            import dearpygui.dearpygui as dpg
            hit = self._canvas_hit_test(dpg)
            if hit is None:
                return
            local_x, _local_y, w, _h = hit
            t = float(self._x_to_time(local_x, int(w)))
            self._scrub_update(t)
        except Exception:
            return

    def _on_mouse_wheel(self, sender, app_data, user_data=None) -> None:
        """Mouse wheel zoom.

        DearPyGui passes the wheel delta in *app_data* (usually an int).
        We zoom around the **mouse position** without moving the playhead.
        """
        try:
            import dearpygui.dearpygui as dpg

            hit = self._canvas_hit_test(dpg)
            if hit is None:
                return
            local_x, _local_y, w, _h = hit

            # Determine zoom center from mouse position
            center_t = float(self._x_to_time(local_x, int(w)))

            # Wheel delta: positive -> zoom in, negative -> zoom out
            try:
                delta = int(app_data)
            except Exception:
                # Some builds may pass a tuple/list; try first element
                try:
                    delta = int(app_data[0])
                except Exception:
                    return

            if delta == 0:
                return

            # Use an exponential mapping so multiple notches compound nicely.
            # One notch -> factor 0.8 (in) or 1.25 (out).
            base_in = 0.8
            if delta > 0:
                factor = base_in ** abs(delta)
            else:
                factor = (1.0 / base_in) ** abs(delta)

            # Do not modify playhead on wheel zoom.
            self._zoom_around_center(center_s=center_t, factor=float(factor))

        except Exception:
            return

    def _on_mouse_drag(self, sender, app_data, user_data=None) -> None:
        """Preferred drag-to-scrub callback (mouse drag handler)."""
        # Note: app_data typically includes drag delta, but we use the current
        # mouse position for precise mapping to time.
        try:
            import dearpygui.dearpygui as dpg
            # If a drag starts outside the canvas, ignore it. If it starts inside,
            # keep updating while inside.
            hit = self._canvas_hit_test(dpg)
            if hit is None:
                return
            # Avoid modifier drags
            if self._is_shift_down(dpg) or self._is_ctrl_down(dpg):
                return
            local_x, _local_y, w, _h = hit
            t = float(self._x_to_time(local_x, int(w)))
            self._scrub_update(t)
        except Exception:
            return

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _canvas_hit_test(self, dpg):
        """Return local mouse coordinates within the canvas if the mouse is inside.

        Parameters
        ----------
        dpg:
            The imported ``dearpygui.dearpygui`` module.

        Returns
        -------
        tuple | None
            ``(local_x, local_y, width, height)`` in **pixels** if the mouse
            is inside the canvas rectangle, else ``None``.

        Notes
        -----
        DearPyGui's "hovered" state is not fully consistent across versions
        and item types (e.g., drawlist-in-child-window). We therefore use a
        rectangle hit test based on item rect geometry.
        """
        # Hit testing strategy
        # --------------------
        # Earlier iterations preferred hit-testing against the drawlist.
        # Unfortunately, DearPyGui's rect geometry for drawlists can be
        # *inconsistent* depending on the container hierarchy (tabs/groups).
        #
        # A common failure mode is that drawlists report a rect-min near
        # (0,0), causing multiple waveform widgets to appear "overlapping" to
        # the hit-test. This would break click-to-seek for target/residual even
        # though the input view works.
        #
        # To make the interaction reliable for all three panels (input/target/
        # residual), we prefer hit-testing against the *child window* (canvas)
        # first, because it has stable screen-space geometry. We only fall back
        # to drawlist geometry if the canvas is not available.
        candidates = []
        if self._canvas_tag and dpg.does_item_exist(self._canvas_tag):
            candidates.append(self._canvas_tag)
        if self._draw_tag and dpg.does_item_exist(self._draw_tag):
            candidates.append(self._draw_tag)
        if not candidates:
            return None

        mp = self._get_mouse_pos_viewport(dpg)
        if mp is None:
            return None
        mx, my = mp

        for target in candidates:
            # We intentionally do NOT rely on `is_item_visible()` / `is_item_shown()`
            # here. In some DearPyGui builds these flags can be inconsistent for
            # items inside tabs, leading to false negatives (e.g., target/residual
            # canvases reporting "not visible" while they are on-screen). A simple
            # rectangle hit-test using rect-min/size is sufficient and more robust.

            try:
                cx, cy = dpg.get_item_rect_min(target)
                w, h = dpg.get_item_rect_size(target)
            except Exception:
                continue

            w = float(w)
            h = float(h)
            if w < 5 or h < 5:
                continue

            local_x = float(mx) - float(cx)
            local_y = float(my) - float(cy)

            if 0.0 <= local_x <= w and 0.0 <= local_y <= h:
                return local_x, local_y, w, h

        return None

    def _key_down_any(self, dpg, names: List[str]) -> bool:
        """Return True if any DearPyGui key constant in *names* is down."""
        for nm in names:
            code = getattr(dpg, nm, None)
            if code is None:
                continue
            try:
                if dpg.is_key_down(code):
                    return True
            except Exception:
                # is_key_down may not exist in very old versions; fall back to
                # False (modifier clicks will behave like normal clicks).
                continue
        return False

    def _is_shift_down(self, dpg) -> bool:
        """Best-effort Shift modifier detection across DearPyGui versions."""
        return self._key_down_any(dpg, ["mvKey_Shift", "mvKey_LShift", "mvKey_RShift"])

    def _is_ctrl_down(self, dpg) -> bool:
        """Best-effort Ctrl modifier detection across DearPyGui versions."""
        return self._key_down_any(dpg, ["mvKey_Control", "mvKey_LControl", "mvKey_RControl"])

    def _scrub_update(self, t: float) -> None:
        """Update playhead during drag-to-scrub.

        To avoid overwhelming the playback backend with high-frequency seek
        events, we throttle callback invocations while still updating the drawn
        playhead position immediately.
        """
        t = float(self._clip_time(t))
        self.set_playhead_seconds(t)

        # Throttle on_seek emission to ~30 Hz or on meaningful changes.
        now = time.time()
        if abs(t - self._scrub_last_t) < 0.005 and (now - self._scrub_last_emit_wall) < 0.05:
            return
        self._scrub_last_t = t
        self._scrub_last_emit_wall = now
        if self.on_seek:
            self.on_seek(t)

    def _clip_time(self, t: float) -> float:
        if self._duration_s <= 0:
            return 0.0
        return max(0.0, min(float(t), float(self._duration_s)))

    def _time_to_x(self, t: float, width_px: int) -> float:
        span = max(1e-9, self._view_end_s - self._view_start_s)
        return (float(t) - self._view_start_s) / span * float(width_px)

    def _x_to_time(self, x: float, width_px: int) -> float:
        span = max(1e-9, self._view_end_s - self._view_start_s)
        t = self._view_start_s + (float(x) / max(1.0, float(width_px))) * span
        return self._clip_time(t)

    def _zoom_around_playhead(self, factor: float) -> None:
        """Zoom the view around the current playhead position."""
        self._zoom_around_center(center_s=float(self._playhead_s), factor=float(factor))

    def _zoom_around_center(self, *, center_s: float, factor: float) -> None:
        """Zoom the view around an arbitrary time position.

        This helper is used for two UX patterns:
        - Button zoom: zoom around the playhead.
        - Mouse wheel zoom: zoom around the mouse position (without moving the playhead).
        """
        if self._duration_s <= 0:
            return

        factor = max(0.05, min(20.0, float(factor)))
        cur_span = max(1e-6, self._view_end_s - self._view_start_s)
        new_span = cur_span * factor
        new_span = max(0.05, min(new_span, float(self._duration_s)))

        center = float(self._clip_time(center_s))
        start = center - new_span / 2.0
        end = center + new_span / 2.0

        if start < 0.0:
            end -= start
            start = 0.0

        if end > self._duration_s:
            start -= (end - self._duration_s)
            end = float(self._duration_s)
            start = max(0.0, start)

        self._view_start_s = float(start)
        self._view_end_s = float(end)
        self._recompute_display()
        self._pending_redraw = True
        self.redraw()

    def _recompute_display(self) -> None:
        """Compute downsampled waveform for the current view."""
        if self._mono is None or len(self._mono) == 0:
            self._disp_x = None
            self._disp_y = None
            return

        start_s = max(0.0, min(self._view_start_s, self._duration_s))
        end_s = max(0.0, min(self._view_end_s, self._duration_s))
        if end_s <= start_s:
            end_s = min(self._duration_s, start_s + 1e-3)

        s0 = int(round(start_s * self._sr))
        s1 = int(round(end_s * self._sr))
        s0 = max(0, min(s0, len(self._mono)))
        s1 = max(0, min(s1, len(self._mono)))
        if s1 <= s0:
            self._disp_x = None
            self._disp_y = None
            return

        seg = self._mono[s0:s1]
        n = len(seg)

        # Determine stride to limit to self.points points.
        pts = max(100, int(self.points))
        stride = max(1, n // pts)

        y = seg[::stride]
        # Corresponding times
        idx = np.arange(0, len(y), dtype=np.int64) * stride + s0
        x = idx.astype(np.float32) / float(self._sr)

        self._disp_x = x
        self._disp_y = y.astype(np.float32, copy=False)

    def _status_text(self) -> str:
        dur = self._duration_s
        ph = self._playhead_s
        v0 = self._view_start_s
        v1 = self._view_end_s
        return f"{self.label}  |  t={ph:.2f}s / {dur:.2f}s  |  view=[{v0:.2f}s..{v1:.2f}s]"
