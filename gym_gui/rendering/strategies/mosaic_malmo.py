"""Renderer strategy for MOSAIC Malmo (Minecraft) environments.

Native-only rendering: mirrors the real Minecraft client window into MOSAIC.
No VideoProducer fallback is used in this mode.

Multi-agent support: when the adapter payload contains ``"multi_agent": True``
and an ``"agents"`` list, the renderer switches to a resizable split-view
using ``QSplitter``.  Each agent gets its own ``_MalmoView`` panel.  The
divider is draggable (resize), collapsible (drag to edge for fullscreen),
and resets to 50/50 on double-click.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Callable, Mapping

from qtpy import QtCore, QtGui, QtWidgets

from gym_gui.core.enums import RenderMode
from gym_gui.rendering.interfaces import RendererContext, RendererStrategy
from gym_gui.services.trainer.video_frame_service import get_frame_store

_LOGGER = logging.getLogger(__name__)

# Upstream Malmo default tick rate (ms)
_DEFAULT_MS_PER_TICK = 50


class MosaicMalmoRendererStrategy(RendererStrategy):
    """Render MOSAIC Malmo — single-agent or multi-agent split-view."""

    mode = RenderMode.MOSAIC_MALMO

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        self._parent = parent

        # Wrapper widget that stays in the stacked widget. Internally swaps
        # between single-agent _MalmoView and multi-agent _MalmoSplitView.
        self._wrapper = QtWidgets.QStackedWidget(parent)

        # Single-agent view (index 0)
        self._view = _MalmoView(self._wrapper)
        self._wrapper.addWidget(self._view)

        # Multi-agent split-view (created on demand, index 1)
        self._split_container: _MalmoSplitView | None = None
        self._multi_agent_mode = False

    # ------------------------------------------------------------------
    # Mouse / interaction callbacks (forwarded from active view)
    # ------------------------------------------------------------------

    def set_mouse_action_callback(self, callback: Callable[[int], None] | None) -> None:
        self._view.set_mouse_action_callback(callback)
        if self._split_container is not None:
            for v in self._split_container.agent_views:
                v.set_mouse_action_callback(callback)

    def set_mouse_delta_callback(self, callback: Callable[[float, float], None] | None) -> None:
        self._view.set_mouse_delta_callback(callback)
        if self._split_container is not None:
            for v in self._split_container.agent_views:
                v.set_mouse_delta_callback(callback)

    def set_mouse_delta_scale(self, scale: float) -> None:
        self._view.set_mouse_delta_scale(scale)
        if self._split_container is not None:
            for v in self._split_container.agent_views:
                v.set_mouse_delta_scale(scale)

    def set_mouse_capture_enabled(self, enabled: bool) -> None:
        self._view.set_mouse_capture_enabled(enabled)
        if self._split_container is not None:
            for v in self._split_container.agent_views:
                v.set_mouse_capture_enabled(enabled)

    # ------------------------------------------------------------------
    # RendererStrategy protocol
    # ------------------------------------------------------------------

    @property
    def widget(self) -> QtWidgets.QWidget:
        return self._wrapper

    def render(
        self, payload: Mapping[str, object], *, context: RendererContext | None = None
    ) -> None:
        is_multi = bool(payload.get("multi_agent"))
        agents = payload.get("agents")

        if is_multi and isinstance(agents, list) and len(agents) >= 2:
            self._ensure_split_view(len(agents))
            self._split_container.render_agents(agents)
            self._wrapper.setCurrentIndex(1)  # Show split view
            return

        # Single-agent fallback
        self._wrapper.setCurrentIndex(0)  # Show single view
        frame = payload.get("rgb")
        if frame is None:
            self.reset()
            return
        self._view.render_frame(frame, payload=payload)

    def supports(self, payload: Mapping[str, object]) -> bool:
        return "rgb" in payload

    def reset(self) -> None:
        self._view.reset()
        if self._split_container is not None:
            self._split_container.reset()

    def cleanup(self) -> None:
        try:
            self._view.reset()
            if self._split_container is not None:
                self._split_container.reset()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Multi-agent split-view management
    # ------------------------------------------------------------------

    def _ensure_split_view(self, agent_count: int) -> None:
        """Create or resize the split-view container on demand."""
        if self._split_container is not None:
            if len(self._split_container.agent_views) == agent_count:
                if not self._multi_agent_mode:
                    self._multi_agent_mode = True
                return
            # Agent count changed — rebuild
            self._wrapper.removeWidget(self._split_container)
            self._split_container.deleteLater()
            self._split_container = None

        self._split_container = _MalmoSplitView(
            agent_count=agent_count, parent=self._wrapper
        )
        self._wrapper.addWidget(self._split_container)  # index 1
        self._multi_agent_mode = True


# ======================================================================
# Multi-agent split-view container
# ======================================================================


class _MalmoSplitView(QtWidgets.QWidget):
    """Resizable split-view container holding one ``_MalmoView`` per agent.

    Uses a ``QSplitter`` for drag-to-resize panels.  Each panel is
    collapsible (drag divider to edge) and the splitter resets to equal
    sizes on double-click of the handle.

    Layout::

        ┌───────────────────────────────────────┐
        │  ┌──────────┬──┬──────────┐           │
        │  │ Agent 0  │◄►│ Agent 1  │           │
        │  │  POV     │  │  POV     │           │
        │  └──────────┴──┴──────────┘           │
        └───────────────────────────────────────┘
    """

    def __init__(
        self,
        agent_count: int = 2,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._splitter = _DoubleClickResetSplitter(QtCore.Qt.Orientation.Horizontal, self)
        self._splitter.setHandleWidth(6)
        self._splitter.setChildrenCollapsible(True)
        self._splitter.setStyleSheet(
            "QSplitter::handle { background: #444; border: 1px solid #555; }"
        )

        self.agent_views: list[_MalmoView] = []
        for role in range(agent_count):
            view = _MalmoView(self._splitter)
            view.setMinimumSize(150, 150)
            self._splitter.addWidget(view)
            self.agent_views.append(view)

        # Start with equal sizes
        self._splitter.setSizes([1] * agent_count)

        # Agent labels overlaid on each panel
        self._labels: list[QtWidgets.QLabel] = []
        for role, view in enumerate(self.agent_views):
            label = QtWidgets.QLabel(f"Agent {role}", view)
            label.setStyleSheet(
                "QLabel {"
                "  color: white;"
                "  background: rgba(0, 0, 0, 160);"
                "  padding: 2px 8px;"
                "  border-radius: 3px;"
                "  font-weight: bold;"
                "  font-size: 11px;"
                "}"
            )
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            label.adjustSize()
            label.move(4, 4)
            label.raise_()
            self._labels.append(label)

        layout.addWidget(self._splitter)

    def render_agents(self, agents: list) -> None:
        """Push per-agent payloads to each panel's ``_MalmoView``."""
        for role, agent_payload in enumerate(agents):
            if role >= len(self.agent_views):
                break
            frame = agent_payload.get("rgb")
            self.agent_views[role].render_frame(frame, payload=agent_payload)

    def reset(self) -> None:
        for view in self.agent_views:
            view.reset()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        # Keep equal split on first show / parent resize if user hasn't dragged
        if not self._splitter._user_has_dragged:
            n = len(self.agent_views)
            if n > 0:
                self._splitter.setSizes([1] * n)


class _DoubleClickResetSplitter(QtWidgets.QSplitter):
    """QSplitter whose handle resets all panels to equal size on double-click."""

    _user_has_dragged: bool = False

    def createHandle(self) -> QtWidgets.QSplitterHandle:
        handle = _ResetHandle(self.orientation(), self)
        return handle

    def _on_user_drag(self) -> None:
        self._user_has_dragged = True

    def reset_equal(self) -> None:
        """Snap back to equal sizes."""
        n = self.count()
        if n > 0:
            self.setSizes([1] * n)
        self._user_has_dragged = False


class _ResetHandle(QtWidgets.QSplitterHandle):
    """Splitter handle that resets to 50/50 on double-click."""

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        splitter = self.splitter()
        if isinstance(splitter, _DoubleClickResetSplitter):
            splitter._on_user_drag()
        super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        splitter = self.splitter()
        if isinstance(splitter, _DoubleClickResetSplitter):
            splitter.reset_equal()
        event.accept()


# ======================================================================
# Single-agent view (also used as panel inside split-view)
# ======================================================================


class _MalmoView(QtWidgets.QWidget):
    """Widget displaying mirrored native Minecraft window content."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.setMinimumSize(300, 250)
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QtGui.QColor(17, 17, 17))
        self.setPalette(palette)

        self._current_pixmap: QtGui.QPixmap | None = None
        self._hud_lines: list[str] = []
        self._image_rect: QtCore.QRect | None = None
        self._native_window_id: int | None = None
        self._last_native_lookup = 0.0
        self._native_lookup_interval_s = 1.0
        self._malmo_pid: int | None = None
        self._malmo_display: str | None = None
        self._last_proc_scan = 0.0
        self._proc_scan_interval_s = 1.0

        # Mouse capture state
        self._mouse_capture_enabled = False
        self._mouse_captured = False
        self._mouse_action_callback: Callable[[int], None] | None = None
        self._mouse_delta_callback: Callable[[float, float], None] | None = None
        self._last_mouse_pos: QtCore.QPoint | None = None
        self._mouse_sensitivity = 5.0
        self._mouse_delta_scale = 0.5
        self._accumulated_delta_x = 0.0
        self._use_delta_mode = False
        # Defaults; env loader can override per mission action space.
        self._turn_left_action = 3   # "turn -1"
        self._turn_right_action = 2  # "turn 1"

        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

        # Direct frame streaming timer (60 FPS)
        self._frame_timer = QtCore.QTimer(self)
        self._frame_timer.timeout.connect(self._check_video_stream)
        self._frame_timer.start(16)
        self._last_stream_frame_num = -1

    def _check_video_stream(self) -> None:
        """Poll the video frame store for fresh frames from Minecraft."""
        store = get_frame_store()
        frame = store.get_latest_frame()
        if frame is None:
            return

        # Skip if we've already rendered this frame
        if frame.frame_number <= self._last_stream_frame_num:
            return

        self._last_stream_frame_num = frame.frame_number
        
        # Convert to QImage directly from bytes (fast path)
        w, h = frame.width, frame.height
        if w <= 0 or h <= 0 or not frame.rgb_data:
            return

        # 3 bytes per pixel for RGB
        total_bytes = w * h * 3
        if len(frame.rgb_data) != total_bytes:
            return

        # Create QImage from raw bytes
        # Note: We must copy() because QImage doesn't own the buffer by default
        # and frame.rgb_data might be reclaimed.
        img = QtGui.QImage(
            frame.rgb_data, 
            w, 
            h, 
            w * 3, 
            QtGui.QImage.Format.Format_RGB888
        ).copy()

        self._current_pixmap = QtGui.QPixmap.fromImage(img)
        
        # Build HUD from frame metadata
        # self._hud_lines = self._build_hud_from_frame(frame)
        self.update()

    # def _build_hud_from_frame(self, frame) -> list[str]:
    #    """Extract HUD overlay text from the video frame metadata."""
    #    lines: list[str] = []
    #
    #    # Agent position
    #    if frame.HasField("agent_x"):
    #        lines.append(f"XYZ: {frame.agent_x:.1f} / {frame.agent_y:.1f} / {frame.agent_z:.1f}")
    #
    #    # Facing
    #    if frame.HasField("agent_yaw"):
    #        lines.append(f"Facing: {frame.agent_yaw:.1f} / {frame.agent_pitch:.1f}")
    #
    #    # Frame stats
    #    lines.append(f"Frame: {frame.frame_number}")
    #
    #    # Status
    #    status_parts: list[str] = []
    #    if frame.HasField("life"):
    #        status_parts.append(f"HP {frame.life:.1f}")
    #    if frame.HasField("food"):
    #        status_parts.append(f"Food {frame.food:.0f}")
    #    if frame.HasField("xp"):
    #        status_parts.append(f"XP {frame.xp}")
    #    
    #    if status_parts:
    #        lines.append(" | ".join(status_parts))
    #        
    #    if frame.HasField("reward"):
    #        lines.append(f"Reward: {frame.reward:.2f}")
    #
    #    return lines


    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(600, 500)

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(200, 200)

    # ------------------------------------------------------------------
    # Mouse interaction (mirrors RgbView)
    # ------------------------------------------------------------------

    def set_mouse_action_callback(self, callback: Callable[[int], None] | None) -> None:
        self._mouse_action_callback = callback
        self._use_delta_mode = False

    def set_mouse_delta_callback(self, callback: Callable[[float, float], None] | None) -> None:
        self._mouse_delta_callback = callback
        self._use_delta_mode = callback is not None

    def set_mouse_delta_scale(self, scale: float) -> None:
        self._mouse_delta_scale = max(0.01, scale)

    def set_mouse_capture_enabled(self, enabled: bool) -> None:
        self._mouse_capture_enabled = enabled
        if not enabled and self._mouse_captured:
            self._release_mouse()

    def set_turn_action_indices(self, turn_left: int, turn_right: int) -> None:
        self._turn_left_action = turn_left
        self._turn_right_action = turn_right

    def set_mouse_sensitivity(self, sensitivity: float) -> None:
        self._mouse_sensitivity = max(0.1, sensitivity)

    def _release_mouse(self) -> None:
        self._mouse_captured = False
        self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        self.releaseMouse()
        self._last_mouse_pos = None
        self._accumulated_delta_x = 0.0

    def _capture_mouse(self) -> None:
        self._mouse_captured = True
        self.setCursor(QtCore.Qt.CursorShape.BlankCursor)
        self.grabMouse()
        center = self.mapToGlobal(self.rect().center())
        QtGui.QCursor.setPos(center)
        self._last_mouse_pos = center
        self._accumulated_delta_x = 0.0

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _scan_malmo_process(self) -> None:
        """Best-effort discover running Minecraft/Malmo Java process."""
        now = time.monotonic()
        if (now - self._last_proc_scan) < self._proc_scan_interval_s:
            return
        self._last_proc_scan = now
        self._malmo_pid = None
        self._malmo_display = None

        proc_root = Path("/proc")
        for proc_dir in proc_root.iterdir():
            if not proc_dir.name.isdigit():
                continue
            pid = int(proc_dir.name)
            try:
                cmdline = (proc_dir / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "ignore")
            except Exception:
                continue
            cmdline_lower = cmdline.lower()
            if "java" not in cmdline_lower:
                continue
            if "gradlestart" not in cmdline_lower and "malmomod" not in cmdline_lower:
                continue
            try:
                env_raw = (proc_dir / "environ").read_bytes()
                env = env_raw.replace(b"\x00", b"\n").decode("utf-8", "ignore")
            except Exception:
                env = ""
            display = None
            for line in env.splitlines():
                if line.startswith("DISPLAY="):
                    display = line.split("=", 1)[1].strip()
                    break
            self._malmo_pid = pid
            self._malmo_display = display
            return

    def _status_message(self) -> str:
        current_display = os.environ.get("DISPLAY", "unset")
        self._scan_malmo_process()
        if self._malmo_pid is None:
            return "MOSAIC Malmo — waiting for Minecraft process..."
        if self._malmo_display and self._malmo_display != current_display:
            # Different displays is fine - we fall back to RGB frames from MalmoEnv
            return (
                f"MOSAIC Malmo — receiving frames from Minecraft (display {self._malmo_display})"
            )
        return "MOSAIC Malmo — waiting for frames..."

    def _find_minecraft_window_id(self) -> int | None:
        """Best-effort locate the native Minecraft window on X11."""
        if not os.environ.get("DISPLAY"):
            return None
        self._scan_malmo_process()
        ids: list[str] = []
        try:
            result = subprocess.run(
                ["xprop", "-root", "_NET_CLIENT_LIST"],
                capture_output=True,
                text=True,
                check=False,
                timeout=0.25,
            )
            if result.returncode == 0:
                ids = re.findall(r"0x[0-9a-fA-F]+", result.stdout)
            if not ids:
                tree = subprocess.run(
                    ["xwininfo", "-root", "-tree"],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=0.4,
                )
                if tree.returncode == 0:
                    ids = re.findall(r"\\b0x[0-9a-fA-F]+\\b", tree.stdout)
            for win_id in reversed(ids):
                if self._malmo_pid is not None:
                    pid_meta = subprocess.run(
                        ["xprop", "-id", win_id, "_NET_WM_PID"],
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=0.25,
                    )
                    pid_match = re.search(r"=\s*(\d+)", pid_meta.stdout or "")
                    if pid_match and int(pid_match.group(1)) == self._malmo_pid:
                        return int(win_id, 16)
                meta = subprocess.run(
                    ["xprop", "-id", win_id, "WM_CLASS", "WM_NAME", "_NET_WM_NAME"],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=0.25,
                )
                text = (meta.stdout + "\n" + meta.stderr).lower()
                if "minecraft" in text or "lwjgl" in text or "malmo" in text or "forge" in text:
                    try:
                        return int(win_id, 16)
                    except ValueError:
                        continue
        except Exception:
            return None
        return None

    def _grab_native_minecraft_pixmap(self) -> QtGui.QPixmap | None:
        """Grab current native Minecraft window contents, if available."""
        now = time.monotonic()
        if self._native_window_id is None or (now - self._last_native_lookup) >= self._native_lookup_interval_s:
            self._native_window_id = self._find_minecraft_window_id()
            self._last_native_lookup = now
        if self._native_window_id is None:
            return None

        screen = self.window().windowHandle().screen() if self.window() and self.window().windowHandle() else QtGui.QGuiApplication.primaryScreen()
        if screen is None:
            return None
        pix = screen.grabWindow(self._native_window_id)
        if pix.isNull():
            self._native_window_id = None
            return None
        return pix

    def render_frame(
        self,
        frame: object,
        *,
        payload: Mapping[str, object] | None = None,
    ) -> None:
        """Render MalmoEnv RGB frames directly from the adapter.

        This strategy only uses RGB frames from MalmoEnv TCP, not native window
        mirroring. This works correctly with Xvfb headless mode where Minecraft
        runs on a different display than MOSAIC.
        """
        if frame is not None and payload is not None:
            rgb = payload.get("rgb")
            if rgb is not None:
                import numpy as np
                arr = np.asarray(rgb, dtype=np.uint8)
                if arr.ndim == 3 and arr.shape[2] in (3, 4):
                    h, w, c = arr.shape
                    # Convert RGB/RGBA to QImage
                    fmt = QtGui.QImage.Format.Format_RGB888 if c == 3 else QtGui.QImage.Format.Format_RGBA8888
                    qimg = QtGui.QImage(arr.tobytes(), w, h, w * c, fmt).copy()
                    self._current_pixmap = QtGui.QPixmap.fromImage(qimg)
                    # self._hud_lines = self._build_hud(payload)
                    self.update()
                    return

        # No frame available - show status
        self._current_pixmap = None
        self._hud_lines = []
        self.update()

    @staticmethod
    def _build_hud(payload: Mapping[str, object]) -> list[str]:
        """Extract HUD overlay text from the render payload."""
        lines: list[str] = []

        def _num(name: str) -> float | None:
            value = payload.get(name)
            try:
                return None if value is None else float(value)
            except (TypeError, ValueError):
                return None

        # Agent position (upstream Malmo debug overlay shows XYZ)
        pos = payload.get("agent_pos")
        if isinstance(pos, (list, tuple)) and len(pos) >= 3:
            lines.append(f"XYZ: {pos[0]:.1f} / {pos[1]:.1f} / {pos[2]:.1f}")

        # Yaw / pitch
        yaw = payload.get("agent_yaw")
        pitch = payload.get("agent_pitch")
        if yaw is not None or pitch is not None:
            y = float(yaw) if yaw is not None else 0.0
            p = float(pitch) if pitch is not None else 0.0
            lines.append(f"Facing: {y:.1f} / {p:.1f}")

        # Tick / timing
        tick = payload.get("tick")
        ms_per_tick = payload.get("ms_per_tick")
        if tick is not None:
            tps = 1000 / int(ms_per_tick) if ms_per_tick else 20
            lines.append(f"Tick: {tick}  ({tps:.0f} tps)")

        # Mission name
        mission = payload.get("mission")
        if mission:
            lines.append(str(mission))

        # Reward
        reward = payload.get("reward")
        if reward is not None:
            lines.append(f"Reward: {float(reward):.2f}")

        life = _num("life")
        food = _num("food")
        air = _num("air")
        xp = _num("xp")
        score = _num("score")
        is_alive = payload.get("is_alive")

        status_parts: list[str] = []
        if life is not None:
            status_parts.append(f"HP {life:.1f}")
        if food is not None:
            status_parts.append(f"Food {food:.0f}")
        if air is not None:
            status_parts.append(f"Air {air:.0f}")
        if xp is not None:
            status_parts.append(f"XP {xp:.0f}")
        if status_parts:
            lines.append(" | ".join(status_parts))
        if score is not None:
            lines.append(f"Score: {score:.0f}")
        if isinstance(is_alive, bool):
            lines.append("Alive" if is_alive else "Dead")

        return lines

    def reset(self) -> None:
        self._current_pixmap = None
        self._hud_lines = []
        self._image_rect = None
        self.update()

    # ------------------------------------------------------------------
    # Qt painting
    # ------------------------------------------------------------------

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, False)

        if self._current_pixmap is None:
            painter.fillRect(self.rect(), QtGui.QColor(17, 17, 17))
            painter.setPen(QtGui.QColor(100, 100, 100))
            painter.drawText(
                self.rect(),
                QtCore.Qt.AlignmentFlag.AlignCenter,
                self._status_message(),
            )
            painter.end()
            return

        # Scale pixmap preserving aspect ratio (nearest-neighbour for pixel art)
        pm = self._current_pixmap
        widget_rect = self.rect()
        scaled = pm.size().scaled(widget_rect.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        x = (widget_rect.width() - scaled.width()) // 2
        y = (widget_rect.height() - scaled.height()) // 2
        target = QtCore.QRect(x, y, scaled.width(), scaled.height())
        self._image_rect = target

        painter.fillRect(widget_rect, QtGui.QColor(17, 17, 17))
        painter.drawPixmap(target, pm)

        # HUD overlay (semi-transparent background, white text)
        if self._hud_lines:
            font = painter.font()
            font.setFamily("Monospace")
            font.setPixelSize(max(12, scaled.height() // 30))
            painter.setFont(font)
            fm = QtGui.QFontMetrics(font)
            line_h = fm.height() + 2
            margin = 6
            hud_h = line_h * len(self._hud_lines) + margin * 2
            max_w = max(fm.horizontalAdvance(ln) for ln in self._hud_lines)
            hud_w = max_w + margin * 2

            hud_rect = QtCore.QRect(x + 4, y + 4, hud_w, hud_h)
            painter.fillRect(hud_rect, QtGui.QColor(0, 0, 0, 140))
            painter.setPen(QtGui.QColor(255, 255, 255))
            for i, line in enumerate(self._hud_lines):
                painter.drawText(
                    x + 4 + margin,
                    y + 4 + margin + (i + 1) * line_h - 2,
                    line,
                )

        painter.end()

    # ------------------------------------------------------------------
    # Mouse events (capture for FPS-style control)
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self._mouse_capture_enabled:
            return super().mousePressEvent(event)
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if not self._mouse_captured:
                self._capture_mouse()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self._mouse_captured or self._last_mouse_pos is None:
            return super().mouseMoveEvent(event)
        current = QtGui.QCursor.pos()
        dx = current.x() - self._last_mouse_pos.x()
        dy = current.y() - self._last_mouse_pos.y()

        if self._use_delta_mode and self._mouse_delta_callback:
            self._mouse_delta_callback(
                dx * self._mouse_delta_scale, dy * self._mouse_delta_scale
            )
        elif self._mouse_action_callback:
            self._accumulated_delta_x += dx
            while abs(self._accumulated_delta_x) >= self._mouse_sensitivity:
                if self._accumulated_delta_x > 0:
                    self._mouse_action_callback(self._turn_right_action)
                    self._accumulated_delta_x -= self._mouse_sensitivity
                else:
                    self._mouse_action_callback(self._turn_left_action)
                    self._accumulated_delta_x += self._mouse_sensitivity

        center = self.mapToGlobal(self.rect().center())
        QtGui.QCursor.setPos(center)
        self._last_mouse_pos = center
        event.accept()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Escape and self._mouse_captured:
            self._release_mouse()
            return
        super().keyPressEvent(event)

    def focusOutEvent(self, event: QtGui.QFocusEvent) -> None:
        if self._mouse_captured:
            self._release_mouse()
        super().focusOutEvent(event)
