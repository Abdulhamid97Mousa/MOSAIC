"""Render container for a single operator in multi-operator mode.

Each operator gets its own render container with:
- Header showing operator name and type badge
- Render area (Grid, Video, or Text)
- Compact telemetry stats (step, episode, reward)
- Status indicator (pending, running, stopped)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from PyQt6.QtCore import pyqtSignal  # type: ignore[attr-defined]
from qtpy import QtCore, QtGui, QtWidgets

from gym_gui.controllers.human_input import (
    _ALE_MAPPINGS,
    _BABAISAI_MAPPINGS,
    _BOX_2D_MAPPINGS,
    _CRAFTER_MAPPINGS,
    _JUMANJI_MAPPINGS,
    _MINIG_GRID_MAPPINGS,
    _MINIHACK_MAPPINGS,
    _NETHACK_MAPPINGS,
    _PROCGEN_MAPPINGS,
    _TOY_TEXT_MAPPINGS,
    _VIZDOOM_MAPPINGS,
)
from gym_gui.core.enums import GameId, RenderMode
from gym_gui.logging_config.helpers import log_constant
from gym_gui.logging_config.log_constants import (
    LOG_HUMAN_ACTION_BUTTON_CLICKED,
    LOG_HUMAN_ACTION_SIGNAL_EMITTED,
)
from gym_gui.rendering import RendererContext, RendererRegistry, create_default_renderer_registry
from gym_gui.rendering.strategies.board_game import BoardGameRendererStrategy
from gym_gui.services.operator import OperatorConfig
from gym_gui.ui.widgets.multi_agent_action_panel import (
    COLOR_PALETTE,
    DEFAULT_AGENT_COLOR_NAMES,
)

# Use operators namespace for dedicated operators.log routing
_LOGGER = logging.getLogger("gym_gui.operators.render_container")


class FlowLayout(QtWidgets.QLayout):
    """A layout that arranges widgets in a flow, wrapping to new lines as needed.

    Based on Qt's FlowLayout example. Widgets are added left-to-right and wrap
    to the next line when they exceed the available width.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, margin: int = 0, spacing: int = -1):
        super().__init__(parent)
        self._item_list: list[QtWidgets.QLayoutItem] = []
        self._h_spacing = spacing if spacing >= 0 else 4
        self._v_spacing = spacing if spacing >= 0 else 4
        self.setContentsMargins(margin, margin, margin, margin)

    def addItem(self, item: QtWidgets.QLayoutItem) -> None:
        self._item_list.append(item)

    def horizontalSpacing(self) -> int:
        return self._h_spacing

    def verticalSpacing(self) -> int:
        return self._v_spacing

    def count(self) -> int:
        return len(self._item_list)

    def itemAt(self, index: int) -> Optional[QtWidgets.QLayoutItem]:
        if 0 <= index < len(self._item_list):
            return self._item_list[index]
        return None

    def takeAt(self, index: int) -> Optional[QtWidgets.QLayoutItem]:
        if 0 <= index < len(self._item_list):
            return self._item_list.pop(index)
        return None

    def expandingDirections(self) -> QtCore.Qt.Orientation:
        return QtCore.Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._do_layout(QtCore.QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QtCore.QRect) -> None:
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QtCore.QSize:
        return self.minimumSize()

    def minimumSize(self) -> QtCore.QSize:
        size = QtCore.QSize()
        for item in self._item_list:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QtCore.QSize(margins.left() + margins.right(), margins.top() + margins.bottom())
        return size

    def _do_layout(self, rect: QtCore.QRect, test_only: bool) -> int:
        """Arrange items in the layout, wrapping as needed.

        Args:
            rect: The available rectangle for layout.
            test_only: If True, just calculate height without moving widgets.

        Returns:
            The total height needed.
        """
        margins = self.contentsMargins()
        effective_rect = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        x = effective_rect.x()
        y = effective_rect.y()
        line_height = 0

        for item in self._item_list:
            widget = item.widget()
            if widget is None:
                continue

            space_x = self._h_spacing
            space_y = self._v_spacing

            next_x = x + item.sizeHint().width() + space_x
            if next_x - space_x > effective_rect.right() and line_height > 0:
                x = effective_rect.x()
                y = y + line_height + space_y
                next_x = x + item.sizeHint().width() + space_x
                line_height = 0

            if not test_only:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), item.sizeHint()))

            x = next_x
            line_height = max(line_height, item.sizeHint().height())

        return y + line_height - rect.y() + margins.bottom()


class _ResizeGrip(QtWidgets.QWidget):
    """Custom resize grip for widgets embedded inside a layout.

    Unlike ``QSizeGrip`` (which resizes the top-level window), this widget
    resizes its *target* widget via ``setFixedSize`` so that the parent layout
    respects the user-chosen dimensions.
    """

    def __init__(
        self,
        target: QtWidgets.QWidget,
        parent: Optional[QtWidgets.QWidget] = None,
        color: str = "#999999",
    ) -> None:
        super().__init__(parent or target)
        self._target = target
        self._dragging = False
        self._start_pos: Optional[QtCore.QPoint] = None
        self._start_size: Optional[QtCore.QSize] = None
        self._dot_color = color
        self.setFixedSize(16, 16)
        self.setCursor(QtCore.Qt.CursorShape.SizeFDiagCursor)

    # -- painting -----------------------------------------------------------

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        """Draw a small triangular grip indicator."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        color = QtGui.QColor(self._dot_color)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(color)
        # Draw three small dots in a diagonal pattern
        for dx, dy in ((10, 14), (14, 14), (14, 10), (8, 10), (10, 10), (10, 8)):
            painter.drawEllipse(dx, dy, 2, 2)
        painter.end()

    # -- mouse handling -----------------------------------------------------

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._dragging = True
            self._start_pos = event.globalPosition().toPoint()
            self._start_size = self._target.size()
            _LOGGER.info(
                "ResizeGrip: drag START on %s, start_size=%dx%d",
                type(self._target).__name__,
                self._start_size.width(), self._start_size.height(),
            )
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:  # noqa: N802
        if self._dragging and self._start_pos is not None and self._start_size is not None:
            delta = event.globalPosition().toPoint() - self._start_pos
            new_w = max(250, self._start_size.width() + delta.x())
            new_h = max(200, self._start_size.height() + delta.y())
            self._target.setFixedSize(new_w, new_h)
            _LOGGER.debug("ResizeGrip: drag MOVE → %dx%d", new_w, new_h)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:  # noqa: N802
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            # Emit resize signal via the target (if it has the method)
            if hasattr(self._target, '_on_grip_resize_finished'):
                self._target._on_grip_resize_finished()
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class _ImageScaleGrip(QtWidgets.QWidget):
    """Grip that scales the rendered image by adjusting ``square_size``.

    Dragging this grip computes a new ``square_size`` proportional to the
    drag distance and writes it into the operator config settings.  The
    owning ``OperatorRenderContainer`` then re-renders the current frame at
    the new scale.
    """

    def __init__(
        self,
        container: "OperatorRenderContainer",
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._container = container
        self._dragging = False
        self._start_pos: Optional[QtCore.QPoint] = None
        self._start_square_size: int = 0
        self.setFixedSize(16, 16)
        self.setCursor(QtCore.Qt.CursorShape.SizeFDiagCursor)

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        # Blue-ish tint to distinguish from the grey container grip
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor("#4488cc"))
        for dx, dy in ((10, 14), (14, 14), (14, 10), (8, 10), (10, 10), (10, 8)):
            painter.drawEllipse(dx, dy, 2, 2)
        painter.end()

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._dragging = True
            self._start_pos = event.globalPosition().toPoint()
            cur = self._container._config.settings.get("square_size", 70)
            self._start_square_size = cur if cur else 70
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:  # noqa: N802
        if self._dragging and self._start_pos is not None:
            delta = event.globalPosition().toPoint() - self._start_pos
            # Use the larger of dx/dy so dragging in either direction works
            pixel_delta = max(delta.x(), delta.y())
            # Every 3 pixels of drag = 1 unit of square_size change
            new_sq = max(10, self._start_square_size + pixel_delta // 3)
            new_sq = min(new_sq, 300)  # cap at 300px per tile
            self._container._config.settings["square_size"] = new_sq
            # Force the renderer to repaint with the new scale
            renderer = self._container._renderer_strategy
            if renderer is not None:
                widget = renderer.widget
                sq_attr = getattr(widget, "_square_size", None)
                if sq_attr is not None or hasattr(widget, "_square_size"):
                    widget._square_size = new_sq
                    widget.update()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:  # noqa: N802
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            new_sq = self._container._config.settings.get("square_size", 70)
            _LOGGER.info("ImageScaleGrip: final square_size=%s", new_sq)
            # Also expand the container if the image now overflows
            if self._container._renderer_strategy:
                widget = self._container._renderer_strategy.widget
                img_rect = getattr(widget, "_image_rect", None)
                if img_rect and not img_rect.isNull():
                    needed_w = img_rect.width() + 40
                    needed_h = img_rect.height() + 100
                    cur_w = self._container.width()
                    cur_h = self._container.height()
                    if needed_w > cur_w or needed_h > cur_h:
                        self._container.setFixedSize(
                            max(cur_w, needed_w), max(cur_h, needed_h),
                        )
            if hasattr(self._container, '_on_grip_resize_finished'):
                self._container._on_grip_resize_finished()
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class _RenderAreaGrip(QtWidgets.QWidget):
    """Grip that resizes the active renderer widget (the dark area with the image).

    On drag it calls ``setFixedSize`` on the renderer widget (e.g. ``_RgbView``)
    and expands the parent ``OperatorRenderContainer`` to accommodate.
    """

    def __init__(
        self,
        container: "OperatorRenderContainer",
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._container = container
        self._dragging = False
        self._start_pos: Optional[QtCore.QPoint] = None
        self._start_size: Optional[QtCore.QSize] = None
        self.setFixedSize(16, 16)
        self.setCursor(QtCore.Qt.CursorShape.SizeFDiagCursor)

    def _renderer_widget(self) -> Optional[QtWidgets.QWidget]:
        strategy = self._container._renderer_strategy
        return strategy.widget if strategy is not None else None

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor("#44aa44"))  # green
        for dx, dy in ((10, 14), (14, 14), (14, 10), (8, 10), (10, 10), (10, 8)):
            painter.drawEllipse(dx, dy, 2, 2)
        painter.end()

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            w = self._renderer_widget()
            if w is not None:
                self._dragging = True
                self._start_pos = event.globalPosition().toPoint()
                self._start_size = w.size()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:  # noqa: N802
        if self._dragging and self._start_pos is not None and self._start_size is not None:
            w = self._renderer_widget()
            if w is None:
                return
            delta = event.globalPosition().toPoint() - self._start_pos
            new_w = max(200, self._start_size.width() + delta.x())
            new_h = max(150, self._start_size.height() + delta.y())
            # Resize the renderer widget (the dark area)
            w.setFixedSize(new_w, new_h)
            # Expand the container if the renderer now needs more room
            needed_w = new_w + 30   # side margins
            needed_h = new_h + 120  # header + stats + grip row
            cw, ch = self._container.width(), self._container.height()
            if needed_w > cw or needed_h > ch:
                self._container.setFixedSize(max(cw, needed_w), max(ch, needed_h))
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:  # noqa: N802
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            if hasattr(self._container, '_on_grip_resize_finished'):
                self._container._on_grip_resize_finished()
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class OperatorRenderContainer(QtWidgets.QFrame):
    """Render container for a single operator.

    Displays:
    - Header with operator name, type badge, and status
    - Render area for visual output (grid/video)
    - Stats bar with step/episode/reward counters

    For Human operators, this container is interactive:
    - Board games: click on board to make moves
    - Other envs: keyboard input to select actions
    """

    status_changed = pyqtSignal(str, str)  # operator_id, new_status
    # Human interaction signals
    human_action_submitted = pyqtSignal(str, int)  # operator_id, action_index
    mouse_delta_input = pyqtSignal(str, int, int)  # operator_id, dx, dy
    board_game_move_made = pyqtSignal(str, str, str)  # operator_id, from_square, to_square
    chess_move_button_clicked = pyqtSignal(str, str)  # operator_id, uci_move (e.g., "e2e4")
    # Resize signal: emitted when user drags the container edge to a new size
    container_resized = pyqtSignal(str, int, int)  # operator_id, width, height

    # Status colors
    STATUS_COLORS = {
        "pending": "#9E9E9E",    # Gray
        "loaded": "#2196F3",     # Blue - environment loaded/ready
        "running": "#4CAF50",    # Green
        "stopped": "#F44336",    # Red
        "error": "#FF9800",      # Orange
    }

    # Type badge colors
    TYPE_COLORS = {
        "llm": "#2196F3",  # Blue
        "vlm": "#00BCD4",  # Cyan
        "rl": "#9C27B0",   # Purple
        "human": "#FF5722",  # Deep Orange - distinct for human operators
    }

    def __init__(
        self,
        config: OperatorConfig,
        renderer_registry: Optional[RendererRegistry] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._status = "pending"
        self._renderer_registry = renderer_registry or create_default_renderer_registry()
        # Support multiple renderer types (GRID for text environments, RGB for visual)
        self._grid_renderer: Optional[Any] = None
        self._rgb_renderer: Optional[Any] = None
        self._board_game_renderer: Optional[BoardGameRendererStrategy] = None
        self._renderer_strategy: Optional[Any] = None  # Currently active renderer
        self._active_render_mode: Optional[RenderMode] = None
        self._is_board_game: bool = False  # Track if current payload is a board game

        # Stats tracking
        self._current_step = 0
        self._current_episode = 0
        self._total_reward = 0.0
        self._episode_reward = 0.0

        # LLM conversation tracking
        self._system_prompt: str = ""
        self._conversation_history: list = []  # List of {"role": "user"|"assistant", "content": str}

        # Human operator interaction tracking
        # Check if this is a human operator:
        # - Single-agent: operator_type == "human"
        # - Multi-agent: any worker has worker_type == "human" or worker_id == "human_worker"
        self._is_interactive: bool = self._has_human_worker(config)
        _LOGGER.debug(
            f"OperatorRenderContainer __init__: operator_id={config.operator_id}, "
            f"_is_interactive={self._is_interactive}, workers={list(config.workers.keys())}",
        )
        self._available_actions: list[int] = []  # Available action indices for human selection
        self._action_labels: list[str] = []  # Human-readable labels for actions
        self._action_buttons: list[QtWidgets.QPushButton] = []  # Action buttons for style updates
        self._selected_action: Optional[int] = None  # Currently selected action index
        self._game_id: Optional[GameId] = None  # Current game for key mappings
        self._key_mappings: dict[int, int] = {}  # Qt key -> action index

        # Mouse capture state
        self._mouse_captured = False
        self._mouse_center = QtCore.QPoint(0, 0)
        # Enable mouse capture for Malmo only (native speed hack)
        env_lower = self._config.env_name.lower() if self._config.env_name else ""
        self._mouse_capture_enabled = "malmo" in env_lower or "minecraft" in env_lower

        self._build_ui()
        self._update_header()

        # Enable keyboard focus for human operators
        if self._is_interactive:
            self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    def _log_extra(self) -> dict:
        """Return extra dict for correlated logging with run_id and operator_id."""
        return {
            "run_id": self._config.run_id or "unknown",
            "agent_id": self._config.operator_id,
        }

    @staticmethod
    def _has_human_worker(config: OperatorConfig) -> bool:
        """Check if the operator has any human workers.

        Args:
            config: The operator configuration.

        Returns:
            True if any worker is a human worker (single-agent or multi-agent).
        """
        # Single-agent mode: check operator_type directly
        if config.operator_type == "human":
            return True

        # Multi-agent mode: check if any worker is human
        for worker in config.workers.values():
            if worker.worker_type == "human" or worker.worker_id == "human_worker":
                return True

        return False

    def _build_ui(self) -> None:
        self.setFrameStyle(QtWidgets.QFrame.Shape.Box | QtWidgets.QFrame.Shadow.Raised)
        self.setLineWidth(1)
        self.setStyleSheet("QFrame { border: 1px solid #ccc; border-radius: 4px; }")

        # Container should expand to fill grid cell (Qt6 best practice)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        # Start with a large minimum size for better initial display
        self.setMinimumSize(400, 350)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header with name, type badge, and status
        self._header = QtWidgets.QWidget(self)
        header_layout = QtWidgets.QHBoxLayout(self._header)
        header_layout.setContentsMargins(4, 2, 4, 2)
        header_layout.setSpacing(8)

        # Operator name
        self._name_label = QtWidgets.QLabel(self._config.display_name, self._header)
        self._name_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        header_layout.addWidget(self._name_label)

        # Type badge (LLM / VLM / RL)
        self._type_badge = QtWidgets.QLabel(self._config.operator_type.upper(), self._header)
        type_color = self.TYPE_COLORS.get(self._config.operator_type, "#666")
        self._type_badge.setStyleSheet(
            f"background-color: {type_color}; color: white; "
            f"padding: 2px 6px; border-radius: 3px; font-size: 9px; font-weight: bold;"
        )
        header_layout.addWidget(self._type_badge)

        # Worker name (e.g., "BALROG LLM Worker")
        worker_name = self._get_worker_display_name()
        self._worker_label = QtWidgets.QLabel(worker_name, self._header)
        self._worker_label.setStyleSheet("color: #333; font-size: 10px;")
        header_layout.addWidget(self._worker_label)

        # Model/Policy info (e.g., "GPT-4o Mini" for LLM, policy name for RL)
        model_info = self._get_model_display_name()
        self._model_label = QtWidgets.QLabel(model_info, self._header)
        self._model_label.setStyleSheet("color: #666; font-size: 10px; font-style: italic;")
        header_layout.addWidget(self._model_label)

        # Environment/Task info
        env_task = f"{self._config.env_name}/{self._config.task}"
        self._env_label = QtWidgets.QLabel(env_task, self._header)
        self._env_label.setStyleSheet("color: #666; font-size: 10px;")
        self._env_label.setToolTip(f"Environment: {self._config.env_name}\nTask: {self._config.task}")
        header_layout.addWidget(self._env_label)

        header_layout.addStretch()

        # Status indicator
        self._status_indicator = QtWidgets.QLabel("PENDING", self._header)
        self._update_status_indicator()
        header_layout.addWidget(self._status_indicator)

        layout.addWidget(self._header)

        # Agent legend strip (multi-agent only)
        self._agent_legend = QtWidgets.QWidget(self)
        legend_layout = QtWidgets.QHBoxLayout(self._agent_legend)
        legend_layout.setContentsMargins(4, 1, 4, 1)
        legend_layout.setSpacing(6)
        self._build_agent_legend(legend_layout)
        layout.addWidget(self._agent_legend)

        # Render area - should expand to fill available space
        self._render_container = QtWidgets.QWidget(self)
        self._render_container.setMinimumSize(350, 280)  # Larger minimum for better display
        self._render_container.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        self._render_layout = QtWidgets.QVBoxLayout(self._render_container)
        self._render_layout.setContentsMargins(2, 2, 2, 2)
        self._render_layout.setSpacing(0)

        # Placeholder label
        self._placeholder = QtWidgets.QLabel("Waiting for data...", self._render_container)
        self._placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: #999; font-style: italic;")
        self._render_layout.addWidget(self._placeholder)

        layout.addWidget(self._render_container, 1)

        # Stats bar
        self._stats_bar = QtWidgets.QWidget(self)
        stats_layout = QtWidgets.QHBoxLayout(self._stats_bar)
        stats_layout.setContentsMargins(4, 2, 4, 2)
        stats_layout.setSpacing(16)

        # Episode counter
        self._episode_label = QtWidgets.QLabel("Episode: 0", self._stats_bar)
        self._episode_label.setStyleSheet("font-size: 10px;")
        stats_layout.addWidget(self._episode_label)

        # Step counter
        self._step_label = QtWidgets.QLabel("Step: 0", self._stats_bar)
        self._step_label.setStyleSheet("font-size: 10px;")
        stats_layout.addWidget(self._step_label)

        # Episode reward
        self._reward_label = QtWidgets.QLabel("Reward: 0.00", self._stats_bar)
        self._reward_label.setStyleSheet("font-size: 10px;")
        stats_layout.addWidget(self._reward_label)

        stats_layout.addStretch()

        # LLM info buttons (only visible for LLM/VLM operators)
        self._prompt_btn = QtWidgets.QPushButton("Step Prompt", self._stats_bar)
        self._prompt_btn.setFixedSize(75, 22)
        self._prompt_btn.setToolTip("View the full prompt sent to LLM at current step")
        self._prompt_btn.setStyleSheet(
            "QPushButton { font-size: 9px; padding: 2px 4px; }"
            "QPushButton:hover { background-color: #e3f2fd; }"
        )
        self._prompt_btn.clicked.connect(self._show_prompt_dialog)
        stats_layout.addWidget(self._prompt_btn)

        self._chat_btn = QtWidgets.QPushButton("Chat", self._stats_bar)
        self._chat_btn.setFixedSize(45, 22)
        self._chat_btn.setToolTip("View LLM conversation history")
        self._chat_btn.setStyleSheet(
            "QPushButton { font-size: 9px; padding: 2px 4px; }"
            "QPushButton:hover { background-color: #e3f2fd; }"
        )
        self._chat_btn.clicked.connect(self._show_chat_dialog)
        stats_layout.addWidget(self._chat_btn)

        # Show/hide LLM buttons based on operator type (not for human operators)
        is_llm = self._config.operator_type in ("llm", "vlm")
        self._prompt_btn.setVisible(is_llm)
        self._chat_btn.setVisible(is_llm)

        # Add "Your Turn" indicator for human operators (hidden initially)
        self._your_turn_label = QtWidgets.QLabel("Your Turn!", self._stats_bar)
        self._your_turn_label.setStyleSheet(
            "QLabel { font-weight: bold; color: white; padding: 2px 8px; "
            "background-color: #FF5722; border-radius: 3px; font-size: 10px; }"
        )
        self._your_turn_label.setVisible(False)
        stats_layout.addWidget(self._your_turn_label)

        layout.addWidget(self._stats_bar)

        # Parallel multi-agent action panel container (hidden by default).
        # When parallel mode is active, the MultiAgentActionPanel is embedded here
        # so the human sees action rows right below the environment render.
        self._parallel_action_container = QtWidgets.QWidget(self)
        self._parallel_action_container_layout = QtWidgets.QVBoxLayout(self._parallel_action_container)
        self._parallel_action_container_layout.setContentsMargins(0, 0, 0, 0)
        self._parallel_action_container.setVisible(False)
        layout.addWidget(self._parallel_action_container)

        # Action panel for human operators (click-to-select actions)
        # Uses FlowLayout so buttons wrap to multiple lines
        self._action_panel = QtWidgets.QWidget(self)
        action_panel_layout = QtWidgets.QVBoxLayout(self._action_panel)
        action_panel_layout.setContentsMargins(4, 2, 4, 2)
        action_panel_layout.setSpacing(2)

        # Header row with "Actions:" label
        header_row = QtWidgets.QWidget(self._action_panel)
        header_layout = QtWidgets.QHBoxLayout(header_row)
        header_layout.setContentsMargins(0, 0, 0, 0)
        action_label = QtWidgets.QLabel("Actions:", header_row)
        action_label.setStyleSheet("font-size: 10px; font-weight: bold;")
        header_layout.addWidget(action_label)
        header_layout.addStretch()
        action_panel_layout.addWidget(header_row)

        # Selection indicator label
        self._selected_action_label = QtWidgets.QLabel("", self._action_panel)
        self._selected_action_label.setStyleSheet("font-size: 9px; color: #4CAF50; font-weight: bold;")
        self._selected_action_label.setVisible(False)
        action_panel_layout.addWidget(self._selected_action_label)

        # Container for action buttons with FlowLayout (wraps to multiple lines)
        self._action_buttons_container = QtWidgets.QWidget(self._action_panel)
        self._action_buttons_layout = FlowLayout(self._action_buttons_container, margin=0, spacing=3)
        action_panel_layout.addWidget(self._action_buttons_container)

        self._action_panel.setVisible(False)  # Hidden by default
        layout.addWidget(self._action_panel)

        # Chess moves panel for Human chess players (clickable legal move buttons)
        self._chess_moves_panel = QtWidgets.QWidget(self)
        chess_panel_layout = QtWidgets.QVBoxLayout(self._chess_moves_panel)
        chess_panel_layout.setContentsMargins(4, 2, 4, 2)
        chess_panel_layout.setSpacing(2)

        # Header row with "Legal Moves:" label and current player indicator
        chess_header = QtWidgets.QWidget(self._chess_moves_panel)
        chess_header_layout = QtWidgets.QHBoxLayout(chess_header)
        chess_header_layout.setContentsMargins(0, 0, 0, 0)
        chess_moves_label = QtWidgets.QLabel("Legal Moves:", chess_header)
        chess_moves_label.setStyleSheet("font-size: 10px; font-weight: bold;")
        chess_header_layout.addWidget(chess_moves_label)
        self._chess_player_label = QtWidgets.QLabel("", chess_header)
        self._chess_player_label.setStyleSheet("font-size: 10px; color: #666;")
        chess_header_layout.addWidget(self._chess_player_label)
        chess_header_layout.addStretch()
        chess_panel_layout.addWidget(chess_header)

        # Container for chess move buttons with FlowLayout
        self._chess_buttons_container = QtWidgets.QWidget(self._chess_moves_panel)
        self._chess_buttons_layout = FlowLayout(self._chess_buttons_container, margin=0, spacing=3)
        chess_panel_layout.addWidget(self._chess_buttons_container)

        self._chess_moves_panel.setVisible(False)  # Hidden by default
        layout.addWidget(self._chess_moves_panel)

        # Track current chess state
        self._chess_legal_moves: list[str] = []
        self._chess_current_player: str = ""

        # --- Three resize grips ---
        # 1) Image grip (blue): scales the environment image via square_size.
        #    Overlaid at the rendered image's bottom-right corner.
        self._image_grip = _ImageScaleGrip(container=self, parent=self._render_container)
        self._image_grip.raise_()
        self._render_container.installEventFilter(self)

        # 2) Render-area grip (green): resizes the renderer widget (_RgbView).
        #    Overlaid at the renderer widget's bottom-right corner.
        self._render_area_grip = _RenderAreaGrip(container=self, parent=self._render_container)
        self._render_area_grip.raise_()

        # 3) Container grip (grey): resizes the whole operator frame.
        #    Sits in a layout row at the very bottom.
        self._container_grip = _ResizeGrip(target=self, parent=self)
        container_grip_row = QtWidgets.QHBoxLayout()
        container_grip_row.setContentsMargins(0, 0, 0, 0)
        container_grip_row.addStretch()
        container_grip_row.addWidget(self._container_grip)
        layout.addLayout(container_grip_row)

        # Track preferred size for resize events
        self._preferred_width: int = 0
        self._preferred_height: int = 0

    def _build_agent_legend(self, legend_layout: QtWidgets.QHBoxLayout) -> None:
        """Build agent identification labels dynamically from operator config."""
        if not self._config.is_multiagent:
            self._agent_legend.hide()
            return

        player_ids = self._config.player_ids
        if not player_ids:
            self._agent_legend.hide()
            return

        for player_id in player_ids:
            worker = self._config.get_worker_for_player(player_id)
            if worker is None:
                continue

            # Extract agent index from player_id (e.g., "agent_2" → 2)
            try:
                agent_idx = int(player_id.split("_")[-1])
            except (ValueError, IndexError):
                agent_idx = 0

            # Resolve agent colour from settings or default palette
            color_name = worker.settings.get("agent_color") if worker.settings else None
            if color_name and color_name != "auto" and color_name in COLOR_PALETTE:
                agent_primary, _ = COLOR_PALETTE[color_name]
            else:
                default_name = DEFAULT_AGENT_COLOR_NAMES.get(player_id)
                if default_name and default_name in COLOR_PALETTE:
                    agent_primary, _ = COLOR_PALETTE[default_name]
                else:
                    agent_primary = "#666"

            type_color = self.TYPE_COLORS.get(worker.worker_type, "#666")
            worker_type_short = worker.worker_type.upper()

            chip = QtWidgets.QLabel(self._agent_legend)
            chip.setText(
                f'<span style="color:{agent_primary};">\u25CF</span>'
                f' <b>A{agent_idx}</b>'
                f' <span style="background:{type_color}; color:white;'
                f' padding:1px 4px; border-radius:2px; font-size:8px;">'
                f'{worker_type_short}</span>'
            )
            chip.setToolTip(
                f"{player_id}\n"
                f"Type: {worker.worker_type}\n"
                f"Worker: {worker.worker_id}\n"
                f"Color: {color_name or 'default'}"
            )
            chip.setStyleSheet("font-size: 10px; padding: 0 2px;")
            legend_layout.addWidget(chip)

        legend_layout.addStretch()
        self._agent_legend.setStyleSheet(
            "background-color: #f5f5f5; border-radius: 3px;"
        )

    def _update_header(self) -> None:
        """Update header with current config."""
        self._name_label.setText(self._config.display_name)
        self._type_badge.setText(self._config.operator_type.upper())
        type_color = self.TYPE_COLORS.get(self._config.operator_type, "#666")
        self._type_badge.setStyleSheet(
            f"background-color: {type_color}; color: white; "
            f"padding: 2px 6px; border-radius: 3px; font-size: 9px; font-weight: bold;"
        )
        # Update worker and model labels
        self._worker_label.setText(self._get_worker_display_name())
        self._model_label.setText(self._get_model_display_name())
        env_task = f"{self._config.env_name}/{self._config.task}"
        self._env_label.setText(env_task)
        # Update LLM button visibility
        is_llm = self._config.operator_type in ("llm", "vlm")
        self._prompt_btn.setVisible(is_llm)
        self._chat_btn.setVisible(is_llm)

    def _update_status_indicator(self) -> None:
        """Update status indicator appearance."""
        status_color = self.STATUS_COLORS.get(self._status, "#666")
        self._status_indicator.setText(self._status.upper())
        self._status_indicator.setStyleSheet(
            f"background-color: {status_color}; color: white; "
            f"padding: 2px 6px; border-radius: 3px; font-size: 9px; font-weight: bold;"
        )

    def _get_worker_display_name(self) -> str:
        """Get a user-friendly worker name from the config.

        Returns:
            Worker display name based on operator type:
            - LLM/VLM: "LLM Worker" (generic, model shown separately)
            - Human: "HUMAN"
            - RL: Specific worker name (e.g., "CleanRL", "Ray")
        """
        op_type = self._config.operator_type
        worker_id = self._config.worker_id

        # For Human operators, show "HUMAN"
        if op_type == "human" or worker_id == "human_worker":
            return "HUMAN"

        # For LLM/VLM operators, show generic "LLM Worker"
        # (the specific model is shown in the model label)
        if op_type in ("llm", "vlm"):
            return "LLM Worker"

        # For RL operators, show the specific worker name
        if worker_id:
            name = worker_id.replace("_worker", "").replace("_", " ")
            return name.upper() if len(name) <= 6 else name.title()
        return ""

    def _get_model_display_name(self) -> str:
        """Get a user-friendly model/policy name from the config.

        Returns:
            For LLM/VLM: Model name (e.g., "GPT-4o Mini", "Llama 3.3 70B")
            For RL: Policy filename or "No policy"
        """
        settings = self._config.settings
        op_type = self._config.operator_type

        if op_type in ("llm", "vlm"):
            # Get model_id from settings
            model_id = settings.get("model_id", "")
            if model_id:
                # Extract display name from model_id
                # e.g., "openai/gpt-4o-mini" -> "gpt-4o-mini"
                # e.g., "meta-llama/llama-3.3-70b-instruct:free" -> "llama-3.3-70b"
                if "/" in model_id:
                    model_id = model_id.split("/")[-1]
                # Remove common suffixes
                for suffix in (":free", "-instruct", "-chat"):
                    model_id = model_id.replace(suffix, "")
                return model_id
            return ""
        elif op_type == "rl":
            # Get policy path
            policy_path = settings.get("policy_path", "")
            if policy_path:
                # Extract filename from path
                import os
                return os.path.basename(policy_path)
            return ""
        return ""

    def set_config(self, config: OperatorConfig) -> None:
        """Update the operator configuration."""
        self._config = config
        # Update interactive flag when config changes (e.g., Human workers added)
        old_interactive = self._is_interactive
        self._is_interactive = self._has_human_worker(config)
        _LOGGER.debug(
            f"set_config: operator_id={config.operator_id}, "
            f"old_interactive={old_interactive}, new_interactive={self._is_interactive}, "
            f"operator_type={config.operator_type}, workers={list(config.workers.keys())}",
            extra=self._log_extra()
        )
        if self._is_interactive and not old_interactive:
            _LOGGER.debug(
                f"set_config: Enabling interactive mode for {config.operator_id}",
                extra=self._log_extra()
            )
            self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
            # Reconnect board game signals if renderer exists
            if self._board_game_renderer:
                self.connect_board_game_signals()
        elif not self._is_interactive and old_interactive:
            # Transitioning from interactive to non-interactive - clear "Your Turn" indicator
            _LOGGER.debug(
                f"set_config: Disabling interactive mode for {config.operator_id}, clearing orange border",
                extra=self._log_extra()
            )
            # Explicitly call set_your_turn to clear the orange border
            self.set_your_turn(False)
        self._update_header()

    def set_status(self, status: str) -> None:
        """Set the operator status.

        Args:
            status: One of "pending", "running", "stopped", "error".
        """
        if status not in self.STATUS_COLORS:
            _LOGGER.warning(f"Unknown status: {status}")
            return

        old_status = self._status
        self._status = status
        self._update_status_indicator()

        if old_status != status:
            self.status_changed.emit(self._config.operator_id, status)

    def set_display_size(self, width: int, height: int) -> None:
        """Set the display size of the render area.

        This controls the size of the render container widget,
        which affects how large the environment is displayed.

        Args:
            width: Width in pixels
            height: Height in pixels
        """
        # Update render container to fixed size
        self._render_container.setMinimumSize(width, height)
        self._render_container.setFixedSize(width, height)

        # Also update the overall widget size (add padding for header/stats)
        total_height = height + 80  # header + stats bar
        total_width = width + 20
        self.setMinimumSize(total_width, total_height)
        self.setFixedSize(total_width, total_height)

        # Force layout update
        self.updateGeometry()
        parent = self.parent()
        if parent is not None and hasattr(parent, 'updateGeometry'):
            parent.updateGeometry()  # type: ignore[union-attr]

        _LOGGER.debug(f"Set display size to {width}x{height} for {self._config.operator_id}")

    def _on_grip_resize_finished(self) -> None:
        """Called by ``_ResizeGrip`` when the user releases the mouse button.

        Emits ``container_resized`` so the config widget dropdown can be
        updated to reflect the new size.
        """
        w = self.width()
        h = self.height()
        self.container_resized.emit(self._config.operator_id, w, h)
        # Reposition the image and render-area grips after resize
        self._reposition_image_grip()
        self._reposition_render_area_grip()
        _LOGGER.debug(
            "Container resized to %dx%d for %s", w, h, self._config.operator_id,
        )

    def eventFilter(self, obj: Any, event: Any) -> bool:  # noqa: N802
        """Reposition grips when the render area resizes."""
        if obj is self._render_container and event.type() == QtCore.QEvent.Type.Resize:
            self._reposition_image_grip()
            self._reposition_render_area_grip()
        return super().eventFilter(obj, event)

    def _reposition_image_grip(self) -> None:
        """Move the image grip to the bottom-right corner of the actual image.

        Falls back to the bottom-right of the render container when the
        image rect is not yet known.
        """
        grip = self._image_grip
        # Try to get the actual image rect from the renderer widget
        img_rect = None
        if self._renderer_strategy is not None:
            widget = self._renderer_strategy.widget
            rect = getattr(widget, "_image_rect", None)
            if rect is not None and not rect.isNull():
                # _image_rect is in renderer-widget coords; map to _render_container
                br = widget.mapTo(self._render_container, rect.bottomRight())
                img_rect = br

        if img_rect is not None:
            grip.move(
                img_rect.x() - grip.width(),
                img_rect.y() - grip.height(),
            )
        else:
            # Fallback: bottom-right of render container
            rc = self._render_container
            grip.move(
                rc.width() - grip.width() - 2,
                rc.height() - grip.height() - 2,
            )
        grip.raise_()

    def _reposition_render_area_grip(self) -> None:
        """Move the render-area grip to the bottom-right of the renderer widget.

        The renderer widget is the ``_RgbView`` (or grid view) — the dark area
        that contains the image.  Falls back to the render container's
        bottom-right when no renderer is active yet.
        """
        grip = self._render_area_grip
        widget = None
        if self._renderer_strategy is not None:
            widget = self._renderer_strategy.widget

        if widget is not None and widget.isVisible():
            # Map renderer widget's bottom-right to _render_container coords
            br = widget.mapTo(self._render_container, QtCore.QPoint(widget.width(), widget.height()))
            grip.move(
                br.x() - grip.width(),
                br.y() - grip.height(),
            )
        else:
            # Fallback: bottom-right of render container
            rc = self._render_container
            grip.move(
                rc.width() - grip.width() - 2,
                rc.height() - grip.height() - 2,
            )
        grip.raise_()

    def display_payload(self, payload: Dict[str, Any]) -> None:
        """Display a render payload from telemetry.

        Args:
            payload: Telemetry payload containing render data.
        """
        _LOGGER.debug(
            "display_payload: received payload keys=%s",
            list(payload.keys()),
            extra=self._log_extra(),
        )
        try:
            # Update stats from payload
            self._update_stats_from_payload(payload)

            # Extract render data
            render_payload = self._extract_render_payload(payload)
            if render_payload is None:
                _LOGGER.debug(
                    "display_payload: render_payload is None, returning",
                    extra=self._log_extra(),
                )
                return

            # Debug: check render_payload structure
            rp_keys = list(render_payload.keys()) if isinstance(render_payload, dict) else "not_dict"
            rp_width = render_payload.get("width") if isinstance(render_payload, dict) else None
            rp_height = render_payload.get("height") if isinstance(render_payload, dict) else None
            rgb = render_payload.get("rgb") if isinstance(render_payload, dict) else None
            rgb_info = f"len={len(rgb)}" if rgb else "None"
            _LOGGER.debug(
                "render_payload: keys=%s, width=%s, height=%s, rgb=%s",
                rp_keys, rp_width, rp_height, rgb_info,
                extra=self._log_extra(),
            )

            # Check if this is a board game payload (chess, go, connect_four, etc.)
            is_board_game = self._is_board_game_payload(render_payload)
            _LOGGER.debug(
                "display_payload: is_board_game=%s",
                is_board_game,
                extra=self._log_extra(),
            )

            if is_board_game:
                # Use BoardGameRendererStrategy for board games
                if not self._board_game_renderer:
                    self._board_game_renderer = BoardGameRendererStrategy(self._render_container)
                    self._render_layout.addWidget(self._board_game_renderer.widget, 1)
                    # Connect board click signals for Human operators
                    _LOGGER.debug(
                        f"Created board game renderer, _is_interactive={self._is_interactive}",
                        extra=self._log_extra()
                    )
                    self.connect_board_game_signals()

                # Switch to board game renderer if needed
                if not self._is_board_game:
                    self._is_board_game = True
                    # Remove the "Waiting for data..." placeholder (mirrors _switch_to_renderer)
                    if self._placeholder and self._placeholder.parent():
                        self._render_layout.removeWidget(self._placeholder)
                        self._placeholder.deleteLater()
                        self._placeholder = None
                    # Hide other renderers
                    if self._grid_renderer:
                        self._grid_renderer.widget.hide()
                    if self._rgb_renderer:
                        self._rgb_renderer.widget.hide()
                    self._board_game_renderer.widget.show()
                    self._renderer_strategy = self._board_game_renderer

                # Render the board game
                if self._board_game_renderer.supports(render_payload):
                    context = RendererContext()
                    # Extract square_size from config settings (if configured via operator widget)
                    square_size = self._config.settings.get("square_size")
                    if square_size:
                        context.square_size = square_size
                    self._board_game_renderer.render(render_payload, context=context)

                    # For Human operators playing chess, extract and show legal moves
                    _LOGGER.debug(
                        f"Chess payload check: _is_interactive={self._is_interactive}, "
                        f"has_chess={'chess' in render_payload}, has_fen={'fen' in render_payload}",
                        extra=self._log_extra()
                    )
                    if self._is_interactive and ("chess" in render_payload or "fen" in render_payload):
                        _LOGGER.debug("Calling _update_chess_legal_moves", extra=self._log_extra())
                        self._update_chess_legal_moves(render_payload)
            else:
                # Use standard RGB or GRID renderer
                if self._is_board_game:
                    self._is_board_game = False
                    if self._board_game_renderer:
                        self._board_game_renderer.widget.hide()

                # Detect payload type and initialize appropriate renderer
                required_mode = self._detect_render_mode(render_payload)
                if required_mode != self._active_render_mode:
                    self._init_renderer(required_mode)

                # Render using the active strategy
                if self._renderer_strategy and self._renderer_strategy.supports(render_payload):
                    context = RendererContext()
                    # Pass square_size for grid environments (same as board game path)
                    square_size = self._config.settings.get("square_size")
                    if square_size:
                        context.square_size = square_size
                    _LOGGER.debug(
                        "RGB render: square_size=%s, settings_keys=%s, strategy=%s",
                        square_size,
                        list(self._config.settings.keys()),
                        type(self._renderer_strategy).__name__,
                        extra=self._log_extra(),
                    )
                    self._renderer_strategy.render(render_payload, context=context)

            # After rendering, reposition grips to track image/widget
            self._reposition_image_grip()
            self._reposition_render_area_grip()

        except Exception as e:
            _LOGGER.error(f"Error displaying payload: {e}")

    def _is_board_game_payload(self, payload: Dict[str, Any]) -> bool:
        """Check if the payload is for a board game (chess, go, connect_four, etc.).

        Args:
            payload: The render payload to analyze.

        Returns:
            True if this is a board game payload that should use BoardGameRendererStrategy.
        """
        # Check for board game specific keys
        if "chess" in payload or "fen" in payload:
            return True
        if "connect_four" in payload:
            return True
        if "go" in payload:
            return True
        if "sudoku" in payload:
            return True
        # Check game_id
        game_id = payload.get("game_id")
        if game_id in ("chess", "connect_four", "go", "tictactoe", "sudoku"):
            return True
        return False

    def _detect_render_mode(self, payload: Dict[str, Any]) -> RenderMode:
        """Detect which render mode is needed for this payload.

        Args:
            payload: The render payload to analyze.

        Returns:
            RenderMode.RGB_ARRAY for RGB frames, RenderMode.GRID otherwise.
        """
        # Check for RGB payload (used by BabyAI, MiniHack, Crafter, etc.)
        if "rgb" in payload or "frame" in payload:
            return RenderMode.RGB_ARRAY
        # Check for grid payload (used by FrozenLake, Taxi, etc.)
        if "grid" in payload:
            return RenderMode.GRID
        # Default to GRID (most basic)
        return RenderMode.GRID

    def _init_renderer(self, mode: RenderMode = RenderMode.GRID) -> None:
        """Initialize the renderer strategy.

        Args:
            mode: Which render mode to initialize (GRID or RGB_ARRAY).
        """
        try:
            # Check if already initialized with this mode
            if mode == RenderMode.GRID and self._grid_renderer:
                self._switch_to_renderer(self._grid_renderer, mode)
                return
            if mode == RenderMode.RGB_ARRAY and self._rgb_renderer:
                self._switch_to_renderer(self._rgb_renderer, mode)
                return

            # Create new renderer
            if not self._renderer_registry.is_registered(mode):
                _LOGGER.warning(f"{mode} renderer not registered")
                return

            new_renderer = self._renderer_registry.create(mode, self._render_container)

            # Store in appropriate slot
            if mode == RenderMode.GRID:
                self._grid_renderer = new_renderer
            elif mode == RenderMode.RGB_ARRAY:
                self._rgb_renderer = new_renderer

            self._switch_to_renderer(new_renderer, mode)
            _LOGGER.info(f"Renderer {mode} initialized for {self._config.operator_id}")

        except Exception as e:
            _LOGGER.error(f"Failed to initialize renderer {mode}: {e}")

    def _switch_to_renderer(self, renderer: Any, mode: RenderMode) -> None:
        """Switch the active renderer displayed in the container."""
        # Hide current renderer widget if any
        if self._renderer_strategy and hasattr(self._renderer_strategy, 'widget'):
            self._renderer_strategy.widget.hide()

        # Remove placeholder if present
        if self._placeholder and self._placeholder.parent():
            self._render_layout.removeWidget(self._placeholder)
            self._placeholder.deleteLater()
            self._placeholder = None

        # Set new active renderer
        self._renderer_strategy = renderer
        self._active_render_mode = mode

        if renderer and hasattr(renderer, 'widget'):
            widget = renderer.widget
            # Qt6 best practice: Use Expanding policy for widgets that should fill space
            widget.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Expanding
            )
            # Add widget with stretch factor=1 to ensure it expands (Qt6 layout docs)
            if widget.parent() != self._render_container:
                self._render_layout.addWidget(widget, 1)
            widget.show()

            # Qt6 best practice: Schedule deferred resize after layout settles
            # This ensures proper geometry before scaling pixmap
            QtCore.QTimer.singleShot(100, lambda: self._trigger_renderer_resize(widget))

    def _trigger_renderer_resize(self, widget: QtWidgets.QWidget) -> None:
        """Trigger a resize on the renderer widget after layout settles.

        This follows Qt6 documentation recommendation for handling initial geometry.
        """
        try:
            # Force the widget to update its geometry
            widget.updateGeometry()
            # If the renderer has a resize handler, trigger it
            if hasattr(widget, 'resizeEvent'):
                from qtpy.QtGui import QResizeEvent
                event = QResizeEvent(widget.size(), widget.size())
                widget.resizeEvent(event)
        except Exception as e:
            _LOGGER.debug(f"Renderer resize trigger: {e}")

    def _extract_render_payload(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract render payload from telemetry payload."""
        # Try render_payload first
        render_payload = payload.get("render_payload")
        if render_payload:
            return render_payload

        # Try render_payload_json
        import json
        render_payload_json = payload.get("render_payload_json")
        if render_payload_json:
            try:
                if isinstance(render_payload_json, str):
                    return json.loads(render_payload_json)
                return render_payload_json
            except (json.JSONDecodeError, TypeError):
                pass

        return None

    def _update_stats_from_payload(self, payload: Dict[str, Any]) -> None:
        """Update stats labels from payload data."""
        # Extract episode/step info
        episode_index = payload.get("episode_index", payload.get("episodeIndex", 0))
        step_index = payload.get("step_index", payload.get("stepIndex", 0))

        # Use episode_reward if available (cumulative from worker), else accumulate step reward
        episode_reward = payload.get("episode_reward")
        step_reward = payload.get("reward", 0.0)

        try:
            new_episode = int(episode_index) + 1  # episode_index is 0-based from worker
            new_step = int(step_index)  # step_index is already 1-based from worker

            # Reset episode reward and conversation when episode changes
            if new_episode != self._current_episode:
                self._episode_reward = 0.0
                self._conversation_history.clear()

            self._current_episode = new_episode
            self._current_step = new_step

            # Prefer episode_reward from worker (already cumulative)
            if episode_reward is not None:
                self._episode_reward = float(episode_reward)
            else:
                self._episode_reward += float(step_reward)
        except (TypeError, ValueError):
            pass

        # Update labels
        self._episode_label.setText(f"Episode: {self._current_episode}")
        self._step_label.setText(f"Step: {self._current_step}")
        self._reward_label.setText(f"Reward: {self._episode_reward:.2f}")

        # Extract LLM conversation data (for LLM/VLM operators)
        if self._config.operator_type in ("llm", "vlm"):
            self._update_conversation_from_payload(payload)

    def _update_conversation_from_payload(self, payload: Dict[str, Any]) -> None:
        """Extract and store LLM conversation data from telemetry payload.

        Args:
            payload: Telemetry payload that may contain observation, action, system_prompt.
        """
        # Extract system prompt (env-family specific instruction, sent once on ready)
        system_prompt = payload.get("system_prompt", "")
        if system_prompt and not self._system_prompt:
            self._system_prompt = system_prompt

        # Detect system prompt from observation at step 0 if not already set
        if not self._system_prompt and payload.get("step_index", -1) == 0:
            obs = payload.get("observation", "")
            if isinstance(obs, str) and obs.startswith("You are"):
                self._system_prompt = obs

        # Extract observation (what was sent to LLM at this step)
        observation = payload.get("observation", "")

        # Extract action/LLM response
        action = payload.get("action", payload.get("llm_response", ""))

        # Add to conversation history
        if observation:
            self._conversation_history.append({
                "role": "user",
                "content": observation
            })

        if action:
            self._conversation_history.append({
                "role": "assistant",
                "content": action
            })

    def reset_stats(self) -> None:
        """Reset stats for a new episode."""
        self._current_step = 0
        self._episode_reward = 0.0
        self._episode_label.setText(f"Episode: {self._current_episode}")
        self._step_label.setText("Step: 0")
        self._reward_label.setText("Reward: 0.00")
        # Clear conversation history for new episode
        self._conversation_history.clear()

    def _show_prompt_dialog(self) -> None:
        """Show dialog with the full prompt sent to LLM at current step."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"Step Prompt - {self._config.display_name}")
        dialog.setMinimumSize(700, 500)

        layout = QtWidgets.QVBoxLayout(dialog)

        # Header with step info
        header = QtWidgets.QLabel(
            f"Step Prompt - Episode {self._current_episode}, Step {self._current_step}",
            dialog
        )
        header.setStyleSheet("font-weight: bold; font-size: 12px; margin-bottom: 8px;")
        layout.addWidget(header)

        # Subheader
        subheader = QtWidgets.QLabel(
            "Full prompt sent to the LLM at current step:",
            dialog
        )
        subheader.setStyleSheet("font-size: 10px; color: #666; margin-bottom: 4px;")
        layout.addWidget(subheader)

        # Prompt text area
        text_edit = QtWidgets.QTextEdit(dialog)
        text_edit.setReadOnly(True)
        text_edit.setStyleSheet("font-family: monospace; font-size: 11px;")

        # Build the full step prompt
        prompt_lines = []
        sep = "=" * 70

        # System Instruction (env-family specific)
        env_family = getattr(self._config, "env_family", "unknown")
        prompt_lines.append(sep)
        prompt_lines.append(f"SYSTEM INSTRUCTION (Env: {env_family})")
        prompt_lines.append(sep)
        if self._system_prompt:
            prompt_lines.append(self._system_prompt)
        else:
            prompt_lines.append("(No system prompt captured yet. Run the operator to see it.)")
        prompt_lines.append("")

        # Current Step Observation (most recent)
        prompt_lines.append(sep)
        prompt_lines.append(f"CURRENT STEP OBSERVATION (Step {self._current_step})")
        prompt_lines.append(sep)
        if self._conversation_history:
            # Find the most recent USER message (observation)
            for msg in reversed(self._conversation_history):
                if msg.get("role") == "user":
                    prompt_lines.append(msg.get("content", "(empty)"))
                    break
            else:
                prompt_lines.append("(No observation yet)")
        else:
            prompt_lines.append("(No observation yet. Run the operator.)")
        prompt_lines.append("")

        # Full Conversation History
        if self._conversation_history:
            prompt_lines.append(sep)
            prompt_lines.append("FULL CONVERSATION HISTORY")
            prompt_lines.append(sep)
            for i, msg in enumerate(self._conversation_history):
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")
                if role == "USER":
                    prompt_lines.append(f"\n[{i+1}] USER (Observation):")
                    prompt_lines.append(content)
                elif role == "ASSISTANT":
                    prompt_lines.append(f"\n[{i+1}] ASSISTANT (Action):")
                    prompt_lines.append(content)
                else:
                    prompt_lines.append(f"\n[{i+1}] {role}:")
                    prompt_lines.append(content)
                prompt_lines.append("-" * 40)

        text_edit.setPlainText("\n".join(prompt_lines))
        layout.addWidget(text_edit)

        # Close button
        close_btn = QtWidgets.QPushButton("Close", dialog)
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.exec()

    def _show_chat_dialog(self) -> None:
        """Show dialog with the full LLM conversation history."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"LLM Conversation - {self._config.display_name}")
        dialog.setMinimumSize(700, 500)

        layout = QtWidgets.QVBoxLayout(dialog)

        # Header with stats
        header = QtWidgets.QLabel(
            f"Conversation History - Episode {self._current_episode}, Step {self._current_step}",
            dialog
        )
        header.setStyleSheet("font-weight: bold; font-size: 12px; margin-bottom: 8px;")
        layout.addWidget(header)

        # Conversation text area
        text_edit = QtWidgets.QTextEdit(dialog)
        text_edit.setReadOnly(True)
        text_edit.setStyleSheet("font-family: monospace; font-size: 11px;")

        if self._conversation_history:
            # Format conversation
            lines = []
            for i, msg in enumerate(self._conversation_history):
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")
                # Color-code by role
                if role == "USER":
                    lines.append(f"[{i+1}] USER (Observation):\n{content}\n")
                elif role == "ASSISTANT":
                    lines.append(f"[{i+1}] ASSISTANT (Action):\n{content}\n")
                else:
                    lines.append(f"[{i+1}] {role}:\n{content}\n")
                lines.append("-" * 60 + "\n")
            text_edit.setPlainText("".join(lines))
        else:
            text_edit.setPlainText("(No conversation captured yet. Run the operator to see the chat.)")

        layout.addWidget(text_edit)

        # Button row
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()

        # Copy button
        copy_btn = QtWidgets.QPushButton("Copy to Clipboard", dialog)
        copy_btn.clicked.connect(lambda: self._copy_conversation_to_clipboard(text_edit.toPlainText()))
        btn_layout.addWidget(copy_btn)

        # Close button
        close_btn = QtWidgets.QPushButton("Close", dialog)
        close_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(close_btn)

        layout.addLayout(btn_layout)

        dialog.exec()

    def _copy_conversation_to_clipboard(self, text: str) -> None:
        """Copy text to clipboard."""
        from qtpy.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        if clipboard:
            clipboard.setText(text)

    def update_llm_data(self, system_prompt: str = "", observation: str = "", action: str = "") -> None:
        """Update LLM conversation data from external source.

        This can be called when receiving telemetry with LLM data.

        Args:
            system_prompt: The system prompt / instruction (set once)
            observation: Current observation sent to LLM
            action: LLM's response / selected action
        """
        if system_prompt and not self._system_prompt:
            self._system_prompt = system_prompt

        if observation:
            self._conversation_history.append({
                "role": "user",
                "content": observation
            })

        if action:
            self._conversation_history.append({
                "role": "assistant",
                "content": action
            })

    # --- Human Operator Interaction Methods ---

    def set_interactive(self, enabled: bool) -> None:
        """Enable or disable interactive mode for human operators.

        Args:
            enabled: True to enable user interaction, False to disable.
        """
        self._is_interactive = enabled
        if enabled:
            self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
            _LOGGER.debug(f"Interactive mode enabled for {self._config.operator_id}")
            # Reconnect board game signals when enabling interactive mode
            if self._board_game_renderer:
                self.connect_board_game_signals()
        else:
            self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

    def set_game_id(self, game_id: GameId) -> None:
        """Set the current game and configure keyboard mappings.

        Args:
            game_id: The GameId for the current environment.
        """
        self._game_id = game_id
        self._key_mappings = {}

        # Look up mappings from human_input.py
        mappings = None
        for mapping_dict in [
            _TOY_TEXT_MAPPINGS,
            _MINIG_GRID_MAPPINGS,
            _BOX_2D_MAPPINGS,
            _VIZDOOM_MAPPINGS,
            _MINIHACK_MAPPINGS,
            _NETHACK_MAPPINGS,
            _CRAFTER_MAPPINGS,
            _BABAISAI_MAPPINGS,
            _PROCGEN_MAPPINGS,
            _JUMANJI_MAPPINGS,
            _ALE_MAPPINGS,
        ]:
            if game_id in mapping_dict:
                mappings = mapping_dict[game_id]
                break

        if mappings:
            # Convert ShortcutMapping tuples to a flat key -> action dict
            for mapping in mappings:
                action = mapping.action
                for sequence in mapping.key_sequences:
                    # QKeySequence stores the key code - need to handle Qt5/Qt6 differences
                    if sequence.count() > 0:
                        # In Qt6, sequence[0] returns QKeyCombination, need .toCombined()
                        # In Qt5, it returns int directly
                        qt_key_raw = sequence[0]
                        qt_key_int: int
                        if hasattr(qt_key_raw, 'toCombined'):
                            qt_key_int = qt_key_raw.toCombined()  # type: ignore[union-attr]
                        elif hasattr(qt_key_raw, 'key'):
                            qt_key_int = int(qt_key_raw.key())  # type: ignore[union-attr]
                        else:
                            qt_key_int = int(qt_key_raw)  # type: ignore[arg-type]
                        self._key_mappings[qt_key_int] = action

            _LOGGER.info(
                f"Configured {len(self._key_mappings)} key mappings for {game_id.value}",
                extra=self._log_extra()
            )
        else:
            _LOGGER.debug(
                f"No specific key mappings found for {game_id.value}, using fallback",
                extra=self._log_extra()
            )

    def set_available_actions(self, actions: list[int], labels: Optional[list[str]] = None) -> None:
        """Set the available actions for human selection.

        Args:
            actions: List of valid action indices.
            labels: Optional human-readable labels for each action.
        """
        self._available_actions = actions
        self._action_labels = labels or [str(a) for a in actions]
        _LOGGER.debug(
            f"Available actions for {self._config.operator_id}: {actions}",
            extra=self._log_extra()
        )

        # Update action buttons for human operators
        if self._is_interactive:
            self._populate_action_buttons(actions, self._action_labels)
            self._action_panel.setVisible(len(actions) > 0)

    def _populate_action_buttons(self, actions: list[int], labels: list[str]) -> None:
        """Populate action panel with clickable buttons.

        Uses FlowLayout so buttons automatically wrap to multiple lines
        when they exceed the container width.

        Args:
            actions: List of action indices.
            labels: Human-readable labels for each action.
        """
        _LOGGER.debug(
            f"_populate_action_buttons called: actions={actions}, labels={labels}",
            extra=self._log_extra()
        )

        # Clear existing buttons
        while self._action_buttons_layout.count():
            item = self._action_buttons_layout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

        # Create buttons for ALL actions (FlowLayout handles wrapping)
        for action_idx, label in zip(actions, labels):
            _LOGGER.debug(
                f"Creating button: action_idx={action_idx}, label='{label}'",
                extra=self._log_extra()
            )
            btn = QtWidgets.QPushButton(label, self._action_buttons_container)
            btn.setFixedHeight(22)
            btn.setMinimumWidth(40)
            btn.setToolTip(f"Action {action_idx}: {label}")
            btn.setStyleSheet(
                "QPushButton { font-size: 9px; padding: 2px 6px; "
                "background-color: #e3f2fd; border: 1px solid #90caf9; border-radius: 3px; }"
                "QPushButton:hover { background-color: #bbdefb; }"
                "QPushButton:pressed { background-color: #90caf9; }"
            )
            # Connect click to emit action
            btn.clicked.connect(lambda checked=False, a=action_idx, lbl=label: self._on_action_button_clicked(a, lbl))
            self._action_buttons_layout.addWidget(btn)

        # Update container geometry to trigger flow layout recalculation
        self._action_buttons_container.updateGeometry()

    def _on_action_button_clicked(self, action: int, button_label: str = "") -> None:
        """Handle action button click.

        Args:
            action: The action index that was clicked.
            button_label: The label on the button that was clicked (for debugging).
        """
        # Update selected action display
        self._selected_action = action
        if self._action_labels and action < len(self._action_labels):
            action_label = self._action_labels[action]
            self._selected_action_label.setText(f"Selected Action: {action_label}")
            self._selected_action_label.setVisible(True)

        log_constant(
            _LOGGER,
            LOG_HUMAN_ACTION_BUTTON_CLICKED,
            message=f"Button clicked: label='{button_label}', action_idx={action}",
            extra=self._log_extra()
        )
        self.human_action_submitted.emit(self._config.operator_id, action)
        log_constant(
            _LOGGER,
            LOG_HUMAN_ACTION_SIGNAL_EMITTED,
            message=f"Signal emitted: operator_id={self._config.operator_id}, action={action}",
            extra=self._log_extra()
        )

    def _resolve_player_color(self, player_id: str) -> tuple[str, str]:
        """Resolve (primary_hex, bg_hex) for a player from config or defaults."""
        worker = self._config.get_worker_for_player(player_id)
        if worker is not None:
            color_name = worker.settings.get("agent_color")
            if color_name and color_name != "auto" and color_name in COLOR_PALETTE:
                return COLOR_PALETTE[color_name]
        default_name = DEFAULT_AGENT_COLOR_NAMES.get(player_id)
        if default_name and default_name in COLOR_PALETTE:
            return COLOR_PALETTE[default_name]
        return ("#666666", "#e0e0e0")

    def set_chess_legal_moves(
        self,
        moves: list[str],
        current_player: str = "",
        fen: str = ""
    ) -> None:
        """Set legal chess moves for human player selection.

        Args:
            moves: List of legal moves in UCI notation (e.g., ["e2e4", "d2d4", ...])
            current_player: Current player ("white" or "black")
            fen: Current FEN position (for SAN conversion)
        """
        self._chess_legal_moves = moves
        self._chess_current_player = current_player

        # Update player label
        if current_player:
            player_display = current_player.title()
            color = "#333" if current_player == "white" else "#666"
            self._chess_player_label.setText(f"({player_display} to move)")
            self._chess_player_label.setStyleSheet(f"font-size: 10px; color: {color};")

        # Populate chess move buttons
        self._populate_chess_buttons(moves, fen)

        # Show panel only for Human operators with legal moves
        show_panel = self._is_interactive and len(moves) > 0
        self._chess_moves_panel.setVisible(show_panel)

        _LOGGER.debug(
            f"Chess legal moves set: {len(moves)} moves for {current_player}",
            extra=self._log_extra()
        )

    def _update_chess_legal_moves(self, render_payload: Dict[str, Any]) -> None:
        """Extract FEN from payload and update legal chess moves for Human player.

        Uses python-chess to compute legal moves from the FEN position.

        Args:
            render_payload: The render payload containing chess data.
        """
        _LOGGER.debug(
            f"_update_chess_legal_moves called, payload keys: {list(render_payload.keys())}",
            extra=self._log_extra()
        )
        try:
            import chess
        except ImportError:
            _LOGGER.debug("python-chess not installed, skipping legal moves display")
            return

        # Extract FEN from payload
        fen = None
        if "fen" in render_payload:
            fen = render_payload["fen"]
        elif "chess" in render_payload:
            chess_data = render_payload["chess"]
            _LOGGER.debug(f"chess_data type: {type(chess_data)}", extra=self._log_extra())
            if isinstance(chess_data, dict):
                fen = chess_data.get("fen")
                _LOGGER.debug(f"Extracted FEN from dict: {fen}", extra=self._log_extra())
            elif isinstance(chess_data, str):
                fen = chess_data

        if not fen:
            _LOGGER.debug("No FEN found in chess payload", extra=self._log_extra())
            self._chess_moves_panel.setVisible(False)
            return

        try:
            # Create board and get legal moves
            board = chess.Board(fen)

            # Get current player
            current_player = "white" if board.turn == chess.WHITE else "black"

            # Get all legal moves in UCI notation
            legal_moves = [move.uci() for move in board.legal_moves]

            _LOGGER.debug(
                f"Chess position: {len(legal_moves)} legal moves for {current_player}",
                extra=self._log_extra()
            )

            # Update the chess moves panel
            self.set_chess_legal_moves(legal_moves, current_player, fen)

        except Exception as e:
            _LOGGER.warning(f"Error parsing chess FEN: {e}")
            self._chess_moves_panel.setVisible(False)

    def _populate_chess_buttons(self, moves: list[str], fen: str = "") -> None:
        """Populate chess move buttons with legal moves.

        Groups moves by source piece for better organization.

        Args:
            moves: List of UCI moves (e.g., ["e2e4", "g1f3"])
            fen: Current FEN position (for piece identification)
        """
        # Clear existing buttons
        while self._chess_buttons_layout.count():
            item = self._chess_buttons_layout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

        if not moves:
            return

        # Group moves by source square for organization
        # e.g., {"e2": ["e2e3", "e2e4"], "g1": ["g1f3", "g1h3"]}
        moves_by_source: dict[str, list[str]] = {}
        for move in sorted(moves):
            source = move[:2]
            if source not in moves_by_source:
                moves_by_source[source] = []
            moves_by_source[source].append(move)

        # Create buttons - group by source square
        for source in sorted(moves_by_source.keys()):
            source_moves = moves_by_source[source]

            for uci_move in source_moves:
                # Format: "e2→e4" or "e2e4" for compact display
                from_sq = uci_move[:2]
                to_sq = uci_move[2:4]
                promotion = uci_move[4:] if len(uci_move) > 4 else ""

                # Display format: compact UCI
                if promotion:
                    label = f"{from_sq}{to_sq}={promotion.upper()}"
                else:
                    label = f"{from_sq}{to_sq}"

                btn = QtWidgets.QPushButton(label, self._chess_buttons_container)
                btn.setFixedHeight(22)
                btn.setMinimumWidth(45)
                btn.setToolTip(f"Move: {from_sq} → {to_sq}" + (f" (promote to {promotion.upper()})" if promotion else ""))
                btn.setStyleSheet(
                    "QPushButton { font-size: 9px; padding: 2px 4px; "
                    "background-color: #fff3e0; border: 1px solid #ffb74d; border-radius: 3px; "
                    "font-family: monospace; }"
                    "QPushButton:hover { background-color: #ffe0b2; }"
                    "QPushButton:pressed { background-color: #ffb74d; }"
                )
                # Connect click to emit the UCI move
                btn.clicked.connect(lambda checked, m=uci_move: self._on_chess_move_clicked(m))
                self._chess_buttons_layout.addWidget(btn)

        # Update geometry
        self._chess_buttons_container.updateGeometry()

    def _on_chess_move_clicked(self, uci_move: str) -> None:
        """Handle chess move button click.

        Args:
            uci_move: The UCI move string (e.g., "e2e4")
        """
        _LOGGER.info(
            f"Chess move via button click: {uci_move}",
            extra=self._log_extra()
        )
        # Emit the signal with operator_id and the UCI move
        self.chess_move_button_clicked.emit(self._config.operator_id, uci_move)

        # Also emit as board_game_move for compatibility with existing handlers
        # Note: UCI moves can be 4 chars (e.g., "e2e4") or 5 chars for promotion (e.g., "c2b1q")
        from_sq = uci_move[:2]
        to_sq = uci_move[2:]  # Keep promotion piece if present (e.g., "b1q" from "c2b1q")
        self.board_game_move_made.emit(self._config.operator_id, from_sq, to_sq)

    def keyPressEvent(self, event) -> None:
        """Handle keyboard input for human operators.

        Uses game-specific key mappings from human_input.py when available,
        falling back to number keys (0-9) and arrow keys.
        """
        if not self._is_interactive:
            super().keyPressEvent(event)
            return

        key = event.key()

        # If mouse is captured, Escape releases it
        if self._mouse_captured and key == QtCore.Qt.Key.Key_Escape:
            self._release_mouse()
            return

        # First, try game-specific mappings from human_input.py (supports WASD, arrows, etc.)
        if key in self._key_mappings:
            action_idx = self._key_mappings[key]
            if action_idx in self._available_actions or not self._available_actions:
                _LOGGER.info(
                    f"Human action via game-specific key: {action_idx} (key={key})",
                    extra=self._log_extra()
                )
                self.human_action_submitted.emit(self._config.operator_id, action_idx)
                return

        # Number keys 0-9 map to actions 0-9 (fallback for any environment)
        if QtCore.Qt.Key.Key_0 <= key <= QtCore.Qt.Key.Key_9:
            action_idx = key - QtCore.Qt.Key.Key_0
            if action_idx in self._available_actions or not self._available_actions:
                _LOGGER.info(
                    f"Human action via number key: {action_idx}",
                    extra=self._log_extra()
                )
                self.human_action_submitted.emit(self._config.operator_id, action_idx)
                return

        # Arrow keys as fallback (FrozenLake convention: Left=0, Down=1, Right=2, Up=3)
        # Only used if no game-specific mappings found
        if not self._key_mappings:
            arrow_mapping = {
                QtCore.Qt.Key.Key_Left: 0,
                QtCore.Qt.Key.Key_Down: 1,
                QtCore.Qt.Key.Key_Right: 2,
                QtCore.Qt.Key.Key_Up: 3,
            }
            if key in arrow_mapping:
                action_idx = arrow_mapping[key]
                _LOGGER.info(
                    f"Human action via arrow key: {action_idx}",
                    extra=self._log_extra()
                )
                self.human_action_submitted.emit(self._config.operator_id, action_idx)
                return

        super().keyPressEvent(event)

    def mousePressEvent(self, event) -> None:
        """Handle mouse clicks - set focus for keyboard input."""
        if self._is_interactive:
            self.setFocus()
            # Capture mouse if enabled and not already captured
            if self._mouse_capture_enabled and not self._mouse_captured:
                if event.button() == QtCore.Qt.MouseButton.LeftButton:
                    self._grab_mouse()
                    event.accept()
                    return

        super().mousePressEvent(event)

    def _grab_mouse(self):
        self._mouse_captured = True
        self.setCursor(QtCore.Qt.CursorShape.BlankCursor)
        self.setMouseTracking(True)
        # Center mouse in widget initially
        self._center_mouse()
        _LOGGER.info(f"Mouse captured for {self._config.operator_id}")

    def _release_mouse(self):
        if self._mouse_captured:
            self._mouse_captured = False
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            self.setMouseTracking(False)
            _LOGGER.info(f"Mouse released for {self._config.operator_id}")

    def _center_mouse(self):
        # Center mouse in widget coordinates
        center = self.rect().center()
        # Map to global coordinates for QCursor.setPos
        global_center = self.mapToGlobal(center)
        self._mouse_center = global_center
        QtGui.QCursor.setPos(global_center)

    def mouseMoveEvent(self, event) -> None:
        if self._mouse_captured:
            global_pos = event.globalPosition().toPoint()
            # Check if this is the synthetic move caused by setPos
            if global_pos == self._mouse_center:
                return

            dx = global_pos.x() - self._mouse_center.x()
            dy = global_pos.y() - self._mouse_center.y()

            if dx != 0 or dy != 0:
                self.mouse_delta_input.emit(self._config.operator_id, dx, dy)
                # Re-center mouse to keep it captured
                self._center_mouse()

            event.accept()
            return

        super().mouseMoveEvent(event)

    def focusOutEvent(self, event) -> None:
        if self._mouse_captured:
            self._release_mouse()
        super().focusOutEvent(event)


    def connect_board_game_signals(self) -> None:
        """Connect board game renderer signals to container signals.

        Call this after the board game renderer is initialized.
        """
        if self._board_game_renderer and self._is_interactive:
            # BoardGameRendererStrategy emits chess_move_made(from_sq, to_sq) signal
            if hasattr(self._board_game_renderer, 'chess_move_made'):
                self._board_game_renderer.chess_move_made.connect(self._on_board_game_move)
                _LOGGER.debug(
                    f"Connected board game signals for {self._config.operator_id}",
                    extra=self._log_extra()
                )

    def _on_board_game_move(self, from_square: str, to_square: str) -> None:
        """Handle board game move from renderer.

        Args:
            from_square: Source square (e.g., "e2")
            to_square: Target square (e.g., "e4")
        """
        _LOGGER.info(
            f"Human board game move: {from_square} -> {to_square}",
            extra=self._log_extra()
        )
        self.board_game_move_made.emit(self._config.operator_id, from_square, to_square)

    def set_your_turn(self, is_your_turn: bool) -> None:
        """Show or hide the 'Your Turn' indicator for human operators.

        Args:
            is_your_turn: True to show indicator, False to hide.
        """
        if self._is_interactive:
            self._your_turn_label.setVisible(is_your_turn)
            if is_your_turn:
                # Visual feedback - highlight the container border
                self.setStyleSheet(
                    "QFrame { border: 2px solid #FF5722; border-radius: 4px; }"
                )
            else:
                self.setStyleSheet(
                    "QFrame { border: 1px solid #ccc; border-radius: 4px; }"
                )

    @property
    def is_interactive(self) -> bool:
        """Check if this container is interactive (human operator)."""
        return self._is_interactive

    # --- Parallel Multi-Agent Action Panel ---

    def set_parallel_action_panel(self, panel: QtWidgets.QWidget) -> None:
        """Embed a MultiAgentActionPanel right below the environment render.

        Hides the single-agent ``_action_panel`` and shows the multi-agent
        panel instead.

        Args:
            panel: The MultiAgentActionPanel widget to embed.
        """
        self.clear_parallel_action_panel()
        self._action_panel.setVisible(False)
        self._parallel_action_container_layout.addWidget(panel)
        self._parallel_action_container.setVisible(True)
        _LOGGER.debug(
            "Parallel action panel embedded in render container for %s",
            self._config.operator_id,
        )

    def clear_parallel_action_panel(self) -> None:
        """Remove the embedded parallel action panel and hide the container."""
        while self._parallel_action_container_layout.count():
            item = self._parallel_action_container_layout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)  # type: ignore[call-overload]
        self._parallel_action_container.setVisible(False)

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._renderer_strategy:
            if hasattr(self._renderer_strategy, 'cleanup'):
                self._renderer_strategy.cleanup()
            self._renderer_strategy = None

    @property
    def config(self) -> OperatorConfig:
        return self._config

    @property
    def status(self) -> str:
        return self._status

    @property
    def operator_id(self) -> str:
        return self._config.operator_id


__all__ = ["OperatorRenderContainer"]
