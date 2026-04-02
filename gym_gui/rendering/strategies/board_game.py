"""Board game renderer strategy for PettingZoo Classic games.

This module provides interactive board rendering for Chess, Go, and Connect Four.
All board games are rendered through the Grid tab using a unified strategy pattern.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Set

from qtpy import QtCore, QtGui, QtWidgets

from gym_gui.core.enums import GameId, RenderMode
from gym_gui.logging_config.helpers import log_constant
from gym_gui.logging_config.log_constants import (
    LOG_CHECKERS_BOARD_CLICK,
    LOG_CHECKERS_BOARD_CLICK_IGNORED,
    LOG_CHECKERS_CELL_SIGNAL_EMITTED,
)
from gym_gui.rendering.interfaces import RendererContext, RendererStrategy

_LOG = logging.getLogger(__name__)


# =============================================================================
# Board Game Renderer Strategy (Main Entry Point)
# =============================================================================


class BoardGameRendererStrategy(RendererStrategy):
    """Renderer strategy for interactive board games (Chess, Go, Connect Four).

    This strategy detects the game type from the payload and delegates rendering
    to the appropriate game-specific renderer. All board games are displayed in
    the Grid tab using this unified strategy.

    Signals are forwarded from the internal widget to allow MainWindow to handle
    game-specific actions (moves, clicks, etc.).
    """

    mode = RenderMode.GRID  # Rendered in Grid tab

    # Supported game IDs
    SUPPORTED_GAMES = frozenset({
        GameId.CHESS, GameId.CONNECT_FOUR, GameId.GO, GameId.TIC_TAC_TOE,
        GameId.JUMANJI_SUDOKU, GameId.OPEN_SPIEL_CHECKERS,
        GameId.AMERICAN_CHECKERS, GameId.RUSSIAN_CHECKERS, GameId.INTERNATIONAL_DRAUGHTS,
    })

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        self._widget = _BoardGameWidget(parent)
        self._current_game: GameId | None = None

    @property
    def widget(self) -> QtWidgets.QWidget:
        return self._widget

    # Signal accessors for MainWindow connection
    @property
    def chess_move_made(self) -> QtCore.SignalInstance:
        """Signal: chess move made (from_sq: str, to_sq: str)."""
        return self._widget.chess_move_made

    @property
    def connect_four_column_clicked(self) -> QtCore.SignalInstance:
        """Signal: Connect Four column clicked (column: int)."""
        return self._widget.connect_four_column_clicked

    @property
    def go_intersection_clicked(self) -> QtCore.SignalInstance:
        """Signal: Go intersection clicked (row: int, col: int)."""
        return self._widget.go_intersection_clicked

    @property
    def go_pass_requested(self) -> QtCore.SignalInstance:
        """Signal: Go pass requested."""
        return self._widget.go_pass_requested

    @property
    def tictactoe_cell_clicked(self) -> QtCore.SignalInstance:
        """Signal: Tic-Tac-Toe cell clicked (row: int, col: int)."""
        return self._widget.tictactoe_cell_clicked

    @property
    def sudoku_cell_selected(self) -> QtCore.SignalInstance:
        """Signal: Sudoku cell selected (row: int, col: int)."""
        return self._widget.sudoku_cell_selected

    @property
    def sudoku_digit_entered(self) -> QtCore.SignalInstance:
        """Signal: Sudoku digit entered (row: int, col: int, digit: int)."""
        return self._widget.sudoku_digit_entered

    @property
    def sudoku_cell_cleared(self) -> QtCore.SignalInstance:
        """Signal: Sudoku cell cleared (row: int, col: int)."""
        return self._widget.sudoku_cell_cleared

    @property
    def checkers_cell_clicked(self) -> QtCore.SignalInstance:
        """Signal: Checkers cell clicked (row: int, col: int)."""
        return self._widget.checkers_cell_clicked

    def set_checkers_selection(
        self,
        selected_cell: Optional[tuple[int, int]],
        legal_destinations: Optional[List[tuple[int, int]]] = None,
    ) -> None:
        """Set the checkers selection highlight.

        Args:
            selected_cell: The (row, col) of selected piece, or None to clear.
            legal_destinations: List of (row, col) tuples for valid destinations.
        """
        self._widget.set_checkers_selection(selected_cell, legal_destinations)

    def set_checkers_moveable_cells(
        self,
        cells: Optional[List[tuple[int, int]]] = None,
    ) -> None:
        """Highlight pieces that can move (for mandatory jump hints).

        Args:
            cells: List of (row, col) tuples for pieces that can move, or None to clear.
        """
        self._widget.set_checkers_moveable_cells(cells)

    def render(
        self, payload: Mapping[str, object], *, context: RendererContext | None = None
    ) -> None:
        """Render the board game payload."""
        game_id = self._detect_game(payload, context)
        if game_id is None:
            self.reset()
            return

        self._widget.render_game(game_id, dict(payload), context=context)
        self._current_game = game_id

    def supports(self, payload: Mapping[str, object]) -> bool:
        """Check if this strategy supports the payload."""
        # Check for board game specific keys
        if "chess" in payload or "fen" in payload:
            return True
        if "connect_four" in payload:
            return True
        if "go" in payload:
            return True
        if "sudoku" in payload:
            return True
        if "checkers" in payload:
            return True
        if "draughts" in payload:
            return True
        # Check game_type field (from adapter payloads)
        game_type = payload.get("game_type")
        if game_type in ("chess", "connect_four", "go", "sudoku", "checkers", "draughts"):
            return True
        # Check game_id
        game_id = payload.get("game_id")
        if isinstance(game_id, GameId) and game_id in self.SUPPORTED_GAMES:
            return True
        if isinstance(game_id, str):
            try:
                return GameId(game_id) in self.SUPPORTED_GAMES
            except ValueError:
                pass
        return False

    def reset(self) -> None:
        """Clear the board."""
        self._widget.reset()
        self._current_game = None

    def cleanup(self) -> None:
        """Clean up resources."""
        self.reset()

    def _detect_game(
        self, payload: Mapping[str, object], context: RendererContext | None
    ) -> GameId | None:
        """Detect which board game this payload represents."""
        # Try context first
        if context and context.game_id in self.SUPPORTED_GAMES:
            return context.game_id

        # Try explicit game_id in payload
        game_id = payload.get("game_id")
        if isinstance(game_id, GameId) and game_id in self.SUPPORTED_GAMES:
            return game_id
        if isinstance(game_id, str):
            try:
                gid = GameId(game_id)
                if gid in self.SUPPORTED_GAMES:
                    return gid
            except ValueError:
                pass

        # Detect from payload keys
        if "chess" in payload or "fen" in payload:
            return GameId.CHESS
        if "connect_four" in payload:
            return GameId.CONNECT_FOUR
        if "go" in payload:
            return GameId.GO
        if "sudoku" in payload:
            return GameId.JUMANJI_SUDOKU
        if "checkers" in payload:
            return GameId.OPEN_SPIEL_CHECKERS
        if "draughts" in payload:
            return _resolve_draughts_variant(payload)

        # Detect from game_type value (adapter payloads use this)
        game_type = payload.get("game_type")
        if game_type == "chess":
            return GameId.CHESS
        if game_type == "connect_four":
            return GameId.CONNECT_FOUR
        if game_type == "go":
            return GameId.GO
        if game_type == "sudoku":
            return GameId.JUMANJI_SUDOKU
        if game_type == "checkers":
            return GameId.OPEN_SPIEL_CHECKERS
        if game_type == "draughts":
            return _resolve_draughts_variant(payload)

        return None

    @staticmethod
    def get_game_from_payload(payload: Mapping[str, Any]) -> GameId | None:
        """Static method to detect game type from payload (for external use)."""
        if "chess" in payload or "fen" in payload:
            return GameId.CHESS
        if "connect_four" in payload:
            return GameId.CONNECT_FOUR
        if "go" in payload:
            return GameId.GO
        if "sudoku" in payload:
            return GameId.JUMANJI_SUDOKU
        if "checkers" in payload:
            return GameId.OPEN_SPIEL_CHECKERS
        if "draughts" in payload:
            return _resolve_draughts_variant(payload)

        # Detect from game_type value (adapter payloads use this)
        game_type = payload.get("game_type")
        if game_type == "chess":
            return GameId.CHESS
        if game_type == "connect_four":
            return GameId.CONNECT_FOUR
        if game_type == "go":
            return GameId.GO
        if game_type == "sudoku":
            return GameId.JUMANJI_SUDOKU
        if game_type == "checkers":
            return GameId.OPEN_SPIEL_CHECKERS
        if game_type == "draughts":
            return _resolve_draughts_variant(payload)

        return None


def _resolve_draughts_variant(payload: Mapping[str, Any]) -> GameId | None:
    """Resolve which draughts variant based on payload."""
    variant = payload.get("variant", "").lower()
    if "american" in variant:
        return GameId.AMERICAN_CHECKERS
    elif "russian" in variant:
        return GameId.RUSSIAN_CHECKERS
    elif "international" in variant:
        return GameId.INTERNATIONAL_DRAUGHTS
    # Default to American if variant not specified
    return GameId.AMERICAN_CHECKERS


# =============================================================================
# Board Game Widget (Internal)
# =============================================================================


class _BoardGameWidget(QtWidgets.QStackedWidget):
    """Widget that manages different board game renderers.

    Uses a stacked widget to switch between Chess, Go, and Connect Four
    renderers on demand.
    """

    # Signals forwarded from child renderers
    chess_move_made = QtCore.Signal(str, str)  # from_sq, to_sq
    connect_four_column_clicked = QtCore.Signal(int)  # column
    go_intersection_clicked = QtCore.Signal(int, int)  # row, col
    go_pass_requested = QtCore.Signal()
    tictactoe_cell_clicked = QtCore.Signal(int, int)  # row, col
    sudoku_cell_selected = QtCore.Signal(int, int)  # row, col
    sudoku_digit_entered = QtCore.Signal(int, int, int)  # row, col, digit
    sudoku_cell_cleared = QtCore.Signal(int, int)  # row, col
    checkers_cell_clicked = QtCore.Signal(int, int)  # row, col

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        # Lazy-loaded renderers
        self._chess_renderer: _ChessBoardRenderer | None = None
        self._connect_four_renderer: _ConnectFourBoardRenderer | None = None
        self._go_renderer: _GoBoardRenderer | None = None
        self._tictactoe_renderer: _TicTacToeBoardRenderer | None = None
        self._sudoku_renderer: _SudokuBoardRenderer | None = None
        self._checkers_renderer: _CheckersBoardRenderer | None = None

        # Placeholder for empty state
        self._placeholder = QtWidgets.QLabel("No board game loaded", self)
        self._placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: #888; font-size: 14px;")
        self.addWidget(self._placeholder)

        self._current_game: GameId | None = None

    def render_game(self, game_id: GameId, payload: Dict[str, Any], context: RendererContext | None = None) -> None:
        """Render the specified board game."""
        # Extract square_size from context if provided
        square_size = context.square_size if context else None

        if game_id == GameId.CHESS:
            renderer = self._get_chess_renderer()
            if square_size:
                renderer.set_square_size(square_size)
            renderer.update_from_payload(payload)
            self.setCurrentWidget(renderer)
        elif game_id == GameId.CONNECT_FOUR:
            renderer = self._get_connect_four_renderer()
            renderer.update_from_payload(payload)
            self.setCurrentWidget(renderer)
        elif game_id == GameId.GO:
            renderer = self._get_go_renderer()
            renderer.update_from_payload(payload)
            self.setCurrentWidget(renderer)
        elif game_id == GameId.TIC_TAC_TOE:
            renderer = self._get_tictactoe_renderer()
            renderer.update_from_payload(payload)
            self.setCurrentWidget(renderer)
        elif game_id == GameId.JUMANJI_SUDOKU:
            renderer = self._get_sudoku_renderer()
            # Extract sudoku data from wrapped or flat payload
            sudoku_data = payload.get("sudoku", payload)
            renderer.update_from_payload(sudoku_data)
            self.setCurrentWidget(renderer)
        elif game_id in (GameId.OPEN_SPIEL_CHECKERS, GameId.AMERICAN_CHECKERS,
                          GameId.RUSSIAN_CHECKERS, GameId.INTERNATIONAL_DRAUGHTS):
            renderer = self._get_checkers_renderer()
            renderer.update_from_payload(payload)
            self.setCurrentWidget(renderer)
        else:
            self.setCurrentWidget(self._placeholder)

        self._current_game = game_id

    def reset(self) -> None:
        """Reset to placeholder state."""
        self.setCurrentWidget(self._placeholder)
        self._current_game = None

    def _get_chess_renderer(self) -> "_ChessBoardRenderer":
        """Lazy-load Chess renderer."""
        if self._chess_renderer is None:
            self._chess_renderer = _ChessBoardRenderer(self)
            self._chess_renderer.move_made.connect(self.chess_move_made)
            self.addWidget(self._chess_renderer)
        return self._chess_renderer

    def _get_connect_four_renderer(self) -> "_ConnectFourBoardRenderer":
        """Lazy-load Connect Four renderer."""
        if self._connect_four_renderer is None:
            self._connect_four_renderer = _ConnectFourBoardRenderer(self)
            self._connect_four_renderer.column_clicked.connect(
                self.connect_four_column_clicked
            )
            self.addWidget(self._connect_four_renderer)
        return self._connect_four_renderer

    def _get_go_renderer(self) -> "_GoBoardRenderer":
        """Lazy-load Go renderer."""
        if self._go_renderer is None:
            self._go_renderer = _GoBoardRenderer(self)
            self._go_renderer.intersection_clicked.connect(self.go_intersection_clicked)
            self._go_renderer.pass_requested.connect(self.go_pass_requested)
            self.addWidget(self._go_renderer)
        return self._go_renderer

    def _get_tictactoe_renderer(self) -> "_TicTacToeBoardRenderer":
        """Lazy-load Tic-Tac-Toe renderer."""
        if self._tictactoe_renderer is None:
            self._tictactoe_renderer = _TicTacToeBoardRenderer(self)
            self._tictactoe_renderer.cell_clicked.connect(self.tictactoe_cell_clicked)
            self.addWidget(self._tictactoe_renderer)
        return self._tictactoe_renderer

    def _get_sudoku_renderer(self) -> "_SudokuBoardRenderer":
        """Lazy-load Sudoku renderer."""
        if self._sudoku_renderer is None:
            self._sudoku_renderer = _SudokuBoardRenderer(self)
            self._sudoku_renderer.cell_selected.connect(self.sudoku_cell_selected)
            self._sudoku_renderer.digit_entered.connect(self.sudoku_digit_entered)
            self._sudoku_renderer.cell_cleared.connect(self.sudoku_cell_cleared)
            self.addWidget(self._sudoku_renderer)
        return self._sudoku_renderer

    def _get_checkers_renderer(self) -> "_CheckersBoardRenderer":
        """Lazy-load Checkers renderer."""
        if self._checkers_renderer is None:
            self._checkers_renderer = _CheckersBoardRenderer(self)
            self._checkers_renderer.cell_clicked.connect(self.checkers_cell_clicked)
            self.addWidget(self._checkers_renderer)
        return self._checkers_renderer

    def set_checkers_selection(
        self,
        selected_cell: Optional[tuple[int, int]],
        legal_destinations: Optional[List[tuple[int, int]]] = None,
    ) -> None:
        """Set the checkers selection highlight.

        Args:
            selected_cell: The (row, col) of selected piece, or None to clear.
            legal_destinations: List of (row, col) tuples for valid destinations.
        """
        renderer = self._get_checkers_renderer()
        renderer.set_selection(selected_cell, legal_destinations)

    def set_checkers_moveable_cells(
        self,
        cells: Optional[List[tuple[int, int]]] = None,
    ) -> None:
        """Highlight pieces that can move (for mandatory jump hints).

        Args:
            cells: List of (row, col) tuples for pieces that can move, or None to clear.
        """
        renderer = self._get_checkers_renderer()
        renderer.set_moveable_cells(cells)


# =============================================================================
# Chess Board Renderer
# =============================================================================

# Unicode chess piece symbols
_CHESS_PIECES: Dict[str, str] = {
    "K": "\u2654",  # White King
    "Q": "\u2655",  # White Queen
    "R": "\u2656",  # White Rook
    "B": "\u2657",  # White Bishop
    "N": "\u2658",  # White Knight
    "P": "\u2659",  # White Pawn
    "k": "\u265A",  # Black King
    "q": "\u265B",  # Black Queen
    "r": "\u265C",  # Black Rook
    "b": "\u265D",  # Black Bishop
    "n": "\u265E",  # Black Knight
    "p": "\u265F",  # Black Pawn
}

# Chess colors
_CHESS_LIGHT_SQUARE = QtGui.QColor(240, 217, 181)
_CHESS_DARK_SQUARE = QtGui.QColor(181, 136, 99)
_CHESS_SELECTED = QtGui.QColor(255, 255, 0, 150)
_CHESS_LEGAL_MOVE = QtGui.QColor(0, 255, 0, 100)
_CHESS_LAST_MOVE = QtGui.QColor(255, 255, 0, 80)
_CHESS_CHECK = QtGui.QColor(255, 0, 0, 100)
_CHESS_HOVER = QtGui.QColor(100, 100, 255, 50)


class _ChessBoardRenderer(QtWidgets.QWidget):
    """Chess board renderer with interactive move input."""

    move_made = QtCore.Signal(str, str)  # from_square, to_square

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        # Board state
        self._fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        self._board: List[List[Optional[str]]] = [[None] * 8 for _ in range(8)]
        self._current_player: str = "white"

        # Interaction state
        self._selected_square: Optional[str] = None
        self._legal_moves: Set[str] = set()
        self._legal_destinations: Set[str] = set()
        self._last_move: Optional[tuple[str, str]] = None
        self._hover_square: Optional[str] = None
        self._is_check: bool = False
        self._king_square: Optional[str] = None
        self._enabled: bool = True

        # Visual settings
        self._square_size: int = 70
        self._margin: int = 25

        self.setMouseTracking(True)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self._update_minimum_size()

    def _update_minimum_size(self) -> None:
        size = 8 * self._square_size + 2 * self._margin
        self.setMinimumSize(size, size)

    def set_square_size(self, size: int) -> None:
        """Set the square size for the chess board.

        Args:
            size: Size of each square in pixels.
        """
        self._square_size = size
        self._update_minimum_size()
        self.update()  # Trigger repaint

    def update_from_payload(self, payload: Dict[str, Any]) -> None:
        """Update chess board from payload."""
        # Handle both wrapped {"chess": {...}} and flat payload from to_dict()
        chess_data = payload.get("chess", {})
        if not chess_data:
            # Flat payload - use top level (from adapter's to_dict())
            chess_data = payload

        # Get FEN
        fen = chess_data.get("fen")
        if fen:
            self._fen = fen
            self._parse_fen(fen)

        # Get legal moves
        legal_moves = chess_data.get("legal_moves", [])
        if isinstance(legal_moves, list):
            self._legal_moves = set(legal_moves)

        # Get last move
        last_move = chess_data.get("last_move")
        if isinstance(last_move, (list, tuple)) and len(last_move) == 2:
            self._last_move = (str(last_move[0]), str(last_move[1]))

        # Get check state
        self._is_check = bool(chess_data.get("is_check", False))
        self._king_square = chess_data.get("king_square")

        # Get current player
        current = chess_data.get("current_player")
        if current:
            self._current_player = current

        self._selected_square = None
        self._legal_destinations.clear()
        self.update()

    def _parse_fen(self, fen: str) -> None:
        """Parse FEN string into board array."""
        parts = fen.split()
        position = parts[0]
        self._current_player = "white" if len(parts) < 2 or parts[1] == "w" else "black"

        self._board = [[None] * 8 for _ in range(8)]
        row, col = 7, 0

        for char in position:
            if char == "/":
                row -= 1
                col = 0
            elif char.isdigit():
                col += int(char)
            else:
                if 0 <= row < 8 and 0 <= col < 8:
                    self._board[row][col] = char
                col += 1

    def _update_legal_destinations(self) -> None:
        """Update legal destinations for selected piece."""
        self._legal_destinations.clear()
        if self._selected_square is None:
            return
        for move in self._legal_moves:
            if move.startswith(self._selected_square):
                dest = move[2:4]
                self._legal_destinations.add(dest)

    def _square_to_coords(self, square: str) -> tuple[int, int]:
        col = ord(square[0]) - ord("a")
        row = int(square[1]) - 1
        return (row, col)

    def _coords_to_square(self, row: int, col: int) -> str:
        return f"{chr(ord('a') + col)}{row + 1}"

    def _pixel_to_square(self, pos: QtCore.QPointF) -> Optional[str]:
        x, y = int(pos.x()) - self._margin, int(pos.y()) - self._margin
        col = x // self._square_size
        row = 7 - (y // self._square_size)
        if 0 <= col < 8 and 0 <= row < 8:
            return self._coords_to_square(row, col)
        return None

    def _square_to_pixel(self, square: str) -> QtCore.QPoint:
        row, col = self._square_to_coords(square)
        x = self._margin + col * self._square_size
        y = self._margin + (7 - row) * self._square_size
        return QtCore.QPoint(x, y)

    def _has_piece_at(self, square: str) -> bool:
        row, col = self._square_to_coords(square)
        return self._board[row][col] is not None

    def _is_own_piece(self, square: str) -> bool:
        row, col = self._square_to_coords(square)
        piece = self._board[row][col]
        if piece is None:
            return False
        is_white_piece = piece.isupper()
        return (self._current_player == "white") == is_white_piece

    # Event handlers
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self._enabled or event.button() != QtCore.Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)

        square = self._pixel_to_square(event.position())
        if square is None:
            return super().mousePressEvent(event)

        if self._selected_square is None:
            if self._has_piece_at(square) and self._is_own_piece(square):
                self._selected_square = square
                self._update_legal_destinations()
                self.update()
        else:
            if square == self._selected_square:
                self._selected_square = None
                self._legal_destinations.clear()
                self.update()
            elif square in self._legal_destinations:
                from_sq = self._selected_square
                self._selected_square = None
                self._legal_destinations.clear()
                self.move_made.emit(from_sq, square)
                self.update()
            elif self._has_piece_at(square) and self._is_own_piece(square):
                self._selected_square = square
                self._update_legal_destinations()
                self.update()
            else:
                self._selected_square = None
                self._legal_destinations.clear()
                self.update()

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        square = self._pixel_to_square(event.position())
        if square != self._hover_square:
            self._hover_square = square
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self._hover_square = None
        self.update()
        super().leaveEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        size = min(event.size().width(), event.size().height())
        board_space = size - 2 * self._margin
        self._square_size = max(20, board_space // 8)
        super().resizeEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        self._draw_squares(painter)
        self._draw_highlights(painter)
        self._draw_pieces(painter)
        self._draw_coordinates(painter)

        painter.end()

    def _draw_squares(self, painter: QtGui.QPainter) -> None:
        for row in range(8):
            for col in range(8):
                x = self._margin + col * self._square_size
                y = self._margin + (7 - row) * self._square_size
                is_light = (row + col) % 2 == 0
                color = _CHESS_LIGHT_SQUARE if is_light else _CHESS_DARK_SQUARE
                painter.fillRect(x, y, self._square_size, self._square_size, color)

    def _draw_highlights(self, painter: QtGui.QPainter) -> None:
        # Last move
        if self._last_move:
            for sq in self._last_move:
                pos = self._square_to_pixel(sq)
                painter.fillRect(
                    pos.x(), pos.y(), self._square_size, self._square_size,
                    _CHESS_LAST_MOVE
                )

        # Check
        if self._is_check and self._king_square:
            pos = self._square_to_pixel(self._king_square)
            painter.fillRect(
                pos.x(), pos.y(), self._square_size, self._square_size,
                _CHESS_CHECK
            )

        # Selected
        if self._selected_square:
            pos = self._square_to_pixel(self._selected_square)
            painter.fillRect(
                pos.x(), pos.y(), self._square_size, self._square_size,
                _CHESS_SELECTED
            )

        # Legal moves
        for dest in self._legal_destinations:
            pos = self._square_to_pixel(dest)
            center_x = pos.x() + self._square_size // 2
            center_y = pos.y() + self._square_size // 2

            if self._has_piece_at(dest):
                painter.setPen(QtGui.QPen(_CHESS_LEGAL_MOVE, 4))
                painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                radius = self._square_size // 2 - 4
            else:
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(_CHESS_LEGAL_MOVE)
                radius = self._square_size // 6

            painter.drawEllipse(QtCore.QPoint(center_x, center_y), radius, radius)

        # Hover
        if self._hover_square and self._hover_square != self._selected_square:
            pos = self._square_to_pixel(self._hover_square)
            painter.fillRect(
                pos.x(), pos.y(), self._square_size, self._square_size,
                _CHESS_HOVER
            )

    def _draw_pieces(self, painter: QtGui.QPainter) -> None:
        font = QtGui.QFont("Arial", int(self._square_size * 0.7))
        painter.setFont(font)

        for row in range(8):
            for col in range(8):
                piece = self._board[row][col]
                if piece is None:
                    continue

                square = self._coords_to_square(row, col)
                pos = self._square_to_pixel(square)
                symbol = _CHESS_PIECES.get(piece, "?")

                if piece.isupper():
                    painter.setPen(QtGui.QColor(255, 255, 255))
                else:
                    painter.setPen(QtGui.QColor(0, 0, 0))

                rect = QtCore.QRect(
                    pos.x(), pos.y(), self._square_size, self._square_size
                )
                painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, symbol)

    def _draw_coordinates(self, painter: QtGui.QPainter) -> None:
        font = QtGui.QFont("Arial", max(10, int(self._square_size * 0.18)))
        painter.setFont(font)
        painter.setPen(QtGui.QColor(40, 40, 40))
        font_metrics = QtGui.QFontMetrics(font)

        for i in range(8):
            # File labels (a-h)
            file_char = chr(ord("a") + i)
            char_width = font_metrics.horizontalAdvance(file_char)
            x = self._margin + i * self._square_size + (self._square_size - char_width) // 2
            painter.drawText(x, self._margin - 8, file_char)
            board_bottom = self._margin + 8 * self._square_size
            painter.drawText(x, board_bottom + font_metrics.ascent() + 5, file_char)

            # Rank labels (1-8)
            rank_char = str(8 - i)
            char_width = font_metrics.horizontalAdvance(rank_char)
            y = self._margin + i * self._square_size + (self._square_size + font_metrics.ascent()) // 2
            painter.drawText(self._margin - char_width - 8, y, rank_char)
            board_right = self._margin + 8 * self._square_size
            painter.drawText(board_right + 8, y, rank_char)


# =============================================================================
# Connect Four Board Renderer
# =============================================================================

_C4_BOARD = QtGui.QColor(0, 100, 200)
_C4_EMPTY = QtGui.QColor(240, 240, 240)
_C4_PLAYER_0 = QtGui.QColor(255, 50, 50)  # Red
_C4_PLAYER_1 = QtGui.QColor(255, 220, 50)  # Yellow
_C4_HOVER = QtGui.QColor(100, 255, 100, 150)
_C4_WIN = QtGui.QColor(255, 255, 255, 180)


class _ConnectFourBoardRenderer(QtWidgets.QWidget):
    """Connect Four board renderer with interactive column selection."""

    column_clicked = QtCore.Signal(int)

    ROWS = 6
    COLS = 7

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        self._board: List[List[int]] = [[0] * self.COLS for _ in range(self.ROWS)]
        self._current_player: str = "player_0"
        self._legal_columns: Set[int] = set(range(self.COLS))
        self._last_column: Optional[int] = None
        self._hover_column: Optional[int] = None
        self._is_game_over: bool = False
        self._winning_positions: List[tuple[int, int]] = []
        self._enabled: bool = True

        self._cell_size: int = 70
        self._padding: int = 10

        self.setMouseTracking(True)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self._update_minimum_size()

    def _update_minimum_size(self) -> None:
        width = self.COLS * self._cell_size + 2 * self._padding
        label_height = int(self._cell_size * 0.4)
        height = (self.ROWS + 1) * self._cell_size + 2 * self._padding + label_height
        self.setMinimumSize(width, height)

    def update_from_payload(self, payload: Dict[str, Any]) -> None:
        """Update Connect Four board from payload."""
        # Handle both wrapped {"connect_four": {...}} and flat payload from to_dict()
        c4_data = payload.get("connect_four", {})
        if not c4_data:
            # Flat payload - use top level (from adapter's to_dict())
            c4_data = payload

        # Get board
        board = c4_data.get("board")
        if isinstance(board, list):
            self._board = [row[:] for row in board]

        # Get legal columns
        legal = c4_data.get("legal_columns", list(range(self.COLS)))
        if isinstance(legal, list):
            self._legal_columns = set(legal)

        # Get current player
        self._current_player = c4_data.get("current_player", "player_0")

        # Get last column
        self._last_column = c4_data.get("last_column")

        # Get game over state
        self._is_game_over = bool(c4_data.get("is_game_over", False))

        # Find winning positions if game over
        if self._is_game_over:
            self._find_winning_positions()
        else:
            self._winning_positions = []

        self.update()

    def _find_winning_positions(self) -> None:
        """Find winning piece positions for highlighting."""
        self._winning_positions = []
        for row in range(self.ROWS):
            for col in range(self.COLS):
                piece = self._board[row][col]
                if piece == 0:
                    continue

                # Horizontal
                if col <= 3 and all(self._board[row][col + i] == piece for i in range(4)):
                    self._winning_positions.extend([(row, col + i) for i in range(4)])

                # Vertical
                if row <= 2 and all(self._board[row + i][col] == piece for i in range(4)):
                    self._winning_positions.extend([(row + i, col) for i in range(4)])

                # Diagonal down-right
                if row <= 2 and col <= 3 and all(
                    self._board[row + i][col + i] == piece for i in range(4)
                ):
                    self._winning_positions.extend([(row + i, col + i) for i in range(4)])

                # Diagonal up-right
                if row >= 3 and col <= 3 and all(
                    self._board[row - i][col + i] == piece for i in range(4)
                ):
                    self._winning_positions.extend([(row - i, col + i) for i in range(4)])

    def _pixel_to_column(self, pos: QtCore.QPointF) -> Optional[int]:
        x = int(pos.x()) - self._padding
        col = x // self._cell_size
        if 0 <= col < self.COLS:
            return col
        return None

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self._enabled or self._is_game_over:
            return super().mousePressEvent(event)
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)

        col = self._pixel_to_column(event.position())
        if col is not None and col in self._legal_columns:
            self.column_clicked.emit(col)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        col = self._pixel_to_column(event.position())
        if col != self._hover_column:
            self._hover_column = col
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self._hover_column = None
        self.update()
        super().leaveEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        width = event.size().width() - 2 * self._padding
        height = event.size().height() - 2 * self._padding
        cell_from_width = width // self.COLS
        cell_from_height = height // (self.ROWS + 1)
        self._cell_size = min(cell_from_width, cell_from_height)
        super().resizeEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        self._draw_preview_row(painter)
        self._draw_board(painter)
        self._draw_pieces(painter)
        self._draw_column_labels(painter)

        painter.end()

    def _draw_preview_row(self, painter: QtGui.QPainter) -> None:
        if self._is_game_over or not self._enabled:
            return
        if self._hover_column is not None and self._hover_column in self._legal_columns:
            col = self._hover_column
            x = self._padding + col * self._cell_size + self._cell_size // 2
            y = self._padding + self._cell_size // 2

            color = _C4_PLAYER_0 if self._current_player == "player_0" else _C4_PLAYER_1
            color = QtGui.QColor(color)
            color.setAlpha(150)

            radius = self._cell_size // 2 - 6
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(QtCore.QPoint(x, y), radius, radius)

    def _draw_board(self, painter: QtGui.QPainter) -> None:
        board_x = self._padding
        board_y = self._padding + self._cell_size
        board_width = self.COLS * self._cell_size
        board_height = self.ROWS * self._cell_size

        painter.setBrush(_C4_BOARD)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawRoundedRect(board_x, board_y, board_width, board_height, 10, 10)

        painter.setBrush(_C4_EMPTY)
        for row in range(self.ROWS):
            for col in range(self.COLS):
                if self._board[row][col] == 0:
                    x = self._padding + col * self._cell_size + self._cell_size // 2
                    y = self._padding + (row + 1) * self._cell_size + self._cell_size // 2
                    radius = self._cell_size // 2 - 6
                    painter.drawEllipse(QtCore.QPoint(x, y), radius, radius)

    def _draw_pieces(self, painter: QtGui.QPainter) -> None:
        for row in range(self.ROWS):
            for col in range(self.COLS):
                piece = self._board[row][col]
                if piece == 0:
                    continue

                x = self._padding + col * self._cell_size + self._cell_size // 2
                y = self._padding + (row + 1) * self._cell_size + self._cell_size // 2
                radius = self._cell_size // 2 - 6

                color = _C4_PLAYER_0 if piece == 1 else _C4_PLAYER_1
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(color)
                painter.drawEllipse(QtCore.QPoint(x, y), radius, radius)

                if (row, col) in self._winning_positions:
                    painter.setBrush(_C4_WIN)
                    painter.drawEllipse(QtCore.QPoint(x, y), radius // 2, radius // 2)

        # Hover highlight
        if (
            self._hover_column is not None
            and self._hover_column in self._legal_columns
            and not self._is_game_over
        ):
            col = self._hover_column
            x = self._padding + col * self._cell_size
            y = self._padding + self._cell_size
            height = self.ROWS * self._cell_size

            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(_C4_HOVER)
            painter.drawRect(x, y, self._cell_size, height)

    def _draw_column_labels(self, painter: QtGui.QPainter) -> None:
        font = QtGui.QFont("Arial", int(self._cell_size * 0.25))
        painter.setFont(font)
        painter.setPen(QtGui.QColor(80, 80, 80))

        board_bottom = self._padding + (self.ROWS + 1) * self._cell_size
        label_margin = int(self._cell_size * 0.3)
        y = board_bottom + label_margin

        for col in range(self.COLS):
            x = self._padding + col * self._cell_size + self._cell_size // 2 - 5
            painter.drawText(x, y, str(col + 1))


# =============================================================================
# Go Board Renderer
# =============================================================================

_GO_BOARD = QtGui.QColor(220, 179, 92)
_GO_LINE = QtGui.QColor(0, 0, 0)
_GO_BLACK = QtGui.QColor(20, 20, 20)
_GO_WHITE = QtGui.QColor(250, 250, 250)
_GO_HOVER = QtGui.QColor(100, 200, 100, 150)
_GO_STAR = QtGui.QColor(0, 0, 0)


class _GoBoardRenderer(QtWidgets.QWidget):
    """Go board renderer with interactive stone placement."""

    intersection_clicked = QtCore.Signal(int, int)  # row, col
    pass_requested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        self._board_size: int = 19
        self._board: List[List[int]] = [[0] * 19 for _ in range(19)]
        self._current_player: str = "black_0"
        self._legal_moves: Set[int] = set()
        self._last_move: Optional[int] = None
        self._hover_pos: Optional[tuple[int, int]] = None
        self._is_game_over: bool = False
        self._enabled: bool = True

        self._cell_size: int = 30
        self._margin: int = 40
        self._stone_radius: int = int(self._cell_size * 0.45)

        self.setMouseTracking(True)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self._update_minimum_size()

    def _update_minimum_size(self) -> None:
        board_width = (self._board_size - 1) * self._cell_size
        size = board_width + 2 * self._margin
        self.setMinimumSize(size, size)

    def update_from_payload(self, payload: Dict[str, Any]) -> None:
        """Update Go board from payload."""
        # Handle both wrapped {"go": {...}} and flat payload from to_dict()
        go_data = payload.get("go", {})
        if not go_data:
            # Flat payload - use top level (from adapter's to_dict())
            go_data = payload

        # Get board size
        size = go_data.get("board_size", 19)
        if size != self._board_size:
            self._board_size = size
            self._board = [[0] * size for _ in range(size)]
            self._update_minimum_size()

        # Get board state
        board = go_data.get("board")
        if isinstance(board, list):
            self._board = [row[:] for row in board]

        # Get legal moves
        legal = go_data.get("legal_moves", [])
        if isinstance(legal, list):
            self._legal_moves = set(legal)

        # Get current player
        self._current_player = go_data.get("current_player", "black_0")

        # Get last move
        self._last_move = go_data.get("last_move")

        # Get game over state
        self._is_game_over = bool(go_data.get("is_game_over", False))

        self.update()

    def action_to_coords(self, action: int) -> Optional[tuple[int, int]]:
        if action < 0 or action >= self._board_size**2:
            return None
        row = action // self._board_size
        col = action % self._board_size
        return (row, col)

    def coords_to_action(self, row: int, col: int) -> int:
        return row * self._board_size + col

    def _pixel_to_intersection(self, pos: QtCore.QPointF) -> Optional[tuple[int, int]]:
        x = int(pos.x()) - self._margin
        y = int(pos.y()) - self._margin

        col = round(x / self._cell_size)
        row = round(y / self._cell_size)

        ix = col * self._cell_size
        iy = row * self._cell_size
        dist = ((x - ix) ** 2 + (y - iy) ** 2) ** 0.5

        if (
            dist <= self._cell_size * 0.4
            and 0 <= row < self._board_size
            and 0 <= col < self._board_size
        ):
            return (row, col)
        return None

    def _intersection_to_pixel(self, row: int, col: int) -> QtCore.QPoint:
        x = self._margin + col * self._cell_size
        y = self._margin + row * self._cell_size
        return QtCore.QPoint(x, y)

    def _get_star_points(self) -> List[tuple[int, int]]:
        if self._board_size == 19:
            return [
                (3, 3), (3, 9), (3, 15),
                (9, 3), (9, 9), (9, 15),
                (15, 3), (15, 9), (15, 15),
            ]
        elif self._board_size == 13:
            return [(3, 3), (3, 9), (6, 6), (9, 3), (9, 9)]
        elif self._board_size == 9:
            return [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)]
        return []

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self._enabled or self._is_game_over:
            return super().mousePressEvent(event)
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)

        intersection = self._pixel_to_intersection(event.position())
        if intersection is not None:
            row, col = intersection
            action = self.coords_to_action(row, col)
            if action in self._legal_moves:
                self.intersection_clicked.emit(row, col)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        intersection = self._pixel_to_intersection(event.position())
        if intersection != self._hover_pos:
            self._hover_pos = intersection
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self._hover_pos = None
        self.update()
        super().leaveEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        available = min(event.size().width(), event.size().height())
        board_space = available - 2 * self._margin
        self._cell_size = max(15, board_space // (self._board_size - 1))
        self._stone_radius = int(self._cell_size * 0.45)
        super().resizeEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        self._draw_board_background(painter)
        self._draw_grid(painter)
        self._draw_star_points(painter)
        self._draw_coordinates(painter)
        self._draw_stones(painter)
        self._draw_hover(painter)
        self._draw_last_move_marker(painter)

        painter.end()

    def _draw_board_background(self, painter: QtGui.QPainter) -> None:
        board_width = (self._board_size - 1) * self._cell_size
        painter.fillRect(
            self._margin - self._cell_size // 2,
            self._margin - self._cell_size // 2,
            board_width + self._cell_size,
            board_width + self._cell_size,
            _GO_BOARD,
        )

    def _draw_grid(self, painter: QtGui.QPainter) -> None:
        painter.setPen(QtGui.QPen(_GO_LINE, 1))
        board_width = (self._board_size - 1) * self._cell_size

        for row in range(self._board_size):
            y = self._margin + row * self._cell_size
            painter.drawLine(self._margin, y, self._margin + board_width, y)

        for col in range(self._board_size):
            x = self._margin + col * self._cell_size
            painter.drawLine(x, self._margin, x, self._margin + board_width)

    def _draw_star_points(self, painter: QtGui.QPainter) -> None:
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(_GO_STAR)
        star_radius = max(3, self._cell_size // 8)

        for row, col in self._get_star_points():
            pos = self._intersection_to_pixel(row, col)
            painter.drawEllipse(pos, star_radius, star_radius)

    def _draw_coordinates(self, painter: QtGui.QPainter) -> None:
        font = QtGui.QFont("Arial", max(9, self._cell_size // 3))
        painter.setFont(font)
        painter.setPen(QtGui.QColor(40, 40, 40))
        font_metrics = QtGui.QFontMetrics(font)

        letters = "ABCDEFGHJKLMNOPQRST"
        for col in range(self._board_size):
            letter = letters[col]
            letter_width = font_metrics.horizontalAdvance(letter)
            x = self._margin + col * self._cell_size - letter_width // 2

            painter.drawText(x, self._margin - self._cell_size // 2 - 5, letter)
            board_bottom = self._margin + (self._board_size - 1) * self._cell_size
            painter.drawText(
                x, board_bottom + self._cell_size // 2 + font_metrics.ascent(), letter
            )

        for row in range(self._board_size):
            label = str(self._board_size - row)
            label_width = font_metrics.horizontalAdvance(label)
            y = self._margin + row * self._cell_size + font_metrics.ascent() // 2

            painter.drawText(
                self._margin - self._cell_size // 2 - label_width - 5, y, label
            )
            board_right = self._margin + (self._board_size - 1) * self._cell_size
            painter.drawText(board_right + self._cell_size // 2 + 5, y, label)

    def _draw_stones(self, painter: QtGui.QPainter) -> None:
        for row in range(self._board_size):
            for col in range(self._board_size):
                stone = self._board[row][col]
                if stone == 0:
                    continue

                pos = self._intersection_to_pixel(row, col)

                if stone == 1:  # Black
                    painter.setPen(QtCore.Qt.PenStyle.NoPen)
                    painter.setBrush(_GO_BLACK)
                else:  # White
                    painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0), 1))
                    painter.setBrush(_GO_WHITE)

                painter.drawEllipse(pos, self._stone_radius, self._stone_radius)

    def _draw_hover(self, painter: QtGui.QPainter) -> None:
        if self._hover_pos is None or self._is_game_over or not self._enabled:
            return

        row, col = self._hover_pos
        action = self.coords_to_action(row, col)

        if action not in self._legal_moves:
            return
        if self._board[row][col] != 0:
            return

        pos = self._intersection_to_pixel(row, col)
        color = QtGui.QColor(
            _GO_BLACK if "black" in self._current_player else _GO_WHITE
        )
        color.setAlpha(100)

        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(pos, self._stone_radius, self._stone_radius)

    def _draw_last_move_marker(self, painter: QtGui.QPainter) -> None:
        if self._last_move is None:
            return

        # Handle both int action (Chess/Connect Four) and tuple coords (Go)
        if isinstance(self._last_move, tuple):
            row, col = self._last_move
        else:
            coords = self.action_to_coords(self._last_move)
            if coords is None:
                return
            row, col = coords
        pos = self._intersection_to_pixel(row, col)
        marker_radius = max(3, self._stone_radius // 3)
        stone = self._board[row][col]

        if stone == 1:
            painter.setBrush(QtGui.QColor(255, 255, 255))
        else:
            painter.setBrush(QtGui.QColor(0, 0, 0))

        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawEllipse(pos, marker_radius, marker_radius)


# =============================================================================
# Tic-Tac-Toe Board Renderer
# =============================================================================

# Tic-Tac-Toe colors
_TTT_GRID_COLOR = QtGui.QColor(50, 50, 50)
_TTT_X_COLOR = QtGui.QColor(50, 100, 200)
_TTT_O_COLOR = QtGui.QColor(200, 50, 50)
_TTT_BG_COLOR = QtGui.QColor(245, 245, 220)
_TTT_HOVER = QtGui.QColor(100, 100, 255, 50)
_TTT_LAST_MOVE = QtGui.QColor(255, 255, 0, 100)
_TTT_HUMAN_WIN_HIGHLIGHT = QtGui.QColor(0, 200, 0, 120)  # Green for human win
_TTT_AI_WIN_HIGHLIGHT = QtGui.QColor(200, 50, 50, 120)  # Red for AI win


class _TicTacToeBoardRenderer(QtWidgets.QWidget):
    """Tic-Tac-Toe board renderer with interactive cell selection."""

    cell_clicked = QtCore.Signal(int, int)  # row, col

    SIZE = 3  # 3x3 board

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        # Board state (0=empty, 1=X, 2=O)
        self._board: List[List[int]] = [[0] * self.SIZE for _ in range(self.SIZE)]
        self._legal_actions: List[int] = list(range(9))
        self._current_player: str = "player_1"
        self._last_row: Optional[int] = None
        self._last_col: Optional[int] = None
        self._winning_positions: Optional[List[tuple[int, int]]] = None
        self._is_game_over: bool = False
        self._winner: Optional[str] = None  # "player_1", "player_2", or "draw"
        self._human_player: str = "player_1"  # Track who human is for win highlight color

        # Rendering settings
        self._cell_size: int = 100
        self._margin: int = 20
        self._line_width: int = 4
        self._enabled: bool = True
        self._hover_pos: Optional[tuple[int, int]] = None

        # Widget setup
        self.setMouseTracking(True)
        self.setMinimumSize(self._margin * 2 + self._cell_size * self.SIZE,
                           self._margin * 2 + self._cell_size * self.SIZE)

    def update_from_payload(self, payload: Dict[str, Any]) -> None:
        """Update board state from adapter payload."""
        # Get board
        board_data = payload.get("board")
        if board_data is not None:
            self._board = [row[:] for row in board_data]

        # Get legal actions
        self._legal_actions = payload.get("legal_actions", [])

        # Get current player
        self._current_player = payload.get("current_player", "player_1")

        # Get last move
        self._last_row = payload.get("last_row")
        self._last_col = payload.get("last_col")

        # Get game over state
        self._is_game_over = bool(payload.get("is_game_over", False))
        self._winning_positions = payload.get("winning_positions")
        self._winner = payload.get("winner")

        # Get human player for win highlight color
        human_player = payload.get("human_player")
        if human_player:
            self._human_player = human_player

        self.update()

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable interaction."""
        self._enabled = enabled

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        """Handle resize to keep board square and centered."""
        super().resizeEvent(event)
        available = min(self.width(), self.height())
        self._cell_size = (available - 2 * self._margin) // self.SIZE
        self._margin = (available - self._cell_size * self.SIZE) // 2

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Paint the Tic-Tac-Toe board."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        # Background
        painter.fillRect(self.rect(), _TTT_BG_COLOR)

        # Draw grid
        self._draw_grid(painter)

        # Draw marks (X and O)
        self._draw_marks(painter)

        # Draw hover
        self._draw_hover(painter)

        # Draw last move highlight
        self._draw_last_move(painter)

        # Draw winning line
        self._draw_winning_line(painter)

        painter.end()

    def _draw_grid(self, painter: QtGui.QPainter) -> None:
        """Draw the grid lines."""
        pen = QtGui.QPen(_TTT_GRID_COLOR, self._line_width)
        painter.setPen(pen)

        # Vertical lines
        for i in range(1, self.SIZE):
            x = self._margin + i * self._cell_size
            painter.drawLine(x, self._margin,
                           x, self._margin + self.SIZE * self._cell_size)

        # Horizontal lines
        for i in range(1, self.SIZE):
            y = self._margin + i * self._cell_size
            painter.drawLine(self._margin, y,
                           self._margin + self.SIZE * self._cell_size, y)

    def _draw_marks(self, painter: QtGui.QPainter) -> None:
        """Draw X and O marks."""
        for row in range(self.SIZE):
            for col in range(self.SIZE):
                mark = self._board[row][col]
                if mark == 0:
                    continue

                cx = self._margin + col * self._cell_size + self._cell_size // 2
                cy = self._margin + row * self._cell_size + self._cell_size // 2
                size = int(self._cell_size * 0.35)

                if mark == 1:  # X
                    pen = QtGui.QPen(_TTT_X_COLOR, self._line_width + 2)
                    painter.setPen(pen)
                    painter.drawLine(cx - size, cy - size, cx + size, cy + size)
                    painter.drawLine(cx - size, cy + size, cx + size, cy - size)
                else:  # O
                    pen = QtGui.QPen(_TTT_O_COLOR, self._line_width + 2)
                    painter.setPen(pen)
                    painter.drawEllipse(QtCore.QPoint(cx, cy), size, size)

    def _draw_hover(self, painter: QtGui.QPainter) -> None:
        """Draw hover highlight on empty cells."""
        if self._hover_pos is None or self._is_game_over or not self._enabled:
            return

        row, col = self._hover_pos
        if self._board[row][col] != 0:
            return

        # Check if legal
        action = col * 3 + row  # PettingZoo column-major indexing
        if action not in self._legal_actions:
            return

        x = self._margin + col * self._cell_size
        y = self._margin + row * self._cell_size
        painter.fillRect(x, y, self._cell_size, self._cell_size, _TTT_HOVER)

    def _draw_last_move(self, painter: QtGui.QPainter) -> None:
        """Draw highlight on last move."""
        if self._last_row is None or self._last_col is None:
            return

        x = self._margin + self._last_col * self._cell_size
        y = self._margin + self._last_row * self._cell_size
        painter.fillRect(x, y, self._cell_size, self._cell_size, _TTT_LAST_MOVE)

    def _draw_winning_line(self, painter: QtGui.QPainter) -> None:
        """Draw highlight on winning positions.

        Uses green for human wins and red for AI wins.
        """
        if self._winning_positions is None:
            return

        # Determine highlight color based on winner
        if self._winner == self._human_player:
            highlight_color = _TTT_HUMAN_WIN_HIGHLIGHT  # Green for human win
        else:
            highlight_color = _TTT_AI_WIN_HIGHLIGHT  # Red for AI win

        for row, col in self._winning_positions:
            x = self._margin + col * self._cell_size
            y = self._margin + row * self._cell_size
            painter.fillRect(x, y, self._cell_size, self._cell_size, highlight_color)

    def _pixel_to_cell(self, pos: QtCore.QPointF) -> Optional[tuple[int, int]]:
        """Convert pixel position to cell coordinates."""
        x = int(pos.x()) - self._margin
        y = int(pos.y()) - self._margin

        if x < 0 or y < 0:
            return None

        col = x // self._cell_size
        row = y // self._cell_size

        if row >= self.SIZE or col >= self.SIZE:
            return None

        return (row, col)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse click to place mark."""
        if not self._enabled or self._is_game_over:
            return super().mousePressEvent(event)
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)

        cell = self._pixel_to_cell(event.position())
        if cell is not None:
            row, col = cell
            # Check if cell is empty and legal
            if self._board[row][col] == 0:
                action = col * 3 + row  # PettingZoo column-major indexing
                if action in self._legal_actions:
                    self.cell_clicked.emit(row, col)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse move for hover effect."""
        cell = self._pixel_to_cell(event.position())
        if cell != self._hover_pos:
            self._hover_pos = cell
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        """Clear hover when mouse leaves."""
        self._hover_pos = None
        self.update()
        super().leaveEvent(event)


# =============================================================================
# Sudoku Board Renderer
# =============================================================================

# Sudoku colors
_SUDOKU_BG = QtGui.QColor(255, 255, 255)
_SUDOKU_GRID_THIN = QtGui.QColor(200, 200, 200)
_SUDOKU_GRID_THICK = QtGui.QColor(0, 0, 0)
_SUDOKU_FIXED = QtGui.QColor(20, 20, 20)  # Pre-filled clues
_SUDOKU_PLACED = QtGui.QColor(50, 100, 200)  # Player-placed digits
_SUDOKU_SELECTED = QtGui.QColor(187, 222, 251)  # Selected cell
_SUDOKU_SAME_ROW_COL_BOX = QtGui.QColor(232, 245, 253)  # Same row/col/box as selected
_SUDOKU_SAME_NUMBER = QtGui.QColor(200, 230, 255)  # Same number highlighted
_SUDOKU_INVALID = QtGui.QColor(255, 200, 200)  # Invalid cell (conflict)
_SUDOKU_HOVER = QtGui.QColor(220, 220, 255)


class _SudokuBoardRenderer(QtWidgets.QWidget):
    """Sudoku board renderer with interactive cell selection and number input.

    Interaction:
    - Click a cell to select it
    - Press 1-9 to enter a digit in the selected cell
    - Press Delete/Backspace to clear a cell (if allowed)
    """

    cell_selected = QtCore.Signal(int, int)  # row, col
    digit_entered = QtCore.Signal(int, int, int)  # row, col, digit (1-9)
    cell_cleared = QtCore.Signal(int, int)  # row, col

    SIZE = 9  # 9x9 grid
    BOX_SIZE = 3  # 3x3 boxes

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        # Board state (0 = empty, 1-9 = digit)
        self._board: List[List[int]] = [[0] * self.SIZE for _ in range(self.SIZE)]
        self._fixed_cells: Set[tuple[int, int]] = set()  # Pre-filled clue positions
        self._invalid_cells: Set[tuple[int, int]] = set()  # Cells with conflicts

        # Action mask from Jumanji (shape: 9*9*9 = 729)
        self._action_mask: Optional[List[bool]] = None

        # Interaction state
        self._selected_cell: Optional[tuple[int, int]] = None
        self._hover_cell: Optional[tuple[int, int]] = None
        self._enabled: bool = True

        # Visual settings
        self._cell_size: int = 50
        self._margin: int = 30
        self._thin_line: int = 1
        self._thick_line: int = 3

        # Widget setup
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self._update_minimum_size()

    def _update_minimum_size(self) -> None:
        size = self.SIZE * self._cell_size + 2 * self._margin
        self.setMinimumSize(size, size)

    def update_from_payload(self, payload: Dict[str, Any]) -> None:
        """Update Sudoku board from adapter payload."""
        # Get board (9x9 array, 0 = empty, 1-9 = digit)
        board = payload.get("board")
        if isinstance(board, (list, tuple)):
            # Handle nested list or numpy array
            new_board = []
            for row in board:
                if hasattr(row, "tolist"):
                    new_board.append(list(row.tolist()))
                else:
                    new_board.append(list(row))
            self._board = new_board

        # Get fixed cells (clues that can't be changed)
        fixed = payload.get("fixed_cells")
        if isinstance(fixed, (list, set)):
            self._fixed_cells = set(tuple(c) for c in fixed)
        elif fixed is None:
            # Infer fixed cells from initial non-zero values
            # Only do this on first load
            if not self._fixed_cells:
                for r in range(self.SIZE):
                    for c in range(self.SIZE):
                        if self._board[r][c] != 0:
                            self._fixed_cells.add((r, c))

        # Get action mask (729 bools)
        action_mask = payload.get("action_mask")
        if action_mask is not None:
            if hasattr(action_mask, "tolist"):
                self._action_mask = action_mask.tolist()
            else:
                self._action_mask = list(action_mask)

        # Get invalid cells (conflicts)
        invalid = payload.get("invalid_cells")
        if isinstance(invalid, (list, set)):
            self._invalid_cells = set(tuple(c) for c in invalid)
        else:
            self._invalid_cells = set()

        self.update()

    def get_valid_digits(self, row: int, col: int) -> List[int]:
        """Get list of valid digits for a cell based on action mask."""
        if self._action_mask is None:
            return list(range(1, 10))

        valid = []
        for digit in range(1, 10):
            action = row * 81 + col * 9 + (digit - 1)
            if action < len(self._action_mask) and self._action_mask[action]:
                valid.append(digit)
        return valid

    def is_cell_editable(self, row: int, col: int) -> bool:
        """Check if a cell can be edited (not a fixed clue)."""
        return (row, col) not in self._fixed_cells

    def _pixel_to_cell(self, pos: QtCore.QPointF) -> Optional[tuple[int, int]]:
        """Convert pixel position to cell coordinates."""
        x = int(pos.x()) - self._margin
        y = int(pos.y()) - self._margin

        if x < 0 or y < 0:
            return None

        col = x // self._cell_size
        row = y // self._cell_size

        if row >= self.SIZE or col >= self.SIZE:
            return None

        return (row, col)

    def _cell_to_pixel(self, row: int, col: int) -> tuple[int, int]:
        """Convert cell coordinates to top-left pixel position."""
        x = self._margin + col * self._cell_size
        y = self._margin + row * self._cell_size
        return (x, y)

    # -------------------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------------------

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self._enabled:
            return super().mousePressEvent(event)
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)

        cell = self._pixel_to_cell(event.position())
        if cell is not None:
            row, col = cell
            self._selected_cell = (row, col)
            self.cell_selected.emit(row, col)
            self.update()

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        cell = self._pixel_to_cell(event.position())
        if cell != self._hover_cell:
            self._hover_cell = cell
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self._hover_cell = None
        self.update()
        super().leaveEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Handle number key input for selected cell."""
        if not self._enabled or self._selected_cell is None:
            return super().keyPressEvent(event)

        row, col = self._selected_cell

        # Check if cell is editable
        if not self.is_cell_editable(row, col):
            return super().keyPressEvent(event)

        key = event.key()

        # Number keys 1-9
        if QtCore.Qt.Key.Key_1 <= key <= QtCore.Qt.Key.Key_9:
            digit = key - QtCore.Qt.Key.Key_1 + 1
            valid_digits = self.get_valid_digits(row, col)
            if digit in valid_digits:
                self.digit_entered.emit(row, col, digit)
            return

        # Numpad 1-9
        if QtCore.Qt.Key.Key_1 <= key - 0x30 <= QtCore.Qt.Key.Key_9 - 0x30:
            # Numpad offset
            pass  # Handle numpad if needed

        # Delete/Backspace to clear
        if key in (QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace):
            if self._board[row][col] != 0:
                self.cell_cleared.emit(row, col)
            return

        # Arrow keys to navigate selection
        if key == QtCore.Qt.Key.Key_Up and row > 0:
            self._selected_cell = (row - 1, col)
            self.cell_selected.emit(row - 1, col)
            self.update()
            return
        if key == QtCore.Qt.Key.Key_Down and row < self.SIZE - 1:
            self._selected_cell = (row + 1, col)
            self.cell_selected.emit(row + 1, col)
            self.update()
            return
        if key == QtCore.Qt.Key.Key_Left and col > 0:
            self._selected_cell = (row, col - 1)
            self.cell_selected.emit(row, col - 1)
            self.update()
            return
        if key == QtCore.Qt.Key.Key_Right and col < self.SIZE - 1:
            self._selected_cell = (row, col + 1)
            self.cell_selected.emit(row, col + 1)
            self.update()
            return

        super().keyPressEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        """Handle resize to keep board square and centered."""
        super().resizeEvent(event)
        available = min(self.width(), self.height())
        self._cell_size = (available - 2 * self._margin) // self.SIZE
        self._margin = (available - self._cell_size * self.SIZE) // 2

    # -------------------------------------------------------------------------
    # Painting
    # -------------------------------------------------------------------------

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        # Background
        painter.fillRect(self.rect(), _SUDOKU_BG)

        # Draw highlights (before grid lines)
        self._draw_highlights(painter)

        # Draw grid
        self._draw_grid(painter)

        # Draw digits
        self._draw_digits(painter)

        # Draw valid digit hints for selected cell
        self._draw_hints(painter)

        painter.end()

    def _draw_highlights(self, painter: QtGui.QPainter) -> None:
        """Draw cell highlights (selection, same row/col/box, hover)."""
        if self._selected_cell is not None:
            sel_row, sel_col = self._selected_cell
            selected_value = self._board[sel_row][sel_col]

            # Highlight same row, column, and 3x3 box
            for r in range(self.SIZE):
                for c in range(self.SIZE):
                    x, y = self._cell_to_pixel(r, c)

                    # Same row, column, or box
                    same_row = r == sel_row
                    same_col = c == sel_col
                    same_box = (r // 3 == sel_row // 3) and (c // 3 == sel_col // 3)

                    if same_row or same_col or same_box:
                        if (r, c) != self._selected_cell:
                            painter.fillRect(
                                x, y, self._cell_size, self._cell_size,
                                _SUDOKU_SAME_ROW_COL_BOX
                            )

                    # Highlight same number
                    if selected_value != 0 and self._board[r][c] == selected_value:
                        if (r, c) != self._selected_cell:
                            painter.fillRect(
                                x, y, self._cell_size, self._cell_size,
                                _SUDOKU_SAME_NUMBER
                            )

            # Selected cell highlight
            x, y = self._cell_to_pixel(sel_row, sel_col)
            painter.fillRect(x, y, self._cell_size, self._cell_size, _SUDOKU_SELECTED)

        # Hover highlight
        if self._hover_cell is not None and self._hover_cell != self._selected_cell:
            hx, hy = self._cell_to_pixel(*self._hover_cell)
            painter.fillRect(hx, hy, self._cell_size, self._cell_size, _SUDOKU_HOVER)

        # Invalid cell highlights
        for r, c in self._invalid_cells:
            x, y = self._cell_to_pixel(r, c)
            painter.fillRect(x, y, self._cell_size, self._cell_size, _SUDOKU_INVALID)

    def _draw_grid(self, painter: QtGui.QPainter) -> None:
        """Draw Sudoku grid lines (thin for cells, thick for 3x3 boxes)."""
        board_size = self.SIZE * self._cell_size

        # Draw thin lines first
        thin_pen = QtGui.QPen(_SUDOKU_GRID_THIN, self._thin_line)
        painter.setPen(thin_pen)

        for i in range(self.SIZE + 1):
            if i % 3 != 0:  # Skip thick line positions
                # Vertical
                x = self._margin + i * self._cell_size
                painter.drawLine(x, self._margin, x, self._margin + board_size)
                # Horizontal
                y = self._margin + i * self._cell_size
                painter.drawLine(self._margin, y, self._margin + board_size, y)

        # Draw thick lines (3x3 box boundaries)
        thick_pen = QtGui.QPen(_SUDOKU_GRID_THICK, self._thick_line)
        painter.setPen(thick_pen)

        for i in range(0, self.SIZE + 1, 3):
            # Vertical
            x = self._margin + i * self._cell_size
            painter.drawLine(x, self._margin, x, self._margin + board_size)
            # Horizontal
            y = self._margin + i * self._cell_size
            painter.drawLine(self._margin, y, self._margin + board_size, y)

    def _draw_digits(self, painter: QtGui.QPainter) -> None:
        """Draw placed digits on the board."""
        font = QtGui.QFont("Arial", int(self._cell_size * 0.55), QtGui.QFont.Weight.Bold)
        painter.setFont(font)

        for row in range(self.SIZE):
            for col in range(self.SIZE):
                digit = self._board[row][col]
                if digit == 0:
                    continue

                x, y = self._cell_to_pixel(row, col)
                rect = QtCore.QRect(x, y, self._cell_size, self._cell_size)

                # Fixed clues are darker, player-placed are blue
                if (row, col) in self._fixed_cells:
                    painter.setPen(_SUDOKU_FIXED)
                else:
                    painter.setPen(_SUDOKU_PLACED)

                painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, str(digit))

    def _draw_hints(self, painter: QtGui.QPainter) -> None:
        """Draw small candidate digits in empty cells (pencil marks style)."""
        if self._selected_cell is None:
            return

        sel_row, sel_col = self._selected_cell
        if self._board[sel_row][sel_col] != 0:
            return  # Cell already has a digit

        if not self.is_cell_editable(sel_row, sel_col):
            return

        valid = self.get_valid_digits(sel_row, sel_col)
        if not valid:
            return

        # Draw small digits at bottom of selected cell
        font = QtGui.QFont("Arial", int(self._cell_size * 0.18))
        painter.setFont(font)
        painter.setPen(QtGui.QColor(100, 100, 100))

        x, y = self._cell_to_pixel(sel_row, sel_col)

        # Layout valid digits in a 3x3 mini-grid within the cell
        mini_size = self._cell_size // 3
        for digit in valid:
            d_row = (digit - 1) // 3
            d_col = (digit - 1) % 3
            dx = x + d_col * mini_size
            dy = y + d_row * mini_size
            rect = QtCore.QRect(dx, dy, mini_size, mini_size)
            painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, str(digit))


# =============================================================================
# Checkers Board Renderer (OpenSpiel via Shimmy)
# =============================================================================

# Checkers colors
_CHECKERS_LIGHT_SQUARE = QtGui.QColor(240, 217, 181)  # Cream
_CHECKERS_DARK_SQUARE = QtGui.QColor(181, 136, 99)    # Brown
_CHECKERS_BLACK_PIECE = QtGui.QColor(50, 50, 50)      # Dark gray/black
_CHECKERS_WHITE_PIECE = QtGui.QColor(255, 255, 255)   # White
_CHECKERS_KING_CROWN = QtGui.QColor(255, 215, 0)      # Gold for king marker
_CHECKERS_SELECTED = QtGui.QColor(255, 255, 0, 150)   # Yellow highlight
_CHECKERS_LEGAL_MOVE = QtGui.QColor(0, 255, 0, 100)   # Green for legal moves
_CHECKERS_LAST_MOVE = QtGui.QColor(255, 255, 0, 80)   # Light yellow for last move
_CHECKERS_HOVER = QtGui.QColor(100, 100, 255, 50)     # Light blue hover


class _CheckersBoardRenderer(QtWidgets.QWidget):
    """Checkers/Draughts board renderer with interactive piece selection.

    Supports both 8x8 (American/Russian) and 10x10 (International) boards.
    Board values:
    - 0: Empty
    - 1: Black piece (player_0)
    - 2: Black king
    - 3: White piece (player_1)
    - 4: White king

    Note: Only dark squares are playable in checkers/draughts.
    """

    cell_clicked = QtCore.Signal(int, int)  # row, col

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        # Board state (dynamically sized - 8x8 or 10x10)
        self._board_size: int = 8  # Default to 8x8
        self._board: List[List[int]] = [[0] * self._board_size for _ in range(self._board_size)]
        self._current_player: str = "player_0"
        self._legal_moves: List[int] = []
        self._last_move: Optional[int] = None
        self._is_game_over: bool = False
        self._winner: Optional[str] = None
        self._move_count: int = 0

        # Interaction state
        self._selected_cell: Optional[tuple[int, int]] = None
        self._legal_destinations: Set[tuple[int, int]] = set()  # Valid destination cells
        self._moveable_cells: Set[tuple[int, int]] = set()  # Cells with pieces that can move
        self._hover_cell: Optional[tuple[int, int]] = None
        self._enabled: bool = True

        # Visual settings
        self._cell_size: int = 60
        self._margin: int = 25

        # Widget setup
        self.setMouseTracking(True)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self._update_minimum_size()

    def _update_minimum_size(self) -> None:
        size = self._board_size * self._cell_size + 2 * self._margin
        self.setMinimumSize(size, size)

    def update_from_payload(self, payload: Dict[str, Any]) -> None:
        """Update Checkers/Draughts board from adapter payload."""
        # Get board size (8x8 or 10x10)
        board_size = payload.get("board_size", 8)
        if board_size != self._board_size:
            self._board_size = board_size
            self._board = [[0] * self._board_size for _ in range(self._board_size)]
            self._update_minimum_size()

        # Get board array
        board = payload.get("board")
        if isinstance(board, list):
            self._board = [row[:] for row in board]

        # Get current player
        self._current_player = payload.get("current_player", "player_0")

        # Get legal moves (action indices)
        legal = payload.get("legal_moves", [])
        if isinstance(legal, list):
            self._legal_moves = list(legal)

        # Get last move
        self._last_move = payload.get("last_move")

        # Get game over state
        self._is_game_over = bool(payload.get("is_game_over", False))
        self._winner = payload.get("winner")
        self._move_count = payload.get("move_count", 0)

        # Clear selection on update (new board state)
        self._selected_cell = None
        self._legal_destinations = set()
        self._moveable_cells = set()

        self.update()

    def set_selection(
        self,
        selected_cell: Optional[tuple[int, int]],
        legal_destinations: Optional[List[tuple[int, int]]] = None,
    ) -> None:
        """Set the currently selected piece and its legal destinations.

        Args:
            selected_cell: The (row, col) of selected piece, or None to clear.
            legal_destinations: List of (row, col) tuples for valid destinations.
        """
        self._selected_cell = selected_cell
        if legal_destinations:
            self._legal_destinations = set(legal_destinations)
        else:
            self._legal_destinations = set()
        # Clear moveable hints when a piece is selected
        self._moveable_cells = set()
        self.update()

    def set_moveable_cells(
        self,
        cells: Optional[List[tuple[int, int]]] = None,
    ) -> None:
        """Highlight cells that have pieces which can move (for mandatory jump hints).

        Args:
            cells: List of (row, col) tuples for pieces that can move, or None to clear.
        """
        if cells:
            self._moveable_cells = set(cells)
        else:
            self._moveable_cells = set()
        self.update()

    def _pixel_to_cell(self, pos: QtCore.QPointF) -> Optional[tuple[int, int]]:
        """Convert pixel position to cell coordinates."""
        x = int(pos.x()) - self._margin
        y = int(pos.y()) - self._margin

        if x < 0 or y < 0:
            return None

        col = x // self._cell_size
        row = y // self._cell_size

        if row >= self._board_size or col >= self._board_size:
            return None

        return (row, col)

    def _cell_to_pixel(self, row: int, col: int) -> tuple[int, int]:
        """Convert cell coordinates to top-left pixel position."""
        x = self._margin + col * self._cell_size
        y = self._margin + row * self._cell_size
        return (x, y)

    def _is_dark_square(self, row: int, col: int) -> bool:
        """Check if a square is dark (playable in checkers)."""
        return (row + col) % 2 == 1

    # -------------------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------------------

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        log_constant(
            _LOG,
            LOG_CHECKERS_BOARD_CLICK,
            extra={"enabled": self._enabled, "game_over": self._is_game_over},
        )
        if not self._enabled or self._is_game_over:
            log_constant(
                _LOG,
                LOG_CHECKERS_BOARD_CLICK_IGNORED,
                extra={"enabled": self._enabled, "game_over": self._is_game_over},
            )
            return super().mousePressEvent(event)
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)

        cell = self._pixel_to_cell(event.position())
        if cell is not None:
            row, col = cell
            is_dark = self._is_dark_square(row, col)
            # Only allow clicks on dark squares
            if is_dark:
                log_constant(
                    _LOG,
                    LOG_CHECKERS_CELL_SIGNAL_EMITTED,
                    extra={"row": row, "col": col},
                )
                self.cell_clicked.emit(row, col)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        cell = self._pixel_to_cell(event.position())
        if cell != self._hover_cell:
            self._hover_cell = cell
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self._hover_cell = None
        self.update()
        super().leaveEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        """Handle resize to keep board square and centered."""
        super().resizeEvent(event)
        available = min(self.width(), self.height())
        self._cell_size = (available - 2 * self._margin) // self._board_size
        self._margin = (available - self._cell_size * self._board_size) // 2

    # -------------------------------------------------------------------------
    # Painting
    # -------------------------------------------------------------------------

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        # Draw board squares
        self._draw_squares(painter)

        # Draw highlights
        self._draw_highlights(painter)

        # Draw pieces
        self._draw_pieces(painter)

        # Draw coordinates
        self._draw_coordinates(painter)

        # Draw game status
        if self._is_game_over:
            self._draw_game_over(painter)

        painter.end()

    def _draw_squares(self, painter: QtGui.QPainter) -> None:
        """Draw the checkerboard pattern."""
        for row in range(self._board_size):
            for col in range(self._board_size):
                x, y = self._cell_to_pixel(row, col)

                if self._is_dark_square(row, col):
                    color = _CHECKERS_DARK_SQUARE
                else:
                    color = _CHECKERS_LIGHT_SQUARE

                painter.fillRect(x, y, self._cell_size, self._cell_size, color)

    def _draw_highlights(self, painter: QtGui.QPainter) -> None:
        """Draw cell highlights (selection, legal moves, moveable hints, hover)."""
        # Moveable pieces hint (pulsing border to show which pieces can move)
        # This is shown when user clicks a piece with no legal moves
        for cell in self._moveable_cells:
            row, col = cell
            x, y = self._cell_to_pixel(row, col)
            # Draw pulsing blue border around moveable pieces
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 150, 255, 200), 4))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawRect(x + 2, y + 2, self._cell_size - 4, self._cell_size - 4)

        # Selected piece highlight
        if self._selected_cell is not None:
            row, col = self._selected_cell
            x, y = self._cell_to_pixel(row, col)
            painter.fillRect(x, y, self._cell_size, self._cell_size, _CHECKERS_SELECTED)

        # Legal move destinations (draw circles like Chess)
        for dest in self._legal_destinations:
            row, col = dest
            x, y = self._cell_to_pixel(row, col)
            center_x = x + self._cell_size // 2
            center_y = y + self._cell_size // 2

            # Check if there's a piece at destination (capture)
            if self._board[row][col] != 0:
                # Draw ring for capture
                painter.setPen(QtGui.QPen(_CHECKERS_LEGAL_MOVE, 4))
                painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                radius = self._cell_size // 2 - 4
            else:
                # Draw dot for empty square
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(_CHECKERS_LEGAL_MOVE)
                radius = self._cell_size // 6

            painter.drawEllipse(QtCore.QPoint(center_x, center_y), radius, radius)

        # Hover highlight (only on dark squares, not on selected)
        if self._hover_cell is not None and not self._is_game_over:
            row, col = self._hover_cell
            if self._is_dark_square(row, col) and self._hover_cell != self._selected_cell:
                x, y = self._cell_to_pixel(row, col)
                painter.fillRect(x, y, self._cell_size, self._cell_size, _CHECKERS_HOVER)

    def _draw_pieces(self, painter: QtGui.QPainter) -> None:
        """Draw checkers pieces on the board."""
        for row in range(self._board_size):
            for col in range(self._board_size):
                piece = self._board[row][col]
                if piece == 0:
                    continue

                x, y = self._cell_to_pixel(row, col)
                center_x = x + self._cell_size // 2
                center_y = y + self._cell_size // 2
                radius = self._cell_size // 2 - 5

                # Determine piece color
                if piece in (1, 2):  # Black pieces
                    piece_color = _CHECKERS_BLACK_PIECE
                    border_color = QtGui.QColor(100, 100, 100)
                else:  # White pieces (3, 4)
                    piece_color = _CHECKERS_WHITE_PIECE
                    border_color = QtGui.QColor(150, 150, 150)

                # Draw piece with border
                painter.setPen(QtGui.QPen(border_color, 2))
                painter.setBrush(piece_color)
                painter.drawEllipse(QtCore.QPoint(center_x, center_y), radius, radius)

                # Draw king crown for kings (2, 4)
                if piece in (2, 4):
                    crown_radius = radius // 3
                    painter.setPen(QtCore.Qt.PenStyle.NoPen)
                    painter.setBrush(_CHECKERS_KING_CROWN)
                    painter.drawEllipse(QtCore.QPoint(center_x, center_y), crown_radius, crown_radius)

                    # Draw "K" on king
                    font = QtGui.QFont("Arial", int(self._cell_size * 0.2), QtGui.QFont.Weight.Bold)
                    painter.setFont(font)
                    if piece == 2:  # Black king
                        painter.setPen(QtGui.QColor(255, 255, 255))
                    else:  # White king
                        painter.setPen(QtGui.QColor(0, 0, 0))
                    text_rect = QtCore.QRect(
                        center_x - crown_radius,
                        center_y - crown_radius,
                        crown_radius * 2,
                        crown_radius * 2,
                    )
                    painter.drawText(text_rect, QtCore.Qt.AlignmentFlag.AlignCenter, "K")

    def _draw_coordinates(self, painter: QtGui.QPainter) -> None:
        """Draw board coordinates (1-8 and a-h)."""
        font = QtGui.QFont("Arial", max(10, int(self._cell_size * 0.18)))
        painter.setFont(font)
        painter.setPen(QtGui.QColor(40, 40, 40))
        font_metrics = QtGui.QFontMetrics(font)

        for i in range(self._board_size):
            # Column labels (a-h)
            file_char = chr(ord("a") + i)
            char_width = font_metrics.horizontalAdvance(file_char)
            x = self._margin + i * self._cell_size + (self._cell_size - char_width) // 2

            painter.drawText(x, self._margin - 8, file_char)
            board_bottom = self._margin + self._board_size * self._cell_size
            painter.drawText(x, board_bottom + font_metrics.ascent() + 5, file_char)

            # Row labels (8-1, top to bottom)
            rank_char = str(self._board_size - i)
            char_width = font_metrics.horizontalAdvance(rank_char)
            y = self._margin + i * self._cell_size + (self._cell_size + font_metrics.ascent()) // 2

            painter.drawText(self._margin - char_width - 8, y, rank_char)
            board_right = self._margin + self._board_size * self._cell_size
            painter.drawText(board_right + 8, y, rank_char)

    def _draw_game_over(self, painter: QtGui.QPainter) -> None:
        """Draw game over overlay."""
        # Semi-transparent overlay
        overlay = QtGui.QColor(0, 0, 0, 100)
        board_size = self._board_size * self._cell_size
        painter.fillRect(
            self._margin, self._margin, board_size, board_size, overlay
        )

        # Winner text
        font = QtGui.QFont("Arial", int(self._cell_size * 0.4), QtGui.QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(255, 255, 255))

        if self._winner == "player_0":
            text = "Black Wins!"
        elif self._winner == "player_1":
            text = "White Wins!"
        elif self._winner == "draw":
            text = "Draw!"
        else:
            text = "Game Over"

        rect = QtCore.QRect(
            self._margin, self._margin, board_size, board_size
        )
        painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, text)


# =============================================================================
# Module exports
# =============================================================================

__all__ = ["BoardGameRendererStrategy"]
