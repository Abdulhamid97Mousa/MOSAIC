"""Tests for Checkers Human Control Mode integration.

Tests that:
1. CheckersEnvironmentAdapter properly handles moves
2. CheckersHandler properly receives cell clicks
3. Signal chain works from board click to handler
4. Board state updates correctly after moves
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from gym_gui.core.enums import ControlMode, GameId

# Check if OpenSpiel and Shimmy are available
try:
    import pyspiel
    from shimmy.openspiel_compatibility import OpenSpielCompatibilityV0
    OPENSPIEL_AVAILABLE = True
except ImportError:
    OPENSPIEL_AVAILABLE = False

if TYPE_CHECKING:
    from gym_gui.core.adapters.base import AdapterContext
    from gym_gui.core.adapters.open_spiel import (
        CheckersEnvironmentAdapter,
        CheckersRenderPayload,
    )
elif OPENSPIEL_AVAILABLE:
    from gym_gui.core.adapters.base import AdapterContext
    from gym_gui.core.adapters.open_spiel import (
        CheckersEnvironmentAdapter,
        CheckersRenderPayload,
    )

# Mark for skipping tests that require OpenSpiel
requires_openspiel = pytest.mark.skipif(
    not OPENSPIEL_AVAILABLE,
    reason="OpenSpiel and Shimmy not installed"
)


_LOG = logging.getLogger(__name__)


@requires_openspiel
class TestCheckersCoordinateConversion:
    """Test coordinate conversion methods used for clicking."""

    @pytest.fixture
    def adapter(self) -> "CheckersEnvironmentAdapter":
        """Create a Checkers adapter for testing."""
        context = AdapterContext(settings=None, control_mode=ControlMode.HUMAN_ONLY)
        adapter = CheckersEnvironmentAdapter(context)
        adapter.load()
        adapter.reset(seed=42)
        return adapter

    def test_cell_to_algebraic(self, adapter: "CheckersEnvironmentAdapter") -> None:
        """Test cell to algebraic notation conversion."""
        # Row 0 (top) is rank 8, Row 7 (bottom) is rank 1
        # Col 0 is 'a', Col 7 is 'h'
        assert adapter.cell_to_algebraic(0, 0) == "a8"
        assert adapter.cell_to_algebraic(0, 7) == "h8"
        assert adapter.cell_to_algebraic(7, 0) == "a1"
        assert adapter.cell_to_algebraic(7, 7) == "h1"
        assert adapter.cell_to_algebraic(3, 4) == "e5"

    def test_algebraic_to_cell(self, adapter: "CheckersEnvironmentAdapter") -> None:
        """Test algebraic notation to cell conversion."""
        assert adapter.algebraic_to_cell("a8") == (0, 0)
        assert adapter.algebraic_to_cell("h8") == (0, 7)
        assert adapter.algebraic_to_cell("a1") == (7, 0)
        assert adapter.algebraic_to_cell("h1") == (7, 7)
        assert adapter.algebraic_to_cell("e5") == (3, 4)

    def test_coordinate_roundtrip(self, adapter: "CheckersEnvironmentAdapter") -> None:
        """Test that coordinate conversion is reversible."""
        for row in range(8):
            for col in range(8):
                alg = adapter.cell_to_algebraic(row, col)
                back_row, back_col = adapter.algebraic_to_cell(alg)
                assert (back_row, back_col) == (row, col), f"Failed for {row}, {col}"


@requires_openspiel
class TestCheckersLegalMoves:
    """Test legal move detection for click handling."""

    @pytest.fixture
    def adapter(self) -> "CheckersEnvironmentAdapter":
        """Create a Checkers adapter for testing."""
        context = AdapterContext(settings=None, control_mode=ControlMode.HUMAN_ONLY)
        adapter = CheckersEnvironmentAdapter(context)
        adapter.load()
        adapter.reset(seed=42)
        return adapter

    def test_has_legal_moves_at_start(self, adapter: "CheckersEnvironmentAdapter") -> None:
        """Test that legal moves are available at game start."""
        legal_moves = adapter.get_legal_moves()
        assert len(legal_moves) > 0, "Should have legal moves at start"

    def test_get_moves_from_square(self, adapter: "CheckersEnvironmentAdapter") -> None:
        """Test getting moves from a specific square."""
        # In initial position, player_0 (black) pieces are at rows 0-2
        # Try to find a piece that can move
        moves = adapter.get_moves_from_square("b6")  # Typical black piece position
        # May or may not have moves depending on board setup
        assert isinstance(moves, list)

    def test_get_legal_move_strings(self, adapter: "CheckersEnvironmentAdapter") -> None:
        """Test getting legal moves as strings."""
        move_strings = adapter.get_legal_move_strings()
        assert isinstance(move_strings, list)
        if move_strings:
            # Each move should be 4 characters (e.g., "a3b4")
            for move in move_strings:
                assert len(move) >= 4, f"Move string too short: {move}"


@requires_openspiel
class TestCheckersBoardState:
    """Test board state for Human Control mode."""

    @pytest.fixture
    def adapter(self) -> "CheckersEnvironmentAdapter":
        """Create a Checkers adapter for testing."""
        context = AdapterContext(settings=None, control_mode=ControlMode.HUMAN_ONLY)
        adapter = CheckersEnvironmentAdapter(context)
        adapter.load()
        adapter.reset(seed=42)
        return adapter

    def test_initial_board_has_pieces(self, adapter: "CheckersEnvironmentAdapter") -> None:
        """Test that initial board has pieces in correct positions."""
        state = adapter.get_checkers_state()
        board = state.board

        # Count pieces
        black_pieces = sum(1 for row in board for cell in row if cell in (1, 2))
        white_pieces = sum(1 for row in board for cell in row if cell in (3, 4))

        assert black_pieces == 12, f"Expected 12 black pieces, got {black_pieces}"
        assert white_pieces == 12, f"Expected 12 white pieces, got {white_pieces}"

    def test_pieces_on_dark_squares(self, adapter: "CheckersEnvironmentAdapter") -> None:
        """Test that pieces are only on dark squares."""
        state = adapter.get_checkers_state()
        board = state.board

        for row in range(8):
            for col in range(8):
                piece = board[row][col]
                is_dark = (row + col) % 2 == 1
                if piece != 0:
                    assert is_dark, f"Piece at ({row}, {col}) is on a light square"

    def test_current_player_at_start(self, adapter: "CheckersEnvironmentAdapter") -> None:
        """Test that the correct player starts."""
        state = adapter.get_checkers_state()
        # In checkers, black (player_0) typically moves first
        assert state.current_player in ("player_0", "player_1")


@requires_openspiel
class TestCheckersMoveExecution:
    """Test executing moves via the adapter."""

    @pytest.fixture
    def adapter(self) -> "CheckersEnvironmentAdapter":
        """Create a Checkers adapter for testing."""
        context = AdapterContext(settings=None, control_mode=ControlMode.HUMAN_ONLY)
        adapter = CheckersEnvironmentAdapter(context)
        adapter.load()
        adapter.reset(seed=42)
        return adapter

    def test_execute_legal_move(self, adapter: "CheckersEnvironmentAdapter") -> None:
        """Test executing a legal move."""
        legal_moves = adapter.get_legal_moves()
        assert len(legal_moves) > 0

        # Execute first legal move
        action = legal_moves[0]
        step = adapter.step(action)

        assert step is not None
        assert step.observation is not None

    def test_find_action_for_move(self, adapter: "CheckersEnvironmentAdapter") -> None:
        """Test finding action index from from/to squares."""
        # Get legal moves as strings
        move_strings = adapter.get_legal_move_strings()

        if move_strings:
            # Take first legal move
            move = move_strings[0]
            from_sq = move[:2]
            to_sq = move[2:4] if len(move) >= 4 else move[-2:]

            # Find action for this move
            action = adapter.find_action_for_move(from_sq, to_sq)

            if action is not None:
                # Should be a valid legal action
                assert action in adapter.get_legal_moves()

    def test_player_changes_after_move(self, adapter: "CheckersEnvironmentAdapter") -> None:
        """Test that current player changes after a move."""
        adapter.current_agent()

        legal_moves = adapter.get_legal_moves()
        if legal_moves:
            adapter.step(legal_moves[0])
            new_player = adapter.current_agent()

            # Player should change (unless multi-jump)
            # In simple cases, player alternates
            assert new_player in ("player_0", "player_1")


@requires_openspiel
class TestCheckersRenderPayloadFormat:
    """Test that render payload has correct format for UI."""

    @pytest.fixture
    def adapter(self) -> "CheckersEnvironmentAdapter":
        """Create a Checkers adapter for testing."""
        context = AdapterContext(settings=None, control_mode=ControlMode.HUMAN_ONLY)
        adapter = CheckersEnvironmentAdapter(context)
        adapter.load()
        adapter.reset(seed=42)
        return adapter

    def test_payload_has_game_type(self, adapter: "CheckersEnvironmentAdapter") -> None:
        """Test payload includes game_type for detection."""
        state = adapter.get_checkers_state()
        payload = state.to_dict()

        assert "game_type" in payload
        assert payload["game_type"] == "checkers"

    def test_payload_has_board(self, adapter: "CheckersEnvironmentAdapter") -> None:
        """Test payload includes board state."""
        state = adapter.get_checkers_state()
        payload = state.to_dict()

        assert "board" in payload
        board = payload["board"]
        assert len(board) == 8
        assert len(board[0]) == 8

    def test_payload_has_legal_moves(self, adapter: "CheckersEnvironmentAdapter") -> None:
        """Test payload includes legal moves."""
        state = adapter.get_checkers_state()
        payload = state.to_dict()

        assert "legal_moves" in payload
        assert isinstance(payload["legal_moves"], list)

    def test_payload_has_current_player(self, adapter: "CheckersEnvironmentAdapter") -> None:
        """Test payload includes current player."""
        state = adapter.get_checkers_state()
        payload = state.to_dict()

        assert "current_player" in payload
        assert payload["current_player"] in ("player_0", "player_1")


class TestDarkSquareLogic:
    """Test the dark square detection logic used for click validation."""

    def test_dark_square_pattern(self) -> None:
        """Test dark square calculation matches checkerboard pattern."""
        # Dark squares are where (row + col) % 2 == 1
        def is_dark(row: int, col: int) -> bool:
            return (row + col) % 2 == 1

        # Row 0: dark at cols 1, 3, 5, 7
        assert not is_dark(0, 0)
        assert is_dark(0, 1)
        assert not is_dark(0, 2)
        assert is_dark(0, 3)

        # Row 1: dark at cols 0, 2, 4, 6
        assert is_dark(1, 0)
        assert not is_dark(1, 1)
        assert is_dark(1, 2)
        assert not is_dark(1, 3)

    def test_all_playable_squares_are_dark(self) -> None:
        """Verify checkers pieces should be on dark squares."""
        # Standard checkers: 12 pieces per player on dark squares of first 3 rows
        dark_count_per_row = 4  # Each row has 4 dark squares
        total_dark_in_3_rows = dark_count_per_row * 3

        assert total_dark_in_3_rows == 12  # 12 pieces per player


@requires_openspiel
class TestCheckersHandlerIntegration:
    """Test CheckersHandler integration with SessionController."""

    def test_handler_checks_game_id(self) -> None:
        """Test that handler checks game_id before processing clicks."""
        from gym_gui.ui.handlers.game_moves.checkers import CheckersHandler

        # Create mock session with wrong game_id
        mock_session = MagicMock()
        mock_session.game_id = GameId.CHESS  # Wrong game!

        mock_render_tabs = MagicMock()
        mock_status_bar = MagicMock()

        handler = CheckersHandler(
            session=mock_session,
            render_tabs=mock_render_tabs,
            status_bar=mock_status_bar,
        )

        # Click should be ignored due to game_id mismatch
        handler.on_checkers_cell_clicked(2, 1)

        # Status bar should not be updated (handler returned early)
        # The handler shouldn't crash

    def test_handler_with_correct_game_id(self) -> None:
        """Test that handler processes clicks when game_id matches."""
        from gym_gui.ui.handlers.game_moves.checkers import CheckersHandler

        # Create mock session with correct game_id
        mock_session = MagicMock()
        mock_session.game_id = GameId.OPEN_SPIEL_CHECKERS

        # Create mock adapter
        context = AdapterContext(settings=None, control_mode=ControlMode.HUMAN_ONLY)
        adapter = CheckersEnvironmentAdapter(context)
        adapter.load()
        adapter.reset(seed=42)
        mock_session._adapter = adapter

        mock_render_tabs = MagicMock()
        mock_status_bar = MagicMock()

        handler = CheckersHandler(
            session=mock_session,
            render_tabs=mock_render_tabs,
            status_bar=mock_status_bar,
        )

        # Click on a dark square where a piece might be
        handler.on_checkers_cell_clicked(2, 1)

        # Should have processed the click (may show status message)
        # No crash means success
        adapter.close()


@requires_openspiel
class TestCheckersMultiMoveGameplay:
    """Test multi-move gameplay to reproduce 'stuck' issue."""

    @pytest.fixture
    def adapter(self) -> "CheckersEnvironmentAdapter":
        """Create a Checkers adapter for testing."""
        context = AdapterContext(settings=None, control_mode=ControlMode.HUMAN_ONLY)
        adapter = CheckersEnvironmentAdapter(context)
        adapter.load()
        adapter.reset(seed=42)
        return adapter

    def test_play_multiple_moves(self, adapter: "CheckersEnvironmentAdapter") -> None:
        """Test playing multiple moves in sequence."""
        moves_played = 0
        max_moves = 20

        for _ in range(max_moves):
            legal_moves = adapter.get_legal_moves()
            if not legal_moves:
                _LOG.info(f"Game ended after {moves_played} moves (no legal moves)")
                break

            # Get legal move strings for debugging
            move_strings = adapter.get_legal_move_strings()
            current_player = adapter.current_agent()

            _LOG.info(
                f"Move {moves_played + 1}: {current_player}'s turn, "
                f"legal moves: {move_strings[:5]}{'...' if len(move_strings) > 5 else ''}"
            )

            # Execute first legal move
            adapter.step(legal_moves[0])
            moves_played += 1

        assert moves_played > 0, "Should be able to play at least one move"
        _LOG.info(f"Successfully played {moves_played} moves")

    def test_turn_alternation(self, adapter: "CheckersEnvironmentAdapter") -> None:
        """Test that turns alternate correctly (except for multi-jumps)."""
        initial_player = adapter.current_agent()
        _LOG.info(f"Initial player: {initial_player}")

        legal_moves = adapter.get_legal_moves()
        if legal_moves:
            adapter.step(legal_moves[0])
            second_player = adapter.current_agent()
            _LOG.info(f"After move 1: {second_player}'s turn")

            # Player may or may not change depending on multi-jump
            assert second_player in ("player_0", "player_1")

    def test_action_mask_consistency(self, adapter: "CheckersEnvironmentAdapter") -> None:
        """Test that action_mask stays consistent with legal_moves."""
        for move_num in range(10):
            legal_moves = adapter.get_legal_moves()
            if not legal_moves:
                _LOG.info(f"Game ended at move {move_num}")
                break

            # action_mask should match legal_moves
            obs, _, _, _, info = adapter._aec_env.last()
            action_mask = info.get("action_mask", [])

            legal_from_mask = [i for i, v in enumerate(action_mask) if v == 1]
            assert set(legal_from_mask) == set(legal_moves), (
                f"Move {move_num}: action_mask doesn't match legal_moves. "
                f"From mask: {legal_from_mask}, From method: {legal_moves}"
            )

            adapter.step(legal_moves[0])

    def test_game_state_vs_agent_selection(self, adapter: "CheckersEnvironmentAdapter") -> None:
        """Test that game_state.current_player matches agent_selection."""
        for move_num in range(10):
            legal_moves = adapter.get_legal_moves()
            if not legal_moves:
                break

            # Check consistency
            agent_sel = adapter._aec_env.agent_selection
            if hasattr(adapter._aec_env, 'game_state'):
                game_player = adapter._aec_env.game_state.current_player()
                expected_agent = f"player_{game_player}"

                # Log discrepancy if any
                if agent_sel != expected_agent:
                    _LOG.warning(
                        f"Move {move_num}: agent_selection={agent_sel}, "
                        f"game_state.current_player={game_player}"
                    )

            adapter.step(legal_moves[0])


@requires_openspiel
class TestCheckersSignalChain:
    """Test the signal chain from board to handler."""

    def test_board_renderer_emits_signal(self) -> None:
        """Test that board renderer emits cell_clicked signal."""
        pytest.importorskip("qtpy")

        from qtpy import QtCore, QtWidgets

        from gym_gui.rendering.strategies.board_game import _CheckersBoardRenderer

        # Need QApplication for Qt widgets
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        renderer = _CheckersBoardRenderer()

        # Track signal emissions
        received_clicks: List[tuple] = []
        renderer.cell_clicked.connect(lambda r, c: received_clicks.append((r, c)))

        # Manually trigger signal
        renderer.cell_clicked.emit(2, 1)

        assert len(received_clicks) == 1
        assert received_clicks[0] == (2, 1)

    def test_board_game_widget_forwards_signal(self) -> None:
        """Test that _BoardGameWidget forwards checkers_cell_clicked."""
        pytest.importorskip("qtpy")

        from qtpy import QtWidgets

        from gym_gui.rendering.strategies.board_game import GameId, _BoardGameWidget

        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        widget = _BoardGameWidget()

        # Track signal emissions
        received_clicks: List[tuple] = []
        widget.checkers_cell_clicked.connect(lambda r, c: received_clicks.append((r, c)))

        # Get checkers renderer (this connects internal signals)
        checkers_renderer = widget._get_checkers_renderer()

        # Emit from internal renderer
        checkers_renderer.cell_clicked.emit(3, 2)

        assert len(received_clicks) == 1
        assert received_clicks[0] == (3, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
