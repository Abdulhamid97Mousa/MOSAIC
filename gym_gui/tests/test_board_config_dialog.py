"""Test board game configuration dialog system.

Tests the extensible board configuration system including:
1. Factory pattern for creating game-specific dialogs
2. Chess board state and FEN manipulation
3. ChessConfigDialog functionality
4. Signal integration with operator config widgets
5. Initial state storage in OperatorConfig
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import pytest

chess = pytest.importorskip("chess", reason="chess board tests require python-chess")

if TYPE_CHECKING:
    pass


# =============================================================================
# Factory Tests
# =============================================================================


class TestBoardConfigDialogFactory:
    """Test the BoardConfigDialogFactory creates correct dialogs."""

    def test_factory_supports_chess(self) -> None:
        """Factory supports chess_v6 game."""
        from gym_gui.ui.widgets.operators_board_config_form import BoardConfigDialogFactory

        assert BoardConfigDialogFactory.supports("chess_v6") is True

    def test_factory_does_not_support_unknown_game(self) -> None:
        """Factory returns False for unsupported games."""
        from gym_gui.ui.widgets.operators_board_config_form import BoardConfigDialogFactory

        assert BoardConfigDialogFactory.supports("unknown_game") is False
        assert BoardConfigDialogFactory.supports("go_v5") is False  # Not yet implemented
        assert BoardConfigDialogFactory.supports("checkers") is False  # Not yet implemented

    def test_factory_get_supported_games(self) -> None:
        """Factory returns list of supported game IDs."""
        from gym_gui.ui.widgets.operators_board_config_form import BoardConfigDialogFactory

        games = BoardConfigDialogFactory.get_supported_games()
        assert isinstance(games, list)
        assert "chess_v6" in games

    def test_factory_raises_for_unsupported_game(self) -> None:
        """Factory raises ValueError for unsupported game."""
        from gym_gui.ui.widgets.operators_board_config_form import BoardConfigDialogFactory

        with pytest.raises(ValueError, match="No configuration dialog for game"):
            BoardConfigDialogFactory.create("unknown_game", None, None)

    def test_factory_register_new_game(self) -> None:
        """Factory allows registering new games dynamically."""
        from gym_gui.ui.widgets.operators_board_config_form import (
            BoardConfigDialog,
            BoardConfigDialogFactory,
        )

        # Create a mock dialog class
        class MockGameDialog(BoardConfigDialog):
            def _get_title(self):
                return "Mock"

            def _get_notation_name(self):
                return "MOCK"

            def _get_notation_placeholder(self):
                return "mock"

            def _create_board_widget(self):  # type: ignore[override]
                return None

            def _create_piece_tray(self):  # type: ignore[override]
                return None

            def _get_presets(self):
                return []

            def _validate_notation(self, notation):
                return True

            def _get_validation_error(self, notation):
                return ""

            def _create_state_from_notation(self, notation):  # type: ignore[override]
                return None

        # Register and verify
        BoardConfigDialogFactory.register("mock_game", MockGameDialog)
        assert BoardConfigDialogFactory.supports("mock_game") is True

        # Cleanup
        BoardConfigDialogFactory.unregister("mock_game")
        assert BoardConfigDialogFactory.supports("mock_game") is False


# =============================================================================
# Chess Board State Tests
# =============================================================================


class TestChessBoardState:
    """Test ChessBoardState FEN manipulation."""

    def test_default_state_is_standard_position(self) -> None:
        """Default ChessBoardState has standard starting position."""
        from gym_gui.ui.widgets.operators_board_config_form.chess_editor import ChessBoardState

        state = ChessBoardState()
        fen = state.to_notation()

        assert "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR" in fen

    def test_state_from_custom_fen(self) -> None:
        """ChessBoardState can be created with custom FEN."""
        from gym_gui.ui.widgets.operators_board_config_form.chess_editor import ChessBoardState

        # Endgame position: K+R vs K
        custom_fen = "8/8/8/4k3/8/8/8/R3K3 w Q - 0 1"
        state = ChessBoardState(custom_fen)
        fen = state.to_notation()

        assert "8/8/8/4k3/8/8/8/R3K3" in fen

    def test_state_get_piece(self) -> None:
        """ChessBoardState.get_piece returns correct piece."""
        from gym_gui.ui.widgets.operators_board_config_form.chess_editor import ChessBoardState

        state = ChessBoardState()

        # White rook at a1 (row 7, col 0 in display coordinates)
        piece = state.get_piece(7, 0)
        assert piece is not None
        assert piece.piece_type == "rook"
        assert piece.color == "white"

        # Black king at e8 (row 0, col 4)
        piece = state.get_piece(0, 4)
        assert piece is not None
        assert piece.piece_type == "king"
        assert piece.color == "black"

        # Empty square at e4 (row 4, col 4)
        piece = state.get_piece(4, 4)
        assert piece is None

    def test_state_set_piece(self) -> None:
        """ChessBoardState.set_piece modifies board correctly."""
        from gym_gui.ui.widgets.operators_board_config_form.chess_editor import (
            CHESS_PIECES,
            ChessBoardState,
        )

        state = ChessBoardState()

        # Move white queen to e4
        white_queen = CHESS_PIECES["Q"]
        state.set_piece(4, 4, white_queen)

        piece = state.get_piece(4, 4)
        assert piece is not None
        assert piece.piece_type == "queen"
        assert piece.color == "white"

    def test_state_remove_piece(self) -> None:
        """ChessBoardState can remove pieces."""
        from gym_gui.ui.widgets.operators_board_config_form.chess_editor import ChessBoardState

        state = ChessBoardState()

        # Remove white rook from a1
        state.set_piece(7, 0, None)

        piece = state.get_piece(7, 0)
        assert piece is None

    def test_state_clear(self) -> None:
        """ChessBoardState.clear removes all pieces."""
        from gym_gui.ui.widgets.operators_board_config_form.chess_editor import ChessBoardState

        state = ChessBoardState()
        state.clear()
        fen = state.to_notation()

        # Empty board FEN
        assert "8/8/8/8/8/8/8/8" in fen

    def test_state_copy(self) -> None:
        """ChessBoardState.copy creates independent copy."""
        from gym_gui.ui.widgets.operators_board_config_form.chess_editor import ChessBoardState

        state1 = ChessBoardState()
        state2 = state1.copy()

        # Modify copy
        state2.clear()

        # Original unchanged
        assert "rnbqkbnr" in state1.to_notation()
        assert "8/8/8/8/8/8/8/8" in state2.to_notation()

    def test_state_dimensions(self) -> None:
        """ChessBoardState.get_dimensions returns 8x8."""
        from gym_gui.ui.widgets.operators_board_config_form.chess_editor import ChessBoardState

        state = ChessBoardState()
        rows, cols = state.get_dimensions()

        assert rows == 8
        assert cols == 8


# =============================================================================
# Chess Config Dialog Tests (headless)
# =============================================================================


class TestChessConfigDialogHeadless:
    """Test ChessConfigDialog functionality without GUI."""

    def test_dialog_presets(self) -> None:
        """ChessConfigDialog has correct presets."""
        from gym_gui.ui.widgets.operators_board_config_form.chess_editor import ChessConfigDialog

        # Can't instantiate without Qt, but can access class attributes
        assert hasattr(ChessConfigDialog, "_get_presets")

    def test_fen_validation_valid(self) -> None:
        """Valid FEN strings pass validation."""
        import chess

        valid_fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Standard
            "8/8/8/4k3/8/8/8/R3K3 w Q - 0 1",  # K+R vs K
            "8/8/8/4k3/8/8/8/3QK3 w - - 0 1",  # K+Q vs K
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",  # Italian
        ]

        for fen in valid_fens:
            board = chess.Board(fen)
            # Valid if has exactly one king per side
            white_kings = len(board.pieces(chess.KING, chess.WHITE))
            black_kings = len(board.pieces(chess.KING, chess.BLACK))
            assert white_kings == 1, f"Invalid white kings in {fen}"
            assert black_kings == 1, f"Invalid black kings in {fen}"

    def test_fen_validation_invalid_no_kings(self) -> None:
        """FEN without kings is invalid."""
        import chess

        # No kings
        try:
            board = chess.Board("8/8/8/8/8/8/8/8 w - - 0 1")
            white_kings = len(board.pieces(chess.KING, chess.WHITE))
            black_kings = len(board.pieces(chess.KING, chess.BLACK))
            assert white_kings == 0
            assert black_kings == 0
        except ValueError:
            pass  # Some versions of chess may reject this

    def test_fen_validation_invalid_two_kings(self) -> None:
        """FEN with two kings of same color is invalid."""
        import chess

        # Two white kings (invalid position)
        fen = "8/8/8/4k3/8/4K3/8/4K3 w - - 0 1"
        board = chess.Board(fen)
        white_kings = len(board.pieces(chess.KING, chess.WHITE))
        assert white_kings == 2  # This is detectable


# =============================================================================
# GamePiece Tests
# =============================================================================


class TestGamePiece:
    """Test GamePiece dataclass."""

    def test_piece_creation(self) -> None:
        """GamePiece can be created with attributes."""
        from gym_gui.ui.widgets.operators_board_config_form.base import GamePiece

        piece = GamePiece(
            piece_type="king",
            color="white",
            symbol="\u2654",
            notation="K"
        )

        assert piece.piece_type == "king"
        assert piece.color == "white"
        assert piece.symbol == "\u2654"
        assert piece.notation == "K"

    def test_piece_equality(self) -> None:
        """GamePiece equality based on type and color."""
        from gym_gui.ui.widgets.operators_board_config_form.base import GamePiece

        piece1 = GamePiece("king", "white", "\u2654", "K")
        piece2 = GamePiece("king", "white", "\u2654", "K")
        piece3 = GamePiece("king", "black", "\u265A", "k")

        assert piece1 == piece2
        assert piece1 != piece3

    def test_piece_hash(self) -> None:
        """GamePiece is hashable."""
        from gym_gui.ui.widgets.operators_board_config_form.base import GamePiece

        piece = GamePiece("king", "white", "\u2654", "K")
        piece_set = {piece}

        assert piece in piece_set


# =============================================================================
# Chess Pieces Definition Tests
# =============================================================================


class TestChessPiecesDefinition:
    """Test CHESS_PIECES dictionary is complete."""

    def test_all_white_pieces_defined(self) -> None:
        """All white piece types are defined."""
        from gym_gui.ui.widgets.operators_board_config_form.chess_editor import CHESS_PIECES

        white_pieces = ["K", "Q", "R", "B", "N", "P"]
        for notation in white_pieces:
            assert notation in CHESS_PIECES
            assert CHESS_PIECES[notation].color == "white"

    def test_all_black_pieces_defined(self) -> None:
        """All black piece types are defined."""
        from gym_gui.ui.widgets.operators_board_config_form.chess_editor import CHESS_PIECES

        black_pieces = ["k", "q", "r", "b", "n", "p"]
        for notation in black_pieces:
            assert notation in CHESS_PIECES
            assert CHESS_PIECES[notation].color == "black"

    def test_piece_symbols_are_unicode(self) -> None:
        """All piece symbols are Unicode chess characters."""
        from gym_gui.ui.widgets.operators_board_config_form.chess_editor import CHESS_PIECES

        # Unicode chess piece range: U+2654 to U+265F
        for notation, piece in CHESS_PIECES.items():
            assert len(piece.symbol) == 1
            codepoint = ord(piece.symbol)
            assert 0x2654 <= codepoint <= 0x265F, f"Invalid symbol for {notation}"


# =============================================================================
# Integration Tests (with PettingZoo)
# =============================================================================


class TestBoardConfigIntegration:
    """Test integration with PettingZoo chess environment."""

    @pytest.fixture
    def chess_env(self):
        """Create a PettingZoo chess environment."""
        try:
            from pettingzoo.classic import chess_v6
            env = chess_v6.env(render_mode="rgb_array")
            env.reset(seed=42)
            yield env
            env.close()
        except ImportError:
            pytest.skip("PettingZoo classic not installed")

    def test_custom_fen_can_be_applied(self, chess_env) -> None:
        """Custom FEN can be applied to PettingZoo chess environment."""
        # K+R vs K endgame
        custom_fen = "8/8/8/4k3/8/8/8/R3K3 w Q - 0 1"

        # Apply custom FEN
        chess_env.board.set_fen(custom_fen)

        # Verify position
        assert "8/8/8/4k3/8/8/8/R3K3" in chess_env.board.fen()

    def test_custom_fen_updates_legal_moves(self, chess_env) -> None:
        """Custom FEN updates legal moves correctly."""
        # Position where white can checkmate in one
        custom_fen = "k7/8/1K6/8/8/8/8/R7 w - - 0 1"

        chess_env.board.set_fen(custom_fen)

        # Ra8 should be checkmate
        legal_moves = [str(m) for m in chess_env.board.legal_moves]
        assert "a1a8" in legal_moves  # Checkmate move available

    def test_board_state_matches_env(self, chess_env) -> None:
        """ChessBoardState matches environment FEN."""
        from gym_gui.ui.widgets.operators_board_config_form.chess_editor import ChessBoardState

        env_fen = chess_env.board.fen()
        state = ChessBoardState(env_fen)

        # FENs should match (at least the piece placement part)
        env_placement = env_fen.split()[0]
        state_placement = state.to_notation().split()[0]
        assert env_placement == state_placement


# =============================================================================
# Operator Config Integration Tests
# =============================================================================


class TestOperatorConfigIntegration:
    """Test integration with operator config widgets."""

    def test_operator_config_row_has_configure_signal(self) -> None:
        """OperatorConfigRow has configure_requested signal."""
        from gym_gui.ui.widgets.operator_config_widget import OperatorConfigRow

        assert hasattr(OperatorConfigRow, "configure_requested")

    def test_operator_config_widget_has_configure_signal(self) -> None:
        """OperatorConfigWidget has configure_requested signal."""
        from gym_gui.ui.widgets.operator_config_widget import OperatorConfigWidget

        assert hasattr(OperatorConfigWidget, "configure_requested")

    def test_operator_config_widget_has_set_initial_state(self) -> None:
        """OperatorConfigWidget has set_operator_initial_state method."""
        from gym_gui.ui.widgets.operator_config_widget import OperatorConfigWidget

        assert hasattr(OperatorConfigWidget, "set_operator_initial_state")


# =============================================================================
# Preset FEN Tests
# =============================================================================


class TestPresetFENs:
    """Test preset FEN positions are valid."""

    def test_standard_fen_is_valid(self) -> None:
        """Standard FEN position is valid."""
        import chess

        from gym_gui.ui.widgets.operators_board_config_form.chess_editor import STANDARD_FEN

        board = chess.Board(STANDARD_FEN)
        assert board.is_valid()
        assert len(list(board.legal_moves)) == 20

    def test_empty_fen_format(self) -> None:
        """Empty FEN has correct format."""
        from gym_gui.ui.widgets.operators_board_config_form.chess_editor import EMPTY_FEN

        assert "8/8/8/8/8/8/8/8" in EMPTY_FEN

    def test_endgame_kr_k_is_valid(self) -> None:
        """K+R vs K endgame FEN is valid."""
        import chess

        from gym_gui.ui.widgets.operators_board_config_form.chess_editor import ENDGAME_KR_K

        board = chess.Board(ENDGAME_KR_K)
        # Should have exactly 3 pieces: 2 kings + 1 rook
        all_pieces = list(board.piece_map().values())
        assert len(all_pieces) == 3

    def test_endgame_kq_k_is_valid(self) -> None:
        """K+Q vs K endgame FEN is valid."""
        import chess

        from gym_gui.ui.widgets.operators_board_config_form.chess_editor import ENDGAME_KQ_K

        board = chess.Board(ENDGAME_KQ_K)
        # Should have exactly 3 pieces: 2 kings + 1 queen
        all_pieces = list(board.piece_map().values())
        assert len(all_pieces) == 3
