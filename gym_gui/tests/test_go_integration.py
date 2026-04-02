"""Integration tests for Go PettingZoo Classic adapter.

Tests cover:
- Adapter loading and initialization
- Render payload structure
- Move mechanics (place stone, pass)
- Game state tracking
- Factory registration
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

pettingzoo = pytest.importorskip("pettingzoo", reason="Go integration requires pettingzoo[classic]")

if TYPE_CHECKING:
    from gym_gui.core.adapters.pettingzoo_classic import GoEnvironmentAdapter

from gym_gui.core.enums import EnvironmentFamily, GameId
from gym_gui.ui.handlers import GoHandler


def _payload(step: Any) -> dict:
    """Extract render_payload from a step, asserting it exists."""
    p = step.render_payload
    assert p is not None, "render_payload should not be None"
    return p


class TestGoAdapterLoads:
    """Test Go adapter initialization."""

    @pytest.fixture
    def go_adapter(self) -> "GoEnvironmentAdapter":
        """Create a fresh Go adapter for each test."""
        from gym_gui.core.adapters.pettingzoo_classic import GoEnvironmentAdapter

        adapter = GoEnvironmentAdapter()
        adapter.load()
        return adapter

    def test_adapter_loads(self, go_adapter: "GoEnvironmentAdapter") -> None:
        """Adapter loads without error."""
        assert go_adapter._aec_env is not None

    def test_adapter_reset(self, go_adapter: "GoEnvironmentAdapter") -> None:
        """Reset returns a valid step result."""
        result = go_adapter.reset(seed=42)

        assert result.observation is not None
        assert result.terminated is False
        assert result.truncated is False

    def test_adapter_has_id(self, go_adapter: "GoEnvironmentAdapter") -> None:
        """Adapter has correct game ID."""
        assert go_adapter.id == "go_v5"


class TestGoRenderPayload:
    """Test Go render payload structure."""

    @pytest.fixture
    def go_adapter(self) -> "GoEnvironmentAdapter":
        """Create and reset adapter for each test."""
        from gym_gui.core.adapters.pettingzoo_classic import GoEnvironmentAdapter

        adapter = GoEnvironmentAdapter()
        adapter.load()
        adapter.reset(seed=42)
        return adapter

    def test_render_payload_has_board(self, go_adapter: "GoEnvironmentAdapter") -> None:
        """Render payload includes board state."""
        step = go_adapter.reset()
        payload = _payload(step)
        assert "board" in payload
        board = payload["board"]
        # Default board is 19x19
        assert len(board) == 19
        assert all(len(row) == 19 for row in board)

    def test_render_payload_has_legal_moves(self, go_adapter: "GoEnvironmentAdapter") -> None:
        """Render payload includes legal moves list."""
        step = go_adapter.reset()
        payload = _payload(step)
        assert "legal_moves" in payload
        assert isinstance(payload["legal_moves"], list)
        # Should have legal moves at start (board positions + pass)
        assert len(payload["legal_moves"]) > 0

    def test_render_payload_has_current_player(self, go_adapter: "GoEnvironmentAdapter") -> None:
        """Render payload includes current player."""
        step = go_adapter.reset()
        payload = _payload(step)
        assert "current_player" in payload
        # Black plays first
        assert payload["current_player"] in ("black_0", "white_0")

    def test_render_payload_has_game_state(self, go_adapter: "GoEnvironmentAdapter") -> None:
        """Render payload includes game state info."""
        step = go_adapter.reset()
        payload = _payload(step)
        assert "is_game_over" in payload
        assert "move_count" in payload
        assert payload["is_game_over"] is False
        assert payload["move_count"] == 0

    def test_render_payload_has_game_type(self, go_adapter: "GoEnvironmentAdapter") -> None:
        """Render payload includes game_type for routing."""
        step = go_adapter.reset()
        payload = _payload(step)
        assert payload.get("game_type") == "go"


class TestGoMoves:
    """Test Go move mechanics."""

    @pytest.fixture
    def go_adapter(self) -> "GoEnvironmentAdapter":
        """Create and reset adapter for each test."""
        from gym_gui.core.adapters.pettingzoo_classic import GoEnvironmentAdapter

        adapter = GoEnvironmentAdapter()
        adapter.load()
        adapter.reset(seed=42)
        return adapter

    def test_valid_move(self, go_adapter: "GoEnvironmentAdapter") -> None:
        """Valid stone placement succeeds."""
        # Place stone at center (9, 9) on 19x19 board
        action = go_adapter.coords_to_action(9, 9)
        step = go_adapter.step(action)
        assert step is not None
        # Board should show a stone at center (non-zero value)
        # After step, observation is from white's perspective, so black=2
        board = _payload(step)["board"]
        assert board[9][9] != 0  # Stone placed

    def test_illegal_move_raises(self, go_adapter: "GoEnvironmentAdapter") -> None:
        """Illegal move raises exception."""
        # Place stone at center
        action = go_adapter.coords_to_action(9, 9)
        go_adapter.step(action)
        # White's turn, try to place on same spot
        with pytest.raises(ValueError):
            go_adapter.step(action)

    def test_is_move_legal_valid(self, go_adapter: "GoEnvironmentAdapter") -> None:
        """is_move_legal returns True for valid moves."""
        action = go_adapter.coords_to_action(0, 0)
        assert go_adapter.is_move_legal(action) is True

    def test_is_move_legal_invalid(self, go_adapter: "GoEnvironmentAdapter") -> None:
        """is_move_legal returns False for invalid moves."""
        # Place stone
        action = go_adapter.coords_to_action(0, 0)
        go_adapter.step(action)
        # Now white's turn - same spot is illegal
        assert go_adapter.is_move_legal(action) is False

    def test_pass_action(self, go_adapter: "GoEnvironmentAdapter") -> None:
        """Pass action works correctly."""
        pass_action = go_adapter.get_pass_action()
        assert pass_action == 19 * 19  # N*N for 19x19 board
        step = go_adapter.step(pass_action)
        assert step is not None
        # After pass, it's white's turn
        assert _payload(step)["current_player"] == "white_0"


class TestGoGameState:
    """Test Go game state tracking."""

    @pytest.fixture
    def go_adapter(self) -> "GoEnvironmentAdapter":
        """Create and reset adapter for each test."""
        from gym_gui.core.adapters.pettingzoo_classic import GoEnvironmentAdapter

        adapter = GoEnvironmentAdapter()
        adapter.load()
        adapter.reset(seed=42)
        return adapter

    def test_move_count_increments(self, go_adapter: "GoEnvironmentAdapter") -> None:
        """Move count increments after each move."""
        step = go_adapter.reset()
        assert _payload(step)["move_count"] == 0

        action = go_adapter.coords_to_action(0, 0)
        step = go_adapter.step(action)
        assert _payload(step)["move_count"] == 1

    def test_current_agent_alternates(self, go_adapter: "GoEnvironmentAdapter") -> None:
        """Current agent alternates between black and white."""
        step = go_adapter.reset()
        assert _payload(step)["current_player"] == "black_0"

        action = go_adapter.coords_to_action(0, 0)
        step = go_adapter.step(action)
        assert _payload(step)["current_player"] == "white_0"

        action = go_adapter.coords_to_action(1, 0)
        step = go_adapter.step(action)
        assert _payload(step)["current_player"] == "black_0"

    def test_get_legal_moves(self, go_adapter: "GoEnvironmentAdapter") -> None:
        """get_legal_moves returns valid move indices."""
        legal_moves = go_adapter.get_legal_moves()
        assert isinstance(legal_moves, list)
        # At start, most positions should be legal (except pass already included)
        assert len(legal_moves) > 300  # 19x19 - 1 = 360 + pass = 361

    def test_get_go_state(self, go_adapter: "GoEnvironmentAdapter") -> None:
        """get_go_state returns GoRenderPayload dataclass."""
        state = go_adapter.get_go_state()
        assert hasattr(state, "board")
        assert hasattr(state, "current_player")
        assert hasattr(state, "is_game_over")
        assert hasattr(state, "move_count")


class TestGoHandlerIntegration:
    """Test Go handler utility functions."""

    def test_handler_is_go_payload(self) -> None:
        """GoHandler.is_go_payload correctly identifies Go payloads."""
        valid_payload = {"game_type": "go", "board": [[0] * 19 for _ in range(19)]}
        invalid_payload = {"game_type": "chess", "fen": "..."}
        empty_payload: dict = {}

        assert GoHandler.is_go_payload(valid_payload) is True
        assert GoHandler.is_go_payload(invalid_payload) is False
        assert GoHandler.is_go_payload(empty_payload) is False


class TestGoAdapterClose:
    """Test Go adapter cleanup."""

    def test_close_cleans_up(self) -> None:
        """Close properly cleans up resources."""
        from gym_gui.core.adapters.pettingzoo_classic import GoEnvironmentAdapter

        adapter = GoEnvironmentAdapter()
        adapter.load()
        adapter.reset(seed=42)
        adapter.close()
        # Should be safe to call
        assert adapter._aec_env is None

    def test_close_multiple_times_safe(self) -> None:
        """Multiple close calls are safe."""
        from gym_gui.core.adapters.pettingzoo_classic import GoEnvironmentAdapter

        adapter = GoEnvironmentAdapter()
        adapter.load()
        adapter.reset(seed=42)
        adapter.close()
        adapter.close()  # Should not raise


class TestGoFactoryRegistration:
    """Test Go game factory registration."""

    def test_go_in_factory_registry(self) -> None:
        """Go is registered in adapter factory."""
        from gym_gui.core.factories.adapters import PETTINGZOO_CLASSIC_ADAPTERS

        assert "go_v5" in PETTINGZOO_CLASSIC_ADAPTERS

    def test_go_game_id_exists(self) -> None:
        """GameId.GO exists in enums."""
        assert hasattr(GameId, "GO")
        assert GameId.GO.value == "go_v5"

    def test_go_in_pettingzoo_classic_family(self) -> None:
        """Go is in PettingZoo Classic family."""
        from gym_gui.core.enums import ENVIRONMENT_FAMILY_BY_GAME

        assert ENVIRONMENT_FAMILY_BY_GAME.get(GameId.GO) == EnvironmentFamily.PETTINGZOO_CLASSIC


class TestGoBoardSizes:
    """Test different Go board sizes."""

    def test_9x9_board(self) -> None:
        """9x9 board initializes correctly."""
        from gym_gui.core.adapters.pettingzoo_classic import GoEnvironmentAdapter

        adapter = GoEnvironmentAdapter(board_size=9)
        adapter.load()
        step = adapter.reset()
        board = _payload(step)["board"]
        assert len(board) == 9
        assert all(len(row) == 9 for row in board)
        adapter.close()

    def test_13x13_board(self) -> None:
        """13x13 board initializes correctly."""
        from gym_gui.core.adapters.pettingzoo_classic import GoEnvironmentAdapter

        adapter = GoEnvironmentAdapter(board_size=13)
        adapter.load()
        step = adapter.reset()
        board = _payload(step)["board"]
        assert len(board) == 13
        assert all(len(row) == 13 for row in board)
        adapter.close()

    def test_19x19_board(self) -> None:
        """19x19 board initializes correctly (default)."""
        from gym_gui.core.adapters.pettingzoo_classic import GoEnvironmentAdapter

        adapter = GoEnvironmentAdapter()
        adapter.load()
        step = adapter.reset()
        board = _payload(step)["board"]
        assert len(board) == 19
        assert all(len(row) == 19 for row in board)
        adapter.close()


class TestGoTwoConsecutivePasses:
    """Test game ending with two consecutive passes."""

    def test_two_passes_ends_game(self) -> None:
        """Two consecutive passes end the game."""
        from gym_gui.core.adapters.pettingzoo_classic import GoEnvironmentAdapter

        adapter = GoEnvironmentAdapter(board_size=9)  # Use smaller board for speed
        adapter.load()
        adapter.reset(seed=42)

        # Black passes
        pass_action = adapter.get_pass_action()
        adapter.step(pass_action)

        # White passes
        step = adapter.step(pass_action)

        # Game should be over after two consecutive passes
        assert _payload(step)["is_game_over"] is True
        adapter.close()
