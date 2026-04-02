"""Tests for the OpenSpiel Checkers adapter integration.

OpenSpiel is a collection of games from Google DeepMind for research in
reinforcement learning, search, and game theory. Shimmy provides PettingZoo-
compatible wrappers for OpenSpiel games.

Repository: https://github.com/google-deepmind/open_spiel
Shimmy: https://shimmy.farama.org/environments/open_spiel/
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pytest

from gym_gui.core.enums import ControlMode, EnvironmentFamily, GameId, RenderMode

# Check if OpenSpiel and Shimmy are available
try:
    import pyspiel
    from shimmy.openspiel_compatibility import OpenSpielCompatibilityV0
    OPENSPIEL_AVAILABLE = True
except ImportError:
    OPENSPIEL_AVAILABLE = False

# Conditionally import adapter classes
if TYPE_CHECKING:
    from gym_gui.core.adapters.base import AdapterContext
    from gym_gui.core.adapters.open_spiel import (
        OPENSPIEL_ADAPTERS,
        CheckersEnvironmentAdapter,
        CheckersRenderPayload,
    )
    from gym_gui.core.factories.adapters import create_adapter, get_adapter_cls
elif OPENSPIEL_AVAILABLE:
    from gym_gui.core.adapters.base import AdapterContext
    from gym_gui.core.adapters.open_spiel import (
        OPENSPIEL_ADAPTERS,
        CheckersEnvironmentAdapter,
        CheckersRenderPayload,
    )
    from gym_gui.core.factories.adapters import create_adapter, get_adapter_cls


# Helper function only available when package is installed
if OPENSPIEL_AVAILABLE:
    def _make_checkers_adapter() -> "CheckersEnvironmentAdapter":
        """Create a Checkers adapter for testing."""
        context = AdapterContext(settings=None, control_mode=ControlMode.HUMAN_ONLY)
        adapter = CheckersEnvironmentAdapter(context)
        adapter.load()
        return adapter


# Mark for skipping tests that require the package
requires_openspiel = pytest.mark.skipif(
    not OPENSPIEL_AVAILABLE,
    reason="OpenSpiel and Shimmy not installed"
)


class TestOpenSpielEnumsRegistered:
    """Test that enums are properly registered."""

    def test_environment_family_registered(self) -> None:
        """Test EnvironmentFamily.OPEN_SPIEL exists."""
        assert hasattr(EnvironmentFamily, "OPEN_SPIEL")
        assert EnvironmentFamily.OPEN_SPIEL.value == "open_spiel"

    def test_game_id_registered(self) -> None:
        """Test GameId.OPEN_SPIEL_CHECKERS exists."""
        assert hasattr(GameId, "OPEN_SPIEL_CHECKERS")
        assert GameId.OPEN_SPIEL_CHECKERS.value == "open_spiel/checkers"


@requires_openspiel
class TestOpenSpielAdaptersRegistry:
    """Test adapter registry integration."""

    def test_adapters_registry_populated(self) -> None:
        """Test that OPENSPIEL_ADAPTERS registry is properly populated."""
        assert GameId.OPEN_SPIEL_CHECKERS in OPENSPIEL_ADAPTERS

    def test_adapter_class_correct(self) -> None:
        """Test adapter class mapping is correct."""
        assert OPENSPIEL_ADAPTERS[GameId.OPEN_SPIEL_CHECKERS] == CheckersEnvironmentAdapter

    def test_factory_get_adapter_cls(self) -> None:
        """Test that factory can look up adapter class."""
        adapter_cls = get_adapter_cls(GameId.OPEN_SPIEL_CHECKERS)
        assert adapter_cls == CheckersEnvironmentAdapter


@requires_openspiel
class TestCheckersRenderPayload:
    """Test CheckersRenderPayload dataclass."""

    def test_payload_creation(self) -> None:
        """Test creating a render payload."""
        board = [[0] * 8 for _ in range(8)]
        board[0][1] = 1  # Black piece
        board[7][6] = 3  # White piece

        payload = CheckersRenderPayload(
            board=board,
            current_player="player_0",
            legal_moves=[1, 2, 3],
            last_move=5,
            is_game_over=False,
            winner=None,
            move_count=10,
        )

        assert payload.board[0][1] == 1
        assert payload.current_player == "player_0"
        assert payload.legal_moves == [1, 2, 3]
        assert payload.move_count == 10

    def test_payload_to_dict(self) -> None:
        """Test conversion to dictionary."""
        board = [[0] * 8 for _ in range(8)]
        payload = CheckersRenderPayload(
            board=board,
            current_player="player_1",
            legal_moves=[],
            is_game_over=True,
            winner="player_1",
            move_count=25,
        )

        data = payload.to_dict()
        assert data["game_type"] == "checkers"
        assert data["current_player"] == "player_1"
        assert data["is_game_over"] is True
        assert data["winner"] == "player_1"


@requires_openspiel
class TestCheckersEnvironmentAdapter:
    """Tests for CheckersEnvironmentAdapter."""

    def test_adapter_creation(self) -> None:
        """Test that CheckersEnvironmentAdapter can be created and loaded."""
        adapter = _make_checkers_adapter()
        try:
            assert adapter.id == GameId.OPEN_SPIEL_CHECKERS.value
            assert adapter.default_render_mode == RenderMode.RGB_ARRAY
            assert ControlMode.HUMAN_ONLY in adapter.supported_control_modes
        finally:
            adapter.close()

    def test_adapter_reset(self) -> None:
        """Test environment reset returns valid observation."""
        adapter = _make_checkers_adapter()
        try:
            step = adapter.reset()
            assert step.observation is not None
            assert step.reward == 0.0
            assert step.terminated is False
            assert step.truncated is False
            assert step.agent_id in ("player_0", "player_1")
        finally:
            adapter.close()

    def test_adapter_get_legal_moves(self) -> None:
        """Test getting legal moves after reset."""
        adapter = _make_checkers_adapter()
        try:
            adapter.reset()
            legal_moves = adapter.get_legal_moves()
            assert isinstance(legal_moves, list)
            assert len(legal_moves) > 0  # Should have legal moves at start
        finally:
            adapter.close()

    def test_adapter_step(self) -> None:
        """Test environment step with a legal action."""
        adapter = _make_checkers_adapter()
        try:
            adapter.reset()
            legal_moves = adapter.get_legal_moves()

            if legal_moves:
                action = legal_moves[0]
                step = adapter.step(action)
                assert step.observation is not None
                assert isinstance(step.reward, (int, float))
                assert isinstance(step.terminated, bool)
                assert isinstance(step.truncated, bool)
        finally:
            adapter.close()

    def test_adapter_multiple_steps(self) -> None:
        """Test multiple environment steps."""
        adapter = _make_checkers_adapter()
        try:
            adapter.reset()

            for _ in range(5):
                legal_moves = adapter.get_legal_moves()
                if not legal_moves:
                    break
                action = legal_moves[0]
                step = adapter.step(action)
                if step.terminated or step.truncated:
                    break

            # Should have completed steps without error
            assert True
        finally:
            adapter.close()

    def test_adapter_current_agent(self) -> None:
        """Test current agent tracking."""
        adapter = _make_checkers_adapter()
        try:
            adapter.reset()
            initial_agent = adapter.current_agent()
            assert initial_agent in ("player_0", "player_1")

            # After a step, the agent should change
            legal_moves = adapter.get_legal_moves()
            if legal_moves:
                adapter.step(legal_moves[0])
                new_agent = adapter.current_agent()
                assert new_agent in ("player_0", "player_1")
                # In checkers, players alternate (unless captures allow multiple jumps)
        finally:
            adapter.close()

    def test_adapter_render(self) -> None:
        """Test environment rendering."""
        adapter = _make_checkers_adapter()
        try:
            adapter.reset()
            frame = adapter.render()

            # Should return an RGB image or None
            if frame is not None:
                assert isinstance(frame, np.ndarray)
                assert len(frame.shape) == 3  # Height x Width x Channels
                assert frame.shape[2] == 3  # RGB
        finally:
            adapter.close()

    def test_adapter_get_checkers_state(self) -> None:
        """Test getting structured checkers state."""
        adapter = _make_checkers_adapter()
        try:
            adapter.reset()
            state = adapter.get_checkers_state()

            assert isinstance(state, CheckersRenderPayload)
            assert len(state.board) == 8
            assert len(state.board[0]) == 8
            assert state.current_player in ("player_0", "player_1")
            assert isinstance(state.legal_moves, list)
        finally:
            adapter.close()

    def test_is_move_legal(self) -> None:
        """Test move legality checking."""
        adapter = _make_checkers_adapter()
        try:
            adapter.reset()
            legal_moves = adapter.get_legal_moves()

            if legal_moves:
                # First legal move should be legal
                assert adapter.is_move_legal(legal_moves[0]) is True

            # Invalid move (negative) should be illegal
            assert adapter.is_move_legal(-1) is False
        finally:
            adapter.close()


@requires_openspiel
class TestFactoryIntegration:
    """Test integration with adapter factory."""

    def test_create_adapter(self) -> None:
        """Test creating adapter through factory."""
        context = AdapterContext(settings=None, control_mode=ControlMode.HUMAN_ONLY)

        adapter = create_adapter(GameId.OPEN_SPIEL_CHECKERS, context)

        assert adapter is not None
        assert isinstance(adapter, CheckersEnvironmentAdapter)

    def test_create_adapter_no_context(self) -> None:
        """Test creating adapter through factory without context."""
        adapter = create_adapter(GameId.OPEN_SPIEL_CHECKERS, None)

        assert adapter is not None
        assert isinstance(adapter, CheckersEnvironmentAdapter)
