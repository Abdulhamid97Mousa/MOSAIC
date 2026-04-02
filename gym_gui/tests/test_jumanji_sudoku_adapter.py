"""Regression tests for Jumanji Sudoku adapter integration.

Tests the JumanjiSudokuAdapter which provides an interactive Sudoku puzzle
environment from the Jumanji JAX-based RL library.

Key functionality tested:
- Adapter creation and loading
- Reset and step with _last_obs tracking
- Render payload structure (sudoku key with board, action_mask, fixed_cells)
- Board game detection for routing to interactive renderer
- Action encoding/decoding helpers
- Documentation mapping
"""

from __future__ import annotations

import os

# Force CPU backend for tests to avoid GPU memory issues
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")

import logging
from typing import Any

import numpy as np
import pytest

# Skip entire module if jumanji is not installed
jumanji = pytest.importorskip("jumanji")

from gym_gui.config.game_configs import JumanjiConfig
from gym_gui.core.adapters.base import AdapterContext
from gym_gui.core.adapters.jumanji import (
    JUMANJI_ADAPTERS,
    JumanjiSudokuAdapter,
)
from gym_gui.core.enums import ControlMode, GameId, RenderMode


def _make_sudoku_adapter(**overrides: Any) -> JumanjiSudokuAdapter:
    """Create a Sudoku adapter with the given configuration."""
    config = JumanjiConfig(env_id="Sudoku-v0", **overrides)
    context = AdapterContext(settings=None, control_mode=ControlMode.HUMAN_ONLY)
    adapter = JumanjiSudokuAdapter(context, config=config)
    adapter.load()
    return adapter


class TestJumanjiSudokuAdapterBasics:
    """Basic adapter functionality tests."""

    def test_sudoku_adapter_creation(self) -> None:
        """Test that JumanjiSudokuAdapter can be created and loaded."""
        adapter = _make_sudoku_adapter()
        try:
            assert adapter._env_id == "Sudoku-v0"
            assert adapter.default_render_mode == RenderMode.RGB_ARRAY
        finally:
            adapter.close()

    def test_sudoku_adapter_in_registry(self) -> None:
        """Test that Sudoku adapter is registered in JUMANJI_ADAPTERS."""
        assert GameId.JUMANJI_SUDOKU in JUMANJI_ADAPTERS
        assert JUMANJI_ADAPTERS[GameId.JUMANJI_SUDOKU] == JumanjiSudokuAdapter

    def test_sudoku_supported_control_modes(self) -> None:
        """Test that Sudoku supports expected control modes."""
        assert ControlMode.HUMAN_ONLY in JumanjiSudokuAdapter.supported_control_modes
        assert ControlMode.AGENT_ONLY in JumanjiSudokuAdapter.supported_control_modes
        assert ControlMode.HYBRID_TURN_BASED in JumanjiSudokuAdapter.supported_control_modes


class TestJumanjiSudokuAdapterReset:
    """Tests for reset functionality and _last_obs tracking."""

    def test_sudoku_reset_basic(self) -> None:
        """Test basic reset functionality."""
        adapter = _make_sudoku_adapter()
        try:
            step = adapter.reset(seed=42)
            assert step.observation is not None
            assert isinstance(step.observation, np.ndarray)
        finally:
            adapter.close()

    def test_sudoku_reset_stores_last_obs(self) -> None:
        """Test that reset stores raw observation in _last_obs."""
        adapter = _make_sudoku_adapter()
        try:
            _ = adapter.reset(seed=42)
            # _last_obs should be set after reset
            assert adapter._last_obs is not None
            assert isinstance(adapter._last_obs, dict)
            assert "board" in adapter._last_obs
            assert "action_mask" in adapter._last_obs
        finally:
            adapter.close()

    def test_sudoku_reset_stores_initial_board(self) -> None:
        """Test that reset captures initial puzzle state for fixed cells."""
        adapter = _make_sudoku_adapter()
        try:
            _ = adapter.reset(seed=42)
            # _initial_board should be captured
            assert adapter._initial_board is not None
            assert isinstance(adapter._initial_board, np.ndarray)
            assert adapter._initial_board.shape == (9, 9)
            # Some cells should be filled (initial clues)
            assert np.sum(adapter._initial_board != 0) > 0
        finally:
            adapter.close()

    def test_sudoku_reset_with_seed_reproducibility(self) -> None:
        """Test that reset with same seed produces same puzzle."""
        adapter = _make_sudoku_adapter()
        try:
            adapter.reset(seed=12345)
            assert adapter._initial_board is not None
            first_board = adapter._initial_board.copy()

            adapter.reset(seed=12345)
            assert adapter._initial_board is not None
            second_board = adapter._initial_board.copy()

            np.testing.assert_array_equal(first_board, second_board)
        finally:
            adapter.close()


class TestJumanjiSudokuAdapterStep:
    """Tests for step functionality.

    Note: Some step tests are marked xfail due to JAX/Jumanji gymnasium wrapper
    issues with action type conversion. The wrapper expects numpy arrays but
    standard gym convention is Python ints.
    """

    @pytest.mark.xfail(reason="Jumanji gymnasium wrapper has action type issues")
    def test_sudoku_step_basic(self) -> None:
        """Test basic step functionality."""
        adapter = _make_sudoku_adapter()
        try:
            _ = adapter.reset(seed=42)

            # Find a valid action from action_mask
            assert adapter._last_obs is not None
            action_mask = adapter._last_obs.get("action_mask", [])
            valid_actions = np.where(np.array(action_mask).flatten())[0]

            if len(valid_actions) > 0:
                action = int(valid_actions[0])
                step = adapter.step(action)
                assert step.observation is not None
                assert isinstance(step.reward, float)
                assert isinstance(step.terminated, bool)
                assert isinstance(step.truncated, bool)
        finally:
            adapter.close()

    @pytest.mark.xfail(reason="Jumanji gymnasium wrapper has action type issues")
    def test_sudoku_step_updates_last_obs(self) -> None:
        """Test that step updates _last_obs."""
        adapter = _make_sudoku_adapter()
        try:
            _ = adapter.reset(seed=42)
            assert adapter._last_obs is not None
            initial_obs = adapter._last_obs

            # Find a valid action
            action_mask = adapter._last_obs.get("action_mask", [])
            valid_actions = np.where(np.array(action_mask).flatten())[0]

            if len(valid_actions) > 0:
                action = int(valid_actions[0])
                _ = adapter.step(action)

                # _last_obs should be updated
                assert adapter._last_obs is not None
                # Board should have changed (one more cell filled)
                new_board = np.array(adapter._last_obs["board"])
                old_board = np.array(initial_obs["board"])
                # Jumanji uses -1 for empty cells
                assert np.sum(new_board != -1) >= np.sum(old_board != -1)
        finally:
            adapter.close()


class TestJumanjiSudokuRenderPayload:
    """Tests for render payload structure - critical for board game routing."""

    def test_sudoku_render_returns_dict(self) -> None:
        """Test that render returns expected dict structure."""
        adapter = _make_sudoku_adapter()
        try:
            _ = adapter.reset(seed=42)
            render_payload = adapter.render()
            assert isinstance(render_payload, dict)
        finally:
            adapter.close()

    def test_sudoku_render_has_sudoku_key(self) -> None:
        """Test that render payload contains 'sudoku' key for board game detection."""
        adapter = _make_sudoku_adapter()
        try:
            _ = adapter.reset(seed=42)
            render_payload = adapter.render()
            assert "sudoku" in render_payload, "Payload must have 'sudoku' key for board game routing"
        finally:
            adapter.close()

    def test_sudoku_render_sudoku_data_structure(self) -> None:
        """Test that sudoku data contains required fields."""
        adapter = _make_sudoku_adapter()
        try:
            _ = adapter.reset(seed=42)
            render_payload = adapter.render()
            sudoku_data = render_payload.get("sudoku", {})

            # Required fields for interactive renderer
            assert "game_type" in sudoku_data
            assert sudoku_data["game_type"] == "sudoku"
            assert "board" in sudoku_data
            assert "action_mask" in sudoku_data
            assert "fixed_cells" in sudoku_data
        finally:
            adapter.close()

    def test_sudoku_render_board_structure(self) -> None:
        """Test that board data has correct structure."""
        adapter = _make_sudoku_adapter()
        try:
            _ = adapter.reset(seed=42)
            render_payload = adapter.render()
            board = render_payload["sudoku"]["board"]

            # Board should be 9x9
            assert len(board) == 9
            for row in board:
                assert len(row) == 9

            # Jumanji uses: -1 = empty, 0-8 = digits 1-9 (zero-indexed)
            for row in board:
                for cell in row:
                    assert -1 <= cell <= 8, f"Cell value {cell} out of range [-1, 8]"
        finally:
            adapter.close()

    def test_sudoku_render_action_mask_structure(self) -> None:
        """Test that action_mask has correct size (9*9*9 = 729)."""
        adapter = _make_sudoku_adapter()
        try:
            _ = adapter.reset(seed=42)
            render_payload = adapter.render()
            action_mask = render_payload["sudoku"]["action_mask"]

            # 9 rows * 9 cols * 9 digits = 729 actions
            assert len(action_mask) == 729

            # All values should be bool-like (0 or 1)
            for val in action_mask:
                assert val in (0, 1, True, False)
        finally:
            adapter.close()

    def test_sudoku_render_fixed_cells_structure(self) -> None:
        """Test that fixed_cells contains initial puzzle clues."""
        adapter = _make_sudoku_adapter()
        try:
            _ = adapter.reset(seed=42)
            render_payload = adapter.render()
            fixed_cells = render_payload["sudoku"]["fixed_cells"]

            # Should have some fixed cells (puzzle clues)
            assert len(fixed_cells) > 0

            # Each fixed cell should be (row, col) tuple
            for cell in fixed_cells:
                assert len(cell) == 2
                row, col = cell
                assert 0 <= row < 9
                assert 0 <= col < 9
        finally:
            adapter.close()

    def test_sudoku_render_has_rgb_frame(self) -> None:
        """Test that render also includes RGB frame from Jumanji viewer."""
        adapter = _make_sudoku_adapter()
        try:
            _ = adapter.reset(seed=42)
            render_payload = adapter.render()

            assert "mode" in render_payload
            assert render_payload["mode"] == "rgb_array"
            assert "rgb" in render_payload

            rgb = render_payload["rgb"]
            assert isinstance(rgb, np.ndarray)
            assert len(rgb.shape) == 3
            assert rgb.shape[2] == 3  # RGB channels
        finally:
            adapter.close()

    def test_sudoku_step_render_payload(self) -> None:
        """Test that step includes render_payload with sudoku data."""
        adapter = _make_sudoku_adapter()
        try:
            step = adapter.reset(seed=42)
            render_payload = step.render_payload

            assert isinstance(render_payload, dict)
            assert "sudoku" in render_payload
            assert render_payload["sudoku"]["game_type"] == "sudoku"
        finally:
            adapter.close()


class TestJumanjiSudokuBoardGameDetection:
    """Tests for board game detection - ensures Sudoku routes to Grid tab."""

    def test_sudoku_detected_by_get_game_from_payload(self) -> None:
        """Test that BoardGameRendererStrategy detects Sudoku payload."""
        from gym_gui.rendering.strategies.board_game import BoardGameRendererStrategy

        adapter = _make_sudoku_adapter()
        try:
            step = adapter.reset(seed=42)
            payload = step.render_payload
            assert payload is not None

            detected = BoardGameRendererStrategy.get_game_from_payload(payload)
            assert detected == GameId.JUMANJI_SUDOKU
        finally:
            adapter.close()

    def test_sudoku_detected_by_sudoku_key(self) -> None:
        """Test detection via 'sudoku' key in payload."""
        from gym_gui.rendering.strategies.board_game import BoardGameRendererStrategy

        # Minimal payload with just sudoku key
        payload = {"sudoku": {"game_type": "sudoku"}}
        detected = BoardGameRendererStrategy.get_game_from_payload(payload)
        assert detected == GameId.JUMANJI_SUDOKU

    def test_sudoku_detected_by_game_type(self) -> None:
        """Test detection via game_type value."""
        from gym_gui.rendering.strategies.board_game import BoardGameRendererStrategy

        # Payload with game_type but no sudoku key (fallback detection)
        payload = {"game_type": "sudoku"}
        detected = BoardGameRendererStrategy.get_game_from_payload(payload)
        assert detected == GameId.JUMANJI_SUDOKU

    def test_sudoku_in_supported_games(self) -> None:
        """Test that JUMANJI_SUDOKU is in SUPPORTED_GAMES."""
        from gym_gui.rendering.strategies.board_game import BoardGameRendererStrategy

        assert GameId.JUMANJI_SUDOKU in BoardGameRendererStrategy.SUPPORTED_GAMES


class TestJumanjiSudokuActionHelpers:
    """Tests for action encoding/decoding helper methods."""

    def test_compute_action_basic(self) -> None:
        """Test compute_action produces correct action index."""
        # Action = row * 81 + col * 9 + (digit - 1)
        # Row 0, Col 0, Digit 1 -> 0*81 + 0*9 + 0 = 0
        assert JumanjiSudokuAdapter.compute_action(0, 0, 1) == 0

        # Row 0, Col 0, Digit 9 -> 0*81 + 0*9 + 8 = 8
        assert JumanjiSudokuAdapter.compute_action(0, 0, 9) == 8

        # Row 0, Col 1, Digit 1 -> 0*81 + 1*9 + 0 = 9
        assert JumanjiSudokuAdapter.compute_action(0, 1, 1) == 9

        # Row 1, Col 0, Digit 1 -> 1*81 + 0*9 + 0 = 81
        assert JumanjiSudokuAdapter.compute_action(1, 0, 1) == 81

        # Row 8, Col 8, Digit 9 -> 8*81 + 8*9 + 8 = 648 + 72 + 8 = 728
        assert JumanjiSudokuAdapter.compute_action(8, 8, 9) == 728

    def test_decode_action_basic(self) -> None:
        """Test decode_action reverses compute_action."""
        # Action 0 -> Row 0, Col 0, Digit 1
        assert JumanjiSudokuAdapter.decode_action(0) == (0, 0, 1)

        # Action 8 -> Row 0, Col 0, Digit 9
        assert JumanjiSudokuAdapter.decode_action(8) == (0, 0, 9)

        # Action 9 -> Row 0, Col 1, Digit 1
        assert JumanjiSudokuAdapter.decode_action(9) == (0, 1, 1)

        # Action 81 -> Row 1, Col 0, Digit 1
        assert JumanjiSudokuAdapter.decode_action(81) == (1, 0, 1)

        # Action 728 -> Row 8, Col 8, Digit 9
        assert JumanjiSudokuAdapter.decode_action(728) == (8, 8, 9)

    def test_compute_decode_roundtrip(self) -> None:
        """Test that compute_action and decode_action are inverses."""
        for row in range(9):
            for col in range(9):
                for digit in range(1, 10):
                    action = JumanjiSudokuAdapter.compute_action(row, col, digit)
                    decoded = JumanjiSudokuAdapter.decode_action(action)
                    assert decoded == (row, col, digit), f"Roundtrip failed for ({row}, {col}, {digit})"

    def test_action_range(self) -> None:
        """Test that all valid (row, col, digit) produce actions in [0, 728]."""
        actions = set()
        for row in range(9):
            for col in range(9):
                for digit in range(1, 10):
                    action = JumanjiSudokuAdapter.compute_action(row, col, digit)
                    assert 0 <= action <= 728
                    actions.add(action)

        # Should have exactly 729 unique actions
        assert len(actions) == 729


class TestJumanjiSudokuDocumentation:
    """Tests for documentation mapping."""

    def test_sudoku_has_documentation(self) -> None:
        """Test that Sudoku has documentation in GAME_INFO."""
        from gym_gui.game_docs import GAME_INFO

        assert GameId.JUMANJI_SUDOKU in GAME_INFO
        doc = GAME_INFO[GameId.JUMANJI_SUDOKU]
        assert isinstance(doc, str)
        assert len(doc) > 0

    def test_sudoku_documentation_content(self) -> None:
        """Test that Sudoku documentation contains expected content."""
        from gym_gui.game_docs import GAME_INFO

        doc = GAME_INFO[GameId.JUMANJI_SUDOKU]
        # Should mention Sudoku
        assert "sudoku" in doc.lower() or "Sudoku" in doc
        # Should have HTML structure
        assert "<" in doc and ">" in doc


class TestJumanjiSudokuConfig:
    """Tests for JumanjiConfig with Sudoku."""

    def test_jumanji_config_sudoku_defaults(self) -> None:
        """Test JumanjiConfig default values for Sudoku."""
        from gym_gui.config.game_configs import DEFAULT_JUMANJI_SUDOKU_CONFIG

        assert DEFAULT_JUMANJI_SUDOKU_CONFIG.env_id == "jumanji/Sudoku-v0"

    def test_jumanji_config_custom_seed(self) -> None:
        """Test JumanjiConfig with custom seed."""
        config = JumanjiConfig(env_id="Sudoku-v0", seed=12345)
        assert config.seed == 12345


class TestJumanjiSudokuStepState:
    """Tests for step state and metrics."""

    def test_sudoku_step_state_metrics(self) -> None:
        """Test that step state contains Sudoku-specific metrics."""
        adapter = _make_sudoku_adapter()
        try:
            step = adapter.reset(seed=42)
            state = step.state

            assert state is not None
            assert state.metrics is not None

            # Should have Sudoku-specific metrics (from _last_obs)
            metrics = state.metrics
            assert "cells_filled" in metrics, f"Missing cells_filled. Got: {metrics}"
            assert "cells_remaining" in metrics, f"Missing cells_remaining. Got: {metrics}"

            # Sanity check values
            assert metrics["cells_filled"] >= 0
            assert metrics["cells_remaining"] >= 0
            assert metrics["cells_filled"] + metrics["cells_remaining"] == 81
        finally:
            adapter.close()

    def test_sudoku_step_state_environment(self) -> None:
        """Test that step state contains environment info."""
        adapter = _make_sudoku_adapter()
        try:
            step = adapter.reset(seed=42)
            state = step.state

            assert state.environment is not None
            # Should contain env_id
            assert "env_id" in state.environment
        finally:
            adapter.close()
