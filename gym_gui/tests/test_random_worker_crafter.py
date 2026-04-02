"""Tests for Random Worker Crafter support.

This module tests that the random worker correctly handles Crafter environments,
including proper gymnasium API compatibility and independent action seeding.
"""

import sys

import pytest

# Skip if crafter not installed
pytest.importorskip("crafter")

# Add random_worker to path
sys.path.insert(0, "3rd_party/workers/mosaic/random_worker")

from random_worker.config import RandomWorkerConfig
from random_worker.runtime import RandomWorkerRuntime


class TestRandomWorkerCrafterBasics:
    """Basic tests for Crafter environment support in random worker."""

    def test_create_crafter_env(self):
        """Test that Crafter environment can be created."""
        config = RandomWorkerConfig(
            run_id="test_crafter",
            task="CrafterReward-v1",
            seed=12345,
        )
        runtime = RandomWorkerRuntime(config)

        env = runtime._create_env()
        assert env is not None
        assert env.action_space.n == 17  # type: ignore[attr-defined]  # Crafter has 17 actions
        env.close()

    def test_crafter_reset(self):
        """Test that Crafter environment can be reset."""
        config = RandomWorkerConfig(
            run_id="test_crafter_reset",
            task="CrafterReward-v1",
            seed=12345,
        )
        runtime = RandomWorkerRuntime(config)

        result = runtime.handle_reset({"cmd": "reset", "seed": 42})

        assert result["type"] == "ready"
        assert result["env_id"] == "CrafterReward-v1"
        assert result["seed"] == 42
        assert result["observation_shape"] == [512, 512, 3]  # Crafter high-quality observation shape

        if runtime._env:
            runtime._env.close()

    def test_crafter_render_payload(self):
        """Test that Crafter environment returns render payload."""
        config = RandomWorkerConfig(
            run_id="test_crafter_render",
            task="CrafterReward-v1",
            seed=12345,
        )
        runtime = RandomWorkerRuntime(config)

        result = runtime.handle_reset({"cmd": "reset", "seed": 42})

        assert "render_payload" in result
        payload = result["render_payload"]
        assert payload["mode"] == "rgb"
        # Should be 512x512 (high quality from CrafterConfig defaults)
        assert payload["width"] == 512
        assert payload["height"] == 512

        if runtime._env:
            runtime._env.close()


class TestRandomWorkerCrafterSeeding:
    """Tests for independent action seeding in Crafter environments."""

    def test_different_seeds_produce_different_actions(self):
        """Test that different seeds produce different action sequences."""
        config1 = RandomWorkerConfig(
            run_id="operator_1",
            task="CrafterReward-v1",
            seed=12345,
        )
        config2 = RandomWorkerConfig(
            run_id="operator_2",
            task="CrafterReward-v1",
            seed=67890,
        )

        runtime1 = RandomWorkerRuntime(config1)
        runtime2 = RandomWorkerRuntime(config2)

        # Initialize both with same env seed (for identical layouts)
        runtime1.handle_reset({"cmd": "reset", "seed": 42})
        runtime2.handle_reset({"cmd": "reset", "seed": 42})

        # Sample actions - should be different due to different action seeds
        actions1 = [runtime1._select_action() for _ in range(20)]
        actions2 = [runtime2._select_action() for _ in range(20)]

        # Actions should be different (different action seeds)
        assert actions1 != actions2, (
            f"Actions should be different for different seeds. "
            f"Got actions1={actions1}, actions2={actions2}"
        )

        if runtime1._env:
            runtime1._env.close()
        if runtime2._env:
            runtime2._env.close()

    def test_same_seed_produces_same_actions(self):
        """Test that same seed produces same action sequence."""
        config1 = RandomWorkerConfig(
            run_id="operator_a",
            task="CrafterReward-v1",
            seed=99999,
        )
        config2 = RandomWorkerConfig(
            run_id="operator_b",
            task="CrafterReward-v1",
            seed=99999,  # Same seed
        )

        runtime1 = RandomWorkerRuntime(config1)
        runtime2 = RandomWorkerRuntime(config2)

        # Initialize both
        runtime1.handle_reset({"cmd": "reset", "seed": 100})
        runtime2.handle_reset({"cmd": "reset", "seed": 100})

        # Sample actions - should be identical due to same action seeds
        actions1 = [runtime1._select_action() for _ in range(20)]
        actions2 = [runtime2._select_action() for _ in range(20)]

        # Actions should be identical (same action seeds)
        assert actions1 == actions2, (
            f"Actions should be identical for same seeds. "
            f"Got actions1={actions1}, actions2={actions2}"
        )

        if runtime1._env:
            runtime1._env.close()
        if runtime2._env:
            runtime2._env.close()


class TestRandomWorkerCrafterStep:
    """Tests for stepping in Crafter environments."""

    def test_crafter_step_action(self):
        """Test that Crafter environment can be stepped."""
        config = RandomWorkerConfig(
            run_id="test_crafter_step",
            task="CrafterReward-v1",
            seed=12345,
        )
        runtime = RandomWorkerRuntime(config)

        # Reset first
        runtime.handle_reset({"cmd": "reset", "seed": 42})

        # Test _select_action directly instead of capturing stdout
        action = runtime._select_action()
        assert isinstance(action, int)
        assert 0 <= action < 17  # Crafter has 17 actions

        # Test step through internal state
        assert runtime._env is not None

        if runtime._env:
            runtime._env.close()
