"""Tests for the PyBullet Drones adapter integration.

gym-pybullet-drones is a PyBullet-based Gymnasium environment for single and
multi-agent reinforcement learning of quadcopter control.

Paper: Panerati, J., et al. (2021). Learning to Fly - a Gym Environment with
       PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

import numpy as np
import pytest

from gym_gui.core.enums import ControlMode, EnvironmentFamily, GameId, RenderMode

# Check if gym-pybullet-drones is available
try:
    import gym_pybullet_drones
    PYBULLET_DRONES_AVAILABLE = True
except ImportError:
    PYBULLET_DRONES_AVAILABLE = False

# Conditionally import adapter classes
if TYPE_CHECKING:
    from gym_gui.core.adapters.base import AdapterContext
    from gym_gui.core.adapters.pybullet_drones import (
        PYBULLET_DRONES_ADAPTERS,
        CtrlAviaryAdapter,
        HoverAviaryAdapter,
        MultiHoverAviaryAdapter,
        PyBulletDronesAdapter,
        PyBulletDronesConfig,
        VelocityAviaryAdapter,
    )
    from gym_gui.core.factories.adapters import create_adapter, get_adapter_cls
elif PYBULLET_DRONES_AVAILABLE:
    from gym_gui.core.adapters.base import AdapterContext
    from gym_gui.core.adapters.pybullet_drones import (
        PYBULLET_DRONES_ADAPTERS,
        CtrlAviaryAdapter,
        HoverAviaryAdapter,
        MultiHoverAviaryAdapter,
        PyBulletDronesAdapter,
        PyBulletDronesConfig,
        VelocityAviaryAdapter,
    )
    from gym_gui.core.factories.adapters import create_adapter, get_adapter_cls


# Helper function only available when package is installed
if PYBULLET_DRONES_AVAILABLE:
    def _make_adapter(
        game_id: GameId = GameId.PYBULLET_HOVER_AVIARY,
        **overrides: Any,
    ) -> PyBulletDronesAdapter:
        """Create a PyBullet Drones adapter with the given configuration."""
        config = PyBulletDronesConfig(env_id=game_id.value, gui=False, **overrides)
        context = AdapterContext(settings=None, control_mode=ControlMode.AGENT_ONLY)
        adapter_class = PYBULLET_DRONES_ADAPTERS.get(game_id, PyBulletDronesAdapter)
        adapter = adapter_class(context, config=config)
        adapter.load()
        return adapter


# Mark for skipping tests that require the package
requires_pybullet_drones = pytest.mark.skipif(
    not PYBULLET_DRONES_AVAILABLE,
    reason="gym-pybullet-drones not installed"
)


class TestPyBulletDronesEnumsRegistered:
    """Test that enums are properly registered."""

    def test_environment_family_registered(self) -> None:
        """Test EnvironmentFamily.PYBULLET_DRONES exists."""
        assert hasattr(EnvironmentFamily, "PYBULLET_DRONES")
        assert EnvironmentFamily.PYBULLET_DRONES.value == "pybullet_drones"

    def test_game_ids_registered(self) -> None:
        """Test all GameId entries exist."""
        assert hasattr(GameId, "PYBULLET_HOVER_AVIARY")
        assert hasattr(GameId, "PYBULLET_MULTIHOVER_AVIARY")
        assert hasattr(GameId, "PYBULLET_CTRL_AVIARY")
        assert hasattr(GameId, "PYBULLET_VELOCITY_AVIARY")

        assert GameId.PYBULLET_HOVER_AVIARY.value == "hover-aviary-v0"
        assert GameId.PYBULLET_MULTIHOVER_AVIARY.value == "multihover-aviary-v0"
        assert GameId.PYBULLET_CTRL_AVIARY.value == "ctrl-aviary-v0"
        assert GameId.PYBULLET_VELOCITY_AVIARY.value == "velocity-aviary-v0"


@requires_pybullet_drones
class TestPyBulletDronesAdaptersRegistry:
    """Test adapter registry integration."""

    def test_adapters_registry_populated(self) -> None:
        """Test that PYBULLET_DRONES_ADAPTERS registry is properly populated."""
        assert GameId.PYBULLET_HOVER_AVIARY in PYBULLET_DRONES_ADAPTERS
        assert GameId.PYBULLET_MULTIHOVER_AVIARY in PYBULLET_DRONES_ADAPTERS
        assert GameId.PYBULLET_CTRL_AVIARY in PYBULLET_DRONES_ADAPTERS
        assert GameId.PYBULLET_VELOCITY_AVIARY in PYBULLET_DRONES_ADAPTERS

    def test_adapter_classes_correct(self) -> None:
        """Test adapter class mappings are correct."""
        assert PYBULLET_DRONES_ADAPTERS[GameId.PYBULLET_HOVER_AVIARY] == HoverAviaryAdapter
        assert PYBULLET_DRONES_ADAPTERS[GameId.PYBULLET_MULTIHOVER_AVIARY] == MultiHoverAviaryAdapter
        assert PYBULLET_DRONES_ADAPTERS[GameId.PYBULLET_CTRL_AVIARY] == CtrlAviaryAdapter
        assert PYBULLET_DRONES_ADAPTERS[GameId.PYBULLET_VELOCITY_AVIARY] == VelocityAviaryAdapter

    def test_factory_get_adapter_cls(self) -> None:
        """Test that factory can look up adapter classes."""
        adapter_cls = get_adapter_cls(GameId.PYBULLET_HOVER_AVIARY)
        assert adapter_cls == HoverAviaryAdapter

        adapter_cls = get_adapter_cls(GameId.PYBULLET_MULTIHOVER_AVIARY)
        assert adapter_cls == MultiHoverAviaryAdapter


@requires_pybullet_drones
class TestPyBulletDronesConfig:
    """Test PyBulletDronesConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = PyBulletDronesConfig()
        assert config.env_id == "hover-aviary-v0"
        assert config.drone_model == "cf2x"
        assert config.num_drones == 1
        assert config.physics == "pyb"
        assert config.pyb_freq == 240
        assert config.ctrl_freq == 30
        assert config.gui is False
        assert config.obs_type == "kin"
        assert config.act_type == "rpm"

    def test_config_to_gym_kwargs(self) -> None:
        """Test config conversion to gym kwargs."""
        config = PyBulletDronesConfig(gui=True, pyb_freq=120, ctrl_freq=15)
        kwargs = config.to_gym_kwargs()

        assert kwargs["gui"] is True
        assert kwargs["pyb_freq"] == 120
        assert kwargs["ctrl_freq"] == 15


@requires_pybullet_drones
class TestHoverAviaryAdapter:
    """Tests for HoverAviaryAdapter (single-agent hover task)."""

    def test_adapter_creation(self) -> None:
        """Test that HoverAviaryAdapter can be created and loaded."""
        adapter = _make_adapter(GameId.PYBULLET_HOVER_AVIARY)
        try:
            assert adapter.id == GameId.PYBULLET_HOVER_AVIARY.value
            assert adapter.default_render_mode == RenderMode.RGB_ARRAY
        finally:
            adapter.close()

    def test_adapter_reset(self) -> None:
        """Test environment reset returns valid observation."""
        adapter = _make_adapter(GameId.PYBULLET_HOVER_AVIARY)
        try:
            step = adapter.reset()
            assert step.observation is not None
            assert isinstance(step.observation, np.ndarray)
            assert step.reward == 0.0
            assert step.terminated is False
            assert step.truncated is False
        finally:
            adapter.close()

    def test_adapter_step(self) -> None:
        """Test environment step with random action."""
        adapter = _make_adapter(GameId.PYBULLET_HOVER_AVIARY)
        try:
            adapter.reset()

            # Get action space from environment
            env = adapter._env
            assert env is not None
            action = env.action_space.sample()

            step = adapter.step(action)
            assert step.observation is not None
            assert isinstance(step.observation, np.ndarray)
            assert isinstance(step.reward, float)
            assert isinstance(step.terminated, bool)
            assert isinstance(step.truncated, bool)
        finally:
            adapter.close()

    def test_adapter_multiple_steps(self) -> None:
        """Test multiple environment steps."""
        adapter = _make_adapter(GameId.PYBULLET_HOVER_AVIARY)
        try:
            adapter.reset()
            env = adapter._env
            assert env is not None

            for _ in range(10):
                action = env.action_space.sample()
                step = adapter.step(action)
                if step.terminated or step.truncated:
                    break

            # Should have completed steps without error
            assert True
        finally:
            adapter.close()

    def test_build_step_state(self) -> None:
        """Test step state building from observation."""
        adapter = _make_adapter(GameId.PYBULLET_HOVER_AVIARY)
        try:
            step = adapter.reset()
            state = adapter.build_step_state(step.observation, step.info)

            assert "env_id" in state.environment
            assert state.environment["env_id"] == GameId.PYBULLET_HOVER_AVIARY.value

            # For kinematic observations, should have position
            if len(step.observation) >= 12:
                assert "position" in state.metrics or "altitude" in state.metrics
        finally:
            adapter.close()


@requires_pybullet_drones
class TestMultiHoverAviaryAdapter:
    """Tests for MultiHoverAviaryAdapter (multi-agent hover task)."""

    def test_adapter_creation(self) -> None:
        """Test that MultiHoverAviaryAdapter can be created."""
        adapter = _make_adapter(GameId.PYBULLET_MULTIHOVER_AVIARY, num_drones=2)
        try:
            assert adapter.id == GameId.PYBULLET_MULTIHOVER_AVIARY.value
            assert ControlMode.MULTI_AGENT_COOP in adapter.supported_control_modes
        finally:
            adapter.close()

    def test_multi_agent_reset(self) -> None:
        """Test multi-agent environment reset."""
        adapter = _make_adapter(GameId.PYBULLET_MULTIHOVER_AVIARY, num_drones=2)
        try:
            step = adapter.reset()
            assert step.observation is not None
            # Multi-agent observations may have different shapes
            assert isinstance(step.observation, np.ndarray)
        finally:
            adapter.close()


@requires_pybullet_drones
class TestCtrlAviaryAdapter:
    """Tests for CtrlAviaryAdapter (low-level control)."""

    def test_adapter_creation(self) -> None:
        """Test that CtrlAviaryAdapter can be created."""
        adapter = _make_adapter(GameId.PYBULLET_CTRL_AVIARY)
        try:
            assert adapter.id == GameId.PYBULLET_CTRL_AVIARY.value
        finally:
            adapter.close()


@requires_pybullet_drones
class TestVelocityAviaryAdapter:
    """Tests for VelocityAviaryAdapter (velocity control)."""

    def test_adapter_creation(self) -> None:
        """Test that VelocityAviaryAdapter can be created."""
        adapter = _make_adapter(GameId.PYBULLET_VELOCITY_AVIARY)
        try:
            assert adapter.id == GameId.PYBULLET_VELOCITY_AVIARY.value
        finally:
            adapter.close()


@requires_pybullet_drones
class TestFactoryIntegration:
    """Test integration with adapter factory."""

    def test_create_adapter_with_config(self) -> None:
        """Test creating adapter through factory with config."""
        config = PyBulletDronesConfig(
            env_id=GameId.PYBULLET_HOVER_AVIARY.value,
            gui=False,
        )
        context = AdapterContext(settings=None, control_mode=ControlMode.AGENT_ONLY)

        adapter = create_adapter(
            GameId.PYBULLET_HOVER_AVIARY,
            context,
            game_config=config,  # type: ignore[arg-type]
        )

        assert adapter is not None
        assert isinstance(adapter, HoverAviaryAdapter)

    def test_create_adapter_without_config(self) -> None:
        """Test creating adapter through factory without config."""
        context = AdapterContext(settings=None, control_mode=ControlMode.AGENT_ONLY)

        adapter = create_adapter(GameId.PYBULLET_HOVER_AVIARY, context)

        assert adapter is not None
        assert isinstance(adapter, HoverAviaryAdapter)
