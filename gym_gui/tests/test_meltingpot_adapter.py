"""Test Melting Pot adapter creation and integration with MOSAIC factory.

Tests that:
1. MeltingPot adapters can be created via factory
2. Adapters have correct configuration
3. Adapters can load environments and perform basic operations
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from gym_gui.core.adapters.meltingpot import MeltingPotAdapter


class TestMeltingPotAdapterFactory:
    """Test that Melting Pot adapters can be created via factory."""

    def test_adapter_registered_in_factory(self) -> None:
        """Melting Pot adapters are registered in adapter factory."""
        from gym_gui.core.enums import GameId
        from gym_gui.core.factories.adapters import available_games

        melting_pot_games = [
            GameId.MELTINGPOT_COLLABORATIVE_COOKING,
            GameId.MELTINGPOT_CLEAN_UP,
            GameId.MELTINGPOT_COMMONS_HARVEST,
            GameId.MELTINGPOT_TERRITORY,
            GameId.MELTINGPOT_KING_OF_THE_HILL,
            GameId.MELTINGPOT_PRISONERS_DILEMMA,
            GameId.MELTINGPOT_STAG_HUNT,
            GameId.MELTINGPOT_ALLELOPATHIC_HARVEST,
        ]

        available = list(available_games())
        for game_id in melting_pot_games:
            assert game_id in available, f"{game_id} not available"

    def test_create_collaborative_cooking_adapter(self) -> None:
        """Can create Collaborative Cooking adapter via factory."""
        from gym_gui.core.enums import GameId
        from gym_gui.core.factories.adapters import create_adapter

        adapter = create_adapter(GameId.MELTINGPOT_COLLABORATIVE_COOKING)
        assert adapter is not None
        assert "collaborative_cooking__circuit" in adapter.id

    def test_create_commons_harvest_adapter(self) -> None:
        """Can create Commons Harvest adapter via factory."""
        from gym_gui.core.enums import GameId
        from gym_gui.core.factories.adapters import create_adapter

        adapter = create_adapter(GameId.MELTINGPOT_COMMONS_HARVEST)
        assert adapter is not None
        assert "commons_harvest__open" in adapter.id

    def test_adapter_with_config(self) -> None:
        """Can create adapter with custom config."""
        from gym_gui.config.game_configs import MeltingPotConfig
        from gym_gui.core.enums import GameId
        from gym_gui.core.factories.adapters import create_adapter

        config = MeltingPotConfig(
            substrate_name="territory__rooms",
            seed=42,
        )

        adapter = create_adapter(
            GameId.MELTINGPOT_TERRITORY,
            game_config=config,
        )

        assert adapter is not None
        assert "territory__rooms" in adapter.id


_HAS_MELTINGPOT_RUNTIME = False
try:
    import dm_env  # noqa: F401
    _HAS_MELTINGPOT_RUNTIME = True
except ImportError:
    pass


class TestMeltingPotAdapterBasics:
    """Test basic adapter operations (requires shimmy[meltingpot] + dm_env)."""

    pytestmark = pytest.mark.skipif(
        not _HAS_MELTINGPOT_RUNTIME,
        reason="shimmy[meltingpot] / dm_env not installed",
    )

    @pytest.fixture
    def adapter(self) -> "MeltingPotAdapter":  # type: ignore[return-type]
        """Create a Melting Pot adapter for testing."""
        from gym_gui.config.game_configs import MeltingPotConfig
        from gym_gui.core.adapters.meltingpot import MeltingPotAdapter

        config = MeltingPotConfig(substrate_name="collaborative_cooking__circuit")
        adapter = MeltingPotAdapter(config=config)
        yield adapter  # type: ignore[misc]
        if adapter._env is not None:
            adapter.close()

    def test_adapter_id(self, adapter: "MeltingPotAdapter") -> None:
        """Adapter has correct ID."""
        assert "collaborative_cooking__circuit" in adapter.id

    def test_adapter_load(self, adapter: "MeltingPotAdapter") -> None:
        """Adapter can load environment."""
        adapter.load()
        assert adapter._env is not None
        assert adapter.num_agents > 0
        assert len(adapter._agent_names) == adapter.num_agents

    def test_adapter_reset(self, adapter: "MeltingPotAdapter") -> None:
        """Adapter can reset environment."""
        adapter.load()
        step = adapter.reset()

        assert step.observation is not None
        assert step.terminated is False
        assert step.truncated is False
        assert adapter._step_counter == 0

    def test_adapter_step(self, adapter: "MeltingPotAdapter") -> None:
        """Adapter can step environment."""
        adapter.load()
        adapter.reset()

        # Sample actions for all agents
        actions = adapter.sample_action()
        step = adapter.step(actions)

        assert step.observation is not None
        assert isinstance(step.reward, float)
        assert adapter._step_counter == 1

    def test_adapter_render(self, adapter: "MeltingPotAdapter") -> None:
        """Adapter can render environment."""
        adapter.load()
        adapter.reset()

        render_result = adapter.render()

        assert "mode" in render_result
        assert "rgb" in render_result
        assert render_result["rgb"].ndim == 3  # RGB array

    def test_adapter_close(self, adapter: "MeltingPotAdapter") -> None:
        """Adapter can close environment."""
        adapter.load()
        adapter.reset()

        assert adapter._env is not None

        adapter.close()
        assert adapter._env is None


class TestMeltingPotAdapterCapabilities:
    """Test adapter capabilities and multi-agent features."""

    def test_adapter_capabilities(self) -> None:
        """Adapter declares correct capabilities."""
        from gym_gui.core.adapters.meltingpot import MeltingPotAdapter
        from gym_gui.core.enums import SteppingParadigm

        caps = MeltingPotAdapter.capabilities

        assert caps.stepping_paradigm == SteppingParadigm.SIMULTANEOUS
        assert SteppingParadigm.SIMULTANEOUS in caps.supported_paradigms
        assert caps.max_agents == 16
        assert "pettingzoo" in caps.env_types
        assert "parallel" in caps.env_types
        assert "discrete" in caps.action_spaces

    def test_action_meanings(self) -> None:
        """Adapter provides action meanings."""
        from gym_gui.config.game_configs import MeltingPotConfig
        from gym_gui.core.adapters.meltingpot import MELTINGPOT_ACTION_NAMES, MeltingPotAdapter

        config = MeltingPotConfig(substrate_name="collaborative_cooking__circuit")
        adapter = MeltingPotAdapter(config=config)

        meanings = adapter.get_action_meanings()
        assert meanings == MELTINGPOT_ACTION_NAMES
        assert "NOOP" in meanings
        assert "FORWARD" in meanings
        assert "INTERACT" in meanings

    @pytest.mark.skipif(
        not _HAS_MELTINGPOT_RUNTIME,
        reason="shimmy[meltingpot] / dm_env not installed",
    )
    def test_sample_actions(self) -> None:
        """Adapter can sample actions."""
        from gym_gui.config.game_configs import MeltingPotConfig
        from gym_gui.core.adapters.meltingpot import MeltingPotAdapter

        config = MeltingPotConfig(substrate_name="collaborative_cooking__circuit")
        adapter = MeltingPotAdapter(config=config)
        adapter.load()
        adapter.reset()

        # Sample actions for all agents
        actions = adapter.sample_action()
        assert isinstance(actions, dict)
        assert len(actions) == adapter.num_agents

        # Sample single action
        single_action = adapter.sample_single_action()
        assert isinstance(single_action, int)
        assert 0 <= single_action < len(adapter.get_action_meanings())


class TestAllMeltingPotAdapters:
    """Test all registered Melting Pot adapters."""

    @pytest.mark.parametrize("game_id, substrate_name", [
        ("MELTINGPOT_COLLABORATIVE_COOKING", "collaborative_cooking__circuit"),
        ("MELTINGPOT_COMMONS_HARVEST", "commons_harvest__open"),
        ("MELTINGPOT_TERRITORY", "territory__rooms"),
        ("MELTINGPOT_PRISONERS_DILEMMA", "prisoners_dilemma_in_the_matrix__repeated"),
        ("MELTINGPOT_STAG_HUNT", "stag_hunt_in_the_matrix__repeated"),
        ("MELTINGPOT_ALLELOPATHIC_HARVEST", "allelopathic_harvest__open"),
    ])
    def test_adapter_creation(self, game_id: str, substrate_name: str) -> None:
        """Test that each adapter can be created and has correct config."""
        from gym_gui.core.enums import GameId
        from gym_gui.core.factories.adapters import create_adapter

        game_id_enum = getattr(GameId, game_id)
        adapter = create_adapter(game_id_enum)

        assert adapter is not None
        assert substrate_name in adapter.id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
