"""Tests for Overcooked-AI adapter integration.

NOTE: These tests require the overcooked_ai package to be installed:
    pip install -e 3rd_party/environments/overcooked_ai/
"""

import numpy as np
import pytest

from gym_gui.config.game_configs import DEFAULT_OVERCOOKED_CRAMPED_ROOM_CONFIG, OvercookedConfig
from gym_gui.core.enums import EnvironmentFamily, GameId
from gym_gui.core.factories.adapters import create_adapter, get_adapter_cls

# Mark all tests as requiring overcooked
pytestmark = pytest.mark.skipif(
    not pytest.importorskip("overcooked_ai_py", reason="overcooked_ai not installed"),
    reason="Overcooked-AI package not available",
)


class TestOvercookedEnums:
    """Test that Overcooked game IDs and families are properly registered."""

    def test_game_ids_exist(self):
        """Verify all 5 Overcooked GameIds are defined."""
        assert hasattr(GameId, "OVERCOOKED_CRAMPED_ROOM")
        assert hasattr(GameId, "OVERCOOKED_ASYMMETRIC_ADVANTAGES")
        assert hasattr(GameId, "OVERCOOKED_COORDINATION_RING")
        assert hasattr(GameId, "OVERCOOKED_FORCED_COORDINATION")
        assert hasattr(GameId, "OVERCOOKED_COUNTER_CIRCUIT")

    def test_environment_family_exists(self):
        """Verify OVERCOOKED environment family is defined."""
        assert hasattr(EnvironmentFamily, "OVERCOOKED")
        assert EnvironmentFamily.OVERCOOKED == "overcooked"

    def test_game_id_values(self):
        """Verify GameId values have correct format."""
        assert GameId.OVERCOOKED_CRAMPED_ROOM == "overcooked/cramped_room"
        assert GameId.OVERCOOKED_ASYMMETRIC_ADVANTAGES == "overcooked/asymmetric_advantages"
        assert GameId.OVERCOOKED_COORDINATION_RING == "overcooked/coordination_ring"
        assert GameId.OVERCOOKED_FORCED_COORDINATION == "overcooked/forced_coordination"
        assert GameId.OVERCOOKED_COUNTER_CIRCUIT == "overcooked/counter_circuit"


class TestOvercookedConfig:
    """Test Overcooked configuration dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = OvercookedConfig()
        assert config.layout_name == "cramped_room"
        assert config.horizon == 400
        assert config.featurization == "lossless_encoding"
        assert config.seed is None

    def test_custom_config(self):
        """Test custom configuration."""
        config = OvercookedConfig(
            layout_name="coordination_ring",
            horizon=500,
            featurization="featurize",
            seed=42,
        )
        assert config.layout_name == "coordination_ring"
        assert config.horizon == 500
        assert config.featurization == "featurize"
        assert config.seed == 42

    def test_config_to_dict(self):
        """Test config serialization."""
        config = OvercookedConfig(layout_name="cramped_room", horizon=400)
        config_dict = config.to_dict()
        assert config_dict["layout_name"] == "cramped_room"
        assert config_dict["horizon"] == 400
        assert config_dict["mdp_params"] == {}
        assert config_dict["env_params"] == {}

    def test_config_from_dict(self):
        """Test config deserialization."""
        config_dict = {
            "layout_name": "forced_coordination",
            "horizon": 300,
            "featurization": "lossless_encoding",
        }
        config = OvercookedConfig.from_dict(config_dict)
        assert config.layout_name == "forced_coordination"
        assert config.horizon == 300

    def test_default_configs_exist(self):
        """Test that default configs are defined for all layouts."""
        assert DEFAULT_OVERCOOKED_CRAMPED_ROOM_CONFIG.layout_name == "cramped_room"


class TestOvercookedFactory:
    """Test adapter factory registration."""

    def test_adapter_registered(self):
        """Test that Overcooked adapters are in the factory registry."""
        from gym_gui.core.factories.adapters import available_games

        games = list(available_games())
        overcooked_games = [g for g in games if "overcooked" in g]

        assert len(overcooked_games) == 5
        assert GameId.OVERCOOKED_CRAMPED_ROOM in overcooked_games
        assert GameId.OVERCOOKED_ASYMMETRIC_ADVANTAGES in overcooked_games
        assert GameId.OVERCOOKED_COORDINATION_RING in overcooked_games
        assert GameId.OVERCOOKED_FORCED_COORDINATION in overcooked_games
        assert GameId.OVERCOOKED_COUNTER_CIRCUIT in overcooked_games

    def test_get_adapter_class(self):
        """Test retrieving adapter class from factory."""
        adapter_cls = get_adapter_cls(GameId.OVERCOOKED_CRAMPED_ROOM)
        assert adapter_cls.__name__ == "CrampedRoomAdapter"

    def test_create_adapter_with_config(self):
        """Test creating adapter with configuration."""
        config = OvercookedConfig(layout_name="cramped_room", horizon=400)
        adapter = create_adapter(GameId.OVERCOOKED_CRAMPED_ROOM, game_config=config)

        assert adapter is not None
        assert adapter.id == "overcooked/cramped_room"
        assert adapter.num_agents == 2  # type: ignore[attr-defined]

    def test_create_all_layout_adapters(self):
        """Test creating adapters for all 5 layouts."""
        layouts = [
            (GameId.OVERCOOKED_CRAMPED_ROOM, "cramped_room"),
            (GameId.OVERCOOKED_ASYMMETRIC_ADVANTAGES, "asymmetric_advantages"),
            (GameId.OVERCOOKED_COORDINATION_RING, "coordination_ring"),
            (GameId.OVERCOOKED_FORCED_COORDINATION, "forced_coordination"),
            (GameId.OVERCOOKED_COUNTER_CIRCUIT, "counter_circuit"),
        ]

        for game_id, layout_name in layouts:
            config = OvercookedConfig(layout_name=layout_name)
            adapter = create_adapter(game_id, game_config=config)
            assert adapter.id == f"overcooked/{layout_name}"


class TestOvercookedAdapter:
    """Test Overcooked adapter functionality."""

    def test_adapter_capabilities(self):
        """Test adapter capability declaration."""
        from gym_gui.core.adapters.overcooked import OvercookedAdapter
        from gym_gui.core.enums import SteppingParadigm

        capabilities = OvercookedAdapter.capabilities

        assert capabilities.stepping_paradigm == SteppingParadigm.SIMULTANEOUS
        assert SteppingParadigm.SIMULTANEOUS in capabilities.supported_paradigms
        assert "overcooked" in capabilities.env_types
        assert "cooperative" in capabilities.env_types
        assert "discrete" in capabilities.action_spaces
        assert "box" in capabilities.observation_spaces
        assert capabilities.max_agents == 2
        assert capabilities.supports_self_play is True
        assert capabilities.supports_record is True

    def test_adapter_load(self):
        """Test loading the Overcooked environment."""
        config = OvercookedConfig(layout_name="cramped_room", horizon=400)
        adapter = create_adapter(GameId.OVERCOOKED_CRAMPED_ROOM, game_config=config)

        adapter.load()
        assert adapter._env is not None
        assert adapter._mdp is not None  # type: ignore[attr-defined]

    def test_adapter_reset(self):
        """Test resetting the environment."""
        config = OvercookedConfig(layout_name="cramped_room", horizon=400)
        adapter = create_adapter(GameId.OVERCOOKED_CRAMPED_ROOM, game_config=config)
        adapter.load()

        step = adapter.reset()

        assert step.observation is not None
        assert len(step.observation) == 2  # 2 agents
        assert isinstance(step.observation[0], np.ndarray)
        assert isinstance(step.observation[1], np.ndarray)
        assert step.reward == 0.0
        assert step.terminated is False
        assert step.truncated is False

    def test_adapter_step_with_list(self):
        """Test stepping with a list of actions."""
        config = OvercookedConfig(layout_name="cramped_room", horizon=400)
        adapter = create_adapter(GameId.OVERCOOKED_CRAMPED_ROOM, game_config=config)
        adapter.load()
        adapter.reset()

        # Step with list of actions (one per agent)
        step = adapter.step([4, 4])  # Both agents STAY

        assert step.observation is not None
        assert len(step.observation) == 2
        assert isinstance(step.reward, (int, float))
        assert isinstance(step.terminated, bool)

    def test_adapter_step_with_broadcast(self):
        """Test stepping with single action (broadcast to all agents)."""
        config = OvercookedConfig(layout_name="cramped_room", horizon=400)
        adapter = create_adapter(GameId.OVERCOOKED_CRAMPED_ROOM, game_config=config)
        adapter.load()
        adapter.reset()

        # Step with single action (broadcast)
        step = adapter.step(4)  # Both agents STAY

        assert step.observation is not None
        assert len(step.observation) == 2

    def test_adapter_episode_completion(self):
        """Test that episode completes at horizon."""
        config = OvercookedConfig(layout_name="cramped_room", horizon=5)
        adapter = create_adapter(GameId.OVERCOOKED_CRAMPED_ROOM, game_config=config)
        adapter.load()
        adapter.reset()

        # Step until done
        done = False
        steps = 0
        while not done and steps < 10:
            step = adapter.step([4, 4])  # Both STAY
            done = step.terminated
            steps += 1

        assert done is True
        assert steps == 5  # Should complete at horizon

    def test_get_agent_observation(self):
        """Test getting observation for specific agent."""
        config = OvercookedConfig(layout_name="cramped_room", horizon=400)
        adapter = create_adapter(GameId.OVERCOOKED_CRAMPED_ROOM, game_config=config)
        adapter.load()
        adapter.reset()

        obs0 = adapter.get_agent_observation(0)  # type: ignore[attr-defined]
        obs1 = adapter.get_agent_observation(1)  # type: ignore[attr-defined]

        assert isinstance(obs0, np.ndarray)
        assert isinstance(obs1, np.ndarray)

    def test_get_agent_reward(self):
        """Test getting reward for specific agent."""
        config = OvercookedConfig(layout_name="cramped_room", horizon=400)
        adapter = create_adapter(GameId.OVERCOOKED_CRAMPED_ROOM, game_config=config)
        adapter.load()
        adapter.reset()
        adapter.step([4, 4])

        reward0 = adapter.get_agent_reward(0)  # type: ignore[attr-defined]
        reward1 = adapter.get_agent_reward(1)  # type: ignore[attr-defined]

        assert isinstance(reward0, (int, float))
        assert isinstance(reward1, (int, float))

    def test_adapter_close(self):
        """Test closing the environment."""
        config = OvercookedConfig(layout_name="cramped_room", horizon=400)
        adapter = create_adapter(GameId.OVERCOOKED_CRAMPED_ROOM, game_config=config)
        adapter.load()

        adapter.close()
        assert adapter._env is None
        assert adapter._mdp is None  # type: ignore[attr-defined]

    def test_featurization_methods(self):
        """Test both featurization methods work."""
        # Test lossless_encoding
        config1 = OvercookedConfig(
            layout_name="cramped_room", featurization="lossless_encoding"
        )
        adapter1 = create_adapter(GameId.OVERCOOKED_CRAMPED_ROOM, game_config=config1)
        adapter1.load()
        step1 = adapter1.reset()
        assert step1.observation is not None

        # Test featurize
        config2 = OvercookedConfig(layout_name="cramped_room", featurization="featurize")
        adapter2 = create_adapter(GameId.OVERCOOKED_CRAMPED_ROOM, game_config=config2)
        adapter2.load()
        step2 = adapter2.reset()
        assert step2.observation is not None


class TestOvercookedActions:
    """Test Overcooked action space."""

    def test_action_names(self):
        """Verify action names are correct."""
        from gym_gui.core.adapters.overcooked import OVERCOOKED_ACTIONS

        expected = ["NORTH", "SOUTH", "EAST", "WEST", "STAY", "INTERACT"]
        assert OVERCOOKED_ACTIONS == expected
        assert len(OVERCOOKED_ACTIONS) == 6

    def test_all_actions_valid(self):
        """Test that all 6 actions are valid."""
        config = OvercookedConfig(layout_name="cramped_room", horizon=400)
        adapter = create_adapter(GameId.OVERCOOKED_CRAMPED_ROOM, game_config=config)
        adapter.load()
        adapter.reset()

        # Try all 6 actions
        for action_idx in range(6):
            step = adapter.step([action_idx, action_idx])
            assert step.observation is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
