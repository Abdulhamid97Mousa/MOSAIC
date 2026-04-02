"""Unit tests for multi-worker Operator system.

This module tests the updated OperatorConfig and WorkerAssignment dataclasses
that support both single-agent and multi-agent (turn-based) environments.

Architecture Overview:
    - Operator = Workers -> Environment binding
    - Single-agent: 1 Worker -> 1 Environment (e.g., BabyAI, MiniGrid)
    - Multi-agent: N Workers -> 1 Environment (e.g., Chess, Go, Connect Four)

Test Categories:
    1. WorkerAssignment dataclass validation
    2. OperatorConfig single-agent factory method
    3. OperatorConfig multi-agent factory method
    4. OperatorConfig backwards compatibility properties
    5. OperatorConfig with_run_id deep copy behavior
    6. OperatorConfig player management methods

Dependencies:
    - gym_gui.services.operator.OperatorConfig
    - gym_gui.services.operator.WorkerAssignment

Author: Claude (AI Assistant)
Date: 2026-01-01
"""

from typing import Any, Dict

import pytest

from gym_gui.services.operator import (
    OperatorConfig,
    WorkerAssignment,
)

# =============================================================================
# WorkerAssignment Tests
# =============================================================================


class TestWorkerAssignment:
    """Test suite for WorkerAssignment dataclass.

    WorkerAssignment represents a single worker assigned to a player slot
    within an operator. It contains worker_id, worker_type, and settings.
    """

    def test_create_llm_worker_assignment(self) -> None:
        """Test creating an LLM worker assignment with typical settings.

        Verifies that:
            - worker_id is correctly stored
            - worker_type is 'llm'
            - settings dict contains expected keys
        """
        assignment = WorkerAssignment(
            worker_id="balrog_worker",
            worker_type="llm",
            settings={
                "client_name": "openrouter",
                "model_id": "openai/gpt-4o",
                "api_key": "test-key-123",
            },
        )

        assert assignment.worker_id == "balrog_worker", \
            "worker_id should be 'balrog_worker'"
        assert assignment.worker_type == "llm", \
            "worker_type should be 'llm'"
        assert assignment.settings["client_name"] == "openrouter", \
            "settings should contain client_name"
        assert assignment.settings["model_id"] == "openai/gpt-4o", \
            "settings should contain model_id"

    def test_create_vlm_worker_assignment(self) -> None:
        """Test creating a VLM (Vision Language Model) worker assignment.

        VLM workers have max_image_history > 0 to enable vision mode.

        Verifies that:
            - worker_type accepts 'vlm'
            - max_image_history setting is preserved
        """
        assignment = WorkerAssignment(
            worker_id="balrog_worker",
            worker_type="vlm",
            settings={
                "client_name": "openai",
                "model_id": "gpt-4o",
                "max_image_history": 1,
            },
        )

        assert assignment.worker_type == "vlm", \
            "worker_type should accept 'vlm'"
        assert assignment.settings["max_image_history"] == 1, \
            "VLM should have max_image_history > 0"

    def test_create_rl_worker_assignment(self) -> None:
        """Test creating an RL (Reinforcement Learning) worker assignment.

        RL workers typically have policy_path and algorithm settings.

        Verifies that:
            - worker_type accepts 'rl'
            - RL-specific settings are preserved
        """
        assignment = WorkerAssignment(
            worker_id="cleanrl_worker",
            worker_type="rl",
            settings={
                "algorithm": "ppo",
                "policy_path": "/path/to/checkpoint.pt",
                "learning_rate": 0.0003,
            },
        )

        assert assignment.worker_id == "cleanrl_worker", \
            "worker_id should be 'cleanrl_worker'"
        assert assignment.worker_type == "rl", \
            "worker_type should be 'rl'"
        assert assignment.settings["algorithm"] == "ppo", \
            "settings should contain algorithm"

    def test_create_human_worker_assignment(self) -> None:
        """Test creating a human worker assignment.

        Human workers represent keyboard/mouse input from a user.
        They typically have minimal settings.

        Verifies that:
            - worker_type accepts 'human'
            - Empty settings dict is valid
        """
        assignment = WorkerAssignment(
            worker_id="human_input",
            worker_type="human",
            settings={},
        )

        assert assignment.worker_type == "human", \
            "worker_type should accept 'human'"
        assert assignment.settings == {}, \
            "Human worker can have empty settings"

    def test_invalid_worker_type_raises_error(self) -> None:
        """Test that invalid worker_type raises ValueError.

        Only 'llm', 'vlm', 'rl', 'human' are valid worker types.
        Any other value should raise ValueError with descriptive message.
        """
        with pytest.raises(ValueError) as exc_info:
            WorkerAssignment(
                worker_id="test_worker",
                worker_type="invalid_type",
                settings={},
            )

        error_message = str(exc_info.value)
        assert "worker_type must be one of" in error_message, \
            "Error message should indicate valid worker types"
        assert "invalid_type" in error_message, \
            "Error message should include the invalid value"

    def test_settings_default_to_empty_dict(self) -> None:
        """Test that settings defaults to empty dict when not provided.

        WorkerAssignment should allow creating instances without
        explicitly passing settings parameter.
        """
        assignment = WorkerAssignment(
            worker_id="balrog_worker",
            worker_type="llm",
        )

        assert assignment.settings == {}, \
            "settings should default to empty dict"
        assert isinstance(assignment.settings, dict), \
            "settings should be a dict instance"


# =============================================================================
# OperatorConfig Single-Agent Tests
# =============================================================================


class TestOperatorConfigSingleAgent:
    """Test suite for OperatorConfig single-agent factory method.

    Single-agent operators bind one worker to one environment.
    The worker is stored in workers["agent"].
    """

    def test_single_agent_factory_creates_config(self) -> None:
        """Test that single_agent() factory creates valid config.

        Verifies:
            - All required fields are set
            - workers dict contains 'agent' key
            - is_multiagent is False
        """
        config = OperatorConfig.single_agent(
            operator_id="op_1",
            display_name="Test Operator",
            worker_id="balrog_worker",
            worker_type="llm",
            env_name="babyai",
            task="BabyAI-GoToObj-v0",
            settings={"model_id": "gpt-4o"},
        )

        assert config.operator_id == "op_1", \
            "operator_id should be set"
        assert config.display_name == "Test Operator", \
            "display_name should be set"
        assert config.env_name == "babyai", \
            "env_name should be set"
        assert config.task == "BabyAI-GoToObj-v0", \
            "task should be set"
        assert "agent" in config.workers, \
            "workers dict should contain 'agent' key for single-agent"
        assert config.is_multiagent is False, \
            "is_multiagent should be False for single-agent"

    def test_single_agent_worker_assignment(self) -> None:
        """Test that single_agent() correctly creates WorkerAssignment.

        The worker settings should be accessible via workers["agent"].
        """
        config = OperatorConfig.single_agent(
            operator_id="op_1",
            display_name="GPT-4 Agent",
            worker_id="balrog_worker",
            worker_type="llm",
            settings={
                "client_name": "openrouter",
                "model_id": "openai/gpt-4o",
            },
        )

        worker = config.workers["agent"]
        assert worker.worker_id == "balrog_worker", \
            "Worker should have correct worker_id"
        assert worker.worker_type == "llm", \
            "Worker should have correct worker_type"
        assert worker.settings["model_id"] == "openai/gpt-4o", \
            "Worker settings should contain model_id"

    def test_single_agent_default_env_and_task(self) -> None:
        """Test that single_agent() uses sensible defaults.

        When env_name and task are not provided, should default to
        babyai environment.
        """
        config = OperatorConfig.single_agent(
            operator_id="op_1",
            display_name="Default Env",
            worker_id="balrog_worker",
            worker_type="llm",
        )

        assert config.env_name == "babyai", \
            "Default env_name should be 'babyai'"
        assert config.task == "BabyAI-GoToRedBall-v0", \
            "Default task should be 'BabyAI-GoToRedBall-v0'"

    def test_single_agent_with_rl_worker(self) -> None:
        """Test single_agent() with RL worker type.

        RL workers are commonly used for trained policy evaluation.
        """
        config = OperatorConfig.single_agent(
            operator_id="op_rl",
            display_name="PPO Agent",
            worker_id="cleanrl_worker",
            worker_type="rl",
            env_name="classic_control",
            task="CartPole-v1",
            settings={
                "algorithm": "ppo",
                "policy_path": "/models/ppo_cartpole.pt",
            },
        )

        assert config.operator_type == "rl", \
            "operator_type property should return 'rl'"
        assert config.workers["agent"].worker_type == "rl", \
            "Worker should have rl type"


# =============================================================================
# OperatorConfig Multi-Agent Tests
# =============================================================================


class TestOperatorConfigMultiAgent:
    """Test suite for OperatorConfig multi-agent factory method.

    Multi-agent operators bind N workers to one environment.
    Each worker is assigned to a player_id (e.g., "player_0", "player_1").
    Used for turn-based games like Chess, Go, Connect Four.
    """

    def test_multi_agent_factory_creates_config(self) -> None:
        """Test that multi_agent() factory creates valid config.

        Verifies:
            - workers dict contains all player keys
            - is_multiagent is True
            - operator_type returns 'multiagent'
        """
        config = OperatorConfig.multi_agent(
            operator_id="game_1",
            display_name="Chess Match",
            env_name="pettingzoo",
            task="chess_v6",
            player_workers={
                "player_0": WorkerAssignment("balrog_worker", "llm", {"model_id": "gpt-4o"}),
                "player_1": WorkerAssignment("balrog_worker", "llm", {"model_id": "llama-3"}),
            },
        )

        assert config.operator_id == "game_1", \
            "operator_id should be set"
        assert config.display_name == "Chess Match", \
            "display_name should be set"
        assert config.env_name == "pettingzoo", \
            "env_name should be 'pettingzoo'"
        assert config.task == "chess_v6", \
            "task should be 'chess_v6'"
        assert config.is_multiagent is True, \
            "is_multiagent should be True"
        assert config.operator_type == "multiagent", \
            "operator_type should return 'multiagent'"

    def test_multi_agent_player_ids(self) -> None:
        """Test that player_ids property returns correct player list.

        player_ids should return all keys from the workers dict.
        """
        config = OperatorConfig.multi_agent(
            operator_id="game_1",
            display_name="Go Match",
            env_name="pettingzoo",
            task="go_v5",
            player_workers={
                "black_0": WorkerAssignment("balrog_worker", "llm", {}),
                "white_0": WorkerAssignment("balrog_worker", "rl", {}),
            },
        )

        player_ids = config.player_ids
        assert "black_0" in player_ids, \
            "player_ids should contain 'black_0'"
        assert "white_0" in player_ids, \
            "player_ids should contain 'white_0'"
        assert len(player_ids) == 2, \
            "Should have exactly 2 players"

    def test_multi_agent_get_worker_for_player(self) -> None:
        """Test get_worker_for_player() method.

        Should return the correct WorkerAssignment for each player_id,
        or None for invalid player_id.
        """
        gpt4_settings = {"model_id": "gpt-4o", "client_name": "openai"}
        llama_settings = {"model_id": "llama-3", "client_name": "vllm"}

        config = OperatorConfig.multi_agent(
            operator_id="game_1",
            display_name="Chess: GPT-4 vs Llama",
            env_name="pettingzoo",
            task="chess_v6",
            player_workers={
                "player_0": WorkerAssignment("balrog_worker", "llm", gpt4_settings),
                "player_1": WorkerAssignment("balrog_worker", "llm", llama_settings),
            },
        )

        # Test valid player_ids
        worker_0 = config.get_worker_for_player("player_0")
        assert worker_0 is not None, \
            "Should return worker for valid player_id"
        assert worker_0.settings["model_id"] == "gpt-4o", \
            "player_0 should have GPT-4o"

        worker_1 = config.get_worker_for_player("player_1")
        assert worker_1 is not None, \
            "Should return worker for valid player_id"
        assert worker_1.settings["model_id"] == "llama-3", \
            "player_1 should have Llama-3"

        # Test invalid player_id
        invalid_worker = config.get_worker_for_player("invalid_player")
        assert invalid_worker is None, \
            "Should return None for invalid player_id"

    def test_multi_agent_mixed_worker_types(self) -> None:
        """Test multi-agent with different worker types per player.

        It's valid to have LLM vs RL, Human vs LLM, etc.
        """
        config = OperatorConfig.multi_agent(
            operator_id="game_1",
            display_name="Human vs AI",
            env_name="pettingzoo",
            task="connect_four_v3",
            player_workers={
                "player_0": WorkerAssignment("human_input", "human", {}),
                "player_1": WorkerAssignment("balrog_worker", "llm", {"model_id": "gpt-4o"}),
            },
        )

        human_worker = config.get_worker_for_player("player_0")
        ai_worker = config.get_worker_for_player("player_1")

        assert human_worker is not None
        assert human_worker.worker_type == "human", \
            "player_0 should be human"
        assert ai_worker is not None
        assert ai_worker.worker_type == "llm", \
            "player_1 should be llm"


# =============================================================================
# OperatorConfig Backwards Compatibility Tests
# =============================================================================


class TestOperatorConfigBackwardsCompatibility:
    """Test suite for backwards compatibility properties.

    Existing code expects operator_type, worker_id, settings as direct
    attributes. These are now properties that read from workers["agent"].
    """

    def test_operator_type_property_single_agent(self) -> None:
        """Test operator_type property returns first worker's type.

        For single-agent, should return the worker_type of workers["agent"].
        """
        config = OperatorConfig.single_agent(
            operator_id="op_1",
            display_name="Test",
            worker_id="balrog_worker",
            worker_type="llm",
        )

        assert config.operator_type == "llm", \
            "operator_type should return 'llm' for single-agent LLM"

    def test_operator_type_property_multi_agent(self) -> None:
        """Test operator_type returns 'multiagent' for multi-agent configs.

        When len(workers) > 1, operator_type should return 'multiagent'.
        """
        config = OperatorConfig.multi_agent(
            operator_id="game_1",
            display_name="Chess",
            env_name="pettingzoo",
            task="chess_v6",
            player_workers={
                "player_0": WorkerAssignment("balrog_worker", "llm", {}),
                "player_1": WorkerAssignment("balrog_worker", "llm", {}),
            },
        )

        assert config.operator_type == "multiagent", \
            "operator_type should return 'multiagent' for multi-agent"

    def test_worker_id_property(self) -> None:
        """Test worker_id property returns first worker's ID.

        For backwards compatibility, worker_id returns the worker_id
        of the first worker in the dict.
        """
        config = OperatorConfig.single_agent(
            operator_id="op_1",
            display_name="Test",
            worker_id="balrog_worker",
            worker_type="llm",
        )

        assert config.worker_id == "balrog_worker", \
            "worker_id property should return 'balrog_worker'"

    def test_settings_property(self) -> None:
        """Test settings property returns first worker's settings.

        For backwards compatibility, settings returns the settings dict
        of the first worker.
        """
        config = OperatorConfig.single_agent(
            operator_id="op_1",
            display_name="Test",
            worker_id="balrog_worker",
            worker_type="llm",
            settings={"model_id": "gpt-4o", "temperature": 0.7},
        )

        assert config.settings["model_id"] == "gpt-4o", \
            "settings property should return model_id"
        assert config.settings["temperature"] == 0.7, \
            "settings property should return temperature"


# =============================================================================
# OperatorConfig with_run_id Tests
# =============================================================================


class TestOperatorConfigWithRunId:
    """Test suite for with_run_id() method.

    with_run_id() creates a deep copy of the config with run_id set.
    Used when launching operators to assign telemetry routing IDs.
    """

    def test_with_run_id_single_agent(self) -> None:
        """Test with_run_id() creates copy with run_id for single-agent.

        Verifies:
            - Original config is unchanged
            - New config has run_id set
            - All other fields are preserved
        """
        original = OperatorConfig.single_agent(
            operator_id="op_1",
            display_name="Test",
            worker_id="balrog_worker",
            worker_type="llm",
            settings={"model_id": "gpt-4o"},
        )

        copied = original.with_run_id("run_abc123")

        # Original unchanged
        assert original.run_id is None, \
            "Original config should not have run_id"

        # Copied has run_id
        assert copied.run_id == "run_abc123", \
            "Copied config should have run_id"

        # Other fields preserved
        assert copied.operator_id == original.operator_id, \
            "operator_id should be preserved"
        assert copied.display_name == original.display_name, \
            "display_name should be preserved"
        assert copied.operator_type == original.operator_type, \
            "operator_type should be preserved"

    def test_with_run_id_deep_copies_workers(self) -> None:
        """Test with_run_id() deep copies the workers dict.

        Modifying the copied config's workers should not affect original.
        """
        original = OperatorConfig.single_agent(
            operator_id="op_1",
            display_name="Test",
            worker_id="balrog_worker",
            worker_type="llm",
            settings={"model_id": "gpt-4o"},
        )

        copied = original.with_run_id("run_123")

        # Modify copied settings
        copied.workers["agent"].settings["model_id"] = "modified"

        # Original should be unchanged
        assert original.settings["model_id"] == "gpt-4o", \
            "Original settings should not be modified"

    def test_with_run_id_multi_agent(self) -> None:
        """Test with_run_id() works correctly for multi-agent configs.

        All workers should be deep copied.
        """
        original = OperatorConfig.multi_agent(
            operator_id="game_1",
            display_name="Chess",
            env_name="pettingzoo",
            task="chess_v6",
            player_workers={
                "player_0": WorkerAssignment("balrog_worker", "llm", {"model_id": "gpt-4o"}),
                "player_1": WorkerAssignment("balrog_worker", "llm", {"model_id": "llama-3"}),
            },
        )

        copied = original.with_run_id("run_game_1")

        assert copied.run_id == "run_game_1", \
            "Copied config should have run_id"
        assert len(copied.workers) == 2, \
            "Should have 2 workers"
        assert copied.is_multiagent is True, \
            "Should still be multiagent"


# =============================================================================
# OperatorConfig Default Constructor Tests
# =============================================================================


class TestOperatorConfigDefaultConstructor:
    """Test suite for default constructor behavior.

    When OperatorConfig is created without workers dict, it should
    create a default single-agent configuration.
    """

    def test_default_constructor_creates_default_worker(self) -> None:
        """Test that default constructor creates default worker.

        When workers dict is empty or not provided, __post_init__
        should create a default worker at workers["agent"].
        """
        config = OperatorConfig(
            operator_id="op_default",
            display_name="Default Config",
        )

        assert "agent" in config.workers, \
            "Should have 'agent' key in workers"
        assert config.workers["agent"].worker_id == "balrog_worker", \
            "Default worker_id should be 'balrog_worker'"
        assert config.workers["agent"].worker_type == "llm", \
            "Default worker_type should be 'llm'"

    def test_constructor_with_workers_dict(self) -> None:
        """Test constructor accepts explicit workers dict.

        When workers dict is provided, should use it directly.
        """
        config = OperatorConfig(
            operator_id="op_1",
            display_name="Custom Workers",
            workers={
                "agent": WorkerAssignment("cleanrl_worker", "rl", {"algorithm": "dqn"}),
            },
        )

        assert config.workers["agent"].worker_id == "cleanrl_worker", \
            "Should use provided worker_id"
        assert config.workers["agent"].worker_type == "rl", \
            "Should use provided worker_type"


# =============================================================================
# Integration Tests
# =============================================================================


class TestOperatorConfigIntegration:
    """Integration tests for OperatorConfig usage scenarios.

    These tests simulate real-world usage patterns.
    """

    def test_widget_usage_pattern(self) -> None:
        """Test the pattern used by operator_config_widget.py.

        The widget creates configs using single_agent() factory.
        """
        # Simulate widget creating config
        operator_type = "llm"
        worker_id = "balrog_worker"
        settings = {
            "client_name": "openrouter",
            "model_id": "openai/gpt-4o-mini",
        }

        config = OperatorConfig.single_agent(
            operator_id="operator_0",
            display_name="Operator 1",
            worker_id=worker_id,
            worker_type=operator_type,
            env_name="babyai",
            task="BabyAI-ActionObjDoor-v0",
            settings=settings,
        )

        # Verify backwards-compatible access
        assert config.operator_type == "llm", \
            "Widget should be able to read operator_type"
        assert config.worker_id == "balrog_worker", \
            "Widget should be able to read worker_id"
        assert config.settings["model_id"] == "openai/gpt-4o-mini", \
            "Widget should be able to read settings"

    def test_multi_operator_comparison_pattern(self) -> None:
        """Test pattern for comparing multiple operators side-by-side.

        Multi-operator view shows N operators running same env.
        """
        configs = [
            OperatorConfig.single_agent(
                operator_id=f"operator_{i}",
                display_name=f"Operator {i+1}",
                worker_id="balrog_worker",
                worker_type="llm",
                env_name="babyai",
                task="BabyAI-GoToObj-v0",
                settings={"model_id": model},
            )
            for i, model in enumerate(["gpt-4o", "claude-3-opus", "llama-3"])
        ]

        assert len(configs) == 3, \
            "Should create 3 operator configs"

        for i, config in enumerate(configs):
            assert config.operator_id == f"operator_{i}", \
                f"Config {i} should have correct operator_id"
            assert config.is_multiagent is False, \
                "Each config should be single-agent"

    def test_chess_match_pattern(self) -> None:
        """Test pattern for setting up a chess match.

        Chess requires 2 workers assigned to player_0 and player_1.
        """
        config = OperatorConfig.multi_agent(
            operator_id="chess_match_1",
            display_name="GPT-4 vs Claude: Chess",
            env_name="pettingzoo",
            task="chess_v6",
            player_workers={
                "player_0": WorkerAssignment(
                    worker_id="balrog_worker",
                    worker_type="llm",
                    settings={
                        "client_name": "openai",
                        "model_id": "gpt-4o",
                    },
                ),
                "player_1": WorkerAssignment(
                    worker_id="balrog_worker",
                    worker_type="llm",
                    settings={
                        "client_name": "anthropic",
                        "model_id": "claude-3-opus",
                    },
                ),
            },
        )

        assert config.env_name == "pettingzoo", \
            "Chess should use pettingzoo env"
        assert config.task == "chess_v6", \
            "Should use chess_v6 task"
        assert config.is_multiagent is True, \
            "Chess is multi-agent"
        assert len(config.player_ids) == 2, \
            "Chess has 2 players"

        # Verify player assignments
        white = config.get_worker_for_player("player_0")
        black = config.get_worker_for_player("player_1")

        assert white is not None
        assert white.settings["model_id"] == "gpt-4o", \
            "White (player_0) should be GPT-4"
        assert black is not None
        assert black.settings["model_id"] == "claude-3-opus", \
            "Black (player_1) should be Claude"


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestOperatorConfigEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_settings_dict(self) -> None:
        """Test that empty settings dict is handled correctly."""
        config = OperatorConfig.single_agent(
            operator_id="op_1",
            display_name="Minimal",
            worker_id="balrog_worker",
            worker_type="llm",
            settings=None,  # None should be converted to {}
        )

        assert config.settings == {}, \
            "None settings should become empty dict"

    def test_single_player_is_not_multiagent(self) -> None:
        """Test that single player in workers is not considered multiagent."""
        config = OperatorConfig(
            operator_id="op_1",
            display_name="Single",
            workers={
                "agent": WorkerAssignment("balrog_worker", "llm", {}),
            },
        )

        assert config.is_multiagent is False, \
            "Single worker should not be multiagent"

    def test_three_player_game(self) -> None:
        """Test support for 3+ player games (if any exist)."""
        config = OperatorConfig.multi_agent(
            operator_id="game_1",
            display_name="3-Player Game",
            env_name="pettingzoo",
            task="some_3_player_game",
            player_workers={
                "player_0": WorkerAssignment("balrog_worker", "llm", {}),
                "player_1": WorkerAssignment("balrog_worker", "llm", {}),
                "player_2": WorkerAssignment("balrog_worker", "rl", {}),
            },
        )

        assert len(config.player_ids) == 3, \
            "Should support 3 players"
        assert config.is_multiagent is True, \
            "3 players is multiagent"


# =============================================================================
# PETTINGZOO_GAMES Constant Tests
# =============================================================================


class TestPettingZooGamesConstant:
    """Test suite for PETTINGZOO_GAMES constant.

    PETTINGZOO_GAMES maps game names to their player configuration.
    Each entry has:
        - family: The PettingZoo environment family (e.g., "classic")
        - players: List of player IDs as used by PettingZoo
        - player_labels: Human-readable labels for each player
    """

    def test_pettingzoo_games_constant_exists(self) -> None:
        """Test that PETTINGZOO_GAMES constant is available."""
        from gym_gui.ui.widgets.operator_config_widget import PETTINGZOO_GAMES

        assert isinstance(PETTINGZOO_GAMES, dict), \
            "PETTINGZOO_GAMES should be a dictionary"
        assert len(PETTINGZOO_GAMES) > 0, \
            "PETTINGZOO_GAMES should have at least one game"

    def test_chess_game_configuration(self) -> None:
        """Test chess game configuration has correct players."""
        from gym_gui.ui.widgets.operator_config_widget import PETTINGZOO_GAMES

        assert "chess_v6" in PETTINGZOO_GAMES, \
            "chess_v6 should be in PETTINGZOO_GAMES"

        chess = PETTINGZOO_GAMES["chess_v6"]
        assert chess["family"] == "classic", \
            "Chess should be in the classic family"
        assert chess["players"] == ["player_0", "player_1"], \
            "Chess should have player_0 and player_1"
        assert chess["player_labels"]["player_0"] == "White", \
            "player_0 should be labeled White"
        assert chess["player_labels"]["player_1"] == "Black", \
            "player_1 should be labeled Black"

    def test_go_game_uses_correct_player_ids(self) -> None:
        """Test Go game uses PettingZoo's actual player IDs."""
        from gym_gui.ui.widgets.operator_config_widget import PETTINGZOO_GAMES

        assert "go_v5" in PETTINGZOO_GAMES, \
            "go_v5 should be in PETTINGZOO_GAMES"

        go = PETTINGZOO_GAMES["go_v5"]
        # Go uses black_0 and white_0 as actual PettingZoo agent names
        assert go["players"] == ["black_0", "white_0"], \
            "Go should have black_0 and white_0 (PettingZoo's actual IDs)"
        assert go["player_labels"]["black_0"] == "Black", \
            "black_0 should be labeled Black"
        assert go["player_labels"]["white_0"] == "White", \
            "white_0 should be labeled White"

    def test_all_games_have_required_fields(self) -> None:
        """Test all games have family, players, and player_labels."""
        from gym_gui.ui.widgets.operator_config_widget import PETTINGZOO_GAMES

        for game_name, game_info in PETTINGZOO_GAMES.items():
            assert "family" in game_info, \
                f"{game_name} should have 'family' field"
            assert "players" in game_info, \
                f"{game_name} should have 'players' field"
            assert "player_labels" in game_info, \
                f"{game_name} should have 'player_labels' field"

            # Players should be a non-empty list
            assert isinstance(game_info["players"], list), \
                f"{game_name} players should be a list"
            assert len(game_info["players"]) >= 2, \
                f"{game_name} should have at least 2 players"

            # Each player should have a label
            for player_id in game_info["players"]:
                assert player_id in game_info["player_labels"], \
                    f"{game_name}: {player_id} should have a label"

    def test_pettingzoo_in_env_families(self) -> None:
        """Test that pettingzoo_classic is included in ENV_FAMILIES."""
        from gym_gui.ui.widgets.operator_config_widget import (
            ENV_FAMILIES,
            PETTINGZOO_GAMES,
        )

        assert "pettingzoo_classic" in ENV_FAMILIES, \
            "pettingzoo_classic should be in ENV_FAMILIES"
        # ENV_FAMILIES["pettingzoo_classic"] should contain all game names
        assert set(ENV_FAMILIES["pettingzoo_classic"]) == set(PETTINGZOO_GAMES.keys()), \
            "ENV_FAMILIES['pettingzoo_classic'] should match PETTINGZOO_GAMES keys"


# =============================================================================
# Integration Test: Multi-agent Config from UI Data Pattern
# =============================================================================


class TestMultiAgentConfigFromUIPattern:
    """Test the pattern of creating multi-agent configs from UI widget data.

    This tests the expected usage pattern where:
    1. User selects pettingzoo environment
    2. User configures workers for each player
    3. Widget builds Dict[str, WorkerAssignment]
    4. OperatorConfig.multi_agent() creates the config
    """

    def test_ui_pattern_chess_gpt4_vs_llama(self) -> None:
        """Test creating chess config with different LLMs for each player."""
        # Simulate what the UI widget would do:
        player_0_assignment = WorkerAssignment(
            worker_id="balrog_worker",
            worker_type="llm",
            settings={
                "client_name": "openrouter",
                "model_id": "openai/gpt-4o",
                "api_key": "sk-xxx",
            },
        )
        player_1_assignment = WorkerAssignment(
            worker_id="balrog_worker",
            worker_type="llm",
            settings={
                "client_name": "vllm",
                "model_id": "meta-llama/Llama-3-70b",
                "base_url": "http://localhost:8000/v1",
            },
        )

        # Build the player_workers dict (as PlayerAssignmentPanel would)
        player_workers = {
            "player_0": player_0_assignment,
            "player_1": player_1_assignment,
        }

        # Create the config (as get_config() would)
        config = OperatorConfig.multi_agent(
            operator_id="operator_1",
            display_name="Chess: GPT-4 vs Llama-3",
            env_name="pettingzoo",
            task="chess_v6",
            player_workers=player_workers,
        )

        # Verify the config
        assert config.is_multiagent is True
        assert config.operator_type == "multiagent"
        assert config.env_name == "pettingzoo"
        assert config.task == "chess_v6"

        # Verify player assignments
        assert config.workers["player_0"].settings["model_id"] == "openai/gpt-4o"
        assert config.workers["player_1"].settings["model_id"] == "meta-llama/Llama-3-70b"

    def test_ui_pattern_connect_four_same_model(self) -> None:
        """Test creating connect-four config with same model for both players."""
        # Same worker assignment for both players
        assignment = WorkerAssignment(
            worker_id="balrog_worker",
            worker_type="llm",
            settings={
                "client_name": "openrouter",
                "model_id": "anthropic/claude-3.5-sonnet",
            },
        )

        player_workers = {
            "player_0": assignment,
            "player_1": assignment,  # Same assignment is fine
        }

        config = OperatorConfig.multi_agent(
            operator_id="operator_2",
            display_name="Connect Four Self-Play",
            env_name="pettingzoo",
            task="connect_four_v3",
            player_workers=player_workers,
        )

        assert config.is_multiagent is True
        assert config.task == "connect_four_v3"
        assert len(config.player_ids) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
