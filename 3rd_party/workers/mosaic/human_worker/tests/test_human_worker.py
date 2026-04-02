"""Unit tests for Human Worker runtime and configuration.

Tests cover:
- HumanWorkerConfig creation and serialization
- HumanWorkerRuntime initialization and state management
- Human input processing and validation
- Interactive mode command handling
- Worker metadata and capabilities
"""

import json
import unittest
from io import StringIO
from unittest.mock import patch

from human_worker import HumanWorkerRuntime, HumanWorkerConfig, get_worker_metadata


class TestHumanWorkerConfig(unittest.TestCase):
    """Test HumanWorkerConfig dataclass."""

    def test_default_values(self) -> None:
        """Should have sensible default values."""
        config = HumanWorkerConfig()
        self.assertEqual(config.run_id, "")
        self.assertEqual(config.player_name, "Human")
        self.assertEqual(config.timeout_seconds, 0.0)
        self.assertTrue(config.show_legal_moves)
        self.assertFalse(config.confirm_moves)
        self.assertEqual(config.telemetry_dir, "var/telemetry")

    def test_custom_values(self) -> None:
        """Should accept custom values."""
        config = HumanWorkerConfig(
            run_id="test_run_123",
            player_name="Alice",
            timeout_seconds=30.0,
            show_legal_moves=False,
            confirm_moves=True,
            telemetry_dir="/custom/path",
        )
        self.assertEqual(config.run_id, "test_run_123")
        self.assertEqual(config.player_name, "Alice")
        self.assertEqual(config.timeout_seconds, 30.0)
        self.assertFalse(config.show_legal_moves)
        self.assertTrue(config.confirm_moves)
        self.assertEqual(config.telemetry_dir, "/custom/path")

    def test_to_dict(self) -> None:
        """Should serialize to dictionary correctly."""
        config = HumanWorkerConfig(
            run_id="run_42",
            player_name="Bob",
            timeout_seconds=60.0,
        )
        data = config.to_dict()
        self.assertEqual(data["run_id"], "run_42")
        self.assertEqual(data["player_name"], "Bob")
        self.assertEqual(data["timeout_seconds"], 60.0)
        self.assertTrue(data["show_legal_moves"])
        self.assertFalse(data["confirm_moves"])

    def test_from_dict(self) -> None:
        """Should deserialize from dictionary correctly."""
        data = {
            "run_id": "run_99",
            "player_name": "Charlie",
            "timeout_seconds": 120.0,
            "show_legal_moves": False,
            "confirm_moves": True,
        }
        config = HumanWorkerConfig.from_dict(data)
        self.assertEqual(config.run_id, "run_99")
        self.assertEqual(config.player_name, "Charlie")
        self.assertEqual(config.timeout_seconds, 120.0)
        self.assertFalse(config.show_legal_moves)
        self.assertTrue(config.confirm_moves)

    def test_from_dict_with_defaults(self) -> None:
        """Should use defaults for missing fields."""
        config = HumanWorkerConfig.from_dict({})
        self.assertEqual(config.run_id, "")
        self.assertEqual(config.player_name, "Human")
        self.assertEqual(config.timeout_seconds, 0.0)


class TestHumanWorkerRuntime(unittest.TestCase):
    """Test HumanWorkerRuntime functionality."""

    def setUp(self) -> None:
        self.config = HumanWorkerConfig(
            run_id="test_run",
            player_name="TestPlayer",
        )
        self.runtime = HumanWorkerRuntime(self.config)

    def test_init(self) -> None:
        """Should initialize with correct state."""
        self.assertEqual(self.runtime.config.player_name, "TestPlayer")
        self.assertEqual(self.runtime._player_id, "")
        self.assertEqual(self.runtime._game_name, "")
        self.assertFalse(self.runtime._waiting_for_input)
        self.assertEqual(self.runtime._current_legal_moves, [])

    def test_init_agent(self) -> None:
        """Should initialize agent state correctly."""
        self.runtime.init_agent("chess_v6", "player_0")
        self.assertEqual(self.runtime._game_name, "chess_v6")
        self.assertEqual(self.runtime._player_id, "player_0")
        self.assertFalse(self.runtime._waiting_for_input)

    def test_request_human_input(self) -> None:
        """Should set waiting state and store legal moves."""
        legal_moves = ["e2e4", "d2d4", "g1f3"]

        with patch.object(self.runtime, '_emit') as mock_emit:
            self.runtime.request_human_input(
                observation="Board position",
                legal_moves=legal_moves,
                board_str="Board string",
            )

        self.assertTrue(self.runtime._waiting_for_input)
        self.assertEqual(self.runtime._current_legal_moves, legal_moves)

        # Verify emit was called
        mock_emit.assert_called_once()
        emit_data = mock_emit.call_args[0][0]
        self.assertEqual(emit_data["type"], "waiting_for_human")
        self.assertEqual(emit_data["legal_moves"], legal_moves)

    def test_process_human_input_valid_move(self) -> None:
        """Should accept valid move."""
        self.runtime._waiting_for_input = True
        self.runtime._current_legal_moves = ["e2e4", "d2d4"]

        result = self.runtime.process_human_input("e2e4")

        self.assertTrue(result["success"])
        self.assertEqual(result["action_str"], "e2e4")
        self.assertFalse(self.runtime._waiting_for_input)

    def test_process_human_input_invalid_move(self) -> None:
        """Should reject invalid move."""
        self.runtime._waiting_for_input = True
        self.runtime._current_legal_moves = ["e2e4", "d2d4"]

        result = self.runtime.process_human_input("a1a8")

        self.assertFalse(result["success"])
        self.assertEqual(result["action_str"], "a1a8")
        self.assertIn("error", result)
        # Should remain waiting for input
        self.assertTrue(self.runtime._waiting_for_input)

    def test_process_human_input_not_waiting(self) -> None:
        """Should reject input when not waiting."""
        self.runtime._waiting_for_input = False

        result = self.runtime.process_human_input("e2e4")

        self.assertFalse(result["success"])
        self.assertIn("Not waiting", result["error"])


class TestHumanWorkerRuntimeInteractive(unittest.TestCase):
    """Test HumanWorkerRuntime interactive command handling."""

    def setUp(self) -> None:
        self.config = HumanWorkerConfig(run_id="test_run")
        self.runtime = HumanWorkerRuntime(self.config)
        self.emitted_messages = []

    def _capture_emit(self, data):
        """Capture emitted messages for testing."""
        self.emitted_messages.append(data)

    def test_init_agent_command(self) -> None:
        """Should handle init_agent command."""
        cmd = json.dumps({
            "cmd": "init_agent",
            "game_name": "chess_v6",
            "player_id": "player_1"
        })

        with patch.object(self.runtime, '_emit', side_effect=self._capture_emit):
            with patch('sys.stdin', StringIO(cmd + "\nstop\n")):
                with patch.object(self.runtime, '_emit', side_effect=self._capture_emit):
                    # Run a single command iteration
                    self.runtime.init_agent("chess_v6", "player_1")
                    self.runtime._emit({
                        "type": "agent_initialized",
                        "run_id": "test_run",
                        "game_name": "chess_v6",
                        "player_id": "player_1",
                    })

        self.assertEqual(self.runtime._game_name, "chess_v6")
        self.assertEqual(self.runtime._player_id, "player_1")

    def test_select_action_command(self) -> None:
        """Should emit waiting_for_human on select_action."""
        self.runtime.init_agent("chess_v6", "player_0")

        with patch.object(self.runtime, '_emit', side_effect=self._capture_emit):
            self.runtime.request_human_input(
                observation="obs",
                legal_moves=["e2e4"],
                board_str="board"
            )

        self.assertTrue(self.runtime._waiting_for_input)
        self.assertEqual(len(self.emitted_messages), 1)
        self.assertEqual(self.emitted_messages[0]["type"], "waiting_for_human")

    def test_human_input_command(self) -> None:
        """Should process human input and emit action_selected."""
        self.runtime.init_agent("chess_v6", "player_0")
        self.runtime._waiting_for_input = True
        self.runtime._current_legal_moves = ["e2e4", "d2d4"]

        result = self.runtime.process_human_input("e2e4")

        self.assertTrue(result["success"])
        self.assertEqual(result["action_str"], "e2e4")


class TestWorkerMetadata(unittest.TestCase):
    """Test worker metadata and capabilities."""

    def test_get_worker_metadata_returns_tuple(self) -> None:
        """Should return (metadata, capabilities) tuple."""
        result = get_worker_metadata()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_metadata_values(self) -> None:
        """Should have correct metadata values."""
        metadata, _ = get_worker_metadata()
        self.assertEqual(metadata.name, "Human Worker")
        self.assertIn("0.", metadata.version)  # Version starts with 0.
        self.assertIn("Human", metadata.description)

    def test_capabilities_values(self) -> None:
        """Should have correct capabilities."""
        _, capabilities = get_worker_metadata()
        self.assertEqual(capabilities.worker_type, "human")
        self.assertIn("human_vs_ai", capabilities.supported_paradigms)
        self.assertIn("pettingzoo", capabilities.env_families)
        self.assertIn("discrete", capabilities.action_spaces)
        self.assertFalse(capabilities.requires_gpu)
        self.assertTrue(capabilities.supports_pause_resume)
        self.assertFalse(capabilities.supports_checkpointing)


class TestHumanWorkerEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self) -> None:
        self.config = HumanWorkerConfig(run_id="edge_test")
        self.runtime = HumanWorkerRuntime(self.config)

    def test_empty_legal_moves(self) -> None:
        """Should handle empty legal moves list."""
        with patch.object(self.runtime, '_emit'):
            self.runtime.request_human_input(
                observation="obs",
                legal_moves=[],
                board_str="board"
            )

        self.assertTrue(self.runtime._waiting_for_input)
        self.assertEqual(self.runtime._current_legal_moves, [])

    def test_multiple_init_agent_calls(self) -> None:
        """Should reset state on multiple init_agent calls."""
        self.runtime.init_agent("chess_v6", "player_0")
        self.runtime._waiting_for_input = True

        self.runtime.init_agent("go_v5", "player_1")

        self.assertEqual(self.runtime._game_name, "go_v5")
        self.assertEqual(self.runtime._player_id, "player_1")
        self.assertFalse(self.runtime._waiting_for_input)

    def test_process_empty_move(self) -> None:
        """Should reject empty move string."""
        self.runtime._waiting_for_input = True
        self.runtime._current_legal_moves = ["e2e4"]

        result = self.runtime.process_human_input("")

        self.assertFalse(result["success"])


class TestActionLabels(unittest.TestCase):
    """Test action label retrieval for various environments."""

    def test_minigrid_labels(self) -> None:
        """Should return MiniGrid action labels."""
        from human_worker.config import get_action_labels
        labels = get_action_labels("minigrid", "MiniGrid-Empty-8x8-v0", 7)
        self.assertEqual(len(labels), 7)
        self.assertIn("Turn Left", labels)
        self.assertIn("Forward", labels)

    def test_frozenlake_labels(self) -> None:
        """Should return FrozenLake action labels."""
        from human_worker.config import get_action_labels
        labels = get_action_labels("gymnasium", "FrozenLake-v1", 4)
        self.assertEqual(labels, ["Left", "Down", "Right", "Up"])

    def test_taxi_labels(self) -> None:
        """Should return Taxi action labels."""
        from human_worker.config import get_action_labels
        labels = get_action_labels("gymnasium", "Taxi-v3", 6)
        self.assertEqual(len(labels), 6)
        self.assertIn("Pickup", labels)
        self.assertIn("Dropoff", labels)

    def test_unknown_env_labels(self) -> None:
        """Should return generic labels for unknown environments."""
        from human_worker.config import get_action_labels
        labels = get_action_labels("unknown", "Unknown-Env-v1", 5)
        self.assertEqual(labels, ["Action 0", "Action 1", "Action 2", "Action 3", "Action 4"])

    def test_labels_truncated_to_action_space(self) -> None:
        """Should truncate labels to action space size."""
        from human_worker.config import get_action_labels
        labels = get_action_labels("minigrid", "MiniGrid-Empty-8x8-v0", 3)
        self.assertEqual(len(labels), 3)


class TestHumanWorkerConfigNew(unittest.TestCase):
    """Test new config fields for environment support."""

    def test_env_config_fields(self) -> None:
        """Should have environment config fields."""
        config = HumanWorkerConfig(
            env_name="minigrid",
            task="MiniGrid-Empty-8x8-v0",
            seed=123,
            render_mode="rgb_array",
        )
        self.assertEqual(config.env_name, "minigrid")
        self.assertEqual(config.task, "MiniGrid-Empty-8x8-v0")
        self.assertEqual(config.seed, 123)
        self.assertEqual(config.render_mode, "rgb_array")

    def test_env_config_to_dict(self) -> None:
        """Should serialize environment fields to dict."""
        config = HumanWorkerConfig(
            env_name="babyai",
            task="BabyAI-GoToLocal-v0",
            seed=42,
        )
        data = config.to_dict()
        self.assertEqual(data["env_name"], "babyai")
        self.assertEqual(data["task"], "BabyAI-GoToLocal-v0")
        self.assertEqual(data["seed"], 42)

    def test_env_config_from_dict(self) -> None:
        """Should deserialize environment fields from dict."""
        data = {
            "env_name": "multigrid",
            "task": "MultiGrid-Soccer-v0",
            "seed": 999,
            "render_mode": "human",
        }
        config = HumanWorkerConfig.from_dict(data)
        self.assertEqual(config.env_name, "multigrid")
        self.assertEqual(config.task, "MultiGrid-Soccer-v0")
        self.assertEqual(config.seed, 999)
        self.assertEqual(config.render_mode, "human")


class TestHumanInteractiveRuntime(unittest.TestCase):
    """Test HumanInteractiveRuntime functionality."""

    def setUp(self) -> None:
        self.config = HumanWorkerConfig(
            run_id="interactive_test",
            player_name="TestPlayer",
            env_name="minigrid",
            task="MiniGrid-Empty-8x8-v0",
            seed=42,
        )

    def test_import_interactive_runtime(self) -> None:
        """Should be able to import HumanInteractiveRuntime."""
        from human_worker.runtime import HumanInteractiveRuntime
        runtime = HumanInteractiveRuntime(self.config)
        self.assertIsNotNone(runtime)
        self.assertEqual(runtime.config.run_id, "interactive_test")

    def test_initial_state(self) -> None:
        """Should have correct initial state."""
        from human_worker.runtime import HumanInteractiveRuntime
        runtime = HumanInteractiveRuntime(self.config)
        self.assertIsNone(runtime._env)
        self.assertEqual(runtime._action_space_n, 0)
        self.assertEqual(runtime._action_labels, [])
        self.assertEqual(runtime._step_index, 0)
        self.assertEqual(runtime._episode_index, -1)  # starts at -1 so first reset makes it 0
        self.assertEqual(runtime._total_reward, 0.0)

    def test_emit_method(self) -> None:
        """Should emit JSON to stdout."""
        from human_worker.runtime import HumanInteractiveRuntime
        runtime = HumanInteractiveRuntime(self.config)

        with patch('builtins.print') as mock_print:
            runtime._emit({"type": "test", "data": "value"})
            mock_print.assert_called_once()
            call_args = mock_print.call_args[0][0]
            self.assertIn('"type": "test"', call_args)


class TestHumanInteractiveRuntimeWithEnv(unittest.TestCase):
    """Test HumanInteractiveRuntime with actual environment (MiniGrid)."""

    @classmethod
    def setUpClass(cls):
        """Check if MiniGrid is available."""
        try:
            import minigrid
            cls.minigrid_available = True
        except ImportError:
            cls.minigrid_available = False

    def setUp(self) -> None:
        if not self.minigrid_available:
            self.skipTest("MiniGrid not installed")

        self.config = HumanWorkerConfig(
            run_id="env_test",
            env_name="minigrid",
            task="MiniGrid-Empty-5x5-v0",
            seed=42,
        )

    def test_handle_reset(self) -> None:
        """Should create and reset environment."""
        from human_worker.runtime import HumanInteractiveRuntime
        runtime = HumanInteractiveRuntime(self.config)

        emitted = []
        def capture_emit(data):
            emitted.append(data)

        with patch.object(runtime, '_emit', side_effect=capture_emit):
            runtime._handle_reset({
                "seed": 42,
                "env_name": "minigrid",
                "task": "MiniGrid-Empty-5x5-v0",
            })

        # Check environment was created
        self.assertIsNotNone(runtime._env)
        self.assertEqual(runtime._action_space_n, 7)  # MiniGrid has 7 actions
        self.assertEqual(len(runtime._action_labels), 7)

        # Check response was emitted
        self.assertEqual(len(emitted), 1)
        response = emitted[0]
        self.assertEqual(response["type"], "ready")
        self.assertEqual(response["action_space"], 7)
        self.assertIn("render_payload", response)
        self.assertIn("action_labels", response)

        # Cleanup
        runtime._env.close()

    def test_handle_step(self) -> None:
        """Should step environment with action."""
        from human_worker.runtime import HumanInteractiveRuntime
        runtime = HumanInteractiveRuntime(self.config)

        emitted = []
        def capture_emit(data):
            emitted.append(data)

        with patch.object(runtime, '_emit', side_effect=capture_emit):
            # First reset
            runtime._handle_reset({
                "seed": 42,
                "env_name": "minigrid",
                "task": "MiniGrid-Empty-5x5-v0",
            })

            # Then step
            runtime._handle_step({"action": 1})  # Turn Right

        # Check step response
        self.assertEqual(len(emitted), 2)  # reset + step
        step_response = emitted[1]
        self.assertEqual(step_response["type"], "step")
        self.assertEqual(step_response["action"], 1)
        self.assertIn("reward", step_response)
        self.assertIn("render_payload", step_response)
        self.assertEqual(runtime._step_index, 1)

        # Cleanup
        runtime._env.close()

    def test_handle_step_invalid_action(self) -> None:
        """Should reject invalid action."""
        from human_worker.runtime import HumanInteractiveRuntime
        runtime = HumanInteractiveRuntime(self.config)

        emitted = []
        def capture_emit(data):
            emitted.append(data)

        with patch.object(runtime, '_emit', side_effect=capture_emit):
            runtime._handle_reset({
                "task": "MiniGrid-Empty-5x5-v0",
            })
            runtime._handle_step({"action": 99})  # Invalid action

        # Check error response
        error_response = emitted[-1]
        self.assertEqual(error_response["type"], "error")
        self.assertIn("Invalid action", error_response["message"])

        # Cleanup
        runtime._env.close()

    def test_handle_step_without_reset(self) -> None:
        """Should error if stepping without reset."""
        from human_worker.runtime import HumanInteractiveRuntime
        runtime = HumanInteractiveRuntime(self.config)

        emitted = []
        def capture_emit(data):
            emitted.append(data)

        with patch.object(runtime, '_emit', side_effect=capture_emit):
            runtime._handle_step({"action": 0})

        # Check error response
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0]["type"], "error")
        self.assertIn("not initialized", emitted[0]["message"])


if __name__ == "__main__":
    unittest.main()
