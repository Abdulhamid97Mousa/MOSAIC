"""Unit tests for the Multi-Operator system (Phase 6).

Tests cover:
- OperatorConfig dataclass validation (updated for multi-worker support)
- MultiOperatorService operator management
- MultiOperatorService state management
- MultiOperatorService run_id management
- MultiOperatorService lifecycle helpers (start_all, stop_all)

Note: These tests have been updated to use the new OperatorConfig.single_agent()
factory method instead of the legacy constructor parameters.
"""

import unittest
from typing import Any, Dict

from gym_gui.services.operator import (
    MultiOperatorService,
    OperatorConfig,
    WorkerAssignment,
)


class TestOperatorConfig(unittest.TestCase):
    """Test OperatorConfig dataclass.

    Updated to use OperatorConfig.single_agent() factory method.
    """

    def test_create_llm_config(self) -> None:
        """Should create a valid LLM operator config using factory method.

        Uses OperatorConfig.single_agent() to create a single-agent LLM config.
        Verifies all fields are correctly set and accessible via backwards-compatible properties.
        """
        config = OperatorConfig.single_agent(
            operator_id="op_0",
            display_name="GPT-4 LLM",
            worker_id="balrog_worker",
            worker_type="llm",
            env_name="babyai",
            task="BabyAI-GoToRedBall-v0",
        )
        self.assertEqual(config.operator_id, "op_0")
        self.assertEqual(config.operator_type, "llm")  # Via property
        self.assertEqual(config.worker_id, "balrog_worker")  # Via property
        self.assertEqual(config.display_name, "GPT-4 LLM")
        self.assertEqual(config.env_name, "babyai")
        self.assertEqual(config.task, "BabyAI-GoToRedBall-v0")
        self.assertIsNone(config.run_id)

    def test_create_rl_config(self) -> None:
        """Should create a valid RL operator config using factory method.

        RL configs typically have algorithm and policy settings.
        """
        config = OperatorConfig.single_agent(
            operator_id="op_1",
            display_name="PPO Agent",
            worker_id="cleanrl_worker",
            worker_type="rl",
            env_name="CartPole-v1",
            task="CartPole-v1",
            settings={"algorithm": "ppo", "learning_rate": 0.0003},
        )
        self.assertEqual(config.operator_type, "rl")  # Via property
        self.assertEqual(config.worker_id, "cleanrl_worker")  # Via property
        self.assertEqual(config.settings["algorithm"], "ppo")  # Via property

    def test_invalid_worker_type_raises_error(self) -> None:
        """Should raise ValueError for invalid worker_type.

        Note: Validation is now done in WorkerAssignment, not OperatorConfig.
        Valid types are: 'llm', 'vlm', 'rl', 'human'.
        """
        with self.assertRaises(ValueError) as ctx:
            # Create WorkerAssignment with invalid type
            WorkerAssignment(
                worker_id="test",
                worker_type="invalid",
                settings={},
            )
        self.assertIn("worker_type must be one of", str(ctx.exception))

    def test_with_run_id(self) -> None:
        """Should create a copy with the run_id set.

        with_run_id() creates a deep copy of the config with run_id assigned.
        Original config should remain unchanged.
        """
        config = OperatorConfig.single_agent(
            operator_id="op_0",
            display_name="GPT-4 LLM",
            worker_id="balrog_worker",
            worker_type="llm",
        )
        new_config = config.with_run_id("run_abc123")

        # Original should be unchanged
        self.assertIsNone(config.run_id)

        # New config should have run_id
        self.assertEqual(new_config.run_id, "run_abc123")
        self.assertEqual(new_config.operator_id, config.operator_id)
        self.assertEqual(new_config.operator_type, config.operator_type)

    def test_default_env_and_task(self) -> None:
        """Should have sensible defaults for env_name and task.

        Default environment is 'babyai' with 'BabyAI-GoToRedBall-v0' task.
        """
        config = OperatorConfig.single_agent(
            operator_id="op_0",
            display_name="Test LLM",
            worker_id="balrog_worker",
            worker_type="llm",
        )
        self.assertEqual(config.env_name, "babyai")
        self.assertEqual(config.task, "BabyAI-GoToRedBall-v0")

    def test_settings_default_to_empty_dict(self) -> None:
        """Settings should default to empty dict when not provided.

        The settings property reads from workers["agent"].settings.
        """
        config = OperatorConfig.single_agent(
            operator_id="op_0",
            display_name="Test LLM",
            worker_id="balrog_worker",
            worker_type="llm",
        )
        self.assertEqual(config.settings, {})


class TestMultiOperatorServiceBasic(unittest.TestCase):
    """Test MultiOperatorService basic functionality.

    Tests add, remove, update, and clear operations.
    """

    def setUp(self) -> None:
        """Create a fresh MultiOperatorService for each test."""
        self.service = MultiOperatorService()

    def test_empty_service(self) -> None:
        """New service should have no operators.

        Verifies initial state of the service is empty.
        """
        self.assertEqual(self.service.operator_count, 0)
        self.assertEqual(self.service.get_active_operators(), {})
        self.assertEqual(self.service.get_operator_ids(), [])

    def test_add_operator(self) -> None:
        """Should add an operator successfully.

        Verifies operator is stored and retrievable by ID.
        """
        config = OperatorConfig.single_agent(
            operator_id="op_0",
            display_name="GPT-4 LLM",
            worker_id="balrog_worker",
            worker_type="llm",
        )
        self.service.add_operator(config)

        self.assertEqual(self.service.operator_count, 1)
        self.assertIn("op_0", self.service.get_operator_ids())

        retrieved = self.service.get_operator("op_0")
        self.assertIsNotNone(retrieved)
        assert retrieved is not None
        self.assertEqual(retrieved.display_name, "GPT-4 LLM")

    def test_add_multiple_operators(self) -> None:
        """Should manage multiple operators.

        Tests adding both LLM and RL operators to the same service.
        """
        config1 = OperatorConfig.single_agent(
            operator_id="llm_1",
            display_name="LLM Agent 1",
            worker_id="balrog_worker",
            worker_type="llm",
        )
        config2 = OperatorConfig.single_agent(
            operator_id="rl_1",
            display_name="RL Agent 1",
            worker_id="cleanrl_worker",
            worker_type="rl",
        )

        self.service.add_operator(config1)
        self.service.add_operator(config2)

        self.assertEqual(self.service.operator_count, 2)
        self.assertIn("llm_1", self.service.get_operator_ids())
        self.assertIn("rl_1", self.service.get_operator_ids())

    def test_remove_operator(self) -> None:
        """Should remove an operator.

        Verifies operator is no longer retrievable after removal.
        """
        config = OperatorConfig.single_agent(
            operator_id="op_0",
            display_name="Test",
            worker_id="balrog_worker",
            worker_type="llm",
        )
        self.service.add_operator(config)
        self.assertEqual(self.service.operator_count, 1)

        self.service.remove_operator("op_0")

        self.assertEqual(self.service.operator_count, 0)
        self.assertIsNone(self.service.get_operator("op_0"))

    def test_remove_nonexistent_operator(self) -> None:
        """Should not raise when removing nonexistent operator.

        Removing a non-existent ID should be a no-op.
        """
        # Should not raise
        self.service.remove_operator("nonexistent")
        self.assertEqual(self.service.operator_count, 0)

    def test_update_operator(self) -> None:
        """Should update an existing operator.

        The update replaces the config while keeping the same operator_id.
        """
        config1 = OperatorConfig.single_agent(
            operator_id="op_0",
            display_name="Original Name",
            worker_id="balrog_worker",
            worker_type="llm",
        )
        self.service.add_operator(config1)

        config2 = OperatorConfig.single_agent(
            operator_id="op_0",
            display_name="Updated Name",
            worker_id="balrog_worker",
            worker_type="llm",
        )
        self.service.update_operator(config2)

        retrieved = self.service.get_operator("op_0")
        assert retrieved is not None
        self.assertEqual(retrieved.display_name, "Updated Name")

    def test_clear_operators(self) -> None:
        """Should clear all operators.

        After clear, service should be empty.
        """
        config1 = OperatorConfig.single_agent(
            operator_id="op_0",
            display_name="LLM 1",
            worker_id="balrog_worker",
            worker_type="llm",
        )
        config2 = OperatorConfig.single_agent(
            operator_id="op_1",
            display_name="RL 1",
            worker_id="cleanrl_worker",
            worker_type="rl",
        )

        self.service.add_operator(config1)
        self.service.add_operator(config2)
        self.assertEqual(self.service.operator_count, 2)

        self.service.clear_operators()

        self.assertEqual(self.service.operator_count, 0)
        self.assertEqual(self.service.get_active_operators(), {})


class TestMultiOperatorServiceIdGeneration(unittest.TestCase):
    """Test MultiOperatorService operator ID generation.

    IDs should be unique and follow 'operator_N' format.
    """

    def setUp(self) -> None:
        """Create a fresh MultiOperatorService for each test."""
        self.service = MultiOperatorService()

    def test_generate_unique_ids(self) -> None:
        """Should generate unique operator IDs.

        Each call to generate_operator_id() returns a different ID.
        """
        id1 = self.service.generate_operator_id()
        id2 = self.service.generate_operator_id()
        id3 = self.service.generate_operator_id()

        self.assertNotEqual(id1, id2)
        self.assertNotEqual(id2, id3)
        self.assertNotEqual(id1, id3)

    def test_id_format(self) -> None:
        """Generated IDs should follow expected format.

        Format is 'operator_N' where N is an incrementing integer.
        """
        id1 = self.service.generate_operator_id()
        self.assertTrue(id1.startswith("operator_"))

        # Verify it contains a number
        parts = id1.split("_")
        self.assertEqual(len(parts), 2)
        self.assertTrue(parts[1].isdigit())


class TestMultiOperatorServiceStateManagement(unittest.TestCase):
    """Test MultiOperatorService state management.

    Operators can be in states: pending, running, stopped, error.
    """

    def setUp(self) -> None:
        """Create service with one operator for state tests."""
        self.service = MultiOperatorService()
        self.config = OperatorConfig.single_agent(
            operator_id="op_0",
            display_name="Test",
            worker_id="balrog_worker",
            worker_type="llm",
        )
        self.service.add_operator(self.config)

    def test_initial_state_is_pending(self) -> None:
        """New operators should start in pending state.

        When added, operators are not yet running.
        """
        state = self.service.get_operator_state("op_0")
        self.assertEqual(state, "pending")

    def test_set_operator_state(self) -> None:
        """Should set operator state.

        State can be changed between valid states.
        """
        self.service.set_operator_state("op_0", "running")
        self.assertEqual(self.service.get_operator_state("op_0"), "running")

        self.service.set_operator_state("op_0", "stopped")
        self.assertEqual(self.service.get_operator_state("op_0"), "stopped")

    def test_invalid_state_raises(self) -> None:
        """Setting invalid state should raise ValueError.

        Only 'pending', 'running', 'stopped', 'error' are valid.
        """
        with self.assertRaises(ValueError):
            self.service.set_operator_state("op_0", "invalid_state")

    def test_get_running_operators(self) -> None:
        """Should return only running operators.

        Filters operators by 'running' state.
        """
        config2 = OperatorConfig.single_agent(
            operator_id="op_1",
            display_name="RL Agent",
            worker_id="cleanrl_worker",
            worker_type="rl",
        )
        config3 = OperatorConfig.single_agent(
            operator_id="op_2",
            display_name="LLM 2",
            worker_id="balrog_worker",
            worker_type="llm",
        )

        self.service.add_operator(config2)
        self.service.add_operator(config3)

        # Set some to running
        self.service.set_operator_state("op_0", "running")
        self.service.set_operator_state("op_2", "running")
        # op_1 stays pending

        running = self.service.get_running_operators()

        self.assertEqual(len(running), 2)
        self.assertIn("op_0", running)
        self.assertIn("op_2", running)
        self.assertNotIn("op_1", running)


class TestMultiOperatorServiceRunIdManagement(unittest.TestCase):
    """Test MultiOperatorService run_id management.

    Run IDs are assigned when operators are launched for telemetry routing.
    """

    def setUp(self) -> None:
        """Create service with one operator for run_id tests."""
        self.service = MultiOperatorService()
        self.config = OperatorConfig.single_agent(
            operator_id="op_0",
            display_name="Test",
            worker_id="balrog_worker",
            worker_type="llm",
        )
        self.service.add_operator(self.config)

    def test_assign_run_id(self) -> None:
        """Should assign run_id to operator.

        The run_id is used for routing telemetry events.
        """
        self.service.assign_run_id("op_0", "run_abc123")

        self.assertEqual(self.service.get_run_id("op_0"), "run_abc123")

        # Config should also be updated with run_id
        updated_config = self.service.get_operator("op_0")
        assert updated_config is not None
        self.assertEqual(updated_config.run_id, "run_abc123")

    def test_get_operator_by_run_id(self) -> None:
        """Should retrieve operator by run_id.

        Allows looking up which operator a telemetry event belongs to.
        """
        self.service.assign_run_id("op_0", "run_abc123")

        config = self.service.get_operator_by_run_id("run_abc123")

        self.assertIsNotNone(config)
        assert config is not None
        self.assertEqual(config.operator_id, "op_0")

    def test_get_operator_by_unknown_run_id(self) -> None:
        """Should return None for unknown run_id.

        Unknown run IDs should not raise errors.
        """
        config = self.service.get_operator_by_run_id("nonexistent_run")
        self.assertIsNone(config)


class TestMultiOperatorServiceLifecycle(unittest.TestCase):
    """Test MultiOperatorService lifecycle helpers.

    Tests start_all and stop_all operations.
    """

    def setUp(self) -> None:
        """Create service with 3 operators for lifecycle tests."""
        self.service = MultiOperatorService()
        # Add multiple operators with alternating types
        for i in range(3):
            config = OperatorConfig.single_agent(
                operator_id=f"op_{i}",
                display_name=f"Agent {i}",
                worker_id="balrog_worker" if i % 2 == 0 else "cleanrl_worker",
                worker_type="llm" if i % 2 == 0 else "rl",
            )
            self.service.add_operator(config)

    def test_start_all_returns_pending(self) -> None:
        """start_all should return all pending operators.

        Returns list of operator IDs ready to start.
        """
        to_start = self.service.start_all()

        self.assertEqual(len(to_start), 3)
        self.assertIn("op_0", to_start)
        self.assertIn("op_1", to_start)
        self.assertIn("op_2", to_start)

    def test_start_all_with_some_running(self) -> None:
        """start_all should only return pending operators.

        Already running operators are excluded.
        """
        # Set one to running
        self.service.set_operator_state("op_1", "running")

        to_start = self.service.start_all()

        self.assertEqual(len(to_start), 2)
        self.assertIn("op_0", to_start)
        self.assertIn("op_2", to_start)
        self.assertNotIn("op_1", to_start)

    def test_stop_all_running(self) -> None:
        """stop_all should stop all running operators.

        All running operators should transition to stopped state.
        """
        # Set all to running
        for i in range(3):
            self.service.set_operator_state(f"op_{i}", "running")

        stopped = self.service.stop_all()

        self.assertEqual(len(stopped), 3)

        # All should now be stopped
        for i in range(3):
            self.assertEqual(
                self.service.get_operator_state(f"op_{i}"),
                "stopped"
            )

    def test_stop_all_with_mixed_states(self) -> None:
        """stop_all should only stop running operators.

        Pending and already stopped operators are not affected.
        """
        # Mixed states
        self.service.set_operator_state("op_0", "running")
        self.service.set_operator_state("op_1", "stopped")
        self.service.set_operator_state("op_2", "running")

        stopped = self.service.stop_all()

        self.assertEqual(len(stopped), 2)
        self.assertIn("op_0", stopped)
        self.assertIn("op_2", stopped)
        self.assertNotIn("op_1", stopped)


class TestMultiOperatorServiceRemovalCleansState(unittest.TestCase):
    """Test that removing an operator cleans up all associated state.

    Removal should clean up: run_id mapping, operator state, and config.
    """

    def setUp(self) -> None:
        """Create operator with run_id and running state."""
        self.service = MultiOperatorService()
        config = OperatorConfig.single_agent(
            operator_id="op_0",
            display_name="Test",
            worker_id="balrog_worker",
            worker_type="llm",
        )
        self.service.add_operator(config)
        self.service.assign_run_id("op_0", "run_abc123")
        self.service.set_operator_state("op_0", "running")

    def test_removal_cleans_run_id(self) -> None:
        """Removing operator should clean up run_id mapping.

        After removal, run_id lookups should return None.
        """
        self.service.remove_operator("op_0")

        self.assertIsNone(self.service.get_run_id("op_0"))
        self.assertIsNone(self.service.get_operator_by_run_id("run_abc123"))

    def test_removal_cleans_state(self) -> None:
        """Removing operator should clean up state.

        After removal, state lookup should return None.
        """
        self.service.remove_operator("op_0")

        self.assertIsNone(self.service.get_operator_state("op_0"))


if __name__ == "__main__":
    unittest.main()
