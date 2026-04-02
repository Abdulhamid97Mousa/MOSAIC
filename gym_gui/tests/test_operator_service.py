"""Unit tests for the Operator protocol and OperatorService.

Tests cover:
- Operator protocol compliance
- OperatorService registration and activation
- Action selection delegation
- Seeding propagation
- HumanOperator and WorkerOperator implementations
"""

import unittest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from gym_gui.services.operator import (
    HumanOperator,
    Operator,
    OperatorDescriptor,
    OperatorService,
    WorkerOperator,
)


@dataclass
class MockOperator:
    """Mock operator for testing OperatorService."""

    id: str
    name: str
    _actions: List[Any] = None  # type: ignore[assignment]
    _action_index: int = 0
    _reset_called: bool = False
    _reset_seed: Optional[int] = None
    _step_results: List[tuple] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._actions is None:
            self._actions = [0]  # Default action
        if self._step_results is None:
            self._step_results = []

    def select_action(
        self,
        observation: Any,
        info: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Return predetermined action for testing."""
        if self._action_index < len(self._actions):
            action = self._actions[self._action_index]
            self._action_index += 1
            return action
        return None

    def reset(self, seed: Optional[int] = None) -> None:
        """Track reset calls for verification."""
        self._reset_called = True
        self._reset_seed = seed
        self._action_index = 0

    def on_step_result(
        self,
        observation: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
    ) -> None:
        """Record step results for verification."""
        self._step_results.append((observation, reward, terminated, truncated, info))


class TestOperatorProtocol(unittest.TestCase):
    """Test that implementations satisfy the Operator protocol."""

    def test_mock_operator_satisfies_protocol(self) -> None:
        """MockOperator should satisfy Operator protocol."""
        operator: Operator = MockOperator(id="test", name="Test")
        self.assertEqual(operator.id, "test")
        self.assertEqual(operator.name, "Test")
        self.assertIsNotNone(operator.select_action({}, None))

    def test_human_operator_satisfies_protocol(self) -> None:
        """HumanOperator should satisfy Operator protocol."""
        operator: Operator = HumanOperator()
        self.assertEqual(operator.id, "human_keyboard")
        self.assertEqual(operator.name, "Human (Keyboard)")
        # Human operator returns None (action comes from UI)
        self.assertIsNone(operator.select_action({}, None))

    def test_worker_operator_satisfies_protocol(self) -> None:
        """WorkerOperator should satisfy Operator protocol."""
        operator: Operator = WorkerOperator(
            id="test_worker",
            name="Test Worker",
            worker_id="worker_123",
        )
        self.assertEqual(operator.id, "test_worker")
        self.assertEqual(operator.name, "Test Worker")
        # Worker operator returns None (worker handles actions)
        self.assertIsNone(operator.select_action({}, None))


class TestOperatorService(unittest.TestCase):
    """Test OperatorService registration and activation."""

    def setUp(self) -> None:
        self.service = OperatorService()

    def test_register_operator(self) -> None:
        """Should register an operator with metadata."""
        operator = MockOperator(id="mock_1", name="Mock One")
        self.service.register_operator(
            operator,
            display_name="Mock Operator 1",
            description="A mock operator for testing",
            category="test",
        )

        self.assertIn("mock_1", self.service.available_operator_ids())
        descriptor = self.service.get_operator_descriptor("mock_1")
        self.assertIsNotNone(descriptor)
        assert descriptor is not None
        self.assertEqual(descriptor.display_name, "Mock Operator 1")
        self.assertEqual(descriptor.description, "A mock operator for testing")
        self.assertEqual(descriptor.category, "test")

    def test_first_operator_becomes_active(self) -> None:
        """First registered operator should become active by default."""
        operator = MockOperator(id="first", name="First")
        self.service.register_operator(operator)

        self.assertEqual(self.service.get_active_operator_id(), "first")
        self.assertIs(self.service.get_active_operator(), operator)

    def test_explicit_activation(self) -> None:
        """activate=True should make operator active."""
        op1 = MockOperator(id="op1", name="Op 1")
        op2 = MockOperator(id="op2", name="Op 2")

        self.service.register_operator(op1)
        self.service.register_operator(op2, activate=True)

        self.assertEqual(self.service.get_active_operator_id(), "op2")

    def test_set_active_operator(self) -> None:
        """Should be able to change active operator."""
        op1 = MockOperator(id="op1", name="Op 1")
        op2 = MockOperator(id="op2", name="Op 2")

        self.service.register_operator(op1)
        self.service.register_operator(op2)

        self.service.set_active_operator("op2")
        self.assertEqual(self.service.get_active_operator_id(), "op2")

        self.service.set_active_operator("op1")
        self.assertEqual(self.service.get_active_operator_id(), "op1")

    def test_set_active_operator_invalid_id(self) -> None:
        """Setting unknown operator ID should raise KeyError."""
        with self.assertRaises(KeyError):
            self.service.set_active_operator("nonexistent")

    def test_describe_operators(self) -> None:
        """Should return descriptors in registration order."""
        op1 = MockOperator(id="alpha", name="Alpha")
        op2 = MockOperator(id="beta", name="Beta")
        op3 = MockOperator(id="gamma", name="Gamma")

        self.service.register_operator(op1, display_name="A")
        self.service.register_operator(op2, display_name="B")
        self.service.register_operator(op3, display_name="C")

        descriptors = self.service.describe_operators()
        self.assertEqual(len(descriptors), 3)
        self.assertEqual(descriptors[0].operator_id, "alpha")
        self.assertEqual(descriptors[1].operator_id, "beta")
        self.assertEqual(descriptors[2].operator_id, "gamma")


class TestOperatorServiceActionSelection(unittest.TestCase):
    """Test action selection delegation."""

    def setUp(self) -> None:
        self.service = OperatorService()
        self.operator = MockOperator(
            id="actor",
            name="Actor",
            _actions=[1, 2, 3],
        )
        self.service.register_operator(self.operator)

    def test_select_action_delegates(self) -> None:
        """Action selection should delegate to active operator."""
        action = self.service.select_action({"obs": 1}, {"info": "test"})
        self.assertEqual(action, 1)

        action = self.service.select_action({"obs": 2})
        self.assertEqual(action, 2)

    def test_select_action_no_active_operator(self) -> None:
        """Should return None when no operator is registered."""
        empty_service = OperatorService()
        action = empty_service.select_action({})
        self.assertIsNone(action)

    def test_notify_step_result(self) -> None:
        """Should forward step results to active operator."""
        self.service.notify_step_result(
            observation={"new_obs": 1},
            reward=10.0,
            terminated=False,
            truncated=False,
            info={"extra": "data"},
        )

        self.assertEqual(len(self.operator._step_results), 1)
        obs, reward, term, trunc, info = self.operator._step_results[0]
        self.assertEqual(obs, {"new_obs": 1})
        self.assertEqual(reward, 10.0)
        self.assertFalse(term)
        self.assertFalse(trunc)
        self.assertEqual(info, {"extra": "data"})


class TestOperatorServiceSeeding(unittest.TestCase):
    """Test seed propagation to operators."""

    def setUp(self) -> None:
        self.service = OperatorService()
        self.op1 = MockOperator(id="op1", name="Op 1")
        self.op2 = MockOperator(id="op2", name="Op 2")
        self.service.register_operator(self.op1)
        self.service.register_operator(self.op2)

    def test_seed_propagates_to_all_operators(self) -> None:
        """Seeding should propagate to all registered operators."""
        self.service.seed(42)

        self.assertTrue(self.op1._reset_called)
        self.assertEqual(self.op1._reset_seed, 42)
        self.assertTrue(self.op2._reset_called)
        self.assertEqual(self.op2._reset_seed, 42)

    def test_last_seed_is_stored(self) -> None:
        """Last seed should be accessible."""
        self.assertIsNone(self.service.last_seed)

        self.service.seed(123)
        self.assertEqual(self.service.last_seed, 123)

        self.service.seed(456)
        self.assertEqual(self.service.last_seed, 456)

    def test_reset_active_operator(self) -> None:
        """Should reset only the active operator."""
        # Reset state
        self.op1._reset_called = False
        self.op2._reset_called = False

        self.service.set_active_operator("op2")
        self.service.reset_active_operator(seed=99)

        self.assertFalse(self.op1._reset_called)
        self.assertTrue(self.op2._reset_called)
        self.assertEqual(self.op2._reset_seed, 99)


class TestOperatorDescriptor(unittest.TestCase):
    """Test OperatorDescriptor dataclass."""

    def test_descriptor_defaults(self) -> None:
        """Descriptor should have sensible defaults."""
        desc = OperatorDescriptor(
            operator_id="test",
            display_name="Test",
        )
        self.assertEqual(desc.operator_id, "test")
        self.assertEqual(desc.display_name, "Test")
        self.assertIsNone(desc.description)
        self.assertEqual(desc.category, "default")
        self.assertFalse(desc.supports_training)
        self.assertFalse(desc.requires_api_key)

    def test_descriptor_is_frozen(self) -> None:
        """Descriptor should be immutable (frozen dataclass)."""
        desc = OperatorDescriptor(
            operator_id="test",
            display_name="Test",
        )
        with self.assertRaises(AttributeError):
            desc.operator_id = "changed"  # type: ignore[misc]


class TestHumanOperator(unittest.TestCase):
    """Test HumanOperator implementation."""

    def setUp(self) -> None:
        self.operator = HumanOperator()

    def test_default_values(self) -> None:
        """Should have correct default ID and name."""
        self.assertEqual(self.operator.id, "human_keyboard")
        self.assertEqual(self.operator.name, "Human (Keyboard)")

    def test_select_action_returns_none(self) -> None:
        """Human action comes from UI, not operator."""
        action = self.operator.select_action({"obs": 1}, {"info": 1})
        self.assertIsNone(action)

    def test_reset_is_noop(self) -> None:
        """Reset should be a no-op for human input."""
        # Should not raise
        self.operator.reset(seed=42)

    def test_on_step_result_is_noop(self) -> None:
        """Step result should be a no-op for human input."""
        # Should not raise
        self.operator.on_step_result({}, 0.0, False, False, {})


class TestWorkerOperator(unittest.TestCase):
    """Test WorkerOperator implementation."""

    def setUp(self) -> None:
        self.operator = WorkerOperator(
            id="balrog_llm",
            name="BALROG LLM",
            worker_id="balrog_worker",
        )

    def test_custom_values(self) -> None:
        """Should store custom ID, name, and worker_id."""
        self.assertEqual(self.operator.id, "balrog_llm")
        self.assertEqual(self.operator.name, "BALROG LLM")
        self.assertEqual(self.operator.worker_id, "balrog_worker")

    def test_select_action_returns_none(self) -> None:
        """Worker handles its own action selection."""
        action = self.operator.select_action({"obs": 1})
        self.assertIsNone(action)

    def test_reset_is_noop(self) -> None:
        """Reset should be a no-op for worker operator."""
        # Should not raise
        self.operator.reset(seed=42)

    def test_on_step_result_is_noop(self) -> None:
        """Step result should be a no-op for worker operator."""
        # Should not raise
        self.operator.on_step_result({}, 0.0, False, False, {})


class TestOperatorServiceMetadata(unittest.TestCase):
    """Test metadata handling for operators."""

    def setUp(self) -> None:
        self.service = OperatorService()

    def test_display_name_fallback(self) -> None:
        """Should use operator.name as fallback for display_name."""
        operator = MockOperator(id="test_op", name="Test Operator")
        self.service.register_operator(operator)

        desc = self.service.get_operator_descriptor("test_op")
        assert desc is not None
        self.assertEqual(desc.display_name, "Test Operator")

    def test_explicit_display_name(self) -> None:
        """Explicit display_name should override operator.name."""
        operator = MockOperator(id="test_op", name="Original Name")
        self.service.register_operator(operator, display_name="Custom Name")

        desc = self.service.get_operator_descriptor("test_op")
        assert desc is not None
        self.assertEqual(desc.display_name, "Custom Name")

    def test_category_metadata(self) -> None:
        """Should store category metadata."""
        human = HumanOperator()
        worker = WorkerOperator(id="w1", name="W1", worker_id="worker_1")

        self.service.register_operator(human, category="human")
        self.service.register_operator(worker, category="worker")

        desc_human = self.service.get_operator_descriptor("human_keyboard")
        assert desc_human is not None
        self.assertEqual(desc_human.category, "human")
        desc_w1 = self.service.get_operator_descriptor("w1")
        assert desc_w1 is not None
        self.assertEqual(desc_w1.category, "worker")

    def test_supports_training_metadata(self) -> None:
        """Should store supports_training metadata."""
        op1 = MockOperator(id="op1", name="No Training")
        op2 = MockOperator(id="op2", name="With Training")

        self.service.register_operator(op1, supports_training=False)
        self.service.register_operator(op2, supports_training=True)

        desc_op1 = self.service.get_operator_descriptor("op1")
        assert desc_op1 is not None
        self.assertFalse(desc_op1.supports_training)
        desc_op2 = self.service.get_operator_descriptor("op2")
        assert desc_op2 is not None
        self.assertTrue(desc_op2.supports_training)

    def test_requires_api_key_metadata(self) -> None:
        """Should store requires_api_key metadata."""
        local_op = MockOperator(id="local", name="Local")
        api_op = MockOperator(id="api", name="API")

        self.service.register_operator(local_op, requires_api_key=False)
        self.service.register_operator(api_op, requires_api_key=True)

        desc_local = self.service.get_operator_descriptor("local")
        assert desc_local is not None
        self.assertFalse(desc_local.requires_api_key)
        desc_api = self.service.get_operator_descriptor("api")
        assert desc_api is not None
        self.assertTrue(desc_api.requires_api_key)


if __name__ == "__main__":
    unittest.main()
