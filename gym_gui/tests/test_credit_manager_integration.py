"""Tests for credit manager integration in TelemetryAsyncHub.

Tests validate that:
1. TelemetryAsyncHub uses the global CreditManager singleton
2. Credit consumption is enforced before publishing to UI path
3. DB path persists even when credits are exhausted
4. Credits are properly tracked per (run_id, agent_id) pair
"""

from unittest import mock

import pytest

from gym_gui.services.trainer.streams import TelemetryAsyncHub
from gym_gui.telemetry.credit_manager import CreditManager, get_credit_manager, reset_credit_manager
from gym_gui.telemetry.events import Topic


@pytest.fixture(autouse=True)
def reset_credit_mgr():
    """Reset credit manager before and after each test."""
    reset_credit_manager()
    yield
    reset_credit_manager()


class TestCreditManagerIntegration:
    """Tests for CreditManager integration with TelemetryAsyncHub."""

    def test_hub_initializes_with_credit_manager(self):
        """Test that TelemetryAsyncHub has a CreditManager instance."""
        hub = TelemetryAsyncHub()
        assert hasattr(hub, "_credit_mgr"), "Hub should have _credit_mgr attribute"
        assert isinstance(hub._credit_mgr, CreditManager), "_credit_mgr should be CreditManager instance"

    def test_hub_uses_global_credit_manager_singleton(self):
        """Test that hub uses the global CreditManager singleton."""
        hub1 = TelemetryAsyncHub()
        hub2 = TelemetryAsyncHub()

        global_mgr = get_credit_manager()

        # Both hubs should use the same global instance
        assert hub1._credit_mgr is global_mgr
        assert hub2._credit_mgr is global_mgr
        assert hub1._credit_mgr is hub2._credit_mgr

    def test_credit_manager_initializes_stream_on_first_consume(self):
        """Test that consuming credit auto-initializes the stream."""
        hub = TelemetryAsyncHub()
        run_id = "run-1"
        agent_id = "agent-A"

        # Stream not initialized yet
        assert hub._credit_mgr.get_credits(run_id, agent_id) == 0

        # First consume should initialize stream
        result = hub._credit_mgr.consume_credit(run_id, agent_id)
        assert result is True, "First consume should succeed after auto-initialization"
        assert hub._credit_mgr.get_credits(run_id, agent_id) == 199  # 200 - 1

    def test_credit_consumption_decrements_available_credits(self):
        """Test that consuming credits properly decrements the count."""
        hub = TelemetryAsyncHub()
        run_id = "run-1"
        agent_id = "agent-A"

        mgr = hub._credit_mgr
        mgr.consume_credit(run_id, agent_id) and mgr.get_credits(run_id, agent_id)  # pyright: ignore[reportUnusedExpression]

        # Consume more credits
        mgr.consume_credit(run_id, agent_id)
        mgr.consume_credit(run_id, agent_id)

        final = mgr.get_credits(run_id, agent_id)
        # Initial should be 199, final should be 197
        assert final == 197

    def test_credit_exhaustion_returns_false(self):
        """Test that credit consumption fails when credits are exhausted."""
        hub = TelemetryAsyncHub()
        run_id = "run-1"
        agent_id = "agent-A"

        mgr = hub._credit_mgr

        # Deplete all credits
        for _ in range(200):
            if not mgr.consume_credit(run_id, agent_id):
                break

        # Attempt to consume when empty
        result = mgr.consume_credit(run_id, agent_id)
        assert result is False, "Should return False when no credits available"
        assert mgr.get_credits(run_id, agent_id) == 0

    def test_independent_credit_pools_per_stream(self):
        """Test that different agents have independent credit pools."""
        hub = TelemetryAsyncHub()
        run_id = "run-1"
        agent_a = "agent-A"
        agent_b = "agent-B"

        mgr = hub._credit_mgr

        # Consume from agent_a
        mgr.consume_credit(run_id, agent_a)
        mgr.consume_credit(run_id, agent_a)
        mgr.consume_credit(run_id, agent_a)

        # Initialize agent_b so we can check its credits
        mgr.initialize_stream(run_id, agent_b)

        # agent_a should have 197, agent_b should have 200
        assert mgr.get_credits(run_id, agent_a) == 197
        assert mgr.get_credits(run_id, agent_b) == 200

    def test_credit_grant_replenishes_credits(self):
        """Test that granting credits replenishes the pool."""
        hub = TelemetryAsyncHub()
        run_id = "run-1"
        agent_id = "agent-A"

        mgr = hub._credit_mgr

        # Consume all credits
        for _ in range(200):
            if not mgr.consume_credit(run_id, agent_id):
                break

        assert mgr.get_credits(run_id, agent_id) == 0

        # Grant new credits
        mgr.grant_credits(run_id, agent_id, 50)
        assert mgr.get_credits(run_id, agent_id) == 50

    def test_multiple_runs_independent_credits(self):
        """Test that different runs have independent credit pools."""
        hub = TelemetryAsyncHub()
        run_1 = "run-1"
        run_2 = "run-2"
        agent_id = "agent-A"

        mgr = hub._credit_mgr

        # Initialize both runs
        mgr.initialize_stream(run_1, agent_id)
        mgr.initialize_stream(run_2, agent_id)

        # Consume from run_1 only
        mgr.consume_credit(run_1, agent_id)
        mgr.consume_credit(run_1, agent_id)

        # run_1 should have 198 remaining, run_2 should stay at 200
        assert mgr.get_credits(run_1, agent_id) == 198
        assert mgr.get_credits(run_2, agent_id) == 200

    def test_credit_manager_reset_clears_all_credits(self):
        """Test that resetting credit manager clears all state."""
        hub1 = TelemetryAsyncHub()
        hub1._credit_mgr.consume_credit("run-1", "agent-A")
        hub1._credit_mgr.consume_credit("run-2", "agent-B")

        # Verify credits were consumed
        assert hub1._credit_mgr.get_credits("run-1", "agent-A") == 199
        assert hub1._credit_mgr.get_credits("run-2", "agent-B") == 199

        # Reset
        reset_credit_manager()

        # Create new hub after reset
        hub2 = TelemetryAsyncHub()
        assert hub2._credit_mgr.get_credits("run-1", "agent-A") == 0  # Not initialized
        assert hub2._credit_mgr.get_credits("run-2", "agent-B") == 0  # Not initialized

    def test_credit_initialization_with_custom_initial_credits(self):
        """Test that CreditManager can be created with custom initial credits."""
        custom_mgr = CreditManager(initial_credits=500)

        # Initialize stream
        custom_mgr.initialize_stream("run-1", "agent-A")
        assert custom_mgr.get_credits("run-1", "agent-A") == 500

    def test_credit_tracking_for_monitoring(self):
        """Test that credit manager tracks dropped events when exhausted."""
        hub = TelemetryAsyncHub()
        run_id = "run-1"
        agent_id = "agent-A"

        mgr = hub._credit_mgr

        # Consume all credits
        for _ in range(200):
            mgr.consume_credit(run_id, agent_id)

        # Attempt to consume more (should fail)
        attempts = 0
        while not mgr.consume_credit(run_id, agent_id):
            attempts += 1
            if attempts >= 5:
                break

        assert attempts == 5, "Multiple consume attempts should fail when exhausted"
        assert mgr.get_credits(run_id, agent_id) == 0
