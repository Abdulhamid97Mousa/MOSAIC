"""Tests for MultiGrid action mapping fix.

This test verifies that the MultiGridKeyCombinationResolver uses the correct
action indices for MultiGrid environments, fixing the bug where actions were
off-by-1 compared to MiniGrid.

Bug Report: User reported "I used one keyboard, and all agents started to move!"
Root Cause: MiniGrid and MultiGrid use different action indices.

MiniGrid actions:
0: LEFT, 1: RIGHT, 2: FORWARD, 3: PICKUP, 4: DROP, 5: TOGGLE, 6: DONE

MultiGrid actions:
0: STILL, 1: LEFT, 2: RIGHT, 3: FORWARD, 4: PICKUP, 5: DROP, 6: TOGGLE, 7: DONE

The fix creates separate resolvers for MiniGrid and MultiGrid.
"""

from __future__ import annotations

from typing import Set

import pytest

from gym_gui.controllers.human_input import (
    _KEY_A,
    _KEY_D,
    _KEY_DOWN,
    _KEY_E,
    _KEY_G,
    _KEY_H,
    _KEY_LEFT,
    _KEY_Q,
    _KEY_RETURN,
    _KEY_RIGHT,
    _KEY_S,
    _KEY_SPACE,
    _KEY_UP,
    _KEY_W,
    MiniGridKeyCombinationResolver,
    MultiGridKeyCombinationResolver,
    get_key_combination_resolver,
)
from gym_gui.core.enums import GameId

# =============================================================================
# MultiGrid Resolver Tests
# =============================================================================

class TestMultiGridKeyCombinationResolver:
    """Test MultiGrid key combination resolver with correct action indices."""

    @pytest.fixture
    def resolver(self) -> MultiGridKeyCombinationResolver:
        """Create a MultiGrid resolver."""
        return MultiGridKeyCombinationResolver()

    # ----- Movement Tests -----

    def test_forward_key_w(self, resolver: MultiGridKeyCombinationResolver):
        """Test W key produces FORWARD action (3)."""
        pressed = {_KEY_W}
        assert resolver.resolve(pressed) == 3  # FORWARD

    def test_forward_key_up(self, resolver: MultiGridKeyCombinationResolver):
        """Test Up arrow produces FORWARD action (3)."""
        pressed = {_KEY_UP}
        assert resolver.resolve(pressed) == 3  # FORWARD

    def test_left_key_a(self, resolver: MultiGridKeyCombinationResolver):
        """Test A key produces LEFT action (1)."""
        pressed = {_KEY_A}
        assert resolver.resolve(pressed) == 1  # LEFT (turn left)

    def test_left_key_left_arrow(self, resolver: MultiGridKeyCombinationResolver):
        """Test Left arrow produces LEFT action (1)."""
        pressed = {_KEY_LEFT}
        assert resolver.resolve(pressed) == 1  # LEFT

    def test_right_key_d(self, resolver: MultiGridKeyCombinationResolver):
        """Test D key produces RIGHT action (2)."""
        pressed = {_KEY_D}
        assert resolver.resolve(pressed) == 2  # RIGHT (turn right)

    def test_right_key_right_arrow(self, resolver: MultiGridKeyCombinationResolver):
        """Test Right arrow produces RIGHT action (2)."""
        pressed = {_KEY_RIGHT}
        assert resolver.resolve(pressed) == 2  # RIGHT

    # ----- Action Button Tests -----

    def test_pickup_space(self, resolver: MultiGridKeyCombinationResolver):
        """Test Space key produces PICKUP action (4)."""
        pressed = {_KEY_SPACE}
        assert resolver.resolve(pressed) == 4  # PICKUP

    def test_pickup_g(self, resolver: MultiGridKeyCombinationResolver):
        """Test G key produces PICKUP action (4)."""
        pressed = {_KEY_G}
        assert resolver.resolve(pressed) == 4  # PICKUP

    def test_drop_h(self, resolver: MultiGridKeyCombinationResolver):
        """Test H key produces DROP action (5)."""
        pressed = {_KEY_H}
        assert resolver.resolve(pressed) == 5  # DROP

    def test_toggle_e(self, resolver: MultiGridKeyCombinationResolver):
        """Test E key produces TOGGLE action (6)."""
        pressed = {_KEY_E}
        assert resolver.resolve(pressed) == 6  # TOGGLE

    def test_toggle_enter(self, resolver: MultiGridKeyCombinationResolver):
        """Test Enter key produces TOGGLE action (6)."""
        pressed = {_KEY_RETURN}
        assert resolver.resolve(pressed) == 6  # TOGGLE

    def test_done_q(self, resolver: MultiGridKeyCombinationResolver):
        """Test Q key produces DONE action (7)."""
        pressed = {_KEY_Q}
        assert resolver.resolve(pressed) == 7  # DONE

    # ----- Priority Tests -----

    def test_action_button_overrides_movement(self, resolver: MultiGridKeyCombinationResolver):
        """Test action buttons have priority over movement keys."""
        # Press both forward and pickup
        pressed = {_KEY_W, _KEY_SPACE}
        assert resolver.resolve(pressed) == 4  # PICKUP (not FORWARD)

    def test_left_overrides_right(self, resolver: MultiGridKeyCombinationResolver):
        """Test left has priority when both left and right are pressed."""
        pressed = {_KEY_LEFT, _KEY_RIGHT}
        assert resolver.resolve(pressed) == 1  # LEFT

    # ----- No Input Tests -----

    def test_no_keys_pressed(self, resolver: MultiGridKeyCombinationResolver):
        """Test no keys pressed returns None (caller should use STILL action 0)."""
        pressed: Set[int] = set()
        assert resolver.resolve(pressed) is None

    def test_unknown_key(self, resolver: MultiGridKeyCombinationResolver):
        """Test unknown keys return None."""
        pressed = {12345}  # Some random key code
        assert resolver.resolve(pressed) is None


# =============================================================================
# MiniGrid Resolver Tests (for comparison)
# =============================================================================

class TestMiniGridKeyCombinationResolver:
    """Test MiniGrid key combination resolver (LEGACY)."""

    @pytest.fixture
    def resolver(self) -> MiniGridKeyCombinationResolver:
        """Create a MiniGrid resolver."""
        return MiniGridKeyCombinationResolver()

    def test_forward_key_w(self, resolver: MiniGridKeyCombinationResolver):
        """Test W key produces move forward action (2) in MiniGrid."""
        pressed = {_KEY_W}
        assert resolver.resolve(pressed) == 2  # FORWARD (MiniGrid index)

    def test_left_key_a(self, resolver: MiniGridKeyCombinationResolver):
        """Test A key produces turn left action (0) in MiniGrid."""
        pressed = {_KEY_A}
        assert resolver.resolve(pressed) == 0  # LEFT (MiniGrid index)

    def test_right_key_d(self, resolver: MiniGridKeyCombinationResolver):
        """Test D key produces turn right action (1) in MiniGrid."""
        pressed = {_KEY_D}
        assert resolver.resolve(pressed) == 1  # RIGHT (MiniGrid index)


# =============================================================================
# Resolver Selection Tests
# =============================================================================

class TestResolverSelection:
    """Test that the correct resolver is selected for each game type."""

    def test_multigrid_uses_multigrid_resolver(self):
        """Test MultiGrid games use MultiGridKeyCombinationResolver."""
        # Create a mock GameId that looks like MOSAIC MultiGrid
        class MockMultiGridGameId:
            value = "MosaicMultiGrid-Soccer-v0"

        resolver = get_key_combination_resolver(MockMultiGridGameId())  # type: ignore[arg-type]
        assert isinstance(resolver, MultiGridKeyCombinationResolver)

    def test_minigrid_uses_minigrid_resolver(self):
        """Test MiniGrid games use MiniGridKeyCombinationResolver."""
        resolver = get_key_combination_resolver(GameId.MINIGRID_EMPTY_RANDOM_5x5)
        assert isinstance(resolver, MiniGridKeyCombinationResolver)


# =============================================================================
# Action Index Difference Tests
# =============================================================================

class TestActionIndexDifference:
    """Test that verifies the action index difference between MiniGrid and MultiGrid."""

    def test_forward_action_difference(self):
        """Test FORWARD action has different indices in MiniGrid vs MultiGrid."""
        minigrid_resolver = MiniGridKeyCombinationResolver()
        multigrid_resolver = MultiGridKeyCombinationResolver()

        pressed = {_KEY_W}

        minigrid_action = minigrid_resolver.resolve(pressed)
        multigrid_action = multigrid_resolver.resolve(pressed)

        # MiniGrid: forward=2, MultiGrid: forward=3
        assert minigrid_action == 2
        assert multigrid_action == 3
        assert multigrid_action == minigrid_action + 1  # Off by 1!

    def test_left_action_difference(self):
        """Test LEFT action has different indices in MiniGrid vs MultiGrid."""
        minigrid_resolver = MiniGridKeyCombinationResolver()
        multigrid_resolver = MultiGridKeyCombinationResolver()

        pressed = {_KEY_A}

        minigrid_action = minigrid_resolver.resolve(pressed)
        multigrid_action = multigrid_resolver.resolve(pressed)

        # MiniGrid: left=0, MultiGrid: left=1
        assert minigrid_action == 0
        assert multigrid_action == 1
        assert multigrid_action == minigrid_action + 1  # Off by 1!

    def test_pickup_action_difference(self):
        """Test PICKUP action has different indices in MiniGrid vs MultiGrid."""
        minigrid_resolver = MiniGridKeyCombinationResolver()
        multigrid_resolver = MultiGridKeyCombinationResolver()

        pressed = {_KEY_SPACE}

        minigrid_action = minigrid_resolver.resolve(pressed)
        multigrid_action = multigrid_resolver.resolve(pressed)

        # MiniGrid: pickup=3, MultiGrid: pickup=4
        assert minigrid_action == 3
        assert multigrid_action == 4
        assert multigrid_action == minigrid_action + 1  # Off by 1!


# =============================================================================
# Integration Tests
# =============================================================================

class TestMultiAgentActionGeneration:
    """Test multi-agent action generation with correct indices."""

    def test_single_agent_action_others_still(self):
        """Test that only the agent with pressed keys gets action, others get STILL."""
        resolver = MultiGridKeyCombinationResolver()

        # Simulate 4 agents, only agent_0 has W pressed
        agent_0_keys = {_KEY_W}
        agent_1_keys: Set[int] = set()
        agent_2_keys: Set[int] = set()
        agent_3_keys: Set[int] = set()

        # Get actions for each agent
        STILL_ACTION = 0

        action_0 = resolver.resolve(agent_0_keys)
        action_1 = resolver.resolve(agent_1_keys) if agent_1_keys else None
        action_2 = resolver.resolve(agent_2_keys) if agent_2_keys else None
        action_3 = resolver.resolve(agent_3_keys) if agent_3_keys else None

        # Build action list (like get_multi_agent_actions does)
        actions = [
            action_0 if action_0 is not None else STILL_ACTION,
            action_1 if action_1 is not None else STILL_ACTION,
            action_2 if action_2 is not None else STILL_ACTION,
            action_3 if action_3 is not None else STILL_ACTION,
        ]

        # Expected: [3, 0, 0, 0] = [FORWARD, STILL, STILL, STILL]
        assert actions == [3, 0, 0, 0]

        # Verify only agent_0 is moving
        assert actions[0] == 3  # FORWARD
        assert actions[1] == 0  # STILL
        assert actions[2] == 0  # STILL
        assert actions[3] == 0  # STILL

    def test_multiple_agents_different_actions(self):
        """Test multiple agents with different actions."""
        resolver = MultiGridKeyCombinationResolver()

        # Simulate 4 agents with different keys
        agent_0_keys = {_KEY_W}      # FORWARD
        agent_1_keys = {_KEY_A}      # LEFT
        agent_2_keys = {_KEY_SPACE}  # PICKUP
        agent_3_keys: Set[int] = set()  # No keys

        STILL_ACTION = 0

        actions = [
            resolver.resolve(agent_0_keys) if agent_0_keys else STILL_ACTION,
            resolver.resolve(agent_1_keys) if agent_1_keys else STILL_ACTION,
            resolver.resolve(agent_2_keys) if agent_2_keys else STILL_ACTION,
            resolver.resolve(agent_3_keys) if agent_3_keys else STILL_ACTION,
        ]

        # Expected: [3, 1, 4, 0] = [FORWARD, LEFT, PICKUP, STILL]
        assert actions == [3, 1, 4, 0]
