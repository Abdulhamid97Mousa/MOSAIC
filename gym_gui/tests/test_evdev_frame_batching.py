"""Tests for evdev singleShot-chained game-tick loop (human multi-keyboard).

Problem:
    When two USB keyboards press the same key simultaneously, the old code
    called ``env.step()`` once per key event.  The first step had incomplete
    state (only one agent's key).  Worse, ``adapter.step()`` calls
    ``render()`` synchronously on the main thread, and a repeating QTimer
    fires the next tick immediately after the slot returns (coalesced),
    starving the Qt event loop of paint events.  The result: agents appear
    frozen, then "teleport" when the event loop finally processes the
    queued paint events.

Solution:
    1. Key press/release events ONLY update per-agent ``_agent_pressed_keys``
       state.  They never call ``env.step()``.
    2. A singleShot-chained game tick reads ALL agents' state and steps once.
    3. After each tick, control returns to the Qt event loop (paint events
       process, the rendered frame appears on screen), then the NEXT tick
       is scheduled via ``QTimer.singleShot(remaining_ms)``.

Test coverage:
    1.  Key events update state only, never step
    2.  Game tick steps with complete multi-agent state
    3.  Simultaneous presses: single tick, both agents move
    4.  Idle tick (no keys held) skipped, chain continues
    5.  Tick rate configuration and clamping
    6.  singleShot chaining: tick schedules next tick (not repeating timer)
    7.  Stop breaks the chain (next tick is a no-op)
    8.  Per-agent signal still fires immediately (Operators tab)
    9.  4- and 8-keyboard scaling
    10. Key release between ticks correctly reflected
    11. Step time subtracted from interval (adaptive pacing)
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Set
from unittest.mock import MagicMock, patch

import pytest

from gym_gui.controllers.human_input import (
    _KEY_A,
    _KEY_D,
    _KEY_S,
    _KEY_W,
    INIMultiGridKeyCombinationResolver,
    MultiGridKeyCombinationResolver,
)

# ---------------------------------------------------------------------------
# Minimal reproduction of the singleShot-chained game-tick logic.
# ---------------------------------------------------------------------------

class FakeGameTickController:
    """Mirrors the evdev game-tick fields/methods from HumanInputController.

    Instead of actually calling ``QTimer.singleShot``, records whether
    the next tick was scheduled and with what delay.
    """

    def __init__(
        self,
        agent_names: List[str],
        resolver,
        default_idle: int,
    ) -> None:
        self.agent_names = agent_names
        self.resolver = resolver
        self.default_idle = default_idle

        self._agent_pressed_keys: Dict[str, Set[int]] = {}
        self._use_evdev = True
        self._evdev_tick_active = False
        self._evdev_tick_interval_ms = 100  # 10 Hz

        # Records
        self.stepped_actions: List[List[int]] = []
        self.per_agent_signals: List[tuple] = []
        self.scheduled_ticks: List[int] = []  # delay_ms for each scheduled tick

        # Simulate step duration (ms)
        self._simulated_step_duration_ms = 0.0

    # -- key press: state-only --
    def simulate_key_press(self, agent_id: str, qt_key: int) -> None:
        if agent_id not in self._agent_pressed_keys:
            self._agent_pressed_keys[agent_id] = set()
        if qt_key not in self._agent_pressed_keys[agent_id]:
            self._agent_pressed_keys[agent_id].add(qt_key)
            pressed = self._agent_pressed_keys.get(agent_id, set())
            resolved = self.resolver.resolve(pressed)
            if resolved is not None:
                self.per_agent_signals.append((agent_id, resolved))

    # -- key release: state-only --
    def simulate_key_release(self, agent_id: str, qt_key: int) -> None:
        if agent_id in self._agent_pressed_keys:
            self._agent_pressed_keys[agent_id].discard(qt_key)

    # -- game tick (mirrors _on_evdev_game_tick with singleShot chaining) --
    def on_game_tick(self) -> None:
        if not self._evdev_tick_active:
            return

        any_keys = any(bool(k) for k in self._agent_pressed_keys.values())
        if any_keys:
            elapsed_ms = self._simulated_step_duration_ms
            actions = self._get_multi_agent_actions()
            if actions:
                self.stepped_actions.append(actions)
            remaining = max(0, self._evdev_tick_interval_ms - elapsed_ms)
        else:
            remaining = self._evdev_tick_interval_ms

        # Record the singleShot scheduling (in real code: QTimer.singleShot)
        self.scheduled_ticks.append(int(remaining))

    def _get_multi_agent_actions(self) -> Optional[List[int]]:
        actions: List[int] = []
        for agent_id in self.agent_names:
            pressed = self._agent_pressed_keys.get(agent_id, set())
            if pressed:
                action = self.resolver.resolve(pressed)
                actions.append(action if action is not None else self.default_idle)
            else:
                actions.append(self.default_idle)
        return actions

    # -- lifecycle --
    def start(self) -> None:
        self._evdev_tick_active = True
        self.scheduled_ticks.append(self._evdev_tick_interval_ms)

    def stop(self) -> None:
        self._evdev_tick_active = False

    def clear(self) -> None:
        self.stop()
        self._agent_pressed_keys.clear()

    def set_tick_rate(self, hz: int) -> None:
        hz = max(1, min(60, hz))
        self._evdev_tick_interval_ms = max(1, 1000 // hz)

    def get_tick_rate(self) -> int:
        interval = self._evdev_tick_interval_ms
        return max(1, 1000 // interval) if interval > 0 else 10


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_agent():
    ctrl = FakeGameTickController(
        ["agent_0", "agent_1"],
        INIMultiGridKeyCombinationResolver(),
        default_idle=6,
    )
    ctrl.start()
    return ctrl


@pytest.fixture
def four_agent():
    ctrl = FakeGameTickController(
        [f"agent_{i}" for i in range(4)],
        INIMultiGridKeyCombinationResolver(),
        default_idle=6,
    )
    ctrl.start()
    return ctrl


@pytest.fixture
def two_agent_legacy():
    ctrl = FakeGameTickController(
        ["agent_0", "agent_1"],
        MultiGridKeyCombinationResolver(),
        default_idle=0,
    )
    ctrl.start()
    return ctrl


# ===========================================================================
# 1. Key Events Update State Only
# ===========================================================================

class TestKeyEventsStateOnly:
    def test_key_press_does_not_step(self, two_agent):
        two_agent.simulate_key_press("agent_0", _KEY_D)
        assert two_agent.stepped_actions == []

    def test_key_press_updates_state(self, two_agent):
        two_agent.simulate_key_press("agent_0", _KEY_D)
        assert _KEY_D in two_agent._agent_pressed_keys["agent_0"]

    def test_key_release_updates_state(self, two_agent):
        two_agent.simulate_key_press("agent_0", _KEY_D)
        two_agent.simulate_key_release("agent_0", _KEY_D)
        assert _KEY_D not in two_agent._agent_pressed_keys.get("agent_0", set())


# ===========================================================================
# 2. Game Tick Steps With Complete State
# ===========================================================================

class TestGameTickStepping:
    def test_tick_produces_one_step(self, two_agent):
        two_agent.simulate_key_press("agent_0", _KEY_D)
        two_agent.on_game_tick()

        assert len(two_agent.stepped_actions) == 1
        assert two_agent.stepped_actions[0][0] == 1   # RIGHT
        assert two_agent.stepped_actions[0][1] == 6   # DONE (idle)

    def test_held_key_multiple_ticks(self, two_agent):
        two_agent.simulate_key_press("agent_0", _KEY_D)
        two_agent.on_game_tick()
        two_agent.on_game_tick()
        two_agent.on_game_tick()

        assert len(two_agent.stepped_actions) == 3
        assert all(a[0] == 1 for a in two_agent.stepped_actions)


# ===========================================================================
# 3. Simultaneous Presses (The Core Bug Fix)
# ===========================================================================

class TestSimultaneousPresses:
    def test_both_d_presses_one_tick_one_step(self, two_agent):
        """THE BUG: two keyboards pressing D used to cause two env.step()
        calls.  With singleShot chaining, it's always one step per tick."""
        two_agent.simulate_key_press("agent_0", _KEY_D)
        two_agent.simulate_key_press("agent_1", _KEY_D)
        two_agent.on_game_tick()

        assert len(two_agent.stepped_actions) == 1

    def test_both_d_presses_both_agents_move(self, two_agent):
        two_agent.simulate_key_press("agent_0", _KEY_D)
        two_agent.simulate_key_press("agent_1", _KEY_D)
        two_agent.on_game_tick()

        actions = two_agent.stepped_actions[0]
        assert actions[0] == 1, "agent_0 -> RIGHT"
        assert actions[1] == 1, "agent_1 -> RIGHT"

    def test_different_keys_simultaneous(self, two_agent):
        two_agent.simulate_key_press("agent_0", _KEY_W)
        two_agent.simulate_key_press("agent_1", _KEY_A)
        two_agent.on_game_tick()

        actions = two_agent.stepped_actions[0]
        assert actions[0] == 2, "agent_0 -> FORWARD"
        assert actions[1] == 0, "agent_1 -> LEFT"

    def test_order_of_key_presses_irrelevant(self, two_agent):
        two_agent.simulate_key_press("agent_1", _KEY_A)
        two_agent.simulate_key_press("agent_0", _KEY_W)
        two_agent.on_game_tick()

        actions = two_agent.stepped_actions[0]
        assert actions[0] == 2, "agent_0 -> FORWARD"
        assert actions[1] == 0, "agent_1 -> LEFT"


# ===========================================================================
# 4. Idle Tick Skipped, Chain Continues
# ===========================================================================

class TestIdleTick:
    def test_no_keys_no_step(self, two_agent):
        two_agent.on_game_tick()
        assert two_agent.stepped_actions == []

    def test_idle_tick_still_schedules_next(self, two_agent):
        """Even when idle, the chain must continue so held keys get picked up."""
        initial_scheduled = len(two_agent.scheduled_ticks)
        two_agent.on_game_tick()  # no keys held
        assert len(two_agent.scheduled_ticks) == initial_scheduled + 1

    def test_keys_released_before_tick(self, two_agent):
        two_agent.simulate_key_press("agent_0", _KEY_D)
        two_agent.simulate_key_release("agent_0", _KEY_D)
        two_agent.on_game_tick()
        assert two_agent.stepped_actions == []


# ===========================================================================
# 5. Tick Rate Configuration
# ===========================================================================

class TestTickRateConfiguration:
    def test_set_tick_rate_20hz(self, two_agent):
        two_agent.set_tick_rate(20)
        assert two_agent._evdev_tick_interval_ms == 50

    def test_set_tick_rate_clamped_low(self, two_agent):
        two_agent.set_tick_rate(0)
        assert two_agent._evdev_tick_interval_ms == 1000  # 1 Hz

    def test_set_tick_rate_clamped_high(self, two_agent):
        two_agent.set_tick_rate(120)
        assert two_agent._evdev_tick_interval_ms == 16  # 60 Hz

    def test_get_tick_rate(self, two_agent):
        assert two_agent.get_tick_rate() == 10


# ===========================================================================
# 6. SingleShot Chaining
# ===========================================================================

class TestSingleShotChaining:
    def test_start_schedules_first_tick(self, two_agent):
        """start() must schedule the initial singleShot to kick off the chain."""
        assert len(two_agent.scheduled_ticks) == 1
        assert two_agent.scheduled_ticks[0] == 100  # initial delay

    def test_tick_schedules_next_tick(self, two_agent):
        two_agent.simulate_key_press("agent_0", _KEY_D)
        two_agent.on_game_tick()

        # start() scheduled 1, on_game_tick() scheduled another
        assert len(two_agent.scheduled_ticks) == 2

    def test_chain_continues_across_multiple_ticks(self, two_agent):
        two_agent.simulate_key_press("agent_0", _KEY_D)
        for _ in range(5):
            two_agent.on_game_tick()

        # 1 from start() + 5 from ticks
        assert len(two_agent.scheduled_ticks) == 6

    def test_idle_tick_schedules_full_interval(self, two_agent):
        two_agent.on_game_tick()  # no keys held
        # Last scheduled delay should be the full interval
        assert two_agent.scheduled_ticks[-1] == 100


# ===========================================================================
# 7. Stop Breaks the Chain
# ===========================================================================

class TestStopBreaksChain:
    def test_stop_sets_flag(self, two_agent):
        two_agent.stop()
        assert two_agent._evdev_tick_active is False

    def test_tick_after_stop_is_noop(self, two_agent):
        two_agent.simulate_key_press("agent_0", _KEY_D)
        two_agent.stop()
        scheduled_before = len(two_agent.scheduled_ticks)
        two_agent.on_game_tick()

        assert two_agent.stepped_actions == []
        # No new tick scheduled
        assert len(two_agent.scheduled_ticks) == scheduled_before

    def test_clear_stops_and_clears(self, two_agent):
        two_agent.simulate_key_press("agent_0", _KEY_D)
        two_agent.clear()
        assert two_agent._agent_pressed_keys == {}
        assert two_agent._evdev_tick_active is False


# ===========================================================================
# 8. Per-Agent Signal Fires Immediately
# ===========================================================================

class TestPerAgentSignal:
    def test_signal_fires_on_press(self, two_agent):
        two_agent.simulate_key_press("agent_0", _KEY_D)
        assert two_agent.per_agent_signals == [("agent_0", 1)]

    def test_signals_fire_for_both_agents(self, two_agent):
        two_agent.simulate_key_press("agent_0", _KEY_D)
        two_agent.simulate_key_press("agent_1", _KEY_W)
        assert two_agent.per_agent_signals == [
            ("agent_0", 1),   # RIGHT
            ("agent_1", 2),   # FORWARD
        ]


# ===========================================================================
# 9. Scaling: 4 and 8 Keyboards
# ===========================================================================

class TestScaling:
    def test_four_keyboards_simultaneous(self, four_agent):
        for i in range(4):
            four_agent.simulate_key_press(f"agent_{i}", _KEY_D)
        four_agent.on_game_tick()

        assert len(four_agent.stepped_actions) == 1
        assert all(a == 1 for a in four_agent.stepped_actions[0])

    def test_eight_keyboards(self):
        agents = [f"agent_{i}" for i in range(8)]
        ctrl = FakeGameTickController(
            agents, INIMultiGridKeyCombinationResolver(), default_idle=6
        )
        ctrl.start()
        for agent in agents:
            ctrl.simulate_key_press(agent, _KEY_D)
        ctrl.on_game_tick()

        assert len(ctrl.stepped_actions) == 1
        assert len(ctrl.stepped_actions[0]) == 8
        assert all(a == 1 for a in ctrl.stepped_actions[0])


# ===========================================================================
# 10. Key Release Between Ticks
# ===========================================================================

class TestKeyReleaseBetweenTicks:
    def test_release_before_tick_reflects(self, two_agent):
        two_agent.simulate_key_press("agent_0", _KEY_D)
        two_agent.simulate_key_press("agent_1", _KEY_D)
        two_agent.simulate_key_release("agent_1", _KEY_D)
        two_agent.on_game_tick()

        actions = two_agent.stepped_actions[0]
        assert actions[0] == 1, "agent_0 -> RIGHT"
        assert actions[1] == 6, "agent_1 -> DONE (idle)"

    def test_held_key_persists_across_ticks(self, two_agent):
        two_agent.simulate_key_press("agent_0", _KEY_W)
        two_agent.on_game_tick()
        two_agent.on_game_tick()
        two_agent.on_game_tick()

        assert len(two_agent.stepped_actions) == 3
        assert all(a[0] == 2 for a in two_agent.stepped_actions)


# ===========================================================================
# 11. Adaptive Pacing (step time subtracted from interval)
# ===========================================================================

class TestAdaptivePacing:
    def test_fast_step_gets_full_remaining(self, two_agent):
        """A 10ms step with 100ms interval -> 90ms remaining."""
        two_agent._simulated_step_duration_ms = 10.0
        two_agent.simulate_key_press("agent_0", _KEY_D)
        two_agent.on_game_tick()

        assert two_agent.scheduled_ticks[-1] == 90

    def test_slow_step_gets_zero_remaining(self, two_agent):
        """A 150ms step with 100ms interval -> 0ms remaining (no negative)."""
        two_agent._simulated_step_duration_ms = 150.0
        two_agent.simulate_key_press("agent_0", _KEY_D)
        two_agent.on_game_tick()

        assert two_agent.scheduled_ticks[-1] == 0

    def test_exact_step_gets_zero_remaining(self, two_agent):
        """A 100ms step with 100ms interval -> 0ms remaining."""
        two_agent._simulated_step_duration_ms = 100.0
        two_agent.simulate_key_press("agent_0", _KEY_D)
        two_agent.on_game_tick()

        assert two_agent.scheduled_ticks[-1] == 0


# ===========================================================================
# 12. Legacy MultiGrid Compatibility
# ===========================================================================

class TestLegacyMultiGrid:
    def test_simultaneous_d_legacy(self, two_agent_legacy):
        two_agent_legacy.simulate_key_press("agent_0", _KEY_D)
        two_agent_legacy.simulate_key_press("agent_1", _KEY_D)
        two_agent_legacy.on_game_tick()

        assert len(two_agent_legacy.stepped_actions) == 1
        assert two_agent_legacy.stepped_actions[0][0] == two_agent_legacy.stepped_actions[0][1]

    def test_idle_is_still(self, two_agent_legacy):
        two_agent_legacy.simulate_key_press("agent_0", _KEY_W)
        two_agent_legacy.on_game_tick()
        assert two_agent_legacy.stepped_actions[0][1] == 0  # STILL
