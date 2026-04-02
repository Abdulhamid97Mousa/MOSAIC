"""Tests for PettingZoo step counter synchronisation and button recovery.

Covers the fixes made to keep the widget "Steps" counter, the render
container's "Step" counter, and the player step buttons in a consistent state.

Scenarios tested
----------------
1. ``OperatorsTab.set_step_count`` updates the label correctly and never goes
   negative (the API call, not the button-click path).
2. ``_on_step_player_clicked`` does NOT increment the step counter — the
   counter must stay at 0 until set_step_count is called externally.
3. ``ControlPanelWidget.set_step_count`` delegates correctly to
   ``OperatorsTab.set_step_count``.
4. Resetting via ``_on_reset_all_clicked`` brings the counter back to 0.
5. ``set_current_player`` correctly enables/disables the Step White / Step
   Black buttons in PettingZoo mode.
6. After a failed step (simulated by calling ``set_current_player`` again
   without any actual env.step), the active player's button is re-enabled.
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# OperatorsTab – step counter behaviour
# ---------------------------------------------------------------------------

class TestOperatorsTabStepCounter:
    """OperatorsTab step counter is driven only by external set_step_count."""

    @pytest.fixture
    def tab(self, qtbot):
        from gym_gui.ui.widgets.operators_tab import OperatorsTab
        t = OperatorsTab()
        qtbot.addWidget(t)
        return t

    def test_initial_step_count_is_zero(self, tab):
        """Step counter starts at 0."""
        assert tab._step_count == 0
        assert tab._step_count_label.text() == "Steps: 0"

    def test_set_step_count_updates_label(self, tab):
        """set_step_count sets both the internal value and the visible label."""
        tab.set_step_count(5)
        assert tab._step_count == 5
        assert tab._step_count_label.text() == "Steps: 5"

    def test_set_step_count_zero_clears_label(self, tab):
        """set_step_count(0) resets the label to 'Steps: 0'."""
        tab.set_step_count(99)
        tab.set_step_count(0)
        assert tab._step_count == 0
        assert tab._step_count_label.text() == "Steps: 0"

    def test_button_click_does_not_increment_counter(self, tab):
        """Clicking a player step button must NOT change _step_count.

        The counter is driven exclusively by set_step_count (called from
        main_window after env.step succeeds).
        """
        # Enable PettingZoo mode so buttons are visible and can be enabled
        tab.set_pettingzoo_mode(True)
        tab.set_current_player("player_0")

        # Capture signal without actually executing a step
        emitted: List[tuple] = []
        tab.step_player_requested.connect(lambda pid, seed: emitted.append((pid, seed)))

        # Simulate click
        tab._on_step_player_clicked("player_0")

        # The signal must have fired
        assert len(emitted) == 1
        assert emitted[0][0] == "player_0"

        # But the step counter must still be 0 — only set_step_count changes it
        assert tab._step_count == 0
        assert tab._step_count_label.text() == "Steps: 0"

    def test_reset_restores_step_count_to_zero(self, tab):
        """_on_reset_all_clicked resets the step counter."""
        tab._is_running = True
        tab.set_step_count(7)
        tab._on_reset_all_clicked()
        assert tab._step_count == 0
        assert tab._step_count_label.text() == "Steps: 0"

    def test_sequential_set_step_count_calls(self, tab):
        """Multiple set_step_count calls reflect each update."""
        for i in range(1, 6):
            tab.set_step_count(i)
            assert tab._step_count == i
            assert tab._step_count_label.text() == f"Steps: {i}"


# ---------------------------------------------------------------------------
# OperatorsTab – PettingZoo button enable / disable
# ---------------------------------------------------------------------------

class TestOperatorsTabPettingZooButtons:
    """Step White / Step Black buttons follow set_current_player correctly."""

    @pytest.fixture
    def tab(self, qtbot):
        from gym_gui.ui.widgets.operators_tab import OperatorsTab
        t = OperatorsTab()
        qtbot.addWidget(t)
        t.set_pettingzoo_mode(True)
        return t

    def test_buttons_visible_in_pettingzoo_mode(self, tab):
        tab.show()
        assert tab._step_player_0_btn.isVisible()
        assert tab._step_player_1_btn.isVisible()

    def test_both_buttons_disabled_after_mode_enable(self, tab):
        """Both buttons are disabled immediately after set_pettingzoo_mode(True)."""
        assert not tab._step_player_0_btn.isEnabled()
        assert not tab._step_player_1_btn.isEnabled()

    def test_set_current_player_0_enables_white_only(self, tab):
        tab.set_current_player("player_0")
        assert tab._step_player_0_btn.isEnabled()
        assert not tab._step_player_1_btn.isEnabled()

    def test_set_current_player_1_enables_black_only(self, tab):
        tab.set_current_player("player_1")
        assert not tab._step_player_0_btn.isEnabled()
        assert tab._step_player_1_btn.isEnabled()

    def test_set_current_player_empty_disables_both(self, tab):
        """Passing '' (game over) disables both buttons."""
        tab.set_current_player("player_0")
        tab.set_current_player("")
        assert not tab._step_player_0_btn.isEnabled()
        assert not tab._step_player_1_btn.isEnabled()

    def test_button_disabled_on_click_then_restored_by_set_current_player(self, tab):
        """After a click disables a button, set_current_player re-enables it (recovery)."""
        tab.set_current_player("player_0")
        assert tab._step_player_0_btn.isEnabled()

        # Simulate the immediate disable that happens on click
        tab._on_step_player_clicked("player_0")
        assert not tab._step_player_0_btn.isEnabled()

        # Simulate failed step recovery: main_window calls set_current_player again
        tab.set_current_player("player_0")
        assert tab._step_player_0_btn.isEnabled()

    def test_wrong_player_click_leaves_correct_button_enabled(self, tab):
        """Clicking the wrong player button: correct button stays enabled after recovery."""
        tab.set_current_player("player_1")  # it is black's turn
        # User mistakenly clicks white's button (which shouldn't even be enabled,
        # but let's test the recovery path)
        tab._on_step_player_clicked("player_0")
        # Recovery: main_window calls set_current_player with the actual current player
        tab.set_current_player("player_1")
        assert not tab._step_player_0_btn.isEnabled()
        assert tab._step_player_1_btn.isEnabled()

    def test_buttons_hidden_when_pettingzoo_mode_disabled(self, tab):
        tab.set_pettingzoo_mode(False)
        assert not tab._step_player_0_btn.isVisible()
        assert not tab._step_player_1_btn.isVisible()


# ---------------------------------------------------------------------------
# ControlPanelWidget – set_step_count passthrough
# ---------------------------------------------------------------------------

class TestControlPanelSetStepCount:
    """ControlPanelWidget.set_step_count delegates to the embedded OperatorsTab."""

    @pytest.fixture
    def panel(self, qtbot):
        from gym_gui.ui.widgets.control_panel import ControlPanelWidget, ControlPanelConfig
        from gym_gui.core.enums import ControlMode, GameId
        from gym_gui.config.game_configs import (
            FrozenLakeConfig, TaxiConfig, CliffWalkingConfig,
            LunarLanderConfig, CarRacingConfig, BipedalWalkerConfig,
            MiniGridConfig,
        )
        config = ControlPanelConfig(
            available_modes={GameId.FROZEN_LAKE: [ControlMode.HUMAN_ONLY]},
            default_mode=ControlMode.HUMAN_ONLY,
            frozen_lake_config=FrozenLakeConfig(),
            taxi_config=TaxiConfig(),
            cliff_walking_config=CliffWalkingConfig(),
            lunar_lander_config=LunarLanderConfig(),
            car_racing_config=CarRacingConfig(),
            bipedal_walker_config=BipedalWalkerConfig(),
            minigrid_empty_config=MiniGridConfig(),
            minigrid_doorkey_5x5_config=MiniGridConfig(),
            minigrid_doorkey_6x6_config=MiniGridConfig(),
            minigrid_doorkey_8x8_config=MiniGridConfig(),
            minigrid_doorkey_16x16_config=MiniGridConfig(),
            minigrid_lavagap_config=MiniGridConfig(),
            default_seed=42,
            allow_seed_reuse=False,
            operators=(),
        )
        p = ControlPanelWidget(config=config)
        qtbot.addWidget(p)
        return p

    def test_set_step_count_updates_operators_tab(self, panel):
        """ControlPanelWidget.set_step_count forwards the value to OperatorsTab."""
        panel.set_step_count(3)
        assert panel._operators_tab._step_count == 3
        assert panel._operators_tab._step_count_label.text() == "Steps: 3"

    def test_set_step_count_zero_resets_operators_tab(self, panel):
        panel.set_step_count(10)
        panel.set_step_count(0)
        assert panel._operators_tab._step_count == 0
        assert panel._operators_tab._step_count_label.text() == "Steps: 0"

    def test_set_step_count_does_not_affect_step_all_button(self, panel):
        """set_step_count must not accidentally enable/disable the Step All button."""
        initial_enabled = panel._operators_tab._step_all_button.isEnabled()
        panel.set_step_count(5)
        assert panel._operators_tab._step_all_button.isEnabled() == initial_enabled


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
