"""UI helpers for MalmoEnv (Minecraft) configuration panels.

Provides controls for overriding mission parameters like:
- Time limit (timeLimitMs)
- Tick rate (MsPerTick)
- Video resolution
- Episode rewards
"""

from __future__ import annotations

from typing import Any, Dict

from PyQt6 import QtWidgets

from gym_gui.core.enums import GameId

# All MalmoEnv game IDs
MALMOENV_FAMILY = {
    GameId.MALMOENV_MOBCHASE,
    GameId.MALMOENV_MAZERUNNER,
    GameId.MALMOENV_VERTICAL,
    GameId.MALMOENV_CLIFFWALKING,
    GameId.MALMOENV_CATCHTHEMOB,
    GameId.MALMOENV_FINDTHEGOAL,
    GameId.MALMOENV_ATTIC,
    GameId.MALMOENV_DEFAULTFLATWORLD,
    GameId.MALMOENV_DEFAULTWORLD,
    GameId.MALMOENV_EATING,
    GameId.MALMOENV_OBSTACLES,
    GameId.MALMOENV_TRICKYARENA,
    GameId.MALMOENV_TREASUREHUNT,
}

# Default MalmoEnv configuration
# Default: NO timeout for exploration (time_limit_ms=0 means unlimited)
DEFAULT_MALMO_CONFIG = {
    "time_limit_ms": 0,  # 0 = no timeout (exploration mode)
    "ms_per_tick": 50,  # 20 tps (mission default)
    "video_width": 320,
    "video_height": 240,
    "reward_goal": 1000,
    "reward_timeout": -1000,
    "reward_death": -10000,
    "port": 9000,
    "server": "localhost",
}


def build_malmo_controls(
    *,
    layout: QtWidgets.QFormLayout,
    group: QtWidgets.QWidget,
    overrides: Dict[str, Any],
    on_change: Any,
) -> None:
    """Build MalmoEnv-specific configuration controls.

    Args:
        layout: Form layout to add controls to
        group: Parent widget for controls
        overrides: Dictionary to store configuration overrides
        on_change: Callback function(name, value) when config changes
    """
    # Set defaults
    for key, value in DEFAULT_MALMO_CONFIG.items():
        overrides.setdefault(key, value)

    def emit(name: str, value: Any) -> None:
        on_change(name, value)

    # --- Mission Timing Section ---
    timing_group = QtWidgets.QGroupBox("Mission Timing", group)
    timing_layout = QtWidgets.QFormLayout(timing_group)

    # Enable/disable time limit checkbox
    timeout_checkbox = QtWidgets.QCheckBox("Enable time limit", timing_group)
    has_timeout = overrides["time_limit_ms"] > 0
    timeout_checkbox.setChecked(has_timeout)
    timing_layout.addRow("", timeout_checkbox)

    # Time limit (seconds, stored internally as ms)
    time_limit_spin = QtWidgets.QDoubleSpinBox(timing_group)
    time_limit_spin.setRange(1.0, 3600.0)  # 1 second to 1 hour
    time_limit_spin.setSingleStep(5.0)
    time_limit_spin.setDecimals(1)
    time_limit_spin.setValue(
        (overrides["time_limit_ms"] / 1000.0) if has_timeout else 60.0
    )
    time_limit_spin.setSuffix(" seconds")
    time_limit_spin.setEnabled(has_timeout)

    def on_timeout_changed(checked: bool) -> None:
        time_limit_spin.setEnabled(checked)
        if checked:
            emit("time_limit_ms", int(time_limit_spin.value() * 1000))
        else:
            emit("time_limit_ms", 0)  # 0 = unlimited

    timeout_checkbox.toggled.connect(on_timeout_changed)

    def on_time_value_changed(value: float) -> None:
        if timeout_checkbox.isChecked():
            emit("time_limit_ms", int(value * 1000))

    time_limit_spin.valueChanged.connect(on_time_value_changed)
    timing_layout.addRow("Time Limit", time_limit_spin)

    # Tick rate (ms per tick)
    tick_spin = QtWidgets.QSpinBox(timing_group)
    tick_spin.setRange(10, 1000)
    tick_spin.setSingleStep(10)
    tick_spin.setValue(int(overrides["ms_per_tick"]))
    tick_spin.setSuffix(" ms")
    tick_spin.valueChanged.connect(lambda value: emit("ms_per_tick", int(value)))

    timing_layout.addRow("Ms Per Tick", tick_spin)


    layout.addRow(timing_group)

    # --- Video Section ---
    video_group = QtWidgets.QGroupBox("Video Settings", group)
    video_layout = QtWidgets.QFormLayout(video_group)

    width_spin = QtWidgets.QSpinBox(video_group)
    width_spin.setRange(84, 1920)
    width_spin.setSingleStep(16)
    width_spin.setValue(int(overrides["video_width"]))
    width_spin.valueChanged.connect(lambda value: emit("video_width", int(value)))
    video_layout.addRow("Width", width_spin)

    height_spin = QtWidgets.QSpinBox(video_group)
    height_spin.setRange(84, 1080)
    height_spin.setSingleStep(16)
    height_spin.setValue(int(overrides["video_height"]))
    height_spin.valueChanged.connect(lambda value: emit("video_height", int(value)))
    video_layout.addRow("Height", height_spin)

    layout.addRow(video_group)

    # --- Rewards Section ---
    rewards_group = QtWidgets.QGroupBox("Reward Overrides", group)
    rewards_layout = QtWidgets.QFormLayout(rewards_group)

    goal_spin = QtWidgets.QSpinBox(rewards_group)
    goal_spin.setRange(-100000, 100000)
    goal_spin.setSingleStep(100)
    goal_spin.setValue(int(overrides["reward_goal"]))
    goal_spin.valueChanged.connect(lambda value: emit("reward_goal", int(value)))
    rewards_layout.addRow("Goal Reward", goal_spin)

    timeout_spin = QtWidgets.QSpinBox(rewards_group)
    timeout_spin.setRange(-100000, 100000)
    timeout_spin.setSingleStep(100)
    timeout_spin.setValue(int(overrides["reward_timeout"]))
    timeout_spin.valueChanged.connect(lambda value: emit("reward_timeout", int(value)))
    rewards_layout.addRow("Timeout Penalty", timeout_spin)

    death_spin = QtWidgets.QSpinBox(rewards_group)
    death_spin.setRange(-100000, 100000)
    death_spin.setSingleStep(1000)
    death_spin.setValue(int(overrides["reward_death"]))
    death_spin.valueChanged.connect(lambda value: emit("reward_death", int(value)))
    rewards_layout.addRow("Death Penalty", death_spin)

    layout.addRow(rewards_group)

    # --- Connection Section (REMOVED) ---
    # User requested removal of manual port/server configuration
    # These defaults are still applied via DEFAULT_MALMO_CONFIG but not shown in UI.

    # conn_group = QtWidgets.QGroupBox("Minecraft Connection", group)
    # conn_layout = QtWidgets.QFormLayout(conn_group)
    # ...


__all__ = ["MALMOENV_FAMILY", "DEFAULT_MALMO_CONFIG", "build_malmo_controls"]
