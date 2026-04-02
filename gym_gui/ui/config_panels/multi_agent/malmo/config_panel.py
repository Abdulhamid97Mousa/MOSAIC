"""Configuration widgets for MalmoEnv multi-agent environments.

Multi-agent Malmo missions (e.g. TreasureHunt) require two or more Minecraft
clients running on consecutive ports.  This panel exposes settings specific
to the multi-agent case:

- Command quota (default: unlimited for human play)
- Time limit
- Tick rate
- Video resolution
- Connection ports
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from PyQt6 import QtWidgets

from gym_gui.core.enums import GameId

# Multi-agent MalmoEnv game IDs
MALMO_MULTIAGENT_GAME_IDS: tuple[GameId, ...] = (
    GameId.MALMOENV_TREASUREHUNT,
)

# Default configuration for multi-agent Malmo
DEFAULT_MALMO_MULTIAGENT_CONFIG: Dict[str, Any] = {
    "time_limit_ms": 0,       # 0 = unlimited
    "command_quota": 0,        # 0 = unlimited (remove from XML)
    "ms_per_tick": 50,         # 20 tps
    "video_width": 320,
    "video_height": 240,
    "port": 9000,
    "server": "localhost",
}


def build_malmo_multiagent_controls(
    *,
    layout: QtWidgets.QFormLayout,
    group: QtWidgets.QWidget,
    overrides: Dict[str, Any],
    on_change: Callable[[], None],
) -> None:
    """Build multi-agent MalmoEnv configuration controls.

    Args:
        layout: Form layout to add controls to.
        group: Parent widget for controls.
        overrides: Dict to store config overrides.
        on_change: Callback when any value changes.
    """
    # Initialize overrides with defaults
    for key, val in DEFAULT_MALMO_MULTIAGENT_CONFIG.items():
        overrides.setdefault(key, val)

    # --- Mission Timing ---
    timing_label = QtWidgets.QLabel("<b>Mission Timing</b>")
    layout.addRow(timing_label)

    # Time limit
    time_limit_check = QtWidgets.QCheckBox("Enable time limit")
    time_limit_spin = QtWidgets.QDoubleSpinBox()
    time_limit_spin.setRange(1.0, 3600.0)
    time_limit_spin.setValue(60.0)
    time_limit_spin.setSuffix(" seconds")
    time_limit_spin.setEnabled(False)

    def _on_time_limit_toggle(checked: bool) -> None:
        time_limit_spin.setEnabled(checked)
        if checked:
            overrides["time_limit_ms"] = int(time_limit_spin.value() * 1000)
        else:
            overrides["time_limit_ms"] = 0
        on_change()

    def _on_time_limit_changed(val: float) -> None:
        if time_limit_check.isChecked():
            overrides["time_limit_ms"] = int(val * 1000)
            on_change()

    time_limit_check.toggled.connect(_on_time_limit_toggle)
    time_limit_spin.valueChanged.connect(_on_time_limit_changed)
    layout.addRow(time_limit_check)
    layout.addRow("Time Limit", time_limit_spin)

    # Command quota
    quota_check = QtWidgets.QCheckBox("Enable command quota per agent")
    quota_spin = QtWidgets.QSpinBox()
    quota_spin.setRange(10, 10000)
    quota_spin.setValue(500)
    quota_spin.setEnabled(False)

    def _on_quota_toggle(checked: bool) -> None:
        quota_spin.setEnabled(checked)
        if checked:
            overrides["command_quota"] = quota_spin.value()
        else:
            overrides["command_quota"] = 0
        on_change()

    def _on_quota_changed(val: int) -> None:
        if quota_check.isChecked():
            overrides["command_quota"] = val
            on_change()

    quota_check.toggled.connect(_on_quota_toggle)
    quota_spin.valueChanged.connect(_on_quota_changed)
    layout.addRow(quota_check)
    layout.addRow("Command Quota", quota_spin)

    # Ms per tick
    tick_spin = QtWidgets.QSpinBox()
    tick_spin.setRange(10, 500)
    tick_spin.setValue(overrides.get("ms_per_tick", 50))
    tick_spin.setSuffix(" ms")

    def _on_tick_changed(val: int) -> None:
        overrides["ms_per_tick"] = val
        on_change()

    tick_spin.valueChanged.connect(_on_tick_changed)
    layout.addRow("Ms Per Tick", tick_spin)

    # --- Video Settings ---
    video_label = QtWidgets.QLabel("<b>Video Settings</b>")
    layout.addRow(video_label)

    width_spin = QtWidgets.QSpinBox()
    width_spin.setRange(80, 1920)
    width_spin.setValue(overrides.get("video_width", 320))

    height_spin = QtWidgets.QSpinBox()
    height_spin.setRange(60, 1080)
    height_spin.setValue(overrides.get("video_height", 240))

    def _on_width(val: int) -> None:
        overrides["video_width"] = val
        on_change()

    def _on_height(val: int) -> None:
        overrides["video_height"] = val
        on_change()

    width_spin.valueChanged.connect(_on_width)
    height_spin.valueChanged.connect(_on_height)
    layout.addRow("Width", width_spin)
    layout.addRow("Height", height_spin)

    # --- Connection ---
    conn_label = QtWidgets.QLabel("<b>Connection</b>")
    layout.addRow(conn_label)

    port_spin = QtWidgets.QSpinBox()
    port_spin.setRange(1024, 65535)
    port_spin.setValue(overrides.get("port", 9000))

    def _on_port(val: int) -> None:
        overrides["port"] = val
        on_change()

    port_spin.valueChanged.connect(_on_port)
    layout.addRow("Base Port", port_spin)

    # Info label
    info = QtWidgets.QLabel(
        "Multi-agent missions require one Minecraft client per agent.\n"
        "Launch with: ./run_malmo.sh --agents 2"
    )
    info.setWordWrap(True)
    info.setStyleSheet("color: #666; font-style: italic;")
    layout.addRow(info)
