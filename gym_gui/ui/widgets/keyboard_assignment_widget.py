"""Keyboard Assignment Widget for multi-human gameplay in the Human Control tab.

Allows multiple humans to play multi-agent games together (e.g., MultiGrid)
by assigning separate USB keyboards to each agent using evdev on Linux.

Device discovery uses human_worker.evdev_input (pure Python, no Qt dependency)
which is the same discovery the subprocess workers use at runtime.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, Optional, TypedDict

from PyQt6.QtCore import pyqtSignal
from qtpy import QtWidgets

from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_KEYBOARD_ASSIGNED,
    LOG_KEYBOARD_DETECTED,
    LOG_KEYBOARD_DETECTION_ERROR,
)

# Import human_worker discovery (Linux only, pure Python, no Qt)
_HAS_EVDEV = False
_discover_keyboards = None

if sys.platform.startswith('linux'):
    try:
        from gym_gui.config.paths import HUMAN_WORKER_PKG_DIR
        _hw_path = str(HUMAN_WORKER_PKG_DIR)
        if _hw_path not in sys.path:
            sys.path.insert(0, _hw_path)
        from human_worker.evdev_input import (
            discover_keyboards as _discover_keyboards,
        )
        _HAS_EVDEV = True
    except ImportError:
        pass


class RowWidgets(TypedDict):
    """Type definition for keyboard row widgets."""
    name: QtWidgets.QLabel
    device_path: QtWidgets.QLabel
    usb_port: QtWidgets.QLabel
    combo: QtWidgets.QComboBox
    status: QtWidgets.QLabel


class KeyboardAssignmentWidget(LogConstantMixin, QtWidgets.QGroupBox):
    """Widget for assigning keyboards to agents for multi-human gameplay using evdev."""

    # Signal: (device_path, agent_id) - per-keyboard, fires N times
    assignment_changed = pyqtSignal(str, object)
    # Signal: (assignments_dict) - fires ONCE with all assignments
    all_assignments_applied = pyqtSignal(dict)
    # Signal: (num_keyboards_detected)
    keyboards_detected = pyqtSignal(int)
    # Signal: (error_message)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        available_agents: Optional[List[str]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__("Keyboard Assignment (Multi-Human Evdev)", parent)

        self._logger = logging.getLogger(__name__)
        self._available_agents = available_agents or ["agent_0", "agent_1"]
        self._keyboards: Dict[str, Any] = {}  # {device_path: device_info}
        self._assignments: Dict[str, Optional[str]] = {}  # {device_path: agent_id}
        self._row_widgets: Dict[str, RowWidgets] = {}

        self._build_ui()

        if not _HAS_EVDEV:
            self._status.setText("Evdev not available (requires Linux with evdev support)")
            self.error_occurred.emit("Evdev support not available")

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        # Info
        info = QtWidgets.QLabel(
            "Assign USB keyboards to agents for multi-human gameplay using evdev.\n"
            "Platform: Linux only. Requires permissions (add user to 'input' group).",
            self
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Grid
        self._grid = QtWidgets.QGridLayout()

        headers = ["Device Name", "Device Path", "USB Port", "Assign To", "Status"]
        for col, text in enumerate(headers):
            lbl = QtWidgets.QLabel(text, self)
            lbl.setStyleSheet("font-weight: bold;")
            self._grid.addWidget(lbl, 0, col)

        layout.addLayout(self._grid)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()

        self._scan_btn = QtWidgets.QPushButton("Scan Keyboards", self)
        self._scan_btn.clicked.connect(self._detect_keyboards)
        btn_layout.addWidget(self._scan_btn)

        self._auto_assign_btn = QtWidgets.QPushButton("Auto-Assign", self)
        self._auto_assign_btn.clicked.connect(self._auto_assign)
        self._auto_assign_btn.setToolTip("Automatically assign keyboards to agents in detection order")
        btn_layout.addWidget(self._auto_assign_btn)

        self._apply_btn = QtWidgets.QPushButton("Apply Assignments", self)
        self._apply_btn.clicked.connect(self._apply_assignments)
        btn_layout.addWidget(self._apply_btn)

        self._clear_btn = QtWidgets.QPushButton("Clear All", self)
        self._clear_btn.clicked.connect(self._clear_assignments)
        btn_layout.addWidget(self._clear_btn)

        btn_layout.addStretch(1)
        layout.addLayout(btn_layout)

        # Agent info label
        self._agent_info = QtWidgets.QLabel("Agents: Waiting for environment...", self)
        self._agent_info.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self._agent_info)

        # Status
        self._status = QtWidgets.QLabel("Ready", self)
        layout.addWidget(self._status)

    def _detect_keyboards(self) -> None:
        """Scan for keyboard devices using human_worker.evdev_input."""
        if not _HAS_EVDEV or _discover_keyboards is None:
            self._status.setText("Evdev not available")
            return

        try:
            self._status.setText("Scanning for keyboards...")
            discovered = _discover_keyboards()

            # Update keyboards dict, preserving existing assignments
            new_keyboards: Dict[str, Any] = {}
            for device in discovered:
                device_path = device.device_path
                if device_path not in self._assignments:
                    self._assignments[device_path] = None
                new_keyboards[device_path] = device

            self._keyboards = new_keyboards
            self._refresh_rows()

            count = len(self._keyboards)
            assigned = sum(1 for agent_id in self._assignments.values() if agent_id)
            self._status.setText(f"Found {count} keyboard(s). {assigned} assigned.")
            self.keyboards_detected.emit(count)

            num_agents = len(self._available_agents)
            unassigned = max(0, num_agents - count)
            self._agent_info.setText(
                f"Environment has {num_agents} agents. "
                f"{count} keyboard(s) detected. "
                f"({unassigned} agent(s) will get NOOP actions)"
            )

            self.log_constant(LOG_KEYBOARD_DETECTED, extra={"count": count})

        except PermissionError:
            error_msg = (
                "Permission denied: Cannot access keyboard devices.\n"
                f"Fix: sudo usermod -a -G input {os.getenv('USER', 'your_username')}\n"
                "Then log out and log back in."
            )
            self._logger.error(error_msg)
            self._status.setText("Permission denied - see logs")
            self.error_occurred.emit(error_msg)
        except Exception as e:
            self.log_constant(LOG_KEYBOARD_DETECTION_ERROR, exc_info=e)
            self._status.setText(f"Error: {e}")
            self.error_occurred.emit(str(e))

    def _refresh_rows(self) -> None:
        """Refresh all keyboard rows in the UI."""
        for device_path in list(self._row_widgets.keys()):
            widgets = self._row_widgets[device_path]
            self._grid.removeWidget(widgets["name"])
            self._grid.removeWidget(widgets["device_path"])
            self._grid.removeWidget(widgets["usb_port"])
            self._grid.removeWidget(widgets["combo"])
            self._grid.removeWidget(widgets["status"])
            widgets["name"].deleteLater()
            widgets["device_path"].deleteLater()
            widgets["usb_port"].deleteLater()
            widgets["combo"].deleteLater()
            widgets["status"].deleteLater()
        self._row_widgets.clear()

        for row_idx, (device_path, device) in enumerate(sorted(self._keyboards.items())):
            grid_row = 1 + row_idx
            self._create_row(device_path, device, grid_row)

    def _create_row(self, device_path: str, device: Any, row: int) -> None:
        """Create a row in the keyboard grid for a device."""
        name_display = device.name if len(device.name) <= 40 else device.name[:37] + "..."
        name_lbl = QtWidgets.QLabel(name_display, self)
        name_lbl.setToolTip(device.name)
        self._grid.addWidget(name_lbl, row, 0)

        path_display = device_path.split("/")[-1] if "/" in device_path else device_path
        path_lbl = QtWidgets.QLabel(path_display, self)
        path_lbl.setToolTip(device_path)
        self._grid.addWidget(path_lbl, row, 1)

        usb_port_display = device.usb_port or "N/A"
        usb_lbl = QtWidgets.QLabel(usb_port_display, self)
        self._grid.addWidget(usb_lbl, row, 2)

        combo = QtWidgets.QComboBox(self)
        combo.addItem("(Unassigned)", None)
        for agent_id in self._available_agents:
            display = f"Agent {agent_id.split('_')[-1]}" if "_" in agent_id else agent_id
            combo.addItem(display, agent_id)

        assigned_agent = self._assignments.get(device_path)
        if assigned_agent:
            idx = combo.findData(assigned_agent)
            if idx >= 0:
                combo.setCurrentIndex(idx)

        combo.currentIndexChanged.connect(
            lambda i, dp=device_path: self._on_assign(dp, i)
        )
        self._grid.addWidget(combo, row, 3)

        status_text = "Assigned" if assigned_agent else "Idle"
        status_lbl = QtWidgets.QLabel(status_text, self)
        self._grid.addWidget(status_lbl, row, 4)

        self._row_widgets[device_path] = {
            "name": name_lbl,
            "device_path": path_lbl,
            "usb_port": usb_lbl,
            "combo": combo,
            "status": status_lbl,
        }

    def _on_assign(self, device_path: str, combo_idx: int) -> None:
        """Handle assignment change for a keyboard."""
        if device_path not in self._keyboards:
            return

        combo = self._row_widgets[device_path]["combo"]
        agent_id = combo.itemData(combo_idx)
        self._assignments[device_path] = agent_id

        status = self._row_widgets[device_path]["status"]
        status.setText("Assigned" if agent_id else "Idle")

        assigned_count = sum(1 for aid in self._assignments.values() if aid)
        self._status.setText(f"Found {len(self._keyboards)} keyboard(s). {assigned_count} assigned.")

        device = self._keyboards[device_path]
        self.log_constant(
            LOG_KEYBOARD_ASSIGNED,
            extra={
                "device_path": device_path,
                "keyboard_name": device.name,
                "agent_id": agent_id or "unassigned",
            }
        )

        self.assignment_changed.emit(device_path, agent_id)

    def _auto_assign(self) -> None:
        """Automatically assign keyboards to agents in detection order."""
        if not self._keyboards:
            self._status.setText("No keyboards detected. Scan first.")
            return

        if not self._available_agents:
            self._status.setText("No agents available.")
            return

        sorted_devices = sorted(self._keyboards.items(), key=lambda x: x[0])

        for i, (device_path, device) in enumerate(sorted_devices):
            if i < len(self._available_agents):
                agent_id = self._available_agents[i]
                self._assignments[device_path] = agent_id
                self._logger.info(f"Auto-assigned: {device.name} -> {agent_id}")
            else:
                self._assignments[device_path] = None

        self._refresh_rows()

        assigned_count = sum(1 for aid in self._assignments.values() if aid)
        self._status.setText(f"Auto-assigned {assigned_count} keyboard(s) to agents")

    def _apply_assignments(self) -> None:
        """Apply current keyboard assignments (emit signal for integration)."""
        assignments = {
            device_path: agent_id
            for device_path, agent_id in self._assignments.items()
            if agent_id is not None
        }

        if not assignments:
            self._status.setText("No keyboards assigned. Use Auto-Assign or assign manually.")
            return

        self._logger.info(f"Applying {len(assignments)} keyboard assignments")

        for device_path, agent_id in assignments.items():
            self.assignment_changed.emit(device_path, agent_id)

        self.all_assignments_applied.emit(assignments)
        self._status.setText(f"Applied {len(assignments)} keyboard assignment(s)")

    def _clear_assignments(self) -> None:
        """Clear all keyboard assignments."""
        for device_path in self._assignments:
            self._assignments[device_path] = None

        self._refresh_rows()
        self._status.setText(f"Cleared all assignments. {len(self._keyboards)} keyboard(s) detected.")
        self._logger.info("Cleared all keyboard assignments")

    # Public API

    def auto_assign_and_apply(self) -> Dict[str, str]:
        """Auto-assign keyboards to agents and apply immediately."""
        self._detect_keyboards()
        self._auto_assign()
        self._apply_assignments()
        return self.get_assignments()

    def get_assignments(self) -> Dict[str, str]:
        """Returns {device_path: agent_id} for assigned keyboards only."""
        return {
            device_path: agent_id
            for device_path, agent_id in self._assignments.items()
            if agent_id is not None
        }

    def get_agent_keyboard(self, agent_id: str) -> Optional[str]:
        """Returns device_path of keyboard assigned to agent."""
        for device_path, assigned_agent in self._assignments.items():
            if assigned_agent == agent_id:
                return device_path
        return None

    def set_available_agents(self, agents: List[str]) -> None:
        """Update available agents list and refresh all combo boxes."""
        self._logger.info(
            f"set_available_agents called with {len(agents)} agents: {agents}"
        )
        self._available_agents = agents
        self._refresh_rows()

        num_agents = len(agents)
        keyboards = len(self._keyboards)
        unassigned = num_agents - min(num_agents, keyboards)
        self._agent_info.setText(
            f"Environment has {num_agents} agents. "
            f"{keyboards} keyboard(s) detected. "
            f"({unassigned} agent(s) will get NOOP actions)"
        )
        self._agent_info.setStyleSheet("color: #333; font-style: normal;")

    def get_detected_keyboards(self) -> List[Any]:
        """Get list of all detected keyboards."""
        return list(self._keyboards.values())

    def cleanup(self) -> None:
        """Cleanup resources when widget is destroyed."""
        pass
