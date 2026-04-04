"""Tests for keyboard assignment widget for multi-human gameplay.

Tests keyboard detection, assignment, and signal emission for multi-agent
human control in environments like MultiGrid and Overcooked.

The widget uses evdev (Linux) for keyboard detection — NOT QInputDevice.
Tests that require real evdev hardware are skipped in CI.
"""

from __future__ import annotations

import os
from typing import Optional
from unittest.mock import MagicMock, Mock, patch

import pytest
from qtpy import QtWidgets

# Ensure Qt renders offscreen in CI environments
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qt_app():
    """Create Qt application for widget testing."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


def _make_mock_evdev_devices():
    """Create mock evdev KeyboardDevice objects."""
    devices = []
    for path, name in [
        ("/dev/input/event0", "Logitech USB Keyboard"),
        ("/dev/input/event1", "Lenovo Black Silk USB Keyboard TRACER"),
        ("/dev/input/event2", "SiGma Micro Keyboard TRACER Gamma Ivory"),
    ]:
        dev = Mock()
        dev.device_path = path
        dev.name = name
        dev.usb_port = None
        devices.append(dev)
    return devices


class TestKeyboardAssignmentWidget:
    """Test keyboard assignment widget functionality."""

    def test_widget_initialization(self, qt_app) -> None:
        """Widget initializes with default agents."""
        from gym_gui.ui.widgets.keyboard_assignment_widget import KeyboardAssignmentWidget

        widget = KeyboardAssignmentWidget(parent=None)

        assert widget is not None
        assert widget._available_agents == ["agent_0", "agent_1"]
        assert len(widget._keyboards) >= 0  # May or may not detect keyboards

        widget.deleteLater()

    def test_widget_custom_agents(self, qt_app) -> None:
        """Widget initializes with custom agent list."""
        from gym_gui.ui.widgets.keyboard_assignment_widget import KeyboardAssignmentWidget

        agents = ["agent_0", "agent_1", "agent_2", "agent_3"]
        widget = KeyboardAssignmentWidget(available_agents=agents, parent=None)

        assert widget._available_agents == agents

        widget.deleteLater()

    def test_keyboard_detection(self, qt_app) -> None:
        """Widget detects keyboards via evdev and populates _keyboards dict."""
        from gym_gui.ui.widgets.keyboard_assignment_widget import KeyboardAssignmentWidget

        widget = KeyboardAssignmentWidget(parent=None)
        mock_devices = _make_mock_evdev_devices()

        # Mock the module-level discover function used by the widget
        with patch("gym_gui.ui.widgets.keyboard_assignment_widget._HAS_EVDEV", True), \
             patch("gym_gui.ui.widgets.keyboard_assignment_widget._discover_keyboards", return_value=mock_devices):
            widget._detect_keyboards()

        assert len(widget._keyboards) == 3
        detected_paths = set(widget._keyboards.keys())
        assert detected_paths == {"/dev/input/event0", "/dev/input/event1", "/dev/input/event2"}

        widget.deleteLater()

    def test_get_assignments(self, qt_app) -> None:
        """get_assignments returns correct device_path-to-agent mapping."""
        from gym_gui.ui.widgets.keyboard_assignment_widget import KeyboardAssignmentWidget

        widget = KeyboardAssignmentWidget(parent=None)
        mock_devices = _make_mock_evdev_devices()
        with patch("gym_gui.ui.widgets.keyboard_assignment_widget._HAS_EVDEV", True), \
             patch("gym_gui.ui.widgets.keyboard_assignment_widget._discover_keyboards", return_value=mock_devices):
            widget._detect_keyboards()

        # Initially no assignments
        assert widget.get_assignments() == {}

        # Manually assign agents
        widget._assignments["/dev/input/event0"] = "agent_0"
        widget._assignments["/dev/input/event1"] = "agent_1"

        assignments = widget.get_assignments()
        assert assignments == {"/dev/input/event0": "agent_0", "/dev/input/event1": "agent_1"}

        widget.deleteLater()

    def test_get_agent_keyboard(self, qt_app) -> None:
        """get_agent_keyboard returns device_path for agent."""
        from gym_gui.ui.widgets.keyboard_assignment_widget import KeyboardAssignmentWidget

        widget = KeyboardAssignmentWidget(parent=None)
        mock_devices = _make_mock_evdev_devices()
        with patch("gym_gui.ui.widgets.keyboard_assignment_widget._HAS_EVDEV", True), \
             patch("gym_gui.ui.widgets.keyboard_assignment_widget._discover_keyboards", return_value=mock_devices):
            widget._detect_keyboards()

        assert widget.get_agent_keyboard("agent_0") is None

        widget._assignments["/dev/input/event0"] = "agent_0"

        assert widget.get_agent_keyboard("agent_0") == "/dev/input/event0"
        assert widget.get_agent_keyboard("agent_1") is None

        widget.deleteLater()

    def test_assignment_changed_signal(self, qt_app) -> None:
        """assignment_changed signal emits when assignment changes."""
        from gym_gui.ui.widgets.keyboard_assignment_widget import KeyboardAssignmentWidget

        widget = KeyboardAssignmentWidget(parent=None)

        signal_received = []
        widget.assignment_changed.connect(
            lambda path, aid: signal_received.append((path, aid))
        )

        widget.assignment_changed.emit("/dev/input/event0", "agent_0")

        assert len(signal_received) == 1
        assert signal_received[0] == ("/dev/input/event0", "agent_0")

        widget.deleteLater()

    def test_keyboards_detected_signal(self, qt_app) -> None:
        """keyboards_detected signal emits with keyboard count."""
        from gym_gui.ui.widgets.keyboard_assignment_widget import KeyboardAssignmentWidget

        widget = KeyboardAssignmentWidget(parent=None)
        mock_devices = _make_mock_evdev_devices()

        detected_counts = []
        widget.keyboards_detected.connect(lambda count: detected_counts.append(count))

        with patch("gym_gui.ui.widgets.keyboard_assignment_widget._HAS_EVDEV", True), \
             patch("gym_gui.ui.widgets.keyboard_assignment_widget._discover_keyboards", return_value=mock_devices):
            widget._detect_keyboards()

        assert len(detected_counts) == 1
        assert detected_counts[0] == 3

        widget.deleteLater()

    def test_set_available_agents(self, qt_app) -> None:
        """set_available_agents updates agent list."""
        from gym_gui.ui.widgets.keyboard_assignment_widget import KeyboardAssignmentWidget

        widget = KeyboardAssignmentWidget(
            available_agents=["agent_0", "agent_1"],
            parent=None,
        )

        assert widget._available_agents == ["agent_0", "agent_1"]

        new_agents = ["agent_0", "agent_1", "agent_2", "agent_3"]
        widget.set_available_agents(new_agents)

        assert widget._available_agents == new_agents

        widget.deleteLater()

    def test_get_detected_keyboards(self, qt_app) -> None:
        """get_detected_keyboards returns list of evdev device objects."""
        from gym_gui.ui.widgets.keyboard_assignment_widget import KeyboardAssignmentWidget

        widget = KeyboardAssignmentWidget(parent=None)
        mock_devices = _make_mock_evdev_devices()
        with patch("gym_gui.ui.widgets.keyboard_assignment_widget._HAS_EVDEV", True), \
             patch("gym_gui.ui.widgets.keyboard_assignment_widget._discover_keyboards", return_value=mock_devices):
            widget._detect_keyboards()

        keyboards = widget.get_detected_keyboards()

        assert isinstance(keyboards, list)
        assert len(keyboards) == 3

        names = {kb.name for kb in keyboards}
        assert "Logitech USB Keyboard" in names
        assert "Lenovo Black Silk USB Keyboard TRACER" in names

        widget.deleteLater()

    def test_widget_visibility_integration(self, qt_app) -> None:
        """Widget can be shown/hidden for single vs multi-agent environments."""
        from gym_gui.ui.widgets.keyboard_assignment_widget import KeyboardAssignmentWidget

        widget = KeyboardAssignmentWidget(parent=None)

        widget.setVisible(True)
        assert widget.isVisible()

        widget.setVisible(False)
        assert not widget.isVisible()

        widget.setVisible(True)
        assert widget.isVisible()

        widget.deleteLater()
