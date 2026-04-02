"""Tests for KeyboardWorkerBridge: subprocess-per-agent keyboard input.

Tests the bridge that manages HumanKeyboardRuntime subprocesses,
one per agent, for human play. All subprocess I/O is mocked;
no real evdev devices or child processes are needed.

Test groups:
    - _KeyboardWorkerHandle: send/read/stop protocol
    - KeyboardWorkerBridge (single-agent): one keyboard, one agent
    - KeyboardWorkerBridge (multi-agent): two keyboards, two agents
    - Lifecycle: start/stop/restart, dead process handling
    - Signal emission: all_actions_ready ordering, action_received per-agent
"""

from __future__ import annotations

import json
import io
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from qtpy import QtCore
from qtpy.QtWidgets import QApplication

from gym_gui.controllers.keyboard_worker_bridge import (
    KeyboardWorkerBridge,
    _KeyboardWorkerHandle,
    _STATE_READY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_process(
    responses: Optional[List[Dict[str, Any]]] = None,
    returncode: Optional[int] = None,
) -> MagicMock:
    """Create a mock subprocess.Popen with controllable stdout/stdin.

    Args:
        responses: JSON messages the process will "emit" on stdout.
        returncode: If set, poll() returns this (process is dead).
    """
    proc = MagicMock()

    # stdin
    proc.stdin = MagicMock()

    # stdout: line-buffered StringIO with the queued responses
    lines = []
    if responses:
        for resp in responses:
            lines.append(json.dumps(resp) + "\n")
    proc.stdout = io.StringIO("".join(lines))

    # poll
    proc.poll.return_value = returncode  # None = running
    proc.wait.return_value = 0
    proc.kill.return_value = None

    return proc


class _SignalCollector:
    """Collects Qt signal emissions for assertions."""

    def __init__(self) -> None:
        self.calls: list = []

    def __call__(self, *args: Any) -> None:
        self.calls.append(args)


# ---------------------------------------------------------------------------
# Ensure QApplication exists (required by QObject/QTimer)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def _qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# ===========================================================================
# _KeyboardWorkerHandle tests
# ===========================================================================

class TestKeyboardWorkerHandle:
    """Unit tests for the per-agent subprocess handle."""

    def test_is_running_when_poll_is_none(self):
        proc = _make_fake_process()
        handle = _KeyboardWorkerHandle("agent_0", "/dev/input/event0", proc)
        assert handle.is_running is True

    def test_is_not_running_when_poll_returns_code(self):
        proc = _make_fake_process(returncode=0)
        handle = _KeyboardWorkerHandle("agent_0", "/dev/input/event0", proc)
        assert handle.is_running is False

    def test_send_command_writes_json_line(self):
        proc = _make_fake_process()
        handle = _KeyboardWorkerHandle("agent_0", "/dev/input/event0", proc)

        result = handle.send_command({"cmd": "select_action", "player_id": "agent_0"})

        assert result is True
        written = proc.stdin.write.call_args[0][0]
        parsed = json.loads(written.strip())
        assert parsed["cmd"] == "select_action"
        assert parsed["player_id"] == "agent_0"

    def test_send_command_fails_on_dead_process(self):
        proc = _make_fake_process(returncode=1)
        handle = _KeyboardWorkerHandle("agent_0", "/dev/input/event0", proc)
        assert handle.send_command({"cmd": "stop"}) is False

    def test_send_command_fails_on_broken_pipe(self):
        proc = _make_fake_process()
        proc.stdin.write.side_effect = BrokenPipeError("pipe gone")
        handle = _KeyboardWorkerHandle("agent_0", "/dev/input/event0", proc)
        assert handle.send_command({"cmd": "stop"}) is False

    def test_try_read_response_returns_json(self):
        responses = [{"type": "init", "version": "1.0"}]
        proc = _make_fake_process(responses=responses)
        handle = _KeyboardWorkerHandle("agent_0", "/dev/input/event0", proc)

        # Patch the select import inside try_read_response
        with patch("select.select", return_value=([proc.stdout], [], [])):
            resp = handle.try_read_response(timeout=1.0)

        assert resp is not None
        assert resp["type"] == "init"

    def test_try_read_response_returns_none_on_dead_process(self):
        proc = _make_fake_process(returncode=0)
        handle = _KeyboardWorkerHandle("agent_0", "/dev/input/event0", proc)
        assert handle.try_read_response(timeout=0.0) is None

    def test_stop_sends_stop_command_and_waits(self):
        proc = _make_fake_process()
        handle = _KeyboardWorkerHandle("agent_0", "/dev/input/event0", proc)
        handle.stop(timeout=1.0)

        # Verify stop command was sent
        written = proc.stdin.write.call_args[0][0]
        parsed = json.loads(written.strip())
        assert parsed["cmd"] == "stop"
        proc.wait.assert_called_once_with(timeout=1.0)

    def test_stop_kills_on_timeout(self):
        proc = _make_fake_process()
        import subprocess as sp
        proc.wait.side_effect = [sp.TimeoutExpired("cmd", 1.0), 0]
        handle = _KeyboardWorkerHandle("agent_0", "/dev/input/event0", proc)
        handle.stop(timeout=1.0)

        proc.kill.assert_called_once()

    def test_stop_closes_log_file(self):
        proc = _make_fake_process()
        handle = _KeyboardWorkerHandle("agent_0", "/dev/input/event0", proc)
        mock_log = MagicMock()
        handle._log_file = mock_log

        handle.stop(timeout=1.0)
        mock_log.close.assert_called_once()
        assert handle._log_file is None


# ===========================================================================
# KeyboardWorkerBridge - Single Agent
# ===========================================================================

class TestBridgeSingleAgent:
    """Tests for bridge with a single keyboard/agent assignment."""

    def test_start_launches_subprocess_and_returns_true(self, _qapp):
        bridge = KeyboardWorkerBridge()

        init_resp = {"type": "init", "version": "1.0"}
        agent_init_resp = {"type": "agent_initialized", "player_id": "agent_0"}
        fake_proc = _make_fake_process(responses=[init_resp, agent_init_resp])

        with patch("gym_gui.controllers.keyboard_worker_bridge.subprocess.Popen", return_value=fake_proc):
            with patch("gym_gui.controllers.keyboard_worker_bridge.VAR_HUMAN_CONTROL_LOGS_DIR") as mock_dir:
                mock_path = MagicMock()
                mock_path.open.return_value = MagicMock()
                mock_dir.__truediv__ = MagicMock(return_value=mock_path)
                mock_dir.mkdir = MagicMock()
                with patch("gym_gui.controllers.keyboard_worker_bridge.ensure_var_directories"):
                    # Patch try_read_response to return from our fake stdout
                    with patch.object(
                        _KeyboardWorkerHandle,
                        "try_read_response",
                        side_effect=[init_resp, agent_init_resp],
                    ):
                        result = bridge.start(
                            assignments={"/dev/input/event0": "agent_0"},
                            env_name="multigrid",
                        )

        assert result is True
        assert bridge.is_active is True
        assert "agent_0" in bridge._handles
        assert bridge._agent_order == ["agent_0"]
        bridge.stop()

    def test_start_is_nonblocking_init_handled_by_poll(self, _qapp):
        """start() is non-blocking: spawns process, returns True, init via poll."""
        bridge = KeyboardWorkerBridge()
        fake_proc = _make_fake_process(responses=[])

        with patch("gym_gui.controllers.keyboard_worker_bridge.subprocess.Popen", return_value=fake_proc):
            with patch("gym_gui.controllers.keyboard_worker_bridge.VAR_HUMAN_CONTROL_LOGS_DIR") as mock_dir:
                mock_path = MagicMock()
                mock_path.open.return_value = MagicMock()
                mock_dir.__truediv__ = MagicMock(return_value=mock_path)
                mock_dir.mkdir = MagicMock()
                with patch("gym_gui.controllers.keyboard_worker_bridge.ensure_var_directories"):
                    result = bridge.start(
                        assignments={"/dev/input/event0": "agent_0"},
                    )

        # start() always returns True if Popen succeeded (init is async)
        assert result is True
        assert bridge.is_active is True
        # Worker is still initializing (not ready yet)
        handle = bridge._handles["agent_0"]
        from gym_gui.controllers.keyboard_worker_bridge import _STATE_WAIT_INIT
        assert handle.state == _STATE_WAIT_INIT
        bridge.stop()

    def test_stop_clears_handles_and_timer(self, _qapp):
        bridge = KeyboardWorkerBridge()
        # Manually inject a handle
        proc = _make_fake_process()
        handle = _KeyboardWorkerHandle("agent_0", "/dev/input/event0", proc)
        bridge._handles["agent_0"] = handle
        bridge._agent_order = ["agent_0"]
        bridge._poll_timer.start()

        bridge.stop()

        assert bridge._handles == {}
        assert bridge._agent_order == []
        assert bridge._poll_timer.isActive() is False

    def test_poll_collects_action_and_emits_signal(self, _qapp):
        bridge = KeyboardWorkerBridge()

        proc = _make_fake_process()
        handle = _KeyboardWorkerHandle("agent_0", "/dev/input/event0", proc)
        handle.state = _STATE_READY
        bridge._handles["agent_0"] = handle
        bridge._agent_order = ["agent_0"]
        bridge._awaiting_response = {"agent_0"}

        # Collect signals
        ready_collector = _SignalCollector()
        action_collector = _SignalCollector()
        bridge.all_actions_ready.connect(ready_collector)
        bridge.action_received.connect(action_collector)

        # Mock the response
        with patch.object(
            handle,
            "try_read_response",
            return_value={"type": "action_selected", "action": 3},
        ):
            bridge._poll_responses()

        # Single agent: should emit immediately
        assert len(ready_collector.calls) == 1
        assert ready_collector.calls[0] == ([3],)

        assert len(action_collector.calls) == 1
        assert action_collector.calls[0] == ("agent_0", 3)

    def test_request_next_round_clears_and_resends(self, _qapp):
        bridge = KeyboardWorkerBridge()

        proc = _make_fake_process()
        handle = _KeyboardWorkerHandle("agent_0", "/dev/input/event0", proc)
        handle.state = _STATE_READY
        bridge._handles["agent_0"] = handle
        bridge._agent_order = ["agent_0"]
        bridge._pending_actions = {"agent_0": 3}
        bridge._awaiting_response = set()

        bridge.request_next_round()

        assert bridge._pending_actions == {}
        assert "agent_0" in bridge._awaiting_response
        # Verify select_action was sent
        written = proc.stdin.write.call_args[0][0]
        parsed = json.loads(written.strip())
        assert parsed["cmd"] == "select_action"


# ===========================================================================
# KeyboardWorkerBridge - Multi Agent
# ===========================================================================

class TestBridgeMultiAgent:
    """Tests for bridge with multiple keyboard/agent assignments."""

    def test_start_launches_two_workers(self, _qapp):
        bridge = KeyboardWorkerBridge()

        init_resp = {"type": "init", "version": "1.0"}
        agent_init_resp_0 = {"type": "agent_initialized", "player_id": "agent_0"}
        agent_init_resp_1 = {"type": "agent_initialized", "player_id": "agent_1"}

        call_count = [0]
        responses_by_call = [
            [init_resp, agent_init_resp_0],  # agent_0
            [init_resp, agent_init_resp_1],  # agent_1
        ]

        def fake_try_read(timeout=0.0):
            # Alternate between agent responses
            idx = call_count[0]
            call_count[0] += 1
            flat_responses = responses_by_call[0] + responses_by_call[1]
            if idx < len(flat_responses):
                return flat_responses[idx]
            return None

        fake_proc = _make_fake_process()

        with patch("gym_gui.controllers.keyboard_worker_bridge.subprocess.Popen", return_value=fake_proc):
            with patch("gym_gui.controllers.keyboard_worker_bridge.VAR_HUMAN_CONTROL_LOGS_DIR") as mock_dir:
                mock_path = MagicMock()
                mock_path.open.return_value = MagicMock()
                mock_dir.__truediv__ = MagicMock(return_value=mock_path)
                mock_dir.mkdir = MagicMock()
                with patch("gym_gui.controllers.keyboard_worker_bridge.ensure_var_directories"):
                    with patch.object(
                        _KeyboardWorkerHandle,
                        "try_read_response",
                        side_effect=fake_try_read,
                    ):
                        result = bridge.start(
                            assignments={
                                "/dev/input/event0": "agent_0",
                                "/dev/input/event1": "agent_1",
                            },
                            env_name="multigrid",
                        )

        assert result is True
        assert len(bridge._handles) == 2
        assert bridge._agent_order == ["agent_0", "agent_1"]
        bridge.stop()

    def test_poll_waits_for_all_agents_before_emitting(self, _qapp):
        bridge = KeyboardWorkerBridge()

        # Set up two handles
        proc0 = _make_fake_process()
        proc1 = _make_fake_process()
        handle0 = _KeyboardWorkerHandle("agent_0", "/dev/input/event0", proc0)
        handle1 = _KeyboardWorkerHandle("agent_1", "/dev/input/event1", proc1)
        handle0.state = _STATE_READY
        handle1.state = _STATE_READY
        bridge._handles = {"agent_0": handle0, "agent_1": handle1}
        bridge._agent_order = ["agent_0", "agent_1"]
        bridge._awaiting_response = {"agent_0", "agent_1"}

        ready_collector = _SignalCollector()
        bridge.all_actions_ready.connect(ready_collector)

        # Poll with only agent_0 responding
        with patch.object(handle0, "try_read_response", return_value={"type": "action_selected", "action": 1}):
            with patch.object(handle1, "try_read_response", return_value=None):
                bridge._poll_responses()

        # Should NOT emit yet (only 1 of 2 agents responded)
        assert len(ready_collector.calls) == 0
        assert bridge._pending_actions == {"agent_0": 1}

        # Now agent_1 responds
        with patch.object(handle0, "try_read_response", return_value=None):
            with patch.object(handle1, "try_read_response", return_value={"type": "action_selected", "action": 2}):
                bridge._poll_responses()

        # NOW it should emit with both actions in order
        assert len(ready_collector.calls) == 1
        assert ready_collector.calls[0] == ([1, 2],)

    def test_action_order_matches_sorted_agent_ids(self, _qapp):
        """Actions list must follow sorted agent_id order, not insertion order."""
        bridge = KeyboardWorkerBridge()

        proc0 = _make_fake_process()
        proc1 = _make_fake_process()
        handle0 = _KeyboardWorkerHandle("agent_0", "/dev/input/event0", proc0)
        handle1 = _KeyboardWorkerHandle("agent_1", "/dev/input/event1", proc1)
        handle0.state = _STATE_READY
        handle1.state = _STATE_READY
        # Insert in reverse order
        bridge._handles = {"agent_1": handle1, "agent_0": handle0}
        bridge._agent_order = ["agent_0", "agent_1"]
        bridge._awaiting_response = {"agent_0", "agent_1"}
        bridge._all_ready_emitted = True

        ready_collector = _SignalCollector()
        bridge.all_actions_ready.connect(ready_collector)

        # agent_1 responds FIRST with action 5, agent_0 with action 3
        with patch.object(handle1, "try_read_response", return_value={"type": "action_selected", "action": 5}):
            with patch.object(handle0, "try_read_response", return_value={"type": "action_selected", "action": 3}):
                bridge._poll_responses()

        # Order should be [agent_0_action, agent_1_action] = [3, 5]
        assert ready_collector.calls[0] == ([3, 5],)

    def test_skip_already_acted_agents(self, _qapp):
        bridge = KeyboardWorkerBridge()

        proc = _make_fake_process()
        handle = _KeyboardWorkerHandle("agent_0", "/dev/input/event0", proc)
        handle.state = _STATE_READY
        bridge._handles = {"agent_0": handle}
        bridge._agent_order = ["agent_0"]
        bridge._pending_actions = {"agent_0": 3}  # Already acted

        # Should not read from this agent
        with patch.object(handle, "try_read_response") as mock_read:
            bridge._poll_responses()
            mock_read.assert_not_called()

    def test_per_agent_action_received_signal(self, _qapp):
        bridge = KeyboardWorkerBridge()

        proc0 = _make_fake_process()
        proc1 = _make_fake_process()
        handle0 = _KeyboardWorkerHandle("agent_0", "/dev/input/event0", proc0)
        handle1 = _KeyboardWorkerHandle("agent_1", "/dev/input/event1", proc1)
        handle0.state = _STATE_READY
        handle1.state = _STATE_READY
        bridge._handles = {"agent_0": handle0, "agent_1": handle1}
        bridge._agent_order = ["agent_0", "agent_1"]
        bridge._awaiting_response = {"agent_0", "agent_1"}
        bridge._all_ready_emitted = True

        action_collector = _SignalCollector()
        bridge.action_received.connect(action_collector)

        with patch.object(handle0, "try_read_response", return_value={"type": "action_selected", "action": 1}):
            with patch.object(handle1, "try_read_response", return_value={"type": "action_selected", "action": 2}):
                bridge._poll_responses()

        # Both per-agent signals emitted
        assert ("agent_0", 1) in action_collector.calls
        assert ("agent_1", 2) in action_collector.calls


# ===========================================================================
# Lifecycle tests
# ===========================================================================

class TestBridgeLifecycle:
    """Start, stop, restart, dead-process scenarios."""

    def test_stop_on_empty_bridge_is_noop(self, _qapp):
        bridge = KeyboardWorkerBridge()
        bridge.stop()  # Should not raise
        assert bridge._handles == {}

    def test_double_stop_is_safe(self, _qapp):
        bridge = KeyboardWorkerBridge()
        proc = _make_fake_process()
        handle = _KeyboardWorkerHandle("agent_0", "/dev/input/event0", proc)
        bridge._handles["agent_0"] = handle
        bridge._agent_order = ["agent_0"]

        bridge.stop()
        bridge.stop()  # Second call should be safe
        assert bridge._handles == {}

    def test_start_stops_previous_workers_first(self, _qapp):
        bridge = KeyboardWorkerBridge()

        # Inject an existing handle
        old_proc = _make_fake_process()
        old_handle = _KeyboardWorkerHandle("old_agent", "/dev/input/event99", old_proc)
        bridge._handles["old_agent"] = old_handle
        bridge._agent_order = ["old_agent"]

        init_resp = {"type": "init", "version": "1.0"}
        agent_init_resp = {"type": "agent_initialized", "player_id": "agent_0"}

        fake_proc = _make_fake_process()

        with patch("gym_gui.controllers.keyboard_worker_bridge.subprocess.Popen", return_value=fake_proc):
            with patch("gym_gui.controllers.keyboard_worker_bridge.VAR_HUMAN_CONTROL_LOGS_DIR") as mock_dir:
                mock_path = MagicMock()
                mock_path.open.return_value = MagicMock()
                mock_dir.__truediv__ = MagicMock(return_value=mock_path)
                mock_dir.mkdir = MagicMock()
                with patch("gym_gui.controllers.keyboard_worker_bridge.ensure_var_directories"):
                    with patch.object(
                        _KeyboardWorkerHandle,
                        "try_read_response",
                        side_effect=[init_resp, agent_init_resp],
                    ):
                        bridge.start(
                            assignments={"/dev/input/event0": "agent_0"},
                        )

        # Old handle should have been stopped (stop sends "stop" cmd)
        old_proc.stdin.write.assert_called()
        written = old_proc.stdin.write.call_args[0][0]
        assert json.loads(written.strip())["cmd"] == "stop"
        bridge.stop()

    def test_dead_worker_skipped_in_send_select_action(self, _qapp):
        bridge = KeyboardWorkerBridge()

        proc = _make_fake_process(returncode=1)  # Dead
        handle = _KeyboardWorkerHandle("agent_0", "/dev/input/event0", proc)
        bridge._handles["agent_0"] = handle
        bridge._agent_order = ["agent_0"]

        bridge._send_select_action_to_all()

        # Should NOT try to send (process is dead)
        assert "agent_0" not in bridge._awaiting_response

    def test_is_active_false_when_no_handles(self, _qapp):
        bridge = KeyboardWorkerBridge()
        assert bridge.is_active is False

    def test_is_active_false_when_timer_stopped(self, _qapp):
        bridge = KeyboardWorkerBridge()
        proc = _make_fake_process()
        bridge._handles["agent_0"] = _KeyboardWorkerHandle("agent_0", "/dev/input/event0", proc)
        # Timer not started
        assert bridge.is_active is False

    def test_poll_ignores_non_action_responses(self, _qapp):
        bridge = KeyboardWorkerBridge()

        proc = _make_fake_process()
        handle = _KeyboardWorkerHandle("agent_0", "/dev/input/event0", proc)
        bridge._handles = {"agent_0": handle}
        bridge._agent_order = ["agent_0"]
        bridge._awaiting_response = {"agent_0"}

        ready_collector = _SignalCollector()
        bridge.all_actions_ready.connect(ready_collector)

        # Return a non-action response
        with patch.object(handle, "try_read_response", return_value={"type": "agent_ready"}):
            bridge._poll_responses()

        # Should NOT emit or record action
        assert len(ready_collector.calls) == 0
        assert "agent_0" not in bridge._pending_actions

    def test_poll_ignores_action_with_none_value(self, _qapp):
        bridge = KeyboardWorkerBridge()

        proc = _make_fake_process()
        handle = _KeyboardWorkerHandle("agent_0", "/dev/input/event0", proc)
        bridge._handles = {"agent_0": handle}
        bridge._agent_order = ["agent_0"]
        bridge._awaiting_response = {"agent_0"}

        ready_collector = _SignalCollector()
        bridge.all_actions_ready.connect(ready_collector)

        with patch.object(handle, "try_read_response", return_value={"type": "action_selected", "action": None}):
            bridge._poll_responses()

        assert len(ready_collector.calls) == 0
        assert "agent_0" not in bridge._pending_actions


# ===========================================================================
# Three-agent scenario
# ===========================================================================

class TestBridgeThreeAgents:
    """Edge case: three agents, actions arrive in arbitrary order."""

    def test_three_agents_collect_in_order(self, _qapp):
        bridge = KeyboardWorkerBridge()

        handles = {}
        for i in range(3):
            proc = _make_fake_process()
            h = _KeyboardWorkerHandle(f"agent_{i}", f"/dev/input/event{i}", proc)
            h.state = _STATE_READY
            handles[f"agent_{i}"] = h

        bridge._handles = handles
        bridge._agent_order = ["agent_0", "agent_1", "agent_2"]
        bridge._awaiting_response = {"agent_0", "agent_1", "agent_2"}

        ready_collector = _SignalCollector()
        bridge.all_actions_ready.connect(ready_collector)

        # agent_2 responds first, then agent_0, then agent_1
        with patch.object(handles["agent_2"], "try_read_response", return_value={"type": "action_selected", "action": 7}):
            with patch.object(handles["agent_0"], "try_read_response", return_value=None):
                with patch.object(handles["agent_1"], "try_read_response", return_value=None):
                    bridge._poll_responses()

        assert len(ready_collector.calls) == 0  # Only 1 of 3

        with patch.object(handles["agent_0"], "try_read_response", return_value={"type": "action_selected", "action": 1}):
            with patch.object(handles["agent_1"], "try_read_response", return_value=None):
                with patch.object(handles["agent_2"], "try_read_response", return_value=None):
                    bridge._poll_responses()

        assert len(ready_collector.calls) == 0  # 2 of 3

        with patch.object(handles["agent_1"], "try_read_response", return_value={"type": "action_selected", "action": 4}):
            with patch.object(handles["agent_0"], "try_read_response", return_value=None):
                with patch.object(handles["agent_2"], "try_read_response", return_value=None):
                    bridge._poll_responses()

        # Now all 3 collected, order is [agent_0, agent_1, agent_2]
        assert len(ready_collector.calls) == 1
        assert ready_collector.calls[0] == ([1, 4, 7],)
