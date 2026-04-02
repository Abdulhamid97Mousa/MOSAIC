"""Bridge between the Human Control tab and HumanKeyboardRuntime subprocesses.

Launches one ``HumanKeyboardRuntime`` subprocess per agent. Each subprocess
owns a single keyboard (and optionally a mouse) via Linux evdev, reads
input in its own OS process, and returns resolved actions via JSON pipe.

The GUI main thread NEVER reads keyboards or mice. It only polls subprocess
pipes (microseconds) and paints frames.

Startup is fully non-blocking. ``start()`` spawns processes, sends init
commands, and starts the 60 Hz poll timer. The poll timer picks up init
responses and action responses alike. No synchronous waits.

Stepping model (round-based):
    1. After all workers report ``agent_initialized``, send ``select_action``
       to every worker.  Each worker blocks reading its keyboard.
    2. Poll workers at ~60 Hz for responses.  As each human presses a
       key, that worker's action is recorded.
    3. When ALL agents have responded, emit ``all_actions_ready`` with
       the complete action list.
    4. The caller steps the environment, then calls
       ``request_next_round()`` to start the cycle again.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from qtpy import QtCore
from qtpy.QtCore import QTimer

from gym_gui.config.paths import (
    VAR_HUMAN_CONTROL_LOGS_DIR,
    ensure_var_directories,
)
from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_HUMAN_CONTROL_BRIDGE_START,
    LOG_HUMAN_CONTROL_BRIDGE_STOP,
    LOG_HUMAN_CONTROL_WORKER_ACTION,
    LOG_HUMAN_CONTROL_WORKER_ERROR,
    LOG_HUMAN_CONTROL_WORKER_LAUNCH,
    LOG_HUMAN_CONTROL_WORKER_READY,
    LOG_HUMAN_CONTROL_WORKER_STEP,
    LOG_HUMAN_CONTROL_WORKER_STOPPED,
)

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Init handshake states for each worker
# ---------------------------------------------------------------------------
_STATE_WAIT_INIT = "wait_init"           # Waiting for {"type": "init"}
_STATE_WAIT_AGENT_INIT = "wait_agent"    # Waiting for {"type": "agent_initialized"}
_STATE_READY = "ready"                   # Worker is ready, accepting select_action


class _KeyboardWorkerHandle:
    """Tracks a single HumanKeyboardRuntime subprocess."""

    def __init__(
        self,
        agent_id: str,
        device_path: str,
        process: subprocess.Popen,
    ) -> None:
        self.agent_id = agent_id
        self.device_path = device_path
        self.process = process
        self.state = _STATE_WAIT_INIT
        self.init_cmd: Optional[Dict[str, Any]] = None  # Queued init_agent command
        self._log_file = None

    @property
    def is_running(self) -> bool:
        return self.process.poll() is None

    @property
    def is_ready(self) -> bool:
        return self.state == _STATE_READY

    def send_command(self, cmd: Dict[str, Any]) -> bool:
        if self.process.stdin is None or not self.is_running:
            return False
        try:
            line = json.dumps(cmd) + "\n"
            self.process.stdin.write(line)
            self.process.stdin.flush()
            return True
        except (BrokenPipeError, OSError) as exc:
            _LOGGER.warning("Send failed for %s: %s", self.agent_id, exc)
            return False

    def try_read_response(self, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        if self.process.stdout is None or not self.is_running:
            return None
        import select as _select
        try:
            readable, _, _ = _select.select([self.process.stdout], [], [], timeout)
            if not readable:
                return None
            line = self.process.stdout.readline()
            if not line:
                return None
            return json.loads(line.strip())
        except (json.JSONDecodeError, OSError, ValueError):
            return None

    def stop(self, timeout: float = 2.0) -> None:
        self.send_command({"cmd": "stop"})
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=1.0)
        if self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None


class KeyboardWorkerBridge(QtCore.QObject, LogConstantMixin):
    """Manages HumanKeyboardRuntime subprocesses for human play.

    One subprocess per agent. Each owns its keyboard exclusively.

    Signals:
        all_actions_ready(list): All agents responded. Ordered action list.
        action_received(str, int): Per-agent action notification.
        mouse_delta_received(str, int, int): agent_id, dx, dy from mouse.
        workers_ready(): All workers completed init handshake.
    """

    all_actions_ready = QtCore.Signal(list)
    action_received = QtCore.Signal(str, int)
    mouse_delta_received = QtCore.Signal(str, int, int)
    raw_key_received = QtCore.Signal(str, int, bool)  # agent_id, keycode, pressed
    workers_ready = QtCore.Signal()

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._logger = _LOGGER
        self._handles: Dict[str, _KeyboardWorkerHandle] = {}
        self._agent_order: List[str] = []
        self._pending_actions: Dict[str, int] = {}
        self._awaiting_response: Set[str] = set()
        self._all_ready_emitted = False
        self._noop_action: int = 0
        self._tick_timeout: float = 0.0  # 0 = blocking (single-agent)

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(16)  # ~60 Hz
        self._poll_timer.timeout.connect(self._poll_responses)

    @property
    def is_active(self) -> bool:
        return bool(self._handles) and self._poll_timer.isActive()

    def start(
        self,
        assignments: Dict[str, str],
        env_name: str = "multigrid",
        action_list: Optional[List[str]] = None,
        mouse_assignments: Optional[Dict[str, str]] = None,
        key_action_map: Optional[List[Dict[str, Any]]] = None,
        noop_action: int = 0,
        tick_timeout: float = 0.0,
    ) -> bool:
        """Launch keyboard worker subprocesses. Non-blocking.

        Spawns one subprocess per agent, sends init commands, starts the
        60 Hz poll timer. The poll timer handles init responses and
        transitions workers to the ready state. No synchronous waits.

        Args:
            assignments: {device_path: agent_id} mapping.
            env_name: Environment family for key resolver fallback.
            action_list: For Malmo, the mission's action space list.
            noop_action: Action index to use as idle/NOOP (varies by env).
            tick_timeout: Seconds to wait for input per tick. 0 = blocking
                (single-agent). >0 = return NOOP after timeout (multi-agent).
            mouse_assignments: {agent_id: mouse_device_path} mapping.
            key_action_map: Dynamic key-to-action mapping from GUI shortcuts.

        Returns:
            True if all subprocesses were spawned (init may still be in progress).
        """
        self.stop()

        self._agent_order = sorted(assignments.values())
        self._all_ready_emitted = False
        self._noop_action = noop_action
        self._tick_timeout = tick_timeout

        ensure_var_directories()
        VAR_HUMAN_CONTROL_LOGS_DIR.mkdir(parents=True, exist_ok=True)

        python_exe = sys.executable
        env = os.environ.copy()
        env.setdefault("QT_DEBUG_PLUGINS", "0")

        for device_path, agent_id in assignments.items():
            run_id = f"keyboard_{agent_id}_{uuid.uuid4().hex[:8]}"

            cmd = [
                python_exe,
                "-m", "human_worker.cli",
                "--mode", "keyboard",
                "--device-path", device_path,
                "--env-name", env_name,
                "--run-id", run_id,
                "--player-name", agent_id,
            ]
            if mouse_assignments and agent_id in mouse_assignments:
                cmd.extend(["--mouse-path", mouse_assignments[agent_id]])

            self.log_constant(
                LOG_HUMAN_CONTROL_WORKER_LAUNCH,
                message=f"Launching keyboard worker for {agent_id}: {device_path}",
                extra={"agent_id": agent_id, "device_path": device_path},
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = VAR_HUMAN_CONTROL_LOGS_DIR / f"keyboard_{agent_id}_{timestamp}.log"
            log_file = log_path.open("a", encoding="utf-8")

            try:
                process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    text=True,
                    bufsize=1,
                    env=env,
                )
            except Exception as exc:
                log_file.close()
                self.log_constant(
                    LOG_HUMAN_CONTROL_WORKER_ERROR,
                    message=f"Failed to launch worker for {agent_id}: {exc}",
                    extra={"agent_id": agent_id, "error": str(exc)},
                )
                self.stop()
                return False

            handle = _KeyboardWorkerHandle(agent_id, device_path, process)
            handle._log_file = log_file
            # Queue the init_agent command to send after we receive {"type": "init"}
            init_cmd: Dict[str, Any] = {
                "cmd": "init_agent",
                "game_name": env_name,
                "player_id": agent_id,
            }
            if action_list is not None:
                init_cmd["action_list"] = action_list
            if key_action_map is not None:
                init_cmd["key_action_map"] = key_action_map
            handle.init_cmd = init_cmd
            self._handles[agent_id] = handle

        # Start polling immediately. The poll loop handles the init handshake.
        self._poll_timer.start()

        self.log_constant(
            LOG_HUMAN_CONTROL_BRIDGE_START,
            message=f"Spawned {len(self._handles)} keyboard worker(s), init in progress",
            extra={"num_workers": len(self._handles), "agents": self._agent_order},
        )
        return True

    def stop(self) -> None:
        """Stop all keyboard worker subprocesses."""
        self._poll_timer.stop()
        self._pending_actions.clear()
        self._awaiting_response.clear()
        self._all_ready_emitted = False

        for agent_id, handle in self._handles.items():
            try:
                handle.stop(timeout=2.0)
                self.log_constant(
                    LOG_HUMAN_CONTROL_WORKER_STOPPED,
                    message=f"Stopped worker for {agent_id}",
                    extra={"agent_id": agent_id},
                )
            except Exception as exc:
                _LOGGER.warning("Error stopping worker for %s: %s", agent_id, exc)

        if self._handles:
            self.log_constant(
                LOG_HUMAN_CONTROL_BRIDGE_STOP,
                message="Keyboard bridge stopped",
            )

        self._handles.clear()
        self._agent_order.clear()

    def request_next_round(self) -> None:
        """Start a new action-collection round after a step completes."""
        self._pending_actions.clear()
        self._awaiting_response.clear()
        self._send_select_action_to_all()

    def _send_select_action_to_all(self) -> None:
        for agent_id, handle in self._handles.items():
            if not handle.is_ready:
                continue
            if agent_id in self._awaiting_response:
                continue
            if not handle.is_running:
                _LOGGER.warning("Worker for %s died", agent_id)
                continue
            cmd: Dict[str, Any] = {
                "cmd": "select_action",
                "player_id": agent_id,
            }
            if self._tick_timeout > 0:
                cmd["noop_action"] = self._noop_action
                cmd["tick_timeout"] = self._tick_timeout
            handle.send_command(cmd)
            self._awaiting_response.add(agent_id)

    def _poll_responses(self) -> None:
        """Poll all workers at ~60 Hz. Handles init handshake and actions."""
        for agent_id, handle in list(self._handles.items()):
            if not handle.is_running:
                continue

            # Skip reading from ready workers that already acted this round
            if handle.is_ready and agent_id in self._pending_actions:
                continue

            resp = handle.try_read_response(timeout=0.0)
            if resp is None:
                continue

            msg_type = resp.get("type", "")

            # --- Init handshake: wait_init -> send init_agent -> wait_agent -> ready ---
            if handle.state == _STATE_WAIT_INIT:
                if msg_type == "init":
                    if handle.init_cmd is not None:
                        handle.send_command(handle.init_cmd)
                        handle.init_cmd = None
                    handle.state = _STATE_WAIT_AGENT_INIT
                continue

            if handle.state == _STATE_WAIT_AGENT_INIT:
                if msg_type == "agent_initialized":
                    handle.state = _STATE_READY
                    self.log_constant(
                        LOG_HUMAN_CONTROL_WORKER_READY,
                        message=f"Worker ready: {agent_id} on {handle.device_path}",
                        extra={"agent_id": agent_id, "device_path": handle.device_path},
                    )
                    if not self._all_ready_emitted and all(
                        h.is_ready for h in self._handles.values()
                    ):
                        self._all_ready_emitted = True
                        _LOGGER.info(
                            "All %d keyboard workers ready", len(self._handles)
                        )
                        self.workers_ready.emit()
                        self._send_select_action_to_all()
                elif msg_type == "error":
                    self.log_constant(
                        LOG_HUMAN_CONTROL_WORKER_ERROR,
                        message=f"Worker {agent_id} init failed: {resp.get('message', '')}",
                        extra={"agent_id": agent_id, "response": str(resp)},
                    )
                continue

            # --- Ready state: handle action responses ---
            if msg_type == "action_selected":
                action = resp.get("action")
                if action is not None:
                    self._pending_actions[agent_id] = int(action)
                    self._awaiting_response.discard(agent_id)
                    self.action_received.emit(agent_id, int(action))

                    mdx = resp.get("mouse_dx", 0)
                    mdy = resp.get("mouse_dy", 0)
                    if mdx != 0 or mdy != 0:
                        self.mouse_delta_received.emit(agent_id, int(mdx), int(mdy))

                    # Forward raw key events for native input (Malmo TCP etc.)
                    for rk in resp.get("raw_keys", []):
                        self.raw_key_received.emit(
                            agent_id, int(rk["keycode"]), bool(rk["pressed"]),
                        )

                    self.log_constant(
                        LOG_HUMAN_CONTROL_WORKER_ACTION,
                        message=f"{agent_id} -> action {action} ({len(self._pending_actions)}/{len(self._handles)})",
                        extra={"agent_id": agent_id, "action": int(action)},
                    )

        # Check if all ready workers have submitted actions
        ready_handles = {aid for aid, h in self._handles.items() if h.is_ready}
        if ready_handles and len(self._pending_actions) >= len(ready_handles):
            actions = [
                self._pending_actions.get(aid, 0) for aid in self._agent_order
            ]
            self.log_constant(
                LOG_HUMAN_CONTROL_WORKER_STEP,
                message=f"All actions collected: {actions}",
                extra={"actions": actions},
            )
            self.all_actions_ready.emit(actions)
