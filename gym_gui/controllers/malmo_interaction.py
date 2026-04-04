from __future__ import annotations

import logging
import socket
import struct
from typing import Any, Optional

from gym_gui.controllers.interaction import InteractionController
from gym_gui.logging_config.helpers import log_constant
from gym_gui.logging_config.log_constants import (
    LOG_MALMO_ACTION_SPACE_DUMP,
    LOG_MALMO_NATIVE_INPUT_CONNECTED,
    LOG_MALMO_NATIVE_INPUT_ERROR,
    LOG_MALMO_NATIVE_KEY_SENT,
    LOG_MALMO_NATIVE_MOUSE_SENT,
    LOG_MALMO_PASSIVE_ACTION_FALLBACK,
    LOG_MALMO_PASSIVE_ACTION_RESOLVED,
)

_LOGGER = logging.getLogger(__name__)


class MalmoInteractionController(InteractionController):
    """Interaction controller for MalmoEnv (Minecraft).

    Sends raw keyboard and mouse events to Minecraft via a TCP side-channel
    socket. Input comes from HumanKeyboardRuntime worker subprocesses via
    evdev. Linux evdev keycodes == LWJGL scan codes, so no translation needed.
    """

    # NativeInputHandler port = MalmoEnv port + 1000 (avoids conflict with
    # multi-agent MalmoEnv ports which use consecutive ports).
    NATIVE_PORT_OFFSET = 1000

    def __init__(self, owner, target_hz: int = 60, malmo_port: int = 9000):
        self._owner = owner
        self._interval_ms = max(1, int(1000 / float(target_hz)))  # ~16ms for 60Hz
        self._native_port = malmo_port + self.NATIVE_PORT_OFFSET
        self._socket = None
        self._connected = False
        self._force_connect()

    def _force_connect(self):
        """Close any existing socket and establish a fresh connection."""
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
        self._connected = False
        self._try_connect()

    def _try_connect(self):
        """Attempt to connect to the native input handler."""
        if self._connected:
            return
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(2.0)  # 2s timeout for connection
            self._socket.connect(('localhost', self._native_port))
            self._socket.settimeout(None)  # Blocking mode for sends
            self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._connected = True
            log_constant(_LOGGER, LOG_MALMO_NATIVE_INPUT_CONNECTED, extra={
                "port": self._native_port,
            })
        except (socket.error, ConnectionRefusedError) as exc:
            self._socket = None
            self._connected = False
            log_constant(_LOGGER, LOG_MALMO_NATIVE_INPUT_ERROR, extra={
                "port": self._native_port,
                "error": str(exc),
            })

    def handle_native_key_evdev(self, evdev_keycode: int, pressed: bool) -> bool:
        """Send raw key event from evdev to Minecraft via TCP side-channel.

        Linux evdev keycodes == LWJGL scan codes (same numbering), so the
        keycode from the worker subprocess goes straight to Minecraft.
        """
        if not self._connected or self._socket is None:
            self._try_connect()

        sock = self._socket
        if sock is None:
            return False

        try:
            # evdev keycode == LWJGL scan code (same numbering)
            data = struct.pack('>IIB', 0, evdev_keycode, 1 if pressed else 0)
            sock.sendall(data)
            log_constant(_LOGGER, LOG_MALMO_NATIVE_KEY_SENT, extra={
                "evdev_keycode": evdev_keycode,
                "lwjgl_code": evdev_keycode,
                "pressed": pressed,
            })
            return True
        except (socket.error, BrokenPipeError) as exc:
            log_constant(_LOGGER, LOG_MALMO_NATIVE_INPUT_ERROR, extra={
                "error": str(exc),
                "context": "key_send_evdev",
            })
            self._connected = False
            if self._socket:
                self._socket.close()
            self._socket = None
            return False

    def handle_native_mouse(self, dx: int, dy: int) -> bool:
        """Send raw mouse delta to Minecraft via side-channel socket."""
        if not self._connected or self._socket is None:
            self._try_connect()

        sock = self._socket
        if sock is None:
            return False

        try:
            data = struct.pack('>iii', 1, dx, dy)
            sock.sendall(data)
            log_constant(_LOGGER, LOG_MALMO_NATIVE_MOUSE_SENT, extra={
                "dx": dx,
                "dy": dy,
            })
            return True
        except (socket.error, BrokenPipeError) as exc:
            log_constant(_LOGGER, LOG_MALMO_NATIVE_INPUT_ERROR, extra={
                "error": str(exc),
                "context": "mouse_send",
            })
            self._connected = False
            if self._socket:
                self._socket.close()
            self._socket = None
            return False

    def idle_interval_ms(self) -> Optional[int]:
        return self._interval_ms

    def should_idle_tick(self) -> bool:
        """Check if we should advance the game this tick (fetch observation)."""
        o = self._owner
        if o._adapter is None or o._game_id is None:
            return False
        if not getattr(o, "_game_started", False):
            return False
        if o._game_paused:
            return False
        # Only tick in HUMAN_ONLY mode where we want continuous updates
        if getattr(o._control_mode, "name", "") != "HUMAN_ONLY":
            return False
        if o._last_step is not None and (o._last_step.terminated or o._last_step.truncated):
            return False
        return True

    def maybe_passive_action(self) -> Optional[Any]:
        """Return NOOP action for MalmoEnv to fetch observation.

        We look for a true no-op command in the action space.  Priority:
        1. "noop 1"  (injected by adapter)
        2. "move 0"  (stop moving — neutral)
        3. "turn 0"  (stop turning — neutral)
        If none found, return 0 but log a warning since action 0 may cause
        unintended movement.
        """
        adapter = getattr(self._owner, "_adapter", None)
        if adapter and adapter.action_space:
            actions = getattr(adapter.action_space, "actions", [])
            log_constant(
                _LOGGER, LOG_MALMO_ACTION_SPACE_DUMP,
                message=f"actions={list(actions)} count={len(actions)}",
            )
            for noop_cmd in ("noop 1", "move 0", "turn 0"):
                try:
                    idx = actions.index(noop_cmd)
                    log_constant(
                        _LOGGER, LOG_MALMO_PASSIVE_ACTION_RESOLVED,
                        message=f"command='{noop_cmd}' action_index={idx}",
                    )
                    return idx
                except ValueError:
                    continue
            fallback_cmd = actions[0] if actions else "EMPTY"
            log_constant(
                _LOGGER, LOG_MALMO_PASSIVE_ACTION_FALLBACK,
                message=f"actions={list(actions)} fallback_index=0 fallback_command='{fallback_cmd}'",
            )
        return 0

    def step_dt(self) -> float:
        return 0.0

    def __del__(self):
        if self._socket is not None:
            try:
                self._socket.close()
            except:
                pass


class MalmoMultiAgentInteractionController(InteractionController):
    """Interaction controller for multi-agent MalmoEnv missions (turn-based).

    Unlike the single-agent ``MalmoInteractionController`` which uses a native
    TCP side-channel, this controller routes actions through the adapter's
    ``step_agent(role, action)`` method.  Each agent acts in turn (role 0
    first, then role 1, etc.) and the controller tracks whose turn it is.

    Keyboard input from evdev is already routed by the existing
    ``HumanInputController`` → ``_on_evdev_agent_action`` pipeline.  This
    controller handles idle ticking to keep observations flowing.
    """

    def __init__(self, owner, agent_count: int = 2, target_hz: int = 20):
        self._owner = owner
        self._agent_count = agent_count
        self._interval_ms = max(1, int(1000 / float(target_hz)))
        self._active_role = 0  # Whose turn it is

    @property
    def active_role(self) -> int:
        """The agent role (0..N-1) whose turn it is to act."""
        return self._active_role

    def advance_turn(self) -> None:
        """Move to the next agent's turn (wraps around)."""
        self._active_role = (self._active_role + 1) % self._agent_count

    def idle_interval_ms(self) -> Optional[int]:
        return self._interval_ms

    def should_idle_tick(self) -> bool:
        o = self._owner
        if o._adapter is None or o._game_id is None:
            return False
        if not getattr(o, "_game_started", False):
            return False
        if o._game_paused:
            return False
        if getattr(o._control_mode, "name", "") != "HUMAN_ONLY":
            return False
        if o._last_step is not None and (o._last_step.terminated or o._last_step.truncated):
            return False
        return True

    def maybe_passive_action(self) -> Optional[Any]:
        """Return NOOP for the active agent to keep observations flowing."""
        adapter = getattr(self._owner, "_adapter", None)
        if adapter is None:
            return 0
        # Use the multi-agent adapter's noop resolver for the active role
        if hasattr(adapter, "_resolve_noop"):
            return adapter._resolve_noop(self._active_role)
        # Fallback to single-agent noop resolution
        if adapter.action_space:
            actions = getattr(adapter.action_space, "actions", [])
            for noop_cmd in ("noop 1", "move 0", "turn 0"):
                try:
                    return actions.index(noop_cmd)
                except ValueError:
                    continue
        return 0

    def step_dt(self) -> float:
        return 0.0
