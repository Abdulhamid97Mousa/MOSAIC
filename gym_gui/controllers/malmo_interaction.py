from __future__ import annotations

import logging
import socket
import struct
from typing import Any, Optional

from qtpy.QtCore import Qt

from gym_gui.controllers.interaction import InteractionController
from gym_gui.logging_config.helpers import log_constant
from gym_gui.logging_config.log_constants import (
    LOG_MALMO_ACTION_SPACE_DUMP,
    LOG_MALMO_NATIVE_INPUT_CONNECTED,
    LOG_MALMO_NATIVE_INPUT_ERROR,
    LOG_MALMO_NATIVE_KEY_SENT,
    LOG_MALMO_NATIVE_KEY_UNMAPPED,
    LOG_MALMO_NATIVE_MOUSE_SENT,
    LOG_MALMO_PASSIVE_ACTION_FALLBACK,
    LOG_MALMO_PASSIVE_ACTION_RESOLVED,
)

_LOGGER = logging.getLogger(__name__)

def _get_qt_key(name: str) -> int:
    """Get Qt key constant by name, handling Qt5/Qt6 differences."""
    key_enum = getattr(Qt, "Key", None)
    if key_enum is not None and hasattr(key_enum, name):
        return int(getattr(key_enum, name))
    legacy = getattr(Qt, name, None)
    if legacy is not None:
        return int(legacy)
    # Return 0 or specific marker if not found, to avoid crashing.
    # But usually we want to know if it's missing.
    # For this mapping table, we can skip missing keys.
    return 0

def _build_qt_to_lwjgl() -> dict[int, int]:
    mapping = {
        "Key_Escape": 1,
        "Key_1": 2, "Key_2": 3, "Key_3": 4, "Key_4": 5, "Key_5": 6,
        "Key_6": 7, "Key_7": 8, "Key_8": 9, "Key_9": 10, "Key_0": 11,
        "Key_Minus": 12, "Key_Equal": 13, "Key_Backspace": 14,
        "Key_Tab": 15,
        "Key_Q": 16, "Key_W": 17, "Key_E": 18, "Key_R": 19, "Key_T": 20,
        "Key_Y": 21, "Key_U": 22, "Key_I": 23, "Key_O": 24, "Key_P": 25,
        "Key_BracketLeft": 26, "Key_BracketRight": 27, "Key_Return": 28,
        "Key_Control": 29,
        "Key_A": 30, "Key_S": 31, "Key_D": 32, "Key_F": 33, "Key_G": 34,
        "Key_H": 35, "Key_J": 36, "Key_K": 37, "Key_L": 38, "Key_Semicolon": 39,
        "Key_Apostrophe": 40, "Key_QuoteLeft": 41,
        "Key_Shift": 42,
        "Key_Backslash": 43,
        "Key_Z": 44, "Key_X": 45, "Key_C": 46, "Key_V": 47, "Key_B": 48,
        "Key_N": 49, "Key_M": 50, "Key_Comma": 51, "Key_Period": 52, "Key_Slash": 53,
        "Key_Alt": 56,
        "Key_Space": 57,
        "Key_F1": 59, "Key_F2": 60, "Key_F3": 61, "Key_F4": 62, "Key_F5": 63,
        "Key_F6": 64, "Key_F7": 65, "Key_F8": 66, "Key_F9": 67, "Key_F10": 68,
        "Key_NumLock": 69,
        "Key_ScrollLock": 70,
        "Key_F11": 87, "Key_F12": 88,
        "Key_Up": 200, "Key_Left": 203, "Key_Right": 205, "Key_Down": 208,
        "Key_Insert": 210, "Key_Delete": 211, "Key_Home": 199, "Key_End": 207,
        "Key_PageUp": 201, "Key_PageDown": 209,
    }

    # Keypad keys - handle potential missing attributes safely
    kp_mapping = {
        "Key_KP_7": 71, "Key_KP_8": 72, "Key_KP_9": 73, "Key_KP_Subtract": 74,
        "Key_KP_4": 75, "Key_KP_5": 76, "Key_KP_6": 77, "Key_KP_Add": 78,
        "Key_KP_1": 79, "Key_KP_2": 80, "Key_KP_3": 81, "Key_KP_0": 82,
        "Key_KP_Decimal": 83,
    }

    result = {}
    for name, code in mapping.items():
        qt_key = _get_qt_key(name)
        if qt_key != 0:
            result[qt_key] = code

    for name, code in kp_mapping.items():
        qt_key = _get_qt_key(name)
        if qt_key != 0:
            result[qt_key] = code

    return result

QT_TO_LWJGL = _build_qt_to_lwjgl()


class MalmoInteractionController(InteractionController):
    """Idle controller for MalmoEnv.

    For MOSAIC Malmo, we want 'native' interaction:
    - No idle ticking (let Minecraft run at its own speed).
    - No passive actions (we don't want to force NO-OPs or repeated keys).
    - The actual input handling is done via the direct TCP service to the native mod,
      bypassing the MalmoEnv step loop for minimal latency.
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

    def handle_native_key(self, key_code: int, pressed: bool) -> bool:
        """Send raw key event to Minecraft via side-channel socket."""
        lwjgl_code = QT_TO_LWJGL.get(key_code)
        if lwjgl_code is None:
            log_constant(_LOGGER, LOG_MALMO_NATIVE_KEY_UNMAPPED, extra={
                "qt_key_code": key_code,
            })
            return False

        if not self._connected or self._socket is None:
            self._try_connect()

        sock = self._socket
        if sock is None:
            return False

        try:
            data = struct.pack('>IIB', 0, lwjgl_code, 1 if pressed else 0)
            sock.sendall(data)
            log_constant(_LOGGER, LOG_MALMO_NATIVE_KEY_SENT, extra={
                "qt_key_code": key_code,
                "lwjgl_code": lwjgl_code,
                "pressed": pressed,
            })
            return True
        except (socket.error, BrokenPipeError) as exc:
            log_constant(_LOGGER, LOG_MALMO_NATIVE_INPUT_ERROR, extra={
                "error": str(exc),
                "context": "key_send",
            })
            self._connected = False
            if self._socket:
                self._socket.close()
            self._socket = None
            return False

    def handle_native_key_evdev(self, evdev_keycode: int, pressed: bool) -> bool:
        """Send raw key event from evdev to Minecraft via side-channel socket.

        Linux evdev keycodes use the same scan code numbering as LWJGL,
        so no Qt-to-LWJGL translation is needed. This is the primary path
        when input comes from HumanKeyboardRuntime worker subprocesses.
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
