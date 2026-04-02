"""Pure-Python evdev keyboard reader and keycode-to-action resolver.

This module provides keyboard input for the human_worker subprocess
WITHOUT any Qt dependency.  It extracts the core evdev reading logic
from ``gym_gui/controllers/evdev_keyboard_monitor.py`` and replaces
Qt.Key-based key resolution with direct Linux keycode mapping.

The module is used by ``HumanKeyboardRuntime`` to read a single
physical USB keyboard in a subprocess and resolve key presses to
game action indices.
"""

from __future__ import annotations

import glob
import logging
import os
import select
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ============================================================================
# Linux input-event-codes.h constants
# ============================================================================

EV_KEY = 0x01
EV_REL = 0x02
EV_VALUE_RELEASE = 0
EV_VALUE_PRESS = 1
EV_VALUE_REPEAT = 2

# Mouse relative-axis codes
REL_X = 0x00
REL_Y = 0x01

# Linux keycodes (from /usr/include/linux/input-event-codes.h)
KEY_ESC = 1
KEY_1 = 2
KEY_2 = 3
KEY_3 = 4
KEY_Q = 16
KEY_W = 17
KEY_E = 18
KEY_R = 19
KEY_A = 30
KEY_S = 31
KEY_D = 32
KEY_F = 33
KEY_G = 34
KEY_H = 35
KEY_Z = 44
KEY_X = 45
KEY_C = 46
KEY_SPACE = 57
KEY_F1 = 59
KEY_UP = 103
KEY_LEFT = 105
KEY_RIGHT = 106
KEY_DOWN = 108
KEY_RETURN = 28

# Directional key sets (Linux keycodes)
KEYS_UP = {KEY_UP, KEY_W}
KEYS_DOWN = {KEY_DOWN, KEY_S}
KEYS_LEFT = {KEY_LEFT, KEY_A}
KEYS_RIGHT = {KEY_RIGHT, KEY_D}

# Qt key name -> Linux keycode mapping (for converting GUI ShortcutMappings)
QT_KEY_TO_LINUX: Dict[str, int] = {
    "Key_Escape": KEY_ESC,
    "Key_1": KEY_1,
    "Key_2": KEY_2,
    "Key_3": KEY_3,
    "Key_Q": KEY_Q,
    "Key_W": KEY_W,
    "Key_E": KEY_E,
    "Key_R": KEY_R,
    "Key_A": KEY_A,
    "Key_S": KEY_S,
    "Key_D": KEY_D,
    "Key_F": KEY_F,
    "Key_G": KEY_G,
    "Key_H": KEY_H,
    "Key_Z": KEY_Z,
    "Key_X": KEY_X,
    "Key_C": KEY_C,
    "Key_Space": KEY_SPACE,
    "Key_F1": KEY_F1,
    "Key_Up": KEY_UP,
    "Key_Down": KEY_DOWN,
    "Key_Left": KEY_LEFT,
    "Key_Right": KEY_RIGHT,
    "Key_Return": KEY_RETURN,
}


# ============================================================================
# Keyboard device discovery (pure Python, no Qt)
# ============================================================================

@dataclass
class KeyboardDeviceInfo:
    """Information about a keyboard input device."""

    device_path: str
    event_path: str
    name: str
    usb_port: Optional[str] = None
    vendor_id: Optional[str] = None
    product_id: Optional[str] = None


def discover_keyboards() -> List[KeyboardDeviceInfo]:
    """Discover all keyboard input devices via /dev/input.

    Scans ``/dev/input/by-path/*kbd`` and ``/dev/input/by-id/*kbd``,
    deduplicates by real path, and reads device names from
    ``/proc/bus/input/devices``.

    Returns:
        List of discovered keyboard devices.
    """
    keyboards: List[KeyboardDeviceInfo] = []
    patterns = ["/dev/input/by-path/*kbd", "/dev/input/by-id/*kbd"]
    seen_real_paths: set[str] = set()

    for pattern in patterns:
        for device_path in sorted(glob.glob(pattern)):
            try:
                real_path = os.path.realpath(device_path)
                if real_path in seen_real_paths:
                    continue
                seen_real_paths.add(real_path)

                name = _get_device_name(real_path)
                usb_port = _extract_usb_port(device_path)
                vendor_id, product_id = _get_device_ids(real_path)

                keyboards.append(
                    KeyboardDeviceInfo(
                        device_path=device_path,
                        event_path=real_path,
                        name=name,
                        usb_port=usb_port,
                        vendor_id=vendor_id,
                        product_id=product_id,
                    )
                )
            except Exception as exc:
                logger.warning("Failed to process device %s: %s", device_path, exc)

    return keyboards


def _get_device_name(event_path: str) -> str:
    """Get human-readable name of an input device."""
    try:
        event_num = Path(event_path).name.replace("event", "")
        with open("/proc/bus/input/devices", "r") as fh:
            content = fh.read()
        for block in content.split("\n\n"):
            if f"event{event_num}" in block:
                for line in block.split("\n"):
                    if line.startswith("N: Name="):
                        return line.split("Name=")[1].strip('"')
    except Exception as exc:
        logger.debug("Could not get device name: %s", exc)
    return "Unknown Device"


def _extract_usb_port(device_path: str) -> Optional[str]:
    """Extract USB port number from a by-path device path."""
    try:
        if "usb-" in device_path:
            usb_part = device_path.split("usb-")[1].split("-")[0]
            parts = usb_part.split(":")
            if len(parts) >= 2:
                return parts[1]
    except Exception as exc:
        logger.debug("Could not extract USB port: %s", exc)
    return None


def _get_device_ids(event_path: str) -> tuple[Optional[str], Optional[str]]:
    """Get vendor and product IDs of an input device."""
    try:
        event_num = Path(event_path).name.replace("event", "")
        with open("/proc/bus/input/devices", "r") as fh:
            content = fh.read()
        for block in content.split("\n\n"):
            if f"event{event_num}" in block:
                for line in block.split("\n"):
                    if line.startswith("I: Bus="):
                        parts = line.split()
                        vendor_id = None
                        product_id = None
                        for part in parts:
                            if part.startswith("Vendor="):
                                vendor_id = part.split("=")[1]
                            elif part.startswith("Product="):
                                product_id = part.split("=")[1]
                        return vendor_id, product_id
    except Exception as exc:
        logger.debug("Could not get device IDs: %s", exc)
    return None, None


# ============================================================================
# Mouse device discovery (pure Python, no Qt)
# ============================================================================

@dataclass
class MouseDeviceInfo:
    """Information about a mouse input device."""

    device_path: str
    event_path: str
    name: str
    usb_port: Optional[str] = None
    vendor_id: Optional[str] = None
    product_id: Optional[str] = None


def discover_mice() -> List[MouseDeviceInfo]:
    """Discover all mouse input devices via /dev/input.

    Scans ``/dev/input/by-path/*event-mouse`` and
    ``/dev/input/by-id/*event-mouse``, deduplicates by real path,
    and reads device names from ``/proc/bus/input/devices``.

    Returns:
        List of discovered mouse devices.
    """
    mice: List[MouseDeviceInfo] = []
    patterns = ["/dev/input/by-path/*event-mouse", "/dev/input/by-id/*event-mouse"]
    seen_real_paths: set[str] = set()

    for pattern in patterns:
        for device_path in sorted(glob.glob(pattern)):
            try:
                real_path = os.path.realpath(device_path)
                if real_path in seen_real_paths:
                    continue
                seen_real_paths.add(real_path)

                name = _get_device_name(real_path)
                usb_port = _extract_usb_port(device_path)
                vendor_id, product_id = _get_device_ids(real_path)

                mice.append(
                    MouseDeviceInfo(
                        device_path=device_path,
                        event_path=real_path,
                        name=name,
                        usb_port=usb_port,
                        vendor_id=vendor_id,
                        product_id=product_id,
                    )
                )
            except Exception as exc:
                logger.warning("Failed to process mouse %s: %s", device_path, exc)

    return mice


def pair_devices_by_usb_port(
    keyboards: List[KeyboardDeviceInfo],
    mice: List[MouseDeviceInfo],
) -> List[Dict[str, Optional[str]]]:
    """Pair keyboards and mice for multi-agent assignment.

    Pairing strategy (in priority order):
    1. **Exact USB port match**: keyboard and mouse on the same port.
    2. **Round-robin**: remaining unpaired mice assigned to keyboards
       in sorted order (by USB port). This handles the common case where
       keyboard and mouse are on different ports of the same hub.

    Keyboards are the primary devices (one per agent). Unmatched keyboards
    still get an entry (with ``mouse_path=None``).

    Returns:
        List of dicts, one per agent, sorted by USB port::

            [
                {"keyboard_path": "/dev/input/...", "mouse_path": "/dev/input/...",
                 "keyboard_name": "...", "mouse_name": "...", "usb_port": "6"},
                ...
            ]
    """
    sorted_kbds = sorted(keyboards, key=lambda k: k.usb_port or "")
    available_mice = list(sorted(mice, key=lambda m: m.usb_port or ""))

    # Phase 1: exact USB port match
    mouse_by_port: Dict[str, MouseDeviceInfo] = {}
    for m in available_mice:
        if m.usb_port:
            mouse_by_port[m.usb_port] = m

    pairs = []
    claimed_mice: set = set()
    for kbd in sorted_kbds:
        mouse = mouse_by_port.get(kbd.usb_port or "") if kbd.usb_port else None
        if mouse:
            claimed_mice.add(mouse.event_path)
        pairs.append({
            "keyboard_path": kbd.device_path,
            "keyboard_name": kbd.name,
            "mouse_path": mouse.device_path if mouse else None,
            "mouse_name": mouse.name if mouse else None,
            "usb_port": kbd.usb_port,
        })

    # Phase 2: round-robin remaining mice to keyboards that have no mouse
    remaining_mice = [m for m in available_mice if m.event_path not in claimed_mice]
    mouse_iter = iter(remaining_mice)
    for pair in pairs:
        if pair["mouse_path"] is None:
            mouse = next(mouse_iter, None)
            if mouse is not None:
                pair["mouse_path"] = mouse.device_path
                pair["mouse_name"] = mouse.name

    return pairs


# ============================================================================
# X11 multi-cursor (MPX) management
# ============================================================================

@dataclass
class MultiCursorState:
    """Tracks X11 master pointers created for multi-agent play."""

    master_names: List[str] = field(default_factory=list)
    reattached: Dict[str, int] = field(default_factory=dict)  # xinput_name -> original_master_id


def setup_multi_cursor(
    mice: List[MouseDeviceInfo],
    agent_names: List[str],
) -> Optional[MultiCursorState]:
    """Create separate X11 cursors for each agent's mouse.

    Uses ``xinput create-master`` and ``xinput reattach`` to give each
    mouse its own independent cursor on screen.

    Args:
        mice: Discovered mice (one per agent, in agent order).
        agent_names: Agent display names (same length as mice).

    Returns:
        State object for teardown, or None if setup failed.
    """
    import subprocess as sp

    if len(mice) < 2:
        return None  # Only one mouse, no MPX needed

    state = MultiCursorState()

    # Get xinput device IDs for each mouse
    try:
        result = sp.run(
            ["xinput", "list", "--id-only"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, sp.TimeoutExpired):
        logger.warning("xinput not available, cannot setup multi-cursor")
        return None

    # For mice beyond the first, create new master pointers and reattach
    for i, mouse in enumerate(mice[1:], start=1):
        agent_name = agent_names[i] if i < len(agent_names) else f"Player{i+1}"
        master_name = f"MOSAIC_{agent_name}"

        try:
            sp.run(
                ["xinput", "create-master", master_name],
                capture_output=True, text=True, timeout=5,
            )
            state.master_names.append(master_name)

            # Find the xinput ID of this mouse by matching its name
            list_result = sp.run(
                ["xinput", "list"],
                capture_output=True, text=True, timeout=5,
            )
            mouse_xinput_id = _find_xinput_id(list_result.stdout, mouse.name)
            master_pointer_id = _find_xinput_id(
                list_result.stdout, f"{master_name} pointer"
            )

            if mouse_xinput_id is not None and master_pointer_id is not None:
                sp.run(
                    ["xinput", "reattach", str(mouse_xinput_id), str(master_pointer_id)],
                    capture_output=True, text=True, timeout=5,
                )
                state.reattached[mouse.name] = 2  # Original "Virtual core pointer" id
                logger.info(
                    "Multi-cursor: %s (id=%d) -> %s pointer (id=%d)",
                    mouse.name, mouse_xinput_id, master_name, master_pointer_id,
                )
            else:
                logger.warning(
                    "Could not find xinput IDs for %s or %s pointer",
                    mouse.name, master_name,
                )
        except (sp.TimeoutExpired, OSError) as exc:
            logger.warning("Failed to create master %s: %s", master_name, exc)

    return state if state.master_names else None


def teardown_multi_cursor(state: Optional[MultiCursorState]) -> None:
    """Restore all mice to the default X11 cursor and remove extra masters.

    Args:
        state: State returned by :func:`setup_multi_cursor`, or None (no-op).
    """
    import subprocess as sp

    if state is None:
        return

    # Reattach mice back to Virtual core pointer (id=2)
    try:
        list_result = sp.run(
            ["xinput", "list"],
            capture_output=True, text=True, timeout=5,
        )
        for mouse_name, original_master in state.reattached.items():
            mouse_id = _find_xinput_id(list_result.stdout, mouse_name)
            if mouse_id is not None:
                sp.run(
                    ["xinput", "reattach", str(mouse_id), str(original_master)],
                    capture_output=True, text=True, timeout=5,
                )
                logger.info("Restored %s to master %d", mouse_name, original_master)
    except (sp.TimeoutExpired, OSError) as exc:
        logger.warning("Error restoring mice: %s", exc)

    # Remove created masters
    for master_name in state.master_names:
        try:
            list_result = sp.run(
                ["xinput", "list"],
                capture_output=True, text=True, timeout=5,
            )
            master_id = _find_xinput_id(list_result.stdout, f"{master_name} pointer")
            if master_id is not None:
                sp.run(
                    ["xinput", "remove-master", str(master_id)],
                    capture_output=True, text=True, timeout=5,
                )
                logger.info("Removed master: %s (id=%d)", master_name, master_id)
        except (sp.TimeoutExpired, OSError) as exc:
            logger.warning("Error removing master %s: %s", master_name, exc)

    state.master_names.clear()
    state.reattached.clear()


def _find_xinput_id(xinput_list_output: str, device_name: str) -> Optional[int]:
    """Extract the xinput device ID for a given device name."""
    import re
    for line in xinput_list_output.split("\n"):
        if device_name in line:
            match = re.search(r"id=(\d+)", line)
            if match:
                return int(match.group(1))
    return None


# ============================================================================
# Mouse device reader (pure Python, no Qt)
# ============================================================================

class MouseDeviceReader:
    """Reads mouse relative-motion events from a ``/dev/input/eventX`` device.

    Opens the device in non-blocking mode and drains ``EV_REL`` events,
    accumulating ``(dx, dy)`` since the last call to :meth:`drain_deltas`.
    """

    EVENT_SIZE = 24  # sizeof(struct input_event) on 64-bit

    def __init__(self, event_path: str) -> None:
        self._event_path = event_path
        self._fd: Optional[int] = None

    @property
    def is_open(self) -> bool:
        return self._fd is not None

    @property
    def fd(self) -> Optional[int]:
        return self._fd

    def open(self) -> None:
        if self._fd is not None:
            return
        self._fd = os.open(self._event_path, os.O_RDONLY | os.O_NONBLOCK)
        logger.info("Opened mouse device: %s (fd=%d)", self._event_path, self._fd)

    def close(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError as exc:
                logger.warning("Error closing mouse fd %d: %s", self._fd, exc)
            self._fd = None

    def drain_deltas(self) -> tuple:
        """Non-blocking drain of accumulated (dx, dy) since last call.

        Returns:
            Tuple ``(dx, dy)`` of accumulated relative motion.
        """
        dx, dy = 0, 0
        if self._fd is None:
            return (dx, dy)

        try:
            while True:
                data = os.read(self._fd, self.EVENT_SIZE)
                if len(data) != self.EVENT_SIZE:
                    break
                _tv_sec, _tv_usec, ev_type, ev_code, ev_value = struct.unpack(
                    "llHHi", data
                )
                if ev_type != EV_REL:
                    continue
                if ev_code == REL_X:
                    dx += ev_value
                elif ev_code == REL_Y:
                    dy += ev_value
        except BlockingIOError:
            pass
        except OSError as exc:
            logger.error("Mouse read error on %s: %s", self._event_path, exc)

        return (dx, dy)


# ============================================================================
# Evdev device reader (pure Python, no Qt)
# ============================================================================

class EvdevDeviceReader:
    """Reads raw evdev events from a single /dev/input/eventX device.

    Pure Python implementation using ``select()`` + ``os.read()`` +
    ``struct.unpack()``.  No Qt, no threads.
    """

    EVENT_SIZE = 24  # sizeof(struct input_event) on x86_64

    def __init__(self, event_path: str) -> None:
        self._event_path = event_path
        self._fd: Optional[int] = None

    @property
    def is_open(self) -> bool:
        return self._fd is not None

    def open(self) -> None:
        """Open the device file descriptor (non-blocking)."""
        if self._fd is not None:
            return
        self._fd = os.open(self._event_path, os.O_RDONLY | os.O_NONBLOCK)
        logger.info("Opened evdev device: %s (fd=%d)", self._event_path, self._fd)

    def close(self) -> None:
        """Close the device file descriptor."""
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError as exc:
                logger.warning("Error closing fd %d: %s", self._fd, exc)
            self._fd = None

    def wait_for_key_press(self, timeout: float = 0.5) -> Optional[int]:
        """Block until a key-press event arrives, then return the keycode.

        Args:
            timeout: Seconds to wait per select() call.

        Returns:
            Linux keycode (int) on press, or None on timeout/no data.
        """
        if self._fd is None:
            return None

        try:
            readable, _, _ = select.select([self._fd], [], [], timeout)
            if not readable:
                return None
            return self._read_key_press()
        except (select.error, OSError) as exc:
            logger.error("select() error on %s: %s", self._event_path, exc)
            return None

    def read_all_pressed(self, timeout: float = 0.0) -> Set[int]:
        """Read all currently pressed keycodes (non-blocking drain).

        Drains the device buffer and tracks press/release state.
        Returns the set of keycodes that are currently held down.

        Args:
            timeout: Seconds to wait for initial data (0 = instant).

        Returns:
            Set of Linux keycodes that are currently pressed.
        """
        pressed: Set[int] = set()
        if self._fd is None:
            return pressed

        try:
            readable, _, _ = select.select([self._fd], [], [], timeout)
            if not readable:
                return pressed
        except (select.error, OSError):
            return pressed

        try:
            while True:
                data = os.read(self._fd, self.EVENT_SIZE)
                if len(data) != self.EVENT_SIZE:
                    break
                _tv_sec, _tv_usec, ev_type, ev_code, ev_value = struct.unpack(
                    "llHHi", data
                )
                if ev_type != EV_KEY:
                    continue
                if ev_value == EV_VALUE_PRESS:
                    pressed.add(ev_code)
                elif ev_value == EV_VALUE_RELEASE:
                    pressed.discard(ev_code)
        except BlockingIOError:
            pass
        except OSError as exc:
            logger.error("Read error on %s: %s", self._event_path, exc)

        return pressed

    def _read_key_press(self) -> Optional[int]:
        """Read events from the fd until an EV_KEY press is found."""
        if self._fd is None:
            return None
        try:
            while True:
                data = os.read(self._fd, self.EVENT_SIZE)
                if len(data) != self.EVENT_SIZE:
                    return None
                _tv_sec, _tv_usec, ev_type, ev_code, ev_value = struct.unpack(
                    "llHHi", data
                )
                if ev_type == EV_KEY and ev_value == EV_VALUE_PRESS:
                    return ev_code
        except BlockingIOError:
            return None
        except OSError as exc:
            logger.error("Read error on %s: %s", self._event_path, exc)
            return None

    def drain_key_events(self) -> List[tuple]:
        """Drain all pending key events (press AND release).

        Returns a list of ``(keycode, pressed)`` tuples where *pressed*
        is ``True`` for key-down and ``False`` for key-up.  This is used
        to forward raw key state to native input handlers (e.g. Malmo
        TCP side-channel) alongside the resolved action integer.
        """
        events: List[tuple] = []
        if self._fd is None:
            return events
        try:
            while True:
                data = os.read(self._fd, self.EVENT_SIZE)
                if len(data) != self.EVENT_SIZE:
                    break
                _tv_sec, _tv_usec, ev_type, ev_code, ev_value = struct.unpack(
                    "llHHi", data
                )
                if ev_type == EV_KEY:
                    if ev_value == EV_VALUE_PRESS:
                        events.append((ev_code, True))
                    elif ev_value == EV_VALUE_RELEASE:
                        events.append((ev_code, False))
        except BlockingIOError:
            pass
        except OSError as exc:
            logger.error("Read error on %s: %s", self._event_path, exc)
        return events


# ============================================================================
# Linux keycode to action resolver (pure Python, no Qt)
# ============================================================================

class LinuxKeycodeActionResolver:
    """Maps Linux keycodes directly to game action indices.

    This replaces the Qt-dependent ``KeyCombinationResolver`` for use in
    subprocesses that have no access to Qt.  It uses raw Linux keycode
    integers from ``/usr/include/linux/input-event-codes.h``.
    """

    def resolve(self, pressed_keycodes: Set[int]) -> Optional[int]:
        """Resolve a set of pressed Linux keycodes to an action index.

        Args:
            pressed_keycodes: Set of currently held Linux keycodes.

        Returns:
            Action index, or None if no recognized combination.
        """
        raise NotImplementedError


class DynamicKeycodeResolver(LinuxKeycodeActionResolver):
    """Generic resolver built from a key-action map passed at runtime.

    The GUI already knows the exact key-to-action mapping for every
    environment (via ``ShortcutMapping`` in ``human_input.py``).  Instead
    of duplicating that knowledge per environment, the GUI converts its
    mappings into a portable ``key_action_map`` and sends it to the worker
    at init time.

    The map format is a list of ``{"keycodes": [int, ...], "action": int}``
    entries.  Each entry maps *any* of the listed keycodes to the given
    action.  First match wins.

    Example::

        [
            {"keycodes": [105, 30], "action": 0},  # Left/A  -> LEFT
            {"keycodes": [108, 31], "action": 1},  # Down/S  -> DOWN
            {"keycodes": [106, 32], "action": 2},  # Right/D -> RIGHT
            {"keycodes": [103, 17], "action": 3},  # Up/W    -> UP
        ]
    """

    def __init__(self, key_action_map: List[Dict]) -> None:
        # Build lookup: keycode -> action (first entry wins)
        self._keycode_to_action: Dict[int, int] = {}
        for entry in key_action_map:
            action = entry["action"]
            for kc in entry["keycodes"]:
                if kc not in self._keycode_to_action:
                    self._keycode_to_action[kc] = action

    def resolve(self, pressed_keycodes: Set[int]) -> Optional[int]:
        for kc in pressed_keycodes:
            if kc in self._keycode_to_action:
                return self._keycode_to_action[kc]
        return None


class MultiGridKeycodeResolver(LinuxKeycodeActionResolver):
    """Resolver for MOSAIC MultiGrid (mosaic_multigrid v6).

    Action space (8 discrete):
        0: STILL (noop)
        1: LEFT (turn left)
        2: RIGHT (turn right)
        3: FORWARD
        4: PICKUP
        5: DROP
        6: TOGGLE
        7: DONE
    """

    def resolve(self, pressed: Set[int]) -> Optional[int]:
        if not pressed:
            return None
        # Action buttons (higher priority)
        if pressed & {KEY_SPACE, KEY_G}:
            return 4  # PICKUP
        if KEY_H in pressed:
            return 5  # DROP
        if pressed & {KEY_E, KEY_RETURN}:
            return 6  # TOGGLE
        if KEY_Q in pressed:
            return 7  # DONE
        # Movement
        if pressed & KEYS_UP:
            return 3  # FORWARD
        if pressed & KEYS_LEFT:
            return 1  # LEFT
        if pressed & KEYS_RIGHT:
            return 2  # RIGHT
        if pressed & KEYS_DOWN:
            return 0  # STILL (backward = noop in multigrid)
        return None


class INIMultiGridKeycodeResolver(LinuxKeycodeActionResolver):
    """Resolver for INI MultiGrid (multigrid package).

    Action space (7 discrete, NO STILL/NOOP):
        0: LEFT (turn left)
        1: RIGHT (turn right)
        2: FORWARD
        3: PICKUP
        4: DROP
        5: TOGGLE
        6: DONE
    """

    def resolve(self, pressed: Set[int]) -> Optional[int]:
        if not pressed:
            return None
        # Action buttons (higher priority)
        if pressed & {KEY_SPACE, KEY_G}:
            return 3  # PICKUP
        if KEY_H in pressed:
            return 4  # DROP
        if pressed & {KEY_E, KEY_RETURN}:
            return 5  # TOGGLE
        if KEY_Q in pressed:
            return 6  # DONE
        # Movement
        if pressed & KEYS_UP:
            return 2  # FORWARD
        if pressed & KEYS_LEFT:
            return 0  # LEFT
        if pressed & KEYS_RIGHT:
            return 1  # RIGHT
        return None


class MiniGridKeycodeResolver(LinuxKeycodeActionResolver):
    """Resolver for MiniGrid/BabyAI.

    Action space (7 discrete):
        0: LEFT (turn left)
        1: RIGHT (turn right)
        2: FORWARD
        3: PICKUP
        4: DROP
        5: TOGGLE
        6: DONE
    """

    def resolve(self, pressed: Set[int]) -> Optional[int]:
        if not pressed:
            return None
        if pressed & {KEY_SPACE, KEY_G}:
            return 3  # PICKUP
        if KEY_H in pressed:
            return 4  # DROP
        if pressed & {KEY_E, KEY_RETURN}:
            return 5  # TOGGLE
        if KEY_Q in pressed:
            return 6  # DONE
        if pressed & KEYS_UP:
            return 2  # FORWARD
        if pressed & KEYS_LEFT:
            return 0  # LEFT
        if pressed & KEYS_RIGHT:
            return 1  # RIGHT
        return None


class ToyTextKeycodeResolver(LinuxKeycodeActionResolver):
    """Resolver for Gymnasium ToyText environments (FrozenLake, CliffWalking, Taxi).

    FrozenLake/CliffWalking action space (4 discrete):
        0: LEFT
        1: DOWN
        2: RIGHT
        3: UP

    Key mapping:
        W / Up    -> 3 (UP)
        A / Left  -> 0 (LEFT)
        S / Down  -> 1 (DOWN)
        D / Right -> 2 (RIGHT)
    """

    def resolve(self, pressed: Set[int]) -> Optional[int]:
        if not pressed:
            return None
        if pressed & KEYS_UP:
            return 3  # UP
        if pressed & KEYS_DOWN:
            return 1  # DOWN
        if pressed & KEYS_LEFT:
            return 0  # LEFT
        if pressed & KEYS_RIGHT:
            return 2  # RIGHT
        return None


class MeltingPotKeycodeResolver(LinuxKeycodeActionResolver):
    """Resolver for MeltingPot.

    Action space (11 discrete):
        0: NOOP
        1: FORWARD
        2: BACKWARD
        3: STRAFE_LEFT
        4: STRAFE_RIGHT
        5: TURN_LEFT
        6: TURN_RIGHT
        7: INTERACT
        8: FIRE_1
        9: FIRE_2
        10: FIRE_3
    """

    def resolve(self, pressed: Set[int]) -> Optional[int]:
        if not pressed:
            return None
        # Combat
        if KEY_1 in pressed:
            return 8  # FIRE_1
        if KEY_2 in pressed:
            return 9  # FIRE_2
        if KEY_3 in pressed:
            return 10  # FIRE_3
        # Interact
        if pressed & {KEY_SPACE, KEY_E, KEY_RETURN}:
            return 7  # INTERACT
        # Movement
        if pressed & KEYS_UP:
            return 1  # FORWARD
        if pressed & KEYS_DOWN:
            return 2  # BACKWARD
        if pressed & KEYS_LEFT:
            return 5  # TURN_LEFT
        if pressed & KEYS_RIGHT:
            return 6  # TURN_RIGHT
        if KEY_Q in pressed:
            return 3  # STRAFE_LEFT
        if KEY_F in pressed:
            return 4  # STRAFE_RIGHT
        return None


class MalmoKeycodeResolver(LinuxKeycodeActionResolver):
    """Resolver for MalmoEnv (Minecraft).

    Unlike other resolvers, Malmo's action space is **dynamic** (varies
    per mission XML).  The action list (e.g. ``['move 1', 'move -1',
    'turn 1', 'turn -1', 'jump 1', 'attack 1', 'use 1', 'noop 1']``)
    is provided at construction time.

    Key bindings mirror ``MalmoEnvKeyCombinationResolver`` from
    ``gym_gui/controllers/malmo_input.py``:
        W/Up    = move 1 (forward)
        S/Down  = move -1 (backward)
        A/Left  = turn -1 (turn left)
        D/Right = turn 1 (turn right)
        Space   = jump 1
        Return  = attack 1
        E/F     = use 1
    """

    def __init__(self, action_list: List[str]) -> None:
        self._index: Dict[str, int] = {
            cmd: idx for idx, cmd in enumerate(action_list)
        }

    def _action(self, command: str) -> Optional[int]:
        return self._index.get(command)

    def resolve(self, pressed: Set[int]) -> Optional[int]:
        if not pressed:
            return None
        # Priority: combat > utility > movement
        if KEY_RETURN in pressed:
            return self._action("attack 1")
        if KEY_SPACE in pressed:
            return self._action("jump 1")
        if pressed & {KEY_E, KEY_F}:
            return self._action("use 1")
        if pressed & KEYS_UP:
            return self._action("move 1")
        if pressed & KEYS_DOWN:
            return self._action("move -1")
        if pressed & KEYS_LEFT:
            return self._action("turn -1")
        if pressed & KEYS_RIGHT:
            return self._action("turn 1")
        return None


# Resolver factory

_RESOLVER_MAP: Dict[str, type] = {
    "mosaic_multigrid": MultiGridKeycodeResolver,
    "multigrid": MultiGridKeycodeResolver,
    "ini_multigrid": INIMultiGridKeycodeResolver,
    "minigrid": MiniGridKeycodeResolver,
    "babyai": MiniGridKeycodeResolver,
    "toy_text": ToyTextKeycodeResolver,
    "classic_control": ToyTextKeycodeResolver,
    "meltingpot": MeltingPotKeycodeResolver,
    "malmo": MalmoKeycodeResolver,
    "malmoenv": MalmoKeycodeResolver,
    "mosaic_malmo": MalmoKeycodeResolver,
}


def create_resolver(
    env_name: str,
    *,
    action_list: Optional[List[str]] = None,
    key_action_map: Optional[List[Dict]] = None,
) -> LinuxKeycodeActionResolver:
    """Create a keycode resolver for the given environment family.

    Args:
        env_name: Environment family name (e.g. "multigrid", "malmo").
        action_list: For Malmo only. The mission's action space list
            (e.g. ``['move 1', 'move -1', 'turn 1', ...]``).
        key_action_map: Dynamic mapping from GUI shortcuts. If provided,
            creates a ``DynamicKeycodeResolver`` and ignores env_name.
            Format: ``[{"keycodes": [int, ...], "action": int}, ...]``

    Returns:
        A resolver instance.
    """
    # Dynamic map takes priority over everything
    if key_action_map:
        logger.info("Using dynamic key-action map (%d entries)", len(key_action_map))
        return DynamicKeycodeResolver(key_action_map)

    cls = _RESOLVER_MAP.get(env_name)
    if cls is None:
        logger.warning(
            "No keycode resolver for env '%s', defaulting to MultiGrid",
            env_name,
        )
        cls = MultiGridKeycodeResolver

    # MalmoKeycodeResolver needs the action list
    if cls is MalmoKeycodeResolver:
        if action_list is None:
            raise ValueError(
                f"Malmo resolver for '{env_name}' requires action_list. "
                f"Pass the mission's action space list."
            )
        return cls(action_list)

    return cls()
