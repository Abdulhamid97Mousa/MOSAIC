from __future__ import annotations

"""Keyboard shortcut management for human control within the Qt shell.

This module provides two input modes:
1. **Shortcut-based** (QShortcut): For turn-based games where single key presses trigger actions
2. **State-based** (key tracking): For real-time games requiring simultaneous key combinations

The state-based mode tracks all currently pressed keys and computes combined actions
(e.g., Up+Right → diagonal movement) on each game tick.

For multi-keyboard support on Linux, uses evdev to bypass X11 keyboard merging.
"""

import logging
import sys
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple, cast

import gymnasium.spaces as spaces
from qtpy import QtCore, QtWidgets
from qtpy.QtCore import Qt
from qtpy.QtGui import QKeyEvent, QKeySequence, QShortcut

from gym_gui.controllers.session import SessionController
from gym_gui.core.enums import ControlMode, EnvironmentFamily, GameId, InputMode
from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_EVDEV_KEY_PRESSED,
    LOG_EVDEV_KEY_RELEASED,
    LOG_INPUT_CONTROLLER_ERROR,
    LOG_INPUT_MODE_CONFIGURED,
    LOG_KEY_RESOLVER_INITIALIZED,
    LOG_KEY_RESOLVER_UNAVAILABLE,
)

# Import evdev support for multi-keyboard (Linux only)
_HAS_EVDEV = False
if sys.platform.startswith('linux'):
    try:
        from gym_gui.controllers.evdev_keyboard_monitor import EvdevKeyboardMonitor, KeyboardDevice
        from gym_gui.controllers.keycode_translation import linux_keycode_to_qt_key
        _HAS_EVDEV = True
    except ImportError:
        pass

_LOGGER = logging.getLogger(__name__)


# =============================================================================
# Key Constants for State-Based Tracking
# =============================================================================
def _get_qt_key(name: str) -> int:
    """Get Qt key constant by name, handling Qt5/Qt6 differences."""
    key_enum = getattr(Qt, "Key", None)
    if key_enum is not None and hasattr(key_enum, name):
        return int(getattr(key_enum, name))
    legacy = getattr(Qt, name, None)
    if legacy is not None:
        return int(legacy)
    raise AttributeError(f"Qt key '{name}' not available")


# Direction keys (both arrow keys and WASD)
_KEY_UP = _get_qt_key("Key_Up")
_KEY_DOWN = _get_qt_key("Key_Down")
_KEY_LEFT = _get_qt_key("Key_Left")
_KEY_RIGHT = _get_qt_key("Key_Right")
_KEY_W = _get_qt_key("Key_W")
_KEY_A = _get_qt_key("Key_A")
_KEY_S = _get_qt_key("Key_S")
_KEY_D = _get_qt_key("Key_D")
_KEY_SPACE = _get_qt_key("Key_Space")
_KEY_Q = _get_qt_key("Key_Q")
_KEY_E = _get_qt_key("Key_E")
_KEY_G = _get_qt_key("Key_G")
_KEY_H = _get_qt_key("Key_H")
_KEY_RETURN = _get_qt_key("Key_Return")
_KEY_Z = _get_qt_key("Key_Z")
_KEY_C = _get_qt_key("Key_C")
_KEY_X = _get_qt_key("Key_X")
_KEY_1 = _get_qt_key("Key_1")
_KEY_2 = _get_qt_key("Key_2")
_KEY_3 = _get_qt_key("Key_3")

# Sets for direction detection
_KEYS_UP = {_KEY_UP, _KEY_W}
_KEYS_DOWN = {_KEY_DOWN, _KEY_S}
_KEYS_LEFT = {_KEY_LEFT, _KEY_A}
_KEYS_RIGHT = {_KEY_RIGHT, _KEY_D}


# =============================================================================
# Key Combination Resolvers for Different Game Families
# =============================================================================
class KeyCombinationResolver:
    """Base class for resolving key combinations to game actions."""

    def resolve(self, pressed_keys: Set[int]) -> Optional[int]:
        """Resolve currently pressed keys to a single game action.

        Args:
            pressed_keys: Set of currently pressed Qt key codes.

        Returns:
            Action index, or None if no recognized action.
        """
        raise NotImplementedError


class MiniGridKeyCombinationResolver(KeyCombinationResolver):
    """Resolve key combinations for MiniGrid environments (LEGACY).

    MiniGrid action space (7 discrete actions):
    0: turn left
    1: turn right
    2: move forward
    3: pick up object
    4: drop object
    5: toggle/activate (open door, use switch)
    6: done/noop

    NOTE: For MultiGrid environments, use MultiGridKeyCombinationResolver instead.
    """

    def resolve(self, pressed_keys: Set[int]) -> Optional[int]:
        """Map pressed keys to MiniGrid actions.

        Priority order:
        1. Action buttons (pickup, drop, toggle, done)
        2. Movement (forward, turn left/right)
        """
        # Check direction keys
        left = bool(pressed_keys & _KEYS_LEFT)
        right = bool(pressed_keys & _KEYS_RIGHT)
        forward = bool(pressed_keys & _KEYS_UP)

        # Action buttons (higher priority)
        if _KEY_G in pressed_keys or _KEY_SPACE in pressed_keys:
            return 3  # pick up
        if _KEY_H in pressed_keys:
            return 4  # drop
        if _KEY_E in pressed_keys or _KEY_RETURN in pressed_keys:
            return 5  # toggle/use
        if _KEY_Q in pressed_keys:
            return 6  # done/noop

        # Movement (only if no action button pressed)
        if left:
            return 0  # turn left
        if right:
            return 1  # turn right
        if forward:
            return 2  # move forward

        return None  # No action - idle


class MultiGridKeyCombinationResolver(KeyCombinationResolver):
    """Resolve key combinations for mosaic_multigrid environments (Soccer, Collect, Basketball).

    mosaic_multigrid v6.0.0 action space (8 discrete actions):
    0: NOOP    - Do nothing (AEC compatibility — non-acting agents wait, inspired by MeltingPot)
    1: LEFT    - Turn left
    2: RIGHT   - Turn right
    3: FORWARD - Move forward
    4: PICKUP  - Pick up object / steal from opponent
    5: DROP    - Drop object / score at goal / teleport pass
    6: TOGGLE  - Toggle/activate object
    7: DONE    - Done action

    NOTE: This is for the mosaic_multigrid package (PyPI: mosaic-multigrid v6.0.0).
    For INI multigrid environments, use INIMultiGridKeyCombinationResolver.
    """

    def resolve(self, pressed_keys: Set[int]) -> Optional[int]:
        """Map pressed keys to mosaic_multigrid v6 actions.

        Priority order:
        1. Action buttons (pickup, drop, toggle, done)
        2. Movement (forward, turn left/right)
        3. No keys = None (caller should use NOOP action 0)
        """
        # Check direction keys
        left = bool(pressed_keys & _KEYS_LEFT)
        right = bool(pressed_keys & _KEYS_RIGHT)
        forward = bool(pressed_keys & _KEYS_UP)

        # Action buttons (higher priority)
        if _KEY_G in pressed_keys or _KEY_SPACE in pressed_keys:
            return 4  # PICKUP
        if _KEY_H in pressed_keys:
            return 5  # DROP
        if _KEY_E in pressed_keys or _KEY_RETURN in pressed_keys:
            return 6  # TOGGLE
        if _KEY_Q in pressed_keys:
            return 7  # DONE

        # Movement (only if no action button pressed)
        if left:
            return 1  # LEFT (turn left)
        if right:
            return 2  # RIGHT (turn right)
        if forward:
            return 3  # FORWARD (move forward)

        return None  # No action - caller should use NOOP (action 0)


class INIMultiGridKeyCombinationResolver(KeyCombinationResolver):
    """Resolve key combinations for INI multigrid environments.

    INI multigrid action space (7 discrete actions - NO STILL):
    0: LEFT - Turn left
    1: RIGHT - Turn right
    2: FORWARD - Move forward
    3: PICKUP - Pick up object
    4: DROP - Drop object
    5: TOGGLE - Toggle/activate object
    6: DONE - Done action

    This is for INI multigrid environments: BlockedUnlockPickup, Empty,
    LockedHallway, RedBlueDoors, Playground.

    Repository: https://github.com/ini/multigrid
    """

    def resolve(self, pressed_keys: Set[int]) -> Optional[int]:
        """Map pressed keys to INI MultiGrid actions.

        Priority order:
        1. Action buttons (pickup, drop, toggle, done)
        2. Movement (forward, turn left/right)
        3. No keys = None (no STILL action in INI multigrid)
        """
        # Check direction keys
        left = bool(pressed_keys & _KEYS_LEFT)
        right = bool(pressed_keys & _KEYS_RIGHT)
        forward = bool(pressed_keys & _KEYS_UP)

        # Action buttons (higher priority)
        if _KEY_G in pressed_keys or _KEY_SPACE in pressed_keys:
            return 3  # PICKUP (action 3 in INI)
        if _KEY_H in pressed_keys:
            return 4  # DROP (action 4 in INI)
        if _KEY_E in pressed_keys or _KEY_RETURN in pressed_keys:
            return 5  # TOGGLE (action 5 in INI)
        if _KEY_Q in pressed_keys:
            return 6  # DONE (action 6 in INI)

        # Movement (only if no action button pressed)
        if left:
            return 0  # LEFT (action 0 in INI - turn left)
        if right:
            return 1  # RIGHT (action 1 in INI - turn right)
        if forward:
            return 2  # FORWARD (action 2 in INI - move forward)

        return None  # No action pressed


class RWAREKeyCombinationResolver(KeyCombinationResolver):
    """Resolve key combinations for RWARE (Robotic Warehouse) environments.

    RWARE action space (5 discrete actions):
    0: NOOP     - Do nothing
    1: FORWARD  - Move forward one cell
    2: LEFT     - Turn left 90 degrees
    3: RIGHT    - Turn right 90 degrees
    4: TOGGLE_LOAD - Pick up / put down a shelf

    Key mappings:
    - Arrow Up / W      -> FORWARD (1)
    - Arrow Left / A    -> LEFT (2)
    - Arrow Right / D   -> RIGHT (3)
    - Space / E / Enter -> TOGGLE_LOAD (4)
    - No keys           -> None (caller uses NOOP)
    """

    def resolve(self, pressed_keys: Set[int]) -> Optional[int]:
        left = bool(pressed_keys & _KEYS_LEFT)
        right = bool(pressed_keys & _KEYS_RIGHT)
        forward = bool(pressed_keys & _KEYS_UP)

        # Action button (higher priority)
        if _KEY_SPACE in pressed_keys or _KEY_E in pressed_keys or _KEY_RETURN in pressed_keys:
            return 4  # TOGGLE_LOAD

        # Movement
        if left:
            return 2  # LEFT
        if right:
            return 3  # RIGHT
        if forward:
            return 1  # FORWARD

        return None  # No action - caller should use NOOP (action 0)


class GriddlyKeyCombinationResolver(KeyCombinationResolver):
    """Resolve key combinations for Griddly single-agent grid world environments.

    IMPORTANT: Griddly's ACTUAL action space differs from its documentation!
    The documentation claims: 0=NOOP, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT
    But the ACTUAL action space is: 0=NOOP, 1=LEFT, 2=UP, 3=RIGHT, 4=DOWN

    Key mappings (corrected for actual Griddly behavior):
    - Arrow Up  / W -> action 2 (UP)
    - Arrow Down / S -> action 4 (DOWN)
    - Arrow Left / A -> action 1 (LEFT)
    - Arrow Right / D -> action 3 (RIGHT)
    - No keys       -> None   (caller uses NOOP)
    """

    def resolve(self, pressed_keys: Set[int]) -> Optional[int]:
        if pressed_keys & _KEYS_UP:
            return 2  # UP (actual Griddly action for moving up)
        if pressed_keys & _KEYS_DOWN:
            return 4  # DOWN (actual Griddly action for moving down)
        if pressed_keys & _KEYS_LEFT:
            return 1  # LEFT (actual Griddly action for moving left)
        if pressed_keys & _KEYS_RIGHT:
            return 3  # RIGHT (actual Griddly action for moving right)
        return None  # No action - caller should use NOOP (action 0)


class MeltingPotKeyCombinationResolver(KeyCombinationResolver):
    """Resolve key combinations for Melting Pot multi-agent environments.

    MeltingPot action space varies by substrate:
    - 8 actions (basic): NOOP, FORWARD, BACKWARD, LEFT, RIGHT, TURN_LEFT, TURN_RIGHT, INTERACT
    - 9 actions: Adds FIRE_1 (e.g., fireZap in some substrates)
    - 11 actions (e.g., clean_up): Adds FIRE_1 (fireZap), FIRE_2 (fireClean), FIRE_3

    Action indices:
    0: NOOP - Do nothing
    1: FORWARD - Move forward
    2: BACKWARD - Move backward
    3: LEFT - Strafe left
    4: RIGHT - Strafe right
    5: TURN_LEFT - Turn left
    6: TURN_RIGHT - Turn right
    7: INTERACT - Interact/use
    8: FIRE_1 - Secondary action (e.g., fireZap in clean_up) - key Z or 1
    9: FIRE_2 - Tertiary action (e.g., fireClean in clean_up) - key C or 2
    10: FIRE_3 - Quaternary action - key X or 3

    This resolver provides FPS-style controls for MeltingPot social scenarios.
    Enables multi-keyboard support for simultaneous multi-agent gameplay.
    """

    def __init__(self, num_actions: int = 8) -> None:
        """Initialize with the number of available actions.

        Args:
            num_actions: Number of discrete actions in the action space.
                         Determines which key bindings are active.
        """
        self._num_actions = num_actions

    def resolve(self, pressed_keys: Set[int]) -> Optional[int]:
        """Map pressed keys to MeltingPot actions.

        Priority order:
        1. Extra fire actions (Z/C/X or 1/2/3) - only if action space supports them
        2. Interact button (space/G)
        3. Turning (Q/E)
        4. Movement (WASD for forward/backward/strafe)
        5. No keys = None (caller should use NOOP action)
        """
        # Check direction keys
        up = bool(pressed_keys & _KEYS_UP)  # W or Up arrow
        down = bool(pressed_keys & _KEYS_DOWN)  # S or Down arrow
        left = bool(pressed_keys & _KEYS_LEFT)  # A or Left arrow
        right = bool(pressed_keys & _KEYS_RIGHT)  # D or Right arrow

        # Extra fire actions (only available in expanded action spaces)
        # FIRE_3 (action 10) - key X or 3
        if self._num_actions > 10:
            if _KEY_X in pressed_keys or _KEY_3 in pressed_keys:
                return 10  # FIRE_3

        # FIRE_2 (action 9) - key C or 2 (e.g., fireClean in clean_up)
        if self._num_actions > 9:
            if _KEY_C in pressed_keys or _KEY_2 in pressed_keys:
                return 9  # FIRE_2

        # FIRE_1 (action 8) - key Z or 1 (e.g., fireZap in clean_up)
        if self._num_actions > 8:
            if _KEY_Z in pressed_keys or _KEY_1 in pressed_keys:
                return 8  # FIRE_1

        # Interact button (highest priority among base actions)
        if _KEY_G in pressed_keys or _KEY_SPACE in pressed_keys:
            return 7  # INTERACT

        # Turning (Q/E keys)
        if _KEY_Q in pressed_keys:
            return 5  # TURN_LEFT
        if _KEY_E in pressed_keys:
            return 6  # TURN_RIGHT

        # Movement (WASD)
        if up:
            return 1  # FORWARD
        if down:
            return 2  # BACKWARD
        if left:
            return 3  # LEFT (strafe)
        if right:
            return 4  # RIGHT (strafe)

        return None  # No action - caller should use NOOP (action 0)


class ProcgenKeyCombinationResolver(KeyCombinationResolver):
    """Resolve key combinations for Procgen environments.

    Procgen action space (15 actions):
    0: down_left, 1: left, 2: up_left, 3: down, 4: noop, 5: up,
    6: down_right, 7: right, 8: up_right,
    9: action_d (fire), 10: action_a, 11: action_w, 12: action_s,
    13: action_q, 14: action_e
    """

    def resolve(self, pressed_keys: Set[int]) -> Optional[int]:
        # Check directions
        up = bool(pressed_keys & _KEYS_UP)
        down = bool(pressed_keys & _KEYS_DOWN)
        left = bool(pressed_keys & _KEYS_LEFT)
        right = bool(pressed_keys & _KEYS_RIGHT)

        # Cancel out opposing directions
        if up and down:
            up = down = False
        if left and right:
            left = right = False

        # Diagonal combinations (check first - more specific)
        if up and right:
            return 8  # up_right
        if up and left:
            return 2  # up_left
        if down and right:
            return 6  # down_right
        if down and left:
            return 0  # down_left

        # Cardinal directions
        if up:
            return 5  # up
        if down:
            return 3  # down
        if left:
            return 1  # left
        if right:
            return 7  # right

        # Action buttons (D is primary fire/interact)
        if _KEY_SPACE in pressed_keys or _KEY_D in pressed_keys:
            return 9  # action_d (fire)

        # Secondary action buttons
        if _KEY_1 in pressed_keys:
            return 13  # action_q
        if _KEY_2 in pressed_keys:
            return 14  # action_e

        return None  # No action - will use NOOP (4) from idle tick


class AleKeyCombinationResolver(KeyCombinationResolver):
    """Resolve key combinations for ALE/Atari environments.

    Standard ALE action space (18 actions):
    0: NOOP, 1: FIRE, 2: UP, 3: RIGHT, 4: LEFT, 5: DOWN,
    6: UPRIGHT, 7: UPLEFT, 8: DOWNRIGHT, 9: DOWNLEFT,
    10: UPFIRE, 11: RIGHTFIRE, 12: LEFTFIRE, 13: DOWNFIRE,
    14: UPRIGHTFIRE, 15: UPLEFTFIRE, 16: DOWNRIGHTFIRE, 17: DOWNLEFTFIRE
    """

    def resolve(self, pressed_keys: Set[int]) -> Optional[int]:
        up = bool(pressed_keys & _KEYS_UP)
        down = bool(pressed_keys & _KEYS_DOWN)
        left = bool(pressed_keys & _KEYS_LEFT)
        right = bool(pressed_keys & _KEYS_RIGHT)
        fire = _KEY_SPACE in pressed_keys

        # Cancel opposing directions
        if up and down:
            up = down = False
        if left and right:
            left = right = False

        # Diagonal + fire combinations
        if fire:
            if up and right:
                return 14  # UPRIGHTFIRE
            if up and left:
                return 15  # UPLEFTFIRE
            if down and right:
                return 16  # DOWNRIGHTFIRE
            if down and left:
                return 17  # DOWNLEFTFIRE
            if up:
                return 10  # UPFIRE
            if right:
                return 11  # RIGHTFIRE
            if left:
                return 12  # LEFTFIRE
            if down:
                return 13  # DOWNFIRE
            return 1  # FIRE only

        # Diagonal combinations (no fire)
        if up and right:
            return 6  # UPRIGHT
        if up and left:
            return 7  # UPLEFT
        if down and right:
            return 8  # DOWNRIGHT
        if down and left:
            return 9  # DOWNLEFT

        # Cardinal directions
        if up:
            return 2  # UP
        if right:
            return 3  # RIGHT
        if left:
            return 4  # LEFT
        if down:
            return 5  # DOWN

        return None  # NOOP


class LunarLanderKeyCombinationResolver(KeyCombinationResolver):
    """Resolve key combinations for LunarLander environment."""

    def resolve(self, pressed_keys: Set[int]) -> Optional[int]:
        # LunarLander: 0=idle, 1=left engine, 2=main engine, 3=right engine
        up = bool(pressed_keys & _KEYS_UP)
        left = bool(pressed_keys & _KEYS_LEFT)
        right = bool(pressed_keys & _KEYS_RIGHT)

        # Priority: main engine > side engines
        if up:
            return 2  # Fire main engine
        if left:
            return 1  # Fire left engine
        if right:
            return 3  # Fire right engine

        return None  # Idle


# Backwards compatibility alias
Box2DKeyCombinationResolver = LunarLanderKeyCombinationResolver


class CarRacingKeyCombinationResolver(KeyCombinationResolver):
    """Resolve key combinations for CarRacing environment.

    CarRacing action mapping (discrete indices to continuous presets):
    0: idle/coast (Space)
    1: steer right (Right/D)
    2: steer left (Left/A)
    3: accelerate (Up/W)
    4: brake (Down/S)
    """

    def resolve(self, pressed_keys: Set[int]) -> Optional[int]:
        up = bool(pressed_keys & _KEYS_UP)
        down = bool(pressed_keys & _KEYS_DOWN)
        left = bool(pressed_keys & _KEYS_LEFT)
        right = bool(pressed_keys & _KEYS_RIGHT)
        space = _KEY_SPACE in pressed_keys

        # Priority: acceleration/brake > steering > idle
        if up:
            return 3  # Accelerate
        if down:
            return 4  # Brake
        if right:
            return 1  # Steer right
        if left:
            return 2  # Steer left
        if space:
            return 0  # Idle/coast

        return None  # No action


class BipedalWalkerKeyCombinationResolver(KeyCombinationResolver):
    """Resolve key combinations for BipedalWalker environment.

    BipedalWalker action mapping:
    0: neutral stance (Space)
    1: lean forward/step (Right/D)
    2: lean backward/step back (Left/A)
    3: crouch/prepare jump (Up/W)
    4: extend legs/hop (Down/S)
    """

    def resolve(self, pressed_keys: Set[int]) -> Optional[int]:
        up = bool(pressed_keys & _KEYS_UP)
        down = bool(pressed_keys & _KEYS_DOWN)
        left = bool(pressed_keys & _KEYS_LEFT)
        right = bool(pressed_keys & _KEYS_RIGHT)
        space = _KEY_SPACE in pressed_keys

        if right:
            return 1  # Lean forward
        if left:
            return 2  # Lean backward
        if up:
            return 3  # Crouch
        if down:
            return 4  # Extend legs
        if space:
            return 0  # Neutral

        return None


class PistonballKeyCombinationResolver(KeyCombinationResolver):
    """Resolve key combinations for Pistonball (PettingZoo Butterfly) environment.

    Pistonball action space (Discrete(3) for discrete mode):
    0: move left  - Move piston left
    1: stay       - Keep piston stationary
    2: move right - Move piston right

    Key mappings:
    - Arrow Left / A  -> move left (0)
    - Arrow Right / D -> move right (2)
    - No keys         -> stay (1)
    """

    def resolve(self, pressed_keys: Set[int]) -> Optional[int]:
        left = bool(pressed_keys & _KEYS_LEFT)
        right = bool(pressed_keys & _KEYS_RIGHT)

        # Cancel opposing directions
        if left and right:
            return 1  # stay

        if left:
            return 0  # move left
        if right:
            return 2  # move right

        return 1  # stay (default action)


class KnightsArchersZombiesKeyCombinationResolver(KeyCombinationResolver):
    """Resolve key combinations for Knights Archers Zombies (PettingZoo Butterfly) environment.

    Knights Archers Zombies action space (Discrete(6)):
    0: noop          - Do nothing
    1: move forward  - Move forward
    2: move backward - Move backward
    3: move left     - Move left (strafe)
    4: move right    - Move right (strafe)
    5: attack/shoot  - Attack or shoot arrow

    Key mappings:
    - Arrow Up / W       -> move forward (1)
    - Arrow Down / S     -> move backward (2)
    - Arrow Left / A     -> move left (3)
    - Arrow Right / D    -> move right (4)
    - Space / E          -> attack/shoot (5)
    - No keys            -> None (caller uses noop)
    """

    def resolve(self, pressed_keys: Set[int]) -> Optional[int]:
        up = bool(pressed_keys & _KEYS_UP)
        down = bool(pressed_keys & _KEYS_DOWN)
        left = bool(pressed_keys & _KEYS_LEFT)
        right = bool(pressed_keys & _KEYS_RIGHT)

        # Attack button (highest priority)
        if _KEY_SPACE in pressed_keys or _KEY_E in pressed_keys:
            return 5  # attack/shoot

        # Movement (cancel opposing directions)
        if up and down:
            up = down = False
        if left and right:
            left = right = False

        if up:
            return 1  # move forward
        if down:
            return 2  # move backward
        if left:
            return 3  # move left
        if right:
            return 4  # move right

        return None  # No action - caller should use noop (action 0)


class CooperativePongKeyCombinationResolver(KeyCombinationResolver):
    """Resolve key combinations for Cooperative Pong (PettingZoo Butterfly) environment.

    Cooperative Pong action space (Discrete(3)):
    0: stay      - Keep paddle stationary
    1: move up   - Move paddle up
    2: move down - Move paddle down

    Key mappings:
    - Arrow Up / W    -> move up (1)
    - Arrow Down / S  -> move down (2)
    - No keys         -> stay (0)
    """

    def resolve(self, pressed_keys: Set[int]) -> Optional[int]:
        up = bool(pressed_keys & _KEYS_UP)
        down = bool(pressed_keys & _KEYS_DOWN)

        # Cancel opposing directions
        if up and down:
            return 0  # stay

        if up:
            return 1  # move up
        if down:
            return 2  # move down

        return 0  # stay (default action)


class RPSKeyCombinationResolver(KeyCombinationResolver):
    """Resolve key combinations for Rock Paper Scissors (PettingZoo Classic) environment.

    RPS action space (Discrete(3)):
    0: rock
    1: paper
    2: scissors

    Key mappings:
    - 1 or R -> rock (0)
    - 2 or P -> paper (1)
    - 3 or S -> scissors (2)
    """

    def resolve(self, pressed_keys: Set[int]) -> Optional[int]:
        _KEY_R = _get_qt_key("Key_R")
        _KEY_P = _get_qt_key("Key_P")
        _KEY_S = _get_qt_key("Key_S")

        if _KEY_1 in pressed_keys or _KEY_R in pressed_keys:
            return 0  # rock
        if _KEY_2 in pressed_keys or _KEY_P in pressed_keys:
            return 1  # paper
        if _KEY_3 in pressed_keys or _KEY_S in pressed_keys:
            return 2  # scissors

        return None  # No action


class PokerKeyCombinationResolver(KeyCombinationResolver):
    """Resolve key combinations for poker games (Leduc Hold'em, Texas Hold'em).

    Poker action space (Discrete(4)):
    0: fold
    1: call
    2: raise
    3: check

    Key mappings:
    - F -> fold (0)
    - C -> call (1)
    - R -> raise (2)
    - Space/Enter -> check (3)
    """

    def resolve(self, pressed_keys: Set[int]) -> Optional[int]:
        _KEY_F = _get_qt_key("Key_F")
        _KEY_C = _get_qt_key("Key_C")
        _KEY_R = _get_qt_key("Key_R")

        if _KEY_F in pressed_keys:
            return 0  # fold
        if _KEY_C in pressed_keys:
            return 1  # call
        if _KEY_R in pressed_keys:
            return 2  # raise
        if _KEY_SPACE in pressed_keys or _KEY_RETURN in pressed_keys:
            return 3  # check

        return None  # No action


class HanabiKeyCombinationResolver(KeyCombinationResolver):
    """Resolve key combinations for Hanabi (PettingZoo Classic) environment.

    Hanabi action space (Discrete(variable)):
    0-4: play card from hand position 0-4
    5-9: discard card from hand position 0-4
    10+: give hint actions (color/rank hints to other players)

    Key mappings:
    - 1-5 -> play card from position 0-4
    - Q, W, E, R, T -> discard card from position 0-4
    - H -> give hint (action 10, first hint action)
    """

    def resolve(self, pressed_keys: Set[int]) -> Optional[int]:
        _KEY_4 = _get_qt_key("Key_4")
        _KEY_5 = _get_qt_key("Key_5")
        _KEY_T = _get_qt_key("Key_T")
        _KEY_W = _get_qt_key("Key_W")
        _KEY_R = _get_qt_key("Key_R")

        # Play card actions (0-4)
        if _KEY_1 in pressed_keys:
            return 0  # play card from position 0
        if _KEY_2 in pressed_keys:
            return 1  # play card from position 1
        if _KEY_3 in pressed_keys:
            return 2  # play card from position 2
        if _KEY_4 in pressed_keys:
            return 3  # play card from position 3
        if _KEY_5 in pressed_keys:
            return 4  # play card from position 4

        # Discard card actions (5-9)
        if _KEY_Q in pressed_keys:
            return 5  # discard card from position 0
        if _KEY_W in pressed_keys:
            return 6  # discard card from position 1
        if _KEY_E in pressed_keys:
            return 7  # discard card from position 2
        if _KEY_R in pressed_keys:
            return 8  # discard card from position 3
        if _KEY_T in pressed_keys:
            return 9  # discard card from position 4

        # Give hint action (10+)
        if _KEY_H in pressed_keys:
            return 10  # give hint (first hint action)

        return None  # No action


class ViZDoomKeyCombinationResolver(KeyCombinationResolver):
    """Resolve key combinations for ViZDoom environments.

    Note: ViZDoom action spaces vary by scenario. This resolver handles common patterns.
    """

    def __init__(self, game_id: GameId):
        self._game_id = game_id

    def resolve(self, pressed_keys: Set[int]) -> Optional[int]:
        # Basic actions for most scenarios
        fire = _KEY_SPACE in pressed_keys
        up = bool(pressed_keys & _KEYS_UP)
        left = bool(pressed_keys & _KEYS_LEFT)
        right = bool(pressed_keys & _KEYS_RIGHT)

        # For now, return single actions (ViZDoom handles combos differently)
        if fire:
            return 0  # ATTACK (common index)
        if up:
            return 3 if self._game_id == GameId.VIZDOOM_DEADLY_CORRIDOR else 0  # MOVE_FORWARD
        if left:
            return 1  # TURN_LEFT / MOVE_LEFT
        if right:
            return 2  # TURN_RIGHT / MOVE_RIGHT

        return None


class MalmoEnvKeyCombinationResolver(KeyCombinationResolver):
    """Resolve key combinations for MalmoEnv (Microsoft Malmo / Minecraft) environments.

    MalmoEnv action indices (matches MALMOENV_ACTIONS in adapters/mosaic_malmo.py):
    0: move forward  (W / Up)
    1: move backward (S / Down)
    2: strafe left   (A / Left)
    3: strafe right  (D / Right)
    4: turn left     (Q)
    5: turn right    (E)
    6: jump          (Space)
    7: attack        (F)
    """

    def resolve(self, pressed_keys: Set[int]) -> Optional[int]:
        _key_f = _get_qt_key("Key_F")

        if _KEY_SPACE in pressed_keys:
            return 6  # jump
        if _key_f in pressed_keys:
            return 7  # attack / break block
        if _KEY_Q in pressed_keys:
            return 4  # turn left
        if _KEY_E in pressed_keys:
            return 5  # turn right
        if pressed_keys & _KEYS_LEFT:
            return 2  # strafe left
        if pressed_keys & _KEYS_RIGHT:
            return 3  # strafe right
        if pressed_keys & _KEYS_UP:
            return 0  # move forward
        if pressed_keys & _KEYS_DOWN:
            return 1  # move backward

        return None


# Map environment families to their resolvers
def get_key_combination_resolver(
    game_id: GameId,
    action_space: Optional[spaces.Space] = None,
) -> Optional[KeyCombinationResolver]:
    """Get the appropriate key combination resolver for a game.

    Returns a resolver that maps key combinations to game actions, or None
    if no resolver is available for this game type.

    Args:
        game_id: The game identifier.
        action_space: Optional action space to determine number of actions.
                      Used for environments with variable action spaces (e.g., MeltingPot).

    The resolver is determined by:
    1. Checking for game-specific resolvers (e.g., CarRacing, LunarLander)
    2. Looking up the game's family in ENVIRONMENT_FAMILY_BY_GAME
    3. Falling back to checking game ID prefixes/patterns
    """
    from gym_gui.core.enums import ENVIRONMENT_FAMILY_BY_GAME

    # Helper to extract number of actions from action space
    def _get_num_actions() -> int:
        if action_space is None:
            return 8  # Default for MeltingPot
        if isinstance(action_space, spaces.Discrete):
            return int(action_space.n)
        return 8  # Fallback

    # Game-specific resolvers for Box2D (each has unique action space)
    if game_id == GameId.CAR_RACING:
        return CarRacingKeyCombinationResolver()
    if game_id == GameId.LUNAR_LANDER:
        return LunarLanderKeyCombinationResolver()
    if game_id == GameId.BIPEDAL_WALKER:
        return BipedalWalkerKeyCombinationResolver()

    # Game-specific resolvers for PettingZoo Classic card/poker games
    if game_id == GameId.RPS:
        return RPSKeyCombinationResolver()
    if game_id == GameId.LEDUC_HOLDEM:
        return PokerKeyCombinationResolver()
    if game_id == GameId.TEXAS_HOLDEM:
        return PokerKeyCombinationResolver()
    if game_id == GameId.HANABI:
        return HanabiKeyCombinationResolver()

    # First, try to get the family from the mapping
    family = ENVIRONMENT_FAMILY_BY_GAME.get(game_id)

    if family == EnvironmentFamily.PROCGEN:
        return ProcgenKeyCombinationResolver()
    if family in (EnvironmentFamily.ALE, EnvironmentFamily.ATARI):
        return AleKeyCombinationResolver()
    if family == EnvironmentFamily.VIZDOOM:
        return ViZDoomKeyCombinationResolver(game_id)
    if family == EnvironmentFamily.MELTINGPOT:
        return MeltingPotKeyCombinationResolver(num_actions=_get_num_actions())
    if family == EnvironmentFamily.MOSAIC_MULTIGRID:
        # MOSAIC MultiGrid environments (8 actions - PyPI package mosaic-multigrid v6.0.0)
        # noop=0, left=1, right=2, forward=3, pickup=4, drop=5, toggle=6, done=7
        return MultiGridKeyCombinationResolver()
    if family == EnvironmentFamily.INI_MULTIGRID:
        # INI multigrid environments (7 actions, no STILL)
        return INIMultiGridKeyCombinationResolver()
    if family == EnvironmentFamily.RWARE:
        # RWARE environments (5 actions: NOOP, FORWARD, LEFT, RIGHT, TOGGLE_LOAD)
        return RWAREKeyCombinationResolver()
    if family == EnvironmentFamily.GRIDDLY:
        # Griddly environments (5 actions: NOOP, UP, DOWN, LEFT, RIGHT)
        return GriddlyKeyCombinationResolver()
    if family == EnvironmentFamily.MALMOENV:
        # MalmoEnv (Microsoft Malmo / Minecraft) environments
        return MalmoEnvKeyCombinationResolver()

    # Fallback: check by game ID prefix/name for games not in the mapping
    game_name = game_id.value if hasattr(game_id, 'value') else str(game_id)

    # PettingZoo Butterfly environments
    if "pistonball" in game_name.lower():
        return PistonballKeyCombinationResolver()
    if "knights_archers_zombies" in game_name.lower():
        return KnightsArchersZombiesKeyCombinationResolver()
    if "cooperative_pong" in game_name.lower():
        return CooperativePongKeyCombinationResolver()

    # MOSAIC MultiGrid environments (8 actions - PyPI package mosaic-multigrid v6.0.0)
    # noop=0, left=1, right=2, forward=3, pickup=4, drop=5, toggle=6, done=7
    if game_name.startswith("MosaicMultiGrid"):
        return MultiGridKeyCombinationResolver()

    # INI MultiGrid environments (7 actions, no noop)
    if game_name.startswith("MultiGrid"):
        return INIMultiGridKeyCombinationResolver()

    # MeltingPot environments (variable actions: 8-11 depending on substrate)
    if game_name.startswith("meltingpot/") or game_name.startswith("MeltingPot"):
        return MeltingPotKeyCombinationResolver(num_actions=_get_num_actions())

    # MiniGrid environments (7 actions: LEFT, RIGHT, FORWARD, PICKUP, DROP, TOGGLE, DONE)
    if game_name.startswith("MiniGrid"):
        return MiniGridKeyCombinationResolver()

    if game_name.startswith("procgen:") or game_name.startswith("procgen/"):
        return ProcgenKeyCombinationResolver()
    if game_name.startswith("ALE/") or game_name.endswith("-v4") or game_name.endswith("-v5"):
        return AleKeyCombinationResolver()
    if game_name.startswith("ViZDoom"):
        return ViZDoomKeyCombinationResolver(game_id)

    return None


@dataclass(frozen=True)
class ShortcutMapping:
    key_sequences: Tuple[QKeySequence, ...]
    action: int


def _qt_key(name: str) -> int:
    key_enum = getattr(QtCore.Qt, "Key", None)
    if key_enum is not None and hasattr(key_enum, name):
        return getattr(key_enum, name)
    legacy = getattr(QtCore.Qt, name, None)
    if legacy is None:  # pragma: no cover - defensive
        raise AttributeError(f"Qt key '{name}' not available")
    return legacy


def _mapping(names: Iterable[str], action: int) -> ShortcutMapping:
    sequences = tuple(QKeySequence(_qt_key(name)) for name in names)
    return ShortcutMapping(sequences, action)


_TOY_TEXT_MAPPINGS: Dict[GameId, Tuple[ShortcutMapping, ...]] = {
    GameId.FROZEN_LAKE: (
        _mapping(("Key_Left", "Key_A"), 0),
        _mapping(("Key_Down", "Key_S"), 1),
        _mapping(("Key_Right", "Key_D"), 2),
        _mapping(("Key_Up", "Key_W"), 3),
    ),
    GameId.CLIFF_WALKING: (
        _mapping(("Key_Up", "Key_W"), 0),     # UP
        _mapping(("Key_Right", "Key_D"), 1),  # RIGHT
        _mapping(("Key_Down", "Key_S"), 2),   # DOWN
        _mapping(("Key_Left", "Key_A"), 3),   # LEFT
    ),
    GameId.TAXI: (
        _mapping(("Key_Down", "Key_S"), 0),   # SOUTH
        _mapping(("Key_Up", "Key_W"), 1),     # NORTH
        _mapping(("Key_Right", "Key_D"), 2),  # EAST
        _mapping(("Key_Left", "Key_A"), 3),   # WEST
        _mapping(("Key_Space",), 4),            # PICKUP
        _mapping(("Key_E",), 5),                # DROPOFF
    ),
    GameId.BLACKJACK: (
        _mapping(("Key_1", "Key_Q"), 0),      # STICK (stop taking cards) - 1 or Q
        _mapping(("Key_2", "Key_E"), 1),      # HIT (take another card) - 2 or E
    ),
}

# MiniGrid action space variants (grouped by which actions are NOT unused)
# Environments are grouped by their actual usable action spaces

# Group A: Movement only (0=LEFT, 1=RIGHT, 2=FORWARD) - 3 actions
# Used by: CrossingEnv, DynamicObstaclesEnv, EmptyEnv, FourRoomsEnv, LavaGapEnv
# Actions 3,4,5,6 are UNUSED - do not map keys to them
_MINIGRID_MOVEMENT_ONLY = (
    _mapping(("Key_Left", "Key_A"), 0),    # turn left
    _mapping(("Key_Right", "Key_D"), 1),   # turn right
    _mapping(("Key_Up", "Key_W"), 2),      # move forward
)

# Group B: Movement + Done (0=LEFT, 1=RIGHT, 2=FORWARD, 6=DONE) - 4 actions
# Used by: GoToDoorEnv, GoToObjectEnv
# Actions 3,4,5 are UNUSED
_MINIGRID_MOVEMENT_DONE = (
    _mapping(("Key_Left", "Key_A"), 0),    # turn left
    _mapping(("Key_Right", "Key_D"), 1),   # turn right
    _mapping(("Key_Up", "Key_W"), 2),      # move forward
    _mapping(("Key_Q",), 6),                # done
)

# Group C: Movement + Pickup (0=LEFT, 1=RIGHT, 2=FORWARD, 3=PICKUP) - 4 actions
# Used by: FetchEnv, KeyCorridorEnv
# Actions 4,5,6 are UNUSED
_MINIGRID_MOVEMENT_PICKUP = (
    _mapping(("Key_Left", "Key_A"), 0),    # turn left
    _mapping(("Key_Right", "Key_D"), 1),   # turn right
    _mapping(("Key_Up", "Key_W"), 2),      # move forward
    _mapping(("Key_G", "Key_Space"), 3),   # pick up
)

# Group D: Movement + Pickup + Toggle (0=LEFT, 1=RIGHT, 2=FORWARD, 3=PICKUP, 5=TOGGLE) - 5 actions
# Used by: DoorKeyEnv, LockedRoomEnv, MemoryEnv, UnlockPickupEnv
# Actions 4=DROP, 6=DONE are UNUSED
_MINIGRID_MOVEMENT_PICKUP_TOGGLE = (
    _mapping(("Key_Left", "Key_A"), 0),    # turn left
    _mapping(("Key_Right", "Key_D"), 1),   # turn right
    _mapping(("Key_Up", "Key_W"), 2),      # move forward
    _mapping(("Key_G", "Key_Space"), 3),   # pick up
    _mapping(("Key_E", "Key_Return"), 5),  # toggle / use
)

# Group E: Movement + Toggle (0=LEFT, 1=RIGHT, 2=FORWARD, 5=TOGGLE) - 4 actions
# Used by: MultiRoomEnv, RedBlueDoorEnv, UnlockEnv
# Actions 3=PICKUP, 4=DROP, 6=DONE are UNUSED
_MINIGRID_MOVEMENT_TOGGLE = (
    _mapping(("Key_Left", "Key_A"), 0),    # turn left
    _mapping(("Key_Right", "Key_D"), 1),   # turn right
    _mapping(("Key_Up", "Key_W"), 2),      # move forward
    _mapping(("Key_E", "Key_Return"), 5),  # toggle / use
)

# Group F: Movement + Pickup + Drop (0=LEFT, 1=RIGHT, 2=FORWARD, 3=PICKUP, 4=DROP) - 5 actions
# Used by: PutNearEnv
# Actions 5=TOGGLE, 6=DONE are UNUSED
_MINIGRID_MOVEMENT_PICKUP_DROP = (
    _mapping(("Key_Left", "Key_A"), 0),    # turn left
    _mapping(("Key_Right", "Key_D"), 1),   # turn right
    _mapping(("Key_Up", "Key_W"), 2),      # move forward
    _mapping(("Key_G", "Key_Space"), 3),   # pick up
    _mapping(("Key_H",), 4),                # drop
)

# Standard MiniGrid action mapping (7 discrete actions - ALL USED)
# MiniGrid action indices: 0=LEFT, 1=RIGHT, 2=FORWARD, 3=PICKUP, 4=DROP, 5=TOGGLE, 6=DONE
_STANDARD_MINIGRID_ACTIONS = (
    _mapping(("Key_Left", "Key_A"), 0),    # turn left
    _mapping(("Key_Right", "Key_D"), 1),   # turn right
    _mapping(("Key_Up", "Key_W"), 2),      # move forward
    _mapping(("Key_G", "Key_Space"), 3),   # pick up
    _mapping(("Key_H",), 4),                # drop
    _mapping(("Key_E", "Key_Return"), 5),  # toggle / use
    _mapping(("Key_Q",), 6),                # done / no-op
)

# mosaic_multigrid v6.0.0 action mapping (8 discrete actions)
# Action indices: 0=NOOP, 1=LEFT, 2=RIGHT, 3=FORWARD, 4=PICKUP, 5=DROP, 6=TOGGLE, 7=DONE
# Used by: Soccer, Collect, Basketball, Solo variants (PyPI: mosaic-multigrid v6.0.0)
# No key pressed → NOOP (action 0) via _get_default_idle_action()
_MOSAIC_MULTIGRID_ACTIONS = (
    _mapping(("Key_Left", "Key_A"), 1),    # turn left
    _mapping(("Key_Right", "Key_D"), 2),   # turn right
    _mapping(("Key_Up", "Key_W"), 3),      # move forward
    _mapping(("Key_G", "Key_Space"), 4),   # pick up
    _mapping(("Key_H",), 5),               # drop
    _mapping(("Key_E", "Key_Return"), 6),  # toggle / use
    _mapping(("Key_Q",), 7),               # done
)

# INI MultiGrid action mapping (7 discrete actions - NO STILL)
# INI MultiGrid action indices: 0=LEFT, 1=RIGHT, 2=FORWARD, 3=PICKUP, 4=DROP, 5=TOGGLE, 6=DONE
# Used by: BlockedUnlockPickup, Empty, LockedHallway, RedBlueDoors, Playground (INI's multigrid)
# NOTE: Same as MiniGrid action mapping!
_INI_MULTIGRID_ACTIONS = (
    _mapping(("Key_Left", "Key_A"), 0),    # turn left (action 0)
    _mapping(("Key_Right", "Key_D"), 1),   # turn right (action 1)
    _mapping(("Key_Up", "Key_W"), 2),      # move forward (action 2)
    _mapping(("Key_G", "Key_Space"), 3),   # pick up (action 3)
    _mapping(("Key_H",), 4),                # drop (action 4)
    _mapping(("Key_E", "Key_Return"), 5),  # toggle / use (action 5)
    _mapping(("Key_Q",), 6),                # done (action 6)
)

_MINIG_GRID_MAPPINGS: Dict[GameId, Tuple[ShortcutMapping, ...]] = {
    # Group A: Movement only (actions 0,1,2) - EmptyEnv, FourRoomsEnv, LavaGapEnv
    GameId.MINIGRID_EMPTY_5x5: _MINIGRID_MOVEMENT_ONLY,
    GameId.MINIGRID_EMPTY_RANDOM_5x5: _MINIGRID_MOVEMENT_ONLY,
    GameId.MINIGRID_EMPTY_6x6: _MINIGRID_MOVEMENT_ONLY,
    GameId.MINIGRID_EMPTY_RANDOM_6x6: _MINIGRID_MOVEMENT_ONLY,
    GameId.MINIGRID_EMPTY_8x8: _MINIGRID_MOVEMENT_ONLY,
    GameId.MINIGRID_EMPTY_16x16: _MINIGRID_MOVEMENT_ONLY,
    GameId.MINIGRID_LAVAGAP_S5: _MINIGRID_MOVEMENT_ONLY,
    GameId.MINIGRID_LAVAGAP_S6: _MINIGRID_MOVEMENT_ONLY,
    GameId.MINIGRID_LAVAGAP_S7: _MINIGRID_MOVEMENT_ONLY,
    # Group A: Movement only - CrossingEnv (LavaCrossing, SimpleCrossing)
    GameId.MINIGRID_LAVA_CROSSING_S9N1: _MINIGRID_MOVEMENT_ONLY,
    GameId.MINIGRID_LAVA_CROSSING_S9N2: _MINIGRID_MOVEMENT_ONLY,
    GameId.MINIGRID_LAVA_CROSSING_S9N3: _MINIGRID_MOVEMENT_ONLY,
    GameId.MINIGRID_LAVA_CROSSING_S11N5: _MINIGRID_MOVEMENT_ONLY,
    GameId.MINIGRID_SIMPLE_CROSSING_S9N1: _MINIGRID_MOVEMENT_ONLY,
    GameId.MINIGRID_SIMPLE_CROSSING_S9N2: _MINIGRID_MOVEMENT_ONLY,
    GameId.MINIGRID_SIMPLE_CROSSING_S9N3: _MINIGRID_MOVEMENT_ONLY,
    GameId.MINIGRID_SIMPLE_CROSSING_S11N5: _MINIGRID_MOVEMENT_ONLY,
    # Group A: Movement only - DynamicObstaclesEnv
    GameId.MINIGRID_DYNAMIC_OBSTACLES_5X5: _MINIGRID_MOVEMENT_ONLY,
    GameId.MINIGRID_DYNAMIC_OBSTACLES_RANDOM_5X5: _MINIGRID_MOVEMENT_ONLY,
    GameId.MINIGRID_DYNAMIC_OBSTACLES_6X6: _MINIGRID_MOVEMENT_ONLY,
    GameId.MINIGRID_DYNAMIC_OBSTACLES_RANDOM_6X6: _MINIGRID_MOVEMENT_ONLY,
    GameId.MINIGRID_DYNAMIC_OBSTACLES_8X8: _MINIGRID_MOVEMENT_ONLY,
    GameId.MINIGRID_DYNAMIC_OBSTACLES_16X16: _MINIGRID_MOVEMENT_ONLY,
    # Group D: Movement + Pickup + Toggle (actions 0,1,2,3,5) - DoorKeyEnv, LockedRoomEnv, MemoryEnv, UnlockPickupEnv
    GameId.MINIGRID_DOORKEY_5x5: _MINIGRID_MOVEMENT_PICKUP_TOGGLE,
    GameId.MINIGRID_DOORKEY_6x6: _MINIGRID_MOVEMENT_PICKUP_TOGGLE,
    GameId.MINIGRID_DOORKEY_8x8: _MINIGRID_MOVEMENT_PICKUP_TOGGLE,
    GameId.MINIGRID_DOORKEY_16x16: _MINIGRID_MOVEMENT_PICKUP_TOGGLE,
    # Group E: Movement + Toggle (actions 0,1,2,5) - MultiRoomEnv, RedBlueDoorEnv, UnlockEnv
    GameId.MINIGRID_MULTIROOM_N2_S4: _MINIGRID_MOVEMENT_TOGGLE,
    GameId.MINIGRID_MULTIROOM_N4_S5: _MINIGRID_MOVEMENT_TOGGLE,
    GameId.MINIGRID_MULTIROOM_N6: _MINIGRID_MOVEMENT_TOGGLE,
    GameId.MINIGRID_REDBLUE_DOORS_6x6: _MINIGRID_MOVEMENT_TOGGLE,
    GameId.MINIGRID_REDBLUE_DOORS_8x8: _MINIGRID_MOVEMENT_TOGGLE,
    # TODO: Need to verify action spaces for remaining environments
    GameId.MINIGRID_BLOCKED_UNLOCK_PICKUP: _STANDARD_MINIGRID_ACTIONS,
    GameId.MINIGRID_OBSTRUCTED_MAZE_1DLHB: _STANDARD_MINIGRID_ACTIONS,
    GameId.MINIGRID_OBSTRUCTED_MAZE_FULL: _STANDARD_MINIGRID_ACTIONS,
}

# MultiGrid has different action indices depending on package version:
# MOSAIC multigrid (Soccer, Collect): 0=LEFT, 1=RIGHT, 2=FORWARD, etc. (7 actions, modernized)
# INI multigrid (BlockedUnlockPickup, Empty, etc.): 0=LEFT, 1=RIGHT, 2=FORWARD, etc. (7 actions, no STILL)
_MULTIGRID_MAPPINGS: Dict[GameId, Tuple[ShortcutMapping, ...]] = {
    # MOSAIC multigrid environments (7 actions - modernized from legacy 8-action version)
    GameId.MOSAIC_MULTIGRID_SOCCER: _INI_MULTIGRID_ACTIONS,
    GameId.MOSAIC_MULTIGRID_COLLECT: _INI_MULTIGRID_ACTIONS,
    GameId.MOSAIC_MULTIGRID_COLLECT2VS2: _INI_MULTIGRID_ACTIONS,
    GameId.MOSAIC_MULTIGRID_SOCCER_2VS2_INDAGOBS: _INI_MULTIGRID_ACTIONS,
    GameId.MOSAIC_MULTIGRID_COLLECT_INDAGOBS: _INI_MULTIGRID_ACTIONS,
    GameId.MOSAIC_MULTIGRID_COLLECT2VS2_INDAGOBS: _INI_MULTIGRID_ACTIONS,
    GameId.MOSAIC_MULTIGRID_SOCCER_2VS2_TEAMOBS: _INI_MULTIGRID_ACTIONS,
    GameId.MOSAIC_MULTIGRID_COLLECT2VS2_TEAMOBS: _INI_MULTIGRID_ACTIONS,
    GameId.MOSAIC_MULTIGRID_BASKETBALL_INDAGOBS: _MOSAIC_MULTIGRID_ACTIONS,
    GameId.MOSAIC_MULTIGRID_BASKETBALL_TEAMOBS: _MOSAIC_MULTIGRID_ACTIONS,
    GameId.MOSAIC_MULTIGRID_BASKETBALL_SOLO_GREEN: _MOSAIC_MULTIGRID_ACTIONS,
    GameId.MOSAIC_MULTIGRID_BASKETBALL_SOLO_BLUE: _MOSAIC_MULTIGRID_ACTIONS,
    # American Football environments (v6.3.0)
    GameId.MOSAIC_MULTIGRID_AMERICAN_FOOTBALL_1V1: _MOSAIC_MULTIGRID_ACTIONS,
    GameId.MOSAIC_MULTIGRID_AMERICAN_FOOTBALL_2V2: _MOSAIC_MULTIGRID_ACTIONS,
    GameId.MOSAIC_MULTIGRID_AMERICAN_FOOTBALL_3V3: _MOSAIC_MULTIGRID_ACTIONS,
    GameId.MOSAIC_MULTIGRID_AMERICAN_FOOTBALL_2V2_TEAMOBS: _MOSAIC_MULTIGRID_ACTIONS,
    GameId.MOSAIC_MULTIGRID_AMERICAN_FOOTBALL_3V3_TEAMOBS: _MOSAIC_MULTIGRID_ACTIONS,
    GameId.MOSAIC_MULTIGRID_AMERICAN_FOOTBALL_SOLO_GREEN: _MOSAIC_MULTIGRID_ACTIONS,
    GameId.MOSAIC_MULTIGRID_AMERICAN_FOOTBALL_SOLO_BLUE: _MOSAIC_MULTIGRID_ACTIONS,
    # INI multigrid environments (7 actions, no STILL - same as MiniGrid)
    GameId.INI_MULTIGRID_BLOCKED_UNLOCK_PICKUP: _INI_MULTIGRID_ACTIONS,
    GameId.INI_MULTIGRID_EMPTY_5X5: _INI_MULTIGRID_ACTIONS,
    GameId.INI_MULTIGRID_EMPTY_RANDOM_5X5: _INI_MULTIGRID_ACTIONS,
    GameId.INI_MULTIGRID_EMPTY_6X6: _INI_MULTIGRID_ACTIONS,
    GameId.INI_MULTIGRID_EMPTY_RANDOM_6X6: _INI_MULTIGRID_ACTIONS,
    GameId.INI_MULTIGRID_EMPTY_8X8: _INI_MULTIGRID_ACTIONS,
    GameId.INI_MULTIGRID_EMPTY_16X16: _INI_MULTIGRID_ACTIONS,
    GameId.INI_MULTIGRID_LOCKED_HALLWAY_2ROOMS: _INI_MULTIGRID_ACTIONS,
    GameId.INI_MULTIGRID_LOCKED_HALLWAY_4ROOMS: _INI_MULTIGRID_ACTIONS,
    GameId.INI_MULTIGRID_LOCKED_HALLWAY_6ROOMS: _INI_MULTIGRID_ACTIONS,
    GameId.INI_MULTIGRID_PLAYGROUND: _INI_MULTIGRID_ACTIONS,
    GameId.INI_MULTIGRID_RED_BLUE_DOORS_6X6: _INI_MULTIGRID_ACTIONS,
    GameId.INI_MULTIGRID_RED_BLUE_DOORS_8X8: _INI_MULTIGRID_ACTIONS,
}

_BOX_2D_MAPPINGS: Dict[GameId, Tuple[ShortcutMapping, ...]] = {
    GameId.LUNAR_LANDER: (
        _mapping(("Key_Space", "Key_S"), 0),  # Idle / cut thrust
        _mapping(("Key_Left", "Key_A"), 1),   # Fire left engine
        _mapping(("Key_Up", "Key_W"), 2),     # Fire main engine
        _mapping(("Key_Right", "Key_D"), 3),  # Fire right engine
    ),
    GameId.CAR_RACING: (
        _mapping(("Key_Space",), 0),           # Neutral / coast
        _mapping(("Key_Right", "Key_D"), 1), # Steer right
        _mapping(("Key_Left", "Key_A"), 2),  # Steer left
        _mapping(("Key_Up", "Key_W"), 3),    # Accelerate
        _mapping(("Key_Down", "Key_S"), 4),  # Brake
    ),
    GameId.BIPEDAL_WALKER: (
        _mapping(("Key_Space",), 0),            # Neutral stance
        _mapping(("Key_Right", "Key_D"), 1),  # Lean forward / step
        _mapping(("Key_Left", "Key_A"), 2),   # Lean backward / step back
        _mapping(("Key_Up", "Key_W"), 3),     # Crouch / prepare jump
        _mapping(("Key_Down", "Key_S"), 4),   # Extend legs / hop
    ),
}

# ALE (Atari) Adventure mappings (Discrete(18))
# ViZDoom mappings - button indices match _available_buttons order in adapter
# Each scenario has different available buttons, so mappings match scenario-specific order
# ViZDoom mouse turn action indices: (turn_left_action, turn_right_action)
# Maps each scenario to the button indices used for turning left/right
# Used for FPS-style mouse capture control
_VIZDOOM_MOUSE_TURN_ACTIONS: Dict[GameId, Tuple[int, int]] = {
    # Basic has no turn - uses MOVE_LEFT(1), MOVE_RIGHT(2) for lateral movement
    GameId.VIZDOOM_BASIC: (1, 2),  # MOVE_LEFT, MOVE_RIGHT (no true turn)
    # DeadlyCorridor: TURN_LEFT(4), TURN_RIGHT(5)
    GameId.VIZDOOM_DEADLY_CORRIDOR: (4, 5),
    # DefendTheCenter: TURN_LEFT(1), TURN_RIGHT(2)
    GameId.VIZDOOM_DEFEND_THE_CENTER: (1, 2),
    # DefendTheLine: TURN_LEFT(1), TURN_RIGHT(2)
    GameId.VIZDOOM_DEFEND_THE_LINE: (1, 2),
    # HealthGathering: TURN_LEFT(1), TURN_RIGHT(2)
    GameId.VIZDOOM_HEALTH_GATHERING: (1, 2),
    # HealthGatheringSupreme: TURN_LEFT(1), TURN_RIGHT(2)
    GameId.VIZDOOM_HEALTH_GATHERING_SUPREME: (1, 2),
    # MyWayHome: TURN_LEFT(1), TURN_RIGHT(2)
    GameId.VIZDOOM_MY_WAY_HOME: (1, 2),
    # PredictPosition: TURN_LEFT(1), TURN_RIGHT(2)
    GameId.VIZDOOM_PREDICT_POSITION: (1, 2),
    # TakeCover has no turn - uses MOVE_LEFT(0), MOVE_RIGHT(1) for lateral movement
    GameId.VIZDOOM_TAKE_COVER: (0, 1),  # MOVE_LEFT, MOVE_RIGHT (no true turn)
    # Deathmatch: TURN_LEFT(6), TURN_RIGHT(7)
    GameId.VIZDOOM_DEATHMATCH: (6, 7),
}


def get_vizdoom_mouse_turn_actions(game_id: GameId) -> Tuple[int, int] | None:
    """Return (turn_left_action, turn_right_action) for a ViZDoom game, or None if not ViZDoom."""
    return _VIZDOOM_MOUSE_TURN_ACTIONS.get(game_id)


_VIZDOOM_MAPPINGS: Dict[GameId, Tuple[ShortcutMapping, ...]] = {
    # Basic: ATTACK(0), MOVE_LEFT(1), MOVE_RIGHT(2)
    GameId.VIZDOOM_BASIC: (
        _mapping(("Key_Space", "Key_Control"), 0),  # ATTACK
        _mapping(("Key_Left", "Key_A"), 1),          # MOVE_LEFT
        _mapping(("Key_Right", "Key_D"), 2),         # MOVE_RIGHT
    ),
    # DeadlyCorridor: ATTACK(0), MOVE_LEFT(1), MOVE_RIGHT(2), MOVE_FORWARD(3), TURN_LEFT(4), TURN_RIGHT(5)
    GameId.VIZDOOM_DEADLY_CORRIDOR: (
        _mapping(("Key_Space", "Key_Control"), 0),  # ATTACK
        _mapping(("Key_A",), 1),                     # MOVE_LEFT (strafe)
        _mapping(("Key_D",), 2),                     # MOVE_RIGHT (strafe)
        _mapping(("Key_Up", "Key_W"), 3),            # MOVE_FORWARD
        _mapping(("Key_Left", "Key_Q"), 4),          # TURN_LEFT
        _mapping(("Key_Right", "Key_E"), 5),         # TURN_RIGHT
    ),
    # DefendTheCenter: ATTACK(0), TURN_LEFT(1), TURN_RIGHT(2)
    GameId.VIZDOOM_DEFEND_THE_CENTER: (
        _mapping(("Key_Space", "Key_Control"), 0),  # ATTACK
        _mapping(("Key_Left", "Key_A"), 1),          # TURN_LEFT
        _mapping(("Key_Right", "Key_D"), 2),         # TURN_RIGHT
    ),
    # DefendTheLine: ATTACK(0), TURN_LEFT(1), TURN_RIGHT(2)
    GameId.VIZDOOM_DEFEND_THE_LINE: (
        _mapping(("Key_Space", "Key_Control"), 0),  # ATTACK
        _mapping(("Key_Left", "Key_A"), 1),          # TURN_LEFT
        _mapping(("Key_Right", "Key_D"), 2),         # TURN_RIGHT
    ),
    # HealthGathering: MOVE_FORWARD(0), TURN_LEFT(1), TURN_RIGHT(2)
    GameId.VIZDOOM_HEALTH_GATHERING: (
        _mapping(("Key_Up", "Key_W"), 0),            # MOVE_FORWARD
        _mapping(("Key_Left", "Key_A"), 1),          # TURN_LEFT
        _mapping(("Key_Right", "Key_D"), 2),         # TURN_RIGHT
    ),
    # HealthGatheringSupreme: MOVE_FORWARD(0), TURN_LEFT(1), TURN_RIGHT(2)
    GameId.VIZDOOM_HEALTH_GATHERING_SUPREME: (
        _mapping(("Key_Up", "Key_W"), 0),            # MOVE_FORWARD
        _mapping(("Key_Left", "Key_A"), 1),          # TURN_LEFT
        _mapping(("Key_Right", "Key_D"), 2),         # TURN_RIGHT
    ),
    # MyWayHome: MOVE_FORWARD(0), TURN_LEFT(1), TURN_RIGHT(2)
    GameId.VIZDOOM_MY_WAY_HOME: (
        _mapping(("Key_Up", "Key_W"), 0),            # MOVE_FORWARD
        _mapping(("Key_Left", "Key_A"), 1),          # TURN_LEFT
        _mapping(("Key_Right", "Key_D"), 2),         # TURN_RIGHT
    ),
    # PredictPosition: ATTACK(0), TURN_LEFT(1), TURN_RIGHT(2)
    GameId.VIZDOOM_PREDICT_POSITION: (
        _mapping(("Key_Space", "Key_Control"), 0),  # ATTACK
        _mapping(("Key_Left", "Key_A"), 1),          # TURN_LEFT
        _mapping(("Key_Right", "Key_D"), 2),         # TURN_RIGHT
    ),
    # TakeCover: MOVE_LEFT(0), MOVE_RIGHT(1)
    GameId.VIZDOOM_TAKE_COVER: (
        _mapping(("Key_Left", "Key_A"), 0),          # MOVE_LEFT
        _mapping(("Key_Right", "Key_D"), 1),         # MOVE_RIGHT
    ),
    # Deathmatch: ATTACK(0), USE(1), MOVE_FORWARD(2), MOVE_BACKWARD(3), MOVE_LEFT(4), MOVE_RIGHT(5), TURN_LEFT(6), TURN_RIGHT(7)
    GameId.VIZDOOM_DEATHMATCH: (
        _mapping(("Key_Space", "Key_Control"), 0),  # ATTACK
        _mapping(("Key_E", "Key_Return"), 1),        # USE
        _mapping(("Key_Up", "Key_W"), 2),            # MOVE_FORWARD
        _mapping(("Key_Down", "Key_S"), 3),          # MOVE_BACKWARD
        _mapping(("Key_A",), 4),                     # MOVE_LEFT (strafe)
        _mapping(("Key_D",), 5),                     # MOVE_RIGHT (strafe)
        _mapping(("Key_Left", "Key_Q"), 6),          # TURN_LEFT
        _mapping(("Key_Right",), 7),                 # TURN_RIGHT
    ),
}

# ===========================================================================
# MiniHack Mappings (roguelike vi-keys + WASD alternatives)
# NLE action indices: 0-7 = 8 compass directions (N, E, S, W, NE, SE, SW, NW)
# Many MiniHack envs use Discrete(8) for basic navigation
# ===========================================================================
def _minihack_nav_mappings() -> Tuple[ShortcutMapping, ...]:
    """Standard 8-direction navigation for MiniHack environments."""
    return (
        _mapping(("Key_K", "Key_Up", "Key_W"), 0),     # North (up)
        _mapping(("Key_L", "Key_Right", "Key_D"), 1),  # East (right)
        _mapping(("Key_J", "Key_Down", "Key_S"), 2),   # South (down)
        _mapping(("Key_H", "Key_Left", "Key_A"), 3),   # West (left)
        _mapping(("Key_U",), 4),                        # Northeast (diag)
        _mapping(("Key_N",), 5),                        # Southeast (diag)
        _mapping(("Key_B",), 6),                        # Southwest (diag)
        _mapping(("Key_Y",), 7),                        # Northwest (diag)
    )


_MINIHACK_MAPPINGS: Dict[GameId, Tuple[ShortcutMapping, ...]] = {
    # Navigation environments (8 directions)
    GameId.MINIHACK_ROOM_5X5: _minihack_nav_mappings(),
    GameId.MINIHACK_ROOM_15X15: _minihack_nav_mappings(),
    GameId.MINIHACK_CORRIDOR_R2: _minihack_nav_mappings(),
    GameId.MINIHACK_CORRIDOR_R3: _minihack_nav_mappings(),
    GameId.MINIHACK_CORRIDOR_R5: _minihack_nav_mappings(),
    GameId.MINIHACK_MAZEWALK_9X9: _minihack_nav_mappings(),
    GameId.MINIHACK_MAZEWALK_15X15: _minihack_nav_mappings(),
    GameId.MINIHACK_MAZEWALK_45X19: _minihack_nav_mappings(),
    GameId.MINIHACK_RIVER: _minihack_nav_mappings(),
    GameId.MINIHACK_RIVER_NARROW: _minihack_nav_mappings(),
    # Exploration environments
    GameId.MINIHACK_EXPLOREMAZE_EASY: _minihack_nav_mappings(),
    GameId.MINIHACK_EXPLOREMAZE_HARD: _minihack_nav_mappings(),
    GameId.MINIHACK_HIDENSEEK: _minihack_nav_mappings(),
    GameId.MINIHACK_MEMENTO_F2: _minihack_nav_mappings(),
    GameId.MINIHACK_MEMENTO_F4: _minihack_nav_mappings(),
    # Skill environments (use fallback for extended actions)
    GameId.MINIHACK_EAT: _minihack_nav_mappings(),
    GameId.MINIHACK_WEAR: _minihack_nav_mappings(),
    GameId.MINIHACK_WIELD: _minihack_nav_mappings(),
    GameId.MINIHACK_ZAP: _minihack_nav_mappings(),
    GameId.MINIHACK_READ: _minihack_nav_mappings(),
    GameId.MINIHACK_QUAFF: _minihack_nav_mappings(),
    GameId.MINIHACK_PUTON: _minihack_nav_mappings(),
    GameId.MINIHACK_LAVACROSS: _minihack_nav_mappings(),
    GameId.MINIHACK_WOD_EASY: _minihack_nav_mappings(),
    GameId.MINIHACK_WOD_MEDIUM: _minihack_nav_mappings(),
    GameId.MINIHACK_WOD_HARD: _minihack_nav_mappings(),
}

# ===========================================================================
# NetHack Mappings (full game via NLE)
# NLE has ~113 actions; mapping core navigation here
# ===========================================================================
_NETHACK_MAPPINGS: Dict[GameId, Tuple[ShortcutMapping, ...]] = {
    # NetHack full game uses same 8-direction navigation as base
    GameId.NETHACK_FULL: _minihack_nav_mappings(),
    GameId.NETHACK_SCORE: _minihack_nav_mappings(),
    GameId.NETHACK_STAIRCASE: _minihack_nav_mappings(),
    GameId.NETHACK_STAIRCASE_PET: _minihack_nav_mappings(),
    GameId.NETHACK_ORACLE: _minihack_nav_mappings(),
    GameId.NETHACK_GOLD: _minihack_nav_mappings(),
    GameId.NETHACK_EAT: _minihack_nav_mappings(),
    GameId.NETHACK_SCOUT: _minihack_nav_mappings(),
}

# ===========================================================================
# Crafter Mappings (open-world survival benchmark)
# 17 discrete actions from crafter/data.yaml:
# 0:noop, 1:move_left, 2:move_right, 3:move_up, 4:move_down, 5:do, 6:sleep,
# 7:place_stone, 8:place_table, 9:place_furnace, 10:place_plant,
# 11:make_wood_pickaxe, 12:make_stone_pickaxe, 13:make_iron_pickaxe,
# 14:make_wood_sword, 15:make_stone_sword, 16:make_iron_sword
# ===========================================================================
def _crafter_mappings() -> Tuple[ShortcutMapping, ...]:
    """Standard 17-action mapping for Crafter environments."""
    return (
        # Note: action 0 (noop) has no key - happens on timeout or idle
        _mapping(("Key_Left", "Key_A"), 1),       # move_left
        _mapping(("Key_Right", "Key_D"), 2),      # move_right
        _mapping(("Key_Up", "Key_W"), 3),         # move_up
        _mapping(("Key_Down", "Key_S"), 4),       # move_down
        _mapping(("Key_Space",), 5),               # do (interact)
        _mapping(("Key_R",), 6),                   # sleep
        _mapping(("Key_1",), 7),                   # place_stone
        _mapping(("Key_2",), 8),                   # place_table
        _mapping(("Key_3",), 9),                   # place_furnace
        _mapping(("Key_4",), 10),                  # place_plant
        _mapping(("Key_Q",), 11),                  # make_wood_pickaxe
        _mapping(("Key_E",), 12),                  # make_stone_pickaxe
        _mapping(("Key_F",), 13),                  # make_iron_pickaxe
        _mapping(("Key_Z",), 14),                  # make_wood_sword
        _mapping(("Key_X",), 15),                  # make_stone_sword
        _mapping(("Key_C",), 16),                  # make_iron_sword
    )


_CRAFTER_MAPPINGS: Dict[GameId, Tuple[ShortcutMapping, ...]] = {
    GameId.CRAFTER_REWARD: _crafter_mappings(),
    GameId.CRAFTER_NO_REWARD: _crafter_mappings(),
}

# ===========================================================================
# BabaIsAI Mappings (rule manipulation puzzle benchmark - ICML 2024)
# 5 discrete actions: 0:UP, 1:DOWN, 2:LEFT, 3:RIGHT, 4:IDLE
# ===========================================================================
def _babaisai_mappings() -> Tuple[ShortcutMapping, ...]:
    """Standard 5-action mapping for BabaIsAI environments."""
    return (
        _mapping(("Key_Up", "Key_W"), 0),         # UP
        _mapping(("Key_Down", "Key_S"), 1),       # DOWN
        _mapping(("Key_Left", "Key_A"), 2),       # LEFT
        _mapping(("Key_Right", "Key_D"), 3),      # RIGHT
        _mapping(("Key_Space",), 4),               # IDLE (wait/skip)
    )


_BABAISAI_MAPPINGS: Dict[GameId, Tuple[ShortcutMapping, ...]] = {
    GameId.BABAISAI_DEFAULT: _babaisai_mappings(),
}

# ===========================================================================
# Procgen Mappings (procedurally generated benchmark - 16 environments)
# 15 discrete actions (button combinations):
# 0:down_left, 1:left, 2:up_left, 3:down, 4:noop, 5:up,
# 6:down_right, 7:right, 8:up_right,
# 9:action_d (fire), 10:action_a, 11:action_w, 12:action_s, 13:action_q, 14:action_e
# ===========================================================================
def _procgen_mappings() -> Tuple[ShortcutMapping, ...]:
    """Standard 15-action mapping for Procgen environments."""
    return (
        # Diagonal: down-left (action 0) - no common key, use numpad 1 or Z
        _mapping(("Key_Z",), 0),                    # down_left
        # Cardinal directions using arrow keys
        _mapping(("Key_Left",), 1),                 # left
        # Diagonal: up-left (action 2) - use Q
        _mapping(("Key_Q",), 2),                    # up_left
        _mapping(("Key_Down",), 3),                 # down
        # Noop (action 4) - space or 0
        _mapping(("Key_0",), 4),                    # noop
        _mapping(("Key_Up",), 5),                   # up
        # Diagonal: down-right (action 6) - use C
        _mapping(("Key_C",), 6),                    # down_right
        _mapping(("Key_Right",), 7),                # right
        # Diagonal: up-right (action 8) - use E
        _mapping(("Key_E",), 8),                    # up_right
        # Game-specific actions (fire/interact buttons)
        _mapping(("Key_Space", "Key_D"), 9),        # action_d (primary fire/interact)
        _mapping(("Key_A",), 10),                   # action_a (secondary)
        _mapping(("Key_W",), 11),                   # action_w (tertiary)
        _mapping(("Key_S",), 12),                   # action_s (quaternary)
        _mapping(("Key_1",), 13),                   # action_q (special 1)
        _mapping(("Key_2",), 14),                   # action_e (special 2)
    )


_PROCGEN_MAPPINGS: Dict[GameId, Tuple[ShortcutMapping, ...]] = {
    GameId.PROCGEN_BIGFISH: _procgen_mappings(),
    GameId.PROCGEN_BOSSFIGHT: _procgen_mappings(),
    GameId.PROCGEN_CAVEFLYER: _procgen_mappings(),
    GameId.PROCGEN_CHASER: _procgen_mappings(),
    GameId.PROCGEN_CLIMBER: _procgen_mappings(),
    GameId.PROCGEN_COINRUN: _procgen_mappings(),
    GameId.PROCGEN_DODGEBALL: _procgen_mappings(),
    GameId.PROCGEN_FRUITBOT: _procgen_mappings(),
    GameId.PROCGEN_HEIST: _procgen_mappings(),
    GameId.PROCGEN_JUMPER: _procgen_mappings(),
    GameId.PROCGEN_LEAPER: _procgen_mappings(),
    GameId.PROCGEN_MAZE: _procgen_mappings(),
    GameId.PROCGEN_MINER: _procgen_mappings(),
    GameId.PROCGEN_NINJA: _procgen_mappings(),
    GameId.PROCGEN_PLUNDER: _procgen_mappings(),
    GameId.PROCGEN_STARPILOT: _procgen_mappings(),
}

_ALE_MAPPINGS: Dict[GameId, Tuple[ShortcutMapping, ...]] = {
    # Minimal but explicit mapping for core and diagonal moves, plus fire variants via letter keys.
    GameId.ADVENTURE_V4: (
        _mapping(("Key_0",), 0),               # NOOP
        _mapping(("Key_Space",), 1),          # FIRE
        _mapping(("Key_Up", "Key_W"), 2),    # UP
        _mapping(("Key_Right", "Key_D"), 3), # RIGHT
        _mapping(("Key_Left", "Key_A"), 4),  # LEFT
        _mapping(("Key_Down", "Key_S"), 5),  # DOWN
        _mapping(("Key_E",), 6),               # UPRIGHT
        _mapping(("Key_Q",), 7),               # UPLEFT
        _mapping(("Key_C",), 8),               # DOWNRIGHT
        _mapping(("Key_Z",), 9),               # DOWNLEFT
        _mapping(("Key_I",), 10),              # UPFIRE
        _mapping(("Key_L",), 11),              # RIGHTFIRE
        _mapping(("Key_J",), 12),              # LEFTFIRE
        _mapping(("Key_K",), 13),              # DOWNFIRE
        _mapping(("Key_O",), 14),              # UPRIGHTFIRE
        _mapping(("Key_U",), 15),              # UPLEFTFIRE
        _mapping(("Key_M",), 16),              # DOWNRIGHTFIRE
        _mapping(("Key_N",), 17),              # DOWNLEFTFIRE
    ),
    GameId.ALE_ADVENTURE_V5: (
        _mapping(("Key_0",), 0),               # NOOP
        _mapping(("Key_Space",), 1),          # FIRE
        _mapping(("Key_Up", "Key_W"), 2),    # UP
        _mapping(("Key_Right", "Key_D"), 3), # RIGHT
        _mapping(("Key_Left", "Key_A"), 4),  # LEFT
        _mapping(("Key_Down", "Key_S"), 5),  # DOWN
        _mapping(("Key_E",), 6),               # UPRIGHT
        _mapping(("Key_Q",), 7),               # UPLEFT
        _mapping(("Key_C",), 8),               # DOWNRIGHT
        _mapping(("Key_Z",), 9),               # DOWNLEFT
        _mapping(("Key_I",), 10),              # UPFIRE
        _mapping(("Key_L",), 11),              # RIGHTFIRE
        _mapping(("Key_J",), 12),              # LEFTFIRE
        _mapping(("Key_K",), 13),              # DOWNFIRE
        _mapping(("Key_O",), 14),              # UPRIGHTFIRE
        _mapping(("Key_U",), 15),              # UPLEFTFIRE
        _mapping(("Key_M",), 16),              # DOWNRIGHTFIRE
        _mapping(("Key_N",), 17),              # DOWNLEFTFIRE
    ),
    # AirRaid uses Discrete(6) by default; map core actions and fire variants
    GameId.AIR_RAID_V4: (
        _mapping(("Key_0",), 0),               # NOOP
        _mapping(("Key_Space",), 1),          # FIRE
        _mapping(("Key_Right", "Key_D"), 2), # RIGHT
        _mapping(("Key_Left", "Key_A"), 3),  # LEFT
        _mapping(("Key_L",), 4),              # RIGHTFIRE
        _mapping(("Key_J",), 5),              # LEFTFIRE
    ),
    GameId.ALE_AIR_RAID_V5: (
        _mapping(("Key_0",), 0),               # NOOP
        _mapping(("Key_Space",), 1),          # FIRE
        _mapping(("Key_Right", "Key_D"), 2), # RIGHT
        _mapping(("Key_Left", "Key_A"), 3),  # LEFT
        _mapping(("Key_L",), 4),              # RIGHTFIRE
        _mapping(("Key_J",), 5),              # LEFTFIRE
    ),
    # Assault uses Discrete(7): NOOP, FIRE, UP, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE
    GameId.ASSAULT_V4: (
        _mapping(("Key_0",), 0),               # NOOP
        _mapping(("Key_Space",), 1),          # FIRE
        _mapping(("Key_Up", "Key_W"), 2),    # UP
        _mapping(("Key_Right", "Key_D"), 3), # RIGHT
        _mapping(("Key_Left", "Key_A"), 4),  # LEFT
        _mapping(("Key_L",), 5),              # RIGHTFIRE
        _mapping(("Key_J",), 6),              # LEFTFIRE
    ),
    GameId.ALE_ASSAULT_V5: (
        _mapping(("Key_0",), 0),               # NOOP
        _mapping(("Key_Space",), 1),          # FIRE
        _mapping(("Key_Up", "Key_W"), 2),    # UP
        _mapping(("Key_Right", "Key_D"), 3), # RIGHT
        _mapping(("Key_Left", "Key_A"), 4),  # LEFT
        _mapping(("Key_L",), 5),              # RIGHTFIRE
        _mapping(("Key_J",), 6),              # LEFTFIRE
    ),
}


# Jumanji Logic Puzzle Environments
# Game2048: 4 actions (up=0, down=1, left=2, right=3)
# Minesweeper: Cell indices (use mouse click, not keyboard for this one)
# RubiksCube: 12 actions (6 faces x 2 directions)
# SlidingPuzzle: 4 actions (move blank up/down/left/right)
# Sudoku: 81 cells x 9 digits (complex, may need mouse)
# GraphColoring: node x color combinations (complex, may need mouse)

def _game2048_mappings() -> Tuple[ShortcutMapping, ...]:
    """Arrow keys for 2048 tile sliding.

    Jumanji Game2048 action space: Discrete(4)
    [0, 1, 2, 3] -> [Up, Right, Down, Left]
    """
    return (
        _mapping(("Key_Up", "Key_W"), 0),      # up
        _mapping(("Key_Right", "Key_D"), 1),   # right
        _mapping(("Key_Down", "Key_S"), 2),    # down
        _mapping(("Key_Left", "Key_A"), 3),    # left
    )


def _sliding_puzzle_mappings() -> Tuple[ShortcutMapping, ...]:
    """Arrow keys for sliding tile puzzle.

    Jumanji SlidingTilePuzzle action space: Discrete(4)
    [0, 1, 2, 3] -> [Up, Right, Down, Left]
    """
    return (
        _mapping(("Key_Up", "Key_W"), 0),      # move blank up
        _mapping(("Key_Right", "Key_D"), 1),   # move blank right
        _mapping(("Key_Down", "Key_S"), 2),    # move blank down
        _mapping(("Key_Left", "Key_A"), 3),    # move blank left
    )


def _rubiks_cube_mappings() -> Tuple[ShortcutMapping, ...]:
    """Letter keys for Rubik's Cube face rotations.

    Standard cube notation:
    R=right, L=left, U=up, D=down, F=front, B=back
    Shift+key for counter-clockwise (prime)
    """
    return (
        _mapping(("Key_R",), 0),   # R (right clockwise)
        _mapping(("Key_T",), 1),   # R' (right counter-clockwise, use T for shift-free)
        _mapping(("Key_L",), 2),   # L (left clockwise)
        _mapping(("Key_K",), 3),   # L' (left counter-clockwise)
        _mapping(("Key_U",), 4),   # U (up clockwise)
        _mapping(("Key_Y",), 5),   # U' (up counter-clockwise)
        _mapping(("Key_D",), 6),   # D (down clockwise)
        _mapping(("Key_E",), 7),   # D' (down counter-clockwise)
        _mapping(("Key_F",), 8),   # F (front clockwise)
        _mapping(("Key_G",), 9),   # F' (front counter-clockwise)
        _mapping(("Key_B",), 10),  # B (back clockwise)
        _mapping(("Key_N",), 11),  # B' (back counter-clockwise)
    )


def _pacman_mappings() -> Tuple[ShortcutMapping, ...]:
    """Arrow keys for PacMan movement.

    Jumanji PacMan action space: Discrete(5)
    - 0: no-op (stay)
    - 1: up
    - 2: right
    - 3: down
    - 4: left
    """
    return (
        _mapping(("Key_Up", "Key_W"), 1),      # up
        _mapping(("Key_Right", "Key_D"), 2),   # right
        _mapping(("Key_Down", "Key_S"), 3),    # down
        _mapping(("Key_Left", "Key_A"), 4),    # left
    )


def _snake_mappings() -> Tuple[ShortcutMapping, ...]:
    """Arrow keys for Snake movement.

    Jumanji Snake action space: Discrete(4)
    - 0: up
    - 1: right
    - 2: down
    - 3: left
    """
    return (
        _mapping(("Key_Up", "Key_W"), 0),      # up
        _mapping(("Key_Right", "Key_D"), 1),   # right
        _mapping(("Key_Down", "Key_S"), 2),    # down
        _mapping(("Key_Left", "Key_A"), 3),    # left
    )


def _jumanji_4dir_mappings() -> Tuple[ShortcutMapping, ...]:
    """Standard 4-direction mapping shared by Maze, Sokoban, and Cleaner.

    Jumanji convention: Discrete(4)
    [0, 1, 2, 3] -> [Up, Right, Down, Left]
    """
    return (
        _mapping(("Key_Up", "Key_W"), 0),      # up
        _mapping(("Key_Right", "Key_D"), 1),   # right
        _mapping(("Key_Down", "Key_S"), 2),    # down
        _mapping(("Key_Left", "Key_A"), 3),    # left
    )


_JUMANJI_MAPPINGS: Dict[GameId, Tuple[ShortcutMapping, ...]] = {
    GameId.JUMANJI_GAME2048: _game2048_mappings(),
    GameId.JUMANJI_SLIDING_PUZZLE: _sliding_puzzle_mappings(),
    GameId.JUMANJI_RUBIKS_CUBE: _rubiks_cube_mappings(),
    GameId.JUMANJI_PACMAN: _pacman_mappings(),
    GameId.JUMANJI_SNAKE: _snake_mappings(),
    GameId.JUMANJI_MAZE: _jumanji_4dir_mappings(),
    GameId.JUMANJI_SOKOBAN: _jumanji_4dir_mappings(),
    GameId.JUMANJI_CLEANER: _jumanji_4dir_mappings(),
    # Minesweeper, Sudoku, and GraphColoring use complex action spaces
    # that are better suited for mouse-based interaction (Tier 3)
    # Tetris uses MultiDiscrete which needs special handling (Tier 2)
}


# ============================================================================
# Qt key-name to Linux keycode mapping (for bridge subprocess)
# ============================================================================

# Reverse mapping: Qt key enum value -> key name string
_QT_KEY_ENUM_TO_NAME: Dict[int, str] = {}


def _build_qt_key_enum_map() -> Dict[int, str]:
    """Build a mapping from Qt key integer values to key name strings."""
    if _QT_KEY_ENUM_TO_NAME:
        return _QT_KEY_ENUM_TO_NAME

    key_names = [
        "Key_Escape", "Key_1", "Key_2", "Key_3", "Key_Q", "Key_W", "Key_E",
        "Key_R", "Key_A", "Key_S", "Key_D", "Key_F", "Key_G", "Key_H",
        "Key_Z", "Key_X", "Key_C", "Key_Space", "Key_F1", "Key_Up",
        "Key_Down", "Key_Left", "Key_Right", "Key_Return",
    ]
    for name in key_names:
        try:
            val = _qt_key(name)
            _QT_KEY_ENUM_TO_NAME[val] = name
        except (AttributeError, TypeError):
            pass
    return _QT_KEY_ENUM_TO_NAME


def build_key_action_map_for_game(
    game_id,
    env_family=None,
    action_space=None,
) -> Optional[List[Dict[str, Any]]]:
    """Convert Qt ``ShortcutMapping`` entries to a portable Linux keycode map.

    Looks up the same mapping dicts the Qt shortcut system uses, then converts
    each ``QKeySequence`` to a Linux keycode via ``QT_KEY_TO_LINUX`` in
    ``evdev_input.py``.

    Args:
        game_id: The ``GameId`` enum value.
        env_family: Optional ``EnvironmentFamily`` for BabyAI fallback.
        action_space: Optional Gymnasium action space for generic fallback.

    Returns:
        A list of ``{"keycodes": [int, ...], "action": int}`` entries,
        or None if no mappings found.
    """
    from gym_gui.config.paths import HUMAN_WORKER_PKG_DIR
    import sys
    _hw = str(HUMAN_WORKER_PKG_DIR)
    if _hw not in sys.path:
        sys.path.insert(0, _hw)
    from human_worker.evdev_input import QT_KEY_TO_LINUX

    # Look up the ShortcutMapping tuple for this game
    mappings = None
    mapping_sources = [
        _TOY_TEXT_MAPPINGS,
        _MINIG_GRID_MAPPINGS,
        _MULTIGRID_MAPPINGS,
        _BOX_2D_MAPPINGS,
        _ALE_MAPPINGS,
        _VIZDOOM_MAPPINGS,
        _MINIHACK_MAPPINGS,
        _NETHACK_MAPPINGS,
        _CRAFTER_MAPPINGS,
        _BABAISAI_MAPPINGS,
        _PROCGEN_MAPPINGS,
        _JUMANJI_MAPPINGS,
    ]
    for source in mapping_sources:
        mappings = source.get(game_id)
        if mappings is not None:
            break

    if mappings is None:
        if env_family == EnvironmentFamily.BABYAI:
            mappings = _STANDARD_MINIGRID_ACTIONS
    if mappings is None and action_space is not None:
        try:
            from gymnasium import spaces
            if isinstance(action_space, spaces.Discrete):
                n = action_space.n
                if n <= 4:
                    mappings = _TOY_TEXT_MAPPINGS.get(GameId.FROZEN_LAKE)
                elif n <= 7:
                    mappings = _STANDARD_MINIGRID_ACTIONS
        except ImportError:
            pass

    if not mappings:
        return None

    # Convert to Linux keycodes
    enum_map = _build_qt_key_enum_map()
    result = []
    for mapping in mappings:
        keycodes = []
        for seq in mapping.key_sequences:
            if len(seq) == 0:
                continue
            combo = seq[0]
            # PyQt6 returns QKeyCombination; PyQt5 returns int
            if hasattr(combo, 'key'):
                key_int = combo.key()
                # key() may return an enum with .value or a plain int
                if hasattr(key_int, 'value'):
                    key_int = key_int.value
                else:
                    key_int = int(key_int)
            else:
                key_int = int(combo)
            qt_name = enum_map.get(key_int)
            if qt_name and qt_name in QT_KEY_TO_LINUX:
                keycodes.append(QT_KEY_TO_LINUX[qt_name])
        if keycodes:
            result.append({"keycodes": keycodes, "action": mapping.action})

    return result if result else None


class HumanInputController(QtCore.QObject, LogConstantMixin):
    """Registers keyboard shortcuts and forwards them to the session controller.

    This controller supports two input modes:

    1. **Shortcut-based** (turn-based games): Uses Qt's QShortcut mechanism for
       single-key actions. Each key press immediately triggers an action.

    2. **State-based** (real-time games): Tracks all currently pressed keys and
       computes combined actions (e.g., Up+Right → diagonal movement) on each
       game tick. This is essential for games requiring simultaneous key presses.

    The mode is automatically selected based on the environment family.
    """

    # Signal emitted when a key combination is detected in state-based mode
    # The session controller listens to this to apply actions during idle ticks
    key_action_available = QtCore.Signal(int)  # action index

    # Signal emitted when an individual agent's action is resolved from evdev input.
    # Used by the Operators tab to route per-agent keyboard actions to the
    # parallel multi-agent step system.  (agent_id: str, action_index: int)
    agent_action_selected = QtCore.Signal(str, int)

    def __init__(self, widget: QtWidgets.QWidget, session: SessionController) -> None:
        super().__init__(widget)
        self._logger = _LOGGER
        self._widget = widget
        self._session = session
        self._shortcuts: List[QShortcut] = []
        self._mode_allows_input = True
        self._requested_enabled = True

        # State-based key tracking
        self._pressed_keys: Set[int] = set()
        self._use_state_based_input = False
        self._key_resolver: Optional[KeyCombinationResolver] = None
        self._current_game_id: Optional[GameId] = None

        # Multi-agent keyboard tracking
        self._agent_pressed_keys: Dict[str, Set[int]] = {}  # {agent_id: set of pressed keys}
        self._keyboard_assignments: Dict[int, str] = {}  # {system_id: agent_id}  # OLD: Qt device IDs
        self._num_agents: int = 1
        self._agent_names: List[str] = []  # Actual agent names from environment (e.g., 'player_0')

        # Evdev multi-keyboard support (Linux only)
        self._use_evdev = False
        self._evdev_monitor: Optional[EvdevKeyboardMonitor] = None
        self._evdev_device_to_agent: Dict[str, str] = {}  # {device_path: agent_id}

        # Tetris cursor state (MultiDiscrete [rotation, column])
        self._tetris_rotation: int = 0
        self._tetris_column: int = 0
        self._tetris_num_cols: int = 10

        # Install event filter for state-based input
        self._widget.installEventFilter(self)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Filter key events for state-based input tracking.

        This captures keyPressEvent and keyReleaseEvent to maintain a set of
        currently pressed keys, enabling simultaneous key combination detection.

        For multi-agent environments, routes keys to agents based on keyboard device.
        """
        # If evdev is active, skip Qt keyboard processing entirely
        if self._use_evdev:
            return False  # Let evdev handle all keyboard input

        if not self._use_state_based_input:
            return False  # Let shortcuts handle it

        if not self._mode_allows_input or not self._requested_enabled:
            return False

        event_type = event.type()

        if event_type == QtCore.QEvent.Type.KeyPress:
            key_event = event  # type: QKeyEvent
            if hasattr(key_event, 'isAutoRepeat') and key_event.isAutoRepeat():
                return False  # Ignore OS auto-repeat

            key = key_event.key()

            # Multi-agent: check which keyboard device sent this event
            if self._num_agents > 1 and self._keyboard_assignments:
                device = key_event.device()
                if device is not None:
                    system_id = device.systemId()
                    agent_id = self._keyboard_assignments.get(system_id)
                    if agent_id:
                        # Track key for specific agent
                        if agent_id not in self._agent_pressed_keys:
                            self._agent_pressed_keys[agent_id] = set()
                        if key not in self._agent_pressed_keys[agent_id]:
                            self._agent_pressed_keys[agent_id].add(key)
                            _LOGGER.debug(f"Key {key} pressed by {agent_id} (keyboard {system_id})")
                            self._emit_multi_agent_action()
                        return True

            # Single-agent or no keyboard assignment
            if key not in self._pressed_keys:
                self._pressed_keys.add(key)
                _LOGGER.debug("Key pressed: %s, pressed_keys=%s", key, self._pressed_keys)
                # Emit immediate action for responsive feel
                self._emit_current_action()
            return True  # Consume the event

        if event_type == QtCore.QEvent.Type.KeyRelease:
            key_event = event  # type: QKeyEvent
            if hasattr(key_event, 'isAutoRepeat') and key_event.isAutoRepeat():
                return False  # Ignore OS auto-repeat

            key = key_event.key()

            # Multi-agent: check which keyboard device sent this event
            if self._num_agents > 1 and self._keyboard_assignments:
                device = key_event.device()
                if device is not None:
                    system_id = device.systemId()
                    agent_id = self._keyboard_assignments.get(system_id)
                    if agent_id and agent_id in self._agent_pressed_keys:
                        self._agent_pressed_keys[agent_id].discard(key)
                        _LOGGER.debug(f"Key {key} released by {agent_id} (keyboard {system_id})")
                        return True

            # Single-agent
            self._pressed_keys.discard(key)
            _LOGGER.debug("Key released: %s, pressed_keys=%s", key, self._pressed_keys)
            return True  # Consume the event

        # Handle focus loss - clear all keys
        if event_type == QtCore.QEvent.Type.FocusOut:
            if self._pressed_keys:
                _LOGGER.debug("Focus lost, clearing pressed keys")
                self._pressed_keys.clear()
            if self._agent_pressed_keys:
                _LOGGER.debug("Focus lost, clearing all agent pressed keys")
                self._agent_pressed_keys.clear()

        return False

    def _emit_current_action(self) -> None:
        """Compute and emit action from currently pressed keys."""
        action = self.get_current_action()
        if action is not None:
            self._session.perform_human_action(action, key_label="combo")

    def _get_default_idle_action(self) -> int:
        """Get the default action to use when an agent has no keyboard input.

        For environments without a NOOP action, returns the least disruptive action.

        Returns:
            Default idle action index:
            - mosaic_multigrid v6: 0 (NOOP) - do nothing
            - MeltingPot: 0 (NOOP) - do nothing
            - INI multigrid: 6 (DONE) - no visible effect, agent stays still
        """
        # INI MultiGrid has NO NOOP action - all actions cause movement/rotation
        # Use DONE (action 6) as the idle since it doesn't cause visible changes
        if isinstance(self._key_resolver, INIMultiGridKeyCombinationResolver):
            return 6  # DONE - no visible effect in INI multigrid

        # Other environments use action 0 as idle:
        # - mosaic_multigrid v6: 0 = NOOP (do nothing)
        # - MeltingPot: 0 = NOOP (do nothing)
        return 0

    def _emit_multi_agent_action(self) -> None:
        """Compute and emit actions for all agents based on their keyboard states."""
        actions = self.get_multi_agent_actions()
        if actions:
            # Build detailed logging: show agent -> action
            agent_names = self._agent_names if self._agent_names else [f"agent_{i}" for i in range(self._num_agents)]
            action_details = []
            agents_with_input = []

            for i, action in enumerate(actions):
                agent_id = agent_names[i] if i < len(agent_names) else f"agent_{i}"

                # Check if this agent has active input (has pressed keys)
                has_pressed_keys = bool(self._agent_pressed_keys.get(agent_id))

                if has_pressed_keys:
                    agents_with_input.append(agent_id)
                    action_name = self._get_action_name(action)
                    action_details.append(f"{agent_id}->{action_name}")
                else:
                    # No input - show the actual idle action being sent
                    idle_action_name = self._get_action_name(action)
                    action_details.append(f"{agent_id}->{idle_action_name}(idle)")

            # Log the complete multi-agent action
            if agents_with_input:
                _LOGGER.info(
                    f"Multi-agent actions: {', '.join(action_details)} | "
                    f"Active agents: {', '.join(agents_with_input)}"
                )
            else:
                _LOGGER.debug("Multi-agent actions: all agents waiting for input")

            self._session.perform_human_action(actions, key_label="multi_agent")

    def _get_action_name(self, action: int) -> str:
        """Get human-readable action name for logging.

        Uses the actual Action enums from the respective packages.

        Args:
            action: Action index

        Returns:
            Action name string
        """
        # INI MultiGrid - use the official Action enum
        if isinstance(self._key_resolver, INIMultiGridKeyCombinationResolver):
            try:
                from multigrid.core.actions import Action as INIAction
                return INIAction(action).name.upper()
            except (ImportError, ValueError):
                pass

        # Legacy mosaic_multigrid - use the Actions class
        if isinstance(self._key_resolver, MultiGridKeyCombinationResolver):
            try:
                from mosaic_multigrid.multigrid import Actions as LegacyActions
                # Reverse lookup: find action name by value
                for name in LegacyActions.available:
                    if getattr(LegacyActions, name, None) == action:
                        return name.upper()
            except (ImportError, AttributeError):
                pass

        # MeltingPot action names (no official enum available)
        if isinstance(self._key_resolver, MeltingPotKeyCombinationResolver):
            meltingpot_actions = [
                "NOOP", "FORWARD", "BACKWARD", "STRAFE_LEFT", "STRAFE_RIGHT",
                "TURN_LEFT", "TURN_RIGHT", "INTERACT", "FIRE_1", "FIRE_2", "FIRE_3"
            ]
            if 0 <= action < len(meltingpot_actions):
                return meltingpot_actions[action]

        # Fallback for other resolvers or out-of-range actions
        return f"action_{action}"

    def get_current_action(self) -> Optional[int]:
        """Get the action corresponding to currently pressed keys.

        Returns:
            Action index if keys are pressed and recognized, None otherwise.
        """
        if not self._pressed_keys or self._key_resolver is None:
            return None
        return self._key_resolver.resolve(self._pressed_keys)

    def get_multi_agent_actions(self) -> Optional[List[int]]:
        """Get actions for all agents based on their keyboard states.

        Returns:
            List of actions (one per agent by index), or None if not in multi-agent mode.
            Agents without keyboard input get a default idle action.

        Note:
            For INI multigrid (no NOOP action), uses LEFT (turn left) as the least
            disruptive default since it doesn't move or interact with objects.
        """
        if self._num_agents <= 1 or self._key_resolver is None:
            return None

        # Get the appropriate idle action for the current game type
        default_idle = self._get_default_idle_action()

        # Use agent names for keyboard lookup, but return list by index
        agent_names = self._agent_names if self._agent_names else [f"agent_{i}" for i in range(self._num_agents)]

        actions: List[int] = []
        for agent_id in agent_names:
            pressed_keys = self._agent_pressed_keys.get(agent_id, set())
            if pressed_keys:
                action = self._key_resolver.resolve(pressed_keys)
                # If resolver returns None (no meaningful keys), use default idle
                actions.append(action if action is not None else default_idle)
            else:
                # No keys pressed for this agent = default idle action
                actions.append(default_idle)

        return actions

    def has_keys_pressed(self) -> bool:
        """Return True if any tracked keys are currently pressed."""
        return bool(self._pressed_keys) or bool(self._agent_pressed_keys)

    def clear_pressed_keys(self) -> None:
        """Clear all tracked pressed keys (e.g., on game pause/stop)."""
        self._pressed_keys.clear()
        self._agent_pressed_keys.clear()

    def set_keyboard_assignments(self, assignments: Dict[int, str]) -> None:
        """Update keyboard-to-agent assignments for multi-human gameplay.

        Args:
            assignments: Dict mapping keyboard system_id to agent_id
        """
        self._keyboard_assignments = assignments
        _LOGGER.info(f"Updated keyboard assignments: {assignments}")

    def set_num_agents(self, num_agents: int) -> None:
        """Update the number of agents for multi-agent environments.

        Args:
            num_agents: Number of agents in the environment
        """
        self._num_agents = num_agents
        _LOGGER.info(f"Set num_agents to {num_agents}")

    def set_agent_names(self, agent_names: List[str]) -> None:
        """Update the agent names for multi-agent environments.

        This enables proper mapping between keyboard assignments and environment agents.
        Different environments use different naming conventions (e.g., 'player_0' for
        MeltingPot, 'agent_0' for MultiGrid).

        Args:
            agent_names: List of agent identifiers as used by the environment.
        """
        self._agent_names = agent_names
        _LOGGER.info(f"Set agent_names to {agent_names}")

    # =========================================================================
    # Evdev Multi-Keyboard Support (Linux only)
    # =========================================================================

    def setup_evdev_keyboards(self, device_assignments: Dict[str, str]) -> bool:
        """Setup evdev keyboard monitoring for multi-keyboard support.

        This enables true multi-keyboard support on Linux by using evdev to read
        keyboard events directly from /dev/input devices, bypassing X11's device merging.

        Args:
            device_assignments: Dict mapping device_path → agent_id
                Example: {
                    "/dev/input/by-path/...usb-0:5.1:1.0-event-kbd": "agent_0",
                    "/dev/input/by-path/...usb-0:5.2:1.0-event-kbd": "agent_1",
                }

        Returns:
            True if evdev monitoring was successfully started
        """
        if not _HAS_EVDEV:
            _LOGGER.warning("Evdev not available (not Linux or module not installed)")
            return False

        if not device_assignments:
            _LOGGER.warning("No keyboard assignments provided for evdev")
            return False

        # IMPORTANT: Stop any existing monitor before creating a new one
        # This prevents duplicate monitors from running concurrently
        if self._evdev_monitor is not None:
            _LOGGER.info("Stopping existing evdev monitor before re-setup")
            self.stop_evdev_monitoring()
            self._evdev_device_to_agent.clear()
            self._agent_pressed_keys.clear()

        _LOGGER.info(f"Setting up evdev keyboard monitoring with {len(device_assignments)} devices")

        try:
            # Create evdev monitor
            self._evdev_monitor = EvdevKeyboardMonitor()

            # Connect signals
            self._evdev_monitor.key_pressed.connect(self._on_evdev_key_pressed)
            self._evdev_monitor.key_released.connect(self._on_evdev_key_released)
            self._evdev_monitor.error_occurred.connect(self._on_evdev_error)

            # Discover keyboards
            keyboards = self._evdev_monitor.discover_keyboards()
            _LOGGER.info(f"Discovered {len(keyboards)} keyboards via evdev")

            # Add assigned keyboards
            added_count = 0
            for kbd in keyboards:
                agent_id = device_assignments.get(kbd.device_path)
                if agent_id:
                    if self._evdev_monitor.add_device(kbd):
                        self._evdev_device_to_agent[kbd.device_path] = agent_id
                        _LOGGER.info(f"Added {kbd.name} for {agent_id}")
                        added_count += 1
                    else:
                        _LOGGER.error(f"Failed to add device {kbd.device_path}")

            if added_count == 0:
                _LOGGER.error("No keyboards could be added (check permissions)")
                return False

            # Start monitoring
            self._evdev_monitor.start_monitoring()
            self._use_evdev = True

            # Disable Qt shortcuts when evdev is active
            self._update_shortcuts_enabled()

            _LOGGER.info(f"Evdev monitoring started for {added_count} keyboard(s)")
            return True

        except Exception as e:
            _LOGGER.error(f"Failed to setup evdev keyboards: {e}", exc_info=True)
            return False

    def stop_evdev_monitoring(self) -> None:
        """Stop evdev keyboard monitoring."""
        if self._evdev_monitor:
            _LOGGER.info("Stopping evdev monitoring")
            self._evdev_monitor.stop_monitoring()
            self._use_evdev = False

            # Re-enable Qt shortcuts when evdev is stopped
            self._update_shortcuts_enabled()

    def _on_evdev_key_pressed(self, device_path: str, keycode: int, timestamp: int) -> None:
        """Handle key press from evdev monitor.

        Args:
            device_path: Path to the keyboard device
            keycode: Linux input keycode
            timestamp: Event timestamp in milliseconds
        """
        if not self._mode_allows_input or not self._requested_enabled:
            return

        # Get agent ID for this device
        agent_id = self._evdev_device_to_agent.get(device_path)
        if not agent_id:
            _LOGGER.debug(f"evdev: Key from unassigned device: {device_path}")
            return

        # Convert Linux keycode to Qt key
        qt_key_enum = linux_keycode_to_qt_key(keycode)
        if qt_key_enum == Qt.Key.Key_unknown:
            _LOGGER.debug(f"evdev: Unknown keycode: {keycode}")
            return

        # CRITICAL: Convert Qt.Key enum to int for consistent comparison
        # The key constants (_KEY_W, _KEY_A, etc.) are integers, so we must
        # store integers in pressed_keys for set intersection to work correctly
        qt_key = int(qt_key_enum)

        # Get device name for logging
        device_name = "Unknown"
        if self._evdev_monitor:
            for kbd in self._evdev_monitor.get_monitored_devices():
                if kbd.device_path == device_path:
                    device_name = kbd.name
                    break

        # Track key for this agent
        if agent_id not in self._agent_pressed_keys:
            self._agent_pressed_keys[agent_id] = set()

        if qt_key not in self._agent_pressed_keys[agent_id]:
            self._agent_pressed_keys[agent_id].add(qt_key)

            # Enhanced logging: keyboard name -> agent -> key
            key_name = Qt.Key(qt_key).name if hasattr(Qt.Key(qt_key), 'name') else f"key_{qt_key}"
            self.log_constant(
                LOG_EVDEV_KEY_PRESSED,
                message=f"[{device_name}] ({device_path}) → {agent_id} → {key_name}",
                extra={
                    "device_path": device_path,
                    "device_name": device_name,
                    "agent_id": agent_id,
                    "key_name": key_name,
                    "keycode": keycode,
                },
            )

            self._emit_multi_agent_action()

            # Emit per-agent action signal for the Operators tab integration.
            # Resolves the action for just this agent based on their pressed keys.
            if self._key_resolver is not None:
                pressed = self._agent_pressed_keys.get(agent_id, set())
                resolved = self._key_resolver.resolve(pressed)
                if resolved is not None:
                    self.agent_action_selected.emit(agent_id, resolved)

    def _on_evdev_key_released(self, device_path: str, keycode: int, timestamp: int) -> None:
        """Handle key release from evdev monitor.

        Args:
            device_path: Path to the keyboard device
            keycode: Linux input keycode
            timestamp: Event timestamp in milliseconds
        """
        # Get agent ID for this device
        agent_id = self._evdev_device_to_agent.get(device_path)
        if not agent_id:
            return

        # Convert Linux keycode to Qt key
        qt_key_enum = linux_keycode_to_qt_key(keycode)
        if qt_key_enum == Qt.Key.Key_unknown:
            return

        # Convert Qt.Key enum to int for consistent comparison (matches key press handler)
        qt_key = int(qt_key_enum)

        # Remove key from agent's pressed keys
        if agent_id in self._agent_pressed_keys:
            self._agent_pressed_keys[agent_id].discard(qt_key)

            # Get device name for logging
            device_name = "Unknown"
            if self._evdev_monitor:
                for kbd in self._evdev_monitor.get_monitored_devices():
                    if kbd.device_path == device_path:
                        device_name = kbd.name
                        break

            key_name = Qt.Key(qt_key).name if hasattr(Qt.Key(qt_key), 'name') else f"key_{qt_key}"
            self.log_constant(
                LOG_EVDEV_KEY_RELEASED,
                message=f"[{device_name}] ({device_path}) → {agent_id} → {key_name}",
                extra={
                    "device_path": device_path,
                    "device_name": device_name,
                    "agent_id": agent_id,
                    "key_name": key_name,
                    "keycode": keycode,
                },
            )

    def _on_evdev_error(self, error_msg: str) -> None:
        """Handle error from evdev monitor.

        Args:
            error_msg: Error message
        """
        _LOGGER.error(f"Evdev error: {error_msg}")

    def is_using_evdev(self) -> bool:
        """Check if evdev monitoring is active.

        Returns:
            True if evdev monitoring is active
        """
        return self._use_evdev

    def get_evdev_devices(self) -> List[KeyboardDevice]:
        """Get list of available evdev keyboard devices.

        Returns:
            List of KeyboardDevice objects, or empty list if evdev not available
        """
        if not _HAS_EVDEV:
            return []

        try:
            monitor = EvdevKeyboardMonitor()
            return monitor.discover_keyboards()
        except Exception as e:
            _LOGGER.error(f"Failed to discover evdev keyboards: {e}")
            return []

    def set_evdev_keyboard_assignments(self, assignments: Dict[str, str]) -> None:
        """Update evdev keyboard-to-agent assignments.

        Args:
            assignments: Dict mapping device_path to agent_id
        """
        if not self._use_evdev or not self._evdev_monitor:
            # Start evdev monitoring with these assignments
            self.setup_evdev_keyboards(assignments)
        else:
            # Update existing assignments
            self._evdev_device_to_agent = assignments
            _LOGGER.info(f"Updated evdev keyboard assignments: {assignments}")

    def clear_evdev_keyboards(self) -> None:
        """Clear all evdev keyboard tracking and stop monitoring."""
        self.stop_evdev_monitoring()
        self._evdev_device_to_agent.clear()
        self._agent_pressed_keys.clear()
        _LOGGER.info("Cleared evdev keyboard tracking")

    def configure(
        self,
        game_id: GameId | None,
        action_space: object | None,
        *,
        overrides: Optional[Dict[str, object]] = None,
    ) -> None:
        """Configure input handling for a specific game.

        Args:
            game_id: The game to configure for, or None to disable.
            action_space: The game's action space.
            overrides: Optional dict of game config overrides, may include 'input_mode'.

        The input mode is determined by the 'input_mode' key in overrides:
        - 'state_based': Track all pressed keys, compute combined actions (for diagonals)
        - 'shortcut_based' (default): Each key triggers immediate action
        """
        self._clear_shortcuts()
        self._pressed_keys.clear()
        self._current_game_id = game_id
        self._key_resolver = None
        self._use_state_based_input = False

        if game_id is None or action_space is None:
            return

        # Get input mode from game configuration (user's choice)
        # Default to shortcut_based for backward compatibility
        input_mode = InputMode.SHORTCUT_BASED.value
        if overrides:
            input_mode = overrides.get("input_mode", InputMode.SHORTCUT_BASED.value)

        # CRITICAL: Force state-based mode for environments that require it
        # Multi-agent: shortcut-based mode conflicts with evdev multi-keyboard monitoring
        # Griddly: ALL Griddly games need state-based mode to use GriddlyKeyCombinationResolver
        #          (no shortcuts are registered for Griddly in shortcut-based mode)
        from gym_gui.core.enums import CONTINUOUS_GRIDDLY_GAMES, ENVIRONMENT_FAMILY_BY_GAME, EnvironmentFamily
        env_family = ENVIRONMENT_FAMILY_BY_GAME.get(game_id)
        is_multi_agent = env_family in (
            EnvironmentFamily.MOSAIC_MULTIGRID,
            EnvironmentFamily.INI_MULTIGRID,
            EnvironmentFamily.MELTINGPOT,
            EnvironmentFamily.OVERCOOKED,
            EnvironmentFamily.RWARE,
        )
        is_griddly = env_family == EnvironmentFamily.GRIDDLY
        requires_state_based = is_multi_agent or is_griddly

        if requires_state_based and input_mode != InputMode.STATE_BASED.value:
            reason = "multi-agent" if is_multi_agent else "Griddly (needs GriddlyKeyCombinationResolver)"
            _LOGGER.warning(
                "Environment %s (%s) REQUIRES state-based input mode. "
                "Forcing input_mode to 'state_based'.",
                game_id.value if hasattr(game_id, 'value') else game_id,
                reason,
            )
            input_mode = InputMode.STATE_BASED.value
            # Update overrides to reflect the forced mode
            if overrides is not None:
                overrides["input_mode"] = InputMode.STATE_BASED.value

        use_state_based = (input_mode == InputMode.STATE_BASED.value)

        self.log_constant(
            LOG_INPUT_MODE_CONFIGURED,
            extra={
                "game_id": game_id.value if hasattr(game_id, 'value') else str(game_id),
                "input_mode": input_mode,
                "forced_multi_agent": is_multi_agent,
            },
        )

        if use_state_based:
            # Try to get a key combination resolver for this game
            # Pass action_space to enable action-space-aware resolvers (e.g., MeltingPot)
            space_arg = action_space if isinstance(action_space, spaces.Space) else None
            self._key_resolver = get_key_combination_resolver(game_id, space_arg)
            if self._key_resolver is not None:
                self._use_state_based_input = True
                self.log_constant(
                    LOG_KEY_RESOLVER_INITIALIZED,
                    extra={
                        "game_id": game_id.value if hasattr(game_id, 'value') else str(game_id),
                        "resolver_type": type(self._key_resolver).__name__,
                    },
                )
                return  # Don't set up shortcuts for state-based games
            else:
                self.log_constant(
                    LOG_KEY_RESOLVER_UNAVAILABLE,
                    extra={
                        "game_id": game_id.value if hasattr(game_id, 'value') else str(game_id),
                    },
                )

        # Tetris uses MultiDiscrete([4, num_cols]) — cursor-based input
        if game_id == GameId.JUMANJI_TETRIS and isinstance(action_space, spaces.MultiDiscrete):
            self._setup_tetris_shortcuts(action_space)
            return

        # Fall back to shortcut-based input for turn-based games
        try:
            mappings = _TOY_TEXT_MAPPINGS.get(game_id)
            if mappings is None:
                mappings = _MINIG_GRID_MAPPINGS.get(game_id)
            if mappings is None:
                mappings = _MULTIGRID_MAPPINGS.get(game_id)
            if mappings is None:
                mappings = _BOX_2D_MAPPINGS.get(game_id)
            if mappings is None:
                mappings = _ALE_MAPPINGS.get(game_id)
            if mappings is None:
                mappings = _VIZDOOM_MAPPINGS.get(game_id)
            if mappings is None:
                mappings = _MINIHACK_MAPPINGS.get(game_id)
            if mappings is None:
                mappings = _NETHACK_MAPPINGS.get(game_id)
            if mappings is None:
                mappings = _CRAFTER_MAPPINGS.get(game_id)
            if mappings is None:
                mappings = _BABAISAI_MAPPINGS.get(game_id)
            if mappings is None:
                mappings = _PROCGEN_MAPPINGS.get(game_id)
            if mappings is None:
                mappings = _JUMANJI_MAPPINGS.get(game_id)
            if mappings is None:
                # BabyAI uses the exact same 7-action space as MiniGrid
                # (0=left, 1=right, 2=forward, 3=pickup, 4=drop, 5=toggle, 6=done)
                if env_family == EnvironmentFamily.BABYAI:
                    mappings = _STANDARD_MINIGRID_ACTIONS
            if mappings is None and isinstance(action_space, spaces.Discrete):
                mappings = self._fallback_mappings(action_space)

            if not mappings:
                return

            for mapping in mappings:
                for sequence in mapping.key_sequences:
                    shortcut = QShortcut(sequence, self._widget)
                    shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
                    shortcut_label = sequence.toString() or repr(sequence)
                    shortcut.activated.connect(self._make_activation(mapping.action, shortcut_label))
                    self._shortcuts.append(shortcut)
            self._update_shortcuts_enabled()
        except Exception as exc:  # pragma: no cover - defensive
            self.log_constant(
                LOG_INPUT_CONTROLLER_ERROR,
                exc_info=exc,
                extra={
                    "game_id": game_id.value if game_id else "unknown",
                    "space_type": type(action_space).__name__ if action_space is not None else "None",
                },
            )
            self._clear_shortcuts()

    def is_state_based(self) -> bool:
        """Return True if using state-based input for current game."""
        return self._use_state_based_input

    def set_enabled(self, enabled: bool) -> None:
        self._requested_enabled = enabled
        self._update_shortcuts_enabled()
        if not enabled:
            self._pressed_keys.clear()

    def _make_activation(self, action: int, shortcut_label: str) -> Callable[[], None]:
        def trigger() -> None:
            _LOGGER.debug("Shortcut activated key='%s' action=%s", shortcut_label, action)
            self._session.perform_human_action(action, key_label=shortcut_label)

        return trigger

    def _clear_shortcuts(self) -> None:
        while self._shortcuts:
            shortcut = self._shortcuts.pop()
            shortcut.deleteLater()

    # ── Tetris cursor-based input (MultiDiscrete [rotation, column]) ───

    def _setup_tetris_shortcuts(self, action_space: spaces.MultiDiscrete) -> None:
        """Set up stateful cursor shortcuts for Tetris.

        Jumanji Tetris uses MultiDiscrete([4, num_cols]) — each action is a
        [rotation, column] pair submitted in one shot.  Arrow/WASD keys adjust
        the cursor, Space/Enter places the piece.
        """
        self._tetris_num_cols = int(action_space.nvec[1])
        self._tetris_rotation = 0
        self._tetris_column = self._tetris_num_cols // 2

        bindings: List[Tuple[Tuple[str, ...], Callable[[], None]]] = [
            (("Key_Left", "Key_A"), self._tetris_move_left),
            (("Key_Right", "Key_D"), self._tetris_move_right),
            (("Key_Up", "Key_W"), self._tetris_rotate_cw),
            (("Key_Down", "Key_S"), self._tetris_rotate_ccw),
            (("Key_Space", "Key_Return"), self._tetris_place),
        ]
        for key_names, callback in bindings:
            for key_name in key_names:
                seq = QKeySequence(_qt_key(key_name))
                shortcut = QShortcut(seq, self._widget)
                shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
                shortcut.activated.connect(callback)
                self._shortcuts.append(shortcut)
        self._update_shortcuts_enabled()
        self._tetris_status_update()

    def _tetris_move_left(self) -> None:
        self._tetris_column = max(0, self._tetris_column - 1)
        self._tetris_status_update()

    def _tetris_move_right(self) -> None:
        self._tetris_column = min(self._tetris_num_cols - 1, self._tetris_column + 1)
        self._tetris_status_update()

    def _tetris_rotate_cw(self) -> None:
        self._tetris_rotation = (self._tetris_rotation + 1) % 4
        self._tetris_status_update()

    def _tetris_rotate_ccw(self) -> None:
        self._tetris_rotation = (self._tetris_rotation - 1) % 4
        self._tetris_status_update()

    def _tetris_place(self) -> None:
        action = [self._tetris_rotation, self._tetris_column]
        label = f"rot={self._tetris_rotation} col={self._tetris_column}"
        _LOGGER.debug("Tetris placement: %s", label)
        self._session.perform_human_action(action, key_label=label)
        # Reset cursor for next piece
        self._tetris_rotation = 0
        self._tetris_column = self._tetris_num_cols // 2
        self._tetris_status_update()

    def _tetris_status_update(self) -> None:
        rot_labels = ["0deg", "90deg", "180deg", "270deg"]
        msg = (
            f"Tetris: Column {self._tetris_column}/{self._tetris_num_cols - 1}, "
            f"Rotation {rot_labels[self._tetris_rotation]} "
            f"[Left/Right=column, Up/Down=rotate, Space=place]"
        )
        self._session.status_message.emit(msg)

    # ── End Tetris cursor-based input ──────────────────────────────────

    @staticmethod
    def _fallback_mappings(action_space: spaces.Discrete) -> Tuple[ShortcutMapping, ...]:
        sequences: List[ShortcutMapping] = []
        base_mappings: List[Tuple[Iterable[str], int]] = [
            (("Key_Left", "Key_A"), 0),
            (("Key_Down", "Key_S"), 1),
            (("Key_Right", "Key_D"), 2),
            (("Key_Up", "Key_W"), 3),
            (("Key_Space",), 4),
            (("Key_E",), 5),
            (("Key_Q",), 6),
        ]
        for keys, action in base_mappings:
            if action >= action_space.n:
                continue
            sequences.append(_mapping(keys, action))
        return tuple(sequences)

    def update_for_mode(self, mode: ControlMode) -> None:
        self._mode_allows_input = mode in {
            ControlMode.HUMAN_ONLY,
            ControlMode.HYBRID_TURN_BASED,
            ControlMode.HYBRID_HUMAN_AGENT,
            ControlMode.MULTI_AGENT_COOP,
            ControlMode.MULTI_AGENT_COMPETITIVE,
        }
        self._update_shortcuts_enabled()
        if not self._mode_allows_input:
            self._pressed_keys.clear()

    def _update_shortcuts_enabled(self) -> None:
        """Enable or disable all shortcuts based on current state."""
        # Disable Qt shortcuts when evdev is active (evdev handles input directly)
        if self._use_evdev:
            enabled = False
            _LOGGER.debug("Shortcuts disabled (evdev mode active)")
        else:
            enabled = self._mode_allows_input and self._requested_enabled
            _LOGGER.debug("Shortcuts enabled=%s (mode_allows=%s, requested=%s)",
                          enabled, self._mode_allows_input, self._requested_enabled)

        for shortcut in self._shortcuts:
            shortcut.setEnabled(enabled)


__all__ = [
    "HumanInputController",
    "get_vizdoom_mouse_turn_actions",
    "get_key_combination_resolver",
    "KeyCombinationResolver",
    "MiniGridKeyCombinationResolver",
    "MultiGridKeyCombinationResolver",
    "INIMultiGridKeyCombinationResolver",
    "MeltingPotKeyCombinationResolver",
    "ProcgenKeyCombinationResolver",
    "AleKeyCombinationResolver",
    "LunarLanderKeyCombinationResolver",
    "CarRacingKeyCombinationResolver",
    "BipedalWalkerKeyCombinationResolver",
    "Box2DKeyCombinationResolver",  # Backwards compatibility alias
]
