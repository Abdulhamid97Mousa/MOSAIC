from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set, Tuple

from gym_gui.controllers.input_utils import (
    _KEY_1,
    _KEY_2,
    _KEY_3,
    _KEY_A,
    _KEY_C,
    _KEY_CONTROL,
    _KEY_D,
    _KEY_DOWN,
    _KEY_E,
    _KEY_F,
    _KEY_G,
    _KEY_H,
    _KEY_LEFT,
    _KEY_Q,
    _KEY_RETURN,
    _KEY_RIGHT,
    _KEY_S,
    _KEY_SPACE,
    _KEY_UP,
    _KEY_W,
    _KEY_X,
    _KEY_Z,
    _KEYS_DOWN,
    _KEYS_LEFT,
    _KEYS_RIGHT,
    _KEYS_UP,
    KeyCombinationResolver,
)
from gym_gui.core.enums import GameId

_LOGGER = logging.getLogger(__name__)


# Map string names (from original mappings) to integer key codes
_QT_KEY_MAP = {
    "Key_Space": _KEY_SPACE,
    "Key_Control": _KEY_CONTROL,
    "Key_Left": _KEY_LEFT,
    "Key_A": _KEY_A,
    "Key_Right": _KEY_RIGHT,
    "Key_D": _KEY_D,
    "Key_Up": _KEY_UP,
    "Key_W": _KEY_W,
    "Key_Down": _KEY_DOWN,
    "Key_S": _KEY_S,
    "Key_Q": _KEY_Q,
    "Key_E": _KEY_E,
    "Key_Return": _KEY_RETURN,
}


@dataclass
class VizDoomKeyMapping:
    keys: Set[int]
    action: int


def _mapping(names: Iterable[str], action: int) -> VizDoomKeyMapping:
    """Helper to create a mapping from key names to action index."""
    key_codes = set()
    for name in names:
        code = _QT_KEY_MAP.get(name)
        if code is not None:
            key_codes.add(code)
    return VizDoomKeyMapping(key_codes, action)


# =============================================================================
# ViZDoom Keyboard Mappings
# =============================================================================
# These mappings define the keyboard controls for each ViZDoom scenario.
# Used by ViZDoomKeyCombinationResolver.
#
# Original comments preserved for documentation value:
_VIZDOOM_KEYBOARD_MAPPINGS: Dict[GameId, Tuple[VizDoomKeyMapping, ...]] = {
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
    # HealthGathering: TURN_LEFT(1), TURN_RIGHT(2)
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


# =============================================================================
# ViZDoom Mouse Turn Actions
# =============================================================================
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


class ViZDoomKeyCombinationResolver(KeyCombinationResolver):
    """
    Key resolver for ViZDoom (Doom).
    Maps WASD/Arrows to scenario-specific Doom actions using _VIZDOOM_KEYBOARD_MAPPINGS.
    """

    def __init__(self, game_id: GameId):
        self._game_id = game_id
        # Default to empty tuple if game not found
        self._mappings = _VIZDOOM_KEYBOARD_MAPPINGS.get(game_id, ())

    def resolve(self, pressed_keys: Set[int]) -> Optional[int]:
        if not self._mappings:
            return None

        # Iterate through mappings to find a matching action.
        # Priority is determined by order in the tuple (first match wins).
        for mapping in self._mappings:
            # Check if ANY key in the mapping is pressed
            if not mapping.keys.isdisjoint(pressed_keys):
                return mapping.action

        return None


def get_vizdoom_mouse_turn_actions(game_id: GameId) -> Optional[Tuple[int, int]]:
    """Return (turn_left_action, turn_right_action) for a ViZDoom game, or None if not ViZDoom."""
    return _VIZDOOM_MOUSE_TURN_ACTIONS.get(game_id)
