"""Keyboard mapping tables and key-action translation for human control.

Single source of truth for "which keys do what in which game."

1. **ShortcutMapping tables** (``_*_MAPPINGS`` dicts): per-game key bindings.
   Used by the GUI display and converted to Linux keycodes for subprocess workers.
2. **build_key_action_map_for_game()**: converts ShortcutMapping entries into
   Linux keycode maps that HumanKeyboardRuntime subprocess workers understand.
3. **HumanInputController**: thin config holder. Does NOT read keyboard input.
   All input goes through worker subprocesses via KeyboardWorkerBridge.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from qtpy import QtCore, QtWidgets
from qtpy.QtGui import QKeySequence

from gym_gui.controllers.session import SessionController
from gym_gui.core.enums import ControlMode, EnvironmentFamily, GameId
from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_INPUT_MODE_CONFIGURED,
)

_LOGGER = logging.getLogger(__name__)


# =============================================================================
# ShortcutMapping Tables (single source of truth for key bindings)
# =============================================================================
@dataclass(frozen=True)
class ShortcutMapping:
    key_sequences: Tuple[QKeySequence, ...]
    action: int


def _qt_key(name: str) -> int:
    key_enum = getattr(QtCore.Qt, "Key", None)
    if key_enum is not None and hasattr(key_enum, name):
        return getattr(key_enum, name)
    legacy = getattr(QtCore.Qt, name, None)
    if legacy is None:
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

# =============================================================================
# Qt-to-Linux keycode conversion for subprocess workers
# =============================================================================
_QT_KEY_ENUM_TO_NAME: Dict[int, str] = {}


def _build_qt_key_enum_map() -> Dict[int, str]:
    """Build reverse mapping from Qt key int value to key name string."""
    if _QT_KEY_ENUM_TO_NAME:
        return _QT_KEY_ENUM_TO_NAME
    key_names = [
        "Key_0", "Key_1", "Key_2", "Key_3", "Key_4",
        "Key_5", "Key_6", "Key_7", "Key_8", "Key_9",
        "Key_A", "Key_B", "Key_C", "Key_D", "Key_E", "Key_F",
        "Key_G", "Key_H", "Key_I", "Key_J", "Key_K", "Key_L",
        "Key_M", "Key_N", "Key_O", "Key_P", "Key_Q", "Key_R",
        "Key_S", "Key_T", "Key_U", "Key_V", "Key_W", "Key_X",
        "Key_Y", "Key_Z",
        "Key_Escape", "Key_Tab", "Key_Backspace", "Key_Return",
        "Key_Enter", "Key_Insert", "Key_Delete", "Key_Home",
        "Key_End", "Key_Shift", "Key_Control", "Key_Alt",
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
    """Convert Qt ShortcutMapping entries to a portable Linux keycode map.

    This is the bridge between the mapping tables defined above and the
    HumanKeyboardRuntime subprocess workers. Each entry becomes a
    ``{"keycodes": [int, ...], "action": int}`` dict that the worker's
    DynamicKeycodeResolver can use directly.

    Args:
        game_id: The GameId enum value.
        env_family: Optional EnvironmentFamily for BabyAI fallback.
        action_space: Optional Gymnasium action space for generic fallback.

    Returns:
        A list of keycode-to-action entries, or None if no mappings found.
    """
    from gym_gui.config.paths import HUMAN_WORKER_PKG_DIR
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
            from gymnasium import spaces as gym_spaces
            if isinstance(action_space, gym_spaces.Discrete):
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
            if hasattr(combo, 'key'):
                key_int = combo.key()
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


# =============================================================================
# HumanInputController (thin config holder, no input reading)
# =============================================================================
class HumanInputController(QtCore.QObject, LogConstantMixin):
    """Configuration holder for human input. Does NOT read keyboard input.

    All keyboard/mouse input goes through HumanKeyboardRuntime subprocesses
    managed by KeyboardWorkerBridge. This class exists to:
    - Store which game is loaded and its action space
    - Store agent count/names for multi-agent environments
    - Provide stub methods so main_window.py callers don't crash
    """

    # Kept for AEC multi-agent signal wiring in main_window.py
    agent_action_selected = QtCore.Signal(str, int)

    def __init__(self, widget: QtWidgets.QWidget, session: SessionController) -> None:
        super().__init__(widget)
        self._logger = _LOGGER
        self._session = session
        self._current_game_id: Optional[GameId] = None
        self._num_agents: int = 1
        self._agent_names: List[str] = []

    def configure(
        self,
        game_id: GameId | None,
        action_space: object | None,
        *,
        overrides: Optional[Dict[str, object]] = None,
    ) -> None:
        """Store game configuration. No shortcuts are created."""
        self._current_game_id = game_id

        if game_id is not None:
            self.log_constant(
                LOG_INPUT_MODE_CONFIGURED,
                extra={
                    "game_id": game_id.value if hasattr(game_id, 'value') else str(game_id),
                    "input_mode": "subprocess",
                    "forced_multi_agent": False,
                },
            )

    def update_for_mode(self, mode: ControlMode) -> None:
        """No-op. Subprocess bridge handles mode transitions."""
        pass

    def set_num_agents(self, num_agents: int) -> None:
        self._num_agents = num_agents

    def set_agent_names(self, agent_names: List[str]) -> None:
        self._agent_names = agent_names

    def set_enabled(self, enabled: bool) -> None:
        """No-op. Subprocess bridge handles enable/disable."""
        pass

    def stop_evdev_monitoring(self) -> None:
        """No-op. Legacy evdev monitor removed."""
        pass

    def is_state_based(self) -> bool:
        """Always False. Subprocess workers handle input mode."""
        return False

    def get_current_action(self) -> Optional[int]:
        """Always None. Subprocess workers handle action resolution."""
        return None


__all__ = [
    "HumanInputController",
    "build_key_action_map_for_game",
    "get_vizdoom_mouse_turn_actions",
    # Mapping tables (imported by operator_render_container.py)
    "_TOY_TEXT_MAPPINGS",
    "_MINIG_GRID_MAPPINGS",
    "_MULTIGRID_MAPPINGS",
    "_VIZDOOM_MAPPINGS",
    "_MINIHACK_MAPPINGS",
    "_NETHACK_MAPPINGS",
    "_CRAFTER_MAPPINGS",
    "_BABAISAI_MAPPINGS",
    "_PROCGEN_MAPPINGS",
    "_ALE_MAPPINGS",
    "_BOX_2D_MAPPINGS",
    "_JUMANJI_MAPPINGS",
]
