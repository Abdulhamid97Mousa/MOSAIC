from __future__ import annotations

import logging
from typing import Any, Optional, Set

from gym_gui.controllers.input_utils import (
    _KEY_E,
    _KEY_F,
    _KEY_RETURN,
    _KEY_SPACE,
    _KEYS_DOWN,
    _KEYS_LEFT,
    _KEYS_RIGHT,
    _KEYS_UP,
    KeyCombinationResolver,
)

_LOGGER = logging.getLogger(__name__)


def _build_action_index(action_space: Any) -> dict[str, int]:
    """Build a command-name → index lookup from a MalmoEnv ActionSpace.

    The ActionSpace has an ``actions`` list like
    ``['move 1', 'move -1', 'turn 1', 'turn -1', 'use 1', 'use 0', 'noop 1']``.
    We map the command string to its index so the resolver never sends an
    out-of-range action.
    """
    actions = getattr(action_space, "actions", [])
    return {cmd: idx for idx, cmd in enumerate(actions)}


class MalmoEnvKeyCombinationResolver(KeyCombinationResolver):
    """Dynamic key resolver for MalmoEnv (Minecraft).

    Reads the actual action space at construction time so key bindings
    always map to valid action indices for the current mission.
    """

    def __init__(self, *, action_space: object = None) -> None:
        self._action_space = action_space
        self._index = _build_action_index(action_space) if action_space else {}
        if self._index:
            _LOGGER.info(
                "MalmoEnvKeyCombinationResolver action index: %s", self._index,
            )
        else:
            _LOGGER.warning(
                "MalmoEnvKeyCombinationResolver: no action_space provided, "
                "key input will be disabled",
            )

    def _action(self, command: str) -> Optional[int]:
        """Look up a Malmo command string, return its index or None."""
        return self._index.get(command)

    def resolve(self, pressed_keys: Set[int]) -> Optional[int]:
        is_up = any(k in pressed_keys for k in _KEYS_UP)
        is_down = any(k in pressed_keys for k in _KEYS_DOWN)
        is_left = any(k in pressed_keys for k in _KEYS_LEFT)
        is_right = any(k in pressed_keys for k in _KEYS_RIGHT)
        is_jump = _KEY_SPACE in pressed_keys
        is_attack = _KEY_RETURN in pressed_keys
        is_use = _KEY_E in pressed_keys or _KEY_F in pressed_keys

        # Priority: combat > utility > movement
        if is_attack:
            return self._action("attack 1")
        if is_jump:
            return self._action("jump 1")
        if is_use:
            return self._action("use 1")

        if is_up:
            return self._action("move 1")
        if is_down:
            return self._action("move -1")
        if is_left:
            return self._action("turn -1")
        if is_right:
            return self._action("turn 1")

        return None
