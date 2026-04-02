from __future__ import annotations

import logging
from typing import Optional, Set

from qtpy.QtCore import Qt

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
try:
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
    _KEY_F = _get_qt_key("Key_F")
    _KEY_G = _get_qt_key("Key_G")
    _KEY_H = _get_qt_key("Key_H")
    _KEY_RETURN = _get_qt_key("Key_Return")
    _KEY_Z = _get_qt_key("Key_Z")
    _KEY_C = _get_qt_key("Key_C")
    _KEY_X = _get_qt_key("Key_X")
    _KEY_1 = _get_qt_key("Key_1")
    _KEY_2 = _get_qt_key("Key_2")
    _KEY_3 = _get_qt_key("Key_3")
    _KEY_CONTROL = _get_qt_key("Key_Control")
except AttributeError:
    _LOGGER.error("Failed to load Qt keys in input_utils.py. Is a QApplication running or Qt installed?")
    # Fallback or re-raise
    raise

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
