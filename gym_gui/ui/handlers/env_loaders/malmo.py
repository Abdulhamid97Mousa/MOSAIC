"""MalmoEnv loader for FPS-style mouse capture.

Mouse horizontal movement is translated to Malmo "turn" actions resolved from
the mission-specific action space.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gym_gui.controllers.session import SessionController
    from gym_gui.ui.widgets.render_tabs import RenderTabs

from gym_gui.core.enums import ENVIRONMENT_FAMILY_BY_GAME, EnvironmentFamily, GameId

_LOG = logging.getLogger(__name__)

# Fallback indices used by most missions in this repository:
# 0=move 1, 1=move -1, 2=turn 1, 3=turn -1.
_FALLBACK_TURN_LEFT_ACTION = 3
_FALLBACK_TURN_RIGHT_ACTION = 2


class MalmoEnvLoader:
    """Loader for MalmoEnv (Microsoft Malmo / Minecraft) environment setup.

    Configures FPS-style mouse capture so that horizontal mouse movement steers
    the Minecraft player view, exactly as physical Minecraft mouse-look works.

    Args:
        render_tabs: The render tabs widget for configuring mouse capture.
    """

    def __init__(self, render_tabs: "RenderTabs") -> None:
        self._render_tabs = render_tabs

    @staticmethod
    def _resolve_turn_actions(session: "SessionController") -> tuple[int, int]:
        """Return (turn_left_action, turn_right_action) for current mission."""
        adapter = getattr(session, "_adapter", None)
        if adapter is None:
            return _FALLBACK_TURN_LEFT_ACTION, _FALLBACK_TURN_RIGHT_ACTION

        try:
            action_space = adapter.action_space
        except Exception:
            return _FALLBACK_TURN_LEFT_ACTION, _FALLBACK_TURN_RIGHT_ACTION

        actions = getattr(action_space, "actions", None)
        if not isinstance(actions, (list, tuple)):
            return _FALLBACK_TURN_LEFT_ACTION, _FALLBACK_TURN_RIGHT_ACTION

        action_to_index = {str(command): idx for idx, command in enumerate(actions)}
        left = action_to_index.get("turn -1")
        right = action_to_index.get("turn 1")
        if left is None or right is None:
            return _FALLBACK_TURN_LEFT_ACTION, _FALLBACK_TURN_RIGHT_ACTION
        return left, right

    def configure_mouse_capture(
        self,
        session: "SessionController",
        sensitivity: float = 5.0,
    ) -> bool:
        """Configure FPS-style mouse capture for MalmoEnv games.

        Click on the Video tab to capture the mouse; ESC or focus-loss releases it.
        Horizontal mouse movement maps to turn-left / turn-right discrete actions.

        Args:
            session: The session controller with the current game.
            sensitivity: Pixels of horizontal mouse movement per turn action (default 5.0).

        Returns:
            True if mouse capture was enabled (MalmoEnv game), False otherwise.
        """
        game_id = session.game_id
        if game_id is None:
            return False

        try:
            gid = GameId(game_id)
        except ValueError:
            return False

        family = ENVIRONMENT_FAMILY_BY_GAME.get(gid)
        if family != EnvironmentFamily.MALMOENV:
            return False

        # Check for native mouse handling (MalmoInteractionController)
        interaction = getattr(session, "_interaction", None)
        is_native = (
            interaction is not None
            and type(interaction).__name__ == "MalmoInteractionController"
        )

        if is_native:
            def native_mouse_callback(dx: float, dy: float) -> None:
                # Send raw mouse deltas to the native handler
                # dx, dy are pixels moved since last frame
                idx = int(dx)
                idy = int(dy)
                if idx != 0 or idy != 0:
                    session.handle_native_mouse(idx, idy)

            self._render_tabs.configure_mouse_capture(
                enabled=True,
                delta_callback=native_mouse_callback,
                delta_scale=1.0,  # 1:1 pixel mapping
            )
            _LOG.debug("MalmoEnv native mouse capture enabled for %s", game_id)
            return True

        turn_left_action, turn_right_action = self._resolve_turn_actions(session)

        def mouse_action_callback(action: int) -> None:
            # Drop stale queued mouse events while a step is already in-flight.
            if not getattr(session, "_awaiting_human", True):
                return
            session.perform_human_action(action, key_label="mouse_turn")

        self._render_tabs.configure_mouse_capture(
            enabled=True,
            action_callback=mouse_action_callback,
            turn_left_action=turn_left_action,
            turn_right_action=turn_right_action,
            sensitivity=sensitivity,
        )
        _LOG.debug(
            "MalmoEnv mouse capture enabled for %s (left=%s right=%s)",
            game_id,
            turn_left_action,
            turn_right_action,
        )
        return True

    def disable_mouse_capture(self) -> None:
        """Disable mouse capture."""
        self._render_tabs.configure_mouse_capture(enabled=False)


__all__ = ["MalmoEnvLoader"]
