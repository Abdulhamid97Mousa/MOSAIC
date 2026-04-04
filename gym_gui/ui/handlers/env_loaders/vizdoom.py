"""ViZDoom environment loader for FPS-style mouse capture.

This loader handles:
- FPS-style mouse capture configuration
- Delta mode (continuous rotation) vs discrete turn actions
- Mouse sensitivity and callback setup
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from gym_gui.controllers.session import SessionController
    from gym_gui.ui.widgets.render_tabs import RenderTabs

from gym_gui.controllers.human_input import get_vizdoom_mouse_turn_actions

_LOG = logging.getLogger(__name__)


class VizdoomEnvLoader:
    """Loader for ViZDoom environment-specific configurations.

    This class encapsulates ViZDoom-specific setup, particularly the FPS-style
    mouse capture for controlling player view rotation.

    Args:
        render_tabs: The render tabs widget for configuring mouse capture.
    """

    def __init__(self, render_tabs: "RenderTabs") -> None:
        self._render_tabs = render_tabs

    def configure_mouse_capture(
        self,
        session: "SessionController",
        delta_scale: float = 0.5,
        sensitivity: float = 5.0,
    ) -> bool:
        """Configure FPS-style mouse capture for ViZDoom games.

        Click on the Video tab to capture the mouse, ESC or focus-loss to release.
        Mouse movement is converted to turn/look actions for controlling the player view.

        Uses delta mode (continuous rotation in degrees) if the adapter supports it,
        otherwise falls back to discrete turn actions.

        Args:
            session: The session controller with the current game.
            delta_scale: Degrees per pixel for delta mode (default 0.5).
            sensitivity: Pixels per action trigger for discrete mode (default 5.0).

        Returns:
            True if mouse capture was enabled (ViZDoom game), False otherwise.
        """
        game_id = session.game_id
        if game_id is None:
            return False

        turn_actions = get_vizdoom_mouse_turn_actions(game_id)
        if turn_actions is None:
            return False

        # Check if adapter supports delta mode (continuous mouse control)
        adapter = session._adapter
        has_delta_support = (
            adapter is not None
            and hasattr(adapter, "has_mouse_delta_support")
            and adapter.has_mouse_delta_support()
        )

        if has_delta_support:
            # Use continuous delta mode for true FPS control (360 degrees)
            def mouse_delta_callback(delta_x: float, delta_y: float) -> None:
                """Route mouse delta to the adapter for smooth rotation."""
                if adapter is not None and hasattr(adapter, "apply_mouse_delta"):
                    adapter.apply_mouse_delta(delta_x, delta_y)

            self._render_tabs.configure_mouse_capture(
                enabled=True,
                delta_callback=mouse_delta_callback,
                delta_scale=delta_scale,
            )
            _LOG.debug(f"ViZDoom mouse capture enabled (delta mode) for {game_id}")
        else:
            # Fallback to discrete turn actions
            turn_left, turn_right = turn_actions

            def mouse_action_callback(action: int) -> None:
                """Route mouse-triggered turn actions to the session controller."""
                session.perform_human_action(action, key_label="mouse_turn")

            self._render_tabs.configure_mouse_capture(
                enabled=True,
                action_callback=mouse_action_callback,
                turn_left_action=turn_left,
                turn_right_action=turn_right,
                sensitivity=sensitivity,
            )
            _LOG.debug(f"ViZDoom mouse capture enabled (discrete mode) for {game_id}")

        return True

    def disable_mouse_capture(self) -> None:
        """Disable mouse capture."""
        self._render_tabs.configure_mouse_capture(enabled=False)


__all__ = ["VizdoomEnvLoader"]
