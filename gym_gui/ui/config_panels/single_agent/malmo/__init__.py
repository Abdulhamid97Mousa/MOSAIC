"""MalmoEnv configuration panel registration."""

from gym_gui.ui.config_panels.single_agent.malmo.config_panel import (
    DEFAULT_MALMO_CONFIG,
    MALMOENV_FAMILY,
    build_malmo_controls,
)

__all__ = ["MALMOENV_FAMILY", "DEFAULT_MALMO_CONFIG", "build_malmo_controls"]

