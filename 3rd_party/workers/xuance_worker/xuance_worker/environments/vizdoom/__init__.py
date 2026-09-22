"""ViZDoom environment family for xuance_worker.

Family-per-directory layout matching gfootball/, smac/, smacv2/. Add new
ViZDoom variants (FastLane sidecar, multi-agent deathmatch wrapper,
per-scenario reward shaping helpers) here rather than at the flat
environments/ level.

Public API mirrors the pattern used by gfootball/__init__.py: consumers
`from xuance_worker.environments.vizdoom import ViZDoom_Env` and get
the same class regardless of internal file layout.
"""

from xuance_worker.environments.vizdoom.scenarios import (
    VIZDOOM_SCENARIOS,
    ViZDoomScenarioSpec,
)
from xuance_worker.environments.vizdoom.vizdoom import ViZDoom_Env

__all__ = ["ViZDoom_Env", "VIZDOOM_SCENARIOS", "ViZDoomScenarioSpec"]
