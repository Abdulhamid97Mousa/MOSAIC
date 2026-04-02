"""Documentation for MOSAIC MalmoEnv environments.

All 13 MalmoEnv missions, each in its own sub-module.
"""
from __future__ import annotations

from .Attic import MALMOENV_ATTIC_HTML, get_attic_html
from .CatchTheMob import MALMOENV_CATCHTHEMOB_HTML, get_catchthemob_html
from .CliffWalking import MALMOENV_CLIFFWALKING_HTML, get_cliffwalking_html
from .DefaultFlatWorld import MALMOENV_DEFAULTFLATWORLD_HTML, get_defaultflatworld_html
from .DefaultWorld import MALMOENV_DEFAULTWORLD_HTML, get_defaultworld_html
from .Eating import MALMOENV_EATING_HTML, get_eating_html
from .FindTheGoal import MALMOENV_FINDTHEGOAL_HTML, get_findthegoal_html
from .MazeRunner import MALMOENV_MAZERUNNER_HTML, get_mazerunner_html
from .MobChase import MALMOENV_MOBCHASE_HTML, get_mobchase_html
from .Obstacles import MALMOENV_OBSTACLES_HTML, get_obstacles_html
from .TrickyArena import MALMOENV_TRICKYARENA_HTML, get_trickyarena_html
from .TreasureHunt import MALMOENV_TREASUREHUNT_HTML, get_treasurehunt_html
from .Vertical import MALMOENV_VERTICAL_HTML, get_vertical_html

# Backward-compatibility aliases (old MARLO_ prefix)
MARLO_ATTIC_HTML = MALMOENV_ATTIC_HTML
MARLO_CATCHTHEMOB_HTML = MALMOENV_CATCHTHEMOB_HTML
MARLO_CLIFFWALKING_HTML = MALMOENV_CLIFFWALKING_HTML
MARLO_DEFAULTFLATWORLD_HTML = MALMOENV_DEFAULTFLATWORLD_HTML
MARLO_DEFAULTWORLD_HTML = MALMOENV_DEFAULTWORLD_HTML
MARLO_EATING_HTML = MALMOENV_EATING_HTML
MARLO_FINDTHEGOAL_HTML = MALMOENV_FINDTHEGOAL_HTML
MARLO_MAZERUNNER_HTML = MALMOENV_MAZERUNNER_HTML
MARLO_OBSTACLES_HTML = MALMOENV_OBSTACLES_HTML
MARLO_TRICKYARENA_HTML = MALMOENV_TRICKYARENA_HTML
MARLO_VERTICAL_HTML = MALMOENV_VERTICAL_HTML

__all__ = [
    # New MALMOENV_ constants
    "MALMOENV_ATTIC_HTML",
    "MALMOENV_CATCHTHEMOB_HTML",
    "MALMOENV_CLIFFWALKING_HTML",
    "MALMOENV_DEFAULTFLATWORLD_HTML",
    "MALMOENV_DEFAULTWORLD_HTML",
    "MALMOENV_EATING_HTML",
    "MALMOENV_FINDTHEGOAL_HTML",
    "MALMOENV_MAZERUNNER_HTML",
    "MALMOENV_MOBCHASE_HTML",
    "MALMOENV_OBSTACLES_HTML",
    "MALMOENV_TRICKYARENA_HTML",
    "MALMOENV_TREASUREHUNT_HTML",
    "MALMOENV_VERTICAL_HTML",
    # Generator functions
    "get_attic_html",
    "get_catchthemob_html",
    "get_cliffwalking_html",
    "get_defaultflatworld_html",
    "get_defaultworld_html",
    "get_eating_html",
    "get_findthegoal_html",
    "get_mazerunner_html",
    "get_mobchase_html",
    "get_obstacles_html",
    "get_trickyarena_html",
    "get_treasurehunt_html",
    "get_vertical_html",
    # Backward-compat aliases
    "MARLO_ATTIC_HTML",
    "MARLO_CATCHTHEMOB_HTML",
    "MARLO_CLIFFWALKING_HTML",
    "MARLO_DEFAULTFLATWORLD_HTML",
    "MARLO_DEFAULTWORLD_HTML",
    "MARLO_EATING_HTML",
    "MARLO_FINDTHEGOAL_HTML",
    "MARLO_MAZERUNNER_HTML",
    "MARLO_OBSTACLES_HTML",
    "MARLO_TRICKYARENA_HTML",
    "MARLO_VERTICAL_HTML",
]
