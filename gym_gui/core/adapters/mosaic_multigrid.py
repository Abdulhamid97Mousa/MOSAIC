"""MOSAIC MultiGrid adapter - Competitive team-based multi-agent environments.

PyPI Package: mosaic-multigrid v6.2.0
GitHub: https://github.com/Abdulhamid97Mousa/mosaic_multigrid
PyPI: https://pypi.org/project/mosaic-multigrid/

NOTE: The upstream mosaic_multigrid package v6.2.0 still imports from legacy `gym`
internally, which triggers deprecation warnings. We suppress these by setting
environment variables before importing. This is harmless - the environments
are registered with Gymnasium and work correctly.

Original Environments (Deprecated):
- MosaicMultiGrid-Soccer-v0: 4 agents (2v2 soccer), zero-sum competitive
- MosaicMultiGrid-Collect-v0: 3 agents, ball collection, competitive
- MosaicMultiGrid-Collect-2vs2-v0: 4 agents (2v2), ball collection
- MosaicMultiGrid-Collect-1vs1-v0: 2 agents (1v1), ball collection

IndAgObs Environments (Individual Agent Observations):
- MosaicMultiGrid-Soccer-2vs2-IndAgObs-v0: Ball respawn, first-to-2-goals, 16x11 FIFA grid
- MosaicMultiGrid-Soccer-1vs1-IndAgObs-v0: 1v1 soccer, same FIFA grid (v4.1.0)
- MosaicMultiGrid-Collect-IndAgObs-v0: Natural termination, 35x faster training
- MosaicMultiGrid-Collect-2vs2-IndAgObs-v0: Natural termination, 7 balls (no draws)
- MosaicMultiGrid-Collect-1vs1-IndAgObs-v0: Natural termination, 3 balls (v4.1.0)
- MosaicMultiGrid-Basketball-3vs3-IndAgObs-v0: 6 agents, 3v3 basketball, court rendering

TeamObs Environments (SMAC-style teammate awareness):
- MosaicMultiGrid-Soccer-2vs2-TeamObs-v0: IndAgObs + teammate positions/directions/has_ball
- MosaicMultiGrid-Collect-2vs2-TeamObs-v0: IndAgObs + teammate features
- MosaicMultiGrid-Basketball-3vs3-TeamObs-v0: IndAgObs + teammate features

Features:
- Gymnasium 1.0+ API (5-tuple dict-keyed observations)
- 8 actions (NOOP, LEFT, RIGHT, FORWARD, PICKUP, DROP, TOGGLE, DONE — noop=0 for AEC)
- view_size=3 (partial observability - competitive challenge)
- Team rewards (positive-only shared), ball passing, teleport passing, stealing mechanics
- Event tracking: goal_scored_by, passes_completed, steals_completed (v4.3.0)
- Agent position and carrying status in info dict per step (v5.0.0)
- Factored one-hot encoding with ball-carrying bit (OneHotObsWrapper)
- PettingZoo AEC + Parallel API support
- FIFA-style rendering for Soccer, court rendering for Basketball

Installation:
    pip install mosaic-multigrid>=5.0.0

Usage:
    from gym_gui.core.adapters.mosaic_multigrid import MultiGridSoccerIndAgObsAdapter
    adapter = MultiGridSoccerIndAgObsAdapter()
    adapter.load()
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from gym_gui.config.game_configs import MultiGridConfig
from gym_gui.core.adapters.base import (
    AdapterContext,
    AdapterStep,
    AgentSnapshot,
    EnvironmentAdapter,
    StepState,
    WorkerCapabilities,
)
from gym_gui.core.enums import ControlMode, GameId, RenderMode, SteppingParadigm
from gym_gui.logging_config.log_constants import (
    LOG_ADAPTER_ENV_CLOSED,
    LOG_ADAPTER_ENV_CREATED,
    LOG_ADAPTER_ENV_RESET,
    LOG_ADAPTER_STEP_SUMMARY,
    LOG_MOSAIC_MULTIGRID_GOAL_SCORED,
    LOG_MOSAIC_MULTIGRID_OBSERVATION,
    LOG_MOSAIC_MULTIGRID_PASS_COMPLETED,
    LOG_MOSAIC_MULTIGRID_STEAL_COMPLETED,
    LOG_MOSAIC_MULTIGRID_VISIBILITY,
)

_log = logging.getLogger(__name__)

try:  # pragma: no cover - import guard
    import gymnasium
except ImportError:  # pragma: no cover
    gymnasium = None  # type: ignore[assignment]

try:  # pragma: no cover - import guard
    # Import mosaic_multigrid.envs module — triggers gymnasium.register() side effect.
    # Each class is fetched with getattr so that a missing class in a given install
    # only nullifies that specific class instead of setting ALL classes to None.
    import mosaic_multigrid.envs as _mmg_envs  # noqa: F401
    _mmg = _mmg_envs
    SoccerGame4HEnv10x15N2          = getattr(_mmg, "SoccerGame4HEnv10x15N2",          None)  # type: ignore[assignment]
    SoccerGame2HIndAgObsEnv16x11N2  = getattr(_mmg, "SoccerGame2HIndAgObsEnv16x11N2",  None)  # type: ignore[assignment]
    CollectGame4HEnv10x10N2         = getattr(_mmg, "CollectGame4HEnv10x10N2",          None)  # type: ignore[assignment]
    CollectGame2HEnv10x10N2         = getattr(_mmg, "CollectGame2HEnv10x10N2",          None)  # type: ignore[assignment]
    SoccerGame4HIndAgObsEnv16x11N2  = getattr(_mmg, "SoccerGame4HIndAgObsEnv16x11N2",  None)  # type: ignore[assignment]
    CollectGame3HIndAgObsEnv10x10N3 = getattr(_mmg, "CollectGame3HIndAgObsEnv10x10N3", None)  # type: ignore[assignment]
    CollectGame4HIndAgObsEnv10x10N2 = getattr(_mmg, "CollectGame4HIndAgObsEnv10x10N2", None)  # type: ignore[assignment]
    CollectGame2HIndAgObsEnv10x10N2 = getattr(_mmg, "CollectGame2HIndAgObsEnv10x10N2", None)  # type: ignore[assignment]
    SoccerTeamObsEnv                = getattr(_mmg, "SoccerTeamObsEnv",                None)  # type: ignore[assignment]
    Collect2vs2TeamObsEnv           = getattr(_mmg, "Collect2vs2TeamObsEnv",           None)  # type: ignore[assignment]
    BasketballGame6HIndAgObsEnv19x11N3 = getattr(_mmg, "BasketballGame6HIndAgObsEnv19x11N3", None)  # type: ignore[assignment]
    Basketball3vs3TeamObsEnv        = getattr(_mmg, "Basketball3vs3TeamObsEnv",        None)  # type: ignore[assignment]
    AmericanFootball1v1Env16x11     = getattr(_mmg, "AmericanFootball1v1Env16x11",     None)  # type: ignore[assignment]
    AmericanFootball2v2Env16x11     = getattr(_mmg, "AmericanFootball2v2Env16x11",     None)  # type: ignore[assignment]
    AmericanFootball3v3Env16x11     = getattr(_mmg, "AmericanFootball3v3Env16x11",     None)  # type: ignore[assignment]
    AmericanFootball2v2TeamObsEnv   = getattr(_mmg, "AmericanFootball2v2TeamObsEnv",   None)  # type: ignore[assignment]
    AmericanFootball3v3TeamObsEnv   = getattr(_mmg, "AmericanFootball3v3TeamObsEnv",   None)  # type: ignore[assignment]
    AmericanFootballSoloGreenEnv16x11 = getattr(_mmg, "AmericanFootballSoloGreenEnv16x11", None)  # type: ignore[assignment]
    AmericanFootballSoloBlueEnv16x11  = getattr(_mmg, "AmericanFootballSoloBlueEnv16x11",  None)  # type: ignore[assignment]
    # Log any classes that came back None (class name not in this install)
    _missing = [
        name for name, obj in [
            ("SoccerGame4HEnv10x15N2",          SoccerGame4HEnv10x15N2),
            ("SoccerGame2HIndAgObsEnv16x11N2",  SoccerGame2HIndAgObsEnv16x11N2),
            ("CollectGame4HEnv10x10N2",         CollectGame4HEnv10x10N2),
            ("CollectGame2HEnv10x10N2",         CollectGame2HEnv10x10N2),
            ("SoccerGame4HIndAgObsEnv16x11N2",  SoccerGame4HIndAgObsEnv16x11N2),
            ("CollectGame3HIndAgObsEnv10x10N3", CollectGame3HIndAgObsEnv10x10N3),
            ("CollectGame4HIndAgObsEnv10x10N2", CollectGame4HIndAgObsEnv10x10N2),
            ("CollectGame2HIndAgObsEnv10x10N2", CollectGame2HIndAgObsEnv10x10N2),
            ("SoccerTeamObsEnv",                SoccerTeamObsEnv),
            ("Collect2vs2TeamObsEnv",           Collect2vs2TeamObsEnv),
            ("BasketballGame6HIndAgObsEnv19x11N3", BasketballGame6HIndAgObsEnv19x11N3),
            ("Basketball3vs3TeamObsEnv",        Basketball3vs3TeamObsEnv),
            ("AmericanFootball1v1Env16x11",     AmericanFootball1v1Env16x11),
            ("AmericanFootball2v2Env16x11",     AmericanFootball2v2Env16x11),
            ("AmericanFootball3v3Env16x11",     AmericanFootball3v3Env16x11),
            ("AmericanFootball2v2TeamObsEnv",   AmericanFootball2v2TeamObsEnv),
            ("AmericanFootball3v3TeamObsEnv",   AmericanFootball3v3TeamObsEnv),
            ("AmericanFootballSoloGreenEnv16x11", AmericanFootballSoloGreenEnv16x11),
            ("AmericanFootballSoloBlueEnv16x11",  AmericanFootballSoloBlueEnv16x11),
        ] if obj is None
    ]
    if _missing:
        _log.warning(
            "mosaic_multigrid installed but missing env classes (install may be outdated): %s",
            ", ".join(_missing),
        )
except ImportError as _import_err:  # pragma: no cover
    _log.warning(
        "mosaic_multigrid import failed (package not installed): %s",
        _import_err,
    )
    SoccerGame4HEnv10x15N2          = None  # type: ignore[assignment, misc]
    SoccerGame2HIndAgObsEnv16x11N2  = None  # type: ignore[assignment, misc]
    CollectGame4HEnv10x10N2         = None  # type: ignore[assignment, misc]
    CollectGame2HEnv10x10N2         = None  # type: ignore[assignment, misc]
    SoccerGame4HIndAgObsEnv16x11N2  = None  # type: ignore[assignment, misc]
    CollectGame3HIndAgObsEnv10x10N3 = None  # type: ignore[assignment, misc]
    CollectGame4HIndAgObsEnv10x10N2 = None  # type: ignore[assignment, misc]
    CollectGame2HIndAgObsEnv10x10N2 = None  # type: ignore[assignment, misc]
    SoccerTeamObsEnv                = None  # type: ignore[assignment, misc]
    Collect2vs2TeamObsEnv           = None  # type: ignore[assignment, misc]
    BasketballGame6HIndAgObsEnv19x11N3 = None  # type: ignore[assignment, misc]
    Basketball3vs3TeamObsEnv        = None  # type: ignore[assignment, misc]
    AmericanFootball1v1Env16x11     = None  # type: ignore[assignment, misc]
    AmericanFootball2v2Env16x11     = None  # type: ignore[assignment, misc]
    AmericanFootball3v3Env16x11     = None  # type: ignore[assignment, misc]
    AmericanFootball2v2TeamObsEnv   = None  # type: ignore[assignment, misc]
    AmericanFootball3v3TeamObsEnv   = None  # type: ignore[assignment, misc]
    AmericanFootballSoloGreenEnv16x11 = None  # type: ignore[assignment, misc]
    AmericanFootballSoloBlueEnv16x11  = None  # type: ignore[assignment, misc]


# MOSAIC multigrid action names (8 actions — noop=0 for AEC compatibility)
# Used by: Soccer, Collect, Basketball (PyPI: mosaic-multigrid v5.0.0+)
# Inspired by MeltingPot NOOP=0 convention (Google DeepMind)
MOSAIC_MULTIGRID_ACTIONS: List[str] = [
    "NOOP",     # 0 - No operation (AEC: non-acting agents wait)
    "LEFT",     # 1 - Turn left
    "RIGHT",    # 2 - Turn right
    "FORWARD",  # 3 - Move forward
    "PICKUP",   # 4 - Pick up object / steal from opponent
    "DROP",     # 5 - Drop object / score at goal / teleport pass
    "TOGGLE",   # 6 - Toggle/activate object
    "DONE",     # 7 - Done
]

# Observation encoding constants (3-channel: TYPE, COLOR, STATE)
_AGENT_TYPE_IDX = 10        # World.OBJECT_TO_IDX['agent']
_BALL_TYPE_IDX = 6          # World.OBJECT_TO_IDX['ball']
_GOAL_TYPE_IDX = 8          # World.OBJECT_TO_IDX['goal']
_BALL_CARRY_OFFSET = 100    # STATE >= 100 means agent is carrying ball

# Human-readable names for observation type channel (index 0)
_TYPE_NAMES: Dict[int, str] = {
    0: "unseen", 1: "empty", 2: "wall", 3: "floor", 4: "door",
    5: "key", 6: "BALL", 7: "box", 8: "GOAL", 9: "lava", 10: "AGENT",
}

# Human-readable names for observation color channel (index 1)
_COLOR_NAMES: Dict[int, str] = {
    0: "red", 1: "green", 2: "blue", 3: "purple", 4: "yellow", 5: "grey",
}

# Log frequency for step events
_MULTIGRID_STEP_LOG_FREQUENCY = 50
# Log frequency for observation grids (every N steps)
_MULTIGRID_OBS_LOG_FREQUENCY = 10


class MultiGridAdapter(EnvironmentAdapter[List[np.ndarray], List[int]]):
    """Adapter for MOSAIC MultiGrid multi-agent environments.

    This adapter handles competitive team-based environments from the mosaic-multigrid
    PyPI package. Key characteristics:
    - Multiple agents acting simultaneously (2-4 agents)
    - Team-based competitive gameplay (Soccer: 2v2, Collect: 3 players)
    - Gymnasium 1.0+ API (5-tuple dict-keyed observations)
    - view_size=3 for partial observability (competitive challenge)
    - 8 actions (noop=0, left=1 … done=7 — noop for AEC compatibility)

    The environment provides:
    - Observations: Dict of encoded grid views per agent
    - Actions: Dict of discrete actions (0-7) per agent
    - Rewards: Dict of float rewards per agent
    - Terminated/Truncated: Dict per agent + __all__ flag
    """

    default_render_mode = RenderMode.RGB_ARRAY
    supported_render_modes = (RenderMode.RGB_ARRAY,)
    supported_control_modes = (
        ControlMode.HUMAN_ONLY,  # Multi-human multi-keyboard gameplay
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    )

    # Multi-agent capability declaration
    capabilities = WorkerCapabilities(
        stepping_paradigm=SteppingParadigm.SIMULTANEOUS,
        supported_paradigms=(SteppingParadigm.SIMULTANEOUS,),
        env_types=("gym", "mosaic_multigrid"),
        action_spaces=("discrete",),
        observation_spaces=("box",),
        max_agents=6,  # Basketball uses 6 agents (3v3), Soccer uses 4 (2v2)
        supports_self_play=True,
        supports_record=True,
    )

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MultiGridConfig | None = None,
    ) -> None:
        """Initialize the MOSAIC MultiGrid adapter.

        Args:
            context: Adapter context with settings and control mode
            config: MultiGrid configuration
        """
        super().__init__(context)
        if config is None:
            config = MultiGridConfig()
        self._config = config
        self._env_id = config.env_id
        self._step_counter = 0
        self._num_agents = 0
        self._agent_observations: List[np.ndarray] = []
        self._agent_rewards: List[float] = []
        self._team_map: Dict[int, int] = {}  # agent_index -> team_index
        self._team_episode_rewards: Dict[int, float] = {}  # team_index -> cumulative reward
        self._color_to_team: Dict[int, int] = {}  # color_index -> team_index (for visibility)

    @property
    def id(self) -> str:  # type: ignore[override]
        """Return the environment identifier."""
        # Map env_id to proper GameId format
        if self._env_id == "soccer":
            return "MosaicMultiGrid-Soccer-v0"
        elif self._env_id == "collect":
            return "MosaicMultiGrid-Collect-v0"
        elif self._env_id == "soccer_indagobs":
            return "MosaicMultiGrid-Soccer-2vs2-IndAgObs-v0"
        elif self._env_id == "soccer_1vs1_indagobs":
            return "MosaicMultiGrid-Soccer-1vs1-IndAgObs-v0"
        elif self._env_id == "collect_indagobs":
            return "MosaicMultiGrid-Collect-IndAgObs-v0"
        elif self._env_id == "collect2vs2_indagobs":
            return "MosaicMultiGrid-Collect-2vs2-IndAgObs-v0"
        elif self._env_id == "collect_1vs1_indagobs":
            return "MosaicMultiGrid-Collect-1vs1-IndAgObs-v0"
        elif self._env_id == "soccer_teamobs":
            return "MosaicMultiGrid-Soccer-2vs2-TeamObs-v0"
        elif self._env_id == "collect2vs2_teamobs":
            return "MosaicMultiGrid-Collect-2vs2-TeamObs-v0"
        elif self._env_id == "basketball_indagobs":
            return "MosaicMultiGrid-Basketball-3vs3-IndAgObs-v0"
        elif self._env_id == "basketball_teamobs":
            return "MosaicMultiGrid-Basketball-3vs3-TeamObs-v0"
        elif self._env_id == "american_football_1v1":
            return "MosaicMultiGrid-AmericanFootball-1v1-v0"
        elif self._env_id == "american_football_2v2":
            return "MosaicMultiGrid-AmericanFootball-2v2-v0"
        elif self._env_id == "american_football_3v3":
            return "MosaicMultiGrid-AmericanFootball-3v3-v0"
        elif self._env_id == "american_football_2v2_teamobs":
            return "MosaicMultiGrid-AmericanFootball-2v2-TeamObs-v0"
        elif self._env_id == "american_football_3v3_teamobs":
            return "MosaicMultiGrid-AmericanFootball-3v3-TeamObs-v0"
        elif self._env_id == "american_football_solo_green":
            return "MosaicMultiGrid-AmericanFootball-Solo-Green-v0"
        elif self._env_id == "american_football_solo_blue":
            return "MosaicMultiGrid-AmericanFootball-Solo-Blue-v0"
        else:
            return self._env_id

    @property
    def num_agents(self) -> int:
        """Return the number of agents in the environment."""
        return self._num_agents

    def load(self) -> None:
        """Instantiate the MOSAIC MultiGrid environment."""
        if gymnasium is None:
            raise RuntimeError(
                "gymnasium package not installed. "
                "Install with: pip install gymnasium"
            )

        # Build optional kwargs (view_size override from config panel)
        extra_kwargs: Dict[str, Any] = {}
        if self._config.view_size is not None:
            extra_kwargs["view_size"] = self._config.view_size

        try:
            # Create environment based on env_id
            # Original environments (deprecated)
            if self._env_id == "soccer" or self._env_id == "MosaicMultiGrid-Soccer-v0":
                if SoccerGame4HEnv10x15N2 is None:
                    raise RuntimeError(
                        "mosaic_multigrid v6.0.0 not installed. "
                        "Install with: pip install mosaic-multigrid==6.0.0"
                    )
                env = SoccerGame4HEnv10x15N2(render_mode='rgb_array', **extra_kwargs)
            elif self._env_id == "collect" or self._env_id == "MosaicMultiGrid-Collect-v0":
                if CollectGame4HEnv10x10N2 is None:
                    raise RuntimeError(
                        "mosaic_multigrid v6.0.0 not installed. "
                        "Install with: pip install mosaic-multigrid==6.0.0"
                    )
                env = CollectGame4HEnv10x10N2(render_mode='rgb_array', **extra_kwargs)
            elif self._env_id == "MosaicMultiGrid-Collect-2vs2-v0":
                if CollectGame4HEnv10x10N2 is None:
                    raise RuntimeError(
                        "mosaic_multigrid v6.0.0 not installed. "
                        "Install with: pip install mosaic-multigrid==6.0.0"
                    )
                env = CollectGame4HEnv10x10N2(render_mode='rgb_array', **extra_kwargs)
            elif self._env_id == "MosaicMultiGrid-Collect-1vs1-v0":
                if CollectGame2HEnv10x10N2 is None:
                    raise RuntimeError(
                        "mosaic_multigrid v6.0.0 not installed. "
                        "Install with: pip install mosaic-multigrid==6.0.0"
                    )
                env = CollectGame2HEnv10x10N2(render_mode='rgb_array', **extra_kwargs)
            # IndAgObs environments (v4.0.0 - Individual Agent Observations)
            elif self._env_id == "soccer_indagobs" or self._env_id == "MosaicMultiGrid-Soccer-2vs2-IndAgObs-v0":
                if SoccerGame4HIndAgObsEnv16x11N2 is None:
                    raise RuntimeError(
                        "mosaic_multigrid v6.0.0 not installed. "
                        "Install with: pip install mosaic-multigrid==6.0.0"
                    )
                env = SoccerGame4HIndAgObsEnv16x11N2(render_mode='rgb_array', **extra_kwargs)
            elif self._env_id == "collect_indagobs" or self._env_id == "MosaicMultiGrid-Collect-IndAgObs-v0":
                if CollectGame3HIndAgObsEnv10x10N3 is None:
                    raise RuntimeError(
                        "mosaic_multigrid v6.0.0 not installed. "
                        "Install with: pip install mosaic-multigrid==6.0.0"
                    )
                env = CollectGame3HIndAgObsEnv10x10N3(render_mode='rgb_array', **extra_kwargs)
            elif self._env_id == "collect2vs2_indagobs" or self._env_id == "MosaicMultiGrid-Collect-2vs2-IndAgObs-v0":
                if CollectGame4HIndAgObsEnv10x10N2 is None:
                    raise RuntimeError(
                        "mosaic_multigrid v6.0.0 not installed. "
                        "Install with: pip install mosaic-multigrid==6.0.0"
                    )
                env = CollectGame4HIndAgObsEnv10x10N2(render_mode='rgb_array', **extra_kwargs)
            elif self._env_id == "collect_1vs1_indagobs" or self._env_id == "MosaicMultiGrid-Collect-1vs1-IndAgObs-v0":
                if CollectGame2HIndAgObsEnv10x10N2 is None:
                    raise RuntimeError(
                        "mosaic_multigrid v6.0.0 not installed. "
                        "Install with: pip install mosaic-multigrid==6.0.0"
                    )
                env = CollectGame2HIndAgObsEnv10x10N2(render_mode='rgb_array', **extra_kwargs)
            elif self._env_id == "soccer_1vs1_indagobs" or self._env_id == "MosaicMultiGrid-Soccer-1vs1-IndAgObs-v0":
                if SoccerGame2HIndAgObsEnv16x11N2 is None:
                    raise RuntimeError(
                        "mosaic_multigrid v6.0.0 not installed. "
                        "Install with: pip install mosaic-multigrid==6.0.0"
                    )
                env = SoccerGame2HIndAgObsEnv16x11N2(render_mode='rgb_array', **extra_kwargs)
            elif self._env_id == "basketball_indagobs" or self._env_id == "MosaicMultiGrid-Basketball-3vs3-IndAgObs-v0":
                if BasketballGame6HIndAgObsEnv19x11N3 is None:
                    raise RuntimeError(
                        "mosaic_multigrid v6.0.0 not installed. "
                        "Install with: pip install mosaic-multigrid==6.0.0"
                    )
                env = BasketballGame6HIndAgObsEnv19x11N3(render_mode='rgb_array', **extra_kwargs)
            # TeamObs environments (v4.0.0 - SMAC-style teammate awareness)
            elif self._env_id == "soccer_teamobs" or self._env_id == "MosaicMultiGrid-Soccer-2vs2-TeamObs-v0":
                if SoccerTeamObsEnv is None:
                    raise RuntimeError(
                        "mosaic_multigrid v6.0.0 not installed. "
                        "Install with: pip install mosaic-multigrid==6.0.0"
                    )
                env = SoccerTeamObsEnv(render_mode='rgb_array', **extra_kwargs)
            elif self._env_id == "collect2vs2_teamobs" or self._env_id == "MosaicMultiGrid-Collect-2vs2-TeamObs-v0":
                if Collect2vs2TeamObsEnv is None:
                    raise RuntimeError(
                        "mosaic_multigrid v6.0.0 not installed. "
                        "Install with: pip install mosaic-multigrid==6.0.0"
                    )
                env = Collect2vs2TeamObsEnv(render_mode='rgb_array', **extra_kwargs)
            elif self._env_id == "basketball_teamobs" or self._env_id == "MosaicMultiGrid-Basketball-3vs3-TeamObs-v0":
                if Basketball3vs3TeamObsEnv is None:
                    raise RuntimeError(
                        "mosaic_multigrid v6.0.0 not installed. "
                        "Install with: pip install mosaic-multigrid==6.0.0"
                    )
                env = Basketball3vs3TeamObsEnv(render_mode='rgb_array', **extra_kwargs)
            # American Football environments (v6.3.0)
            elif self._env_id == "american_football_1v1" or self._env_id == "MosaicMultiGrid-AmericanFootball-1v1-v0":
                if AmericanFootball1v1Env16x11 is None:
                    raise RuntimeError(
                        "mosaic_multigrid v6.3.0 not installed. "
                        "Install with: pip install mosaic-multigrid==6.3.0"
                    )
                env = AmericanFootball1v1Env16x11(render_mode='rgb_array', **extra_kwargs)
            elif self._env_id == "american_football_2v2" or self._env_id == "MosaicMultiGrid-AmericanFootball-2v2-v0":
                if AmericanFootball2v2Env16x11 is None:
                    raise RuntimeError(
                        "mosaic_multigrid v6.3.0 not installed. "
                        "Install with: pip install mosaic-multigrid==6.3.0"
                    )
                env = AmericanFootball2v2Env16x11(render_mode='rgb_array', **extra_kwargs)
            elif self._env_id == "american_football_3v3" or self._env_id == "MosaicMultiGrid-AmericanFootball-3v3-v0":
                if AmericanFootball3v3Env16x11 is None:
                    raise RuntimeError(
                        "mosaic_multigrid v6.3.0 not installed. "
                        "Install with: pip install mosaic-multigrid==6.3.0"
                    )
                env = AmericanFootball3v3Env16x11(render_mode='rgb_array', **extra_kwargs)
            elif self._env_id == "american_football_2v2_teamobs" or self._env_id == "MosaicMultiGrid-AmericanFootball-2v2-TeamObs-v0":
                if AmericanFootball2v2TeamObsEnv is None:
                    raise RuntimeError(
                        "mosaic_multigrid v6.3.0 not installed. "
                        "Install with: pip install mosaic-multigrid==6.3.0"
                    )
                env = AmericanFootball2v2TeamObsEnv(render_mode='rgb_array', **extra_kwargs)
            elif self._env_id == "american_football_3v3_teamobs" or self._env_id == "MosaicMultiGrid-AmericanFootball-3v3-TeamObs-v0":
                if AmericanFootball3v3TeamObsEnv is None:
                    raise RuntimeError(
                        "mosaic_multigrid v6.3.0 not installed. "
                        "Install with: pip install mosaic-multigrid==6.3.0"
                    )
                env = AmericanFootball3v3TeamObsEnv(render_mode='rgb_array', **extra_kwargs)
            elif self._env_id == "american_football_solo_green" or self._env_id == "MosaicMultiGrid-AmericanFootball-Solo-Green-v0":
                if AmericanFootballSoloGreenEnv16x11 is None:
                    raise RuntimeError(
                        "mosaic_multigrid v6.3.0 not installed. "
                        "Install with: pip install mosaic-multigrid==6.3.0"
                    )
                env = AmericanFootballSoloGreenEnv16x11(render_mode='rgb_array', **extra_kwargs)
            elif self._env_id == "american_football_solo_blue" or self._env_id == "MosaicMultiGrid-AmericanFootball-Solo-Blue-v0":
                if AmericanFootballSoloBlueEnv16x11 is None:
                    raise RuntimeError(
                        "mosaic_multigrid v6.3.0 not installed. "
                        "Install with: pip install mosaic-multigrid==6.3.0"
                    )
                env = AmericanFootballSoloBlueEnv16x11(render_mode='rgb_array', **extra_kwargs)
            else:
                # Try to make via gymnasium.make if registered (e.g., Solo variants)
                try:
                    env = gymnasium.make(self._env_id, render_mode='rgb_array', **extra_kwargs)
                except Exception as e:
                    raise RuntimeError(
                        f"Unknown MOSAIC MultiGrid environment: {self._env_id}. "
                        f"Available: Soccer/Collect/Basketball/AmericanFootball IndAgObs, TeamObs, and Solo variants. Error: {e}"
                    )

            self._env = env
            self._num_agents = len(env.unwrapped.agents)

            # Build agent-to-team mapping for per-team reward tracking
            self._team_map = {}
            for agent in env.unwrapped.agents:
                self._team_map[agent.index] = agent.team_index
            unique_teams = sorted(set(self._team_map.values()))
            self._team_episode_rewards = {t: 0.0 for t in unique_teams}

            self.log_constant(
                LOG_ADAPTER_ENV_CREATED,
                extra={
                    "env_id": self._env_id,
                    "num_agents": self._num_agents,
                    "grid_size": f"{env.unwrapped.width}x{env.unwrapped.height}",
                },
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create MOSAIC MultiGrid environment '{self._env_id}': {exc}"
            ) from exc

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> AdapterStep[List[np.ndarray]]:
        """Reset the environment.

        Args:
            seed: Optional random seed
            options: Additional reset options

        Returns:
            Initial step result with list of observations (one per agent)
        """
        env = self._require_env()

        # MOSAIC MultiGrid uses Gymnasium API: reset(seed=seed) -> (obs, info)
        if seed is not None:
            reset_result = env.reset(seed=seed)
        else:
            reset_result = env.reset()

        # Gymnasium API returns (obs, info) tuple
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            raw_obs = reset_result[0]
        else:
            raw_obs = reset_result

        # Convert observations dict to list (uses {0: obs0, 1: obs1})
        if isinstance(raw_obs, dict):
            self._agent_observations = [raw_obs[i] for i in range(self._num_agents)]
        else:
            self._agent_observations = list(raw_obs)

        self._agent_rewards = [0.0] * self._num_agents
        self._step_counter = 0
        self._episode_step = 0
        self._episode_return = 0.0
        self._team_episode_rewards = {t: 0.0 for t in self._team_episode_rewards}

        # Build color-to-team mapping AFTER reset (colors not set until _gen_grid)
        self._color_to_team = {}
        for agent in env.unwrapped.agents:
            color_idx = (
                agent.color.to_index()
                if hasattr(agent.color, "to_index")
                else int(agent.color)
            )
            self._color_to_team[color_idx] = agent.team_index

        info: Dict[str, Any] = {
            "num_agents": self._num_agents,
            "agent_observations": self._agent_observations,
            "env_id": self._env_id,
        }

        self.log_constant(
            LOG_ADAPTER_ENV_RESET,
            extra={
                "env_id": self._env_id,
                "num_agents": self._num_agents,
                "seed": seed if seed is not None else "None",
            },
        )

        return self._package_step(self._agent_observations, 0.0, False, False, info)

    def step(self, action: List[int] | int) -> AdapterStep[List[np.ndarray]]:
        """Execute actions for all agents simultaneously.

        Args:
            action: List of actions (one per agent) or single action.
                    If single action, it's broadcast to all agents.

        Returns:
            Step result with list of observations, sum of rewards, and info
        """
        env = self._require_env()

        # Handle single action (broadcast to all agents)
        if isinstance(action, int):
            actions = [action] * self._num_agents
        else:
            actions = list(action)
            if len(actions) != self._num_agents:
                raise ValueError(
                    f"Expected {self._num_agents} actions, got {len(actions)}"
                )

        # MOSAIC MultiGrid uses Gymnasium API with dict actions
        # Convert list to dict: [2, 6] -> {0: 2, 1: 6}
        actions_dict = {i: actions[i] for i in range(len(actions))}
        step_result = env.step(actions_dict)

        # Gymnasium API returns 5 values
        if len(step_result) == 5:
            raw_obs, rewards, terminated, truncated, info = step_result
        else:
            # Fallback for unexpected format
            raw_obs, rewards, done, info = step_result[:4]
            terminated = bool(done)
            truncated = False

        # Convert dict terminated/truncated to bool
        if isinstance(terminated, dict):
            terminated = terminated.get("__all__", any(terminated.values()))
        if isinstance(truncated, dict):
            truncated = truncated.get("__all__", any(truncated.values()))

        # Convert observations dict to list
        if isinstance(raw_obs, dict):
            self._agent_observations = [raw_obs[i] for i in range(self._num_agents)]
        else:
            self._agent_observations = list(raw_obs)

        # Handle rewards dict (uses integer keys: {0: 0.5, 1: 0.5})
        if isinstance(rewards, dict):
            self._agent_rewards = []
            for i in range(self._num_agents):
                if i in rewards:
                    self._agent_rewards.append(float(rewards[i]))
                else:
                    self._agent_rewards.append(0.0)
        else:
            self._agent_rewards = [float(r) for r in rewards] if hasattr(rewards, '__iter__') else [float(rewards)] * self._num_agents

        # Prepare info dict
        step_info: Dict[str, Any] = dict(info) if isinstance(info, dict) else {}
        step_info["num_agents"] = self._num_agents
        step_info["agent_observations"] = self._agent_observations
        step_info["agent_rewards"] = self._agent_rewards
        step_info["actions"] = actions
        step_info["action_names"] = [
            MOSAIC_MULTIGRID_ACTIONS[a] if 0 <= a < len(MOSAIC_MULTIGRID_ACTIONS) else str(a)
            for a in actions
        ]

        # Sum rewards for total episode reward tracking
        total_reward = float(sum(self._agent_rewards))

        # Compute per-team step rewards and accumulate episode totals
        team_step_rewards: Dict[int, float] = {t: 0.0 for t in self._team_episode_rewards}
        for agent_idx, reward_val in enumerate(self._agent_rewards):
            team_idx = self._team_map.get(agent_idx, 0)
            team_step_rewards[team_idx] += reward_val
        for t, r in team_step_rewards.items():
            self._team_episode_rewards[t] += r
        step_info["team_episode_rewards"] = dict(self._team_episode_rewards)

        # Update episode tracking
        self._step_counter += 1
        self._episode_step += 1
        self._episode_return += total_reward
        step_info["episode_step"] = self._episode_step
        step_info["episode_score"] = self._episode_return

        # Analyze inter-agent visibility from observations
        visibility = self._analyze_visibility()
        step_info["agent_visibility"] = visibility

        # Log visibility at INFO when any agent sees another
        if visibility:
            obs_model = "TeamObs" if any(
                isinstance(o, dict) and "teammate_positions" in o
                for o in self._agent_observations
            ) else "IndAgObs"
            vis_parts = []
            for agent_idx, sightings in sorted(visibility.items()):
                vis_parts.append(f"agent_{agent_idx}: {', '.join(sightings)}")
            vis_text = " | ".join(vis_parts)
            self.log_constant(
                LOG_MOSAIC_MULTIGRID_VISIBILITY,
                message=f"{obs_model} step {self._step_counter + 1} | {vis_text}",
                extra={
                    "env_id": self._env_id,
                    "step": self._step_counter + 1,
                    "obs_model": obs_model,
                    "sightings": vis_text,
                },
            )

        # Log observation grids periodically so user can verify what agent sees
        if self._step_counter % _MULTIGRID_OBS_LOG_FREQUENCY == 1:
            for agent_idx, obs in enumerate(self._agent_observations):
                if not isinstance(obs, dict) or "image" not in obs:
                    continue
                image = obs["image"]  # shape: (view_size, view_size, 3)
                rows, cols = image.shape[0], image.shape[1]
                grid_lines = []
                sees_ball = False
                sees_goal = False
                for r in range(rows):
                    cells = []
                    for c in range(cols):
                        t = int(image[r, c, 0])
                        color = int(image[r, c, 1])
                        state = int(image[r, c, 2])
                        name = _TYPE_NAMES.get(t, f"?{t}")
                        if t == _BALL_TYPE_IDX:
                            sees_ball = True
                            cells.append(f"BALL({_COLOR_NAMES.get(color, color)})")
                        elif t == _GOAL_TYPE_IDX:
                            sees_goal = True
                            cells.append(f"GOAL({_COLOR_NAMES.get(color, color)})")
                        elif t == _AGENT_TYPE_IDX:
                            carrying = " +ball" if state >= _BALL_CARRY_OFFSET else ""
                            cells.append(f"AGT({_COLOR_NAMES.get(color, color)}{carrying})")
                        else:
                            cells.append(name)
                    grid_lines.append(" | ".join(cells))
                grid_text = "\n".join(grid_lines)
                summary = f"agent_{agent_idx} [{rows}x{cols}]"
                if sees_ball:
                    summary += " sees_ball=YES"
                else:
                    summary += " sees_ball=NO"
                if sees_goal:
                    summary += " sees_goal=YES"
                self.log_constant(
                    LOG_MOSAIC_MULTIGRID_OBSERVATION,
                    message=f"step {self._step_counter} {summary}\n{grid_text}",
                    extra={
                        "env_id": self._env_id,
                        "step": self._step_counter,
                        "agent": agent_idx,
                        "sees_ball": sees_ball,
                        "sees_goal": sees_goal,
                        "view_size": rows,
                    },
                )

        # Log v4.3.0 event tracking (goal, pass, steal) from per-agent info
        agent0_info = info.get(0, {}) if isinstance(info, dict) else {}
        if isinstance(agent0_info, dict):
            if "goal_scored_by" in agent0_info:
                evt = agent0_info["goal_scored_by"]
                self.log_constant(
                    LOG_MOSAIC_MULTIGRID_GOAL_SCORED,
                    message=(
                        f"agent_{evt.get('scorer')} scored for team {evt.get('team')}"
                        f" at step {evt.get('step')}"
                    ),
                    extra={
                        "env_id": self._env_id,
                        "step": evt.get("step"),
                        "scorer": f"agent_{evt.get('scorer')}",
                        "team": evt.get("team"),
                        "visibility": visibility,
                    },
                )
            if "pass_completed" in agent0_info:
                evt = agent0_info["pass_completed"]
                self.log_constant(
                    LOG_MOSAIC_MULTIGRID_PASS_COMPLETED,
                    message=(
                        f"agent_{evt.get('passer')} -> agent_{evt.get('receiver')}"
                        f" (team {evt.get('team')}) at step {evt.get('step')}"
                    ),
                    extra={
                        "env_id": self._env_id,
                        "step": evt.get("step"),
                        "passer": f"agent_{evt.get('passer')}",
                        "receiver": f"agent_{evt.get('receiver')}",
                        "team": evt.get("team"),
                        "visibility": visibility,
                    },
                )
            if "steal_completed" in agent0_info:
                evt = agent0_info["steal_completed"]
                self.log_constant(
                    LOG_MOSAIC_MULTIGRID_STEAL_COMPLETED,
                    message=(
                        f"agent_{evt.get('stealer')} stole from agent_{evt.get('victim')}"
                        f" (team {evt.get('team')}) at step {evt.get('step')}"
                    ),
                    extra={
                        "env_id": self._env_id,
                        "step": evt.get("step"),
                        "stealer": f"agent_{evt.get('stealer')}",
                        "victim": f"agent_{evt.get('victim')}",
                        "team": evt.get("team"),
                        "visibility": visibility,
                    },
                )

        if self._step_counter % _MULTIGRID_STEP_LOG_FREQUENCY == 1:
            self.log_constant(
                LOG_ADAPTER_STEP_SUMMARY,
                extra={
                    "env_id": self._env_id,
                    "step": self._step_counter,
                    "actions": step_info["action_names"],
                    "rewards": self._agent_rewards,
                    "total_reward": total_reward,
                    "terminated": terminated,
                    "visibility": visibility,
                },
            )

        return self._package_step(
            self._agent_observations, total_reward, terminated, truncated, step_info
        )

    def _analyze_visibility(self) -> Dict[int, List[str]]:
        """Analyze each agent's awareness of other agents.

        Two observation models produce different awareness levels:

        IndAgObs -- agent only knows about others if they appear in its
        3x3 local view.  Sightings are prefixed with ``[view]``.

        TeamObs -- agent additionally receives global teammate features
        (positions, directions, has_ball) regardless of distance.
        These are prefixed with ``[team]``.  Opponents are still only
        visible through the 3x3 view.

        Returns:
            Dict mapping agent index to list of awareness descriptions.
            Only agents with non-empty awareness appear as keys.
        """
        env = self._require_env()
        agents = env.unwrapped.agents

        # Build color -> list of agent indices for specific identification
        color_agents: Dict[int, List[int]] = {}
        for agent in agents:
            cidx = (
                agent.color.to_index()
                if hasattr(agent.color, "to_index")
                else int(agent.color)
            )
            color_agents.setdefault(cidx, []).append(agent.index)

        # Build team -> sorted list of member indices
        team_members: Dict[int, List[int]] = {}
        for idx, team in self._team_map.items():
            team_members.setdefault(team, []).append(idx)

        visibility: Dict[int, List[str]] = {}
        for i, obs in enumerate(self._agent_observations):
            if not isinstance(obs, dict) or "image" not in obs:
                continue

            image = obs["image"]
            observer_team = self._team_map.get(i, 0)
            sightings: List[str] = []
            is_teamobs = "teammate_positions" in obs

            # --- 3x3 view scan (both IndAgObs and TeamObs) ---
            for r in range(image.shape[0]):
                for c in range(image.shape[1]):
                    type_idx = int(image[r, c, 0])
                    if type_idx != _AGENT_TYPE_IDX:
                        continue

                    color_idx = int(image[r, c, 1])
                    state = int(image[r, c, 2])
                    spotted_team = self._color_to_team.get(color_idx, -1)
                    is_teammate = spotted_team == observer_team
                    has_ball = state >= _BALL_CARRY_OFFSET

                    candidates = [
                        idx for idx in color_agents.get(color_idx, [])
                        if idx != i
                    ]
                    role = "teammate" if is_teammate else "opponent"
                    if len(candidates) == 1:
                        desc = f"[view] {role} agent_{candidates[0]}"
                    else:
                        desc = f"[view] {role}"
                    if has_ball:
                        desc += " (ball)"
                    sightings.append(desc)

            # --- TeamObs: global teammate awareness (same team only) ---
            if is_teamobs:
                positions = obs.get("teammate_positions")   # (N, 2)
                directions = obs.get("teammate_directions")  # (N,)
                has_balls = obs.get("teammate_has_ball")     # (N,)

                # Teammates of agent i, excluding i itself
                teammates = [t for t in team_members.get(observer_team, []) if t != i]

                if positions is not None:
                    for k, mate_idx in enumerate(teammates):
                        if k >= len(positions):
                            break
                        dx, dy = int(positions[k][0]), int(positions[k][1])
                        d_dir = int(directions[k]) if directions is not None and k < len(directions) else -1
                        d_ball = bool(has_balls[k]) if has_balls is not None and k < len(has_balls) else False
                        dir_names = {0: "right", 1: "down", 2: "left", 3: "up"}
                        parts = [f"[team] teammate agent_{mate_idx} at ({dx},{dy})"]
                        if d_dir in dir_names:
                            parts.append(dir_names[d_dir])
                        if d_ball:
                            parts.append("(ball)")
                        sightings.append(" ".join(parts))

            if sightings:
                visibility[i] = sightings

        return visibility

    def render(self) -> Dict[str, Any]:
        """Render the environment.

        Returns:
            Dictionary with RGB array and metadata
        """
        env = self._require_env()
        try:
            # Gymnasium API: render_mode is set during env creation
            frame = env.render()

            if frame is None:
                return {
                    "mode": RenderMode.RGB_ARRAY.value,
                    "rgb": np.zeros((320, 480, 3), dtype=np.uint8),
                    "game_id": self._env_id,
                    "num_agents": self._num_agents,
                }

            array = np.asarray(frame)
            return {
                "mode": RenderMode.RGB_ARRAY.value,
                "rgb": array,
                "game_id": self._env_id,
                "num_agents": self._num_agents,
                "step": self._step_counter,
            }
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Render failed for {self._env_id}: {e}")
            return {
                "mode": RenderMode.RGB_ARRAY.value,
                "rgb": np.zeros((320, 480, 3), dtype=np.uint8),
                "game_id": self._env_id,
            }

    def close(self) -> None:
        """Close the environment."""
        if self._env is not None:
            self.log_constant(
                LOG_ADAPTER_ENV_CLOSED,
                extra={"env_id": self.id},
            )
            if hasattr(self._env, "close"):
                self._env.close()
            self._env = None

    def get_agent_observation(self, agent_idx: int) -> np.ndarray:
        """Get observation for a specific agent.

        Args:
            agent_idx: Index of the agent (0 to num_agents-1)

        Returns:
            Observation array for the specified agent
        """
        if agent_idx < 0 or agent_idx >= self._num_agents:
            raise IndexError(f"Agent index {agent_idx} out of range [0, {self._num_agents})")
        return self._agent_observations[agent_idx]

    def get_agent_reward(self, agent_idx: int) -> float:
        """Get last reward for a specific agent.

        Args:
            agent_idx: Index of the agent (0 to num_agents-1)

        Returns:
            Reward for the specified agent from last step
        """
        if agent_idx < 0 or agent_idx >= self._num_agents:
            raise IndexError(f"Agent index {agent_idx} out of range [0, {self._num_agents})")
        return self._agent_rewards[agent_idx]

    def build_step_state(
        self,
        observation: List[np.ndarray],
        info: Mapping[str, Any],
    ) -> StepState:
        """Construct the canonical StepState for the current step."""
        env = self._require_env()

        # Build agent snapshots
        agent_snapshots = []
        for i, agent in enumerate(env.unwrapped.agents):
            agent_snapshots.append(
                AgentSnapshot(
                    name=f"agent_{i}",
                    role=f"team_{agent.index}" if hasattr(agent, "index") else None,
                    position=tuple(agent.pos) if hasattr(agent, "pos") else None,
                    orientation=str(agent.dir) if hasattr(agent, "dir") else None,
                    info={
                        "carrying": str(agent.carrying) if hasattr(agent, "carrying") and agent.carrying else None,
                        "color": agent.color if hasattr(agent, "color") else None,
                    },
                )
            )

        return StepState(
            active_agent=None,  # All agents act simultaneously
            agents=tuple(agent_snapshots),
            metrics={
                "step_count": self._step_counter,
                "num_agents": self._num_agents,
                "agent_rewards": self._agent_rewards,
            },
            environment={
                "env_id": self._env_id,
                "grid_size": f"{env.unwrapped.width}x{env.unwrapped.height}",
                "stepping_paradigm": "simultaneous",
            },
            raw=dict(info) if isinstance(info, Mapping) else {},
        )

    def get_action_meanings(self) -> List[str]:
        """Get human-readable action names.

        Returns:
            List of 7 action names
        """
        return MOSAIC_MULTIGRID_ACTIONS.copy()

    def sample_action(self) -> List[int]:
        """Sample random actions for all agents.

        Returns:
            List of random action indices
        """
        import random
        return [random.randint(0, len(MOSAIC_MULTIGRID_ACTIONS) - 1) for _ in range(self._num_agents)]

    def sample_single_action(self) -> int:
        """Sample a random action for one agent.

        Returns:
            Random action index
        """
        import random
        return random.randint(0, len(MOSAIC_MULTIGRID_ACTIONS) - 1)


# Specific adapter classes for each environment variant
class MultiGridSoccerAdapter(MultiGridAdapter):
    """Adapter for MOSAIC MultiGrid Soccer environment (4 players, 2v2)."""

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MultiGridConfig | None = None,
    ) -> None:
        if config is None:
            config = MultiGridConfig(env_id="soccer")
        super().__init__(context, config=config)


class MultiGridCollect3HAdapter(MultiGridAdapter):
    """Adapter for MOSAIC MultiGrid Collect environment (3 agents, individual).

    Maps to CollectGame3HEnv10x10N3 environment class.
    """

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MultiGridConfig | None = None,
    ) -> None:
        if config is None:
            config = MultiGridConfig(env_id="MosaicMultiGrid-Collect-v0")
        super().__init__(context, config=config)


class MultiGridCollect4HAdapter(MultiGridAdapter):
    """Adapter for MOSAIC MultiGrid Collect-2vs2 environment (4 agents, 2v2 teams).

    Maps to CollectGame4HEnv10x10N2 environment class.
    """

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MultiGridConfig | None = None,
    ) -> None:
        if config is None:
            config = MultiGridConfig(env_id="MosaicMultiGrid-Collect-2vs2-v0")
        super().__init__(context, config=config)


class MultiGridCollect1vs1Adapter(MultiGridAdapter):
    """Adapter for MOSAIC MultiGrid Collect-1vs1 deprecated environment (2 agents, 1v1).

    Maps to CollectGame2HEnv10x10N2 environment class.
    Use MultiGridCollect1vs1IndAgObsAdapter for training (natural termination, 3 balls).
    """

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MultiGridConfig | None = None,
    ) -> None:
        if config is None:
            config = MultiGridConfig(env_id="MosaicMultiGrid-Collect-1vs1-v0")
        super().__init__(context, config=config)


# IndAgObs adapter classes (v4.0.0 - Individual Agent Observations)
class MultiGridSoccerIndAgObsAdapter(MultiGridAdapter):
    """Adapter for MOSAIC MultiGrid Soccer IndAgObs environment (4 players, 2v2).

    Features: Ball respawn, first-to-2-goals, dual cooldown, 16x11 FIFA grid,
    FIFA-style rendering, ~50x faster training.
    """

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MultiGridConfig | None = None,
    ) -> None:
        if config is None:
            config = MultiGridConfig(env_id="soccer_indagobs")
        super().__init__(context, config=config)


class MultiGridCollectIndAgObsAdapter(MultiGridAdapter):
    """Adapter for MOSAIC MultiGrid Collect IndAgObs environment (3 agents, individual).

    Features: Natural termination, ~35x faster training.
    """

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MultiGridConfig | None = None,
    ) -> None:
        if config is None:
            config = MultiGridConfig(env_id="collect_indagobs")
        super().__init__(context, config=config)


class MultiGridCollect2vs2IndAgObsAdapter(MultiGridAdapter):
    """Adapter for MOSAIC MultiGrid Collect2vs2 IndAgObs environment (4 agents, 2v2).

    Features: Natural termination, 7 balls (no draws), ~35x faster training.
    """

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MultiGridConfig | None = None,
    ) -> None:
        if config is None:
            config = MultiGridConfig(env_id="collect2vs2_indagobs")
        super().__init__(context, config=config)


class MultiGridBasketballIndAgObsAdapter(MultiGridAdapter):
    """Adapter for MOSAIC MultiGrid Basketball 3vs3 IndAgObs environment (6 players, 3v3).

    Features: 19x11 court, basketball-style rendering, dunking mechanics.
    """

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MultiGridConfig | None = None,
    ) -> None:
        if config is None:
            config = MultiGridConfig(env_id="basketball_indagobs")
        super().__init__(context, config=config)


# 1vs1 adapter classes
class MultiGridCollect1vs1IndAgObsAdapter(MultiGridAdapter):
    """Adapter for MOSAIC MultiGrid Collect 1vs1 IndAgObs environment (2 agents, 1v1).

    Features: Natural termination, 3 balls (no draws), fastest training variant.
    """

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MultiGridConfig | None = None,
    ) -> None:
        if config is None:
            config = MultiGridConfig(env_id="collect_1vs1_indagobs")
        super().__init__(context, config=config)


class MultiGridSoccer1vs1IndAgObsAdapter(MultiGridAdapter):
    """Adapter for MOSAIC MultiGrid Soccer 1vs1 IndAgObs environment (2 players, 1v1).

    Features: Same FIFA grid as 2v2, no teleport passing (no teammates),
    pure individual play, fastest soccer training variant.
    """

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MultiGridConfig | None = None,
    ) -> None:
        if config is None:
            config = MultiGridConfig(env_id="soccer_1vs1_indagobs")
        super().__init__(context, config=config)


# TeamObs adapter classes (v4.0.0 - SMAC-style teammate awareness)
class MultiGridSoccerTeamObsAdapter(MultiGridAdapter):
    """Adapter for MOSAIC MultiGrid Soccer TeamObs environment (4 players, 2v2).

    Adds teammate_positions, teammate_directions, teammate_has_ball to obs.
    FIFA-style rendering.
    """

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MultiGridConfig | None = None,
    ) -> None:
        if config is None:
            config = MultiGridConfig(env_id="soccer_teamobs")
        super().__init__(context, config=config)


class MultiGridCollect2vs2TeamObsAdapter(MultiGridAdapter):
    """Adapter for MOSAIC MultiGrid Collect2vs2 TeamObs environment (4 agents, 2v2).

    Adds teammate_positions, teammate_directions, teammate_has_ball to obs.
    """

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MultiGridConfig | None = None,
    ) -> None:
        if config is None:
            config = MultiGridConfig(env_id="collect2vs2_teamobs")
        super().__init__(context, config=config)


class MultiGridBasketballTeamObsAdapter(MultiGridAdapter):
    """Adapter for MOSAIC MultiGrid Basketball 3vs3 TeamObs environment (6 players, 3v3).

    Adds teammate_positions, teammate_directions, teammate_has_ball to obs.
    Basketball court rendering.
    """

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MultiGridConfig | None = None,
    ) -> None:
        if config is None:
            config = MultiGridConfig(env_id="basketball_teamobs")
        super().__init__(context, config=config)


# Solo environment adapter classes (v6.0.0 - single-agent, no opponent)

class MultiGridSoccerSoloGreenAdapter(MultiGridAdapter):
    """Adapter for Soccer Solo Green (1 agent, scores right, no opponent)."""

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MultiGridConfig | None = None,
    ) -> None:
        if config is None:
            config = MultiGridConfig(env_id="MosaicMultiGrid-Soccer-Solo-Green-IndAgObs-v0")
        super().__init__(context, config=config)


class MultiGridSoccerSoloBlueAdapter(MultiGridAdapter):
    """Adapter for Soccer Solo Blue (1 agent, scores left, no opponent)."""

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MultiGridConfig | None = None,
    ) -> None:
        if config is None:
            config = MultiGridConfig(env_id="MosaicMultiGrid-Soccer-Solo-Blue-IndAgObs-v0")
        super().__init__(context, config=config)


class MultiGridBasketballSoloGreenAdapter(MultiGridAdapter):
    """Adapter for Basketball Solo Green (1 agent, scores right, no opponent)."""

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MultiGridConfig | None = None,
    ) -> None:
        if config is None:
            config = MultiGridConfig(env_id="MosaicMultiGrid-Basketball-Solo-Green-IndAgObs-v0")
        super().__init__(context, config=config)


class MultiGridBasketballSoloBlueAdapter(MultiGridAdapter):
    """Adapter for Basketball Solo Blue (1 agent, scores left, no opponent)."""

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MultiGridConfig | None = None,
    ) -> None:
        if config is None:
            config = MultiGridConfig(env_id="MosaicMultiGrid-Basketball-Solo-Blue-IndAgObs-v0")
        super().__init__(context, config=config)


# American Football adapter classes (v6.3.0)
class MultiGridAmericanFootball1v1Adapter(MultiGridAdapter):
    """Adapter for MOSAIC MultiGrid American Football 1v1 environment (2 players, 1v1).

    Features: 16x11 brown field, end zones, touchdown scoring, ball stealing mechanics.
    """

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MultiGridConfig | None = None,
    ) -> None:
        if config is None:
            config = MultiGridConfig(env_id="MosaicMultiGrid-AmericanFootball-1v1-v0")
        super().__init__(context, config=config)


class MultiGridAmericanFootball2v2Adapter(MultiGridAdapter):
    """Adapter for MOSAIC MultiGrid American Football 2v2 environment (4 players, 2v2).

    Features: 16x11 brown field, end zones, touchdown scoring, team-based gameplay.
    """

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MultiGridConfig | None = None,
    ) -> None:
        if config is None:
            config = MultiGridConfig(env_id="MosaicMultiGrid-AmericanFootball-2v2-v0")
        super().__init__(context, config=config)


class MultiGridAmericanFootball3v3Adapter(MultiGridAdapter):
    """Adapter for MOSAIC MultiGrid American Football 3v3 environment (6 players, 3v3).

    Features: 16x11 brown field, end zones, touchdown scoring, 3v3 team-based gameplay.
    """

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MultiGridConfig | None = None,
    ) -> None:
        if config is None:
            config = MultiGridConfig(env_id="MosaicMultiGrid-AmericanFootball-3v3-v0")
        super().__init__(context, config=config)


class MultiGridAmericanFootball2v2TeamObsAdapter(MultiGridAdapter):
    """Adapter for MOSAIC MultiGrid American Football 2v2 TeamObs environment (4 players, 2v2).

    Adds teammate_positions, teammate_directions, teammate_has_ball to obs.
    American Football field rendering.
    """

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MultiGridConfig | None = None,
    ) -> None:
        if config is None:
            config = MultiGridConfig(env_id="MosaicMultiGrid-AmericanFootball-2v2-TeamObs-v0")
        super().__init__(context, config=config)


class MultiGridAmericanFootball3v3TeamObsAdapter(MultiGridAdapter):
    """Adapter for MOSAIC MultiGrid American Football 3v3 TeamObs environment (6 players, 3v3).

    Adds teammate_positions, teammate_directions, teammate_has_ball to obs.
    American Football field rendering.
    """

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MultiGridConfig | None = None,
    ) -> None:
        if config is None:
            config = MultiGridConfig(env_id="MosaicMultiGrid-AmericanFootball-3v3-TeamObs-v0")
        super().__init__(context, config=config)


class MultiGridAmericanFootballSoloGreenAdapter(MultiGridAdapter):
    """Adapter for American Football Solo Green (1 agent, scores in end zone, no opponent)."""

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MultiGridConfig | None = None,
    ) -> None:
        if config is None:
            config = MultiGridConfig(env_id="MosaicMultiGrid-AmericanFootball-Solo-Green-v0")
        super().__init__(context, config=config)


class MultiGridAmericanFootballSoloBlueAdapter(MultiGridAdapter):
    """Adapter for American Football Solo Blue (1 agent, scores in end zone, no opponent)."""

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MultiGridConfig | None = None,
    ) -> None:
        if config is None:
            config = MultiGridConfig(env_id="MosaicMultiGrid-AmericanFootball-Solo-Blue-v0")
        super().__init__(context, config=config)


# MOSAIC MultiGrid adapter registry - competitive team-based environments from PyPI
MOSAIC_MULTIGRID_ADAPTERS: Dict[GameId, type[MultiGridAdapter]] = {
    # Original environments (deprecated - kept for backward compatibility)
    GameId.MOSAIC_MULTIGRID_SOCCER: MultiGridSoccerAdapter,
    GameId.MOSAIC_MULTIGRID_COLLECT: MultiGridCollect3HAdapter,
    GameId.MOSAIC_MULTIGRID_COLLECT2VS2: MultiGridCollect4HAdapter,
    GameId.MOSAIC_MULTIGRID_COLLECT_1VS1: MultiGridCollect1vs1Adapter,
    # IndAgObs environments (v4.0.0 - Individual Agent Observations)
    GameId.MOSAIC_MULTIGRID_SOCCER_2VS2_INDAGOBS: MultiGridSoccerIndAgObsAdapter,
    GameId.MOSAIC_MULTIGRID_SOCCER_1VS1_INDAGOBS: MultiGridSoccer1vs1IndAgObsAdapter,
    GameId.MOSAIC_MULTIGRID_COLLECT_INDAGOBS: MultiGridCollectIndAgObsAdapter,
    GameId.MOSAIC_MULTIGRID_COLLECT2VS2_INDAGOBS: MultiGridCollect2vs2IndAgObsAdapter,
    GameId.MOSAIC_MULTIGRID_COLLECT_1VS1_INDAGOBS: MultiGridCollect1vs1IndAgObsAdapter,
    GameId.MOSAIC_MULTIGRID_BASKETBALL_INDAGOBS: MultiGridBasketballIndAgObsAdapter,
    # TeamObs environments (v4.0.0 - SMAC-style teammate awareness)
    GameId.MOSAIC_MULTIGRID_SOCCER_2VS2_TEAMOBS: MultiGridSoccerTeamObsAdapter,
    GameId.MOSAIC_MULTIGRID_COLLECT2VS2_TEAMOBS: MultiGridCollect2vs2TeamObsAdapter,
    GameId.MOSAIC_MULTIGRID_BASKETBALL_TEAMOBS: MultiGridBasketballTeamObsAdapter,
    # Solo environments (v6.0.0 - single-agent, no opponent)
    GameId.MOSAIC_MULTIGRID_SOCCER_SOLO_GREEN: MultiGridSoccerSoloGreenAdapter,
    GameId.MOSAIC_MULTIGRID_SOCCER_SOLO_BLUE: MultiGridSoccerSoloBlueAdapter,
    GameId.MOSAIC_MULTIGRID_BASKETBALL_SOLO_GREEN: MultiGridBasketballSoloGreenAdapter,
    GameId.MOSAIC_MULTIGRID_BASKETBALL_SOLO_BLUE: MultiGridBasketballSoloBlueAdapter,
    # American Football environments (v6.3.0)
    GameId.MOSAIC_MULTIGRID_AMERICAN_FOOTBALL_1V1: MultiGridAmericanFootball1v1Adapter,
    GameId.MOSAIC_MULTIGRID_AMERICAN_FOOTBALL_2V2: MultiGridAmericanFootball2v2Adapter,
    GameId.MOSAIC_MULTIGRID_AMERICAN_FOOTBALL_3V3: MultiGridAmericanFootball3v3Adapter,
    GameId.MOSAIC_MULTIGRID_AMERICAN_FOOTBALL_2V2_TEAMOBS: MultiGridAmericanFootball2v2TeamObsAdapter,
    GameId.MOSAIC_MULTIGRID_AMERICAN_FOOTBALL_3V3_TEAMOBS: MultiGridAmericanFootball3v3TeamObsAdapter,
    GameId.MOSAIC_MULTIGRID_AMERICAN_FOOTBALL_SOLO_GREEN: MultiGridAmericanFootballSoloGreenAdapter,
    GameId.MOSAIC_MULTIGRID_AMERICAN_FOOTBALL_SOLO_BLUE: MultiGridAmericanFootballSoloBlueAdapter,
}

__all__ = [
    "MultiGridAdapter",
    "MultiGridSoccerAdapter",
    "MultiGridCollect3HAdapter",
    "MultiGridCollect4HAdapter",
    "MultiGridCollect1vs1Adapter",
    "MultiGridSoccerIndAgObsAdapter",
    "MultiGridSoccer1vs1IndAgObsAdapter",
    "MultiGridCollectIndAgObsAdapter",
    "MultiGridCollect2vs2IndAgObsAdapter",
    "MultiGridCollect1vs1IndAgObsAdapter",
    "MultiGridBasketballIndAgObsAdapter",
    "MultiGridSoccerTeamObsAdapter",
    "MultiGridCollect2vs2TeamObsAdapter",
    "MultiGridBasketballTeamObsAdapter",
    "MultiGridSoccerSoloGreenAdapter",
    "MultiGridSoccerSoloBlueAdapter",
    "MultiGridBasketballSoloGreenAdapter",
    "MultiGridBasketballSoloBlueAdapter",
    "MultiGridAmericanFootball1v1Adapter",
    "MultiGridAmericanFootball2v2Adapter",
    "MultiGridAmericanFootball3v3Adapter",
    "MultiGridAmericanFootball2v2TeamObsAdapter",
    "MultiGridAmericanFootball3v3TeamObsAdapter",
    "MultiGridAmericanFootballSoloGreenAdapter",
    "MultiGridAmericanFootballSoloBlueAdapter",
    "MOSAIC_MULTIGRID_ADAPTERS",
    "MOSAIC_MULTIGRID_ACTIONS",
]
