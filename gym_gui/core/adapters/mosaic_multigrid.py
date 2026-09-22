"""MOSAIC MultiGrid adapter - Competitive team-based multi-agent environments.

PyPI Package: mosaic_multigrid v7.0.0
GitHub: https://github.com/Abdulhamid97Mousa/mosaic_multigrid
PyPI: https://pypi.org/project/mosaic_multigrid/

Sports covered: Soccer (S), Basketball (BB), American Football (AF), Collect (C).

Environment naming (v7.0.0+):
    MosaicMultiGrid-<Sport>-[Team-]<Format>-[IndAgObs-]v1

    Sport:      S (Soccer) | BB (Basketball) | AF (AmericanFootball) | C (Collect)
    Team:       G (Green only) | B (Blue only) — omitted for symmetric matchups
    Format:     NvM  (e.g. 1v0, 0v1, 1v1, 2v2, 3v3, 4v4, 2v0, 0v2, 3v0, 0v3,
                     asymmetric 1v2, 2v1, 1v3, 3v1, 1v4, 4v1, 2v3, 3v2, 2v4, 4v2)
    ObsVariant: IndAgObs — omitted for solo (1v0 / 0v1) envs. TeamObs was
                removed in v7.0.0.

All environments are registered with Gymnasium as a side effect of importing
``mosaic_multigrid.envs``, so this adapter drives every variant uniformly via
``gymnasium.make()`` instead of importing individual environment classes.

Features:
- Gymnasium 1.0+ API (5-tuple dict-keyed observations)
- 8 actions (NOOP, LEFT, RIGHT, FORWARD, PICKUP, DROP, TOGGLE, DONE — noop=0 for AEC)
- view_size=3 (partial observability - competitive challenge)
- Team rewards (positive-only shared), ball passing, teleport passing, stealing mechanics
- Event tracking: goal_scored_by, passes_completed, steals_completed
- Agent position and carrying status in info dict per step
- PettingZoo AEC + Parallel API support
- FIFA-style rendering for Soccer, court rendering for Basketball, field rendering
  for American Football

Installation:
    pip install mosaic_multigrid>=7.0.0
    # or from the vendored submodule:
    pip install -e 3rd_party/environments/mosaic_multigrid

Usage:
    from gym_gui.core.adapters.mosaic_multigrid import MultiGridSoccerIndAgObsAdapter
    adapter = MultiGridSoccerIndAgObsAdapter()
    adapter.load()
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping

import numpy as np

from gym_gui.core.adapters.base import (
    AdapterContext,
    AdapterStep,
    AgentSnapshot,
    EnvironmentAdapter,
    StepState,
    WorkerCapabilities,
)
from gym_gui.core.enums import ControlMode, GameId, RenderMode, SteppingParadigm
from gym_gui.core.ui.game_config.game_configs import MultiGridConfig
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

_MOSAIC_MULTIGRID_AVAILABLE = False
if gymnasium is not None:
    try:  # pragma: no cover - import guard
        # Import mosaic_multigrid.envs module — triggers gymnasium.register()
        # side effect for all MosaicMultiGrid-* env IDs.
        import mosaic_multigrid.envs  # noqa: F401

        _MOSAIC_MULTIGRID_AVAILABLE = True
    except ImportError as _import_err:  # pragma: no cover
        _log.warning(
            "mosaic_multigrid import failed (package not installed): %s. "
            "Install with: pip install mosaic_multigrid>=7.0.0",
            _import_err,
        )


# MOSAIC multigrid action names (8 actions — noop=0 for AEC compatibility)
# Used by: Soccer, Basketball, American Football, Collect (PyPI: mosaic_multigrid v7.0.0+)
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

    This adapter handles competitive team-based environments from the
    mosaic_multigrid PyPI package. Key characteristics:
    - Multiple agents acting simultaneously (2-8 agents)
    - Team-based competitive gameplay (Soccer, Basketball, American Football: NvM;
      Collect: individual/team)
    - Gymnasium 1.0+ API (5-tuple dict-keyed observations)
    - view_size=3 for partial observability (competitive challenge)
    - 8 actions (noop=0, left=1 … done=7 — noop for AEC compatibility)

    The environment provides:
    - Observations: Dict of encoded grid views per agent
    - Actions: Dict of discrete actions (0-7) per agent
    - Rewards: Dict of float rewards per agent
    - Terminated/Truncated: Dict per agent + __all__ flag

    All environment variants are driven via ``gymnasium.make(env_id, ...)``
    since ``mosaic_multigrid.envs`` registers every variant with Gymnasium.
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
        max_agents=8,  # v7 supports symmetric 4v4 variants
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
        """Return the Gymnasium environment identifier."""
        return self._env_id

    @property
    def num_agents(self) -> int:
        """Return the number of agents in the environment."""
        return self._num_agents

    def load(self) -> None:
        """Instantiate the MOSAIC MultiGrid environment via gymnasium.make()."""
        if gymnasium is None:
            raise RuntimeError(
                "gymnasium package not installed. "
                "Install with: pip install gymnasium"
            )
        if not _MOSAIC_MULTIGRID_AVAILABLE:
            raise RuntimeError(
                "mosaic_multigrid not installed. "
                "Install with: pip install mosaic_multigrid>=7.0.0"
            )

        # Build optional kwargs (view_size override from config panel)
        extra_kwargs: Dict[str, Any] = {}
        if self._config.view_size is not None:
            extra_kwargs["view_size"] = self._config.view_size

        try:
            env = gymnasium.make(self._env_id, render_mode="rgb_array", **extra_kwargs)

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
            vis_parts = []
            for agent_idx, sightings in sorted(visibility.items()):
                vis_parts.append(f"agent_{agent_idx}: {', '.join(sightings)}")
            vis_text = " | ".join(vis_parts)
            self.log_constant(
                LOG_MOSAIC_MULTIGRID_VISIBILITY,
                message=f"IndAgObs step {self._step_counter + 1} | {vis_text}",
                extra={
                    "env_id": self._env_id,
                    "step": self._step_counter + 1,
                    "obs_model": "IndAgObs",
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

        Under the IndAgObs observation model (the only surviving model
        in mosaic_multigrid v7.0.0), an agent only knows about others if
        they appear in its 3x3 local view. Sightings are prefixed with
        ``[view]``.

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

        visibility: Dict[int, List[str]] = {}
        for i, obs in enumerate(self._agent_observations):
            if not isinstance(obs, dict) or "image" not in obs:
                continue

            image = obs["image"]
            observer_team = self._team_map.get(i, 0)
            sightings: List[str] = []

            # --- 3x3 view scan ---
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
            List of 8 action names
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


# ---------------------------------------------------------------------------
# Specific adapter classes for each environment variant
#
# All variants share MultiGridAdapter's load()/reset()/step() logic and just
# differ by which env_id (a MosaicMultiGrid-* Gymnasium ID) they default to.
# ---------------------------------------------------------------------------

class _EnvIdAdapter(MultiGridAdapter):
    """Base class for adapters that only need to fix a default env_id."""

    _default_env_id: str = ""

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MultiGridConfig | None = None,
    ) -> None:
        if config is None:
            config = MultiGridConfig(env_id=self._default_env_id)
        super().__init__(context, config=config)


# Soccer (S)
class MultiGridSoccerIndAgObsAdapter(_EnvIdAdapter):
    """Soccer 2v2 IndAgObs (4 players). Ball respawn, first-to-2-goals, 16x11 FIFA grid."""
    _default_env_id = "MosaicMultiGrid-S-2v2-IndAgObs-v1"


class MultiGridSoccer1vs1IndAgObsAdapter(_EnvIdAdapter):
    """Soccer 1v1 IndAgObs (2 players). Same FIFA grid, no teleport passing."""
    _default_env_id = "MosaicMultiGrid-S-1v1-IndAgObs-v1"


class MultiGridSoccerSoloGreenAdapter(_EnvIdAdapter):
    """Soccer Solo Green (1 agent, scores right, no opponent)."""
    _default_env_id = "MosaicMultiGrid-S-G-1v0-v1"


class MultiGridSoccerSoloBlueAdapter(_EnvIdAdapter):
    """Soccer Solo Blue (1 agent, scores left, no opponent)."""
    _default_env_id = "MosaicMultiGrid-S-B-0v1-v1"


# Basketball (BB)
class MultiGridBasketballIndAgObsAdapter(_EnvIdAdapter):
    """Basketball 3v3 IndAgObs (6 players). 19x11 court, dunking mechanics."""
    _default_env_id = "MosaicMultiGrid-BB-3v3-IndAgObs-v1"


class MultiGridBasketballSoloGreenAdapter(_EnvIdAdapter):
    """Basketball Solo Green (1 agent, scores right, no opponent)."""
    _default_env_id = "MosaicMultiGrid-BB-G-1v0-v1"


class MultiGridBasketballSoloBlueAdapter(_EnvIdAdapter):
    """Basketball Solo Blue (1 agent, scores left, no opponent)."""
    _default_env_id = "MosaicMultiGrid-BB-B-0v1-v1"


# American Football (AF)
class MultiGridAmericanFootball1v1Adapter(_EnvIdAdapter):
    """American Football 1v1 (2 players). 16x11 field, end zones, touchdown scoring."""
    _default_env_id = "MosaicMultiGrid-AF-1v1-IndAgObs-v1"


class MultiGridAmericanFootball2v2Adapter(_EnvIdAdapter):
    """American Football 2v2 (4 players). 16x11 field, team-based gameplay."""
    _default_env_id = "MosaicMultiGrid-AF-2v2-IndAgObs-v1"


class MultiGridAmericanFootball3v3Adapter(_EnvIdAdapter):
    """American Football 3v3 (6 players). 16x11 field, 3v3 team-based gameplay."""
    _default_env_id = "MosaicMultiGrid-AF-3v3-IndAgObs-v1"


class MultiGridAmericanFootballSoloGreenAdapter(_EnvIdAdapter):
    """American Football Solo Green (1 agent, scores in end zone, no opponent)."""
    _default_env_id = "MosaicMultiGrid-AF-G-1v0-v1"


class MultiGridAmericanFootballSoloBlueAdapter(_EnvIdAdapter):
    """American Football Solo Blue (1 agent, scores in end zone, no opponent)."""
    _default_env_id = "MosaicMultiGrid-AF-B-0v1-v1"


# Collect (C)
class MultiGridCollectIndAgObsAdapter(_EnvIdAdapter):
    """Collect (3 agents, individual). Natural termination, ~35x faster training."""
    _default_env_id = "MosaicMultiGrid-C-IndAgObs-v1"


class MultiGridCollect2vs2IndAgObsAdapter(_EnvIdAdapter):
    """Collect 2v2 IndAgObs (4 agents). Natural termination, 7 balls (no draws)."""
    _default_env_id = "MosaicMultiGrid-C-2v2-IndAgObs-v1"


class MultiGridCollect1vs1IndAgObsAdapter(_EnvIdAdapter):
    """Collect 1v1 IndAgObs (2 agents). Natural termination, 3 balls, fastest variant."""
    _default_env_id = "MosaicMultiGrid-C-1v1-IndAgObs-v1"


# MOSAIC MultiGrid adapter registry — v7.0.0 (no TeamObs)
MOSAIC_MULTIGRID_ADAPTERS: Dict[GameId, type[MultiGridAdapter]] = {
    # Soccer
    GameId.MOSAIC_MULTIGRID_S_1V1_INDAGOBS: MultiGridSoccer1vs1IndAgObsAdapter,
    GameId.MOSAIC_MULTIGRID_S_2V2_INDAGOBS: MultiGridSoccerIndAgObsAdapter,
    GameId.MOSAIC_MULTIGRID_S_G_1V0: MultiGridSoccerSoloGreenAdapter,
    GameId.MOSAIC_MULTIGRID_S_B_0V1: MultiGridSoccerSoloBlueAdapter,
    # Basketball
    GameId.MOSAIC_MULTIGRID_BB_3V3_INDAGOBS: MultiGridBasketballIndAgObsAdapter,
    GameId.MOSAIC_MULTIGRID_BB_G_1V0: MultiGridBasketballSoloGreenAdapter,
    GameId.MOSAIC_MULTIGRID_BB_B_0V1: MultiGridBasketballSoloBlueAdapter,
    # American Football
    GameId.MOSAIC_MULTIGRID_AF_1V1_INDAGOBS: MultiGridAmericanFootball1v1Adapter,
    GameId.MOSAIC_MULTIGRID_AF_2V2_INDAGOBS: MultiGridAmericanFootball2v2Adapter,
    GameId.MOSAIC_MULTIGRID_AF_3V3_INDAGOBS: MultiGridAmericanFootball3v3Adapter,
    GameId.MOSAIC_MULTIGRID_AF_G_1V0: MultiGridAmericanFootballSoloGreenAdapter,
    GameId.MOSAIC_MULTIGRID_AF_B_0V1: MultiGridAmericanFootballSoloBlueAdapter,
    # Collect
    GameId.MOSAIC_MULTIGRID_C_INDAGOBS: MultiGridCollectIndAgObsAdapter,
    GameId.MOSAIC_MULTIGRID_C_2V2_INDAGOBS: MultiGridCollect2vs2IndAgObsAdapter,
    GameId.MOSAIC_MULTIGRID_C_1V1_INDAGOBS: MultiGridCollect1vs1IndAgObsAdapter,
}


# ---------------------------------------------------------------------------
# Full-matrix parity (v7.0.0)
# ---------------------------------------------------------------------------
# mosaic_multigrid v7.0.0 registers 81 MosaicMultiGrid-* IDs with Gymnasium.
# The subset above has dedicated adapter classes; the rest are pointed at the
# base MultiGridAdapter, which drives every registered ID uniformly via
# gymnasium.make(). The GUI passes MultiGridConfig(env_id=game_id.value) into
# the adapter, and the no-config path in create_adapter derives the same
# env_id from the GameId. No new adapter class is needed per variant.
#
# Variants that lack a dedicated adapter class: Soccer 3v3, Basketball 1v1/2v2,
# the one-sided cooperative Nv0 / 0vN sets across all three sports (2v0..6v0),
# the 4v4 symmetric competitive variants, and the 30 new asymmetric variants
# added in v7.0.0 (1v2, 2v1, 1v3, 3v1, 1v4, 4v1, 2v3, 3v2, 2v4, 4v2 across
# Basketball / American Football / Soccer).
_MOSAIC_FULL_MATRIX_IDS: tuple[GameId, ...] = (
    # Soccer
    GameId.MOSAIC_MULTIGRID_S_3V3_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_S_G_2V0_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_S_G_3V0_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_S_B_0V2_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_S_B_0V3_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_S_G_4V0_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_S_B_0V4_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_S_G_5V0_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_S_B_0V5_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_S_G_6V0_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_S_B_0V6_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_S_4V4_INDAGOBS,
    # Soccer asymmetric competitive (v7.0.0)
    GameId.MOSAIC_MULTIGRID_S_1V2_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_S_2V1_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_S_1V3_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_S_3V1_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_S_1V4_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_S_4V1_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_S_2V3_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_S_3V2_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_S_2V4_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_S_4V2_INDAGOBS,
    # Basketball
    GameId.MOSAIC_MULTIGRID_BB_1V1_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_BB_2V2_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_BB_G_2V0_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_BB_G_3V0_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_BB_B_0V2_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_BB_B_0V3_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_BB_G_4V0_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_BB_B_0V4_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_BB_G_5V0_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_BB_B_0V5_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_BB_G_6V0_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_BB_B_0V6_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_BB_4V4_INDAGOBS,
    # Basketball asymmetric competitive (v7.0.0)
    GameId.MOSAIC_MULTIGRID_BB_1V2_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_BB_2V1_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_BB_1V3_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_BB_3V1_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_BB_1V4_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_BB_4V1_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_BB_2V3_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_BB_3V2_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_BB_2V4_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_BB_4V2_INDAGOBS,
    # American Football
    GameId.MOSAIC_MULTIGRID_AF_G_2V0_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_AF_G_3V0_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_AF_B_0V2_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_AF_B_0V3_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_AF_G_4V0_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_AF_B_0V4_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_AF_G_5V0_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_AF_B_0V5_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_AF_G_6V0_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_AF_B_0V6_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_AF_4V4_INDAGOBS,
    # American Football asymmetric competitive (v7.0.0)
    GameId.MOSAIC_MULTIGRID_AF_1V2_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_AF_2V1_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_AF_1V3_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_AF_3V1_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_AF_1V4_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_AF_4V1_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_AF_2V3_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_AF_3V2_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_AF_2V4_INDAGOBS,
    GameId.MOSAIC_MULTIGRID_AF_4V2_INDAGOBS,
)

for _gid in _MOSAIC_FULL_MATRIX_IDS:
    MOSAIC_MULTIGRID_ADAPTERS.setdefault(_gid, MultiGridAdapter)


__all__ = [
    "MultiGridAdapter",
    "MultiGridSoccerIndAgObsAdapter",
    "MultiGridSoccer1vs1IndAgObsAdapter",
    "MultiGridSoccerSoloGreenAdapter",
    "MultiGridSoccerSoloBlueAdapter",
    "MultiGridBasketballIndAgObsAdapter",
    "MultiGridBasketballSoloGreenAdapter",
    "MultiGridBasketballSoloBlueAdapter",
    "MultiGridAmericanFootball1v1Adapter",
    "MultiGridAmericanFootball2v2Adapter",
    "MultiGridAmericanFootball3v3Adapter",
    "MultiGridAmericanFootballSoloGreenAdapter",
    "MultiGridAmericanFootballSoloBlueAdapter",
    "MultiGridCollectIndAgObsAdapter",
    "MultiGridCollect2vs2IndAgObsAdapter",
    "MultiGridCollect1vs1IndAgObsAdapter",
    "MOSAIC_MULTIGRID_ADAPTERS",
    "MOSAIC_MULTIGRID_ACTIONS",
]
