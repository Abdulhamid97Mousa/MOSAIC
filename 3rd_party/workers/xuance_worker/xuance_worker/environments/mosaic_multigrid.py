"""MultiGrid environment wrapper for XuanCe multi-agent training.

This module provides XuanCe-compatible wrappers for MOSAIC multigrid environments
with flexible training mode support: competitive, cooperative, or independent.

MOSAIC multigrid: Modernized multi-agent gridworld package
Location: 3rd_party/environments/mosaic_multigrid/

Training Modes:
---------------
- COMPETITIVE: Per-team policies (uses environment's team structure)
  → XuanCe uses runner_competition.py
  → Example: Soccer [1,1,2,2] → 2 policies (Green team, Blue team)

- COOPERATIVE: All agents share one policy
  → XuanCe uses runner_basic.py with parameter sharing
  → Example: Soccer [1,1,2,2] → 1 policy (all 4 agents share)

- INDEPENDENT: Each agent has its own policy
  → XuanCe uses runner_competition.py with n groups of 1
  → Example: Soccer [1,1,2,2] → 4 policies (one per agent)

Configuration:
--------------
    config.training_mode = "competitive"  # or "cooperative" or "independent"

Example:
--------
    config = SimpleNamespace(
        env_name="multigrid",
        env_id="soccer",
        env_seed=42,
        training_mode="competitive",  # Per-team policies
    )
    env = MultiGrid_Env(config)

    # For cooperative (shared policy):
    config.training_mode = "cooperative"
    env = MultiGrid_Env(config)
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
from gymnasium import spaces

_logger = logging.getLogger(__name__)

try:
    from xuance.environment import RawMultiAgentEnv
except ImportError:
    RawMultiAgentEnv = object  # Fallback for type hints

try:
    from gym_gui.core.wrappers import ReproducibleMultiGridWrapper
except ImportError:
    ReproducibleMultiGridWrapper = None

# FastLane integration for visualization during training
try:
    from xuance_worker.fastlane import maybe_wrap_env, is_fastlane_enabled
except ImportError:
    maybe_wrap_env = None
    is_fastlane_enabled = lambda: False


# =============================================================================
# Gymnasium Compatibility Wrapper
# =============================================================================

# Import gymnasium for proper Env inheritance
try:
    import gymnasium as _gymnasium
    _HAS_GYMNASIUM = True
except ImportError:
    _gymnasium = None
    _HAS_GYMNASIUM = False


class GymToGymnasiumWrapper(_gymnasium.Env if _HAS_GYMNASIUM else object):
    """Compatibility wrapper: normalises Gym/Gymnasium API differences.

    mosaic_multigrid v7.0.0 uses Gymnasium API natively, but this wrapper
    still handles both old Gym (4-tuple step, single-value reset) and
    Gymnasium (5-tuple step, 2-tuple reset) transparently so older env
    classes continue to work.

    FastLane expects the Gymnasium API:
        reset() -> (obs, info)
        step()  -> (obs, reward, terminated, truncated, info)

    IMPORTANT: Inherits from gymnasium.Env so it passes isinstance() checks
    required by gymnasium.Wrapper (used by FastLaneTelemetryWrapper).
    """

    # Gymnasium metadata
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, env: Any) -> None:
        self._env = env
        self._step_count = 0
        # Get max_steps from env if available
        self._max_steps = getattr(env, 'max_steps', 10000)

        # Copy spaces from wrapped env (required by gymnasium.Env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        # Set render_mode if available
        self.render_mode = getattr(env, 'render_mode', None)

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to wrapped environment."""
        # Avoid infinite recursion for _env
        if name == '_env':
            raise AttributeError(name)
        return getattr(self._env, name)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Any, Dict[str, Any]]:
        """Reset returning Gymnasium-style 2-tuple.

        Auto-detects whether the wrapped env uses old Gym API (returns obs)
        or Gymnasium API (returns (obs, info) tuple) and normalises to the
        Gymnasium 2-tuple format.
        """
        self._step_count = 0

        # Try Gymnasium-style reset (seed as keyword argument)
        try:
            result = self._env.reset(seed=seed)
        except TypeError:
            # Old Gym API: seed() + reset() separately
            if seed is not None and hasattr(self._env, 'seed'):
                self._env.seed(seed)
            result = self._env.reset()

        # If already Gymnasium format (obs, info) tuple, pass through
        if isinstance(result, tuple) and len(result) == 2:
            return result

        # Old Gym format: wrap as (obs, {})
        return result, {}

    def step(self, action: Any) -> Tuple[Any, Any, bool, bool, Dict[str, Any]]:
        """Step returning Gymnasium-style 5-tuple.

        Converts from old Gym 4-tuple:
            (obs, reward, done, info)
        To new Gymnasium 5-tuple:
            (obs, reward, terminated, truncated, info)
        """
        result = self._env.step(action)

        if len(result) == 4:
            obs, reward, done, info = result
            # Determine if terminated (episode end) or truncated (max steps)
            self._step_count += 1
            truncated = self._step_count >= self._max_steps and not done
            terminated = done
        else:
            # Already 5-tuple (shouldn't happen but handle gracefully)
            obs, reward, terminated, truncated, info = result

        return obs, reward, terminated, truncated, info

    def render(self) -> Any:
        """Render the environment."""
        return self._env.render()

    def close(self) -> None:
        """Close the environment."""
        if hasattr(self._env, 'close'):
            self._env.close()

    @property
    def unwrapped(self) -> Any:
        """Return the base unwrapped environment."""
        return getattr(self._env, 'unwrapped', self._env)


# =============================================================================
# Training Mode Enumeration
# =============================================================================

class TrainingMode(str, Enum):
    """Training mode determines how agent policies are structured.

    COMPETITIVE: Per-team policies based on environment's team structure.
        - Soccer: 2 policies (Green team shares π_A, Blue team shares π_B)
        - Uses runner_competition.py in XuanCe
        - Best for: Adversarial games, team vs team scenarios

    COOPERATIVE: All agents share a single policy.
        - Soccer: 1 policy (all 4 agents share π)
        - Uses runner_basic.py with parameter sharing
        - Best for: Cooperative tasks, homogeneous agents

    INDEPENDENT: Each agent has its own separate policy.
        - Soccer: 4 policies (π_0, π_1, π_2, π_3)
        - Uses runner_competition.py with n groups of 1
        - Best for: Heterogeneous agents, individual learning
    """
    COMPETITIVE = "competitive"
    COOPERATIVE = "cooperative"
    INDEPENDENT = "independent"


# =============================================================================
# Environment Registry
# =============================================================================

# Lazy-loaded environment classes to avoid import errors
_ENV_CLASSES_CACHE: Dict[str, Optional[Type]] = {}


def _get_env_class(env_id: str) -> Optional[Type]:
    """Lazily load and cache environment classes.

    This prevents import errors if mosaic_multigrid is not installed.
    """
    if env_id in _ENV_CLASSES_CACHE:
        return _ENV_CLASSES_CACHE[env_id]

    env_cls = None

    # Try to import from mosaic_multigrid
    try:
        if env_id.lower() in ("soccer", "soccergame4henv10x15n2"):
            from mosaic_multigrid.envs import SoccerGame4HEnv10x15N2
            env_cls = SoccerGame4HEnv10x15N2

        elif env_id.lower() in ("soccer_1vs1", "soccergame2hindagobsenv16x11n2"):
            from mosaic_multigrid.envs import SoccerGame2HIndAgObsEnv16x11N2
            env_cls = SoccerGame2HIndAgObsEnv16x11N2

        elif env_id.lower() in ("collect", "collectgame4henv10x10n2"):
            from mosaic_multigrid.envs import CollectGame4HEnv10x10N2
            env_cls = CollectGame4HEnv10x10N2

        elif env_id.lower() in ("collect_1vs1", "collectgame2hindagobsenv10x10n2"):
            from mosaic_multigrid.envs import CollectGame2HIndAgObsEnv10x10N2
            env_cls = CollectGame2HIndAgObsEnv10x10N2

        # --- 2vs2 IndAgObs variants ---
        elif env_id.lower() in ("soccer_2vs2_indagobs", "soccergame4hindagobsenv16x11n2"):
            from mosaic_multigrid.envs import SoccerGame4HIndAgObsEnv16x11N2
            env_cls = SoccerGame4HIndAgObsEnv16x11N2

        elif env_id.lower() in ("collect_2vs2_indagobs", "collectgame4hindagobsenv10x10n2"):
            from mosaic_multigrid.envs import CollectGame4HIndAgObsEnv10x10N2
            env_cls = CollectGame4HIndAgObsEnv10x10N2

        # --- 3vs3 IndAgObs variants ---
        elif env_id.lower() in ("basketball_3vs3_indagobs", "basketballgame6hindagobsenv19x11n3"):
            from mosaic_multigrid.envs import BasketballGame6HIndAgObsEnv19x11N3
            env_cls = BasketballGame6HIndAgObsEnv19x11N3

        # --- TeamObs variants (IndAgObs + teammate features) ---
        elif env_id.lower() in ("soccer_2vs2_teamobs", "soccerteamobsenv"):
            from mosaic_multigrid.envs import SoccerTeamObsEnv
            env_cls = SoccerTeamObsEnv

        elif env_id.lower() in ("collect_2vs2_teamobs", "collect2vs2teamobsenv"):
            from mosaic_multigrid.envs import Collect2vs2TeamObsEnv
            env_cls = Collect2vs2TeamObsEnv

        elif env_id.lower() in ("basketball_3vs3_teamobs", "basketball3vs3teamobsenv"):
            from mosaic_multigrid.envs import Basketball3vs3TeamObsEnv
            env_cls = Basketball3vs3TeamObsEnv

        # --- Solo variants (v6.0.0, single-agent, no opponent) ---
        elif env_id.lower() in ("soccer_solo_green", "soccersologreenindagobsenv16x11"):
            from mosaic_multigrid.envs import SoccerSoloGreenIndAgObsEnv16x11
            env_cls = SoccerSoloGreenIndAgObsEnv16x11

        elif env_id.lower() in ("soccer_solo_blue", "soccersoloblueindagobsenv16x11"):
            from mosaic_multigrid.envs import SoccerSoloBlueIndAgObsEnv16x11
            env_cls = SoccerSoloBlueIndAgObsEnv16x11

        elif env_id.lower() in ("basketball_solo_green", "basketballsologreenindagobsenv19x11"):
            from mosaic_multigrid.envs import BasketballSoloGreenIndAgObsEnv19x11
            env_cls = BasketballSoloGreenIndAgObsEnv19x11

        elif env_id.lower() in ("basketball_solo_blue", "basketballsoloblueindagobsenv19x11"):
            from mosaic_multigrid.envs import BasketballSoloBlueIndAgObsEnv19x11
            env_cls = BasketballSoloBlueIndAgObsEnv19x11

    except ImportError as e:
        _logger.warning(f"Could not import mosaic_multigrid environment '{env_id}': {e}")

    _ENV_CLASSES_CACHE[env_id] = env_cls
    return env_cls


def get_available_environments() -> List[str]:
    """Return short IDs backed by mosaic_multigrid v7 environments."""
    available = []
    for env_id in [
        "soccer_1vs1", "collect_1vs1",
        "soccer_2vs2_indagobs", "collect_2vs2_indagobs",
        "basketball_3vs3_indagobs",
    ]:
        if _get_env_class(env_id) is not None:
            available.append(env_id)
    return available


# Environment metadata: describes team structure and recommended training modes
MULTIGRID_ENV_INFO = {
    "soccer": {
        "full_name": "SoccerGame4HEnv10x15N2",
        "description": "2v2 Soccer - score in opponent's goal",
        "num_agents": 4,
        "default_teams": [[0, 1], [2, 3]],  # Green vs Blue
        "team_names": ["green", "blue"],
        "recommended_mode": TrainingMode.COMPETITIVE,
        "zero_sum": True,
    },
    "soccer_1vs1": {
        "full_name": "SoccerGame2HIndAgObsEnv16x11N2",
        "description": "1v1 Soccer - IndAgObs, 16x11 FIFA grid, first-to-2-goals",
        "num_agents": 2,
        "default_teams": [[0], [1]],  # Green vs Blue (1 agent each)
        "team_names": ["green", "blue"],
        "recommended_mode": TrainingMode.COMPETITIVE,
        "zero_sum": False,
    },
    "collect": {
        "full_name": "CollectGame4HEnv10x10N2",
        "description": "3-player ball collection - free-for-all",
        "num_agents": 3,
        "default_teams": [[0], [1], [2]],  # Each agent is their own "team"
        "team_names": ["agent_0", "agent_1", "agent_2"],
        "recommended_mode": TrainingMode.INDEPENDENT,
        "zero_sum": True,
    },
    "collect_1vs1": {
        "full_name": "CollectGame2HIndAgObsEnv10x10N2",
        "description": "1v1 ball collection - IndAgObs, 10x10, 3 balls, zero-sum",
        "num_agents": 2,
        "default_teams": [[0], [1]],
        "team_names": ["green", "red"],
        "recommended_mode": TrainingMode.COMPETITIVE,
        "zero_sum": True,
    },
    # --- 2vs2 IndAgObs variants ---
    "soccer_2vs2_indagobs": {
        "full_name": "SoccerGame4HIndAgObsEnv16x11N2",
        "description": "2v2 Soccer - IndAgObs, 16x11 FIFA grid, ball respawn",
        "num_agents": 4,
        "default_teams": [[0, 1], [2, 3]],
        "team_names": ["green", "blue"],
        "recommended_mode": TrainingMode.COMPETITIVE,
        "zero_sum": True,
    },
    "collect_2vs2_indagobs": {
        "full_name": "CollectGame4HIndAgObsEnv10x10N2",
        "description": "2v2 ball collection - IndAgObs, 10x10, 7 balls",
        "num_agents": 4,
        "default_teams": [[0, 1], [2, 3]],
        "team_names": ["green", "red"],
        "recommended_mode": TrainingMode.COMPETITIVE,
        "zero_sum": True,
    },
    # --- 3vs3 IndAgObs variants ---
    "basketball_3vs3_indagobs": {
        "full_name": "BasketballGame6HIndAgObsEnv19x11N3",
        "description": "3v3 Basketball - IndAgObs, 19x11 court, score in opponent hoop",
        "num_agents": 6,
        "default_teams": [[0, 1, 2], [3, 4, 5]],
        "team_names": ["green", "blue"],
        "recommended_mode": TrainingMode.COMPETITIVE,
        "zero_sum": True,
    },
    # --- TeamObs variants (IndAgObs + teammate features) ---
    "soccer_2vs2_teamobs": {
        "full_name": "SoccerTeamObsEnv",
        "description": "2v2 Soccer - TeamObs (image + teammate positions/directions/ball)",
        "num_agents": 4,
        "default_teams": [[0, 1], [2, 3]],
        "team_names": ["green", "blue"],
        "recommended_mode": TrainingMode.COMPETITIVE,
        "zero_sum": True,
        "obs_type": "teamobs",
    },
    "collect_2vs2_teamobs": {
        "full_name": "Collect2vs2TeamObsEnv",
        "description": "2v2 Collect - TeamObs (image + teammate positions/directions/ball)",
        "num_agents": 4,
        "default_teams": [[0, 1], [2, 3]],
        "team_names": ["green", "red"],
        "recommended_mode": TrainingMode.COMPETITIVE,
        "zero_sum": True,
        "obs_type": "teamobs",
    },
    "basketball_3vs3_teamobs": {
        "full_name": "Basketball3vs3TeamObsEnv",
        "description": "3v3 Basketball - TeamObs (image + teammate positions/directions/ball)",
        "num_agents": 6,
        "default_teams": [[0, 1, 2], [3, 4, 5]],
        "team_names": ["green", "blue"],
        "recommended_mode": TrainingMode.COMPETITIVE,
        "zero_sum": True,
        "obs_type": "teamobs",
    },
    # --- Solo variants (v6.0.0, single-agent, no opponent) ---
    "soccer_solo_green": {
        "full_name": "SoccerSoloGreenIndAgObsEnv16x11",
        "description": "Solo Soccer (Green) - score in blue goal, no opponent",
        "num_agents": 1,
        "default_teams": [[0]],
        "team_names": ["green"],
        "recommended_mode": TrainingMode.INDEPENDENT,
        "zero_sum": False,
        "solo": True,
    },
    "soccer_solo_blue": {
        "full_name": "SoccerSoloBlueIndAgObsEnv16x11",
        "description": "Solo Soccer (Blue) - score in green goal, no opponent",
        "num_agents": 1,
        "default_teams": [[0]],
        "team_names": ["blue"],
        "recommended_mode": TrainingMode.INDEPENDENT,
        "zero_sum": False,
        "solo": True,
    },
    "basketball_solo_green": {
        "full_name": "BasketballSoloGreenIndAgObsEnv19x11",
        "description": "Solo Basketball (Green) - score in blue hoop, no opponent",
        "num_agents": 1,
        "default_teams": [[0]],
        "team_names": ["green"],
        "recommended_mode": TrainingMode.INDEPENDENT,
        "zero_sum": False,
        "solo": True,
    },
    "basketball_solo_blue": {
        "full_name": "BasketballSoloBlueIndAgObsEnv19x11",
        "description": "Solo Basketball (Blue) - score in green hoop, no opponent",
        "num_agents": 1,
        "default_teams": [[0]],
        "team_names": ["blue"],
        "recommended_mode": TrainingMode.INDEPENDENT,
        "zero_sum": False,
        "solo": True,
    },
}


# =============================================================================
# Solo Environment Wrapper (single-agent PPO)
# =============================================================================

class SoloMultiGrid_Env(_gymnasium.Env if _HAS_GYMNASIUM else object):
    """Standard Gymnasium wrapper for solo mosaic_multigrid environments.

    Solo environments (v6.0.0) have 1 agent with no opponent. They use the
    multi-agent API internally ({0: obs}, {0: reward}, etc.) but this wrapper
    converts them to standard single-agent Gymnasium API for use with XuanCe's
    PPO/DRL runner.

    Observation: 3x3x3 IndAgObs image → flattened to (27,) float32
    Action: Discrete(8) — left, right, forward, pickup, drop, toggle, done, still

    Example:
        config = SimpleNamespace(env_id="soccer_solo_green")
        env = SoloMultiGrid_Env(config)
        obs, info = env.reset()  # obs: np.ndarray shape (27,)
        obs, reward, terminated, truncated, info = env.step(3)
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: Any) -> None:
        env_id = getattr(config, 'env_id', 'soccer_solo_green')
        env_seed = getattr(config, 'env_seed', None)
        self.render_mode = getattr(config, 'render_mode', None)

        env_cls = _get_env_class(env_id)
        if env_cls is None:
            raise ValueError(
                f"Unknown solo MultiGrid environment: '{env_id}'. "
                f"Available solo envs: soccer_solo_green, soccer_solo_blue, "
                f"basketball_solo_green, basketball_solo_blue."
            )

        # FastLane render mode
        fastlane_active = maybe_wrap_env is not None and is_fastlane_enabled()
        render_mode = "rgb_array" if fastlane_active else None
        self._inner = env_cls(render_mode=render_mode)

        self._seed = env_seed
        self._base = getattr(self._inner, 'unwrapped', self._inner)

        # Single-agent: extract obs/action spaces from agent 0
        inner_obs_space = self._inner.observation_space[0]
        img_space = inner_obs_space['image']
        self._obs_flat_dim = int(np.prod(img_space.shape))

        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(self._obs_flat_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self._inner.action_space[0].n)

        # Wrap with FastLane if enabled
        if maybe_wrap_env is not None and fastlane_active:
            self._inner = GymToGymnasiumWrapper(self._inner)
            self._inner = maybe_wrap_env(self._inner)
            _logger.info("Applied FastLane wrapper for solo training visualization")

        self.max_episode_steps = getattr(self._base, 'max_steps', 200)
        self._episode_step = 0
        self.env_id = env_id

        _logger.info(
            "SoloMultiGrid_Env initialized: env=%s, obs_dim=%d, actions=%d",
            env_id, self._obs_flat_dim, self.action_space.n,
        )

    def _flatten_obs(self, multi_obs: dict) -> np.ndarray:
        """Convert {0: {'image': (3,3,3), ...}} → flat (27,) float32."""
        agent_obs = multi_obs[0]
        img = agent_obs['image']
        return img.flatten().astype(np.float32)

    def reset(self, seed=None, options=None):
        kwargs = {}
        if seed is not None:
            kwargs['seed'] = seed
        elif self._seed is not None:
            kwargs['seed'] = self._seed

        result = self._inner.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs, info = result, {}

        self._episode_step = 0
        return self._flatten_obs(obs), info

    def step(self, action):
        # Solo env expects list of actions: [action_for_agent_0]
        result = self._inner.step([int(action)])
        obs, rewards, terminated, truncated, info = result

        self._episode_step += 1

        # Extract scalar values from agent-0 dicts
        flat_obs = self._flatten_obs(obs)
        reward = rewards[0] if isinstance(rewards, dict) else float(rewards)
        done = terminated[0] if isinstance(terminated, dict) else bool(terminated)
        trunc = truncated[0] if isinstance(truncated, dict) else bool(truncated)
        step_info = info.get(0, info) if isinstance(info, dict) else info

        return flat_obs, float(reward), bool(done), bool(trunc), step_info

    def render(self):
        return self._inner.render()

    def close(self):
        if hasattr(self._inner, 'close'):
            self._inner.close()

    @property
    def unwrapped(self):
        return self._base


# =============================================================================
# Main Environment Wrapper (multi-agent)
# =============================================================================

class MultiGrid_Env(RawMultiAgentEnv):
    """XuanCe-compatible wrapper for mosaic_multigrid environments.

    This wrapper converts mosaic_multigrid's list-based multi-agent API to
    XuanCe's dict-based API, with flexible training mode support.

    Attributes:
        env: The wrapped mosaic_multigrid environment
        agents: List of agent IDs (e.g., ['agent_0', 'agent_1', ...])
        training_mode: How policies are structured (competitive/cooperative/independent)
        agent_groups: List of agent groups (structure depends on training_mode)
        observation_space: Dict mapping agent ID to observation space
        action_space: Dict mapping agent ID to action space

    Example:
        # Competitive (per-team policies)
        config = SimpleNamespace(env_id="soccer", training_mode="competitive")
        env = MultiGrid_Env(config)
        print(env.groups_info['num_groups'])  # 2 (Green team, Blue team)

        # Cooperative (shared policy)
        config = SimpleNamespace(env_id="soccer", training_mode="cooperative")
        env = MultiGrid_Env(config)
        print(env.groups_info['num_groups'])  # 1 (all agents share)

        # Independent (per-agent policies)
        config = SimpleNamespace(env_id="soccer", training_mode="independent")
        env = MultiGrid_Env(config)
        print(env.groups_info['num_groups'])  # 4 (one per agent)
    """

    def __init__(self, config: Any) -> None:
        """Initialize the MultiGrid environment wrapper.

        Args:
            config: Configuration object with attributes:
                - env_name: Environment family name (default: "multigrid")
                - env_id: Specific environment (e.g., "soccer", "collect")
                - env_seed: Random seed for reproducibility (optional)
                - render_mode: Render mode ("human", "rgb_array", or None)
                - training_mode: Policy structure - "competitive", "cooperative",
                                 or "independent" (default: "competitive")
                - custom_teams: Optional custom team assignments (list of lists)
        """
        super().__init__()

        # Extract configuration
        env_name = getattr(config, 'env_name', 'multigrid')
        env_id = getattr(config, 'env_id', 'soccer')
        env_seed = getattr(config, 'env_seed', None)
        self.render_mode = getattr(config, 'render_mode', None)

        # Parse training mode
        training_mode_str = getattr(config, 'training_mode', 'competitive')
        try:
            self.training_mode = TrainingMode(training_mode_str.lower())
        except ValueError:
            _logger.warning(
                f"Unknown training_mode '{training_mode_str}', "
                f"defaulting to 'competitive'"
            )
            self.training_mode = TrainingMode.COMPETITIVE

        # Custom team assignments (optional override)
        self._custom_teams: Optional[List[List[int]]] = getattr(
            config, 'custom_teams', None
        )

        # Store env_id for reference
        self.env_id = env_id.lower()

        # Create environment
        env_cls = _get_env_class(env_id)
        if env_cls is None:
            available = get_available_environments()
            raise ValueError(
                f"Unknown or unavailable MultiGrid environment: '{env_id}'. "
                f"Available: {available}. "
                f"Make sure mosaic_multigrid is installed."
            )

        # Pass render_mode="rgb_array" when FastLane is enabled so that
        # env.render() returns a numpy array (required by FastLaneTelemetryWrapper).
        # Without this, render() returns None and the shared memory buffer is
        # never created, causing "fastlane-unavailable" on the GUI side.
        fastlane_active = maybe_wrap_env is not None and is_fastlane_enabled()
        render_mode = "rgb_array" if fastlane_active else None

        # Allow view_size override via MOSAIC_VIEW_SIZE environment variable.
        # With view_size=7 (vs default 3), agents see 49 cells instead of 9,
        # changing obs from (3,3,3)=27 to (7,7,3)=147 flattened features.
        env_extra_kwargs: dict[str, Any] = {}
        view_size_str = os.environ.get("MOSAIC_VIEW_SIZE", "")
        if view_size_str:
            env_extra_kwargs["view_size"] = int(view_size_str)
            try:
                from gym_gui.logging_config.helpers import log_constant
                from gym_gui.logging_config.log_constants import (
                    LOG_OPERATOR_VIEW_SIZE_CONFIGURED,
                )
                log_constant(
                    _logger,
                    LOG_OPERATOR_VIEW_SIZE_CONFIGURED,
                    extra={
                        "view_size": int(view_size_str),
                        "env_id": env_id,
                    },
                )
            except ImportError:
                _logger.info(
                    "Using view_size=%d from MOSAIC_VIEW_SIZE",
                    int(view_size_str),
                )
        self.env = env_cls(render_mode=render_mode, **env_extra_kwargs)

        # mosaic_multigrid uses Gymnasium API - no wrapper needed (already reproducible)
        # Seed will be passed to reset() instead of calling env.seed()
        self._seed = env_seed  # Store seed for use in reset()

        # =====================================================================
        # IMPORTANT: Store env metadata BEFORE wrapping with FastLane
        # FastLane wrapper doesn't forward all attributes (agents, max_steps, etc.)
        # TeamObs envs are gymnasium.Wrappers — look through to unwrapped env.
        # =====================================================================
        _base = getattr(self.env, 'unwrapped', self.env)
        self._base_env_agents = _base.agents  # mosaic_multigrid agent objects
        self._base_max_steps = getattr(_base, 'max_steps', getattr(self.env, 'max_steps', 10000))

        # Store scenario name for reference
        self.scenario_name = f"{env_name}.{env_id}"

        # Build agent list: ['agent_0', 'agent_1', ...]
        self.num_agents = len(self._base_env_agents)
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]

        # Wrap with FastLane for visualization during training
        # mosaic_multigrid uses Gymnasium API (5-tuple step return) - no conversion needed
        if maybe_wrap_env is not None and is_fastlane_enabled():
            # Apply Gymnasium compatibility wrapper before FastLane
            self.env = GymToGymnasiumWrapper(self.env)
            self.env = maybe_wrap_env(self.env)
            _logger.info("Applied FastLane wrapper for training visualization")

        # Build agent_groups based on training mode
        self.agent_groups = self._build_agent_groups()
        self.n_groups = len(self.agent_groups)

        # Log configuration
        _logger.info(
            f"MultiGrid_Env initialized: env={env_id}, "
            f"training_mode={self.training_mode.value}, "
            f"num_agents={self.num_agents}, "
            f"num_groups={self.n_groups}"
        )

        # Build observation and action spaces (dict format for XuanCe)
        # mosaic_multigrid uses dict observation_space: {0: Dict[image, direction, mission], 1: ...}
        # We extract the image space from agent 0 as template
        obs_space_0 = self.env.observation_space[0]  # Dict space for agent 0
        img_space = obs_space_0['image']  # Box space for image

        # Detect TeamObs: teammate_* keys present alongside image
        self._is_teamobs = 'teammate_positions' in obs_space_0.spaces

        # Flatten observation for MLP representations.
        # IndAgObs: image only → (3,3,3) = 27
        # TeamObs:  image + teammate_directions + teammate_has_ball + teammate_positions
        #   2vs2 (1 teammate):  27 + 1 + 1 + 2 = 31    (excl. direction)
        #   3vs3 (2 teammates): 27 + 2 + 2 + 4 = 35    (excl. direction)
        # XuanCe Basic_MLP reads obs_shape[0] as input dimension, so a flat
        # shape ensures the first Linear layer has the correct in_features.
        self._obs_flat_dim = int(np.prod(img_space.shape))
        if self._is_teamobs:
            for key in ('teammate_directions', 'teammate_has_ball', 'teammate_positions'):
                if key in obs_space_0.spaces:
                    self._obs_flat_dim += int(np.prod(obs_space_0[key].shape))

        self.observation_space = {
            agent: spaces.Box(
                low=0,
                high=255,
                shape=(self._obs_flat_dim,),
                dtype=np.float32,
            )
            for agent in self.agents
        }

        # mosaic_multigrid uses dict action_space: {0: Discrete(7), 1: Discrete(7), ...}
        act_space_0 = self.env.action_space[0]  # Discrete space for agent 0

        self.action_space = {
            agent: spaces.Discrete(act_space_0.n)
            for agent in self.agents
        }

        # State space (global state) - concatenated agent observations
        self.state_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_agents * self._obs_flat_dim,),
            dtype=np.float32
        )

        # Episode tracking (use stored value, not wrapped env)
        self.max_episode_steps = self._base_max_steps
        self.individual_episode_reward = {k: 0.0 for k in self.agents}
        self._episode_step = 0
        self._last_obs: Optional[List[np.ndarray]] = None

    def _build_agent_groups(self) -> List[List[str]]:
        """Build agent_groups based on training mode.

        Returns:
            List of agent groups. Structure depends on training_mode:
            - COMPETITIVE: Groups based on environment's team structure
            - COOPERATIVE: Single group with all agents
            - INDEPENDENT: Each agent in their own group
        """
        if self.training_mode == TrainingMode.COOPERATIVE:
            # All agents share one policy
            return [self.agents.copy()]

        elif self.training_mode == TrainingMode.INDEPENDENT:
            # Each agent has their own policy
            return [[agent] for agent in self.agents]

        else:  # COMPETITIVE (default)
            # Use custom teams if provided
            if self._custom_teams is not None:
                return [
                    [f"agent_{i}" for i in team]
                    for team in self._custom_teams
                ]

            # Otherwise, use environment's natural team structure
            return self._build_teams_from_env()

    def _build_teams_from_env(self) -> List[List[str]]:
        """Build team groups from mosaic_multigrid's agent.index assignments.

        Returns:
            List of agent groups based on environment's team structure.
            Example for Soccer: [['agent_0', 'agent_1'], ['agent_2', 'agent_3']]
        """
        team_to_agents: Dict[int, List[str]] = defaultdict(list)

        # Use stored base env agents (not wrapped env which may not forward .agents)
        for i, agent in enumerate(self._base_env_agents):
            team_idx = agent.team_index  # Team assignment (1, 2, etc.)
            team_to_agents[team_idx].append(f"agent_{i}")

        # Sort by team index and return as list of lists
        sorted_teams = sorted(team_to_agents.keys())
        return [team_to_agents[team] for team in sorted_teams]

    def close(self) -> None:
        """Close the environment."""
        if hasattr(self.env, 'close'):
            self.env.close()

    def render(self, *args: Any, **kwargs: Any) -> Optional[np.ndarray]:
        """Render the environment."""
        return self.env.render(*args, **kwargs)

    def _flatten_agent_obs(self, agent_obs: dict) -> np.ndarray:
        """Flatten a single agent's observation dict to a 1-D float32 array.

        IndAgObs: image only → (27,)
        TeamObs:  image + teammate_directions + teammate_has_ball + teammate_positions
        """
        parts = [np.asarray(agent_obs['image']).flatten()]
        if self._is_teamobs:
            for key in ('teammate_directions', 'teammate_has_ball', 'teammate_positions'):
                if key in agent_obs:
                    parts.append(np.asarray(agent_obs[key]).flatten())
        return np.concatenate(parts).astype(np.float32)

    def reset(self, **kwargs: Any) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment to initial state.

        Returns:
            Tuple of (observations_dict, info_dict)
        """
        # Pass seed on first reset (Gymnasium API)
        reset_kwargs = {}
        if hasattr(self, '_seed') and self._seed is not None:
            reset_kwargs['seed'] = self._seed
            self._seed = None  # Only use seed once

        result = self.env.reset(**reset_kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs_dict, _info = result
        else:
            obs_dict = result

        # mosaic_multigrid returns dict {0: {image, direction, mission, ...}, 1: ...}
        # Flatten observations (handles both IndAgObs and TeamObs)
        observations = {
            f"agent_{i}": self._flatten_agent_obs(obs_dict[i])
            for i in range(self.num_agents)
        }

        # Store full dict for state() function
        self._last_obs = obs_dict

        self._episode_step = 0
        self.individual_episode_reward = {k: 0.0 for k in self.agents}

        reset_info = {
            "infos": {},
            "individual_episode_rewards": self.individual_episode_reward.copy(),
            "state": self.state(),
            "avail_actions": self.avail_actions(),
        }

        return observations, reset_info

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], bool, Dict[str, Any]]:
        """Execute actions for all agents.

        Args:
            actions: Dict mapping agent ID to action

        Returns:
            Tuple of (observations, rewards, terminated, truncated, info)
        """
        # mosaic_multigrid uses dict actions: {0: action, 1: action, ...}
        actions_dict = {i: actions[f"agent_{i}"] for i in range(self.num_agents)}

        # mosaic_multigrid uses Gymnasium API (5-tuple dict-keyed return)
        obs_dict, rewards_dict, terminated_dict, truncated_dict, info = self.env.step(actions_dict)

        # Flatten observations (handles both IndAgObs and TeamObs)
        observations = {
            f"agent_{i}": self._flatten_agent_obs(obs_dict[i])
            for i in range(self.num_agents)
        }

        rewards = {
            f"agent_{i}": float(rewards_dict[i])
            for i in range(self.num_agents)
        }

        for agent, reward in rewards.items():
            self.individual_episode_reward[agent] += reward
        self._episode_step += 1
        self._last_obs = obs_dict

        # Check if any agent is terminated (episode done)
        done = any(terminated_dict.values())

        terminated = {agent: bool(done) for agent in self.agents}
        truncated = self._episode_step >= self.max_episode_steps

        step_info = {
            "infos": info,
            "individual_episode_rewards": self.individual_episode_reward.copy(),
            "state": self.state(),
            "avail_actions": self.avail_actions(),
            # Required by OnPolicyMARLAgents.store_experience():
            "agent_mask": self.agent_mask(),
            "episode_step": self._episode_step,
            # Required by MARL runners for episode-end logging:
            "episode_score": self.individual_episode_reward.copy(),
        }

        return observations, rewards, terminated, truncated, step_info

    def state(self) -> np.ndarray:
        """Returns the global state of the environment."""
        if self._last_obs is None:
            return np.zeros(self.state_space.shape, dtype=np.float32)

        # _last_obs is dict {0: {image, direction, mission, ...}, 1: ...}
        # Flatten and concatenate all agent observations
        parts = [self._flatten_agent_obs(self._last_obs[i]) for i in range(self.num_agents)]
        return np.concatenate(parts).astype(np.float32)

    def agent_mask(self) -> Dict[str, bool]:
        """Returns boolean mask indicating which agents are alive."""
        return {agent: True for agent in self.agents}

    def avail_actions(self) -> Dict[str, np.ndarray]:
        """Returns available actions mask for each agent.

        When MOSAIC_MASK_INVALID_ACTIONS=1 is set, masks actions that are
        irrelevant for Soccer/Basketball (noop=0, toggle=6, done=7), reducing
        the effective action space from 8 to 5 actions while keeping the same
        policy output dimensions (clean deployment, no wrappers needed).

        Set MOSAIC_INVALID_ACTIONS=0,6,7 to customise which actions to mask.
        """
        num_actions = self.action_space[self.agents[0]].n
        if os.environ.get("MOSAIC_MASK_INVALID_ACTIONS", "0") == "1":
            invalid = [
                int(a)
                for a in os.environ.get("MOSAIC_INVALID_ACTIONS", "0,6,7").split(",")
            ]
            mask = np.ones(num_actions, dtype=np.bool_)
            for idx in invalid:
                if 0 <= idx < num_actions:
                    mask[idx] = False
            return {agent: mask.copy() for agent in self.agents}
        return {
            agent: np.ones(num_actions, dtype=np.bool_)
            for agent in self.agents
        }

    @property
    def env_info(self) -> Dict[str, Any]:
        """Return environment information dict."""
        return {
            'state_space': self.state_space,
            'observation_space': self.observation_space,
            'action_space': self.action_space,
            'agents': self.agents,
            'num_agents': self.num_agents,
            'max_episode_steps': self.max_episode_steps,
            'training_mode': self.training_mode.value,
            'num_groups': len(self.agent_groups),
        }

    @property
    def groups_info(self) -> Dict[str, Any]:
        """Return groups information for XuanCe runners.

        This is the key interface that determines policy architecture:
        - runner_competition.py uses this to create per-group policies
        - Structure depends on training_mode setting

        Returns:
            Dict with group structure information.
        """
        return {
            'num_groups': len(self.agent_groups),
            'agent_groups': self.agent_groups,
            'observation_space_groups': [
                {agent: self.observation_space[agent] for agent in group}
                for group in self.agent_groups
            ],
            'action_space_groups': [
                {agent: self.action_space[agent] for agent in group}
                for group in self.agent_groups
            ],
            'num_agents_groups': [len(group) for group in self.agent_groups],
        }

    def get_training_mode_info(self) -> Dict[str, Any]:
        """Return detailed information about current training mode.

        Useful for debugging and logging.
        """
        mode_descriptions = {
            TrainingMode.COMPETITIVE: "Per-team policies (adversarial training)",
            TrainingMode.COOPERATIVE: "Shared policy (parameter sharing)",
            TrainingMode.INDEPENDENT: "Per-agent policies (individual learning)",
        }

        return {
            "training_mode": self.training_mode.value,
            "description": mode_descriptions[self.training_mode],
            "num_policies": len(self.agent_groups),
            "policy_structure": [
                {
                    "policy_id": i,
                    "agents": group,
                    "num_agents": len(group),
                }
                for i, group in enumerate(self.agent_groups)
            ],
        }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "MultiGrid_Env",
    "TrainingMode",
    "MULTIGRID_ENV_INFO",
    "get_available_environments",
]
