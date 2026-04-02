"""Ray RLlib multi-agent training runtime.

This module provides the core training logic for multi-agent environments
using Ray RLlib. It supports various PettingZoo environment families and
multiple policy configurations.

Supported Environment Families:
- SISL: Cooperative continuous control (Multiwalker, Waterworld, Pursuit)
- Classic: Board games (Chess, Go, Connect Four, Tic-Tac-Toe)
- Butterfly: Mixed cooperative/competitive
- MPE: Multi-agent particle environments

Supported Policy Configurations:
- Parameter Sharing: All agents share one policy (for homogeneous cooperative)
- Independent: Each agent has its own policy
- Self-Play: Agent plays against copies of itself
- Shared Value Function: CTDE (Centralized Training, Decentralized Execution)

Analytics Integration:
- TensorBoard: Logs to var/trainer/runs/{run_id}/tensorboard/
- WandB: Optional integration with Weights & Biases
- analytics.json: Manifest file for GUI integration
"""

from __future__ import annotations

import json as _json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set, TYPE_CHECKING

import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.impala import IMPALAConfig
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv, ParallelPettingZooEnv
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.replay_buffers.multi_agent_replay_buffer import MultiAgentReplayBuffer, ReplayBuffer
from ray.tune.registry import register_env

try:
    import gymnasium as gym
except ImportError:
    import gym


# =============================================================================
# Ray RLlib Bug Workaround: ABCMeta type check in replay buffer creation
# =============================================================================
# Bug: Ray RLlib 2.52.1 has a bug in _create_local_replay_buffer_if_necessary()
# where it checks `if "EpisodeReplayBuffer" in config["replay_buffer_config"]["type"]`
# but the `type` field is sometimes converted to a class (ABCMeta) instead of
# remaining as a string, causing TypeError: argument of type 'ABCMeta' is not iterable.
#
# This affects DQN and SAC algorithms when using the old API stack.
# Fix: Monkey-patch the method to handle both string and class types.
# =============================================================================

def _patched_create_local_replay_buffer_if_necessary(
    self, config
) -> Optional[MultiAgentReplayBuffer]:
    """Patched version that handles both string and class types for replay buffer.

    This fixes Ray RLlib bug where the replay buffer type string is converted to
    an ABCMeta class before the string check, causing a TypeError.
    """
    if not config.get("replay_buffer_config") or config["replay_buffer_config"].get(
        "no_local_replay_buffer"
    ):
        return None

    buffer_type = config["replay_buffer_config"].get("type", "")

    # Handle both string and class types - this is the fix
    if isinstance(buffer_type, str):
        type_name = buffer_type
    elif isinstance(buffer_type, type):
        type_name = buffer_type.__name__
    else:
        type_name = str(buffer_type)

    # Check for EpisodeReplayBuffer using the type name string
    if "EpisodeReplayBuffer" in type_name:
        config["replay_buffer_config"][
            "metrics_num_episodes_for_smoothing"
        ] = self.config.metrics_num_episodes_for_smoothing

    return from_config(ReplayBuffer, config["replay_buffer_config"])


def _apply_ray_replay_buffer_patch() -> None:
    """Apply monkey-patch to fix Ray RLlib replay buffer bug.

    Only patches if the bug exists (checks for the problematic code pattern).
    This is idempotent - safe to call multiple times.
    """
    import inspect
    import logging as _logging

    # Check if already patched
    if getattr(Algorithm, "_replay_buffer_patched", False):
        return

    try:
        # Get the source of the original method
        source = inspect.getsource(Algorithm._create_local_replay_buffer_if_necessary)

        # Check if the buggy pattern exists
        if '"EpisodeReplayBuffer" in config["replay_buffer_config"]["type"]' in source:
            Algorithm._create_local_replay_buffer_if_necessary = (
                _patched_create_local_replay_buffer_if_necessary
            )
            Algorithm._replay_buffer_patched = True
            _logging.getLogger(__name__).debug(
                "Applied Ray RLlib replay buffer bug workaround (ABCMeta fix)"
            )
    except Exception as e:
        _logging.getLogger(__name__).debug(
            f"Could not check/apply replay buffer patch: {e}"
        )


# Apply the patch at module import time (for DQN/SAC to work)
_apply_ray_replay_buffer_patch()

from gym_gui.core.worker import TelemetryEmitter
from gym_gui.logging_config.helpers import log_constant
from gym_gui.logging_config.log_constants import (
    LOG_WORKER_RAY_RUNTIME_STARTED,
    LOG_WORKER_RAY_RUNTIME_COMPLETED,
    LOG_WORKER_RAY_RUNTIME_FAILED,
    LOG_WORKER_RAY_CLUSTER_STARTED,
    LOG_WORKER_RAY_RLLIB_TRAINING_STARTED,
    LOG_WORKER_RAY_HEARTBEAT,
    LOG_WORKER_RAY_TENSORBOARD_ENABLED,
    LOG_WORKER_RAY_WANDB_ENABLED,
    LOG_WORKER_RAY_CHECKPOINT_SAVED,
    LOG_WORKER_RAY_ANALYTICS_MANIFEST_CREATED,
    LOG_WORKER_RAY_LOG_FILE_CREATED,
    LOG_WORKER_RAY_ALGORITHM_BUILT,
)

from .config import (
    RayWorkerConfig,
    TrainingConfig,
    PolicyConfiguration,
    PettingZooAPIType,
)
from .fastlane import maybe_wrap_env, set_fastlane_env_vars, is_fastlane_enabled, maybe_wrap_parallel_env
from .analytics import write_analytics_manifest
from .algo_params import (
    get_algorithm_fields,
    get_field_names,
    merge_with_defaults,
    filter_params_for_algorithm,
    RLLIB_PARAM_MAPPING,
)

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

# Standard LogRecord attribute names (used to identify custom extra fields).
_STANDARD_LOG_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
)

_HUMAN_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


class _StructuredLogFormatter(logging.Formatter):
    """Formatter that emits pipe-delimited output for ``log_constant()`` records.

    Records produced by :func:`log_constant` carry a ``log_code`` attribute in
    *extra*.  This formatter detects that attribute and renders the line as::

        LOG_CODE | human message | extra={"key": "value", ...}

    which the dispatcher's ``_parse_structured_log_line()`` can parse.

    All other records fall back to the standard human-readable format.
    """

    def __init__(self) -> None:
        super().__init__(_HUMAN_FMT, datefmt=_DATE_FMT)

    def format(self, record: logging.LogRecord) -> str:
        code: str | None = getattr(record, "log_code", None)
        if not code:
            return super().format(record)

        # Build message text — strip the leading "LOG_CODE " prefix that
        # log_constant() prepends via ``"%s %s" % (code, text)``.
        msg = record.getMessage()
        prefix = f"{code} "
        if msg.startswith(prefix):
            msg = msg[len(prefix):]

        # Collect the extra payload (all non-standard record attributes).
        extra: dict[str, Any] = {}
        for key, val in record.__dict__.items():
            if key not in _STANDARD_LOG_ATTRS:
                extra[key] = val

        return f"{code} | {msg} | extra={_json.dumps(extra, default=str)}"


_LOGGER = logging.getLogger(__name__)


def _wrap_env_for_ray(env: Any, api_type: PettingZooAPIType, worker_index: int = 0) -> Any:
    """Wrap a PettingZoo environment for Ray RLlib (module-level for picklability).

    This function is defined at module level to avoid capturing `self` in closures,
    which would cause pickle errors with SummaryWriter's thread locks.

    Args:
        env: Raw PettingZoo environment (AEC or Parallel)
        api_type: API type of the environment (may be overridden by actual env type)
        worker_index: Ray worker index for unique FastLane stream per worker

    Returns:
        Ray-compatible wrapped environment:
        - PettingZooEnv for AEC environments
        - ParallelPettingZooEnv for Parallel environments
    """
    # Handle standard Gym environments (single agent)
    if api_type == PettingZooAPIType.GYM:
        # TODO: Add FastLane wrapper for Gym environments
        # from .fastlane import maybe_wrap_gym_env
        # return maybe_wrap_gym_env(env, worker_index=worker_index)
        return env

    # Detect actual environment type - some envs (like Classic games) are always AEC
    # regardless of what the config says. AEC envs have agent_selection attribute.
    is_aec_env = hasattr(env, "agent_selection") or hasattr(env, "agent_iter")

    # Use detected type: if env is AEC, always use PettingZooEnv wrapper
    use_parallel = (api_type == PettingZooAPIType.PARALLEL) and not is_aec_env

    if use_parallel:
        # Wrap with FastLane for live visualization (Parallel version)
        # Each worker gets its own stream: {run_id}-worker-{worker_index}
        env = maybe_wrap_parallel_env(env, worker_index=worker_index)
        # Use Ray's ParallelPettingZooEnv for Parallel API
        return ParallelPettingZooEnv(env)
    else:
        # Wrap with FastLane for live visualization (AEC version)
        env = maybe_wrap_env(env, worker_index=worker_index)
        # Use Ray's PettingZooEnv for AEC API
        return PettingZooEnv(env)


class EnvironmentFactory:
    """Factory for creating PettingZoo environments.

    Supports both API types:
    - AEC (Agent Environment Cycle): Turn-based, agents act sequentially
    - Parallel: All agents act simultaneously

    API Type Selection by Family:
    - SISL (cooperative): Defaults to Parallel (native API)
    - Classic (board games): AEC only (turn-based by nature)
    - Butterfly: Both supported, user selects
    - MPE: Defaults to Parallel (native API)
    """

    # Environment families and their default/supported API types
    FAMILY_DEFAULTS = {
        "sisl": PettingZooAPIType.PARALLEL,      # Cooperative, simultaneous
        "classic": PettingZooAPIType.AEC,        # Turn-based games
        "butterfly": PettingZooAPIType.PARALLEL, # Mixed, but mostly parallel
        "mpe": PettingZooAPIType.PARALLEL,       # Continuous, simultaneous
        "gym": PettingZooAPIType.GYM,            # Standard Gym
    }

    # Families that only support AEC (turn-based games)
    AEC_ONLY_FAMILIES = {"classic"}

    @staticmethod
    def create_gym_env(env_id: str, **kwargs) -> Any:
        """Create a standard Gym/Gymnasium environment."""
        if "render_mode" in kwargs and kwargs["render_mode"] == "rgb_array":
             # Ensure we don't pass render_mode if env doesn't support it or if it's already set
             pass
        return gym.make(env_id, **kwargs)

    @staticmethod
    def create_sisl_env(env_id: str, api_type: PettingZooAPIType, **kwargs) -> Any:
        """Create a SISL environment.

        SISL (Stanford Intelligent Systems Lab) environments are cooperative
        multi-agent benchmarks that natively support both AEC and Parallel APIs.
        Default: Parallel (simultaneous actions).
        """
        if env_id == "waterworld_v4":
            from pettingzoo.sisl import waterworld_v4
            if api_type == PettingZooAPIType.PARALLEL:
                return waterworld_v4.parallel_env(**kwargs)
            return waterworld_v4.env(**kwargs)
        elif env_id == "multiwalker_v9":
            from pettingzoo.sisl import multiwalker_v9
            if api_type == PettingZooAPIType.PARALLEL:
                return multiwalker_v9.parallel_env(**kwargs)
            return multiwalker_v9.env(**kwargs)
        elif env_id == "pursuit_v4":
            from pettingzoo.sisl import pursuit_v4
            if api_type == PettingZooAPIType.PARALLEL:
                return pursuit_v4.parallel_env(**kwargs)
            return pursuit_v4.env(**kwargs)
        else:
            raise ValueError(f"Unknown SISL environment: {env_id}")

    @staticmethod
    def create_classic_env(env_id: str, api_type: PettingZooAPIType, **kwargs) -> Any:
        """Create a Classic (board game) environment.

        Classic environments are turn-based games (chess, go, etc.)
        and only support AEC API.
        """
        # Classic games are always AEC (turn-based)
        if api_type == PettingZooAPIType.PARALLEL:
            _LOGGER.warning(
                f"Classic environment {env_id} only supports AEC API. "
                "Ignoring Parallel request and using AEC."
            )

        if env_id == "chess_v6":
            from pettingzoo.classic import chess_v6
            return chess_v6.env(**kwargs)
        elif env_id == "go_v5":
            from pettingzoo.classic import go_v5
            return go_v5.env(**kwargs)
        elif env_id == "connect_four_v3":
            from pettingzoo.classic import connect_four_v3
            return connect_four_v3.env(**kwargs)
        elif env_id == "tictactoe_v3":
            from pettingzoo.classic import tictactoe_v3
            return tictactoe_v3.env(**kwargs)
        else:
            raise ValueError(f"Unknown Classic environment: {env_id}")

    @staticmethod
    def create_butterfly_env(env_id: str, api_type: PettingZooAPIType, **kwargs) -> Any:
        """Create a Butterfly environment.

        Butterfly environments support both AEC and Parallel APIs.
        """
        if env_id == "knights_archers_zombies_v10":
            from pettingzoo.butterfly import knights_archers_zombies_v10
            if api_type == PettingZooAPIType.PARALLEL:
                return knights_archers_zombies_v10.parallel_env(**kwargs)
            return knights_archers_zombies_v10.env(**kwargs)
        elif env_id == "cooperative_pong_v5":
            from pettingzoo.butterfly import cooperative_pong_v5
            if api_type == PettingZooAPIType.PARALLEL:
                return cooperative_pong_v5.parallel_env(**kwargs)
            return cooperative_pong_v5.env(**kwargs)
        elif env_id == "pistonball_v6":
            from pettingzoo.butterfly import pistonball_v6
            if api_type == PettingZooAPIType.PARALLEL:
                return pistonball_v6.parallel_env(**kwargs)
            return pistonball_v6.env(**kwargs)
        else:
            raise ValueError(f"Unknown Butterfly environment: {env_id}")

    @staticmethod
    def create_mpe_env(env_id: str, api_type: PettingZooAPIType, **kwargs) -> Any:
        """Create an MPE (Multi-Particle Environment).

        MPE environments support both AEC and Parallel APIs.
        Default: Parallel (continuous control, simultaneous).
        """
        if env_id == "simple_spread_v3":
            from pettingzoo.mpe import simple_spread_v3
            if api_type == PettingZooAPIType.PARALLEL:
                return simple_spread_v3.parallel_env(**kwargs)
            return simple_spread_v3.env(**kwargs)
        elif env_id == "simple_adversary_v3":
            from pettingzoo.mpe import simple_adversary_v3
            if api_type == PettingZooAPIType.PARALLEL:
                return simple_adversary_v3.parallel_env(**kwargs)
            return simple_adversary_v3.env(**kwargs)
        elif env_id == "simple_tag_v3":
            from pettingzoo.mpe import simple_tag_v3
            if api_type == PettingZooAPIType.PARALLEL:
                return simple_tag_v3.parallel_env(**kwargs)
            return simple_tag_v3.env(**kwargs)
        else:
            raise ValueError(f"Unknown MPE environment: {env_id}")

    @classmethod
    def get_default_api_type(cls, family: str) -> PettingZooAPIType:
        """Get the default API type for an environment family."""
        return cls.FAMILY_DEFAULTS.get(family.lower(), PettingZooAPIType.AEC)

    @classmethod
    def is_aec_only(cls, family: str) -> bool:
        """Check if a family only supports AEC API."""
        return family.lower() in cls.AEC_ONLY_FAMILIES

    @classmethod
    def create_env(
        cls,
        family: str,
        env_id: str,
        api_type: PettingZooAPIType = PettingZooAPIType.PARALLEL,
        **kwargs,
    ) -> Any:
        """Create a PettingZoo environment by family and ID.

        Args:
            family: Environment family (sisl, classic, butterfly, mpe)
            env_id: Environment identifier
            api_type: AEC or Parallel API type
            **kwargs: Additional environment arguments

        Returns:
            PettingZoo environment instance (AEC or Parallel based on api_type)

        Note:
            For Ray RLlib compatibility:
            - AEC envs: Use PettingZooEnv wrapper
            - Parallel envs: Use ParallelPettingZooEnv wrapper OR convert to AEC

        API Type by Family:
            - SISL: Parallel (default) - cooperative, simultaneous
            - Classic: AEC only - turn-based board games
            - Butterfly: Both - user selects
            - MPE: Parallel (default) - continuous, simultaneous
        """
        family_lower = family.lower()

        # Normalize api_type to string for logging (may be enum or string)
        api_type_str = api_type.value if hasattr(api_type, 'value') else str(api_type)

        # Log the API type being used
        _LOGGER.info(f"Creating {family}/{env_id} with {api_type_str} API")

        if family_lower == "sisl":
            env = cls.create_sisl_env(env_id, api_type, **kwargs)
        elif family_lower == "classic":
            env = cls.create_classic_env(env_id, api_type, **kwargs)
        elif family_lower == "butterfly":
            env = cls.create_butterfly_env(env_id, api_type, **kwargs)
        elif family_lower == "mpe":
            env = cls.create_mpe_env(env_id, api_type, **kwargs)
        elif family_lower == "gym":
            env = cls.create_gym_env(env_id, **kwargs)
        else:
            raise ValueError(f"Unknown environment family: {family}")

        return env


# Algorithm configuration mapping
ALGORITHM_CONFIGS = {
    "PPO": PPOConfig,
    "IMPALA": IMPALAConfig,
    "APPO": APPOConfig,
    "DQN": DQNConfig,
    "SAC": SACConfig,
}

# Algorithm suitability for different environments
ALGORITHM_INFO = {
    "PPO": {
        "description": "Proximal Policy Optimization - stable, general purpose",
        "action_space": ["discrete", "continuous"],
        "on_policy": True,
        "multi_agent": True,
    },
    "IMPALA": {
        "description": "Importance Weighted Actor-Learner - distributed, fast",
        "action_space": ["discrete", "continuous"],
        "on_policy": False,
        "multi_agent": True,
    },
    "APPO": {
        "description": "Async PPO - higher throughput than PPO",
        "action_space": ["discrete", "continuous"],
        "on_policy": False,
        "multi_agent": True,
    },
    "DQN": {
        "description": "Deep Q-Network - off-policy, sample efficient",
        "action_space": ["discrete"],
        "on_policy": False,
        "multi_agent": True,
    },
    "SAC": {
        "description": "Soft Actor-Critic - off-policy, continuous actions",
        "action_space": ["continuous"],
        "on_policy": False,
        "multi_agent": True,
    },
}


# =============================================================================
# Environment-Specific Model Configurations
# =============================================================================
# Ray RLlib only has default CNN configs for: 42x42, 84x84, 64x64, 10x10
# We need custom configs for other observation shapes.

ENV_MODEL_CONFIGS = {
    # Classic board games (8x8 board with many channels)
    "chess_v6": {
        "conv_filters": [
            [64, [3, 3], 1],   # 8x8 -> 6x6
            [64, [3, 3], 1],   # 6x6 -> 4x4
            [64, [2, 2], 2],   # 4x4 -> 2x2
        ],
        "conv_activation": "relu",
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
    },
    "go_v5": {
        # Go board: 19x19 or configurable with many channels
        "conv_filters": [
            [64, [3, 3], 2],   # 19x19 -> 9x9
            [128, [3, 3], 2],  # 9x9 -> 4x4
            [128, [2, 2], 2],  # 4x4 -> 2x2
        ],
        "conv_activation": "relu",
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
    },
    "connect_four_v3": {
        # Connect Four: 6x7 board
        "conv_filters": [
            [32, [3, 3], 1],   # 6x7 -> 4x5
            [64, [3, 3], 1],   # 4x5 -> 2x3
        ],
        "conv_activation": "relu",
        "fcnet_hiddens": [128, 128],
        "fcnet_activation": "relu",
    },
    "tictactoe_v3": {
        # Tic-tac-toe: 3x3 board
        "conv_filters": [
            [16, [2, 2], 1],   # 3x3 -> 2x2
        ],
        "conv_activation": "relu",
        "fcnet_hiddens": [64, 64],
        "fcnet_activation": "relu",
    },
    # SISL environments (small observation spaces)
    "pursuit_v4": {
        # Pursuit: 7x7x3
        "conv_filters": [
            [16, [3, 3], 2],  # 7x7 -> 3x3
            [32, [3, 3], 2],  # 3x3 -> 1x1
        ],
        "conv_activation": "relu",
        "fcnet_hiddens": [64, 64],
        "fcnet_activation": "relu",
    },
    "waterworld_v4": {
        # Waterworld: Continuous observation space - uses MLP, not CNN
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "tanh",
    },
    "multiwalker_v9": {
        # Multiwalker: Continuous observation space - uses MLP, not CNN
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "tanh",
    },
}


def _get_model_config_for_env(env_id: str) -> Dict[str, Any]:
    """Get appropriate model configuration for an environment.

    Args:
        env_id: Environment ID (e.g., 'chess_v6', 'pursuit_v4')

    Returns:
        Model config dict suitable for RLlib's config.training(model=...)
    """
    if env_id in ENV_MODEL_CONFIGS:
        return ENV_MODEL_CONFIGS[env_id]

    # Default for small image observations (7x7 like pursuit)
    return {
        "conv_filters": [
            [16, [3, 3], 2],  # Generic for small images
            [32, [3, 3], 2],
        ],
        "conv_activation": "relu",
        "fcnet_hiddens": [64, 64],
        "fcnet_activation": "relu",
    }


def get_algorithm_config_class(algorithm: str) -> type:
    """Get the algorithm config class for a given algorithm name.

    Args:
        algorithm: Algorithm name (PPO, IMPALA, APPO, DQN, SAC)

    Returns:
        Algorithm config class

    Raises:
        ValueError: If algorithm is not supported
    """
    algorithm_upper = algorithm.upper()
    if algorithm_upper not in ALGORITHM_CONFIGS:
        supported = ", ".join(ALGORITHM_CONFIGS.keys())
        raise ValueError(
            f"Unsupported algorithm: {algorithm}. Supported: {supported}"
        )
    return ALGORITHM_CONFIGS[algorithm_upper]


class RayWorkerRuntime:
    """Runtime for multi-agent training with Ray RLlib.

    This class handles:
    - Environment registration with Ray
    - Algorithm configuration based on policy configuration
    - Training loop with checkpointing
    - Progress reporting for UI integration
    - TensorBoard and WandB logging
    """

    def __init__(self, config: RayWorkerConfig) -> None:
        """Initialize the runtime with configuration.

        Args:
            config: Worker configuration
        """
        self.config = config
        self._algorithm: Optional[Algorithm] = None
        self._env_name = f"{config.environment.family}_{config.environment.env_id}"
        self._agent_ids: Set[str] = set()
        # Analytics/logging
        self._writer: Optional["SummaryWriter"] = None
        self._wandb_run: Optional[Any] = None
        # Telemetry emitter for lifecycle events
        self._emitter = TelemetryEmitter(run_id=config.run_id, logger=_LOGGER)

    def _create_env_factory(self) -> Callable:
        """Create environment factory function for Ray registration."""
        config = self.config

        def env_creator(_config: dict) -> Any:
            # Get worker index from Ray's config
            worker_index = getattr(_config, "worker_index", 0)

            # Create environment
            env = EnvironmentFactory.create_env(
                family=config.environment.family,
                env_id=config.environment.env_id,
                api_type=config.environment.api_type,
                **config.environment.env_kwargs,
            )

            # Wrap with FastLane and Ray wrappers
            wrapped_env = self._wrap_env_for_ray(env, worker_index=worker_index)
            return wrapped_env

        return env_creator

    def _wrap_env_for_ray(self, env: Any, worker_index: int = 0) -> Any:
        """Wrap PettingZoo environment for Ray RLlib.

        Delegates to the module-level _wrap_env_for_ray function which correctly
        handles both AEC and Parallel API types with FastLane support.

        Args:
            env: Raw PettingZoo environment (AEC or Parallel)
            worker_index: Ray worker index for unique FastLane stream

        Returns:
            Ray-compatible wrapped environment (PettingZooEnv or ParallelPettingZooEnv)
        """
        # Use the module-level function which handles both API types
        return _wrap_env_for_ray(env, self.config.environment.api_type, worker_index)

    def _get_agent_ids(self) -> Set[str]:
        """Get agent IDs from the environment."""
        if self._agent_ids:
            return self._agent_ids

        # Standard Gym environments are single-agent
        if self.config.environment.api_type == PettingZooAPIType.GYM:
            self._agent_ids = {"default_policy"}
            return self._agent_ids

        # Create a temporary env to get agent IDs
        # Use render_mode=None to avoid opening pygame windows
        temp_kwargs = dict(self.config.environment.env_kwargs)
        temp_kwargs["render_mode"] = None  # No rendering needed for agent detection
        env = EnvironmentFactory.create_env(
            family=self.config.environment.family,
            env_id=self.config.environment.env_id,
            api_type=self.config.environment.api_type,
            **temp_kwargs,
        )

        if hasattr(env, "possible_agents"):
            self._agent_ids = set(env.possible_agents)
        elif hasattr(env, "agents"):
            self._agent_ids = set(env.agents)
        else:
            # Fallback: try to get from reset
            if hasattr(env, "reset"):
                obs = env.reset()
                if isinstance(obs, dict):
                    self._agent_ids = set(obs.keys())
                elif isinstance(obs, tuple) and isinstance(obs[0], dict):
                    self._agent_ids = set(obs[0].keys())

        if hasattr(env, "close"):
            env.close()

        _LOGGER.info(f"Detected agents: {self._agent_ids}")
        return self._agent_ids

    def _build_multi_agent_config(self, base_config: PPOConfig) -> PPOConfig:
        """Configure multi-agent settings based on policy configuration.

        Uses the OLD RLlib API (api_stack with enable_rl_module_and_learner=False).
        The new API's .rl_module() method is not compatible with the old stack.

        Args:
            base_config: Base algorithm config

        Returns:
            Configured algorithm config with multi-agent settings
        """
        # Skip multi-agent config for Gym environments (single agent)
        if self.config.environment.api_type == PettingZooAPIType.GYM:
            return base_config

        agent_ids = self._get_agent_ids()
        policy_config = self.config.policy_configuration

        if policy_config == PolicyConfiguration.PARAMETER_SHARING:
            # All agents share a single policy
            return base_config.multi_agent(
                policies={"shared"},
                policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared",
            )

        elif policy_config == PolicyConfiguration.INDEPENDENT:
            # Each agent has its own policy (1:1 mapping)
            policies = {aid for aid in agent_ids}
            return base_config.multi_agent(
                policies=policies,
                policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
            )

        elif policy_config == PolicyConfiguration.SELF_PLAY:
            # Self-play: main policy plays against frozen copies
            # For now, use simple self-play where all agents use same policy
            # Advanced league training would require more complex setup
            return base_config.multi_agent(
                policies={"main"},
                policy_mapping_fn=lambda agent_id, *args, **kwargs: "main",
            )

        elif policy_config == PolicyConfiguration.SHARED_VALUE_FUNCTION:
            # CTDE: Shared critic, separate actor policies
            # Each agent has own policy but they share value function
            policies = {aid for aid in agent_ids}
            return base_config.multi_agent(
                policies=policies,
                policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
            )

        else:
            raise ValueError(f"Unknown policy configuration: {policy_config}")

    def _build_algorithm_config(self) -> AlgorithmConfig:
        """Build the complete algorithm configuration.

        Supports multiple algorithms: PPO, IMPALA, APPO, DQN, SAC

        Returns:
            Fully configured algorithm config
        """
        tc = self.config.training
        rc = self.config.resources
        algorithm = tc.algorithm.upper()

        # Note: FastLane env vars are set in run() before ray.init()
        # so that Ray workers can inherit them via runtime_env

        # Extract all values needed for env creation BEFORE defining the closure.
        # This avoids capturing `self` which contains non-picklable objects (SummaryWriter with locks).
        env_family = self.config.environment.family
        env_id = self.config.environment.env_id
        env_api_type = self.config.environment.api_type
        base_env_kwargs = dict(self.config.environment.env_kwargs)
        fastlane_enabled = self.config.fastlane_enabled

        # Create wrapped env factory - captures only primitive/picklable values
        def wrapped_env_creator(_config: dict) -> Any:
            # Include render_mode for FastLane frame capture
            env_kwargs = dict(base_env_kwargs)
            if fastlane_enabled and "render_mode" not in env_kwargs:
                env_kwargs["render_mode"] = "rgb_array"

            env = EnvironmentFactory.create_env(
                family=env_family,
                env_id=env_id,
                api_type=env_api_type,
                **env_kwargs,
            )

            # Get worker_index from RLlib config for unique FastLane stream per worker
            # RLlib passes worker_index: 0 for local worker, 1+ for remote workers
            # NOTE: RLlib's EnvContext stores worker_index as an ATTRIBUTE, not a dict key
            if hasattr(_config, "worker_index"):
                worker_index = _config.worker_index
            elif isinstance(_config, dict):
                worker_index = _config.get("worker_index", 0)
            else:
                worker_index = 0

            # Wrap for Ray - using the module-level function with API type and worker index
            return _wrap_env_for_ray(env, env_api_type, worker_index=worker_index)

        # Register environment with Ray
        register_env(self._env_name, wrapped_env_creator)

        # Get the appropriate config class for the algorithm
        ConfigClass = get_algorithm_config_class(algorithm)
        _LOGGER.info(f"Using algorithm: {algorithm} ({ConfigClass.__name__})")

        # Build base config with common settings
        # Use OLD API stack for all algorithms (better compatibility with various obs shapes)
        base_config = (
            ConfigClass()
            .environment(self._env_name)
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            # Disable validation to allow custom replay buffer configs
            .experimental(_validate_config=False)
            .env_runners(
                num_env_runners=rc.num_workers,
                num_cpus_per_env_runner=rc.num_cpus_per_worker,
                num_gpus_per_env_runner=rc.num_gpus_per_worker,
                # Create env on local worker (W0) so it can sample for FastLane
                create_env_on_local_worker=True,
            )
            .resources(
                num_gpus=rc.num_gpus,
            )
            # Environment-specific model config (CNN/MLP architecture)
            # Ray RLlib only has defaults for 42x42, 84x84, 64x64, 10x10 images
            # We define custom configs for chess (8x8x111), pursuit (7x7x3), etc.
            .training(
                model=_get_model_config_for_env(env_id)
            )
        )

        # Apply algorithm-specific training parameters
        base_config = self._apply_training_config(base_config, algorithm, tc)

        # Apply multi-agent configuration
        config = self._build_multi_agent_config(base_config)

        return config

    def _apply_training_config(
        self,
        config: AlgorithmConfig,
        algorithm: str,
        tc: TrainingConfig,
    ) -> AlgorithmConfig:
        """Apply algorithm-specific training parameters dynamically.

        Uses the algo_params module to validate and filter parameters based on
        the algorithm schema. This prevents runtime crashes from invalid
        parameter combinations (e.g., passing minibatch_size to APPO).

        Args:
            config: Base algorithm config
            algorithm: Algorithm name (PPO, IMPALA, APPO, DQN, SAC)
            tc: Training configuration with algo_params dict

        Returns:
            Config with training parameters applied
        """
        # Start with parameters from config, merge with schema defaults
        algo_params = merge_with_defaults(algorithm, tc.algo_params)

        # Filter to only valid parameters for this algorithm
        valid_params = filter_params_for_algorithm(algorithm, algo_params)

        # Parameters that should NOT be passed to config.training()
        # These are either not RLlib params or belong to other config methods
        NON_TRAINING_PARAMS = {
            "total_timesteps",  # Our custom field, not RLlib
            "num_workers",      # Goes to env_runners()
            "num_gpus",         # Goes to resources()
        }

        # Build the RLlib training params dict with proper name mapping
        training_params = {}

        for name, value in valid_params.items():
            if value is None:
                continue

            # Skip non-training parameters
            if name in NON_TRAINING_PARAMS:
                _LOGGER.debug(f"Skipping non-training param: {name}")
                continue

            # Map schema name to RLlib config name
            rllib_name = RLLIB_PARAM_MAPPING.get(name, name)
            training_params[rllib_name] = value

            _LOGGER.debug(f"Training param: {name} -> {rllib_name} = {value}")

        _LOGGER.info(f"Applying {len(training_params)} training params for {algorithm}")
        _LOGGER.debug(f"Training params: {training_params}")

        # DQN and SAC require explicit replay buffer config with old API stack
        # Note: The _apply_ray_replay_buffer_patch() fixes a Ray RLlib bug that caused
        # TypeError when checking replay buffer type (ABCMeta vs string issue)
        #
        # Memory considerations for DQN/SAC:
        # - Chess obs: 8x8x111 = 7104 floats ≈ 28KB per transition
        # - 10000 capacity ≈ 280MB (manageable), 50000 ≈ 1.4GB (can cause OOM)
        #
        # Note: We use MultiAgentReplayBuffer (non-prioritized) instead of
        # MultiAgentPrioritizedReplayBuffer because the prioritized version has
        # an assertion error in update_priorities_in_replay_buffer with multi-agent
        # environments (len(batch_indices) != len(td_error)).
        if algorithm in ("DQN", "SAC"):
            # Use smaller buffer to prevent OOM on large observation spaces
            buffer_capacity = training_params.pop("replay_buffer_capacity", 10000)
            training_params["replay_buffer_config"] = {
                "type": "MultiAgentReplayBuffer",  # Non-prioritized for multi-agent
                "capacity": buffer_capacity,
            }
            _LOGGER.info(
                f"Configured MultiAgentReplayBuffer for {algorithm} "
                f"(capacity={buffer_capacity})"
            )

        # Apply training parameters
        return config.training(**training_params)

    def _setup_logging(self) -> None:
        """Set up logging for the training run.

        Configures:
        * **stdout** — uses :class:`_StructuredLogFormatter` so that
          ``log_constant()`` records are emitted in the pipe-delimited format
          the dispatcher's ``_parse_structured_log_line()`` expects.
        * **File handler** — writes human-readable logs to
          ``{logs_dir}/worker.log`` for post-mortem diagnostics.
        """
        root = logging.getLogger()

        # Remove any handlers left over from cli.py's basicConfig so we can
        # set up our own stdout handler with the structured formatter.
        for h in list(root.handlers):
            root.removeHandler(h)
        root.setLevel(logging.DEBUG)

        # Stdout handler — structured format for dispatcher parsing
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.setFormatter(_StructuredLogFormatter())
        root.addHandler(stdout_handler)

        # File handler — human-readable format for post-mortem
        try:
            logs_dir = self.config.logs_dir
            logs_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(
                logs_dir / "worker.log", encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(
                logging.Formatter(_HUMAN_FMT, datefmt=_DATE_FMT)
            )
            root.addHandler(file_handler)
            log_constant(
                _LOGGER,
                LOG_WORKER_RAY_LOG_FILE_CREATED,
                extra={
                    "run_id": self.config.run_id,
                    "log_file": str(logs_dir / "worker.log"),
                },
            )
        except OSError as exc:
            _LOGGER.warning("Failed to create log file handler: %s", exc)

    def _setup_analytics(self) -> None:
        """Set up TensorBoard and WandB logging.

        Creates:
        - var/trainer/runs/{run_id}/tensorboard/ for TensorBoard logs
        - analytics.json manifest for GUI integration
        - Optional WandB run if enabled
        """
        # Ensure run directories exist
        self.config.ensure_run_directories()

        # Initialize TensorBoard writer if enabled
        if self.config.tensorboard:
            tb_dir = self.config.tensorboard_log_dir
            if tb_dir:
                tb_dir.mkdir(parents=True, exist_ok=True)
                try:
                    from torch.utils.tensorboard import SummaryWriter
                    self._writer = SummaryWriter(log_dir=str(tb_dir))
                    _LOGGER.info(f"TensorBoard logging to: {tb_dir}")
                    log_constant(
                        _LOGGER,
                        LOG_WORKER_RAY_TENSORBOARD_ENABLED,
                        extra={
                            "run_id": self.config.run_id,
                            "tensorboard_dir": str(tb_dir),
                        },
                    )
                except ImportError:
                    _LOGGER.warning(
                        "tensorboard not installed, TensorBoard logging disabled. "
                        "Install with: pip install tensorboard"
                    )

        # Initialize WandB if enabled
        wandb_run_path = None
        if self.config.wandb:
            try:
                import wandb
                # Build WandB config with dynamic algo_params
                wandb_config = {
                    "run_id": self.config.run_id,
                    "environment": self.config.environment.full_env_id,
                    "policy_configuration": self.config.policy_configuration.value,
                    "algorithm": self.config.training.algorithm,
                    "total_timesteps": self.config.training.total_timesteps,
                }
                # Add all algo_params to WandB config
                wandb_config.update(self.config.training.algo_params)

                self._wandb_run = wandb.init(
                    project=self.config.wandb_project or "ray-marl",
                    entity=self.config.wandb_entity,
                    name=self.config.wandb_run_name or self.config.run_id,
                    config=wandb_config,
                    sync_tensorboard=self.config.tensorboard,
                    reinit=True,
                )
                wandb_run_path = self._wandb_run.path if self._wandb_run else None
                _LOGGER.info(f"WandB run initialized: {wandb_run_path}")
                log_constant(
                    _LOGGER,
                    LOG_WORKER_RAY_WANDB_ENABLED,
                    extra={
                        "run_id": self.config.run_id,
                        "wandb_project": self.config.wandb_project or "ray-marl",
                        "wandb_run_path": wandb_run_path,
                    },
                )
            except ImportError:
                _LOGGER.warning(
                    "wandb not installed, WandB logging disabled. "
                    "Install with: pip install wandb"
                )
            except Exception as e:
                _LOGGER.warning(f"Failed to initialize WandB: {e}")

        # Write analytics manifest for GUI integration
        num_agents = len(self._get_agent_ids()) if self._agent_ids else None
        manifest_path = write_analytics_manifest(
            self.config,
            wandb_run_path=wandb_run_path,
            num_agents=num_agents,
            notes=f"Ray RLlib {self.config.training.algorithm} training on {self.config.environment.full_env_id}",
        )
        _LOGGER.info(f"Analytics manifest written: {manifest_path}")
        log_constant(
            _LOGGER,
            LOG_WORKER_RAY_ANALYTICS_MANIFEST_CREATED,
            extra={
                "run_id": self.config.run_id,
                "manifest_path": str(manifest_path),
                "num_agents": num_agents,
            },
        )

    def _log_metrics(self, result: Dict[str, Any], global_step: int) -> None:
        """Log training metrics to TensorBoard and WandB.

        Args:
            result: Training result dictionary from algorithm.train()
            global_step: Current global step (timesteps)
        """
        # Extract metrics from result (handle both old and new Ray API)
        env_runners = result.get("env_runners", {})

        # Core metrics
        metrics = {
            "train/episode_reward_mean": env_runners.get(
                "episode_return_mean",
                env_runners.get("episode_reward_mean", result.get("episode_reward_mean", 0))
            ),
            "train/episode_len_mean": env_runners.get(
                "episode_len_mean", result.get("episode_len_mean", 0)
            ),
            "train/num_episodes": env_runners.get(
                "num_episodes_lifetime", result.get("episodes_total", 0)
            ),
        }

        # Additional env_runner metrics
        if "num_env_steps_sampled_lifetime" in env_runners:
            metrics["train/env_steps_sampled"] = env_runners["num_env_steps_sampled_lifetime"]
        if "num_agent_steps_sampled_lifetime" in env_runners:
            metrics["train/agent_steps_sampled"] = env_runners["num_agent_steps_sampled_lifetime"]

        # Learner metrics (Ray 2.x structure: learners -> {policy_name} -> metrics)
        learners = result.get("learners", {})
        if learners:
            # Try multiple possible policy names used by Ray
            # - "shared" for parameter_sharing policy configuration
            # - "default_policy" for default single policy
            # - "main" for self-play
            # - First key if none of those exist
            policy_keys = ["shared", "default_policy", "main", "default_learner", "learner"]
            learner_data = None

            for key in policy_keys:
                if key in learners:
                    learner_data = learners[key]
                    break

            # If no known key, try first available policy
            if learner_data is None and learners:
                first_key = next(iter(learners.keys()), None)
                if first_key:
                    learner_data = learners[first_key]

            if isinstance(learner_data, dict):
                # PPO loss metrics
                if "total_loss" in learner_data:
                    metrics["train/total_loss"] = learner_data["total_loss"]
                if "policy_loss" in learner_data:
                    metrics["train/policy_loss"] = learner_data["policy_loss"]
                if "vf_loss" in learner_data:
                    metrics["train/vf_loss"] = learner_data["vf_loss"]
                if "entropy" in learner_data:
                    metrics["train/entropy"] = learner_data["entropy"]

                # Additional PPO metrics
                if "kl_loss" in learner_data:
                    metrics["train/kl_loss"] = learner_data["kl_loss"]
                if "vf_loss_unclipped" in learner_data:
                    metrics["train/vf_loss_unclipped"] = learner_data["vf_loss_unclipped"]
                if "mean_kl_loss" in learner_data:
                    metrics["train/mean_kl"] = learner_data["mean_kl_loss"]
                if "curr_kl_coeff" in learner_data:
                    metrics["train/kl_coeff"] = learner_data["curr_kl_coeff"]

                # Learning rate
                for key in learner_data:
                    if "learning_rate" in key.lower() or key == "lr":
                        metrics["train/learning_rate"] = learner_data[key]
                        break

        # Timing metrics
        if "time_total_s" in result:
            metrics["perf/time_total_s"] = result["time_total_s"]
        if "timers" in result:
            timers = result["timers"]
            if "learn_time_ms" in timers:
                metrics["perf/learn_time_ms"] = timers["learn_time_ms"]
            if "sample_time_ms" in timers:
                metrics["perf/sample_time_ms"] = timers["sample_time_ms"]

        # Filter out None values
        metrics = {k: v for k, v in metrics.items() if v is not None}

        # Debug: log available keys on first iteration to understand Ray result structure
        if global_step <= 1000:
            _LOGGER.debug(f"Ray result keys: {list(result.keys())}")
            if learners:
                _LOGGER.debug(f"Learners keys: {list(learners.keys())}")
                for lk, lv in learners.items():
                    if isinstance(lv, dict):
                        _LOGGER.debug(f"Learner '{lk}' metrics: {list(lv.keys())}")

        # Log to TensorBoard
        if self._writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self._writer.add_scalar(key, value, global_step)
            self._writer.flush()

        # Log to WandB
        if self._wandb_run:
            try:
                import wandb
                wandb.log(metrics, step=global_step)
            except Exception as e:
                _LOGGER.debug(f"WandB logging error: {e}")

    def _cleanup_analytics(self) -> None:
        """Clean up analytics resources."""
        if self._writer:
            self._writer.close()
            self._writer = None
            _LOGGER.info("TensorBoard writer closed")

        if self._wandb_run:
            try:
                import wandb
                wandb.finish()
                _LOGGER.info("WandB run finished")
            except Exception as e:
                _LOGGER.debug(f"WandB cleanup error: {e}")

    def _report_progress(
        self,
        iteration: int,
        result: Dict[str, Any],
        total_iterations: int,
    ) -> None:
        """Report training progress to stdout (for UI capture).

        Args:
            iteration: Current iteration number
            result: Training result dictionary
            total_iterations: Total expected iterations
        """
        # Handle both old and new Ray API result keys
        # New API: num_env_steps_sampled_lifetime, env_runners.num_episodes_lifetime
        # Old API: timesteps_total, episodes_total
        timesteps = result.get(
            "num_env_steps_sampled_lifetime",
            result.get("timesteps_total", 0)
        )

        # Try env_runners nested dict first, then top-level
        env_runners = result.get("env_runners", {})
        episode_reward_mean = env_runners.get(
            "episode_return_mean",
            env_runners.get(
                "episode_reward_mean",
                result.get("episode_reward_mean", 0)
            )
        )
        # Handle case where reward is None
        if episode_reward_mean is None:
            episode_reward_mean = 0.0

        episodes = env_runners.get(
            "num_episodes_lifetime",
            result.get("episodes_total", 0)
        )

        # Calculate estimated progress
        progress_pct = (iteration / total_iterations) * 100 if total_iterations > 0 else 0

        # Print in a format that can be parsed by the UI
        print(
            f"[PROGRESS] iteration={iteration}/{total_iterations} "
            f"({progress_pct:.1f}%) | "
            f"timesteps={timesteps} | "
            f"episodes={episodes} | "
            f"reward_mean={episode_reward_mean:.3f}"
        )
        sys.stdout.flush()

    def run(self) -> Dict[str, Any]:
        """Run the training loop.

        Returns:
            Final training result
        """
        self._setup_logging()

        # Disable pygame display to prevent popup windows
        # This must be set BEFORE pygame is imported by PettingZoo environments
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"

        log_constant(
            _LOGGER,
            LOG_WORKER_RAY_RLLIB_TRAINING_STARTED,
            extra={
                "run_id": self.config.run_id,
                "env": f"{self.config.environment.family}/{self.config.environment.env_id}",
                "policy": self.config.policy_configuration.value,
                "total_timesteps": self.config.training.total_timesteps,
            },
        )

        # Emit run started event
        self._emitter.run_started(
            {
                "worker_type": "ray",
                "env_family": self.config.environment.family,
                "env_id": self.config.environment.env_id,
                "policy_configuration": self.config.policy_configuration.value,
                "algorithm": self.config.training.algorithm,
                "total_timesteps": self.config.training.total_timesteps,
                "num_workers": self.config.resources.num_workers,
            },
            constant=LOG_WORKER_RAY_RUNTIME_STARTED,
        )

        # Set up TensorBoard, WandB, and analytics manifest
        self._setup_analytics()

        # Build runtime_env with all required environment variables
        # Get run_dir early so we can redirect ALL Ray outputs there
        run_dir = self.config.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)

        runtime_env_vars = {
            # Disable pygame windows (use dummy video/audio driver)
            "SDL_VIDEODRIVER": "dummy",
            "SDL_AUDIODRIVER": "dummy",
            # CRITICAL: Redirect Ray's internal cache from ~/ray_results to our run directory
            # This ensures ALL Ray artifacts (checkpoints, logs, trial metadata) stay in var/trainer/runs/
            "RAY_AIR_LOCAL_CACHE_DIR": str(run_dir),
        }

        # Add FastLane env vars if enabled (so Ray workers can use FastLane)
        if self.config.fastlane_enabled:
            runtime_env_vars["RAY_FASTLANE_ENABLED"] = "1"
            runtime_env_vars["RAY_FASTLANE_RUN_ID"] = self.config.run_id
            runtime_env_vars["RAY_FASTLANE_ENV_NAME"] = f"{self.config.environment.family}/{self.config.environment.env_id}"
            runtime_env_vars["RAY_FASTLANE_THROTTLE_MS"] = str(self.config.fastlane_throttle_ms)
            _LOGGER.info(f"FastLane enabled for run: {self.config.run_id}")

        # Also set env vars in the main process (for local env runners)
        for key, value in runtime_env_vars.items():
            os.environ[key] = value

        # Raise Ray's OOM kill threshold to avoid premature worker kills.
        # Default 0.95 is too aggressive when the host runs other services
        # (UI, TensorBoard, gRPC proxy). Set to 0.98 so workers survive init.
        os.environ.setdefault("RAY_memory_usage_threshold", "0.98")

        # Initialize Ray with env vars for all workers
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                log_to_driver=True,
                runtime_env={"env_vars": runtime_env_vars},
            )
            log_constant(
                _LOGGER,
                LOG_WORKER_RAY_CLUSTER_STARTED,
                extra={"run_id": self.config.run_id},
            )

        try:
            # Build algorithm (use build_algo() for RLlib 2.x+)
            algo_config = self._build_algorithm_config()

            # Create custom logger_creator to redirect Ray's logs to the tensorboard
            # subdirectory so they don't pollute the run root with stray tfevents files.
            tb_log_dir = run_dir / "tensorboard"
            tb_log_dir.mkdir(parents=True, exist_ok=True)

            def custom_logger_creator(config):
                from ray.tune.logger import UnifiedLogger
                return UnifiedLogger(config, str(tb_log_dir), loggers=None)

            try:
                self._algorithm = algo_config.build_algo(logger_creator=custom_logger_creator)
            except IndexError as e:
                # Ray's get_spaces() fails with IndexError when ALL env runners
                # were OOM-killed before init. Surface a clear message.
                raise RuntimeError(
                    f"Ray algorithm build failed ({e}). This usually means "
                    f"RolloutWorker actors were killed by Ray's OOM monitor. "
                    f"Try reducing num_env_runners or freeing memory on the host."
                ) from e

            log_constant(
                _LOGGER,
                LOG_WORKER_RAY_ALGORITHM_BUILT,
                extra={
                    "run_id": self.config.run_id,
                    "tensorboard_dir": str(tb_log_dir),
                },
            )

            # Calculate iterations using train_batch_size from algo_params
            train_batch_size = self.config.training.algo_params.get("train_batch_size", 4000)
            total_iterations = max(
                1,
                self.config.training.total_timesteps // train_batch_size
            )

            _LOGGER.info(f"Training for {total_iterations} iterations")

            # Create checkpoint directory
            checkpoint_dir = self.config.checkpoint_dir
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Training loop
            final_result = {}
            for i in range(total_iterations):
                result = self._algorithm.train()
                final_result = result

                # Get current timesteps for logging
                current_timesteps = result.get(
                    "num_env_steps_sampled_lifetime",
                    result.get("timesteps_total", 0)
                )

                # Report progress to stdout (for UI)
                self._report_progress(i + 1, result, total_iterations)

                # Log metrics to TensorBoard/WandB
                self._log_metrics(result, current_timesteps)

                # Emit heartbeat event
                self._emitter.heartbeat(
                    {
                        "iteration": i + 1,
                        "timesteps": current_timesteps,
                        "episode_reward_mean": result.get("env_runners", {}).get(
                            "episode_return_mean",
                            result.get("episode_reward_mean", 0.0)
                        ),
                    },
                    constant=LOG_WORKER_RAY_HEARTBEAT,
                )

                # Checkpoint (only if checkpoint_freq > 0)
                if self.config.checkpoint.checkpoint_freq > 0 and (i + 1) % self.config.checkpoint.checkpoint_freq == 0:
                    ckpt_result = self._algorithm.save(str(checkpoint_dir))
                    # Extract checkpoint path from TrainingResult (Ray new API)
                    if hasattr(ckpt_result, 'checkpoint') and hasattr(ckpt_result.checkpoint, 'path'):
                        checkpoint_path = ckpt_result.checkpoint.path
                    else:
                        checkpoint_path = str(ckpt_result)
                    log_constant(
                        _LOGGER,
                        LOG_WORKER_RAY_CHECKPOINT_SAVED,
                        extra={
                            "run_id": self.config.run_id,
                            "checkpoint_path": str(checkpoint_path),
                            "iteration": i + 1,
                            "timesteps": current_timesteps,
                        },
                    )
                    print(f"[CHECKPOINT] path={checkpoint_path}")
                    sys.stdout.flush()

                # Check if we've reached target timesteps
                if current_timesteps >= self.config.training.total_timesteps:
                    _LOGGER.info("Reached target timesteps, stopping training")
                    break

            # Final checkpoint
            if self.config.checkpoint.checkpoint_at_end:
                result = self._algorithm.save(str(checkpoint_dir))
                # Extract checkpoint path from TrainingResult (Ray new API)
                if hasattr(result, 'checkpoint') and hasattr(result.checkpoint, 'path'):
                    final_checkpoint = result.checkpoint.path
                else:
                    final_checkpoint = str(result)
                log_constant(
                    _LOGGER,
                    LOG_WORKER_RAY_CHECKPOINT_SAVED,
                    extra={
                        "run_id": self.config.run_id,
                        "checkpoint_path": str(final_checkpoint),
                        "final": True,
                    },
                )
                print(f"[CHECKPOINT] path={final_checkpoint} final=true")
                sys.stdout.flush()

            _LOGGER.info("Training completed successfully")
            print(f"[COMPLETE] run_id={self.config.run_id} status=success")
            sys.stdout.flush()

            # Emit run completed event
            self._emitter.run_completed(
                {
                    "final_result": {
                        k: v for k, v in final_result.items()
                        if k in ["timesteps_total", "episode_reward_mean", "episodes_total"]
                    }
                },
                constant=LOG_WORKER_RAY_RUNTIME_COMPLETED,
            )

            return final_result

        except Exception as e:
            log_constant(
                _LOGGER,
                LOG_WORKER_RAY_RUNTIME_FAILED,
                message=str(e),
                extra={"run_id": self.config.run_id, "error_type": type(e).__name__},
                exc_info=e,
            )
            print(f"[ERROR] run_id={self.config.run_id} error={str(e)}")
            sys.stdout.flush()
            # Emit run failed event
            self._emitter.run_failed(
                {"error": str(e), "error_type": type(e).__name__},
                constant=LOG_WORKER_RAY_RUNTIME_FAILED,
            )
            raise

        finally:
            # Cleanup analytics (TensorBoard writer, WandB run)
            self._cleanup_analytics()
            # Cleanup algorithm and Ray
            if self._algorithm is not None:
                self._algorithm.stop()
            ray.shutdown()

    def stop(self) -> None:
        """Stop training and cleanup resources."""
        self._cleanup_analytics()
        if self._algorithm is not None:
            self._algorithm.stop()
            self._algorithm = None


__all__ = [
    "EnvironmentFactory",
    "RayWorkerRuntime",
]
