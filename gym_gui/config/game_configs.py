"""Game-specific configuration dataclasses following separation of concerns.

Each game has its own configuration class with game-specific parameters.
These configs are separate from global application settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, TypeAlias

from gym_gui.constants.constants_game import (
    BLACKJACK_DEFAULTS,
    CLIFF_WALKING_DEFAULTS,
    FROZEN_LAKE_DEFAULTS,
    FROZEN_LAKE_V2_DEFAULTS,
)
from gym_gui.core.enums import GameId


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class FrozenLakeConfig:
    """Configuration for FrozenLake environment."""

    is_slippery: bool = True
    """If True, agent moves in intended direction with probability specified by success_rate,
    else moves in perpendicular directions with equal probability.
    If False, agent always moves in intended direction."""

    success_rate: float = 1.0 / 3.0
    """Probability of moving in intended direction when is_slippery=True.
    For example, with success_rate=1/3:
    - P(intended direction) = 1/3
    - P(perpendicular direction 1) = 1/3
    - P(perpendicular direction 2) = 1/3"""

    reward_schedule: tuple[float, float, float] = (1.0, 0.0, 0.0)
    """Reward amounts for reaching tiles: (Goal, Hole, Frozen).
    Default (1, 0, 0) gives +1 for goal, 0 for hole/frozen."""

    grid_height: int = 4
    """Grid height (number of rows). Default is 4 for FrozenLake-v1, 8 for FrozenLake-v2."""

    grid_width: int = 4
    """Grid width (number of columns). Default is 4 for FrozenLake-v1, 8 for FrozenLake-v2."""

    start_position: tuple[int, int] | None = None
    """Starting position (row, col). If None, defaults to (0, 0)."""

    goal_position: tuple[int, int] | None = None
    """Goal position (row, col). If None, defaults to bottom-right corner."""

    hole_count: int | None = None
    """Number of holes. If None, uses Gymnasium default (4 for 4×4, 10 for 8×8)."""

    random_holes: bool = False
    """If True, holes are placed randomly. If False, uses fixed Gymnasium default map patterns.
    Only applies to FrozenLake-v2 with standard 4×4 or 8×8 grids."""

    def to_gym_kwargs(self) -> Dict[str, Any]:
        """Convert to Gymnasium environment kwargs.

        For FrozenLake-v1: Pass only is_slippery (Gymnasium 1.0.0 compatible).
                          Do NOT pass map_name or grid dimensions.

        For FrozenLake-v2: Custom map generation handled by adapter.

        Note: success_rate and reward_schedule require Gymnasium >= 1.1.0
        and are NOT passed to gym.make() in current implementation.
        """
        kwargs: Dict[str, Any] = {
            "is_slippery": self.is_slippery,
        }

        # Note: grid_height, grid_width, start_position, goal_position, and hole_count
        # should only be used by FrozenLakeV2Adapter._generate_map_descriptor().
        # DO NOT pass map_name to Gymnasium for v1 (causes initialization failure).
        # FrozenLakeV2Adapter handles custom map generation separately via gym_kwargs()
        # in its subclass override.

        return kwargs


@dataclass(frozen=True)
class TaxiConfig:
    """Configuration for Taxi-v3 environment.

    Note: Taxi-v3 in Gymnasium does not support is_raining or fickle_passenger
    parameters. These were present in older versions but removed in modern Gymnasium.
    The environment always uses deterministic movement.
    """

    is_raining: bool = False
    """[NOT SUPPORTED] If True, the cab would move in intended direction with 80% probability.
    This parameter is kept for UI compatibility but has no effect."""

    fickle_passenger: bool = False
    """[NOT SUPPORTED] If True, passenger would change destinations randomly.
    This parameter is kept for UI compatibility but has no effect."""

    def to_gym_kwargs(self) -> Dict[str, Any]:
        """Convert to Gymnasium environment kwargs.

        Note: Returns empty dict as Taxi-v3 doesn't accept custom parameters.
        """
        # Taxi-v3 doesn't support is_raining or fickle_passenger in current Gymnasium
        return {}


@dataclass(frozen=True)
class CliffWalkingConfig:
    """Configuration for CliffWalking environment."""

    is_slippery: bool = False
    """If True, the cliff can be slippery so the player may move perpendicular
    to the intended direction sometimes. If False (default), player always moves
    in intended direction."""

    def to_gym_kwargs(self) -> Dict[str, Any]:
        """Convert to Gymnasium environment kwargs."""
        return {"is_slippery": self.is_slippery}


@dataclass(frozen=True)
class BlackjackConfig:
    """Configuration for Blackjack environment."""

    natural: bool = False
    """If True, give an additional reward for starting with a natural blackjack
    (ace and ten, sum is 21). Natural gives 1.5 reward instead of 1.0."""

    sab: bool = False
    """If True, follow the exact rules from Sutton and Barto's book.
    When sab=True, the natural parameter is ignored. If the player achieves
    a natural blackjack and the dealer does not, the player wins (+1 reward).
    If both get a natural, it's a draw (0 reward)."""

    def to_gym_kwargs(self) -> Dict[str, Any]:
        """Convert to Gymnasium environment kwargs."""
        return {"natural": self.natural, "sab": self.sab}


@dataclass(frozen=True)
class LunarLanderConfig:
    """Configuration for LunarLander environment."""

    continuous: bool = False
    gravity: float = -10.0
    enable_wind: bool = False
    wind_power: float = 15.0
    turbulence_power: float = 1.5
    max_episode_steps: int | None = None

    def to_gym_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "continuous": self.continuous,
            "gravity": _clamp(self.gravity, -12.0, 0.0),
        }
        if self.enable_wind:
            kwargs.update(
                enable_wind=True,
                wind_power=max(0.0, self.wind_power),
                turbulence_power=max(0.0, self.turbulence_power),
            )
        else:
            kwargs["enable_wind"] = False
        return kwargs

    def sanitized_step_limit(self) -> int | None:
        steps = self.max_episode_steps
        if steps is None:
            return None
        if isinstance(steps, bool):
            return None
        try:
            value = int(steps)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None


@dataclass(frozen=True)
class CarRacingConfig:
    """Configuration for CarRacing environment."""

    continuous: bool = False
    domain_randomize: bool = False
    lap_complete_percent: float = 0.95
    max_episode_steps: int | None = None
    max_episode_seconds: float | None = None

    @classmethod
    def from_env(cls) -> "CarRacingConfig":
        steps_raw = os.getenv("CAR_RACING_MAX_EPISODE_STEPS")
        seconds_raw = os.getenv("CAR_RACING_MAX_EPISODE_SECONDS")
        continuous_raw = os.getenv("CAR_RACING_CONTINUOUS")
        domain_raw = os.getenv("CAR_RACING_DOMAIN_RANDOMIZE")
        lap_raw = os.getenv("CAR_RACING_LAP_COMPLETE_PERCENT")

        steps: int | None
        seconds: float | None
        continuous = False
        domain_randomize = False
        lap_percent = 0.95

        try:
            steps = int(steps_raw) if steps_raw not in (None, "", "0") else None
        except (TypeError, ValueError):
            steps = None

        try:
            seconds = float(seconds_raw) if seconds_raw not in (None, "", "0") else None
        except (TypeError, ValueError):
            seconds = None

        if continuous_raw is not None:
            continuous = continuous_raw.strip().lower() in {"1", "true", "yes", "on"}
        if domain_raw is not None:
            domain_randomize = domain_raw.strip().lower() in {"1", "true", "yes", "on"}
        if lap_raw:
            try:
                lap_percent = float(lap_raw)
            except (TypeError, ValueError):
                lap_percent = 0.95

        return cls(
            continuous=continuous,
            domain_randomize=domain_randomize,
            lap_complete_percent=lap_percent,
            max_episode_steps=steps,
            max_episode_seconds=seconds,
        )

    def to_gym_kwargs(self) -> Dict[str, Any]:
        percent = _clamp(self.lap_complete_percent, 0.5, 1.0)
        kwargs: Dict[str, Any] = {
            "continuous": self.continuous,
            "domain_randomize": self.domain_randomize,
            "lap_complete_percent": percent,
        }
        return kwargs

    def sanitized_time_limits(self) -> tuple[int | None, float | None]:
        steps = int(self.max_episode_steps) if self.max_episode_steps and self.max_episode_steps > 0 else None
        seconds = (
            float(self.max_episode_seconds)
            if self.max_episode_seconds and self.max_episode_seconds > 0
            else None
        )
        return steps, seconds


@dataclass(frozen=True)
class BipedalWalkerConfig:
    """Configuration for BipedalWalker environment."""

    hardcore: bool = False
    max_episode_steps: int | None = None
    max_episode_seconds: float | None = None

    @classmethod
    def from_env(cls) -> "BipedalWalkerConfig":
        hardcore_raw = os.getenv("BIPEDAL_HARDCORE")
        steps_raw = os.getenv("BIPEDAL_MAX_EPISODE_STEPS")
        seconds_raw = os.getenv("BIPEDAL_MAX_EPISODE_SECONDS")

        hardcore = False
        steps: int | None
        seconds: float | None

        if hardcore_raw is not None:
            hardcore = hardcore_raw.strip().lower() in {"1", "true", "yes", "on"}

        try:
            steps = int(steps_raw) if steps_raw not in (None, "", "0") else None
        except (TypeError, ValueError):
            steps = None

        try:
            seconds = float(seconds_raw) if seconds_raw not in (None, "", "0") else None
        except (TypeError, ValueError):
            seconds = None

        return cls(
            hardcore=hardcore,
            max_episode_steps=steps,
            max_episode_seconds=seconds,
        )

    def to_gym_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"hardcore": self.hardcore}
        return kwargs

    def sanitized_time_limits(self) -> tuple[int | None, float | None]:
        steps = int(self.max_episode_steps) if self.max_episode_steps and self.max_episode_steps > 0 else None
        seconds = (
            float(self.max_episode_seconds)
            if self.max_episode_seconds and self.max_episode_seconds > 0
            else None
        )
        return steps, seconds


@dataclass(frozen=True)
class MiniGridConfig:
    """Configuration payload for MiniGrid environments."""

    env_id: str = GameId.MINIGRID_EMPTY_5x5.value
    """Gymnasium environment identifier (e.g., ``MiniGrid-Empty-5x5-v0``)."""

    partial_observation: bool = True
    """Apply :class:`RGBImgPartialObsWrapper` to expose agent-centric views."""

    image_observation: bool = True
    """Convert observations to RGB image arrays using :class:`ImgObsWrapper`."""

    reward_multiplier: float = 10.0
    """Scalar applied to environment rewards (aligned with xuance baseline)."""

    agent_view_size: int | None = None
    """Optional override for agent view size (MiniGrid default is 7)."""

    max_episode_steps: int | None = None
    """Override max episode steps; ``None`` preserves environment default."""

    seed: int | None = None
    """Default seed forwarded to :meth:`gymnasium.Env.reset`."""

    render_mode: str = "rgb_array"
    """Render mode requested during environment creation."""

    append_direction: bool = True
    """When True, append the agent's direction to the flattened observation
    vector (image.flatten() + [direction]). Matches xuance baseline behaviour."""

    def to_gym_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"render_mode": self.render_mode}
        if self.agent_view_size is not None:
            kwargs["agent_view_size"] = int(self.agent_view_size)
        if self.max_episode_steps is not None and self.max_episode_steps > 0:
            kwargs["max_episode_steps"] = int(self.max_episode_steps)
        return kwargs


# GameConfig type alias is defined after all config classes (see below)


# Default configurations for each game
DEFAULT_FROZEN_LAKE_CONFIG = FrozenLakeConfig(
    is_slippery=FROZEN_LAKE_DEFAULTS.slippery,
    success_rate=1.0 / 3.0,
    reward_schedule=(1.0, 0.0, 0.0),
    grid_height=FROZEN_LAKE_DEFAULTS.grid_height,
    grid_width=FROZEN_LAKE_DEFAULTS.grid_width,
    start_position=FROZEN_LAKE_DEFAULTS.start,
    goal_position=FROZEN_LAKE_DEFAULTS.goal,
    hole_count=FROZEN_LAKE_DEFAULTS.hole_count,
    random_holes=FROZEN_LAKE_DEFAULTS.random_holes,
)
DEFAULT_FROZEN_LAKE_V2_CONFIG = FrozenLakeConfig(
    is_slippery=FROZEN_LAKE_V2_DEFAULTS.slippery,
    success_rate=1.0 / 3.0,
    reward_schedule=(1.0, 0.0, 0.0),
    grid_height=FROZEN_LAKE_V2_DEFAULTS.grid_height,
    grid_width=FROZEN_LAKE_V2_DEFAULTS.grid_width,
    start_position=FROZEN_LAKE_V2_DEFAULTS.start,
    goal_position=FROZEN_LAKE_V2_DEFAULTS.goal,
    hole_count=FROZEN_LAKE_V2_DEFAULTS.hole_count,
    random_holes=FROZEN_LAKE_V2_DEFAULTS.random_holes,
)
DEFAULT_TAXI_CONFIG = TaxiConfig(is_raining=False, fickle_passenger=False)
DEFAULT_CLIFF_WALKING_CONFIG = CliffWalkingConfig(
    is_slippery=CLIFF_WALKING_DEFAULTS.slippery
)
DEFAULT_BLACKJACK_CONFIG = BlackjackConfig()
DEFAULT_LUNAR_LANDER_CONFIG = LunarLanderConfig()
DEFAULT_CAR_RACING_CONFIG = CarRacingConfig.from_env()
DEFAULT_BIPEDAL_WALKER_CONFIG = BipedalWalkerConfig.from_env()
DEFAULT_MINIGRID_EMPTY_5x5_CONFIG = MiniGridConfig(env_id=GameId.MINIGRID_EMPTY_5x5.value)
DEFAULT_MINIGRID_EMPTY_RANDOM_5x5_CONFIG = MiniGridConfig(env_id=GameId.MINIGRID_EMPTY_RANDOM_5x5.value)
DEFAULT_MINIGRID_EMPTY_6x6_CONFIG = MiniGridConfig(env_id=GameId.MINIGRID_EMPTY_6x6.value)
DEFAULT_MINIGRID_EMPTY_RANDOM_6x6_CONFIG = MiniGridConfig(env_id=GameId.MINIGRID_EMPTY_RANDOM_6x6.value)
DEFAULT_MINIGRID_EMPTY_8x8_CONFIG = MiniGridConfig(env_id=GameId.MINIGRID_EMPTY_8x8.value)
DEFAULT_MINIGRID_EMPTY_16x16_CONFIG = MiniGridConfig(env_id=GameId.MINIGRID_EMPTY_16x16.value)
DEFAULT_MINIGRID_DOORKEY_5x5_CONFIG = MiniGridConfig(
    env_id=GameId.MINIGRID_DOORKEY_5x5.value,
    agent_view_size=5,
)
DEFAULT_MINIGRID_DOORKEY_6x6_CONFIG = MiniGridConfig(
    env_id=GameId.MINIGRID_DOORKEY_6x6.value,
    agent_view_size=7,
)
DEFAULT_MINIGRID_DOORKEY_8x8_CONFIG = MiniGridConfig(
    env_id=GameId.MINIGRID_DOORKEY_8x8.value,
    agent_view_size=7,
)
DEFAULT_MINIGRID_DOORKEY_16x16_CONFIG = MiniGridConfig(
    env_id=GameId.MINIGRID_DOORKEY_16x16.value,
    agent_view_size=9,
)
DEFAULT_MINIGRID_LAVAGAP_S7_CONFIG = MiniGridConfig(
    env_id=GameId.MINIGRID_LAVAGAP_S7.value,
    agent_view_size=7,
    partial_observation=True,
)

# RedBlueDoors environments
DEFAULT_MINIGRID_REDBLUE_DOORS_6x6_CONFIG = MiniGridConfig(
    env_id=GameId.MINIGRID_REDBLUE_DOORS_6x6.value,
    agent_view_size=7,
)
DEFAULT_MINIGRID_REDBLUE_DOORS_8x8_CONFIG = MiniGridConfig(
    env_id=GameId.MINIGRID_REDBLUE_DOORS_8x8.value,
    agent_view_size=7,
)


# ---------------------------------------------------------------------------
# ALE (Atari) configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CrafterConfig:
    """Configuration payload for Crafter environments.

    Crafter is an open world survival game benchmark for reinforcement learning
    that evaluates a wide range of agent capabilities within a single environment.

    Paper: Hafner, D. (2022). Benchmarking the Spectrum of Agent Capabilities. ICLR 2022.
    Repository: https://github.com/danijar/crafter
    """

    env_id: str = GameId.CRAFTER_REWARD.value
    """Gymnasium environment identifier (e.g., ``CrafterReward-v1``)."""

    area: tuple[int, int] = (64, 64)
    """World dimensions (width, height). Default is 64x64."""

    view: tuple[int, int] = (9, 9)
    """Agent viewport dimensions (width, height). Default is 9x9."""

    size: tuple[int, int] = (512, 512)
    """Rendered image size (width, height). Default is 512x512 for balanced quality/performance."""

    reward: bool = True
    """Enable rewards. Set to False for CrafterNoReward-v1 variant."""

    length: int = 10000
    """Maximum episode steps. Default is 10,000."""

    seed: int | None = None
    """Default seed forwarded to :meth:`gymnasium.Env.reset`."""

    render_mode: str = "rgb_array"
    """Render mode requested during environment creation."""

    reward_multiplier: float = 1.0
    """Scalar applied to environment rewards."""

    def to_gym_kwargs(self) -> Dict[str, Any]:
        """Convert to Gymnasium environment kwargs."""
        kwargs: Dict[str, Any] = {"render_mode": self.render_mode}
        # Note: area, view, size, reward, length are typically passed
        # during environment creation for custom configurations
        return kwargs


@dataclass(frozen=True)
class ProcgenConfig:
    """Configuration payload for Procgen environments.

    Procgen provides 16 procedurally-generated game-like environments designed
    to measure sample efficiency and generalization in reinforcement learning.

    Paper: Cobbe et al. (2019). Leveraging Procedural Generation to Benchmark RL.
    Repository: https://github.com/openai/procgen
    """

    env_name: str = "coinrun"
    """Procgen game name (one of 16: bigfish, bossfight, caveflyer, etc.)."""

    num_levels: int = 0
    """Number of unique levels (0 = unlimited levels for generalization testing)."""

    start_level: int = 0
    """Starting level seed for reproducibility."""

    distribution_mode: str = "hard"
    """Difficulty mode: 'easy', 'hard', 'extreme', 'memory', 'exploration'."""

    use_backgrounds: bool = True
    """Use human-designed backgrounds (False = pure black)."""

    center_agent: bool = True
    """Center observations on agent."""

    use_sequential_levels: bool = False
    """Progress through levels sequentially (like gym-retro)."""

    paint_vel_info: bool = False
    """Paint velocity info on observations (game-specific)."""

    render_mode: str = "rgb_array"
    """Render mode requested during environment creation."""

    render_scale: int = 1
    """Scale factor for 512x512 info["rgb"] (1 = no scaling, fast rendering)."""

    seed: int | None = None
    """Default seed forwarded to :meth:`gymnasium.Env.reset`."""

    def to_gym_kwargs(self) -> Dict[str, Any]:
        """Convert to Gymnasium environment kwargs."""
        return {
            "env_name": self.env_name,
            "num_levels": self.num_levels,
            "start_level": self.start_level,
            "distribution_mode": self.distribution_mode,
            "use_backgrounds": self.use_backgrounds,
            "center_agent": self.center_agent,
            "use_sequential_levels": self.use_sequential_levels,
            "paint_vel_info": self.paint_vel_info,
            "render_mode": self.render_mode,
        }


@dataclass(frozen=True)
class ALEConfig:
    """Configuration payload for ALE Atari environments.

    Mirrors common ALE kwargs used by Gymnasium:
    - obs_type: "rgb" | "ram" | "grayscale"
    - frameskip: int or (min, max) tuple
    - repeat_action_probability: float (aka stickiness / RAP)
    - difficulty, mode: integers selecting game flavour
    - full_action_space: request full 18-action set when True
    - render_mode: rendering mode (default "rgb_array")
    - env_id: Gymnasium environment id
    """

    env_id: str = GameId.ADVENTURE_V4.value
    obs_type: str = "rgb"
    frameskip: int | tuple[int, int] | None = None
    repeat_action_probability: float | None = None
    difficulty: int | None = None
    mode: int | None = None
    full_action_space: bool = False
    render_mode: str = "rgb_array"

    def to_gym_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "render_mode": self.render_mode,
            "obs_type": self.obs_type,
        }
        if self.frameskip is not None:
            kwargs["frameskip"] = self.frameskip
        if self.repeat_action_probability is not None:
            kwargs["repeat_action_probability"] = float(self.repeat_action_probability)
        if self.difficulty is not None:
            kwargs["difficulty"] = int(self.difficulty)
        if self.mode is not None:
            kwargs["mode"] = int(self.mode)
        if self.full_action_space:
            kwargs["full_action_space"] = True
        return kwargs


@dataclass(frozen=True)
class TextWorldConfig:
    """Configuration payload for TextWorld text-based game environments.

    TextWorld is a Microsoft Research sandbox for training RL agents on
    text-based games. It generates and simulates text-based adventure games
    for research in language understanding and sequential decision making.

    Paper: Cote et al. (2018). TextWorld: A Learning Environment for Text-based Games.
    Repository: https://github.com/microsoft/TextWorld
    """

    env_id: str = "TextWorld-Simple-v0"
    """Environment identifier for the TextWorld game type."""

    challenge_type: str = "simple"
    """Challenge type: 'simple', 'coin_collector', 'treasure_hunter', 'cooking'."""

    level: int = 1
    """Difficulty level for built-in challenges (1-300 for coin_collector)."""

    nb_rooms: int = 5
    """Number of rooms for custom game generation."""

    nb_objects: int = 10
    """Number of objects for custom game generation."""

    quest_length: int = 5
    """Quest length for custom game generation."""

    max_episode_steps: int = 100
    """Maximum steps per episode."""

    gamefile: str | None = None
    """Path to a pre-generated game file (.ulx or .z8). If provided, skips generation."""

    seed: int | None = None
    """Random seed for game generation and reproducibility."""

    reward_multiplier: float = 1.0
    """Scalar applied to environment rewards."""

    intermediate_reward: bool = True
    """Enable intermediate rewards for making progress toward the goal."""

    render_mode: str = "ansi"
    """Render mode (TextWorld uses 'ansi' for text output)."""

    def to_gym_kwargs(self) -> Dict[str, Any]:
        """Convert to Gymnasium environment kwargs."""
        return {
            "render_mode": self.render_mode,
        }


@dataclass(frozen=True)
class JumanjiConfig:
    """Configuration payload for Jumanji JAX-based logic puzzle environments.

    Jumanji is a suite of JAX-based reinforcement learning environments that
    provides logic puzzle games like 2048, Minesweeper, Rubik's Cube, Sudoku,
    and more.

    Repository: https://github.com/google-deepmind/jumanji
    """

    env_id: str = "jumanji/Game2048-v1"
    """Gymnasium environment identifier (e.g., ``jumanji/Game2048-v1``)."""

    seed: int | None = None
    """Random seed for JAX PRNG reproducibility."""

    flatten_obs: bool = False
    """If True, flatten structured observations to 1D arrays for RL training."""

    backend: str | None = None
    """JAX backend ('cpu', 'gpu', 'tpu') or None for auto-detection."""

    render_mode: str = "rgb_array"
    """Render mode (Jumanji supports 'rgb_array' for visualization)."""

    def to_gym_kwargs(self) -> Dict[str, Any]:
        """Convert to jumanji_worker.gymnasium_adapter.make_jumanji_gym_env kwargs."""
        return {
            "seed": self.seed or 0,
            "flatten_obs": self.flatten_obs,
            "backend": self.backend,
            "render_mode": self.render_mode,
        }


@dataclass
class MultiGridConfig:
    """Configuration payload for mosaic_multigrid multi-agent environments.

    mosaic_multigrid is a multi-agent extension of MiniGrid for training cooperative
    and competitive multi-agent RL policies. All agents act simultaneously.

    Repository: https://github.com/Abdulhamid97Mousa/mosaic_multigrid
    Package: pip install mosaic-multigrid

    IMPORTANT: MultiGrid environments REQUIRE state-based input mode for multi-keyboard
    support. Shortcut-based mode is incompatible with evdev multi-keyboard monitoring
    and will cause all agents to respond to any keyboard input.
    """

    env_id: str = "soccer"
    """Environment variant: 'soccer' (2v2, 4 agents) or 'collect' (3 agents)."""

    num_agents: int | None = None
    """Number of agents in the environment. If None, uses environment default.
    For INI multigrid environments, defaults to 1 if not specified.
    For legacy environments (Soccer, Collect), this is ignored (fixed agent count)."""

    seed: int | None = None
    """Random seed for reproducibility."""

    highlight: bool = True
    """Whether to highlight agent view cones in render."""

    view_size: int | None = None
    """Agent view size override. MOSAIC default is 3, INI default is 7.
    When None, uses the environment's built-in default.
    Passed as view_size kwarg to the environment constructor."""

    env_kwargs: Dict[str, Any] | None = None
    """Additional environment-specific kwargs."""

    @property
    def required_input_mode(self) -> str:
        """Return the required input mode for MultiGrid environments.

        MultiGrid environments MUST use state-based input mode to support
        multi-keyboard control via evdev. Shortcut-based mode conflicts with
        per-device keyboard monitoring.

        Returns:
            "state_based" - Always returns state-based mode requirement
        """
        return "state_based"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        result: Dict[str, Any] = {
            "env_id": self.env_id,
            "seed": self.seed,
            "highlight": self.highlight,
            "env_kwargs": self.env_kwargs or {},
        }
        if self.view_size is not None:
            result["view_size"] = self.view_size
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiGridConfig":
        """Create config from dictionary."""
        view_size = data.get("view_size")
        return cls(
            env_id=data.get("env_id", "soccer"),
            seed=data.get("seed"),
            highlight=data.get("highlight", True),
            view_size=int(view_size) if view_size is not None else None,
            env_kwargs=data.get("env_kwargs"),
        )


@dataclass(frozen=True)
class MeltingPotConfig:
    """Configuration payload for Melting Pot multi-agent environments.

    Melting Pot is a suite of test scenarios for multi-agent reinforcement learning
    developed by Google DeepMind. It assesses generalization to novel social situations
    involving both familiar and unfamiliar individuals, using the Shimmy PettingZoo wrapper.

    Repository: https://github.com/google-deepmind/meltingpot
    Shimmy: https://shimmy.farama.org/environments/meltingpot/

    NOTE: Linux/macOS only (Windows NOT supported)

    IMPORTANT: MeltingPot environments REQUIRE state-based input mode for multi-keyboard
    support. Shortcut-based mode is incompatible with evdev multi-keyboard monitoring
    and will cause all agents to respond to any keyboard input.
    """

    substrate_name: str = "collaborative_cooking__circuit"
    """Substrate identifier (e.g., 'collaborative_cooking__circuit', 'commons_harvest__open').
    Available substrates: collaborative_cooking, clean_up, commons_harvest, territory,
    king_of_the_hill, prisoners_dilemma_in_the_matrix, stag_hunt_in_the_matrix,
    allelopathic_harvest."""

    seed: int | None = None
    """Random seed for reproducibility."""

    render_scale: int = 2
    """Scale factor for rendered image (1 = native, 2 = 2x, 4 = 4x).
    Native resolution varies by substrate (40×72 to 312×184).
    Higher values improve visibility but may impact performance."""

    env_kwargs: Dict[str, Any] | None = None
    """Additional environment-specific kwargs passed to Shimmy wrapper."""

    @property
    def required_input_mode(self) -> str:
        """Return the required input mode for MeltingPot environments.

        MeltingPot environments MUST use state-based input mode to support
        multi-keyboard control via evdev. Shortcut-based mode conflicts with
        per-device keyboard monitoring.

        Returns:
            "state_based" - Always returns state-based mode requirement
        """
        return "state_based"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "substrate_name": self.substrate_name,
            "seed": self.seed,
            "render_scale": self.render_scale,
            "env_kwargs": self.env_kwargs or {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MeltingPotConfig":
        """Create config from dictionary."""
        return cls(
            substrate_name=data.get("substrate_name", "collaborative_cooking__circuit"),
            seed=data.get("seed"),
            render_scale=data.get("render_scale", 2),
            env_kwargs=data.get("env_kwargs"),
        )


@dataclass(frozen=True)
class OvercookedConfig:
    """Configuration payload for Overcooked-AI cooperative cooking environments.

    Overcooked-AI is a benchmark environment for fully cooperative human-AI task performance,
    based on the cooperative cooking game. Two agents must coordinate to prepare and deliver
    soups by collecting ingredients, placing them in pots, waiting for cooking, and serving.

    Repository: https://github.com/HumanCompatibleAI/overcooked_ai
    Paper: https://arxiv.org/abs/1910.05789 (NeurIPS 2019)

    Research focus: Human-AI coordination, zero-shot coordination, behavior cloning

    IMPORTANT: Overcooked environments REQUIRE state-based input mode for multi-keyboard
    support. Shortcut-based mode is incompatible with evdev multi-keyboard monitoring
    and will cause all agents to respond to any keyboard input.
    """

    layout_name: str = "cramped_room"
    """Layout identifier (e.g., 'cramped_room', 'asymmetric_advantages', 'coordination_ring').
    Available research layouts: cramped_room, asymmetric_advantages, coordination_ring,
    forced_coordination, counter_circuit (plus 45+ others)."""

    horizon: int = 400
    """Maximum episode length in timesteps."""

    mdp_params: Dict[str, Any] | None = None
    """MDP parameters passed to OvercookedGridworld.from_layout_name().
    Common params: {'old_dynamics': True/False, 'start_positions': [(x,y), (x,y)]}."""

    env_params: Dict[str, Any] | None = None
    """Environment parameters passed to OvercookedEnv.from_mdp().
    Common params: {'reward_shaping_params': dict, 'mlam_params': dict}."""

    featurization: str = "lossless_encoding"
    """State featurization method: 'lossless_encoding' (default) or 'featurize'."""

    seed: int | None = None
    """Random seed for reproducibility (used in MDP generation)."""

    @property
    def required_input_mode(self) -> str:
        """Return the required input mode for Overcooked environments.

        Overcooked environments MUST use state-based input mode to support
        multi-keyboard control via evdev. Shortcut-based mode conflicts with
        per-device keyboard monitoring.

        Returns:
            "state_based" - Always returns state-based mode requirement
        """
        return "state_based"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "layout_name": self.layout_name,
            "horizon": self.horizon,
            "mdp_params": self.mdp_params or {},
            "env_params": self.env_params or {},
            "featurization": self.featurization,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OvercookedConfig":
        """Create config from dictionary."""
        return cls(
            layout_name=data.get("layout_name", "cramped_room"),
            horizon=data.get("horizon", 400),
            mdp_params=data.get("mdp_params"),
            env_params=data.get("env_params"),
            featurization=data.get("featurization", "lossless_encoding"),
            seed=data.get("seed"),
        )


@dataclass
class SMACConfig:
    """Configuration for SMAC/SMACv2 StarCraft Multi-Agent Challenge environments.

    Shared by both EnvironmentFamily.SMAC (v1 hand-designed maps) and
    EnvironmentFamily.SMACV2 (procedural generation).  The adapter import
    path determines which SMAC package is used; this config is family-agnostic
    (same pattern as MultiGridConfig for MOSAIC vs INI MultiGrid).

    Repositories:
        - SMAC v1: https://github.com/oxwhirl/smac
        - SMACv2:  https://github.com/oxwhirl/smacv2

    Requirements: StarCraft II Linux binary + smac or smacv2 pip package.
    """

    map_name: str = "3s5z"
    """SMAC map name passed to StarCraft2Env(map_name=...).
    v1 examples: '3m', '8m', '2s3z', '3s5z', '5m_vs_6m', 'MMM2'.
    v2 examples: '10gen_terran', '10gen_protoss', '10gen_zerg'."""

    difficulty: str = "7"
    """StarCraft II built-in AI difficulty (1-10).
    Default '7' is the standard SMAC benchmark difficulty."""

    reward_sparse: bool = False
    """If True, use sparse reward (+1 win, 0 otherwise).
    If False (default), use shaped reward (damage dealt + kills + win bonus)."""

    reward_only_positive: bool = True
    """If True, clip negative rewards to zero."""

    reward_scale: bool = True
    """If True, normalize total episode reward by max_reward."""

    reward_scale_rate: float = 20.0
    """Scaling factor for reward normalization."""

    obs_own_health: bool = True
    """Include the agent's own health in its observation vector."""

    obs_pathing_grid: bool = False
    """Include pathing grid features in observation (increases obs size)."""

    obs_terrain_height: bool = False
    """Include terrain height features in observation."""

    seed: int | None = None
    """Random seed for reproducibility."""

    episode_limit: int | None = None
    """Override episode step limit. If None, uses map default."""

    sc2_path: str | None = None
    """Path to StarCraft II installation. If None, uses SC2PATH env var."""

    renderer: str = "3d"
    """Renderer style: '3d' (GPU-rendered SC2 engine), 'heatmap' (multi-panel feature layers),
    or 'classic' (PyGame circles)."""


@dataclass
class RWAREConfig:
    """Configuration for Robotic Warehouse (RWARE) multi-agent environments.

    RWARE simulates a warehouse with autonomous robots that cooperatively
    pick up shelves, deliver them to goal workstations, and return them.

    Paper: Papoudakis et al. (2021). "Benchmarking Multi-Agent Deep RL
           Algorithms in Cooperative Tasks"
    Source: 3rd_party/environments/robotic-warehouse/
    """

    observation_type: str = "flattened"
    """Observation format: 'flattened' (1D vector), 'dict' (nested dict),
    'image' (multi-channel grid), 'image_dict' (image + dict)."""

    sensor_range: int = 1
    """How many cells each agent can see around itself (1-5)."""

    reward_type: str = "individual"
    """Reward distribution: 'global' (shared), 'individual' (per-agent),
    'two_stage' (delivery + return)."""

    msg_bits: int = 0
    """Communication bits per agent (0 = silent, >0 = message channels)."""

    max_steps: int = 500
    """Maximum steps per episode."""

    seed: int | None = None
    """Random seed for reproducibility (None = random)."""

    render_mode: str = "rgb_array"
    """Render mode for MOSAIC display."""


@dataclass
class GriddlyConfig:
    """Configuration for Griddly grid world environments.

    Griddly provides a C++ backend with Vulkan rendering for high-throughput
    grid world research (30 000+ FPS headless).

    Paper: Bamford et al. (2021). "Griddly: A Platform for AI Research in Games"
    Repo:  https://github.com/Bam4d/Griddly
    """

    level: int = 0
    """Level index within the game (0 = default level)."""

    max_steps: int = 1000
    """Maximum steps per episode before truncation."""

    seed: int | None = None
    """Random seed for reproducibility (None = random)."""

    render_mode: str = "rgb_array"
    """Render mode for MOSAIC display."""


@dataclass
class GFootballConfig:
    """Configuration for Google Research Football (GRF) environments.

    GRF is a multi-agent football (soccer) environment developed by the
    Google Brain team, designed for RL research in cooperative and
    competitive settings with realistic physics.

    Paper: Kurach et al. (2020). "Google Research Football: A Novel RL
           Environment"
    Source: 3rd_party/environments/football/
    """

    representation: str = "simple115v2"
    """Observation representation: 'simple115' (115-d vector), 'simple115v2'
    (fixed positions), 'extracted' (super minimap), 'pixels', 'pixels_gray',
    or 'raw' (dict with full game state)."""

    rewards: str = "scoring"
    """Comma-separated reward components: 'scoring' (goal reward),
    'checkpoints' (dense distance-based reward)."""

    number_of_left_players_agent_controls: int = 1
    """Number of left team players controlled by the agent (0-11)."""

    number_of_right_players_agent_controls: int = 0
    """Number of right team players controlled by the agent (0-11)."""

    stacked: bool = False
    """Whether to stack 4 consecutive observations (pixels/extracted only)."""

    action_set: str = "default"
    """Action set: 'default' (19 actions), 'v2' (20 with builtin_ai),
    or 'full' (33 with all release actions)."""

    render: bool = True
    """Whether to render game frames (required for pixel observations
    and for MOSAIC Render View)."""

    max_steps: int = 3000
    """Maximum steps per episode."""

    seed: int | None = None
    """Random seed for reproducibility (None = random)."""

GameConfig: TypeAlias = (
    FrozenLakeConfig
    | TaxiConfig
    | CliffWalkingConfig
    | BlackjackConfig
    | LunarLanderConfig
    | CarRacingConfig
    | BipedalWalkerConfig
    | MiniGridConfig
    | CrafterConfig
    | ProcgenConfig
    | ALEConfig
    | TextWorldConfig
    | JumanjiConfig
    | MultiGridConfig
    | MeltingPotConfig
    | OvercookedConfig
    | SMACConfig
    | RWAREConfig
    | GriddlyConfig
    | GFootballConfig
)


# Default Jumanji configurations for each logic puzzle environment
DEFAULT_JUMANJI_GAME2048_CONFIG = JumanjiConfig(env_id="jumanji/Game2048-v1")
DEFAULT_JUMANJI_MINESWEEPER_CONFIG = JumanjiConfig(env_id="jumanji/Minesweeper-v0")
DEFAULT_JUMANJI_RUBIKS_CUBE_CONFIG = JumanjiConfig(env_id="jumanji/RubiksCube-v0")
DEFAULT_JUMANJI_SLIDING_PUZZLE_CONFIG = JumanjiConfig(env_id="jumanji/SlidingTilePuzzle-v0")
DEFAULT_JUMANJI_SUDOKU_CONFIG = JumanjiConfig(env_id="jumanji/Sudoku-v0")
DEFAULT_JUMANJI_GRAPH_COLORING_CONFIG = JumanjiConfig(env_id="jumanji/GraphColoring-v1")

# Default MultiGrid configurations for each multi-agent environment
DEFAULT_MULTIGRID_SOCCER_CONFIG = MultiGridConfig(env_id="soccer", highlight=True)
DEFAULT_MULTIGRID_COLLECT_CONFIG = MultiGridConfig(env_id="collect", highlight=True)

# Default Melting Pot configurations for each substrate
DEFAULT_MELTINGPOT_COLLABORATIVE_COOKING_CONFIG = MeltingPotConfig(substrate_name="collaborative_cooking__circuit")
DEFAULT_MELTINGPOT_CLEAN_UP_CONFIG = MeltingPotConfig(substrate_name="clean_up")
DEFAULT_MELTINGPOT_COMMONS_HARVEST_CONFIG = MeltingPotConfig(substrate_name="commons_harvest__open")
DEFAULT_MELTINGPOT_TERRITORY_CONFIG = MeltingPotConfig(substrate_name="territory__rooms")
DEFAULT_MELTINGPOT_KING_OF_THE_HILL_CONFIG = MeltingPotConfig(substrate_name="king_of_the_hill__repeated")
DEFAULT_MELTINGPOT_PRISONERS_DILEMMA_CONFIG = MeltingPotConfig(substrate_name="prisoners_dilemma_in_the_matrix__repeated")
DEFAULT_MELTINGPOT_STAG_HUNT_CONFIG = MeltingPotConfig(substrate_name="stag_hunt_in_the_matrix__repeated")
DEFAULT_MELTINGPOT_ALLELOPATHIC_HARVEST_CONFIG = MeltingPotConfig(substrate_name="allelopathic_harvest__open")

# Default Overcooked configurations for core research layouts
DEFAULT_OVERCOOKED_CRAMPED_ROOM_CONFIG = OvercookedConfig(layout_name="cramped_room", horizon=400)
DEFAULT_OVERCOOKED_ASYMMETRIC_ADVANTAGES_CONFIG = OvercookedConfig(layout_name="asymmetric_advantages", horizon=400)
DEFAULT_OVERCOOKED_COORDINATION_RING_CONFIG = OvercookedConfig(layout_name="coordination_ring", horizon=400)
DEFAULT_OVERCOOKED_FORCED_COORDINATION_CONFIG = OvercookedConfig(layout_name="forced_coordination", horizon=400)
DEFAULT_OVERCOOKED_COUNTER_CIRCUIT_CONFIG = OvercookedConfig(layout_name="counter_circuit", horizon=400)


__all__ = [
    "FrozenLakeConfig",
    "TaxiConfig",
    "CliffWalkingConfig",
    "BlackjackConfig",
    "LunarLanderConfig",
    "CarRacingConfig",
    "BipedalWalkerConfig",
    "MiniGridConfig",
    "CrafterConfig",
    "ProcgenConfig",
    "ALEConfig",
    "TextWorldConfig",
    "JumanjiConfig",
    "MultiGridConfig",
    "MeltingPotConfig",
    "OvercookedConfig",
    "GameConfig",
    "DEFAULT_FROZEN_LAKE_CONFIG",
    "DEFAULT_FROZEN_LAKE_V2_CONFIG",
    "DEFAULT_TAXI_CONFIG",
    "DEFAULT_CLIFF_WALKING_CONFIG",
    "DEFAULT_BLACKJACK_CONFIG",
    "DEFAULT_LUNAR_LANDER_CONFIG",
    "DEFAULT_CAR_RACING_CONFIG",
    "DEFAULT_BIPEDAL_WALKER_CONFIG",
    "DEFAULT_MINIGRID_EMPTY_5x5_CONFIG",
    "DEFAULT_MINIGRID_EMPTY_RANDOM_5x5_CONFIG",
    "DEFAULT_MINIGRID_EMPTY_6x6_CONFIG",
    "DEFAULT_MINIGRID_EMPTY_RANDOM_6x6_CONFIG",
    "DEFAULT_MINIGRID_EMPTY_8x8_CONFIG",
    "DEFAULT_MINIGRID_EMPTY_16x16_CONFIG",
    "DEFAULT_MINIGRID_DOORKEY_5x5_CONFIG",
    "DEFAULT_MINIGRID_DOORKEY_6x6_CONFIG",
    "DEFAULT_MINIGRID_DOORKEY_8x8_CONFIG",
    "DEFAULT_MINIGRID_DOORKEY_16x16_CONFIG",
    "DEFAULT_MINIGRID_LAVAGAP_S7_CONFIG",
    "DEFAULT_MINIGRID_REDBLUE_DOORS_6x6_CONFIG",
    "DEFAULT_MINIGRID_REDBLUE_DOORS_8x8_CONFIG",
    "DEFAULT_JUMANJI_GAME2048_CONFIG",
    "DEFAULT_JUMANJI_MINESWEEPER_CONFIG",
    "DEFAULT_JUMANJI_RUBIKS_CUBE_CONFIG",
    "DEFAULT_JUMANJI_SLIDING_PUZZLE_CONFIG",
    "DEFAULT_JUMANJI_SUDOKU_CONFIG",
    "DEFAULT_JUMANJI_GRAPH_COLORING_CONFIG",
    "DEFAULT_MULTIGRID_SOCCER_CONFIG",
    "DEFAULT_MULTIGRID_COLLECT_CONFIG",
    "DEFAULT_MELTINGPOT_COLLABORATIVE_COOKING_CONFIG",
    "DEFAULT_MELTINGPOT_CLEAN_UP_CONFIG",
    "DEFAULT_MELTINGPOT_COMMONS_HARVEST_CONFIG",
    "DEFAULT_MELTINGPOT_TERRITORY_CONFIG",
    "DEFAULT_MELTINGPOT_KING_OF_THE_HILL_CONFIG",
    "DEFAULT_MELTINGPOT_PRISONERS_DILEMMA_CONFIG",
    "DEFAULT_MELTINGPOT_STAG_HUNT_CONFIG",
    "DEFAULT_MELTINGPOT_ALLELOPATHIC_HARVEST_CONFIG",
    "DEFAULT_OVERCOOKED_CRAMPED_ROOM_CONFIG",
    "DEFAULT_OVERCOOKED_ASYMMETRIC_ADVANTAGES_CONFIG",
    "DEFAULT_OVERCOOKED_COORDINATION_RING_CONFIG",
    "DEFAULT_OVERCOOKED_FORCED_COORDINATION_CONFIG",
    "DEFAULT_OVERCOOKED_COUNTER_CIRCUIT_CONFIG",
]
