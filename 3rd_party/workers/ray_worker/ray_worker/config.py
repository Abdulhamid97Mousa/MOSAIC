"""Configuration dataclasses for Ray RLlib multi-agent worker.

This module defines the configuration structure for multi-agent training
with Ray RLlib, supporting various policy configurations:
- Parameter Sharing: All agents share one policy (homogeneous cooperative)
- Independent Learning: Each agent has its own policy (heterogeneous)
- Self-Play: Agent plays against copies of itself (competitive)

Integrates with MOSAIC's trainer infrastructure:
- var/trainer/runs/{run_id}/ for run artifacts
- var/trainer/runs/{run_id}/tensorboard/ for TensorBoard logs
- var/trainer/runs/{run_id}/checkpoints/ for policy checkpoints
- var/trainer/runs/{run_id}/logs/ for worker logs
- var/trainer/runs/{run_id}/analytics.json for analytics manifest
- var/trainer/trainer.sqlite for run metadata

See Also:
    - gym_gui/config/paths.py for path constants
    - gym_gui/constants/constants_trainer.py for trainer defaults
    - gym_gui/constants/constants_tensorboard.py for TensorBoard helpers
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import paths from gym_gui if available, otherwise use defaults
try:
    from gym_gui.config.paths import (
        VAR_ROOT,
        VAR_TRAINER_DIR,
        VAR_TENSORBOARD_DIR,
        VAR_WANDB_DIR,
        ensure_var_directories,
    )
    from gym_gui.constants.constants_tensorboard import (
        build_tensorboard_log_dir,
        build_tensorboard_relative_path,
    )
    _USING_GYM_PATHS = True
except ImportError:
    # Fallback for standalone worker execution
    _REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
    VAR_ROOT = _REPO_ROOT / "var"
    VAR_TRAINER_DIR = VAR_ROOT / "trainer"
    VAR_TENSORBOARD_DIR = VAR_TRAINER_DIR / "runs"
    VAR_WANDB_DIR = VAR_TRAINER_DIR / "runs"
    _USING_GYM_PATHS = False

    def ensure_var_directories() -> None:
        """Create writable directories if they don't exist."""
        for path in (VAR_ROOT, VAR_TRAINER_DIR, VAR_TENSORBOARD_DIR, VAR_WANDB_DIR):
            path.mkdir(parents=True, exist_ok=True)

    def build_tensorboard_log_dir(run_id: str, worker_id: Optional[str] = None) -> Path:
        """Return absolute path for TensorBoard directory."""
        return (VAR_TENSORBOARD_DIR / run_id / "tensorboard").resolve()

    def build_tensorboard_relative_path(run_id: str, worker_id: Optional[str] = None) -> str:
        """Return relative path for TensorBoard directory."""
        return f"var/trainer/runs/{run_id}/tensorboard"


class PolicyConfiguration(str, Enum):
    """Multi-agent policy configuration strategy.

    Defines how agents map to policies in multi-agent training:
    - PARAMETER_SHARING: All agents share one policy (homogeneous cooperative)
    - INDEPENDENT: Each agent has its own policy (heterogeneous learning)
    - SELF_PLAY: Agents train against copies of themselves (competitive)
    - SHARED_VALUE_FUNCTION: CTDE - Centralized Training, Decentralized Execution
    """

    PARAMETER_SHARING = "parameter_sharing"
    INDEPENDENT = "independent"
    SELF_PLAY = "self_play"
    SHARED_VALUE_FUNCTION = "shared_value_function"


# Backwards compatibility alias (deprecated - use PolicyConfiguration)
TrainingParadigm = PolicyConfiguration


class PettingZooAPIType(str, Enum):
    """PettingZoo environment API type."""

    AEC = "aec"        # Agent Environment Cycle (turn-based)
    PARALLEL = "parallel"  # Simultaneous actions
    GYM = "gym"        # Standard Gymnasium/Gym API (single agent)


@dataclass(frozen=True)
class ResourceConfig:
    """Resource allocation configuration."""

    num_workers: int = 2
    num_gpus: int = 0
    num_cpus_per_worker: int = 1
    num_gpus_per_worker: float = 0.0


@dataclass
class TrainingConfig:
    """Training hyperparameters with dynamic algorithm parameters.

    The algo_params field contains algorithm-specific parameters that are
    validated against schemas defined in metadata/ray_rllib/schemas.json.
    This ensures only valid parameters for the selected algorithm are used,
    preventing runtime crashes from invalid parameter combinations.

    Example:
        # PPO-specific parameters
        TrainingConfig(
            algorithm="PPO",
            algo_params={
                "lr": 0.0003,
                "gamma": 0.99,
                "clip_param": 0.3,
                "sgd_minibatch_size": 128,  # PPO only
                "num_sgd_iter": 30,         # PPO only
            }
        )

        # APPO parameters (no minibatch_size or num_sgd_iter!)
        TrainingConfig(
            algorithm="APPO",
            algo_params={
                "lr": 0.0005,
                "gamma": 0.99,
                "vtrace": True,  # APPO only
            }
        )
    """

    algorithm: str = "PPO"
    total_timesteps: int = 1_000_000
    algo_params: Dict[str, Any] = field(default_factory=dict)

    def get_param(self, name: str, default: Any = None) -> Any:
        """Get an algorithm parameter value.

        Args:
            name: Parameter name
            default: Default value if not set

        Returns:
            Parameter value or default
        """
        return self.algo_params.get(name, default)

    def with_defaults(self) -> "TrainingConfig":
        """Return a new TrainingConfig with schema defaults merged in.

        Uses algo_params.merge_with_defaults() to fill in missing values.
        """
        try:
            from .algo_params import merge_with_defaults
            merged_params = merge_with_defaults(self.algorithm, self.algo_params)
            return TrainingConfig(
                algorithm=self.algorithm,
                total_timesteps=self.total_timesteps,
                algo_params=merged_params,
            )
        except ImportError:
            # Fallback if algo_params module not available
            return self


@dataclass(frozen=True)
class CheckpointConfig:
    """Checkpoint and logging configuration."""

    checkpoint_freq: int = 10  # Save every N iterations
    checkpoint_at_end: bool = True
    keep_checkpoints_num: int = 5
    export_policy: bool = True


@dataclass(frozen=True)
class EnvironmentConfig:
    """PettingZoo environment configuration."""

    family: str  # "sisl", "classic", "butterfly", "mpe"
    env_id: str  # "waterworld_v4", "chess_v6"
    api_type: PettingZooAPIType = PettingZooAPIType.PARALLEL
    render_mode: Optional[str] = None
    env_kwargs: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_env_id(self) -> str:
        """Get the full environment identifier."""
        return f"{self.family}/{self.env_id}"


@dataclass
class RayWorkerConfig:
    """Complete configuration for Ray RLlib multi-agent worker.

    This configuration is passed via JSON file from the UI to the worker CLI.

    Example:
        config = RayWorkerConfig(
            run_id="waterworld_ppo_20241213",
            environment=EnvironmentConfig(
                family="sisl",
                env_id="waterworld_v4",
                api_type=PettingZooAPIType.PARALLEL,
            ),
            policy_configuration=PolicyConfiguration.PARAMETER_SHARING,
        )
    """

    # Run identification
    run_id: str

    # Environment configuration
    environment: EnvironmentConfig

    # Policy configuration strategy
    policy_configuration: PolicyConfiguration = PolicyConfiguration.PARAMETER_SHARING

    # Training hyperparameters
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Resource allocation
    resources: ResourceConfig = field(default_factory=ResourceConfig)

    # Checkpointing
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    # Output directory (defaults to var/trainer/runs)
    output_dir: Optional[str] = None  # None means use VAR_TRAINER_DIR/runs

    # Random seed (optional)
    seed: Optional[int] = None

    # TensorBoard logging
    tensorboard: bool = True
    tensorboard_dir: Optional[str] = None

    # WandB logging
    wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # FastLane live visualization
    fastlane_enabled: bool = True  # Enable live rendering in GUI
    fastlane_throttle_ms: int = 33  # ~30 FPS

    # Extra configuration passed through
    extras: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate required fields on initialization."""
        if not self.run_id:
            raise ValueError("ray_worker config missing required field 'run_id'")

        if not isinstance(self.environment, EnvironmentConfig):
            raise TypeError("environment must be EnvironmentConfig instance")

        if not self.environment.family:
            raise ValueError("environment.family is required")

        if not self.environment.env_id:
            raise ValueError("environment.env_id is required")

        if not isinstance(self.policy_configuration, PolicyConfiguration):
            raise TypeError(f"policy_configuration must be PolicyConfiguration enum, got {type(self.policy_configuration)}")

        # Protocol compliance assertion
        from gym_gui.core.worker import WorkerConfig as WorkerConfigProtocol
        assert isinstance(self, WorkerConfigProtocol), (
            "RayWorkerConfig must implement WorkerConfig protocol"
        )

    # --- Path resolution methods (integrates with var/trainer infrastructure) ---

    @property
    def run_dir(self) -> Path:
        """Get the run directory path (var/trainer/runs/{run_id})."""
        ensure_var_directories()
        if self.output_dir:
            # Always resolve to absolute path (Ray requires absolute paths for checkpoints)
            return (Path(self.output_dir) / self.run_id).resolve()
        return (VAR_TRAINER_DIR / "runs" / self.run_id).resolve()

    @property
    def checkpoint_dir(self) -> Path:
        """Get the checkpoint directory path."""
        return self.run_dir / "checkpoints"

    @property
    def logs_dir(self) -> Path:
        """Get the logs directory path."""
        return self.run_dir / "logs"

    @property
    def videos_dir(self) -> Path:
        """Get the videos directory path."""
        return self.run_dir / "videos"

    @property
    def tensorboard_log_dir(self) -> Optional[Path]:
        """Get the TensorBoard log directory path."""
        if not self.tensorboard:
            return None
        if self.tensorboard_dir:
            return Path(self.tensorboard_dir)
        # Use run_dir/tensorboard to respect output_dir setting
        return self.run_dir / "tensorboard"

    @property
    def tensorboard_relative_path(self) -> Optional[str]:
        """Get the relative TensorBoard path for analytics manifest."""
        if not self.tensorboard:
            return None
        return "tensorboard"  # Relative to run_dir

    @property
    def analytics_manifest_path(self) -> Path:
        """Get the path for analytics.json manifest."""
        return self.run_dir / "analytics.json"

    def ensure_run_directories(self) -> None:
        """Create all necessary run directories."""
        for path in (self.run_dir, self.checkpoint_dir, self.logs_dir, self.videos_dir):
            path.mkdir(parents=True, exist_ok=True)
        if self.tensorboard_log_dir:
            self.tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "environment": {
                "family": self.environment.family,
                "env_id": self.environment.env_id,
                "api_type": self.environment.api_type.value,
                "render_mode": self.environment.render_mode,
                "env_kwargs": self.environment.env_kwargs,
            },
            "policy_configuration": self.policy_configuration.value,
            "paradigm": self.policy_configuration.value,  # Backwards compatibility (deprecated)
            "training": {
                "algorithm": self.training.algorithm,
                "total_timesteps": self.training.total_timesteps,
                "algo_params": self.training.algo_params,
            },
            "resources": {
                "num_workers": self.resources.num_workers,
                "num_gpus": self.resources.num_gpus,
                "num_cpus_per_worker": self.resources.num_cpus_per_worker,
                "num_gpus_per_worker": self.resources.num_gpus_per_worker,
            },
            "checkpoint": {
                "checkpoint_freq": self.checkpoint.checkpoint_freq,
                "checkpoint_at_end": self.checkpoint.checkpoint_at_end,
                "keep_checkpoints_num": self.checkpoint.keep_checkpoints_num,
                "export_policy": self.checkpoint.export_policy,
            },
            "output_dir": self.output_dir,
            "seed": self.seed,
            "tensorboard": self.tensorboard,
            "tensorboard_dir": self.tensorboard_dir,
            "wandb": self.wandb,
            "wandb_project": self.wandb_project,
            "wandb_entity": self.wandb_entity,
            "wandb_run_name": self.wandb_run_name,
            "fastlane_enabled": self.fastlane_enabled,
            "fastlane_throttle_ms": self.fastlane_throttle_ms,
            "extras": self.extras,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RayWorkerConfig":
        """Create config from dictionary (loaded from JSON)."""
        env_data = data.get("environment", {})
        environment = EnvironmentConfig(
            family=env_data.get("family", ""),
            env_id=env_data.get("env_id", ""),
            api_type=PettingZooAPIType(env_data.get("api_type", "parallel")),
            render_mode=env_data.get("render_mode"),
            env_kwargs=env_data.get("env_kwargs", {}),
        )

        training_data = data.get("training", {})

        # Support both new format (algo_params dict) and legacy format (flat fields)
        if "algo_params" in training_data:
            # New format: algorithm params in nested dict
            algo_params = training_data.get("algo_params", {})
        else:
            # Legacy format: extract known fields into algo_params
            # This ensures backward compatibility with old config files
            legacy_fields = [
                "lr", "gamma", "lambda_", "clip_param", "vf_loss_coeff",
                "entropy_coeff", "train_batch_size", "sgd_minibatch_size",
                "num_sgd_iter", "vtrace", "use_kl_loss", "tau", "initial_alpha",
                "n_step", "target_network_update_freq", "double_q", "dueling",
            ]
            algo_params = {
                k: v for k, v in training_data.items()
                if k in legacy_fields and v is not None
            }

        training = TrainingConfig(
            algorithm=training_data.get("algorithm", "PPO"),
            total_timesteps=training_data.get("total_timesteps", 1_000_000),
            algo_params=algo_params,
        )

        resources_data = data.get("resources", {})
        resources = ResourceConfig(
            num_workers=resources_data.get("num_workers", 2),
            num_gpus=resources_data.get("num_gpus", 0),
            num_cpus_per_worker=resources_data.get("num_cpus_per_worker", 1),
            num_gpus_per_worker=resources_data.get("num_gpus_per_worker", 0.0),
        )

        checkpoint_data = data.get("checkpoint", {})
        checkpoint = CheckpointConfig(
            checkpoint_freq=checkpoint_data.get("checkpoint_freq", 10),
            checkpoint_at_end=checkpoint_data.get("checkpoint_at_end", True),
            keep_checkpoints_num=checkpoint_data.get("keep_checkpoints_num", 5),
            export_policy=checkpoint_data.get("export_policy", True),
        )

        # Support both new (policy_configuration) and old (paradigm) keys for backwards compatibility
        policy_config_str = data.get("policy_configuration") or data.get("paradigm", "parameter_sharing")
        policy_configuration = PolicyConfiguration(policy_config_str)

        return cls(
            run_id=data.get("run_id", ""),
            environment=environment,
            policy_configuration=policy_configuration,
            training=training,
            resources=resources,
            checkpoint=checkpoint,
            output_dir=data.get("output_dir"),  # None = use var/trainer/runs/
            seed=data.get("seed"),
            tensorboard=data.get("tensorboard", True),
            tensorboard_dir=data.get("tensorboard_dir"),
            wandb=data.get("wandb", False),
            wandb_project=data.get("wandb_project"),
            wandb_entity=data.get("wandb_entity"),
            wandb_run_name=data.get("wandb_run_name"),
            fastlane_enabled=data.get("fastlane_enabled", True),
            fastlane_throttle_ms=data.get("fastlane_throttle_ms", 33),
            extras=data.get("extras", {}),
        )

    def save(self, path: Path) -> None:
        """Save config to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "RayWorkerConfig":
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


def load_worker_config(config_path: str) -> RayWorkerConfig:
    """Load worker configuration from a JSON file path.

    This is the main entry point for loading config from CLI.

    Args:
        config_path: Path to the JSON configuration file

    Returns:
        Parsed RayWorkerConfig object
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        raw_data = json.load(f)

    # Handle nested metadata.worker.config structure from UI
    if "metadata" in raw_data and "worker" in raw_data["metadata"]:
        worker_data = raw_data["metadata"]["worker"]
        config_data = worker_data.get("config", {})
    else:
        # Direct config format
        config_data = raw_data

    return RayWorkerConfig.from_dict(config_data)


__all__ = [
    "PolicyConfiguration",
    "TrainingParadigm",  # Backwards compatibility alias (deprecated)
    "PettingZooAPIType",
    "ResourceConfig",
    "TrainingConfig",
    "CheckpointConfig",
    "EnvironmentConfig",
    "RayWorkerConfig",
    "load_worker_config",
]
