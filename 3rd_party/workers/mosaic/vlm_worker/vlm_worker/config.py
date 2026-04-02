"""Configuration dataclass for VLM Worker.

Defines all configuration options for running VLM (Vision-Language Model) agents
on RL environments with image observation support.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional


# Valid environment names
# Note: "minigrid" uses the same wrapper as "babyai" (BabyAI is built on MiniGrid)
# Note: "multigrid" is multi-agent extension of MiniGrid with simultaneous stepping
# Note: "pettingzoo" is used for multi-agent games (chess, go, connect-four, etc.)
#       In pettingzoo mode, the worker acts as action-selector (doesn't own env)
ENV_NAMES = ("babyai", "minigrid", "multigrid", "mosaic_multigrid", "minihack", "crafter", "nle", "textworld", "pettingzoo")

# Valid LLM client names
# OpenRouter provides unified access to all major model providers
# Zhipu AI (api.z.ai / bigmodel.cn) uses OpenAI-compatible API
CLIENT_NAMES = ("openrouter", "openai", "anthropic", "google", "vllm", "zhipu")

# Valid agent types
AGENT_TYPES = ("naive", "cot", "robust_naive", "robust_cot", "few_shot", "dummy")


@dataclass
class VLMWorkerConfig:
    """Configuration for VLM Worker subprocess.

    This config controls which environment, LLM client, and agent type to use,
    along with episode limits, vision settings, and telemetry output.

    Attributes:
        run_id: Unique identifier for this run (from GUI).
        env_name: Environment to use (babyai, minihack, crafter, etc.).
        task: Specific task/level within the environment.
        client_name: LLM client to use (openai, anthropic, google, vllm).
        model_id: Model identifier (e.g., "gpt-4o-mini", "claude-3-5-sonnet").
        agent_type: Agent strategy (naive, cot, robust_naive, etc.).
        num_episodes: Number of episodes to run.
        max_steps: Maximum steps per episode before truncation.
        temperature: LLM sampling temperature.
        api_key: API key for the LLM client (can also use env vars).
        base_url: Custom base URL for LLM API (for vLLM or proxies).
        telemetry_dir: Directory to write telemetry output.
        emit_jsonl: Whether to emit JSONL telemetry files.
        seed: Random seed for reproducibility.
        render_mode: Rendering mode for environments (None, "human", "rgb_array").
    """

    run_id: str
    env_name: Literal["babyai", "minigrid", "multigrid", "mosaic_multigrid", "minihack", "crafter", "nle", "textworld", "pettingzoo"] = "babyai"
    task: str = "BabyAI-GoToRedBall-v0"
    client_name: Literal["openrouter", "openai", "anthropic", "google", "vllm"] = "openrouter"
    model_id: str = "openai/gpt-4o-mini"  # OpenRouter format: provider/model
    agent_type: Literal["naive", "cot", "robust_naive", "robust_cot", "few_shot", "dummy"] = "naive"
    num_episodes: int = 5
    max_steps: int = 100
    temperature: float = 0.7
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    telemetry_dir: Optional[str] = None  # Resolved in __post_init__ to VAR_TELEMETRY_DIR
    emit_jsonl: bool = True
    seed: Optional[int] = None
    render_mode: Optional[str] = None
    # Advanced options
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 60.0
    alternate_roles: bool = True
    # VLM (Vision-Language Model) settings
    # Default to 1 for VLM worker (vision-enabled)
    # Set to 0 for text-only fallback mode
    max_image_history: int = 1
    # MultiGrid-specific settings (for multi-agent environments)
    agent_id: int = 0  # Agent index for multi-agent environments
    observation_mode: str = "visible_teammates"  # "egocentric" or "visible_teammates"
    coordination_level: int = 1  # 1=Emergent, 2=Basic Hints, 3=Role-Based
    role: Optional[str] = None  # Agent role for Level 3 (e.g., "forward", "defender")

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.env_name not in ENV_NAMES:
            raise ValueError(
                f"Invalid env_name '{self.env_name}'. "
                f"Must be one of: {ENV_NAMES}"
            )
        if self.client_name not in CLIENT_NAMES:
            raise ValueError(
                f"Invalid client_name '{self.client_name}'. "
                f"Must be one of: {CLIENT_NAMES}"
            )
        if self.agent_type not in AGENT_TYPES:
            raise ValueError(
                f"Invalid agent_type '{self.agent_type}'. "
                f"Must be one of: {AGENT_TYPES}"
            )
        if self.num_episodes < 1:
            raise ValueError("num_episodes must be >= 1")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")

        # Resolve telemetry_dir to VAR_OPERATORS_TELEMETRY_DIR if not specified
        if self.telemetry_dir is None:
            try:
                from gym_gui.config.paths import VAR_OPERATORS_TELEMETRY_DIR
                object.__setattr__(self, "telemetry_dir", str(VAR_OPERATORS_TELEMETRY_DIR))
            except ImportError:
                object.__setattr__(self, "telemetry_dir", "./telemetry")

        # Protocol compliance assertion (only if gym_gui available)
        try:
            from gym_gui.core.worker import WorkerConfig as WorkerConfigProtocol
            assert isinstance(self, WorkerConfigProtocol), (
                "VLMWorkerConfig must implement WorkerConfig protocol"
            )
        except ImportError:
            pass  # gym_gui not available, skip protocol check

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "run_id": self.run_id,
            "env_name": self.env_name,
            "task": self.task,
            "client_name": self.client_name,
            "model_id": self.model_id,
            "agent_type": self.agent_type,
            "num_episodes": self.num_episodes,
            "max_steps": self.max_steps,
            "temperature": self.temperature,
            "api_key": "***" if self.api_key else None,  # Mask for safety
            "base_url": self.base_url,
            "telemetry_dir": self.telemetry_dir,
            "emit_jsonl": self.emit_jsonl,
            "seed": self.seed,
            "render_mode": self.render_mode,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "timeout": self.timeout,
            "alternate_roles": self.alternate_roles,
            "observation_mode": self.observation_mode,
            "coordination_level": self.coordination_level,
            "role": self.role,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert config to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VLMWorkerConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json_file(cls, path: str | Path) -> "VLMWorkerConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_llm_config(self) -> Dict[str, Any]:
        """Convert to OmegaConf structure for AgentFactory.

        Returns a dict that can be converted to OmegaConf for use with
        AgentFactory and make_env functions.
        """
        return {
            "client": {
                "client_name": self.client_name,
                "model_id": self.model_id,
                "base_url": self.base_url,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
                "delay": self.retry_delay,
                "alternate_roles": self.alternate_roles,
                "generate_kwargs": {
                    "temperature": self.temperature,
                },
            },
            "agent": {
                "type": self.agent_type,
                "max_icl_history": 5,  # For few_shot agent
                "max_text_history": 10,  # Max text history entries
                "max_image_history": self.max_image_history,  # 0 for text-only, >= 1 for VLM
                "max_cot_history": 5,  # Max chain-of-thought history entries
            },
            "envs": {
                "names": self.env_name,
                "max_steps": self.max_steps,
            },
            "eval": {
                "num_episodes": self.num_episodes,
                "seed": self.seed,
            },
        }


def load_worker_config(config_path: str) -> VLMWorkerConfig:
    """Load worker configuration from a JSON file path.

    This is the main entry point for loading config from CLI.
    Handles both direct config format and nested metadata.worker.config structure.

    Args:
        config_path: Path to the JSON configuration file

    Returns:
        Parsed VLMWorkerConfig object
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

    return VLMWorkerConfig.from_dict(config_data)


__all__ = [
    "VLMWorkerConfig",
    "load_worker_config",
    "ENV_NAMES",
    "CLIENT_NAMES",
    "AGENT_TYPES",
]
