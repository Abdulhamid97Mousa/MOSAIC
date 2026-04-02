"""Configuration for RLtools worker."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class RLToolsWorkerConfig:
    """Configuration for RLtools training runs."""

    run_id: str
    algo: str  # "sac", "td3", "ppo"
    env_id: str
    total_timesteps: int
    seed: Optional[int] = None
    extras: Dict[str, Any] = field(default_factory=dict)
    worker_id: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if not self.run_id:
            raise ValueError("run_id is required")
        if not self.algo:
            raise ValueError("algo is required")
        if not self.env_id:
            raise ValueError("env_id is required")
        if self.total_timesteps < 1:
            raise ValueError("total_timesteps must be >= 1")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "algo": self.algo,
            "env_id": self.env_id,
            "total_timesteps": self.total_timesteps,
            "seed": self.seed,
            "extras": self.extras,
            "worker_id": self.worker_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RLToolsWorkerConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def load_worker_config(config_path: str) -> RLToolsWorkerConfig:
    path = Path(config_path)
    with open(path) as f:
        raw_data = json.load(f)
    if "metadata" in raw_data and "worker" in raw_data["metadata"]:
        config_data = raw_data["metadata"]["worker"].get("config", {})
    else:
        config_data = raw_data
    return RLToolsWorkerConfig.from_dict(config_data)
