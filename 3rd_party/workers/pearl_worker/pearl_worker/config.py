"""Configuration parsing helpers for the Pearl worker shim."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace, asdict
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

try:
    from gym_gui.core.worker import WorkerConfig as WorkerConfigProtocol
except ImportError:
    WorkerConfigProtocol = None  # type: ignore[assignment,misc]


@dataclass
class PearlWorkerConfig:
    """Structured configuration for launching a Pearl worker run.

    Implements the MOSAIC WorkerConfig protocol for standardization.
    """

    run_id: str
    algo: str
    env_id: str
    total_timesteps: int
    seed: Optional[int] = None
    extras: dict[str, Any] = field(default_factory=dict)
    worker_id: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Validate required fields on initialization."""
        if not self.run_id:
            raise ValueError("pearl_worker config missing required field 'run_id'")
        if not self.algo:
            raise ValueError("pearl_worker config missing required field 'algo'")
        if not self.env_id:
            raise ValueError("pearl_worker config missing required field 'env_id'")
        if self.total_timesteps is None:
            raise ValueError("pearl_worker config missing required field 'total_timesteps'")

        # Assert protocol compliance (structural typing check)
        if WorkerConfigProtocol is not None:
            assert isinstance(self, WorkerConfigProtocol), (
                "PearlWorkerConfig must implement WorkerConfig protocol"
            )

    def with_overrides(
        self,
        *,
        algo: Optional[str] = None,
        env_id: Optional[str] = None,
        total_timesteps: Optional[int] = None,
        seed: Optional[int] = None,
        worker_id: Optional[str] = None,
        extras: Optional[Mapping[str, Any]] = None,
    ) -> "PearlWorkerConfig":
        """Return a new config applying CLI overrides."""

        merged_extras = dict(self.extras)
        if extras:
            merged_extras.update(extras)

        return replace(
            self,
            algo=algo or self.algo,
            env_id=env_id or self.env_id,
            total_timesteps=total_timesteps or self.total_timesteps,
            seed=self.seed if seed is None else seed,
            worker_id=worker_id if worker_id is not None else self.worker_id,
            extras=merged_extras,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to dictionary (WorkerConfig protocol requirement).

        Returns:
            Dictionary representation of configuration
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PearlWorkerConfig":
        """Deserialize configuration from dictionary (WorkerConfig protocol requirement).

        Handles both flat and nested formats for backwards compatibility.

        Args:
            data: Dictionary containing configuration values

        Returns:
            PearlWorkerConfig instance
        """
        # Extract extras if present
        extras_field = data.get("extras")
        extras: dict[str, Any]
        if isinstance(extras_field, MutableMapping):
            extras = dict(extras_field)
        else:
            extras = {}

        # Handle total_timesteps special case (eval mode)
        total_timesteps_value = data.get("total_timesteps")
        total_timesteps = int(total_timesteps_value or 0)

        if total_timesteps <= 0:
            eval_episodes = extras.get("eval_episodes")
            try:
                total_timesteps = max(1, int(eval_episodes)) if eval_episodes is not None else 1
            except (TypeError, ValueError):
                total_timesteps = 1

        return cls(
            run_id=str(data["run_id"]),
            algo=str(data["algo"]),
            env_id=str(data["env_id"]),
            total_timesteps=total_timesteps,
            seed=int(data["seed"]) if data.get("seed") is not None else None,
            extras=extras,
            worker_id=str(data["worker_id"]).strip() or None if data.get("worker_id") else None,
            raw=dict(data),
        )


def load_worker_config(path: Path) -> PearlWorkerConfig:
    """Load and parse a trainer-generated worker config JSON file.

    Args:
        path: Path to JSON configuration file

    Returns:
        PearlWorkerConfig instance
    """
    payload = json.loads(path.read_text(encoding="utf-8"))

    # Handle nested metadata structure if present (from main app trainer)
    if isinstance(payload, Mapping) and "metadata" in payload:
        worker_section = payload.get("metadata", {}).get("worker", {})
        worker_config = worker_section.get("config", {})
        return PearlWorkerConfig.from_dict(worker_config)

    return PearlWorkerConfig.from_dict(payload)


# Type alias for backwards compatibility
WorkerConfig = PearlWorkerConfig
