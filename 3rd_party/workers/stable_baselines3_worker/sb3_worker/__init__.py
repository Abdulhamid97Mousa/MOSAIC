"""Stable-Baselines3 worker integration scaffolding."""

from __future__ import annotations

from .config import WorkerConfig, load_worker_config
from .runtime import SB3WorkerRuntime

try:
    from gym_gui.core.worker import WorkerMetadata, WorkerCapabilities
except ImportError:
    WorkerMetadata = None
    WorkerCapabilities = None


def get_worker_metadata():
    """Get SB3 worker metadata and capabilities for MOSAIC discovery.

    This function is called by the worker discovery system to populate
    the worker registry with Stable-Baselines3 metadata and capabilities.

    Returns:
        tuple: (WorkerMetadata, WorkerCapabilities)
    """
    if WorkerMetadata is None or WorkerCapabilities is None:
        raise ImportError(
            "gym_gui.core.worker is required for worker metadata discovery"
        )

    metadata = WorkerMetadata(
        name="Stable-Baselines3 Worker",
        version=__version__,
        description="Reliable RL implementations (Stable-Baselines3)",
        author="MOSAIC Team",
        homepage="https://github.com/DLR-RM/stable-baselines3",
        upstream_library="stable-baselines3",
        upstream_version="2.4.0",
        license="MIT",
    )

    capabilities = WorkerCapabilities(
        worker_type="sb3",
        supported_paradigms=("sequential",),
        env_families=("gymnasium", "atari", "mujoco"),
        action_spaces=("discrete", "continuous", "multi_discrete", "multi_binary"),
        observation_spaces=("vector", "image", "dict"),
        max_agents=1,
        supports_self_play=False,
        supports_population=False,
        supports_checkpointing=True,
        supports_pause_resume=False,
        requires_gpu=False,
        gpu_memory_mb=None,
        cpu_cores=1,
        estimated_memory_mb=512,
    )

    return metadata, capabilities


__all__ = [
    "SB3WorkerRuntime",
    "WorkerConfig",
    "load_worker_config",
    "get_worker_metadata",
]

__version__ = "0.1.0"
