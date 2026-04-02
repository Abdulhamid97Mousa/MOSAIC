"""Tianshou worker integration scaffolding."""

from __future__ import annotations

from .config import WorkerConfig, load_worker_config
from .runtime import TianshouWorkerRuntime
# from . import wrappers  # Uncomment when wrappers are implemented

from gym_gui.core.worker import WorkerMetadata, WorkerCapabilities


def get_worker_metadata() -> tuple[WorkerMetadata, WorkerCapabilities]:
    """Get Tianshou worker metadata and capabilities for MOSAIC discovery.

    This function is called by the worker discovery system to populate
    the worker registry with Tianshou's metadata and capabilities.

    Returns:
        tuple: (WorkerMetadata, WorkerCapabilities)
    """
    metadata = WorkerMetadata(
        name="Tianshou Worker",
        version=__version__,
        description="Deep Reinforcement Learning Platform (Tianshou)",
        author="MOSAIC Team",
        homepage="https://github.com/thu-ml/tianshou",
        upstream_library="tianshou",
        upstream_version="2.0.0",
        license="MIT",
    )

    capabilities = WorkerCapabilities(
        worker_type="tianshou",
        supported_paradigms=("sequential",),
        env_families=("gymnasium", "atari", "mujoco", "pettingzoo"),
        action_spaces=("discrete", "continuous"),
        observation_spaces=("vector", "image"),
        max_agents=1, # Tianshou supports MARL but starting simple
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
    "TianshouWorkerRuntime",
    "WorkerConfig",
    "load_worker_config",
    "get_worker_metadata",
]

__version__ = "0.1.0"
