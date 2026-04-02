"""MushroomRL worker integration scaffolding."""

from __future__ import annotations

from .config import WorkerConfig, load_worker_config
from .runtime import MushroomRLWorkerRuntime

try:
    from gym_gui.core.worker import WorkerMetadata, WorkerCapabilities
except ImportError:  # pragma: no cover
    WorkerMetadata = None  # type: ignore[assignment,misc]
    WorkerCapabilities = None  # type: ignore[assignment,misc]


def get_worker_metadata() -> tuple:
    """Get MushroomRL worker metadata and capabilities for MOSAIC discovery.

    This function is called by the worker discovery system to populate
    the worker registry with MushroomRL's metadata and capabilities.

    Returns:
        tuple: (WorkerMetadata, WorkerCapabilities)
    """
    if WorkerMetadata is None or WorkerCapabilities is None:
        raise RuntimeError(
            "gym_gui.core.worker is not available; "
            "install the full MOSAIC package to use worker discovery"
        )

    metadata = WorkerMetadata(
        name="MushroomRL Worker",
        version=__version__,
        description="Reinforcement Learning library (MushroomRL)",
        author="MOSAIC Team",
        homepage="https://github.com/MushroomRL/mushroom-rl",
        upstream_library="mushroom_rl",
        upstream_version="1.9.0",
        license="MIT",
    )

    capabilities = WorkerCapabilities(
        worker_type="mushroomrl",
        supported_paradigms=("sequential",),
        env_families=("gymnasium",),
        action_spaces=("continuous",),
        observation_spaces=("vector",),
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
    "MushroomRLWorkerRuntime",
    "WorkerConfig",
    "load_worker_config",
    "get_worker_metadata",
]

__version__ = "0.1.0"
