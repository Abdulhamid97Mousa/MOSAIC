"""Pearl worker integration scaffolding."""

from __future__ import annotations

from .config import WorkerConfig, load_worker_config
from .runtime import PearlWorkerRuntime

try:
    from gym_gui.core.worker import WorkerMetadata, WorkerCapabilities
except ImportError:
    WorkerMetadata = None  # type: ignore[assignment,misc]
    WorkerCapabilities = None  # type: ignore[assignment,misc]


def get_worker_metadata() -> tuple:
    """Get Pearl worker metadata and capabilities for MOSAIC discovery.

    This function is called by the worker discovery system to populate
    the worker registry with Pearl's metadata and capabilities.

    Returns:
        tuple: (WorkerMetadata, WorkerCapabilities)
    """
    if WorkerMetadata is None or WorkerCapabilities is None:
        raise RuntimeError(
            "gym_gui.core.worker is not available; "
            "install the main MOSAIC package to use discovery"
        )

    metadata = WorkerMetadata(
        name="Pearl Worker",
        version=__version__,
        description="Production-Ready Reinforcement Learning Agent (Meta Pearl)",
        author="MOSAIC Team",
        homepage="https://github.com/facebookresearch/Pearl",
        upstream_library="pearl",
        upstream_version="0.1.0",
        license="MIT",
    )

    capabilities = WorkerCapabilities(
        worker_type="pearl",
        supported_paradigms=("sequential",),
        env_families=("gymnasium",),
        action_spaces=("discrete", "continuous"),
        observation_spaces=("vector",),
        max_agents=1,
        supports_self_play=False,
        supports_population=False,
        supports_checkpointing=False,
        supports_pause_resume=False,
        requires_gpu=False,
        gpu_memory_mb=None,
        cpu_cores=1,
        estimated_memory_mb=512,
    )

    return metadata, capabilities


__all__ = [
    "PearlWorkerRuntime",
    "WorkerConfig",
    "load_worker_config",
    "get_worker_metadata",
]

__version__ = "0.1.0"
