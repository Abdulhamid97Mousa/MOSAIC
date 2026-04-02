"""SBX (Stable-Baselines Jax) worker integration scaffolding."""

from __future__ import annotations

from .config import WorkerConfig, load_worker_config
from .runtime import SBXWorkerRuntime

try:
    from gym_gui.core.worker import WorkerMetadata, WorkerCapabilities
except ImportError:
    WorkerMetadata = None
    WorkerCapabilities = None


def get_worker_metadata():
    """Get SBX worker metadata and capabilities for MOSAIC discovery.

    This function is called by the worker discovery system to populate
    the worker registry with SBX metadata and capabilities.

    Returns:
        tuple: (WorkerMetadata, WorkerCapabilities)
    """
    if WorkerMetadata is None or WorkerCapabilities is None:
        raise ImportError(
            "gym_gui.core.worker is required for worker metadata discovery"
        )

    metadata = WorkerMetadata(
        name="SBX Worker",
        version=__version__,
        description="JAX-based RL implementations (SBX, drop-in replacement for SB3)",
        author="MOSAIC Team",
        homepage="https://github.com/araffin/sbx",
        upstream_library="sbx",
        upstream_version="0.18.0",
        license="MIT",
    )

    capabilities = WorkerCapabilities(
        worker_type="sbx",
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
    "SBXWorkerRuntime",
    "WorkerConfig",
    "load_worker_config",
    "get_worker_metadata",
]

__version__ = "0.1.0"
