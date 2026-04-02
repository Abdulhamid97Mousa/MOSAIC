"""Human Worker - Human-in-the-loop action selection for multi-agent games."""

__version__ = "0.1.0"

from .runtime import HumanKeyboardRuntime, HumanWorkerRuntime
from .config import HumanWorkerConfig


def get_worker_metadata() -> tuple:
    """Return worker metadata for MOSAIC registry.

    Returns:
        Tuple of (WorkerMetadata, WorkerCapabilities)
    """
    from gym_gui.core.worker import WorkerMetadata, WorkerCapabilities

    metadata = WorkerMetadata(
        name="Human Worker",
        version=__version__,
        description="Human-in-the-loop action selection via GUI clicks",
        author="MOSAIC Team",
        homepage="https://github.com/Abdulhamid97Mousa/MOSAIC",
        upstream_library=None,
        upstream_version=None,
        license="MIT",
    )

    capabilities = WorkerCapabilities(
        worker_type="human",
        supported_paradigms=("human_vs_ai", "human_vs_human"),
        env_families=("pettingzoo", "gymnasium"),
        action_spaces=("discrete",),
        observation_spaces=("structured", "rgb"),
        max_agents=1,
        supports_self_play=False,
        supports_population=False,
        supports_checkpointing=False,
        supports_pause_resume=True,
        requires_gpu=False,
        gpu_memory_mb=None,
        cpu_cores=1,
        estimated_memory_mb=256,
    )

    return metadata, capabilities


__all__ = [
    "HumanKeyboardRuntime",
    "HumanWorkerRuntime",
    "HumanWorkerConfig",
    "__version__",
    "get_worker_metadata",
]
