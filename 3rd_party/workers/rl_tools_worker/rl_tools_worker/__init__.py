"""RLtools MOSAIC Worker.

Wraps the rltools Python interface (JIT-compiled C++ RL library) for
benchmarking and integration with MOSAIC's agent-mixing platform.
"""

__version__ = "0.1.0"


def get_worker_metadata():
    """Return worker metadata for MOSAIC discovery."""
    try:
        from gym_gui.core.worker import WorkerMetadata, WorkerCapabilities
    except ImportError:
        return None

    metadata = WorkerMetadata(
        name="RLtools Worker",
        version=__version__,
        description="JIT-compiled C++ RL: fastest training via rltools Python wrapper",
        author="Jonas Eschmann, Dario Albani, Giuseppe Loianno",
        homepage="https://rl.tools",
        upstream_library="rltools",
        upstream_version="2.2.1",
        license="MIT",
    )

    capabilities = WorkerCapabilities(
        worker_type="rltools",
        supported_paradigms=("sequential",),
        env_families=("gymnasium",),
        action_spaces=("continuous",),
        observation_spaces=("vector",),
        max_agents=1,
        supports_checkpointing=True,
        requires_gpu=False,
    )

    return metadata, capabilities


__all__ = ["get_worker_metadata"]
