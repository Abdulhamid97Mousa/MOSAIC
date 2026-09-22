"""JaxMARL-based multi-agent RL worker for MOSAIC.

Algorithms: IPPO (independent), MAPPO (global centralized critic).
Native JAX/Flax — GPU-vectorized training over hundreds of parallel envs.
Sports: Soccer, American Football, Basketball.
"""

__version__ = "0.2.0"


def get_worker_metadata():
    """Return (WorkerMetadata, WorkerCapabilities) for MOSAIC discovery."""
    try:
        from gym_gui.core.worker import WorkerCapabilities, WorkerMetadata

        metadata = WorkerMetadata(
            name="JaxMARL Worker",
            version=__version__,
            description=(
                "GPU-accelerated multi-agent RL using JAX/Flax. "
                "Algorithms: IPPO, MAPPO (global critic). "
                "Native MOSAIC sports environments: Soccer, American Football, Basketball."
            ),
            author="MOSAIC Team",
            homepage="https://github.com/FLAIROx/JaxMARL",
            upstream_library="jaxmarl",
            upstream_version="0.0.4",
            license="Apache-2.0",
        )
        capabilities = WorkerCapabilities(
            worker_type="jaxmarl",
            supported_paradigms=("independent", "parameter_sharing"),
            env_families=("multigrid", "soccer", "american_football", "basketball"),
            action_spaces=("discrete",),
            observation_spaces=("vector",),
            max_agents=6,
            supports_self_play=False,
            supports_population=False,
            supports_checkpointing=True,
            supports_pause_resume=False,
            requires_gpu=False,
            gpu_memory_mb=2048,
            cpu_cores=1,
            estimated_memory_mb=2048,
        )
        return metadata, capabilities

    except ImportError:
        from dataclasses import dataclass

        @dataclass
        class _Meta:
            name: str = "JaxMARL Worker"
            version: str = __version__

        @dataclass
        class _Caps:
            algorithms: tuple = ("ippo", "mappo")
            frameworks: tuple = ("jax", "flax")

        return _Meta(), _Caps()


__all__ = ["__version__", "get_worker_metadata"]
