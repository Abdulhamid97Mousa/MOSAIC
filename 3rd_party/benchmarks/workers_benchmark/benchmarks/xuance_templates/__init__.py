"""XuanCe worker benchmarks - template implementations."""

# Note: These are template implementations that need to be adapted
# based on XuanCe's specific API and worker implementation.

from pathlib import Path
import time
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.benchmark_base import BenchmarkResult


def run_native_benchmark(config):
    """Run native XuanCe PPO benchmark."""
    print("⚠️  XuanCe native benchmark needs implementation")
    print("TODO: Implement direct XuanCe PPO training without worker wrapper")

    # Placeholder result
    return BenchmarkResult(
        worker_name="xuance",
        scenario="native",
        env_id=config.env_id,
        total_timesteps=config.total_timesteps,
        wall_time_seconds=0.0,
        steps_per_second=0.0,
        peak_memory_mb=0.0,
        seed=config.seed,
        num_envs=config.num_envs,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


def run_worker_benchmark(config):
    """Run XuanCe worker benchmark with tensorboard logging."""
    print("⚠️  XuanCe worker benchmark needs implementation")
    print("TODO: Implement XuanCe worker training with logging")

    # Placeholder result
    return BenchmarkResult(
        worker_name="xuance",
        scenario="worker",
        env_id=config.env_id,
        total_timesteps=config.total_timesteps,
        wall_time_seconds=0.0,
        steps_per_second=0.0,
        peak_memory_mb=0.0,
        seed=config.seed,
        num_envs=config.num_envs,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


def run_fastlane_benchmark(config):
    """Run XuanCe worker benchmark with fastlane visual streaming."""
    print("⚠️  XuanCe fastlane benchmark needs implementation")
    print("TODO: Implement XuanCe worker training with fastlane")

    # Placeholder result
    return BenchmarkResult(
        worker_name="xuance",
        scenario="fastlane",
        env_id=config.env_id,
        total_timesteps=config.total_timesteps,
        wall_time_seconds=0.0,
        steps_per_second=0.0,
        peak_memory_mb=0.0,
        seed=config.seed,
        num_envs=config.num_envs,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


__all__ = ["run_native_benchmark", "run_worker_benchmark", "run_fastlane_benchmark"]
