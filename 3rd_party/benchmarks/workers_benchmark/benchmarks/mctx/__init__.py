"""MCTX worker benchmarks - template implementations."""

from pathlib import Path
import time

from workers_benchmark.configs import get_benchmark_config
from workers_benchmark.utils import BenchmarkTimer, BenchmarkResult, print_benchmark_header


def run_native_benchmark(config):
    """Run native MCTX benchmark."""
    print_benchmark_header(config)
    print("⚠️  MCTX native benchmark needs implementation")
    print("TODO: Implement direct MCTX training without worker wrapper")

    return BenchmarkResult(
        worker_name="mctx",
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
    """Run MCTX worker benchmark with tensorboard logging."""
    print_benchmark_header(config)
    print("⚠️  MCTX worker benchmark needs implementation")
    print("TODO: Implement MCTX worker training with logging")

    return BenchmarkResult(
        worker_name="mctx",
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
    """Run MCTX worker benchmark with fastlane visual streaming."""
    print_benchmark_header(config)
    print("⚠️  MCTX fastlane benchmark needs implementation")
    print("TODO: Implement MCTX worker training with fastlane")

    return BenchmarkResult(
        worker_name="mctx",
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
