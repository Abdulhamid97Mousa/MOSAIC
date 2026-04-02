"""Jumanji worker benchmarks - template implementations."""

from pathlib import Path
import time

from workers_benchmark.configs import get_benchmark_config
from workers_benchmark.utils import BenchmarkTimer, BenchmarkResult, print_benchmark_header


def run_native_benchmark(config):
    """Run native Jumanji benchmark."""
    print_benchmark_header(config)
    print("⚠️  Jumanji native benchmark needs implementation")
    print("TODO: Implement direct Jumanji training without worker wrapper")

    return BenchmarkResult(
        worker_name="jumanji",
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
    """Run Jumanji worker benchmark with tensorboard logging."""
    print_benchmark_header(config)
    print("⚠️  Jumanji worker benchmark needs implementation")
    print("TODO: Implement Jumanji worker training with logging")

    return BenchmarkResult(
        worker_name="jumanji",
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
    """Run Jumanji worker benchmark with fastlane visual streaming."""
    print_benchmark_header(config)
    print("⚠️  Jumanji fastlane benchmark needs implementation")
    print("TODO: Implement Jumanji worker training with fastlane")

    return BenchmarkResult(
        worker_name="jumanji",
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
