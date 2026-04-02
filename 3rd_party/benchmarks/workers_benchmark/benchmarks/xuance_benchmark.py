#!/usr/bin/env python3
"""
Xuance Worker Overhead Benchmark

Benchmarks Xuance PPO with:
- Native: Direct execution via XuanCe runner
- Worker: Via MOSAIC worker wrapper
- Fastlane: Worker with visual streaming

Environment: CartPole-v1, 100k running_steps, 1 parallel environment
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.benchmark_base import AbstractBenchmark, BenchmarkConfig

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class XuanceBenchmark(AbstractBenchmark):
    """Benchmark Xuance PPO implementations."""
    
    def __init__(self, config: BenchmarkConfig):
        super().__init__(config, "xuance")
        
        # Initialize results dict
        self.results = {
            "native": [],
            "worker": [],
            "fastlane": [],
        }
    
    def run_benchmark(self) -> None:
        """Run the complete benchmark."""
        print("=" * 80)
        print(" " * 15 + "XUANCE WORKER OVERHEAD BENCHMARK")
        print("=" * 80)
        print(f"  Environment: {self.config.env_id}")
        print(f"  Running Steps: {self.config.total_timesteps:,}")
        print(f"  Parallel Environments: {self.config.num_envs}")
        print(f"  Iterations: {self.config.num_iterations}")
        print("=" * 80)
        
        for i in range(self.config.num_iterations):
            seed = self.config.seed + i
            print(f"\n--- Iteration {i+1}/{self.config.num_iterations} (seed={seed}) ---")
            
            # Native
            try:
                native_time = self._run_native(seed)
                self.results["native"].append(native_time)
                print(f"  Native:   {native_time:8.2f}s")
            except Exception as e:
                print(f"  Native:   ERROR - {e}")
                self.results["native"].append(None)
            
            # Worker
            try:
                worker_time = self._run_worker(seed)
                self.results["worker"].append(worker_time)
                print(f"  Worker:   {worker_time:8.2f}s")
            except Exception as e:
                print(f"  Worker:   ERROR - {e}")
                self.results["worker"].append(None)
            
            # Fastlane
            try:
                fastlane_time = self._run_fastlane(seed)
                self.results["fastlane"].append(fastlane_time)
                print(f"  Fastlane: {fastlane_time:8.2f}s")
            except Exception as e:
                print(f"  Fastlane: ERROR - {e}")
                self.results["fastlane"].append(None)
        
        # Save and display results
        self.save_results(["native", "worker", "fastlane"])
        self.print_summary(["native", "worker", "fastlane"])
        self.plot_results(["native", "worker", "fastlane"])
    
    def _run_native(self, seed: int) -> float:
        """Run native Xuance PPO."""
        env = os.environ.copy()
        env["WANDB_MODE"] = "disabled"
        
        # Xuance expects config to be loaded via XuanCe's config system
        # For simplicity, we'll use the worker runtime but mark it as native mode
        try:
            from xuance_worker.config import XuanCeWorkerConfig
            from xuance_worker.runtime import XuanCeWorkerRuntime
        except ImportError:
            raise ImportError("xuance_worker not installed")
        
        config = XuanCeWorkerConfig(
            run_id=f"bench_native_{seed}",
            method="ppo",
            env="classic_control",
            env_id=self.config.env_id,
            running_steps=self.config.total_timesteps,
            seed=seed,
            device="cpu",
            parallels=self.config.num_envs,
        )
        
        runtime = XuanCeWorkerRuntime(config=config, dry_run=False)
        
        start = time.perf_counter()
        try:
            result = runtime.run()
            elapsed = time.perf_counter() - start
        except Exception as e:
            elapsed = time.perf_counter() - start
            raise e
        
        return elapsed
    
    def _run_worker(self, seed: int) -> float:
        """Run Xuance via MOSAIC worker wrapper."""
        try:
            from xuance_worker.config import XuanCeWorkerConfig
            from xuance_worker.runtime import XuanCeWorkerRuntime
        except ImportError:
            raise ImportError("xuance_worker not installed")
        
        config = XuanCeWorkerConfig(
            run_id=f"bench_worker_{seed}",
            method="ppo",
            env="classic_control",
            env_id=self.config.env_id,
            running_steps=self.config.total_timesteps,
            seed=seed,
            device="cpu",
            parallels=self.config.num_envs,
        )
        
        runtime = XuanCeWorkerRuntime(config=config, dry_run=False)
        
        start = time.perf_counter()
        try:
            result = runtime.run()
            elapsed = time.perf_counter() - start
        except Exception as e:
            elapsed = time.perf_counter() - start
            raise e
        
        return elapsed
    
    def _run_fastlane(self, seed: int) -> float:
        """Run Xuance via worker with fastlane enabled."""
        try:
            from xuance_worker.config import XuanCeWorkerConfig
            from xuance_worker.runtime import XuanCeWorkerRuntime
        except ImportError:
            raise ImportError("xuance_worker not installed")
        
        config = XuanCeWorkerConfig(
            run_id=f"bench_fastlane_{seed}",
            method="ppo",
            env="classic_control",
            env_id=self.config.env_id,
            running_steps=self.config.total_timesteps,
            seed=seed,
            device="cpu",
            parallels=self.config.num_envs,
        )
        
        runtime = XuanCeWorkerRuntime(config=config, dry_run=False)
        
        # Enable FastLane
        env = os.environ.copy()
        env["GYM_GUI_FASTLANE_ONLY"] = "1"
        env["GYM_GUI_FASTLANE_SLOT"] = "0"
        
        # Set environment
        for key, value in env.items():
            os.environ[key] = value
        
        start = time.perf_counter()
        try:
            result = runtime.run()
            elapsed = time.perf_counter() - start
        except Exception as e:
            elapsed = time.perf_counter() - start
            raise e
        
        return elapsed


def load_config(config_path: Path) -> BenchmarkConfig:
    """Load config from YAML file."""
    if not HAS_YAML:
        raise ImportError("PyYAML not installed. Install via: pip install pyyaml")
    
    with open(config_path) as f:
        data = yaml.safe_load(f)
    
    # Extract from nested structure
    env_config = data.get("environment", {})
    train_config = data.get("training", {})
    
    return BenchmarkConfig(
        env_id=env_config.get("env_id", "CartPole-v1"),
        total_timesteps=train_config.get("running_steps", 100000),
        num_envs=train_config.get("parallels", 1),
        num_iterations=train_config.get("num_iterations", 3),
        seed=train_config.get("seed", 42),
    )


def main():
    """Main entry point."""
    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "xuance.yaml"
    config = load_config(config_path)
    config.output_dir = Path(__file__).parent.parent / "logs"
    
    # Run benchmark
    benchmark = XuanceBenchmark(config)
    benchmark.run_benchmark()


if __name__ == "__main__":
    main()
