#!/usr/bin/env python3
"""
CleanRL Worker Overhead Benchmark

Benchmarks CleanRL PPO with:
- Native: Direct execution
- Worker: Via MOSAIC worker wrapper
- Fastlane: Worker with visual streaming

Environment: CartPole-v1, 100k timesteps, 1 environment
"""

import os
import sys
import subprocess
import tempfile
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


class CleanRLBenchmark(AbstractBenchmark):
    """Benchmark CleanRL PPO implementations."""
    
    def __init__(
        self,
        config: BenchmarkConfig,
        cleanrl_repo: Optional[Path] = None,
    ):
        super().__init__(config, "cleanrl")
        
        # Find CleanRL repo
        if cleanrl_repo is None:
            cleanrl_repo = Path(
                "/home/hamid/Desktop/software/mosaic/3rd_party/workers/cleanrl_worker/cleanrl"
            )
        
        self.cleanrl_repo = cleanrl_repo
        self.ppo_script = cleanrl_repo / "cleanrl" / "ppo.py"
        
        if not self.ppo_script.exists():
            raise FileNotFoundError(f"PPO script not found: {self.ppo_script}")
        
        # Initialize results dict
        self.results = {
            "native": [],
            "worker": [],
            "fastlane": [],
        }
    
    def run_benchmark(self) -> None:
        """Run the complete benchmark."""
        print("=" * 80)
        print(" " * 15 + "CLEANRL WORKER OVERHEAD BENCHMARK")
        print("=" * 80)
        print(f"  Environment: {self.config.env_id}")
        print(f"  Timesteps: {self.config.total_timesteps:,}")
        print(f"  Environments: {self.config.num_envs}")
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
        """Run native CleanRL PPO."""
        env = os.environ.copy()
        env["WANDB_MODE"] = "disabled"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                sys.executable,
                str(self.ppo_script),
                f"--env-id={self.config.env_id}",
                f"--total-timesteps={self.config.total_timesteps}",
                f"--num-envs={self.config.num_envs}",
                f"--seed={seed}",
                "--no-track",
                "--no-capture-video",
            ]
            
            start = time.perf_counter()
            result = subprocess.run(
                cmd,
                cwd=tmpdir,
                capture_output=True,
                text=True,
                env=env,
                timeout=3600,  # 1 hour timeout
            )
            elapsed = time.perf_counter() - start
            
            if result.returncode != 0:
                print(f"    Warning: Non-zero exit code {result.returncode}")
                if result.stderr:
                    print(f"    Stderr: {result.stderr[:200]}")
            
            return elapsed
    
    def _run_worker(self, seed: int) -> float:
        """Run CleanRL via MOSAIC worker wrapper."""
        try:
            from cleanrl_worker.config import CleanRLWorkerConfig
            from cleanrl_worker.runtime import CleanRLWorkerRuntime
        except ImportError:
            raise ImportError("cleanrl_worker not installed. Install via: pip install cleanrl_worker")
        
        config = CleanRLWorkerConfig(
            run_id=f"bench_worker_{seed}",
            algo="ppo",
            env_id=self.config.env_id,
            total_timesteps=self.config.total_timesteps,
            seed=seed,
            extras={"cuda": False, "num_envs": self.config.num_envs},
        )
        runtime = CleanRLWorkerRuntime(
            config=config,
            use_grpc=False,
            grpc_target="",
            dry_run=True,
        )
        args = runtime.build_cleanrl_args()
        module_name = runtime.resolve_entrypoint()[0]
        
        env = os.environ.copy()
        env["WANDB_MODE"] = "disabled"
        
        start = time.perf_counter()
        result = subprocess.run(
            [sys.executable, "-m", module_name] + args + ["--no-track", "--no-capture-video"],
            capture_output=True,
            text=True,
            env=env,
            timeout=3600,
        )
        elapsed = time.perf_counter() - start
        
        if result.returncode != 0:
            print(f"    Warning: Non-zero exit code {result.returncode}")
        
        return elapsed
    
    def _run_fastlane(self, seed: int) -> float:
        """Run CleanRL via worker with fastlane enabled."""
        try:
            from cleanrl_worker.config import CleanRLWorkerConfig
            from cleanrl_worker.runtime import CleanRLWorkerRuntime
        except ImportError:
            raise ImportError("cleanrl_worker not installed")
        
        config = CleanRLWorkerConfig(
            run_id=f"bench_fastlane_{seed}",
            algo="ppo",
            env_id=self.config.env_id,
            total_timesteps=self.config.total_timesteps,
            seed=seed,
            extras={"cuda": False, "num_envs": self.config.num_envs},
        )
        runtime = CleanRLWorkerRuntime(
            config=config,
            use_grpc=False,
            grpc_target="",
            dry_run=True,
        )
        args = runtime.build_cleanrl_args()
        module_name = runtime.resolve_entrypoint()[0]
        
        env = os.environ.copy()
        env["WANDB_MODE"] = "disabled"
        env["GYM_GUI_FASTLANE_ONLY"] = "1"
        env["GYM_GUI_FASTLANE_SLOT"] = "0"
        
        start = time.perf_counter()
        result = subprocess.run(
            [sys.executable, "-m", module_name] + args + ["--no-track", "--no-capture-video"],
            capture_output=True,
            text=True,
            env=env,
            timeout=3600,
        )
        elapsed = time.perf_counter() - start
        
        if result.returncode != 0:
            print(f"    Warning: Non-zero exit code {result.returncode}")
        
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
        total_timesteps=train_config.get("total_timesteps", 100000),
        num_envs=train_config.get("num_envs", 1),
        num_iterations=train_config.get("num_iterations", 3),
        seed=train_config.get("seed", 42),
    )


def main():
    """Main entry point."""
    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "cleanrl.yaml"
    config = load_config(config_path)
    config.output_dir = Path(__file__).parent.parent / "logs"
    
    # Run benchmark
    benchmark = CleanRLBenchmark(config)
    benchmark.run_benchmark()


if __name__ == "__main__":
    main()
