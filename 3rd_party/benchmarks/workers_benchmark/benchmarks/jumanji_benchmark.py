#!/usr/bin/env python3
"""
Jumanji Worker Overhead Benchmark

Benchmarks Jumanji A2C with:
- Native: Direct JAX A2C execution
- Worker: Via MOSAIC worker wrapper
- Fastlane: Worker with visual streaming

Environment: Game2048-v1, 100k steps, 1 parallel environment
"""

import os
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.benchmark_base import AbstractBenchmark, BenchmarkConfig

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class JumanjiBenchmark(AbstractBenchmark):
    """Benchmark Jumanji A2C implementations."""
    
    def __init__(self, config: BenchmarkConfig):
        super().__init__(config, "jumanji")
        
        # Initialize results dict
        self.results = {
            "native": [],
            "worker": [],
            "fastlane": [],
        }
    
    def run_benchmark(self) -> None:
        """Run the complete benchmark."""
        print("=" * 80)
        print(" " * 16 + "JUMANJI WORKER OVERHEAD BENCHMARK")
        print("=" * 80)
        print(f"  Environment: {self.config.env_id}")
        print(f"  Training Steps: {self.config.total_timesteps:,}")
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
                print(f"  Native:   ERROR - {str(e)[:80]}")
                self.results["native"].append(None)
            
            # Worker
            try:
                worker_time = self._run_worker(seed)
                self.results["worker"].append(worker_time)
                print(f"  Worker:   {worker_time:8.2f}s")
            except Exception as e:
                print(f"  Worker:   ERROR - {str(e)[:80]}")
                self.results["worker"].append(None)
            
            # Fastlane
            try:
                fastlane_time = self._run_fastlane(seed)
                self.results["fastlane"].append(fastlane_time)
                print(f"  Fastlane: {fastlane_time:8.2f}s")
            except Exception as e:
                print(f"  Fastlane: ERROR - {str(e)[:80]}")
                self.results["fastlane"].append(None)
        
        # Save and display results
        self.save_results(["native", "worker", "fastlane"])
        self.print_summary(["native", "worker", "fastlane"])
        self.plot_results(["native", "worker", "fastlane"])
    
    def _run_native(self, seed: int) -> float:
        """Run native Jumanji A2C."""
        try:
            from jumanji_worker.config import JumanjiWorkerConfig
            from jumanji_worker.runtime import JumanjiWorkerRuntime
        except ImportError:
            raise ImportError("jumanji_worker not installed. Install with: pip install -e 3rd_party/workers/jumanji_worker")
        
        config = JumanjiWorkerConfig(
            run_id=f"bench_native_{seed}",
            agent="a2c",  # A2C is the only algorithm available
            env_id=self.config.env_id,
            seed=seed,
        )
        
        runtime = JumanjiWorkerRuntime(config=config)
        
        start = time.perf_counter()
        try:
            runtime.run()
            elapsed = time.perf_counter() - start
        except Exception as e:
            elapsed = time.perf_counter() - start
            raise e
        
        return elapsed
    
    def _run_worker(self, seed: int) -> float:
        """Run Jumanji via MOSAIC worker wrapper."""
        try:
            from jumanji_worker.config import JumanjiWorkerConfig
            from jumanji_worker.runtime import JumanjiWorkerRuntime
        except ImportError:
            raise ImportError("jumanji_worker not installed")
        
        config = JumanjiWorkerConfig(
            run_id=f"bench_worker_{seed}",
            agent="a2c",
            env_id=self.config.env_id,
            seed=seed,
        )
        
        runtime = JumanjiWorkerRuntime(config=config)
        
        start = time.perf_counter()
        try:
            runtime.run()
            elapsed = time.perf_counter() - start
        except Exception as e:
            elapsed = time.perf_counter() - start
            raise e
        
        return elapsed
    
    def _run_fastlane(self, seed: int) -> float:
        """Run Jumanji via worker with fastlane enabled."""
        try:
            from jumanji_worker.config import JumanjiWorkerConfig
            from jumanji_worker.runtime import JumanjiWorkerRuntime
        except ImportError:
            raise ImportError("jumanji_worker not installed")
        
        config = JumanjiWorkerConfig(
            run_id=f"bench_fastlane_{seed}",
            agent="a2c",
            env_id=self.config.env_id,
            seed=seed,
        )
        
        runtime = JumanjiWorkerRuntime(config=config)
        
        # Enable FastLane via environment variable
        os.environ["GYM_GUI_FASTLANE_ONLY"] = "1"
        os.environ["GYM_GUI_FASTLANE_SLOT"] = "0"
        
        start = time.perf_counter()
        try:
            runtime.run()
            elapsed = time.perf_counter() - start
        except Exception as e:
            elapsed = time.perf_counter() - start
            raise e
        finally:
            # Clean up env vars
            os.environ.pop("GYM_GUI_FASTLANE_ONLY", None)
            os.environ.pop("GYM_GUI_FASTLANE_SLOT", None)
        
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
        env_id=env_config.get("env_id", "Game2048-v1"),
        total_timesteps=train_config.get("num_epochs", 100),  # Jumanji uses epochs not timesteps
        num_envs=1,
        num_iterations=train_config.get("num_iterations", 3),
        seed=train_config.get("seed", 42),
    )


def main():
    """Main entry point."""
    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "jumanji.yaml"
    config = load_config(config_path)
    config.output_dir = Path(__file__).parent.parent / "logs"
    
    # Run benchmark
    benchmark = JumanjiBenchmark(config)
    benchmark.run_benchmark()


if __name__ == "__main__":
    main()
