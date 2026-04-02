#!/usr/bin/env python3
"""
MOSAIC Worker Overhead Benchmark

Compares performance of:
1. Native: Direct CleanRL execution
2. Worker: CleanRL via MOSAIC worker wrapper
3. Fastlane: Worker with visual streaming enabled

Saves results as CSV and generates publication-ready plots.
"""

import os
import subprocess
import sys
import tempfile
import time
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import platform

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Plots will not be generated.")


class BenchmarkRunner:
    def __init__(
        self,
        cleanrl_repo: Path,
        env_id: str = "CartPole-v1",
        total_timesteps: int = 100000,
        num_envs: int = 1,
        iterations: int = 3,
        output_dir: Path = None,
    ):
        self.cleanrl_repo = cleanrl_repo
        self.ppo_script = cleanrl_repo / "cleanrl" / "ppo.py"
        self.env_id = env_id
        self.total_timesteps = total_timesteps
        self.num_envs = num_envs
        self.iterations = iterations
        self.output_dir = output_dir or Path.cwd() / "benchmark_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: Dict[str, List[float]] = {
            "native": [],
            "worker": [],
            "fastlane": [],
        }
        self.metadata = self._collect_metadata()

    def _collect_metadata(self) -> Dict:
        """Collect system and execution metadata."""
        return {
            "timestamp": datetime.now().isoformat(),
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "env_id": self.env_id,
            "total_timesteps": self.total_timesteps,
            "num_envs": self.num_envs,
            "iterations": self.iterations,
            "ppo_script": str(self.ppo_script),
        }

    def _run_native(self, seed: int) -> float:
        """Run native CleanRL PPO."""
        env = os.environ.copy()
        env["WANDB_MODE"] = "disabled"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                sys.executable,
                str(self.ppo_script),
                f"--env-id={self.env_id}",
                f"--total-timesteps={self.total_timesteps}",
                f"--num-envs={self.num_envs}",
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
            )
            elapsed = time.perf_counter() - start
            
            if result.returncode != 0:
                print(f"Warning: Native run failed with return code {result.returncode}")
            
            return elapsed

    def _run_worker(self, seed: int) -> float:
        """Run CleanRL via MOSAIC worker wrapper."""
        from cleanrl_worker.config import CleanRLWorkerConfig
        from cleanrl_worker.runtime import CleanRLWorkerRuntime
        
        config = CleanRLWorkerConfig(
            run_id=f"bench_worker_{seed}",
            algo="ppo",
            env_id=self.env_id,
            total_timesteps=self.total_timesteps,
            seed=seed,
            extras={"cuda": False, "num_envs": self.num_envs},
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
        )
        elapsed = time.perf_counter() - start
        
        if result.returncode != 0:
            print(f"Warning: Worker run failed with return code {result.returncode}")
        
        return elapsed

    def _run_fastlane(self, seed: int) -> float:
        """Run CleanRL via worker with fastlane (visual streaming) enabled."""
        from cleanrl_worker.config import CleanRLWorkerConfig
        from cleanrl_worker.runtime import CleanRLWorkerRuntime
        
        config = CleanRLWorkerConfig(
            run_id=f"bench_fastlane_{seed}",
            algo="ppo",
            env_id=self.env_id,
            total_timesteps=self.total_timesteps,
            seed=seed,
            extras={"cuda": False, "num_envs": self.num_envs},
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
        )
        elapsed = time.perf_counter() - start
        
        if result.returncode != 0:
            print(f"Warning: Fastlane run failed with return code {result.returncode}")
        
        return elapsed

    def run(self):
        """Run the complete benchmark."""
        print("=" * 80)
        print(" " * 20 + "MOSAIC WORKER OVERHEAD BENCHMARK")
        print("=" * 80)
        print(f"  Environment: {self.env_id}")
        print(f"  Timesteps: {self.total_timesteps:,}")
        print(f"  Environments: {self.num_envs}")
        print(f"  Iterations: {self.iterations}")
        print("=" * 80)
        
        for i in range(self.iterations):
            seed = 42 + i
            print(f"\n--- Iteration {i+1}/{self.iterations} (seed={seed}) ---")
            
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
        
        self._save_results()
        self._print_summary()

    def _save_results(self):
        """Save results to CSV and JSON."""
        # Save as CSV
        csv_file = self.output_dir / "benchmark_results.csv"
        
        # Filter out None values for statistics
        native_times = [t for t in self.results["native"] if t is not None]
        worker_times = [t for t in self.results["worker"] if t is not None]
        fastlane_times = [t for t in self.results["fastlane"] if t is not None]
        
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Iteration", "Native (s)", "Worker (s)", "Fastlane (s)"])
            
            for i in range(self.iterations):
                writer.writerow([
                    i + 1,
                    self.results["native"][i] if self.results["native"][i] is not None else "",
                    self.results["worker"][i] if self.results["worker"][i] is not None else "",
                    self.results["fastlane"][i] if self.results["fastlane"][i] is not None else "",
                ])
            
            # Add statistics
            writer.writerow([])
            writer.writerow(["Metric", "Native", "Worker", "Fastlane"])
            
            if native_times:
                writer.writerow([
                    "Average (s)",
                    f"{np.mean(native_times):.2f}",
                    f"{np.mean(worker_times):.2f}" if worker_times else "",
                    f"{np.mean(fastlane_times):.2f}" if fastlane_times else "",
                ])
                writer.writerow([
                    "Std Dev (s)",
                    f"{np.std(native_times):.2f}",
                    f"{np.std(worker_times):.2f}" if worker_times else "",
                    f"{np.std(fastlane_times):.2f}" if fastlane_times else "",
                ])
                
                # Overhead percentages
                avg_native = np.mean(native_times)
                writer.writerow([
                    "Worker Overhead (%)",
                    "baseline",
                    f"{((np.mean(worker_times) - avg_native) / avg_native * 100):.1f}%" if worker_times else "",
                    f"{((np.mean(fastlane_times) - avg_native) / avg_native * 100):.1f}%" if fastlane_times else "",
                ])
        
        print(f"✓ Results saved to: {csv_file}")
        
        # Save metadata as JSON
        json_file = self.output_dir / "benchmark_metadata.json"
        with open(json_file, "w") as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"✓ Metadata saved to: {json_file}")

    def _print_summary(self):
        """Print summary statistics."""
        native_times = [t for t in self.results["native"] if t is not None]
        worker_times = [t for t in self.results["worker"] if t is not None]
        fastlane_times = [t for t in self.results["fastlane"] if t is not None]
        
        if not native_times:
            print("\nNo valid native results!")
            return
        
        avg_native = np.mean(native_times)
        std_native = np.std(native_times)
        
        print("\n" + "=" * 80)
        print(" " * 30 + "SUMMARY")
        print("=" * 80)
        print(f"\n  Native")
        print(f"    Average: {avg_native:8.2f}s")
        print(f"    Std Dev: {std_native:8.2f}s")
        
        if worker_times:
            avg_worker = np.mean(worker_times)
            std_worker = np.std(worker_times)
            overhead = (avg_worker - avg_native) / avg_native * 100
            print(f"\n  Worker")
            print(f"    Average: {avg_worker:8.2f}s")
            print(f"    Std Dev: {std_worker:8.2f}s")
            print(f"    Overhead: {overhead:7.1f}%")
        
        if fastlane_times:
            avg_fastlane = np.mean(fastlane_times)
            std_fastlane = np.std(fastlane_times)
            overhead = (avg_fastlane - avg_native) / avg_native * 100
            print(f"\n  Fastlane")
            print(f"    Average: {avg_fastlane:8.2f}s")
            print(f"    Std Dev: {std_fastlane:8.2f}s")
            print(f"    Overhead: {overhead:7.1f}%")
        
        print("\n" + "=" * 80)

    def plot_results(self):
        """Generate publication-ready plots."""
        if not HAS_MATPLOTLIB:
            print("Skipping plots: matplotlib not installed")
            return
        
        native_times = [t for t in self.results["native"] if t is not None]
        worker_times = [t for t in self.results["worker"] if t is not None]
        fastlane_times = [t for t in self.results["fastlane"] if t is not None]
        
        if not native_times:
            print("No valid results to plot")
            return
        
        # Set publication-ready style
        plt.style.use("seaborn-v0_8-darkgrid")
        
        # Figure 1: Bar plot with error bars
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scenarios = ["Native", "Worker", "Fastlane"]
        means = [np.mean(native_times)]
        stds = [np.std(native_times)]
        
        if worker_times:
            means.append(np.mean(worker_times))
            stds.append(np.std(worker_times))
        else:
            scenarios.pop(1)
        
        if fastlane_times:
            means.append(np.mean(fastlane_times))
            stds.append(np.std(fastlane_times))
        else:
            if len(scenarios) > 1:
                scenarios.pop(-1)
        
        x_pos = np.arange(len(scenarios))
        colors = ["#2ecc71", "#e74c3c", "#3498db"]
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors[:len(scenarios)],
                      alpha=0.8, edgecolor="black", linewidth=1.5)
        
        ax.set_ylabel("Time (seconds)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"MOSAIC Worker Overhead Benchmark\n{self.env_id}, {self.total_timesteps:,} timesteps",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(scenarios, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        
        # Add value labels on bars
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{mean:.2f}s",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        
        plt.tight_layout()
        plot_file = self.output_dir / "benchmark_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        print(f"✓ Plot saved to: {plot_file}")
        
        # Figure 2: Overhead percentage
        fig, ax = plt.subplots(figsize=(10, 6))
        
        baseline = np.mean(native_times)
        overhead_scenarios = []
        overhead_values = []
        overhead_colors = []
        
        if worker_times:
            overhead_scenarios.append("Worker")
            overhead_values.append((np.mean(worker_times) - baseline) / baseline * 100)
            overhead_colors.append("#e74c3c")
        
        if fastlane_times:
            overhead_scenarios.append("Fastlane")
            overhead_values.append((np.mean(fastlane_times) - baseline) / baseline * 100)
            overhead_colors.append("#3498db")
        
        if overhead_scenarios:
            x_pos = np.arange(len(overhead_scenarios))
            bars = ax.bar(x_pos, overhead_values, color=overhead_colors, alpha=0.8,
                         edgecolor="black", linewidth=1.5)
            
            ax.axhline(y=0, color="black", linestyle="-", linewidth=1)
            ax.set_ylabel("Overhead (%)", fontsize=12, fontweight="bold")
            ax.set_title(
                f"MOSAIC Worker Overhead (vs. Native)\n{self.env_id}, {self.total_timesteps:,} timesteps",
                fontsize=14,
                fontweight="bold",
            )
            ax.set_xticks(x_pos)
            ax.set_xticklabels(overhead_scenarios, fontsize=11)
            ax.grid(axis="y", alpha=0.3)
            
            # Add value labels
            for bar, val in zip(bars, overhead_values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{val:.1f}%",
                    ha="center",
                    va="bottom" if height >= 0 else "top",
                    fontsize=10,
                    fontweight="bold",
                )
            
            plt.tight_layout()
            plot_file = self.output_dir / "benchmark_overhead.png"
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            print(f"✓ Overhead plot saved to: {plot_file}")


def main():
    # Configuration
    cleanrl_repo = Path("/home/hamid/Desktop/software/mosaic/3rd_party/workers/cleanrl_worker/cleanrl")
    output_dir = Path("/home/hamid/Desktop/software/mosaic/3rd_party/benchmarks/workers_benchmark/logs")
    
    # Run benchmark
    runner = BenchmarkRunner(
        cleanrl_repo=cleanrl_repo,
        env_id="CartPole-v1",
        total_timesteps=100000,
        num_envs=1,
        iterations=3,
        output_dir=output_dir,
    )
    
    runner.run()
    runner.plot_results()


if __name__ == "__main__":
    main()
