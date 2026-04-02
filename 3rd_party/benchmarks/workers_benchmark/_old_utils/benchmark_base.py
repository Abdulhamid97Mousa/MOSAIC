"""
Base benchmark class with common functionality.

Provides:
- CSV export with statistics
- Metadata collection
- Publication-ready plotting
- Reproducible runs
"""

import os
import sys
import time
import json
import csv
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import platform
from dataclasses import dataclass, asdict

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: numpy not installed")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    env_id: str
    total_timesteps: int
    num_envs: int
    num_iterations: int
    seed: int = 42
    output_dir: Optional[Path] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class AbstractBenchmark(ABC):
    """Abstract base class for all benchmarks."""
    
    def __init__(self, config: BenchmarkConfig, worker_name: str):
        """
        Initialize benchmark.
        
        Args:
            config: BenchmarkConfig instance
            worker_name: Name of the worker being benchmarked
        """
        self.config = config
        self.worker_name = worker_name
        self.output_dir = config.output_dir or Path.cwd() / "benchmark_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results: Dict[str, List[Optional[float]]] = {}
        self.metadata = self._collect_metadata()
        
    def _collect_metadata(self) -> Dict:
        """Collect system and execution metadata."""
        config_dict = self.config.to_dict()
        # Convert Path objects to strings for JSON serialization
        if isinstance(config_dict.get("output_dir"), Path):
            config_dict["output_dir"] = str(config_dict["output_dir"])
        
        return {
            "timestamp": datetime.now().isoformat(),
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "worker": self.worker_name,
            **config_dict,
        }
    
    @abstractmethod
    def run_benchmark(self) -> None:
        """Run the benchmark. Subclasses must implement."""
        pass
    
    def save_results(self, scenarios: List[str]) -> None:
        """
        Save results to CSV and JSON.
        
        Args:
            scenarios: List of scenario names (e.g., ["native", "worker", "fastlane"])
        """
        # Save as CSV
        csv_file = self.output_dir / f"{self.worker_name}_results.csv"
        
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            
            # Header
            header = ["Iteration"] + scenarios
            writer.writerow(header)
            
            # Data rows
            max_iterations = max(len(self.results.get(s, [])) for s in scenarios)
            for i in range(max_iterations):
                row = [i + 1]
                for scenario in scenarios:
                    times = self.results.get(scenario, [])
                    if i < len(times) and times[i] is not None:
                        row.append(f"{times[i]:.2f}")
                    else:
                        row.append("")
                writer.writerow(row)
            
            # Statistics section
            writer.writerow([])
            writer.writerow(["Statistic"] + scenarios)
            
            # Calculate stats for each scenario
            for metric_name, metric_func in [
                ("Average (s)", lambda t: np.mean([x for x in t if x is not None])),
                ("Std Dev (s)", lambda t: np.std([x for x in t if x is not None])),
            ]:
                row = [metric_name]
                for scenario in scenarios:
                    times = [x for x in self.results.get(scenario, []) if x is not None]
                    if times:
                        row.append(f"{metric_func(times):.2f}")
                    else:
                        row.append("")
                writer.writerow(row)
            
            # Overhead percentages
            writer.writerow([])
            native_times = [x for x in self.results.get("native", []) if x is not None]
            if native_times:
                avg_native = np.mean(native_times)
                row = ["Overhead (%)"]
                for scenario in scenarios:
                    if scenario == "native":
                        row.append("baseline")
                    else:
                        times = [x for x in self.results.get(scenario, []) if x is not None]
                        if times:
                            overhead = (np.mean(times) - avg_native) / avg_native * 100
                            row.append(f"{overhead:.1f}%")
                        else:
                            row.append("")
                writer.writerow(row)
        
        print(f"✓ Results saved to: {csv_file}")
        
        # Save metadata as JSON
        json_file = self.output_dir / f"{self.worker_name}_metadata.json"
        with open(json_file, "w") as f:
            json.dump(self.metadata, f, indent=2)
        print(f"✓ Metadata saved to: {json_file}")
    
    def print_summary(self, scenarios: List[str]) -> None:
        """
        Print summary statistics.
        
        Args:
            scenarios: List of scenario names
        """
        print("\n" + "=" * 80)
        print(f" " * (30 + len(self.worker_name) // 2) + self.worker_name.upper())
        print("=" * 80)
        
        native_times = [x for x in self.results.get("native", []) if x is not None]
        
        if not native_times:
            print("No valid native results!")
            return
        
        avg_native = np.mean(native_times)
        std_native = np.std(native_times)
        
        print(f"\n  Native")
        print(f"    Average: {avg_native:8.2f}s")
        print(f"    Std Dev: {std_native:8.2f}s")
        
        for scenario in scenarios:
            if scenario == "native":
                continue
            
            times = [x for x in self.results.get(scenario, []) if x is not None]
            if times:
                avg = np.mean(times)
                std = np.std(times)
                overhead = (avg - avg_native) / avg_native * 100
                print(f"\n  {scenario.capitalize()}")
                print(f"    Average: {avg:8.2f}s")
                print(f"    Std Dev: {std:8.2f}s")
                print(f"    Overhead: {overhead:7.1f}%")
        
        print("\n" + "=" * 80)
    
    def plot_results(self, scenarios: List[str]) -> None:
        """
        Generate publication-ready plots.
        
        Args:
            scenarios: List of scenario names to plot
        """
        if not HAS_MATPLOTLIB:
            print("Skipping plots: matplotlib not installed")
            return
        
        # Collect valid times
        scenario_data = {}
        for scenario in scenarios:
            times = [x for x in self.results.get(scenario, []) if x is not None]
            if times:
                scenario_data[scenario] = times
        
        if not scenario_data:
            print("No valid results to plot")
            return
        
        # Set publication-ready style
        plt.style.use("seaborn-v0_8-darkgrid")
        
        # Figure 1: Bar plot with error bars
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scenario_names = list(scenario_data.keys())
        means = [np.mean(scenario_data[s]) for s in scenario_names]
        stds = [np.std(scenario_data[s]) for s in scenario_names]
        
        colors = ["#2ecc71", "#e74c3c", "#3498db"]  # green, red, blue
        
        x_pos = np.arange(len(scenario_names))
        bars = ax.bar(
            x_pos,
            means,
            yerr=stds,
            capsize=5,
            color=colors[:len(scenario_names)],
            alpha=0.8,
            edgecolor="black",
            linewidth=1.5,
        )
        
        ax.set_ylabel("Time (seconds)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"{self.worker_name.upper()} Overhead Benchmark\n"
            f"{self.config.env_id}, {self.config.total_timesteps:,} timesteps",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels([s.capitalize() for s in scenario_names], fontsize=11)
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
        plot_file = self.output_dir / f"{self.worker_name}_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        print(f"✓ Plot saved to: {plot_file}")
        plt.close()
        
        # Figure 2: Overhead percentage (if not native)
        if "native" in scenario_data:
            overhead_scenarios = [s for s in scenario_names if s != "native"]
            if overhead_scenarios:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                baseline = np.mean(scenario_data["native"])
                overhead_values = [
                    (np.mean(scenario_data[s]) - baseline) / baseline * 100
                    for s in overhead_scenarios
                ]
                overhead_colors = ["#e74c3c", "#3498db"]  # red, blue
                
                x_pos = np.arange(len(overhead_scenarios))
                bars = ax.bar(
                    x_pos,
                    overhead_values,
                    color=overhead_colors[:len(overhead_scenarios)],
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=1.5,
                )
                
                ax.axhline(y=0, color="black", linestyle="-", linewidth=1)
                ax.set_ylabel("Overhead (%)", fontsize=12, fontweight="bold")
                ax.set_title(
                    f"{self.worker_name.upper()} Overhead (vs. Native)\n"
                    f"{self.config.env_id}, {self.config.total_timesteps:,} timesteps",
                    fontsize=14,
                    fontweight="bold",
                )
                ax.set_xticks(x_pos)
                ax.set_xticklabels([s.capitalize() for s in overhead_scenarios], fontsize=11)
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
                plot_file = self.output_dir / f"{self.worker_name}_overhead.png"
                plt.savefig(plot_file, dpi=300, bbox_inches="tight")
                print(f"✓ Overhead plot saved to: {plot_file}")
                plt.close()
