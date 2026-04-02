"""Run all benchmark scenarios (native/worker/fastlane) for a single worker.

Repeats each scenario N times (like hyperfine) and collects statistics.
"""

import sys
from pathlib import Path

from workers_benchmark.configs import get_benchmark_config, BENCHMARK_ENVS
from workers_benchmark.benchmarks import AVAILABLE_WORKERS

SCENARIOS = ["native", "worker", "fastlane"]


def _import_worker_module(worker_name: str):
    import importlib

    if worker_name not in AVAILABLE_WORKERS:
        raise ValueError(f"Unknown worker: {worker_name}. Available: {AVAILABLE_WORKERS}")
    return importlib.import_module(f"workers_benchmark.benchmarks.{worker_name}")


def run_worker_benchmarks(worker_name: str, env_name: str = "cartpole", seed: int = 42):
    """Run all scenarios for a worker with multiple iterations."""
    config = get_benchmark_config(worker_name, "native", env_name, seed)
    iterations = config.iterations
    module = _import_worker_module(worker_name)

    print(f"\n{'=' * 70}")
    print(f"  {worker_name.upper()} OVERHEAD BENCHMARK")
    print(f"  {config.env_id}  |  {config.total_timesteps:,} steps  |  "
          f"{iterations} iterations  |  seed={seed}")
    print(f"{'=' * 70}")

    # {scenario: [BenchmarkResult, ...]}
    all_results = {s: [] for s in SCENARIOS}

    for i in range(1, iterations + 1):
        iter_seed = seed + i - 1
        for scenario in SCENARIOS:
            cfg = get_benchmark_config(worker_name, scenario, env_name, iter_seed)
            cfg._current_iteration = i  # noqa: SLF001

            run_fn = {
                "native": module.run_native_benchmark,
                "worker": module.run_worker_benchmark,
                "fastlane": module.run_fastlane_benchmark,
            }[scenario]

            try:
                result = run_fn(cfg)
                all_results[scenario].append(result)
                # Save individual result
                results_dir = Path(__file__).parent.parent / "results"
                result.save(results_dir)
            except Exception as e:
                print(f"    FAILED: {e}")

    _print_summary(worker_name, all_results)
    _save_csv(worker_name, config, all_results)
    return all_results


def _print_summary(worker_name: str, all_results: dict):
    """Print summary with multiplier labels (like rl-tools Figure 1)."""
    import numpy as np

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY: {worker_name.upper()}")
    print(f"{'=' * 70}")
    print(f"  {'Scenario':<12} {'Mean (s)':>10} {'Std (s)':>10} "
          f"{'SPS':>10} {'Memory':>10} {'Overhead':>10}")
    print(f"  {'-' * 62}")

    native_times = [r.wall_time_seconds for r in all_results.get("native", [])]
    native_mean = float(np.mean(native_times)) if native_times else 0.0

    for scenario in SCENARIOS:
        results = all_results.get(scenario, [])
        if not results:
            print(f"  {scenario:<12} {'--':>10}")
            continue

        times = [r.wall_time_seconds for r in results]
        sps_vals = [r.steps_per_second for r in results]
        mem_vals = [r.peak_memory_mb for r in results]

        mean_t = float(np.mean(times))
        std_t = float(np.std(times))
        mean_sps = float(np.mean(sps_vals))
        mean_mem = float(np.mean(mem_vals))

        if scenario == "native":
            overhead = "1.00x"
        elif native_mean > 0:
            overhead = f"{mean_t / native_mean:.2f}x"
        else:
            overhead = "N/A"

        print(f"  {scenario:<12} {mean_t:>10.2f} {std_t:>10.2f} "
              f"{mean_sps:>10.0f} {mean_mem:>8.0f}MB {overhead:>10}")

    print(f"{'=' * 70}\n")


def _save_csv(worker_name: str, config, all_results: dict):
    """Save results as CSV for visualization (pandas/matplotlib/spreadsheet)."""
    import csv
    import numpy as np

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{worker_name}_benchmark.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "worker", "scenario", "iteration", "seed",
            "env_id", "total_timesteps", "num_envs",
            "wall_time_s", "steps_per_second", "peak_memory_mb",
        ])
        for scenario in SCENARIOS:
            for r in all_results.get(scenario, []):
                writer.writerow([
                    r.worker_name, r.scenario, r.iteration, r.seed,
                    r.env_id, r.total_timesteps, r.num_envs,
                    f"{r.wall_time_seconds:.3f}",
                    f"{r.steps_per_second:.1f}",
                    f"{r.peak_memory_mb:.1f}",
                ])

    print(f"  CSV saved: {csv_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark a single MOSAIC worker")
    parser.add_argument("worker", choices=AVAILABLE_WORKERS)
    parser.add_argument("--env", choices=list(BENCHMARK_ENVS.keys()), default="cartpole")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_worker_benchmarks(args.worker, args.env, args.seed)


if __name__ == "__main__":
    main()
