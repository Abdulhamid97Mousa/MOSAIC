"""Run benchmarks for all available workers."""

from workers_benchmark.configs import BENCHMARK_ENVS
from workers_benchmark.benchmarks import AVAILABLE_WORKERS
from workers_benchmark.scripts.run_worker import run_worker_benchmarks


def run_all_benchmarks(env_name: str = "cartpole", seed: int = 42):
    """Benchmark every registered worker."""
    print(f"\n{'#' * 70}")
    print(f"  MOSAIC WORKER OVERHEAD BENCHMARK -- ALL WORKERS")
    print(f"  Environment: {BENCHMARK_ENVS[env_name]['env_id']}")
    print(f"  Workers: {', '.join(AVAILABLE_WORKERS)}")
    print(f"{'#' * 70}\n")

    combined = {}
    for worker in AVAILABLE_WORKERS:
        try:
            combined[worker] = run_worker_benchmarks(worker, env_name, seed)
        except Exception as e:
            print(f"  SKIPPED {worker}: {e}")
            combined[worker] = {}

    # Cross-worker comparison
    _print_cross_worker_summary(combined)
    _save_combined_csv(combined, env_name)


def _print_cross_worker_summary(combined: dict):
    """Print horizontal comparison (like rl-tools Figure 1)."""
    import numpy as np

    print(f"\n{'#' * 70}")
    print(f"  CROSS-WORKER COMPARISON (native only)")
    print(f"{'#' * 70}")
    print(f"  {'Worker':<12} {'Mean (s)':>10} {'Std (s)':>10} "
          f"{'SPS':>10} {'vs fastest':>12}")
    print(f"  {'-' * 56}")

    native_means = {}
    for worker, results in combined.items():
        native_results = results.get("native", [])
        if native_results:
            times = [r.wall_time_seconds for r in native_results]
            sps_vals = [r.steps_per_second for r in native_results]
            native_means[worker] = {
                "mean": float(np.mean(times)),
                "std": float(np.std(times)),
                "sps": float(np.mean(sps_vals)),
            }

    if not native_means:
        print("  No native results collected.")
        return

    fastest = min(v["mean"] for v in native_means.values())

    for worker, stats in sorted(native_means.items(), key=lambda x: x[1]["mean"]):
        multiplier = f"{stats['mean'] / fastest:.1f}x"
        print(f"  {worker:<12} {stats['mean']:>10.2f} {stats['std']:>10.2f} "
              f"{stats['sps']:>10.0f} {multiplier:>12}")

    print(f"{'#' * 70}\n")


def _save_combined_csv(combined: dict, env_name: str):
    """Save all results to a single CSV for cross-worker visualization."""
    import csv
    from pathlib import Path

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"all_workers_{env_name}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "worker", "scenario", "iteration", "seed",
            "env_id", "total_timesteps", "num_envs",
            "wall_time_s", "steps_per_second", "peak_memory_mb",
        ])
        for worker, results in combined.items():
            for scenario, rlist in results.items():
                for r in rlist:
                    writer.writerow([
                        r.worker_name, r.scenario, r.iteration, r.seed,
                        r.env_id, r.total_timesteps, r.num_envs,
                        f"{r.wall_time_seconds:.3f}",
                        f"{r.steps_per_second:.1f}",
                        f"{r.peak_memory_mb:.1f}",
                    ])

    print(f"\n  Combined CSV saved: {csv_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark all MOSAIC workers")
    parser.add_argument("--env", choices=list(BENCHMARK_ENVS.keys()), default="cartpole")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_all_benchmarks(args.env, args.seed)


if __name__ == "__main__":
    main()
