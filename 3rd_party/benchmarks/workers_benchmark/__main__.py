"""CLI entry point: python -m workers_benchmark <command>

Commands:
  worker   -- benchmark a single worker (native/worker/fastlane)
  all      -- benchmark all available workers
  analyze  -- analyze saved results + optional plots
"""

import argparse
import sys

from workers_benchmark.configs import BENCHMARK_ENVS
from workers_benchmark.benchmarks import AVAILABLE_WORKERS


def main():
    parser = argparse.ArgumentParser(
        prog="workers_benchmark",
        description="MOSAIC Worker Overhead Benchmark",
    )
    sub = parser.add_subparsers(dest="command")

    # -- worker subcommand --
    wp = sub.add_parser("worker", help="Benchmark a single worker")
    wp.add_argument("name", choices=AVAILABLE_WORKERS)
    wp.add_argument("--env", choices=list(BENCHMARK_ENVS.keys()), default="cartpole")
    wp.add_argument("--seed", type=int, default=42)

    # -- all subcommand --
    ap = sub.add_parser("all", help="Benchmark all workers")
    ap.add_argument("--env", choices=list(BENCHMARK_ENVS.keys()), default="cartpole")
    ap.add_argument("--seed", type=int, default=42)

    # -- analyze subcommand --
    rp = sub.add_parser("analyze", help="Analyze saved results")
    rp.add_argument("--results-dir", type=str, default=None)
    rp.add_argument("--plot", action="store_true")

    args = parser.parse_args()

    if args.command == "worker":
        from workers_benchmark.scripts.run_worker import run_worker_benchmarks
        run_worker_benchmarks(args.name, args.env, args.seed)

    elif args.command == "all":
        from workers_benchmark.scripts.run_all import run_all_benchmarks
        run_all_benchmarks(args.env, args.seed)

    elif args.command == "analyze":
        from pathlib import Path
        from workers_benchmark.scripts.analyze_results import (
            load_results, analyze, plot_results,
        )
        results_dir = Path(args.results_dir) if args.results_dir else (
            Path(__file__).parent / "results"
        )
        results = load_results(results_dir)
        analyze(results)
        if args.plot:
            plot_results(results, results_dir / "plots")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
