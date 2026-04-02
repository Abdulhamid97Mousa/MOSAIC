#!/usr/bin/env python3
"""Quick example demonstrating the benchmark system."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from workers_benchmark.configs import get_benchmark_config


def main():
    """Run a quick example benchmark."""
    print("\n" + "="*70)
    print("  WORKERS BENCHMARK - QUICK EXAMPLE")
    print("="*70 + "\n")

    print("This example demonstrates the benchmark system.")
    print("It will run a quick test of CleanRL worker in all three scenarios:\n")
    print("  1. Native: Direct framework execution (no wrapper)")
    print("  2. Worker: Framework wrapped with tensorboard logging")
    print("  3. Fastlane: Framework with visual streaming enabled\n")

    print("Environment: MiniGrid-Empty-8x8-v0")
    print("Total timesteps: 10,000 (quick test)")
    print("Num envs: 2\n")

    response = input("Press Enter to start benchmark (or Ctrl+C to cancel)...")

    try:
        from workers_benchmark.benchmarks import cleanrl

        # Run all three scenarios
        scenarios = ["native", "worker", "fastlane"]
        results = []

        for scenario in scenarios:
            print(f"\n{'='*70}")
            print(f"  Running: {scenario.upper()}")
            print(f"{'='*70}\n")

            config = get_benchmark_config("cleanrl", scenario, "quick_test")

            if scenario == "native":
                result = cleanrl.run_native_benchmark(config)
            elif scenario == "worker":
                result = cleanrl.run_worker_benchmark(config)
            elif scenario == "fastlane":
                result = cleanrl.run_fastlane_benchmark(config)

            results.append(result)

            # Save result
            results_dir = Path(__file__).parent.parent / "results"
            result.save(results_dir)

        # Print comparison
        print(f"\n{'='*70}")
        print("  COMPARISON")
        print(f"{'='*70}\n")

        native_time = results[0].wall_time_seconds

        for result in results:
            overhead = ""
            if result.scenario != "native" and native_time > 0:
                overhead_pct = ((result.wall_time_seconds - native_time) / native_time) * 100
                overhead = f" (+{overhead_pct:.1f}% overhead)"

            print(f"{result.scenario:12s}: {result.wall_time_seconds:8.2f}s | "
                  f"{result.steps_per_second:8.2f} steps/s{overhead}")

        print(f"\n{'='*70}\n")
        print("✓ Example complete! Results saved to results/ directory")
        print("\nTo analyze all results, run:")
        print("  python -m workers_benchmark --analyze\n")

    except KeyboardInterrupt:
        print("\n\nBenchmark cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error running benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
