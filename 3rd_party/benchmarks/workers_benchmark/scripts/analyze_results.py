"""Analyze benchmark results from JSON files and produce summary tables.

Reads results saved by run_worker.py and computes:
- Per-worker overhead tables with multiplier labels (like rl-tools paper)
- Cross-worker native comparison
- Optional matplotlib plots
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

DISPLAY = {
    'cleanrl': 'CleanRL',
    'xuance': 'XuanCe',
    'ray': 'RLlib',
    'tianshou': 'Tianshou',
    'sb3': 'SB3',
    'sbx': 'SBX',
    'torchrl': 'TorchRL',
    'rltools': 'RLtools',
    'jumanji': 'Jumanji (JAX)',
}


def load_results(results_dir: Path) -> List[Dict[str, Any]]:
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return []
    results = []
    for f in sorted(results_dir.glob("*.json")):
        try:
            with open(f) as fp:
                results.append(json.load(fp))
        except Exception as e:
            print(f"  skip {f.name}: {e}")
    return results


def analyze(results: List[Dict[str, Any]]):
    if not results:
        print("No results to analyze.")
        return

    # Group: worker -> scenario -> [result, ...]
    grouped: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        grouped[r["worker_name"]][r["scenario"]].append(r)

    # --- Per-worker overhead table ---
    print(f"\n{'=' * 80}")
    print("  OVERHEAD ANALYSIS (per worker)")
    print(f"{'=' * 80}")
    print(f"  {'Worker':<12} {'Scenario':<16} {'Mean (s)':>10} +/- {'Std':>7} "
          f"{'SPS':>10} {'Memory':>8} {'Overhead':>10}")
    print(f"  {'-' * 78}")

    all_scenarios = ["native", "worker", "fastlane", "fastlane_gymnax", "fastlane_gym"]

    for worker in sorted(grouped):
        scenarios = grouped[worker]
        native_times = [r["wall_time_seconds"] for r in scenarios.get("native", [])]
        native_mean = float(np.mean(native_times)) if native_times else 0.0

        for scenario in all_scenarios:
            rlist = scenarios.get(scenario, [])
            if not rlist:
                continue
            times = [r["wall_time_seconds"] for r in rlist]
            sps = [r["steps_per_second"] for r in rlist]
            mem = [r["peak_memory_mb"] for r in rlist]
            mean_t = float(np.mean(times))
            std_t = float(np.std(times))
            mean_sps = float(np.mean(sps))
            mean_mem = float(np.mean(mem))

            if scenario == "native":
                overhead = "1.00x"
            elif native_mean > 0:
                overhead = f"{mean_t / native_mean:.2f}x"
            else:
                overhead = "N/A"

            print(f"  {worker:<12} {scenario:<16} {mean_t:>10.2f}     {std_t:>7.2f} "
                  f"{mean_sps:>10.0f} {mean_mem:>6.0f}MB {overhead:>10}")
        print()

    # --- Cross-worker native comparison ---
    print(f"{'=' * 80}")
    print("  NATIVE COMPARISON (horizontal, like rl-tools Figure 1)")
    print(f"{'=' * 80}")
    print(f"  {'Worker':<12} {'Mean (s)':>10} +/- {'Std':>7} {'SPS':>10} {'vs fastest':>12}")
    print(f"  {'-' * 55}")

    native_stats = {}
    for worker in sorted(grouped):
        rlist = grouped[worker].get("native", [])
        if rlist:
            times = [r["wall_time_seconds"] for r in rlist]
            sps = [r["steps_per_second"] for r in rlist]
            native_stats[worker] = {
                "mean": float(np.mean(times)),
                "std": float(np.std(times)),
                "sps": float(np.mean(sps)),
            }

    if native_stats:
        fastest = min(v["mean"] for v in native_stats.values())
        for worker, s in sorted(native_stats.items(), key=lambda x: x[1]["mean"]):
            mult = f"{s['mean'] / fastest:.1f}x"
            print(f"  {worker:<12} {s['mean']:>10.2f}     {s['std']:>7.2f} "
                  f"{s['sps']:>10.0f} {mult:>12}")

    print(f"\n{'=' * 80}\n")


def plot_results(results: List[Dict[str, Any]], output_dir: Path):
    """Generate clean bar charts (no error bars, no boxes)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return

    grouped: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        grouped[r["worker_name"]][r["scenario"]].append(r)

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: Native comparison bar chart ---
    native_stats = {}
    for worker in sorted(grouped):
        rlist = grouped[worker].get("native", [])
        if rlist:
            times = [r["wall_time_seconds"] for r in rlist]
            native_stats[worker] = {"mean": np.mean(times), "std": np.std(times)}

    if native_stats:
        # Sort by mean time ascending
        sorted_workers = sorted(native_stats.keys(), key=lambda w: native_stats[w]["mean"])
        fastest_mean = native_stats[sorted_workers[0]]["mean"]

        # Color gradient: green (fastest) to dark blue (slowest)
        n = len(sorted_workers)
        cmap = plt.cm.get_cmap('YlGnBu', n + 2)
        colors = [cmap(2 + i) for i in range(n)]

        fig, ax = plt.subplots(figsize=(max(8, n * 1.6), 5))
        display_names = [DISPLAY.get(w, w) for w in sorted_workers]
        means = [native_stats[w]["mean"] for w in sorted_workers]

        bars = ax.bar(display_names, means, color=colors, edgecolor='#333', linewidth=0.6)

        for bar, w in zip(bars, sorted_workers):
            mult = native_stats[w]["mean"] / fastest_mean
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{mult:.1f}x", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_ylabel("Training time [s] (smaller is better)", fontsize=11, fontweight='bold')
        ax.set_title("Native Framework Comparison: PPO on CartPole-v1 (100K steps)",
                     fontsize=13, fontweight='bold')
        ax.grid(axis="y", alpha=0.25, linestyle='--')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis='x', labelsize=10.5)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        plt.tight_layout()

        path = output_dir / "native_comparison.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")

    # --- Figure 2: Per-worker overhead charts ---
    # Colors matching plot_publication.py
    C_NATIVE = '#9E9E9E'
    C_WORKER = '#4A90D9'
    C_FASTLANE = '#1B5E9E'
    C_FASTLANE_GYMNAX = '#D4763A'
    C_FASTLANE_GYM = '#1B5E9E'

    all_scenarios = ["native", "worker", "fastlane", "fastlane_gymnax", "fastlane_gym"]
    scenario_colors = {
        "native": C_NATIVE,
        "worker": C_WORKER,
        "fastlane": C_FASTLANE,
        "fastlane_gymnax": C_FASTLANE_GYMNAX,
        "fastlane_gym": C_FASTLANE_GYM,
    }
    scenario_labels = {
        "native": "Native",
        "worker": "MOSAIC Worker",
        "fastlane": "FastLane",
        "fastlane_gymnax": "FastLane (gymnax)",
        "fastlane_gym": "FastLane (Gymnasium)",
    }
    scenario_hatch = {
        "native": '',
        "worker": '//',
        "fastlane": '',
        "fastlane_gymnax": '\\\\',
        "fastlane_gym": '',
    }

    for worker in sorted(grouped):
        scenarios = grouped[worker]
        data = {}
        for s in all_scenarios:
            rlist = scenarios.get(s, [])
            if rlist:
                times = [r["wall_time_seconds"] for r in rlist]
                data[s] = {"mean": np.mean(times), "std": np.std(times)}

        if len(data) < 2:
            continue

        labels = list(data.keys())
        display = [scenario_labels.get(s, s) for s in labels]
        means = [data[s]["mean"] for s in labels]
        bar_colors = [scenario_colors.get(s, '#999') for s in labels]
        hatches = [scenario_hatch.get(s, '') for s in labels]

        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
        bars = ax.bar(display, means, color=bar_colors, edgecolor='#333', linewidth=0.6)
        for bar, h in zip(bars, hatches):
            bar.set_hatch(h)

        native_mean = data.get("native", {}).get("mean", 0)
        for bar, s in zip(bars, labels):
            if native_mean > 0:
                mult = data[s]["mean"] / native_mean
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{mult:.2f}x", ha="center", va="bottom", fontsize=9, fontweight="bold")

        worker_display = DISPLAY.get(worker, worker.upper())
        ax.set_ylabel("Training time [s] (smaller is better)", fontsize=11, fontweight='bold')
        ax.set_title(f"{worker_display} Wrapper Overhead", fontsize=13, fontweight='bold')
        ax.grid(axis="y", alpha=0.25, linestyle='--')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis='x', labelsize=10)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        plt.tight_layout()

        path = output_dir / f"{worker}_overhead.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--results-dir", type=Path,
                        default=Path(__file__).parent.parent / "results")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    args = parser.parse_args()

    results = load_results(args.results_dir)
    analyze(results)

    if args.plot:
        plot_results(results, args.results_dir / "plots")


if __name__ == "__main__":
    main()
