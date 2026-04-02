"""Generate publication-quality benchmark charts.

Produces a combined bar chart in the style of the rl-tools paper (Eschmann et al.),
showing native vs MOSAIC Worker vs MOSAIC Worker+FastLane for all workers.

Jumanji has 4 bars: native (gymnax), worker (gymnax), FastLane gymnax,
FastLane gymnasium -- with a distinct color for the gymnasium FastLane bar.

Usage:
    python -m workers_benchmark.scripts.plot_publication [--results-dir PATH] [--env cartpole]
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# Display names for workers
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

# Publication colors (colorblind-safe, print-friendly)
C_NATIVE   = '#9E9E9E'    # neutral gray
C_WORKER   = '#4A90D9'    # steel blue (hatched)
C_FASTLANE = '#1B5E9E'    # deep blue
C_FASTLANE_GYM = '#D4763A'  # warm orange (gymnasium FastLane, distinct)


def load_results(results_dir, env_filter=None):
    """Load and group results, optionally filtering by env_id."""
    results = []
    for f in sorted(results_dir.glob('*.json')):
        try:
            with open(f) as fp:
                r = json.load(fp)
                if r.get('wall_time_seconds', 0) > 0:
                    if env_filter and r.get('env_id') != env_filter:
                        continue
                    results.append(r)
        except Exception:
            pass

    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        grouped[r['worker_name']][r['scenario']].append(r)
    return grouped


def compute_stats(grouped):
    """Compute mean/std for each worker/scenario."""
    stats = {}
    for w in grouped:
        s = {}
        for scenario in ['native', 'worker', 'fastlane', 'fastlane_gymnax', 'fastlane_gym']:
            rlist = grouped[w].get(scenario, [])
            if rlist:
                times = [r['wall_time_seconds'] for r in rlist]
                s[scenario] = {'mean': np.mean(times), 'std': np.std(times), 'n': len(rlist)}
            else:
                s[scenario] = {'mean': 0, 'std': 0, 'n': 0}
        if s['native']['mean'] > 0:
            stats[w] = s
    return stats


def _has_extra_bars(stats, worker):
    """Check if a worker has 4 bars (gymnax + gymnasium FastLane)."""
    s = stats[worker]
    return s['fastlane_gymnax']['mean'] > 0 and s['fastlane_gym']['mean'] > 0


def plot_combined(stats, output_path, env_id='CartPole-v1', total_timesteps=100000):
    """Combined bar chart: native / worker / fastlane for all workers.

    Jumanji gets 4 bars (gymnax FL + gymnasium FL) with a wider group.
    """
    sorted_workers = sorted(stats.keys(), key=lambda w: stats[w]['native']['mean'])

    # Compute x positions: wider group for Jumanji (4 bars)
    bw = 0.20
    x_positions = []
    pos = 0.0
    for w in sorted_workers:
        x_positions.append(pos)
        if _has_extra_bars(stats, w):
            pos += 1.4  # wider gap for 4-bar group
        else:
            pos += 1.0
    x = np.array(x_positions)

    fig, ax = plt.subplots(figsize=(max(11, len(sorted_workers) * 1.8), 5.5))

    # Draw bars per worker
    for i, w in enumerate(sorted_workers):
        nm = stats[w]['native']['mean']
        wm = stats[w]['worker']['mean']

        if _has_extra_bars(stats, w):
            # 4 bars: native, worker, fastlane_gymnax, fastlane_gym
            fgx_m = stats[w]['fastlane_gymnax']['mean']
            fgy_m = stats[w]['fastlane_gym']['mean']

            offsets = [-1.5*bw, -0.5*bw, 0.5*bw, 1.5*bw]
            ax.bar(x[i] + offsets[0], nm, bw, color=C_NATIVE,
                   edgecolor='#333', linewidth=0.6,
                   label='Native' if i == 0 else '')
            ax.bar(x[i] + offsets[1], wm, bw, color=C_WORKER,
                   edgecolor='#333', linewidth=0.6, hatch='//',
                   label='MOSAIC Worker' if i == 0 else '')
            ax.bar(x[i] + offsets[2], fgx_m, bw, color=C_FASTLANE_GYM,
                   edgecolor='#333', linewidth=0.6, hatch='\\\\',
                   label='FastLane (gymnax)' if i == 0 else '')
            ax.bar(x[i] + offsets[3], fgy_m, bw, color=C_FASTLANE,
                   edgecolor='#333', linewidth=0.6,
                   label='FastLane (Gymnasium)' if i == 0 else '')
        else:
            # Standard 3 bars
            fm = stats[w]['fastlane']['mean']
            offsets = [-bw, 0, bw]

            b_n = ax.bar(x[i] + offsets[0], nm, bw, color=C_NATIVE,
                         edgecolor='#333', linewidth=0.6,
                         label='Native' if i == 0 else '')
            b_w = ax.bar(x[i] + offsets[1], wm, bw, color=C_WORKER,
                         edgecolor='#333', linewidth=0.6, hatch='//',
                         label='MOSAIC Worker' if i == 0 else '')
            b_f = ax.bar(x[i] + offsets[2], fm, bw, color=C_FASTLANE,
                         edgecolor='#333', linewidth=0.6,
                         label='MOSAIC Worker + FastLane' if i == 0 else '')

    ax.set_ylabel('Training time [s] (smaller is better)', fontsize=11, fontweight='bold')
    steps_k = total_timesteps // 1000
    ax.set_title(f'MOSAIC Worker Overhead: PPO on {env_id} ({steps_k}K steps)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY.get(w, w) for w in sorted_workers],
                       fontsize=10.5, fontweight='bold')

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen and l:
            seen.add(l)
            unique_handles.append(h)
            unique_labels.append(l)
    ax.legend(unique_handles, unique_labels, fontsize=9, loc='upper left',
              framealpha=0.9, edgecolor='#ccc', fancybox=False)

    ax.grid(axis='y', alpha=0.25, linestyle='--')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {output_path}')


def plot_overhead_horizontal(stats, output_path):
    """Horizontal bar chart showing overhead ratios.

    Workers with gymnax FastLane data get 3 bars (worker, FL gymnax, FL gym).
    Others get 2 bars (worker, fastlane).
    """
    sorted_workers = sorted(stats.keys(), key=lambda w: stats[w]['worker']['mean'] / stats[w]['native']['mean']
                            if stats[w]['worker']['mean'] > 0 else 999)

    labels = []
    ow_vals = []    # worker overhead
    ofgx_vals = []  # fastlane gymnax overhead (or standard fastlane)
    ofgy_vals = []  # fastlane gymnasium overhead (None if not available)
    for w in sorted_workers:
        nm = stats[w]['native']['mean']
        wm = stats[w]['worker']['mean']
        if wm <= 0 or nm <= 0:
            continue
        labels.append(DISPLAY.get(w, w))
        ow_vals.append(wm / nm)
        # Check for gymnax/gymnasium split
        fgx = stats[w].get('fastlane_gymnax', {}).get('mean', 0)
        fgy = stats[w].get('fastlane_gym', {}).get('mean', 0)
        if fgx > 0:
            ofgx_vals.append(fgx / nm)
            ofgy_vals.append(fgy / nm if fgy > 0 else None)
        else:
            fm = stats[w]['fastlane']['mean']
            ofgx_vals.append(fm / nm)
            ofgy_vals.append(None)

    y = np.arange(len(labels))
    # Use 3 bars for workers that have gymnasium FastLane, else 2
    has_gym_fl = any(v is not None for v in ofgy_vals)
    n_bars = 3 if has_gym_fl else 2
    h = 0.25 if has_gym_fl else 0.35

    fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.8)))

    if has_gym_fl:
        ax.barh(y + h, ow_vals, h, label='MOSAIC Worker', color=C_WORKER,
                edgecolor='#333', linewidth=0.5, hatch='//')
        ax.barh(y, ofgx_vals, h, label='FastLane (gymnax)', color=C_FASTLANE_GYM,
                edgecolor='#333', linewidth=0.5, hatch='\\\\')
        # Gymnasium FL bars (only where available)
        gym_vals = [v if v is not None else 0 for v in ofgy_vals]
        ax.barh(y - h, gym_vals, h, label='FastLane (Gymnasium)', color=C_FASTLANE,
                edgecolor='#333', linewidth=0.5)

        for i in range(len(labels)):
            ax.text(ow_vals[i] + 0.02, y[i] + h, f'{ow_vals[i]:.2f}x',
                    va='center', fontsize=8.5, fontweight='bold')
            ax.text(ofgx_vals[i] + 0.02, y[i], f'{ofgx_vals[i]:.2f}x',
                    va='center', fontsize=8.5, fontweight='bold')
            if ofgy_vals[i] is not None:
                ax.text(ofgy_vals[i] + 0.02, y[i] - h, f'{ofgy_vals[i]:.2f}x',
                        va='center', fontsize=8.5, fontweight='bold')
    else:
        ax.barh(y + h/2, ow_vals, h, label='MOSAIC Worker', color=C_WORKER,
                edgecolor='#333', linewidth=0.5, hatch='//')
        ax.barh(y - h/2, ofgx_vals, h, label='MOSAIC Worker + FastLane', color=C_FASTLANE,
                edgecolor='#333', linewidth=0.5)
        for i in range(len(labels)):
            ax.text(ow_vals[i] + 0.02, y[i] + h/2, f'{ow_vals[i]:.2f}x',
                    va='center', fontsize=9, fontweight='bold')
            ax.text(ofgx_vals[i] + 0.02, y[i] - h/2, f'{ofgx_vals[i]:.2f}x',
                    va='center', fontsize=9, fontweight='bold')

    ax.axvline(x=1.0, color='#CC3333', linestyle='--', linewidth=1.5, alpha=0.6,
               label='No overhead (1.0x)')

    ax.set_xlabel('Overhead ratio (1.0x = no overhead)', fontsize=11, fontweight='bold')
    ax.set_title('MOSAIC Wrapper Overhead by Framework', fontsize=13, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10.5, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='x', alpha=0.25, linestyle='--')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    all_vals = ow_vals + ofgx_vals + [v for v in ofgy_vals if v is not None]
    ax.set_xlim(0, max(all_vals) * 1.15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {output_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path,
                        default=Path(__file__).parent.parent / 'results')
    parser.add_argument('--env', default='CartPole-v1')
    parser.add_argument('--timesteps', type=int, default=100000)
    args = parser.parse_args()

    plots_dir = args.results_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    grouped = load_results(args.results_dir, env_filter=args.env)
    stats = compute_stats(grouped)

    if not stats:
        print(f'No results found for env={args.env}')
        return

    print(f'Found {len(stats)} workers with {args.env} results')

    plot_combined(stats, plots_dir / 'combined_overhead.png',
                  env_id=args.env, total_timesteps=args.timesteps)
    plot_overhead_horizontal(stats, plots_dir / 'overhead_ratios.png')


if __name__ == '__main__':
    main()
