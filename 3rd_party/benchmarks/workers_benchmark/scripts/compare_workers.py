#!/usr/bin/env python3
"""
Compare all worker benchmarks in a unified visualization.
Shows native vs worker vs fastlane across all frameworks.
"""

import os
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

# Get benchmark logs directory
BENCHMARK_ROOT = Path(__file__).parent.parent
LOGS_DIR = BENCHMARK_ROOT / "logs"


def load_results(framework: str) -> Dict:
    """Load CSV results for a framework."""
    csv_file = LOGS_DIR / f"{framework}_results.csv"
    if not csv_file.exists():
        return None
    
    results = {"iterations": [], "native": [], "worker": [], "fastlane": []}
    stats = {}
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        lines = list(reader)
    
    # Parse iterations
    header = lines[0]
    for i, line in enumerate(lines[1:4], 1):
        results["iterations"].append(i)
        results["native"].append(float(line[1]) if line[1] else None)
        results["worker"].append(float(line[2]) if line[2] else None)
        results["fastlane"].append(float(line[3]) if line[3] else None)
    
    # Parse statistics
    for line in lines[5:]:
        if line and len(line) >= 4:
            stat_name = line[0]
            if stat_name == "Average (s)":
                stats["native_avg"] = float(line[1]) if line[1] else None
                stats["worker_avg"] = float(line[2]) if line[2] else None
                stats["fastlane_avg"] = float(line[3]) if line[3] else None
            elif stat_name == "Std Dev (s)":
                stats["native_std"] = float(line[1]) if line[1] else None
                stats["worker_std"] = float(line[2]) if line[2] else None
                stats["fastlane_std"] = float(line[3]) if line[3] else None
    
    results.update(stats)
    return results


def create_comparison_graphs():
    """Create comprehensive comparison visualizations."""
    
    # Load results
    cleanrl = load_results("cleanrl")
    xuance = load_results("xuance")
    
    if not cleanrl or not xuance:
        print("❌ Missing results. Run benchmarks first:")
        print("   python benchmarks/cleanrl_benchmark.py")
        print("   python benchmarks/xuance_benchmark.py")
        return
    
    # Extract averages
    frameworks = ["CleanRL", "Xuance"]
    native_times = [cleanrl["native_avg"], xuance["native_avg"]]
    worker_times = [cleanrl["worker_avg"], xuance["worker_avg"]]
    fastlane_times = [cleanrl["fastlane_avg"], xuance["fastlane_avg"]]
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # ========== 1. Side-by-side bars (all scenarios) ==========
    ax1 = plt.subplot(2, 3, 1)
    x = np.arange(len(frameworks))
    width = 0.25
    
    bars1 = ax1.bar(x - width, native_times, width, label='Native', color='#2ecc71', alpha=0.8)
    bars2 = ax1.bar(x, worker_times, width, label='Worker', color='#3498db', alpha=0.8)
    bars3 = ax1.bar(x + width, fastlane_times, width, label='FastLane', color='#e74c3c', alpha=0.8)
    
    ax1.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax1.set_title('Execution Time: All Scenarios', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(frameworks)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}s', ha='center', va='bottom', fontsize=9)
    
    # ========== 2. Overhead comparison ==========
    ax2 = plt.subplot(2, 3, 2)
    overhead_worker = [
        ((worker_times[0] - native_times[0]) / native_times[0] * 100),
        ((worker_times[1] - native_times[1]) / native_times[1] * 100)
    ]
    overhead_fastlane = [
        ((fastlane_times[0] - native_times[0]) / native_times[0] * 100),
        ((fastlane_times[1] - native_times[1]) / native_times[1] * 100)
    ]
    
    x = np.arange(len(frameworks))
    width = 0.35
    
    colors_worker = ['#e74c3c' if oh > 0 else '#2ecc71' for oh in overhead_worker]
    colors_fastlane = ['#e74c3c' if oh > 0 else '#2ecc71' for oh in overhead_fastlane]
    
    bars1 = ax2.bar(x - width/2, overhead_worker, width/2, label='Worker', color=colors_worker, alpha=0.8)
    bars2 = ax2.bar(x + width/2, overhead_fastlane, width/2, label='FastLane', color=colors_fastlane, alpha=0.8)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_ylabel('Overhead (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Overhead vs Native', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(frameworks)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', 
                    va='bottom' if height > 0 else 'top', fontsize=9)
    
    # ========== 3. Speed ranking (all modes) ==========
    ax3 = plt.subplot(2, 3, 3)
    all_times = [
        cleanrl["native_avg"], cleanrl["worker_avg"], cleanrl["fastlane_avg"],
        xuance["native_avg"], xuance["worker_avg"], xuance["fastlane_avg"]
    ]
    labels = [
        "CleanRL\nNative", "CleanRL\nWorker", "CleanRL\nFastLane",
        "Xuance\nNative", "Xuance\nWorker", "Xuance\nFastLane"
    ]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#2ecc71', '#3498db', '#e74c3c']
    
    sorted_indices = sorted(range(len(all_times)), key=lambda i: all_times[i])
    sorted_times = [all_times[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]
    
    bars = ax3.barh(range(len(sorted_times)), sorted_times, color=sorted_colors, alpha=0.8)
    ax3.set_yticks(range(len(sorted_labels)))
    ax3.set_yticklabels(sorted_labels)
    ax3.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax3.set_title('Speed Ranking (Fastest → Slowest)', fontsize=12, fontweight='bold')
    ax3.invert_yaxis()
    ax3.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, time) in enumerate(zip(bars, sorted_times)):
        ax3.text(time, bar.get_y() + bar.get_height()/2.,
                f' {time:.1f}s', va='center', fontsize=9, fontweight='bold')
    
    # ========== 4. Native performance baseline ==========
    ax4 = plt.subplot(2, 3, 4)
    ax4.bar(frameworks, native_times, color='#2ecc71', alpha=0.8, width=0.5)
    ax4.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax4.set_title('Native Performance (Baseline)', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    for i, (fw, t) in enumerate(zip(frameworks, native_times)):
        ax4.text(i, t, f'{t:.1f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # ========== 5. Worker efficiency (time saved/lost) ==========
    ax5 = plt.subplot(2, 3, 5)
    time_diff_worker = [native_times[i] - worker_times[i] for i in range(len(frameworks))]
    colors_diff = ['#2ecc71' if d > 0 else '#e74c3c' for d in time_diff_worker]
    
    bars = ax5.bar(frameworks, time_diff_worker, color=colors_diff, alpha=0.8, width=0.5)
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax5.set_ylabel('Time Difference (s)', fontsize=11, fontweight='bold')
    ax5.set_title('Worker vs Native (Positive = Faster)', fontsize=12, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    
    for bar, diff in zip(bars, time_diff_worker):
        ax5.text(bar.get_x() + bar.get_width()/2., diff,
                f'{diff:+.2f}s', ha='center', 
                va='bottom' if diff > 0 else 'top', fontsize=10, fontweight='bold')
    
    # ========== 6. Summary table ==========
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Create summary data
    summary_data = [
        ["Framework", "Native (s)", "Worker (s)", "Overhead"],
        ["CleanRL", f"{cleanrl['native_avg']:.2f}", f"{cleanrl['worker_avg']:.2f}", 
         f"{overhead_worker[0]:+.1f}%"],
        ["Xuance", f"{xuance['native_avg']:.2f}", f"{xuance['worker_avg']:.2f}", 
         f"{overhead_worker[1]:+.1f}%"],
        ["", "", "", ""],
        ["Fastest Worker", "Xuance", f"{min(worker_times):.2f}s", 
         f"{((min(worker_times) - max(worker_times)) / max(worker_times) * 100):.1f}% faster"]
    ]
    
    table = ax6.table(cellText=summary_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style summary rows
    for i in range(4):
        table[(1, i)].set_facecolor('#ecf0f1')
        table[(2, i)].set_facecolor('#ecf0f1')
    
    # Highlight fastest
    for i in range(4):
        table[(4, i)].set_facecolor('#2ecc71')
        table[(4, i)].set_text_props(weight='bold')
    
    plt.suptitle('MOSAIC Worker Benchmark Comparison\nCartPole-v1 (100k timesteps)', 
                fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    output_path = LOGS_DIR / "workers_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Comparison graph saved to: {output_path}")
    
    # Also create a simpler version for quick reference
    fig2, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(frameworks))
    width = 0.25
    
    bars1 = ax.bar(x - width, native_times, width, label='Native', color='#2ecc71', alpha=0.85)
    bars2 = ax.bar(x, worker_times, width, label='Worker', color='#3498db', alpha=0.85)
    bars3 = ax.bar(x + width, fastlane_times, width, label='FastLane', color='#e74c3c', alpha=0.85)
    
    ax.set_ylabel('Time (seconds)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Framework', fontsize=13, fontweight='bold')
    ax.set_title('Worker Performance Comparison\nCartPole-v1, 100k timesteps, 3 iterations', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(frameworks, fontsize=12, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add overhead percentages as text
    ax.text(0 - width, native_times[0] + 5, 'baseline', ha='center', fontsize=9, style='italic')
    ax.text(0, native_times[0] + 5, f'+{overhead_worker[0]:.1f}%', ha='center', fontsize=9, style='italic')
    ax.text(0 + width, native_times[0] + 5, f'{overhead_fastlane[0]:+.1f}%', ha='center', fontsize=9, style='italic')
    
    ax.text(1 - width, native_times[1] + 2, 'baseline', ha='center', fontsize=9, style='italic')
    ax.text(1, native_times[1] + 2, f'{overhead_worker[1]:+.1f}%', ha='center', fontsize=9, style='italic')
    ax.text(1 + width, native_times[1] + 2, f'{overhead_fastlane[1]:+.1f}%', ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    simple_output = LOGS_DIR / "workers_comparison_simple.png"
    plt.savefig(simple_output, dpi=150, bbox_inches='tight')
    print(f"✓ Simple comparison graph saved to: {simple_output}")
    
    plt.show()


if __name__ == "__main__":
    create_comparison_graphs()
