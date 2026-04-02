#!/usr/bin/env python3
"""
Worker benchmark comparison using line graphs.
Shows native vs worker vs fastlane as line plots.
"""

import os
import csv
from pathlib import Path
from typing import Dict, List
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
    
    results.update(stats)
    return results


def create_line_graphs():
    """Create line graph visualizations."""
    
    # Load results
    cleanrl = load_results("cleanrl")
    xuance = load_results("xuance")
    
    if not cleanrl or not xuance:
        print("❌ Missing results. Run benchmarks first:")
        print("   python benchmarks/cleanrl_benchmark.py")
        print("   python benchmarks/xuance_benchmark.py")
        return
    
    # ========== Graph 1: Individual framework line graphs ==========
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, (framework, data) in enumerate([("CleanRL", cleanrl), ("Xuance", xuance)]):
        ax = axes[idx]
        iterations = data["iterations"]
        
        # Plot lines
        ax.plot(iterations, data["native"], 'o-', linewidth=2.5, markersize=8, 
               label='Native', color='#2ecc71', markerfacecolor='#2ecc71', markeredgecolor='#27ae60', markeredgewidth=2)
        ax.plot(iterations, data["worker"], 's-', linewidth=2.5, markersize=8, 
               label='Worker', color='#3498db', markerfacecolor='#3498db', markeredgecolor='#2980b9', markeredgewidth=2)
        ax.plot(iterations, data["fastlane"], '^-', linewidth=2.5, markersize=8, 
               label='FastLane', color='#e74c3c', markerfacecolor='#e74c3c', markeredgecolor='#c0392b', markeredgewidth=2)
        
        # Add value labels
        for i, it in enumerate(iterations):
            ax.text(it, data["native"][i], f'{data["native"][i]:.1f}s', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.text(it, data["worker"][i], f'{data["worker"][i]:.1f}s', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.text(it, data["fastlane"][i], f'{data["fastlane"][i]:.1f}s', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title(f'{framework} - CartPole-v1 (100k timesteps)', fontsize=13, fontweight='bold')
        ax.set_xticks(iterations)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add average lines
        ax.axhline(y=data["native_avg"], color='#2ecc71', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.axhline(y=data["worker_avg"], color='#3498db', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.axhline(y=data["fastlane_avg"], color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.5)
    
    plt.suptitle('Worker Performance Across Iterations (Line View)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output1 = LOGS_DIR / "workers_lines_individual.png"
    plt.savefig(output1, dpi=150, bbox_inches='tight')
    print(f"✓ Individual framework line graphs saved to: {output1}")
    
    # ========== Graph 2: Overlay all frameworks ==========
    fig, ax = plt.subplots(figsize=(14, 7))
    
    cleanrl_iter = cleanrl["iterations"]
    xuance_iter = xuance["iterations"]
    
    # Plot CleanRL
    ax.plot(cleanrl_iter, cleanrl["native"], 'o--', linewidth=2, markersize=7, 
           label='CleanRL Native', color='#2ecc71', alpha=0.8)
    ax.plot(cleanrl_iter, cleanrl["worker"], 's--', linewidth=2, markersize=7, 
           label='CleanRL Worker', color='#3498db', alpha=0.8)
    ax.plot(cleanrl_iter, cleanrl["fastlane"], '^--', linewidth=2, markersize=7, 
           label='CleanRL FastLane', color='#e74c3c', alpha=0.8)
    
    # Shift Xuance iterations slightly for visibility
    xuance_iter_offset = [x + 0.15 for x in xuance_iter]
    ax.plot(xuance_iter_offset, xuance["native"], 'o-', linewidth=2, markersize=7, 
           label='Xuance Native', color='#2ecc71', alpha=0.8)
    ax.plot(xuance_iter_offset, xuance["worker"], 's-', linewidth=2, markersize=7, 
           label='Xuance Worker', color='#3498db', alpha=0.8)
    ax.plot(xuance_iter_offset, xuance["fastlane"], '^-', linewidth=2, markersize=7, 
           label='Xuance FastLane', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('All Workers Comparison - Overlay View', fontsize=13, fontweight='bold')
    ax.set_xticks([1, 2, 3])
    ax.legend(fontsize=10, loc='best', ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output2 = LOGS_DIR / "workers_lines_overlay.png"
    plt.savefig(output2, dpi=150, bbox_inches='tight')
    print(f"✓ Overlay line graph saved to: {output2}")
    
    # ========== Graph 3: Comparison by scenario (Native/Worker/FastLane) ==========
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    scenarios = ["native", "worker", "fastlane"]
    scenario_labels = ["Native", "Worker", "FastLane"]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    for idx, (scenario, label, color) in enumerate(zip(scenarios, scenario_labels, colors)):
        ax = axes[idx]
        
        cleanrl_data = cleanrl[scenario]
        xuance_data = xuance[scenario]
        
        # Plot lines
        ax.plot(cleanrl["iterations"], cleanrl_data, 'o-', linewidth=2.5, markersize=8,
               label='CleanRL', color='#9b59b6', markerfacecolor='#9b59b6', markeredgecolor='#8e44ad', markeredgewidth=2)
        ax.plot(xuance["iterations"], xuance_data, 's-', linewidth=2.5, markersize=8,
               label='Xuance', color='#f39c12', markerfacecolor='#f39c12', markeredgecolor='#d68910', markeredgewidth=2)
        
        # Add value labels
        for i, it in enumerate(cleanrl["iterations"]):
            ax.text(it - 0.05, cleanrl_data[i], f'{cleanrl_data[i]:.1f}s', 
                   ha='right', va='bottom', fontsize=9, fontweight='bold')
        for i, it in enumerate(xuance["iterations"]):
            ax.text(it + 0.05, xuance_data[i], f'{xuance_data[i]:.1f}s', 
                   ha='left', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title(f'{label} Mode Comparison', fontsize=13, fontweight='bold')
        ax.set_xticks(cleanrl["iterations"])
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle('Performance by Execution Mode', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output3 = LOGS_DIR / "workers_lines_by_mode.png"
    plt.savefig(output3, dpi=150, bbox_inches='tight')
    print(f"✓ Mode comparison line graphs saved to: {output3}")
    
    # ========== Graph 4: Simple average comparison ==========
    fig, ax = plt.subplots(figsize=(12, 7))
    
    frameworks = ["CleanRL", "Xuance"]
    native_avgs = [cleanrl["native_avg"], xuance["native_avg"]]
    worker_avgs = [cleanrl["worker_avg"], xuance["worker_avg"]]
    fastlane_avgs = [cleanrl["fastlane_avg"], xuance["fastlane_avg"]]
    
    x = np.arange(len(frameworks))
    offset = 0.3
    
    ax.plot(x, native_avgs, 'o-', linewidth=3, markersize=12, 
           label='Native', color='#2ecc71', markerfacecolor='#2ecc71', markeredgecolor='#27ae60', markeredgewidth=2)
    ax.plot(x, worker_avgs, 's-', linewidth=3, markersize=12, 
           label='Worker', color='#3498db', markerfacecolor='#3498db', markeredgecolor='#2980b9', markeredgewidth=2)
    ax.plot(x, fastlane_avgs, '^-', linewidth=3, markersize=12, 
           label='FastLane', color='#e74c3c', markerfacecolor='#e74c3c', markeredgecolor='#c0392b', markeredgewidth=2)
    
    # Add value labels
    for i, (n, w, f) in enumerate(zip(native_avgs, worker_avgs, fastlane_avgs)):
        ax.text(i - 0.1, n, f'{n:.1f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.text(i, w, f'{w:.1f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.text(i + 0.1, f, f'{f:.1f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Framework', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Execution Time (seconds)', fontsize=13, fontweight='bold')
    ax.set_title('Average Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(frameworks, fontsize=12, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    output4 = LOGS_DIR / "workers_lines_average.png"
    plt.savefig(output4, dpi=150, bbox_inches='tight')
    print(f"✓ Average comparison line graph saved to: {output4}")
    
    print("\n📊 All line graphs created successfully!")


if __name__ == "__main__":
    create_line_graphs()
