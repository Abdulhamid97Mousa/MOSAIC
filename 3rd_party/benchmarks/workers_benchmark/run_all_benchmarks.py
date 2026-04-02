#!/usr/bin/env python3
"""
Master Benchmark Script for MOSAIC Workers
"""

import sys
import os
import time
import subprocess
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any

# Add worker paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
WORKER_PATHS = [
    str(REPO_ROOT / "3rd_party/workers/cleanrl_worker"),
    str(REPO_ROOT / "3rd_party/workers/xuance_worker"),
    str(REPO_ROOT / "3rd_party/workers/ray_worker"),
    str(REPO_ROOT / "3rd_party/workers/jumanji_worker"),
]
sys.path.extend(WORKER_PATHS)

OUTPUT_DIR = Path("logs/benchmark_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_env_with_pythonpath(base_env: Dict[str, str]) -> Dict[str, str]:
    env = base_env.copy()
    current_pp = env.get("PYTHONPATH", "")
    new_pp = os.pathsep.join(WORKER_PATHS + [current_pp]) if current_pp else os.pathsep.join(WORKER_PATHS)
    env["PYTHONPATH"] = new_pp
    return env

def run_command(cmd: List[str], env: Dict[str, str], timeout: int = 3600) -> float:
    """Run command and return execution time."""
    # Ensure PYTHONPATH includes workers
    env = get_env_with_pythonpath(env)
    
    start = time.perf_counter()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - start
    
    if result.returncode != 0:
        print(f"FAILED: {' '.join(cmd)}")
        print(f"STDERR: {result.stderr[-500:]}")
        raise RuntimeError("Command failed")
        
    return elapsed

def benchmark_cleanrl(steps: int = 100000) -> Dict[str, float]:
    print("\n--- Benchmarking CleanRL ---")
    results = {}
    
    # FastLane
    try:
        print("Running FastLane...")
        env = os.environ.copy()
        env["GYM_GUI_FASTLANE_ONLY"] = "1"
        env["GYM_GUI_FASTLANE_SLOT"] = "0"
        
        # Construct command using worker module directly
        cmd = [
            sys.executable, "-m", "cleanrl_worker.launcher",
            "cleanrl_worker.algorithms.ppo",
            "--env-id=CartPole-v1",
            f"--total-timesteps={steps}",
            "--num-envs=1",
            "--no-cuda",
            "--no-track",
            "--no-capture-video",
        ]
        
        t = run_command(cmd, env)
        results["FastLane"] = steps / t
        print(f"FastLane: {t:.2f}s ({steps/t:.0f} SPS)")
    except Exception as e:
        print(f"FastLane failed: {e}")
        results["FastLane"] = 0

    return results

def benchmark_xuance(steps: int = 100000) -> Dict[str, float]:
    print("\n--- Benchmarking XuanCe ---")
    results = {}
    
    try:
        print("Running FastLane (Default for XuanCe)...")
        env = os.environ.copy()
        
        script = f"""
from xuance_worker.config import XuanCeWorkerConfig
from xuance_worker.runtime import XuanCeWorkerRuntime
import sys

config = XuanCeWorkerConfig(
    run_id="bench_xuance",
    method="ppo",
    env="classic_control",
    env_id="CartPole-v1",
    running_steps={steps},
)
runtime = XuanCeWorkerRuntime(config)
runtime.run()
"""
        cmd = [sys.executable, "-c", script]
        t = run_command(cmd, env)
        results["FastLane"] = steps / t
        print(f"FastLane: {t:.2f}s ({steps/t:.0f} SPS)")
    except Exception as e:
        print(f"FastLane failed: {e}")
        results["FastLane"] = 0
        
    return results

def benchmark_ray(steps: int = 100000) -> Dict[str, float]:
    print("\n--- Benchmarking Ray ---")
    results = {}
    
    try:
        print("Running FastLane (Default for Ray)...")
        env = os.environ.copy()
        
        script = f"""
from ray_worker.config import RayWorkerConfig, EnvironmentConfig, TrainingConfig
from ray_worker.runtime import RayWorkerRuntime
import ray

ray.init(local_mode=True, include_dashboard=False, logging_level="ERROR")

config = RayWorkerConfig(
    run_id="bench_ray",
    environment=EnvironmentConfig(
        family="classic",
        env_id="CartPole-v1",
    ),
    training=TrainingConfig(
        algorithm="PPO",
        total_timesteps={steps},
    ),
)
runtime = RayWorkerRuntime(config)
runtime.run()
ray.shutdown()
"""
        cmd = [sys.executable, "-c", script]
        t = run_command(cmd, env)
        results["FastLane"] = steps / t
        print(f"FastLane: {t:.2f}s ({steps/t:.0f} SPS)")
    except Exception as e:
        print(f"FastLane failed: {e}")
        results["FastLane"] = 0
        
    return results

def benchmark_jumanji(steps: int = 100000) -> Dict[str, float]:
    print("\n--- Benchmarking Jumanji ---")
    results = {}
    
    try:
        print("Running FastLane (Default for Jumanji)...")
        env = os.environ.copy()
        
        script_fixed = f"""
from jumanji_worker.config import JumanjiWorkerConfig
from jumanji_worker.runtime import JumanjiWorkerRuntime

# 100 epochs * 1000 batch = 100,000 steps
config = JumanjiWorkerConfig(
    run_id="bench_jumanji",
    env_id="Snake-v1", 
    agent="a2c",
    num_epochs=100, 
    total_batch_size=1000,
)
runtime = JumanjiWorkerRuntime(config)
runtime.run()
"""
        
        cmd = [sys.executable, "-c", script_fixed]
        t = run_command(cmd, env)
        total_steps = 100 * 1000
        results["FastLane"] = total_steps / t
        print(f"FastLane: {t:.2f}s ({total_steps/t:.0f} SPS)")
    except Exception as e:
        print(f"FastLane failed: {e}")
        results["FastLane"] = 0
        
    return results

def main():
    data = []
    
    # CleanRL
    res = benchmark_cleanrl()
    for mode, sps in res.items():
        data.append({"Worker": "CleanRL", "Mode": mode, "SPS": sps})
        
    # XuanCe
    res = benchmark_xuance()
    for mode, sps in res.items():
        data.append({"Worker": "XuanCe", "Mode": mode, "SPS": sps})
        
    # Ray
    res = benchmark_ray()
    for mode, sps in res.items():
        data.append({"Worker": "Ray", "Mode": mode, "SPS": sps})
        
    # Jumanji
    res = benchmark_jumanji()
    for mode, sps in res.items():
        data.append({"Worker": "Jumanji", "Mode": mode, "SPS": sps})
        
    # Create DataFrame
    df = pd.DataFrame(data)
    print("\nResults:")
    print(df)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Worker", y="SPS", hue="Mode")
    plt.title("Worker Execution Speed (Steps Per Second)")
    plt.ylabel("Steps Per Second (SPS)")
    plt.xlabel("Worker Framework")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "worker_benchmark_sps.png")
    print(f"\nPlot saved to {OUTPUT_DIR / 'worker_benchmark_sps.png'}")
    
    # Line graph as requested (SPS per worker)
    plt.figure(figsize=(10, 6))
    # Simple line plot connecting the tops of the bars essentially, or point plot
    sns.pointplot(data=df, x="Worker", y="SPS", join=True)
    plt.title("Worker Execution Speed Comparison")
    plt.ylabel("Steps Per Second (SPS)")
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / "worker_benchmark_line.png")
    print(f"Line graph saved to {OUTPUT_DIR / 'worker_benchmark_line.png'}")

if __name__ == "__main__":
    main()
