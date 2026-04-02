"""XuanCe fastlane benchmark -- worker + SHM frame streaming."""

import os
import sys
import time

from workers_benchmark.utils import (
    BenchmarkResult, run_subprocess_timed, print_run_header, print_run_result,
)


def run_fastlane_benchmark(config) -> BenchmarkResult:
    """Run XuanCe PPO via worker with FastLane enabled."""
    print_run_header(config.worker_name, "fastlane", config.env_id,
                     config.total_timesteps, config.num_envs, config.seed,
                     getattr(config, "_current_iteration", 1),
                     config.iterations)

    script = f"""\
import os
os.environ["WANDB_MODE"] = "disabled"
from xuance_worker.config import XuanCeWorkerConfig
from xuance_worker.runtime import XuanCeWorkerRuntime

config = XuanCeWorkerConfig(
    run_id="bench_xuance_fl_{config.seed}",
    method="ppo",
    env="classic_control",
    env_id="{config.env_id}",
    running_steps={config.total_timesteps},
    seed={config.seed},
    parallels={config.num_envs},
    device="cpu",
)
runtime = XuanCeWorkerRuntime(config)
runtime.run()
"""
    cmd = [sys.executable, "-c", script]
    env_overrides = {
        "GYM_GUI_FASTLANE_ONLY": "1",
        "GYM_GUI_FASTLANE_SLOT": "0",
        "GYM_GUI_FASTLANE_VIDEO_MODE": "single",
        "XUANCE_RUN_ID": f"bench_xuance_fl_{config.seed}",
    }

    elapsed, peak_mb, stdout, _ = run_subprocess_timed(cmd, env=env_overrides)
    sps = config.total_timesteps / elapsed if elapsed > 0 else 0.0

    result = BenchmarkResult(
        worker_name="xuance",
        scenario="fastlane",
        env_id=config.env_id,
        total_timesteps=config.total_timesteps,
        wall_time_seconds=elapsed,
        steps_per_second=sps,
        peak_memory_mb=peak_mb,
        seed=config.seed,
        num_envs=config.num_envs,
        iteration=getattr(config, "_current_iteration", 1),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    print_run_result(result)
    return result
