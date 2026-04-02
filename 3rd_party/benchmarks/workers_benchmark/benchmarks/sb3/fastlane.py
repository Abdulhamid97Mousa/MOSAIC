"""sb3 fastlane benchmark -- worker launcher + SHM frame streaming."""

import json
import sys
import tempfile
import time
from pathlib import Path

from workers_benchmark.utils import (
    BenchmarkResult, run_subprocess_timed, print_run_header, print_run_result,
)


def run_fastlane_benchmark(config) -> BenchmarkResult:
    """Run sb3 PPO via worker launcher with FastLane enabled."""
    print_run_header(config.worker_name, "fastlane", config.env_id,
                     config.total_timesteps, config.num_envs, config.seed,
                     getattr(config, "_current_iteration", 1),
                     config.iterations)

    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        log_dir = work_dir / "logs"
        log_dir.mkdir()

        config_dict = {
            "run_id": f"bench_sb3_fl_{config.seed}",
            "algo": "ppo",
            "env_id": config.env_id,
            "total_timesteps": config.total_timesteps,
            "seed": config.seed,
            "extras": {
                "num_envs": config.num_envs,
                "device": "cpu",
                "lr": config.learning_rate,
            },
        }
        config_path = work_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f)

        cmd = [
            sys.executable, "-m", "sb3_worker.launcher",
            "--run-id", config_dict["run_id"],
            "--algo", "ppo",
            "--env-id", config.env_id,
            "--total-timesteps", str(config.total_timesteps),
            "--seed", str(config.seed),
            "--config-file", str(config_path),
            "--work-dir", str(work_dir),
            "--log-dir", str(log_dir),
        ]

        env_overrides = {
            "GYM_GUI_FASTLANE_ONLY": "1",
            "GYM_GUI_FASTLANE_SLOT": "0",
            "GYM_GUI_FASTLANE_VIDEO_MODE": "single",
        }

        elapsed, peak_mb, stdout, _ = run_subprocess_timed(
            cmd, env=env_overrides, timeout=1800,
        )

    sps = config.total_timesteps / elapsed if elapsed > 0 else 0.0

    result = BenchmarkResult(
        worker_name="sb3",
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
