"""RLtools worker benchmark -- SAC through rl_tools_worker.launcher."""

import json
import sys
import tempfile
import time
from pathlib import Path

from workers_benchmark.utils import (
    BenchmarkResult, run_subprocess_timed, print_run_header, print_run_result,
)


def run_worker_benchmark(config) -> BenchmarkResult:
    """Run RLtools SAC via the worker launcher subprocess."""
    print_run_header(config.worker_name, "worker", config.env_id,
                     config.total_timesteps, config.num_envs, config.seed,
                     getattr(config, "_current_iteration", 1),
                     config.iterations)

    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        log_dir = work_dir / "logs"
        log_dir.mkdir()

        config_dict = {
            "run_id": f"bench_rltools_worker_{config.seed}",
            "algo": "sac",
            "env_id": config.env_id,
            "total_timesteps": config.total_timesteps,
            "seed": config.seed,
            "extras": {},
        }
        config_path = work_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f)

        cmd = [
            sys.executable, "-m", "rl_tools_worker.launcher",
            "--run-id", config_dict["run_id"],
            "--algo", "sac",
            "--env-id", config.env_id,
            "--total-timesteps", str(config.total_timesteps),
            "--seed", str(config.seed),
            "--config-file", str(config_path),
            "--work-dir", str(work_dir),
            "--log-dir", str(log_dir),
        ]

        env_overrides = {"GYM_GUI_FASTLANE_ONLY": "0"}
        elapsed, peak_mb, stdout, _ = run_subprocess_timed(
            cmd, env=env_overrides, timeout=1800,
        )

    sps = config.total_timesteps / elapsed if elapsed > 0 else 0.0

    result = BenchmarkResult(
        worker_name="rltools",
        scenario="worker",
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
