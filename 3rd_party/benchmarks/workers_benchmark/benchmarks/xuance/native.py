"""XuanCe native benchmark -- runs PPO directly via XuanCe's get_runner().

Baseline: no MOSAIC wrapping, no subprocess isolation.
Calls XuanCe's API in-process, same as a researcher would use it standalone.
"""

import os
import sys
import time

from workers_benchmark.utils import (
    BenchmarkResult, run_subprocess_timed, print_run_header, print_run_result,
)

os.environ["WANDB_MODE"] = "disabled"


def run_native_benchmark(config) -> BenchmarkResult:
    """Run XuanCe PPO directly via subprocess (no MOSAIC worker wrapping)."""
    print_run_header(config.worker_name, "native", config.env_id,
                     config.total_timesteps, config.num_envs, config.seed,
                     getattr(config, "_current_iteration", 1),
                     config.iterations)

    # Run XuanCe directly via its own API, no worker shim
    script = f"""\
import os
os.environ["WANDB_MODE"] = "disabled"
from xuance import get_runner
from argparse import Namespace

args = Namespace(
    method="ppo",
    env="classic_control",
    env_id="{config.env_id}",
    running_steps={config.total_timesteps},
    seed={config.seed},
    parallels={config.num_envs},
    dl_toolbox="torch",
    test_mode=False,
    device="cpu",
    benchmark=False,
)
runner = get_runner(
    algo="ppo",
    env="classic_control",
    env_id="{config.env_id}",
    parser_args=args,
)
runner.run(mode="train")
"""
    cmd = [sys.executable, "-c", script]

    # No MOSAIC env vars -- pure XuanCe
    elapsed, peak_mb, stdout, _ = run_subprocess_timed(cmd)
    sps = config.total_timesteps / elapsed if elapsed > 0 else 0.0

    result = BenchmarkResult(
        worker_name="xuance",
        scenario="native",
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
