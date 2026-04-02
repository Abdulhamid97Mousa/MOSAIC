"""RLtools native benchmark -- SAC via JIT-compiled C++ backend.

RLtools compiles C++ at runtime for near-native performance.
This is the fastest RL library in the rl-tools paper benchmark.
"""

import sys
import time

from workers_benchmark.utils import (
    BenchmarkResult, run_subprocess_timed, print_run_header, print_run_result,
)


def run_native_benchmark(config) -> BenchmarkResult:
    """Run RLtools SAC directly via subprocess."""
    print_run_header(config.worker_name, "native", config.env_id,
                     config.total_timesteps, config.num_envs, config.seed,
                     getattr(config, "_current_iteration", 1),
                     config.iterations)

    script = f"""\
import rltools
import gymnasium as gym
import time

def env_factory():
    env = gym.make("{config.env_id}")
    if isinstance(env.action_space, gym.spaces.Box):
        env = gym.wrappers.RescaleAction(env, -1.0, 1.0)
    return env

module = rltools.SAC(env_factory, STEP_LIMIT={config.total_timesteps}, verbose=False)
module.set_environment_factory(env_factory)
state = module.State({config.seed})

finished = False
while not finished:
    finished = state.step()
"""
    cmd = [sys.executable, "-c", script]
    elapsed, peak_mb, stdout, _ = run_subprocess_timed(cmd, timeout=1800)
    sps = config.total_timesteps / elapsed if elapsed > 0 else 0.0

    result = BenchmarkResult(
        worker_name="rltools",
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
