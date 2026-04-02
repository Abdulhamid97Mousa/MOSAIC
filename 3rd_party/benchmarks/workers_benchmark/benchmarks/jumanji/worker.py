"""Jumanji worker benchmark -- gymnax JIT-compiled PPO + MOSAIC worker imports.

Same fully JIT-compiled gymnax PPO as native.py, but with MOSAIC worker
infrastructure imported (config, telemetry, metadata) to measure the
overhead of running inside the MOSAIC worker subprocess.
"""

import sys
import time

from workers_benchmark.utils import (
    BenchmarkResult, run_subprocess_timed, print_run_header, print_run_result,
)

from .native import _build_gymnax_ppo_script, _parse_elapsed


def run_worker_benchmark(config) -> BenchmarkResult:
    """Run gymnax PPO with MOSAIC worker imports."""
    print_run_header(config.worker_name, "worker", config.env_id,
                     config.total_timesteps, config.num_envs, config.seed,
                     getattr(config, "_current_iteration", 1),
                     config.iterations)

    # Get the gymnax script and prepend MOSAIC worker imports
    gymnax_script = _build_gymnax_ppo_script(
        env_id=config.env_id,
        total_timesteps=config.total_timesteps,
        num_envs=config.num_envs,
        seed=config.seed,
        learning_rate=config.learning_rate,
        num_steps=config.num_steps,
    )

    # Strip os.environ lines from gymnax script (preamble sets them)
    lines = gymnax_script.split('\n')
    script_body = '\n'.join(
        l for l in lines
        if not l.startswith('os.environ') and l != 'import os'
    )

    worker_preamble = '''\
import os, logging
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# ---- MOSAIC worker overhead: import worker infrastructure ----
try:
    from jumanji_worker.config import JumanjiWorkerConfig
    from jumanji_worker import get_worker_metadata
    _meta = get_worker_metadata()
except ImportError:
    pass
try:
    from gym_gui.core.worker import TelemetryEmitter
    _emitter = TelemetryEmitter(run_id="bench_jumanji_worker", logger=logging.getLogger())
    _emitter.run_started({"worker_type": "jumanji", "backend": "gymnax"})
except Exception:
    _emitter = None
'''

    full_script = worker_preamble + script_body
    full_script += '''
if _emitter:
    try:
        _emitter.run_completed({"status": "completed"})
    except Exception:
        pass
'''

    cmd = [sys.executable, "-c", full_script]
    env_overrides = {"GYM_GUI_FASTLANE_ONLY": "0"}

    elapsed, peak_mb, stdout, _ = run_subprocess_timed(cmd, env=env_overrides, timeout=1800)

    inner = _parse_elapsed(stdout)
    if inner > 0:
        elapsed = inner

    sps = config.total_timesteps / elapsed if elapsed > 0 else 0.0

    result = BenchmarkResult(
        worker_name="jumanji",
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
