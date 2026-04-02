"""Benchmark utilities: timing, memory measurement, and result storage.

Design follows rl-tools' approach: OS-level wall-clock timing via
time.perf_counter, plus periodic VmPeak sampling via a background thread
for true peak-memory tracking (the old code only checked at enter/exit).
"""

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Dict, Any, List


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    worker_name: str          # e.g. "cleanrl", "xuance"
    scenario: str             # "native", "worker", or "fastlane"
    env_id: str
    total_timesteps: int

    # Timing (like rl-tools' hyperfine output)
    wall_time_seconds: float
    steps_per_second: float

    # Memory (like rl-tools' maxresident from /usr/bin/time)
    peak_memory_mb: float

    # Optional training metrics
    final_episode_return: Optional[float] = None
    final_episode_length: Optional[float] = None

    # Metadata
    seed: int = 42
    num_envs: int = 1
    iteration: int = 1
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        env_safe = self.env_id.replace("/", "_").replace("-", "_")
        fname = f"{self.worker_name}_{self.scenario}_{env_safe}_i{self.iteration}.json"
        path = output_dir / fname
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path


# ---------------------------------------------------------------------------
# Timer with background memory sampling
# ---------------------------------------------------------------------------

class BenchmarkTimer:
    """High-resolution timer with periodic peak-memory sampling.

    Uses time.perf_counter (not time.time) for wall-clock measurement,
    matching rl-tools' use of std::chrono::high_resolution_clock.

    A daemon thread polls /proc/self/status VmPeak every 0.5 s so that
    short-lived memory spikes during training are not missed (the old
    code only checked RSS at __enter__ and __exit__).
    """

    def __init__(self) -> None:
        self._start: float = 0.0
        self._end: Optional[float] = None
        self._peak_kb: float = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "BenchmarkTimer":
        self._peak_kb = _read_vm_rss_kb()
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._end = time.perf_counter()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._peak_kb = max(self._peak_kb, _read_vm_rss_kb())

    @property
    def elapsed_seconds(self) -> float:
        end = self._end if self._end is not None else time.perf_counter()
        return end - self._start

    @property
    def peak_memory_mb(self) -> float:
        return self._peak_kb / 1024.0

    def _sample(self) -> None:
        while not self._stop.is_set():
            sample = _read_vm_rss_kb()
            if sample > self._peak_kb:
                self._peak_kb = sample
            self._stop.wait(0.5)


def _read_vm_rss_kb() -> float:
    """Read VmRSS (current resident memory) from /proc/self/status.

    We use VmRSS not VmPeak because VmPeak is monotonically increasing
    for the lifetime of the process — useless for in-process benchmarks
    that run multiple iterations. We track our own max over VmRSS samples.

    Falls back to resource module, then psutil.
    """
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1])
    except (FileNotFoundError, OSError):
        pass
    try:
        import resource
        # ru_maxrss is in KB on Linux
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Subprocess runner (like rl-tools' benchmark.sh wrapping /usr/bin/time)
# ---------------------------------------------------------------------------

def run_subprocess_timed(
    cmd: List[str],
    env: Optional[Dict[str, str]] = None,
    timeout: int = 3600,
) -> tuple:
    """Run a subprocess and return (elapsed_seconds, peak_memory_mb, stdout, stderr).

    Tries /usr/bin/time -v for OS-level metrics (like rl-tools does),
    falls back to pure perf_counter timing.
    """
    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    # Disable wandb everywhere
    run_env["WANDB_MODE"] = "disabled"
    run_env["PYTHONUNBUFFERED"] = "1"

    # Try /usr/bin/time for maxresident (like rl-tools' benchmark.sh)
    time_bin = "/usr/bin/time"
    use_time = os.path.exists(time_bin)

    if use_time:
        full_cmd = [time_bin, "-v"] + cmd
    else:
        full_cmd = cmd

    start = time.perf_counter()
    result = subprocess.run(
        full_cmd,
        capture_output=True,
        text=True,
        env=run_env,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - start

    peak_mb = 0.0
    stderr = result.stderr

    if use_time:
        # Parse "Maximum resident set size (kbytes): 123456"
        for line in stderr.splitlines():
            if "Maximum resident set size" in line:
                try:
                    peak_mb = float(line.split(":")[-1].strip()) / 1024.0
                except (ValueError, IndexError):
                    pass

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}):\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stderr (last 500 chars): {stderr[-500:]}"
        )

    return elapsed, peak_mb, result.stdout, stderr


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_run_header(worker: str, scenario: str, env_id: str,
                     timesteps: int, num_envs: int, seed: int,
                     iteration: int, total_iterations: int) -> None:
    print(f"\n  [{worker}/{scenario}] {env_id}  "
          f"{timesteps:,} steps  {num_envs} envs  seed={seed}  "
          f"({iteration}/{total_iterations})")


def print_run_result(result: BenchmarkResult) -> None:
    print(f"    -> {result.wall_time_seconds:.2f}s  "
          f"{result.steps_per_second:.0f} SPS  "
          f"{result.peak_memory_mb:.0f} MB")
