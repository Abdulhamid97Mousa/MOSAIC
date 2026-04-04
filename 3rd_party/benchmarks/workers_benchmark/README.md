# MOSAIC Worker Overhead Benchmark Suite

Measures the overhead of wrapping RL frameworks with MOSAIC's worker
abstraction and FastLane visual streaming pipeline.

Three scenarios per framework:

| Scenario | What it measures |
|----------|------------------|
| **Native** | Framework running directly, no MOSAIC wrapping |
| **Worker** | Framework wrapped by MOSAIC worker (logging, config, telemetry) |
| **FastLane** | Worker + real-time frame streaming via shared memory |

## Frameworks benchmarked

| Worker | Framework | Architecture | Status |
|--------|-----------|-------------|--------|
| `cleanrl` | CleanRL | Single-process | Complete |
| `xuance` | XuanCe | Single-process | Complete |
| `ray` | Ray/RLlib | Distributed (multi-process) | Complete |
| `tianshou` | Tianshou | Single-process | Complete |
| `sb3` | Stable-Baselines3 | Single-process | Complete |
| `sbx` | SBX (JAX) | Single-process | Complete |
| `torchrl` | TorchRL | Single-process | Complete |
| `rltools` | RLtools (C++) | Single-process | Complete |
| `jumanji` | Jumanji (JAX) | Single-process (JIT) | Complete |

## Prerequisites

```bash
# From the mosaic root directory
cd /path/to/mosaic

# Activate the virtual environment
source .venv/bin/activate

# Set PYTHONPATH so all worker packages are importable
export PYTHONPATH="3rd_party/benchmarks:3rd_party/workers/cleanrl_worker:3rd_party/workers/ray_worker:3rd_party/workers/xuance_worker:3rd_party/workers/xuance_worker/xuance:$PYTHONPATH"
```

### Per-worker dependencies

Each worker has its own requirements file in `requirements/`:

```bash
pip install -r requirements/base.txt           # shared deps
pip install -r requirements/cleanrl_worker.txt  # CleanRL
pip install -r requirements/ray_worker.txt      # Ray/RLlib
# ... etc.
```

### System requirements

- **RAM**: 8 GB minimum, 16 GB recommended. Ray/RLlib spawns multiple
  worker processes (~1.1 GB each). Running under memory pressure inflates
  wall times due to swap thrashing.
- **CPU**: 4+ cores. Ray benchmarks use `num_cpus=4`.

## Running benchmarks

### Single worker (recommended starting point)

```bash
# Run all 3 scenarios (native/worker/fastlane) x 5 seeds for CleanRL
python -m workers_benchmark worker cleanrl --env cartpole --seed 42

# Quick smoke test (10K steps, 3 iterations)
python -m workers_benchmark worker cleanrl --env quick_test
```

### All 9 workers

```bash
python -m workers_benchmark all --env cartpole --seed 42
```

### Individual worker scripts

```bash
python -m workers_benchmark.scripts.run_worker cleanrl --env cartpole --seed 42
python -m workers_benchmark.scripts.run_worker ray --env cartpole --seed 42
```

## Analyzing results

### Summary tables

```bash
python -m workers_benchmark analyze --plot
```

Prints per-worker overhead tables and cross-worker native comparison,
then generates PNG plots in `results/plots/`.

### Publication charts

```bash
python -m workers_benchmark.scripts.plot_publication --env CartPole-v1
```

Generates:
- `results/plots/combined_overhead.png` -- all frameworks side by side
- `results/plots/overhead_ratios.png` -- horizontal overhead ratio bars

### Per-worker charts

Generated automatically by `analyze --plot`:
- `results/plots/{worker}_overhead.png` -- native vs worker vs fastlane
- `results/plots/native_comparison.png` -- all frameworks native only

## Environment presets

| Preset | Env | Steps | Envs | Iterations | Purpose |
|--------|-----|-------|------|------------|---------|
| `quick_test` | CartPole-v1 | 10,000 | 1 | 3 | Smoke test (~5 s) |
| `cartpole` | CartPole-v1 | 100,000 | 4 | 5 | Standard benchmark |
| `pendulum` | Pendulum-v1 | 300,000 | 1 | 10 | Matches rl-tools paper |

## Metrics collected

- **Wall time**: `time.perf_counter()` (high-resolution, not CPU time)
- **Steps per second**: `total_timesteps / wall_time`
- **Peak memory**: Background thread samples `VmPeak` from `/proc/self/status`
  every 0.5 s. Fallback: `resource.getrusage` or `psutil`.
- **Overhead ratio**: `scenario_mean / native_mean` (1.0x = no overhead)

## Output files

### JSON (one file per run)

Naming: `{worker}_{scenario}_{env}_i{iteration}.json`

```
results/
  cleanrl_native_CartPole_v1_i1.json
  cleanrl_worker_CartPole_v1_i1.json
  cleanrl_fastlane_CartPole_v1_i1.json
  ...
```

### CSV (one file per worker)

Naming: `{worker}_benchmark.csv`

Columns: `worker, scenario, iteration, seed, env_id, total_timesteps,
num_envs, wall_time_s, steps_per_second, peak_memory_mb`

### PNG plots

```
results/plots/
  combined_overhead.png      # all frameworks, 3 bars each
  overhead_ratios.png        # horizontal ratio bars
  native_comparison.png      # native-only ranking
  cleanrl_overhead.png       # per-worker detail
  ray_overhead.png
  ...
```

## Reproducing results from scratch

```bash
# 1. Set up environment
source .venv/bin/activate
export PYTHONPATH="3rd_party/benchmarks:3rd_party/workers/cleanrl_worker:3rd_party/workers/ray_worker:3rd_party/workers/xuance_worker:3rd_party/workers/xuance_worker/xuance:$PYTHONPATH"

# 2. Clear old results (optional)
rm -f 3rd_party/benchmarks/workers_benchmark/results/*.json
rm -f 3rd_party/benchmarks/workers_benchmark/results/*.csv

# 3. Run all workers
python -m workers_benchmark all --env cartpole --seed 42

# 4. Generate publication charts
python -m workers_benchmark.scripts.plot_publication --env CartPole-v1

# 5. Generate analysis + per-worker charts
python -m workers_benchmark analyze --plot
```

**Estimated time**: ~3 hours for all 9 workers on a 4-core machine with
16 GB RAM. Ray/RLlib is the slowest (~25 min for 5 seeds x 3 scenarios).

## Directory structure

```
workers_benchmark/
  __main__.py                 # CLI entry (python -m workers_benchmark)
  utils.py                    # BenchmarkResult, BenchmarkTimer, run_subprocess_timed
  configs/
    common.py                 # BenchmarkConfig dataclass, environment presets
  benchmarks/
    __init__.py               # AVAILABLE_WORKERS list
    cleanrl/                  # native.py, worker.py, fastlane.py
    xuance/
    ray/
    tianshou/
    sb3/
    sbx/
    torchrl/
    rltools/
    jumanji/
  scripts/
    run_worker.py             # Orchestrate 1 worker x 3 scenarios x N seeds
    run_all.py                # Orchestrate all 9 workers
    analyze_results.py        # Load JSONs, compute stats, plot
    plot_publication.py        # Publication-quality combined charts
    compare_workers.py        # Multi-subplot comparison
    compare_workers_lines.py  # Line graphs across iterations
  results/
    *.json                    # Individual run results
    *.csv                     # Per-worker and combined CSVs
    plots/                    # Generated PNG charts
```

## Adding a new worker

1. Create `benchmarks/<name>/` with `__init__.py`, `native.py`,
   `worker.py`, `fastlane.py`.
2. Each file exports a `run_{scenario}_benchmark(config) -> BenchmarkResult`.
3. Add the worker name to `AVAILABLE_WORKERS` in `benchmarks/__init__.py`.
4. See `benchmarks/cleanrl/` as reference.
