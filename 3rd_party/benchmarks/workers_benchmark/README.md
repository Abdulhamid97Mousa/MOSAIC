# Workers Benchmark Suite

Comprehensive benchmarking system for MOSAIC workers to measure:
1. **Native Performance**: Framework running directly without wrappers
2. **Worker Performance**: Framework wrapped with logging
3. **Worker + Fastlane Performance**: Framework with visual streaming enabled

## Quick Start

```bash
cd /home/hamid/Desktop/software/mosaic/3rd_party/benchmarks

# Run all scenarios for CleanRL (native, worker, fastlane)
python -m workers_benchmark cleanrl --env quick_test

# Run specific scenario
python -m workers_benchmark cleanrl --scenario native --env quick_test

# Analyze results
python -m workers_benchmark --analyze
```

## Available Commands

```bash
# Run all scenarios for a worker
python -m workers_benchmark cleanrl
python -m workers_benchmark xuance  # (placeholder)
python -m workers_benchmark ray     # (placeholder)

# Run specific scenario
python -m workers_benchmark cleanrl --scenario native
python -m workers_benchmark cleanrl --scenario worker
python -m workers_benchmark cleanrl --scenario fastlane

# Run with different environment presets
python -m workers_benchmark cleanrl --env quick_test   # 10K steps, CartPole
python -m workers_benchmark cleanrl --env cartpole     # 50K steps, CartPole

# Analyze all results
python -m workers_benchmark --analyze
```

## Environment Presets

| Preset | Environment | Timesteps | Num Envs | Description |
|--------|-------------|-----------|----------|-------------|
| `quick_test` | CartPole-v1 | 10,000 | 2 | Quick smoke test |
| `cartpole` | CartPole-v1 | 50,000 | 4 | Classic control |

## Benchmark Scenarios

### 1. Native Performance
Tests the raw framework performance without any MOSAIC wrapper overhead.
- No tensorboard logging
- No fastlane visual streaming
- Minimal instrumentation

### 2. Worker Performance
Tests framework wrapped as MOSAIC worker with logging enabled.
- Uses CleanRLWorkerRuntime
- Standard training configuration

### 3. Worker + Fastlane Performance
Tests full MOSAIC worker with visual streaming.
- Fastlane visual streaming enabled
- Full instrumentation

## Metrics Collected

- **Wall time**: Total wall-clock time to complete training
- **Steps per second**: Training throughput
- **Peak memory**: Maximum memory usage during training
- **Overhead percentage**: Relative to native performance

## Example Output

```
======================================================================
  SUMMARY: CLEANRL
======================================================================
native      :     6.60s |  1514.99 steps/s |  1335.86 MB
worker      :    12.11s |   825.87 steps/s |  1540.05 MB (+83.4% overhead)
fastlane    :    15.58s |   641.75 steps/s |  1533.31 MB (+136.1% overhead)
======================================================================
```

## Structure

```
workers_benchmark/
├── __main__.py          # CLI entry point
├── example.py           # Quick example script
├── utils.py             # Timing and metrics utilities
├── configs/
│   ├── __init__.py
│   └── common.py        # Benchmark configurations
├── benchmarks/
│   ├── __init__.py
│   ├── cleanrl/         # CleanRL benchmarks (implemented)
│   │   ├── __init__.py
│   │   ├── native.py
│   │   ├── worker.py
│   │   └── fastlane.py
│   ├── xuance/          # Xuance benchmarks (placeholder)
│   ├── ray/             # Ray benchmarks (placeholder)
│   ├── jumanji/         # Jumanji benchmarks (placeholder)
│   └── mctx/            # MCTX benchmarks (placeholder)
├── scripts/
│   ├── run_worker.py    # Run all scenarios for a worker
│   ├── run_all.py       # Run all workers
│   └── analyze_results.py  # Analyze and compare results
└── results/             # JSON benchmark results
```

## Implementing New Worker Benchmarks

To implement benchmarks for a new worker:

1. Create `benchmarks/<worker_name>/__init__.py` with three functions:
   ```python
   def run_native_benchmark(config) -> BenchmarkResult:
       # Run framework directly
       ...

   def run_worker_benchmark(config) -> BenchmarkResult:
       # Run with worker wrapper
       ...

   def run_fastlane_benchmark(config) -> BenchmarkResult:
       # Run with fastlane visual streaming
       ...
   ```

2. Follow the CleanRL implementation as reference.

## Notes

- Results are saved as JSON files in `results/` directory
- The native benchmark runs a self-contained PPO implementation to measure baseline performance
- Worker and fastlane benchmarks use the actual cleanrl_worker runtime
- Fastlane requires gym_gui to be installed
