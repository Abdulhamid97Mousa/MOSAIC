# Quick Start Guide

## Installation

The benchmark suite is part of the MOSAIC workers directory. No additional installation is required beyond the worker dependencies.

## Running Benchmarks

### Quick Test (Recommended for first run)

Run a quick benchmark of CleanRL worker:

```bash
cd /home/hamid/Desktop/software/mosaic/3rd_party/benchmarks/workers_benchmark
python example.py
```

This will run all three scenarios (native, worker, fastlane) with a small environment for quick testing.

### Run Specific Worker

```bash
# Run all scenarios for CleanRL
python -m workers_benchmark cleanrl

# Run specific scenario
python -m workers_benchmark cleanrl --scenario native
python -m workers_benchmark cleanrl --scenario worker
python -m workers_benchmark cleanrl --scenario fastlane
```

### Run All Workers

```bash
# Quick test on all workers
python -m workers_benchmark --all

# Full benchmark on all workers
python -m workers_benchmark --all --env minigrid_empty
```

### Analyze Results

```bash
python -m workers_benchmark --analyze
```

## Environment Presets

- `quick_test`: 10K timesteps, 2 envs (fast, for testing)
- `minigrid_empty`: 100K timesteps, 4 envs (standard benchmark)
- `babyai_goto`: 200K timesteps, 4 envs (instruction-following)

## Understanding Results

The benchmark measures three key metrics:

1. **Wall time**: Total time to complete training
2. **Steps/second**: Training throughput
3. **Peak memory**: Maximum memory usage during training

### Overhead Analysis

- **Native**: Baseline performance (framework only)
- **Worker**: Overhead from tensorboard logging
- **Fastlane**: Additional overhead from visual streaming

Expected overhead:
- Worker: 5-15% slower than native
- Fastlane: 10-25% slower than native

## Implementing New Worker Benchmarks

The system provides templates for xuance, ray, jumanji, and mctx workers. To implement:

1. Navigate to `benchmarks/<worker_name>/`
2. Edit `__init__.py` to implement the three benchmark functions:
   - `run_native_benchmark()`: Direct framework execution
   - `run_worker_benchmark()`: Worker-wrapped execution
   - `run_fastlane_benchmark()`: Worker with fastlane

3. Follow the CleanRL implementation as a reference:
   - `benchmarks/cleanrl/native.py`
   - `benchmarks/cleanrl/worker.py`
   - `benchmarks/cleanrl/fastlane.py`

## Troubleshooting

### Import Errors

Make sure you're running from the workers_benchmark directory or have it in your PYTHONPATH:

```bash
export PYTHONPATH=/home/hamid/Desktop/software/mosaic/3rd_party/benchmarks:$PYTHONPATH
```

### Missing Dependencies

Install worker dependencies:

```bash
cd ../cleanrl_worker && pip install -e .
cd ../xuance_worker && pip install -e .
# etc.
```

### Fastlane Not Working

Fastlane requires the gym_gui package. If not available, the benchmark will fall back to a no-op implementation.

## Next Steps

1. Run the quick example: `python example.py`
2. Implement benchmarks for other workers (xuance, ray, etc.)
3. Run full benchmarks: `python -m workers_benchmark --all --env minigrid_empty`
4. Analyze and compare results: `python -m workers_benchmark --analyze`
