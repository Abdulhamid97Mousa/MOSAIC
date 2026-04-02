# MOSAIC Worker Benchmarks - Phase 1 Foundation Complete

## Overview

This directory contains a comprehensive benchmarking system to compare MOSAIC workers' performance across three execution modes:
1. **Native**: Framework running directly without wrappers
2. **Worker**: Framework wrapped with MOSAIC worker interface
3. **FastLane**: Worker mode with live visualization streaming

## Phase 1: Foundation - Complete ✓

### What's Done

#### 1. Research & Analysis
- Analyzed source code for all 4 workers (CleanRL, Xuance, Ray, Jumanji)
- Documented actual APIs, not guesses
- Created comprehensive reference guide

#### 2. Configuration System
```
configs/
├── cleanrl.yaml      # CleanRL PPO parameters
├── xuance.yaml       # Xuance algorithm parameters (46+ algorithms supported)
├── ray.yaml          # RLlib algorithm parameters (5 algorithms)
└── jumanji.yaml      # Jumanji A2C parameters (A2C already exists!)
```

Each config file includes:
- Environment settings (CartPole-v1, 100k timesteps, 1 environment)
- Algorithm hyperparameters (based on actual upstream defaults)
- FastLane configuration
- Benchmark settings (timeout, warmup)

#### 3. Reusable Base Template
```
utils/benchmark_base.py
├── AbstractBenchmark (base class for all benchmarks)
│   ├── run_benchmark() - abstract method
│   ├── save_results() - automatic CSV + JSON export
│   ├── print_summary() - statistics (mean, std, overhead %)
│   └── plot_results() - publication-ready PNG plots
├── BenchmarkConfig (dataclass for benchmark parameters)
└── Automatic metadata collection (system info, timestamps, git state)
```

#### 4. Example Implementation
```
benchmarks/cleanrl_benchmark.py
├── CleanRLBenchmark class (extends AbstractBenchmark)
├── run_benchmark() - 3 iterations with fixed seeds
├── Scenario testing:
│   ├── _run_native() - Direct execution
│   ├── _run_worker() - Via CleanRLWorkerRuntime
│   └── _run_fastlane() - Worker + GYM_GUI_FASTLANE_ONLY=1
└── Automatic result saving + plotting
```

Run it:
```bash
python benchmarks/cleanrl_benchmark.py
```

Output:
```
logs/
├── cleanrl_results.csv        # Timing data + statistics
├── cleanrl_metadata.json      # System info, config, reproducibility
├── cleanrl_comparison.png     # Bar chart (native vs worker vs fastlane)
└── cleanrl_overhead.png       # Overhead percentage chart
```

#### 5. Comprehensive Reference
```
CONFIG_REFERENCE.md - Complete guide covering:
├── CleanRL Worker
│   ├── Config structure (env_id, total_timesteps, etc.)
│   ├── PPO hyperparameters
│   ├── Execution modes (native/worker/fastlane)
│   └── FastLane environment variables
├── Xuance Worker
│   ├── Configuration (method, env, running_steps)
│   ├── Supported algorithms (46+ including multi-agent)
│   ├── Execution modes
│   └── FastLane support
├── Ray Worker
│   ├── RayWorkerConfig structure
│   ├── Algorithms (PPO, IMPALA, APPO, DQN, SAC)
│   ├── Policy configurations (PARAMETER_SHARING, INDEPENDENT, etc.)
│   └── FastLane integration
└── Jumanji Worker
    ├── JumanjiWorkerConfig structure
    ├── A2C hyperparameters
    ├── Available environments (24 total)
    └── JAX framework details
```

## Phase 2: Benchmark Implementations - Next

### To Build

1. **xuance_benchmark.py** - Adapt template for Xuance
   - Use XuanceWorkerConfig + XuanceWorkerRuntime
   - Test native, worker, fastlane modes
   - Generate CSV + plots

2. **ray_benchmark.py** - Adapt template for Ray/RLlib
   - Use RayWorkerConfig + RayWorkerRuntime
   - Handle Ray distributed setup
   - Generate CSV + plots

3. **jumanji_benchmark.py** - Adapt template for Jumanji
   - Use JumanjiWorkerConfig + JumanjiWorkerRuntime
   - Test A2C algorithm (already exists!)
   - Handle JAX pmap vectorization
   - Generate CSV + plots

### Running Phase 1 Benchmarks

```bash
# CleanRL (example - only this works now)
python benchmarks/cleanrl_benchmark.py

# After implementing Phase 2:
python benchmarks/xuance_benchmark.py
python benchmarks/ray_benchmark.py
python benchmarks/jumanji_benchmark.py
```

## Phase 3: Learning Benchmarks - Next

### MiniGrid Benchmark

Measure real learning performance on MiniGrid-DoorKey-16x16-v0:
- Wall-clock time to reach target reward
- Steps/second during training
- Learning curves convergence

Create: `learning_benchmarks/minigrid_benchmark.py`

## Phase 4: Consolidated Dashboard - Next

### Results Analysis & Comparison

Create: `analysis/results_analyzer.py`

Generate:
- Overhead comparison across all workers
- Learning speed comparison
- Best worker recommendations per environment
- Master report (HTML/PDF)

## Key Findings

### No Custom Implementation Needed!

**Major discovery during research:**
- Jumanji **already has A2C algorithm** - no need to create custom PPO
- All workers **already support FastLane** - no changes needed to workers
- All worker APIs are **well-documented and stable**

### Performance Insights

From your earlier benchmark results:
- CleanRL worker overhead: ~10% (reasonable for unified interface)
- FastLane overhead: negligible for CartPole
- Workers are efficient abstractions!

## File Structure

```
3rd_party/benchmarks/workers_benchmark/
├── CONFIG_REFERENCE.md              ← Start here for understanding configs
├── INDEX.md                         ← This file
│
├── configs/
│   ├── cleanrl.yaml                 ✓ Accurate, code-analyzed
│   ├── xuance.yaml                  ✓ Accurate, code-analyzed
│   ├── ray.yaml                     ✓ Accurate, code-analyzed
│   └── jumanji.yaml                 ✓ Accurate, code-analyzed
│
├── utils/
│   ├── benchmark_base.py            ✓ Complete reusable base class
│   └── __init__.py
│
├── benchmarks/
│   ├── cleanrl_benchmark.py         ✓ Working example
│   ├── xuance_benchmark.py          [ ] TODO: Adapt template
│   ├── ray_benchmark.py             [ ] TODO: Adapt template
│   └── jumanji_benchmark.py         [ ] TODO: Adapt template
│
├── learning_benchmarks/
│   └── minigrid_benchmark.py        [ ] TODO: Learning speed tests
│
├── analysis/
│   ├── results_analyzer.py          [ ] TODO: Consolidated dashboard
│   └── plot_utils.py                [ ] TODO: Shared plotting
│
└── logs/
    ├── cleanrl_results.csv          (Generated at runtime)
    ├── cleanrl_metadata.json        (Generated at runtime)
    ├── cleanrl_comparison.png       (Generated at runtime)
    └── ... (other workers' results)
```

## How to Use

### 1. Understand Configurations
```bash
cat CONFIG_REFERENCE.md
```

### 2. Review Example
```bash
cat benchmarks/cleanrl_benchmark.py
```

### 3. Run Phase 1 (CleanRL)
```bash
python benchmarks/cleanrl_benchmark.py
ls logs/
```

### 4. Create Worker Benchmarks (Phase 2)
- Copy cleanrl_benchmark.py → xuance_benchmark.py
- Adapt methods:
  - `_run_native()` → Use correct worker API
  - `_run_worker()` → Use correct WorkerConfig/Runtime
  - `_run_fastlane()` → Enable FastLane for that worker
- Update imports

### 5. Run Learning Benchmarks (Phase 3)
```bash
python learning_benchmarks/minigrid_benchmark.py
```

### 6. Generate Dashboard (Phase 4)
```bash
python analysis/results_analyzer.py
```

## Important Notes

### Reproducibility
- All benchmarks use fixed seeds
- Metadata saved: system info, Python version, git commit, timestamps
- Results CSV includes statistics (mean, std, overhead %)

### Paper-Ready Plots
- High DPI (300) PNG output
- Clean matplotlib styling
- Proper axis labels and legends
- Overhead percentages displayed

### Timeout Protection
- All runs have 1-hour timeout (configurable per worker)
- Prevents hanging on long benchmarks
- Graceful error handling with warnings

## Next Steps

1. Review `CONFIG_REFERENCE.md` for all worker details
2. Study `benchmarks/cleanrl_benchmark.py` as template
3. Create `xuance_benchmark.py` by adapting template
4. Create `ray_benchmark.py` by adapting template
5. Create `jumanji_benchmark.py` by adapting template
6. Create `minigrid_benchmark.py` for learning benchmarks
7. Create `analysis/results_analyzer.py` for dashboard

## Questions?

Refer to:
- **CONFIG_REFERENCE.md** - All parameter mappings and APIs
- **utils/benchmark_base.py** - Base class documentation
- **benchmarks/cleanrl_benchmark.py** - Working example with comments
