# MOSAIC Worker Benchmarks - Configuration Reference

This document describes the actual configuration options for each worker, based on upstream code analysis.

## CleanRL Worker

**Config File:** `configs/cleanrl.yaml`

**Key Parameters:**
- `env_id`: Environment identifier (e.g., "CartPole-v1")
- `total_timesteps`: Total training steps
- `num_envs`: Number of parallel environments
- `seed`: Random seed for reproducibility

**PPO Hyperparameters:**
- `learning_rate`: LR for policy and value network
- `num_steps`: Steps per environment per rollout (typically 2048)
- `num_minibatches`: Number of mini-batches per update
- `update_epochs`: Number of epochs per update
- `gamma`: Discount factor (0.99)
- `gae_lambda`: GAE advantage discount (0.95)
- `clip_coef`: PPO clipping parameter (0.2)
- `ent_coef`: Entropy coefficient
- `vf_coef`: Value function coefficient

**Execution Modes:**
1. **Native**: Direct execution via `python -m cleanrl.ppo`
2. **Worker**: Via CleanRLWorkerConfig + CleanRLWorkerRuntime
3. **FastLane**: Worker mode with `GYM_GUI_FASTLANE_ONLY=1` environment variable

**FastLane Configuration:**
- `GYM_GUI_FASTLANE_SLOT`: Which parallel env to record (default: 0)
- `GYM_GUI_FASTLANE_VIDEO_MODE`: "single", "grid", or "off"
- `GYM_GUI_FASTLANE_GRID_LIMIT`: Max envs in composite grid (default: 4)

---

## Xuance Worker

**Config File:** `configs/xuance.yaml`

**Key Parameters:**
- `env`: Environment family (e.g., "classic_control", "mpe", "smac")
- `env_id`: Specific environment ID
- `method`: Algorithm name (normalized to config folder: "ppo", "mappo", "dqn", etc.)
- `running_steps`: Total training steps (note: not timesteps)
- `parallels`: Number of parallel environments
- `dl_toolbox`: Deep learning backend ("torch", "tensorflow", "mindspore")
- `device`: "cpu" or "cuda:0", etc.

**Supported Algorithms (46+):**
- Single-agent: PG, A2C, PPO_Clip, PPO_KL, DQN, DDPG, TD3, SAC, etc.
- Multi-agent: IPPO, MAPPO, MADDPG, QMIX, VDN, etc.
- Competition: SELF_PLAY, SHARED_VALUE_FUNCTION variants

**PPO Configuration:**
- Similar to CleanRL (learning_rate, gamma, gae_lambda, clip_range, etc.)
- Architecture: representation networks, actor/critic networks
- Loss weights: entropy_coef, value_loss_coef

**Execution Modes:**
1. **Native**: Standard training with `XuanCeWorkerRuntime.run()`
2. **Worker**: Via MOSAIC worker wrapper
3. **FastLane**: Enabled via environment variables

**FastLane in Xuance:**
- Auto-detects: `GYM_GUI_FASTLANE_ONLY` or `MOSAIC_FASTLANE_ENABLED`
- Full support for grid video mode and multi-slot recording

---

## Ray Worker (RLlib)

**Config File:** `configs/ray.yaml`

**Key Parameters:**
- `env_id`: Gymnasium environment ID
- `algorithm`: "PPO", "IMPALA", "APPO", "DQN", or "SAC"
- `total_timesteps`: Total training steps
- `num_workers`: Number of remote Ray workers
- `num_gpus`: Total GPUs to use
- `num_cpus_per_worker`: CPUs per worker

**Ray-Specific:**
- Uses **OLD RLlib API stack** (not new API)
- Why: Compatibility with diverse observation shapes (chess 8x8, pursuit-evasion, etc.)
- Policy configuration: PARAMETER_SHARING (shared weights), INDEPENDENT, SELF_PLAY, SHARED_VALUE_FUNCTION

**PPO Configuration (RLlib style):**
- `lr`: Learning rate (default: 5e-5)
- `gamma`: Discount factor
- `lambda`: GAE lambda
- `clip_param`: PPO clipping (0.2)
- `sgd_minibatch_size`: Mini-batch size
- `num_sgd_iter`: SGD iterations per iteration
- `train_batch_size`: Total batch size per iteration

**Execution Modes:**
1. **Native**: Direct RLlib `Algorithm.train()`
2. **Worker**: Via RayWorkerConfig + RayWorkerRuntime
3. **FastLane**: Enabled via config flag `fastlane_enabled: true`

**FastLane in Ray:**
- Set via runtime environment variables
- Per-worker streams for distributed training
- Supports frame downscaling and grid visualization

---

## Jumanji Worker (JAX-based)

**Config File:** `configs/jumanji.yaml`

**Key Parameters:**
- `env_id`: Jumanji environment (24 total)
  - Phase 1 (Logic): Game2048, GraphColoring, Minesweeper, RubiksCube, SlidingTilePuzzle, Sudoku
  - Phase 2 (Packing): BinPack, FlatPack, JobShop, Knapsack, Tetris
  - Phase 3 (Routing): Cleaner, Connector, CVRP, Maze, MMST, MultiCVRP, PacMan, RobotWarehouse, Snake, Sokoban, TSP
- `agent`: "a2c" (Advantage Actor-Critic) or "random" (baseline)

**A2C Hyperparameters:**
- `learning_rate`: 0.001 (default)
- `discount_factor`: 0.99 (gamma)
- `bootstrapping_factor`: 1.0 (lambda for TD)
- `l_pg`: Policy gradient loss weight
- `l_td`: Temporal difference loss weight
- `l_en`: Entropy regularization weight

**JAX-Specific:**
- All training is JIT-compiled with `jax.pmap()` (data-parallel mapping)
- Per-environment networks built with Haiku
- Fully vectorized across devices
- Internal vectorization (no traditional `num_envs` parameter)

**Available JAX Libraries:**
- `distrax`: Distribution library for policy parametrization
- `dm-haiku`: Neural network library
- `optax`: JAX optimizers
- `rlax`: JAX RL utilities (TD losses, advantage computation)

**Execution Modes:**
1. **Native**: Direct training via `jumanji/training/train.py`
2. **Worker**: Via JumanjiWorkerConfig + JumanjiWorkerRuntime
3. **FastLane**: Supported (frame capture from agent observations)

---

## Benchmark Structure

All configurations follow the same pattern:

```yaml
environment:
  env_id: "..."
  # environment-specific params

training:
  algorithm/method/agent: "..."
  num_iterations: 3
  seed: 42

[algorithm]_config:
  learning_rate: ...
  gamma: ...
  # algorithm-specific hyperparameters

scenarios:
  - native      # Direct algorithm execution
  - worker      # Via MOSAIC worker wrapper
  - fastlane    # Worker + live visualization

benchmark_settings:
  warmup_runs: 0
  timeout_seconds: 3600
```

## Running Benchmarks

Each worker has a dedicated benchmark script:

```bash
# CleanRL
python 3rd_party/benchmarks/workers_benchmark/benchmarks/cleanrl_benchmark.py

# Xuance
python 3rd_party/benchmarks/workers_benchmark/benchmarks/xuance_benchmark.py

# Ray
python 3rd_party/benchmarks/workers_benchmark/benchmarks/ray_benchmark.py

# Jumanji
python 3rd_party/benchmarks/workers_benchmark/benchmarks/jumanji_benchmark.py
```

## Results Output

Each benchmark produces:
- **CSV**: `{worker}_results.csv` with timing data and statistics
- **JSON**: `{worker}_metadata.json` with system info and config
- **PNG**: `{worker}_comparison.png` and `{worker}_overhead.png` with plots

---

## Next Steps

1. Create benchmark scripts using the actual configurations
2. Implement learning benchmarks (MiniGrid-DoorKey-16x16-v0)
3. Create consolidated dashboard comparing all workers
