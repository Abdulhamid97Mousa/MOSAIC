"""Benchmark configuration.

Environment presets follow rl-tools' paper methodology:
- Pendulum-v1 as canonical benchmark env (same as rl-tools Figure 1/2)
- 300K steps for meaningful training (matching rl-tools' PPO benchmark)
- 10 iterations for statistical confidence (mean +/- sigma)
- CartPole kept only for quick smoke tests
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    env_id: str
    total_timesteps: int
    num_envs: int
    seed: int

    worker_name: str
    scenario: str  # "native", "worker", "fastlane"

    # How many times to repeat for statistics (like hyperfine's default 10)
    iterations: int = 5

    # PPO hyperparams (kept consistent across all frameworks)
    learning_rate: float = 2.5e-4
    num_steps: int = 128

    # Logging toggles (disabled by default to isolate overhead)
    enable_tensorboard: bool = False
    enable_fastlane: bool = False


# rl-tools uses Pendulum-v1 @ 300K for PPO horizontal benchmark (Figure 1).
# We match that for the "standard" preset. CartPole is a quick sanity check.
BENCHMARK_ENVS = {
    "quick_test": {
        "env_id": "CartPole-v1",
        "total_timesteps": 10_000,
        "num_envs": 1,
        "iterations": 3,
        "description": "Smoke test (~5 s)",
    },
    "cartpole": {
        "env_id": "CartPole-v1",
        "total_timesteps": 100_000,
        "num_envs": 4,
        "iterations": 5,
        "description": "CartPole 100K steps",
    },
    "pendulum": {
        "env_id": "Pendulum-v1",
        "total_timesteps": 300_000,
        "num_envs": 1,
        "iterations": 10,
        "description": "Pendulum 300K steps (matches rl-tools paper)",
    },
}

# Workers that use their own environment (not CartPole/Pendulum)
WORKER_ENV_OVERRIDES = {
    "jumanji": "Game2048-v1",
    "rltools": "Pendulum-v1",
}


def get_benchmark_config(
    worker_name: str,
    scenario: str,
    env_name: str = "cartpole",
    seed: int = 42,
) -> BenchmarkConfig:
    """Create a BenchmarkConfig from a preset name."""
    preset = BENCHMARK_ENVS[env_name]
    return BenchmarkConfig(
        env_id=preset["env_id"],
        total_timesteps=preset["total_timesteps"],
        num_envs=preset["num_envs"],
        seed=seed,
        worker_name=worker_name,
        scenario=scenario,
        iterations=preset.get("iterations", 5),
        enable_fastlane=(scenario == "fastlane"),
    )
