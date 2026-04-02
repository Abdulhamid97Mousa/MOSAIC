"""Launcher for SBX (Stable-Baselines Jax) algorithms."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict

# Load MOSAIC site customizations (FastLane gym.make patch) before gym import
try:
    import sbx_worker.sitecustomize  # noqa: F401
except Exception:
    pass

import gymnasium as gym
from sbx import PPO, DQN, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from .config import SBXWorkerConfig

LOGGER = logging.getLogger(__name__)


def run_ppo(config: SBXWorkerConfig, work_dir: Path, log_dir: Path):
    """Run PPO algorithm (JAX backend)."""
    args = config.extras

    n_envs = args.get("n_envs", 4)
    vec_env = make_vec_env(config.env_id, n_envs=n_envs, seed=config.seed)

    # Build model kwargs from extras
    model_kwargs: Dict[str, Any] = {}
    for key in ("learning_rate", "n_steps", "batch_size", "n_epochs",
                "gamma", "gae_lambda", "clip_range", "ent_coef",
                "vf_coef", "max_grad_norm"):
        if key in args:
            model_kwargs[key] = args[key]

    policy = args.get("policy", "MlpPolicy")
    model = PPO(
        policy,
        vec_env,
        tensorboard_log=str(log_dir),
        seed=config.seed,
        verbose=1,
        **model_kwargs,
    )

    # Optional eval callback
    eval_env = None
    callbacks = []
    if args.get("eval_freq"):
        eval_env = make_vec_env(config.env_id, n_envs=1, seed=config.seed + 1000 if config.seed is not None else None)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(work_dir / "best_model"),
            log_path=str(log_dir / "eval"),
            eval_freq=args["eval_freq"],
            n_eval_episodes=args.get("n_eval_episodes", 5),
            deterministic=True,
        )
        callbacks.append(eval_callback)

    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks if callbacks else None,
    )

    # Save final model
    model.save(str(work_dir / "final_model"))
    print(f"PPO (JAX) training finished. Model saved to {work_dir / 'final_model'}")

    vec_env.close()
    if eval_env is not None:
        eval_env.close()



# Note: SBX does not provide A2C. Use SB3 worker for A2C.


def run_dqn(config: SBXWorkerConfig, work_dir: Path, log_dir: Path):
    """Run DQN algorithm (JAX backend)."""
    args = config.extras

    env = gym.make(config.env_id)
    if not isinstance(env.action_space, gym.spaces.Discrete):
        env.close()
        raise ValueError(
            f"DQN only supports discrete action spaces, but {config.env_id} has {env.action_space}"
        )
    env.close()

    vec_env = make_vec_env(config.env_id, n_envs=1, seed=config.seed)

    model_kwargs: Dict[str, Any] = {}
    for key in ("learning_rate", "buffer_size", "learning_starts",
                "batch_size", "tau", "gamma", "train_freq",
                "target_update_interval", "exploration_fraction",
                "exploration_initial_eps", "exploration_final_eps"):
        if key in args:
            model_kwargs[key] = args[key]

    policy = args.get("policy", "MlpPolicy")
    model = DQN(
        policy,
        vec_env,
        tensorboard_log=str(log_dir),
        seed=config.seed,
        verbose=1,
        **model_kwargs,
    )

    model.learn(total_timesteps=config.total_timesteps)

    model.save(str(work_dir / "final_model"))
    print(f"DQN (JAX) training finished. Model saved to {work_dir / 'final_model'}")

    vec_env.close()


def run_sac(config: SBXWorkerConfig, work_dir: Path, log_dir: Path):
    """Run SAC algorithm (JAX backend)."""
    args = config.extras

    vec_env = make_vec_env(config.env_id, n_envs=1, seed=config.seed)

    model_kwargs: Dict[str, Any] = {}
    for key in ("learning_rate", "buffer_size", "learning_starts",
                "batch_size", "tau", "gamma", "train_freq",
                "ent_coef", "target_entropy"):
        if key in args:
            model_kwargs[key] = args[key]

    policy = args.get("policy", "MlpPolicy")
    model = SAC(
        policy,
        vec_env,
        tensorboard_log=str(log_dir),
        seed=config.seed,
        verbose=1,
        **model_kwargs,
    )

    model.learn(total_timesteps=config.total_timesteps)

    model.save(str(work_dir / "final_model"))
    print(f"SAC (JAX) training finished. Model saved to {work_dir / 'final_model'}")

    vec_env.close()


ALGO_MAP: Dict[str, Callable[[SBXWorkerConfig, Path, Path], None]] = {
    "ppo": run_ppo,
    "dqn": run_dqn,
    "sac": run_sac,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--algo", type=str, required=True)
    parser.add_argument("--env-id", type=str, required=True)
    parser.add_argument("--total-timesteps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--worker-id", type=str)
    parser.add_argument("--config-file", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--log-dir", type=Path)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Load config if provided
    if args.config_file:
        with open(args.config_file, "r") as f:
            config_dict = json.load(f)
            config = SBXWorkerConfig.from_dict(config_dict)
    else:
        # Construct config from CLI args (fallback)
        config = SBXWorkerConfig(
            run_id=args.run_id,
            algo=args.algo,
            env_id=args.env_id,
            total_timesteps=args.total_timesteps,
            seed=args.seed,
            worker_id=args.worker_id,
        )

    runner = ALGO_MAP.get(config.algo)
    if not runner:
        supported = ", ".join(sorted(ALGO_MAP.keys()))
        raise ValueError(
            f"Algorithm '{config.algo}' not supported by SBX worker. Supported: {supported}"
        )

    work_dir = args.work_dir or Path(os.getcwd())
    log_dir = args.log_dir or work_dir / "log"

    runner(config, work_dir, log_dir)


if __name__ == "__main__":
    main()
