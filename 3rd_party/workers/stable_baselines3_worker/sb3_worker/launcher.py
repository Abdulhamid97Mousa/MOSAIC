"""Launcher for Stable-Baselines3 algorithms."""

import argparse
import json
import logging
import os
import sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("WANDB_MODE", "disabled")
from pathlib import Path
from typing import Any, Callable, Dict

import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from .config import SB3WorkerConfig

LOGGER = logging.getLogger(__name__)


def run_ppo(config: SB3WorkerConfig, work_dir: Path, log_dir: Path):
    """Run PPO algorithm."""
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
    print(f"PPO training finished. Model saved to {work_dir / 'final_model'}")

    vec_env.close()
    if eval_env is not None:
        eval_env.close()


def run_a2c(config: SB3WorkerConfig, work_dir: Path, log_dir: Path):
    """Run A2C algorithm."""
    args = config.extras

    n_envs = args.get("n_envs", 4)
    vec_env = make_vec_env(config.env_id, n_envs=n_envs, seed=config.seed)

    model_kwargs: Dict[str, Any] = {}
    for key in ("learning_rate", "n_steps", "gamma", "gae_lambda",
                "ent_coef", "vf_coef", "max_grad_norm", "use_rms_prop"):
        if key in args:
            model_kwargs[key] = args[key]

    policy = args.get("policy", "MlpPolicy")
    model = A2C(
        policy,
        vec_env,
        tensorboard_log=str(log_dir),
        seed=config.seed,
        verbose=1,
        **model_kwargs,
    )

    model.learn(total_timesteps=config.total_timesteps)

    model.save(str(work_dir / "final_model"))
    print(f"A2C training finished. Model saved to {work_dir / 'final_model'}")

    vec_env.close()


def run_dqn(config: SB3WorkerConfig, work_dir: Path, log_dir: Path):
    """Run DQN algorithm."""
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
    print(f"DQN training finished. Model saved to {work_dir / 'final_model'}")

    vec_env.close()


def run_sac(config: SB3WorkerConfig, work_dir: Path, log_dir: Path):
    """Run SAC algorithm."""
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
    print(f"SAC training finished. Model saved to {work_dir / 'final_model'}")

    vec_env.close()


ALGO_MAP: Dict[str, Callable[[SB3WorkerConfig, Path, Path], None]] = {
    "ppo": run_ppo,
    "a2c": run_a2c,
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
            config = SB3WorkerConfig.from_dict(config_dict)
    else:
        # Construct config from CLI args (fallback)
        config = SB3WorkerConfig(
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
            f"Algorithm '{config.algo}' not supported by SB3 worker. Supported: {supported}"
        )

    work_dir = args.work_dir or Path(os.getcwd())
    log_dir = args.log_dir or work_dir / "log"

    runner(config, work_dir, log_dir)


if __name__ == "__main__":
    main()
