"""Launcher for OmniSafe algorithms."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict

import omnisafe

from .config import OmniSafeWorkerConfig, ENV_MAP

LOGGER = logging.getLogger(__name__)


def run_ppolag(config: OmniSafeWorkerConfig, work_dir: Path, log_dir: Path):
    """Run PPOLag (PPO-Lagrangian) algorithm using OmniSafe.

    Maps standard gym env IDs to Safety-Gymnasium equivalents via config.safe_env_id.
    """
    args = config.extras
    seed = config.seed if config.seed is not None else 0
    total_timesteps = config.total_timesteps
    num_envs = args.get("num_envs", 1)
    steps_per_epoch = args.get("steps_per_epoch", 1000)
    update_iters = args.get("update_iters", 10)

    omnisafe_env = config.safe_env_id
    LOGGER.info(
        "Mapped env_id=%s to omnisafe_env=%s", config.env_id, omnisafe_env,
    )

    agent = omnisafe.Agent(
        "PPOLag",
        omnisafe_env,
        custom_cfgs={
            "seed": seed,
            "train_cfgs": {
                "total_steps": total_timesteps,
                "vector_env_nums": num_envs,
            },
            "algo_cfgs": {
                "steps_per_epoch": steps_per_epoch,
                "update_iters": update_iters,
            },
            "logger_cfgs": {
                "use_wandb": False,
                "use_tensorboard": True,
                "log_dir": str(log_dir),
            },
        },
    )

    agent.learn()
    agent.close()

    print(f"PPOLag training finished. Total steps: {total_timesteps}")


ALGO_MAP: Dict[str, Callable[[OmniSafeWorkerConfig, Path, Path], None]] = {
    "ppolag": run_ppolag,
    "ppo": run_ppolag,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--algo", type=str, required=True)
    parser.add_argument("--env-id", type=str, required=True)
    parser.add_argument("--total-timesteps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
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
            config = OmniSafeWorkerConfig.from_dict(config_dict)
    else:
        # Construct config from CLI args (fallback)
        config = OmniSafeWorkerConfig(
            run_id=args.run_id,
            algo=args.algo,
            env_id=args.env_id,
            total_timesteps=args.total_timesteps,
            seed=args.seed,
            worker_id=args.worker_id,
        )

    runner = ALGO_MAP.get(config.algo)
    if not runner:
        raise ValueError(f"Algorithm {config.algo} not supported by OmniSafe worker yet.")

    work_dir = args.work_dir or Path(os.getcwd())
    log_dir = args.log_dir or work_dir / "log"

    runner(config, work_dir, log_dir)


if __name__ == "__main__":
    main()
