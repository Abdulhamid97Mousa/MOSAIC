"""Launcher for Pearl algorithms."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict

import gymnasium as gym
import numpy as np

from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.sequential_decision_making.ppo import (
    ProximalPolicyOptimization,
    PPOReplayBuffer,
)
from pearl.utils.functional_utils.train_and_eval.online_learning import online_learning
from pearl.utils.instantiations.environments.gym_environment import GymEnvironment

from .config import PearlWorkerConfig

LOGGER = logging.getLogger(__name__)


def run_ppo(config: PearlWorkerConfig, work_dir: Path, log_dir: Path):
    """Run PPO algorithm using Pearl.

    Pearl uses its own agent abstraction built around PearlAgent, which
    wraps the policy learner, replay buffer, and exploration module into
    a single entity trained via online_learning.

    Note: Pearl does not provide a built-in TensorBoard logger, so
    TensorBoard integration is skipped for this worker.
    """
    extras = config.extras
    seed = config.seed if config.seed is not None else 0
    total_timesteps = config.total_timesteps

    # Environment
    env = GymEnvironment(config.env_id)

    # Build agent
    agent = PearlAgent(
        policy_learner=ProximalPolicyOptimization(
            state_dim=env.observation_space.shape[0],
            action_space=env.action_space,
            actor_hidden_dims=extras.get("actor_hidden_dims", [64, 64]),
            critic_hidden_dims=extras.get("critic_hidden_dims", [64, 64]),
            training_rounds=extras.get("training_rounds", 10),
            batch_size=extras.get("batch_size", 64),
            epsilon=extras.get("epsilon", 0.2),
        ),
        replay_buffer=PPOReplayBuffer(
            capacity=extras.get("buffer_capacity", 10000),
        ),
    )

    # Training
    info = online_learning(
        agent=agent,
        env=env,
        number_of_steps=total_timesteps,
        seed=seed,
    )

    print(f"PPO training finished. Info: {info}")


ALGO_MAP: Dict[str, Callable[[PearlWorkerConfig, Path, Path], None]] = {
    "ppo": run_ppo,
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
            config = PearlWorkerConfig.from_dict(config_dict)
    else:
        # Construct config from CLI args (fallback)
        config = PearlWorkerConfig(
            run_id=args.run_id,
            algo=args.algo,
            env_id=args.env_id,
            total_timesteps=args.total_timesteps,
            seed=args.seed,
            worker_id=args.worker_id,
        )

    runner = ALGO_MAP.get(config.algo)
    if not runner:
        raise ValueError(f"Algorithm {config.algo} not supported by Pearl worker yet.")

    work_dir = args.work_dir or Path(os.getcwd())
    log_dir = args.log_dir or work_dir / "log"

    runner(config, work_dir, log_dir)


if __name__ == "__main__":
    main()
