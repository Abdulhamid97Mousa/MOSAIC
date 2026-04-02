"""Launcher for MushroomRL algorithms."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mushroom_rl.core import Core
from mushroom_rl.algorithms.actor_critic import PPO
from mushroom_rl.policy import GaussianTorchPolicy
try:
    from mushroom_rl.environments import Gymnasium
except ImportError:
    from mushroom_rl.environments import Gym as Gymnasium

from .config import MushroomRLWorkerConfig

LOGGER = logging.getLogger(__name__)


class Network(nn.Module):
    """Simple feedforward network for MushroomRL policies."""

    def __init__(self, input_shape, output_shape, n_features=64, **kwargs):
        super().__init__()
        n_input = input_shape[-1]
        n_output = output_shape[0]
        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

    def forward(self, state, **kwargs):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        return self._h3(features2)


def run_ppo(config: MushroomRLWorkerConfig, work_dir: Path, log_dir: Path):
    """Run PPO algorithm using MushroomRL.

    MushroomRL PPO works best with continuous action spaces.
    Default environment is Pendulum-v1.
    """
    args = config.extras
    seed = config.seed if config.seed is not None else 1
    total_timesteps = config.total_timesteps
    n_features = args.get("n_features", 64)
    batch_size = args.get("batch_size", 64)
    lr = args.get("lr", 3e-4)
    n_epochs_policy = args.get("n_epochs_policy", 10)
    eps_ppo = args.get("eps_ppo", 0.2)
    lam = args.get("lam", 0.95)
    horizon = args.get("horizon", 500)
    gamma = args.get("gamma", 0.99)
    steps_per_collect = args.get("steps_per_collect", 2048)

    # Seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Environment setup (use Gymnasium wrapper, not Gym)
    mdp = Gymnasium(config.env_id, horizon=horizon, gamma=gamma)

    # Policy
    policy = GaussianTorchPolicy(
        Network,
        mdp.info.observation_space.shape,
        mdp.info.action_space.shape,
        std_0=1.0,
    )

    # Critic parameters
    critic_params = dict(
        network=Network,
        optimizer={"class": optim.Adam, "params": {"lr": lr}},
        loss=F.mse_loss,
        n_features=n_features,
        batch_size=batch_size,
        input_shape=mdp.info.observation_space.shape,
        output_shape=(1,),
    )

    # Agent
    agent = PPO(
        mdp.info,
        policy,
        actor_optimizer={"class": optim.Adam, "params": {"lr": lr}},
        critic_params=critic_params,
        n_epochs_policy=n_epochs_policy,
        batch_size=batch_size,
        eps_ppo=eps_ppo,
        lam=lam,
    )

    # Core
    core = Core(agent, mdp)

    # Train in chunks
    steps_done = 0
    epoch = 0
    while steps_done < total_timesteps:
        core.learn(n_steps=steps_per_collect, n_steps_per_fit=steps_per_collect)
        steps_done += steps_per_collect
        epoch += 1
        print(f"[MushroomRL PPO] epoch={epoch} steps_done={steps_done}/{total_timesteps}")

    print(f"PPO training finished. Total steps: {steps_done}")


ALGO_MAP: Dict[str, Callable[[MushroomRLWorkerConfig, Path, Path], None]] = {
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
            config = MushroomRLWorkerConfig.from_dict(config_dict)
    else:
        # Construct config from CLI args (fallback)
        config = MushroomRLWorkerConfig(
            run_id=args.run_id,
            algo=args.algo,
            env_id=args.env_id,
            total_timesteps=args.total_timesteps,
            seed=args.seed,
            worker_id=args.worker_id,
        )

    runner = ALGO_MAP.get(config.algo)
    if not runner:
        raise ValueError(f"Algorithm {config.algo} not supported by MushroomRL worker yet.")

    work_dir = args.work_dir or Path(os.getcwd())
    log_dir = args.log_dir or work_dir / "log"

    runner(config, work_dir, log_dir)


if __name__ == "__main__":
    main()
