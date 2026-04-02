"""Launcher for TorchRL algorithms."""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict

# Load MOSAIC site customizations (FastLane gym.make patch) before gym import
try:
    import torchrl_worker.sitecustomize  # noqa: F401
except Exception:
    pass

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim

from torchrl.envs import GymEnv, TransformedEnv, Compose, StepCounter
from torchrl.collectors import SyncDataCollector
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import OneHotCategorical, TanhNormal
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.record.loggers.tensorboard import TensorboardLogger
from tensordict.nn import TensorDictModule

from .config import TorchRLWorkerConfig

LOGGER = logging.getLogger(__name__)


def _detect_action_type(env_id: str) -> str:
    """Detect whether the environment has discrete or continuous actions."""
    probe = gym.make(env_id)
    action_type = "discrete" if isinstance(probe.action_space, gym.spaces.Discrete) else "continuous"
    probe.close()
    return action_type


def run_ppo(config: TorchRLWorkerConfig, work_dir: Path, log_dir: Path):
    """Run PPO algorithm using TorchRL.

    Builds actor-critic networks, a data collector, and the ClipPPOLoss
    objective.  For discrete envs we use OneHotCategorical; for continuous
    envs we use TanhNormal.

    IMPORTANT (TorchRL v0.11): the loss uses ``entropy_coeff`` and
    ``critic_coeff`` (not ``entropy_coef`` / ``critic_coef``).
    """
    extras = config.extras
    seed = config.seed if config.seed is not None else 0
    device = torch.device(extras.get("device", "cpu"))

    torch.manual_seed(seed)
    np.random.seed(seed)

    # -- Environment ---------------------------------------------------------
    base_env = GymEnv(config.env_id, device=device)
    env = TransformedEnv(base_env, Compose(StepCounter()))

    obs_spec = env.observation_spec
    action_spec = env.action_spec

    action_type = _detect_action_type(config.env_id)
    hidden_sizes = extras.get("hidden_sizes", [64, 64])
    lr = extras.get("lr", 3e-4)

    # -- Actor ---------------------------------------------------------------
    if action_type == "discrete":
        n_actions = action_spec.space.n
        actor_net = MLP(
            in_features=obs_spec["observation"].shape[-1],
            out_features=n_actions,
            num_cells=hidden_sizes,
            activation_class=nn.Tanh,
        )
        actor_module = TensorDictModule(
            actor_net,
            in_keys=["observation"],
            out_keys=["logits"],
        )
        actor = ProbabilisticActor(
            module=actor_module,
            in_keys=["logits"],
            out_keys=["action"],
            distribution_class=OneHotCategorical,
            return_log_prob=True,
        )
    else:
        action_dim = action_spec.shape[-1]
        # Output both loc and scale for TanhNormal
        actor_net = MLP(
            in_features=obs_spec["observation"].shape[-1],
            out_features=2 * action_dim,
            num_cells=hidden_sizes,
            activation_class=nn.Tanh,
        )
        actor_module = TensorDictModule(
            actor_net,
            in_keys=["observation"],
            out_keys=["loc_and_scale"],
        )

        # Split loc / scale inside a small wrapper
        class _SplitLocScale(nn.Module):
            def __init__(self, action_dim: int):
                super().__init__()
                self.action_dim = action_dim

            def forward(self, loc_and_scale):
                loc = loc_and_scale[..., : self.action_dim]
                scale = torch.clamp(
                    loc_and_scale[..., self.action_dim :].exp(), min=1e-4, max=10.0
                )
                return loc, scale

        split_module = TensorDictModule(
            _SplitLocScale(action_dim),
            in_keys=["loc_and_scale"],
            out_keys=["loc", "scale"],
        )

        from tensordict.nn import TensorDictSequential

        actor_seq = TensorDictSequential(actor_module, split_module)
        actor = ProbabilisticActor(
            module=actor_seq,
            in_keys=["loc", "scale"],
            out_keys=["action"],
            distribution_class=TanhNormal,
            return_log_prob=True,
        )

    # -- Critic --------------------------------------------------------------
    critic_net = MLP(
        in_features=obs_spec["observation"].shape[-1],
        out_features=1,
        num_cells=hidden_sizes,
        activation_class=nn.Tanh,
    )
    critic = ValueOperator(
        module=critic_net,
        in_keys=["observation"],
    )

    # -- Collector -----------------------------------------------------------
    frames_per_batch = extras.get("frames_per_batch", 2048)
    total_frames = config.total_timesteps

    collector = SyncDataCollector(
        env,
        actor,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
    )

    # -- Loss & Optimizer ----------------------------------------------------
    advantage_module = GAE(
        gamma=extras.get("gamma", 0.99),
        lmbda=extras.get("gae_lambda", 0.95),
        value_network=critic,
    )

    # TorchRL v0.11: use entropy_coeff / critic_coeff (not entropy_coef / critic_coef)
    loss_fn = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=extras.get("clip_epsilon", 0.2),
        entropy_coeff=extras.get("entropy_coeff", 0.01),
        critic_coeff=extras.get("critic_coeff", 0.5),
    )

    optimizer = optim.Adam(loss_fn.parameters(), lr=lr)

    # -- Logger --------------------------------------------------------------
    tb_logger = TensorboardLogger(exp_name=config.run_id, log_dir=str(log_dir))

    # -- Training loop -------------------------------------------------------
    num_epochs = extras.get("num_epochs", 10)
    batch_size = extras.get("batch_size", 64)
    global_step = 0

    for i, batch in enumerate(collector):
        # Compute advantage
        advantage_module(batch)

        # Mini-batch updates
        for _epoch in range(num_epochs):
            num_mini = max(1, batch.shape[0] // batch_size)
            for mb_idx in range(num_mini):
                start = mb_idx * batch_size
                end = min(start + batch_size, batch.shape[0])
                mb = batch[start:end]

                loss_vals = loss_fn(mb)
                total_loss = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                optimizer.zero_grad()
                total_loss.backward()
                max_grad_norm = extras.get("max_grad_norm", 0.5)
                nn.utils.clip_grad_norm_(loss_fn.parameters(), max_grad_norm)
                optimizer.step()

        global_step += batch.numel()

        # Logging
        tb_logger.log_scalar("train/loss_objective", loss_vals["loss_objective"].item(), global_step)
        tb_logger.log_scalar("train/loss_critic", loss_vals["loss_critic"].item(), global_step)
        tb_logger.log_scalar("train/loss_entropy", loss_vals["loss_entropy"].item(), global_step)
        tb_logger.log_scalar("train/total_loss", total_loss.item(), global_step)

        if "episode_reward" in batch.keys():
            mean_reward = batch["episode_reward"][batch["episode_reward"] != 0].mean().item()
            if not math.isnan(mean_reward):
                tb_logger.log_scalar("train/episode_reward", mean_reward, global_step)

        print(
            f"Batch {i}: frames={global_step}/{total_frames} "
            f"loss={total_loss.item():.4f}"
        )

    collector.shutdown()
    print(f"PPO training finished. Total frames collected: {global_step}")


ALGO_MAP: Dict[str, Callable[[TorchRLWorkerConfig, Path, Path], None]] = {
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
            config = TorchRLWorkerConfig.from_dict(config_dict)
    else:
        # Construct config from CLI args (fallback)
        config = TorchRLWorkerConfig(
            run_id=args.run_id,
            algo=args.algo,
            env_id=args.env_id,
            total_timesteps=args.total_timesteps,
            seed=args.seed,
            worker_id=args.worker_id,
        )

    runner = ALGO_MAP.get(config.algo)
    if not runner:
        raise ValueError(f"Algorithm {config.algo} not supported by TorchRL worker yet.")

    work_dir = args.work_dir or Path(os.getcwd())
    log_dir = args.log_dir or work_dir / "log"

    runner(config, work_dir, log_dir)


if __name__ == "__main__":
    main()
