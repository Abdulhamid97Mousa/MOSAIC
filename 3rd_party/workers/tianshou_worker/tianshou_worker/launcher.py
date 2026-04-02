"""Launcher for Tianshou algorithms."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict

# Load MOSAIC site customizations (FastLane gym.make patch) before gym import
try:
    import tianshou_worker.sitecustomize  # noqa: F401
except Exception:
    pass

import gymnasium as gym
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv

# Tianshou 2.0 imports
from tianshou.algorithm.modelfree.ppo import PPO
from tianshou.algorithm.modelfree.dqn import DQN, DiscreteQLearningPolicy
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.trainer import OnPolicyTrainerParams, OffPolicyTrainerParams
from tianshou.algorithm.optim import AdamOptimizerFactory

from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import DiscreteActor as Actor, DiscreteCritic as Critic
from tianshou.utils.net.continuous import ContinuousActorProbabilistic as ActorProbabilistic, ContinuousCritic

from .config import TianshouWorkerConfig

LOGGER = logging.getLogger(__name__)


def make_env(env_id: str, seed: int, idx: int, capture_video: bool, run_name: str):
    """Factory for creating gym environments."""
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def run_ppo(config: TianshouWorkerConfig, work_dir: Path, log_dir: Path):
    """Run PPO algorithm."""
    # args from config
    args = config.extras
    
    # Validation
    if args.get("num_envs", 1) < 1:
        raise ValueError("num_envs must be >= 1")
    
    # Environment setup
    env = gym.make(config.env_id)
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0] if isinstance(env.action_space, gym.spaces.Box) else 0

    train_envs = SubprocVectorEnv(
        [make_env(config.env_id, config.seed + i, i, False, config.run_id) for i in range(args.get("num_envs", 1))]
    )
    test_envs = SubprocVectorEnv(
        [make_env(config.env_id, config.seed + 100 + i, i, False, config.run_id) for i in range(args.get("num_test_envs", 1))]
    )

    # Seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    train_envs.seed(config.seed)
    test_envs.seed(config.seed + 100)

    # Network
    device = args.get("device", "cpu")
    net = Net(state_shape=state_shape, hidden_sizes=args.get("hidden_sizes", [64, 64]))

    if isinstance(env.action_space, gym.spaces.Discrete):
        actor_net = Actor(preprocess_net=net, action_shape=action_shape).to(device)
        critic_net = Critic(preprocess_net=Net(state_shape=state_shape, hidden_sizes=args.get("hidden_sizes", [64, 64]))).to(device)
        actor_optim_factory = AdamOptimizerFactory(lr=args.get("lr", 3e-4))
        critic_optim_factory = AdamOptimizerFactory(lr=args.get("lr", 3e-4))

        dist = torch.distributions.Categorical

        # Policy component (Actor)
        policy_actor = ProbabilisticActorPolicy(
            actor=actor_net,
            dist_fn=lambda x: dist(logits=x),
            action_space=env.action_space,
            deterministic_eval=True,
            action_scaling=False,
        )

        # Algorithm component (PPO)
        algorithm = PPO(
            policy=policy_actor,
            critic=critic_net,
            optim=actor_optim_factory, 
            eps_clip=args.get("eps_clip", 0.2),
            dual_clip=args.get("dual_clip", None),
            value_clip=args.get("value_clip", False),
            advantage_normalization=args.get("norm_adv", True),
            recompute_advantage=args.get("recompute_adv", False),
            vf_coef=args.get("vf_coef", 0.5),
            ent_coef=args.get("ent_coef", 0.01),
            max_grad_norm=args.get("max_grad_norm", 0.5),
            gae_lambda=args.get("gae_lambda", 0.95),
            max_batchsize=args.get("max_batchsize", 256),
            return_scaling=False,
        )
    else:
        # Continuous
        actor_net = ActorProbabilistic(net, action_shape, max_action=max_action).to(device)
        critic_net = ContinuousCritic(Net(state_shape=state_shape, hidden_sizes=args.get("hidden_sizes", [64, 64]))).to(device)
        actor_optim_factory = AdamOptimizerFactory(lr=args.get("lr", 3e-4))
        
        # Policy component
        policy_actor = ProbabilisticActorPolicy(
            actor=actor_net,
            dist_fn=None, # handled by ActorProbabilistic
            action_space=env.action_space,
            deterministic_eval=True,
            action_scaling=True,
            action_bound_method="clip",
        )
        
        # Algorithm component
        algorithm = PPO(
            policy=policy_actor,
            critic=critic_net,
            optim=actor_optim_factory,
            eps_clip=args.get("eps_clip", 0.2),
            dual_clip=args.get("dual_clip", None),
            value_clip=args.get("value_clip", False),
            advantage_normalization=args.get("norm_adv", True),
            recompute_advantage=args.get("recompute_adv", False),
            vf_coef=args.get("vf_coef", 0.5),
            ent_coef=args.get("ent_coef", 0.01),
            max_grad_norm=args.get("max_grad_norm", 0.5),
            gae_lambda=args.get("gae_lambda", 0.95),
            max_batchsize=args.get("max_batchsize", 256),
            return_scaling=True,
        )

    # Collector
    train_collector = Collector(algorithm, train_envs, VectorReplayBuffer(args.get("buffer_size", 20000), len(train_envs)))
    test_collector = Collector(algorithm, test_envs)

    # Loggers
    from tianshou.utils import TensorboardLogger
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir)
    logger = TensorboardLogger(writer)

    # Training
    result = algorithm.run_training(
        OnPolicyTrainerParams(
            training_collector=train_collector,
            test_collector=test_collector,
            max_epochs=args.get("epoch", 10),
            epoch_num_steps=args.get("step_per_epoch", 10000),
            collection_step_num_env_steps=args.get("step_per_collect", 2048),
            test_step_num_episodes=args.get("test_num", 10),
            batch_size=args.get("batch_size", 64),
            update_step_num_repetitions=args.get("repeat_per_collect", 10),
            logger=logger,
        )
    )
    
    print(f"PPO training finished. Result: {result}")


def run_dqn(config: TianshouWorkerConfig, work_dir: Path, log_dir: Path):
    """Run DQN algorithm."""
    args = config.extras
    
    # Environment setup
    env = gym.make(config.env_id)
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError(f"DQN only supports discrete action spaces, but {config.env_id} has {env.action_space}")

    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n

    train_envs = SubprocVectorEnv(
        [make_env(config.env_id, config.seed + i, i, False, config.run_id) for i in range(args.get("num_envs", 1))]
    )
    test_envs = SubprocVectorEnv(
        [make_env(config.env_id, config.seed + 100 + i, i, False, config.run_id) for i in range(args.get("num_test_envs", 1))]
    )
    
    # Seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    train_envs.seed(config.seed)
    test_envs.seed(config.seed + 100)

    # Network
    device = args.get("device", "cpu")
    net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=args.get("hidden_sizes", [128, 128])).to(device)
    optim_factory = AdamOptimizerFactory(lr=args.get("lr", 1e-3))
    
    # Policy component
    policy = DiscreteQLearningPolicy(
        model=net,
        action_space=env.action_space,
        observation_space=env.observation_space,
        eps_training=args.get("eps_train", 0.1),
        eps_inference=args.get("eps_test", 0.05),
    )
    
    # Algorithm component
    algorithm = DQN(
        policy=policy,
        optim=optim_factory,
        gamma=args.get("gamma", 0.99),
        n_step_return_horizon=args.get("n_step", 3),
        target_update_freq=args.get("target_update_freq", 320),
        is_double=args.get("is_double", True),
    )

    # Collector
    train_collector = Collector(algorithm, train_envs, VectorReplayBuffer(args.get("buffer_size", 20000), len(train_envs)), exploration_noise=True)
    test_collector = Collector(algorithm, test_envs, exploration_noise=True)

    # Loggers
    from tianshou.utils import TensorboardLogger
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir)
    logger = TensorboardLogger(writer)

    # Pre-collect
    train_collector.collect(n_step=args.get("batch_size", 64) * args.get("training_num", 10))
    
    # Training
    result = algorithm.run_training(
        OffPolicyTrainerParams(
            training_collector=train_collector,
            test_collector=test_collector,
            max_epochs=args.get("epoch", 10),
            epoch_num_steps=args.get("step_per_epoch", 10000),
            collection_step_num_env_steps=args.get("step_per_collect", 10),
            test_step_num_episodes=args.get("test_num", 10),
            batch_size=args.get("batch_size", 64),
            update_step_num_gradient_steps_per_sample=args.get("update_per_step", 0.1),
            test_in_training=False,
            logger=logger,
        )
    )

    print(f"DQN training finished. Result: {result}")


ALGO_MAP: Dict[str, Callable[[TianshouWorkerConfig, Path, Path], None]] = {
    "ppo": run_ppo,
    "dqn": run_dqn,
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
            config = TianshouWorkerConfig.from_dict(config_dict)
    else:
        # Construct config from CLI args (fallback)
        config = TianshouWorkerConfig(
            run_id=args.run_id,
            algo=args.algo,
            env_id=args.env_id,
            total_timesteps=args.total_timesteps,
            seed=args.seed,
            worker_id=args.worker_id,
        )

    runner = ALGO_MAP.get(config.algo)
    if not runner:
        raise ValueError(f"Algorithm {config.algo} not supported by Tianshou worker yet.")
    
    work_dir = args.work_dir or Path(os.getcwd())
    log_dir = args.log_dir or work_dir / "log"
    
    runner(config, work_dir, log_dir)


if __name__ == "__main__":
    main()
