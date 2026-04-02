"""Tianshou native benchmark -- PPO directly via Tianshou 2.0 API, no MOSAIC wrapping.

Baseline: uses the actual Tianshou 2.0 API (no device param on Net,
DiscreteActor/DiscreteCritic instead of Actor/Critic).
"""

import sys
import time

from workers_benchmark.utils import (
    BenchmarkResult, run_subprocess_timed, print_run_header, print_run_result,
)


def run_native_benchmark(config) -> BenchmarkResult:
    """Run Tianshou PPO directly via subprocess (no MOSAIC worker wrapping)."""
    print_run_header(config.worker_name, "native", config.env_id,
                     config.total_timesteps, config.num_envs, config.seed,
                     getattr(config, "_current_iteration", 1),
                     config.iterations)

    script = f"""\
import os, gymnasium as gym, numpy as np, torch
os.environ["WANDB_MODE"] = "disabled"

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.algorithm.modelfree.ppo import PPO
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.trainer import OnPolicyTrainerParams
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import DiscreteActor, DiscreteCritic

env = gym.make("{config.env_id}")
state_shape = env.observation_space.shape
action_shape = env.action_space.n
seed = {config.seed}
num_envs = {config.num_envs}

def make_env(idx):
    def thunk():
        e = gym.make("{config.env_id}")
        e = gym.wrappers.RecordEpisodeStatistics(e)
        e.action_space.seed(seed + idx)
        e.observation_space.seed(seed + idx)
        return e
    return thunk

train_envs = SubprocVectorEnv([make_env(i) for i in range(num_envs)])
test_envs = SubprocVectorEnv([make_env(100)])

np.random.seed(seed)
torch.manual_seed(seed)
train_envs.seed(seed)
test_envs.seed(seed + 100)

net = Net(state_shape=state_shape, hidden_sizes=[64, 64])
actor_net = DiscreteActor(preprocess_net=net, action_shape=action_shape)
critic_net = DiscreteCritic(preprocess_net=Net(state_shape=state_shape, hidden_sizes=[64, 64]))

policy = ProbabilisticActorPolicy(
    actor=actor_net,
    dist_fn=lambda x: torch.distributions.Categorical(logits=x),
    action_space=env.action_space,
    deterministic_eval=True,
    action_scaling=False,
)

algorithm = PPO(
    policy=policy,
    critic=critic_net,
    optim=AdamOptimizerFactory(lr={config.learning_rate}),
    eps_clip=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
    max_grad_norm=0.5,
    gae_lambda=0.95,
    max_batchsize=256,
    return_scaling=False,
)

train_collector = Collector(algorithm, train_envs, VectorReplayBuffer(20000, num_envs))
test_collector = Collector(algorithm, test_envs)

algorithm.run_training(OnPolicyTrainerParams(
    training_collector=train_collector,
    test_collector=test_collector,
    max_epochs=1,
    epoch_num_steps={config.total_timesteps},
    collection_step_num_env_steps=2048,
    test_step_num_episodes=5,
    batch_size=64,
    update_step_num_repetitions=10,
))

env.close()
train_envs.close()
test_envs.close()
"""
    cmd = [sys.executable, "-c", script]
    elapsed, peak_mb, stdout, _ = run_subprocess_timed(cmd, timeout=1800)
    sps = config.total_timesteps / elapsed if elapsed > 0 else 0.0

    result = BenchmarkResult(
        worker_name="tianshou",
        scenario="native",
        env_id=config.env_id,
        total_timesteps=config.total_timesteps,
        wall_time_seconds=elapsed,
        steps_per_second=sps,
        peak_memory_mb=peak_mb,
        seed=config.seed,
        num_envs=config.num_envs,
        iteration=getattr(config, "_current_iteration", 1),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    print_run_result(result)
    return result
