"""TorchRL native benchmark -- PPO via TorchRL API, no MOSAIC wrapping.

pip install torchrl
"""

import sys
import time

from workers_benchmark.utils import (
    BenchmarkResult, run_subprocess_timed, print_run_header, print_run_result,
)


def run_native_benchmark(config) -> BenchmarkResult:
    """Run TorchRL PPO directly via subprocess."""
    print_run_header(config.worker_name, "native", config.env_id,
                     config.total_timesteps, config.num_envs, config.seed,
                     getattr(config, "_current_iteration", 1),
                     config.iterations)

    script = f"""\
import torch
torch.manual_seed({config.seed})

from torchrl.envs import GymEnv, TransformedEnv, Compose, StepCounter
from torchrl.collectors import Collector
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator, OneHotCategorical
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tensordict.nn import TensorDictModule

def make_env():
    env = GymEnv("{config.env_id}")
    env = TransformedEnv(env, Compose(StepCounter()))
    return env

env = make_env()
obs_size = env.observation_spec["observation"].shape[-1]

is_discrete = env.action_spec.dtype in (torch.int64, torch.int32, torch.long)

if is_discrete:
    n_actions = env.action_spec.space.n
    actor_net = MLP(in_features=obs_size, out_features=n_actions, num_cells=[64, 64])
    actor_module = TensorDictModule(actor_net, in_keys=["observation"], out_keys=["logits"])
    actor = ProbabilisticActor(
        module=actor_module,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=OneHotCategorical,
        return_log_prob=True,
    )
else:
    from torchrl.modules import TanhNormal
    act_size = env.action_spec.shape[-1]
    actor_net = MLP(in_features=obs_size, out_features=2 * act_size, num_cells=[64, 64])
    actor_module = TensorDictModule(actor_net, in_keys=["observation"], out_keys=["loc", "scale"])
    actor = ProbabilisticActor(
        module=actor_module,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        return_log_prob=True,
    )

critic_net = MLP(in_features=obs_size, out_features=1, num_cells=[64, 64])
critic = ValueOperator(module=critic_net, in_keys=["observation"])

collector = Collector(
    create_env_fn=make_env,
    policy=actor,
    frames_per_batch={config.num_steps},
    total_frames={config.total_timesteps},
)

advantage = GAE(gamma=0.99, lmbda=0.95, value_network=critic)
loss_fn = ClipPPOLoss(
    actor_network=actor,
    critic_network=critic,
    clip_epsilon=0.2,
    entropy_coeff=0.01,
    critic_coeff=0.5,
)
optim = torch.optim.Adam(loss_fn.parameters(), lr={config.learning_rate})

for batch in collector:
    advantage(batch)
    for _ in range(10):
        loss = loss_fn(batch)
        total_loss = loss["loss_objective"] + loss["loss_critic"] + loss["loss_entropy"]
        optim.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), 0.5)
        optim.step()

collector.shutdown()
env.close()
"""
    cmd = [sys.executable, "-c", script]
    elapsed, peak_mb, stdout, _ = run_subprocess_timed(cmd, timeout=1800)
    sps = config.total_timesteps / elapsed if elapsed > 0 else 0.0

    result = BenchmarkResult(
        worker_name="torchrl",
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
