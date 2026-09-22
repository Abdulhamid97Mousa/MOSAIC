"""MAT-CNN on SocialJax coin_game -- CNN encoder + transformer (paper-faithful).

Same Multi-Agent Transformer as coop_mining MAT but for coin_game (2 agents,
obs_c=14 channels). CNN encoder replaces flat linear projection:

  flat Linear(1694->128)   ==>   Conv(32,5x5)->Conv(32,3x3)->Conv(32,3x3)
                                  ->flatten->Dense(128)

Config: 32 envs x 9375 updates x 1000 steps = 300M env-steps.

Run (GPU 1):
  CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \\
    python -m jaxmarl_worker.algorithms.socialjax.coins.mat_cnn_scan_socialjax
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import sys
import time
from pathlib import Path

import jax
import numpy as np

from jaxmarl_worker.networks.mat_actor import MATCNNActor
from jaxmarl_worker.algorithms.multigrid_sports.mat_scan import make_train
from jaxmarl_worker.environments.socialjax.coin_game import SocialJaxCoinGame


def _make_env(num_agents: int, num_inner_steps: int) -> SocialJaxCoinGame:
    return SocialJaxCoinGame(
        num_agents      = num_agents,
        num_inner_steps = num_inner_steps,
        shared_rewards  = True,
    )


def parse_args():
    _default_run_dir = str(
        Path(__file__).resolve().parents[7]
        / "var" / "trainer" / "socialjax" / "coins" / "MAT_CNN"
    )
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir",       type=str,   default=_default_run_dir)
    p.add_argument("--total-updates", type=int,   default=9375,
                   help="9375 * 32envs * 1000steps = 300M env-steps.")
    p.add_argument("--n-envs",        type=int,   default=32)
    p.add_argument("--n-steps",       type=int,   default=1000)
    p.add_argument("--n-epochs",      type=int,   default=2)
    p.add_argument("--n-minibatches", type=int,   default=16)
    p.add_argument("--n-embd",        type=int,   default=128)
    p.add_argument("--n-head",        type=int,   default=4)
    p.add_argument("--n-block",       type=int,   default=2)
    p.add_argument("--lr",            type=float, default=5e-4)
    p.add_argument("--gamma",         type=float, default=0.99)
    p.add_argument("--gae-lambda",    type=float, default=0.95)
    p.add_argument("--clip-eps",      type=float, default=0.05,
                   help="MAT paper: smaller clip eps (0.05 vs PPO 0.2)")
    p.add_argument("--vf-coef",       type=float, default=0.5)
    p.add_argument("--ent-coef",      type=float, default=0.01)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--training-type",   type=bool,  default=True,
                   help="Must be True - enforces training_reward parameter usage")
    p.add_argument("--training-reward", type=str,   default="cooperative-no-opponent",
                   choices=["cooperative-no-opponent"],
                   help="SocialJax environments are always cooperative (no-opponent)")
    p.add_argument("--num-agents",,    type=int,   default=2)
    p.add_argument("--num-inner-steps", type=int, default=1000)
    return p.parse_args()


def main():
    args     = parse_args()
    assert args.training_type == True, "training_type must be True"
    assert args.training_reward == "cooperative-no-opponent", f"SocialJax only supports cooperative-no-opponent, got {args.training_reward}"
    assert args.max_steps == 1000, f"SocialJax episode length must be exactly 1000 steps, got {args.max_steps}"
    env      = _make_env(args.num_agents, args.num_inner_steps)
    run_dir  = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    try:
        from torch.utils.tensorboard import SummaryWriter as _SW
        tb_writer = _SW(log_dir=str(run_dir / "tensorboard"))
        print(f"TensorBoard live at {run_dir / 'tensorboard'}")
    except ImportError:
        tb_writer = None
        print("torch not available — TensorBoard disabled")

    _lf = open(run_dir / "train.log", "w", buffering=1)
    class _Tee:
        def write(self, s): _lf.write(s); sys.__stdout__.write(s)
        def flush(self): _lf.flush(); sys.__stdout__.flush()
        def fileno(self): return sys.__stdout__.fileno()
    sys.stdout = _Tee()

    env_steps   = args.total_updates * args.n_envs * args.n_steps
    agent_steps = env_steps * env.num_agents
    print(f"[MAT-CNN-SocialJax] coin_game  n_agents={env.num_agents}  "
          f"obs_dim={env._obs_dim}  n_envs={args.n_envs}  total_updates={args.total_updates}  "
          f"(~{env_steps/1e6:.0f}M env-steps == {agent_steps/1e6:.0f}M agent-steps)")
    print(f"  arch: CNN(32,5x5)->CNN(32,3x3)->CNN(32,3x3)->Dense({args.n_embd}) + "
          f"transformer E{args.n_embd}H{args.n_head}B{args.n_block}")

    _total_updates = args.total_updates
    _n_steps = args.n_steps
    _step_size = args.n_envs * args.n_steps

    def _log_callback(step, ret, loss):
        print(f"  update {int(step):4d}/{_total_updates}  "
              f"ep_return={float(ret) * _n_steps:.1f}  loss={float(loss):.4f}", flush=True)
        if tb_writer is not None:
            env_step = int(step) * _step_size
            tb_writer.add_scalar("train/episode_return", float(ret) * _n_steps, env_step)
            tb_writer.flush()

    def _cnn_factory(action_dim, n_agent, n_embd, n_head, n_block):
        return MATCNNActor(action_dim=action_dim, n_agent=n_agent,
                           n_embd=n_embd, n_head=n_head, n_block=n_block,
                           obs_h=11, obs_w=11, obs_c=14)

    config = {
        "ENV":             env,
        "N_ENVS":          args.n_envs,
        "N_STEPS":         args.n_steps,
        "N_EPOCHS":        args.n_epochs,
        "N_MINIBATCHES":   args.n_minibatches,
        "N_EMBD":          args.n_embd,
        "N_HEAD":          args.n_head,
        "N_BLOCK":         args.n_block,
        "LR":              args.lr,
        "GAMMA":           args.gamma,
        "GAE_LAMBDA":      args.gae_lambda,
        "CLIP_EPS":        args.clip_eps,
        "VF_COEF":         args.vf_coef,
        "ENT_COEF":        args.ent_coef,
        "TOTAL_UPDATES":   args.total_updates,
        "NETWORK_FACTORY": _cnn_factory,
        "LOG_CALLBACK":    _log_callback,
        "ACTION_DIM":      7,
    }

    train_fn = make_train(config)
    print("[MAT-CNN-SocialJax] JIT-compiling...", flush=True)
    t0 = time.time()
    key = jax.random.PRNGKey(args.seed)
    runner_state, metrics = jax.block_until_ready(train_fn(key))
    if tb_writer is not None:
        tb_writer.close()
    t1 = time.time()
    print(f"[MAT-CNN-SocialJax] done in {t1-t0:.1f}s  ({env_steps/(t1-t0):,.0f} env-steps/sec)")

    ep_rets = np.array(metrics["mean_episode_return"]).ravel() * args.n_steps
    print(f"  final ep_ret (last 50): {ep_rets[-50:].mean():.2f}")
    np.save(str(run_dir / "metrics_ep_return.npy"), ep_rets)

    leaves, _ = jax.tree_util.tree_flatten(
        jax.tree_util.tree_map(np.array, runner_state.train_state.params)
    )
    final_path = ckpt_dir / "final.npz"
    np.savez(str(final_path), *leaves)
    print(f"[MAT-CNN-SocialJax] checkpoint -> {final_path}  ({len(leaves)} leaves)")


if __name__ == "__main__":
    main()
