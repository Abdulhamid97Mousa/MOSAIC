"""IPPO on SocialJax CoopMining (Anakin: jax.lax.scan + jax.vmap).

Reuses the generic make_train from ippo_scan.py; this file only provides
the SocialJax env factory and a CLI with coop_mining-appropriate defaults.

Hyperparameter defaults match the SocialJax IPPO paper config
adapted to 6 agents (paper uses 4; we match MAPPO's 6-agent setup for the six-agent setup).

Usage (smoke -- ~5 min):
  XLA_PYTHON_CLIENT_PREALLOCATE=false \\
  python -m jaxmarl_worker.algorithms.ippo_scan_socialjax \\
    --run-dir var/trainer/socialjax/coop_mining/IPPO --total-updates 50 --n-envs 8

Usage (full -- ~300M steps):
  XLA_PYTHON_CLIENT_PREALLOCATE=false \\
  python -m jaxmarl_worker.algorithms.ippo_scan_socialjax \\
    --run-dir var/trainer/socialjax/coop_mining/IPPO --total-updates 1562 --n-envs 32
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

import argparse
import time
from pathlib import Path

import jax
import numpy as np

from jaxmarl_worker.algorithms.multigrid_sports.ippo_scan import make_train
from jaxmarl_worker.environments.socialjax.coop_mining import SocialJaxCoopMining


def _make_env(num_agents: int, num_inner_steps: int) -> SocialJaxCoopMining:
    return SocialJaxCoopMining(
        num_agents         = num_agents,
        num_inner_steps    = num_inner_steps,
        shared_rewards     = True,
        regrowth_prob_iron = 0.0004,
        regrowth_prob_gold = 0.00016,
        reward_iron        = 1.0,
        reward_gold        = 8.0,
        gold_mining_window = 3,
        min_gold_miners    = 2,
        max_miners         = 4,
    )


def parse_args():
    p = argparse.ArgumentParser()
    _default_run_dir = str(Path(__file__).resolve().parents[7] / "var" / "trainer" / "socialjax" / "coop_mining" / "IPPO")
    p.add_argument("--run-dir",       type=str,   default=_default_run_dir)
    p.add_argument("--total-updates", type=int,   default=195,
                   help="Scan iterations. 195 * 256envs * 1000steps * 6agents ~= 300M steps")
    p.add_argument("--n-envs",        type=int,   default=256,
                   help="SocialJax paper IPPO base config: 256 parallel envs")
    p.add_argument("--n-steps",       type=int,   default=1000,
                   help="Must match num_inner_steps (episode length)")
    p.add_argument("--n-epochs",      type=int,   default=2)
    p.add_argument("--n-minibatches", type=int,   default=500,
                   help="SocialJax paper IPPO base config: 500 minibatches")
    p.add_argument("--hidden-dim",    type=int,   default=256)
    p.add_argument("--lr",            type=float, default=5e-4)
    p.add_argument("--gamma",         type=float, default=0.99)
    p.add_argument("--gae-lambda",    type=float, default=0.95)
    p.add_argument("--clip-eps",      type=float, default=0.2)
    p.add_argument("--vf-coef",       type=float, default=0.5)
    p.add_argument("--ent-coef",      type=float, default=0.01)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--training-type",   type=bool,  default=True,
                   help="Must be True - enforces training_reward parameter usage")
    p.add_argument("--training-reward", type=str,   default="cooperative-no-opponent",
                   choices=["cooperative-no-opponent"],
                   help="SocialJax environments are always cooperative (no-opponent)")
    p.add_argument("--num-agents",,    type=int,   default=6)
    p.add_argument("--num-inner-steps", type=int, default=1000)
    return p.parse_args()


def main():
    args   = parse_args()
    assert args.training_type == True, "training_type must be True"
    assert args.training_reward == "cooperative-no-opponent", f"SocialJax only supports cooperative-no-opponent, got {args.training_reward}"
    assert args.max_steps == 1000, f"SocialJax episode length must be exactly 1000 steps, got {args.max_steps}"
    env    = _make_env(args.num_agents, args.num_inner_steps)
    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    total_steps = args.total_updates * args.n_envs * args.n_steps * env.num_agents
    print(
        f"[IPPO-SocialJax] coop_mining  n_agents={env.num_agents}  "
        f"obs_dim={env._obs_dim}  n_envs={args.n_envs}  "
        f"n_steps={args.n_steps}  total_updates={args.total_updates}  "
        f"(~{total_steps/1e6:.0f}M agent-steps)"
    )

    config = {
        "ENV":            env,
        "N_ENVS":         args.n_envs,
        "N_STEPS":        args.n_steps,
        "N_EPOCHS":       args.n_epochs,
        "N_MINIBATCHES":  args.n_minibatches,
        "HIDDEN_DIM":     args.hidden_dim,
        "LR":             args.lr,
        "GAMMA":          args.gamma,
        "GAE_LAMBDA":     args.gae_lambda,
        "CLIP_EPS":       args.clip_eps,
        "VF_COEF":        args.vf_coef,
        "ENT_COEF":       args.ent_coef,
        "TOTAL_UPDATES":  args.total_updates,
    }

    train_fn = make_train(config)
    print("[IPPO-SocialJax] JIT-compiling...", flush=True)
    t0 = time.time()
    key = jax.random.PRNGKey(args.seed)
    runner_state, metrics = jax.block_until_ready(train_fn(key))
    t1 = time.time()
    print(f"[IPPO-SocialJax] done in {t1-t0:.1f}s  ({total_steps/(t1-t0):,.0f} steps/sec)")

    ep_rets = np.array(metrics["mean_episode_return"])
    print(f"  final ep_ret (last 50): {ep_rets[-50:].mean():.4f}")

    leaves, _ = jax.tree_util.tree_flatten(
        jax.tree_util.tree_map(np.array, runner_state.train_state.params)
    )
    final_path = ckpt_dir / "final.npz"
    np.savez(str(final_path), *leaves)
    print(f"[IPPO-SocialJax] checkpoint -> {final_path}")


if __name__ == "__main__":
    main()
