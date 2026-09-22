"""MAT on SocialJax CoopMining (Anakin: jax.lax.scan + jax.vmap).

Reuses the generic make_train from mat_scan.py (MATActor with autoregressive
action sampling). This file provides the SocialJax env factory and a CLI
with coop_mining-appropriate defaults.

Obs: flat 1452-dim symbolic grid (11*11*12). MAT sees this as a flat token
sequence -- no CNN needed. The transformer learns spatial structure from
position implicitly via self-attention.

Architecture: n_embd=128, n_head=4, n_block=2 (E128H4B2).
  Wider embed (128 vs 64) handles the 1452-dim obs without bottleneck.
  n_block=2 over n_block=3: 2 wider blocks beat 3 narrow ones.

Gradient updates: n_minibatches=16 gives ~50k gradient steps at 300M env-steps,
  matching MAPPO and avoiding the degenerate policy caused by only 4 minibatches.

Usage (smoke):
  XLA_PYTHON_CLIENT_PREALLOCATE=false \\
  python -m jaxmarl_worker.algorithms.socialjax.coop_mining.mat_scan \\
    --run-dir var/trainer/socialjax/coop_mining/MAT --total-updates 50 --n-envs 8

Usage (full -- ~300M steps):
  XLA_PYTHON_CLIENT_PREALLOCATE=false \\
  python -m jaxmarl_worker.algorithms.socialjax.coop_mining.mat_scan \\
    --run-dir var/trainer/socialjax/coop_mining/MAT
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

from jaxmarl_worker.algorithms.multigrid_sports.mat_scan import make_train
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
    _default_run_dir = str(Path(__file__).resolve().parents[7] / "var" / "trainer" / "socialjax" / "coop_mining" / "MAT")
    p.add_argument("--run-dir",       type=str,   default=_default_run_dir)
    p.add_argument("--total-updates", type=int,   default=1562,
                   help="Scan iterations. 1562 * 32envs * 1000steps * 6agents ~= 300M steps")
    p.add_argument("--n-envs",        type=int,   default=32)
    p.add_argument("--n-steps",       type=int,   default=1000,
                   help="Must equal episode length (1000 for coop_mining)")
    p.add_argument("--n-epochs",      type=int,   default=2)
    p.add_argument("--n-minibatches", type=int,   default=16,
                   help="16 minibatches -> ~50k grad steps at 300M env-steps (matches MAPPO)")
    p.add_argument("--n-embd",        type=int,   default=128,
                   help="E128H4B2: wider embed handles 1452-dim obs without bottleneck")
    p.add_argument("--n-head",        type=int,   default=4)
    p.add_argument("--n-block",       type=int,   default=2,
                   help="E128H4B2: 2 wider blocks beat 3 narrow 64-dim blocks")
    p.add_argument("--lr",            type=float, default=5e-4)
    p.add_argument("--gamma",         type=float, default=0.99)
    p.add_argument("--gae-lambda",    type=float, default=0.95)
    p.add_argument("--clip-eps",      type=float, default=0.05,
                   help="MAT paper uses smaller clip eps (0.05 vs PPO 0.2)")
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
    args    = parse_args()
    assert args.training_type == True, "training_type must be True"
    assert args.training_reward == "cooperative-no-opponent", f"SocialJax only supports cooperative-no-opponent, got {args.training_reward}"
    assert args.max_steps == 1000, f"SocialJax episode length must be exactly 1000 steps, got {args.max_steps}"
    env     = _make_env(args.num_agents, args.num_inner_steps)
    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    total_steps = args.total_updates * args.n_envs * args.n_steps * env.num_agents
    print(
        f"[MAT-SocialJax] coop_mining  n_agents={env.num_agents}  "
        f"obs_dim={env._obs_dim}  n_envs={args.n_envs}  "
        f"n_steps={args.n_steps}  total_updates={args.total_updates}  "
        f"(~{total_steps/1e6:.0f}M agent-steps)"
    )
    print(
        f"  MAT arch: n_embd={args.n_embd}  n_head={args.n_head}  n_block={args.n_block}"
    )

    config = {
        "ENV":            env,
        "N_ENVS":         args.n_envs,
        "N_STEPS":        args.n_steps,
        "N_EPOCHS":       args.n_epochs,
        "N_MINIBATCHES":  args.n_minibatches,
        "N_EMBD":         args.n_embd,
        "N_HEAD":         args.n_head,
        "N_BLOCK":        args.n_block,
        "LR":             args.lr,
        "GAMMA":          args.gamma,
        "GAE_LAMBDA":     args.gae_lambda,
        "CLIP_EPS":       args.clip_eps,
        "VF_COEF":        args.vf_coef,
        "ENT_COEF":       args.ent_coef,
        "TOTAL_UPDATES":  args.total_updates,
    }

    train_fn = make_train(config)
    print("[MAT-SocialJax] JIT-compiling...", flush=True)
    t0 = time.time()
    key = jax.random.PRNGKey(args.seed)
    runner_state, metrics = jax.block_until_ready(train_fn(key))
    t1 = time.time()
    print(f"[MAT-SocialJax] done in {t1-t0:.1f}s  ({total_steps/(t1-t0):,.0f} steps/sec)")

    ep_rets = np.array(metrics["mean_episode_return"])
    print(f"  final ep_ret (last 50): {ep_rets[-50:].mean():.4f}")

    leaves, _ = jax.tree_util.tree_flatten(
        jax.tree_util.tree_map(np.array, runner_state.train_state.params)
    )
    final_path = ckpt_dir / "final.npz"
    np.savez(str(final_path), *leaves)
    print(f"[MAT-SocialJax] checkpoint -> {final_path}")


if __name__ == "__main__":
    main()
