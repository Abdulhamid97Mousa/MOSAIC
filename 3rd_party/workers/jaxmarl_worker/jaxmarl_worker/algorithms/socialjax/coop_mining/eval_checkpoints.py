"""Eval trained IPPO / MAPPO / MAT checkpoints on coop_mining.

All rollouts use jax.lax.scan + jax.vmap -- fully GPU-accelerated, same as training.
Runs N episodes in parallel and reports:
  - Mean / std episode return
  - Mine rate (fraction of agent-steps spent mining)
  - Gold reward fraction (gold events vs iron events)
  - Action distribution

Usage:
  cd jaxmarl_worker
  XLA_PYTHON_CLIENT_PREALLOCATE=false \\
  python -m jaxmarl_worker.algorithms.socialjax.coop_mining.eval_checkpoints \\
    --episodes 50

  # Single algorithm only
  python -m jaxmarl_worker.algorithms.socialjax.coop_mining.eval_checkpoints \\
    --algo IPPO --episodes 20
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import time
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

from jaxmarl_worker.environments.socialjax.coop_mining import SocialJaxCoopMining

_CKPT_ROOT = Path(__file__).resolve().parents[7] / "var" / "trainer" / "socialjax" / "coop_mining"

N_STEPS   = 1000   # episode length (matches training)
N_AGENTS  = 6
OBS_DIM   = 1452
ACTION_DIM = 8

ACTION_NAMES = ["turn_left", "turn_right", "step_left", "step_right",
                "forward",   "backward",   "stay",      "mine"]


# ---------------------------------------------------------------------------
# Network definitions (must match training scripts exactly)
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    action_dim: int = ACTION_DIM
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x):
        x = nn.tanh(nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)),
                             bias_init=constant(0.0))(x))
        x = nn.tanh(nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)),
                             bias_init=constant(0.0))(x))
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01),
                          bias_init=constant(0.0))(x)
        value  = nn.Dense(1, kernel_init=orthogonal(1.0),
                          bias_init=constant(0.0))(x)
        return logits, value.squeeze(-1)


class Actor(nn.Module):
    action_dim: int = ACTION_DIM
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x):
        x = nn.tanh(nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)),
                             bias_init=constant(0.0))(x))
        x = nn.tanh(nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)),
                             bias_init=constant(0.0))(x))
        return nn.Dense(self.action_dim, kernel_init=orthogonal(0.01),
                        bias_init=constant(0.0))(x)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load_params(npz_path: Path, network: nn.Module, dummy_input):
    dummy_params = network.init(jax.random.PRNGKey(0), dummy_input)
    _, treedef   = jax.tree_util.tree_flatten(dummy_params)
    ck     = np.load(str(npz_path))
    leaves = [ck[f"arr_{i}"] for i in range(len(ck.files))]
    return jax.tree_util.tree_unflatten(treedef, leaves)


def _load_mat_params(npz_path: Path, network, dummy_obs, dummy_sa):
    dummy_params = network.init(jax.random.PRNGKey(0), dummy_obs, dummy_sa)
    _, treedef   = jax.tree_util.tree_flatten(dummy_params)
    ck     = np.load(str(npz_path))
    leaves = [ck[f"arr_{i}"] for i in range(len(ck.files))]
    return jax.tree_util.tree_unflatten(treedef, leaves)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def _make_env() -> SocialJaxCoopMining:
    return SocialJaxCoopMining(
        num_agents         = N_AGENTS,
        num_inner_steps    = N_STEPS,
        shared_rewards     = True,
        regrowth_prob_iron = 0.0004,
        regrowth_prob_gold = 0.00016,
        reward_iron        = 1.0,
        reward_gold        = 8.0,
        gold_mining_window = 3,
        min_gold_miners    = 2,
        max_miners         = 4,
    )


# ---------------------------------------------------------------------------
# GPU-accelerated rollout via lax.scan
# ---------------------------------------------------------------------------

def _make_ippo_rollout(env, net, params):
    agents = [f"agent_{i}" for i in range(N_AGENTS)]

    @jax.jit
    def run_episode(key):
        key, rk = jax.random.split(key)
        obs_dict, state = env.reset(rk)

        def step_fn(carry, _):
            obs_dict, state, key = carry
            obs_flat = jnp.stack([obs_dict[a] for a in agents])   # (N, obs_dim)
            logits, _ = net.apply(params, obs_flat)                 # (N, 8)
            actions_arr = jnp.argmax(logits, axis=-1)               # greedy
            actions = {a: actions_arr[i] for i, a in enumerate(agents)}
            key, sk = jax.random.split(key)
            obs_dict, state, rewards, dones, _ = env.step(sk, state, actions)
            step_reward = sum(rewards.values()) / N_AGENTS
            return (obs_dict, state, key), (step_reward, actions_arr)

        (_, _, _), (rewards_t, actions_t) = jax.lax.scan(
            step_fn, (obs_dict, state, key), None, length=N_STEPS
        )
        # rewards_t: (T,)   actions_t: (T, N)
        ep_return      = rewards_t.sum()
        action_counts  = jnp.zeros(ACTION_DIM).at[actions_t.reshape(-1)].add(1)
        mine_steps     = (actions_t == 7).sum()
        # gold heuristic: step reward > 4/N_AGENTS means a gold event fired
        gold_mask      = rewards_t > (4.0 / N_AGENTS)
        gold_rewards   = (rewards_t * gold_mask).sum()
        iron_rewards   = (rewards_t * ~gold_mask).sum()
        return ep_return, mine_steps, gold_rewards, iron_rewards, action_counts

    return jax.vmap(run_episode)


def _make_mappo_rollout(env, net, actor_params):
    agents = [f"agent_{i}" for i in range(N_AGENTS)]

    @jax.jit
    def run_episode(key):
        key, rk = jax.random.split(key)
        obs_dict, state = env.reset(rk)

        def step_fn(carry, _):
            obs_dict, state, key = carry
            obs_flat    = jnp.stack([obs_dict[a] for a in agents])
            logits      = net.apply(actor_params, obs_flat)
            actions_arr = jnp.argmax(logits, axis=-1)
            actions     = {a: actions_arr[i] for i, a in enumerate(agents)}
            key, sk     = jax.random.split(key)
            obs_dict, state, rewards, dones, _ = env.step(sk, state, actions)
            step_reward = sum(rewards.values()) / N_AGENTS
            return (obs_dict, state, key), (step_reward, actions_arr)

        (_, _, _), (rewards_t, actions_t) = jax.lax.scan(
            step_fn, (obs_dict, state, key), None, length=N_STEPS
        )
        ep_return     = rewards_t.sum()
        action_counts = jnp.zeros(ACTION_DIM).at[actions_t.reshape(-1)].add(1)
        mine_steps    = (actions_t == 7).sum()
        gold_mask     = rewards_t > (4.0 / N_AGENTS)
        gold_rewards  = (rewards_t * gold_mask).sum()
        iron_rewards  = (rewards_t * ~gold_mask).sum()
        return ep_return, mine_steps, gold_rewards, iron_rewards, action_counts

    return jax.vmap(run_episode)


def _make_mat_rollout(env, net, params):
    from jaxmarl_worker.networks.mat_actor import mat_get_actions
    agents = [f"agent_{i}" for i in range(N_AGENTS)]

    @jax.jit
    def run_episode(key):
        key, rk = jax.random.split(key)
        obs_dict, state = env.reset(rk)

        def step_fn(carry, _):
            obs_dict, state, key = carry
            obs_flat = jnp.stack([obs_dict[a] for a in agents])  # (N, obs_dim)
            obs_b    = obs_flat[jnp.newaxis]                      # (1, N, obs_dim)
            key, ak  = jax.random.split(key)
            actions_b, _, _ = mat_get_actions(
                net.apply, params, obs_b, ak, ACTION_DIM, deterministic=True
            )
            actions_arr = actions_b.squeeze(0)                    # (N,)
            actions     = {a: actions_arr[i] for i, a in enumerate(agents)}
            key, sk     = jax.random.split(key)
            obs_dict, state, rewards, dones, _ = env.step(sk, state, actions)
            step_reward = sum(rewards.values()) / N_AGENTS
            return (obs_dict, state, key), (step_reward, actions_arr)

        (_, _, _), (rewards_t, actions_t) = jax.lax.scan(
            step_fn, (obs_dict, state, key), None, length=N_STEPS
        )
        ep_return     = rewards_t.sum()
        action_counts = jnp.zeros(ACTION_DIM).at[actions_t.reshape(-1)].add(1)
        mine_steps    = (actions_t == 7).sum()
        gold_mask     = rewards_t > (4.0 / N_AGENTS)
        gold_rewards  = (rewards_t * gold_mask).sum()
        iron_rewards  = (rewards_t * ~gold_mask).sum()
        return ep_return, mine_steps, gold_rewards, iron_rewards, action_counts

    return jax.vmap(run_episode)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_report(algo, ep_returns, mine_steps, gold_r, iron_r, action_counts):
    n = len(ep_returns)
    action_pct = 100.0 * action_counts / action_counts.sum()
    gold_frac  = gold_r.sum() / (gold_r.sum() + iron_r.sum() + 1e-8)
    mine_rate  = mine_steps.sum() / (n * N_STEPS * N_AGENTS)

    print(f"\n{'='*60}")
    print(f"  {algo}  ({n} episodes)")
    print(f"{'='*60}")
    print(f"  Episode return   : {ep_returns.mean():7.2f}  (+/- {ep_returns.std():.2f})")
    print(f"  Gold reward frac : {gold_frac*100:6.1f}%  (higher = more gold cooperation)")
    print(f"  Mine rate        : {mine_rate*100:6.1f}%  (% of agent-steps spent mining)")
    print()
    print("  Action distribution:")
    for i, name in enumerate(ACTION_NAMES):
        bar = "#" * int(action_pct[i] / 2)
        print(f"    {name:<12} {action_pct[i]:5.1f}%  {bar}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--algo",     type=str, default="all",
                   choices=["all", "IPPO", "MAPPO", "MAT"])
    p.add_argument("--seed",     type=int, default=0)
    args = p.parse_args()

    env  = _make_env()
    run_algos = ["IPPO", "MAPPO", "MAT"] if args.algo == "all" else [args.algo]

    # episode keys: one key per episode, vmap runs them all in parallel
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.episodes)

    for algo in run_algos:
        ckpt_dir = _CKPT_ROOT / algo / "checkpoints"

        print(f"\n[{algo}] Loading checkpoint and JIT-compiling...", flush=True)
        t0 = time.time()

        if algo == "IPPO":
            net    = ActorCritic()
            params = _load_params(ckpt_dir / "final.npz", net,
                                  jnp.zeros((N_AGENTS, OBS_DIM)))
            rollout_fn = _make_ippo_rollout(env, net, params)

        elif algo == "MAPPO":
            net    = Actor()
            params = _load_params(ckpt_dir / "actor_final.npz", net,
                                  jnp.zeros((N_AGENTS, OBS_DIM)))
            rollout_fn = _make_mappo_rollout(env, net, params)

        elif algo == "MAT":
            from jaxmarl_worker.networks.mat_actor import MATActor
            net    = MATActor(action_dim=ACTION_DIM, n_agent=N_AGENTS,
                              n_embd=128, n_head=4, n_block=2)
            params = _load_mat_params(
                ckpt_dir / "final.npz", net,
                jnp.zeros((1, N_AGENTS, OBS_DIM)),
                jnp.zeros((1, N_AGENTS, ACTION_DIM + 1)),
            )
            rollout_fn = _make_mat_rollout(env, net, params)

        # First call triggers JIT compilation
        ep_returns, mine_steps, gold_r, iron_r, action_counts = jax.block_until_ready(
            rollout_fn(keys)
        )
        t1 = time.time()
        print(f"[{algo}] Done in {t1-t0:.1f}s  ({args.episodes} episodes)")

        ep_returns    = np.array(ep_returns)
        mine_steps    = np.array(mine_steps)
        gold_r        = np.array(gold_r)
        iron_r        = np.array(iron_r)
        action_counts = np.array(action_counts).sum(axis=0)  # sum over episodes

        _print_report(algo, ep_returns, mine_steps, gold_r, iron_r, action_counts)

    print("\nDone.")


if __name__ == "__main__":
    main()
