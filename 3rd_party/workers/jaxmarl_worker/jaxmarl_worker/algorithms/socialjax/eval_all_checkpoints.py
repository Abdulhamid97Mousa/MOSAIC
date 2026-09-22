"""Evaluate IPPO-CNN / MAPPO-CNN / MAT-CNN vs RANDOM baseline on all 3 SocialJax envs.

All algorithms use greedy (deterministic) policy. RANDOM uses uniform sampling.
Runs N episodes in parallel via jax.vmap for speed.

Usage:
  cd /home/hamid/projects/2x6000/mosaic
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 \\
    .venv/bin/python -m jaxmarl_worker.algorithms.socialjax.eval_all_checkpoints \\
    --episodes 100

  # Single env:
    --env coop_mining   (or coins, cleanup)
  # Single algo:
    --algo IPPO         (or MAPPO, MAT, RANDOM)
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

from jaxmarl_worker.environments.socialjax.coop_mining import SocialJaxCoopMining
from jaxmarl_worker.environments.socialjax.coin_game import SocialJaxCoinGame as SocialJaxCoins
from jaxmarl_worker.environments.socialjax.cleanup  import SocialJaxCleanup

_CKPT_ROOT = Path(__file__).resolve().parents[6] / "var" / "trainer" / "socialjax"


# ---------------------------------------------------------------------------
# CNN networks (match training exactly)
# ---------------------------------------------------------------------------

class _CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv(32, (5, 5), kernel_init=orthogonal(np.sqrt(2)),
                            bias_init=constant(0.0))(x))
        x = nn.relu(nn.Conv(32, (3, 3), kernel_init=orthogonal(np.sqrt(2)),
                            bias_init=constant(0.0))(x))
        x = nn.relu(nn.Conv(32, (3, 3), kernel_init=orthogonal(np.sqrt(2)),
                            bias_init=constant(0.0))(x))
        return nn.relu(nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)),
                                bias_init=constant(0.0))(x.reshape((x.shape[0], -1))))


class _IPPONet(nn.Module):
    """Full ActorCritic (must match training to load checkpoint); eval uses only logits."""
    action_dim: int

    @nn.compact
    def __call__(self, x):
        emb    = _CNN()(x)
        actor  = nn.relu(nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)),
                                  bias_init=constant(0.0))(emb))
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01),
                          bias_init=constant(0.0))(actor)
        critic = nn.relu(nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)),
                                  bias_init=constant(0.0))(emb))
        _value = nn.Dense(1, kernel_init=orthogonal(1.0),
                          bias_init=constant(0.0))(critic)
        return logits  # only logits used at eval time


class _MAPPOActor(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        emb    = _CNN()(x)
        hidden = nn.relu(nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)),
                                  bias_init=constant(0.0))(emb))
        return nn.Dense(self.action_dim, kernel_init=orthogonal(0.01),
                        bias_init=constant(0.0))(hidden)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load(path: Path, net: nn.Module, dummy):
    _, treedef = jax.tree_util.tree_flatten(net.init(jax.random.PRNGKey(0), dummy))
    ck = np.load(str(path))
    return jax.tree_util.tree_unflatten(treedef, [ck[f"arr_{i}"] for i in range(len(ck.files))])


def _load_mat(path: Path, net, dummy_obs, dummy_sa):
    _, treedef = jax.tree_util.tree_flatten(net.init(jax.random.PRNGKey(0), dummy_obs, dummy_sa))
    ck = np.load(str(path))
    return jax.tree_util.tree_unflatten(treedef, [ck[f"arr_{i}"] for i in range(len(ck.files))])


def _detect_mat_action_dim(path: Path) -> int:
    """Infer action_dim from a MAT checkpoint by finding the logits kernel shape (n_embd, action_dim)."""
    ck = np.load(str(path))
    for f in ck.files:
        s = ck[f].shape
        # logits kernel: 2D, first dim is n_embd (large), second dim is action_dim (small, > 1)
        if len(s) == 2 and s[0] >= 64 and 1 < s[1] <= 32:
            return int(s[1])
    raise ValueError(f"Cannot auto-detect action_dim from MAT checkpoint: {path}")


# ---------------------------------------------------------------------------
# Rollout builders
# ---------------------------------------------------------------------------

def _make_rollout(env, algo, net, params, cfg, stochastic=False):
    n_agents   = cfg["n_agents"]
    n_actions  = cfg["n_actions"]
    n_steps    = cfg["n_steps"]
    obs_h, obs_w, obs_c = cfg["obs_h"], cfg["obs_w"], cfg["obs_c"]
    agents     = [f"agent_{i}" for i in range(n_agents)]
    # MAT checkpoints may have been trained with a different action_dim than the env
    mat_act_dim = cfg.get("mat_action_dim", n_actions)

    mat_get_actions_fn = None
    if algo == "MAT":
        from jaxmarl_worker.networks.mat_actor import mat_get_actions
        mat_get_actions_fn = mat_get_actions

    # Pass params as explicit argument to avoid JAX JIT closure-caching issues
    # (same __code__ across calls to _make_rollout would share the cache key)
    def run_episode(key, p):
        key, rk = jax.random.split(key)
        obs_dict, state = env.reset(rk)

        def step_fn(carry, _):
            obs_dict, state, key = carry
            obs_flat    = jnp.stack([obs_dict[a] for a in agents])
            obs_spatial = obs_flat.reshape(n_agents, obs_h, obs_w, obs_c)

            if algo == "RANDOM":
                key, ak = jax.random.split(key)
                actions_arr = jax.random.randint(ak, (n_agents,), 0, n_actions)
            elif algo == "MAT":
                obs_b = obs_flat[jnp.newaxis]
                key, ak = jax.random.split(key)
                actions_b, _, _ = mat_get_actions_fn(
                    net.apply, p, obs_b, ak, mat_act_dim,
                    deterministic=not stochastic,
                )
                raw = actions_b.squeeze(0)
                # If checkpoint was trained with more actions than env has, clip to valid range
                actions_arr = jnp.clip(raw, 0, n_actions - 1)
            else:
                logits = net.apply(p, obs_spatial)
                if stochastic:
                    key, ak = jax.random.split(key)
                    # sample each agent's action independently from softmax distribution
                    actions_arr = jax.vmap(
                        lambda l, k: jax.random.categorical(k, l)
                    )(logits, jax.random.split(ak, n_agents))
                else:
                    actions_arr = jnp.argmax(logits, axis=-1)

            actions = {a: actions_arr[i] for i, a in enumerate(agents)}
            key, sk = jax.random.split(key)
            obs_dict, state, rewards, dones, _ = env.step(sk, state, actions)
            step_r = sum(rewards.values()) / n_agents
            return (obs_dict, state, key), (step_r, actions_arr)

        (_, _, _), (rewards_t, actions_t) = jax.lax.scan(
            step_fn, (obs_dict, state, key), None, length=n_steps
        )
        ep_return     = rewards_t.sum()
        action_counts = jnp.zeros(n_actions).at[actions_t.reshape(-1)].add(1)
        return ep_return, action_counts

    # vmap over keys (axis 0), broadcast params (axis None)
    vmapped = jax.vmap(run_episode, in_axes=(0, None))
    return lambda keys: jax.jit(vmapped)(keys, params)


# ---------------------------------------------------------------------------
# Environment / checkpoint config per env
# ---------------------------------------------------------------------------

def _build_env_cfg():
    return {
        "coop_mining": {
            "env": SocialJaxCoopMining(num_agents=4, num_inner_steps=1000, shared_rewards=True,
                                       regrowth_prob_iron=0.0004, regrowth_prob_gold=0.00016,
                                       reward_iron=1.0, reward_gold=8.0,
                                       gold_mining_window=3, min_gold_miners=2, max_miners=4),
            "n_agents": 4, "obs_h": 11, "obs_w": 11, "obs_c": 12,
            "n_actions": 8, "n_steps": 1000,
            "paper_target": 726.0,
            "ckpts": {
                "IPPO":  _CKPT_ROOT / "coop_mining/IPPO_CNN/checkpoints/final.npz",
                "MAPPO": _CKPT_ROOT / "coop_mining/MAPPO_CNN/checkpoints/actor_final.npz",
                "MAT":   _CKPT_ROOT / "coop_mining/MAT_CNN/checkpoints/final.npz",
            },
        },
        "coins": {
            "env": SocialJaxCoins(num_agents=2, num_inner_steps=1000, shared_rewards=True),
            "n_agents": 2, "obs_h": 11, "obs_w": 11, "obs_c": 14,
            "n_actions": 7, "n_steps": 1000,
            "paper_target": 164.0,
            "ckpts": {
                "IPPO":  _CKPT_ROOT / "coins/IPPO_CNN/checkpoints/final.npz",
                "MAPPO": _CKPT_ROOT / "coins/MAPPO_CNN/checkpoints/actor_final.npz",
                "MAT":   _CKPT_ROOT / "coins/MAT_CNN/checkpoints/final.npz",
            },
        },
        "cleanup": {
            "env": SocialJaxCleanup(num_agents=7, num_inner_steps=1000, shared_rewards=True),
            "n_agents": 7, "obs_h": 11, "obs_w": 11, "obs_c": 19,
            "n_actions": 9, "n_steps": 1000,
            "paper_target": 970.0,
            "ckpts": {
                "IPPO":  _CKPT_ROOT / "cleanup/IPPO_CNN/checkpoints/final.npz",
                "MAPPO": _CKPT_ROOT / "cleanup/MAPPO_CNN/checkpoints/actor_final.npz",
                "MAT":   _CKPT_ROOT / "cleanup/MAT_CNN/checkpoints/final.npz",
            },
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes",   type=int,  default=50)
    p.add_argument("--env",        type=str,  default="all",
                   choices=["all", "coop_mining", "coins", "cleanup"])
    p.add_argument("--algo",       type=str,  default="all",
                   choices=["all", "IPPO", "MAPPO", "MAT", "RANDOM"])
    p.add_argument("--seed",       type=int,  default=0)
    p.add_argument("--stochastic", action="store_true",
                   help="Sample from softmax instead of greedy argmax (matches paper eval)")
    args = p.parse_args()

    env_cfgs   = _build_env_cfg()
    run_envs   = list(env_cfgs.keys()) if args.env == "all" else [args.env]
    run_algos  = ["IPPO", "MAPPO", "MAT", "RANDOM"] if args.algo == "all" else [args.algo]
    keys       = jax.random.split(jax.random.PRNGKey(args.seed), args.episodes)

    policy_mode = "stochastic" if args.stochastic else "greedy"
    results = {}

    for env_name in run_envs:
        cfg = env_cfgs[env_name]
        env = cfg["env"]
        results[env_name] = {}

        print(f"\n{'#'*65}")
        print(f"  ENV: {env_name}   n_agents={cfg['n_agents']}  "
              f"obs=({cfg['obs_h']},{cfg['obs_w']},{cfg['obs_c']})  "
              f"n_actions={cfg['n_actions']}  paper_target={cfg['paper_target']:.0f}  "
              f"policy={policy_mode}")
        print(f"{'#'*65}")

        for algo in run_algos:
            print(f"\n  [{algo}] ", end="", flush=True)
            t0 = time.time()

            if algo == "RANDOM":
                net = params = None
            elif algo == "IPPO":
                net    = _IPPONet(action_dim=cfg["n_actions"])
                dummy  = jnp.zeros((cfg["n_agents"], cfg["obs_h"], cfg["obs_w"], cfg["obs_c"]))
                params = _load(cfg["ckpts"]["IPPO"], net, dummy)
            elif algo == "MAPPO":
                net    = _MAPPOActor(action_dim=cfg["n_actions"])
                dummy  = jnp.zeros((cfg["n_agents"], cfg["obs_h"], cfg["obs_w"], cfg["obs_c"]))
                params = _load(cfg["ckpts"]["MAPPO"], net, dummy)
            elif algo == "MAT":
                from jaxmarl_worker.networks.mat_actor import MATCNNActor
                n_obs_flat  = cfg["obs_h"] * cfg["obs_w"] * cfg["obs_c"]
                mat_act_dim = _detect_mat_action_dim(cfg["ckpts"]["MAT"])
                if mat_act_dim != cfg["n_actions"]:
                    print(f"  [MAT WARN] checkpoint action_dim={mat_act_dim} != env n_actions={cfg['n_actions']} "
                          f"(training misconfiguration; extra actions will be clipped)", flush=True)
                cfg = dict(cfg, mat_action_dim=mat_act_dim)
                net = MATCNNActor(
                    action_dim=mat_act_dim, n_agent=cfg["n_agents"],
                    n_embd=128, n_head=4, n_block=2,
                    obs_h=cfg["obs_h"], obs_w=cfg["obs_w"], obs_c=cfg["obs_c"],
                )
                dummy_obs = jnp.zeros((1, cfg["n_agents"], n_obs_flat))
                dummy_sa  = jnp.zeros((1, cfg["n_agents"], mat_act_dim + 1))
                params    = _load_mat(cfg["ckpts"]["MAT"], net, dummy_obs, dummy_sa)

            rollout_fn = _make_rollout(env, algo, net, params, cfg, stochastic=args.stochastic)
            ep_returns, action_counts = jax.block_until_ready(rollout_fn(keys))
            t1 = time.time()

            ep_returns    = np.array(ep_returns)
            action_counts = np.array(action_counts).sum(axis=0)
            action_pct    = 100.0 * action_counts / action_counts.sum()

            mean_ret = ep_returns.mean()
            std_ret  = ep_returns.std()
            results[env_name][algo] = mean_ret

            print(f"done in {t1-t0:.1f}s")
            print(f"    return: {mean_ret:7.2f} +/- {std_ret:.2f}   "
                  f"(paper target: {cfg['paper_target']:.0f})")
            top3 = np.argsort(action_pct)[::-1][:3]
            acts = [f"a{i}={action_pct[i]:.0f}%" for i in top3]
            print(f"    top actions: {', '.join(acts)}")

    # Summary comparison table
    print(f"\n\n{'='*65}")
    print("  SUMMARY -- mean episode return / agent")
    print(f"{'='*65}")
    header = f"  {'ENV':<14}" + "".join(f"{a:>10}" for a in ["RANDOM","IPPO","MAPPO","MAT","Paper"])
    print(header)
    print("  " + "-"*63)
    for env_name in run_envs:
        cfg = env_cfgs[env_name]
        row = f"  {env_name:<14}"
        for algo in ["RANDOM", "IPPO", "MAPPO", "MAT"]:
            val = results[env_name].get(algo, float("nan"))
            row += f"{val:>10.1f}" if not np.isnan(val) else f"{'--':>10}"
        row += f"{cfg['paper_target']:>10.0f}"
        print(row)

    # vs-random lift
    if "RANDOM" in run_algos:
        print(f"\n  Lift vs RANDOM:")
        for env_name in run_envs:
            rand_r = results[env_name].get("RANDOM", float("nan"))
            for algo in ["IPPO", "MAPPO", "MAT"]:
                val = results[env_name].get(algo, float("nan"))
                if not (np.isnan(val) or np.isnan(rand_r)) and rand_r > 0:
                    lift = (val - rand_r) / abs(rand_r) * 100
                    print(f"    {env_name}/{algo}: {lift:+.0f}% vs random  "
                          f"({val:.1f} vs {rand_r:.1f})")
    print()


if __name__ == "__main__":
    main()
