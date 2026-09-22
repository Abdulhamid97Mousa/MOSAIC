"""JAX MAGIC training on PredatorPreyJAX.

Adapted from jaxmarl_worker/algorithms/magic_scan.py for the Predator-Prey
benchmark comparison.  Key differences vs the soccer/football version:
  - Env is PredatorPreyJAX (n_agents=3, n_actions=5, obs_dim=261)
  - n_actions changed from 8 → 5 everywhere
  - Tracking mean_team_reward per update
  - Saving results to /tmp/magic_comparison/jax_results.npz

Run from /usrhome/Hamid/projects/mosaic/3rd_party/workers/jaxmarl_worker:
  .venv/bin/python benchmark/train_magic_jax.py

DO NOT modify the original magic_scan.py — this is a self-contained copy.
"""

from __future__ import annotations

import os
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

import sys
import time
from pathlib import Path
from typing import NamedTuple, Dict, Tuple, Any

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

# ---------------------------------------------------------------------------
# Make sure benchmark/ and jaxmarl_worker/ are importable regardless of cwd
# ---------------------------------------------------------------------------
_WORKER_ROOT = Path(__file__).resolve().parent.parent
if str(_WORKER_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKER_ROOT))

from benchmark.pp_env_jax import PredatorPreyJAX  # noqa: E402

# ---------------------------------------------------------------------------
# Network modules  (identical to magic_scan.py — copied to allow n_actions=5)
# ---------------------------------------------------------------------------

class GraphAttention(nn.Module):
    """Graph Attention layer (JAX/Flax port of MAGIC gnn_layers.py)."""
    out_features: int
    num_heads: int = 4
    negative_slope: float = 0.2
    average: bool = False

    @nn.compact
    def __call__(self, h: chex.Array, adj: chex.Array) -> chex.Array:
        N, in_f = h.shape
        W = self.param('W', nn.initializers.glorot_normal(),
                       (in_f, self.num_heads * self.out_features))
        a_i = self.param('a_i', nn.initializers.glorot_normal(),
                         (self.num_heads, self.out_features))
        a_j = self.param('a_j', nn.initializers.glorot_normal(),
                         (self.num_heads, self.out_features))
        out_dim = self.out_features if self.average else self.num_heads * self.out_features
        bias = self.param('bias', nn.initializers.zeros, (out_dim,))

        h_t = (h @ W).reshape(N, self.num_heads, self.out_features)
        ci = jnp.einsum('nhf,hf->nh', h_t, a_i)
        cj = jnp.einsum('nhf,hf->nh', h_t, a_j)
        e = ci[:, None, :] + cj[None, :, :]
        e = jax.nn.leaky_relu(e, negative_slope=self.negative_slope)

        adj3 = adj[:, :, None]
        attn = jax.nn.softmax(e * adj3, axis=1) * adj3
        out = jnp.einsum('ijh,jhf->ihf', attn, h_t)

        if self.average:
            return out.mean(axis=1) + bias
        return out.reshape(N, self.num_heads * self.out_features) + bias


class SubScheduler(nn.Module):
    """MLP → binary adjacency via straight-through Gumbel-Softmax."""
    hid_size: int

    @nn.compact
    def __call__(self, h: chex.Array, key: chex.PRNGKey) -> chex.Array:
        N = h.shape[0]
        h_i = jnp.tile(h[:, None, :], (1, N, 1))
        h_j = jnp.tile(h[None, :, :], (N, 1, 1))
        pairs = jnp.concatenate([h_i, h_j], axis=-1)

        x = nn.relu(nn.Dense(self.hid_size // 2)(pairs))
        x = nn.relu(nn.Dense(self.hid_size // 8)(x))
        x = nn.Dense(2)(x)

        gumbel = -jnp.log(-jnp.log(jax.random.uniform(key, x.shape) + 1e-20) + 1e-20)
        y_soft = jax.nn.softmax(x + gumbel, axis=-1)
        y_hard = jax.nn.one_hot(jnp.argmax(y_soft, axis=-1), 2)
        y = y_hard + y_soft - jax.lax.stop_gradient(y_soft)
        return y[..., 1]


class MAGICNet(nn.Module):
    """MAGIC: LSTM + 2-round GAT + action/value heads."""
    hid_size: int
    n_agents: int
    n_actions: int
    gat_hid_size: int = 64
    gat_num_heads: int = 4
    gat_num_heads_out: int = 1

    @nn.compact
    def __call__(
        self,
        obs: chex.Array,
        hx: chex.Array,
        cx: chex.Array,
        key: chex.PRNGKey,
    ) -> Tuple:
        key, sk1, sk2 = jax.random.split(key, 3)

        enc = nn.Dense(self.hid_size, name='obs_encoder')(obs)

        lstm = nn.LSTMCell(self.hid_size, name='lstm')
        (new_cx, new_hx), _ = lstm((cx, hx), enc)

        comm_ori = new_hx
        adj1 = SubScheduler(self.hid_size, name='sched1')(comm_ori, sk1)
        comm = nn.elu(
            GraphAttention(self.gat_hid_size, self.gat_num_heads,
                           average=False, name='gat1')(comm_ori, adj1)
        )

        adj2 = SubScheduler(self.hid_size, name='sched2')(comm_ori, sk2)
        comm = GraphAttention(
            self.hid_size, self.gat_num_heads_out, average=True, name='gat2'
        )(comm, adj2)

        h_cat = jnp.concatenate([new_hx, comm], axis=-1)
        logits = nn.Dense(self.n_actions, name='action_head')(h_cat)
        value  = nn.Dense(1, name='value_head')(h_cat).squeeze(-1)

        return logits, value, new_hx, new_cx


# ---------------------------------------------------------------------------
# Trajectory + Runner state
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    obs:      chex.Array
    hx:       chex.Array
    cx:       chex.Array
    action:   chex.Array
    log_prob: chex.Array
    value:    chex.Array
    reward:   chex.Array
    done:     chex.Array


class RunnerState(NamedTuple):
    train_state: TrainState
    env_state:   Any
    obs:         Dict
    hx:          chex.Array
    cx:          chex.Array
    key:         chex.PRNGKey


# ---------------------------------------------------------------------------
# Training function factory
# ---------------------------------------------------------------------------

def make_train(config: dict):
    env      = config["ENV"]
    n_agents = env.num_agents       # 3
    n_actions = config["N_ACTIONS"] # 5
    n_envs   = config["N_ENVS"]
    n_steps  = config["N_STEPS"]
    gamma    = config["GAMMA"]
    vf_coef  = config["VF_COEF"]
    ent_coef = config["ENT_COEF"]
    hid_size = config["HID_SIZE"]
    gat_hid  = config["GAT_HID_SIZE"]
    gat_heads = config["GAT_NUM_HEADS"]
    lr       = config["LR"]

    obs_dim  = env._obs_dim  # 261

    net = MAGICNet(
        hid_size=hid_size,
        n_agents=n_agents,
        n_actions=n_actions,
        gat_hid_size=gat_hid,
        gat_num_heads=gat_heads,
        gat_num_heads_out=1,
    )

    def _net_apply_batch(params, obs_b, hx_b, cx_b, keys_b):
        return jax.vmap(lambda o, h, c, k: net.apply(params, o, h, c, k))(
            obs_b, hx_b, cx_b, keys_b
        )

    def train_chunk(runner_state: RunnerState, n_updates_chunk: int):
        """Run `n_updates_chunk` update steps, returning (runner_state, info_chunk)."""

        def _update_step(runner_state: RunnerState, _):
            # -- collect N_STEPS transitions --
            def _env_step(runner_state: RunnerState, _):
                train_state, env_state, obs, hx, cx, key = runner_state

                obs_arr = jnp.stack(
                    [obs[f"agent_{i}"] for i in range(n_agents)], axis=1
                )  # (N_ENVS, N_AGENTS, obs_dim)

                key, net_key = jax.random.split(key)
                net_keys = jax.random.split(net_key, n_envs)

                logits, values, new_hx, new_cx = _net_apply_batch(
                    train_state.params, obs_arr, hx, cx, net_keys
                )  # (N_ENVS, N_AGENTS, n_actions), (N_ENVS, N_AGENTS)

                key, action_key = jax.random.split(key)
                logits_flat = logits.reshape(n_envs * n_agents, n_actions)
                actions_flat = jax.random.categorical(action_key, logits_flat)
                log_probs_flat = jax.nn.log_softmax(logits_flat)[
                    jnp.arange(n_envs * n_agents), actions_flat
                ]

                actions_arr  = actions_flat.reshape(n_envs, n_agents)
                actions_dict = {f"agent_{i}": actions_arr[:, i] for i in range(n_agents)}

                key, step_key = jax.random.split(key)
                step_keys = jax.random.split(step_key, n_envs)
                next_obs, next_env_state, rewards, dones, _ = jax.vmap(env.step)(
                    step_keys, env_state, actions_dict
                )

                reward_arr = jnp.stack(
                    [rewards[f"agent_{i}"] for i in range(n_agents)], axis=1
                )  # (N_ENVS, N_AGENTS)

                done_flag = dones["__all__"]              # (N_ENVS,)
                alive = (~done_flag)[:, None, None]       # (N_ENVS, 1, 1)
                new_hx = new_hx * alive
                new_cx = new_cx * alive

                transition = Transition(
                    obs      = obs_arr,
                    hx       = hx,
                    cx       = cx,
                    action   = actions_arr,
                    log_prob = log_probs_flat.reshape(n_envs, n_agents),
                    value    = values,
                    reward   = reward_arr,
                    done     = done_flag,
                )
                return RunnerState(train_state, next_env_state, next_obs, new_hx, new_cx, key), transition

            runner_state, traj = jax.lax.scan(_env_step, runner_state, None, n_steps)
            # traj shapes: (N_STEPS, N_ENVS, ...)

            # -- bootstrap last value --
            train_state, env_state, obs, hx, cx, key = runner_state
            obs_arr = jnp.stack([obs[f"agent_{i}"] for i in range(n_agents)], axis=1)
            key, net_key = jax.random.split(key)
            net_keys = jax.random.split(net_key, n_envs)
            _, last_vals, _, _ = _net_apply_batch(
                train_state.params, obs_arr, hx, cx, net_keys
            )

            # -- discounted returns via reverse scan --
            def _return_step(carry, t: Transition):
                ret = carry
                done_ba = jnp.broadcast_to(t.done[:, None], (n_envs, n_agents))
                ret = t.reward + gamma * ret * (1.0 - done_ba)
                return ret, ret

            _, returns = jax.lax.scan(
                _return_step, last_vals, traj, reverse=True
            )  # (N_STEPS, N_ENVS, N_AGENTS)

            advantages = returns - traj.value

            # -- gradient update --
            key, upd_key = jax.random.split(key)
            T_E = n_steps * n_envs
            upd_keys = jax.random.split(upd_key, T_E)

            def loss_fn(params):
                obs_flat = traj.obs.reshape(T_E, n_agents, obs_dim)
                hx_flat  = traj.hx.reshape(T_E, n_agents, hid_size)
                cx_flat  = traj.cx.reshape(T_E, n_agents, hid_size)

                logits_new, vals_new, _, _ = _net_apply_batch(
                    params, obs_flat, hx_flat, cx_flat, upd_keys
                )

                lp = jax.nn.log_softmax(logits_new, axis=-1)
                acts = traj.action.reshape(T_E, n_agents)
                logp_new = lp[
                    jnp.arange(T_E)[:, None],
                    jnp.arange(n_agents)[None, :],
                    acts,
                ]

                adv_flat = advantages.reshape(T_E, n_agents)
                adv_norm = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
                rets_flat = returns.reshape(T_E, n_agents)

                pg_loss = -(adv_norm * logp_new).mean()
                vf_loss = 0.5 * ((vals_new - rets_flat) ** 2).mean()
                probs   = jax.nn.softmax(logits_new, axis=-1)
                entropy = -(probs * lp).sum(-1).mean()
                total   = pg_loss + vf_coef * vf_loss - ent_coef * entropy
                return total, (pg_loss, vf_loss, entropy)

            (total_loss, (pg_l, vf_l, ent)), grads = jax.value_and_grad(
                loss_fn, has_aux=True
            )(train_state.params)
            train_state = train_state.apply_gradients(grads=grads)

            runner_state = RunnerState(train_state, env_state, obs, hx, cx, key)

            # mean_team_reward: mean over steps, envs, agents
            mean_team_reward = traj.reward.mean()
            info = {
                "mean_team_reward": mean_team_reward,
                "total_loss":       total_loss,
                "pg_loss":          pg_l,
                "vf_loss":          vf_l,
                "entropy":          ent,
            }
            return runner_state, info

        runner_state, info_chunk = jax.lax.scan(
            _update_step, runner_state, None, n_updates_chunk
        )
        return runner_state, info_chunk

    return jax.jit(train_chunk, static_argnums=(1,))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Hyperparameters matching the spec
    HID_SIZE       = 64
    GAT_HID_SIZE   = 32
    GAT_NUM_HEADS  = 4
    LR             = 1e-3
    GAMMA          = 1.0
    VF_COEF        = 0.01
    ENT_COEF       = 0.01
    N_ENVS         = 32
    N_STEPS        = 20
    TOTAL_UPDATES  = 1000
    SEED           = 42

    # Chunking for wall-time tracking
    CHUNK_SIZE     = 50    # 20 chunks of 50 updates each
    N_CHUNKS       = TOTAL_UPDATES // CHUNK_SIZE

    env = PredatorPreyJAX()

    config = {
        "ENV":           env,
        "N_ACTIONS":     5,
        "N_ENVS":        N_ENVS,
        "N_STEPS":       N_STEPS,
        "GAMMA":         GAMMA,
        "VF_COEF":       VF_COEF,
        "ENT_COEF":      ENT_COEF,
        "HID_SIZE":      HID_SIZE,
        "GAT_HID_SIZE":  GAT_HID_SIZE,
        "GAT_NUM_HEADS": GAT_NUM_HEADS,
        "LR":            LR,
    }

    print(f"[JAX-MAGIC-PP] n_agents={env.num_agents}  n_actions=5  obs_dim={env._obs_dim}")
    print(f"  n_envs={N_ENVS}  n_steps={N_STEPS}  total_updates={TOTAL_UPDATES}")
    total_env_steps = TOTAL_UPDATES * N_ENVS * N_STEPS * env.num_agents
    print(f"  total env steps = {total_env_steps:,}")

    train_chunk_fn = make_train(config)

    # Initialise network and env
    key = jax.random.PRNGKey(SEED)

    net = MAGICNet(
        hid_size=HID_SIZE,
        n_agents=env.num_agents,
        n_actions=5,
        gat_hid_size=GAT_HID_SIZE,
        gat_num_heads=GAT_NUM_HEADS,
        gat_num_heads_out=1,
    )

    key, nk, gk = jax.random.split(key, 3)
    dummy_obs = jnp.zeros((env.num_agents, env._obs_dim))
    dummy_hx  = jnp.zeros((env.num_agents, HID_SIZE))
    dummy_cx  = jnp.zeros((env.num_agents, HID_SIZE))
    params = net.init(nk, dummy_obs, dummy_hx, dummy_cx, gk)

    lr_sched = optax.linear_schedule(LR, LR / 10, TOTAL_UPDATES)
    tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr_sched, eps=1e-5))
    train_state = TrainState.create(apply_fn=net.apply, params=params, tx=tx)

    key, reset_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, N_ENVS)
    obs, env_state = jax.vmap(env.reset)(reset_keys)

    hx = jnp.zeros((N_ENVS, env.num_agents, HID_SIZE))
    cx = jnp.zeros((N_ENVS, env.num_agents, HID_SIZE))
    runner_state = RunnerState(train_state, env_state, obs, hx, cx, key)

    print("[JAX-MAGIC-PP] JIT-compiling first chunk...", flush=True)

    # Storage for results across chunks
    all_team_rewards = []
    all_wall_times   = []

    t_start = time.time()
    t_last  = t_start

    for chunk_idx in range(N_CHUNKS):
        runner_state, info_chunk = jax.block_until_ready(
            train_chunk_fn(runner_state, CHUNK_SIZE)
        )

        t_now = time.time()
        chunk_wall_time = t_now - t_last
        t_last = t_now

        chunk_rewards = np.array(info_chunk["mean_team_reward"])  # (CHUNK_SIZE,)
        all_team_rewards.append(chunk_rewards)

        # Cumulative wall time at the end of each update in this chunk
        # Linearly interpolate within the chunk for per-update time estimates
        for i in range(CHUNK_SIZE):
            frac = (i + 1) / CHUNK_SIZE
            all_wall_times.append(t_now - t_start - chunk_wall_time * (1.0 - frac))

        # Print every 50 updates (i.e. every chunk since CHUNK_SIZE=50)
        if True:
            update_num  = (chunk_idx + 1) * CHUNK_SIZE
            mean_reward = chunk_rewards.mean()
            elapsed     = t_now - t_start
            steps_so_far = update_num * N_ENVS * N_STEPS * env.num_agents
            sps = steps_so_far / elapsed if elapsed > 0 else 0
            print(f"  update={update_num:4d}  mean_team_reward={mean_reward:+.4f}  "
                  f"steps/sec={sps:,.0f}", flush=True)

    t_end = time.time()
    total_wall = t_end - t_start
    total_steps = TOTAL_UPDATES * N_ENVS * N_STEPS * env.num_agents
    steps_per_sec = total_steps / total_wall

    update_rewards = np.concatenate(all_team_rewards)  # (TOTAL_UPDATES,)
    wall_times     = np.array(all_wall_times)           # (TOTAL_UPDATES,)

    print(f"\n[JAX-MAGIC-PP] Training complete in {total_wall:.1f}s  ({steps_per_sec:,.0f} steps/sec)")
    final_10pct = update_rewards[int(0.9 * TOTAL_UPDATES):]
    print(f"  final 10% mean_team_reward: {final_10pct.mean():+.4f}")

    out_dir = Path("/tmp/magic_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "jax_results.npz"
    np.savez(
        str(out_path),
        update_rewards = update_rewards,
        wall_times     = wall_times,
        steps_per_sec  = np.float64(steps_per_sec),
    )
    print(f"  results → {out_path}")


if __name__ == "__main__":
    main()
