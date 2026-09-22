"""CommNet + PPO (GPU-accelerated via jax.lax.scan).

Architecture (Sukhbaatar et al., NeurIPS 2016 -- faithful to Xuance reference):
  - Shared GRU encoder per agent (parameter sharing)
  - C-layer applied to h_{t-1} (zero-init, comm starts silent):
      c_i = C(h_i^{t-1}),  comm_i = (1/(N-1)) * sum_{j!=i} c_j
  - Comm injected into GRU input (paper: h^{t+1} = sigma(H*h + C*c + W_e*s)):
      new_h = GRU(h_{t-1}, tanh(W_e * obs) + comm)
  - Per-agent PPO loss (IPPO-style): no centralized critic
  - Hidden state carried across steps; reset at episode boundaries

Checkpointing:
  Only the final checkpoint is written, to <ckpt_dir>/final.npz.
  Every run trains from scratch and saves exactly one file at the end.

Checkpoint format: commnet_{leaf_idx}
"""

import os
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

import argparse
import time
from pathlib import Path
from typing import Any, NamedTuple, Dict

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

SAVE_EVERY  = 5000  # updates per training chunk (JIT scan length, 4 chunks for 20k run)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class CommNet(nn.Module):
    hid_dim:    int = 256
    action_dim: int = 8

    @nn.compact
    def __call__(self, hidden, obs, done):
        """
        hidden : (n_envs, n_agents, hid_dim)
        obs    : (n_envs, n_agents, obs_dim)
        done   : (n_envs,)
        Returns new_hidden, logits, values.

        Faithful to Sukhbaatar et al. 2016 + Xuance reference:
          C applied to h_{t-1} (zero-init, comm starts silent)
          comm_i = (1/(N-1)) * sum_{j!=i} C(h_j^{t-1})
          new_h  = GRU(h_{t-1}, tanh(W_e * obs) + comm)
        Injects comm into GRU input gate, equivalent to
        h^{t+1} = sigma(H*h + C*c + W_e*s) with GRU replacing H and sigma.
        """
        n_envs, n_agents, obs_dim = obs.shape

        done_ag = jnp.broadcast_to(done[:, None, None], hidden.shape)
        hidden  = jnp.where(done_ag, jnp.zeros_like(hidden), hidden)

        h_flat = hidden.reshape(n_envs * n_agents, self.hid_dim)
        o_flat = obs.reshape(n_envs * n_agents, obs_dim)

        # C applied to h_{t-1}; zero-init so comm starts silent and is learned gradually
        c_flat = nn.Dense(self.hid_dim,
                          kernel_init=constant(0.0),
                          bias_init=constant(0.0))(h_flat)
        c = c_flat.reshape(n_envs, n_agents, self.hid_dim)
        c_sum = c.sum(axis=1, keepdims=True)
        if n_agents > 1:
            comm = (c_sum - c) / (n_agents - 1)
        else:
            comm = jnp.zeros_like(c)
        comm_flat = comm.reshape(n_envs * n_agents, self.hid_dim)

        x = nn.Dense(self.hid_dim,
                     kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(o_flat)
        x = nn.tanh(x)

        # Inject comm into GRU input: implements H*h + W_e*s + C*c in one pass
        new_h_flat, _ = nn.GRUCell(self.hid_dim)(h_flat, x + comm_flat)
        new_h = new_h_flat.reshape(n_envs, n_agents, self.hid_dim)

        logits = nn.Dense(self.action_dim,
                          kernel_init=orthogonal(0.01),
                          bias_init=constant(0.0))(new_h_flat)
        value  = jnp.squeeze(
            nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(new_h_flat),
            axis=-1,
        )
        return (
            new_h,
            logits.reshape(n_envs, n_agents, self.action_dim),
            value.reshape(n_envs, n_agents),
        )


# ---------------------------------------------------------------------------
# Training structures
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    obs:      chex.Array
    hidden:   chex.Array
    action:   chex.Array
    log_prob: chex.Array
    value:    chex.Array
    reward:   chex.Array
    done:     chex.Array


class RunnerState(NamedTuple):
    train_state: TrainState
    hidden:      chex.Array
    env_state:   Any
    obs:         Dict
    key:         chex.PRNGKey


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _leaves(x):
    ls, _ = jax.tree_util.tree_flatten(jax.tree_util.tree_map(np.array, x))
    return ls


# ---------------------------------------------------------------------------
# make_train  — returns a JIT fn that runs exactly SAVE_EVERY update steps
# ---------------------------------------------------------------------------

def make_train(config: dict):
    env        = config["ENV"]
    n_agents   = env.num_agents
    n_envs     = config["N_ENVS"]
    n_steps    = config["N_STEPS"]
    n_epochs   = config["N_EPOCHS"]
    n_mb       = config["N_MINIBATCHES"]
    gamma      = config["GAMMA"]
    gae_lam    = config["GAE_LAMBDA"]
    clip_eps   = config["CLIP_EPS"]
    vf_coef    = config["VF_COEF"]
    ent_coef   = config["ENT_COEF"]
    hid_dim    = config["HIDDEN_DIM"]
    n_updates  = config["TOTAL_UPDATES"]
    lr         = config["LR"]

    obs_dim      = env._obs_dim
    total_per_ag = n_steps * n_envs
    mb_size      = total_per_ag // n_mb

    net = CommNet(hid_dim=hid_dim, action_dim=8)

    def _make_tx():
        sched = optax.linear_schedule(lr, lr / 10, n_updates * n_epochs * n_mb)
        return optax.chain(optax.clip_by_global_norm(0.5), optax.adam(sched, eps=1e-5))

    tx = _make_tx()

    # ------------------------------------------------------------------ #
    def _update_step(runner_state: RunnerState, _):

        def _env_step(runner_state: RunnerState, _):
            ts, hidden, env_state, obs, key = runner_state
            obs_arr   = jnp.stack([obs[f"agent_{i}"] for i in range(n_agents)], axis=1)
            done_prev = jnp.zeros(n_envs)
            new_hidden, logits, values = ts.apply_fn(ts.params, hidden, obs_arr, done_prev)

            key, k = jax.random.split(key)
            lf = logits.reshape(n_envs * n_agents, 8)
            af = jax.random.categorical(k, lf)
            lp = jax.nn.log_softmax(lf)[jnp.arange(n_envs * n_agents), af]
            acts = af.reshape(n_envs, n_agents)
            logp = lp.reshape(n_envs, n_agents)

            key, sk = jax.random.split(key)
            next_obs, next_env_state, rewards, dones, _ = jax.vmap(env.step)(
                jax.random.split(sk, n_envs), env_state,
                {f"agent_{i}": acts[:, i] for i in range(n_agents)},
            )
            reward_arr = jnp.stack([rewards[f"agent_{i}"] for i in range(n_agents)], axis=1)
            done_all   = dones["__all__"]

            done_ag = jnp.broadcast_to(done_all[:, None, None], new_hidden.shape)
            new_hidden = jnp.where(done_ag, jnp.zeros_like(new_hidden), new_hidden)

            transition = Transition(obs=obs_arr, hidden=hidden,
                                    action=acts, log_prob=logp, value=values,
                                    reward=reward_arr, done=done_all)
            return RunnerState(ts, new_hidden, next_env_state, next_obs, key), transition

        runner_state, traj = jax.lax.scan(_env_step, runner_state, None, n_steps)
        ts, hidden, env_state, obs, key = runner_state

        obs_arr = jnp.stack([obs[f"agent_{i}"] for i in range(n_agents)], axis=1)
        _, _, last_val = ts.apply_fn(ts.params, hidden, obs_arr, jnp.zeros(n_envs))

        def _gae(carry, t: Transition):
            last_gae, next_val = carry
            done_ba = jnp.broadcast_to(t.done[:, None], (n_envs, n_agents))
            delta   = t.reward + gamma * next_val * (1 - done_ba) - t.value
            gae     = delta + gamma * gae_lam * (1 - done_ba) * last_gae
            return (gae, t.value), gae

        _, advantages = jax.lax.scan(
            _gae, (jnp.zeros((n_envs, n_agents)), last_val),
            traj, reverse=True, unroll=16,
        )
        targets = advantages + traj.value

        def _flatten(x):
            return x.reshape(n_steps * n_envs, *x.shape[2:])

        obs_flat  = _flatten(traj.obs)
        h_flat    = _flatten(traj.hidden)
        act_flat  = _flatten(traj.action)
        logp_flat = _flatten(traj.log_prob)
        adv_flat  = _flatten(advantages)
        ret_flat  = _flatten(targets)
        done_flat = _flatten(traj.done)

        def _epoch(ts, _):
            key_e, _ = jax.random.split(jax.random.PRNGKey(ts.step))
            perm     = jax.random.permutation(key_e, total_per_ag)
            o_mb  = obs_flat[perm].reshape(n_mb, mb_size, n_agents, obs_dim)
            h_mb  = h_flat[perm].reshape(n_mb, mb_size, n_agents, hid_dim)
            a_mb  = act_flat[perm].reshape(n_mb, mb_size, n_agents)
            lp_mb = logp_flat[perm].reshape(n_mb, mb_size, n_agents)
            adv_mb= adv_flat[perm].reshape(n_mb, mb_size, n_agents)
            ret_mb= ret_flat[perm].reshape(n_mb, mb_size, n_agents)
            d_mb  = done_flat[perm].reshape(n_mb, mb_size)

            def _mb_update(ts, batch):
                o_b, h_b, a_b, lp_b, adv_b, ret_b, d_b = batch

                def loss_fn(params):
                    _, logits_b, val_b = net.apply(params, h_b, o_b, d_b)
                    B, N, A = logits_b.shape
                    lps      = jax.nn.log_softmax(logits_b)
                    lps_flat = lps.reshape(B * N, A)
                    new_lp   = lps_flat[jnp.arange(B * N),
                                        a_b.reshape(B * N)].reshape(B, N)
                    ratio  = jnp.exp(new_lp - lp_b)
                    adv_n  = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)
                    pg     = jnp.maximum(
                        -adv_n * ratio,
                        -adv_n * jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps),
                    ).mean()
                    vf  = 0.5 * jnp.mean((val_b - ret_b) ** 2)
                    ent = -(jax.nn.softmax(logits_b) * lps).sum(-1).mean()
                    return pg + vf_coef * vf - ent_coef * ent, (pg, vf, ent)

                (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(ts.params)
                ts = ts.apply_gradients(grads=grads)
                return ts, {"loss": loss, "pg": aux[0], "vf": aux[1], "ent": aux[2]}

            ts, mb_m = jax.lax.scan(_mb_update, ts,
                                     (o_mb, h_mb, a_mb, lp_mb, adv_mb, ret_mb, d_mb))
            return ts, {k: v.mean() for k, v in mb_m.items()}

        ts, ep_m = jax.lax.scan(_epoch, ts, None, n_epochs)
        info = {"mean_episode_return": traj.reward.mean(),
                **{k: v.mean() for k, v in ep_m.items()}}
        runner_state = RunnerState(ts, hidden, env_state, obs, key)
        return runner_state, info

    # Chunk function: runs exactly SAVE_EVERY update steps
    def train_chunk(runner_state: RunnerState):
        return jax.lax.scan(_update_step, runner_state, None, SAVE_EVERY)

    return jax.jit(train_chunk), net, tx


def init_runner_state(config: dict, key: chex.PRNGKey, net, tx) -> RunnerState:
    """Create a fresh RunnerState (used when no checkpoint exists)."""
    env      = config["ENV"]
    n_agents = env.num_agents
    n_envs   = config["N_ENVS"]
    hid_dim  = config["HIDDEN_DIM"]
    obs_dim  = env._obs_dim

    key, ik, rk = jax.random.split(key, 3)
    dummy_h = jnp.zeros((1, n_agents, hid_dim))
    dummy_o = jnp.zeros((1, n_agents, obs_dim))
    dummy_d = jnp.zeros((1,))
    params  = net.init(ik, dummy_h, dummy_o, dummy_d)
    ts      = TrainState.create(apply_fn=net.apply, params=params, tx=tx)

    obs, env_state = jax.vmap(env.reset)(jax.random.split(rk, n_envs))
    hidden = jnp.zeros((n_envs, n_agents, hid_dim))
    return RunnerState(ts, hidden, env_state, obs, key)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_env(family, variant, view_size=7, ball_coef=0.01, max_steps=256, goal_rows=None):
    if family == "af":
        from jaxmarl_worker.environments.mosaic_multigrid.AF.american_football_jax import AmericanFootballJAX
        return AmericanFootballJAX(variant=variant, view_size=view_size,
                                   ball_coef=ball_coef, max_steps=max_steps, goal_rows=goal_rows)
    elif family == "bb":
        from jaxmarl_worker.environments.mosaic_multigrid.BB.basketball_jax import BasketballJAX
        return BasketballJAX(variant=variant, view_size=view_size,
                             ball_coef=ball_coef, max_steps=max_steps, goal_rows=goal_rows)
    elif family == "soccer":
        from jaxmarl_worker.environments.mosaic_multigrid.S.soccer_jax import SoccerJAX
        return SoccerJAX(variant=variant, view_size=view_size,
                         ball_coef=ball_coef, max_steps=max_steps, goal_rows=goal_rows)
    raise ValueError(f"Unknown env family: {family}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env-family",    type=str, default="af", choices=["af","bb","soccer"])
    p.add_argument("--variant",       type=str, default="2v2")
    p.add_argument("--run-dir",       type=str, required=True)
    p.add_argument("--total-updates", type=int, default=2000)
    p.add_argument("--n-envs",        type=int, default=256)
    p.add_argument("--n-steps",       type=int, default=256)
    p.add_argument("--n-epochs",      type=int, default=4)
    p.add_argument("--n-minibatches", type=int, default=4)
    p.add_argument("--hidden-dim",    type=int, default=256)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--gamma",         type=float, default=0.99)
    p.add_argument("--gae-lambda",    type=float, default=0.95)
    p.add_argument("--clip-eps",      type=float, default=0.2)
    p.add_argument("--vf-coef",       type=float, default=0.5)
    p.add_argument("--ent-coef",      type=float, default=0.005)
    p.add_argument("--seed",          type=int, default=1)
    p.add_argument("--log-every",     type=int, default=1,
                   help="TensorBoard write stride in updates (1 = every update)")
    p.add_argument("--save-every",    type=int, default=5000)
    p.add_argument("--tensorboard",   action="store_true")
    p.add_argument("--view-size",          type=int, default=7)
    p.add_argument("--ball-approach-coef", type=float, default=0.01)
    p.add_argument("--max-steps",          type=int, default=256)
    p.add_argument("--goal-rows",          type=int, nargs="+", default=None)
    p.add_argument("--cooperative", action="store_true",
                   help="Compatibility flag accepted by legacy launchers (no effect)")
    return p.parse_args()


def main():
    args     = parse_args()
    run_dir  = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints" / f"{args.env_family}-{args.variant}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    env = _make_env(args.env_family, args.variant, view_size=args.view_size,
                    ball_coef=args.ball_approach_coef, max_steps=args.max_steps,
                    goal_rows=args.goal_rows)

    n_updates  = args.total_updates
    save_every = args.save_every
    assert n_updates % save_every == 0, \
        f"--total-updates ({n_updates}) must be divisible by --save-every ({save_every})"

    print(f"[CommNet-scan] {args.env_family}-{args.variant}  "
          f"n_agents={env.num_agents}  n_envs={args.n_envs}  "
          f"obs_dim={env._obs_dim}  hid_dim={args.hidden_dim}")
    print(f"  Communication: mean-pool (CommNet, Sukhbaatar et al. 2016)")
    print(f"  total env steps = "
          f"{n_updates * args.n_envs * args.n_steps * env.num_agents:,}")
    print(f"  logging every {save_every} updates")

    config = dict(
        ENV=env, N_ENVS=args.n_envs, N_STEPS=args.n_steps,
        N_EPOCHS=args.n_epochs, N_MINIBATCHES=args.n_minibatches,
        GAMMA=args.gamma, GAE_LAMBDA=args.gae_lambda, CLIP_EPS=args.clip_eps,
        VF_COEF=args.vf_coef, ENT_COEF=args.ent_coef, HIDDEN_DIM=args.hidden_dim,
        TOTAL_UPDATES=n_updates, LR=args.lr,
    )

    train_chunk_fn, net, tx = make_train(config)

    # ---- fresh start (no checkpoint loading: every run trains from scratch) ----
    key = jax.random.PRNGKey(args.seed)
    print("[CommNet-scan] Fresh start — JIT-compiling...", flush=True)
    runner_state = init_runner_state(config, key, net, tx)

    tb_writer = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    # ---- chunked training loop ----
    n_chunks = n_updates // save_every
    t0 = time.time()

    for chunk_idx in range(n_chunks):
        runner_state, metrics = jax.block_until_ready(train_chunk_fn(runner_state))
        update_end = (chunk_idx + 1) * save_every

        # Logging
        er  = float(np.array(metrics["mean_episode_return"]).mean())
        lo  = float(np.array(metrics["loss"]).mean())
        ent = float(np.array(metrics["ent"]).mean())
        elapsed = time.time() - t0
        steps_done = update_end * args.n_envs * args.n_steps * env.num_agents
        sps = steps_done / max(elapsed, 1)
        print(f"  update {update_end:>6}/{n_updates}  "
              f"ep_ret={er:.4f}  loss={lo:.4f}  ent={ent:.4f}  "
              f"sps={sps:,.0f}  {elapsed:.0f}s", flush=True)

        if tb_writer is not None:
            er_arr  = np.array(metrics["mean_episode_return"])
            lo_arr  = np.array(metrics["loss"])
            ent_arr = np.array(metrics["ent"])
            for i in range(0, save_every, args.log_every):
                step = chunk_idx * save_every + i + 1
                tb_writer.add_scalar("train/episode_return", float(er_arr[i]), step)
                tb_writer.add_scalar("train/loss", float(lo_arr[i]), step)
                tb_writer.add_scalar("train/entropy", float(ent_arr[i]), step)

    if tb_writer is not None:
        tb_writer.close()

    # ---- save final checkpoint ----
    leaves = _leaves(runner_state.train_state.params)
    np.savez(str(ckpt_dir / "final.npz"), **{f"commnet_{i}": v for i, v in enumerate(leaves)})
    print(f"[CommNet-scan] checkpoint → {ckpt_dir}/final.npz  ({len(leaves)} leaves)")
    print(f"[CommNet-scan] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
