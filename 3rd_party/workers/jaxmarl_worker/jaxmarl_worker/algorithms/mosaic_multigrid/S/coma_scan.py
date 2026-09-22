"""GPU-accelerated COMA (Counterfactual Multi-Agent Policy Gradients), faithful to
Foerster et al. 2018 (arXiv:1705.08926) and the reference implementation
github.com/matteokarldonati/Counterfactual-Multi-Agent-Policy-Gradients.

Faithful COMA specifics (vs a PPO wrapper):
  - Actor loss is PLAIN policy gradient: -mean(advantage * log pi(a_taken)).
    No PPO clipping, no importance ratio. => single on-policy pass per rollout
    (--n-epochs 1).
  - Counterfactual advantage from a centralized critic:
        A_i = Q_target(s, a_i) - sum_a pi_i(a) Q_target(s, (a, a_{-i})).
    Both terms use the TARGET critic (detached), exactly as the reference's
    `Q_taken_target` / `baseline`.
  - Critic target is TD(0) with the target-network bootstrap:
        r_t = reward_t + gamma * Q_taken_target_{t+1}   (r_t = reward_t if done).
    critic_loss = mean((r_t - Q_taken)^2), Q_taken from the CURRENT critic.
  - Hard target network updated every --target-update-steps updates.

Deviations, deliberate and flagged:
  - Critic input MASKS the acting agent's own action (feeds a_{-i}), per the
    original paper. The matteokarldonati repo leaves a_i unmasked; masking is the
    theoretically-correct counterfactual and the reviewer-safe choice.
  - The ACTOR is the mosaic-standard feed-forward ActorCritic (256, tanh, value
    head), identical to ippo_scan, so the saved checkpoint loads through the
    existing per-agent inference path with no new loader. The value head is never
    trained (COMA uses the counterfactual critic, not a state-value head); it
    stays at init and is ignored at deployment, which reads only the logits.

At execution COMA is a per-agent actor a_i ~ pi(.|o_i); direct overwrite is
sufficient (AC = IC = 0), like MAPPO/IPPO/QMIX/HAPPO.

Usage (same scale as the ippo_scan V2 / m256_20k recipe):
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m jaxmarl_worker.algorithms.multigrid_sports.coma_scan \
    --env-family af --variant 2v2 --run-dir <path> \
    --n-envs 256 --n-steps 256 --max-steps 256 --total-updates 20000 \
    --n-epochs 1 --target-update-steps 200 --view-size 7 \
    --goal-rows 1 2 3 4 5 6 7 8 9 --tensorboard
"""

import os
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

import argparse
import time
from pathlib import Path
from typing import NamedTuple, Dict

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

ACTION_DIM = 8

# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """Identical to ippo_scan.ActorCritic so the saved actor is loader-compatible.
    Only the logits path is trained under COMA; the value head is unused."""
    action_dim: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        value = jnp.squeeze(
            nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x), axis=-1)
        return logits, value


class COMACritic(nn.Module):
    """Counterfactual critic (3-layer MLP, as in the reference). Input per acting
    agent i: [ global state (all obs flat), a_{-i} one-hot (agent i masked out),
    agent-i identity one-hot ]. Output: Q over agent i's ACTION_DIM actions."""
    action_dim: int
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x))
        x = nn.relu(nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x))
        q = nn.Dense(self.action_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return q


class Transition(NamedTuple):
    obs:    chex.Array  # (N_ENVS, N_AGENTS, OBS_DIM)
    action: chex.Array  # (N_ENVS, N_AGENTS)
    reward: chex.Array  # (N_ENVS, N_AGENTS)
    done:   chex.Array  # (N_ENVS,)


class RunnerState(NamedTuple):
    actor_state:   TrainState
    critic_state:  TrainState
    target_params: any
    env_state:     any
    obs:           Dict
    key:           chex.PRNGKey


# ---------------------------------------------------------------------------
# make_train
# ---------------------------------------------------------------------------

def make_train(config: dict):
    env       = config["ENV"]
    n_agents  = env.num_agents
    n_envs    = config["N_ENVS"]
    n_steps   = config["N_STEPS"]
    n_epochs  = config["N_EPOCHS"]
    n_mb      = config["N_MINIBATCHES"]
    gamma     = config["GAMMA"]
    ent_coef  = config["ENT_COEF"]
    hidden_dim = config["HIDDEN_DIM"]
    critic_hidden = config["CRITIC_HIDDEN"]
    n_updates = config["TOTAL_UPDATES"]
    lr        = config["LR"]
    tgt_every = config["TARGET_UPDATE_STEPS"]

    obs_dim   = env._obs_dim
    state_dim = n_agents * obs_dim
    critic_in_dim = state_dim + n_agents * ACTION_DIM + n_agents
    total = n_steps * n_envs * n_agents
    mb_size = total // n_mb

    actor_net  = ActorCritic(action_dim=ACTION_DIM, hidden_dim=hidden_dim)
    critic_net = COMACritic(action_dim=ACTION_DIM, hidden_dim=critic_hidden)
    agent_ids  = jnp.eye(n_agents)
    mask_rows  = 1.0 - jnp.eye(n_agents)[None, :, :, None]  # zero agent i's own action row

    def _critic_inputs(obs_arr, action_arr):
        """obs_arr (B,n,obs_dim), action_arr (B,n) -> (B,n,critic_in_dim), a_i masked."""
        B = obs_arr.shape[0]
        state = jnp.broadcast_to(obs_arr.reshape(B, state_dim)[:, None, :], (B, n_agents, state_dim))
        ja = jax.nn.one_hot(action_arr, ACTION_DIM)                    # (B,n,A)
        ja_b = jnp.broadcast_to(ja[:, None, :, :], (B, n_agents, n_agents, ACTION_DIM))
        ja_masked = (ja_b * mask_rows).reshape(B, n_agents, n_agents * ACTION_DIM)
        ids = jnp.broadcast_to(agent_ids[None, :, :], (B, n_agents, n_agents))
        return jnp.concatenate([state, ja_masked, ids], axis=-1)

    def train(key: chex.PRNGKey):
        key, ak, ck = jax.random.split(key, 3)
        actor_params  = actor_net.init(ak, jnp.zeros((1, obs_dim)))
        critic_params = critic_net.init(ck, jnp.zeros((1, critic_in_dim)))
        target_params = critic_params  # hard copy at init

        sched = optax.linear_schedule(lr, lr / 10, n_updates * max(n_epochs, 1))
        actor_tx  = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(sched, eps=1e-5))
        critic_tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(sched, eps=1e-5))
        actor_state  = TrainState.create(apply_fn=actor_net.apply,  params=actor_params,  tx=actor_tx)
        critic_state = TrainState.create(apply_fn=critic_net.apply, params=critic_params, tx=critic_tx)

        key, rk = jax.random.split(key)
        obs, env_state = jax.vmap(env.reset)(jax.random.split(rk, n_envs))
        runner_state = RunnerState(actor_state, critic_state, target_params, env_state, obs, key)

        def _update_step(runner_state: RunnerState, upd_idx):
            def _env_step(rs: RunnerState, _):
                a_st, c_st, tp, env_state, obs, key = rs
                obs_arr = jnp.stack([obs[f"agent_{i}"] for i in range(n_agents)], axis=1)
                logits, _ = actor_net.apply(a_st.params, obs_arr.reshape(n_envs * n_agents, obs_dim))
                logits = jnp.asarray(logits)
                key, akey = jax.random.split(key)
                act = jax.random.categorical(akey, logits).reshape(n_envs, n_agents)
                adict = {f"agent_{i}": act[:, i] for i in range(n_agents)}
                key, skey = jax.random.split(key)
                nobs, nenv, rew, done, _ = jax.vmap(env.step)(jax.random.split(skey, n_envs), env_state, adict)
                rarr = jnp.stack([rew[f"agent_{i}"] for i in range(n_agents)], axis=1)
                return (RunnerState(a_st, c_st, tp, nenv, nobs, key),
                        Transition(obs_arr, act, rarr, done["__all__"]))

            runner_state, traj = jax.lax.scan(_env_step, runner_state, None, n_steps)
            a_st, c_st, tp, env_state, obs, key = runner_state

            # ---- target-critic Q over the whole rollout ---------------------
            cin = jax.vmap(_critic_inputs)(traj.obs, traj.action)          # (T,e,n,cin)
            cin_flat = cin.reshape(total, critic_in_dim)
            qT = critic_net.apply(tp, cin_flat).reshape(n_steps, n_envs, n_agents, ACTION_DIM)  # target net
            act_oh = jax.nn.one_hot(traj.action, ACTION_DIM)
            q_taken_T = jnp.sum(qT * act_oh, axis=-1)                       # (T,e,n)

            # counterfactual advantage (target critic + rollout policy), detached
            logits_all, _ = actor_net.apply(a_st.params, traj.obs.reshape(-1, obs_dim))
            pi = jax.nn.softmax(jnp.asarray(logits_all)).reshape(n_steps, n_envs, n_agents, ACTION_DIM)
            baseline = jnp.sum(pi * qT, axis=-1)
            advantage = jax.lax.stop_gradient(q_taken_T - baseline)

            # ---- TD(0) target with target-net bootstrap ---------------------
            last_obs = jnp.stack([obs[f"agent_{i}"] for i in range(n_agents)], axis=1)
            l_logits, _ = actor_net.apply(a_st.params, last_obs.reshape(n_envs * n_agents, obs_dim))
            key, bkey = jax.random.split(key)
            last_act = jax.random.categorical(bkey, jnp.asarray(l_logits)).reshape(n_envs, n_agents)
            l_cin = _critic_inputs(last_obs, last_act).reshape(n_envs * n_agents, critic_in_dim)
            qT_last = critic_net.apply(tp, l_cin).reshape(n_envs, n_agents, ACTION_DIM)
            q_taken_T_last = jnp.sum(qT_last * jax.nn.one_hot(last_act, ACTION_DIM), axis=-1)  # (e,n)

            next_q = jnp.concatenate([q_taken_T[1:], q_taken_T_last[None]], axis=0)  # (T,e,n)
            done_ba = jnp.broadcast_to(traj.done[:, :, None], (n_steps, n_envs, n_agents))
            td_target = jax.lax.stop_gradient(traj.reward + gamma * (1.0 - done_ba) * next_q)

            # ---- flatten -----------------------------------------------------
            f_obs = traj.obs.reshape(total, obs_dim)
            f_cin = cin.reshape(total, critic_in_dim)
            f_act = traj.action.reshape(total)
            f_adv = advantage.reshape(total)
            f_tgt = td_target.reshape(total)

            def _epoch(carry, _):
                a_st, c_st, key = carry
                key, sk = jax.random.split(key)
                perm = jax.random.permutation(sk, total)
                mb = lambda a: a[perm].reshape(n_mb, mb_size, *a.shape[1:])
                batches = (mb(f_obs), mb(f_cin), mb(f_act), mb(f_adv), mb(f_tgt))

                def _mb(carry, batch):
                    a_st, c_st = carry
                    obs_b, cin_b, act_b, adv_b, tgt_b = batch

                    def actor_loss(p):
                        logits, _ = actor_net.apply(p, obs_b)
                        logits = jnp.asarray(logits)
                        logp = jax.nn.log_softmax(logits)
                        chosen = logp[jnp.arange(mb_size), act_b]
                        pg = -(adv_b * chosen).mean()                   # PLAIN policy gradient
                        ent = -(jax.nn.softmax(logits) * logp).sum(-1).mean()
                        return pg - ent_coef * ent, (pg, ent)

                    (al, (pg, ent)), ag = jax.value_and_grad(actor_loss, has_aux=True)(a_st.params)
                    a_st = a_st.apply_gradients(grads=ag)

                    def critic_loss(p):
                        q = critic_net.apply(p, cin_b)
                        q_taken = q[jnp.arange(mb_size), act_b]
                        return jnp.mean((q_taken - tgt_b) ** 2)

                    cl, cg = jax.value_and_grad(critic_loss)(c_st.params)
                    c_st = c_st.apply_gradients(grads=cg)
                    return (a_st, c_st), {"pg_loss": pg, "entropy": ent, "critic_loss": cl}

                (a_st, c_st), m = jax.lax.scan(_mb, (a_st, c_st), batches)
                return (a_st, c_st, key), m

            (a_st, c_st, key), metrics = jax.lax.scan(_epoch, (a_st, c_st, key), None, n_epochs)

            # ---- hard target-network update every tgt_every updates ----------
            tp = jax.lax.cond(((upd_idx + 1) % tgt_every) == 0,
                              lambda _: c_st.params, lambda _: tp, None)

            runner_state = RunnerState(a_st, c_st, tp, env_state, obs, key)
            info = {"mean_episode_return": traj.reward.mean(),
                    "pg_loss": metrics["pg_loss"].mean(),
                    "entropy": metrics["entropy"].mean(),
                    "critic_loss": metrics["critic_loss"].mean()}
            return runner_state, info

        runner_state, hist = jax.lax.scan(_update_step, runner_state, jnp.arange(n_updates))
        return runner_state, hist

    return jax.jit(train)


# ---------------------------------------------------------------------------
# CLI (mirrors ippo_scan; adds --target-update-steps, --critic-hidden)
# ---------------------------------------------------------------------------

def _make_env(family, variant, view_size=7, ball_coef=0.01, max_steps=256, goal_rows=None):
    if family == "af":
        from jaxmarl_worker.environments.american_football_jax import AmericanFootballJAX
        return AmericanFootballJAX(variant=variant, view_size=view_size, ball_coef=ball_coef,
                                   max_steps=max_steps, goal_rows=goal_rows)
    elif family == "bb":
        from jaxmarl_worker.environments.basketball_jax import BasketballJAX
        return BasketballJAX(variant=variant, view_size=view_size, ball_coef=ball_coef,
                             max_steps=max_steps, goal_rows=goal_rows)
    elif family == "soccer":
        from jaxmarl_worker.environments.soccer_jax import SoccerJAX
        return SoccerJAX(variant=variant, view_size=view_size, ball_coef=ball_coef,
                         max_steps=max_steps, goal_rows=goal_rows)
    elif family == "coop_mining":
        from jaxmarl_worker.environments.coop_mining_jax import CoopMiningJAX
        return CoopMiningJAX(max_steps=max_steps)
    else:
        raise ValueError(f"Unknown env family: {family}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--training-type",   type=bool,  default=True,
                   help="Must be True - enforces training_reward parameter usage")
    p.add_argument("--training-reward", type=str,   default="zero-sum",
                   choices=["zero-sum", "cooperative-no-opponent", "general-sum"],
                   help="Reward mode: zero-sum (equal teams), cooperative-no-opponent (any teams), general-sum (equal teams)")
    p.add_argument("--variant", type=str, default="2v2")
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--total-updates", type=int, default=20000)
    p.add_argument("--n-envs", type=int, default=256)
    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=1, help="COMA is on-policy plain PG: keep 1")
    p.add_argument("--n-minibatches", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=256, help="actor hidden (deploy-compatible)")
    p.add_argument("--critic-hidden", type=int, default=128, help="COMA critic hidden (ref uses 64)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--ent-coef", type=float, default=0.0, help="reference COMA uses no entropy bonus")
    p.add_argument("--target-update-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--tensorboard", action="store_true")
    p.add_argument("--view-size", type=int, default=7)
    p.add_argument("--ball-approach-coef", type=float, default=0.01)
    p.add_argument("--max-steps", type=int, default=256)
    p.add_argument("--goal-rows", type=int, nargs='+', default=None)
    return p.parse_args()


def main():
    args = parse_args()
    assert args.training_type == True, "training_type must be True"
    assert args.training_reward in ["zero-sum", "cooperative-no-opponent", "general-sum"], \
        f"training_reward must be one of: zero-sum, cooperative-no-opponent, general-sum, got {args.training_reward}"
    assert "soccer" in ["af", "bb", "soccer"], "env_family must be 'af', 'bb', or 'soccer', got soccer"
    assert args.max_steps == 256, f"Episode length must be exactly 256 steps, got {args.max_steps}"

    # Validate team sizes based on training_reward
    try:
        team_a, team_b = map(int, args.variant.split('v'))
    except (ValueError, IndexError):
        raise ValueError(f"Invalid variant format: {args.variant}. Expected format: NvM (e.g., 2v2)")

    if args.training_reward in ["zero-sum", "general-sum"]:
        assert team_a == team_b, \
            f"Variant {args.variant}: {args.training_reward} requires equal team sizes, got {team_a}v{team_b}"

    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints" / args.training_reward / f"soccer-{args.variant}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    env = _make_env("soccer", args.variant, view_size=args.view_size,
                    ball_coef=args.ball_approach_coef, max_steps=args.max_steps, goal_rows=args.goal_rows)
    print(f"[COMA-scan] soccer-{args.variant}  n_agents={env.num_agents}  "
          f"n_envs={args.n_envs}  n_steps={args.n_steps}  total_updates={args.total_updates}  "
          f"target_every={args.target_update_steps}")
    print(f"  total env steps = {args.total_updates * args.n_envs * args.n_steps * env.num_agents:,}")

    config = {
        "ENV": env, "N_ENVS": args.n_envs, "N_STEPS": args.n_steps, "N_EPOCHS": args.n_epochs,
        "N_MINIBATCHES": args.n_minibatches, "GAMMA": args.gamma, "ENT_COEF": args.ent_coef,
        "HIDDEN_DIM": args.hidden_dim, "CRITIC_HIDDEN": args.critic_hidden,
        "TOTAL_UPDATES": args.total_updates, "LR": args.lr,
        "TARGET_UPDATE_STEPS": args.target_update_steps,
    }

    tb_writer = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    train_fn = make_train(config)
    print("[COMA-scan] JIT-compiling...", flush=True)
    t0 = time.time()
    runner_state, metrics = jax.block_until_ready(train_fn(jax.random.PRNGKey(args.seed)))
    t1 = time.time()
    steps = args.total_updates * args.n_envs * args.n_steps * env.num_agents
    print(f"[COMA-scan] done in {t1-t0:.1f}s  ({steps/(t1-t0):,.0f} steps/sec)")

    ep = np.array(metrics["mean_episode_return"])
    print(f"  final ep_ret (last 50): {ep[-50:].mean():.4f}")
    print(f"  pg_loss={float(metrics['pg_loss'][-1].mean()):.4f}  "
          f"entropy={float(metrics['entropy'][-1].mean()):.4f}  "
          f"critic_loss={float(metrics['critic_loss'][-1].mean()):.4f}")

    if tb_writer:
        pl = np.array(metrics["pg_loss"]).reshape(args.total_updates, -1).mean(-1)
        en = np.array(metrics["entropy"]).reshape(args.total_updates, -1).mean(-1)
        cl = np.array(metrics["critic_loss"]).reshape(args.total_updates, -1).mean(-1)
        for i, (er, p_, e_, c_) in enumerate(zip(ep, pl, en, cl)):
            step = i * args.n_envs * args.n_steps * env.num_agents
            tb_writer.add_scalar("train/episode_return", float(er), step)
            tb_writer.add_scalar("train/pg_loss", float(p_), step)
            tb_writer.add_scalar("train/entropy", float(e_), step)
            tb_writer.add_scalar("train/critic_loss", float(c_), step)
        tb_writer.close()

    leaves, _ = jax.tree_util.tree_flatten(
        jax.tree_util.tree_map(np.array, runner_state.actor_state.params))
    final_path = ckpt_dir / "final.npz"
    np.savez(str(final_path), *leaves)
    print(f"[COMA-scan] actor checkpoint -> {final_path}")


if __name__ == "__main__":
    main()
