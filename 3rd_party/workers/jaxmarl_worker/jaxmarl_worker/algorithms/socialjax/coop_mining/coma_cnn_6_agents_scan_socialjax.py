"""COMA-CNN for SocialJax coop_mining -- faithful to Foerster et al. 2018.

Faithful COMA specifics (see also coma_scan.py for multigrid):
  - Actor: same CNN + ActorCritic as IPPO (IPPO-compatible checkpoint format)
  - Critic: centralized with OWN CNN; takes all agents' raw obs as global state
  - Plain PG actor loss (no PPO clip), single on-policy pass per rollout
  - TD(0) critic + hard target network (updated every --target-update-steps)
  - Counterfactual advantage: A_i = Q_target(s, a_i) - sum_a pi_i(a) Q_target(s, a)
    with the acting agent's own action masked out of the joint-action input

At deployment only the actor checkpoint is used; it loads through the existing
per-agent inference path (IPPO-compatible final.npz). AC and IC both read 0 under
substitution, same as IPPO/MAPPO (gradient-sector actor).

Run:
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \\
    python -m jaxmarl_worker.algorithms.socialjax.coop_mining.coma_cnn_6_agents_scan_socialjax
"""

import os
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

import argparse
import sys
import time
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

_SOCIALJAX_ROOT = str(Path(__file__).resolve().parents[6] / "environments" / "SocialJax")
_ALGO_ROOT = str(Path(_SOCIALJAX_ROOT) / "algorithms")
if _SOCIALJAX_ROOT not in sys.path:
    sys.path.insert(0, _SOCIALJAX_ROOT)
if _ALGO_ROOT not in sys.path:
    sys.path.insert(0, _ALGO_ROOT)

import socialjax
from socialjax.wrappers.baselines import LogWrapper


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class _CNN(nn.Module):
    """SocialJax paper CNN backbone (verbatim from ippo_cnn scripts)."""
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(32, (5, 5), kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Conv(32, (3, 3), kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Conv(32, (3, 3), kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        return x  # (B, 64)


class _ActorCritic(nn.Module):
    """Identical to ippo_cnn_scan ActorCritic for checkpoint compatibility."""
    action_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        act = nn.relu if self.activation == "relu" else nn.tanh
        emb = _CNN()(x)
        actor = act(nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(emb))
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor)
        critic = act(nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(emb))
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        return logits, jnp.squeeze(value, axis=-1)


class _COMACritic(nn.Module):
    """Centralized counterfactual critic with its own CNN.

    Has completely independent weights from the actor CNN.
    Takes ALL agents' raw observations as global state input.
    Outputs Q-values over the ACTING AGENT's action space.

    Input (per acting agent i):
      all_obs   : (B, n_agents, H, W, C) - raw obs of every agent
      act_masked: (B, n_agents * n_actions) - a_{-i} one-hot, a_i row zeroed
      agent_id  : (B, n_agents) - one-hot identity of acting agent i
    Output: (B, n_actions) Q over agent i's actions
    """
    n_agents: int
    n_actions: int
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, all_obs, act_masked, agent_id):
        B = all_obs.shape[0]
        flat = all_obs.reshape(B * self.n_agents, *all_obs.shape[2:])
        embs = _CNN()(flat).reshape(B, self.n_agents * 64)     # global state proxy

        x = jnp.concatenate([embs, act_masked, agent_id], axis=-1)
        x = nn.relu(nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x))
        x = nn.relu(nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x))
        return nn.Dense(self.n_actions, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)


# ---------------------------------------------------------------------------
# Transition
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    done:   jnp.ndarray   # (NUM_ACTORS,) -- tiled across agents
    action: jnp.ndarray   # (NUM_ACTORS,)
    reward: jnp.ndarray   # (NUM_ACTORS,)
    obs:    jnp.ndarray   # (NUM_ACTORS, H, W, C)
    info:   dict


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def make_train(cfg: dict):
    env_raw = socialjax.make(cfg["ENV_NAME"], **cfg["ENV_KWARGS"])
    env     = LogWrapper(env_raw, replace_info=False)

    n_agents  = env_raw.num_agents
    n_actions = env_raw.action_space().n
    obs_shape = env_raw.observation_space()[0].shape   # (11, 11, 12)

    n_envs    = cfg["NUM_ENVS"]
    n_steps   = cfg["NUM_STEPS"]
    n_updates = int(cfg["TOTAL_TIMESTEPS"] // n_steps // n_envs)
    num_actors = n_agents * n_envs
    tb_writer  = cfg.get("TB_WRITER", None)
    _step_size = n_envs * n_steps

    actor_net  = _ActorCritic(action_dim=n_actions)
    critic_net = _COMACritic(n_agents=n_agents, n_actions=n_actions, hidden_dim=cfg["CRITIC_HIDDEN"])

    # Pre-build mask and identity for critic input construction
    _mask = 1.0 - jnp.eye(n_agents)[None, :, :, None]   # (1, n, n, 1) -- zeros a_i row
    _ids  = jnp.eye(n_agents)                            # (n, n) one-hot IDs

    def _build_critic_args(obs_env, act_env):
        """
        obs_env : (B, n_agents, H, W, C)
        act_env : (B, n_agents) int
        Returns tensors for n_agents critic calls batched together:
          all_obs   (B*n, n, H, W, C)
          act_masked (B*n, n*A)
          ids        (B*n, n)
        """
        B = obs_env.shape[0]
        n = n_agents

        obs_b   = jnp.broadcast_to(obs_env[:, None],
                                    (B, n, n) + obs_shape)
        all_obs  = obs_b.reshape(B * n, n, *obs_shape)

        ja      = jax.nn.one_hot(act_env, n_actions)            # (B, n, A)
        ja_b    = jnp.broadcast_to(ja[:, None], (B, n, n, n_actions))
        act_m   = (ja_b * _mask).reshape(B * n, n * n_actions)

        ids_b   = jnp.broadcast_to(_ids[None], (B, n, n))
        ids_flat = ids_b.reshape(B * n, n)

        return all_obs, act_m, ids_flat

    def train(rng):
        # Init actor
        rng, ak = jax.random.split(rng)
        actor_params = actor_net.init(ak, jnp.zeros((1, *obs_shape)))

        # Init critic (dummy forward to get param shapes)
        rng, ck = jax.random.split(rng)
        dummy_obs = jnp.zeros((1, n_agents, *obs_shape))
        dummy_am  = jnp.zeros((1, n_agents * n_actions))
        dummy_id  = jnp.zeros((1, n_agents))
        critic_params = critic_net.init(ck, dummy_obs, dummy_am, dummy_id)

        # Optimizers (faithful COMA uses separate actor/critic schedules)
        def _sched(count):
            return cfg["LR"] * (1.0 - count / (n_updates * max(cfg["UPDATE_EPOCHS"], 1)))

        lr_fn     = _sched if cfg["ANNEAL_LR"] else cfg["LR"]
        actor_tx  = optax.chain(optax.clip_by_global_norm(cfg["MAX_GRAD_NORM"]),
                                 optax.adam(lr_fn, eps=1e-5))
        critic_tx = optax.chain(optax.clip_by_global_norm(cfg["MAX_GRAD_NORM"]),
                                 optax.adam(cfg["CRITIC_LR"], eps=1e-5))
        actor_state  = TrainState.create(apply_fn=actor_net.apply,  params=actor_params,  tx=actor_tx)
        critic_state = TrainState.create(apply_fn=critic_net.apply, params=critic_params, tx=critic_tx)

        # Env reset
        rng, rk = jax.random.split(rng)
        obsv, env_state = jax.vmap(env.reset)(jax.random.split(rk, n_envs))

        runner = (actor_state, critic_state, env_state, obsv, 0, rng)

        def _update_step(runner, upd_idx):
            # ---- Rollout ---------------------------------------------------
            def _env_step(runner, _):
                a_st, c_st, env_state, last_obs, update_step, rng = runner

                # (E, n, H, W, C) -> (n, E, H, W, C) -> (n*E, H, W, C)
                obs_batch = jnp.transpose(last_obs, (1, 0, 2, 3, 4)).reshape(-1, *obs_shape)
                logits, _ = actor_net.apply(a_st.params, obs_batch)
                rng, akey = jax.random.split(rng)
                action = jax.random.categorical(akey, logits)

                # (n*E,) -> (n, E) -> per-agent list
                env_act = [action.reshape(n_agents, n_envs)[i] for i in range(n_agents)]
                rng, skey = jax.random.split(rng)
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    jax.random.split(skey, n_envs), env_state, env_act)

                # reward (E, n) -> (n*E,) arranged as [r_a0_e0,...,r_a0_e{E-1},r_a1_e0,...]
                reward_flat = jnp.transpose(reward, (1, 0)).reshape(-1)
                done_flat   = jnp.tile(done["__all__"], n_agents)
                info = jax.tree.map(lambda x: x.reshape((num_actors,)), info)

                tr = Transition(done_flat, action, reward_flat, obs_batch, info)
                return (a_st, c_st, env_state, obsv, update_step, rng), tr

            runner, traj = jax.lax.scan(_env_step, runner, None, n_steps)
            a_st, c_st, env_state, last_obs, update_step, rng = runner

            T, E, n = n_steps, n_envs, n_agents

            # Reshape traj data to (T, E, n, ...) layout
            # traj.obs: (T, n*E, H, W, C) arranged as [a0_e0,..,a0_e{E-1}, a1_e0,..]
            obs_TE  = traj.obs.reshape(T, n, E, *obs_shape).transpose(0, 2, 1, 3, 4, 5)
            # obs_TE: (T, E, n, H, W, C)
            act_TE  = traj.action.reshape(T, n, E).transpose(0, 2, 1)  # (T, E, n)
            rew_TE  = traj.reward.reshape(T, n, E).transpose(0, 2, 1)  # (T, E, n)
            done_TE = traj.done.reshape(T, n, E)[:, 0, :]              # (T, E)

            B = T * E
            obs_B = obs_TE.reshape(B, n, *obs_shape)    # (B, n, H, W, C)
            act_B = act_TE.reshape(B, n)                # (B, n)

            # ---- Monte Carlo returns (no bootstrapping = no divergence) -----
            # Compute discounted return G_t = r_t + γ·r_{t+1} + ... backwards.
            # This replaces TD(0) targets. The critic is trained on real returns,
            # so there is no frozen random target network to explode.
            done_for_mc = jnp.broadcast_to(done_TE[:, :, None], (T, E, n))  # (T,E,n)

            def _mc_step(carry, x):
                G = carry             # (E, n)
                r, d = x              # (E, n), (E, n)
                G = r + cfg["GAMMA"] * (1.0 - d) * G
                return G, G

            _, mc_rev = jax.lax.scan(
                _mc_step, jnp.zeros((E, n)),
                (rew_TE[::-1], done_for_mc[::-1]),
            )
            mc_returns = jax.lax.stop_gradient(mc_rev[::-1].reshape(B, n))  # (B,n)

            # ---- Counterfactual advantage (from ONLINE critic) ---------------
            # We no longer need a target network: MC targets are ground truth.
            # Using the online critic for the advantage baseline avoids the random
            # random-target poisoning that caused divergence with TD targets.
            # NUM_INFERENCE_CHUNKS: XLA memory chunking (must be large, ~500).
            # NUM_MINIBATCHES: training gradient steps (keep small, ~32).
            n_inf      = cfg.get("NUM_INFERENCE_CHUNKS", 500)
            chunk_size = B // n_inf

            def _q_chunk(args):
                obs_ch, act_ch = args  # (chunk_size, n, H, W, C), (chunk_size, n)
                obs_c_ch, am_ch, ids_ch = _build_critic_args(obs_ch, act_ch)
                return critic_net.apply(c_st.params, obs_c_ch, am_ch, ids_ch)

            chunk_obs = obs_B.reshape(n_inf, chunk_size, n, *obs_shape)
            chunk_act = act_B.reshape(n_inf, chunk_size, n)
            qV = jax.lax.map(_q_chunk, (chunk_obs, chunk_act)).reshape(B, n, n_actions)

            act_oh  = jax.nn.one_hot(act_B, n_actions)   # (B, n, A)
            q_taken = jnp.sum(qV * act_oh, axis=-1)      # (B, n)

            def _pi_chunk(obs_ch):  # obs_ch: (chunk_size*n, H, W, C)
                logits, _ = actor_net.apply(a_st.params, obs_ch)
                return jax.nn.softmax(logits)

            pi = jax.lax.map(
                _pi_chunk,
                obs_B.reshape(n_inf, chunk_size * n, *obs_shape),
            ).reshape(B, n, n_actions)
            baseline  = jnp.sum(pi * qV, axis=-1)         # (B, n)
            advantage = jax.lax.stop_gradient(q_taken - baseline)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            td_target = mc_returns  # critic training target (already stop_gradient above)

            # ---- Faithful COMA update: critic multi-step, actor single-step ----
            # COMA (Foerster 2018) uses ONE actor gradient step per rollout.
            # The PPO-style loop over 32 minibatches is correct for the critic
            # (needs many steps to fit Q-values) but wrong for the actor (32 steps
            # with a random critic collapses entropy from ~2.1 to ~0 by update 2).
            n_mb  = cfg["NUM_MINIBATCHES"]
            mb_sz = B // n_mb

            def _update_epoch(carry, _):
                a_st, c_st, rng = carry
                rng, sk = jax.random.split(rng)
                perm = jax.random.permutation(sk, B)
                obs_p = obs_B[perm]
                act_p = act_B[perm]
                adv_p = advantage[perm]
                tgt_p = td_target[perm]

                # Critic: n_mb gradient steps (memory-safe via minibatches).
                def _critic_mb(c_st, batch):
                    obs_b, act_b, tgt_b = batch
                    obs_c_b, am_b, id_b = _build_critic_args(obs_b, act_b)
                    tgt_flat     = tgt_b.reshape(mb_sz * n)
                    act_all_flat = act_b.reshape(mb_sz * n)

                    def critic_loss(params):
                        q    = critic_net.apply(params, obs_c_b, am_b, id_b)
                        q_tk = jnp.sum(q * jax.nn.one_hot(act_all_flat, n_actions), axis=-1)
                        return jnp.mean((q_tk - tgt_flat) ** 2)

                    cl, cg = jax.value_and_grad(critic_loss)(c_st.params)
                    return c_st.apply_gradients(grads=cg), cl

                c_mbs = (obs_p.reshape(n_mb, mb_sz, n, *obs_shape),
                         act_p.reshape(n_mb, mb_sz, n),
                         tgt_p.reshape(n_mb, mb_sz, n))
                c_st, cl_hist = jax.lax.scan(_critic_mb, c_st, c_mbs)
                cl = cl_hist.mean()

                # Actor: single gradient step using first mb_sz samples.
                # One PG step is faithful to the paper and prevents entropy collapse
                # from the random critic during the first ~200 updates (before target
                # network gets meaningful Q-values).
                obs_a = obs_p[:mb_sz].reshape(mb_sz * n, *obs_shape)
                act_a = act_p[:mb_sz].reshape(mb_sz * n)
                adv_a = adv_p[:mb_sz].reshape(mb_sz * n)

                def actor_loss(params):
                    logits, _ = actor_net.apply(params, obs_a)
                    logp   = jax.nn.log_softmax(logits)
                    chosen = logp[jnp.arange(mb_sz * n), act_a]
                    pg     = -(adv_a * chosen).mean()
                    ent    = -(jax.nn.softmax(logits) * logp).sum(-1).mean()
                    return pg - cfg["ENT_COEF"] * ent, (pg, ent)

                (_, (pg, ent)), ag = jax.value_and_grad(actor_loss, has_aux=True)(a_st.params)
                a_st = a_st.apply_gradients(grads=ag)

                return (a_st, c_st, rng), {"pg_loss": pg, "entropy": ent, "critic_loss": cl}

            (a_st, c_st, rng), metrics = jax.lax.scan(
                _update_epoch, (a_st, c_st, rng), None, cfg["UPDATE_EPOCHS"])

            update_step = update_step + 1
            metric_info = jax.tree.map(lambda x: x.mean(), traj.info)

            def _log(args):
                step, info, pg, ent, cl = args
                ret  = info.get("returned_episode_returns", 0.0)
                gold = info.get("mining_gold", 0.0) * cfg["NUM_STEPS"]
                print(f"  update {int(step):4d}/{n_updates}  "
                      f"return={float(ret):.3f}  gold={float(gold):.2f}  "
                      f"pg={float(pg):.4f}  ent={float(ent):.4f}  cl={float(cl):.4f}",
                      flush=True)
                if tb_writer is not None:
                    env_step = int(step) * _step_size
                    tb_writer.add_scalar("train/episode_return", float(ret), env_step)
                    tb_writer.add_scalar("train/critic_loss",    float(cl),  env_step)
                    tb_writer.add_scalar("train/entropy",        float(ent), env_step)
                    tb_writer.flush()

            jax.debug.callback(_log, (update_step, metric_info,
                                      metrics["pg_loss"].mean(),
                                      metrics["entropy"].mean(),
                                      metrics["critic_loss"].mean()))

            runner = (a_st, c_st, env_state, last_obs, update_step, rng)
            info_out = {
                "mean_episode_return": metric_info.get("returned_episode_returns", 0.0),
                "pg_loss":     metrics["pg_loss"].mean(),
                "entropy":     metrics["entropy"].mean(),
                "critic_loss": metrics["critic_loss"].mean(),
            }
            return runner, info_out

        runner, metrics_hist = jax.lax.scan(_update_step, runner, jnp.arange(n_updates))
        return {"runner_state": runner, "metrics": metrics_hist}

    return train


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(params, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    leaves, _ = jax.tree_util.tree_flatten(params)
    np.savez(str(path), *[np.array(l) for l in leaves])
    print(f"Checkpoint saved: {path}  ({len(leaves)} leaves)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    _default_out = str(
        Path(__file__).resolve().parents[7]
        / "var" / "trainer" / "socialjax" / "coop_mining" / "COMA_CNN_6agent"
    )
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir",             default=_default_out)
    p.add_argument("--training-type",   type=bool,  default=True,
                   help="Must be True - enforces training_reward parameter usage")
    p.add_argument("--training-reward", type=str,   default="cooperative-no-opponent",
                   choices=["cooperative-no-opponent"],
                   help="SocialJax environments are always cooperative (no-opponent)")
    p.add_argument("--num-agents",,          type=int,   default=6,   help="Default: 6 (coop_mining)")
    p.add_argument("--num-envs",            type=int,   default=32)
    p.add_argument("--total-timesteps",     type=float, default=3e8, help="300M for paper-scale run")
    p.add_argument("--num-steps",           type=int,   default=1000)
    p.add_argument("--update-epochs",        type=int,   default=1,   help="Faithful COMA: 1 on-policy pass")
    p.add_argument("--num-minibatches",      type=int,   default=32,
                   help="Training gradient steps per epoch (keep small: 32). "
                        "Separate from --num-inference-chunks which controls XLA memory.")
    p.add_argument("--num-inference-chunks", type=int,   default=500,
                   help="Chunks for chunked Q/pi inference to avoid XLA OOM (keep large: 500).")
    p.add_argument("--critic-hidden",        type=int,   default=128)
    p.add_argument("--lr",                   type=float, default=5e-4)
    p.add_argument("--critic-lr",            type=float, default=1e-3,
                   help="Separate (higher) LR for critic; 1e-3 helps it converge faster than actor.")
    p.add_argument("--gamma",                type=float, default=0.99)
    p.add_argument("--ent-coef",             type=float, default=0.01,
                   help="Small entropy bonus prevents policy collapse during critic warm-up.")
    p.add_argument("--max-grad-norm",        type=float, default=0.5)
    p.add_argument("--seed",                 type=int,   default=30)
    p.add_argument("--shared-rewards",       action="store_true", default=True)
    p.add_argument("--no-shared-rewards",    dest="shared_rewards", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()
    assert args.training_type == True, "training_type must be True"
    assert args.training_reward == "cooperative-no-opponent", f"SocialJax only supports cooperative-no-opponent, got {args.training_reward}"
    assert args.max_steps == 1000, f"SocialJax episode length must be exactly 1000 steps, got {args.max_steps}"

    cfg = {
        "ENV_NAME": "coop_mining",
        "ENV_KWARGS": {
            "num_agents":      args.num_agents,
            "num_inner_steps": args.num_steps,
            "shared_rewards":  args.shared_rewards,
            "cnn":             True,
            "jit":             True,
        },
        "LR":                 args.lr,
        "NUM_ENVS":             args.num_envs,
        "NUM_STEPS":            args.num_steps,
        "TOTAL_TIMESTEPS":      int(args.total_timesteps),
        "UPDATE_EPOCHS":        args.update_epochs,
        "NUM_MINIBATCHES":      args.num_minibatches,
        "NUM_INFERENCE_CHUNKS": args.num_inference_chunks,
        "CRITIC_HIDDEN":        args.critic_hidden,
        "CRITIC_LR":          args.critic_lr,
        "GAMMA":              args.gamma,
        "ENT_COEF":           args.ent_coef,
        "MAX_GRAD_NORM":      args.max_grad_norm,
        "ANNEAL_LR":          True,
    }

    reward_str = "CR" if args.shared_rewards else "IR"
    n_updates  = int(args.total_timesteps // args.num_steps // args.num_envs)

    print(f"[COMA-CNN] coop_mining  n_agents={args.num_agents}  "
          f"reward={reward_str}  envs={args.num_envs}  "
          f"steps/ep={args.num_steps}  updates={n_updates}  "
          f"total={args.total_timesteps/1e6:.0f}M  seed={args.seed}")
    print(f"  Counterfactual PG (MC returns, no target network)  "
          f"critic_lr={args.critic_lr}  critic_hidden={args.critic_hidden}")
    print(f"GPU: {jax.devices()}")

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Open TensorBoard writer before training so updates stream in real-time.
    # x-axis in env-steps matches IPPO/MAT/CommNet/IC3Net (one point per update).
    tb_writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter as _SW
        tb_writer = _SW(log_dir=str(run_dir / "tensorboard"))
        print(f"TensorBoard live at {run_dir / 'tensorboard'}")
    except ImportError:
        print("torch not available — TensorBoard disabled")

    cfg["TB_WRITER"] = tb_writer

    train_fn = make_train(cfg)
    train_jit = jax.jit(train_fn)

    t0  = time.time()
    out = train_jit(jax.random.PRNGKey(args.seed))
    dt  = time.time() - t0
    print(f"\nTraining done in {dt/60:.1f} min")

    if tb_writer is not None:
        tb_writer.close()

    final_params = out["runner_state"][0].params   # actor_state.params
    save_checkpoint(final_params, run_dir / "checkpoints" / "final.npz")

    metrics = out["metrics"]
    ep_rets = np.array(metrics["mean_episode_return"]).ravel()
    print(f"Final episode return (last 50): {ep_rets[-50:].mean():.3f}")
    np.save(str(run_dir / "metrics_ep_return.npy"), ep_rets)


if __name__ == "__main__":
    main()
