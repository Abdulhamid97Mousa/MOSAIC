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
    python -m jaxmarl_worker.algorithms.socialjax.coop_mining.coma_cnn_4_agents_scan_socialjax
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
    tgt_every  = cfg["TARGET_UPDATE_STEPS"]

    actor_net  = _ActorCritic(n_actions)
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
        target_params = critic_params   # hard-copy target at init

        # Optimizers (faithful COMA uses separate actor/critic schedules)
        def _sched(count):
            return cfg["LR"] * (1.0 - count / (n_updates * max(cfg["UPDATE_EPOCHS"], 1)))

        lr_fn     = _sched if cfg["ANNEAL_LR"] else cfg["LR"]
        actor_tx  = optax.chain(optax.clip_by_global_norm(cfg["MAX_GRAD_NORM"]),
                                 optax.adam(lr_fn, eps=1e-5))
        critic_tx = optax.chain(optax.clip_by_global_norm(cfg["MAX_GRAD_NORM"]),
                                 optax.adam(lr_fn, eps=1e-5))
        actor_state  = TrainState.create(apply_fn=actor_net.apply,  params=actor_params,  tx=actor_tx)
        critic_state = TrainState.create(apply_fn=critic_net.apply, params=critic_params, tx=critic_tx)

        # Env reset
        rng, rk = jax.random.split(rng)
        obsv, env_state = jax.vmap(env.reset)(jax.random.split(rk, n_envs))

        runner = (actor_state, critic_state, target_params, env_state, obsv, 0, rng)

        def _update_step(runner, upd_idx):
            # ---- Rollout ---------------------------------------------------
            def _env_step(runner, _):
                a_st, c_st, tp, env_state, last_obs, update_step, rng = runner

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
                return (a_st, c_st, tp, env_state, obsv, update_step, rng), tr

            runner, traj = jax.lax.scan(_env_step, runner, None, n_steps)
            a_st, c_st, tp, env_state, last_obs, update_step, rng = runner

            T, E, n = n_steps, n_envs, n_agents

            # Reshape traj data to (T, E, n, ...) layout
            # traj.obs: (T, n*E, H, W, C) arranged as [a0_e0,..,a0_e{E-1}, a1_e0,..]
            obs_TE  = traj.obs.reshape(T, n, E, *obs_shape).transpose(0, 2, 1, 3, 4, 5)
            # obs_TE: (T, E, n, H, W, C)
            act_TE  = traj.action.reshape(T, n, E).transpose(0, 2, 1)  # (T, E, n)
            rew_TE  = traj.reward.reshape(T, n, E).transpose(0, 2, 1)  # (T, E, n)
            done_TE = traj.done.reshape(T, n, E)[:, 0, :]              # (T, E)

            B = T * E
            obs_B  = obs_TE.reshape(B, n, *obs_shape)    # (B, n, H, W, C)
            act_B  = act_TE.reshape(B, n)                # (B, n)
            rew_B  = rew_TE.reshape(B, n)                # (B, n)
            done_B = done_TE.reshape(B)                  # (B,)

            # ---- Counterfactual advantage (from target critic) --------------
            obs_c, act_m, ids = _build_critic_args(obs_B, act_B)  # (B*n, ...) each
            qT = critic_net.apply(tp, obs_c, act_m, ids).reshape(B, n, n_actions)

            act_oh   = jax.nn.one_hot(act_B, n_actions)          # (B, n, A)
            q_taken  = jnp.sum(qT * act_oh, axis=-1)             # (B, n) from target

            logits_all, _ = actor_net.apply(
                a_st.params, obs_B.reshape(B * n, *obs_shape))
            pi = jax.nn.softmax(logits_all).reshape(B, n, n_actions)
            baseline  = jnp.sum(pi * qT, axis=-1)               # (B, n)
            advantage = jax.lax.stop_gradient(q_taken - baseline)

            # ---- TD(0) target (target-net bootstrap at last step) -----------
            last_obs_en  = last_obs  # (E, n, H, W, C) already correct layout
            last_logits, _ = actor_net.apply(
                a_st.params, last_obs_en.reshape(E * n, *obs_shape))
            rng, bkey = jax.random.split(rng)
            last_act = jax.random.categorical(bkey, last_logits).reshape(E, n)
            obs_cl, act_ml, ids_l = _build_critic_args(last_obs_en, last_act)
            q_last = critic_net.apply(tp, obs_cl, act_ml, ids_l).reshape(E, n, n_actions)
            q_taken_last = jnp.sum(q_last * jax.nn.one_hot(last_act, n_actions), axis=-1)  # (E, n)

            # next Q: shift q_taken along T axis; last timestep uses bootstrap
            q_next = jnp.concatenate([
                q_taken.reshape(T, E, n)[1:],          # (T-1, E, n)
                q_taken_last[None],                     # (1, E, n)
            ], axis=0).reshape(B, n)                   # (T*E, n)

            done_ba  = jnp.broadcast_to(done_B[:, None], (B, n))
            td_target = jax.lax.stop_gradient(rew_B + cfg["GAMMA"] * (1.0 - done_ba) * q_next)

            # ---- Single on-policy update pass (faithful: one epoch) ---------
            mb_sz = B // cfg["NUM_MINIBATCHES"]

            def _update_epoch(carry, _):
                a_st, c_st, rng = carry
                rng, sk = jax.random.split(rng)
                perm = jax.random.permutation(sk, B)
                # Permute over (T*E) keeping n agents together
                obs_p  = obs_B[perm]      # (B, n, H, W, C)
                act_p  = act_B[perm]      # (B, n)
                adv_p  = advantage[perm]  # (B, n)
                tgt_p  = td_target[perm]  # (B, n)

                def _mb(carry, batch):
                    a_st, c_st = carry
                    obs_b, act_b, adv_b, tgt_b = batch  # each (mb_sz, n, ...)

                    # Actor update: plain PG (no clip)
                    obs_flat = obs_b.reshape(mb_sz * n, *obs_shape)
                    act_flat = act_b.reshape(mb_sz * n)
                    adv_flat = adv_b.reshape(mb_sz * n)

                    def actor_loss(params):
                        logits, _ = actor_net.apply(params, obs_flat)
                        logp = jax.nn.log_softmax(logits)
                        chosen = logp[jnp.arange(mb_sz * n), act_flat]
                        pg = -(adv_flat * chosen).mean()
                        ent = -(jax.nn.softmax(logits) * logp).sum(-1).mean()
                        return pg - cfg["ENT_COEF"] * ent, (pg, ent)

                    (al, (pg, ent)), ag = jax.value_and_grad(actor_loss, has_aux=True)(a_st.params)
                    a_st = a_st.apply_gradients(grads=ag)

                    # Critic update: TD(0)
                    obs_c_b, am_b, id_b = _build_critic_args(obs_b, act_b)
                    tgt_flat = tgt_b.reshape(mb_sz * n)
                    act_all_flat = act_b.reshape(mb_sz * n)

                    def critic_loss(params):
                        q = critic_net.apply(params, obs_c_b, am_b, id_b)  # (mb*n, A)
                        q_tk = jnp.sum(q * jax.nn.one_hot(act_all_flat, n_actions), axis=-1)
                        return jnp.mean((q_tk - tgt_flat) ** 2)

                    cl, cg = jax.value_and_grad(critic_loss)(c_st.params)
                    c_st = c_st.apply_gradients(grads=cg)

                    return (a_st, c_st), {"pg_loss": pg, "entropy": ent, "critic_loss": cl}

                minibatches = (
                    obs_p.reshape(cfg["NUM_MINIBATCHES"], mb_sz, n, *obs_shape),
                    act_p.reshape(cfg["NUM_MINIBATCHES"], mb_sz, n),
                    adv_p.reshape(cfg["NUM_MINIBATCHES"], mb_sz, n),
                    tgt_p.reshape(cfg["NUM_MINIBATCHES"], mb_sz, n),
                )
                (a_st, c_st), m = jax.lax.scan(_mb, (a_st, c_st), minibatches)
                return (a_st, c_st, rng), m

            (a_st, c_st, rng), metrics = jax.lax.scan(
                _update_epoch, (a_st, c_st, rng), None, cfg["UPDATE_EPOCHS"])

            # Hard target update every tgt_every steps
            tp = jax.lax.cond(
                ((upd_idx + 1) % tgt_every) == 0,
                lambda _: c_st.params,
                lambda _: tp, None)

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

            jax.debug.callback(_log, (update_step, metric_info,
                                      metrics["pg_loss"].mean(),
                                      metrics["entropy"].mean(),
                                      metrics["critic_loss"].mean()))

            runner = (a_st, c_st, tp, env_state, last_obs, update_step, rng)
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
        / "var" / "trainer" / "socialjax" / "coop_mining" / "COMA_CNN"
    )
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir",             default=_default_out)
    p.add_argument("--training-type",   type=bool,  default=True,
                   help="Must be True - enforces training_reward parameter usage")
    p.add_argument("--training-reward", type=str,   default="cooperative-no-opponent",
                   choices=["cooperative-no-opponent"],
                   help="SocialJax environments are always cooperative (no-opponent)")
    p.add_argument("--num-agents",,          type=int,   default=4,   help="Default: 4 (SocialJax paper)")
    p.add_argument("--num-envs",            type=int,   default=256)
    p.add_argument("--total-timesteps",     type=float, default=3e8, help="300M for paper-scale run")
    p.add_argument("--num-steps",           type=int,   default=1000)
    p.add_argument("--update-epochs",       type=int,   default=1,   help="Faithful COMA: 1 on-policy pass")
    p.add_argument("--num-minibatches",     type=int,   default=500)
    p.add_argument("--critic-hidden",       type=int,   default=128)
    p.add_argument("--target-update-steps", type=int,   default=200)
    p.add_argument("--lr",                  type=float, default=5e-4)
    p.add_argument("--gamma",               type=float, default=0.99)
    p.add_argument("--ent-coef",            type=float, default=0.0, help="Reference COMA uses 0")
    p.add_argument("--max-grad-norm",       type=float, default=0.5)
    p.add_argument("--seed",                type=int,   default=30)
    p.add_argument("--shared-rewards",      action="store_true", default=True)
    p.add_argument("--no-shared-rewards",   dest="shared_rewards", action="store_false")
    p.add_argument("--tensorboard",         action="store_true", default=False,
                   help="Write training curves to tensorboard/")
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
        "NUM_ENVS":           args.num_envs,
        "NUM_STEPS":          args.num_steps,
        "TOTAL_TIMESTEPS":    int(args.total_timesteps),
        "UPDATE_EPOCHS":      args.update_epochs,
        "NUM_MINIBATCHES":    args.num_minibatches,
        "CRITIC_HIDDEN":      args.critic_hidden,
        "TARGET_UPDATE_STEPS": args.target_update_steps,
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
    print(f"  Counterfactual PG (faithful Foerster 2018)  "
          f"target_update_every={args.target_update_steps}  "
          f"critic_hidden={args.critic_hidden}")
    print(f"GPU: {jax.devices()}")

    train_fn = make_train(cfg)
    train_jit = jax.jit(train_fn)

    t0  = time.time()
    out = train_jit(jax.random.PRNGKey(args.seed))
    dt  = time.time() - t0
    print(f"\nTraining done in {dt/60:.1f} min")

    run_dir = Path(args.run_dir)
    final_params = out["runner_state"][0].params   # actor_state.params
    save_checkpoint(final_params, run_dir / "checkpoints" / "final.npz")

    metrics = out["metrics"]
    last_ret = float(jax.tree.map(lambda x: x[-1].mean(), metrics).get("mean_episode_return", 0))
    print(f"Final episode return: {last_ret:.3f}")

    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter as _SW
        _tb = _SW(log_dir=str(run_dir / "tensorboard"))
        import numpy as _np
        rets = _np.array(metrics["mean_episode_return"]).ravel()
        step_size = cfg["NUM_ENVS"] * cfg["NUM_STEPS"]
        for _i, _er in enumerate(rets):
            _tb.add_scalar("train/episode_return", float(_er), _i * step_size)
        _tb.close()


if __name__ == "__main__":
    main()
