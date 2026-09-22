"""HAPPO — Heterogeneous-Agent PPO (GPU-accelerated via jax.lax.scan).

Architecture:
  - Shared centralised critic (global state = all agents' obs concatenated)
  - Shared actor with agent one-hot in obs (EPyMARL obs_agent_id)
  - Sequential policy updates per PPO epoch:
      * Agents updated in a random order σ(1), σ(2), ..., σ(n)
      * Agent σ(1) uses standard GAE advantage
      * Agent σ(m) uses: A_σ(m) *= prod_{k<m} r_σ(k)
        where r_σ(k) = pi_new(a_σ(k)|o_σ(k)) / pi_old(a_σ(k)|o_σ(k))

References:
  Kuba et al. (2021) — Trust Region Policy Optimisation in Multi-Agent RL
"""

import os
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

import re as _re
import time
from pathlib import Path
from typing import NamedTuple, Dict

import chex
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from omegaconf import OmegaConf
from jaxmarl_worker.algorithms.mosaic_multigrid.structured_configs import TrainConfig, register
from jaxmarl_worker.environments.wrappers import AgentIDWrapper


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    action_dim: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        return nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)


class CentralizedCritic(nn.Module):
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        return jnp.squeeze(
            nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x), axis=-1
        )


# ---------------------------------------------------------------------------
# Trajectory storage
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    obs:      chex.Array  # (N_ENVS, N_AGENTS, OBS_DIM + N_AGENTS)  obs + agent one-hot
    action:   chex.Array  # (N_ENVS, N_AGENTS)
    log_prob: chex.Array  # (N_ENVS, N_AGENTS)
    value:    chex.Array  # (N_ENVS, N_AGENTS)
    reward:   chex.Array  # (N_ENVS, N_AGENTS)
    done:     chex.Array  # (N_ENVS,)


class RunnerState(NamedTuple):
    actor_state:  TrainState
    critic_state: TrainState
    env_state:    object
    obs:          Dict
    key:          chex.PRNGKey


# ---------------------------------------------------------------------------
# make_train
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
    hidden_dim = config["HIDDEN_DIM"]
    n_updates  = config["TOTAL_UPDATES"]
    log_every  = config["LOG_EVERY"]
    lr         = config["LR"]

    obs_dim      = env._obs_dim
    obs_dim_net  = obs_dim
    gs_dim       = n_agents * obs_dim           # critic: raw obs concatenated
    total_per_ag = n_steps * n_envs             # samples per agent per rollout
    mb_per_ag    = total_per_ag // n_mb

    actor  = Actor(action_dim=8, hidden_dim=hidden_dim)
    critic = CentralizedCritic(hidden_dim=hidden_dim)

    def _make_tx():
        lr_sched = optax.linear_schedule(lr, lr / 10, n_updates * n_epochs)
        return optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr_sched, eps=1e-5))

    def init_runner_state(key: chex.PRNGKey) -> RunnerState:
        key, ak, ck = jax.random.split(key, 3)
        a_params = actor.init(ak, jnp.zeros((1, obs_dim_net)))
        c_params = critic.init(ck, jnp.zeros((1, gs_dim)))
        a_state  = TrainState.create(apply_fn=actor.apply,  params=a_params, tx=_make_tx())
        c_state  = TrainState.create(apply_fn=critic.apply, params=c_params, tx=_make_tx())

        key, rk = jax.random.split(key)
        obs, env_state = jax.vmap(env.reset)(jax.random.split(rk, n_envs))
        return RunnerState(a_state, c_state, env_state, obs, key)

    def _update_step(runner_state: RunnerState, _):

        def _env_step(runner_state: RunnerState, _):
            a_st, c_st, env_state, obs, key = runner_state

            obs_arr  = jnp.stack([obs[f"agent_{i}"] for i in range(n_agents)], axis=1)
            gs       = obs_arr.reshape(n_envs, gs_dim)
            obs_flat = obs_arr.reshape(n_envs * n_agents, obs_dim_net)

            logits = jnp.asarray(actor.apply(a_st.params, obs_flat))
            key, k = jax.random.split(key)
            acts   = jax.random.categorical(k, logits)
            logp   = jax.nn.log_softmax(logits)[jnp.arange(n_envs * n_agents), acts]
            val_e  = jnp.asarray(critic.apply(c_st.params, gs))
            val    = jnp.broadcast_to(val_e[:, None], (n_envs, n_agents))

            acts_2d      = acts.reshape(n_envs, n_agents)
            actions_dict = {f"agent_{i}": acts_2d[:, i] for i in range(n_agents)}
            key, sk  = jax.random.split(key)
            next_obs, next_env_state, rewards, dones, _ = jax.vmap(env.step)(
                jax.random.split(sk, n_envs), env_state, actions_dict
            )
            reward_arr = jnp.stack([rewards[f"agent_{i}"] for i in range(n_agents)], axis=1)
            transition  = Transition(
                obs=obs_arr, action=acts_2d,
                log_prob=logp.reshape(n_envs, n_agents),
                value=val, reward=reward_arr, done=dones["__all__"],
            )
            return RunnerState(a_st, c_st, next_env_state, next_obs, key), transition

        runner_state, traj = jax.lax.scan(_env_step, runner_state, None, n_steps)
        a_st, c_st, env_state, obs, key = runner_state

        # Bootstrap last value (critic uses raw obs, no agent ID)
        obs_arr  = jnp.stack([obs[f"agent_{i}"] for i in range(n_agents)], axis=1)
        gs       = obs_arr.reshape(n_envs, gs_dim)
        lv_e     = jnp.asarray(critic.apply(c_st.params, gs))
        last_val = jnp.broadcast_to(lv_e[:, None], (n_envs, n_agents))

        # GAE
        def _gae_step(carry, t: Transition):
            last_gae, next_val = carry
            done_ba = jnp.broadcast_to(t.done[:, None], (n_envs, n_agents))
            delta   = t.reward + gamma * next_val * (1.0 - done_ba) - t.value
            gae     = delta + gamma * gae_lam * (1.0 - done_ba) * last_gae
            return (gae, t.value), gae

        _, advantages = jax.lax.scan(
            _gae_step, (jnp.zeros((n_envs, n_agents)), last_val),
            traj, reverse=True, unroll=16,
        )
        targets = advantages + traj.value

        # Per-agent data: (N_AGENTS, total_per_ag, ...)
        ag_obs  = jnp.transpose(traj.obs, (2, 0, 1, 3)).reshape(n_agents, total_per_ag, obs_dim_net)
        ag_act  = jnp.transpose(traj.action,   (2, 0, 1)).reshape(n_agents, total_per_ag)
        ag_logp = jnp.transpose(traj.log_prob, (2, 0, 1)).reshape(n_agents, total_per_ag)
        ag_adv  = jnp.transpose(advantages,    (2, 0, 1)).reshape(n_agents, total_per_ag)
        ag_ret  = jnp.transpose(targets,        (2, 0, 1)).reshape(n_agents, total_per_ag)
        ag_val  = jnp.transpose(traj.value,    (2, 0, 1)).reshape(n_agents, total_per_ag)
        # Global state for critic (raw obs, no agent ID)
        gs_flat = traj.obs[..., :obs_dim].reshape(n_steps * n_envs, n_agents, obs_dim) \
                                          .reshape(n_steps * n_envs, gs_dim)

        def _epoch(carry, _):
            a_st, c_st, key = carry

            key, sk = jax.random.split(key)
            agent_order = jax.random.permutation(sk, n_agents)

            # Critic update: clipped VF loss over all agents' data
            key, sk = jax.random.split(key)
            all_ret_flat = ag_ret.reshape(n_agents * total_per_ag)
            all_val_flat = ag_val.reshape(n_agents * total_per_ag)
            perm_c   = jax.random.permutation(sk, n_agents * total_per_ag)
            gs_tiled = jnp.repeat(gs_flat, n_agents, axis=0)

            def _critic_mb(c_st, batch):
                gs_b, ret_b, val_b = batch
                def critic_loss_fn(params):
                    v        = jnp.asarray(critic.apply(params, gs_b))
                    v_clip   = val_b + jnp.clip(v - val_b, -clip_eps, clip_eps)
                    vf_loss  = 0.5 * jnp.maximum(
                        jnp.square(v - ret_b),
                        jnp.square(v_clip - ret_b),
                    ).mean()
                    return vf_coef * vf_loss
                cl, cg = jax.value_and_grad(critic_loss_fn)(c_st.params)
                return c_st.apply_gradients(grads=cg), cl

            total_c  = n_agents * total_per_ag
            mb_c     = total_c // n_mb
            s_gs_c   = gs_tiled[perm_c].reshape(n_mb, mb_c, gs_dim)
            s_ret_c  = all_ret_flat[perm_c].reshape(n_mb, mb_c)
            s_val_c  = all_val_flat[perm_c].reshape(n_mb, mb_c)
            c_st, cl_all = jax.lax.scan(_critic_mb, c_st, (s_gs_c, s_ret_c, s_val_c))

            # Sequential actor update over agents in random order
            def _agent_update(carry, agent_idx):
                a_st, cum_ratio, key = carry

                obs_i  = ag_obs[agent_idx]
                act_i  = ag_act[agent_idx]
                logp_i = ag_logp[agent_idx]
                adv_i  = ag_adv[agent_idx] * cum_ratio

                key, sk = jax.random.split(key)
                perm    = jax.random.permutation(sk, total_per_ag)
                s_obs   = obs_i[perm];  s_act  = act_i[perm]
                s_logp  = logp_i[perm]; s_adv  = adv_i[perm]

                mb_obs_i  = s_obs.reshape(n_mb, mb_per_ag, obs_dim_net)
                mb_act_i  = s_act.reshape(n_mb, mb_per_ag)
                mb_logp_i = s_logp.reshape(n_mb, mb_per_ag)
                mb_adv_i  = s_adv.reshape(n_mb, mb_per_ag)

                def _mb_actor(a_st, batch):
                    obs_b, act_b, logp_b, adv_b = batch
                    def actor_loss_fn(params):
                        logits    = jnp.asarray(actor.apply(params, obs_b))
                        log_probs = jax.nn.log_softmax(logits)
                        logprob   = log_probs[jnp.arange(mb_per_ag), act_b]
                        ratio     = jnp.exp(logprob - logp_b)
                        adv_n     = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)
                        pg_loss   = jnp.maximum(
                            -adv_n * ratio,
                            -adv_n * jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
                        ).mean()
                        entropy   = -(jax.nn.softmax(logits) * log_probs).sum(-1).mean()
                        return pg_loss - ent_coef * entropy, entropy
                    (al, ent), ag = jax.value_and_grad(actor_loss_fn, has_aux=True)(a_st.params)
                    return a_st.apply_gradients(grads=ag), {"actor_loss": al, "entropy": ent}

                a_st, mb_m = jax.lax.scan(
                    _mb_actor, a_st, (mb_obs_i, mb_act_i, mb_logp_i, mb_adv_i))

                # IS ratio for this agent (needed by subsequent agents)
                logits_new = jnp.asarray(actor.apply(a_st.params, obs_i))
                logp_new   = jax.nn.log_softmax(logits_new)[jnp.arange(total_per_ag), act_i]
                ratio_i    = jnp.exp(logp_new - logp_i)
                metrics    = {k: v.mean() for k, v in mb_m.items()}
                return (a_st, cum_ratio * ratio_i, key), metrics

            init_cum_ratio = jnp.ones(total_per_ag)
            (a_st, _, key), ag_metrics = jax.lax.scan(
                _agent_update, (a_st, init_cum_ratio, key), agent_order
            )

            epoch_metrics = {
                "actor_loss":  ag_metrics["actor_loss"].mean(),
                "critic_loss": cl_all.mean(),
                "entropy":     ag_metrics["entropy"].mean(),
            }
            return (a_st, c_st, key), epoch_metrics

        (a_st, c_st, key), raw = jax.lax.scan(_epoch, (a_st, c_st, key), None, n_epochs)
        metrics = {k: v.mean() for k, v in raw.items()}

        runner_state = RunnerState(a_st, c_st, env_state, obs, key)
        info = {
            "mean_episode_return": traj.reward.mean(),
            "actor_loss":          metrics["actor_loss"],
            "critic_loss":         metrics["critic_loss"],
            "entropy":             metrics["entropy"],
        }
        return runner_state, info

    def train_chunk(runner_state: RunnerState):
        runner_state, metrics_history = jax.lax.scan(
            _update_step, runner_state, None, log_every
        )
        return runner_state, metrics_history

    return jax.jit(init_runner_state), jax.jit(train_chunk)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_env(family: str, variant: str, view_size: int = 7, ball_coef: float = 0.01,
              max_steps: int = 256, goal_rows=None):
    if family == "af":
        from jaxmarl_worker.environments.mosaic_multigrid.AF.american_football_jax import AmericanFootballJAX
        return AmericanFootballJAX(variant=variant, view_size=view_size, ball_coef=ball_coef,
                                   max_steps=max_steps, goal_rows=goal_rows)
    elif family == "bb":
        from jaxmarl_worker.environments.mosaic_multigrid.BB.basketball_jax import BasketballJAX
        return BasketballJAX(variant=variant, view_size=view_size, ball_coef=ball_coef,
                             max_steps=max_steps, goal_rows=goal_rows)
    elif family == "soccer":
        from jaxmarl_worker.environments.mosaic_multigrid.S.soccer_jax import SoccerJAX
        return SoccerJAX(variant=variant, view_size=view_size, ball_coef=ball_coef,
                         max_steps=max_steps, goal_rows=goal_rows)
    elif family == "coop_mining":
        from jaxmarl_worker.environments.coop_mining_jax import CoopMiningJAX
        return CoopMiningJAX(max_steps=max_steps)
    else:
        raise ValueError(f"Unknown env family: {family}")


register()


@hydra.main(version_base=None, config_path="conf", config_name="happo_config")
def main(cfg: TrainConfig) -> None:
    assert cfg.training_reward in ["zero-sum", "cooperative-no-opponent", "general-sum"]

    m = _re.search(r'(\d+)v(\d+)', cfg.env.variant)
    if m is None:
        raise ValueError(f"Invalid variant: {cfg.env.variant}. Expected NvM (e.g., G-2v0, 2v2)")
    team_a, team_b = int(m.group(1)), int(m.group(2))
    if cfg.training_reward in ["zero-sum", "general-sum"]:
        assert team_a == team_b, \
            f"Variant {cfg.env.variant}: {cfg.training_reward} requires equal teams"

    run_dir  = Path(cfg.run_dir)
    ckpt_dir = run_dir / "checkpoints" / f"{cfg.env.family}-{cfg.env.variant}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    env = _make_env(cfg.env.family, cfg.env.variant, view_size=cfg.env.view_size,
                    ball_coef=cfg.env.ball_approach_coef, max_steps=cfg.env.max_steps,
                    goal_rows=OmegaConf.to_container(cfg.env.goal_rows) if cfg.env.goal_rows is not None else None)
    if cfg.agent_id.enabled:
        env = AgentIDWrapper(env)
    print(f"[HAPPO-scan] {cfg.env.family}-{cfg.env.variant}  "
          f"n_agents={env.num_agents}  n_envs={cfg.ppo.n_envs}  "
          f"n_steps={cfg.ppo.n_steps}  total_updates={cfg.total_updates}  "
          f"obs_dim={env._obs_dim}  obs_dim_net={env._obs_dim}  "
          f"gs_dim={env.num_agents * env._obs_dim}")
    print(f"  total env steps = "
          f"{cfg.total_updates * cfg.ppo.n_envs * cfg.ppo.n_steps * env.num_agents:,}")

    train_config = {
        "ENV":           env,
        "N_ENVS":        cfg.ppo.n_envs,
        "N_STEPS":       cfg.ppo.n_steps,
        "N_EPOCHS":      cfg.ppo.n_epochs,
        "N_MINIBATCHES": cfg.ppo.n_minibatches,
        "GAMMA":         cfg.ppo.gamma,
        "GAE_LAMBDA":    cfg.ppo.gae_lambda,
        "CLIP_EPS":      cfg.ppo.clip_eps,
        "VF_COEF":       cfg.ppo.vf_coef,
        "ENT_COEF":      cfg.ppo.ent_coef,
        "HIDDEN_DIM":    cfg.ppo.hidden_dim,
        "TOTAL_UPDATES": cfg.total_updates,
        "LOG_EVERY":     cfg.log_every,
        "LR":            cfg.ppo.lr,
    }
    assert cfg.total_updates % cfg.log_every == 0, \
        f"total_updates ({cfg.total_updates}) must be divisible by log_every ({cfg.log_every})"
    n_chunks = cfg.total_updates // cfg.log_every

    tb_writer = None
    if cfg.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    init_fn, train_chunk_fn = make_train(train_config)
    print("[HAPPO-scan] JIT-compiling...", flush=True)
    t0  = time.time()
    key = jax.random.PRNGKey(cfg.seed)
    runner_state = jax.block_until_ready(init_fn(key))

    ep_rets_chunks, al_chunks, cl_chunks, ent_chunks = [], [], [], []
    for chunk_idx in range(n_chunks):
        runner_state, chunk_metrics = jax.block_until_ready(train_chunk_fn(runner_state))
        ep_ret_c = np.array(chunk_metrics["mean_episode_return"])
        al_c     = np.array(chunk_metrics["actor_loss"])
        cl_c     = np.array(chunk_metrics["critic_loss"])
        ent_c    = np.array(chunk_metrics["entropy"])
        ep_rets_chunks.append(ep_ret_c); al_chunks.append(al_c)
        cl_chunks.append(cl_c);          ent_chunks.append(ent_c)

        if tb_writer:
            for i in range(cfg.log_every):
                update_idx = chunk_idx * cfg.log_every + i
                step = update_idx * cfg.ppo.n_envs * cfg.ppo.n_steps * env.num_agents
                tb_writer.add_scalar("train/episode_return", float(ep_ret_c[i]), step)
                tb_writer.add_scalar("train/actor_loss",     float(al_c[i]),     step)
                tb_writer.add_scalar("train/critic_loss",    float(cl_c[i]),     step)
                tb_writer.add_scalar("train/entropy",        float(ent_c[i]),    step)
            tb_writer.flush()

        elapsed = time.time() - t0
        updates_done = (chunk_idx + 1) * cfg.log_every
        sps = updates_done * cfg.ppo.n_envs * cfg.ppo.n_steps * env.num_agents / elapsed
        print(f"[HAPPO-scan] chunk {chunk_idx + 1}/{n_chunks}  "
              f"updates={updates_done}/{cfg.total_updates}  "
              f"ep_ret={ep_ret_c.mean():.4f}  actor_loss={al_c.mean():.4f}  "
              f"critic_loss={cl_c.mean():.4f}  entropy={ent_c.mean():.4f}  "
              f"({sps:,.0f} steps/sec, {elapsed:.1f}s elapsed)", flush=True)

    t1  = time.time()
    total_steps = cfg.total_updates * cfg.ppo.n_envs * cfg.ppo.n_steps * env.num_agents
    print(f"[HAPPO-scan] Done in {t1 - t0:.1f}s  ({total_steps / (t1-t0):,.0f} steps/sec)")
    ep_rets = np.concatenate(ep_rets_chunks)
    print(f"  final ep_ret (last 50): {ep_rets[-50:].mean():.4f}")

    if tb_writer:
        tb_writer.add_hparams(
            hparam_dict={
                "alg":                 "HAPPO",
                "env/sport":           cfg.env.family,
                "env/variant":         cfg.env.variant,
                "env/n_agents":        env.num_agents,
                "env/obs_dim":         env._obs_dim,
                "env/obs_dim_net":     env._obs_dim,
                "env/view_size":       cfg.env.view_size,
                "train/reward":        cfg.training_reward,
                "train/n_envs":        cfg.ppo.n_envs,
                "train/n_steps":       cfg.ppo.n_steps,
                "train/total_updates": cfg.total_updates,
                "net/hidden_dim":      cfg.ppo.hidden_dim,
                "net/n_epochs":        cfg.ppo.n_epochs,
                "net/n_minibatches":   cfg.ppo.n_minibatches,
                "opt/lr":              cfg.ppo.lr,
                "opt/gamma":           cfg.ppo.gamma,
                "opt/gae_lambda":      cfg.ppo.gae_lambda,
                "opt/clip_eps":        cfg.ppo.clip_eps,
                "opt/vf_coef":         cfg.ppo.vf_coef,
                "opt/ent_coef":        cfg.ppo.ent_coef,
            },
            metric_dict={"hparam/final_ep_return": float(ep_rets[-50:].mean())},
        )
        tb_writer.close()

    def _leaves(params):
        ls, _ = jax.tree_util.tree_flatten(jax.tree_util.tree_map(np.array, params))
        return ls

    final_path = ckpt_dir / "final.npz"
    np.savez(str(final_path),
             **{f"actor_{i}":  v for i, v in enumerate(_leaves(runner_state.actor_state.params))},
             **{f"critic_{i}": v for i, v in enumerate(_leaves(runner_state.critic_state.params))})
    print(f"[HAPPO-scan] checkpoint → {final_path}")


if __name__ == "__main__":
    main()
