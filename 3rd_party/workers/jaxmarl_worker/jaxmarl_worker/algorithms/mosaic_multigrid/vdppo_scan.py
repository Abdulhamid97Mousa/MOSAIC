"""VDPPO — Value-Decomposition PPO (GPU-accelerated via jax.lax.scan).

Implements VDAC (Value Decomposition Actor-Critic) from:
  "Value Decomposition Multi-Agent Actor-Critic" (Wang et al., AAAI 2021)

Each agent has its own decentralised actor and critic (shared params), both
receiving obs + agent one-hot (EPyMARL obs_agent_id). A mixer (VDN or QMIX)
combines individual V values into a global value function for GAE and critic
learning, propagating gradients back through the mixer.

Key properties:
  - Cooperative credit assignment via value mixing
  - Decentralised execution with local obs only
  - Agent one-hot in actor + individual critic obs (symmetry breaking)
  - Mixer options: VDN (additive) or QMIX (monotonic hypernetworks)
  - PPO2-style clipped value loss

References:
  VDAC: Wang et al. (AAAI 2021)
  QMIX: Rashid et al. (2018)
  VDN: Sunehag et al. (2018)
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
from jaxmarl_worker.algorithms.mosaic_multigrid.structured_configs import VDPPOTrainConfig, register
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


class IndividualCritic(nn.Module):
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


class VDNMixer(nn.Module):
    @nn.compact
    def __call__(self, agent_vs, state=None):
        return agent_vs.sum(axis=-1)


class QMIXMixer(nn.Module):
    n_agents:     int
    hidden_dim:   int = 32
    hyper_hidden: int = 32

    @nn.compact
    def __call__(self, agent_vs, state):
        batch = agent_vs.shape[0]

        w1 = nn.Dense(self.hyper_hidden, kernel_init=orthogonal(np.sqrt(2)))(state)
        w1 = nn.relu(w1)
        w1 = jnp.abs(nn.Dense(self.n_agents * self.hidden_dim)(w1))
        w1 = w1.reshape(batch, self.n_agents, self.hidden_dim)
        b1 = nn.Dense(self.hidden_dim)(state).reshape(batch, 1, self.hidden_dim)

        w2 = nn.Dense(self.hyper_hidden, kernel_init=orthogonal(np.sqrt(2)))(state)
        w2 = nn.relu(w2)
        w2 = jnp.abs(nn.Dense(self.hidden_dim)(w2)).reshape(batch, self.hidden_dim, 1)
        b2 = nn.Dense(self.hyper_hidden, kernel_init=orthogonal(np.sqrt(2)))(state)
        b2 = nn.relu(b2)
        b2 = nn.Dense(1)(b2).reshape(batch, 1, 1)

        qs = agent_vs.reshape(batch, 1, self.n_agents)
        hidden = nn.elu(jnp.matmul(qs, w1) + b1)
        return jnp.matmul(hidden, w2).reshape(batch) + b2.reshape(batch)


# ---------------------------------------------------------------------------
# Trajectory storage
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    obs:       chex.Array  # (N_ENVS, N_AGENTS, OBS_DIM + N_AGENTS)  obs + agent one-hot
    action:    chex.Array  # (N_ENVS, N_AGENTS)
    log_prob:  chex.Array  # (N_ENVS, N_AGENTS)
    ind_value: chex.Array  # (N_ENVS, N_AGENTS)  individual critic outputs
    mix_value: chex.Array  # (N_ENVS,)            mixed global value
    reward:    chex.Array  # (N_ENVS, N_AGENTS)
    done:      chex.Array  # (N_ENVS,)


class RunnerState(NamedTuple):
    actor_state:  TrainState
    critic_state: TrainState
    mixer_state:  TrainState
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
    mixer_type = config.get("MIXER", "QMIX")
    mix_hidden = config.get("MIX_HIDDEN", 32)
    hyper_hid  = config.get("HYPER_HIDDEN", 32)

    obs_dim       = env._obs_dim
    obs_dim_net   = obs_dim
    gs_dim        = n_agents * obs_dim           # mixer global state: raw obs (no agent ID)
    total_rollout = n_steps * n_envs * n_agents
    mb_size       = total_rollout // n_mb

    actor  = Actor(action_dim=8, hidden_dim=hidden_dim)
    critic = IndividualCritic(hidden_dim=hidden_dim)
    if mixer_type == "VDN":
        mixer = VDNMixer()
    else:
        mixer = QMIXMixer(n_agents=n_agents, hidden_dim=mix_hidden, hyper_hidden=hyper_hid)

    def _make_tx():
        lr_sched = optax.linear_schedule(lr, lr / 10, n_updates * n_epochs)
        return optax.chain(optax.clip_by_global_norm(10.0), optax.adam(lr_sched, eps=1e-5))

    def init_runner_state(key: chex.PRNGKey) -> RunnerState:
        key, ak, ck, mk = jax.random.split(key, 4)
        a_params = actor.init(ak, jnp.zeros((1, obs_dim_net)))
        c_params = critic.init(ck, jnp.zeros((1, obs_dim_net)))
        if mixer_type == "VDN":
            m_params = mixer.init(mk, jnp.zeros((1, n_agents)))
        else:
            m_params = mixer.init(mk, jnp.zeros((1, n_agents)), jnp.zeros((1, gs_dim)))
        a_state = TrainState.create(apply_fn=actor.apply,  params=a_params, tx=_make_tx())
        c_state = TrainState.create(apply_fn=critic.apply, params=c_params, tx=_make_tx())
        m_state = TrainState.create(apply_fn=mixer.apply,  params=m_params, tx=_make_tx())

        key, rk = jax.random.split(key)
        obs, env_state = jax.vmap(env.reset)(jax.random.split(rk, n_envs))
        return RunnerState(a_state, c_state, m_state, env_state, obs, key)

    def _update_step(runner_state: RunnerState, _):

        def _env_step(runner_state: RunnerState, _):
            a_st, c_st, m_st, env_state, obs, key = runner_state

            obs_arr   = jnp.stack([obs[f"agent_{i}"] for i in range(n_agents)], axis=1)
            gs        = obs_arr.reshape(n_envs, gs_dim)
            obs_flat  = obs_arr.reshape(n_envs * n_agents, obs_dim_net)

            logits = jnp.asarray(actor.apply(a_st.params, obs_flat))
            key, k = jax.random.split(key)
            acts   = jax.random.categorical(k, logits)
            logp   = jax.nn.log_softmax(logits)[jnp.arange(n_envs * n_agents), acts]

            ind_v = jnp.asarray(critic.apply(c_st.params, obs_flat)).reshape(n_envs, n_agents)
            if mixer_type == "VDN":
                mix_v = mixer.apply(m_st.params, ind_v)
            else:
                mix_v = mixer.apply(m_st.params, ind_v, gs)

            acts_2d      = acts.reshape(n_envs, n_agents)
            actions_dict = {f"agent_{i}": acts_2d[:, i] for i in range(n_agents)}
            key, sk = jax.random.split(key)
            next_obs, next_env_state, rewards, dones, _ = jax.vmap(env.step)(
                jax.random.split(sk, n_envs), env_state, actions_dict
            )
            reward_arr = jnp.stack([rewards[f"agent_{i}"] for i in range(n_agents)], axis=1)
            transition  = Transition(
                obs=obs_arr, action=acts_2d,
                log_prob=logp.reshape(n_envs, n_agents),
                ind_value=ind_v, mix_value=mix_v,
                reward=reward_arr, done=dones["__all__"],
            )
            return RunnerState(a_st, c_st, m_st, next_env_state, next_obs, key), transition

        runner_state, traj = jax.lax.scan(_env_step, runner_state, None, n_steps)
        a_st, c_st, m_st, env_state, obs, key = runner_state

        # Bootstrap last mixed value (raw obs for mixer)
        obs_arr    = jnp.stack([obs[f"agent_{i}"] for i in range(n_agents)], axis=1)
        gs         = obs_arr.reshape(n_envs, gs_dim)
        last_ind_v = jnp.asarray(critic.apply(c_st.params, obs_arr.reshape(n_envs * n_agents, obs_dim_net))).reshape(n_envs, n_agents)
        if mixer_type == "VDN":
            last_mix_v = mixer.apply(m_st.params, last_ind_v)
        else:
            last_mix_v = mixer.apply(m_st.params, last_ind_v, gs)

        # GAE over mixed global value
        def _gae_step(carry, t: Transition):
            last_gae, next_mix = carry
            team_r = t.reward.mean(axis=-1)
            delta  = team_r + gamma * next_mix * (1.0 - t.done) - t.mix_value
            gae    = delta + gamma * gae_lam * (1.0 - t.done) * last_gae
            return (gae, t.mix_value), gae

        _, mix_adv = jax.lax.scan(
            _gae_step, (jnp.zeros(n_envs), last_mix_v),
            traj, reverse=True, unroll=16,
        )
        advantages  = jnp.broadcast_to(mix_adv[:, :, None], (n_steps, n_envs, n_agents))
        mix_targets = mix_adv + traj.mix_value

        # Flatten for minibatch training
        flat_obs     = traj.obs.reshape(total_rollout, obs_dim_net)       # actor + critic (with ID)
        flat_gs      = jnp.repeat(
            traj.obs[..., :obs_dim].reshape(n_steps * n_envs, n_agents, obs_dim)
                                   .reshape(n_steps * n_envs, gs_dim),
            n_agents, axis=0,
        )                                                                   # mixer (raw obs)
        flat_act     = traj.action.reshape(total_rollout)
        flat_logp    = traj.log_prob.reshape(total_rollout)
        flat_adv     = advantages.reshape(total_rollout)
        flat_mix_ret = jnp.repeat(mix_targets.reshape(n_steps * n_envs), n_agents, axis=0)
        flat_mix_val = jnp.repeat(traj.mix_value.reshape(n_steps * n_envs), n_agents, axis=0)

        def _epoch(carry, _):
            a_st, c_st, m_st, key = carry
            key, sk = jax.random.split(key)
            perm = jax.random.permutation(sk, total_rollout)

            s_obs     = flat_obs[perm];     s_act     = flat_act[perm]
            s_logp    = flat_logp[perm];    s_adv     = flat_adv[perm]
            s_gs      = flat_gs[perm];      s_mix_ret = flat_mix_ret[perm]
            s_mix_val = flat_mix_val[perm]

            mb_obs     = s_obs.reshape(n_mb, mb_size, obs_dim_net)
            mb_act     = s_act.reshape(n_mb, mb_size)
            mb_logp    = s_logp.reshape(n_mb, mb_size)
            mb_adv     = s_adv.reshape(n_mb, mb_size)
            mb_gs      = s_gs.reshape(n_mb, mb_size, gs_dim)
            mb_mix_ret = s_mix_ret.reshape(n_mb, mb_size)
            mb_mix_val = s_mix_val.reshape(n_mb, mb_size)

            def _mb(carry, batch):
                a_st, c_st, m_st = carry
                obs_b, act_b, logp_b, adv_b, gs_b, mix_ret_b, mix_val_b = batch

                def actor_loss_fn(params):
                    logits    = jnp.asarray(actor.apply(params, obs_b))
                    log_probs = jax.nn.log_softmax(logits)
                    logprob   = log_probs[jnp.arange(mb_size), act_b]
                    ratio     = jnp.exp(logprob - logp_b)
                    adv_n     = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)
                    pg_loss   = jnp.maximum(
                        -adv_n * ratio,
                        -adv_n * jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
                    ).mean()
                    entropy   = -(jax.nn.softmax(logits) * log_probs).sum(-1).mean()
                    return pg_loss - ent_coef * entropy, (pg_loss, entropy)

                def critic_mixer_loss_fn(c_params, m_params):
                    # Per-agent values from obs (obs already includes agent ID when wrapper is active)
                    ag_raw = gs_b.reshape(mb_size, n_agents, obs_dim)
                    ind_vs = jnp.asarray(critic.apply(
                        c_params, ag_raw.reshape(mb_size * n_agents, obs_dim_net)
                    )).reshape(mb_size, n_agents)

                    if mixer_type == "VDN":
                        mix_vs = mixer.apply(m_params, ind_vs)
                    else:
                        mix_vs = mixer.apply(m_params, ind_vs, gs_b)

                    # Clipped value loss (PPO2 style)
                    mix_v_clip = mix_val_b + jnp.clip(mix_vs - mix_val_b, -clip_eps, clip_eps)
                    vf_loss    = 0.5 * jnp.maximum(
                        jnp.square(mix_vs - mix_ret_b),
                        jnp.square(mix_v_clip - mix_ret_b),
                    ).mean()
                    return vf_coef * vf_loss, vf_loss

                (al, (pgl, ent)), ag = jax.value_and_grad(actor_loss_fn, has_aux=True)(a_st.params)
                (cl, vfl), (cg, mg) = jax.value_and_grad(
                    critic_mixer_loss_fn, argnums=(0, 1), has_aux=True)(c_st.params, m_st.params)

                a_st = a_st.apply_gradients(grads=ag)
                c_st = c_st.apply_gradients(grads=cg)
                m_st = m_st.apply_gradients(grads=mg)
                return (a_st, c_st, m_st), {
                    "actor_loss": al, "critic_loss": cl,
                    "entropy": ent, "pg_loss": pgl, "vf_loss": vfl,
                }

            (a_st, c_st, m_st), mb_m = jax.lax.scan(
                _mb, (a_st, c_st, m_st),
                (mb_obs, mb_act, mb_logp, mb_adv, mb_gs, mb_mix_ret, mb_mix_val),
            )
            return (a_st, c_st, m_st, key), mb_m

        (a_st, c_st, m_st, key), raw = jax.lax.scan(
            _epoch, (a_st, c_st, m_st, key), None, n_epochs)
        metrics = {k: v.mean() for k, v in raw.items()}

        runner_state = RunnerState(a_st, c_st, m_st, env_state, obs, key)
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


@hydra.main(version_base=None, config_path="conf", config_name="vdppo_config")
def main(cfg: VDPPOTrainConfig) -> None:
    assert cfg.training_reward in ["zero-sum", "cooperative-no-opponent", "general-sum"]

    m = _re.search(r'(\d+)v(\d+)', cfg.env.variant)
    if m is None:
        raise ValueError(f"Invalid variant: {cfg.env.variant}. Expected NvM (e.g., G-2v0, 2v2)")
    team_a, team_b = int(m.group(1)), int(m.group(2))
    if cfg.training_reward in ["zero-sum", "general-sum"]:
        assert team_a == team_b, f"Variant {cfg.env.variant}: {cfg.training_reward} requires equal teams"

    run_dir  = Path(cfg.run_dir)
    ckpt_dir = run_dir / "checkpoints" / f"{cfg.env.family}-{cfg.env.variant}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    env = _make_env(cfg.env.family, cfg.env.variant, view_size=cfg.env.view_size,
                    ball_coef=cfg.env.ball_approach_coef, max_steps=cfg.env.max_steps,
                    goal_rows=OmegaConf.to_container(cfg.env.goal_rows) if cfg.env.goal_rows is not None else None)
    if cfg.agent_id.enabled:
        env = AgentIDWrapper(env)
    print(f"[VDPPO-scan] {cfg.env.family}-{cfg.env.variant}  "
          f"n_agents={env.num_agents}  n_envs={cfg.ppo.n_envs}  "
          f"n_steps={cfg.ppo.n_steps}  total_updates={cfg.total_updates}  "
          f"obs_dim={env._obs_dim}  obs_dim_net={env._obs_dim}  "
          f"mixer={cfg.ppo.mixer}")
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
        "MIXER":         cfg.ppo.mixer,
        "MIX_HIDDEN":    cfg.ppo.mix_hidden,
        "HYPER_HIDDEN":  cfg.ppo.hyper_hidden,
    }
    assert cfg.total_updates % cfg.log_every == 0, \
        f"total_updates ({cfg.total_updates}) must be divisible by log_every ({cfg.log_every})"
    n_chunks = cfg.total_updates // cfg.log_every

    tb_writer = None
    if cfg.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    init_fn, train_chunk_fn = make_train(train_config)
    print("[VDPPO-scan] JIT-compiling...", flush=True)
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
        print(f"[VDPPO-scan] chunk {chunk_idx + 1}/{n_chunks}  "
              f"updates={updates_done}/{cfg.total_updates}  "
              f"ep_ret={ep_ret_c.mean():.4f}  actor_loss={al_c.mean():.4f}  "
              f"critic_loss={cl_c.mean():.4f}  entropy={ent_c.mean():.4f}  "
              f"({sps:,.0f} steps/sec, {elapsed:.1f}s elapsed)", flush=True)

    t1  = time.time()
    total_steps = cfg.total_updates * cfg.ppo.n_envs * cfg.ppo.n_steps * env.num_agents
    print(f"[VDPPO-scan] Done in {t1 - t0:.1f}s  ({total_steps / (t1 - t0):,.0f} steps/sec)")
    ep_rets = np.concatenate(ep_rets_chunks)
    print(f"  final ep_ret (last 50): {ep_rets[-50:].mean():.4f}")

    if tb_writer:
        tb_writer.add_hparams(
            hparam_dict={
                "alg":                 "VDPPO",
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
                "net/mixer":           cfg.ppo.mixer,
                "net/mix_hidden":      cfg.ppo.mix_hidden,
                "net/hyper_hidden":    cfg.ppo.hyper_hidden,
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
             **{f"critic_{i}": v for i, v in enumerate(_leaves(runner_state.critic_state.params))},
             **{f"mixer_{i}":  v for i, v in enumerate(_leaves(runner_state.mixer_state.params))})
    print(f"[VDPPO-scan] checkpoint → {final_path}")


if __name__ == "__main__":
    main()
