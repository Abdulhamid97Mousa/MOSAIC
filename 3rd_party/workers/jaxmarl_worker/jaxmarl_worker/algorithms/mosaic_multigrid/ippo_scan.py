"""Full GPU-accelerated IPPO using jax.lax.scan + jax.vmap.

Architecture:
  - jax.vmap(env.reset/step)  → N_ENVS parallel environments on GPU
  - jax.lax.scan              → rollout loop entirely on GPU (no Python)
  - Single shared ActorCritic network (actor + value head)
  - Parameter sharing: all agents share one network; agent one-hot in obs
  - GAE computed via reverse jax.lax.scan
  - PPO minibatch updates via scan over shuffled minibatches

Difference from MAPPO:
  - No centralized critic. Each agent estimates its own value from local obs.
  - One TrainState instead of two (actor + critic merged).

Usage:
  python -m jaxmarl_worker.algorithms.multigrid_sports.ippo_scan \
    --env-family af --variant 2v2 \
    --run-dir /home/hamid/Desktop/Projects/GUI_BDI_RL/var/trainer/AF/IPPO \
    --total-updates 2000 --n-envs 512
"""

import os
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

import re
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
# Network — single actor-critic with split heads
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    action_dim: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x):
        # separate actor trunk (following JaxMARL ippo_ff_mpe.py)
        a = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        a = nn.tanh(a)
        a = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(a)
        a = nn.tanh(a)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(a)
        # separate critic trunk
        c = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        c = nn.tanh(c)
        c = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(c)
        c = nn.tanh(c)
        value = jnp.squeeze(
            nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(c), axis=-1
        )
        return logits, value

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
    train_state: TrainState
    env_state:   any
    obs:         Dict
    key:         chex.PRNGKey

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
    gae_lam   = config["GAE_LAMBDA"]
    clip_eps  = config["CLIP_EPS"]
    vf_coef   = config["VF_COEF"]
    ent_coef  = config["ENT_COEF"]
    hidden_dim = config["HIDDEN_DIM"]
    n_updates = config["TOTAL_UPDATES"]
    log_every = config["LOG_EVERY"]
    lr        = config["LR"]

    obs_dim     = env._obs_dim
    obs_dim_net = obs_dim
    total_per_rollout = n_steps * n_envs * n_agents
    mb_size  = total_per_rollout // n_mb

    net = ActorCritic(action_dim=8, hidden_dim=hidden_dim)

    def init_runner_state(key: chex.PRNGKey) -> RunnerState:
        # ---- initialise network --------------------------------------------
        key, nk = jax.random.split(key)
        dummy_obs = jnp.zeros((1, obs_dim_net))
        params = net.init(nk, dummy_obs)

        lr_sched = optax.linear_schedule(lr, lr / 10, n_updates * n_epochs)
        tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr_sched, eps=1e-5))
        train_state = TrainState.create(apply_fn=net.apply, params=params, tx=tx)

        # ---- initialise environments (vmapped) -----------------------------
        key, reset_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, n_envs)
        obs, env_state = jax.vmap(env.reset)(reset_keys)

        return RunnerState(train_state, env_state, obs, key)

    # ---- one rollout + update step (scanned over log_every per chunk) ------
    def _update_step(runner_state: RunnerState, _):
        # ---- collect N_STEPS transitions --------------------------------
        def _env_step(runner_state: RunnerState, _):
            train_state, env_state, obs, key = runner_state

            obs_arr = jnp.stack(
                [obs[f"agent_{i}"] for i in range(n_agents)], axis=1
            )  # (N_ENVS, N_AGENTS, OBS_DIM)
            obs_flat = obs_arr.reshape(n_envs * n_agents, obs_dim_net)
            logits, values_flat = net.apply(train_state.params, obs_flat)
            logits = jnp.asarray(logits)
            values_flat = jnp.asarray(values_flat)

            key, action_key = jax.random.split(key)
            actions_flat = jax.random.categorical(action_key, logits)
            log_probs_flat = jax.nn.log_softmax(logits)[
                jnp.arange(n_envs * n_agents), actions_flat
            ]

            actions_arr = actions_flat.reshape(n_envs, n_agents)
            actions_dict = {f"agent_{i}": actions_arr[:, i] for i in range(n_agents)}
            key, step_key = jax.random.split(key)
            step_keys = jax.random.split(step_key, n_envs)
            next_obs, next_env_state, rewards, dones, _ = jax.vmap(env.step)(
                step_keys, env_state, actions_dict
            )

            reward_arr = jnp.stack(
                [rewards[f"agent_{i}"] for i in range(n_agents)], axis=1
            )

            transition = Transition(
                obs      = obs_arr,
                action   = actions_arr,
                log_prob = log_probs_flat.reshape(n_envs, n_agents),
                value    = values_flat.reshape(n_envs, n_agents),
                reward   = reward_arr,
                done     = dones["__all__"],
            )
            return RunnerState(train_state, next_env_state, next_obs, key), transition

        runner_state, traj = jax.lax.scan(_env_step, runner_state, None, n_steps)

        # ---- bootstrap last value --------------------------------------
        train_state, env_state, obs, key = runner_state
        obs_arr = jnp.stack([obs[f"agent_{i}"] for i in range(n_agents)], axis=1)
        obs_flat = obs_arr.reshape(n_envs * n_agents, obs_dim_net)
        _, last_val_flat = net.apply(train_state.params, obs_flat)
        last_val = jnp.asarray(last_val_flat).reshape(n_envs, n_agents)

        # ---- GAE via reverse scan --------------------------------------
        def _gae_step(carry, transition: Transition):
            last_gae, next_val = carry
            done_ba = jnp.broadcast_to(transition.done[:, None], (n_envs, n_agents))
            delta   = transition.reward + gamma * next_val * (1.0 - done_ba) - transition.value
            gae     = delta + gamma * gae_lam * (1.0 - done_ba) * last_gae
            return (gae, transition.value), gae

        _, advantages = jax.lax.scan(
            _gae_step,
            (jnp.zeros((n_envs, n_agents)), last_val),
            traj,
            reverse=True,
            unroll=16,
        )
        targets = advantages + traj.value

        # ---- PPO update ------------------------------------------------
        flat_obs  = traj.obs.reshape(total_per_rollout, obs_dim_net)
        flat_act  = traj.action.reshape(total_per_rollout)
        flat_logp = traj.log_prob.reshape(total_per_rollout)
        flat_adv  = advantages.reshape(total_per_rollout)
        flat_ret  = targets.reshape(total_per_rollout)
        flat_val  = traj.value.reshape(total_per_rollout)  # old values for clipped VF loss

        def _update_epoch(carry, _):
            train_state, key = carry
            key, subkey = jax.random.split(key)
            perm = jax.random.permutation(subkey, total_per_rollout)

            s_obs  = flat_obs[perm];  s_act  = flat_act[perm]
            s_logp = flat_logp[perm]; s_adv  = flat_adv[perm]
            s_ret  = flat_ret[perm];  s_val  = flat_val[perm]

            mb_obs  = s_obs.reshape(n_mb, mb_size, obs_dim_net)
            mb_act  = s_act.reshape(n_mb, mb_size)
            mb_logp = s_logp.reshape(n_mb, mb_size)
            mb_adv  = s_adv.reshape(n_mb, mb_size)
            mb_ret  = s_ret.reshape(n_mb, mb_size)
            mb_val  = s_val.reshape(n_mb, mb_size)

            def _update_minibatch(carry, batch):
                train_st = carry
                obs_b, act_b, logp_b, adv_b, ret_b, val_b = batch

                def loss_fn(params):
                    logits, vals = net.apply(params, obs_b)
                    logits = jnp.asarray(logits)
                    vals   = jnp.asarray(vals)
                    log_probs = jax.nn.log_softmax(logits)
                    logprob   = log_probs[jnp.arange(mb_size), act_b]
                    ratio     = jnp.exp(logprob - logp_b)
                    adv_n     = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)
                    pg_loss   = jnp.maximum(
                        -adv_n * ratio,
                        -adv_n * jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
                    ).mean()
                    entropy   = -(jax.nn.softmax(logits) * log_probs).sum(-1).mean()
                    # clipped value loss (PPO2 / JaxMARL style)
                    val_clipped = val_b + jnp.clip(vals - val_b, -clip_eps, clip_eps)
                    vf_loss     = 0.5 * jnp.maximum(
                        jnp.square(vals - ret_b),
                        jnp.square(val_clipped - ret_b),
                    ).mean()
                    total     = pg_loss + vf_coef * vf_loss - ent_coef * entropy
                    return total, (pg_loss, vf_loss, entropy)

                (total_loss, (pg_l, vf_l, ent)), grads = jax.value_and_grad(
                    loss_fn, has_aux=True)(train_st.params)
                train_st = train_st.apply_gradients(grads=grads)
                metrics  = {"total_loss": total_loss, "pg_loss": pg_l,
                            "vf_loss": vf_l, "entropy": ent}
                return train_st, metrics

            train_state, mb_metrics = jax.lax.scan(
                _update_minibatch,
                train_state,
                (mb_obs, mb_act, mb_logp, mb_adv, mb_ret, mb_val),
            )
            return (train_state, key), mb_metrics

        (train_state, key), metrics = jax.lax.scan(
            _update_epoch,
            (train_state, key),
            None,
            n_epochs,
        )

        runner_state = RunnerState(train_state, env_state, obs, key)
        info = {
            "mean_episode_return": traj.reward.mean(),
            "total_loss": metrics["total_loss"].mean(),
            "pg_loss":    metrics["pg_loss"].mean(),
            "vf_loss":    metrics["vf_loss"].mean(),
            "entropy":    metrics["entropy"].mean(),
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


@hydra.main(version_base=None, config_path="conf", config_name="ippo_config")
def main(cfg: TrainConfig) -> None:
    assert cfg.training_reward in ["zero-sum", "cooperative-no-opponent", "general-sum"], \
        f"training_reward must be one of: zero-sum, cooperative-no-opponent, general-sum, got {cfg.training_reward}"
    assert cfg.env.max_steps == 256, f"Episode length must be exactly 256 steps, got {cfg.env.max_steps}"

    m = re.search(r'(\d+)v(\d+)', cfg.env.variant)
    if m is None:
        raise ValueError(f"Invalid variant: {cfg.env.variant}. Expected NvM (e.g., G-2v0, 2v2)")
    team_a, team_b = int(m.group(1)), int(m.group(2))
    if cfg.training_reward in ["zero-sum", "general-sum"]:
        assert team_a == team_b, \
            f"Variant {cfg.env.variant}: {cfg.training_reward} requires equal team sizes, got {team_a}v{team_b}"

    run_dir  = Path(cfg.run_dir)
    ckpt_dir = run_dir / "checkpoints" / f"{cfg.env.family}-{cfg.env.variant}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    env = _make_env(cfg.env.family, cfg.env.variant, view_size=cfg.env.view_size,
                    ball_coef=cfg.env.ball_approach_coef, max_steps=cfg.env.max_steps,
                    goal_rows=OmegaConf.to_container(cfg.env.goal_rows) if cfg.env.goal_rows is not None else None)
    if cfg.agent_id.enabled:
        env = AgentIDWrapper(env)
    print(f"[IPPO-scan] {cfg.env.family}-{cfg.env.variant}  "
          f"n_agents={env.num_agents}  n_envs={cfg.ppo.n_envs}  "
          f"n_steps={cfg.ppo.n_steps}  total_updates={cfg.total_updates}")
    print(f"  total env steps = {cfg.total_updates * cfg.ppo.n_envs * cfg.ppo.n_steps * env.num_agents:,}")

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

    print("[IPPO-scan] JIT-compiling training function...", flush=True)
    t0 = time.time()
    key = jax.random.PRNGKey(cfg.seed)
    runner_state = jax.block_until_ready(init_fn(key))

    ep_rets_chunks, tl_chunks, pl_chunks, vl_chunks, ent_chunks = [], [], [], [], []
    for chunk_idx in range(n_chunks):
        runner_state, chunk_metrics = jax.block_until_ready(train_chunk_fn(runner_state))
        ep_ret_c = np.array(chunk_metrics["mean_episode_return"])
        tl_c     = np.array(chunk_metrics["total_loss"])
        pl_c     = np.array(chunk_metrics["pg_loss"])
        vl_c     = np.array(chunk_metrics["vf_loss"])
        ent_c    = np.array(chunk_metrics["entropy"])
        ep_rets_chunks.append(ep_ret_c); tl_chunks.append(tl_c)
        pl_chunks.append(pl_c); vl_chunks.append(vl_c); ent_chunks.append(ent_c)

        if tb_writer:
            for i in range(cfg.log_every):
                update_idx = chunk_idx * cfg.log_every + i
                step = update_idx * cfg.ppo.n_envs * cfg.ppo.n_steps * env.num_agents
                tb_writer.add_scalar("train/episode_return", float(ep_ret_c[i]), step)
                tb_writer.add_scalar("train/total_loss",     float(tl_c[i]),     step)
                tb_writer.add_scalar("train/pg_loss",        float(pl_c[i]),     step)
                tb_writer.add_scalar("train/vf_loss",        float(vl_c[i]),     step)
                tb_writer.add_scalar("train/entropy",        float(ent_c[i]),    step)
            tb_writer.flush()

        elapsed = time.time() - t0
        updates_done = (chunk_idx + 1) * cfg.log_every
        sps = updates_done * cfg.ppo.n_envs * cfg.ppo.n_steps * env.num_agents / elapsed
        print(f"[IPPO-scan] chunk {chunk_idx + 1}/{n_chunks}  "
              f"updates={updates_done}/{cfg.total_updates}  "
              f"ep_ret={ep_ret_c.mean():.4f}  total_loss={tl_c.mean():.4f}  "
              f"pg_loss={pl_c.mean():.4f}  vf_loss={vl_c.mean():.4f}  entropy={ent_c.mean():.4f}  "
              f"({sps:,.0f} steps/sec, {elapsed:.1f}s elapsed)", flush=True)

    t1 = time.time()
    total_steps = cfg.total_updates * cfg.ppo.n_envs * cfg.ppo.n_steps * env.num_agents
    sps = total_steps / (t1 - t0)
    print(f"[IPPO-scan] Training complete in {t1-t0:.1f}s  ({sps:,.0f} steps/sec)")

    ep_rets = np.concatenate(ep_rets_chunks)
    print(f"  final ep_ret (last 50 updates): {ep_rets[-50:].mean():.4f}")
    print(f"  total_loss: {float(tl_chunks[-1][-1]):.4f}")
    print(f"  pg_loss:    {float(pl_chunks[-1][-1]):.4f}")
    print(f"  vf_loss:    {float(vl_chunks[-1][-1]):.4f}")
    print(f"  entropy:    {float(ent_chunks[-1][-1]):.4f}")

    if tb_writer:
        tb_writer.add_hparams(
            hparam_dict={
                "alg":                 "IPPO",
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

    leaves, _ = jax.tree_util.tree_flatten(
        jax.tree_util.tree_map(np.array, runner_state.train_state.params))
    final_path = ckpt_dir / "final.npz"
    np.savez(str(final_path), *leaves)
    print(f"[IPPO-scan] checkpoint → {final_path}")


if __name__ == "__main__":
    main()
