"""Full GPU-accelerated MAT (Multi-Agent Transformer) training via jax.lax.scan.

Mirrors ippo_scan.py exactly (same JAX-native envs, jax.vmap(env.step), lax.scan
rollout, GAE, PPO) but the actor is the MAT transformer (faithful JAX port of
Wen et al. 2022). Unlike per-agent IPPO/MAPPO, MAT processes the JOINT agent set:
  - rollout: mat_get_actions  -> autoregressive action sampling over agents
  - update:  mat_eval_actions -> teacher-forced parallel evaluation (keeps agent dim)

This trainer uses MATActor with standard autoregressive action sampling for training and evaluation.

Usage (smoke first, then full):
  python -m jaxmarl_worker.algorithms.multigrid_sports.mat_scan --env-family soccer --variant 2v2 \
    --run-dir var/trainer/multigrid_sports/S_m256_20k/MAT/VIEWSIZE7/indagobs/adversarial/V2/2v2 \
    --total-updates 50 --n-envs 64        # smoke
  # then --total-updates 20000 --n-envs 256 for the real run
"""
import os
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
# Do not pre-grab ~75% of GPU memory; allocate on demand (the MAT is tiny).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import time
from pathlib import Path
from typing import NamedTuple, Dict

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from jaxmarl_worker.networks.mat_actor import MATActor, mat_get_actions, mat_eval_actions

ACTION_DIM = 8  # fixed for all multigrid_sports envs


# ---------------------------------------------------------------------------
# ValueNorm (faithful JAX port of PKU-MARL mat/utils/valuenorm.py)
# The MAT value head outputs NORMALIZED values; returns are normalized to match
# in the value loss. GAE de-normalizes the value to real scale for bootstrapping.
# Pure functions over a (running_mean, running_mean_sq, debiasing_term) pytree.
# ---------------------------------------------------------------------------

class ValueNormState(NamedTuple):
    running_mean:    chex.Array   # (1,)
    running_mean_sq: chex.Array   # (1,)
    debiasing_term:  chex.Array   # (1,)


_VN_BETA = 0.99999
_VN_EPS  = 1e-5


def vn_init() -> ValueNormState:
    return ValueNormState(jnp.zeros(1), jnp.zeros(1), jnp.zeros(1))


def _vn_mean_var(state: ValueNormState):
    debias = jnp.maximum(state.debiasing_term, _VN_EPS)
    mean   = state.running_mean / debias
    mean_sq = state.running_mean_sq / debias
    var    = jnp.maximum(mean_sq - mean ** 2, 1e-2)   # PKU clamps var to >= 1e-2
    return mean, var


def vn_normalize(state: ValueNormState, x):
    mean, var = _vn_mean_var(state)
    return (x - mean) / jnp.sqrt(var)


def vn_denormalize(state: ValueNormState, x):
    mean, var = _vn_mean_var(state)
    return x * jnp.sqrt(var) + mean


def vn_update(state: ValueNormState, x) -> ValueNormState:
    """EMA update over a batch of returns x (any shape). Matches PKU ValueNorm.update
    with per_element_update=False (weight = beta)."""
    batch_mean    = jnp.mean(x)
    batch_sq_mean = jnp.mean(x ** 2)
    w = _VN_BETA
    return ValueNormState(
        running_mean    = state.running_mean * w + batch_mean    * (1.0 - w),
        running_mean_sq = state.running_mean_sq * w + batch_sq_mean * (1.0 - w),
        debiasing_term  = state.debiasing_term * w + (1.0 - w),
    )


class Transition(NamedTuple):
    obs:      chex.Array  # (N_ENVS, N_AGENTS, OBS_DIM)
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
    vn:          ValueNormState


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
    n_updates = config["TOTAL_UPDATES"]
    lr        = config["LR"]

    obs_dim = env._obs_dim
    n_joint = n_steps * n_envs           # joint (step,env) samples; agent dim kept
    mb_size = n_joint // n_mb
    rew_shaping_horizon = config.get("REW_SHAPING_HORIZON", 0)  # 0 = disabled
    action_dim = config.get("ACTION_DIM", ACTION_DIM)  # allow per-env override

    _net_factory = config.get("NETWORK_FACTORY")
    if _net_factory is None:
        _net_factory = lambda **kw: MATActor(action_dim=kw["action_dim"], n_agent=kw["n_agent"],
                                             n_embd=kw["n_embd"], n_head=kw["n_head"], n_block=kw["n_block"])
    net = _net_factory(action_dim=action_dim, n_agent=n_agents,
                       n_embd=config["N_EMBD"], n_head=config["N_HEAD"], n_block=config["N_BLOCK"])

    def _empty_shifted(b):
        return jnp.zeros((b, n_agents, action_dim + 1)).at[:, 0, 0].set(1.0)

    def train(key: chex.PRNGKey):
        key, nk = jax.random.split(key)
        dummy_obs = jnp.zeros((1, n_agents, obs_dim))
        dummy_sa = jnp.zeros((1, n_agents, action_dim + 1))
        params = net.init(nk, dummy_obs, dummy_sa)

        lr_sched = optax.linear_schedule(lr, lr / 10, n_updates * n_epochs)
        tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr_sched, eps=1e-5))
        train_state = TrainState.create(apply_fn=net.apply, params=params, tx=tx)

        key, reset_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, n_envs)
        obs, env_state = jax.vmap(env.reset)(reset_keys)
        runner_state = RunnerState(train_state, env_state, obs, key, vn_init())

        _log_cb = config.get("LOG_CALLBACK")

        def _update_step(runner_state: RunnerState, step_idx):
            def _env_step(runner_state: RunnerState, _):
                train_state, env_state, obs, key, vn = runner_state
                obs_arr = jnp.stack([obs[f"agent_{i}"] for i in range(n_agents)], axis=1)

                key, ak = jax.random.split(key)
                actions_arr, logp_arr, values_arr = mat_get_actions(
                    net.apply, train_state.params, obs_arr, ak, action_dim)

                actions_dict = {f"agent_{i}": actions_arr[:, i] for i in range(n_agents)}
                key, step_key = jax.random.split(key)
                step_keys = jax.random.split(step_key, n_envs)
                next_obs, next_env_state, rewards, dones, info = jax.vmap(env.step)(
                    step_keys, env_state, actions_dict)

                reward_arr = jnp.stack([rewards[f"agent_{i}"] for i in range(n_agents)], axis=1)
                if rew_shaping_horizon > 0:
                    anneal_w = jnp.maximum(0.0, 1.0 - step_idx * n_envs * n_steps / rew_shaping_horizon)
                    reward_arr = reward_arr + info["clean_action_info"] * anneal_w
                transition = Transition(
                    obs=obs_arr, action=actions_arr, log_prob=logp_arr,
                    value=values_arr, reward=reward_arr, done=dones["__all__"])
                return RunnerState(train_state, next_env_state, next_obs, key, vn), transition

            runner_state, traj = jax.lax.scan(_env_step, runner_state, None, n_steps)

            # ---- bootstrap last value (encoder value head; action-independent) ----
            train_state, env_state, obs, key, vn = runner_state
            obs_arr = jnp.stack([obs[f"agent_{i}"] for i in range(n_agents)], axis=1)
            _, last_val = net.apply(train_state.params, obs_arr, _empty_shifted(n_envs))

            # ---- GAE in REAL return scale: de-normalize the (normalized) net values ----
            traj_val_real = vn_denormalize(vn, traj.value)          # (n_steps, n_envs, n_agents)
            last_val_real = vn_denormalize(vn, last_val)            # (n_envs, n_agents)

            def _gae_step(carry, transition_real):
                last_gae, next_val = carry
                reward, done, val_real = transition_real
                done_ba = jnp.broadcast_to(done[:, None], (n_envs, n_agents))
                delta = reward + gamma * next_val * (1.0 - done_ba) - val_real
                gae = delta + gamma * gae_lam * (1.0 - done_ba) * last_gae
                return (gae, val_real), gae

            _, advantages = jax.lax.scan(
                _gae_step, (jnp.zeros((n_envs, n_agents)), last_val_real),
                (traj.reward, traj.done, traj_val_real),
                reverse=True, unroll=16)
            targets = advantages + traj_val_real                       # real-scale returns

            # ---- flatten over (step, env), KEEP agent dim ----
            flat_obs  = traj.obs.reshape(n_joint, n_agents, obs_dim)
            flat_act  = traj.action.reshape(n_joint, n_agents)
            flat_logp = traj.log_prob.reshape(n_joint, n_agents)
            flat_adv  = advantages.reshape(n_joint, n_agents)
            flat_ret  = targets.reshape(n_joint, n_agents)
            flat_val  = traj.value.reshape(n_joint, n_agents)          # NORMALIZED old value (for clip)

            def _update_epoch(carry, _):
                train_state, vn, key = carry
                key, subkey = jax.random.split(key)
                perm = jax.random.permutation(subkey, n_joint)
                s_obs, s_act = flat_obs[perm], flat_act[perm]
                s_logp, s_adv, s_ret, s_val = flat_logp[perm], flat_adv[perm], flat_ret[perm], flat_val[perm]

                mb_obs  = s_obs.reshape(n_mb, mb_size, n_agents, obs_dim)
                mb_act  = s_act.reshape(n_mb, mb_size, n_agents)
                mb_logp = s_logp.reshape(n_mb, mb_size, n_agents)
                mb_adv  = s_adv.reshape(n_mb, mb_size, n_agents)
                mb_ret  = s_ret.reshape(n_mb, mb_size, n_agents)
                mb_val  = s_val.reshape(n_mb, mb_size, n_agents)        # normalized old value

                def _update_minibatch(carry, batch):
                    train_st, vn_st = carry
                    obs_b, act_b, logp_b, adv_b, ret_b, val_b = batch

                    # update ValueNorm with this minibatch's real-scale returns
                    vn_new = vn_update(vn_st, ret_b)
                    ret_norm = vn_normalize(vn_new, ret_b)              # normalized targets

                    def loss_fn(params):
                        logp, entropy, vals = mat_eval_actions(
                            net.apply, params, obs_b, act_b, action_dim)  # vals: NORMALIZED
                        ratio = jnp.exp(logp - logp_b)
                        adv_n = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)
                        pg_loss = jnp.maximum(
                            -adv_n * ratio,
                            -adv_n * jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)).mean()
                        # value loss in normalized space, with value clipping (PKU cal_value_loss)
                        val_clipped = val_b + (vals - val_b).clip(-clip_eps, clip_eps)
                        v_loss = 0.5 * jnp.maximum((vals - ret_norm) ** 2,
                                                   (val_clipped - ret_norm) ** 2).mean()
                        ent = entropy.mean()
                        total = pg_loss + vf_coef * v_loss - ent_coef * ent
                        return total, (pg_loss, v_loss, ent)

                    (total_loss, (pg_l, vf_l, ent)), grads = jax.value_and_grad(
                        loss_fn, has_aux=True)(train_st.params)
                    train_st = train_st.apply_gradients(grads=grads)
                    return (train_st, vn_new), {"total_loss": total_loss, "pg_loss": pg_l,
                                                "vf_loss": vf_l, "entropy": ent}

                (train_state, vn), mb_metrics = jax.lax.scan(
                    _update_minibatch, (train_state, vn),
                    (mb_obs, mb_act, mb_logp, mb_adv, mb_ret, mb_val))
                return (train_state, vn, key), mb_metrics

            (train_state, vn, key), metrics = jax.lax.scan(
                _update_epoch, (train_state, vn, key), None, n_epochs)

            runner_state = RunnerState(train_state, env_state, obs, key, vn)
            info = {
                "mean_episode_return": traj.reward.mean(),
                "total_loss": metrics["total_loss"].mean(),
                "pg_loss":    metrics["pg_loss"].mean(),
                "vf_loss":    metrics["vf_loss"].mean(),
                "entropy":    metrics["entropy"].mean(),
            }
            if _log_cb is not None:
                jax.debug.callback(
                    _log_cb,
                    step_idx + 1,
                    info["mean_episode_return"],
                    info["total_loss"],
                )
            return runner_state, info

        runner_state, metrics_history = jax.lax.scan(
            _update_step, runner_state, jnp.arange(n_updates))
        return runner_state, metrics_history

    return jax.jit(train)


def _make_env(family: str, variant: str, view_size: int = 7, ball_coef: float = 0.01,
              max_steps: int = 256, goal_rows=None, cooperative: bool = False):
    if family == "af":
        from jaxmarl_worker.environments.american_football_jax import AmericanFootballJAX
        return AmericanFootballJAX(variant=variant, view_size=view_size, ball_coef=ball_coef,
                                   max_steps=max_steps, goal_rows=goal_rows, cooperative=cooperative)
    elif family == "bb":
        from jaxmarl_worker.environments.basketball_jax import BasketballJAX
        return BasketballJAX(variant=variant, view_size=view_size, ball_coef=ball_coef,
                             max_steps=max_steps, goal_rows=goal_rows, cooperative=cooperative)
    elif family == "soccer":
        from jaxmarl_worker.environments.soccer_jax import SoccerJAX
        return SoccerJAX(variant=variant, view_size=view_size, ball_coef=ball_coef,
                         max_steps=max_steps, goal_rows=goal_rows, cooperative=cooperative)
    raise ValueError(f"Unknown env family: {family}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--training-type",   type=bool,  default=True,
                   help="Must be True - enforces training_reward parameter usage")
    p.add_argument("--training-reward", type=str,   default="zero-sum",
                   choices=["zero-sum", "cooperative-no-opponent", "general-sum"],
                   help="Reward mode: zero-sum (equal teams), cooperative-no-opponent (any teams), general-sum (equal teams)")
    p.add_argument("--variant",       type=str, default="2v2")
    p.add_argument("--run-dir",       type=str, required=True)
    p.add_argument("--total-updates", type=int, default=2000)
    p.add_argument("--n-envs",        type=int, default=512)
    p.add_argument("--n-steps",       type=int, default=256)
    p.add_argument("--n-epochs",      type=int, default=4)
    p.add_argument("--n-minibatches", type=int, default=4)
    p.add_argument("--n-embd",        type=int, default=64)
    p.add_argument("--n-head",        type=int, default=1)
    p.add_argument("--n-block",       type=int, default=1)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--gamma",         type=float, default=0.99)
    p.add_argument("--gae-lambda",    type=float, default=0.95)
    p.add_argument("--clip-eps",      type=float, default=0.2)
    p.add_argument("--vf-coef",       type=float, default=0.5)
    p.add_argument("--ent-coef",      type=float, default=0.01)
    p.add_argument("--seed",          type=int, default=1)
    p.add_argument("--view-size",     type=int, default=7)
    p.add_argument("--ball-approach-coef", type=float, default=0.01)
    p.add_argument("--max-steps",     type=int, default=256)
    p.add_argument("--goal-rows",     type=int, nargs='+', default=None)
    p.add_argument("--cooperative",   action="store_true",
                   help="Non-zero-sum reward: own scoring only, no opponent-scored penalty, no steal bonus (EXP-B deconfound)")
    p.add_argument("--tensorboard",   action="store_true")
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
    run_dir = run_dir / ("zero-sum_false" if args.cooperative else "zero-sum_true")
    ckpt_dir = run_dir / "checkpoints" / args.training_reward / f"soccer-{args.variant}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    env = _make_env("soccer", args.variant, view_size=args.view_size,
                    ball_coef=args.ball_approach_coef, max_steps=args.max_steps,
                    goal_rows=args.goal_rows, cooperative=args.cooperative)
    print(f"[MAT-scan] soccer-{args.variant}  n_agents={env.num_agents}  "
          f"cooperative={args.cooperative}  "
          f"n_envs={args.n_envs}  n_steps={args.n_steps}  total_updates={args.total_updates}")
    print(f"  MAT: n_embd={args.n_embd} n_head={args.n_head} n_block={args.n_block}")

    config = {
        "ENV": env, "N_ENVS": args.n_envs, "N_STEPS": args.n_steps,
        "N_EPOCHS": args.n_epochs, "N_MINIBATCHES": args.n_minibatches,
        "GAMMA": args.gamma, "GAE_LAMBDA": args.gae_lambda, "CLIP_EPS": args.clip_eps,
        "VF_COEF": args.vf_coef, "ENT_COEF": args.ent_coef, "TOTAL_UPDATES": args.total_updates,
        "LR": args.lr, "N_EMBD": args.n_embd, "N_HEAD": args.n_head, "N_BLOCK": args.n_block,
    }

    tb_writer = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            from tensorboardX import SummaryWriter
        tb_log_dir = run_dir / "tb"
        tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
        print(f"[MAT-scan] TensorBoard logging to {tb_log_dir}")

        def _tb_callback(step, ep_ret, loss):
            # Convert update index -> real env-steps, matching IPPO/MAPPO/COMA's
            # x-axis convention so TensorBoard curves are directly comparable
            # across algorithms.
            env_step = int(step) * args.n_envs * args.n_steps * env.num_agents
            tb_writer.add_scalar("train/episode_return", float(ep_ret), env_step)
            tb_writer.add_scalar("train/total_loss", float(loss), env_step)
            tb_writer.flush()

        config["LOG_CALLBACK"] = _tb_callback

    train_fn = make_train(config)
    print("[MAT-scan] JIT-compiling...", flush=True)
    t0 = time.time()
    key = jax.random.PRNGKey(args.seed)
    runner_state, metrics = jax.block_until_ready(train_fn(key))
    t1 = time.time()
    total_steps = args.total_updates * args.n_envs * args.n_steps * env.num_agents
    print(f"[MAT-scan] done in {t1-t0:.1f}s  ({total_steps/(t1-t0):,.0f} steps/sec)")

    ep_rets = np.array(metrics["mean_episode_return"])
    print(f"  final ep_ret (last 50): {ep_rets[-50:].mean():.4f}")
    print(f"  total_loss={float(metrics['total_loss'][-1].mean()):.4f}  "
          f"entropy={float(metrics['entropy'][-1].mean()):.4f}")

    leaves, _ = jax.tree_util.tree_flatten(
        jax.tree_util.tree_map(np.array, runner_state.train_state.params))
    final_path = ckpt_dir / "final.npz"
    np.savez(str(final_path), *leaves)
    print(f"[MAT-scan] checkpoint → {final_path}")

    if tb_writer is not None:
        tb_writer.close()


if __name__ == "__main__":
    main()
