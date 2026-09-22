"""IC3Net-CNN for SocialJax clean_up.

IC3Net (Singh et al. 2019) with SocialJax paper CNN backbone:
  CNN: Conv(32,5x5)->Conv(32,3x3)->Conv(32,3x3)->Dense(64)->Dense(hid_dim)->GRU
  Gated comm: binary Categorical gate per agent, REINFORCE gate loss (gate_coef=0.1)
  PPO actor-critic, shared params (IPPO-CR)
  7 agents, shared rewards, 300M env steps, annealed reward shaping (30M horizon)
  32 envs x 9375 updates x 1000 steps = 300M env-steps

Run:
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \\
    python -m jaxmarl_worker.algorithms.socialjax.cleanup.ic3net_cnn_scan_socialjax
"""

import os
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

import argparse
import sys
import time
from pathlib import Path
from typing import NamedTuple, Any

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
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

GATE_COEF = 0.1


class _CNN(nn.Module):
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
        return x


class IC3NetCNN(nn.Module):
    hid_dim:    int = 256
    action_dim: int = 9

    @nn.compact
    def __call__(self, hidden, obs_img, done, gate_action):
        n_envs, n_agents = obs_img.shape[:2]
        obs_hw = obs_img.shape[2:]

        done_ag = jnp.broadcast_to(done[:, None, None], hidden.shape)
        hidden  = jnp.where(done_ag, jnp.zeros_like(hidden), hidden)

        h_flat = hidden.reshape(n_envs * n_agents, self.hid_dim)
        o_flat = obs_img.reshape(n_envs * n_agents, *obs_hw)

        gate_logits_flat = nn.Dense(2, kernel_init=orthogonal(1.0),
                                    bias_init=constant(0.0))(h_flat)
        gate_logits = gate_logits_flat.reshape(n_envs, n_agents, 2)

        gate_s  = gate_action.astype(jnp.float32).reshape(n_envs, n_agents, 1)
        gated_h = hidden * gate_s
        if n_agents > 1:
            gh_sum = gated_h.sum(axis=1, keepdims=True)
            comm   = (gh_sum - gated_h) / (n_agents - 1)
        else:
            comm = jnp.zeros_like(hidden)
        comm_flat = comm.reshape(n_envs * n_agents, self.hid_dim)

        cnn_out = _CNN()(o_flat)
        x = nn.Dense(self.hid_dim, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(cnn_out)
        x = nn.tanh(x)

        new_h_flat, _ = nn.GRUCell(self.hid_dim)(h_flat, x + comm_flat)
        new_h = new_h_flat.reshape(n_envs, n_agents, self.hid_dim)

        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01),
                          bias_init=constant(0.0))(new_h_flat)
        value  = jnp.squeeze(
            nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(new_h_flat),
            axis=-1,
        )
        return (
            new_h,
            logits.reshape(n_envs, n_agents, self.action_dim),
            value.reshape(n_envs, n_agents),
            gate_logits,
        )


class Transition(NamedTuple):
    obs:           chex.Array
    hidden:        chex.Array
    action:        chex.Array
    log_prob:      chex.Array
    value:         chex.Array
    reward:        chex.Array
    done:          chex.Array
    gate_prev:     chex.Array
    gate_log_prob: chex.Array
    info:          Any


class RunnerState(NamedTuple):
    train_state: TrainState
    hidden:      chex.Array
    prev_gate:   chex.Array
    env_state:   Any
    obs:         chex.Array
    update_step: int
    key:         chex.PRNGKey


def make_train(cfg: dict):
    env_raw  = socialjax.make(cfg["ENV_NAME"], **cfg["ENV_KWARGS"])
    env      = LogWrapper(env_raw, replace_info=False)
    n_agents = env_raw.num_agents

    n_envs    = cfg["NUM_ENVS"]
    n_steps   = cfg["NUM_STEPS"]
    n_epochs  = cfg["UPDATE_EPOCHS"]
    n_mb      = cfg["NUM_MINIBATCHES"]
    hid_dim   = cfg["HIDDEN_DIM"]
    gate_coef = cfg["GATE_COEF"]
    gamma     = cfg["GAMMA"]
    gae_lam   = cfg["GAE_LAMBDA"]
    clip_eps  = cfg["CLIP_EPS"]
    vf_coef   = cfg["VF_COEF"]
    ent_coef  = cfg["ENT_COEF"]

    n_updates    = int(cfg["TOTAL_TIMESTEPS"] // n_steps // n_envs)
    total_per_ag = n_steps * n_envs
    mb_size      = total_per_ag // n_mb
    rew_shaping_horizon = cfg.get("REW_SHAPING_HORIZON", 0)  # 0 = disabled

    obs_shape = env_raw.observation_space()[0].shape
    n_actions = env_raw.action_space().n

    net = IC3NetCNN(hid_dim=hid_dim, action_dim=n_actions)

    def linear_lr(count):
        frac = 1.0 - (count // (n_mb * n_epochs)) / n_updates
        return cfg["LR"] * frac

    def train(rng):
        rng, ik, rk = jax.random.split(rng, 3)
        dummy_h    = jnp.zeros((1, n_agents, hid_dim))
        dummy_o    = jnp.zeros((1, n_agents, *obs_shape))
        dummy_d    = jnp.zeros((1,))
        dummy_gate = jnp.ones((1, n_agents), dtype=jnp.int32)
        params = net.init(ik, dummy_h, dummy_o, dummy_d, dummy_gate)

        tx = optax.chain(
            optax.clip_by_global_norm(cfg["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_lr if cfg["ANNEAL_LR"] else cfg["LR"], eps=1e-5),
        )
        ts = TrainState.create(apply_fn=net.apply, params=params, tx=tx)

        obs, env_state = jax.vmap(env.reset, in_axes=(0,))(jax.random.split(rk, n_envs))
        hidden    = jnp.zeros((n_envs, n_agents, hid_dim))
        prev_gate = jnp.ones((n_envs, n_agents), dtype=jnp.int32)

        rng, _rng = jax.random.split(rng)
        runner_state = RunnerState(ts, hidden, prev_gate, env_state, obs, 0, _rng)

        def _update_step(runner_state, _unused):

            def _env_step(runner_state, _unused):
                ts, hidden, prev_gate, env_state, obs, update_step, key = runner_state
                key, k_act, k_gate, k_step = jax.random.split(key, 4)

                new_hidden, logits, values, gate_logits = ts.apply_fn(
                    ts.params, hidden, obs, jnp.zeros(n_envs), prev_gate)

                lf   = logits.reshape(n_envs * n_agents, n_actions)
                af   = jax.random.categorical(k_act, lf)
                lp   = jax.nn.log_softmax(lf)[jnp.arange(n_envs * n_agents), af]
                acts = af.reshape(n_envs, n_agents)
                logp = lp.reshape(n_envs, n_agents)

                gf        = gate_logits.reshape(n_envs * n_agents, 2)
                gate_next = jax.random.categorical(k_gate, gf).reshape(n_envs, n_agents)
                gate_lp   = jax.nn.log_softmax(gf)[
                    jnp.arange(n_envs * n_agents), gate_next.reshape(-1)
                ].reshape(n_envs, n_agents)

                env_act  = [acts[:, i] for i in range(n_agents)]
                rng_step = jax.random.split(k_step, n_envs)
                next_obs, next_env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)

                done_all = done["__all__"]
                done_ag  = jnp.broadcast_to(done_all[:, None, None], new_hidden.shape)
                new_hidden = jnp.where(done_ag, jnp.zeros_like(new_hidden), new_hidden)

                if rew_shaping_horizon > 0:
                    anneal_w = jnp.maximum(0.0, 1.0 - update_step * n_steps * n_envs / rew_shaping_horizon)
                    reward = reward + info["clean_action_info"] * anneal_w

                transition = Transition(
                    obs=obs, hidden=hidden, action=acts, log_prob=logp, value=values,
                    reward=reward, done=done_all, gate_prev=prev_gate,
                    gate_log_prob=gate_lp, info=info,
                )
                return RunnerState(ts, new_hidden, gate_next, next_env_state,
                                   next_obs, update_step, key), transition

            runner_state, traj = jax.lax.scan(_env_step, runner_state, None, n_steps)
            ts, hidden, prev_gate, env_state, obs, update_step, key = runner_state

            _, _, last_val, _ = ts.apply_fn(ts.params, hidden, obs, jnp.zeros(n_envs), prev_gate)

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

            obs_flat     = _flatten(traj.obs)
            h_flat_      = _flatten(traj.hidden)
            act_flat     = _flatten(traj.action)
            logp_flat    = _flatten(traj.log_prob)
            adv_flat     = _flatten(advantages)
            ret_flat     = _flatten(targets)
            done_flat    = _flatten(traj.done)
            gate_flat    = _flatten(traj.gate_prev)
            gate_lp_flat = _flatten(traj.gate_log_prob)

            def _epoch(ts, _):
                key_e, _ = jax.random.split(jax.random.PRNGKey(ts.step))
                perm     = jax.random.permutation(key_e, total_per_ag)
                o_mb   = obs_flat[perm].reshape(n_mb, mb_size, n_agents, *obs_shape)
                h_mb   = h_flat_[perm].reshape(n_mb, mb_size, n_agents, hid_dim)
                a_mb   = act_flat[perm].reshape(n_mb, mb_size, n_agents)
                lp_mb  = logp_flat[perm].reshape(n_mb, mb_size, n_agents)
                adv_mb = adv_flat[perm].reshape(n_mb, mb_size, n_agents)
                ret_mb = ret_flat[perm].reshape(n_mb, mb_size, n_agents)
                d_mb   = done_flat[perm].reshape(n_mb, mb_size)
                g_mb   = gate_flat[perm].reshape(n_mb, mb_size, n_agents)
                glp_mb = gate_lp_flat[perm].reshape(n_mb, mb_size, n_agents)

                def _mb_update(ts, batch):
                    o_b, h_b, a_b, lp_b, adv_b, ret_b, d_b, g_b, glp_b = batch

                    def loss_fn(params):
                        _, logits_b, val_b, _ = net.apply(params, h_b, o_b, d_b, g_b)
                        B, N, A = logits_b.shape
                        lps_flat = jax.nn.log_softmax(logits_b).reshape(B * N, A)
                        new_lp   = lps_flat[jnp.arange(B * N), a_b.reshape(B * N)].reshape(B, N)
                        ratio    = jnp.exp(new_lp - lp_b)
                        adv_n    = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)
                        pg  = jnp.maximum(
                            -adv_n * ratio,
                            -adv_n * jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps),
                        ).mean()
                        vf  = 0.5 * jnp.mean((val_b - ret_b) ** 2)
                        ent = -(jax.nn.softmax(logits_b) *
                                jax.nn.log_softmax(logits_b)).sum(-1).mean()
                        gate_loss = -(glp_b * adv_n).mean()
                        total = pg + vf_coef * vf - ent_coef * ent + gate_coef * gate_loss
                        return total, (pg, vf, ent, gate_loss)

                    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(ts.params)
                    ts = ts.apply_gradients(grads=grads)
                    return ts, {"loss": loss, "pg": aux[0], "vf": aux[1],
                                "ent": aux[2], "gate_loss": aux[3]}

                ts, mb_m = jax.lax.scan(
                    _mb_update, ts,
                    (o_mb, h_mb, a_mb, lp_mb, adv_mb, ret_mb, d_mb, g_mb, glp_mb),
                )
                return ts, {k: v.mean() for k, v in mb_m.items()}

            ts, ep_m = jax.lax.scan(_epoch, ts, None, n_epochs)
            ep_m_s = {k: v.mean() for k, v in ep_m.items()}

            update_step = update_step + 1
            metric = jax.tree.map(lambda x: x.mean(), traj.info)

            def _log(args):
                step, ret, loss, gate_l = args
                print(f"  update {int(step):4d}/{n_updates}  "
                      f"return={float(ret):.3f}  loss={float(loss):.4f}  "
                      f"gate={float(gate_l):.4f}", flush=True)

            jax.debug.callback(_log, (
                update_step,
                metric.get("returned_episode_returns", jnp.array(0.0)),
                ep_m_s["loss"],
                ep_m_s["gate_loss"],
            ))

            runner_state = RunnerState(ts, hidden, prev_gate, env_state, obs, update_step, key)
            return runner_state, {**metric, **ep_m_s, "mean_step_reward": traj.reward.mean()}

        return runner_state, _update_step

    return train, n_agents, n_updates


def save_checkpoint(params, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    leaves, _ = jax.tree_util.tree_flatten(params)
    np.savez(str(path), **{f"ic3net_{i}": np.array(v) for i, v in enumerate(leaves)})
    print(f"Checkpoint saved: {path}  ({len(leaves)} leaves)")


def parse_args():
    _default_out = str(
        Path(__file__).resolve().parents[7]
        / "var" / "trainer" / "socialjax" / "cleanup" / "IC3Net_CNN"
    )
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir",           default=_default_out)
    p.add_argument("--training-type",   type=bool,  default=True,
                   help="Must be True - enforces training_reward parameter usage")
    p.add_argument("--training-reward", type=str,   default="cooperative-no-opponent",
                   choices=["cooperative-no-opponent"],
                   help="SocialJax environments are always cooperative (no-opponent)")
    p.add_argument("--num-agents",,        type=int,   default=7)
    p.add_argument("--num-envs",          type=int,   default=32,
                   help="32 envs → 3125 updates × 1000 steps = 100M env-steps (consistent with all algorithms)")
    p.add_argument("--total-timesteps",   type=float, default=3e8)
    p.add_argument("--num-steps",         type=int,   default=1000)
    p.add_argument("--update-epochs",     type=int,   default=2)
    p.add_argument("--num-minibatches",   type=int,   default=500)
    p.add_argument("--hidden-dim",        type=int,   default=256)
    p.add_argument("--lr",                type=float, default=5e-4)
    p.add_argument("--gamma",             type=float, default=0.99)
    p.add_argument("--gae-lambda",        type=float, default=0.95)
    p.add_argument("--clip-eps",          type=float, default=0.2)
    p.add_argument("--ent-coef",          type=float, default=0.01)
    p.add_argument("--vf-coef",           type=float, default=0.5)
    p.add_argument("--max-grad-norm",     type=float, default=0.5)
    p.add_argument("--gate-coef",         type=float, default=GATE_COEF)
    p.add_argument("--seed",              type=int,   default=30)
    p.add_argument("--shared-rewards",    action="store_true", default=True)
    p.add_argument("--no-shared-rewards", dest="shared_rewards", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()
    assert args.training_type == True, "training_type must be True"
    assert args.training_reward == "cooperative-no-opponent", f"SocialJax only supports cooperative-no-opponent, got {args.training_reward}"
    assert args.max_steps == 1000, f"SocialJax episode length must be exactly 1000 steps, got {args.max_steps}"

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _lf = open(run_dir / "train.log", "w", buffering=1)
    class _Tee:
        def write(self, s): _lf.write(s); sys.__stdout__.write(s)
        def flush(self): _lf.flush(); sys.__stdout__.flush()
        def fileno(self): return sys.__stdout__.fileno()
    sys.stdout = _Tee()

    cfg = {
        "ENV_NAME": "clean_up",
        "ENV_KWARGS": {
            "num_agents":      args.num_agents,
            "num_inner_steps": args.num_steps,
            "shared_rewards":  args.shared_rewards,
            "cnn":             True,
            "jit":             True,
        },
        "LR":              args.lr,
        "NUM_ENVS":        args.num_envs,
        "NUM_STEPS":       args.num_steps,
        "TOTAL_TIMESTEPS": int(args.total_timesteps),
        "UPDATE_EPOCHS":   args.update_epochs,
        "NUM_MINIBATCHES": args.num_minibatches,
        "GAMMA":           args.gamma,
        "GAE_LAMBDA":      args.gae_lambda,
        "CLIP_EPS":        args.clip_eps,
        "ENT_COEF":        args.ent_coef,
        "VF_COEF":         args.vf_coef,
        "MAX_GRAD_NORM":   args.max_grad_norm,
        "HIDDEN_DIM":          args.hidden_dim,
        "GATE_COEF":           args.gate_coef,
        "ANNEAL_LR":           True,
        "REW_SHAPING_HORIZON": 30_000_000,
    }

    train_fn, _, n_updates = make_train(cfg)
    reward_str = "CR" if args.shared_rewards else "IR"

    print(f"[IC3Net-CNN] clean_up  n_agents={args.num_agents}  "
          f"reward={reward_str}  envs={args.num_envs}  "
          f"steps/ep={args.num_steps}  updates={n_updates}  "
          f"total={int(args.total_timesteps)/1e6:.0f}M  seed={args.seed}")
    print(f"  Gated comm (IC3Net, Singh et al. 2019)  gate_coef={args.gate_coef}")
    print(f"GPU: {jax.devices()}")

    try:
        from torch.utils.tensorboard import SummaryWriter as _SW
        tb_writer = _SW(log_dir=str(run_dir / "tensorboard"))
        print(f"TensorBoard live at {run_dir / 'tensorboard'}")
    except ImportError:
        tb_writer = None
        print("torch not available — TensorBoard disabled")

    step_size = args.num_envs * args.num_steps
    THROTTLE = 3.0
    t0 = time.time()
    runner_state, _update_step = train_fn(jax.random.PRNGKey(args.seed))
    _jit_step = jax.jit(_update_step)
    ep_rets = []
    for _i in range(n_updates):
        runner_state, last_metric = _jit_step(runner_state, None)
        jax.block_until_ready(runner_state)
        ret = float(np.array(last_metric["mean_step_reward"])) * args.num_steps
        ep_rets.append(ret)
        env_step = (_i + 1) * step_size
        print(f"  update {_i+1:5d}/{n_updates}  return={ret:.3f}", flush=True)
        if tb_writer is not None:
            tb_writer.add_scalar("train/episode_return", ret, env_step)
            tb_writer.flush()
        time.sleep(THROTTLE)
    dt = time.time() - t0
    print(f"\nTraining done in {dt/60:.1f} min")

    if tb_writer is not None:
        tb_writer.close()

    save_checkpoint(runner_state.train_state.params, run_dir / "checkpoints" / "final.npz")

    ep_rets_np = np.array(ep_rets)
    np.save(str(run_dir / "metrics_ep_return.npy"), ep_rets_np)
    print(f"Final return/agent (last 50): {ep_rets_np[-50:].mean():.3f}")


if __name__ == "__main__":
    main()
