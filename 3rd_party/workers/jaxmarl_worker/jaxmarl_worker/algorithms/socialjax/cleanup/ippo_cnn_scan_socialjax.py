"""IPPO-CNN for SocialJax clean_up -- paper-faithful replication.

Exactly matches the SocialJax paper (Guo et al., ICLR 2026) config:
  - CNN backbone: Conv(32,5x5) -> Conv(32,3x3) -> Conv(32,3x3) -> Dense(64)
  - 7 agents, shared parameters, common reward (shared_rewards=True)
  - 300M env steps, 256 parallel envs, 1000 steps/episode
  - Paper note: reward shaping code exists but is commented out -- raw reward only

Run:
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \\
    python -m jaxmarl_worker.algorithms.socialjax.cleanup.ippo_cnn_scan_socialjax
"""
import argparse
import sys
import time
from pathlib import Path

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
# CNN + ActorCritic (verbatim from SocialJax/algorithms/utils/networks.py)
# ---------------------------------------------------------------------------

class _CNN(nn.Module):
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        act = nn.relu if self.activation == "relu" else nn.tanh
        x = nn.Conv(32, (5, 5), kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = act(x)
        x = nn.Conv(32, (3, 3), kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = act(x)
        x = nn.Conv(32, (3, 3), kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = act(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = act(x)
        return x


class _Categorical:
    """Minimal Categorical distribution backed by pure JAX (no distrax)."""
    def __init__(self, logits):
        self.logits = logits

    def sample(self, seed):
        return jax.random.categorical(seed, self.logits)

    def log_prob(self, actions):
        log_probs = jax.nn.log_softmax(self.logits)
        return log_probs[jnp.arange(actions.shape[0]), actions]

    def entropy(self):
        probs = jax.nn.softmax(self.logits)
        log_probs = jax.nn.log_softmax(self.logits)
        return -(probs * log_probs).sum(-1)


class _ActorCritic(nn.Module):
    action_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        act = nn.relu if self.activation == "relu" else nn.tanh
        embedding = _CNN(self.activation)(x)

        actor_mean = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        actor_mean = act(actor_mean)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)

        critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        critic = act(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return logits, jnp.squeeze(critic, axis=-1)


# ---------------------------------------------------------------------------
# Transition namedtuple
# ---------------------------------------------------------------------------

from typing import NamedTuple

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: dict


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def make_train(cfg: dict):
    env_raw = socialjax.make(cfg["ENV_NAME"], **cfg["ENV_KWARGS"])
    env     = LogWrapper(env_raw, replace_info=False)

    n_agents = env_raw.num_agents
    cfg["NUM_ACTORS"]     = n_agents * cfg["NUM_ENVS"]
    cfg["NUM_UPDATES"]    = int(cfg["TOTAL_TIMESTEPS"] // cfg["NUM_STEPS"] // cfg["NUM_ENVS"])
    cfg["MINIBATCH_SIZE"] = cfg["NUM_ACTORS"] * cfg["NUM_STEPS"] // cfg["NUM_MINIBATCHES"]

    tb_writer  = cfg.get("TB_WRITER", None)
    _step_size = cfg["NUM_ENVS"] * cfg["NUM_STEPS"]
    rew_shaping_horizon = cfg.get("REW_SHAPING_HORIZON", 0)  # 0 = disabled

    obs_shape = env_raw.observation_space()[0].shape  # (11, 11, 19)
    n_actions = env_raw.action_space().n              # 9

    def linear_lr(count):
        frac = 1.0 - (count // (cfg["NUM_MINIBATCHES"] * cfg["UPDATE_EPOCHS"])) / cfg["NUM_UPDATES"]
        return cfg["LR"] * frac

    def train(rng):
        network = _ActorCritic(n_actions, activation=cfg["ACTIVATION"])
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *obs_shape))
        params = network.init(_rng, init_x)

        tx = optax.chain(
            optax.clip_by_global_norm(cfg["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_lr if cfg["ANNEAL_LR"] else cfg["LR"], eps=1e-5),
        )
        train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, cfg["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        def _update_step(runner_state, _unused):
            def _env_step(runner_state, _unused):
                ts, env_state, last_obs, update_step, rng = runner_state
                rng, _rng = jax.random.split(rng)

                # (NUM_ENVS, n_agents, H, W, C) -> (n_agents, NUM_ENVS, H, W, C) -> (NUM_ACTORS, H, W, C)
                obs_batch = jnp.transpose(last_obs, (1, 0, 2, 3, 4)).reshape(-1, *obs_shape)
                logits, value = network.apply(ts.params, obs_batch)
                pi       = _Categorical(logits)
                action   = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # action: (NUM_ACTORS,) = (n_agents * NUM_ENVS,) arranged [a0_env0, a0_env1, ..., a1_env0, ...]
                env_act = [action.reshape(n_agents, cfg["NUM_ENVS"])[i] for i in range(n_agents)]

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, cfg["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_step, env_state, env_act
                )
                # reward: (NUM_ENVS, n_agents)  done: {'__all__': (NUM_ENVS,), '0':..., ...}
                if rew_shaping_horizon > 0:
                    anneal_w = jnp.maximum(0.0, 1.0 - update_step * cfg["NUM_STEPS"] * cfg["NUM_ENVS"] / rew_shaping_horizon)
                    reward = reward + info["clean_action_info"] * anneal_w
                info        = jax.tree.map(lambda x: x.reshape((cfg["NUM_ACTORS"],)), info)
                reward_flat = jnp.transpose(reward, (1, 0)).reshape(-1)
                done_flat   = jnp.tile(done["__all__"], n_agents)

                transition = Transition(done_flat, action, value, reward_flat, log_prob, obs_batch, info)
                return (ts, env_state, obsv, update_step, rng), transition

            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, cfg["NUM_STEPS"])
            ts, env_state, last_obs, update_step, rng = runner_state

            last_obs_batch = jnp.transpose(last_obs, (1, 0, 2, 3, 4)).reshape(-1, *obs_shape)
            _, last_val = network.apply(ts.params, last_obs_batch)

            def _gae(traj_batch, last_val):
                def _get_adv(carry, transition):
                    gae, next_val = carry
                    delta = transition.reward + cfg["GAMMA"] * next_val * (1 - transition.done) - transition.value
                    gae   = delta + cfg["GAMMA"] * cfg["GAE_LAMBDA"] * (1 - transition.done) * gae
                    return (gae, transition.value), gae
                _, advantages = jax.lax.scan(
                    _get_adv, (jnp.zeros_like(last_val), last_val), traj_batch, reverse=True, unroll=16
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _gae(traj_batch, last_val)

            def _update_epoch(update_state, _unused):
                def _update_minibatch(ts, batch_info):
                    traj, adv, tgt = batch_info
                    def _loss(params, traj, adv, tgt):
                        logits, value = network.apply(params, traj.obs)
                        pi       = _Categorical(logits)
                        log_prob = pi.log_prob(traj.action)
                        v_clip = traj.value + (value - traj.value).clip(-cfg["CLIP_EPS"], cfg["CLIP_EPS"])
                        v_loss = 0.5 * jnp.maximum(jnp.square(value - tgt), jnp.square(v_clip - tgt)).mean()
                        ratio   = jnp.exp(log_prob - traj.log_prob)
                        adv_n   = (adv - adv.mean()) / (adv.std() + 1e-8)
                        pg_loss = -jnp.minimum(ratio * adv_n,
                                               jnp.clip(ratio, 1 - cfg["CLIP_EPS"], 1 + cfg["CLIP_EPS"]) * adv_n).mean()
                        entropy = pi.entropy().mean()
                        total   = pg_loss + cfg["VF_COEF"] * v_loss - cfg["ENT_COEF"] * entropy
                        return total, (v_loss, pg_loss, entropy)
                    grads, aux = jax.grad(_loss, has_aux=True)(ts.params, traj, adv, tgt)
                    ts = ts.apply_gradients(grads=grads)
                    return ts, aux

                ts, traj, adv, tgt, rng = update_state
                rng, _rng = jax.random.split(rng)
                B = cfg["MINIBATCH_SIZE"] * cfg["NUM_MINIBATCHES"]
                perm  = jax.random.permutation(_rng, B)
                batch = jax.tree_util.tree_map(lambda x: x.reshape((B,) + x.shape[2:]), (traj, adv, tgt))
                batch = jax.tree_util.tree_map(lambda x: jnp.take(x, perm, axis=0), batch)
                minibatches = jax.tree_util.tree_map(
                    lambda x: x.reshape([cfg["NUM_MINIBATCHES"], -1] + list(x.shape[1:])), batch
                )
                ts, loss_info = jax.lax.scan(_update_minibatch, ts, minibatches)
                return (ts, traj, adv, tgt, rng), loss_info

            update_state = (ts, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, cfg["UPDATE_EPOCHS"]
            )
            ts  = update_state[0]
            rng = update_state[-1]

            update_step = update_step + 1
            metric = jax.tree.map(lambda x: x.mean(), traj_batch.info)

            def _log(args):
                step, mean_rew = args
                ret = float(mean_rew) * cfg["NUM_STEPS"]
                print(f"  update {int(step):4d}/{cfg['NUM_UPDATES']}  "
                      f"return={ret:.2f}", flush=True)
                if tb_writer is not None:
                    env_step = int(step) * _step_size
                    tb_writer.add_scalar("train/episode_return", ret, env_step)
                    tb_writer.flush()

            jax.debug.callback(_log, (update_step, traj_batch.reward.mean()))

            return (ts, env_state, last_obs, update_step, rng), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, 0, _rng)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, cfg["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train, n_agents


# ---------------------------------------------------------------------------
# Checkpoint helpers
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
        / "var" / "trainer" / "socialjax" / "cleanup" / "IPPO_CNN"
    )
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir",         default=_default_out)
    p.add_argument("--training-type",   type=bool,  default=True,
                   help="Must be True - enforces training_reward parameter usage")
    p.add_argument("--training-reward", type=str,   default="cooperative-no-opponent",
                   choices=["cooperative-no-opponent"],
                   help="SocialJax environments are always cooperative (no-opponent)")
    p.add_argument("--num-agents",,      type=int,   default=7,    help="Paper default: 7")
    p.add_argument("--num-envs",        type=int,   default=32,
                   help="32 envs → 3125 updates × 1000 steps = 100M env-steps (consistent with all algorithms)")
    p.add_argument("--total-timesteps", type=float, default=3e8,  help="300M env-steps → 9375 updates with 32 envs × 1000 steps")
    p.add_argument("--num-steps",       type=int,   default=1000)
    p.add_argument("--update-epochs",   type=int,   default=2)
    p.add_argument("--num-minibatches", type=int,   default=500)
    p.add_argument("--lr",              type=float, default=5e-4)
    p.add_argument("--gamma",           type=float, default=0.99)
    p.add_argument("--gae-lambda",      type=float, default=0.95)
    p.add_argument("--clip-eps",        type=float, default=0.2)
    p.add_argument("--ent-coef",        type=float, default=0.01)
    p.add_argument("--vf-coef",         type=float, default=0.5)
    p.add_argument("--max-grad-norm",   type=float, default=0.5)
    p.add_argument("--seed",            type=int,   default=30)
    p.add_argument("--shared-rewards",  action="store_true", default=True,
                   help="Common reward (paper IPPO-CR). Pass --no-shared-rewards for IPPO-IR.")
    p.add_argument("--no-shared-rewards", dest="shared_rewards", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()
    assert args.training_type == True, "training_type must be True"
    assert args.training_reward == "cooperative-no-opponent", f"SocialJax only supports cooperative-no-opponent, got {args.training_reward}"
    assert args.max_steps == 1000, f"SocialJax episode length must be exactly 1000 steps, got {args.max_steps}"

    cfg = {
        "ENV_NAME": "clean_up",
        "ENV_KWARGS": {
            "num_agents":      args.num_agents,
            "num_inner_steps": args.num_steps,
            "shared_rewards":  args.shared_rewards,
            "cnn":             True,
            "jit":             True,
        },
        "LR":               args.lr,
        "NUM_ENVS":         args.num_envs,
        "NUM_STEPS":        args.num_steps,
        "TOTAL_TIMESTEPS":  int(args.total_timesteps),
        "UPDATE_EPOCHS":    args.update_epochs,
        "NUM_MINIBATCHES":  args.num_minibatches,
        "GAMMA":            args.gamma,
        "GAE_LAMBDA":       args.gae_lambda,
        "CLIP_EPS":         args.clip_eps,
        "ENT_COEF":         args.ent_coef,
        "VF_COEF":          args.vf_coef,
        "MAX_GRAD_NORM":    args.max_grad_norm,
        "ACTIVATION":          "relu",
        "ANNEAL_LR":           True,
        "REW_SHAPING_HORIZON": 30_000_000,
    }

    reward_str = "CR" if args.shared_rewards else "IR"
    total_steps = cfg["TOTAL_TIMESTEPS"]
    n_updates   = int(total_steps // cfg["NUM_STEPS"] // cfg["NUM_ENVS"])

    print(f"[IPPO-CNN] clean_up  n_agents={args.num_agents}  "
          f"reward={reward_str}  envs={args.num_envs}  "
          f"steps/ep={args.num_steps}  updates={n_updates}  "
          f"total={total_steps/1e6:.0f}M  seed={args.seed}")
    print(f"GPU: {jax.devices()}")

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter as _SW
        tb_writer = _SW(log_dir=str(run_dir / "tensorboard"))
        print(f"TensorBoard live at {run_dir / 'tensorboard'}")
    except ImportError:
        tb_writer = None
        print("torch not available — TensorBoard disabled")
    cfg["TB_WRITER"] = tb_writer

    train_fn, _ = make_train(cfg)
    train_jit   = jax.jit(train_fn)

    t0  = time.time()
    out = train_jit(jax.random.PRNGKey(args.seed))
    dt  = time.time() - t0
    print(f"\nTraining done in {dt/60:.1f} min")

    if tb_writer is not None:
        tb_writer.close()

    final_params = out["runner_state"][0].params
    save_checkpoint(final_params, run_dir / "checkpoints" / "final.npz")

    ep_rets = np.array(out["metrics"].get("returned_episode_returns", [])).ravel()
    np.save(str(run_dir / "metrics_ep_return.npy"), ep_rets)
    print(f"Final return/agent (last 50): {ep_rets[-50:].mean():.3f}")


if __name__ == "__main__":
    main()
