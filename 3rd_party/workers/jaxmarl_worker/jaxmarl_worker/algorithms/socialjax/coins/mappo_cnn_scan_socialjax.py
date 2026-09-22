"""MAPPO-CNN for SocialJax coin_game -- paper-faithful replication.

Exactly matches the SocialJax paper (Guo et al., ICLR 2026) MAPPO config:
  - SmallActor: CNN backbone on per-agent obs (11,11,14)
  - SmallCritic: CNN backbone on centralized world_state (11,11,28)
  - 2 agents, common reward (shared_rewards=True)
  - 300M env steps, 256 parallel envs, 1000 steps/episode
  - LR=1e-3, 64 minibatches (MAPPO base config)
  - Scaled clip eps: CLIP_EPS / num_agents = 0.2/2 = 0.1

Run:
  CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \\
    python -m jaxmarl_worker.algorithms.socialjax.coins.mappo_cnn_scan_socialjax
"""
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
if _SOCIALJAX_ROOT not in sys.path:
    sys.path.insert(0, _SOCIALJAX_ROOT)

import socialjax
from socialjax.wrappers.baselines import LogWrapper


# ---------------------------------------------------------------------------
# CNN backbone (verbatim from SocialJax/algorithms/utils/networks.py)
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


class _SmallActor(nn.Module):
    action_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs):
        act = nn.relu if self.activation == "relu" else nn.tanh
        embedding = _CNN(self.activation)(obs)
        x = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        x = act(x)
        return nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)


class _SmallCritic(nn.Module):
    activation: str = "relu"

    @nn.compact
    def __call__(self, world_state):
        act = nn.relu if self.activation == "relu" else nn.tanh
        embedding = _CNN(self.activation)(world_state)
        x = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        x = act(x)
        return jnp.squeeze(
            nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x), axis=-1
        )


class Transition(NamedTuple):
    done:        jnp.ndarray
    action:      jnp.ndarray
    value:       jnp.ndarray
    reward:      jnp.ndarray
    log_prob:    jnp.ndarray
    obs:         jnp.ndarray
    world_state: jnp.ndarray
    info:        dict


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

    obs_shape = env_raw.observation_space()[0].shape          # (11, 11, 14)
    ws_shape  = (*obs_shape[:-1], obs_shape[-1] * n_agents)   # (11, 11, 28)
    n_actions = env_raw.action_space().n
    clip_eps  = cfg["CLIP_EPS"] / n_agents if cfg["SCALE_CLIP_EPS"] else cfg["CLIP_EPS"]

    def linear_lr(count):
        frac = 1.0 - (count // (cfg["NUM_MINIBATCHES"] * cfg["UPDATE_EPOCHS"])) / cfg["NUM_UPDATES"]
        return cfg["LR"] * frac

    def train(rng):
        actor  = _SmallActor(n_actions, activation=cfg["ACTIVATION"])
        critic = _SmallCritic(activation=cfg["ACTIVATION"])

        rng, _rng = jax.random.split(rng)
        a_params = actor.init(_rng, jnp.zeros((1, *obs_shape)))
        rng, _rng = jax.random.split(rng)
        c_params = critic.init(_rng, jnp.zeros((1, *ws_shape)))

        tx = optax.chain(
            optax.clip_by_global_norm(cfg["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_lr if cfg["ANNEAL_LR"] else cfg["LR"], eps=1e-5),
        )
        actor_ts  = TrainState.create(apply_fn=actor.apply,  params=a_params, tx=tx)
        critic_ts = TrainState.create(apply_fn=critic.apply, params=c_params, tx=tx)

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, cfg["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        def _update_step(runner_state, _unused):
            def _env_step(runner_state, _unused):
                a_ts, c_ts, env_state, last_obs, update_step, rng = runner_state
                rng, _rng = jax.random.split(rng)

                obs_batch = jnp.transpose(last_obs, (1, 0, 2, 3, 4)).reshape(-1, *obs_shape)
                logits = actor.apply(a_ts.params, obs_batch)
                pi     = _Categorical(logits)
                action   = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                world_state = jnp.transpose(last_obs, (0, 2, 3, 1, 4)).reshape(
                    cfg["NUM_ENVS"], *ws_shape)
                value_env = critic.apply(c_ts.params, world_state)

                env_act = [action.reshape(n_agents, cfg["NUM_ENVS"])[i] for i in range(n_agents)]

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, cfg["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_step, env_state, env_act
                )

                info        = jax.tree.map(lambda x: x.reshape((cfg["NUM_ACTORS"],)), info)
                reward_flat = jnp.transpose(reward, (1, 0)).reshape(-1)
                done_flat   = jnp.tile(done["__all__"], n_agents)
                value_flat  = jnp.tile(value_env, n_agents)

                transition = Transition(done_flat, action, value_flat, reward_flat,
                                        log_prob, obs_batch,
                                        jnp.tile(world_state, (n_agents, 1, 1, 1)), info)
                return (a_ts, c_ts, env_state, obsv, update_step, rng), transition

            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, cfg["NUM_STEPS"])
            a_ts, c_ts, env_state, last_obs, update_step, rng = runner_state

            last_ws      = jnp.transpose(last_obs, (0, 2, 3, 1, 4)).reshape(cfg["NUM_ENVS"], *ws_shape)
            last_val_env = critic.apply(c_ts.params, last_ws)
            last_val     = jnp.tile(last_val_env, n_agents)

            def _gae(traj_batch, last_val):
                def _get_adv(carry, transition):
                    gae, next_val = carry
                    delta = transition.reward + cfg["GAMMA"] * next_val * (1 - transition.done) - transition.value
                    gae   = delta + cfg["GAMMA"] * cfg["GAE_LAMBDA"] * (1 - transition.done) * gae
                    return (gae, transition.value), gae
                _, advantages = jax.lax.scan(
                    _get_adv, (jnp.zeros_like(last_val), last_val), traj_batch, reverse=True, unroll=16)
                return advantages, advantages + traj_batch.value

            advantages, targets = _gae(traj_batch, last_val)

            def _update_epoch(update_state, _unused):
                def _update_minibatch(states, batch_info):
                    a_ts, c_ts = states
                    traj, adv, tgt = batch_info

                    def _actor_loss(a_params, traj, adv):
                        logits   = actor.apply(a_params, traj.obs)
                        pi       = _Categorical(logits)
                        log_prob = pi.log_prob(traj.action)
                        ratio    = jnp.exp(log_prob - traj.log_prob)
                        adv_n    = (adv - adv.mean()) / (adv.std() + 1e-8)
                        pg       = -jnp.minimum(ratio * adv_n,
                                                 jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv_n).mean()
                        entropy  = pi.entropy().mean()
                        return pg - cfg["ENT_COEF"] * entropy, entropy

                    def _critic_loss(c_params, traj, tgt):
                        ws      = traj.world_state[::n_agents]
                        value   = critic.apply(c_params, ws)
                        tgt_env = tgt[::n_agents]
                        v_clip  = traj.value[::n_agents] + (value - traj.value[::n_agents]).clip(-clip_eps, clip_eps)
                        v_loss  = 0.5 * jnp.maximum(jnp.square(value - tgt_env),
                                                     jnp.square(v_clip - tgt_env)).mean()
                        return v_loss, v_loss

                    a_grads, ent = jax.grad(_actor_loss, has_aux=True)(a_ts.params, traj, adv)
                    c_grads, _   = jax.grad(_critic_loss, has_aux=True)(c_ts.params, traj, tgt)
                    a_ts = a_ts.apply_gradients(grads=a_grads)
                    c_ts = c_ts.apply_gradients(grads=c_grads)
                    return (a_ts, c_ts), (ent, 0.0)

                a_ts, c_ts, traj, adv, tgt, rng = update_state
                rng, _rng = jax.random.split(rng)
                B     = cfg["MINIBATCH_SIZE"] * cfg["NUM_MINIBATCHES"]
                perm  = jax.random.permutation(_rng, B)
                batch = jax.tree_util.tree_map(lambda x: x.reshape((B,) + x.shape[2:]), (traj, adv, tgt))
                batch = jax.tree_util.tree_map(lambda x: jnp.take(x, perm, axis=0), batch)
                minibatches = jax.tree_util.tree_map(
                    lambda x: x.reshape([cfg["NUM_MINIBATCHES"], -1] + list(x.shape[1:])), batch)
                states, loss_info = jax.lax.scan(_update_minibatch, (a_ts, c_ts), minibatches)
                (a_ts, c_ts) = states
                return (a_ts, c_ts, traj, adv, tgt, rng), loss_info

            update_state = (a_ts, c_ts, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, cfg["UPDATE_EPOCHS"])
            a_ts, c_ts = update_state[0], update_state[1]
            rng = update_state[-1]

            update_step = update_step + 1
            metric = jax.tree.map(lambda x: x.mean(), traj_batch.info)

            def _log(args):
                step, mean_rew = args
                ret = float(mean_rew) * cfg["NUM_STEPS"]
                print(f"  update {int(step):5d}/{cfg['NUM_UPDATES']}  "
                      f"return={ret:.2f}", flush=True)
                if tb_writer is not None:
                    env_step = int(step) * _step_size
                    tb_writer.add_scalar("train/episode_return", ret, env_step)
                    tb_writer.flush()

            jax.debug.callback(_log, (update_step, traj_batch.reward.mean()))
            return (a_ts, c_ts, env_state, last_obs, update_step, rng), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (actor_ts, critic_ts, env_state, obsv, 0, _rng)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, cfg["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train, n_agents


def save_checkpoint(params, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    leaves, _ = jax.tree_util.tree_flatten(params)
    np.savez(str(path), *[np.array(l) for l in leaves])
    print(f"Checkpoint saved: {path}  ({len(leaves)} leaves)")


def parse_args():
    _default_out = str(
        Path(__file__).resolve().parents[7]
        / "var" / "trainer" / "socialjax" / "coins" / "MAPPO_CNN"
    )
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir",         default=_default_out)
    p.add_argument("--training-type",   type=bool,  default=True,
                   help="Must be True - enforces training_reward parameter usage")
    p.add_argument("--training-reward", type=str,   default="cooperative-no-opponent",
                   choices=["cooperative-no-opponent"],
                   help="SocialJax environments are always cooperative (no-opponent)")
    p.add_argument("--num-agents",,      type=int,   default=2)
    p.add_argument("--num-envs",        type=int,   default=32,   help="128 envs (300M / 128 / 1000 = 2344 updates; matches coop_mining MAPPO)")
    p.add_argument("--total-timesteps", type=float, default=3e8,  help="300M env-steps → 9375 updates with 32 envs × 1000 steps")
    p.add_argument("--num-steps",       type=int,   default=1000)
    p.add_argument("--update-epochs",   type=int,   default=2)
    p.add_argument("--num-minibatches", type=int,   default=64,   help="Paper MAPPO base: 64")
    p.add_argument("--lr",              type=float, default=1e-3, help="Paper MAPPO base: 1e-3")
    p.add_argument("--clip-eps",        type=float, default=0.2,  help="Pre-scaling; divided by n_agents")
    p.add_argument("--scale-clip",      action="store_true", default=True)
    p.add_argument("--no-scale-clip",   dest="scale_clip", action="store_false")
    p.add_argument("--gamma",           type=float, default=0.99)
    p.add_argument("--gae-lambda",      type=float, default=0.95)
    p.add_argument("--ent-coef",        type=float, default=0.01)
    p.add_argument("--vf-coef",         type=float, default=0.5)
    p.add_argument("--max-grad-norm",   type=float, default=0.5)
    p.add_argument("--seed",            type=int,   default=30)
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
        "ENV_NAME": "coin_game",
        "ENV_KWARGS": {
            "num_agents":      args.num_agents,
            "num_inner_steps": args.num_steps,
            "shared_rewards":  True,
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
        "SCALE_CLIP_EPS":  args.scale_clip,
        "ENT_COEF":        args.ent_coef,
        "VF_COEF":         args.vf_coef,
        "MAX_GRAD_NORM":   args.max_grad_norm,
        "ACTIVATION":      "relu",
        "ANNEAL_LR":       True,
    }

    n_updates  = int(cfg["TOTAL_TIMESTEPS"] // cfg["NUM_STEPS"] // cfg["NUM_ENVS"])
    clip_eff   = cfg["CLIP_EPS"] / args.num_agents
    print(f"[MAPPO-CNN] coin_game  n_agents={args.num_agents}  "
          f"envs={args.num_envs}  steps/ep={args.num_steps}  updates={n_updates}  "
          f"total={cfg['TOTAL_TIMESTEPS']/1e6:.0f}M  clip={clip_eff:.4f}  seed={args.seed}")
    print(f"GPU: {jax.devices()}")

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

    actor_params  = out["runner_state"][0].params
    critic_params = out["runner_state"][1].params
    save_checkpoint(actor_params,  run_dir / "checkpoints" / "actor_final.npz")
    save_checkpoint(critic_params, run_dir / "checkpoints" / "critic_final.npz")

    ep_rets = np.array(out["metrics"].get("returned_episode_returns", [])).ravel()
    np.save(str(run_dir / "metrics_ep_return.npy"), ep_rets)
    print(f"Final return/agent (last 50): {ep_rets[-50:].mean():.3f}")


if __name__ == "__main__":
    main()
