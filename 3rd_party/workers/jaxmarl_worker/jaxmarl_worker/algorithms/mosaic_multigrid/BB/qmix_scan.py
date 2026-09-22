"""QMIX for MOSAIC multigrid Basketball environment (GRU Q-networks + monotonic mixing).

Architecture:
  - Per-agent RNNQNetwork: GRU hidden state → Q(a | o_id, h)
    obs_id = [obs | one-hot agent ID]  (EPyMARL obs_agent_id)
  - MixingNetwork: QMIX hypernetwork over global state → monotonic Q_tot

Key properties vs IPPO/MAPPO:
  - Off-policy with flashbax trajectory replay buffer
  - ε-greedy exploration (not policy gradient)
  - GRU hidden state maintained across collection steps
  - Agent IDs appended inside algorithm (buffer stores raw obs)

Checkpoint format (final.npz):
  agent_q_{i}  — RNNQNetwork params (leaves, jax.tree_util.tree_flatten order)
  mixer_{i}    — MixingNetwork params (leaves)
"""

import os
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

import argparse
import re as _re
import time
from functools import partial
from pathlib import Path
from typing import Any

import chex
import flashbax as fbx
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jaxmarl.wrappers.baselines import CTRolloutManager


# ---------------------------------------------------------------------------
# Networks (ported from JaxMARL qmix_rnn.py)
# ---------------------------------------------------------------------------

class ScannedRNN(nn.Module):
    @partial(nn.scan, variable_broadcast="params", in_axes=0, out_axes=0,
             split_rngs={"params": False})
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        hidden_size = ins.shape[-1]
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(hidden_size, *ins.shape[:-1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(hidden_size)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        return nn.GRUCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (*batch_size, hidden_size)
        )


class RNNQNetwork(nn.Module):
    action_dim: int
    hidden_dim: int
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, hidden, obs, dones):
        embedding = nn.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale),
                             bias_init=constant(0.0))(obs)
        embedding = nn.relu(embedding)
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        q_vals = nn.Dense(self.action_dim, kernel_init=orthogonal(self.init_scale),
                          bias_init=constant(0.0))(embedding)
        return hidden, q_vals


class HyperNetwork(nn.Module):
    hidden_dim: int
    output_dim: int
    init_scale: float

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale),
                     bias_init=constant(0.0))(x)
        x = nn.relu(x)
        return nn.Dense(self.output_dim, kernel_init=orthogonal(self.init_scale),
                        bias_init=constant(0.0))(x)


class MixingNetwork(nn.Module):
    embedding_dim: int
    hypernet_hidden_dim: int
    init_scale: float

    @nn.compact
    def __call__(self, q_vals, states):
        n_agents, time_steps, batch_size = q_vals.shape
        q_vals = jnp.transpose(q_vals, (1, 2, 0))  # (T, B, n_agents)

        w_1 = HyperNetwork(self.hypernet_hidden_dim, self.embedding_dim * n_agents,
                           self.init_scale)(states)
        b_1 = nn.Dense(self.embedding_dim, kernel_init=orthogonal(self.init_scale),
                       bias_init=constant(0.0))(states)
        w_2 = HyperNetwork(self.hypernet_hidden_dim, self.embedding_dim, self.init_scale)(states)
        b_2 = HyperNetwork(self.embedding_dim, 1, self.init_scale)(states)

        w_1 = jnp.abs(w_1.reshape(time_steps, batch_size, n_agents, self.embedding_dim))
        b_1 = b_1.reshape(time_steps, batch_size, 1, self.embedding_dim)
        w_2 = jnp.abs(w_2.reshape(time_steps, batch_size, self.embedding_dim, 1))
        b_2 = b_2.reshape(time_steps, batch_size, 1, 1)

        hidden = nn.elu(jnp.matmul(q_vals[:, :, None, :], w_1) + b_1)
        q_tot = jnp.matmul(hidden, w_2) + b_2
        return q_tot.squeeze()  # (T, B)


# ---------------------------------------------------------------------------
# Train state
# ---------------------------------------------------------------------------

@chex.dataclass(frozen=True)
class Timestep:
    obs: dict
    actions: dict
    rewards: dict
    dones: dict
    avail_actions: dict


class CustomTrainState(TrainState):
    target_network_params: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def make_train(config):
    env = config["ENV"]

    wrapped_env = CTRolloutManager(env, batch_size=config["NUM_ENVS"])
    test_env    = CTRolloutManager(env, batch_size=config["TEST_NUM_ENVS"])

    num_updates = config["NUM_UPDATES"]
    n_agents    = env.num_agents
    # obs_dim_net: raw obs + one-hot agent ID (EPyMARL obs_agent_id)
    obs_dim_net = wrapped_env.obs_size + n_agents

    eps_scheduler = optax.linear_schedule(
        init_value=config["EPS_START"],
        end_value=config["EPS_FINISH"],
        transition_steps=config["EPS_DECAY"] * num_updates,
    )

    def get_greedy_actions(q_vals, valid_actions):
        q_vals = q_vals - (1 - valid_actions) * 1e10
        return jnp.argmax(q_vals, axis=-1)

    def eps_greedy_exploration(rng, q_vals, eps, valid_actions):
        rng_a, rng_e = jax.random.split(rng)
        greedy = get_greedy_actions(q_vals, valid_actions)

        def rand_action(rng, va):
            return jax.random.choice(rng, jnp.arange(va.shape[-1]),
                                     p=va / jnp.sum(va, axis=-1))

        _rngs = jax.random.split(rng_a, valid_actions.shape[0])
        random = jax.vmap(rand_action)(_rngs, valid_actions)
        return jnp.where(jax.random.uniform(rng_e, greedy.shape) < eps, random, greedy)

    def batchify(x: dict):
        return jnp.stack([x[a] for a in env.agents], axis=0)

    def unbatchify(x):
        return {a: x[i] for i, a in enumerate(env.agents)}

    def add_agent_ids(obs_batched):
        """Append one-hot agent IDs to batched obs.

        obs_batched: (n_agents, T, B, obs_dim)
        returns:     (n_agents, T, B, obs_dim + n_agents)
        """
        eye = jnp.eye(n_agents)                          # (n_agents, n_agents)
        agent_ids = eye[:, np.newaxis, np.newaxis, :]    # (n_agents, 1, 1, n_agents)
        agent_ids = jnp.broadcast_to(agent_ids, (*obs_batched.shape[:3], n_agents))
        return jnp.concatenate([obs_batched, agent_ids], axis=-1)

    def train(rng):
        # Sample a trajectory to infer buffer shape (stores raw obs, no agent IDs)
        def _env_sample_step(env_state, _):
            rng0 = jax.random.PRNGKey(0)
            ks = jax.random.split(rng0, env.num_agents + 1)
            actions = {a: wrapped_env.batch_sample(ks[i], a)
                       for i, a in enumerate(env.agents)}
            avail = wrapped_env.get_valid_actions(env_state)
            obs, env_state, rewards, dones, infos = wrapped_env.batch_step(
                ks[-1], env_state, actions)
            return env_state, Timestep(obs=obs, actions=actions, rewards=rewards,
                                       dones=dones, avail_actions=avail)

        rng, _rng = jax.random.split(rng)
        _init_obs, _env_state = wrapped_env.batch_reset(_rng)
        _, sample_traj = jax.lax.scan(_env_sample_step, _env_state, None,
                                       config["NUM_STEPS"])
        sample_traj_ub = jax.tree.map(lambda x: x[:, 0], sample_traj)

        # Network init — Q-network receives obs + agent one-hot
        network = RNNQNetwork(action_dim=wrapped_env.max_action_space,
                              hidden_dim=config["HIDDEN_DIM"])
        mixer   = MixingNetwork(config["MIXER_EMBEDDING_DIM"],
                                config["MIXER_HYPERNET_HIDDEN_DIM"],
                                config["MIXER_INIT_SCALE"])

        rng, _rng = jax.random.split(rng)
        init_hs = ScannedRNN.initialize_carry(config["HIDDEN_DIM"], 1)
        agent_params = network.init(_rng, init_hs,
                                    jnp.zeros((1, 1, obs_dim_net)),
                                    jnp.zeros((1, 1)))
        rng, _rng = jax.random.split(rng)
        state_dim = sample_traj.obs["__all__"].shape[-1]
        mixer_params = mixer.init(_rng,
                                  jnp.zeros((env.num_agents, 1, 1)),
                                  jnp.zeros((1, 1, state_dim)))
        network_params = {"agent": agent_params, "mixer": mixer_params}

        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.radam(learning_rate=config["LR"]),
        )
        train_state = CustomTrainState.create(
            apply_fn=network.apply,
            params=network_params,
            target_network_params=network_params,
            tx=tx,
        )

        # Replay buffer stores raw obs (agent IDs added on-the-fly during training)
        buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=config["BUFFER_SIZE"] // config["NUM_ENVS"],
            min_length_time_axis=config["BUFFER_BATCH_SIZE"],
            sample_batch_size=config["BUFFER_BATCH_SIZE"],
            add_batch_size=config["NUM_ENVS"],
            sample_sequence_length=1,
            period=1,
        )
        buffer_state = buffer.init(sample_traj_ub)

        def _update_step(runner_state, _):
            train_state, buffer_state, rng = runner_state

            # --- Collection ---
            def _step_env(carry, _):
                hs, last_obs, last_dones, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)

                _obs   = batchify(last_obs)[:, np.newaxis]    # (n_agents, 1, n_envs, obs_dim)
                _dones = batchify(last_dones)[:, np.newaxis]  # (n_agents, 1, n_envs)
                # Append agent one-hot IDs before Q-network (EPyMARL obs_agent_id)
                _obs_id = add_agent_ids(_obs)                 # (n_agents, 1, n_envs, obs_dim_net)
                new_hs, q_vals = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                    train_state.params["agent"], hs, _obs_id, _dones)
                q_vals = q_vals.squeeze(axis=1)  # (n_agents, n_envs, n_actions)

                avail = wrapped_env.get_valid_actions(env_state)
                eps = eps_scheduler(train_state.n_updates)
                _rngs = jax.random.split(rng_a, env.num_agents)
                actions = jax.vmap(eps_greedy_exploration, in_axes=(0, 0, None, 0))(
                    _rngs, q_vals, eps, batchify(avail))
                actions = unbatchify(actions)

                new_obs, new_state, rewards, dones, infos = wrapped_env.batch_step(
                    rng_s, env_state, actions)
                # Store raw obs — agent IDs added in learning phase
                ts = Timestep(obs=last_obs, actions=actions, rewards=rewards,
                              dones=last_dones, avail_actions=avail)
                return (new_hs, new_obs, dones, new_state, rng), (ts, infos)

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = wrapped_env.batch_reset(_rng)
            init_dones = {a: jnp.zeros(config["NUM_ENVS"], dtype=bool)
                          for a in env.agents + ["__all__"]}
            init_hs = ScannedRNN.initialize_carry(
                config["HIDDEN_DIM"], env.num_agents, config["NUM_ENVS"])
            rng, _rng = jax.random.split(rng)
            _, (timesteps, infos) = jax.lax.scan(
                _step_env, (init_hs, init_obs, init_dones, env_state, _rng),
                None, config["NUM_STEPS"])

            train_state = train_state.replace(
                timesteps=train_state.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"])

            buf_batch = jax.tree.map(
                lambda x: jnp.swapaxes(x, 0, 1)[:, np.newaxis], timesteps)
            buffer_state = buffer.add(buffer_state, buf_batch)

            # --- Learning ---
            def _learn_phase(carry, _):
                train_state, rng = carry
                rng, _rng = jax.random.split(rng)
                mb = buffer.sample(buffer_state, _rng).experience
                # (batch, 1, NUM_STEPS, ...) → (NUM_STEPS, batch, ...)
                mb = jax.tree.map(lambda x: jnp.swapaxes(x[:, 0], 0, 1), mb)

                init_hs_learn = ScannedRNN.initialize_carry(
                    config["HIDDEN_DIM"], env.num_agents, config["BUFFER_BATCH_SIZE"])
                _obs   = batchify(mb.obs)            # (n_agents, NUM_STEPS, batch, obs_dim)
                _dones = batchify(mb.dones)
                _acts  = batchify(mb.actions)
                _avail = batchify(mb.avail_actions)

                # Add agent one-hot IDs before Q-network (EPyMARL obs_agent_id)
                _obs_id = add_agent_ids(_obs)        # (n_agents, NUM_STEPS, batch, obs_dim_net)

                _, q_next_target = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                    train_state.target_network_params["agent"],
                    init_hs_learn, _obs_id, _dones)

                def _loss_fn(params):
                    _, q_vals = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                        params["agent"], init_hs_learn, _obs_id, _dones)

                    chosen_q = jnp.take_along_axis(q_vals, _acts[..., np.newaxis],
                                                   axis=-1).squeeze(-1)
                    valid_q_next = q_next_target - (1 - _avail) * 1e10
                    q_next = jnp.take_along_axis(
                        q_next_target,
                        jnp.argmax(valid_q_next, axis=-1)[..., np.newaxis],
                        axis=-1).squeeze(-1)

                    # Mixer uses global state (raw concatenated obs, no agent IDs)
                    q_tot_next   = mixer.apply(train_state.target_network_params["mixer"],
                                               q_next,   mb.obs["__all__"])
                    q_tot_target = (mb.rewards["__all__"][:-1]
                                    + (1 - mb.dones["__all__"][:-1])
                                    * config["GAMMA"]
                                    * q_tot_next[1:])
                    q_tot = mixer.apply(params["mixer"], chosen_q, mb.obs["__all__"])[:-1]
                    loss = jnp.mean((q_tot - jax.lax.stop_gradient(q_tot_target)) ** 2)
                    return loss, chosen_q.mean()

                (loss, qmean), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
                    train_state.params)
                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(grad_steps=train_state.grad_steps + 1)
                return (train_state, rng), (loss, qmean)

            rng, _rng = jax.random.split(rng)
            is_learn = buffer.can_sample(buffer_state) & (
                train_state.timesteps > config["LEARNING_STARTS"])
            (train_state, rng), (loss, qmean) = jax.lax.cond(
                is_learn,
                lambda ts, r: jax.lax.scan(_learn_phase, (ts, r), None, config["NUM_EPOCHS"]),
                lambda ts, r: ((ts, r), (jnp.zeros(config["NUM_EPOCHS"]),
                                         jnp.zeros(config["NUM_EPOCHS"]))),
                train_state, _rng)

            # Target network update
            train_state = jax.lax.cond(
                train_state.n_updates % config["TARGET_UPDATE_INTERVAL"] == 0,
                lambda ts: ts.replace(target_network_params=optax.incremental_update(
                    ts.params, ts.target_network_params, config["TAU"])),
                lambda ts: ts,
                operand=train_state)

            train_state = train_state.replace(n_updates=train_state.n_updates + 1)

            metrics = {
                "env_step":  train_state.timesteps,
                "n_updates": train_state.n_updates,
                "loss":      loss.mean(),
                "qmean":     qmean.mean(),
                "epsilon":   eps_scheduler(train_state.n_updates),
            }
            metrics.update(jax.tree.map(lambda x: x.mean(), infos))

            return (train_state, buffer_state, rng), metrics

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, buffer_state, _rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"])
        return runner_state, metrics

    return train


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------

def _make_env(variant: str, view_size: int = 7, ball_coef: float = 0.01,
              max_steps: int = 256, goal_rows=None):
    from jaxmarl_worker.environments.mosaic_multigrid.BB.basketball_jax import BasketballJAX
    return BasketballJAX(variant=variant, view_size=view_size, ball_coef=ball_coef,
                         max_steps=max_steps, goal_rows=goal_rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="QMIX training for BB Basketball (MOSAIC multigrid)")
    p.add_argument("--training-reward", type=str, default="zero-sum",
                   choices=["zero-sum", "cooperative-no-opponent", "general-sum"])
    p.add_argument("--variant",       type=str,   default="2v2")
    p.add_argument("--run-dir",       type=str,   required=True)
    p.add_argument("--total-updates", type=int,   default=20_000)
    p.add_argument("--n-envs",        type=int,   default=256)
    p.add_argument("--n-steps",       type=int,   default=256)
    p.add_argument("--hidden-dim",    type=int,   default=256)
    p.add_argument("--mixer-embed-dim",    type=int,   default=64)
    p.add_argument("--mixer-hypernet-dim", type=int,   default=256)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--gamma",         type=float, default=0.99)
    p.add_argument("--eps-start",     type=float, default=1.0)
    p.add_argument("--eps-finish",    type=float, default=0.05)
    p.add_argument("--eps-decay",     type=float, default=0.1)
    p.add_argument("--buffer-size",   type=int,   default=12_800)
    p.add_argument("--buffer-batch-size", type=int, default=32)
    p.add_argument("--learning-starts", type=int, default=10_000)
    p.add_argument("--n-epochs",      type=int,   default=8)
    p.add_argument("--max-grad-norm", type=float, default=10.0)
    p.add_argument("--target-update-interval", type=int, default=10)
    p.add_argument("--tau",           type=float, default=1.0)
    p.add_argument("--test-num-envs", type=int,   default=64)
    p.add_argument("--seed",          type=int,   default=1)
    p.add_argument("--log-every",     type=int,   default=50)
    p.add_argument("--tensorboard",   action="store_true")
    p.add_argument("--view-size",          type=int,   default=7)
    p.add_argument("--ball-approach-coef", type=float, default=0.01)
    p.add_argument("--max-steps",          type=int,   default=256)
    p.add_argument("--goal-rows",          type=int,   nargs="+", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    assert args.training_reward in ["zero-sum", "cooperative-no-opponent", "general-sum"]

    m = _re.search(r'(\d+)v(\d+)', args.variant)
    if m is None:
        raise ValueError(f"Invalid variant: {args.variant}. Expected NvM (e.g., G-2v0, 2v2)")
    team_a, team_b = int(m.group(1)), int(m.group(2))
    if args.training_reward in ["zero-sum", "general-sum"]:
        assert team_a == team_b, \
            f"Variant {args.variant}: {args.training_reward} requires equal teams"

    run_dir  = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints" / f"bb-{args.variant}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    env = _make_env(args.variant, view_size=args.view_size,
                    ball_coef=args.ball_approach_coef, max_steps=args.max_steps,
                    goal_rows=args.goal_rows)

    obs_dim_net = env._obs_dim + env.num_agents
    total_env_steps = args.total_updates * args.n_envs * args.n_steps * env.num_agents

    print(f"[QMIX-scan] bb-{args.variant}  "
          f"n_agents={env.num_agents}  n_envs={args.n_envs}  "
          f"n_steps={args.n_steps}  total_updates={args.total_updates}  "
          f"obs_dim={env._obs_dim}  obs_dim_net={obs_dim_net}")
    print(f"  total env steps = {total_env_steps:,}")

    config = {
        "ENV":                    env,
        "NUM_ENVS":               args.n_envs,
        "NUM_STEPS":              args.n_steps,
        "NUM_UPDATES":            args.total_updates,
        "HIDDEN_DIM":             args.hidden_dim,
        "MIXER_EMBEDDING_DIM":    args.mixer_embed_dim,
        "MIXER_HYPERNET_HIDDEN_DIM": args.mixer_hypernet_dim,
        "MIXER_INIT_SCALE":       0.001,
        "LR":                     args.lr,
        "GAMMA":                  args.gamma,
        "EPS_START":              args.eps_start,
        "EPS_FINISH":             args.eps_finish,
        "EPS_DECAY":              args.eps_decay,
        "BUFFER_SIZE":            args.buffer_size,
        "BUFFER_BATCH_SIZE":      args.buffer_batch_size,
        "LEARNING_STARTS":        args.learning_starts,
        "NUM_EPOCHS":             args.n_epochs,
        "MAX_GRAD_NORM":          args.max_grad_norm,
        "TARGET_UPDATE_INTERVAL": args.target_update_interval,
        "TAU":                    args.tau,
        "TEST_NUM_ENVS":          args.test_num_envs,
        "TEST_NUM_STEPS":         args.max_steps,
    }

    tb_writer = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
        except ImportError:
            print("[QMIX-scan] tensorboard not available, skipping")

    train_fn = jax.jit(make_train(config))

    print("[QMIX-scan] JIT-compiling...", flush=True)
    t0 = time.time()
    rng = jax.random.PRNGKey(args.seed)
    (runner_state, buffer_state, _), metrics = jax.block_until_ready(train_fn(rng))
    t1 = time.time()

    print(f"[QMIX-scan] Done in {t1 - t0:.1f}s  ({total_env_steps / (t1 - t0):,.0f} ts/sec)")
    losses = np.array(metrics["loss"])
    print(f"  final loss (last 50): {losses[-50:].mean():.4f}")

    if tb_writer:
        for i, (loss, qm) in enumerate(zip(
                np.array(metrics["loss"]), np.array(metrics["qmean"]))):
            step = i * args.n_envs * args.n_steps
            tb_writer.add_scalar("train/loss",  float(loss), step)
            tb_writer.add_scalar("train/qmean", float(qm),   step)
        if "returned_episode_returns" in metrics:
            for i, er in enumerate(np.array(metrics["returned_episode_returns"])):
                tb_writer.add_scalar("train/episode_return", float(er),
                                     i * args.n_envs * args.n_steps)
        tb_writer.add_hparams(
            hparam_dict={
                "alg":                 "QMIX",
                "env/sport":           "bb",
                "env/variant":         args.variant,
                "env/n_agents":        env.num_agents,
                "env/obs_dim":         env._obs_dim,
                "env/obs_dim_net":     obs_dim_net,
                "env/view_size":       args.view_size,
                "train/reward":        args.training_reward,
                "train/n_envs":        args.n_envs,
                "train/n_steps":       args.n_steps,
                "train/total_updates": args.total_updates,
                "net/hidden_dim":      args.hidden_dim,
                "net/mixer_embed":     args.mixer_embed_dim,
                "net/mixer_hyper":     args.mixer_hypernet_dim,
                "opt/lr":              args.lr,
                "opt/gamma":           args.gamma,
                "opt/eps_start":       args.eps_start,
                "opt/eps_finish":      args.eps_finish,
                "opt/eps_decay":       args.eps_decay,
                "opt/n_epochs":        args.n_epochs,
                "opt/buffer_size":     args.buffer_size,
                "opt/buffer_batch":    args.buffer_batch_size,
                "opt/learn_start":     args.learning_starts,
            },
            metric_dict={"hparam/final_loss": float(losses[-50:].mean())},
        )
        tb_writer.close()

    # Save checkpoint — named leaves for unambiguous evaluation loading
    train_state = runner_state
    agent_leaves, _ = jax.tree_util.tree_flatten(
        jax.tree_util.tree_map(np.array, train_state.params["agent"]))
    mixer_leaves, _ = jax.tree_util.tree_flatten(
        jax.tree_util.tree_map(np.array, train_state.params["mixer"]))

    final_path = ckpt_dir / "final.npz"
    np.savez(str(final_path),
             **{f"agent_q_{i}": v for i, v in enumerate(agent_leaves)},
             **{f"mixer_{i}":   v for i, v in enumerate(mixer_leaves)})
    print(f"[QMIX-scan] checkpoint → {final_path}")
    print(f"  agent_q params: {len(agent_leaves)} leaves")
    print(f"  mixer params:   {len(mixer_leaves)} leaves")


if __name__ == "__main__":
    main()
