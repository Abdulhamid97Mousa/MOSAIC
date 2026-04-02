"""Jumanji FastLane benchmarks -- two variants.

1. gymnax FastLane: fully JIT-compiled PPO with jax.debug.callback injecting
   FastLaneWriter.publish() into the XLA scan loop. Every step fires a
   Python callback that writes an 84x84 RGB frame to shared memory.

2. Gymnasium FastLane: Python-loop JAX PPO on Gymnasium CartPole with
   env.render() + FastLaneWriter.publish() every step. This is the
   standard MOSAIC FastLane pathway for environments with renderers.
"""

import sys
import time

from workers_benchmark.utils import (
    BenchmarkResult, run_subprocess_timed, print_run_header, print_run_result,
)

from .native import _build_gymnax_ppo_script, _parse_elapsed


# -----------------------------------------------------------------------
# gymnax FastLane (jax.debug.callback)
# -----------------------------------------------------------------------

def run_fastlane_gymnax_benchmark(config) -> BenchmarkResult:
    """Run gymnax PPO with real FastLane frame streaming via jax.debug.callback."""
    print_run_header(config.worker_name, "fastlane_gymnax", config.env_id,
                     config.total_timesteps, config.num_envs, config.seed,
                     getattr(config, "_current_iteration", 1),
                     config.iterations)

    script = _build_gymnax_fastlane_script(
        total_timesteps=config.total_timesteps,
        num_envs=config.num_envs,
        seed=config.seed,
        learning_rate=config.learning_rate,
        num_steps=config.num_steps,
    )

    cmd = [sys.executable, "-c", script]
    env_overrides = {
        "GYM_GUI_FASTLANE_ONLY": "1",
        "GYM_GUI_FASTLANE_SLOT": "0",
        "GYM_GUI_FASTLANE_VIDEO_MODE": "single",
    }

    elapsed, peak_mb, stdout, _ = run_subprocess_timed(cmd, env=env_overrides, timeout=1800)

    inner = _parse_elapsed(stdout)
    if inner > 0:
        elapsed = inner

    sps = config.total_timesteps / elapsed if elapsed > 0 else 0.0

    result = BenchmarkResult(
        worker_name="jumanji",
        scenario="fastlane_gymnax",
        env_id=config.env_id,
        total_timesteps=config.total_timesteps,
        wall_time_seconds=elapsed,
        steps_per_second=sps,
        peak_memory_mb=peak_mb,
        seed=config.seed,
        num_envs=config.num_envs,
        iteration=getattr(config, "_current_iteration", 1),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    print_run_result(result)
    return result


# -----------------------------------------------------------------------
# Gymnasium FastLane (Python loop + render + publish)
# -----------------------------------------------------------------------

def run_fastlane_gymnasium_benchmark(config) -> BenchmarkResult:
    """Run Gymnasium+JAX PPO with real FastLane frame streaming."""
    print_run_header(config.worker_name, "fastlane_gym", config.env_id,
                     config.total_timesteps, config.num_envs, config.seed,
                     getattr(config, "_current_iteration", 1),
                     config.iterations)

    script = _build_gymnasium_jax_fastlane_script(
        env_id=config.env_id,
        total_timesteps=config.total_timesteps,
        num_envs=config.num_envs,
        seed=config.seed,
        learning_rate=config.learning_rate,
        num_steps=config.num_steps,
    )

    cmd = [sys.executable, "-c", script]
    env_overrides = {
        "GYM_GUI_FASTLANE_ONLY": "1",
        "GYM_GUI_FASTLANE_SLOT": "0",
        "GYM_GUI_FASTLANE_VIDEO_MODE": "single",
    }

    elapsed, peak_mb, stdout, _ = run_subprocess_timed(cmd, env=env_overrides, timeout=1800)

    sps = config.total_timesteps / elapsed if elapsed > 0 else 0.0

    result = BenchmarkResult(
        worker_name="jumanji",
        scenario="fastlane_gym",
        env_id=config.env_id,
        total_timesteps=config.total_timesteps,
        wall_time_seconds=elapsed,
        steps_per_second=sps,
        peak_memory_mb=peak_mb,
        seed=config.seed,
        num_envs=config.num_envs,
        iteration=getattr(config, "_current_iteration", 1),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    print_run_result(result)
    return result


# -----------------------------------------------------------------------
# Script builders
# -----------------------------------------------------------------------

def _build_gymnax_fastlane_script(
    total_timesteps: int,
    num_envs: int,
    seed: int,
    learning_rate: float = 2.5e-4,
    num_steps: int = 128,
) -> str:
    """Fully JIT-compiled gymnax PPO with FastLane via jax.debug.callback."""
    return f'''\
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import time, numpy as np
import jax, jax.numpy as jnp, flax.linen as nn, optax, gymnax
from flax.training.train_state import TrainState
from typing import NamedTuple

# MOSAIC worker imports
try:
    from jumanji_worker.config import JumanjiWorkerConfig
    from jumanji_worker import get_worker_metadata
    _meta = get_worker_metadata()
except ImportError:
    pass

# FastLane writer + real Gymnasium renderer for actual frames
_fl_writer = None
_render_env = None
try:
    from gym_gui.fastlane.buffer import FastLaneWriter, FastLaneConfig, FastLaneMetrics
    import gymnasium as _gym
    _render_env = _gym.make("CartPole-v1", render_mode="rgb_array")
    _render_env.reset()
    # Get actual frame dimensions from the renderer
    _sample_frame = _render_env.render()
    _h, _w, _c = _sample_frame.shape
    _fl_config = FastLaneConfig(width=_w, height=_h, channels=_c, capacity=4)
    _fl_writer = FastLaneWriter.create("bench_gymnax_fl_{seed}", _fl_config)
except Exception:
    pass

def _publish_callback(obs):
    \"\"\"Render real CartPole frame by syncing gymnax state to Gymnasium env.\"\"\"
    if _fl_writer is None or _render_env is None:
        return
    # obs is (NUM_ENVS, 4) JAX array; take env 0 state: [x, x_dot, theta, theta_dot]
    state = np.asarray(obs[0])
    _render_env.unwrapped.state = state
    frame = _render_env.render()
    if frame is not None:
        _fl_writer.publish(
            frame.tobytes(),
            metrics=FastLaneMetrics(last_reward=0.0, rolling_return=0.0, step_rate_hz=0.0),
        )

class ActorCritic(nn.Module):
    action_dim: int
    @nn.compact
    def __call__(self, x):
        a = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        a = nn.tanh(a)
        a = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(a)
        a = nn.tanh(a)
        logits = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(a)
        c = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        c = nn.tanh(c)
        c = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(c)
        c = nn.tanh(c)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(c)
        return logits, value.squeeze(-1)

class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    log_prob: jnp.ndarray
    value: jnp.ndarray

NUM_ENVS = {num_envs}
NUM_STEPS = {num_steps}
TOTAL = {total_timesteps}
LR = {learning_rate}
GAMMA, GAE_LAMBDA = 0.99, 0.95
UPDATE_EPOCHS, NUM_MB = 4, 4
CLIP, ENT, VF, MAXGN = 0.2, 0.01, 0.5, 0.5
BATCH = NUM_ENVS * NUM_STEPS
MB_SIZE = BATCH // NUM_MB
N_ITER = TOTAL // BATCH
SEED = {seed}

env, env_params = gymnax.make("CartPole-v1")
key = jax.random.PRNGKey(SEED)
network = ActorCritic(action_dim=env.num_actions)
key, ik = jax.random.split(key)
params = network.init(ik, jnp.zeros((1, *env.obs_shape)))
tx = optax.chain(optax.clip_by_global_norm(MAXGN), optax.adam(LR, eps=1e-5))
ts = TrainState.create(apply_fn=network.apply, params=params, tx=tx)
key, *rks = jax.random.split(key, NUM_ENVS + 1)
obs, es = jax.vmap(env.reset, in_axes=(0, None))(jnp.stack(rks), env_params)

def _env_step(carry, _):
    ts, obs, es, key = carry
    key, ak, sk = jax.random.split(key, 3)
    logits, value = ts.apply_fn(ts.params, obs)
    action = jax.random.categorical(ak, logits)
    log_prob = jax.nn.log_softmax(logits)[jnp.arange(NUM_ENVS), action]
    sks = jax.random.split(sk, NUM_ENVS)
    nobs, nes, rew, done, _ = jax.vmap(env.step, in_axes=(0,0,0,None))(sks, es, action, env_params)
    jax.debug.callback(_publish_callback, obs)
    return (ts, nobs, nes, key), Transition(obs, action, rew, done, log_prob, value)

def _ppo_loss(params, apply_fn, mo, ma, ml, mad, mr, mv):
    logits, nv = apply_fn(params, mo)
    nlp = jax.nn.log_softmax(logits)[jnp.arange(ma.shape[0]), ma]
    ent = -(jax.nn.softmax(logits) * jax.nn.log_softmax(logits)).sum(-1).mean()
    ratio = jnp.exp(nlp - ml)
    an = (mad - mad.mean()) / (mad.std() + 1e-8)
    pg = jnp.maximum(-an * ratio, -an * jnp.clip(ratio, 1-CLIP, 1+CLIP)).mean()
    vc = mv + jnp.clip(nv - mv, -CLIP, CLIP)
    vl = 0.5 * jnp.maximum((nv - mr)**2, (vc - mr)**2).mean()
    return pg - ENT * ent + vl * VF

def _ppo_iter(carry, _):
    ts, obs, es, key = carry
    (ts, obs, es, key), traj = jax.lax.scan(_env_step, (ts, obs, es, key), None, length=NUM_STEPS)
    _, lv = ts.apply_fn(ts.params, obs)
    def _gae(carry, t):
        lg, nv = carry
        d = t.reward + GAMMA * nv * (1-t.done) - t.value
        lg = d + GAMMA * GAE_LAMBDA * (1-t.done) * lg
        return (lg, t.value), lg
    _, adv = jax.lax.scan(_gae, (jnp.zeros(NUM_ENVS), lv), traj, reverse=True)
    ret = adv + traj.value
    bo = traj.obs.reshape((BATCH, *env.obs_shape))
    ba = traj.action.reshape(BATCH)
    bl = traj.log_prob.reshape(BATCH)
    bv = traj.value.reshape(BATCH)
    bad = adv.reshape(BATCH)
    br = ret.reshape(BATCH)
    def _epoch(carry, _):
        ts, key = carry
        key, pk = jax.random.split(key)
        p = jax.random.permutation(pk, BATCH)
        so = bo[p].reshape(NUM_MB, MB_SIZE, *env.obs_shape)
        sa = ba[p].reshape(NUM_MB, MB_SIZE)
        sl = bl[p].reshape(NUM_MB, MB_SIZE)
        sv = bv[p].reshape(NUM_MB, MB_SIZE)
        sd = bad[p].reshape(NUM_MB, MB_SIZE)
        sr = br[p].reshape(NUM_MB, MB_SIZE)
        def _mb(ts, mb):
            g = jax.value_and_grad(_ppo_loss)
            _, grads = g(ts.params, ts.apply_fn, *mb)
            return ts.apply_gradients(grads=grads), None
        ts, _ = jax.lax.scan(_mb, ts, (so, sa, sl, sd, sr, sv))
        return (ts, key), None
    (ts, key), _ = jax.lax.scan(_epoch, (ts, key), None, length=UPDATE_EPOCHS)
    return (ts, obs, es, key), None

@jax.jit
def train(ts, obs, es, key):
    (ts, obs, es, key), _ = jax.lax.scan(_ppo_iter, (ts, obs, es, key), None, length=N_ITER)
    return ts

# Compile
_ = train(ts, obs, es, key)
jax.block_until_ready(_)

# Pure execution with FastLane
key2 = jax.random.PRNGKey(SEED + 1000)
key2, *rk2 = jax.random.split(key2, NUM_ENVS + 1)
o2, e2 = jax.vmap(env.reset, in_axes=(0, None))(jnp.stack(rk2), env_params)
t0 = time.perf_counter()
_ = train(ts, o2, e2, key2)
jax.block_until_ready(_)
elapsed = time.perf_counter() - t0
print(f"GYMNAX_RESULT elapsed={{elapsed:.6f}}")

if _fl_writer is not None:
    try:
        _fl_writer.close()
    except Exception:
        pass
'''


def _build_gymnasium_jax_fastlane_script(
    env_id: str,
    total_timesteps: int,
    num_envs: int,
    seed: int,
    learning_rate: float = 2.5e-4,
    num_steps: int = 128,
) -> str:
    """JAX PPO on Gymnasium CartPole with real FastLane frame publishing.

    Uses CleanRL's sitecustomize + FastLaneTelemetryWrapper to inject
    render_mode="rgb_array" and wrap each sub-env. Only slot 0 renders
    and publishes frames, matching the real MOSAIC FastLane pathway.
    """
    return f'''\
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CLEANRL_RUN_ID"] = "bench_jumanji_gymfl_{seed}"
os.environ["RUN_ID"] = "bench_jumanji_gymfl_{seed}"

import sys, time, logging
import numpy as np

# Load sitecustomize BEFORE importing gym -- patches gym.make to inject
# render_mode="rgb_array" and wrap with FastLaneTelemetryWrapper
import cleanrl_worker.sitecustomize  # noqa: F401

import gymnasium as gym
import jax, jax.numpy as jnp, flax.linen as nn, optax
from flax.training.train_state import TrainState

# MOSAIC worker imports
try:
    from jumanji_worker.config import JumanjiWorkerConfig
    from jumanji_worker import get_worker_metadata
    _meta = get_worker_metadata()
except ImportError:
    pass

class ActorCritic(nn.Module):
    action_dim: int
    @nn.compact
    def __call__(self, x):
        a = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        a = nn.tanh(a)
        a = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(a)
        a = nn.tanh(a)
        logits = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(a)
        c = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        c = nn.tanh(c)
        c = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(c)
        c = nn.tanh(c)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(c)
        return logits, value.squeeze(-1)

seed = {seed}; num_envs = {num_envs}; num_steps = {num_steps}; lr = {learning_rate}
total_timesteps = {total_timesteps}
batch_size = num_envs * num_steps
num_iterations = total_timesteps // batch_size

np.random.seed(seed)
key = jax.random.PRNGKey(seed)

# gym.make is patched by sitecustomize: auto-injects render_mode="rgb_array"
# and wraps with FastLaneTelemetryWrapper (only slot 0 renders/publishes)
envs = gym.vector.SyncVectorEnv([
    lambda i=i: gym.make("{env_id}") for i in range(num_envs)
])
obs_shape = envs.single_observation_space.shape
n_actions = envs.single_action_space.n

network = ActorCritic(action_dim=n_actions)
key, init_key = jax.random.split(key)
params = network.init(init_key, jnp.zeros((1,) + obs_shape))
tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr, eps=1e-5))
train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

@jax.jit
def get_action_and_value(state, obs, rng):
    logits, value = state.apply_fn(state.params, obs)
    rng, subkey = jax.random.split(rng)
    action = jax.random.categorical(subkey, logits)
    log_prob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
    return action, log_prob, value, rng

@jax.jit
def get_value(state, obs):
    _, value = state.apply_fn(state.params, obs)
    return value

def ppo_loss_fn(params, apply_fn, mb_obs, mb_act, mb_logp, mb_adv, mb_ret, mb_val):
    logits, newvalue = apply_fn(params, mb_obs)
    newlogp = jax.nn.log_softmax(logits)[jnp.arange(mb_act.shape[0]), mb_act]
    entropy = -(jax.nn.softmax(logits) * jax.nn.log_softmax(logits)).sum(-1).mean()
    ratio = jnp.exp(newlogp - mb_logp)
    adv_n = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
    pg1 = -adv_n * ratio
    pg2 = -adv_n * jnp.clip(ratio, 1 - 0.2, 1 + 0.2)
    pg_loss = jnp.maximum(pg1, pg2).mean()
    v_clipped = mb_val + jnp.clip(newvalue - mb_val, -0.2, 0.2)
    v_loss = 0.5 * jnp.maximum((newvalue - mb_ret)**2, (v_clipped - mb_ret)**2).mean()
    return pg_loss - 0.01 * entropy + v_loss * 0.5

@jax.jit
def update_step(state, mb_obs, mb_act, mb_logp, mb_adv, mb_ret, mb_val):
    grad_fn = jax.value_and_grad(ppo_loss_fn)
    loss, grads = grad_fn(state.params, state.apply_fn, mb_obs, mb_act, mb_logp, mb_adv, mb_ret, mb_val)
    return state.apply_gradients(grads=grads), loss

# Warmup
obs_jax = jnp.zeros((num_envs,) + obs_shape)
_ = get_action_and_value(train_state, obs_jax, key)
_ = get_value(train_state, obs_jax)
dummy_mb = jnp.zeros((128,) + obs_shape)
dummy_act = jnp.zeros(128, dtype=jnp.int32)
dummy_f = jnp.zeros(128)
_ = update_step(train_state, dummy_mb, dummy_act, dummy_f, dummy_f, dummy_f, dummy_f)
jax.block_until_ready(_)

obs_buf = np.zeros((num_steps, num_envs) + obs_shape, dtype=np.float32)
act_buf = np.zeros((num_steps, num_envs), dtype=np.int32)
logp_buf = np.zeros((num_steps, num_envs), dtype=np.float32)
rew_buf = np.zeros((num_steps, num_envs), dtype=np.float32)
done_buf = np.zeros((num_steps, num_envs), dtype=np.float32)
val_buf = np.zeros((num_steps, num_envs), dtype=np.float32)

next_obs, _ = envs.reset(seed=seed)
next_done = np.zeros(num_envs)

start = time.perf_counter()

for iteration in range(1, num_iterations + 1):
    for step in range(num_steps):
        obs_buf[step] = next_obs
        done_buf[step] = next_done
        obs_jax = jnp.array(next_obs)
        action, log_prob, value, key = get_action_and_value(train_state, obs_jax, key)
        act_np = np.array(action)
        logp_buf[step] = np.array(log_prob)
        val_buf[step] = np.array(value)
        act_buf[step] = act_np
        # FastLaneTelemetryWrapper.step() auto-renders and publishes for slot 0
        next_obs, reward, terminations, truncations, infos = envs.step(act_np)
        next_done = np.logical_or(terminations, truncations).astype(np.float32)
        rew_buf[step] = reward

    next_value = np.array(get_value(train_state, jnp.array(next_obs)))
    advantages = np.zeros_like(rew_buf)
    lastgaelam = 0
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - done_buf[t + 1]
            nextvalues = val_buf[t + 1]
        delta = rew_buf[t] + 0.99 * nextvalues * nextnonterminal - val_buf[t]
        advantages[t] = lastgaelam = delta + 0.99 * 0.95 * nextnonterminal * lastgaelam
    returns = advantages + val_buf

    b_obs = jnp.array(obs_buf.reshape((-1,) + obs_shape))
    b_act = jnp.array(act_buf.reshape(-1))
    b_logp = jnp.array(logp_buf.reshape(-1))
    b_adv = jnp.array(advantages.reshape(-1))
    b_ret = jnp.array(returns.reshape(-1))
    b_val = jnp.array(val_buf.reshape(-1))

    b_inds = np.arange(batch_size)
    for _ in range(4):
        np.random.shuffle(b_inds)
        for si in range(0, batch_size, batch_size // 4):
            mb = b_inds[si:si + batch_size // 4]
            train_state, loss = update_step(
                train_state, b_obs[mb], b_act[mb], b_logp[mb],
                b_adv[mb], b_ret[mb], b_val[mb],
            )

elapsed = time.perf_counter() - start
sps = total_timesteps / elapsed
print(f"Gymnasium+JAX PPO + FastLane: {{elapsed:.2f}}s, {{sps:.0f}} SPS")
'''
