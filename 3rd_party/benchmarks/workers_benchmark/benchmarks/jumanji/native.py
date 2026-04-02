"""Jumanji native benchmark -- PPO via gymnax (fully JIT-compiled) on CartPole-v1.

Baseline: the entire training loop (environment stepping, rollout collection,
GAE computation, PPO gradient updates) is compiled into a single XLA program
via jax.lax.scan. No Python loop, no host-device transfer, no Gymnasium.

This is what JAX brings to the table: the env IS JAX, so the whole loop
compiles to one XLA HLO graph and runs on-device without interpreter overhead.

Uses gymnax (Lange, 2022) for the JAX-native CartPole-v1 implementation.
Same PPO hyperparameters as all other workers for fair comparison.
"""

import sys
import time

from workers_benchmark.utils import (
    BenchmarkResult, run_subprocess_timed, print_run_header, print_run_result,
)


def run_native_benchmark(config) -> BenchmarkResult:
    """Run fully JIT-compiled JAX PPO on gymnax CartPole-v1."""
    print_run_header(config.worker_name, "native", config.env_id,
                     config.total_timesteps, config.num_envs, config.seed,
                     getattr(config, "_current_iteration", 1),
                     config.iterations)

    script = _build_gymnax_ppo_script(
        env_id=config.env_id,
        total_timesteps=config.total_timesteps,
        num_envs=config.num_envs,
        seed=config.seed,
        learning_rate=config.learning_rate,
        num_steps=config.num_steps,
    )

    cmd = [sys.executable, "-c", script]
    elapsed, peak_mb, stdout, _ = run_subprocess_timed(cmd, timeout=1800)

    # Parse inner elapsed (post-JIT pure execution, excludes compilation)
    inner = _parse_elapsed(stdout)
    if inner > 0:
        elapsed = inner

    sps = config.total_timesteps / elapsed if elapsed > 0 else 0.0

    result = BenchmarkResult(
        worker_name="jumanji",
        scenario="native",
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


def _parse_elapsed(stdout: str) -> float:
    """Parse GYMNAX_RESULT elapsed=X.XX from subprocess stdout."""
    for line in stdout.splitlines():
        if "GYMNAX_RESULT" in line:
            for part in line.split():
                if part.startswith("elapsed="):
                    try:
                        return float(part.split("=")[1])
                    except (IndexError, ValueError):
                        pass
    return 0.0


def _build_gymnax_ppo_script(
    env_id: str,
    total_timesteps: int,
    num_envs: int,
    seed: int,
    learning_rate: float = 2.5e-4,
    num_steps: int = 128,
) -> str:
    """Build a fully JIT-compiled PPO training script using gymnax.

    The entire training loop (env.step + rollout + GAE + PPO update) is
    compiled into a single XLA program via jax.lax.scan. Zero Python
    interpreter overhead during training.
    """
    return f'''\
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax, jax.numpy as jnp, flax.linen as nn, optax, gymnax
from flax.training.train_state import TrainState
from typing import NamedTuple

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

# Pure execution
key2 = jax.random.PRNGKey(SEED + 1000)
key2, *rk2 = jax.random.split(key2, NUM_ENVS + 1)
o2, e2 = jax.vmap(env.reset, in_axes=(0, None))(jnp.stack(rk2), env_params)
import time
t0 = time.perf_counter()
_ = train(ts, o2, e2, key2)
jax.block_until_ready(_)
elapsed = time.perf_counter() - t0
print(f"GYMNAX_RESULT elapsed={{elapsed:.6f}}")
'''
