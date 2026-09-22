"""Multi-Agent Transformer (MAT) actor in flax -- a faithful JAX port of
Wen et al. 2022, "Multi-Agent RL is a Sequence Modeling Problem"
(reference: 3rd_party/environments/Multi-Agent-Transformer/mat/algorithms/mat/algorithm/ma_transformer.py).

Ported component-by-component to match the reference so it can be cited as such:
  - SelfAttention: forward(key, value, query); 1/sqrt(hs) scaling; tril causal mask.
  - EncodeBlock (POST-LN): x = ln1(x + attn(x,x,x)); x = ln2(x + mlp(x)); attn unmasked.
  - DecodeBlock (POST-LN): x = ln1(x + attn1(x,x,x));                 # masked self-attn
                           x = ln2(rep_enc + attn2(key=x,value=x,query=rep_enc));  # masked cross-attn
                           x = ln3(x + mlp(x))
  - Encoder: obs_encoder = LN -> Linear -> GELU; rep = blocks(ln(emb)); value head.
  - Decoder: action_encoder = Linear(bias=False) -> GELU; x = ln(emb); blocks; head.
  - init_: orthogonal(gain), gain = relu-gain (sqrt2) if `activate` else 0.01; bias 0.
  - GELU is exact (erf), matching torch nn.GELU default; LayerNorm eps = 1e-5 (torch default).

Used with encode_state=False (observation-encoding variant). MLP inner dim = n_embd
(1x), as in the reference.
"""
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import orthogonal, constant


def _gelu(x):
    # match torch nn.GELU() default (exact erf), not the tanh approximation
    return jax.nn.gelu(x, approximate=False)


def _init_dense(features, activate=False, bias=True):
    """Mirror PKU `init_`: orthogonal(gain) with gain = relu-gain if activate else 0.01."""
    gain = np.sqrt(2.0) if activate else 0.01
    return nn.Dense(features, use_bias=bias,
                    kernel_init=orthogonal(gain), bias_init=constant(0.0))


def _ln():
    return nn.LayerNorm(epsilon=1e-5)  # torch LayerNorm default eps


class SelfAttention(nn.Module):
    """Faithful to PKU SelfAttention: __call__(key, value, query)."""
    n_embd: int
    n_head: int
    n_agent: int
    masked: bool = False

    @nn.compact
    def __call__(self, key, value, query):
        B, Lq, D = query.shape
        Lk = key.shape[1]
        h, hs = self.n_head, D // self.n_head
        # projections (PKU: key, query, value, proj -- all init_ default gain 0.01)
        k = _init_dense(D)(key).reshape(B, Lk, h, hs).transpose(0, 2, 1, 3)
        q = _init_dense(D)(query).reshape(B, Lq, h, hs).transpose(0, 2, 1, 3)
        v = _init_dense(D)(value).reshape(B, Lk, h, hs).transpose(0, 2, 1, 3)
        att = (q @ jnp.swapaxes(k, -1, -2)) * (1.0 / jnp.sqrt(hs))
        if self.masked:
            mask = jnp.tril(jnp.ones((Lq, Lk)))
            att = jnp.where(mask[None, None] == 0, -jnp.inf, att)
        att = jax.nn.softmax(att, axis=-1)
        y = (att @ v).transpose(0, 2, 1, 3).reshape(B, Lq, D)
        return _init_dense(D)(y)  # output projection


class _MLP(nn.Module):
    n_embd: int

    @nn.compact
    def __call__(self, x):
        x = _gelu(_init_dense(self.n_embd, activate=True)(x))
        return _init_dense(self.n_embd)(x)


class EncodeBlock(nn.Module):
    """POST-LN, unmasked self-attention (PKU EncodeBlock)."""
    n_embd: int
    n_head: int
    n_agent: int

    @nn.compact
    def __call__(self, x):
        attn = SelfAttention(self.n_embd, self.n_head, self.n_agent, masked=False)
        x = _ln()(x + attn(x, x, x))
        x = _ln()(x + _MLP(self.n_embd)(x))
        return x


class DecodeBlock(nn.Module):
    """POST-LN, masked self-attn + masked cross-attn (PKU DecodeBlock)."""
    n_embd: int
    n_head: int
    n_agent: int

    @nn.compact
    def __call__(self, x, rep_enc):
        attn1 = SelfAttention(self.n_embd, self.n_head, self.n_agent, masked=True)
        attn2 = SelfAttention(self.n_embd, self.n_head, self.n_agent, masked=True)
        x = _ln()(x + attn1(x, x, x))
        x = _ln()(rep_enc + attn2(key=x, value=x, query=rep_enc))
        x = _ln()(x + _MLP(self.n_embd)(x))
        return x


class MATActor(nn.Module):
    """Encoder + Decoder, encode_state=False (obs-encoding variant)."""
    action_dim: int
    n_agent: int
    n_embd: int = 64
    n_head: int = 1
    n_block: int = 1

    @nn.compact
    def __call__(self, obs, shifted_actions):
        """obs: (B, N, obs_dim). shifted_actions: (B, N, action_dim+1).
        Returns logits (B, N, action_dim), values (B, N)."""
        # ---- Encoder ----
        obs_emb = _gelu(_init_dense(self.n_embd, activate=True)(_ln()(obs)))
        rep = _ln()(obs_emb)
        for _ in range(self.n_block):
            rep = EncodeBlock(self.n_embd, self.n_head, self.n_agent)(rep)
        vh = _gelu(_init_dense(self.n_embd, activate=True)(rep))
        value = jnp.squeeze(_init_dense(1)(_ln()(vh)), axis=-1)
        # ---- Decoder ----
        act_emb = _gelu(_init_dense(self.n_embd, activate=True, bias=False)(shifted_actions))
        x = _ln()(act_emb)
        for _ in range(self.n_block):
            x = DecodeBlock(self.n_embd, self.n_head, self.n_agent)(x, rep)
        xh = _gelu(_init_dense(self.n_embd, activate=True)(x))
        logits = _init_dense(self.action_dim)(_ln()(xh))
        return logits, value


class MATCNNActor(nn.Module):
    """MAT with a CNN per-agent encoder (replaces the flat Linear projection).

    Identical transformer (EncodeBlock/DecodeBlock/value+action heads) to
    MATActor; only the obs->embedding step changes:
        flat Linear(1452->n_embd)   ==>   Conv(32,5x5)->Conv(32,3x3)->Conv(32,3x3)
                                          ->flatten->Dense(n_embd)
    Same __call__(obs_flat, shifted_actions) signature so mat_get_actions /
    mat_eval_actions work unchanged. obs_flat (B,N,1452) is reshaped internally
    to (B,N,11,11,12).
    """
    action_dim: int
    n_agent: int
    n_embd: int = 64
    n_head: int = 1
    n_block: int = 1
    obs_h: int = 11
    obs_w: int = 11
    obs_c: int = 12

    @nn.compact
    def __call__(self, obs_flat, shifted_actions):
        B, N, D = obs_flat.shape
        grid = jnp.reshape(obs_flat, (B, N, self.obs_h, self.obs_w, self.obs_c))
        grid = jnp.reshape(grid, (B * N, self.obs_h, self.obs_w, self.obs_c))
        x = grid
        x = nn.Conv(32, (5, 5), kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x); x = _gelu(x)
        x = nn.Conv(32, (3, 3), kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x); x = _gelu(x)
        x = nn.Conv(32, (3, 3), kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x); x = _gelu(x)
        x = x.reshape((B * N, -1))
        obs_emb = jnp.reshape(_gelu(_init_dense(self.n_embd, activate=True)(x)), (B, N, self.n_embd))

        rep = _ln()(obs_emb)
        for _ in range(self.n_block):
            rep = EncodeBlock(self.n_embd, self.n_head, self.n_agent)(rep)
        vh = _gelu(_init_dense(self.n_embd, activate=True)(_ln()(rep)))
        value = jnp.squeeze(_init_dense(1)(_ln()(vh)), axis=-1)

        act_emb = _gelu(_init_dense(self.n_embd, activate=True, bias=False)(shifted_actions))
        x = _ln()(act_emb)
        for _ in range(self.n_block):
            x = DecodeBlock(self.n_embd, self.n_head, self.n_agent)(x, rep)
        xh = _gelu(_init_dense(self.n_embd, activate=True)(x))
        logits = _init_dense(self.action_dim)(_ln()(xh))
        return logits, value


# ---------------------------------------------------------------------------
# Autoregressive helpers shared by evaluation and rollout.
# ---------------------------------------------------------------------------

def _empty_shifted(B, N, action_dim):
    s = jnp.zeros((B, N, action_dim + 1))
    return s.at[:, 0, 0].set(1.0)


def mat_logits_autoreg(apply_fn, params, obs, action_dim):
    """Per-agent logits from a deterministic autoregressive decode."""
    B, N, _ = obs.shape
    shifted = _empty_shifted(B, N, action_dim)
    logits_all = []
    for i in range(N):
        logits, _ = apply_fn(params, obs, shifted)
        li = logits[:, i, :]
        logits_all.append(li)
        ai = jnp.argmax(li, axis=-1)
        if i + 1 < N:
            shifted = shifted.at[:, i + 1, 1:].set(jax.nn.one_hot(ai, action_dim))
    return jnp.stack(logits_all, axis=1)


def mat_get_actions(
    apply_fn, params, obs, key, action_dim, deterministic=False
):
    """Sample a joint action autoregressively.

    Returns ``(actions, logp, values)``.
    """
    B, N, _ = obs.shape
    shifted = _empty_shifted(B, N, action_dim)
    actions, logps, values = [], [], None
    for i in range(N):
        logits, values = apply_fn(params, obs, shifted)
        li = logits[:, i, :]
        if deterministic:
            ai = jnp.argmax(li, axis=-1)
        else:
            key, sk = jax.random.split(key)
            ai = jax.random.categorical(sk, li)
        logps.append(jax.nn.log_softmax(li)[jnp.arange(B), ai])
        actions.append(ai)
        if i + 1 < N:
            shifted = shifted.at[:, i + 1, 1:].set(jax.nn.one_hot(ai, action_dim))
    return jnp.stack(actions, axis=1), jnp.stack(logps, axis=1), values


def mat_eval_actions(apply_fn, params, obs, actions, action_dim):
    """Teacher-forced parallel evaluation (for the PPO update). Returns
    logp (B, N), entropy (B, N), values (B, N)."""
    B, N, _ = obs.shape
    oh = jax.nn.one_hot(actions, action_dim)
    shifted = _empty_shifted(B, N, action_dim)
    shifted = shifted.at[:, 1:, 1:].set(oh[:, :-1, :])
    logits, values = apply_fn(params, obs, shifted)
    logp_all = jax.nn.log_softmax(logits)
    logp = jnp.take_along_axis(logp_all, actions[..., None], axis=-1)[..., 0]
    entropy = -(jax.nn.softmax(logits) * logp_all).sum(-1)
    return logp, entropy, values
