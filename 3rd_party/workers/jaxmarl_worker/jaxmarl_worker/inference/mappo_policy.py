"""Frozen MAPPO actor for cooperative mosaic_multigrid inference.

Co-located with jaxmarl_worker so the Actor definition stays in sync with
the training algorithms (algorithms/mosaic_multigrid/BB/mappo_scan.py etc.).

Checkpoint key format (verified 2026-09-10, MAPPO_GLOBAL_INDAGOBS cooperative):
  arr_* keys, alphabetical Flax traversal order:
    arr_0  Dense_0.bias   (hidden_dim,)
    arr_1  Dense_0.kernel (obs_dim, hidden_dim)
    arr_2  Dense_1.bias   (hidden_dim,)
    arr_3  Dense_1.kernel (hidden_dim, hidden_dim)
    arr_4  Dense_2.bias   (action_dim,)
    arr_5  Dense_2.kernel (hidden_dim, action_dim)
  Adversarial checkpoints use actor_green_* / actor_blue_* keys.
"""
from __future__ import annotations

from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal


class _Actor(nn.Module):
    action_dim: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim,
                     kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim,
                     kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        return nn.Dense(self.action_dim,
                        kernel_init=orthogonal(0.01),
                        bias_init=constant(0.0))(x)


class MAPPOPolicy:
    """Frozen MAPPO actor for inference. Stateless feedforward MLP.

    Works for both cooperative (arr_* keys) and adversarial (actor_green/blue_*)
    checkpoint formats.
    """

    def __init__(self, actor: _Actor, params) -> None:
        self._apply = jax.jit(
            lambda obs: jnp.argmax(jnp.asarray(actor.apply(params, obs[None])), axis=-1)[0]
        )

        def _with_info(obs):
            logits  = jnp.asarray(actor.apply(params, obs[None]))[0]
            probs   = jax.nn.softmax(logits)
            entropy = -jnp.sum(probs * jnp.log(probs + 1e-8))
            return jnp.argmax(probs), probs, entropy

        self._apply_with_info = jax.jit(_with_info)

    def act(self, obs: jnp.ndarray) -> int:
        """Return greedy action int. Feedforward, no hidden state."""
        return int(self._apply(obs))

    def act_with_info(self, obs: jnp.ndarray) -> tuple[int, np.ndarray, float]:
        """Return (action_int, probs_np, entropy_float). Feedforward only.

        probs_np: shape (action_dim,), float32, sums to 1.0
        entropy_float: scalar entropy in nats
        """
        action, probs, entropy = self._apply_with_info(obs)
        return int(action), np.array(probs, dtype=np.float32), float(entropy)

    @classmethod
    def load(
        cls,
        ckpt_path: Path,
        obs_dim: int,
        role: str = "shared",
        hidden_dim: int = 256,
        action_dim: int = 8,
    ) -> "MAPPOPolicy":
        """Load checkpoint and return a ready-to-use policy.

        role: 'shared' -> arr_* keys (cooperative MAPPO_GLOBAL_INDAGOBS)
              'green'  -> actor_green_* keys (adversarial, green team)
              'blue'   -> actor_blue_*  keys (adversarial, blue team)
        """
        actor = _Actor(action_dim=action_dim, hidden_dim=hidden_dim)
        dummy = actor.init(jax.random.PRNGKey(0), jnp.zeros((1, obs_dim)))
        _, treedef = jax.tree_util.tree_flatten(dummy)
        ck = np.load(str(ckpt_path))
        n  = len(jax.tree_util.tree_leaves(dummy))

        if f"actor_{role}_0" in ck:
            leaves = [jnp.array(ck[f"actor_{role}_{i}"]) for i in range(n)]
        elif "actor_green_0" in ck:
            import warnings
            warnings.warn(
                f"role='{role}' not found; falling back to actor_green_*.",
                stacklevel=2,
            )
            leaves = [jnp.array(ck[f"actor_green_{i}"]) for i in range(n)]
        else:
            leaves = [jnp.array(ck[f"arr_{i}"]) for i in range(n)]

        return cls(actor, jax.tree_util.tree_unflatten(treedef, leaves))
