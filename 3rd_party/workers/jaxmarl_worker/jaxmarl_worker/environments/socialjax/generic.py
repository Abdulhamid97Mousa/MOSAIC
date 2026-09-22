"""Generic wrapper for any SocialJax environment.

Adapts any socialjax.make(env_id) environment to the jaxmarl_worker
InteractiveRuntime interface (dict observations, step_env name, etc.).

Obs and action dimensions are discovered dynamically via a dummy reset,
so this wrapper works for all 8 registered SocialJax environments without
per-env hardcoding.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

_SOCIALJAX_ROOT = str(
    Path(__file__).resolve().parents[5] / "environments" / "SocialJax"
)
if _SOCIALJAX_ROOT not in sys.path:
    sys.path.insert(0, _SOCIALJAX_ROOT)


class SocialJaxGenericWrapper:
    """jaxmarl_worker-compatible wrapper for any SocialJax environment.

    Discovers obs_dim and action_dim at init time via a dummy reset so
    it works for all registered env_ids without per-env hardcoding.
    """

    def __init__(self, env_id: str, **kwargs):
        import socialjax
        self._socialjax = socialjax.make(env_id, **kwargs)
        self._env_id = env_id

        self.num_agents: int = self._socialjax.num_agents

        # Discover action_dim from the env's action_space
        try:
            self.action_dim: int = self._socialjax.action_space(0).n
        except Exception:
            # Fallback: count via action_space without agent arg
            try:
                self.action_dim = self._socialjax.action_space().n
            except Exception:
                self.action_dim = 8  # conservative default

        # Discover obs_dim via a dummy reset (one JIT compile, ~50ms)
        dummy_key = jax.random.PRNGKey(0)
        obs_arr, _ = self._socialjax.reset(dummy_key)
        # obs_arr shape: (num_agents, ...) -- flatten all dims after axis 0
        per_agent = obs_arr[0]
        self._obs_dim: int = int(jnp.array(per_agent).size)

        # Canonical agent names expected by InteractiveRuntime
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _obs_to_dict(self, obs_arr: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """(N, ...) array -> {agent_i: flat_obs} dict."""
        return {f"agent_{i}": obs_arr[i].flatten() for i in range(self.num_agents)}

    def _dict_to_actions(self, actions_dict: Dict[str, Any]) -> jnp.ndarray:
        """Dict of per-agent actions -> (N,) int32 array."""
        return jnp.stack(
            [jnp.int32(actions_dict[f"agent_{i}"]) for i in range(self.num_agents)]
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, key) -> Tuple[Dict[str, jnp.ndarray], Any]:
        obs_arr, state = self._socialjax.reset(key)
        return self._obs_to_dict(obs_arr), state

    def step_env(
        self,
        key,
        state: Any,
        actions_dict: Dict[str, Any],
    ) -> Tuple[Dict, Any, Dict, Dict, Dict]:
        """Step the environment with dict actions.

        Returns (obs_dict, new_state, rewards_dict, dones_dict, info).
        dones_dict includes dones["__all__"] as a scalar bool.
        """
        actions_arr = self._dict_to_actions(actions_dict)
        obs_arr, new_state, rewards_arr, done_dict, info = self._socialjax.step_env(
            key, state, actions_arr
        )
        obs_out = self._obs_to_dict(obs_arr)

        # Normalise rewards to dict
        if hasattr(rewards_arr, "__len__"):
            rew_out = {f"agent_{i}": float(rewards_arr[i]) for i in range(self.num_agents)}
        else:
            rew_out = {f"agent_{i}": float(rewards_arr) for i in range(self.num_agents)}

        # Normalise dones dict -- ensure __all__ key exists
        if isinstance(done_dict, dict):
            done_out = done_dict
        else:
            done_scalar = bool(done_dict)
            done_out = {f"agent_{i}": done_scalar for i in range(self.num_agents)}
            done_out["__all__"] = done_scalar

        return obs_out, new_state, rew_out, done_out, info

    def step(self, key, state: Any, actions_dict: Dict[str, Any]):
        """Alias for step_env -- no auto-reset (use for eval loops)."""
        return self.step_env(key, state, actions_dict)
