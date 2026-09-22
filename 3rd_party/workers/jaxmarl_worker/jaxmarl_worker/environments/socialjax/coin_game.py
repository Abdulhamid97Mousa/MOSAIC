"""SocialJax CoinGame wrapper for jaxmarl_worker MAT training.

Adapts SocialJax coin_game to jaxmarl_worker's dict-obs API
so it can be used with mat_scan.make_train.

Observation per agent: float32 (1694,) -- flat 11*11*14 CNN grid
Actions: discovered from env.action_space() at init
"""
import sys
from pathlib import Path
from typing import Dict, Tuple

import jax
import jax.numpy as jnp

_SOCIALJAX_ROOT = str(
    Path(__file__).resolve().parents[5] / "environments" / "SocialJax"
)
if _SOCIALJAX_ROOT not in sys.path:
    sys.path.insert(0, _SOCIALJAX_ROOT)

import socialjax

OBS_H, OBS_W, OBS_C = 11, 11, 14
OBS_DIM    = OBS_H * OBS_W * OBS_C   # 1694
NUM_AGENTS = 2


class SocialJaxCoinGame:
    """jaxmarl_worker-compatible wrapper around SocialJax coin_game.

    Provides flat (OBS_DIM,) observations and dict interface for
    mat_scan.make_train. Auto-resets on episode end (Anakin-compatible).
    """

    def __init__(
        self,
        num_agents:      int  = NUM_AGENTS,
        num_inner_steps: int  = 1000,
        shared_rewards:  bool = True,
    ):
        self._socialjax = socialjax.make(
            "coin_game",
            num_agents      = num_agents,
            num_inner_steps = num_inner_steps,
            shared_rewards  = shared_rewards,
            cnn             = True,
            jit             = True,
        )
        self.num_agents = num_agents
        self._obs_dim   = OBS_DIM
        self.agents     = [f"agent_{i}" for i in range(num_agents)]

    def _obs_to_dict(self, obs_arr: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """(N, 11, 11, 14) -> {f'agent_{i}': (1694,) float32}"""
        return {f"agent_{i}": obs_arr[i].flatten() for i in range(self.num_agents)}

    def reset(self, key) -> Tuple[Dict[str, jnp.ndarray], object]:
        obs_arr, state = self._socialjax.reset(key)
        return self._obs_to_dict(obs_arr), state

    def step(
        self,
        key,
        state,
        actions_dict: Dict[str, jnp.ndarray],
    ) -> Tuple[Dict, object, Dict, Dict, Dict]:
        """Anakin-compatible step with built-in auto-reset."""
        actions_arr = jnp.stack(
            [actions_dict[f"agent_{i}"] for i in range(self.num_agents)]
        )

        key, step_key, reset_key = jax.random.split(key, 3)

        obs_arr, new_state, rewards_arr, done_dict, info = self._socialjax.step_env(
            step_key, state, actions_arr
        )
        done = done_dict["__all__"]

        # Auto-reset: swap in a fresh episode when done
        obs_reset, state_reset = self._socialjax.reset(reset_key)
        final_obs   = jnp.where(done, obs_reset, obs_arr)
        final_state = jax.tree_util.tree_map(
            lambda r, s: jnp.where(done, r, s), state_reset, new_state
        )

        obs_out  = self._obs_to_dict(final_obs)
        rew_out  = {f"agent_{i}": rewards_arr[i] for i in range(self.num_agents)}
        done_out = {f"agent_{i}": done for i in range(self.num_agents)}
        done_out["__all__"] = done

        return obs_out, final_state, rew_out, done_out, info

    def step_env(self, key, state, actions_dict):
        return self.step(key, state, actions_dict)
