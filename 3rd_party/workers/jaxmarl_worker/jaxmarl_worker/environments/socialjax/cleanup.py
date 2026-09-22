"""SocialJax Cleanup wrapper for jaxmarl_worker training.

Adapts SocialJax clean_up to jaxmarl_worker's dict-obs API so it can be
used with mat_scan.make_train.

Observation per agent: float32 (2299,) -- flat 11*11*19 CNN grid
  11x11 egocentric window, 19 channels:
    ch 0-8:  item one-hot (9 item types, len(Items)-1 = 9)
    ch 9-18: agent/direction/inventory encoding (+10 channels)

Actions (9): 0=turn_left 1=turn_right 2=left 3=right 4=up 5=down
             6=stay 7=zap_forward 8=zap_clean
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

OBS_H, OBS_W, OBS_C = 11, 11, 19
OBS_DIM     = OBS_H * OBS_W * OBS_C  # 2299
NUM_AGENTS  = 7
NUM_ACTIONS = 9


class SocialJaxCleanup:
    """jaxmarl_worker-compatible wrapper around SocialJax clean_up.

    Auto-resets on episode end (Anakin-compatible).
    """

    def __init__(
        self,
        num_agents:      int  = NUM_AGENTS,
        num_inner_steps: int  = 1000,
        shared_rewards:  bool = True,
    ):
        self._socialjax = socialjax.make(
            "clean_up",
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
        """(N, 11, 11, 19) -> {f'agent_{i}': (2299,) float32}"""
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
