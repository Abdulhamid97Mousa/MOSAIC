"""SocialJax CoopMining wrapper for jaxmarl_worker Anakin training.

Adapts SocialJax coop_mining to jaxmarl_worker's dict-obs API
so it can be used directly with ippo_scan, mappo_indagobs_scan, and mat_scan.

Observation per agent: float32 (1452,) -- flat 11*11*12 symbolic grid
  11x11 egocentric window, 12 channels:
    ch 0-5:  item one-hot (empty/wall/ore_wait/spawn/iron/gold)
    ch 6:    "this is me" flag
    ch 7:    "other agent" flag
    ch 8-11: agent orientation one-hot (N/E/S/W)

Actions (8): 0=noop 1=fwd 2=bck 3=strafe-L 4=strafe-R 5=turn-L 6=turn-R 7=mine

Auto-reset on episode end is handled internally via jnp.where over the state pytree,
so this wrapper is fully compatible with jax.vmap(env.step) inside lax.scan loops.
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

from socialjax.environments.coop_mining.coop_mining import CoopMining  # noqa: E402

OBS_H, OBS_W, OBS_C = 11, 11, 12
OBS_DIM    = OBS_H * OBS_W * OBS_C  # 1452
NUM_AGENTS = 6
NUM_ACTIONS = 8


class SocialJaxCoopMining:
    """jaxmarl_worker-compatible wrapper around SocialJax CoopMining.

    Hyperparameter defaults match the official SocialJax MAPPO paper config
    (Guo et al.).
    """

    def __init__(
        self,
        num_agents:          int   = NUM_AGENTS,
        num_inner_steps:     int   = 1000,
        shared_rewards:      bool  = True,
        regrowth_prob_iron:  float = 0.0004,
        regrowth_prob_gold:  float = 0.00016,
        reward_iron:         float = 1.0,
        reward_gold:         float = 8.0,
        gold_mining_window:  int   = 3,
        min_gold_miners:     int   = 2,
        max_miners:          int   = 4,
        num_outer_steps:     int   = 1,
    ):
        self._socialjax = CoopMining(
            num_agents         = num_agents,
            num_inner_steps    = num_inner_steps,
            shared_rewards     = shared_rewards,
            regrowth_prob_iron = regrowth_prob_iron,
            regrowth_prob_gold = regrowth_prob_gold,
            reward_iron        = reward_iron,
            reward_gold        = reward_gold,
            gold_mining_window = gold_mining_window,
            min_gold_miners    = min_gold_miners,
            max_miners         = max_miners,
            num_outer_steps    = num_outer_steps,
        )
        self.num_agents = num_agents
        self._obs_dim   = OBS_DIM
        self.agents     = [f"agent_{i}" for i in range(num_agents)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _obs_to_dict(self, obs_arr: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """(N, 11, 11, 12) -> {f'agent_{i}': (1452,) float32}"""
        return {f"agent_{i}": obs_arr[i].flatten() for i in range(self.num_agents)}

    # ------------------------------------------------------------------
    # Public API (mirrors JaxMARL MultiAgentEnv interface)
    # ------------------------------------------------------------------

    def reset(self, key) -> Tuple[Dict[str, jnp.ndarray], object]:
        obs_arr, state = self._socialjax.reset(key)
        return self._obs_to_dict(obs_arr), state

    def step(
        self,
        key,
        state,
        actions_dict: Dict[str, jnp.ndarray],
    ) -> Tuple[Dict, object, Dict, Dict, Dict]:
        """Anakin-compatible step with built-in auto-reset.

        Args:
            key:          JAX PRNGKey
            state:        SocialJax State pytree from a prior reset/step
            actions_dict: {f'agent_{i}': int32 scalar} for each agent

        Returns:
            (obs, new_state, rewards, dones, info)
            where obs/rewards/dones are {f'agent_{i}': ...} dicts.
            dones["__all__"] is a scalar bool.
        """
        actions_arr = jnp.stack(
            [actions_dict[f"agent_{i}"] for i in range(self.num_agents)]
        )

        key, step_key, reset_key = jax.random.split(key, 3)

        obs_arr, new_state, rewards_arr, done_dict, info = self._socialjax.step_env(
            step_key, state, actions_arr
        )
        done = done_dict["__all__"]

        # Auto-reset: when done, swap in a fresh episode's obs and state
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

    def step_env(
        self,
        key,
        state,
        actions_dict: Dict[str, jnp.ndarray],
    ) -> Tuple[Dict, object, Dict, Dict, Dict]:
        """Alias for step() -- satisfies the jaxmarl_worker InteractiveRuntime interface."""
        return self.step(key, state, actions_dict)
