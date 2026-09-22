"""Observation preprocessing wrappers for mosaic multigrid JAX environments.

These wrappers sit between the base JAX environment and the training script.
They are applied once after _make_env() and are transparent to all downstream
code that reads env._obs_dim, env.num_agents, env.agents, etc.
"""

from typing import Dict, Tuple

import chex
import jax
import jax.numpy as jnp
from jaxmarl.environments.multi_agent_env import MultiAgentEnv, State
from jaxmarl.environments.spaces import Discrete
from jaxmarl.wrappers.baselines import JaxMARLWrapper


class AgentIDWrapper(JaxMARLWrapper):
    """Appends a one-hot agent ID to each agent's flat observation.

    Motivation
    ----------
    All mosaic_multigrid training scripts use parameter sharing: every agent
    runs through the same network weights.  Without an identity signal, two
    agents at positions that produce identical egocentric views receive identical
    action logits.  The one-hot ID breaks this symmetry at zero policy cost.

    After wrapping
    --------------
      env._obs_dim              = base_obs_dim + n_agents
      env.observation_spaces[a] = Discrete(base_obs_dim + n_agents)
      env.reset() / env.step()  return obs[a] of the new dim

    Training scripts need no changes beyond inserting:
        env = AgentIDWrapper(env)
    immediately after _make_env(), because all downstream code reads env._obs_dim.

    Dec-POMDP note
    --------------
    The one-hot ID is part of the observation o_i, not the global state s.
    It is known to agent i at execution time (it is a fixed property of the
    agent, not a private feature of s).  This is formally correct.
    """

    def __init__(self, env: MultiAgentEnv):
        super().__init__(env)
        n = env.num_agents
        self._base_obs_dim = env._obs_dim
        self._obs_dim = env._obs_dim + n

        # Static one-hot vectors, one per agent. Stored as a dict so they can
        # be used inside jax.vmap without recomputation.
        self._one_hots: Dict[str, chex.Array] = {
            a: jnp.eye(n, dtype=jnp.float32)[i]
            for i, a in enumerate(env.agents)
        }

        # Shadow the base env's observation_spaces so space-aware code sees
        # the augmented dimension (JaxMARLWrapper.__getattr__ forwards attrs
        # not on self to self._env, so we must set this explicitly).
        self.observation_spaces = {
            a: Discrete(self._obs_dim) for a in env.agents
        }

    def _add_ids(self, obs: Dict[str, chex.Array]) -> Dict[str, chex.Array]:
        return {
            a: jnp.concatenate([obs[a].astype(jnp.float32), self._one_hots[a]])
            for a in self.agents
        }

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict, State]:
        obs, state = self._env.reset(key)
        return self._add_ids(obs), state

    def step(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict,
    ) -> Tuple[Dict, State, Dict, Dict, Dict]:
        obs, state, reward, done, info = self._env.step(key, state, actions)
        return self._add_ids(obs), state, reward, done, info
