"""MeltingPot Prisoner's Dilemma environment wrapper for XuanCe multi-agent training.

Wraps MeltingPot's PD substrate via shimmy into XuanCe's RawMultiAgentEnv interface.
Flattens pixel RGB + inventory observations into a single vector for MLP training,
or keeps them structured for CNN training.

Usage:
    config = SimpleNamespace(env_id="prisoners_dilemma_in_the_matrix__repeated")
    env = MeltingPot_PD_Env(config)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces

_logger = logging.getLogger(__name__)

try:
    from xuance.environment import RawMultiAgentEnv
except ImportError:
    RawMultiAgentEnv = object

try:
    from shimmy import MeltingPotCompatibilityV0
    _HAS_SHIMMY = True
except ImportError:
    _HAS_SHIMMY = False


class MeltingPot_PD_Env(RawMultiAgentEnv):
    """XuanCe-compatible wrapper for MeltingPot Prisoner's Dilemma.

    Observation: Flattened RGB (40*40*3=4800) + INVENTORY(2) + READY_TO_SHOOT(1)
                 + INTERACTION_INVENTORIES(4) = 4807 total features.

    For CNN mode (use_cnn=True): Returns dict with 'image' and 'extra' keys.
    For MLP mode (use_cnn=False): Returns flattened vector of 4807 features.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()

        if not _HAS_SHIMMY:
            raise ImportError("shimmy not installed. Install with: pip install shimmy[meltingpot]")

        raw_name = getattr(config, 'env_id', 'prisoners_dilemma_in_the_matrix__repeated')
        # Short name mapping
        _NAME_MAP = {
            'prisoners_dilemma': 'prisoners_dilemma_in_the_matrix__repeated',
            'pd_repeated': 'prisoners_dilemma_in_the_matrix__repeated',
            'stag_hunt': 'stag_hunt_in_the_matrix__repeated',
            'chicken': 'chicken_in_the_matrix__repeated',
            'rps': 'running_with_scissors_in_the_matrix__repeated',
        }
        substrate_name = _NAME_MAP.get(raw_name, raw_name)
        env_seed = getattr(config, 'env_seed', None)
        self.use_cnn = getattr(config, 'use_cnn', False)

        self.env = MeltingPotCompatibilityV0(
            substrate_name=substrate_name,
            render_mode=None,
        )

        self._shimmy_agents = list(self.env.possible_agents)
        self.n_agents = len(self._shimmy_agents)
        # Use shimmy's agent names so XuanCe wrapper can key into action_space
        self.agents = self._shimmy_agents

        # Observation space: flatten everything for MLP
        # RGB: (40,40,3) = 4800, INVENTORY: 2, READY_TO_SHOOT: 1, INTERACTION_INVENTORIES: 4
        self._rgb_size = 40 * 40 * 3
        self._extra_size = 2 + 1 + 4  # inventory + ready_to_shoot + interaction_inventories
        self._obs_size = self._rgb_size + self._extra_size  # 4807

        # XuanCe interface
        self.dim_obs = [self._obs_size] * self.n_agents
        self.dim_act = [8] * self.n_agents  # 8 discrete actions
        self.obs_space = [
            spaces.Box(low=0, high=1, shape=(self._obs_size,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]
        self.act_space = [
            spaces.Discrete(8)
            for _ in range(self.n_agents)
        ]
        self.n_adversaries = 1  # 1v1 game, 1 adversary per agent

        # Groups: 2 independent players
        self.agent_ids = list(range(self.n_agents))
        self.agent_keys = list(self._shimmy_agents)  # ['player_0', 'player_1']
        self.num_agents = self.n_agents  # XuanCe uses num_agents
        self.max_episode_steps = 200  # Keep short for RNN buffer memory
        self.max_cycles = 200

        # Dict-keyed spaces for XuanCe's wrapper (keyed by shimmy agent names)
        self.observation_space = {
            agent: spaces.Box(low=0, high=1, shape=(self._obs_size,), dtype=np.float32)
            for agent in self.agents
        }
        self.action_space = {
            agent: spaces.Discrete(8)
            for agent in self.agents
        }

        # State space for centralized critic (XuanCe requires this)
        self._state_size = self._obs_size * self.n_agents
        self.state_space = spaces.Box(
            low=0, high=1, shape=(self._state_size,), dtype=np.float32
        )
        self.dim_state = self._state_size

        _logger.info(
            f"MeltingPot_PD_Env initialized: substrate={substrate_name}, "
            f"agents={self.n_agents}, obs_dim={self._obs_size}, act_dim=8"
        )

    def _flatten_obs(self, obs: dict, agent: str) -> np.ndarray:
        """Flatten MeltingPot observation dict into a single vector."""
        rgb = obs[agent]['RGB'].astype(np.float32).flatten() / 255.0
        inv = obs[agent]['INVENTORY'].astype(np.float32).flatten()
        rts = np.array([obs[agent]['READY_TO_SHOOT']], dtype=np.float32).flatten()
        ii = obs[agent]['INTERACTION_INVENTORIES'].astype(np.float32).flatten()
        return np.concatenate([rgb, inv, rts, ii])

    def reset(self, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        obs, info = self.env.reset()

        flat_obs = {}
        for i, agent in enumerate(self.agents):
            flat_obs[self.agent_keys[i]] = self._flatten_obs(obs, agent)

        self._last_obs = flat_obs
        return flat_obs, {}

    def step(self, actions: Dict[str, int]) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        # Convert XuanCe actions to shimmy format
        shimmy_actions = {}
        for i, agent in enumerate(self.agents):
            key = self.agent_keys[i]
            if key in actions:
                shimmy_actions[agent] = int(actions[key])
            elif i in actions:
                shimmy_actions[agent] = int(actions[i])
            else:
                shimmy_actions[agent] = 0  # NOOP

        obs, rewards, terms, truncs, infos = self.env.step(shimmy_actions)

        flat_obs = {}
        flat_rewards = {}
        flat_terms = {}
        flat_truncs = {}
        flat_infos = {}

        for i, agent in enumerate(self.agents):
            key = self.agent_keys[i]
            flat_obs[key] = self._flatten_obs(obs, agent)
            flat_rewards[key] = float(rewards[agent])
            flat_terms[key] = bool(terms[agent])
            flat_truncs[key] = bool(truncs[agent])
            flat_infos[key] = infos.get(agent, {})

        self._last_obs = flat_obs
        return flat_obs, flat_rewards, flat_terms, flat_truncs, flat_infos

    def get_agent_mask(self) -> np.ndarray:
        """All agents always active in PD."""
        return np.ones(self.n_agents, dtype=bool)

    def state(self) -> np.ndarray:
        """Global state for centralized critic (concat all obs)."""
        if hasattr(self, '_last_obs') and self._last_obs is not None:
            parts = [self._last_obs[agent] for agent in self.agent_keys]
            return np.concatenate(parts)
        return np.zeros(self._state_size, dtype=np.float32)

    def close(self):
        self.env.close()

    def render(self, mode=None):
        return self.env.render()
