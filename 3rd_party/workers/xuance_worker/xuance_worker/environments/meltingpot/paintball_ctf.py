"""MeltingPot Paintball Capture-the-Flag wrapper for XuanCe multi-agent training.

This wrapper exposes Melting Pot's ``paintball__capture_the_flag`` substrate
through the ``RawMultiAgentEnv`` interface so XuanCe's MARL runner can train
MAPPO with a CNN representation on the raw 88x88x3 egocentric RGB view.

Key differences vs ``pd.py`` (the MeltingPot PD wrapper in this same family directory):

* PD flattens RGB + inventory into a single vector and uses ``Basic_MLP``.
  Paintball CTF keeps the image as a 3D ``(88, 88, 3)`` ``uint8`` tensor per
  agent so ``Basic_CNN`` (``xuance.torch.representations.cnn``) can convolve
  over it.
* PD relies on shimmy's ``MeltingPotCompatibilityV0(substrate_name=...)``
  loader, which calls ``meltingpot.substrate.build`` with the default config.
  Paintball CTF builds the substrate explicitly so it can set
  ``config.shaping_kwargs`` before the Lua level is assembled. This is the
  only way to turn on the dense reward shaping the Melting Pot 2.0 tech
  report recommends for bootstrapping self-play on this substrate.

Observation per agent:
    - RGB:             shape (88, 88, 3), dtype uint8, range [0, 255]
    - READY_TO_SHOOT:  currently ignored (first-pass CNN config). Can be
                       folded into a 4th channel later.

Action per agent:
    - Discrete(9): Paintball CTF's flattened ACTION_SET. The wrapper reads
      the count from the substrate's action_spec, so this is future-proof
      if Melting Pot adds or removes actions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from gymnasium import spaces

_logger = logging.getLogger(__name__)

try:
    from xuance.environment import RawMultiAgentEnv
except ImportError:
    RawMultiAgentEnv = object  # type: ignore[assignment,misc]


_DEFAULT_SHAPING_KWARGS: Mapping[str, float] = {
    # Values taken verbatim from Melting Pot 2.0 tech report (Agapiou et al.
    # 2023), which states: "We used these pseudorewards to train the
    # background population bots in the test scenarios" for Paintball CTF.
    "defaultTeamReward": 25.0,
    "rewardForZapping": 1.0,
    "extraRewardForZappingFlagCarrier": 1.0,
    "rewardForReturningFlag": 3.0,
    "rewardForPickingUpOpposingFlag": 5.0,
}


class MeltingPot_PaintballCaptureTheFlag_Env(RawMultiAgentEnv):
    """XuanCe-compatible wrapper for Melting Pot Paintball: Capture the Flag.

    Expected config attributes:
        env_id:         substrate name, default 'paintball__capture_the_flag'.
        env_seed:       optional int seed for the substrate.
        shaping_kwargs: optional dict of Paintball CTF reward-shaping overrides.
                        If None or omitted, the default dense shaping above
                        is applied. Set to {} to get the sparse zero-sum
                        reward (flag capture only).
        render_mode:    optional render mode forwarded to shimmy.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()

        substrate_name = getattr(config, "env_id", "paintball__capture_the_flag")

        # The canonical substrate name is paintball__capture_the_flag
        # (double underscore, matching Melting Pot's naming). Use that
        # exact string in config.env_id. No short aliases on purpose:
        # reviewer memory prefers one canonical name per substrate, not a
        # sprawl of nicknames that have to be kept in sync.

        shaping_kwargs = getattr(config, "shaping_kwargs", None)
        if shaping_kwargs is None:
            shaping_kwargs = dict(_DEFAULT_SHAPING_KWARGS)
        elif shaping_kwargs == {}:
            # Explicit opt-out: train on the sparse zero-sum reward.
            shaping_kwargs = None

        render_mode = getattr(config, "render_mode", None)
        self.render_mode = render_mode

        # Build the substrate ourselves (bypassing shimmy's default-config
        # shortcut) so we can inject shaping_kwargs before the Lua level is
        # constructed. See meltingpot/configs/substrates/paintball__capture
        # _the_flag.py:765-770 and :799-820 for where shaping_kwargs flows.
        try:
            import meltingpot
            from meltingpot import substrate as mp_substrate
        except ImportError as e:
            raise ImportError(
                "meltingpot is required for Paintball CTF training. "
                "Install with: pip install dm-meltingpot"
            ) from e

        try:
            from shimmy import MeltingPotCompatibilityV0
        except ImportError as e:
            raise ImportError(
                "shimmy[meltingpot] is required. "
                "Install with: pip install 'shimmy[meltingpot]'"
            ) from e

        mp_config = mp_substrate.get_config(substrate_name)
        # get_config() returns a locked ConfigDict; unlock to mutate shaping.
        with mp_config.unlocked():
            mp_config.shaping_kwargs = shaping_kwargs
        roles = mp_config.default_player_roles  # ("default",) * 8 for CTF
        dm_env_substrate = mp_substrate.build_from_config(
            mp_config, roles=roles
        )

        self.env = MeltingPotCompatibilityV0(
            env=dm_env_substrate,
            render_mode=render_mode,
        )

        # shimmy exposes a PettingZoo ParallelEnv-like API:
        #   env.possible_agents, env.reset(), env.step(actions_dict)
        self.agents: List[str] = list(self.env.possible_agents)
        self.num_agents: int = len(self.agents)

        assert self.num_agents == 8, (
            f"Paintball CTF substrate returned {self.num_agents} agents, "
            f"expected 8 (4 red vs 4 blue)."
        )

        # Team assignment for Paintball CTF is deterministic in the substrate:
        #   even player indices -> red team, odd indices -> blue team
        #   (see paintball__capture_the_flag.py:712-726, _even_vs_odd_team_
        #   assignment, which is the default assignment function).
        self.agent_groups: List[List[str]] = [
            [a for i, a in enumerate(self.agents) if i % 2 == 0],  # red
            [a for i, a in enumerate(self.agents) if i % 2 == 1],  # blue
        ]

        # The substrate renders each agent's 11x11 egocentric window at
        # spriteSize=8 pixels per cell, giving an 88x88 RGB tensor. We read
        # the first agent's observation_space from shimmy to pick up the RGB
        # key in case a future version renames it.
        shimmy_obs_space = self.env.observation_space(self.agents[0])
        assert isinstance(shimmy_obs_space, spaces.Dict), (
            f"Expected Dict observation_space, got {type(shimmy_obs_space)}"
        )
        assert "RGB" in shimmy_obs_space.spaces, (
            f"Paintball CTF observation is missing RGB key; have "
            f"{list(shimmy_obs_space.spaces.keys())}"
        )
        rgb_space = shimmy_obs_space.spaces["RGB"]
        self._rgb_shape: Tuple[int, int, int] = tuple(rgb_space.shape)  # (88,88,3)

        per_agent_obs_space = spaces.Box(
            low=0,
            high=255,
            shape=self._rgb_shape,
            dtype=np.uint8,
        )
        self.observation_space: Dict[str, spaces.Space] = {
            agent: per_agent_obs_space for agent in self.agents
        }

        # Action space: 8 discrete actions. Melting Pot returns a Discrete
        # space per agent already, but we force it to Discrete(8) for
        # determinism against the YAML's actor_hidden_size assumptions.
        shimmy_act_space = self.env.action_space(self.agents[0])
        if isinstance(shimmy_act_space, spaces.Discrete):
            self._n_actions = int(shimmy_act_space.n)
        else:
            raise TypeError(
                f"Expected Discrete action space for Paintball CTF, got "
                f"{type(shimmy_act_space)}"
            )
        self.action_space: Dict[str, spaces.Space] = {
            agent: spaces.Discrete(self._n_actions) for agent in self.agents
        }

        # Maximum episode length. The substrate sets maxEpisodeLengthFrames=
        # 1000 (see paintball__capture_the_flag.py:809). shimmy honors that
        # via max_cycles=1000 by default.
        # episode_length in the YAML overrides max_cycles so the RNN buffer
        # allocates (n_envs, episode_length, H, W, C) per agent rather than
        # (n_envs, 1000, H, W, C) which exceeds available RAM.
        self.max_episode_steps: int = int(
            getattr(config, "episode_length", getattr(self.env, "max_cycles", 1000))
        )

        # Centralized state: xuance's MAPPO can run with use_global_state=
        # False, in which case state() is not used for the critic. We still
        # need a valid state_space so get_env_info() returns a consistent
        # shape. Use a heavily downsampled view of one agent's image (stride-
        # 8, single channel) as a cheap placeholder, matching the pattern
        # in xuance/environment/multi_agent_env/atari.py:49,74.
        ds_h = self._rgb_shape[0] // 8  # 88 // 8 = 11
        ds_w = self._rgb_shape[1] // 8  # 88 // 8 = 11
        self._state_shape = (ds_h * ds_w,)  # 121
        self.state_space = spaces.Box(
            low=0,
            high=255,
            shape=self._state_shape,
            dtype=np.uint8,
        )
        self.global_state: np.ndarray = np.zeros(
            self._state_shape, dtype=np.uint8
        )

        self.individual_episode_reward: Dict[str, float] = {
            k: 0.0 for k in self.agents
        }
        self._episode_step: int = 0

        _logger.info(
            "MeltingPot_PaintballCaptureTheFlag_Env initialized: "
            "substrate=%s, agents=%d (red=%d, blue=%d), rgb=%s, "
            "actions=%d, shaping=%s",
            substrate_name,
            self.num_agents,
            len(self.agent_groups[0]),
            len(self.agent_groups[1]),
            self._rgb_shape,
            self._n_actions,
            "on" if shaping_kwargs else "sparse",
        )

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _extract_rgb(
        self, obs_dict: Mapping[str, Mapping[str, Any]]
    ) -> Dict[str, np.ndarray]:
        """Pull the RGB tensor out of each agent's observation dict.

        shimmy returns something like:
            {'player_0': {'RGB': ndarray(88,88,3), 'READY_TO_SHOOT': scalar},
             'player_1': {...}, ...}

        We return just the RGB tensor per agent as uint8, HWC layout. The
        xuance ``Basic_CNN`` representation divides by 255 and permutes to
        ``(C, H, W)`` internally, so we deliberately do NOT normalize here.
        """
        out: Dict[str, np.ndarray] = {}
        for agent in self.agents:
            agent_obs = obs_dict[agent]
            rgb = np.asarray(agent_obs["RGB"], dtype=np.uint8)
            out[agent] = rgb
        return out

    def _update_global_state(self, rgb_obs: Dict[str, np.ndarray]) -> None:
        """Downsample one agent's RGB to a flat 121-length uint8 state."""
        ref = rgb_obs[self.agents[0]]
        # stride-8 downsample of the red channel, matches atari.py pattern
        self.global_state = ref[::8, ::8, 0].reshape(-1).astype(np.uint8)

    # ------------------------------------------------------------------
    # RawMultiAgentEnv API
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        observations, infos = self.env.reset()
        rgb_obs = self._extract_rgb(observations)
        self._update_global_state(rgb_obs)
        self._episode_step = 0
        for k in self.agents:
            self.individual_episode_reward[k] = 0.0
        reset_info = {
            "infos": infos,
            "individual_episode_rewards": dict(self.individual_episode_reward),
        }
        return rgb_obs, reset_info

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        bool,
        Dict[str, Any],
    ]:
        # shimmy expects a dict keyed by the PettingZoo agent names we already
        # use, so pass actions through directly after coercing to int.
        shimmy_actions = {
            agent: int(actions.get(agent, 0)) for agent in self.agents
        }
        observations, rewards, terminated, truncated, infos = self.env.step(
            shimmy_actions
        )
        rgb_obs = self._extract_rgb(observations)
        self._update_global_state(rgb_obs)
        self._episode_step += 1

        # Per-agent rewards and terminated signals stay as dicts because
        # xuance's DummyVecMultiAgentEnv calls ``.values()`` on the
        # terminated dict at dummy_vec_maenv.py:77. Rewards are consumed
        # agent-by-agent downstream.
        rewards_out: Dict[str, float] = {}
        terms_out: Dict[str, bool] = {}
        for agent in self.agents:
            r = float(rewards[agent])
            rewards_out[agent] = r
            self.individual_episode_reward[agent] += r
            terms_out[agent] = bool(terminated[agent])

        # CRITICAL: truncated must be a scalar bool, not a per-agent dict.
        # xuance's DummyVecMultiAgentEnv.step_wait initializes
        # ``truncated = [False for _ in self.envs]`` and then evaluates
        # ``or truncated[e]`` as a plain truthiness check. A non-empty
        # dict is always truthy in Python, so returning a dict here
        # triggers ``self.envs[e].reset()`` on every single tick, which
        # rebuilds the Melting Pot Lua substrate (~300-400 ms) and
        # dominates wall time. MPE's reference wrapper does exactly this
        # scalar pattern at xuance/environment/multi_agent_env/mpe.py:82.
        truncated_scalar = bool(self._episode_step >= self.max_episode_steps)

        # step_info follows the MPE convention at
        # xuance/environment/multi_agent_env/mpe.py:79-80. The
        # XuanCeMultiAgentEnvWrapper at wrapper.py:169-180 then adds
        # episode_step, episode_score, agent_mask, avail_actions, and
        # state keys to this dict before it reaches DummyVec.
        step_info: Dict[str, Any] = {
            "infos": infos,
            "individual_episode_rewards": dict(self.individual_episode_reward),
        }

        return rgb_obs, rewards_out, terms_out, truncated_scalar, step_info

    def state(self) -> np.ndarray:
        return self.global_state

    def agent_mask(self) -> Dict[str, bool]:
        return {agent: True for agent in self.agents}

    def avail_actions(self) -> Dict[str, np.ndarray]:
        return {
            agent: np.ones(self._n_actions, dtype=np.bool_)
            for agent in self.agents
        }

    def render(self, *args, **kwargs):
        return self.env.render()

    def close(self) -> None:
        self.env.close()
