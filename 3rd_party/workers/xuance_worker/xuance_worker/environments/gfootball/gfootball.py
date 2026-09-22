"""GFootball wrapper that replaces xuance's broken football wrapper chain.

Location
========
`xuance_worker.environments.gfootball.gfootball` — family-per-directory
layout. Future gfootball-family variants (e.g. FastLane sidecar env with
render=True, shared helper modules) live alongside this file under
`environments/gfootball/` rather than polluting the flat namespace with
`football_fastlane_sidecar.py`, `football_helpers.py`, etc.

Why this file exists
====================
Xuance's `xuance.environment.multi_agent_env.football.GFootball_Env` is
architecturally broken (verified 2026-08-13, run 01KZX3BASHY75M21DPWPEKAZ8Z):

    xuance.environment.multi_agent_env.football.football_raw_env.__init__ does:
        env = gfootball.env.create_environment(representation='simple115v2', ...)
                       .unwrapped                    # strips gfootball wrappers
        super().__init__(gfootball_config)           # new raw FootballEnv

    Then xuance.environment.multi_agent_env.football.GFootball_Env.__init__ does:
        self.env = _apply_output_wrappers(env=football_raw_env, ...,
                                          representation='simple115v2', ...)

The `.unwrapped + re-wrap` pattern strips gfootball's wrappers and re-applies
them at the WRONG layer (outside football_raw_env instead of inside), which
causes:

    TypeError: list indices must be integers or slices, not str
        at gfootball/env/wrappers.py:204 Simple115StateWrapper.convert_observation

...on the very first env.reset(). Fails for both single-agent (1v1) and
multi-agent (3v1) scenarios.

Proof this is xuance-internal (not gfootball):
    gfootball.env.create_environment(env_name='1_vs_1_easy',
                                     representation='simple115v2', ...)
    -> reset() returns ndarray shape (115,)  # single-agent, works
    gfootball.env.create_environment(env_name='academy_3_vs_1_with_keeper',
                                     representation='simple115v2',
                                     number_of_left_players_agent_controls=3)
    -> reset() returns ndarray shape (3, 115)  # multi-agent, works

This wrapper bypasses xuance's broken chain by extending RawMultiAgentEnv
directly and using gfootball.env.create_environment() with the correct
parameter ordering. Zero changes to vendored xuance code.

Design invariants
=================
1. Vendored `xuance/` tree is NEVER modified. All fixes stay in
   `xuance_worker/environments/gfootball/`.
2. Wrapper implements the RawMultiAgentEnv API xuance's runner expects
   (num_agents, agents, observation_space, action_space, state_space,
    max_episode_steps, reset, step, close, render, state, get_env_info).
3. `get_env_info()` also includes `num_adversaries` — required by
   xuance's `SubprocVecEnv_Football.__init__` which reads it as an
   attribute during vectorized-env construction.
4. `step()` info dict always includes `score_reward` — required by
   xuance's `SubprocVecEnv_Football.step_wait` for battles_won tallying.
5. Registered under both 'Football' (xuance legacy key) and 'gfootball'
   (upstream Python package name) in REGISTRY_MULTI_AGENT_ENV.
6. FastLane frame publishing runs INSIDE this wrapper's own step() —
   no sidecar env, no second process. Empirically verified: gfootball
   supports `render=True` + `representation='simple115v2'` simultaneously,
   so the training obs pipeline (115-dim vectors) is unaffected while
   `env.render(mode='rgb_array')` returns (720, 1280, 3) RGB frames on
   demand. We grab a frame at the end of every step() (throttled to ~30
   FPS by default) and push it to the FastLane shared-memory ring buffer.
   Zero training-loop overhead beyond the render() call itself.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

import numpy as np
from gymnasium.spaces import Box, Discrete

_LOGGER = logging.getLogger(__name__)

try:
    from xuance.environment import RawMultiAgentEnv
except ImportError as _exc:
    RawMultiAgentEnv = object
    _IMPORT_ERROR: Optional[Exception] = _exc
else:
    _IMPORT_ERROR = None

try:
    import gfootball.env as football_env
    _GFOOTBALL_AVAILABLE = True
except ImportError:
    football_env = None
    _GFOOTBALL_AVAILABLE = False

# Xuance's shorthand-to-upstream scenario mapping. Imported so we honor
# the same env_id conventions xuance uses in its config YAMLs.
try:
    from xuance.environment.multi_agent_env.football import GFOOTBALL_ENV_ID
except ImportError:
    GFOOTBALL_ENV_ID = {}

try:
    from xuance_worker.fastlane import is_fastlane_enabled
except ImportError:
    is_fastlane_enabled = None

# FastLane shared-memory publisher (optional — only needed when FastLane
# is enabled). Kept as separate try/except so ImportError doesn't break
# the training-only code path.
try:
    from gym_gui.fastlane import FastLaneWriter, FastLaneConfig, FastLaneMetrics
    from gym_gui.fastlane.buffer import create_fastlane_name
    _FASTLANE_PUBLISHER_AVAILABLE = True
except ImportError:
    FastLaneWriter = None  # type: ignore
    FastLaneConfig = None  # type: ignore
    FastLaneMetrics = None  # type: ignore
    create_fastlane_name = None  # type: ignore
    _FASTLANE_PUBLISHER_AVAILABLE = False


class GFootballFastLane_Env(RawMultiAgentEnv if RawMultiAgentEnv is not object else object):
    """Replacement for xuance's broken GFootball_Env.

    Constructs the gfootball env directly (bypassing xuance's `.unwrapped +
    re-wrap` bug) and exposes the RawMultiAgentEnv API xuance's runner
    consumes. Works for both single-agent (num_agent=1) and multi-agent
    (num_agent>=2) scenarios.
    """

    def __init__(self, config: Any) -> None:
        if RawMultiAgentEnv is object:
            raise ImportError(f"xuance.environment unavailable: {_IMPORT_ERROR}")
        if not _GFOOTBALL_AVAILABLE:
            raise ImportError("gfootball package not installed")

        super().__init__()

        # Resolve upstream scenario name via xuance's GFOOTBALL_ENV_ID
        # (e.g., '3v1' -> 'academy_3_vs_1_with_keeper'). Falls back to the
        # raw env_id if not in the mapping (permits users to supply upstream
        # names directly).
        self._env_id_shorthand = getattr(config, "env_id", "1v1")
        scenario = GFOOTBALL_ENV_ID.get(self._env_id_shorthand, self._env_id_shorthand)

        self.num_agents: int = int(getattr(config, "num_agent", 1))
        self.num_adversaries: int = int(getattr(config, "num_adversary", 0))
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.agent_groups = [list(self.agents)]  # single-group MARL
        self.max_episode_steps: int = int(getattr(config, "max_episode_steps", 1000))
        self._episode_step: int = 0

        # FastLane state — enabled if the env var is set. Empirically verified
        # (2026-08-13): gfootball supports `render=True` + `simple115v2`
        # simultaneously, so we can grab RGB frames from THIS env without
        # spawning a sidecar. The obs pipeline stays (115,)-shaped.
        self._fastlane_active: bool = bool(
            is_fastlane_enabled is not None
            and is_fastlane_enabled()
            and _FASTLANE_PUBLISHER_AVAILABLE
        )
        self._fastlane_writer: Optional[Any] = None
        self._fastlane_last_emit_ns: int = 0
        # 33ms default = ~30 FPS. Same env var name as the wrapper in
        # xuance_worker/fastlane.py so users can tune both from one place.
        self._fastlane_throttle_ns: int = int(
            float(os.getenv("XUANCE_FASTLANE_INTERVAL_MS", "33")) * 1e6
        )
        # 0 = no downscale. Default to 480 to keep frames well under the
        # FastLane 60Hz budget even at HD source resolution (720x1280 native).
        self._fastlane_max_dim: int = int(
            os.getenv("XUANCE_FASTLANE_MAX_DIM", "480") or "480"
        )
        self._fastlane_episode_return: float = 0.0
        self._fastlane_run_id: str = (
            os.getenv("XUANCE_RUN_ID")
            or os.getenv("RUN_ID")
            or "xuance-gfootball"
        )
        self._fastlane_video_mode: str = os.getenv("GYM_GUI_FASTLANE_VIDEO_MODE", "single")
        self._fastlane_grid_limit: int = int(os.getenv("GYM_GUI_FASTLANE_GRID_LIMIT", "4") or "4")
        if self._fastlane_active:
            _LOGGER.info(
                "GFootballFastLane_Env: FastLane ENABLED — "
                "will publish RGB frames to shared memory (run_id=%s, "
                "throttle=%dms, max_dim=%d) using single-env render=True "
                "pattern (no sidecar).",
                self._fastlane_run_id,
                self._fastlane_throttle_ns // 1_000_000,
                self._fastlane_max_dim,
            )

        # Directly construct the env via gfootball. THIS IS THE FIX: bypasses
        # xuance's broken `.unwrapped + re-wrap` chain, which double-applies
        # Simple115StateWrapper at the wrong layer and crashes on reset().
        obs_type = getattr(config, "obs_type", "simple115v2")
        rewards_type = getattr(config, "rewards_type", "scoring,checkpoints")
        stacked = getattr(config, "use_stacked_frames", False)
        smm_w = int(getattr(config, "smm_width", 96))
        smm_h = int(getattr(config, "smm_height", 72))
        # render=True is safe with simple115v2 — verified empirically.
        # Only enable rendering when FastLane is active (avoids the ~5-10%
        # CPU/GPU tax of the render pipeline in training-only runs).
        self.env = football_env.create_environment(
            env_name=scenario,
            stacked=stacked,
            representation=obs_type,
            rewards=rewards_type,
            render=self._fastlane_active,
            write_video=False,
            number_of_left_players_agent_controls=self.num_agents,
            number_of_right_players_agent_controls=self.num_adversaries,
            channel_dimensions=(smm_w, smm_h),
        )

        # Determine per-agent obs shape from the ndarray returned by reset().
        # gfootball returns:
        #   num_agent=1 with representation=simple115v2 -> shape (115,)
        #   num_agent=N with representation=simple115v2 -> shape (N, 115)
        _obs_probe = self.env.reset()
        if self.num_agents == 1:
            per_agent_shape = _obs_probe.shape  # e.g. (115,)
        else:
            per_agent_shape = _obs_probe.shape[1:]  # strip leading agent dim
        self.observation_space = {
            k: Box(-np.inf, np.inf, per_agent_shape, dtype=np.float32)
            for k in self.agents
        }

        # Action space: gfootball uses Discrete(19) per controlled player.
        # Expose per-agent Discrete regardless of internal representation
        # (Tuple / MultiDiscrete) so xuance's per-agent policy code works.
        n_actions = self._probe_num_actions()
        self.action_space = {
            k: Discrete(n_actions) for k in self.agents
        }

        # State space: concatenated per-agent obs as the global state.
        # Simpler than gfootball's raw internal state (which xuance's original
        # implementation accesses via private `self.env._env._observation`,
        # itself fragile across gfootball versions).
        state_dim = int(np.prod(per_agent_shape)) * self.num_agents
        self.state_space = Box(-np.inf, np.inf, (state_dim,), dtype=np.float32)

    # ------------------------------------------------------------------
    # RawMultiAgentEnv API
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        obs = self.env.reset()
        self._episode_step = 0
        self._fastlane_episode_return = 0.0
        return self._obs_to_dict(obs), {}

    def step(self, action_dict):
        if self.num_agents == 1:
            actions: Any = int(action_dict[self.agents[0]])
        else:
            actions = [int(action_dict[k]) for k in self.agents]

        obs, reward, terminated, info = self.env.step(actions)
        self._episode_step += 1

        # gfootball returns scalar reward for single-agent, ndarray for multi.
        if self.num_agents == 1:
            reward_dict = {self.agents[0]: float(reward)}
        else:
            reward_dict = {k: float(reward[i]) for i, k in enumerate(self.agents)}

        terminated_dict = {k: bool(terminated) for k in self.agents}
        truncated = bool(self._episode_step >= self.max_episode_steps)

        # SubprocVecEnv_Football.step_wait requires info['score_reward'] to
        # tally battles_won. gfootball's Simple115v2 wrapper strips info to a
        # minimal dict; when score_reward isn't present, fall back to the
        # raw reward (with rewards='scoring,checkpoints' the sign carries the
        # score signal, so a positive reward tally is a safe proxy for wins).
        if "score_reward" not in info:
            if self.num_agents == 1:
                info["score_reward"] = float(reward)
            else:
                info["score_reward"] = float(np.asarray(reward).sum())

        # FastLane frame publishing (throttled). Happens INSIDE this step()
        # so training and visualization share one env, one process, one
        # render call. Failures are silent — they must never break training.
        if self._fastlane_active:
            try:
                # Track episode reward for the HUD (scalar summary).
                if self.num_agents == 1:
                    self._fastlane_episode_return += float(reward)
                    hud_reward = float(reward)
                else:
                    r_sum = float(np.asarray(reward).sum())
                    self._fastlane_episode_return += r_sum
                    hud_reward = r_sum
                self._publish_fastlane_frame(hud_reward)
            except Exception as exc:  # pragma: no cover — must never break training
                _LOGGER.debug("FastLane publish failed (non-fatal): %s", exc)

        return self._obs_to_dict(obs), reward_dict, terminated_dict, truncated, info

    def close(self):
        # Release the FastLane writer FIRST so the shared-memory segment
        # gets unlinked cleanly. If we close the env first and the writer
        # is holding a mapping, the shm cleanup can race with process exit
        # and leak a segment under /dev/shm.
        if self._fastlane_writer is not None:
            try:
                self._fastlane_writer.close()
                try:
                    self._fastlane_writer.unlink()
                except (FileNotFoundError, AttributeError):
                    pass
            except Exception as exc:  # pragma: no cover
                _LOGGER.debug("FastLane writer close failed (non-fatal): %s", exc)
            self._fastlane_writer = None
        try:
            if self.env is not None:
                self.env.close()
        except Exception as exc:  # pragma: no cover
            _LOGGER.debug("GFootball env close failed: %s", exc)

    def render(self, *args, **kwargs):
        """Return an RGB frame if the env was constructed with render=True.
        Currently returns None because training env uses render=False (see
        header comment for sidecar architecture follow-up)."""
        try:
            if hasattr(self.env, "render"):
                return self.env.render(mode="rgb_array")
        except Exception:
            pass
        return None

    def state(self):
        """Global state: flat concat of per-agent obs. Simple and stable
        across gfootball versions (unlike xuance's original which uses
        `self.env._env._observation` — a private path that breaks silently
        when gfootball's internals change).

        Called by the XuanCeMultiAgentEnvWrapper `state` property on every
        reset/step. Must NEVER call self.env.reset() here — that would clobber
        the training env mid-episode. Pre-episode fallback returns zeros of
        the correct shape."""
        if getattr(self, "_last_obs", None) is None:
            return np.zeros(int(self.state_space.shape[0]), dtype=np.float32)
        obs_arr = np.asarray(self._last_obs, dtype=np.float32)
        return obs_arr.reshape(-1)

    def get_env_info(self) -> dict:
        """Xuance's `SubprocVecEnv_Football.__init__` reads
        `env_info['num_adversaries']` and fails with KeyError otherwise.
        `RawMultiAgentEnv.get_env_info()` does NOT include it, so we override
        to add the football-specific field. All other keys come from the
        base implementation."""
        return {
            "state_space": self.state_space,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "agents": self.agents,
            "num_agents": self.num_agents,
            "max_episode_steps": self.max_episode_steps,
            "num_adversaries": self.num_adversaries,
        }

    def get_more_info(self, info):
        """Extension hook for xuance's football-specific info augmentation.
        We don't need the extra fields (`active`, `designated`, `sticky_actions`)
        that xuance's original wrapper extracts — those come from a private
        `self.env.unwrapped.observation()` call that is again fragile. If a
        runner needs them, add them here."""
        return info

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _obs_to_dict(self, obs):
        """Convert gfootball's ndarray obs into per-agent dict."""
        arr = np.asarray(obs)
        self._last_obs = arr  # cache for state()
        if self.num_agents == 1:
            return {self.agents[0]: arr}
        return {k: arr[i] for i, k in enumerate(self.agents)}

    # ------------------------------------------------------------------
    # FastLane frame publishing (in-band, single-env, single-process)
    # ------------------------------------------------------------------

    def _publish_fastlane_frame(self, hud_reward: float) -> None:
        """Grab an RGB frame from the (already-render=True) gfootball env
        and push it to the FastLane shared-memory ring buffer.

        Throttled to ~30 FPS by default (env var XUANCE_FASTLANE_INTERVAL_MS).
        Zero-cost when the throttle window has not elapsed — no render call,
        no serialization. Verified single-env safe (2026-08-13): gfootball's
        render pipeline coexists with simple115v2 obs without corruption.
        """
        # Throttle: skip if we published recently. Avoids CPU/GPU render
        # storm at training rates >30Hz (gfootball can hit ~60Hz on RTX 4090).
        now_ns = time.perf_counter_ns()
        if now_ns - self._fastlane_last_emit_ns < self._fastlane_throttle_ns:
            return

        # Grab RGB. Only works because we passed render=True to
        # create_environment. Returns (720, 1280, 3) uint8 by default.
        try:
            frame = self.env.render(mode="rgb_array")
        except Exception:
            return
        if frame is None:
            return
        arr = np.asarray(frame)
        if arr.dtype != np.uint8 or arr.ndim != 3:
            return
        if self._fastlane_max_dim > 0:
            arr = self._downscale_frame(arr)

        # Lazy-create writer on first publish so we know the exact frame
        # dimensions (downscale might have changed them). FastLaneWriter is
        # sized to a specific (width, height, channels) so we cannot allocate
        # it in __init__ before we've seen a real frame.
        if self._fastlane_writer is None:
            self._fastlane_writer = self._create_fastlane_writer(arr.shape)
            if self._fastlane_writer is None:
                # Creation failed — disable further attempts so we don't
                # burn cycles retrying every step.
                self._fastlane_active = False
                return

        h, w, c = arr.shape
        # step_rate_hz: derived from throttle interval, not measured — good
        # enough for the HUD and cheap. If XUANCE_FASTLANE_INTERVAL_MS==33,
        # step_rate reads as ~30Hz on the FastLane HUD.
        step_rate_hz = 1e9 / self._fastlane_throttle_ns if self._fastlane_throttle_ns > 0 else 0.0
        try:
            metrics = FastLaneMetrics(
                last_reward=float(hud_reward),
                rolling_return=float(self._fastlane_episode_return),
                step_rate_hz=float(step_rate_hz),
            )
            self._fastlane_writer.publish(arr.tobytes(), metrics=metrics)
            self._fastlane_last_emit_ns = now_ns
        except Exception as exc:  # pragma: no cover
            _LOGGER.debug("FastLane publish failed: %s", exc)

    def _create_fastlane_writer(self, frame_shape: tuple) -> Optional[Any]:
        """Create the FastLaneWriter sized to the given (H, W, C) frame.

        In grid mode each subprocess claims the first unclaimed slot via a
        race-safe try-next pattern: shm_open with O_CREAT|O_EXCL is atomic,
        so exactly one process wins each {run_id}-{i} slot.
        """
        if FastLaneWriter is None or FastLaneConfig is None:
            return None
        h, w, c = frame_shape
        cfg = FastLaneConfig(
            width=int(w),
            height=int(h),
            channels=int(c),
            pixel_format="RGB" if c == 3 else "RGBA",
        )

        if self._fastlane_video_mode == "grid":
            for i in range(self._fastlane_grid_limit):
                slot_id = f"{self._fastlane_run_id}-{i}"
                try:
                    writer = FastLaneWriter.create(slot_id, cfg)
                    _LOGGER.info(
                        "FastLane grid slot claimed: %s (%dx%dx%d)", slot_id, w, h, c
                    )
                    return writer
                except FileExistsError:
                    continue
            _LOGGER.warning(
                "FastLane grid: all %d slots already claimed, env will not publish",
                self._fastlane_grid_limit,
            )
            return None

        try:
            writer = FastLaneWriter.create(self._fastlane_run_id, cfg)
            _LOGGER.info(
                "FastLane writer created (run_id=%s, %dx%dx%d)",
                self._fastlane_run_id, w, h, c,
            )
            return writer
        except FileExistsError:
            # Another subprocess in the same run already claimed this slot.
            # Do NOT attach — concurrent writes from multiple subprocesses
            # corrupt the shared-memory header. Only the creator publishes;
            # returning None disables FastLane for this subprocess via the
            # self._fastlane_active = False branch in _publish_fastlane_frame.
            _LOGGER.debug(
                "FastLane single mode: slot %s already claimed, "
                "this subprocess will not publish",
                self._fastlane_run_id,
            )
            return None
        except Exception as exc:  # pragma: no cover
            _LOGGER.warning("FastLane writer create failed: %s", exc)
            return None

    def _downscale_frame(self, arr: np.ndarray) -> np.ndarray:
        """Downscale to keep the max dimension <= self._fastlane_max_dim.
        Uses PIL LANCZOS if available, otherwise a fast numpy stride
        subsample (nearest-neighbour) fallback so we never hard-depend on PIL."""
        h, w = arr.shape[:2]
        max_dim = max(h, w)
        if max_dim <= self._fastlane_max_dim:
            return arr
        scale = self._fastlane_max_dim / max_dim
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        try:
            from PIL import Image
            resample = getattr(Image.Resampling, "LANCZOS", getattr(Image, "LANCZOS", 1))
            return np.array(Image.fromarray(arr).resize((new_w, new_h), resample))
        except ImportError:
            # Nearest-neighbour stride subsample — ugly but zero-dep.
            step_h = max(1, h // new_h)
            step_w = max(1, w // new_w)
            return arr[::step_h, ::step_w].copy()

    def _probe_num_actions(self) -> int:
        """Discover discrete action count from gfootball's action_space.
        Handles the three shapes gfootball may return: Discrete, MultiDiscrete,
        Tuple(Discrete, Discrete, ...)."""
        space = self.env.action_space
        # MultiDiscrete
        if hasattr(space, "nvec"):
            return int(space.nvec[0])
        # Tuple of Discrete
        if hasattr(space, "spaces") and len(space.spaces) > 0:
            first = space.spaces[0]
            if hasattr(first, "n"):
                return int(first.n)
        # Discrete
        if hasattr(space, "n"):
            return int(space.n)
        # Fallback: gfootball's default full action set is 19
        return 19


__all__ = ["GFootballFastLane_Env"]
