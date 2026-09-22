"""ViZDoom single-agent wrapper for xuance_worker.

Location
========
`xuance_worker.environments.vizdoom.vizdoom` (family-per-directory layout,
matching gfootball/gfootball.py and smac/). Future variants (deathmatch
multi-agent wrapper, FastLane sidecar) live alongside this file under
`environments/vizdoom/` rather than polluting the flat namespace.

Why this file exists
====================
Xuance's REGISTRY_ENV has no vizdoom entry (verified 2026-09-02: grep
across vendored xuance/ tree returned zero hits). The MOSAIC UI at
gym_gui/ui/widgets/xuance_train_form.py:338 lists 10 ViZDoom scenarios
in the training dropdown, but selecting one crashed at runtime because
xuance could not construct the env. This wrapper closes that gap.

Design invariants
=================
1. Vendored xuance/ tree is NEVER modified. Registration happens
   externally via environments/__init__.py::register_mosaic_environments.
2. Extends gymnasium.Env directly (single-agent path). Deathmatch, the
   only multi-agent ViZDoom scenario, needs a separate RawMultiAgentEnv
   wrapper and is out of scope here.
3. Observation is the RGB screen buffer (uint8, H x W x 3). Format is
   fixed by the scenario spec; we do not expose configuration for now.
4. Action space is Discrete(len(available_buttons)); each integer maps
   to a one-hot ViZDoom button vector at step() time. This matches
   xuance's Discrete-action policies (PPO, DQN, A2C).
5. Frame repeat is a scenario-level parameter (default 4). One xuance
   step corresponds to N ViZDoom engine tics, so wall-clock training
   throughput is roughly N times higher than a per-tic wrapper.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete

from xuance_worker.environments.vizdoom.scenarios import (
    VIZDOOM_SCENARIOS,
    ViZDoomScenarioSpec,
)

try:
    from xuance_worker.fastlane import is_fastlane_enabled
except ImportError:
    is_fastlane_enabled = None

# FastLane shared-memory publisher (optional). Import is guarded so the
# ViZDoom_Env stays functional in environments where gym_gui is not on the
# import path (e.g. headless remote workers).
try:
    from gym_gui.fastlane import FastLaneWriter, FastLaneConfig, FastLaneMetrics
    _FASTLANE_PUBLISHER_AVAILABLE = True
except ImportError:
    FastLaneWriter = None  # type: ignore
    FastLaneConfig = None  # type: ignore
    FastLaneMetrics = None  # type: ignore
    _FASTLANE_PUBLISHER_AVAILABLE = False

_LOGGER = logging.getLogger(__name__)

_VAR_VIZDOOM_INI = Path(
    "/home/hamid/Desktop/software/mosaic/var/data/vizdoom/_vizdoom.ini"
)


def _resolution_to_hw(resolution: str) -> tuple[int, int]:
    """Turn 'RES_320X240' into (240, 320)."""
    if not resolution.startswith("RES_"):
        raise ValueError(f"Unrecognized ViZDoom resolution: {resolution}")
    w_str, h_str = resolution.removeprefix("RES_").split("X")
    return int(h_str), int(w_str)


class ViZDoom_Env(Env):
    """Single-agent ViZDoom environment for xuance training.

    Config object contract (duck-typed, matching xuance conventions):
        env_id: str        e.g. "ViZDoom-Basic-v0"
        render_mode: str   "rgb_array" or "human"; only "rgb_array" tested
        frame_repeat: int  optional; falls back to scenario spec default
        seed: int          optional; forwarded to the ViZDoom engine
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, config: Any) -> None:
        import vizdoom as vzd

        super().__init__()

        env_id: str = getattr(config, "env_id", "ViZDoom-Basic-v0")
        if env_id not in VIZDOOM_SCENARIOS:
            raise KeyError(
                f"Unknown ViZDoom env_id {env_id!r}; "
                f"supported: {sorted(VIZDOOM_SCENARIOS)}"
            )
        spec: ViZDoomScenarioSpec = VIZDOOM_SCENARIOS[env_id]
        self._spec = spec
        self._env_id = env_id
        self._frame_repeat = int(getattr(config, "frame_repeat", spec.frame_repeat))
        self._seed = int(getattr(config, "seed", 0))

        _VAR_VIZDOOM_INI.parent.mkdir(parents=True, exist_ok=True)

        game = vzd.DoomGame()
        game.set_doom_config_path(str(_VAR_VIZDOOM_INI))
        game.load_config(str(Path(vzd.scenarios_path) / spec.scenario_cfg))
        game.set_screen_format(getattr(vzd.ScreenFormat, spec.screen_format))
        game.set_screen_resolution(getattr(vzd.ScreenResolution, spec.screen_resolution))
        game.set_window_visible(False)
        game.set_mode(vzd.Mode.PLAYER)
        game.set_seed(self._seed)

        game.clear_available_buttons()
        for name in spec.available_buttons:
            game.add_available_button(getattr(vzd.Button, name))

        game.init()
        self._game: "vzd.DoomGame" = game
        self._closed: bool = False

        h, w = _resolution_to_hw(spec.screen_resolution)
        self.observation_space = Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8)
        self.action_space = Discrete(len(spec.available_buttons))
        self.max_episode_steps = int(game.get_episode_timeout() or 2100)

        self._last_frame: Optional[np.ndarray] = None
        self._episode_step: int = 0

        # FastLane state. Mirrors the pattern in
        # xuance_worker/environments/gfootball/gfootball.py::__init__ so
        # both wrappers publish to the same shared-memory layout and honour
        # the same set of tuning env vars.
        self._fastlane_active: bool = bool(
            is_fastlane_enabled is not None
            and is_fastlane_enabled()
            and _FASTLANE_PUBLISHER_AVAILABLE
        )
        self._fastlane_writer: Optional[Any] = None
        self._fastlane_last_emit_ns: int = 0
        # 33ms = ~30 FPS. Same env var as gfootball wrapper so both can be
        # tuned from one place.
        self._fastlane_throttle_ns: int = int(
            float(os.getenv("XUANCE_FASTLANE_INTERVAL_MS", "33")) * 1e6
        )
        # 0 = no downscale. Default 480 keeps native 320x240 ViZDoom frames
        # untouched; a user bumping ViZDoom to RES_1024X768 gets downscale
        # for free without a code change.
        self._fastlane_max_dim: int = int(
            os.getenv("XUANCE_FASTLANE_MAX_DIM", "480") or "480"
        )
        self._fastlane_episode_return: float = 0.0
        self._fastlane_run_id: str = (
            os.getenv("XUANCE_RUN_ID")
            or os.getenv("RUN_ID")
            or "xuance-vizdoom"
        )
        self._fastlane_video_mode: str = os.getenv("GYM_GUI_FASTLANE_VIDEO_MODE", "single")
        self._fastlane_grid_limit: int = int(os.getenv("GYM_GUI_FASTLANE_GRID_LIMIT", "4") or "4")
        if self._fastlane_active:
            _LOGGER.info(
                "ViZDoom_Env: FastLane ENABLED for %s (run_id=%s, throttle=%dms, max_dim=%d)",
                self._env_id,
                self._fastlane_run_id,
                self._fastlane_throttle_ns // 1_000_000,
                self._fastlane_max_dim,
            )

    def _current_frame(self) -> np.ndarray:
        state = self._game.get_state()
        if state is None or state.screen_buffer is None:
            if self._last_frame is not None:
                return self._last_frame
            h, w = _resolution_to_hw(self._spec.screen_resolution)
            return np.zeros((h, w, 3), dtype=np.uint8)
        frame = state.screen_buffer
        if frame.ndim == 3 and frame.shape[0] == 3:
            frame = np.transpose(frame, (1, 2, 0))
        self._last_frame = frame.astype(np.uint8, copy=False)
        return self._last_frame

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        del options
        if seed is not None:
            self._game.set_seed(int(seed))
        self._game.new_episode()
        self._episode_step = 0
        self._last_frame = None
        self._fastlane_episode_return = 0.0
        return self._current_frame(), {"env_id": self._env_id}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        button_count = len(self._spec.available_buttons)
        one_hot = [0] * button_count
        one_hot[int(action)] = 1
        reward = float(self._game.make_action(one_hot, self._frame_repeat))
        self._episode_step += 1
        terminated = bool(self._game.is_episode_finished())
        truncated = self._episode_step >= self.max_episode_steps
        info = {
            "env_id": self._env_id,
            "episode_step": self._episode_step,
        }
        frame = self._current_frame()

        # FastLane frame publishing (throttled). Failures are silent so
        # visualization issues never break the training loop.
        if self._fastlane_active:
            try:
                self._fastlane_episode_return += reward
                self._publish_fastlane_frame(frame, reward)
            except Exception as exc:  # pragma: no cover
                _LOGGER.debug("FastLane publish failed (non-fatal): %s", exc)

        return frame, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        return self._current_frame()

    def close(self) -> None:
        # Release the FastLane writer FIRST so the shared-memory segment
        # gets unlinked cleanly. Closing the DoomGame first while the writer
        # still holds a mapping can race with process exit and leak an
        # entry under /dev/shm.
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
        if not self._closed:
            self._game.close()
            self._closed = True
            _LOGGER.debug("Closed ViZDoom_Env for %s", self._env_id)

    def get_env_info(self) -> dict:
        return {
            "num_agents": 1,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "max_episode_steps": self.max_episode_steps,
            "env_id": self._env_id,
        }

    # ------------------------------------------------------------------
    # FastLane frame publishing (in-band, single-env, single-process)
    # ------------------------------------------------------------------

    def _publish_fastlane_frame(self, frame: np.ndarray, hud_reward: float) -> None:
        """Publish an RGB frame to the FastLane shared-memory ring buffer.

        Throttled to ~30 FPS by default (env var XUANCE_FASTLANE_INTERVAL_MS).
        Zero-cost when the throttle window has not elapsed. Unlike the
        gfootball wrapper we do not need a separate env.render() call:
        ViZDoom already produces the RGB observation, so we reuse it.
        """
        # Explicit guard so pyright can narrow FastLaneMetrics from Optional
        # to concrete. Also cheap defensive check: if these symbols are None
        # then _fastlane_active should never have been True in the first
        # place, but the guard costs one branch and prevents an obscure
        # AttributeError if init-time invariants ever break.
        if FastLaneMetrics is None or not _FASTLANE_PUBLISHER_AVAILABLE:
            self._fastlane_active = False
            return

        now_ns = time.perf_counter_ns()
        if now_ns - self._fastlane_last_emit_ns < self._fastlane_throttle_ns:
            return

        arr = np.asarray(frame)
        if arr.dtype != np.uint8 or arr.ndim != 3:
            return
        if self._fastlane_max_dim > 0:
            arr = self._downscale_frame(arr)

        if self._fastlane_writer is None:
            self._fastlane_writer = self._create_fastlane_writer(arr.shape)
            if self._fastlane_writer is None:
                # Creation failed. Disable further attempts to avoid burning
                # cycles retrying on every step.
                self._fastlane_active = False
                return

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
        """Create a FastLaneWriter sized to the given (H, W, C) frame.

        In grid mode each subprocess claims the first unclaimed slot via a
        race-safe try-next pattern. In single mode only the creator of the
        slot publishes; other subprocesses returning FileExistsError disable
        their own FastLane emission so we never corrupt the shared buffer
        header with concurrent writes.
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
            _LOGGER.debug(
                "FastLane single mode: slot %s already claimed, this subprocess will not publish",
                self._fastlane_run_id,
            )
            return None
        except Exception as exc:  # pragma: no cover
            _LOGGER.warning("FastLane writer create failed: %s", exc)
            return None

    def _downscale_frame(self, arr: np.ndarray) -> np.ndarray:
        """Downscale so max(H, W) <= self._fastlane_max_dim.

        Uses PIL LANCZOS if available; falls back to a numpy stride subsample
        (nearest-neighbour) so PIL is optional.
        """
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
            step_h = max(1, h // new_h)
            step_w = max(1, w // new_w)
            return arr[::step_h, ::step_w].copy()
