"""StarCraft II (SMAC) wrapper with FastLane frame publishing.

Location
========
`xuance_worker.environments.smac.smac` — family-per-directory layout,
mirrors the `environments/gfootball/` structure added 2026-08-13.

Why this file exists
====================
XuanCe's `xuance.environment.multi_agent_env.starcraft2.StarCraft2_Env`
works correctly for training, but publishes no live frames to the FastLane
shared-memory pipeline. This wrapper adds FastLane support while keeping
the full SMAC-specific API (avail_actions, agent_mask, get_env_info with
num_enemies) intact.

Two rendering modes are supported (selected via MOSAIC_SMAC_RENDER_MODE):

- ``"3d"`` (default): SC2 native GPU frames extracted from
  ``env._obs.observation.render_data.map`` after ``_apply_3d_render_patch``
  monkey-patches ``_launch()`` to pass ``want_rgb=True`` and add a
  ``SpatialCameraSetup``. SMAC's own ``render()`` is PyGame-only and never
  reads render_data — the data must be read from the protobuf directly.

- ``"heatmap"``: Pure-numpy 2x2 panel rendered by
  ``gym_gui.rendering.smac_heatmap``, which reads SC2's raw protobuf
  observation (``env._obs.observation.raw_data.units``). Fully headless,
  no 3D pipeline required.

Design invariants
=================
1. Vendored `xuance/` tree is NEVER modified.
2. Full SMAC-specific API preserved: avail_actions(), agent_mask(),
   get_env_info() with num_enemies key.
3. FastLane frame publishing is mode-aware: 3D frames from render_data,
   heatmap frames from raw_data.units; PyGame render() is the final
   fallback for both modes.
4. close() releases the FastLane writer BEFORE closing the SMAC env,
   matching the close-order invariant in GFootballFastLane_Env.
5. Registered under 'StarCraft2' in REGISTRY_MULTI_AGENT_ENV, overriding
   xuance's upstream entry (which is always identical in behaviour when
   FastLane is disabled, so the override is safe for non-FastLane runs).
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
    from smac.env import StarCraft2Env as _StarCraft2Env
    _SMAC_AVAILABLE = True
except Exception:
    # Broad catch: pysc2/s2clientprotocol can raise TypeError at import
    # time when the protobuf runtime version is incompatible (e.g. >=4.x vs
    # the generated pb2 code targeting <3.20). ImportError alone is not
    # sufficient.
    _StarCraft2Env = None  # type: ignore
    _SMAC_AVAILABLE = False

try:
    from xuance_worker.fastlane import is_fastlane_enabled
except ImportError:
    is_fastlane_enabled = None

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


class StarCraft2FastLane_Env(RawMultiAgentEnv if RawMultiAgentEnv is not object else object):
    """SMAC wrapper that adds FastLane frame publishing to training runs.

    Wraps `smac.env.StarCraft2Env` directly and exposes the full
    `RawMultiAgentEnv` API that XuanCe's SMAC runner consumes, including
    the SMAC-specific methods `avail_actions()` and `agent_mask()`.

    When `MOSAIC_FASTLANE_ENABLED=1` is set in the subprocess environment,
    a frame is rendered after each training step and published to the
    FastLane shared-memory ring buffer. The render mode is controlled by
    ``MOSAIC_SMAC_RENDER_MODE``: ``"3d"`` extracts SC2 native GPU frames
    from the protobuf render_data; ``"heatmap"`` uses pure-numpy feature
    panels from raw_data.units. When FastLane is disabled this class is
    behaviourally identical to XuanCe's upstream StarCraft2_Env.
    """

    def __init__(self, config: Any) -> None:
        if RawMultiAgentEnv is object:
            raise ImportError(f"xuance.environment unavailable: {_IMPORT_ERROR}")
        if not _SMAC_AVAILABLE:
            raise ImportError("smac package not installed")

        super().__init__()

        self.env = _StarCraft2Env(map_name=config.env_id)
        self.env_info = self.env.get_env_info()

        self.num_agents: int = self.env_info["n_agents"]
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.state_space = Box(
            low=-np.inf, high=np.inf,
            shape=(self.env_info["state_shape"],),
            dtype=np.float32,
        )
        self.observation_space = {
            k: Box(low=-np.inf, high=np.inf, shape=(self.env_info["obs_shape"],), dtype=np.float32)
            for k in self.agents
        }
        self.action_space = {
            k: Discrete(n=self.env_info["n_actions"]) for k in self.agents
        }
        self.max_episode_steps: int = self.env_info["episode_limit"]
        self._episode_step: int = 0

        # Render mode: "heatmap" (default, headless pure-numpy) or "3d"
        # (SC2 native GPU render via want_rgb=True _launch patch).
        self._smac_render_mode: str = os.getenv("MOSAIC_SMAC_RENDER_MODE", "heatmap")
        self._smac_render_size: int = int(
            os.getenv("MOSAIC_SMAC_RENDER_SIZE", "1024") or "1024"
        )

        # FastLane state — enabled if the env var is set.
        self._fastlane_active: bool = bool(
            is_fastlane_enabled is not None
            and is_fastlane_enabled()
            and _FASTLANE_PUBLISHER_AVAILABLE
        )
        self._fastlane_writer: Optional[Any] = None
        self._fastlane_last_emit_ns: int = 0
        self._fastlane_throttle_ns: int = int(
            float(os.getenv("XUANCE_FASTLANE_INTERVAL_MS", "33")) * 1e6
        )
        self._fastlane_max_dim: int = int(
            os.getenv("XUANCE_FASTLANE_MAX_DIM", "480") or "480"
        )
        self._fastlane_episode_return: float = 0.0
        self._fastlane_run_id: str = (
            os.getenv("XUANCE_RUN_ID")
            or os.getenv("RUN_ID")
            or "xuance-smac"
        )
        self._fastlane_video_mode: str = os.getenv("GYM_GUI_FASTLANE_VIDEO_MODE", "single")
        self._fastlane_grid_limit: int = int(os.getenv("GYM_GUI_FASTLANE_GRID_LIMIT", "4") or "4")
        # Lazy-created on first frame publish.
        self._heatmap_renderer: Optional[Any] = None
        # Populated after first env.reset() triggers _launch().
        self._playable_area: Optional[tuple] = None

        if self._fastlane_active:
            _LOGGER.info(
                "StarCraft2FastLane_Env: FastLane ENABLED — "
                "render_mode=%s, render_size=%d, throttle=%dms, max_dim=%d (run_id=%s).",
                self._smac_render_mode,
                self._smac_render_size,
                self._fastlane_throttle_ns // 1_000_000,
                self._fastlane_max_dim,
                self._fastlane_run_id,
            )

        # Apply 3D GPU render patch BEFORE first reset() triggers _launch().
        # The patch replaces self.env._launch at instance level so that SC2
        # starts with want_rgb=True and returns RGB frames via EGL.
        if self._fastlane_active and self._smac_render_mode == "3d":
            self._apply_3d_render_patch()

        # Trigger SC2 process launch (mirrors xuance's StarCraft2_Env.__init__).
        try:
            self.env.reset(seed=config.env_seed)
        except Exception:
            self.env.reset()

        # Extract playable area bounds that the heatmap renderer needs.
        # SMAC's _launch() stores max_distance_x/y but not the origin, so
        # we approximate with (0, 0, map_x, map_y) — good enough for the
        # heatmap's unit-position normalisation.
        self._playable_area = self._get_playable_area()

    # ------------------------------------------------------------------
    # RawMultiAgentEnv API
    # ------------------------------------------------------------------

    def get_env_info(self) -> dict:
        return {
            "state_space": self.state_space,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "agents": self.agents,
            "num_agents": self.env_info["n_agents"],
            "max_episode_steps": self.max_episode_steps,
            "num_enemies": self.env.n_enemies,
        }

    def reset(self, **kwargs):
        obs, _ = self.env.reset()
        obs_dict = {key: obs[index] for index, key in enumerate(self.agents)}
        self._episode_step = 0
        self._fastlane_episode_return = 0.0
        return obs_dict, {}

    def step(self, actions):
        actions_list = [actions[key] for key in self.agents]
        reward, terminated, info = self.env.step(actions_list)
        if not info:
            info = {"battle_won": 0, "dead_allies": 0, "dead_enemies": 0}

        reward_dict = {k: reward for k in self.agents}
        terminated_dict = {k: terminated for k in self.agents}
        obs = self.env.get_obs()
        obs_dict = {key: obs[index] for index, key in enumerate(self.agents)}
        self._episode_step += 1
        truncated = bool(self._episode_step >= self.max_episode_steps)

        if self._fastlane_active:
            try:
                self._fastlane_episode_return += float(reward)
                self._publish_fastlane_frame(float(reward))
            except Exception as exc:  # pragma: no cover — must never break training
                _LOGGER.debug("FastLane publish failed (non-fatal): %s", exc)

        return obs_dict, reward_dict, terminated_dict, truncated, info

    def close(self) -> None:
        # Release writer FIRST to avoid shm race on process exit.
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
            _LOGGER.debug("SMAC env close failed: %s", exc)

    def render(self, mode=None):
        return self._get_smac_render_frame()

    def state(self):
        return self.env.get_state()

    def agent_mask(self) -> dict:
        """Boolean alive-mask per agent. Mirrors xuance's upstream default."""
        return {agent: True for agent in self.agents}

    def avail_actions(self) -> dict:
        """Boolean action-availability mask per agent (SMAC-specific)."""
        masks = self.env.get_avail_actions()
        return {key: masks[index] for index, key in enumerate(self.agents)}

    # ------------------------------------------------------------------
    # Internals — playable area + heatmap render
    # ------------------------------------------------------------------

    def _get_playable_area(self) -> tuple:
        """Return (min_x, min_y, max_x, max_y) for the heatmap renderer.

        SMAC's _launch() stores max_distance_x/y (width/height of the
        playable area) but does not expose the origin. We approximate the
        origin as (0, 0), which is correct on most SMAC maps and merely
        shifts heatmap coordinates by the origin offset on the rest.
        """
        try:
            return (
                0.0, 0.0,
                float(self.env.max_distance_x),
                float(self.env.max_distance_y),
            )
        except AttributeError:
            pass
        return (
            0.0, 0.0,
            float(getattr(self.env, "map_x", 32)),
            float(getattr(self.env, "map_y", 32)),
        )

    def _apply_3d_render_patch(self) -> None:
        """Monkey-patch SMAC's _launch() for SC2 native 3D GPU rendering.

        Imports `_patch_launch_for_3d` from the gym_gui adapter and applies
        it at instance level. On failure (gym_gui unavailable, missing pysc2
        deps, etc.) falls back silently to heatmap mode so training is never
        blocked.
        """
        try:
            from gym_gui.core.adapters.smac import _patch_launch_for_3d
            _patch_launch_for_3d(self.env, render_size=self._smac_render_size)
            _LOGGER.info(
                "StarCraft2FastLane_Env: 3D GPU render patch applied "
                "(render_size=%d px square).",
                self._smac_render_size,
            )
        except Exception as exc:
            _LOGGER.warning(
                "StarCraft2FastLane_Env: 3D GPU patch failed — "
                "falling back to heatmap: %s",
                exc,
            )
            self._smac_render_mode = "heatmap"

    def _get_smac_render_frame(self) -> Optional[np.ndarray]:
        """Produce an RGB frame from the current SMAC state.

        When render mode is '3d', reads SC2 native GPU frames directly from
        ``self.env._obs.observation.render_data.map`` (the protobuf observation
        produced by SC2's EGL pipeline after ``_apply_3d_render_patch`` has
        replaced ``_launch``). SMAC's own ``render()`` is PyGame-only and never
        reads render_data — calling it in 3D mode returns a PyGame circle frame,
        not the SC2 3D scene.

        Primary fallback: `gym_gui.rendering.smac_heatmap` (pure numpy,
        fully headless). Final fallback: SMAC's PyGame render().
        """
        # 3D GPU path — extract RGB from SC2 protobuf render_data.
        # Steps:
        #   1. Reposition the SC2 camera to the unit centroid via
        #      _center_camera_on_units (camera_move action + observe).
        #      This does NOT call controller.step(), so the game loop is
        #      unaffected. It also refreshes env._obs so render_data.map
        #      reflects the new camera position.
        #   2. Extract the raw frame from render_data.map.
        #   3. Composite the minimap inset onto the frame.
        if self._smac_render_mode == "3d":
            try:
                from gym_gui.core.adapters.smac import (
                    _center_camera_on_units,
                    _composite_minimap_inset,
                )
            except ImportError:
                _center_camera_on_units = None  # type: ignore[assignment]
                _composite_minimap_inset = None  # type: ignore[assignment]
            try:
                camera_center = (
                    _center_camera_on_units(self.env)
                    if _center_camera_on_units is not None
                    else None
                )
                obs = getattr(self.env, "_obs", None)
                if obs is not None and obs.observation.HasField("render_data"):
                    map_img = obs.observation.render_data.map
                    if len(map_img.data) > 0:
                        channels = map_img.bits_per_pixel // 8
                        frame = np.frombuffer(map_img.data, dtype=np.uint8).reshape(
                            map_img.size.y, map_img.size.x, channels
                        )
                        if channels == 4:
                            frame = frame[:, :, :3]
                        if camera_center is not None and _composite_minimap_inset is not None:
                            try:
                                frame = _composite_minimap_inset(
                                    frame,
                                    obs,
                                    self._playable_area or (0.0, 0.0, 32.0, 32.0),
                                    camera_center,
                                    asset_family="SMAC",
                                )
                            except Exception:
                                pass
                        if not hasattr(self, "_3d_frame_logged"):
                            self._3d_frame_logged = True
                            _LOGGER.debug(
                                "StarCraft2FastLane_Env: first 3D frame — "
                                "%dx%d px, %d channels",
                                map_img.size.x, map_img.size.y, channels,
                            )
                        return frame
            except Exception:
                pass

        # Primary: heatmap renderer — reads env._obs protobuf, no SC2
        # 3D render pipeline required.
        try:
            from gym_gui.rendering.smac_heatmap import (
                SMACHeatmapRenderer,
                extract_frame_data,
            )
            if self._heatmap_renderer is None:
                self._heatmap_renderer = SMACHeatmapRenderer()
            area = self._playable_area or (0.0, 0.0, 32.0, 32.0)
            fd = extract_frame_data(
                self.env,
                self._episode_step,
                getattr(self.env, "map_name", "smac"),
                area,
            )
            if fd is not None:
                frame = self._heatmap_renderer.render(fd)
                if isinstance(frame, np.ndarray):
                    return frame
        except Exception:
            pass

        # Fallback: SMAC's built-in PyGame render (needs a display or
        # SDL_VIDEODRIVER=dummy). May return None or raise if pygame is
        # not available — both are handled silently.
        try:
            frame = self.env.render(mode="rgb_array")
            if isinstance(frame, np.ndarray):
                return frame
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # FastLane frame publishing (in-band, single-env, single-process)
    # ------------------------------------------------------------------

    def _publish_fastlane_frame(self, hud_reward: float) -> None:
        """Render a heatmap frame and push it to the FastLane ring buffer.

        Throttled to ~30 FPS by default (XUANCE_FASTLANE_INTERVAL_MS).
        When the throttle window has not elapsed, returns immediately with
        no render call and no serialisation overhead.
        """
        now_ns = time.perf_counter_ns()
        if now_ns - self._fastlane_last_emit_ns < self._fastlane_throttle_ns:
            return

        frame = self._get_smac_render_frame()
        if frame is None:
            return
        arr = np.asarray(frame)
        if arr.dtype != np.uint8 or arr.ndim != 3:
            return
        if self._fastlane_max_dim > 0:
            arr = self._downscale_frame(arr)

        if self._fastlane_writer is None:
            self._fastlane_writer = self._create_fastlane_writer(arr.shape)
            if self._fastlane_writer is None:
                self._fastlane_active = False
                return

        step_rate_hz = (
            1e9 / self._fastlane_throttle_ns if self._fastlane_throttle_ns > 0 else 0.0
        )
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
            # returning None disables FastLane for this subprocess.
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
        """Downscale to keep max dimension <= self._fastlane_max_dim.
        PIL LANCZOS when available; numpy stride subsample as fallback."""
        h, w = arr.shape[:2]
        max_dim = max(h, w)
        if max_dim <= self._fastlane_max_dim:
            return arr
        scale = self._fastlane_max_dim / max_dim
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        try:
            from PIL import Image
            resample = getattr(
                Image.Resampling, "LANCZOS", getattr(Image, "LANCZOS", 1)
            )
            return np.array(Image.fromarray(arr).resize((new_w, new_h), resample))
        except ImportError:
            step_h = max(1, h // new_h)
            step_w = max(1, w // new_w)
            return arr[::step_h, ::step_w].copy()


__all__ = ["StarCraft2FastLane_Env"]
