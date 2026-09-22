"""StarCraft II v2 (SMACv2) wrapper with FastLane frame publishing.

SMACv2 uses ``smacv2.env.StarCraftCapabilityEnvWrapper`` which injects
procedurally generated unit compositions at each episode reset via SC2's
debug API. This is fundamentally different from SMAC v1 (units pre-placed
in .SC2Map files) — hence the separate class.

Map name convention: ``{race}_{n_agents}_vs_{n_enemies}``
e.g. ``protoss_5_vs_5``, ``terran_10_vs_11``, ``zerg_20_vs_23``.

The base SC2 map is always ``32x32_flat`` (empty flat terrain); unit
composition is sampled from the race's weighted unit pool each episode.

For 3D GPU rendering, the ``_launch`` patch must target the inner
``smacv2.env.StarCraft2Env`` (``self.env.env``), not the capability wrapper:
setting an attribute on the wrapper leaves the inner env's ``_launch``
untouched because ``StarCraftCapabilityEnvWrapper.__getattr__`` only
intercepts GET operations, not SET.

Registered under ``'StarCraft2v2'`` in ``REGISTRY_MULTI_AGENT_ENV``.
XuanCe's ``RunnerStarCraft2`` is reused; YAML configs must have
``env_name: "StarCraft2v2"``.
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
    from smacv2.env.starcraft2.wrapper import (
        StarCraftCapabilityEnvWrapper as _StarCraftCapabilityEnvWrapper,
    )
    _SMACV2_AVAILABLE = True
except Exception:
    _StarCraftCapabilityEnvWrapper = None  # type: ignore
    _SMACV2_AVAILABLE = False

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


# ---------------------------------------------------------------------------
# SMACv2 capability configs (EPyMARL benchmark standard weights)
# ---------------------------------------------------------------------------

_SMACV2_RACE_CONFIGS: dict = {
    "terran": {
        "base_map": "10gen_terran",
        "team_gen": {
            "dist_type": "weighted_teams",
            "unit_types": ["marine", "marauder", "medivac"],
            "weights": [0.45, 0.45, 0.1],
            "observe": True,
            "exception_unit_types": ["medivac"],
        },
    },
    "protoss": {
        "base_map": "10gen_protoss",
        "team_gen": {
            "dist_type": "weighted_teams",
            "unit_types": ["stalker", "zealot", "colossus"],
            "weights": [0.45, 0.45, 0.1],
            "observe": True,
            "exception_unit_types": ["colossus"],
        },
    },
    "zerg": {
        "base_map": "10gen_zerg",
        "team_gen": {
            "dist_type": "weighted_teams",
            "unit_types": ["zergling", "hydralisk", "baneling"],
            "weights": [0.45, 0.45, 0.1],
            "observe": True,
            "exception_unit_types": ["baneling"],
        },
    },
}

_SMACV2_START_POSITIONS: dict = {
    "dist_type": "surrounded_and_reflect",
    "p": 0.5,
    "map_x": 32,
    "map_y": 32,
}


def _parse_smacv2_map_name(map_name: str) -> "tuple[str, int, int] | None":
    """Parse '{race}_{n}_vs_{m}' → (race, n_agents, n_enemies) or None."""
    parts = map_name.split("_")
    if len(parts) != 4 or parts[2] != "vs":
        return None
    race = parts[0]
    if race not in _SMACV2_RACE_CONFIGS:
        return None
    try:
        return race, int(parts[1]), int(parts[3])
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class StarCraft2v2FastLane_Env(RawMultiAgentEnv if RawMultiAgentEnv is not object else object):
    """SMACv2 wrapper with FastLane frame publishing.

    See module docstring for design rationale and 3D patch note.
    """

    def __init__(self, config: Any) -> None:
        if RawMultiAgentEnv is object:
            raise ImportError(f"xuance.environment unavailable: {_IMPORT_ERROR}")
        if not _SMACV2_AVAILABLE:
            raise ImportError(
                "smacv2 package not installed — "
                "pip install the smacv2 package to use SMACv2 environments"
            )

        super().__init__()

        parsed = _parse_smacv2_map_name(config.env_id)
        if parsed is None:
            raise ValueError(
                f"env_id '{config.env_id}' is not a valid SMACv2 map name. "
                f"Expected format: {{race}}_{{n}}_vs_{{m}}, "
                f"e.g. 'protoss_5_vs_5', 'terran_10_vs_11'."
            )
        _race, _n_units, _n_enemies = parsed
        _race_cfg = _SMACV2_RACE_CONFIGS[_race]

        capability_config = {
            "n_units": _n_units,
            "n_enemies": _n_enemies,
            "team_gen": _race_cfg["team_gen"],
            "start_positions": _SMACV2_START_POSITIONS,
        }
        self.env = _StarCraftCapabilityEnvWrapper(
            capability_config=capability_config,
            map_name=_race_cfg["base_map"],
        )
        _LOGGER.info(
            "StarCraft2v2FastLane_Env: map=%s race=%s %dv%d base_map=%s.",
            config.env_id, _race, _n_units, _n_enemies, _race_cfg["base_map"],
        )

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

        self._smac_render_mode: str = os.getenv("MOSAIC_SMAC_RENDER_MODE", "heatmap")
        self._smac_render_size: int = int(
            os.getenv("MOSAIC_SMAC_RENDER_SIZE", "1024") or "1024"
        )
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
            os.getenv("XUANCE_RUN_ID") or os.getenv("RUN_ID") or "xuance-smacv2"
        )
        self._fastlane_video_mode: str = os.getenv("GYM_GUI_FASTLANE_VIDEO_MODE", "single")
        self._fastlane_grid_limit: int = int(os.getenv("GYM_GUI_FASTLANE_GRID_LIMIT", "4") or "4")
        self._heatmap_renderer: Optional[Any] = None
        # SMACv2 always uses 32x32_flat empty terrain.
        self._playable_area: tuple = (0.0, 0.0, 32.0, 32.0)

        # Patch the INNER smacv2.env.StarCraft2Env's _launch (not the wrapper).
        # SC2 calls self._launch() on the inner env; setting it on the wrapper
        # is a no-op because __getattr__ only intercepts GET, not SET.
        if self._fastlane_active and self._smac_render_mode == "3d":
            self._apply_3d_render_patch()

        try:
            self.env.reset(seed=config.env_seed)
        except Exception:
            self.env.reset()

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
            except Exception as exc:
                _LOGGER.debug("FastLane publish failed (non-fatal): %s", exc)

        return obs_dict, reward_dict, terminated_dict, truncated, info

    def close(self) -> None:
        if self._fastlane_writer is not None:
            try:
                self._fastlane_writer.close()
                try:
                    self._fastlane_writer.unlink()
                except (FileNotFoundError, AttributeError):
                    pass
            except Exception as exc:
                _LOGGER.debug("FastLane writer close failed (non-fatal): %s", exc)
            self._fastlane_writer = None
        try:
            if self.env is not None:
                self.env.close()
        except Exception as exc:
            _LOGGER.debug("SMACv2 env close failed: %s", exc)

    def render(self, mode=None):
        return self._get_render_frame()

    def state(self):
        return self.env.get_state()

    def agent_mask(self) -> dict:
        return {agent: True for agent in self.agents}

    def avail_actions(self) -> dict:
        masks = self.env.get_avail_actions()
        return {key: masks[index] for index, key in enumerate(self.agents)}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _apply_3d_render_patch(self) -> None:
        """Patch the inner smacv2.env.StarCraft2Env's _launch for EGL rendering."""
        try:
            from gym_gui.core.adapters.smac import _patch_launch_for_3d
            _inner = self.env.env  # smacv2.env.StarCraft2Env, not the wrapper
            _patch_launch_for_3d(_inner, render_size=self._smac_render_size)
            _LOGGER.info(
                "StarCraft2v2FastLane_Env: 3D GPU render patch applied "
                "(render_size=%d px square).",
                self._smac_render_size,
            )
        except Exception as exc:
            _LOGGER.warning(
                "StarCraft2v2FastLane_Env: 3D GPU patch failed — "
                "falling back to heatmap: %s",
                exc,
            )
            self._smac_render_mode = "heatmap"

    def _get_render_frame(self) -> Optional[np.ndarray]:
        """Produce an RGB frame from the current SMACv2 state."""
        _inner = self.env.env  # smacv2.env.StarCraft2Env — has _obs and _controller

        if self._smac_render_mode == "3d":
            try:
                from gym_gui.core.adapters.smac import (
                    _center_camera_on_units,
                    _composite_minimap_inset,
                )
            except ImportError:
                _center_camera_on_units = None
                _composite_minimap_inset = None
            try:
                camera_center = (
                    _center_camera_on_units(_inner)
                    if _center_camera_on_units is not None
                    else None
                )
                obs = getattr(_inner, "_obs", None)
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
                                    frame, obs,
                                    self._playable_area,
                                    camera_center,
                                    asset_family="SMAC",
                                )
                            except Exception:
                                pass
                        return frame
            except Exception:
                pass

        # Heatmap fallback — reads env._obs protobuf, fully headless.
        try:
            from gym_gui.rendering.smac_heatmap import SMACHeatmapRenderer, extract_frame_data
            if self._heatmap_renderer is None:
                self._heatmap_renderer = SMACHeatmapRenderer()
            fd = extract_frame_data(
                _inner,
                self._episode_step,
                "smacv2",
                self._playable_area,
            )
            if fd is not None:
                frame = self._heatmap_renderer.render(fd)
                if isinstance(frame, np.ndarray):
                    return frame
        except Exception:
            pass
        return None

    def _publish_fastlane_frame(self, reward: float) -> None:
        now_ns = time.perf_counter_ns()
        if now_ns - self._fastlane_last_emit_ns < self._fastlane_throttle_ns:
            return
        self._fastlane_last_emit_ns = now_ns
        frame = self._get_render_frame()
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
            return
        try:
            metrics = FastLaneMetrics(
                last_reward=float(reward),
                rolling_return=float(self._fastlane_episode_return),
                step_rate_hz=float(1e9 / self._fastlane_throttle_ns) if self._fastlane_throttle_ns > 0 else 0.0,
            ) if FastLaneMetrics is not None else None
            self._fastlane_writer.publish(arr.tobytes(), metrics=metrics)
        except Exception as exc:
            _LOGGER.debug("FastLane write failed (non-fatal): %s", exc)

    def _create_fastlane_writer(self, frame_shape: tuple) -> Optional[Any]:
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
                "FastLane single mode: slot %s already claimed, "
                "this subprocess will not publish",
                self._fastlane_run_id,
            )
            return None
        except Exception as exc:
            _LOGGER.warning("FastLane writer create failed: %s", exc)
            return None

    def _downscale_frame(self, arr: np.ndarray) -> np.ndarray:
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


__all__ = ["StarCraft2v2FastLane_Env"]
