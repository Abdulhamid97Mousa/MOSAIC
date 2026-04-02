"""FastLane telemetry helpers for the CleanRL worker."""

from __future__ import annotations

import logging
import os
import time
import sys
from dataclasses import dataclass
from itertools import count
from multiprocessing import shared_memory
from time import perf_counter
from typing import Any, Optional

import numpy as np
import gymnasium as gym

_LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - relies on repo layout
    from gym_gui.fastlane import FastLaneWriter, FastLaneConfig, FastLaneMetrics
    from gym_gui.fastlane.buffer import create_fastlane_name
    from gym_gui.fastlane.tiling import tile_frames
    from gym_gui.telemetry.semconv import VideoModes, TelemetryEnv
    from gym_gui.logging_config.log_constants import (
        LOG_FASTLANE_WRITER_CLOSED,
        LOG_FASTLANE_SHM_STALE_UNLINKED,
        LOG_FASTLANE_SHM_UNLINK_FAILED,
        LOG_FASTLANE_SHM_LEAK_RISK,
    )
    from gym_gui.logging_config.helpers import log_constant
    _HAS_LOG_CONSTANTS = True
except ImportError:  # pragma: no cover - best effort fallback
    FastLaneWriter = None  # type: ignore
    FastLaneConfig = None  # type: ignore
    FastLaneMetrics = None  # type: ignore
    create_fastlane_name = None  # type: ignore
    _HAS_LOG_CONSTANTS = False
    log_constant = None  # type: ignore

    class _VideoModes:
        SINGLE = "single"
        GRID = "grid"
        OFF = "off"

    class _TelemetryEnv:
        FASTLANE_ONLY = "GYM_GUI_FASTLANE_ONLY"
        FASTLANE_SLOT = "GYM_GUI_FASTLANE_SLOT"
        FASTLANE_VIDEO_MODE = "GYM_GUI_FASTLANE_VIDEO_MODE"
        FASTLANE_GRID_LIMIT = "GYM_GUI_FASTLANE_GRID_LIMIT"

    VideoModes = _VideoModes()  # type: ignore
    TelemetryEnv = _TelemetryEnv()  # type: ignore

    def tile_frames(frames):
        if not frames:
            raise ValueError("tile_frames requires frames")
        return frames[0]


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return False


def is_fastlane_enabled() -> bool:
    return _truthy(os.getenv("GYM_GUI_FASTLANE_ONLY"))


@dataclass(frozen=True)
class _FastLaneConfig:
    enabled: bool
    slot: int
    run_id: str
    agent_id: str
    worker_id: str
    seed: int
    total_envs: int
    video_mode: str
    grid_limit: int


def _resolve_config() -> _FastLaneConfig:
    enabled = is_fastlane_enabled()
    try:
        slot = int(os.getenv("GYM_GUI_FASTLANE_SLOT", "0"))
    except ValueError:
        slot = 0
    try:
        total_envs = max(1, int(os.getenv("CLEANRL_NUM_ENVS", "1")))
    except ValueError:
        total_envs = 1
    slot = max(0, min(slot, total_envs - 1))
    run_id = os.getenv("CLEANRL_RUN_ID") or os.getenv("RUN_ID") or "cleanrl-run"
    agent_id = os.getenv("CLEANRL_AGENT_ID") or os.getenv("AGENT_ID") or "cleanrl-agent"
    worker_id = (
        os.getenv("WORKER_ID")
        or os.getenv("CLEANRL_WORKER_ID")
        or os.getenv("CLEANRL_AGENT_ID")
        or "cleanrl-worker"
    )
    try:
        seed = int(os.getenv("CLEANRL_SEED", "0"))
    except ValueError:
        seed = 0
    video_mode = os.getenv(TelemetryEnv.FASTLANE_VIDEO_MODE, VideoModes.SINGLE)
    if video_mode not in {VideoModes.SINGLE, VideoModes.GRID, VideoModes.OFF}:
        video_mode = VideoModes.SINGLE
    try:
        grid_limit = int(os.getenv(TelemetryEnv.FASTLANE_GRID_LIMIT, "4"))
    except ValueError:
        grid_limit = 4
    grid_limit = max(1, min(grid_limit, total_envs))
    return _FastLaneConfig(
        enabled=enabled,
        slot=slot,
        run_id=run_id,
        agent_id=agent_id,
        worker_id=worker_id,
        seed=seed,
        total_envs=total_envs,
        video_mode=video_mode,
        grid_limit=grid_limit,
    )


_CONFIG = _resolve_config()
_ENV_SLOT_COUNTER = count()
_GRID_COORDINATORS: dict[str, "_GridCoordinator"] = {}


def reset_slot_counter() -> None:
    """Reset the FastLane slot counter.

    Call this at the start of training to ensure slot assignments start from 0.
    This is important for curriculum learning where environments may be recreated
    during task transitions - without resetting, replacement envs get slots
    >= grid_limit and won't contribute to GRID mode rendering.
    """
    global _ENV_SLOT_COUNTER
    _ENV_SLOT_COUNTER = count()


def maybe_wrap_env(env: Any) -> Any:
    """Attach the FastLane wrapper if fastlane streaming is enabled.

    Note: If FASTLANE_SKIP_WRAP=1 is set, wrapping is skipped. This is used
    by curriculum learning to prevent re-wrapping replacement environments.
    """
    if not _CONFIG.enabled or _CONFIG.video_mode == VideoModes.OFF:
        return env

    # Skip wrapping if explicitly disabled (e.g., for replacement envs in curriculum)
    if os.getenv("FASTLANE_SKIP_WRAP") == "1":
        return env

    slot_index = next(_ENV_SLOT_COUNTER)
    return FastLaneTelemetryWrapper(env, _CONFIG, slot_index)


class FastLaneTelemetryWrapper(gym.Wrapper):
    """Gym wrapper that emits FastLane-ready telemetry on every step."""

    def __init__(self, env: gym.Env, config: _FastLaneConfig, slot_index: int) -> None:
        super().__init__(env)
        self._config = config
        self._slot = slot_index
        self._video_mode = config.video_mode
        self._grid_limit = config.grid_limit
        self._grid_coordinator = _get_grid_coordinator(config) if self._video_mode == VideoModes.GRID else None
        if self._video_mode == VideoModes.GRID:
            self._active = config.enabled and slot_index == 0
            self._grid_contributor = slot_index < self._grid_limit
        else:
            self._active = config.enabled and slot_index == config.slot
            self._grid_contributor = False
        self._episode_index = 0
        self._step_index = 0
        self._episode_return = 0.0
        self._last_emit_ns = 0
        self._throttle_interval_ns = int(float(os.getenv("CLEANRL_FASTLANE_INTERVAL_MS", "0")) * 1e6)
        self._downscale_max_dim = int(os.getenv("CLEANRL_FASTLANE_MAX_DIM", "0"))
        self._writer: Optional[FastLaneWriter] = None  # type: ignore[assignment]
        self._last_metrics_ts: Optional[float] = None
        self._debug_counter = 0
        self._log_debug(
            f"FastLane wrapper active={self._active} slot={slot_index}/{config.slot} run={config.run_id}"
        )

    # ------------------------------------------------------------------
    def reset(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        result = self.env.reset(*args, **kwargs)
        self._step_index = 0
        self._episode_return = 0.0
        if self._grid_coordinator is not None and self._slot == 0:
            self._grid_coordinator.reset()
        return result

    def step(self, action: Any):  # type: ignore[override]
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward_f = float(reward)
        if self._video_mode == VideoModes.GRID:
            self._handle_grid_step(reward_f)
        elif self._active:
            self._emit_single_step(reward_f)

        if self._active:
            self._step_index += 1
            self._episode_return += reward_f
            if terminated or truncated:
                self._emit_episode()
                self._episode_index += 1
                self._step_index = 0
                self._episode_return = 0.0
        return obs, reward, terminated, truncated, info

    def close(self):  # type: ignore[override]
        self._close_writer()
        if self._grid_coordinator is not None and self._slot == 0:
            _GRID_COORDINATORS.pop(self._config.run_id, None)
        return super().close()

    # ------------------------------------------------------------------
    def _emit_single_step(self, reward: float) -> None:
        frame_array = self._grab_frame()
        if frame_array is None:
            self._log_debug("FastLane frame skipped (render failed)", limit=5)
            return
        if self._should_throttle():
            return
        self._publish_array_frame(frame_array, reward)

    def _handle_grid_step(self, reward: float) -> None:
        if self._grid_coordinator is None or not self._grid_contributor:
            return
        frame_array = self._grab_frame()
        if frame_array is None:
            return
        self._grid_coordinator.collect(self._slot, frame_array)
        if not self._active:
            return
        result = self._grid_coordinator.compose_if_ready(reward)
        if result is None:
            return
        composite, publisher_reward = result
        if self._should_throttle():
            return
        self._publish_array_frame(composite, publisher_reward)

    def _emit_episode(self) -> None:
        # Metrics stream already tracks totals; no extra emission needed
        return None

    def _grab_frame(self) -> Optional[np.ndarray]:
        try:
            frame = self.env.render()
        except Exception:
            return None
        if isinstance(frame, (tuple, list)) and frame:
            frame = frame[0]
        if frame is None:
            return None
        arr = None
        if isinstance(frame, np.ndarray):
            arr = frame
        elif hasattr(frame, "__array__"):
            try:
                arr = np.asarray(frame)
            except Exception:
                arr = None
        if arr is None:
            return None
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)
        if arr.ndim != 3:
            return None
        if self._downscale_max_dim and max(arr.shape[:2]) > self._downscale_max_dim:
            scale = self._downscale_max_dim / max(arr.shape[:2])
            new_h = max(1, int(arr.shape[0] * scale))
            new_w = max(1, int(arr.shape[1] * scale))
            arr = _resize_nearest(arr, new_h, new_w)
        arr = np.ascontiguousarray(arr.astype(np.uint8, copy=False))
        return arr

    def _should_throttle(self) -> bool:
        now_ns = time.time_ns()
        if self._throttle_interval_ns and self._last_emit_ns:
            if now_ns - self._last_emit_ns < self._throttle_interval_ns:
                return True
        self._last_emit_ns = now_ns
        return False

    def _publish_array_frame(self, frame_array: np.ndarray, reward: float) -> None:
        if FastLaneWriter is None or FastLaneConfig is None or FastLaneMetrics is None:
            return
        height, width, channels = frame_array.shape
        frame_bytes = frame_array.tobytes()
        writer = self._writer
        if writer is None:
            config = FastLaneConfig(
                width=width,
                height=height,
                channels=channels,
                pixel_format="RGB" if channels == 3 else "RGBA",
            )
            writer = self._create_writer(config)
            if writer is None:
                return
            self._writer = writer

        now = perf_counter()
        if self._last_metrics_ts is None:
            step_rate = 0.0
        else:
            delta = now - self._last_metrics_ts
            step_rate = 1.0 / delta if delta > 0 else 0.0
        self._last_metrics_ts = now

        metrics = FastLaneMetrics(
            last_reward=float(reward),
            rolling_return=float(self._episode_return + reward),
            step_rate_hz=step_rate,
        )
        try:
            writer.publish(frame_bytes, metrics=metrics)
        except Exception:
            self._close_writer()
        else:
            self._log_debug("FastLane frame published", limit=10)

    def _create_writer(self, config: Any) -> Optional[FastLaneWriter]:  # type: ignore[name-defined]
        try:
            return FastLaneWriter.create(self._config.run_id, config)  # type: ignore[union-attr]
        except FileExistsError:
            if create_fastlane_name is None:
                return None
            try:
                name = create_fastlane_name(self._config.run_id)
                # Unlink stale segment from a previous crashed run, then recreate
                try:
                    stale = shared_memory.SharedMemory(name=name, create=False)
                    stale.close()
                    stale.unlink()
                    self._log_structured(
                        LOG_FASTLANE_SHM_STALE_UNLINKED if _HAS_LOG_CONSTANTS else None,
                        f"Unlinked stale shm segment '{name}' from previous run",
                        level="warning",
                    )
                except FileNotFoundError:
                    pass
                return FastLaneWriter.create(self._config.run_id, config)  # type: ignore[union-attr]
            except Exception:
                self._log_structured(
                    LOG_FASTLANE_SHM_UNLINK_FAILED if _HAS_LOG_CONSTANTS else None,
                    f"Failed to recover stale shm for run={self._config.run_id}",
                    level="error",
                )
                return None

    def _close_writer(self) -> None:
        writer = self._writer
        if writer is None:
            return
        try:
            writer.close()
            writer.unlink()
        except FileNotFoundError:
            pass  # already unlinked
        except Exception:
            self._log_structured(
                LOG_FASTLANE_SHM_UNLINK_FAILED if _HAS_LOG_CONSTANTS else None,
                f"Failed to unlink shm for run={self._config.run_id} — "
                "orphaned /dev/shm segment may cause resource_tracker memory leak",
                level="error",
            )
        finally:
            self._writer = None
            self._log_structured(
                LOG_FASTLANE_WRITER_CLOSED if _HAS_LOG_CONSTANTS else None,
                "FastLane writer closed and unlinked",
            )

    def _log_debug(self, message: str, *, limit: int | None = None) -> None:
        if limit is not None and self._debug_counter >= limit:
            return
        self._debug_counter += 1
        _LOGGER.debug("[FASTLANE] %s", message)

    def _log_structured(
        self,
        constant: Any,
        fallback_msg: str,
        *,
        level: str = "info",
    ) -> None:
        """Log via log_constant when available, else fall back to stdlib."""
        extra = {"run_id": self._config.run_id, "slot": self._slot}
        if _HAS_LOG_CONSTANTS and constant is not None and log_constant is not None:
            log_constant(_LOGGER, constant, extra=extra)
        else:
            getattr(_LOGGER, level, _LOGGER.info)("[FASTLANE] %s", fallback_msg)


class _GridCoordinator:
    """Accumulates per-env frames and emits a composite when all arrive."""

    def __init__(self, grid_limit: int) -> None:
        self._grid_limit = max(1, grid_limit)
        self._frames: dict[int, np.ndarray] = {}

    def collect(self, slot: int, frame: np.ndarray) -> None:
        if slot >= self._grid_limit:
            return
        self._frames[slot] = frame

    def compose_if_ready(self, reward: float) -> Optional[tuple[np.ndarray, float]]:
        if len(self._frames) < self._grid_limit:
            return None
        ordered: list[np.ndarray] = []
        for idx in range(self._grid_limit):
            frame = self._frames.get(idx)
            if frame is None:
                return None
            ordered.append(frame)
        self._frames.clear()
        composite = tile_frames(ordered)
        return composite, reward

    def reset(self) -> None:
        self._frames.clear()

    @property
    def grid_limit(self) -> int:
        return self._grid_limit


def _get_grid_coordinator(config: _FastLaneConfig) -> _GridCoordinator:
    coord = _GRID_COORDINATORS.get(config.run_id)
    if coord is None or coord.grid_limit != config.grid_limit:
        coord = _GridCoordinator(config.grid_limit)
        _GRID_COORDINATORS[config.run_id] = coord
    return coord


__all__ = ["is_fastlane_enabled", "maybe_wrap_env"]


def reload_fastlane_config() -> _FastLaneConfig:
    """Recompute FastLane config from environment variables."""

    global _CONFIG
    _CONFIG = _resolve_config()
    return _CONFIG


__all__.append("reload_fastlane_config")


def _resize_nearest(arr: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Resize an HWC image via nearest-neighbour without pulling heavy deps."""

    h, w, c = arr.shape
    row_idx = (np.linspace(0, h - 1, new_h)).astype(np.int64)
    col_idx = (np.linspace(0, w - 1, new_w)).astype(np.int64)
    return arr[row_idx][:, col_idx]
