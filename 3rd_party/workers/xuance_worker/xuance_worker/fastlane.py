"""FastLane telemetry helpers for the XuanCe worker.

Adapted from cleanrl_worker/cleanrl_worker/fastlane.py
Key changes:
- CLEANRL_NUM_ENVS → XUANCE_PARALLELS (number of vectorized envs)
- CLEANRL_RUN_ID → XUANCE_RUN_ID
- CLEANRL_AGENT_ID → XUANCE_AGENT_ID
- Works with both Gymnasium and XuanCe's environment wrappers
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from itertools import count
from typing import Any, Optional

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym  # type: ignore

try:  # pragma: no cover - relies on repo layout
    from gym_gui.fastlane import FastLaneWriter, FastLaneConfig, FastLaneMetrics
    from gym_gui.fastlane.buffer import create_fastlane_name
    from gym_gui.fastlane.tiling import tile_frames
    from gym_gui.telemetry.semconv import VideoModes, TelemetryEnv
except ImportError:  # pragma: no cover - best effort fallback
    FastLaneWriter = None  # type: ignore
    FastLaneConfig = None  # type: ignore
    FastLaneMetrics = None  # type: ignore
    create_fastlane_name = None  # type: ignore

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


_LOGGER = logging.getLogger(__name__)


def _truthy(value: Any) -> bool:
    """Check if a value is truthy (handles strings like '1', 'true', etc.)."""
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
    """Check if FastLane is enabled via environment variables."""
    return _truthy(os.getenv("GYM_GUI_FASTLANE_ONLY")) or _truthy(os.getenv("MOSAIC_FASTLANE_ENABLED"))


@dataclass(frozen=True)
class _FastLaneConfig:
    """FastLane configuration for XuanCe worker."""
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
    """Resolve FastLane configuration from environment variables.

    Adapted from CleanRL, but uses XuanCe-specific env vars:
    - XUANCE_PARALLELS instead of CLEANRL_NUM_ENVS
    - XUANCE_RUN_ID instead of CLEANRL_RUN_ID
    - etc.
    """
    enabled = is_fastlane_enabled()

    try:
        slot = int(os.getenv("GYM_GUI_FASTLANE_SLOT", "0"))
    except ValueError:
        slot = 0

    # XuanCe uses 'parallels' instead of 'num_envs'.
    # Check both XUANCE_PARALLELS (set by train form) and XUANCE_NUM_ENVS
    # (set by shell scripts) to support grid mode from either launch path.
    try:
        total_envs = max(1, int(
            os.getenv("XUANCE_PARALLELS")
            or os.getenv("XUANCE_NUM_ENVS", "1")
        ))
    except ValueError:
        total_envs = 1

    slot = max(0, min(slot, total_envs - 1))

    # XuanCe-specific environment variables
    run_id = os.getenv("XUANCE_RUN_ID") or os.getenv("RUN_ID") or "xuance-run"
    agent_id = os.getenv("XUANCE_AGENT_ID") or os.getenv("AGENT_ID") or "xuance-agent"
    worker_id = (
        os.getenv("WORKER_ID")
        or os.getenv("XUANCE_WORKER_ID")
        or os.getenv("XUANCE_AGENT_ID")
        or "xuance-worker"
    )

    try:
        seed = int(os.getenv("XUANCE_SEED", "0"))
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


def cleanup_fastlane_buffer(run_id: Optional[str] = None) -> None:
    """Unlink the FastLane shared memory buffer so a new one can be created.

    Call this during curriculum phase transitions when the new environment
    produces frames of a different size (e.g. collect_1vs1 10x10 -> soccer_1vs1
    16x11).  The old buffer is sized for the old frame dimensions; without
    unlinking, new writers fall back to the old buffer and corrupt the rendering.

    POSIX semantics: ``shm_unlink`` removes the name from /dev/shm but existing
    mappings (e.g. the GUI reader) stay valid until they close.  The GUI will
    reconnect when the new buffer appears under the same name.
    """
    target_run_id = run_id or _CONFIG.run_id
    if create_fastlane_name is None:
        return
    name = create_fastlane_name(target_run_id)
    try:
        from multiprocessing import shared_memory as _shm
        shm = _shm.SharedMemory(name=name, create=False)

        # Signal the GUI reader to detach before we unlink the name.
        # The reader's mmap points to the same physical memory, so it will
        # see the flag on the very next poll (≤16 ms).
        try:
            from gym_gui.fastlane.buffer import FastLaneBase
            base = FastLaneBase(shm)
            base.invalidate()
            # Release our memoryview/buffer references so shm.close() works
            base._mv.release()
            base._buffer.release()
            _LOGGER.info("Marked FastLane buffer as invalidated: %s", name)
        except Exception as inv_exc:
            _LOGGER.debug("Could not invalidate buffer header: %s", inv_exc)

        shm.close()
        shm.unlink()
        _LOGGER.info("Unlinked FastLane buffer: %s", name)
    except FileNotFoundError:
        _LOGGER.debug("FastLane buffer not found (already cleaned): %s", name)
    except Exception as exc:
        _LOGGER.warning("Could not unlink FastLane buffer %s: %s", name, exc)

    # Clear stale grid coordinator so it doesn't hold old frame references
    _GRID_COORDINATORS.pop(target_run_id, None)


def reload_fastlane_config() -> _FastLaneConfig:
    """Recompute FastLane config from environment variables.

    Useful for testing or when environment variables change after module import.
    """
    global _CONFIG
    _CONFIG = _resolve_config()
    return _CONFIG


def maybe_wrap_env(env: Any) -> Any:
    """Attach the FastLane wrapper if fastlane streaming is enabled.

    Args:
        env: Environment to potentially wrap (Gymnasium or XuanCe wrapped env)

    Returns:
        Wrapped or original environment

    Note: If FASTLANE_SKIP_WRAP=1 is set, wrapping is skipped. This is used
    by curriculum learning to prevent re-wrapping replacement environments.
    When using curriculum learning (e.g., Syllabus ReinitTaskWrapper), apply
    FastLane wrapper at the OUTERMOST level after all curriculum wrappers.
    """
    global _CONFIG

    _LOGGER.info(
        "maybe_wrap_env called: _CONFIG.enabled=%s is_fastlane_enabled()=%s video_mode=%s",
        _CONFIG.enabled, is_fastlane_enabled(), _CONFIG.video_mode
    )

    # Check env vars dynamically - they may have been set after module import
    # This is important because subprocess env vars are available at runtime
    # but the module may have been imported before they were set
    if not _CONFIG.enabled and is_fastlane_enabled():
        # Env vars are now set, reload config
        _CONFIG = _resolve_config()
        _LOGGER.info("FastLane config reloaded: enabled=%s run_id=%s", _CONFIG.enabled, _CONFIG.run_id)

    if not _CONFIG.enabled or _CONFIG.video_mode == VideoModes.OFF:
        _LOGGER.info("FastLane wrapping SKIPPED: enabled=%s video_mode=%s", _CONFIG.enabled, _CONFIG.video_mode)
        return env

    # Skip wrapping if explicitly disabled (e.g., for replacement envs in curriculum)
    if os.getenv("FASTLANE_SKIP_WRAP") == "1":
        _LOGGER.info("FastLane wrapping SKIPPED: FASTLANE_SKIP_WRAP=1")
        return env

    slot_index = next(_ENV_SLOT_COUNTER)
    _LOGGER.info("FastLane wrapping env with slot_index=%d run_id=%s", slot_index, _CONFIG.run_id)
    return FastLaneTelemetryWrapper(env, _CONFIG, slot_index)


class FastLaneTelemetryWrapper(gym.Wrapper):
    """Wrapper that emits FastLane telemetry on every step.

    Works with both Gymnasium and XuanCe's environment wrappers.
    Inherits from gym.Wrapper to be compatible with standard Gym wrapper chains.
    """

    def __init__(self, env: Any, config: _FastLaneConfig, slot_index: int) -> None:
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

        # XuanCe-specific throttling env var
        # Default to 33ms (~30fps) to prevent CPU starvation from rendering every step
        self._throttle_interval_ns = int(float(os.getenv("XUANCE_FASTLANE_INTERVAL_MS", "33")) * 1e6)
        self._downscale_max_dim = int(os.getenv("XUANCE_FASTLANE_MAX_DIM", "0"))

        self._writer: Optional[Any] = None  # FastLaneWriter when created
        self._last_metrics_ts: Optional[float] = None
        self._debug_counter = 0

        _LOGGER.debug(
            "FastLane wrapper initialized: active=%s slot=%d/%d run=%s",
            self._active, slot_index, config.slot, config.run_id
        )

    def reset(self, *args: Any, **kwargs: Any):
        """Reset environment and FastLane state."""
        result = self.env.reset(*args, **kwargs)
        self._step_index = 0
        self._episode_return = 0.0
        if self._grid_coordinator is not None and self._slot == 0:
            self._grid_coordinator.reset()
        return result

    def step(self, action: Any):
        """Step environment and emit FastLane frame."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Handle both single value and array rewards
        if hasattr(reward, '__iter__') and not isinstance(reward, (str, bytes)):
            reward_f = float(reward[0]) if len(reward) > 0 else 0.0
        else:
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

    def close(self):
        """Close environment and FastLane writer."""
        self._close_writer()
        if self._grid_coordinator is not None and self._slot == 0:
            _GRID_COORDINATORS.pop(self._config.run_id, None)
        return self.env.close()

    def _emit_single_step(self, reward: float) -> None:
        """Emit a single frame for this step."""
        # Check throttle BEFORE rendering to avoid CPU starvation
        if self._should_throttle():
            return
        frame_array = self._grab_frame()
        if frame_array is None:
            if self._debug_counter < 5:
                _LOGGER.debug("FastLane frame skipped (render failed)")
            return
        self._publish_array_frame(frame_array, reward)

    def _handle_grid_step(self, reward: float) -> None:
        """Handle grid mode (multiple envs in one frame)."""
        if self._grid_coordinator is None or not self._grid_contributor:
            return
        # Check throttle BEFORE rendering to avoid CPU starvation
        # Only the active slot (slot 0) checks throttling to coordinate grid timing
        if self._active and self._should_throttle():
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
        self._publish_array_frame(composite, publisher_reward)

    def _emit_episode(self) -> None:
        """Called at episode end (metrics already tracked)."""
        return None

    def _grab_frame(self) -> Optional[np.ndarray]:
        """Grab frame from environment render."""
        try:
            frame = self.env.render()
        except Exception:
            return None

        # Handle list/tuple of frames (vectorized envs)
        if isinstance(frame, (tuple, list)) and frame:
            frame = frame[0]

        if frame is None:
            return None

        # Convert to numpy array
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

        # Ensure 3D (H, W, C)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)
        if arr.ndim != 3:
            return None

        # Ensure uint8
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)

        # Downscale if needed
        if self._downscale_max_dim > 0:
            arr = self._downscale(arr)

        return arr

    def _downscale(self, arr: np.ndarray) -> np.ndarray:
        """Downscale frame if dimensions exceed limit."""
        h, w = arr.shape[:2]
        max_dim = max(h, w)
        if max_dim <= self._downscale_max_dim:
            return arr

        try:
            from PIL import Image
            scale = self._downscale_max_dim / max_dim
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = Image.fromarray(arr)
            # Use Resampling.LANCZOS for PIL 10+ or fallback to LANCZOS constant
            resample = getattr(Image.Resampling, "LANCZOS", getattr(Image, "LANCZOS", 1))
            img = img.resize((new_w, new_h), resample)
            return np.array(img)
        except ImportError:
            return arr

    def _should_throttle(self) -> bool:
        """Check if we should skip this frame due to throttling."""
        if self._throttle_interval_ns <= 0:
            return False
        now_ns = time.perf_counter_ns()
        if now_ns - self._last_emit_ns < self._throttle_interval_ns:
            return True
        self._last_emit_ns = now_ns
        return False

    def _publish_array_frame(self, frame: np.ndarray, reward: float) -> None:
        """Publish frame to FastLane shared memory.

        Uses the same API as cleanrl_worker's fastlane.py for compatibility.
        """
        if FastLaneWriter is None or FastLaneConfig is None or FastLaneMetrics is None:
            return

        height, width, channels = frame.shape
        frame_bytes = frame.tobytes()

        if self._writer is None:
            config = FastLaneConfig(
                width=width,
                height=height,
                channels=channels,
                pixel_format="RGB" if channels == 3 else "RGBA",
            )
            self._writer = self._create_writer(config)
            if self._writer is None:
                return

        # Calculate step rate
        now = time.perf_counter()
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
            self._writer.publish(frame_bytes, metrics=metrics)
            self._debug_counter += 1
            if self._debug_counter % 100 == 0:
                _LOGGER.debug(
                    "FastLane frame emitted: episode=%d step=%d reward=%.2f",
                    self._episode_index, self._step_index, reward
                )
        except Exception as exc:
            if self._debug_counter < 10:
                _LOGGER.warning("Failed to write FastLane frame: %s", exc, exc_info=True)
            self._close_writer()

    def _create_writer(self, config: Any) -> Optional[Any]:
        """Create FastLane writer with proper error handling.

        Matches cleanrl_worker's implementation for compatibility.
        """
        _LOGGER.info("Creating FastLane writer for run_id=%s", self._config.run_id)
        try:
            writer = FastLaneWriter.create(self._config.run_id, config)  # type: ignore[union-attr]
            _LOGGER.info("FastLane writer created successfully for run_id=%s", self._config.run_id)
            return writer
        except FileExistsError:
            _LOGGER.info("FastLane buffer already exists, connecting to existing for run_id=%s", self._config.run_id)
            # Buffer already exists, try to connect to it
            if create_fastlane_name is None:
                return None
            try:
                from multiprocessing import shared_memory
                name = create_fastlane_name(self._config.run_id)
                shm = shared_memory.SharedMemory(name=name, create=False)
                return FastLaneWriter(shm, config)  # type: ignore[call-arg]
            except Exception:
                return None
        except Exception as exc:
            _LOGGER.error("Failed to create FastLane writer: %s", exc, exc_info=True)
            return None

    def _close_writer(self) -> None:
        """Close FastLane writer and unlink shared memory segment."""
        if self._writer is not None:
            try:
                self._writer.close()
                try:
                    self._writer.unlink()
                except (FileNotFoundError, AttributeError):
                    pass
            except Exception:
                pass
            self._writer = None

    # Note: __getattr__ removed - gym.Wrapper handles attribute proxying


# Grid coordinator for tiling multiple environment views
class _GridCoordinator:
    """Coordinates frame collection from multiple envs for grid view."""

    def __init__(self, config: _FastLaneConfig):
        self._config = config
        self._frames: dict[int, np.ndarray] = {}
        self._reward_sum = 0.0
        self._count = 0

    def reset(self) -> None:
        self._frames.clear()
        self._reward_sum = 0.0
        self._count = 0

    def collect(self, slot: int, frame: np.ndarray) -> None:
        self._frames[slot] = frame

    def compose_if_ready(self, reward: float) -> Optional[tuple[np.ndarray, float]]:
        needed = min(self._config.grid_limit, self._config.total_envs)
        if len(self._frames) < needed:
            return None

        frame_list = [self._frames[i] for i in range(needed) if i in self._frames]
        if not frame_list:
            return None

        composite = tile_frames(frame_list)
        self._reward_sum += reward
        self._count += 1
        avg_reward = self._reward_sum / self._count if self._count > 0 else 0.0

        self._frames.clear()
        return composite, avg_reward


def _get_grid_coordinator(config: _FastLaneConfig) -> _GridCoordinator:
    """Get or create grid coordinator for a run."""
    if config.run_id not in _GRID_COORDINATORS:
        _GRID_COORDINATORS[config.run_id] = _GridCoordinator(config)
    return _GRID_COORDINATORS[config.run_id]


__all__ = [
    "is_fastlane_enabled",
    "maybe_wrap_env",
    "reset_slot_counter",
    "cleanup_fastlane_buffer",
    "reload_fastlane_config",
    "FastLaneTelemetryWrapper",
]
