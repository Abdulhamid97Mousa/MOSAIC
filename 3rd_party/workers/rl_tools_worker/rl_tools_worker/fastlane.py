"""FastLane telemetry helpers for the RLtools worker.

Reuses CleanRL's FastLaneTelemetryWrapper to provide real-time frame
streaming for RLtools training runs.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from itertools import count
from typing import Any

_LOGGER = logging.getLogger(__name__)

try:
    from cleanrl_worker.fastlane import FastLaneTelemetryWrapper
except ImportError:
    FastLaneTelemetryWrapper = None  # type: ignore[assignment, misc]

try:
    from gym_gui.telemetry.semconv import VideoModes, TelemetryEnv
except ImportError:

    class _VideoModes:
        SINGLE = "single"
        GRID = "grid"
        OFF = "off"

    class _TelemetryEnv:
        FASTLANE_ONLY = "GYM_GUI_FASTLANE_ONLY"
        FASTLANE_SLOT = "GYM_GUI_FASTLANE_SLOT"
        FASTLANE_VIDEO_MODE = "GYM_GUI_FASTLANE_VIDEO_MODE"
        FASTLANE_GRID_LIMIT = "GYM_GUI_FASTLANE_GRID_LIMIT"

    VideoModes = _VideoModes()  # type: ignore[assignment]
    TelemetryEnv = _TelemetryEnv()  # type: ignore[assignment]


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
    """Return True when the orchestrator requests FastLane streaming."""
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
        total_envs = max(1, int(os.getenv("RLTOOLS_NUM_ENVS", os.getenv("NUM_ENVS", "1"))))
    except ValueError:
        total_envs = 1

    slot = max(0, min(slot, total_envs - 1))

    run_id = os.getenv("RLTOOLS_RUN_ID") or os.getenv("RUN_ID") or "rltools-run"
    agent_id = os.getenv("RLTOOLS_AGENT_ID") or os.getenv("AGENT_ID") or "rltools-agent"
    worker_id = (
        os.getenv("WORKER_ID")
        or os.getenv("RLTOOLS_WORKER_ID")
        or "rltools-worker"
    )

    try:
        seed = int(os.getenv("RLTOOLS_SEED", os.getenv("SEED", "0")))
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


def reset_slot_counter() -> None:
    """Reset the slot counter so new environments start from slot 0."""
    global _ENV_SLOT_COUNTER
    _ENV_SLOT_COUNTER = count()


def maybe_wrap_env(env: Any) -> Any:
    """Wrap *env* with FastLaneTelemetryWrapper when streaming is enabled."""
    if FastLaneTelemetryWrapper is None:
        return env
    if not _CONFIG.enabled or _CONFIG.video_mode == VideoModes.OFF:
        return env

    slot_index = next(_ENV_SLOT_COUNTER)
    try:
        return FastLaneTelemetryWrapper(env, _CONFIG, slot_index)
    except Exception:
        _LOGGER.debug("[RLTOOLS-FASTLANE] Failed to wrap env at slot %d", slot_index)
        return env


__all__ = ["is_fastlane_enabled", "maybe_wrap_env"]
