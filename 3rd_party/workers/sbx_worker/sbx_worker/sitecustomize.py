"""MOSAIC SBX site customizations.

This module is auto-imported by Python when present on PYTHONPATH. It patches
gymnasium.make so that stable-baselines3's make_vec_env (which calls gym.make
internally) automatically gets FastLane frame streaming when enabled.
"""

from __future__ import annotations

try:  # pragma: no cover - handle import context differences
    from .fastlane import is_fastlane_enabled, maybe_wrap_env
except ImportError:  # pragma: no cover - fallback when package context missing
    from sbx_worker.fastlane import is_fastlane_enabled, maybe_wrap_env  # type: ignore

try:  # pragma: no cover - gym optional
    import gymnasium as gym

    if hasattr(gym.make, "_mosaic_wrapped"):
        _ORIG_MAKE = getattr(gym.make, "_mosaic_orig_make")
    else:
        _ORIG_MAKE = gym.make

    def _wrapped_make(env_id, *args, **kwargs):
        render_kwargs = dict(kwargs)

        if is_fastlane_enabled() and "render_mode" not in render_kwargs:
            try:
                env = _ORIG_MAKE(env_id, *args, render_mode="rgb_array", **render_kwargs)
            except TypeError:
                env = _ORIG_MAKE(env_id, *args, **render_kwargs)
        else:
            env = _ORIG_MAKE(env_id, *args, **render_kwargs)

        env = maybe_wrap_env(env)
        return env

    _wrapped_make._mosaic_wrapped = True  # type: ignore[attr-defined]
    _wrapped_make._mosaic_orig_make = _ORIG_MAKE  # type: ignore[attr-defined]
    gym.make = _wrapped_make

except Exception:  # pragma: no cover - gym optional
    pass
