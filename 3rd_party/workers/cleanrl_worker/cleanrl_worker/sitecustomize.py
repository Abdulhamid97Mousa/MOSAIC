"""MOSAIC CleanRL site customizations.

This module is auto-imported by Python when present on PYTHONPATH. We use it to
set sane WANDB defaults and provide optional video hooks without patching the
vendored CleanRL sources.
"""

from __future__ import annotations

import os
from pathlib import Path

try:  # pragma: no cover - handle import context differences
    from .fastlane import is_fastlane_enabled, maybe_wrap_env  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - fallback when package context missing
    from cleanrl_worker.fastlane import (  # type: ignore
        is_fastlane_enabled,
        maybe_wrap_env,
    )

# --- WANDB defaults -------------------------------------------------------
os.environ.setdefault("WANDB_START_METHOD", "thread")
os.environ.setdefault("WANDB__SERVICE", "disabled")
os.environ.setdefault("WANDB_DISABLE_SERVICE", "true")
os.environ.setdefault("WANDB_DISABLE_GYM", "true")
os.environ.setdefault("WANDB_MODE", os.environ.get("WANDB_MODE", "offline"))

try:  # pragma: no cover - wandb optional
    import wandb

    _ORIG_INIT = wandb.init

    def _patched_init(*args, **kwargs):
        kwargs.setdefault("reinit", True)
        if "project" not in kwargs:
            default_project = os.getenv("WANDB_PROJECT") or os.getenv("WANDB_PROJECT_NAME")
            if default_project:
                kwargs["project"] = default_project
        if "entity" not in kwargs:
            default_entity = os.getenv("WANDB_ENTITY") or os.getenv("WANDB_ENTITY_NAME")
            if default_entity:
                kwargs["entity"] = default_entity
        if "name" not in kwargs:
            default_name = os.getenv("WANDB_NAME")
            if default_name:
                kwargs["name"] = default_name
        return _ORIG_INIT(*args, **kwargs)

    wandb.init = _patched_init
except Exception:  # pragma: no cover - wandb optional
    pass

# --- Gymnasium RecordVideo helper ----------------------------------------

try:  # pragma: no cover - gym optional
    import gymnasium as gym
    from gymnasium.wrappers import RecordVideo

    if hasattr(gym.make, "_mosaic_wrapped"):
        _ORIG_MAKE = getattr(gym.make, "_mosaic_orig_make")
    else:
        _ORIG_MAKE = gym.make
    _RGB_MODES = {"rgb_array", "rgb_array_list"}

    if not hasattr(RecordVideo, "enabled"):

        def _get_enabled(self):
            return getattr(self, "_mosaic_recording_enabled", True)

        def _set_enabled(self, value):
            self._mosaic_recording_enabled = bool(value)

        RecordVideo.enabled = property(_get_enabled, _set_enabled)  # type: ignore[attr-defined]

    if not hasattr(RecordVideo, "path"):
        _ORIG_STOP_RECORDING = RecordVideo.stop_recording

        def _wrapped_stop_recording(self):
            video_folder = getattr(self, "video_folder", None)
            video_name = getattr(self, "_video_name", None)
            result = _ORIG_STOP_RECORDING(self)
            if video_folder and video_name:
                candidate = os.path.join(video_folder, f"{video_name}.mp4")
                self._mosaic_video_path = candidate
            return result

        def _get_path(self):
            return getattr(self, "_mosaic_video_path", None)

        def _set_path(self, value):
            self._mosaic_video_path = value

        RecordVideo.stop_recording = _wrapped_stop_recording
        RecordVideo.path = property(_get_path, _set_path)  # type: ignore[attr-defined]

    def _wrapped_make(env_id, *args, **kwargs):
        # Auto-import environment packages to register their environments
        _is_minigrid_env = False
        if isinstance(env_id, str):
            if env_id.startswith("MiniGrid") or env_id.startswith("BabyAI"):
                _is_minigrid_env = True
                try:
                    import minigrid  # noqa: F401
                    if hasattr(minigrid, "register_minigrid_envs"):
                        minigrid.register_minigrid_envs()
                except ImportError:
                    pass

        render_kwargs = dict(kwargs)
        env = None
        if is_fastlane_enabled() and "render_mode" not in render_kwargs:
            try:
                env = _ORIG_MAKE(env_id, *args, render_mode="rgb_array", **render_kwargs)
            except TypeError:
                env = _ORIG_MAKE(env_id, *args, **render_kwargs)
        else:
            env = _ORIG_MAKE(env_id, *args, **render_kwargs)

        # Wrap with ProceduralGenerationWrapper if requested
        procedural_gen = os.getenv("CLEANRL_PROCEDURAL_GENERATION", "").lower()
        if procedural_gen in ("1", "true", "yes") and env is not None:
            try:
                from cleanrl_worker.wrappers.procedural_generation import ProceduralGenerationWrapper
                # Enable procedural generation (different levels each episode)
                # Don't pass fixed_seed - let each parallel env get its own random RNG
                # This ensures different layouts across parallel environments
                env = ProceduralGenerationWrapper(
                    env,
                    procedural=True,
                    fixed_seed=None
                )
            except Exception:
                pass
        elif procedural_gen in ("0", "false", "no") and env is not None:
            try:
                from cleanrl_worker.wrappers.procedural_generation import ProceduralGenerationWrapper
                # Get seed from environment variable
                seed_str = os.getenv("CLEANRL_SEED")
                seed = int(seed_str) if seed_str else 42  # Default seed for fixed mode

                # Disable procedural generation (same level each episode)
                env = ProceduralGenerationWrapper(
                    env,
                    procedural=False,
                    fixed_seed=seed
                )
            except Exception:
                pass

        if os.getenv("MOSAIC_CAPTURE_VIDEO") == "1" and not isinstance(env, RecordVideo):
            render_mode = getattr(env, "render_mode", None)
            if render_mode in _RGB_MODES:
                video_dir = os.getenv("MOSAIC_VIDEOS_DIR", "videos")
                try:
                    env = RecordVideo(env, video_folder=video_dir)
                except Exception:
                    pass

        env = maybe_wrap_env(env)
        return env

    _wrapped_make._mosaic_wrapped = True  # type: ignore[attr-defined]
    _wrapped_make._mosaic_orig_make = _ORIG_MAKE  # type: ignore[attr-defined]
    gym.make = _wrapped_make

except Exception as e:  # pragma: no cover - gym optional
    pass

try:  # pragma: no cover - gym optional
    import gymnasium as gym
    from gymnasium.wrappers import TransformObservation as _OrigTransformObservation

    class _MosaicTransformObservation(_OrigTransformObservation):
        def __init__(self, env, func, observation_space=None, **kwargs):
            obs_space = observation_space or getattr(env, "observation_space", None)
            if obs_space is None:
                raise ValueError("TransformObservation requires an observation_space")
            super().__init__(env, func, observation_space=obs_space, **kwargs)

    gym.wrappers.TransformObservation = _MosaicTransformObservation
except Exception:  # pragma: no cover - gym optional
    pass

try:  # pragma: no cover - torch optional
    import torch

    _ORIG_TORCH_SAVE = torch.save

    def _mosaic_torch_save(obj, f, *args, **kwargs):
        if isinstance(f, (str, Path)):
            Path(f).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        return _ORIG_TORCH_SAVE(obj, f, *args, **kwargs)

    torch.save = _mosaic_torch_save
except Exception:  # pragma: no cover - torch optional
    pass

# --- Checkpoint resume auto-loading ---------------------------------------
# Patches torch.nn.Module.to() to automatically load checkpoint weights when
# CLEANRL_RESUME_PATH is set. Works for any algorithm transparently.
try:  # pragma: no cover - torch optional
    import torch
    import torch.nn as nn

    # Guard against double-patching on module reload
    _MOSAIC_MODULE_TO_PATCHED = getattr(nn.Module.to, "_mosaic_patched", False)

    if not _MOSAIC_MODULE_TO_PATCHED:
        _ORIG_MODULE_TO = nn.Module.to
    else:
        # Already patched - _ORIG_MODULE_TO should exist from prior import
        _ORIG_MODULE_TO = getattr(nn.Module, "_orig_to", nn.Module.to)

    _RESUME_CHECKPOINT_LOADED = False

    def _mosaic_module_to(self, *args, **kwargs):
        global _RESUME_CHECKPOINT_LOADED
        result = _ORIG_MODULE_TO(self, *args, **kwargs)

        resume_path = os.getenv("CLEANRL_RESUME_PATH")
        if not resume_path or _RESUME_CHECKPOINT_LOADED:
            return result

        checkpoint_file = Path(resume_path).expanduser().resolve()
        if not checkpoint_file.exists():
            return result

        # Determine device from args
        device = None
        if args:
            device = args[0]
        elif "device" in kwargs:
            device = kwargs["device"]

        try:
            checkpoint = torch.load(
                checkpoint_file,
                map_location=device,
                weights_only=True,
            )

            # Handle both old format (just state_dict) and new format (dict with model_state_dict)
            model_keys = set(self.state_dict().keys())

            # New format: {"model_state_dict": ..., "global_step": ..., "iteration": ...}
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                checkpoint_keys = set(checkpoint["model_state_dict"].keys())
                if model_keys == checkpoint_keys:
                    self.load_state_dict(checkpoint["model_state_dict"])
                    _RESUME_CHECKPOINT_LOADED = True
                    # Set env var for global_step restoration
                    if "global_step" in checkpoint:
                        os.environ["CLEANRL_RESUME_GLOBAL_STEP"] = str(checkpoint["global_step"])
                    print(f"[MOSAIC] Resumed from checkpoint: {checkpoint_file}")
                    if "global_step" in checkpoint:
                        print(f"[MOSAIC] Resume global_step: {checkpoint['global_step']}")
            else:
                # Old format: just the state_dict directly
                checkpoint_keys = set(checkpoint.keys())
                if model_keys == checkpoint_keys:
                    self.load_state_dict(checkpoint)
                    _RESUME_CHECKPOINT_LOADED = True
                    print(f"[MOSAIC] Resumed from checkpoint: {checkpoint_file}")
        except Exception as exc:
            # Silently skip if checkpoint doesn't match this module
            pass

        return result

    # Apply patch and mark as patched to prevent recursion on reload
    _mosaic_module_to._mosaic_patched = True  # type: ignore[attr-defined]
    nn.Module._orig_to = _ORIG_MODULE_TO  # type: ignore[attr-defined]  # Store original for reload access
    nn.Module.to = _mosaic_module_to  # type: ignore[assignment]
except Exception:  # pragma: no cover - torch optional
    pass

try:  # pragma: no cover - cleanrl optional
    import os
    from cleanrl_utils.evals import ppo_eval as _ppo_eval

    _ORIG_PPO_EVALUATE = _ppo_eval.evaluate

    def _mosaic_ppo_evaluate(model_path, make_env, env_id, eval_episodes, run_name, Model, *, device: "str | torch.device" = "cpu", capture_video=True, gamma=0.99):
        override = os.getenv("MOSAIC_CLEANRL_EVAL_EPISODES")
        if override is not None:
            try:
                eval_episodes = int(override)
            except ValueError:
                pass
        return _ORIG_PPO_EVALUATE(
            model_path,
            make_env,
            env_id,
            eval_episodes,
            run_name,
            Model,
            device=device,
            capture_video=capture_video,
            gamma=gamma,
        )

    _ppo_eval.evaluate = _mosaic_ppo_evaluate
except Exception:  # pragma: no cover - cleanrl optional
    pass

# --- TensorBoard log redirection ----------------------------------------
try:  # pragma: no cover - tensorboard optional
    from torch.utils import tensorboard as _tb_pkg
    from torch.utils.tensorboard import SummaryWriter as _TorchSummaryWriter
    from torch.utils.tensorboard import writer as _tb_writer_mod

    def _resolve_tensorboard_logdir(root: str, log_dir: str | os.PathLike | None) -> str:
        base = Path(root).resolve()
        if not log_dir:
            return str(base)
        resolved = Path(log_dir).resolve()
        # If log_dir is already under the override root, don't redirect again
        try:
            resolved.relative_to(base)
            return str(resolved)
        except ValueError:
            pass
        leaf = Path(log_dir).name or "events"
        return str(base / leaf)

    class _MosaicSummaryWriter(_TorchSummaryWriter):
        def __init__(self, log_dir=None, *args, **kwargs):
            override_root = os.getenv("CLEANRL_TENSORBOARD_DIR")
            if override_root:
                log_dir = _resolve_tensorboard_logdir(override_root, log_dir)
            super().__init__(log_dir=log_dir, *args, **kwargs)

    _tb_writer_mod.SummaryWriter = _MosaicSummaryWriter
    _tb_pkg.SummaryWriter = _MosaicSummaryWriter
except Exception:  # pragma: no cover - tensorboard optional
    pass
