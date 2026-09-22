"""MOSAIC XuanCe site customizations.

This module is auto-imported by Python when present on PYTHONPATH. We use it to
set sane defaults and provide optional hooks without patching the vendored
XuanCe sources.
"""

from __future__ import annotations

import logging as _logging  # Import at module level to ensure it's always available
import os
from pathlib import Path

# --- WANDB defaults -------------------------------------------------------
os.environ.setdefault("WANDB_START_METHOD", "thread")
os.environ.setdefault("WANDB__SERVICE", "disabled")
os.environ.setdefault("WANDB_DISABLE_SERVICE", "true")
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

# --- Gymnasium environment wrapper ----------------------------------------
try:  # pragma: no cover - gym optional
    import gymnasium as gym

    # Import FastLane helpers
    try:
        from xuance_worker.fastlane import is_fastlane_enabled, maybe_wrap_env
    except ImportError:
        from .fastlane import is_fastlane_enabled, maybe_wrap_env

    if hasattr(gym.make, "_mosaic_wrapped"):
        _ORIG_MAKE = getattr(gym.make, "_mosaic_orig_make")
    else:
        _ORIG_MAKE = gym.make

    def _wrapped_make(env_id, *args, **kwargs):
        """Wrap gym.make to support MOSAIC environment variables and FastLane."""
        render_kwargs = dict(kwargs)

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

        # mosaic_multigrid v7 registers environments when envs is imported.
        if isinstance(env_id, str) and env_id.startswith("MosaicMultiGrid"):
            try:
                import mosaic_multigrid.envs  # noqa: F401
            except ImportError:
                pass

        # CRITICAL: Force rgb_array for FastLane telemetry
        # XuanCe defaults to render_mode="human" but FastLane needs rgb_array
        # We must OVERRIDE whatever render_mode was passed, not just add if missing
        if is_fastlane_enabled():
            render_kwargs["render_mode"] = "rgb_array"
            try:
                env = _ORIG_MAKE(env_id, *args, **render_kwargs)
            except TypeError:
                # Fallback if env doesn't support render_mode kwarg
                render_kwargs.pop("render_mode", None)
                env = _ORIG_MAKE(env_id, *args, **render_kwargs)
        else:
            env = _ORIG_MAKE(env_id, *args, **render_kwargs)

        # Wrap MiniGrid/BabyAI environments to flatten Dict observation to Box
        # SKIP for XuanCe - it has its own MiniGridEnv wrapper that handles observation space
        # XuanCe's MiniGridEnv expects raw Dict obs and does its own ImgObsWrapper handling
        _is_xuance = os.getenv("XUANCE_RUN_ID") is not None
        if _is_minigrid_env and env is not None and not _is_xuance:
            try:
                from minigrid.wrappers import ImgObsWrapper
                # ImgObsWrapper extracts just the image (7,7,3), then flatten to (147,)
                env = ImgObsWrapper(env)
                env = gym.wrappers.FlattenObservation(env)
            except Exception:
                pass

        # Wrap with ProceduralGenerationWrapper if requested
        procedural_gen = os.getenv("CLEANRL_PROCEDURAL_GENERATION", "").lower()
        if procedural_gen in ("1", "true", "yes") and env is not None:
            try:
                from cleanrl_worker.wrappers.procedural_generation import ProceduralGenerationWrapper  # type: ignore[import-not-found]
                # Enable procedural generation (different levels each episode)
                env = ProceduralGenerationWrapper(
                    env,
                    procedural=True,
                    fixed_seed=None
                )
            except Exception:
                pass
        elif procedural_gen in ("0", "false", "no") and env is not None:
            try:
                from cleanrl_worker.wrappers.procedural_generation import ProceduralGenerationWrapper  # type: ignore[import-not-found]
                seed_str = os.getenv("CLEANRL_SEED")
                seed = int(seed_str) if seed_str else 42
                env = ProceduralGenerationWrapper(
                    env,
                    procedural=False,
                    fixed_seed=seed
                )
            except Exception:
                pass

        # Wrap with FastLane telemetry (THIS WAS MISSING!)
        env = maybe_wrap_env(env)
        return env

    _wrapped_make._mosaic_wrapped = True  # type: ignore[attr-defined]
    _wrapped_make._mosaic_orig_make = _ORIG_MAKE  # type: ignore[attr-defined]
    gym.make = _wrapped_make
except Exception:  # pragma: no cover - gym optional
    pass

# --- XuanCe FastLane environment wrapping ---------------------------------
# NOTE: XuanCeEnvWrapper patching is DISABLED because importing xuance at
# module load time can hang indefinitely (xuance initialization issue).
# The gym.make() patching above is sufficient for FastLane wrapping since
# XuanCe internally uses gym.make() to create environments.
#
# If you need to re-enable XuanCe wrapper patching, it should be done lazily
# (e.g., only when the wrapper is first instantiated) to avoid blocking imports.
#
# Original code kept for reference:
# try:
#     from xuance.environment.utils import XuanCeEnvWrapper, ...
#     if is_fastlane_enabled():
#         # Patch XuanCeEnvWrapper.__init__ to call maybe_wrap_env()
#         ...
# except Exception:
#     pass

# --- PyTorch save helper ----------------------------------------
try:  # pragma: no cover - torch optional
    import torch

    _ORIG_TORCH_SAVE = torch.save

    def _mosaic_torch_save(obj, f, *args, **kwargs):
        """Ensure parent directories exist before saving."""
        if isinstance(f, (str, Path)):
            Path(f).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        return _ORIG_TORCH_SAVE(obj, f, *args, **kwargs)

    torch.save = _mosaic_torch_save
except Exception:  # pragma: no cover - torch optional
    pass

# --- TensorBoard log redirection ----------------------------------------
try:  # pragma: no cover - tensorboard optional
    from torch.utils import tensorboard as _tb_pkg
    from torch.utils.tensorboard import SummaryWriter as _TorchSummaryWriter
    from torch.utils.tensorboard import writer as _tb_writer_mod

    def _resolve_tensorboard_logdir(root: str, log_dir: str | os.PathLike | None) -> str:
        base = Path(root)
        if not log_dir:
            return str(base)
        leaf = Path(log_dir).name or "events"
        return str(base / leaf)

    class _MosaicSummaryWriter(_TorchSummaryWriter):
        def __init__(self, log_dir=None, *args, **kwargs):
            override_root = os.getenv("XUANCE_TENSORBOARD_DIR")
            if override_root:
                log_dir = _resolve_tensorboard_logdir(override_root, log_dir)
            super().__init__(log_dir=log_dir, *args, **kwargs)

    _tb_writer_mod.SummaryWriter = _MosaicSummaryWriter
    _tb_pkg.SummaryWriter = _MosaicSummaryWriter
except Exception:  # pragma: no cover - tensorboard optional
    pass

# --- XuanCe config override support ----------------------------------------
# NOTE: XuanCe config patching is DISABLED because importing xuance at
# module load time can hang indefinitely (xuance initialization issue).
# Config overrides are now handled via environment variables in the worker
# config (XuanCeWorkerConfig) and passed through parser_args.
#
# Original code kept for reference:
# try:
#     import xuance
#     if hasattr(xuance, "get_config"):
#         # Patch xuance.get_config to apply env var overrides
#         ...
# except Exception:
#     pass

# --- Checkpoint resume support ----------------------------------------
try:  # pragma: no cover - torch optional
    import torch
    import torch.nn as nn

    _MOSAIC_MODULE_TO_PATCHED = getattr(nn.Module.to, "_mosaic_patched", False)

    if not _MOSAIC_MODULE_TO_PATCHED:
        _ORIG_MODULE_TO = nn.Module.to
    else:
        _ORIG_MODULE_TO = getattr(nn.Module, "_orig_to", nn.Module.to)

    _RESUME_CHECKPOINT_LOADED = False

    def _mosaic_module_to(self, *args, **kwargs):
        global _RESUME_CHECKPOINT_LOADED
        result = _ORIG_MODULE_TO(self, *args, **kwargs)

        resume_path = os.getenv("XUANCE_RESUME_PATH")
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

            model_keys = set(self.state_dict().keys())

            # Handle dict format with model_state_dict
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                checkpoint_keys = set(checkpoint["model_state_dict"].keys())
                if model_keys == checkpoint_keys:
                    self.load_state_dict(checkpoint["model_state_dict"])
                    _RESUME_CHECKPOINT_LOADED = True
                    if "global_step" in checkpoint:
                        os.environ["XUANCE_RESUME_GLOBAL_STEP"] = str(checkpoint["global_step"])
                    print(f"[MOSAIC] Resumed from checkpoint: {checkpoint_file}")
            else:
                # Direct state_dict format
                checkpoint_keys = set(checkpoint.keys())
                if model_keys == checkpoint_keys:
                    self.load_state_dict(checkpoint)
                    _RESUME_CHECKPOINT_LOADED = True
                    print(f"[MOSAIC] Resumed from checkpoint: {checkpoint_file}")
        except Exception:
            pass

        return result

    _mosaic_module_to._mosaic_patched = True  # type: ignore[attr-defined]
    nn.Module._orig_to = _ORIG_MODULE_TO  # type: ignore[attr-defined]
    nn.Module.to = _mosaic_module_to
except Exception:  # pragma: no cover - torch optional
    pass
