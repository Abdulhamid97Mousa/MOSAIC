"""MOSAIC JaxMARL site customizations.

Auto-imported by Python when present on PYTHONPATH. Provides sane defaults
and hooks without patching upstream JaxMARL sources.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- WandB defaults -------------------------------------------------------
os.environ.setdefault("WANDB_START_METHOD", "thread")
os.environ.setdefault("WANDB__SERVICE", "disabled")
os.environ.setdefault("WANDB_DISABLE_SERVICE", "true")
os.environ.setdefault("WANDB_MODE", os.environ.get("WANDB_MODE", "offline"))

try:
    import wandb

    _ORIG_INIT = wandb.init

    def _patched_init(*args, **kwargs):
        kwargs.setdefault("reinit", True)
        for env_key, kwarg_key in [
            ("WANDB_PROJECT", "project"),
            ("JAXMARL_WANDB_PROJECT", "project"),
            ("WANDB_ENTITY", "entity"),
            ("WANDB_NAME", "name"),
        ]:
            if kwarg_key not in kwargs:
                val = os.getenv(env_key)
                if val:
                    kwargs[kwarg_key] = val
        return _ORIG_INIT(*args, **kwargs)

    wandb.init = _patched_init
except Exception:
    pass

# --- NumPy savez parent-dir creation --------------------------------------
try:
    import numpy as np

    _ORIG_SAVEZ = np.savez

    def _mosaic_savez(file, *args, **kwargs):
        if isinstance(file, (str, Path)):
            Path(file).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        return _ORIG_SAVEZ(file, *args, **kwargs)

    np.savez = _mosaic_savez

    _ORIG_SAVEZ_COMPRESSED = np.savez_compressed

    def _mosaic_savez_compressed(file, *args, **kwargs):
        if isinstance(file, (str, Path)):
            Path(file).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        return _ORIG_SAVEZ_COMPRESSED(file, *args, **kwargs)

    np.savez_compressed = _mosaic_savez_compressed
except Exception:
    pass

# --- TensorBoard log redirection ------------------------------------------
try:
    from torch.utils import tensorboard as _tb_pkg
    from torch.utils.tensorboard import SummaryWriter as _TorchSummaryWriter
    from torch.utils.tensorboard import writer as _tb_writer_mod

    def _resolve_logdir(root: str, log_dir) -> str:
        base = Path(root)
        if not log_dir:
            return str(base)
        leaf = Path(log_dir).name or "events"
        return str(base / leaf)

    class _MosaicSummaryWriter(_TorchSummaryWriter):
        def __init__(self, log_dir=None, *args, **kwargs):
            override = os.getenv("JAXMARL_TENSORBOARD_DIR")
            if override:
                log_dir = _resolve_logdir(override, log_dir)
            super().__init__(log_dir=log_dir, *args, **kwargs)

    _tb_writer_mod.SummaryWriter = _MosaicSummaryWriter
    _tb_pkg.SummaryWriter = _MosaicSummaryWriter
except Exception:
    pass
