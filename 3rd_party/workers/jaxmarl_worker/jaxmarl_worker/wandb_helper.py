"""Optional WandB integration for jaxmarl_worker training scans.

All calls are safe no-ops if wandb is not installed OR
JAXMARL_WANDB_ENABLED != '1', so scan files can call these unconditionally
without needing to gate on a flag themselves.

Env vars consumed (set by the training form / trainer daemon):
  JAXMARL_WANDB_ENABLED    '1' to enable, anything else to disable
  JAXMARL_WANDB_PROJECT    project name (default: 'jaxmarl')
  JAXMARL_WANDB_ENTITY     team/user name (optional)
  JAXMARL_WANDB_RUN_NAME   optional run name override (defaults to run_id)
  WANDB_API_KEY            standard wandb auth (form or shell can set)
  HTTP_PROXY / HTTPS_PROXY standard proxy env vars (form can set)

Usage from a scan file:

    from jaxmarl_worker.wandb_helper import init_wandb, log_wandb, finish_wandb

    def main():
        args = parse_args()
        init_wandb(run_id=args.run_id or 'jaxmarl_run', config=vars(args))
        try:
            # ... training loop ...
            log_wandb({'reward': r, 'loss': l}, step=step)
        finally:
            finish_wandb()
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

_LOGGER = logging.getLogger("jaxmarl_worker.wandb_helper")

# Module-level singleton for the active run.  wandb itself has a global run
# concept, but we track our own reference for the no-op-when-not-inited path.
_wandb_run: Any = None


def is_wandb_enabled() -> bool:
    """Return True iff env flag is set AND wandb package is importable."""
    if os.environ.get("JAXMARL_WANDB_ENABLED") != "1":
        return False
    try:
        import wandb  # noqa: F401
    except ImportError:
        _LOGGER.warning(
            "JAXMARL_WANDB_ENABLED=1 but wandb package not installed; disabling."
        )
        return False
    return True


def init_wandb(
    run_id: str,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
) -> bool:
    """Initialize WandB if enabled.

    Returns:
        True if initialization actually happened, False if disabled or failed.
    """
    global _wandb_run
    if not is_wandb_enabled():
        return False
    try:
        import wandb
    except ImportError:
        return False

    project = os.environ.get("JAXMARL_WANDB_PROJECT") or "jaxmarl"
    entity = os.environ.get("JAXMARL_WANDB_ENTITY") or None
    name = os.environ.get("JAXMARL_WANDB_RUN_NAME") or run_id

    try:
        _wandb_run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=config or {},
            tags=tags,
            reinit=True,
        )
        _LOGGER.info(
            "wandb run initialized: project=%s entity=%s name=%s",
            project, entity, name,
        )
        return True
    except Exception as exc:  # pragma: no cover
        _LOGGER.warning("wandb.init failed: %s", exc)
        _wandb_run = None
        return False


def log_wandb(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log a metrics dict. No-op if wandb was never initialized."""
    if _wandb_run is None:
        return
    try:
        import wandb
        if step is not None:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)
    except Exception as exc:  # pragma: no cover
        _LOGGER.warning("wandb.log failed: %s", exc)


def finish_wandb() -> None:
    """Close the wandb run. No-op if never initialized."""
    global _wandb_run
    if _wandb_run is None:
        return
    try:
        import wandb
        wandb.finish()
    except Exception as exc:  # pragma: no cover
        _LOGGER.warning("wandb.finish failed: %s", exc)
    finally:
        _wandb_run = None


def _reset_for_tests() -> None:
    """Test helper: force _wandb_run back to None between tests."""
    global _wandb_run
    _wandb_run = None


__all__ = [
    "is_wandb_enabled",
    "init_wandb",
    "log_wandb",
    "finish_wandb",
    "_reset_for_tests",
]
