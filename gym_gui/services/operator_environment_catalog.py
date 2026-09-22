"""Environment inventories shared by the operator configuration UIs."""

from __future__ import annotations

import logging
from typing import Tuple

from gym_gui.core.enums import GameId

_LOGGER = logging.getLogger(__name__)


# Keep the fallback inventory in the same order as GameId. GameId is already the
# application-wide source of truth used by the adapter factory and contains the
# complete mosaic_multigrid v7.0.0 inventory.
MOSAIC_MULTIGRID_ENVIRONMENTS: Tuple[str, ...] = tuple(
    game_id.value
    for game_id in GameId
    if game_id.value.startswith("MosaicMultiGrid-")
)


def get_mosaic_multigrid_environments() -> Tuple[str, ...]:
    """Return valid installed mosaic_multigrid v7 environment IDs."""
    try:
        import gymnasium
        import mosaic_multigrid.envs  # noqa: F401 - registers environments

        registered = {
            env_id
            for env_id in gymnasium.registry
            if env_id.startswith("MosaicMultiGrid-")
        }
        if not registered:
            return MOSAIC_MULTIGRID_ENVIRONMENTS

        return tuple(
            env_id
            for env_id in MOSAIC_MULTIGRID_ENVIRONMENTS
            if env_id in registered
        )
    except ImportError:
        _LOGGER.debug("mosaic_multigrid is not installed; using the v7 catalog")
        return MOSAIC_MULTIGRID_ENVIRONMENTS
    except Exception as exc:
        _LOGGER.warning(
            "Could not inspect the mosaic_multigrid registry; using the v7 catalog: %s",
            exc,
        )
        return MOSAIC_MULTIGRID_ENVIRONMENTS


__all__ = [
    "MOSAIC_MULTIGRID_ENVIRONMENTS",
    "get_mosaic_multigrid_environments",
]
