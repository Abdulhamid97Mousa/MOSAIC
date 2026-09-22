"""MOSAIC multigrid environment family.

Family-per-directory layout. Add new multigrid variants (e.g. alternative
observation encoders, curriculum helpers, reward-shaping wrappers) under
this directory rather than at the flat ``environments/`` level.

The underlying upstream package (installed via pip from
``3rd_party/environments/mosaic_multigrid/``) is ``multigrid_sports``.
This module is our XuanCe-facing wrapper on top of it; the file was
renamed away from ``multigrid_sports.py`` to avoid a name collision with
the upstream package in ``import`` statements.

Public API mirrors a single-file layout: consumers
``from xuance_worker.environments.multigrid import MultiGrid_Env``.
"""

from xuance_worker.environments.multigrid.multigrid import (
    GymToGymnasiumWrapper,
    MultiGrid_Env,
    SoloMultiGrid_Env,
    TrainingMode,
)

__all__ = [
    "GymToGymnasiumWrapper",
    "MultiGrid_Env",
    "SoloMultiGrid_Env",
    "TrainingMode",
]
