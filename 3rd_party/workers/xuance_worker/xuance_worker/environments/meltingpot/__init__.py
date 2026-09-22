"""MeltingPot environment family.

Family-per-directory layout. Each substrate lives in its own file under
this directory (``pd.py``, ``paintball_ctf.py``, ...). New substrates or
shared MeltingPot helpers (custom shaping, config builders, common
observation preprocessors) belong here rather than at the flat
``environments/`` level.

Public API mirrors a single-file layout: consumers
``from xuance_worker.environments.meltingpot import MeltingPot_PD_Env,
MeltingPot_PaintballCaptureTheFlag_Env``.
"""

from xuance_worker.environments.meltingpot.paintball_ctf import (
    MeltingPot_PaintballCaptureTheFlag_Env,
)
from xuance_worker.environments.meltingpot.pd import MeltingPot_PD_Env

__all__ = [
    "MeltingPot_PD_Env",
    "MeltingPot_PaintballCaptureTheFlag_Env",
]
