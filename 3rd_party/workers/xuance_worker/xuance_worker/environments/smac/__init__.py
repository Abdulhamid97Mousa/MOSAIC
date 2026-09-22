"""StarCraft II (SMAC) environment family.

Family-per-directory layout. Add new SMAC variants (e.g. SMACv2 FastLane
wrapper, alternative reward shaping helpers) under this directory rather
than at the flat `environments/` level.

Public API mirrors a single-file layout: consumers
`from xuance_worker.environments.smac import StarCraft2FastLane_Env`.
"""

from xuance_worker.environments.smac.smac import StarCraft2FastLane_Env

__all__ = ["StarCraft2FastLane_Env"]
