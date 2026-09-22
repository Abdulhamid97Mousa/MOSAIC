"""StarCraft II v2 (SMACv2) environment family.

Family-per-directory layout, mirrors environments/smac/ for SMAC v1.

SMACv2 fundamentally differs from SMAC v1: units are procedurally injected
at each episode reset via SC2's debug API (StarCraftCapabilityEnvWrapper),
whereas SMAC v1 has units pre-placed inside the .SC2Map file. The two cannot
share a wrapper class.

Public API:
    from xuance_worker.environments.smacv2 import StarCraft2v2FastLane_Env
"""

from xuance_worker.environments.smacv2.starcraft2v2 import StarCraft2v2FastLane_Env

__all__ = ["StarCraft2v2FastLane_Env"]
