"""Google Research Football (gfootball) environment family.

Family-per-directory layout. Add new gfootball variants (e.g. FastLane
sidecar env with render=True, alternative reward shaping helpers) here
under this directory rather than at the flat `environments/` level.

Public API mirrors the single-file layout that preceded it: consumers
`from xuance_worker.environments.gfootball import GFootballFastLane_Env`
and get the same class regardless of internal file layout.
"""

from xuance_worker.environments.gfootball.gfootball import GFootballFastLane_Env

__all__ = ["GFootballFastLane_Env"]
