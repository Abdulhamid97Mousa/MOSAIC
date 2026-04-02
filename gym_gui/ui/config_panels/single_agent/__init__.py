"""Single-agent environment UI helpers."""

# Re-export gym family helpers
# Re-export ALE helpers (Atari via ALE)
from .ale import (
    ALE_GAME_IDS,
    build_ale_controls,
)
from .ale import (
    ControlCallbacks as ALEControlCallbacks,
)

# Re-export Crafter helpers
from .crafter import (
    CRAFTER_GAME_IDS,
    build_crafter_controls,
)
from .crafter import (
    ControlCallbacks as CrafterControlCallbacks,
)
from .gym import (
    BOX2D_FAMILY,
    TOY_TEXT_FAMILY,
    build_bipedal_controls,
    build_car_racing_controls,
    build_cliff_controls,
    build_frozenlake_controls,
    build_frozenlake_v2_controls,
    build_lunarlander_controls,
    build_taxi_controls,
)

# Re-export MalmoEnv helpers
from .malmo import (
    DEFAULT_MALMO_CONFIG,
    MALMOENV_FAMILY,
    build_malmo_controls,
)

# Re-export minigrid helpers
from .minigrid import (
    MINIGRID_GAME_IDS,
    ControlCallbacks,
    build_minigrid_controls,
    resolve_default_config,
)
from .vizdoom import (
    VIZDOOM_GAME_IDS,
    build_vizdoom_controls,
)
from .vizdoom import (
    ControlCallbacks as ViZDoomControlCallbacks,
)

__all__ = [
    # Gym family
    "TOY_TEXT_FAMILY",
    "BOX2D_FAMILY",
    "build_frozenlake_controls",
    "build_frozenlake_v2_controls",
    "build_taxi_controls",
    "build_cliff_controls",
    "build_lunarlander_controls",
    "build_car_racing_controls",
    "build_bipedal_controls",
    # MiniGrid family
    "MINIGRID_GAME_IDS",
    "ControlCallbacks",
    "build_minigrid_controls",
    "resolve_default_config",
    # ALE family
    "ALE_GAME_IDS",
    "ALEControlCallbacks",
    "build_ale_controls",
    "VIZDOOM_GAME_IDS",
    "ViZDoomControlCallbacks",
    "build_vizdoom_controls",
    # Crafter family
    "CRAFTER_GAME_IDS",
    "CrafterControlCallbacks",
    "build_crafter_controls",
    # MalmoEnv family
    "MALMOENV_FAMILY",
    "DEFAULT_MALMO_CONFIG",
    "build_malmo_controls",
]
