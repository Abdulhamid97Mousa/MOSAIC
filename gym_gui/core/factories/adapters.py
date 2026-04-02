"""Factory helpers for environment adapters."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, TypeVar

from gym_gui.cache.memory import memoize
from gym_gui.config.game_configs import (
    ALEConfig,
    BipedalWalkerConfig,
    BlackjackConfig,
    CarRacingConfig,
    CliffWalkingConfig,
    FrozenLakeConfig,
    GameConfig,
    GriddlyConfig,
    LunarLanderConfig,
    MeltingPotConfig,
    MiniGridConfig,
    MultiGridConfig,
    OvercookedConfig,
    RWAREConfig,
    SMACConfig,
    TaxiConfig,
)
from gym_gui.core.adapters.ale import ALE_ADAPTERS, ALEAdapter
from gym_gui.core.adapters.babyai import BABYAI_ADAPTERS
from gym_gui.core.adapters.base import AdapterContext, EnvironmentAdapter
from gym_gui.core.adapters.box2d import BOX2D_ADAPTERS
from gym_gui.core.adapters.minigrid import MINIGRID_ADAPTERS
from gym_gui.core.adapters.toy_text import TOY_TEXT_ADAPTERS

# TYPE_CHECKING imports removed - using GameConfig type alias instead

try:  # Optional dependency
    from gym_gui.core.adapters.vizdoom import (  # pragma: no cover - optional
        VIZDOOM_ADAPTERS,
        ViZDoomAdapter,
        ViZDoomConfig,
    )
except Exception:  # pragma: no cover - vizdoom optional
    VIZDOOM_ADAPTERS: dict[Any, Any] = {}
    ViZDoomAdapter = None  # type: ignore[misc, assignment]
    ViZDoomConfig = None  # type: ignore[misc, assignment]

try:  # Optional dependency - PettingZoo Classic
    from gym_gui.core.adapters.pettingzoo_classic import (
        PETTINGZOO_CLASSIC_ADAPTERS,
    )
except Exception:  # pragma: no cover - pettingzoo optional
    PETTINGZOO_CLASSIC_ADAPTERS: dict[Any, Any] = {}

try:  # Optional dependency - MiniHack (sandbox RL on NLE)
    from gym_gui.core.adapters.minihack import (  # pragma: no cover - optional
        MINIHACK_ADAPTERS,
        MiniHackAdapter,
        MiniHackConfig,
    )
except Exception:  # pragma: no cover - minihack optional
    MINIHACK_ADAPTERS: dict[Any, Any] = {}
    MiniHackAdapter = None  # type: ignore[misc, assignment]
    MiniHackConfig = None  # type: ignore[misc, assignment]

try:  # Optional dependency - NetHack (full game via NLE)
    from gym_gui.core.adapters.nethack import (  # pragma: no cover - optional
        NETHACK_ADAPTERS,
        NetHackAdapter,
        NetHackConfig,
    )
except Exception:  # pragma: no cover - nethack optional
    NETHACK_ADAPTERS: dict[Any, Any] = {}
    NetHackAdapter = None  # type: ignore[misc, assignment]
    NetHackConfig = None  # type: ignore[misc, assignment]

try:  # Optional dependency - Crafter (open world survival benchmark)
    from gym_gui.config.game_configs import CrafterConfig
    from gym_gui.core.adapters.crafter import (  # pragma: no cover - optional
        CRAFTER_ADAPTERS,
        CrafterAdapter,
    )
except Exception:  # pragma: no cover - crafter optional
    CRAFTER_ADAPTERS: dict[Any, Any] = {}
    CrafterAdapter = None  # type: ignore[misc, assignment]
    CrafterConfig = None  # type: ignore[misc, assignment]

try:  # Optional dependency - Procgen (procedurally generated benchmark)
    from gym_gui.config.game_configs import ProcgenConfig
    from gym_gui.core.adapters.procgen import (  # pragma: no cover - optional
        PROCGEN_ADAPTERS,
        ProcgenAdapter,
    )
except Exception:  # pragma: no cover - procgen optional
    PROCGEN_ADAPTERS: dict[Any, Any] = {}
    ProcgenAdapter = None  # type: ignore[misc, assignment]
    ProcgenConfig = None  # type: ignore[misc, assignment]

try:  # Optional dependency - TextWorld (text-based game environments)
    from gym_gui.config.game_configs import TextWorldConfig
    from gym_gui.core.adapters.textworld import (  # pragma: no cover - optional
        TEXTWORLD_ADAPTERS,
        TextWorldAdapter,
    )
except Exception:  # pragma: no cover - textworld optional
    TEXTWORLD_ADAPTERS: dict[Any, Any] = {}
    TextWorldAdapter = None  # type: ignore[misc, assignment]
    TextWorldConfig = None  # type: ignore[misc, assignment]

try:  # Optional dependency - Jumanji (JAX-based logic puzzle environments)
    from gym_gui.config.game_configs import JumanjiConfig
    from gym_gui.core.adapters.jumanji import (  # pragma: no cover - optional
        JUMANJI_ADAPTERS,
        JumanjiAdapter,
    )
except Exception:  # pragma: no cover - jumanji optional
    JUMANJI_ADAPTERS: dict[Any, Any] = {}
    JumanjiAdapter = None  # type: ignore[misc, assignment]
    JumanjiConfig = None  # type: ignore[misc, assignment]

try:  # Optional dependency - PyBullet Drones (quadcopter control environments)
    from gym_gui.core.adapters.pybullet_drones import (  # pragma: no cover - optional
        PYBULLET_DRONES_ADAPTERS,
        PyBulletDronesAdapter,
        PyBulletDronesConfig,
    )
except Exception:  # pragma: no cover - pybullet-drones optional
    PYBULLET_DRONES_ADAPTERS: dict[Any, Any] = {}
    PyBulletDronesAdapter = None  # type: ignore[misc, assignment]
    PyBulletDronesConfig = None  # type: ignore[misc, assignment]

try:  # Optional dependency - OpenSpiel (board games via Shimmy)
    from gym_gui.core.adapters.open_spiel import (  # pragma: no cover - optional
        OPENSPIEL_ADAPTERS,
        CheckersEnvironmentAdapter,
    )
except Exception:  # pragma: no cover - openspiel optional
    OPENSPIEL_ADAPTERS: dict[Any, Any] = {}
    CheckersEnvironmentAdapter = None  # type: ignore[misc, assignment]

try:  # Draughts/Checkers variants with proper rule implementations
    from gym_gui.core.adapters.draughts import (  # pragma: no cover - draughts
        DRAUGHTS_ADAPTERS,
        AmericanCheckersAdapter,
        InternationalDraughtsAdapter,
        RussianCheckersAdapter,
    )
except Exception:  # pragma: no cover - draughts adapters
    DRAUGHTS_ADAPTERS: dict[Any, Any] = {}
    AmericanCheckersAdapter = None  # type: ignore[misc, assignment]
    RussianCheckersAdapter = None  # type: ignore[misc, assignment]
    InternationalDraughtsAdapter = None  # type: ignore[misc, assignment]

try:  # Optional dependency - BabaIsAI (rule manipulation puzzle benchmark)
    from gym_gui.core.adapters.babaisai import (  # pragma: no cover - optional
        BABAISAI_ADAPTERS,
        BabaIsAIAdapter,
        BabaIsAIConfig,
    )
except Exception:  # pragma: no cover - babaisai optional
    BABAISAI_ADAPTERS: dict[Any, Any] = {}
    BabaIsAIAdapter = None  # type: ignore[misc, assignment]
    BabaIsAIConfig = None  # type: ignore[misc, assignment]

try:  # Optional dependency - MOSAIC MultiGrid (competitive team-based PyPI package)
    from gym_gui.core.adapters.mosaic_multigrid import (  # pragma: no cover - optional
        MOSAIC_MULTIGRID_ADAPTERS,
    )
    from gym_gui.core.adapters.mosaic_multigrid import (
        MultiGridAdapter as MosaicMultiGridAdapter,
    )
except Exception:  # pragma: no cover - mosaic_multigrid optional
    MOSAIC_MULTIGRID_ADAPTERS: dict[Any, Any] = {}
    MosaicMultiGridAdapter = None  # type: ignore[misc, assignment]

try:  # Optional dependency - MalmoEnv (Microsoft Malmo, Java-based Minecraft)
    from gym_gui.core.adapters.mosaic_malmo import (  # pragma: no cover - optional
        MALMOENV_ADAPTERS,
        MOSAIC_MALMO_ADAPTERS,  # legacy alias
        MalmoEnvAdapter,
    )
except Exception:  # pragma: no cover - malmoenv optional
    MALMOENV_ADAPTERS: dict[Any, Any] = {}
    MOSAIC_MALMO_ADAPTERS: dict[Any, Any] = {}
    MalmoEnvAdapter = None  # type: ignore[misc, assignment]

try:  # Optional dependency - INI MultiGrid (cooperative exploration local package)
    from gym_gui.core.adapters.ini_multigrid import (  # pragma: no cover - optional
        INI_MULTIGRID_ADAPTERS,
    )
    from gym_gui.core.adapters.ini_multigrid import (
        MultiGridAdapter as INIMultiGridAdapter,
    )
except Exception:  # pragma: no cover - ini_multigrid optional
    INI_MULTIGRID_ADAPTERS: dict[Any, Any] = {}
    INIMultiGridAdapter = None  # type: ignore[misc, assignment]

try:  # Optional dependency - Melting Pot (multi-agent social scenarios via Shimmy)
    from gym_gui.core.adapters.meltingpot import (  # pragma: no cover - optional
        MELTINGPOT_ADAPTERS,
        MeltingPotAdapter,
    )
except Exception:  # pragma: no cover - meltingpot optional
    MELTINGPOT_ADAPTERS: dict[Any, Any] = {}
    MeltingPotAdapter = None  # type: ignore[misc, assignment]

try:  # pragma: no cover - optional dep: overcooked
    from gym_gui.core.adapters.overcooked import (  # pragma: no cover - optional
        OVERCOOKED_ADAPTERS,
        OvercookedAdapter,
    )
except Exception:  # pragma: no cover - overcooked optional
    OVERCOOKED_ADAPTERS: dict[Any, Any] = {}
    OvercookedAdapter = None  # type: ignore[misc, assignment]

try:  # Optional dependency - SMAC v1 (StarCraft Multi-Agent Challenge)
    from gym_gui.core.adapters.smac import (  # pragma: no cover - optional
        SMAC_ADAPTERS,
        SMACAdapter,
    )
except Exception:  # pragma: no cover - smac optional
    SMAC_ADAPTERS: dict[Any, Any] = {}
    SMACAdapter = None  # type: ignore[misc, assignment]

try:  # Optional dependency - SMACv2 (procedural unit generation)
    from gym_gui.core.adapters.smacv2 import (  # pragma: no cover - optional
        SMACV2_ADAPTERS,
        SMACv2Adapter,
    )
except Exception:  # pragma: no cover - smacv2 optional
    SMACV2_ADAPTERS: dict[Any, Any] = {}
    SMACv2Adapter = None  # type: ignore[misc, assignment]

try:  # Optional dependency - RWARE (Robotic Warehouse)
    from gym_gui.core.adapters.rware import (  # pragma: no cover - optional
        RWARE_ADAPTERS,
        RWAREAdapter,
    )
except Exception:  # pragma: no cover - rware optional
    RWARE_ADAPTERS: dict[Any, Any] = {}
    RWAREAdapter = None  # type: ignore[misc, assignment]

try:  # Optional dependency - Griddly (C++ backend grid world platform)
    from gym_gui.core.adapters.griddly import (  # pragma: no cover - optional
        GRIDDLY_ADAPTERS,
        GriddlyAdapter,
    )
except Exception:  # pragma: no cover - griddly optional
    GRIDDLY_ADAPTERS: dict[Any, Any] = {}
    GriddlyAdapter = None  # type: ignore[misc, assignment]

try:  # Optional dependency - HeMAC (Heterogeneous Multi-Agent Challenge)
    from gym_gui.core.adapters.hemac import (  # pragma: no cover - optional
        HEMAC_ADAPTERS,
        HeMACEnvironmentAdapter,
    )
except Exception:  # pragma: no cover - hemac optional
    HEMAC_ADAPTERS: dict[Any, Any] = {}
    HeMACEnvironmentAdapter = None  # type: ignore[misc, assignment]

from gym_gui.core.enums import GameId

AdapterT = TypeVar("AdapterT", bound=EnvironmentAdapter)


class AdapterFactoryError(KeyError):
    """Raised when an adapter cannot be created for the requested game."""


@memoize()
def _registry() -> Mapping[GameId, type[EnvironmentAdapter]]:
    """Build the adapter registry on first use.

    Toy-text adapters are the initial entries; future phases can extend this by
    importing additional modules and updating the mapping.
    """

    return {
        **TOY_TEXT_ADAPTERS,
        **BOX2D_ADAPTERS,
        **MINIGRID_ADAPTERS,
        **BABYAI_ADAPTERS,
        **ALE_ADAPTERS,
        **VIZDOOM_ADAPTERS,
        **PETTINGZOO_CLASSIC_ADAPTERS,
        **MINIHACK_ADAPTERS,
        **NETHACK_ADAPTERS,
        **CRAFTER_ADAPTERS,
        **PROCGEN_ADAPTERS,
        **TEXTWORLD_ADAPTERS,
        **JUMANJI_ADAPTERS,
        **PYBULLET_DRONES_ADAPTERS,
        **OPENSPIEL_ADAPTERS,
        **DRAUGHTS_ADAPTERS,
        **BABAISAI_ADAPTERS,
        **MOSAIC_MULTIGRID_ADAPTERS,
        **MOSAIC_MALMO_ADAPTERS,
        **INI_MULTIGRID_ADAPTERS,
        **MELTINGPOT_ADAPTERS,
        **OVERCOOKED_ADAPTERS,
        **SMAC_ADAPTERS,
        **SMACV2_ADAPTERS,
        **RWARE_ADAPTERS,
        **GRIDDLY_ADAPTERS,
        **HEMAC_ADAPTERS,
    }


def available_games() -> Iterable[GameId]:
    """Return the set of :class:`GameId` values with registered adapters."""

    return _registry().keys()


def get_adapter_cls(game_id: GameId) -> type[EnvironmentAdapter]:
    """Look up the adapter class registered for ``game_id``."""

    try:
        return _registry()[game_id]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise AdapterFactoryError(f"No adapter registered for '{game_id.value}'") from exc


def create_adapter(
    game_id: GameId,
    context: AdapterContext | None = None,
    *,
    game_config: GameConfig | None = None,
) -> EnvironmentAdapter:
    """Instantiate the adapter bound to the optional ``context`` and ``game_config``."""

    adapter_cls = get_adapter_cls(game_id)

    # Import adapter classes to check if game_config is supported
    from gym_gui.core.adapters.ale import (
        AdventureV4Adapter,
        AdventureV5Adapter,
    )
    from gym_gui.core.adapters.box2d import (
        BipedalWalkerAdapter,
        CarRacingAdapter,
        LunarLanderAdapter,
    )
    from gym_gui.core.adapters.minigrid import (
        MiniGridAdapter,
        MiniGridDoorKey5x5Adapter,
        MiniGridDoorKey6x6Adapter,
        MiniGridDoorKey8x8Adapter,
        MiniGridDoorKey16x16Adapter,
        MiniGridEmpty5x5Adapter,
        MiniGridEmpty6x6Adapter,
        MiniGridEmpty8x8Adapter,
        MiniGridEmpty16x16Adapter,
        MiniGridEmptyRandom5x5Adapter,
        MiniGridEmptyRandom6x6Adapter,
        MiniGridLavaGapS5Adapter,
        MiniGridLavaGapS6Adapter,
        MiniGridLavaGapS7Adapter,
    )
    from gym_gui.core.adapters.toy_text import (
        BlackjackAdapter,
        CliffWalkingAdapter,
        FrozenLakeAdapter,
        FrozenLakeV2Adapter,
        TaxiAdapter,
    )

    # Pass game config to appropriate adapter constructor
    if game_config is not None:
        if adapter_cls is FrozenLakeAdapter and isinstance(game_config, FrozenLakeConfig):
            adapter = FrozenLakeAdapter(context, game_config=game_config)
        elif adapter_cls is FrozenLakeV2Adapter and isinstance(game_config, FrozenLakeConfig):
            adapter = FrozenLakeV2Adapter(context, game_config=game_config)
        elif adapter_cls is TaxiAdapter and isinstance(game_config, TaxiConfig):
            adapter = TaxiAdapter(context, game_config=game_config)
        elif adapter_cls is CliffWalkingAdapter and isinstance(game_config, CliffWalkingConfig):
            adapter = CliffWalkingAdapter(context, game_config=game_config)
        elif adapter_cls is BlackjackAdapter and isinstance(game_config, BlackjackConfig):
            adapter = BlackjackAdapter(context, game_config=game_config)
        elif adapter_cls is LunarLanderAdapter and isinstance(game_config, LunarLanderConfig):
            adapter = LunarLanderAdapter(context, config=game_config)
        elif adapter_cls is CarRacingAdapter and isinstance(game_config, CarRacingConfig):
            adapter = CarRacingAdapter(context, config=game_config)
        elif adapter_cls is BipedalWalkerAdapter and isinstance(game_config, BipedalWalkerConfig):
            adapter = BipedalWalkerAdapter(context, config=game_config)
        elif issubclass(adapter_cls, MiniGridAdapter) and isinstance(game_config, MiniGridConfig):
            adapter = adapter_cls(context, config=game_config)
        elif issubclass(adapter_cls, ALEAdapter) and isinstance(game_config, ALEConfig):
            adapter = adapter_cls(context, config=game_config)
        elif (
            ViZDoomAdapter is not None
            and issubclass(adapter_cls, ViZDoomAdapter)
            and ViZDoomConfig is not None
            and isinstance(game_config, ViZDoomConfig)
        ):
            adapter = adapter_cls(context, config=game_config)  # type: ignore[arg-type]
        elif (
            MiniHackAdapter is not None
            and issubclass(adapter_cls, MiniHackAdapter)
            and MiniHackConfig is not None
            and isinstance(game_config, MiniHackConfig)
        ):
            adapter = adapter_cls(context, config=game_config)  # type: ignore[arg-type]
        elif (
            NetHackAdapter is not None
            and issubclass(adapter_cls, NetHackAdapter)
            and NetHackConfig is not None
            and isinstance(game_config, NetHackConfig)
        ):
            adapter = adapter_cls(context, config=game_config)  # type: ignore[arg-type]
        elif (
            CrafterAdapter is not None
            and issubclass(adapter_cls, CrafterAdapter)
            and CrafterConfig is not None
            and isinstance(game_config, CrafterConfig)
        ):
            adapter = adapter_cls(context, config=game_config)  # type: ignore[arg-type]
        elif (
            ProcgenAdapter is not None
            and issubclass(adapter_cls, ProcgenAdapter)
            and ProcgenConfig is not None
            and isinstance(game_config, ProcgenConfig)
        ):
            adapter = adapter_cls(context, config=game_config)  # type: ignore[arg-type]
        elif (
            TextWorldAdapter is not None
            and issubclass(adapter_cls, TextWorldAdapter)
            and TextWorldConfig is not None
            and isinstance(game_config, TextWorldConfig)
        ):
            adapter = adapter_cls(context, config=game_config)  # type: ignore[arg-type]
        elif (
            JumanjiAdapter is not None
            and issubclass(adapter_cls, JumanjiAdapter)
            and JumanjiConfig is not None
            and isinstance(game_config, JumanjiConfig)
        ):
            adapter = adapter_cls(context, config=game_config)  # type: ignore[arg-type]
        elif (
            PyBulletDronesAdapter is not None
            and issubclass(adapter_cls, PyBulletDronesAdapter)
            and PyBulletDronesConfig is not None
            and isinstance(game_config, PyBulletDronesConfig)
        ):
            adapter = adapter_cls(context, config=game_config)  # type: ignore[arg-type]
        elif (
            MosaicMultiGridAdapter is not None
            and issubclass(adapter_cls, MosaicMultiGridAdapter)
            and isinstance(game_config, MultiGridConfig)
        ):
            adapter = adapter_cls(context, config=game_config)  # type: ignore[arg-type]
        elif (
            INIMultiGridAdapter is not None
            and issubclass(adapter_cls, INIMultiGridAdapter)
            and isinstance(game_config, MultiGridConfig)
        ):
            adapter = adapter_cls(context, config=game_config)  # type: ignore[arg-type]
        elif (
            MeltingPotAdapter is not None
            and issubclass(adapter_cls, MeltingPotAdapter)
            and MeltingPotConfig is not None
            and isinstance(game_config, MeltingPotConfig)
        ):
            adapter = adapter_cls(context, config=game_config)  # type: ignore[arg-type]
        elif (
            OvercookedAdapter is not None
            and issubclass(adapter_cls, OvercookedAdapter)
            and OvercookedConfig is not None
            and isinstance(game_config, OvercookedConfig)
        ):
            adapter = adapter_cls(context, config=game_config)  # type: ignore[arg-type]
        elif (
            SMACAdapter is not None
            and issubclass(adapter_cls, SMACAdapter)
            and isinstance(game_config, SMACConfig)
        ):
            adapter = adapter_cls(context, config=game_config)  # type: ignore[arg-type]
        elif (
            SMACv2Adapter is not None
            and issubclass(adapter_cls, SMACv2Adapter)
            and isinstance(game_config, SMACConfig)
        ):
            adapter = adapter_cls(context, config=game_config)  # type: ignore[arg-type]
        elif (
            RWAREAdapter is not None
            and issubclass(adapter_cls, RWAREAdapter)
            and isinstance(game_config, RWAREConfig)
        ):
            adapter = adapter_cls(context, config=game_config)  # type: ignore[arg-type]
        elif (
            GriddlyAdapter is not None
            and issubclass(adapter_cls, GriddlyAdapter)
            and isinstance(game_config, GriddlyConfig)
        ):
            adapter = adapter_cls(context, config=game_config)  # type: ignore[arg-type]
        else:
            adapter = adapter_cls(context)
    else:
        # For MeltingPot, derive substrate config from GameId value
        if (
            MeltingPotAdapter is not None
            and issubclass(adapter_cls, MeltingPotAdapter)
            and MeltingPotConfig is not None
        ):
            substrate = game_id.value.removeprefix("meltingpot/")
            adapter = adapter_cls(context, config=MeltingPotConfig(substrate_name=substrate))
        else:
            adapter = adapter_cls(context)

    if context is not None:
        adapter.ensure_control_mode(context.control_mode)
    return adapter


__all__ = [
    "AdapterFactoryError",
    "available_games",
    "create_adapter",
    "get_adapter_cls",
]
