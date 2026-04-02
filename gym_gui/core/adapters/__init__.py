"""Environment adapter base classes and concrete implementations."""

from .ale import (
    ALE_ADAPTERS,
    AdventureV4Adapter,
    AdventureV5Adapter,
    AirRaidV4Adapter,
    AirRaidV5Adapter,
    ALEAdapter,
    AssaultV4Adapter,
    AssaultV5Adapter,
)
from .babyai import (
    BABYAI_ADAPTERS,
    BabyAIActionObjDoorAdapter,
    BabyAIBlockedUnlockPickupAdapter,
    BabyAIBossLevelAdapter,
    BabyAIBossLevelNoUnlockAdapter,
    BabyAIFindObjS5Adapter,
    BabyAIGoToAdapter,
    BabyAIGoToDoorAdapter,
    BabyAIGoToImpUnlockAdapter,
    BabyAIGoToLocalAdapter,
    BabyAIGoToObjAdapter,
    BabyAIGoToObjDoorAdapter,
    BabyAIGoToRedBallAdapter,
    BabyAIGoToRedBallGreyAdapter,
    BabyAIGoToRedBallNoDistsAdapter,
    BabyAIGoToRedBlueBallAdapter,
    BabyAIGoToSeqAdapter,
    BabyAIKeyCorridorS3R1Adapter,
    BabyAIKeyCorridorS3R2Adapter,
    BabyAIKeyCorridorS3R3Adapter,
    BabyAIKeyInBoxAdapter,
    BabyAIMiniBossLevelAdapter,
    BabyAIMoveTwoAcrossS8N9Adapter,
    BabyAIOneRoomS8Adapter,
    BabyAIOpenAdapter,
    BabyAIOpenDoorAdapter,
    BabyAIOpenDoorsOrderN2Adapter,
    BabyAIOpenDoorsOrderN4Adapter,
    BabyAIOpenRedDoorAdapter,
    BabyAIOpenTwoDoorsAdapter,
    BabyAIPickupAboveAdapter,
    BabyAIPickupAdapter,
    BabyAIPickupDistAdapter,
    BabyAIPickupLocAdapter,
    BabyAIPutNextAdapter,
    BabyAIPutNextLocalAdapter,
    BabyAISynthAdapter,
    BabyAISynthLocAdapter,
    BabyAISynthSeqAdapter,
    BabyAIUnblockPickupAdapter,
    BabyAIUnlockAdapter,
    BabyAIUnlockLocalAdapter,
    BabyAIUnlockPickupAdapter,
    BabyAIUnlockToUnlockAdapter,
)
from .base import (
    AdapterContext,
    AdapterNotReadyError,
    AdapterStep,
    EnvironmentAdapter,
    UnsupportedModeError,
    WorkerCapabilities,
)
from .box2d import (
    BOX2D_ADAPTERS,
    BipedalWalkerAdapter,
    Box2DAdapter,
    CarRacingAdapter,
    LunarLanderAdapter,
)
from .minigrid import (
    MINIGRID_ADAPTERS,
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
    MiniGridLavaCrossingS9N1Adapter,
    MiniGridLavaCrossingS9N2Adapter,
    MiniGridLavaCrossingS9N3Adapter,
    MiniGridLavaCrossingS11N5Adapter,
    MiniGridLavaGapS5Adapter,
    MiniGridLavaGapS6Adapter,
    MiniGridLavaGapS7Adapter,
    MiniGridSimpleCrossingS9N1Adapter,
    MiniGridSimpleCrossingS9N2Adapter,
    MiniGridSimpleCrossingS9N3Adapter,
    MiniGridSimpleCrossingS11N5Adapter,
)
from .paradigm import (
    ParadigmAdapter,
    ParadigmStepResult,
    SequentialParadigmAdapter,
    SimultaneousParadigmAdapter,
    SingleAgentParadigmAdapter,
    create_paradigm_adapter,
)
from .toy_text import (
    TOY_TEXT_ADAPTERS,
    CliffWalkingAdapter,
    FrozenLakeAdapter,
    FrozenLakeV2Adapter,
    TaxiAdapter,
)

try:  # Optional dependency
    from .vizdoom import (  # pragma: no cover - exercised in integration
        VIZDOOM_ADAPTERS,
        ViZDoomAdapter,
        ViZDoomBasicAdapter,
        ViZDoomConfig,
        ViZDoomDeadlyCorridorAdapter,
        ViZDoomDeathmatchAdapter,
        ViZDoomDefendTheCenterAdapter,
        ViZDoomDefendTheLineAdapter,
        ViZDoomHealthGatheringAdapter,
        ViZDoomHealthGatheringSupremeAdapter,
        ViZDoomMyWayHomeAdapter,
        ViZDoomPredictPositionAdapter,
        ViZDoomTakeCoverAdapter,
    )
    _VIZDOOM_AVAILABLE = True
except Exception:  # pragma: no cover - vizdoom optional
    ViZDoomAdapter = None  # type: ignore
    ViZDoomBasicAdapter = None  # type: ignore
    ViZDoomConfig = None  # type: ignore
    ViZDoomDeadlyCorridorAdapter = None  # type: ignore
    ViZDoomDefendTheCenterAdapter = None  # type: ignore
    ViZDoomDefendTheLineAdapter = None  # type: ignore
    ViZDoomDeathmatchAdapter = None  # type: ignore
    ViZDoomHealthGatheringAdapter = None  # type: ignore
    ViZDoomHealthGatheringSupremeAdapter = None  # type: ignore
    ViZDoomMyWayHomeAdapter = None  # type: ignore
    ViZDoomPredictPositionAdapter = None  # type: ignore
    ViZDoomTakeCoverAdapter = None  # type: ignore
    VIZDOOM_ADAPTERS = {}
    _VIZDOOM_AVAILABLE = False

try:  # Optional dependency - PettingZoo multi-agent environments
    from .pettingzoo import (  # pragma: no cover - pettingzoo optional
        PETTINGZOO_ADAPTERS,
        ChessAdapter,
        ConnectFourAdapter,
        GoAdapter,
        MultiwalkerAdapter,
        PettingZooAdapter,
        PettingZooConfig,
        PistonballAdapter,
        SimpleSpreadAdapter,
        SimpleTagAdapter,
        TicTacToeAdapter,
        create_pettingzoo_adapter,
    )
    _PETTINGZOO_AVAILABLE = True
except Exception:  # pragma: no cover - pettingzoo optional
    PettingZooAdapter = None  # type: ignore
    PettingZooConfig = None  # type: ignore
    ChessAdapter = None  # type: ignore
    ConnectFourAdapter = None  # type: ignore
    TicTacToeAdapter = None  # type: ignore
    GoAdapter = None  # type: ignore
    SimpleSpreadAdapter = None  # type: ignore
    SimpleTagAdapter = None  # type: ignore
    PistonballAdapter = None  # type: ignore
    MultiwalkerAdapter = None  # type: ignore
    PETTINGZOO_ADAPTERS = {}
    create_pettingzoo_adapter = None  # type: ignore
    _PETTINGZOO_AVAILABLE = False

try:  # Optional dependency - MiniHack (sandbox RL on NLE)
    from .minihack import (  # pragma: no cover - minihack optional
        MINIHACK_ADAPTERS,
        MiniHackAdapter,
        MiniHackConfig,
        MiniHackCorridorR2Adapter,
        MiniHackCorridorR3Adapter,
        MiniHackCorridorR5Adapter,
        MiniHackEatAdapter,
        MiniHackExploreMazeEasyAdapter,
        MiniHackExploreMazeHardAdapter,
        MiniHackHideNSeekAdapter,
        MiniHackLavaCrossAdapter,
        MiniHackMazeWalk9x9Adapter,
        MiniHackMazeWalk15x15Adapter,
        MiniHackMazeWalk45x19Adapter,
        MiniHackMementoF2Adapter,
        MiniHackMementoF4Adapter,
        MiniHackPutOnAdapter,
        MiniHackQuaffAdapter,
        MiniHackReadAdapter,
        MiniHackRiverAdapter,
        MiniHackRiverNarrowAdapter,
        MiniHackRoom5x5Adapter,
        MiniHackRoom15x15Adapter,
        MiniHackWearAdapter,
        MiniHackWieldAdapter,
        MiniHackWoDEasyAdapter,
        MiniHackWoDHardAdapter,
        MiniHackWoDMediumAdapter,
        MiniHackZapAdapter,
    )
    _MINIHACK_AVAILABLE = True
except Exception:  # pragma: no cover - minihack optional
    MiniHackAdapter = None  # type: ignore
    MiniHackConfig = None  # type: ignore
    MiniHackRoom5x5Adapter = None  # type: ignore
    MiniHackRoom15x15Adapter = None  # type: ignore
    MiniHackCorridorR2Adapter = None  # type: ignore
    MiniHackCorridorR3Adapter = None  # type: ignore
    MiniHackCorridorR5Adapter = None  # type: ignore
    MiniHackMazeWalk9x9Adapter = None  # type: ignore
    MiniHackMazeWalk15x15Adapter = None  # type: ignore
    MiniHackMazeWalk45x19Adapter = None  # type: ignore
    MiniHackRiverAdapter = None  # type: ignore
    MiniHackRiverNarrowAdapter = None  # type: ignore
    MiniHackEatAdapter = None  # type: ignore
    MiniHackWearAdapter = None  # type: ignore
    MiniHackWieldAdapter = None  # type: ignore
    MiniHackZapAdapter = None  # type: ignore
    MiniHackReadAdapter = None  # type: ignore
    MiniHackQuaffAdapter = None  # type: ignore
    MiniHackPutOnAdapter = None  # type: ignore
    MiniHackLavaCrossAdapter = None  # type: ignore
    MiniHackWoDEasyAdapter = None  # type: ignore
    MiniHackWoDMediumAdapter = None  # type: ignore
    MiniHackWoDHardAdapter = None  # type: ignore
    MiniHackExploreMazeEasyAdapter = None  # type: ignore
    MiniHackExploreMazeHardAdapter = None  # type: ignore
    MiniHackHideNSeekAdapter = None  # type: ignore
    MiniHackMementoF2Adapter = None  # type: ignore
    MiniHackMementoF4Adapter = None  # type: ignore
    MINIHACK_ADAPTERS = {}
    _MINIHACK_AVAILABLE = False

try:  # Optional dependency - NetHack (full game via NLE)
    from .nethack import (  # pragma: no cover - nethack optional
        NETHACK_ADAPTERS,
        NetHackAdapter,
        NetHackChallengeAdapter,
        NetHackConfig,
        NetHackGoldAdapter,
        NetHackOracleAdapter,
        NetHackScoreAdapter,
        NetHackScoutAdapter,
        NetHackStaircaseAdapter,
        NetHackStaircasePetAdapter,
    )
    from .nethack import (
        NetHackEatAdapter as NetHackEatTaskAdapter,  # Avoid conflict with MiniHack
    )
    _NETHACK_AVAILABLE = True
except Exception:  # pragma: no cover - nethack optional
    NetHackAdapter = None  # type: ignore
    NetHackConfig = None  # type: ignore
    NetHackChallengeAdapter = None  # type: ignore
    NetHackScoreAdapter = None  # type: ignore
    NetHackStaircaseAdapter = None  # type: ignore
    NetHackStaircasePetAdapter = None  # type: ignore
    NetHackOracleAdapter = None  # type: ignore
    NetHackGoldAdapter = None  # type: ignore
    NetHackEatTaskAdapter = None  # type: ignore
    NetHackScoutAdapter = None  # type: ignore
    NETHACK_ADAPTERS = {}
    _NETHACK_AVAILABLE = False

try:  # Optional dependency - Crafter (open world survival benchmark)
    from .crafter import (  # pragma: no cover - crafter optional
        CRAFTER_ACHIEVEMENTS,
        CRAFTER_ACTIONS,
        CRAFTER_ADAPTERS,
        CrafterAdapter,
        CrafterNoRewardAdapter,
        CrafterRewardAdapter,
    )
    _CRAFTER_AVAILABLE = True
except Exception:  # pragma: no cover - crafter optional
    CrafterAdapter = None  # type: ignore
    CrafterRewardAdapter = None  # type: ignore
    CrafterNoRewardAdapter = None  # type: ignore
    CRAFTER_ADAPTERS = {}
    CRAFTER_ACHIEVEMENTS = []  # type: ignore
    CRAFTER_ACTIONS = []  # type: ignore
    _CRAFTER_AVAILABLE = False

try:  # Optional dependency - TextWorld (text-based game environments)
    from .textworld import (  # pragma: no cover - textworld optional
        TEXTWORLD_ADAPTERS,
        TEXTWORLD_CHALLENGES,
        TextWorldAdapter,
        TextWorldCoinCollectorAdapter,
        TextWorldCookingAdapter,
        TextWorldSimpleAdapter,
        TextWorldTreasureHunterAdapter,
    )
    _TEXTWORLD_AVAILABLE = True
except Exception:  # pragma: no cover - textworld optional
    TextWorldAdapter = None  # type: ignore
    TextWorldSimpleAdapter = None  # type: ignore
    TextWorldCoinCollectorAdapter = None  # type: ignore
    TextWorldTreasureHunterAdapter = None  # type: ignore
    TextWorldCookingAdapter = None  # type: ignore
    TEXTWORLD_ADAPTERS = {}
    TEXTWORLD_CHALLENGES = []  # type: ignore
    _TEXTWORLD_AVAILABLE = False

try:  # Optional dependency - Jumanji (JAX-based logic puzzle environments)
    from .jumanji import (  # pragma: no cover - jumanji optional
        GAME2048_ACTIONS,
        JUMANJI_ADAPTERS,
        JUMANJI_ENV_NAMES,
        RUBIKS_CUBE_ACTIONS,
        JumanjiAdapter,
        JumanjiGame2048Adapter,
        JumanjiGraphColoringAdapter,
        JumanjiMinesweeperAdapter,
        JumanjiRubiksCubeAdapter,
        JumanjiSlidingPuzzleAdapter,
        JumanjiSudokuAdapter,
    )
    _JUMANJI_AVAILABLE = True
except Exception:  # pragma: no cover - jumanji optional
    JumanjiAdapter = None  # type: ignore
    JumanjiGame2048Adapter = None  # type: ignore
    JumanjiMinesweeperAdapter = None  # type: ignore
    JumanjiRubiksCubeAdapter = None  # type: ignore
    JumanjiSlidingPuzzleAdapter = None  # type: ignore
    JumanjiSudokuAdapter = None  # type: ignore
    JumanjiGraphColoringAdapter = None  # type: ignore
    JUMANJI_ADAPTERS = {}
    JUMANJI_ENV_NAMES = []  # type: ignore
    GAME2048_ACTIONS = []  # type: ignore
    RUBIKS_CUBE_ACTIONS = []  # type: ignore
    _JUMANJI_AVAILABLE = False

try:  # Optional dependency - Procgen (procedural generation benchmark)
    from .procgen import (  # pragma: no cover - procgen optional
        PROCGEN_ACTIONS,
        PROCGEN_ADAPTERS,
        PROCGEN_ENV_NAMES,
        ProcgenAdapter,
        ProcgenBigfishAdapter,
        ProcgenBossfightAdapter,
        ProcgenCaveflyerAdapter,
        ProcgenChaserAdapter,
        ProcgenClimberAdapter,
        ProcgenCoinrunAdapter,
        ProcgenDodgeballAdapter,
        ProcgenFruitbotAdapter,
        ProcgenHeistAdapter,
        ProcgenJumperAdapter,
        ProcgenLeaperAdapter,
        ProcgenMazeAdapter,
        ProcgenMinerAdapter,
        ProcgenNinjaAdapter,
        ProcgenPlunderAdapter,
        ProcgenStarpilotAdapter,
    )
    _PROCGEN_AVAILABLE = True
except Exception:  # pragma: no cover - procgen optional
    ProcgenAdapter = None  # type: ignore[misc, assignment]
    PROCGEN_ADAPTERS = {}  # type: ignore[misc]
    PROCGEN_ENV_NAMES = []  # type: ignore[misc]
    PROCGEN_ACTIONS = []  # type: ignore[misc]
    _PROCGEN_AVAILABLE = False

try:  # Optional dependency - BabaIsAI (puzzle game environment)
    from .babaisai import (  # pragma: no cover - babaisai optional
        BABAISAI_ACTIONS,
        BABAISAI_ADAPTERS,
        BabaIsAIAdapter,
        BabaIsAIConfig,
        create_babaisai_adapter,
    )
    _BABAISAI_AVAILABLE = True
except Exception:  # pragma: no cover - babaisai optional
    BabaIsAIAdapter = None  # type: ignore[misc, assignment]
    BabaIsAIConfig = None  # type: ignore[misc, assignment]
    BABAISAI_ADAPTERS = {}  # type: ignore[misc]
    BABAISAI_ACTIONS = []  # type: ignore[misc]
    create_babaisai_adapter = None  # type: ignore[misc, assignment]
    _BABAISAI_AVAILABLE = False

try:  # Optional dependency - MOSAIC MultiGrid (competitive team-based)
    from .mosaic_multigrid import (
        MOSAIC_MULTIGRID_ACTIONS,
        MOSAIC_MULTIGRID_ADAPTERS,
        MultiGridBasketballIndAgObsAdapter,
        MultiGridBasketballTeamObsAdapter,
        MultiGridCollect1vs1IndAgObsAdapter,
        MultiGridCollect2vs2IndAgObsAdapter,
        MultiGridCollect2vs2TeamObsAdapter,
        MultiGridCollect3HAdapter,
        MultiGridCollect4HAdapter,
        MultiGridCollectIndAgObsAdapter,
        MultiGridSoccer1vs1IndAgObsAdapter,
        MultiGridSoccerAdapter,
        MultiGridSoccerIndAgObsAdapter,
        MultiGridSoccerTeamObsAdapter,
    )
    from .mosaic_multigrid import (  # pragma: no cover - mosaic_multigrid optional
        MultiGridAdapter as MosaicMultiGridAdapter,
    )
    _MOSAIC_MULTIGRID_AVAILABLE = True
except Exception:  # pragma: no cover - mosaic_multigrid optional
    MosaicMultiGridAdapter = None  # type: ignore[misc, assignment]
    MOSAIC_MULTIGRID_ADAPTERS = {}  # type: ignore[misc]
    MOSAIC_MULTIGRID_ACTIONS = []  # type: ignore[misc]
    _MOSAIC_MULTIGRID_AVAILABLE = False

try:  # Optional dependency - MalmoEnv (Microsoft Malmo, Java-based Minecraft)
    from .mosaic_malmo import (
        MALMOENV_ACTIONS,
        MALMOENV_ADAPTERS,
        MALMOENV_MULTI_AGENT_ADAPTERS,
        MOSAIC_MALMO_ACTIONS,  # legacy alias
        MOSAIC_MALMO_ADAPTERS,  # legacy alias
        MalmoEnvAdapter,
        MalmoEnvAtticAdapter,
        MalmoEnvCatchTheMobAdapter,
        MalmoEnvCliffWalkingAdapter,
        MalmoEnvDefaultFlatWorldAdapter,
        MalmoEnvDefaultWorldAdapter,
        MalmoEnvEatingAdapter,
        MalmoEnvFindTheGoalAdapter,
        MalmoEnvMazeRunnerAdapter,
        MalmoEnvMobChaseAdapter,
        MalmoEnvMultiAgentAdapter,
        MalmoEnvObstaclesAdapter,
        MalmoEnvTreasureHuntAdapter,
        MalmoEnvTreasureHuntMultiAgentAdapter,
        MalmoEnvTrickyArenaAdapter,
        MalmoEnvVerticalAdapter,
    )
    _MOSAIC_MALMO_AVAILABLE = True
except Exception:  # pragma: no cover - malmoenv optional
    MalmoEnvAdapter = None  # type: ignore[misc, assignment]
    MalmoEnvMultiAgentAdapter = None  # type: ignore[misc, assignment]
    MalmoEnvMobChaseAdapter = None  # type: ignore[misc, assignment]
    MalmoEnvMazeRunnerAdapter = None  # type: ignore[misc, assignment]
    MalmoEnvVerticalAdapter = None  # type: ignore[misc, assignment]
    MalmoEnvCliffWalkingAdapter = None  # type: ignore[misc, assignment]
    MalmoEnvCatchTheMobAdapter = None  # type: ignore[misc, assignment]
    MalmoEnvFindTheGoalAdapter = None  # type: ignore[misc, assignment]
    MalmoEnvAtticAdapter = None  # type: ignore[misc, assignment]
    MalmoEnvDefaultFlatWorldAdapter = None  # type: ignore[misc, assignment]
    MalmoEnvDefaultWorldAdapter = None  # type: ignore[misc, assignment]
    MalmoEnvEatingAdapter = None  # type: ignore[misc, assignment]
    MalmoEnvObstaclesAdapter = None  # type: ignore[misc, assignment]
    MalmoEnvTrickyArenaAdapter = None  # type: ignore[misc, assignment]
    MalmoEnvTreasureHuntAdapter = None  # type: ignore[misc, assignment]
    MalmoEnvTreasureHuntMultiAgentAdapter = None  # type: ignore[misc, assignment]
    MALMOENV_ADAPTERS = {}  # type: ignore[misc]
    MALMOENV_MULTI_AGENT_ADAPTERS = {}  # type: ignore[misc]
    MALMOENV_ACTIONS = []  # type: ignore[misc]
    MOSAIC_MALMO_ADAPTERS = {}  # type: ignore[misc]
    MOSAIC_MALMO_ACTIONS = []  # type: ignore[misc]
    _MOSAIC_MALMO_AVAILABLE = False

try:  # Optional dependency - INI MultiGrid (cooperative multi-agent)
    from .ini_multigrid import (
        INI_MULTIGRID_ACTIONS,
        INI_MULTIGRID_ADAPTERS,
    )
    from .ini_multigrid import (  # pragma: no cover - ini_multigrid optional
        MultiGridAdapter as INIMultiGridAdapter,
    )
    _INI_MULTIGRID_AVAILABLE = True
except Exception:  # pragma: no cover - ini_multigrid optional
    INIMultiGridAdapter = None  # type: ignore[misc, assignment]
    INI_MULTIGRID_ADAPTERS = {}  # type: ignore[misc]
    INI_MULTIGRID_ACTIONS = []  # type: ignore[misc]
    _INI_MULTIGRID_AVAILABLE = False

try:  # Optional dependency - MeltingPot (multi-agent social scenarios)
    from .meltingpot import (  # pragma: no cover - meltingpot optional
        MELTINGPOT_ACTION_NAMES,
        MELTINGPOT_ADAPTERS,
        AllelopathicHarvestAdapter,
        CleanUpAdapter,
        CollaborativeCookingAdapter,
        CommonsHarvestAdapter,
        KingOfTheHillAdapter,
        MeltingPotAdapter,
        PrisonersDilemmaAdapter,
        StagHuntAdapter,
        TerritoryAdapter,
        create_meltingpot_adapter,
    )
    _MELTINGPOT_AVAILABLE = True
except Exception:  # pragma: no cover - meltingpot optional
    MeltingPotAdapter = None  # type: ignore[misc, assignment]
    MELTINGPOT_ADAPTERS = {}  # type: ignore[misc]
    MELTINGPOT_ACTION_NAMES = []  # type: ignore[misc]
    create_meltingpot_adapter = None  # type: ignore[misc, assignment]
    _MELTINGPOT_AVAILABLE = False

try:  # Optional dependency - Overcooked (cooperative cooking)
    from .overcooked import (  # pragma: no cover - overcooked optional
        OVERCOOKED_ACTIONS,
        OVERCOOKED_ADAPTERS,
        AsymmetricAdvantagesAdapter,
        CoordinationRingAdapter,
        CounterCircuitAdapter,
        CrampedRoomAdapter,
        ForcedCoordinationAdapter,
        OvercookedAdapter,
    )
    _OVERCOOKED_AVAILABLE = True
except Exception:  # pragma: no cover - overcooked optional
    OvercookedAdapter = None  # type: ignore[misc, assignment]
    OVERCOOKED_ADAPTERS = {}  # type: ignore[misc]
    OVERCOOKED_ACTIONS = []  # type: ignore[misc]
    _OVERCOOKED_AVAILABLE = False

try:  # Optional dependency - OpenSpiel (game theory environments)
    from .open_spiel import (  # pragma: no cover - open_spiel optional
        OPENSPIEL_ADAPTERS,
        CheckersEnvironmentAdapter,
        CheckersRenderPayload,
    )
    _OPENSPIEL_AVAILABLE = True
except Exception:  # pragma: no cover - open_spiel optional
    CheckersEnvironmentAdapter = None  # type: ignore[misc, assignment]
    CheckersRenderPayload = None  # type: ignore[misc, assignment]
    OPENSPIEL_ADAPTERS = {}  # type: ignore[misc]
    _OPENSPIEL_AVAILABLE = False

try:  # Optional dependency - PyBullet Drones (quadcopter simulation)
    from .pybullet_drones import (  # pragma: no cover - pybullet_drones optional
        PYBULLET_DRONES_ADAPTERS,
        CtrlAviaryAdapter,
        HoverAviaryAdapter,
        MultiHoverAviaryAdapter,
        PyBulletDronesAdapter,
        PyBulletDronesConfig,
        VelocityAviaryAdapter,
    )
    _PYBULLET_DRONES_AVAILABLE = True
except Exception:  # pragma: no cover - pybullet_drones optional
    PyBulletDronesAdapter = None  # type: ignore[misc, assignment]
    PyBulletDronesConfig = None  # type: ignore[misc, assignment]
    PYBULLET_DRONES_ADAPTERS = {}  # type: ignore[misc]
    _PYBULLET_DRONES_AVAILABLE = False

try:  # Optional dependency - PettingZoo Classic (turn-based board games)
    from .pettingzoo_classic import (  # pragma: no cover - pettingzoo_classic optional
        PETTINGZOO_CLASSIC_ADAPTERS,
        ChessEnvironmentAdapter,
        ChessRenderPayload,
        ConnectFourEnvironmentAdapter,
        ConnectFourRenderPayload,
        GoEnvironmentAdapter,
        GoRenderPayload,
    )
    _PETTINGZOO_CLASSIC_AVAILABLE = True
except Exception:  # pragma: no cover - pettingzoo_classic optional
    ChessEnvironmentAdapter = None  # type: ignore[misc, assignment]
    ChessRenderPayload = None  # type: ignore[misc, assignment]
    ConnectFourEnvironmentAdapter = None  # type: ignore[misc, assignment]
    ConnectFourRenderPayload = None  # type: ignore[misc, assignment]
    GoEnvironmentAdapter = None  # type: ignore[misc, assignment]
    GoRenderPayload = None  # type: ignore[misc, assignment]
    PETTINGZOO_CLASSIC_ADAPTERS = {}  # type: ignore[misc]
    _PETTINGZOO_CLASSIC_AVAILABLE = False

try:  # Optional dependency - Draughts (checkers variants)
    from .draughts import (  # pragma: no cover - draughts optional
        DRAUGHTS_ADAPTERS,
        AmericanCheckersAdapter,
        BaseDraughtsAdapter,
        DraughtsRenderPayload,
        DraughtsState,
        InternationalDraughtsAdapter,
        RussianCheckersAdapter,
    )
    _DRAUGHTS_AVAILABLE = True
except Exception:  # pragma: no cover - draughts optional
    BaseDraughtsAdapter = None  # type: ignore[misc, assignment]
    AmericanCheckersAdapter = None  # type: ignore[misc, assignment]
    RussianCheckersAdapter = None  # type: ignore[misc, assignment]
    InternationalDraughtsAdapter = None  # type: ignore[misc, assignment]
    DraughtsState = None  # type: ignore[misc, assignment]
    DraughtsRenderPayload = None  # type: ignore[misc, assignment]
    DRAUGHTS_ADAPTERS = {}  # type: ignore[misc]
    _DRAUGHTS_AVAILABLE = False

try:  # Optional dependency - RWARE (robotic warehouse)
    from .rware import (  # pragma: no cover - rware optional
        ALL_RWARE_GAME_IDS,
        RWARE_ADAPTERS,
        RWAREAdapter,
    )
    _RWARE_AVAILABLE = True
except Exception:  # pragma: no cover - rware optional
    RWAREAdapter = None  # type: ignore[misc, assignment]
    RWARE_ADAPTERS = {}  # type: ignore[misc]
    ALL_RWARE_GAME_IDS = ()  # type: ignore[misc]
    _RWARE_AVAILABLE = False

try:
    from .smac import (
        SMAC_ADAPTERS,
        SMAC2S3ZAdapter,
        SMAC3MAdapter,
        SMAC3S5ZAdapter,
        SMAC5Mvs6MAdapter,
        SMAC8MAdapter,
        SMACAdapter,
        SMACMMM2Adapter,
    )
    _SMAC_AVAILABLE = True
except Exception:
    SMACAdapter = None  # type: ignore[misc, assignment]
    SMAC_ADAPTERS = {}  # type: ignore[misc]
    _SMAC_AVAILABLE = False

try:
    from .smacv2 import (
        SMACV2_ADAPTERS,
        SMACv2Adapter,
        SMACv2ProtossAdapter,
        SMACv2TerranAdapter,
        SMACv2ZergAdapter,
    )
    _SMACV2_AVAILABLE = True
except Exception:
    SMACv2Adapter = None  # type: ignore[misc, assignment]
    SMACV2_ADAPTERS = {}  # type: ignore[misc]
    _SMACV2_AVAILABLE = False

# Standalone adapters for Human vs Agent mode (simplified state-based interfaces)
from .chess_adapter import ChessAdapter as ChessHvAAdapter
from .chess_adapter import ChessState
from .connect_four_adapter import ConnectFourAdapter as ConnectFourHvAAdapter
from .connect_four_adapter import ConnectFourState
from .go_adapter import GoAdapter as GoHvAAdapter
from .go_adapter import GoState
from .tictactoe_adapter import TicTacToeAdapter as TicTacToeHvAAdapter
from .tictactoe_adapter import TicTacToeState

__all__ = [
    # Base classes
    "AdapterContext",
    "AdapterNotReadyError",
    "AdapterStep",
    "EnvironmentAdapter",
    "UnsupportedModeError",
    "WorkerCapabilities",
    # Paradigm adapters
    "ParadigmAdapter",
    "ParadigmStepResult",
    "SingleAgentParadigmAdapter",
    "SequentialParadigmAdapter",
    "SimultaneousParadigmAdapter",
    "create_paradigm_adapter",
    # Toy text adapters
    "FrozenLakeAdapter",
    "FrozenLakeV2Adapter",
    "CliffWalkingAdapter",
    "TaxiAdapter",
    "TOY_TEXT_ADAPTERS",
    "Box2DAdapter",
    "LunarLanderAdapter",
    "CarRacingAdapter",
    "BipedalWalkerAdapter",
    "BOX2D_ADAPTERS",
    "MiniGridAdapter",
    "MiniGridEmpty5x5Adapter",
    "MiniGridEmptyRandom5x5Adapter",
    "MiniGridEmpty6x6Adapter",
    "MiniGridEmptyRandom6x6Adapter",
    "MiniGridEmpty8x8Adapter",
    "MiniGridEmpty16x16Adapter",
    "MiniGridDoorKey5x5Adapter",
    "MiniGridDoorKey6x6Adapter",
    "MiniGridDoorKey8x8Adapter",
    "MiniGridDoorKey16x16Adapter",
    "MiniGridLavaGapS5Adapter",
    "MiniGridLavaGapS6Adapter",
    "MiniGridLavaGapS7Adapter",
    "MiniGridLavaCrossingS9N1Adapter",
    "MiniGridLavaCrossingS9N2Adapter",
    "MiniGridLavaCrossingS9N3Adapter",
    "MiniGridLavaCrossingS11N5Adapter",
    "MiniGridSimpleCrossingS9N1Adapter",
    "MiniGridSimpleCrossingS9N2Adapter",
    "MiniGridSimpleCrossingS9N3Adapter",
    "MiniGridSimpleCrossingS11N5Adapter",
    "MINIGRID_ADAPTERS",
    # BabyAI adapters
    "BabyAIGoToRedBallGreyAdapter",
    "BabyAIGoToRedBallAdapter",
    "BabyAIGoToRedBallNoDistsAdapter",
    "BabyAIGoToObjAdapter",
    "BabyAIGoToLocalAdapter",
    "BabyAIGoToAdapter",
    "BabyAIGoToImpUnlockAdapter",
    "BabyAIGoToSeqAdapter",
    "BabyAIGoToRedBlueBallAdapter",
    "BabyAIGoToDoorAdapter",
    "BabyAIGoToObjDoorAdapter",
    "BabyAIOpenAdapter",
    "BabyAIOpenRedDoorAdapter",
    "BabyAIOpenDoorAdapter",
    "BabyAIOpenTwoDoorsAdapter",
    "BabyAIOpenDoorsOrderN2Adapter",
    "BabyAIOpenDoorsOrderN4Adapter",
    "BabyAIPickupAdapter",
    "BabyAIUnblockPickupAdapter",
    "BabyAIPickupLocAdapter",
    "BabyAIPickupDistAdapter",
    "BabyAIPickupAboveAdapter",
    "BabyAIUnlockAdapter",
    "BabyAIUnlockLocalAdapter",
    "BabyAIKeyInBoxAdapter",
    "BabyAIUnlockPickupAdapter",
    "BabyAIBlockedUnlockPickupAdapter",
    "BabyAIUnlockToUnlockAdapter",
    "BabyAIPutNextLocalAdapter",
    "BabyAIPutNextAdapter",
    "BabyAIActionObjDoorAdapter",
    "BabyAIFindObjS5Adapter",
    "BabyAIKeyCorridorS3R1Adapter",
    "BabyAIKeyCorridorS3R2Adapter",
    "BabyAIKeyCorridorS3R3Adapter",
    "BabyAIOneRoomS8Adapter",
    "BabyAIMoveTwoAcrossS8N9Adapter",
    "BabyAISynthAdapter",
    "BabyAISynthLocAdapter",
    "BabyAISynthSeqAdapter",
    "BabyAIMiniBossLevelAdapter",
    "BabyAIBossLevelAdapter",
    "BabyAIBossLevelNoUnlockAdapter",
    "BABYAI_ADAPTERS",
    "ALEAdapter",
    "AdventureV4Adapter",
    "AdventureV5Adapter",
    "AirRaidV4Adapter",
    "AirRaidV5Adapter",
    "AssaultV4Adapter",
    "AssaultV5Adapter",
    "ALE_ADAPTERS",
]

if _VIZDOOM_AVAILABLE:
    __all__ += [
        "ViZDoomAdapter",
        "ViZDoomConfig",
        "ViZDoomBasicAdapter",
        "ViZDoomDeadlyCorridorAdapter",
        "ViZDoomDefendTheCenterAdapter",
        "ViZDoomDefendTheLineAdapter",
        "ViZDoomDeathmatchAdapter",
        "ViZDoomHealthGatheringAdapter",
        "ViZDoomHealthGatheringSupremeAdapter",
        "ViZDoomMyWayHomeAdapter",
        "ViZDoomPredictPositionAdapter",
        "ViZDoomTakeCoverAdapter",
        "VIZDOOM_ADAPTERS",
    ]

if _PETTINGZOO_AVAILABLE:
    __all__ += [
        "PettingZooAdapter",
        "PettingZooConfig",
        "ChessAdapter",
        "ConnectFourAdapter",
        "TicTacToeAdapter",
        "GoAdapter",
        "SimpleSpreadAdapter",
        "SimpleTagAdapter",
        "PistonballAdapter",
        "MultiwalkerAdapter",
        "PETTINGZOO_ADAPTERS",
        "create_pettingzoo_adapter",
    ]

if _MINIHACK_AVAILABLE:
    __all__ += [
        "MiniHackAdapter",
        "MiniHackConfig",
        "MiniHackRoom5x5Adapter",
        "MiniHackRoom15x15Adapter",
        "MiniHackCorridorR2Adapter",
        "MiniHackCorridorR3Adapter",
        "MiniHackCorridorR5Adapter",
        "MiniHackMazeWalk9x9Adapter",
        "MiniHackMazeWalk15x15Adapter",
        "MiniHackMazeWalk45x19Adapter",
        "MiniHackRiverAdapter",
        "MiniHackRiverNarrowAdapter",
        "MiniHackEatAdapter",
        "MiniHackWearAdapter",
        "MiniHackWieldAdapter",
        "MiniHackZapAdapter",
        "MiniHackReadAdapter",
        "MiniHackQuaffAdapter",
        "MiniHackPutOnAdapter",
        "MiniHackLavaCrossAdapter",
        "MiniHackWoDEasyAdapter",
        "MiniHackWoDMediumAdapter",
        "MiniHackWoDHardAdapter",
        "MiniHackExploreMazeEasyAdapter",
        "MiniHackExploreMazeHardAdapter",
        "MiniHackHideNSeekAdapter",
        "MiniHackMementoF2Adapter",
        "MiniHackMementoF4Adapter",
        "MINIHACK_ADAPTERS",
    ]

if _NETHACK_AVAILABLE:
    __all__ += [
        "NetHackAdapter",
        "NetHackConfig",
        "NetHackChallengeAdapter",
        "NetHackScoreAdapter",
        "NetHackStaircaseAdapter",
        "NetHackStaircasePetAdapter",
        "NetHackOracleAdapter",
        "NetHackGoldAdapter",
        "NetHackEatTaskAdapter",
        "NetHackScoutAdapter",
        "NETHACK_ADAPTERS",
    ]

if _CRAFTER_AVAILABLE:
    __all__ += [
        "CrafterAdapter",
        "CrafterRewardAdapter",
        "CrafterNoRewardAdapter",
        "CRAFTER_ADAPTERS",
        "CRAFTER_ACHIEVEMENTS",
        "CRAFTER_ACTIONS",
    ]

if _TEXTWORLD_AVAILABLE:
    __all__ += [
        "TextWorldAdapter",
        "TextWorldSimpleAdapter",
        "TextWorldCoinCollectorAdapter",
        "TextWorldTreasureHunterAdapter",
        "TextWorldCookingAdapter",
        "TEXTWORLD_CHALLENGES",
    ]

if _JUMANJI_AVAILABLE:
    __all__ += [
        "JumanjiAdapter",
        "JumanjiGame2048Adapter",
        "JumanjiMinesweeperAdapter",
        "JumanjiRubiksCubeAdapter",
        "JumanjiSlidingPuzzleAdapter",
        "JumanjiSudokuAdapter",
        "JumanjiGraphColoringAdapter",
        "JUMANJI_ADAPTERS",
        "JUMANJI_ENV_NAMES",
        "GAME2048_ACTIONS",
        "RUBIKS_CUBE_ACTIONS",
    ]

if _PROCGEN_AVAILABLE:
    __all__ += [
        "ProcgenAdapter",
        "ProcgenBigfishAdapter",
        "ProcgenBossfightAdapter",
        "ProcgenCaveflyerAdapter",
        "ProcgenChaserAdapter",
        "ProcgenClimberAdapter",
        "ProcgenCoinrunAdapter",
        "ProcgenDodgeballAdapter",
        "ProcgenFruitbotAdapter",
        "ProcgenHeistAdapter",
        "ProcgenJumperAdapter",
        "ProcgenLeaperAdapter",
        "ProcgenMazeAdapter",
        "ProcgenMinerAdapter",
        "ProcgenNinjaAdapter",
        "ProcgenPlunderAdapter",
        "ProcgenStarpilotAdapter",
        "PROCGEN_ADAPTERS",
        "PROCGEN_ENV_NAMES",
        "PROCGEN_ACTIONS",
    ]

if _BABAISAI_AVAILABLE:
    __all__ += [
        "BabaIsAIAdapter",
        "BabaIsAIConfig",
        "BABAISAI_ADAPTERS",
        "BABAISAI_ACTIONS",
        "create_babaisai_adapter",
    ]

if _MOSAIC_MULTIGRID_AVAILABLE:
    __all__ += [
        "MosaicMultiGridAdapter",
        "MultiGridSoccerAdapter",
        "MultiGridCollect3HAdapter",
        "MultiGridCollect4HAdapter",
        "MultiGridSoccerIndAgObsAdapter",
        "MultiGridSoccer1vs1IndAgObsAdapter",
        "MultiGridCollectIndAgObsAdapter",
        "MultiGridCollect2vs2IndAgObsAdapter",
        "MultiGridCollect1vs1IndAgObsAdapter",
        "MultiGridBasketballIndAgObsAdapter",
        "MultiGridSoccerTeamObsAdapter",
        "MultiGridCollect2vs2TeamObsAdapter",
        "MultiGridBasketballTeamObsAdapter",
        "MOSAIC_MULTIGRID_ADAPTERS",
        "MOSAIC_MULTIGRID_ACTIONS",
    ]

if _MOSAIC_MALMO_AVAILABLE:
    __all__ += [
        "MalmoEnvAdapter",
        "MalmoEnvMobChaseAdapter",
        "MalmoEnvMazeRunnerAdapter",
        "MalmoEnvVerticalAdapter",
        "MalmoEnvCliffWalkingAdapter",
        "MalmoEnvCatchTheMobAdapter",
        "MalmoEnvFindTheGoalAdapter",
        "MalmoEnvAtticAdapter",
        "MalmoEnvDefaultFlatWorldAdapter",
        "MalmoEnvDefaultWorldAdapter",
        "MalmoEnvEatingAdapter",
        "MalmoEnvObstaclesAdapter",
        "MalmoEnvTrickyArenaAdapter",
        "MalmoEnvTreasureHuntAdapter",
        "MalmoEnvMultiAgentAdapter",
        "MalmoEnvTreasureHuntMultiAgentAdapter",
        "MALMOENV_ADAPTERS",
        "MALMOENV_MULTI_AGENT_ADAPTERS",
        "MALMOENV_ACTIONS",
        "MOSAIC_MALMO_ADAPTERS",
        "MOSAIC_MALMO_ACTIONS",
    ]

if _INI_MULTIGRID_AVAILABLE:
    __all__ += [
        "INIMultiGridAdapter",
        "INI_MULTIGRID_ADAPTERS",
        "INI_MULTIGRID_ACTIONS",
    ]

if _MELTINGPOT_AVAILABLE:
    __all__ += [
        "MeltingPotAdapter",
        "CollaborativeCookingAdapter",
        "CleanUpAdapter",
        "CommonsHarvestAdapter",
        "TerritoryAdapter",
        "KingOfTheHillAdapter",
        "PrisonersDilemmaAdapter",
        "StagHuntAdapter",
        "AllelopathicHarvestAdapter",
        "MELTINGPOT_ADAPTERS",
        "MELTINGPOT_ACTION_NAMES",
        "create_meltingpot_adapter",
    ]

if _OVERCOOKED_AVAILABLE:
    __all__ += [
        "OvercookedAdapter",
        "CrampedRoomAdapter",
        "AsymmetricAdvantagesAdapter",
        "CoordinationRingAdapter",
        "ForcedCoordinationAdapter",
        "CounterCircuitAdapter",
        "OVERCOOKED_ADAPTERS",
        "OVERCOOKED_ACTIONS",
    ]

if _OPENSPIEL_AVAILABLE:
    __all__ += [
        "CheckersEnvironmentAdapter",
        "CheckersRenderPayload",
        "OPENSPIEL_ADAPTERS",
    ]

if _PYBULLET_DRONES_AVAILABLE:
    __all__ += [
        "PyBulletDronesAdapter",
        "PyBulletDronesConfig",
        "HoverAviaryAdapter",
        "MultiHoverAviaryAdapter",
        "CtrlAviaryAdapter",
        "VelocityAviaryAdapter",
        "PYBULLET_DRONES_ADAPTERS",
    ]

if _PETTINGZOO_CLASSIC_AVAILABLE:
    __all__ += [
        "ChessEnvironmentAdapter",
        "ChessRenderPayload",
        "ConnectFourEnvironmentAdapter",
        "ConnectFourRenderPayload",
        "GoEnvironmentAdapter",
        "GoRenderPayload",
        "PETTINGZOO_CLASSIC_ADAPTERS",
    ]

if _DRAUGHTS_AVAILABLE:
    __all__ += [
        "BaseDraughtsAdapter",
        "AmericanCheckersAdapter",
        "RussianCheckersAdapter",
        "InternationalDraughtsAdapter",
        "DraughtsState",
        "DraughtsRenderPayload",
        "DRAUGHTS_ADAPTERS",
    ]

if _RWARE_AVAILABLE:
    __all__ += [
        "RWAREAdapter",
        "RWARE_ADAPTERS",
        "ALL_RWARE_GAME_IDS",
    ]

if _SMAC_AVAILABLE:
    __all__ += [
        "SMACAdapter",
        "SMAC_ADAPTERS",
        "SMAC3MAdapter",
        "SMAC8MAdapter",
        "SMAC2S3ZAdapter",
        "SMAC3S5ZAdapter",
        "SMAC5Mvs6MAdapter",
        "SMACMMM2Adapter",
    ]

if _SMACV2_AVAILABLE:
    __all__ += [
        "SMACv2Adapter",
        "SMACV2_ADAPTERS",
        "SMACv2TerranAdapter",
        "SMACv2ProtossAdapter",
        "SMACv2ZergAdapter",
    ]

# Human vs Agent mode adapters (always available)
__all__ += [
    "ChessHvAAdapter",
    "ChessState",
    "ConnectFourHvAAdapter",
    "ConnectFourState",
    "GoHvAAdapter",
    "GoState",
    "TicTacToeHvAAdapter",
    "TicTacToeState",
]
