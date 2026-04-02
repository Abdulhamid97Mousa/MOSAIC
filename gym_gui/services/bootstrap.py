from __future__ import annotations

"""Helpers to bootstrap shared services for the application."""

import logging
import os

from gym_gui.config.paths import VAR_REPLAY_DIR, VAR_ROOT, VAR_TELEMETRY_DIR, ensure_var_directories
from gym_gui.constants import (
    DB_SINK_BATCH_SIZE,
    DB_SINK_CHECKPOINT_INTERVAL,
    DB_SINK_WRITER_QUEUE_SIZE,
    OPERATOR_CATEGORY_HUMAN,
    OPERATOR_CATEGORY_LLM,
    OPERATOR_CATEGORY_RL,
    OPERATOR_DESCRIPTION_BALROG_LLM,
    OPERATOR_DESCRIPTION_HUMAN,
    OPERATOR_DISPLAY_NAME_BALROG_LLM,
    OPERATOR_DISPLAY_NAME_HUMAN,
    TELEMETRY_HUB_BUFFER_SIZE,
    TELEMETRY_HUB_MAX_QUEUE,
    WORKER_ID_BALROG,
)
from gym_gui.controllers.live_telemetry_controllers import LiveTelemetryController
from gym_gui.rendering import RendererRegistry, create_default_renderer_registry
from gym_gui.services.action_mapping import ContinuousActionMapper, create_default_action_mapper
from gym_gui.services.actor import (
    ActorService,
    CleanRLWorkerActor,
    HumanKeyboardActor,
    RayRLlibWorkerActor,
    XuanCeWorkerActor,
)
from gym_gui.services.operator import (
    HumanOperator,
    OperatorService,
    WorkerOperator,
)
from gym_gui.services.policy_mapping import PolicyMappingService
from gym_gui.services.service_locator import ServiceLocator, get_service_locator
from gym_gui.services.storage import StorageRecorderService
from gym_gui.services.telemetry import TelemetryService
from gym_gui.services.trainer import RunRegistry, TrainerClient, TrainerClientConfig, TrainerClientRunner
from gym_gui.services.trainer.launcher import TrainerDaemonHandle, ensure_trainer_daemon_running
from gym_gui.services.trainer.streams import TelemetryAsyncHub
from gym_gui.telemetry import TelemetrySQLiteStore
from gym_gui.telemetry.db_sink import TelemetryDBSink
from gym_gui.telemetry.run_bus import get_bus


def bootstrap_default_services() -> ServiceLocator:
    """Register the default service set into the shared locator."""

    locator = get_service_locator()

    storage = StorageRecorderService()
    telemetry = TelemetryService()
    telemetry.attach_storage(storage)
    ensure_var_directories()
    telemetry_db = VAR_TELEMETRY_DIR / "telemetry.sqlite"
    telemetry_store = TelemetrySQLiteStore(telemetry_db)
    telemetry.attach_store(telemetry_store)
    if os.getenv("GYM_GUI_RESET_TELEMETRY") == "1":
        telemetry_store.delete_all_episodes(wait=True)

    legacy_trainer_db = VAR_ROOT / "trainer" / "trainer.sqlite"
    if legacy_trainer_db.exists():
        logging.getLogger(__name__).warning(
            "Detected legacy trainer database at %s – new runs write to %s. Consider migrating or deleting the old file to avoid confusion.",
            legacy_trainer_db,
            VAR_TELEMETRY_DIR.parent / "trainer" / "trainer.sqlite",
        )

    actors = ActorService()
    actors.register_actor(
        HumanKeyboardActor(),
        display_name="Human (Keyboard)",
        description="Forward keyboard input captured by the UI.",
        policy_label="Manual control",
        backend_label="Qt keyboard input",
        activate=True,
    )
    actors.register_actor(
        CleanRLWorkerActor(),
        display_name="CleanRL Worker",
        description="Delegates decisions to a CleanRL policy running in the worker process.",
        policy_label="External CleanRL policy",
        backend_label="Trainer-managed worker",
    )
    actors.register_actor(
        XuanCeWorkerActor(),
        display_name="XuanCe Worker",
        description="Comprehensive RL library with 46+ algorithms (DQN, PPO, SAC, MAPPO, QMIX).",
        policy_label="External XuanCe policy",
        backend_label="Trainer-managed worker",
    )
    actors.register_actor(
        RayRLlibWorkerActor(),
        display_name="Ray RLlib Worker",
        description="Distributed multi-agent RL with various training paradigms.",
        policy_label="External Ray RLlib policy",
        backend_label="Ray distributed runtime",
    )

    # -------------------------------------------------------------------------
    # Operator Service (new abstraction replacing actors in the UI)
    # -------------------------------------------------------------------------
    operators = OperatorService()
    operators.register_operator(
        HumanOperator(),
        display_name=OPERATOR_DISPLAY_NAME_HUMAN,
        description=OPERATOR_DESCRIPTION_HUMAN,
        category=OPERATOR_CATEGORY_HUMAN,
        activate=True,
    )
    operators.register_operator(
        WorkerOperator(
            id="balrog_llm",
            name="BALROG LLM",
            worker_id=WORKER_ID_BALROG,
        ),
        display_name=OPERATOR_DISPLAY_NAME_BALROG_LLM,
        description=OPERATOR_DESCRIPTION_BALROG_LLM,
        category=OPERATOR_CATEGORY_LLM,
        requires_api_key=True,
    )
    # Register existing workers as operators for backward compatibility
    operators.register_operator(
        WorkerOperator(id="cleanrl_worker", name="CleanRL Worker", worker_id="cleanrl_worker"),
        display_name="CleanRL Worker",
        description="Delegates decisions to a CleanRL policy running in the worker process.",
        category=OPERATOR_CATEGORY_RL,
        supports_training=True,
    )
    operators.register_operator(
        WorkerOperator(id="xuance_worker", name="XuanCe Worker", worker_id="xuance_worker"),
        display_name="XuanCe Worker",
        description="Comprehensive RL library with 46+ algorithms (DQN, PPO, SAC, MAPPO, QMIX).",
        category=OPERATOR_CATEGORY_RL,
        supports_training=True,
    )
    operators.register_operator(
        WorkerOperator(id="ray_worker", name="Ray RLlib Worker", worker_id="ray_worker"),
        display_name="Ray RLlib Worker",
        description="Distributed multi-agent RL with various training paradigms.",
        category=OPERATOR_CATEGORY_RL,
        supports_training=True,
    )
    operators.register_operator(
        WorkerOperator(id="tianshou_worker", name="Tianshou Worker", worker_id="tianshou_worker"),
        display_name="Tianshou Worker",
        description="Deep RL platform based on PyTorch with 20+ algorithms (PPO, DQN, SAC, TD3, A2C).",
        category=OPERATOR_CATEGORY_RL,
        supports_training=True,
    )
    operators.register_operator(
        WorkerOperator(id="sb3_worker", name="Stable-Baselines3 Worker", worker_id="sb3_worker"),
        display_name="Stable-Baselines3 Worker",
        description="Reliable RL implementations (PPO, A2C, DQN, SAC, TD3, HER). PyTorch backend.",
        category=OPERATOR_CATEGORY_RL,
        supports_training=True,
    )
    operators.register_operator(
        WorkerOperator(id="sbx_worker", name="SBX (JAX) Worker", worker_id="sbx_worker"),
        display_name="SBX (JAX) Worker",
        description="JAX-accelerated RL (PPO, DQN, SAC, TQC, CrossQ). Drop-in SB3 replacement.",
        category=OPERATOR_CATEGORY_RL,
        supports_training=True,
    )
    operators.register_operator(
        WorkerOperator(id="mushroomrl_worker", name="MushroomRL Worker", worker_id="mushroomrl_worker"),
        display_name="MushroomRL Worker",
        description="Model-free and model-based RL (PPO, TRPO, SAC, TD3). Continuous control focus.",
        category=OPERATOR_CATEGORY_RL,
        supports_training=True,
    )
    operators.register_operator(
        WorkerOperator(id="pearl_worker", name="Pearl Worker", worker_id="pearl_worker"),
        display_name="Pearl Worker",
        description="Production-ready RL from Meta (PPO, DQN, SAC, bandits). Modular agent design.",
        category=OPERATOR_CATEGORY_RL,
        supports_training=True,
    )
    operators.register_operator(
        WorkerOperator(id="torchrl_worker", name="TorchRL Worker", worker_id="torchrl_worker"),
        display_name="TorchRL Worker",
        description="PyTorch official RL library with modular collector/loss/advantage design.",
        category=OPERATOR_CATEGORY_RL,
        supports_training=True,
    )
    operators.register_operator(
        WorkerOperator(id="omnisafe_worker", name="OmniSafe Worker", worker_id="omnisafe_worker"),
        display_name="OmniSafe Worker",
        description="Safe RL (PPOLag, CPO, FOCOPS, CUP). 30+ constrained RL algorithms.",
        category=OPERATOR_CATEGORY_RL,
        supports_training=True,
    )
    operators.register_operator(
        WorkerOperator(id="mctx_worker", name="MCTX GPU Worker", worker_id="mctx_worker"),
        display_name="MCTX GPU Worker",
        description="GPU-accelerated MCTS (AlphaZero, MuZero, Gumbel MuZero) via JAX.",
        category=OPERATOR_CATEGORY_RL,
        supports_training=True,
    )
    operators.register_operator(
        WorkerOperator(id="marllib_worker", name="MARLlib Worker", worker_id="marllib_worker"),
        display_name="MARLlib Worker",
        description="Multi-Agent RLlib with 18 algorithms across IL, CC, and VD paradigms.",
        category=OPERATOR_CATEGORY_RL,
        supports_training=True,
    )
    operators.register_operator(
        WorkerOperator(id="chess_worker", name="Chess LLM Worker", worker_id="chess_worker"),
        display_name="Chess LLM Worker",
        description="LLM-based chess using multi-turn prompting. Works with PettingZoo chess_v6.",
        category=OPERATOR_CATEGORY_LLM,
    )

    action_mapper: ContinuousActionMapper = create_default_action_mapper()
    renderer_registry: RendererRegistry = create_default_renderer_registry()  # default strategies

    # Create PolicyMappingService wrapping ActorService for multi-agent support
    policy_mapping = PolicyMappingService(actors)

    locator.register(StorageRecorderService, storage)
    locator.register(TelemetryService, telemetry)
    locator.register(TelemetrySQLiteStore, telemetry_store)
    locator.register(ActorService, actors)
    locator.register(OperatorService, operators)
    locator.register(PolicyMappingService, policy_mapping)
    locator.register(ContinuousActionMapper, action_mapper)

    client_config = TrainerClientConfig()

    daemon_handle = ensure_trainer_daemon_running(target=client_config.target)

    trainer_client = TrainerClient(client_config)
    trainer_runner = TrainerClientRunner(trainer_client)

    # Create run registry for training run management
    # Uses the same database as the trainer daemon (VAR_TRAINER_DIR/trainer.sqlite)
    run_registry = RunRegistry()

    # Initialize telemetry hub for live streaming
    # Increased buffer_size from 256 to 512 to handle high-frequency training (episode 658+)
    # max_queue=2048 handles gRPC stream buffering, buffer_size=512 handles UI processing lag
    telemetry_hub = TelemetryAsyncHub(
        max_queue=TELEMETRY_HUB_MAX_QUEUE,
        buffer_size=TELEMETRY_HUB_BUFFER_SIZE,
    )
    # Hub will auto-start on first subscribe_run call

    # Create live telemetry controller and start RunBus subscription
    # UI queue size: 64 events (responsive rendering)
    live_controller = LiveTelemetryController(telemetry_hub, trainer_client)
    live_controller.start()

    # Initialize and start database sink for durable persistence
    # Writer queue is larger (512) to handle backlog, UI queue is smaller (64)
    # HDF5 replay storage enabled: frames/observations → var/replay/, SQLite stores refs only
    bus = get_bus()
    db_sink = TelemetryDBSink(
        telemetry_store,
        bus,
        batch_size=DB_SINK_BATCH_SIZE,
        checkpoint_interval=DB_SINK_CHECKPOINT_INTERVAL,
        writer_queue_size=DB_SINK_WRITER_QUEUE_SIZE,
        replay_dir=VAR_REPLAY_DIR,  # Enable HDF5 storage for frames/observations
    )
    db_sink.start()

    locator.register(RendererRegistry, renderer_registry)
    locator.register(TrainerClient, trainer_client)
    locator.register(TrainerClientRunner, trainer_runner)
    locator.register(RunRegistry, run_registry)
    locator.register(TelemetryAsyncHub, telemetry_hub)
    locator.register(LiveTelemetryController, live_controller)
    locator.register(TelemetryDBSink, db_sink)
    locator.register(TrainerDaemonHandle, daemon_handle)

    # Also register under string keys for convenience in legacy code.
    locator.register("storage", storage)
    locator.register("telemetry", telemetry)
    locator.register("telemetry_store", telemetry_store)
    locator.register("actors", actors)
    locator.register("operators", operators)
    locator.register("policy_mapping", policy_mapping)
    locator.register("action_mapper", action_mapper)
    locator.register("renderer_registry", renderer_registry)
    locator.register("trainer_client", trainer_client)
    locator.register("trainer_client_runner", trainer_runner)
    locator.register("run_registry", run_registry)
    locator.register("telemetry_hub", telemetry_hub)
    locator.register("live_telemetry_controller", live_controller)
    locator.register("trainer_daemon_handle", daemon_handle)

    return locator


__all__ = ["bootstrap_default_services"]
