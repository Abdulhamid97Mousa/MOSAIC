"""Shared structured log constants for the Gym GUI project.

This module centralises the error and event identifiers used across the
application so that emitters can attach consistent metadata (component,
subcomponent, tags) alongside a stable log code and human readable message.

Each constant is represented by :class:`LogConstant`, which aligns with the
expectations of the helper utilities defined throughout the codebase.

## Usage

Log constants are designed to be used with the `_log_constant()` helper pattern:

    from gym_gui.logging_config.log_constants import LOG_SESSION_STEP_ERROR, _log_constant

    try:
        adapter.step()
    except Exception as exc:
        _log_constant(LOG_SESSION_STEP_ERROR, exc_info=exc, extra={"run_id": run_id})

## Lookup & Filtering

The module provides helpers for discovering constants at runtime:

    from gym_gui.logging_config.log_constants import get_constant_by_code, list_known_components

    # Retrieve a constant by code
    const = get_constant_by_code("LOG401")  # → LOG_SESSION_ADAPTER_LOAD_ERROR

    # List all known components
    components = list_known_components()  # → ["Controller", "Adapter", "Service", ...]

    # Build dynamic component/severity filters for the GUI
    component_snapshot = get_component_snapshot()  # → {"Controller": {"Session", "LiveTelemetry", ...}, ...}

## Validation

To validate that all log constants conform to logging standards:

    from gym_gui.logging_config.log_constants import validate_log_constants

    errors = validate_log_constants()
    if errors:
        raise ValueError(f"Invalid log constants: {errors}")

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Set, Tuple


@dataclass(frozen=True, slots=True)
class LogConstant:
    """Container describing a structured log record template."""

    code: str
    level: int | str
    message: str
    component: str
    subcomponent: str
    tags: tuple[str, ...] = ()


def _tags(*values: str) -> tuple[str, ...]:
    return tuple(v for v in values if v)


def _constant(
    code: str,
    level: int | str,
    message: str,
    *,
    component: str,
    subcomponent: str,
    tags: Iterable[str] = (),
) -> LogConstant:
    return LogConstant(
        code=code,
        level=level,
        message=message,
        component=component,
        subcomponent=subcomponent,
        tags=tuple(tags),
    )


# ---------------------------------------------------------------------------
# Controller constants (LOG401–LOG429)
# ---------------------------------------------------------------------------
LOG_SESSION_ADAPTER_LOAD_ERROR = _constant(
    "LOG401",
    "ERROR",
    "Session failed to load adapter",
    component="Controller",
    subcomponent="Session",
    tags=_tags("session", "adapter", "error"),
)

LOG_SESSION_STEP_ERROR = _constant(
    "LOG402",
    "ERROR",
    "Adapter step raised an exception",
    component="Controller",
    subcomponent="Session",
    tags=_tags("session", "adapter", "step"),
)

LOG_SESSION_EPISODE_ERROR = _constant(
    "LOG403",
    "ERROR",
    "Failed to finalise episode state",
    component="Controller",
    subcomponent="Session",
    tags=_tags("session", "episode"),
)

LOG_SESSION_TIMER_PRECISION_WARNING = _constant(
    "LOG404",
    "WARNING",
    "Session timer precision degraded",
    component="Controller",
    subcomponent="Session",
    tags=_tags("session", "timer"),
)

LOG_INPUT_CONTROLLER_ERROR = _constant(
    "LOG405",
    "ERROR",
    "Input controller encountered an error",
    component="Controller",
    subcomponent="Input",
    tags=_tags("input", "controller"),
)

LOG_KEYBOARD_DETECTED = _constant(
    "LOG406",
    "INFO",
    "Keyboard device detected for multi-human gameplay",
    component="Controller",
    subcomponent="Input",
    tags=_tags("keyboard", "multi-agent", "device"),
)

LOG_KEYBOARD_ASSIGNED = _constant(
    "LOG407",
    "INFO",
    "Keyboard assigned to agent",
    component="Controller",
    subcomponent="Input",
    tags=_tags("keyboard", "multi-agent", "assignment"),
)

LOG_KEYBOARD_DETECTION_ERROR = _constant(
    "LOG4071",
    "ERROR",
    "Keyboard detection failed",
    component="Controller",
    subcomponent="Input",
    tags=_tags("keyboard", "multi-agent", "error"),
)

LOG_KEYBOARD_EVDEV_SETUP_START = _constant(
    "LOG4072",
    "INFO",
    "Starting evdev keyboard monitoring setup",
    component="Controller",
    subcomponent="Input",
    tags=_tags("keyboard", "evdev", "multi-agent", "setup"),
)

LOG_KEYBOARD_EVDEV_SETUP_SUCCESS = _constant(
    "LOG4073",
    "INFO",
    "Evdev keyboard monitoring setup completed successfully",
    component="Controller",
    subcomponent="Input",
    tags=_tags("keyboard", "evdev", "multi-agent", "setup"),
)

LOG_KEYBOARD_EVDEV_SETUP_FAILED = _constant(
    "LOG4074",
    "ERROR",
    "Evdev keyboard monitoring setup failed",
    component="Controller",
    subcomponent="Input",
    tags=_tags("keyboard", "evdev", "multi-agent", "error"),
)

LOG_KEY_RESOLVER_INITIALIZED = _constant(
    "LOG4075",
    "INFO",
    "Key combination resolver initialized for game",
    component="Controller",
    subcomponent="Input",
    tags=_tags("input", "keyboard", "resolver"),
)

LOG_KEY_RESOLVER_UNAVAILABLE = _constant(
    "LOG4076",
    "WARNING",
    "No key combination resolver available for game",
    component="Controller",
    subcomponent="Input",
    tags=_tags("input", "keyboard", "resolver", "warning"),
)

LOG_INPUT_MODE_CONFIGURED = _constant(
    "LOG4077",
    "INFO",
    "Input mode configured for game",
    component="Controller",
    subcomponent="Input",
    tags=_tags("input", "keyboard", "mode"),
)

LOG_EVDEV_KEY_PRESSED = _constant(
    "LOG4078",
    "INFO",
    "Evdev key pressed",
    component="Controller",
    subcomponent="Input",
    tags=_tags("evdev", "keyboard", "key", "multi-agent"),
)

LOG_EVDEV_KEY_RELEASED = _constant(
    "LOG4079",
    "DEBUG",
    "Evdev key released",
    component="Controller",
    subcomponent="Input",
    tags=_tags("evdev", "keyboard", "key", "multi-agent"),
)

# Human Control keyboard worker subprocess events (LOG48xx)
LOG_HUMAN_CONTROL_WORKER_LAUNCH = _constant(
    "LOG4801",
    "INFO",
    "Human control keyboard worker launched",
    component="Controller",
    subcomponent="HumanControl",
    tags=_tags("human_control", "keyboard", "subprocess", "lifecycle"),
)
LOG_HUMAN_CONTROL_WORKER_READY = _constant(
    "LOG4802",
    "INFO",
    "Human control keyboard worker initialized and ready",
    component="Controller",
    subcomponent="HumanControl",
    tags=_tags("human_control", "keyboard", "subprocess", "lifecycle"),
)
LOG_HUMAN_CONTROL_WORKER_ACTION = _constant(
    "LOG4803",
    "DEBUG",
    "Action received from keyboard worker",
    component="Controller",
    subcomponent="HumanControl",
    tags=_tags("human_control", "keyboard", "action"),
)
LOG_HUMAN_CONTROL_WORKER_STEP = _constant(
    "LOG4804",
    "INFO",
    "All keyboard worker actions collected, stepping environment",
    component="Controller",
    subcomponent="HumanControl",
    tags=_tags("human_control", "keyboard", "step"),
)
LOG_HUMAN_CONTROL_WORKER_STOPPED = _constant(
    "LOG4805",
    "INFO",
    "Human control keyboard worker stopped",
    component="Controller",
    subcomponent="HumanControl",
    tags=_tags("human_control", "keyboard", "subprocess", "lifecycle"),
)
LOG_HUMAN_CONTROL_WORKER_ERROR = _constant(
    "LOG4806",
    "ERROR",
    "Human control keyboard worker error",
    component="Controller",
    subcomponent="HumanControl",
    tags=_tags("human_control", "keyboard", "error"),
)
LOG_HUMAN_CONTROL_BRIDGE_START = _constant(
    "LOG4810",
    "INFO",
    "Human control keyboard bridge started",
    component="Controller",
    subcomponent="HumanControl",
    tags=_tags("human_control", "bridge", "lifecycle"),
)
LOG_HUMAN_CONTROL_BRIDGE_STOP = _constant(
    "LOG4811",
    "INFO",
    "Human control keyboard bridge stopped",
    component="Controller",
    subcomponent="HumanControl",
    tags=_tags("human_control", "bridge", "lifecycle"),
)
LOG_HUMAN_CONTROL_AUTO_ASSIGN = _constant(
    "LOG4812",
    "INFO",
    "Auto-assigned keyboards to agents",
    component="Controller",
    subcomponent="HumanControl",
    tags=_tags("human_control", "keyboard", "assignment"),
)
LOG_HUMAN_CONTROL_INPUT_DISABLED = _constant(
    "LOG4813",
    "INFO",
    "Main-thread keyboard input disabled (subprocess workers active)",
    component="Controller",
    subcomponent="HumanControl",
    tags=_tags("human_control", "keyboard", "lifecycle"),
)

LOG_HUMAN_CONTROL_RAW_KEY_NO_HANDLER = _constant(
    "LOG4814",
    "WARNING",
    "Raw key event received but no native handler available",
    component="Controller",
    subcomponent="HumanControl",
    tags=_tags("human_control", "keyboard", "native_input"),
)

LOG_LIVE_CONTROLLER_INITIALIZED = _constant(
    "LOG408",
    "INFO",
    "Live telemetry controller initialised",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "lifecycle"),
)

LOG_LIVE_CONTROLLER_THREAD_STARTED = _constant(
    "LOG409",
    "INFO",
    "Live telemetry worker thread started",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "thread"),
)

LOG_LIVE_CONTROLLER_THREAD_STOPPED = _constant(
    "LOG410",
    "INFO",
    "Live telemetry worker thread stopped",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "thread"),
)

LOG_LIVE_CONTROLLER_THREAD_STOP_TIMEOUT = _constant(
    "LOG411",
    "WARNING",
    "Timed out waiting for live telemetry thread to stop",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "thread", "timeout"),
)

LOG_LIVE_CONTROLLER_ALREADY_RUNNING = _constant(
    "LOG412",
    "WARNING",
    "Live telemetry controller already running",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "lifecycle"),
)

LOG_LIVE_CONTROLLER_RUN_SUBSCRIBED = _constant(
    "LOG413",
    "INFO",
    "Subscribed to run telemetry stream",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "subscription"),
)

LOG_LIVE_CONTROLLER_RUN_UNSUBSCRIBED = _constant(
    "LOG414",
    "INFO",
    "Unsubscribed from run telemetry stream",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "subscription"),
)

LOG_LIVE_CONTROLLER_RUN_ALREADY_SUBSCRIBED = _constant(
    "LOG415",
    "WARNING",
    "Run telemetry stream already subscribed",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "subscription"),
)

LOG_LIVE_CONTROLLER_RUNBUS_SUBSCRIBED = _constant(
    "LOG416",
    "INFO",
    "RunBus subscription established",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "runbus"),
)

LOG_LIVE_CONTROLLER_RUN_COMPLETED = _constant(
    "LOG417",
    "INFO",
    "Run telemetry stream completed",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "lifecycle"),
)

LOG_LIVE_CONTROLLER_QUEUE_OVERFLOW = _constant(
    "LOG418",
    "WARNING",
    "Live telemetry queue overflowed",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "queue", "overflow"),
)

LOG_LIVE_CONTROLLER_BUFFER_STEPS_FLUSHED = _constant(
    "LOG419",
    "INFO",
    "Flushed buffered step events to UI",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "buffer", "step"),
)

LOG_LIVE_CONTROLLER_BUFFER_EPISODES_FLUSHED = _constant(
    "LOG420",
    "INFO",
    "Flushed buffered episode events to UI",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "buffer", "episode"),
)

LOG_BUFFER_DROP = _constant(
    "LOG421",
    "WARNING",
    "Dropped buffered telemetry event due to limit",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "buffer", "drop"),
)

LOG_TELEMETRY_CONTROLLER_THREAD_ERROR = _constant(
    "LOG422",
    "ERROR",
    "Telemetry controller thread crashed",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "thread", "error"),
)

LOG_LIVE_CONTROLLER_SIGNAL_EMIT_FAILED = _constant(
    "LOG423",
    "ERROR",
    "Failed to emit telemetry update signal",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "qt", "signal"),
)

LOG_LIVE_CONTROLLER_TAB_ADD_FAILED = _constant(
    "LOG424",
    "ERROR",
    "Failed to add telemetry tab to UI",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "ui"),
)

LOG_CREDIT_STARVED = _constant(
    "LOG425",
    "INFO",
    "Telemetry credits exhausted for stream",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "credits", "backpressure"),
)

LOG_CREDIT_RESUMED = _constant(
    "LOG426",
    "INFO",
    "Telemetry credits replenished for stream",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "credits", "backpressure"),
)

LOG_LIVE_CONTROLLER_LOOP_EXITED = _constant(
    "LOG427",
    "INFO",
    "Telemetry controller loop exited",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "lifecycle"),
)

LOG_TELEMETRY_SUBSCRIBE_ERROR = _constant(
    "LOG428",
    "ERROR",
    "Failed to subscribe to telemetry stream",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "subscription", "error"),
)

LOG_LIVE_CONTROLLER_EVENT_FOR_DELETED_RUN = _constant(
    "LOG429",
    "WARNING",
    "Received telemetry for run marked deleted",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "deleted", "diagnostics"),
)

LOG_LIVE_CONTROLLER_TAB_REQUESTED = _constant(
    "LOG430",
    "INFO",
    "Requested creation of live telemetry tab",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "tab", "request"),
)


# ---------------------------------------------------------------------------
# Adapter constants (LOG501–LOG513)
# ---------------------------------------------------------------------------
LOG_ADAPTER_INIT_ERROR = _constant(
    "LOG501",
    "ERROR",
    "Adapter initialisation failed",
    component="Adapter",
    subcomponent="Lifecycle",
    tags=_tags("adapter", "initialise"),
)

LOG_ADAPTER_STEP_ERROR = _constant(
    "LOG502",
    "ERROR",
    "Adapter step execution failed",
    component="Adapter",
    subcomponent="Step",
    tags=_tags("adapter", "step"),
)

LOG_ADAPTER_PAYLOAD_ERROR = _constant(
    "LOG503",
    "ERROR",
    "Adapter produced invalid payload",
    component="Adapter",
    subcomponent="Payload",
    tags=_tags("adapter", "payload"),
)

LOG_ADAPTER_STATE_INVALID = _constant(
    "LOG504",
    "ERROR",
    "Adapter emitted invalid state snapshot",
    component="Adapter",
    subcomponent="State",
    tags=_tags("adapter", "state"),
)

LOG_ADAPTER_RENDER_ERROR = _constant(
    "LOG505",
    "ERROR",
    "Adapter render call failed",
    component="Adapter",
    subcomponent="Render",
    tags=_tags("adapter", "render"),
)

LOG_ADAPTER_ENV_CREATED = _constant(
    "LOG510",
    "INFO",
    "Adapter environment created",
    component="Adapter",
    subcomponent="Lifecycle",
    tags=_tags("adapter", "environment"),
)

LOG_ADAPTER_ENV_RESET = _constant(
    "LOG511",
    "INFO",
    "Adapter environment reset",
    component="Adapter",
    subcomponent="Lifecycle",
    tags=_tags("adapter", "environment"),
)

LOG_ADAPTER_STEP_SUMMARY = _constant(
    "LOG512",
    "DEBUG",
    "Adapter step summary",
    component="Adapter",
    subcomponent="Step",
    tags=_tags("adapter", "step"),
)

LOG_ADAPTER_ENV_CLOSED = _constant(
    "LOG513",
    "INFO",
    "Adapter environment closed",
    component="Adapter",
    subcomponent="Lifecycle",
    tags=_tags("adapter", "environment"),
)

LOG_ADAPTER_MAP_GENERATION = _constant(
    "LOG514",
    "INFO",
    "Adapter map generation details",
    component="Adapter",
    subcomponent="MapGeneration",
    tags=_tags("adapter", "map", "generation"),
)

LOG_ADAPTER_HOLE_PLACEMENT = _constant(
    "LOG515",
    "DEBUG",
    "Hole placement configuration",
    component="Adapter",
    subcomponent="MapGeneration",
    tags=_tags("adapter", "map", "holes"),
)

LOG_ADAPTER_GOAL_OVERRIDE = _constant(
    "LOG516",
    "INFO",
    "Goal position override applied",
    component="Adapter",
    subcomponent="MapGeneration",
    tags=_tags("adapter", "map", "goal"),
)

LOG_ADAPTER_RENDER_PAYLOAD = _constant(
    "LOG517",
    "DEBUG",
    "Adapter render payload generated",
    component="Adapter",
    subcomponent="Render",
    tags=_tags("adapter", "render", "payload"),
)


LOG_ENV_MINIGRID_BOOT = _constant(
    "LOG518",
    "INFO",
    "MiniGrid adapter bootstrapped",
    component="Adapter",
    subcomponent="MiniGrid",
    tags=_tags("minigrid", "environment", "boot"),
)

LOG_ENV_MINIGRID_STEP = _constant(
    "LOG519",
    "DEBUG",
    "MiniGrid step checkpoint",
    component="Adapter",
    subcomponent="MiniGrid",
    tags=_tags("minigrid", "step"),
)

LOG_ENV_MINIGRID_ERROR = _constant(
    "LOG520",
    "ERROR",
    "MiniGrid adapter failure",
    component="Adapter",
    subcomponent="MiniGrid",
    tags=_tags("minigrid", "error"),
)

LOG_ENV_MINIGRID_RENDER_WARNING = _constant(
    "LOG521",
    "WARNING",
    "MiniGrid render payload unavailable",
    component="Adapter",
    subcomponent="MiniGrid",
    tags=_tags("minigrid", "render", "warning"),
)


# ALE adapter specific constants
LOG_ADAPTER_ALE_NAMESPACE_IMPORT_FAILED = _constant(
    "LOG522",
    "WARNING",
    "ALE optional namespace import failed; environments may not be auto-registered",
    component="Adapter",
    subcomponent="ALE",
    tags=_tags("ale", "import", "namespace", "warning"),
)

LOG_ADAPTER_ALE_METADATA_PROBE_FAILED = _constant(
    "LOG523",
    "DEBUG",
    "ALE metadata probe failed; continuing without optional metrics",
    component="Adapter",
    subcomponent="ALE",
    tags=_tags("ale", "metadata", "probe"),
)


# MOSAIC MultiGrid event tracking constants (LOG525-LOG527)
LOG_MOSAIC_MULTIGRID_GOAL_SCORED = _constant(
    "LOG525",
    "INFO",
    "MOSAIC MultiGrid goal scored",
    component="Adapter",
    subcomponent="MosaicMultiGrid",
    tags=_tags("mosaic", "multigrid", "goal", "event"),
)

LOG_MOSAIC_MULTIGRID_PASS_COMPLETED = _constant(
    "LOG526",
    "INFO",
    "MOSAIC MultiGrid pass completed",
    component="Adapter",
    subcomponent="MosaicMultiGrid",
    tags=_tags("mosaic", "multigrid", "pass", "event"),
)

LOG_MOSAIC_MULTIGRID_STEAL_COMPLETED = _constant(
    "LOG527",
    "INFO",
    "MOSAIC MultiGrid steal completed",
    component="Adapter",
    subcomponent="MosaicMultiGrid",
    tags=_tags("mosaic", "multigrid", "steal", "event"),
)

LOG_MOSAIC_MULTIGRID_VISIBILITY = _constant(
    "LOG528",
    "INFO",
    "MOSAIC MultiGrid agent visibility",
    component="Adapter",
    subcomponent="MosaicMultiGrid",
    tags=_tags("mosaic", "multigrid", "visibility", "observation"),
)

LOG_MOSAIC_MULTIGRID_OBSERVATION = _constant(
    "LOG529",
    "DEBUG",
    "MOSAIC MultiGrid agent observation grid",
    component="Adapter",
    subcomponent="MosaicMultiGrid",
    tags=_tags("mosaic", "multigrid", "observation", "grid"),
)


# ---------------------------------------------------------------------------
# Crafter adapter specific constants (LOG530–LOG534)
# ---------------------------------------------------------------------------
LOG_ENV_CRAFTER_BOOT = _constant(
    "LOG530",
    "INFO",
    "Crafter adapter bootstrapped",
    component="Adapter",
    subcomponent="Crafter",
    tags=_tags("crafter", "environment", "boot"),
)

LOG_ENV_CRAFTER_STEP = _constant(
    "LOG531",
    "DEBUG",
    "Crafter step checkpoint",
    component="Adapter",
    subcomponent="Crafter",
    tags=_tags("crafter", "step"),
)

LOG_ENV_CRAFTER_ERROR = _constant(
    "LOG532",
    "ERROR",
    "Crafter adapter failure",
    component="Adapter",
    subcomponent="Crafter",
    tags=_tags("crafter", "error"),
)

LOG_ENV_CRAFTER_RENDER_WARNING = _constant(
    "LOG533",
    "WARNING",
    "Crafter render payload unavailable",
    component="Adapter",
    subcomponent="Crafter",
    tags=_tags("crafter", "render", "warning"),
)

LOG_ENV_CRAFTER_ACHIEVEMENT = _constant(
    "LOG534",
    "INFO",
    "Crafter achievement unlocked",
    component="Adapter",
    subcomponent="Crafter",
    tags=_tags("crafter", "achievement", "milestone"),
)


# ---------------------------------------------------------------------------
# Procgen adapter specific constants (LOG540–LOG544)
# ---------------------------------------------------------------------------
LOG_ENV_PROCGEN_BOOT = _constant(
    "LOG540",
    "INFO",
    "Procgen adapter bootstrapped",
    component="Adapter",
    subcomponent="Procgen",
    tags=_tags("procgen", "bootstrap", "lifecycle"),
)

LOG_ENV_PROCGEN_STEP = _constant(
    "LOG541",
    "DEBUG",
    "Procgen step checkpoint",
    component="Adapter",
    subcomponent="Procgen",
    tags=_tags("procgen", "step", "checkpoint"),
)

LOG_ENV_PROCGEN_ERROR = _constant(
    "LOG542",
    "ERROR",
    "Procgen environment error",
    component="Adapter",
    subcomponent="Procgen",
    tags=_tags("procgen", "error"),
)

LOG_ENV_PROCGEN_RENDER_WARNING = _constant(
    "LOG543",
    "WARNING",
    "Procgen render mode constraint",
    component="Adapter",
    subcomponent="Procgen",
    tags=_tags("procgen", "render", "warning"),
)

LOG_ENV_PROCGEN_LEVEL_COMPLETE = _constant(
    "LOG544",
    "INFO",
    "Procgen level completed",
    component="Adapter",
    subcomponent="Procgen",
    tags=_tags("procgen", "level", "complete"),
)


# ---------------------------------------------------------------------------
# TextWorld adapter specific constants (LOG550–LOG554)
# ---------------------------------------------------------------------------
LOG_ENV_TEXTWORLD_BOOT = _constant(
    "LOG550",
    "INFO",
    "TextWorld adapter bootstrapped",
    component="Adapter",
    subcomponent="TextWorld",
    tags=_tags("textworld", "bootstrap", "lifecycle"),
)

LOG_ENV_TEXTWORLD_STEP = _constant(
    "LOG551",
    "DEBUG",
    "TextWorld step checkpoint",
    component="Adapter",
    subcomponent="TextWorld",
    tags=_tags("textworld", "step", "checkpoint"),
)

LOG_ENV_TEXTWORLD_ERROR = _constant(
    "LOG552",
    "ERROR",
    "TextWorld environment error",
    component="Adapter",
    subcomponent="TextWorld",
    tags=_tags("textworld", "error"),
)

LOG_ENV_TEXTWORLD_GAME_GENERATED = _constant(
    "LOG553",
    "INFO",
    "TextWorld game generated",
    component="Adapter",
    subcomponent="TextWorld",
    tags=_tags("textworld", "game", "generation"),
)

LOG_ENV_TEXTWORLD_COMMAND = _constant(
    "LOG554",
    "DEBUG",
    "TextWorld command executed",
    component="Adapter",
    subcomponent="TextWorld",
    tags=_tags("textworld", "command", "action"),
)


# ---------------------------------------------------------------------------
# MalmoEnv (Minecraft) adapter specific constants (LOG560–LOG569)
# ---------------------------------------------------------------------------
LOG_ENV_MALMO_BOOT = _constant(
    "LOG560",
    "INFO",
    "MalmoEnv adapter bootstrapped",
    component="Adapter",
    subcomponent="MalmoEnv",
    tags=_tags("malmo", "minecraft", "bootstrap", "lifecycle"),
)

LOG_ENV_MALMO_STEP = _constant(
    "LOG561",
    "DEBUG",
    "MalmoEnv step checkpoint",
    component="Adapter",
    subcomponent="MalmoEnv",
    tags=_tags("malmo", "minecraft", "step", "checkpoint"),
)

LOG_ENV_MALMO_ERROR = _constant(
    "LOG562",
    "ERROR",
    "MalmoEnv environment error",
    component="Adapter",
    subcomponent="MalmoEnv",
    tags=_tags("malmo", "minecraft", "error"),
)

LOG_ENV_MALMO_CONNECTION_REFUSED = _constant(
    "LOG563",
    "WARNING",
    "MalmoEnv TCP connection refused - Minecraft may not be ready",
    component="Adapter",
    subcomponent="MalmoEnv",
    tags=_tags("malmo", "minecraft", "connection", "tcp"),
)

LOG_ENV_MALMO_CONNECTION_TIMEOUT = _constant(
    "LOG564",
    "WARNING",
    "MalmoEnv TCP connection timed out",
    component="Adapter",
    subcomponent="MalmoEnv",
    tags=_tags("malmo", "minecraft", "connection", "timeout"),
)

LOG_ENV_MALMO_MISSION_INIT = _constant(
    "LOG565",
    "INFO",
    "MalmoEnv mission initialized",
    component="Adapter",
    subcomponent="MalmoEnv",
    tags=_tags("malmo", "minecraft", "mission", "init"),
)

LOG_ENV_MALMO_MISSION_RESET = _constant(
    "LOG566",
    "INFO",
    "MalmoEnv mission reset",
    component="Adapter",
    subcomponent="MalmoEnv",
    tags=_tags("malmo", "minecraft", "mission", "reset"),
)

LOG_ENV_MALMO_MISSION_END = _constant(
    "LOG567",
    "INFO",
    "MalmoEnv mission ended",
    component="Adapter",
    subcomponent="MalmoEnv",
    tags=_tags("malmo", "minecraft", "mission", "end"),
)

LOG_ENV_MALMO_RENDER_FRAME = _constant(
    "LOG568",
    "DEBUG",
    "MalmoEnv render frame received",
    component="Adapter",
    subcomponent="MalmoEnv",
    tags=_tags("malmo", "minecraft", "render", "frame"),
)

LOG_ENV_MALMO_PACKAGE_MISSING = _constant(
    "LOG569",
    "ERROR",
    "MalmoEnv package not installed - run pip install",
    component="Adapter",
    subcomponent="MalmoEnv",
    tags=_tags("malmo", "minecraft", "import", "error"),
)

# Malmo interaction / input constants (LOG570–LOG579)

LOG_MALMO_PASSIVE_ACTION_RESOLVED = _constant(
    "LOG570",
    "DEBUG",
    "Malmo passive action resolved",
    component="Controller",
    subcomponent="MalmoInteraction",
    tags=_tags("malmo", "minecraft", "input", "passive", "action"),
)

LOG_MALMO_PASSIVE_ACTION_FALLBACK = _constant(
    "LOG571",
    "WARNING",
    "Malmo passive action fallback — no noop/move 0/turn 0 in action space",
    component="Controller",
    subcomponent="MalmoInteraction",
    tags=_tags("malmo", "minecraft", "input", "passive", "fallback"),
)

LOG_MALMO_ACTION_SPACE_DUMP = _constant(
    "LOG572",
    "DEBUG",
    "Malmo action space contents",
    component="Controller",
    subcomponent="MalmoInteraction",
    tags=_tags("malmo", "minecraft", "input", "action_space", "debug"),
)

LOG_MALMO_NATIVE_KEY_SENT = _constant(
    "LOG573",
    "DEBUG",
    "Malmo native key event sent to Java",
    component="Controller",
    subcomponent="MalmoInteraction",
    tags=_tags("malmo", "minecraft", "input", "key", "native"),
)

LOG_MALMO_NATIVE_KEY_UNMAPPED = _constant(
    "LOG574",
    "WARNING",
    "Malmo native key unmapped — Qt key has no LWJGL equivalent",
    component="Controller",
    subcomponent="MalmoInteraction",
    tags=_tags("malmo", "minecraft", "input", "key", "unmapped"),
)

LOG_MALMO_NATIVE_MOUSE_SENT = _constant(
    "LOG575",
    "DEBUG",
    "Malmo native mouse event sent to Java",
    component="Controller",
    subcomponent="MalmoInteraction",
    tags=_tags("malmo", "minecraft", "input", "mouse", "native"),
)

LOG_MALMO_NATIVE_INPUT_CONNECTED = _constant(
    "LOG576",
    "INFO",
    "Malmo native input handler connected",
    component="Controller",
    subcomponent="MalmoInteraction",
    tags=_tags("malmo", "minecraft", "input", "connection", "established"),
)

LOG_MALMO_NATIVE_INPUT_ERROR = _constant(
    "LOG577",
    "ERROR",
    "Malmo native input handler error",
    component="Controller",
    subcomponent="MalmoInteraction",
    tags=_tags("malmo", "minecraft", "input", "connection", "error"),
)


# ---------------------------------------------------------------------------
# Service and telemetry constants (LOG601–LOG650)
# ---------------------------------------------------------------------------
LOG_SERVICE_TELEMETRY_STEP_REJECTED = _constant(
    "LOG601",
    "WARNING",
    "Telemetry step rejected by validation",
    component="Service",
    subcomponent="Telemetry",
    tags=_tags("telemetry", "validation"),
)

LOG_SERVICE_TELEMETRY_ASYNC_ERROR = _constant(
    "LOG602",
    "ERROR",
    "Telemetry asynchronous persistence failed",
    component="Service",
    subcomponent="Telemetry",
    tags=_tags("telemetry", "async", "error"),
)

LOG_SERVICE_DB_SINK_INITIALIZED = _constant(
    "LOG610",
    "INFO",
    "Telemetry DB sink initialised",
    component="Service",
    subcomponent="DBSink",
    tags=_tags("telemetry", "db", "lifecycle"),
)

LOG_SERVICE_DB_SINK_STARTED = _constant(
    "LOG611",
    "INFO",
    "Telemetry DB sink worker started",
    component="Service",
    subcomponent="DBSink",
    tags=_tags("telemetry", "db", "worker"),
)

LOG_SERVICE_DB_SINK_ALREADY_RUNNING = _constant(
    "LOG612",
    "WARNING",
    "Telemetry DB sink already running",
    component="Service",
    subcomponent="DBSink",
    tags=_tags("telemetry", "db", "lifecycle"),
)

LOG_SERVICE_DB_SINK_STOPPED = _constant(
    "LOG613",
    "INFO",
    "Telemetry DB sink worker stopped",
    component="Service",
    subcomponent="DBSink",
    tags=_tags("telemetry", "db", "worker"),
)

LOG_SERVICE_DB_SINK_STOP_TIMEOUT = _constant(
    "LOG614",
    "WARNING",
    "Timed out waiting for telemetry DB sink to stop",
    component="Service",
    subcomponent="DBSink",
    tags=_tags("telemetry", "db", "timeout"),
)

LOG_SERVICE_DB_SINK_LOOP_EXITED = _constant(
    "LOG615",
    "INFO",
    "Telemetry DB sink loop exited",
    component="Service",
    subcomponent="DBSink",
    tags=_tags("telemetry", "db", "lifecycle"),
)

LOG_SERVICE_DB_SINK_FATAL = _constant(
    "LOG616",
    "ERROR",
    "Telemetry DB sink encountered fatal error",
    component="Service",
    subcomponent="DBSink",
    tags=_tags("telemetry", "db", "error"),
)

LOG_SERVICE_DB_SINK_QUEUE_DEPTH = _constant(
    "LOG931",
    "DEBUG",
    "Telemetry DB sink queue depth snapshot",
    component="Service",
    subcomponent="DBSink",
    tags=_tags("telemetry", "db", "queue"),
)

LOG_SERVICE_DB_SINK_QUEUE_PRESSURE = _constant(
    "LOG932",
    "WARNING",
    "Telemetry DB sink queue high-water mark",
    component="Service",
    subcomponent="DBSink",
    tags=_tags("telemetry", "db", "queue", "pressure"),
)

LOG_SERVICE_DB_SINK_FLUSH_STATS = _constant(
    "LOG933",
    "DEBUG",
    "Telemetry DB sink flush statistics",
    component="Service",
    subcomponent="DBSink",
    tags=_tags("telemetry", "db", "flush"),
)

LOG_SERVICE_DB_SINK_FLUSH_LATENCY = _constant(
    "LOG934",
    "WARNING",
    "Telemetry DB sink flush latency exceeded threshold",
    component="Service",
    subcomponent="DBSink",
    tags=_tags("telemetry", "db", "flush", "latency"),
)

LOG_SERVICE_SQLITE_DEBUG = _constant(
    "LOG617",
    "DEBUG",
    "Telemetry SQLite debug event",
    component="Service",
    subcomponent="SQLite",
    tags=_tags("telemetry", "sqlite", "debug"),
)

LOG_SERVICE_SQLITE_INFO = _constant(
    "LOG618",
    "INFO",
    "Telemetry SQLite info event",
    component="Service",
    subcomponent="SQLite",
    tags=_tags("telemetry", "sqlite"),
)

LOG_SERVICE_SQLITE_WARNING = _constant(
    "LOG619",
    "WARNING",
    "Telemetry SQLite warning",
    component="Service",
    subcomponent="SQLite",
    tags=_tags("telemetry", "sqlite", "warning"),
)

LOG_SERVICE_SQLITE_WORKER_STARTED = _constant(
    "LOG620",
    "INFO",
    "Telemetry SQLite worker started",
    component="Service",
    subcomponent="SQLite",
    tags=_tags("telemetry", "sqlite", "worker"),
)

LOG_SERVICE_SQLITE_WORKER_STOPPED = _constant(
    "LOG621",
    "INFO",
    "Telemetry SQLite worker stopped",
    component="Service",
    subcomponent="SQLite",
    tags=_tags("telemetry", "sqlite", "worker"),
)

LOG_SERVICE_SQLITE_WRITE_ERROR = _constant(
    "LOG622",
    "ERROR",
    "Telemetry SQLite write failed",
    component="Service",
    subcomponent="SQLite",
    tags=_tags("telemetry", "sqlite", "error"),
)

LOG_SERVICE_SQLITE_DISK_IO_ERROR = _constant(
    "LOG622A",
    "ERROR",
    "Telemetry SQLite disk I/O error - database may be corrupted. Delete var/telemetry/telemetry.sqlite* files and restart.",
    component="Service",
    subcomponent="SQLite",
    tags=_tags("telemetry", "sqlite", "disk_io", "corruption", "error"),
)

LOG_SERVICE_SQLITE_INIT_ERROR = _constant(
    "LOG622B",
    "ERROR",
    "Telemetry SQLite initialization failed",
    component="Service",
    subcomponent="SQLite",
    tags=_tags("telemetry", "sqlite", "init", "error"),
)

LOG_SERVICE_SQLITE_DESERIALIZATION_FAILED = _constant(
    "LOG623",
    "ERROR",
    "Failed to deserialize telemetry record from queue",
    component="Service",
    subcomponent="SQLite",
    tags=_tags("telemetry", "sqlite", "error"),
)

# Telemetry Bridge (TelemetryBridge class) constants
LOG_SERVICE_TELEMETRY_BRIDGE_STEP_QUEUED = _constant(
    "LOG624",
    "DEBUG",
    "Telemetry step event queued for delivery",
    component="Service",
    subcomponent="TelemetryBridge",
    tags=_tags("telemetry", "bridge", "step"),
)

LOG_SERVICE_TELEMETRY_BRIDGE_EPISODE_QUEUED = _constant(
    "LOG625",
    "DEBUG",
    "Telemetry episode event queued for delivery",
    component="Service",
    subcomponent="TelemetryBridge",
    tags=_tags("telemetry", "bridge", "episode"),
)

LOG_SERVICE_TELEMETRY_BRIDGE_STEP_DELIVERED = _constant(
    "LOG626",
    "DEBUG",
    "Telemetry step event delivered from bridge",
    component="Service",
    subcomponent="TelemetryBridge",
    tags=_tags("telemetry", "bridge", "step"),
)

LOG_SERVICE_TELEMETRY_BRIDGE_EPISODE_DELIVERED = _constant(
    "LOG627",
    "DEBUG",
    "Telemetry episode event delivered from bridge",
    component="Service",
    subcomponent="TelemetryBridge",
    tags=_tags("telemetry", "bridge", "episode"),
)

LOG_SERVICE_TELEMETRY_BRIDGE_OVERFLOW = _constant(
    "LOG628",
    "WARNING",
    "Telemetry bridge queue overflow - events dropped",
    component="Service",
    subcomponent="TelemetryBridge",
    tags=_tags("telemetry", "bridge", "overflow"),
)

LOG_SERVICE_TELEMETRY_BRIDGE_RUN_COMPLETED = _constant(
    "LOG629",
    "INFO",
    "Training run completed - signaling via bridge",
    component="Service",
    subcomponent="TelemetryBridge",
    tags=_tags("telemetry", "bridge", "completion"),
)

# Telemetry Async Hub (TelemetryAsyncHub class) constants
LOG_SERVICE_TELEMETRY_HUB_STARTED = _constant(
    "LOG640",
    "INFO",
    "Telemetry async hub started",
    component="Service",
    subcomponent="TelemetryAsyncHub",
    tags=_tags("telemetry", "hub", "lifecycle"),
)

LOG_SERVICE_TELEMETRY_HUB_SUBSCRIBED = _constant(
    "LOG641",
    "INFO",
    "Subscribed to telemetry stream for run",
    component="Service",
    subcomponent="TelemetryAsyncHub",
    tags=_tags("telemetry", "hub", "subscription"),
)

LOG_SERVICE_TELEMETRY_HUB_TRACE = _constant(
    "LOG642",
    "DEBUG",
    "Telemetry hub operation trace",
    component="Service",
    subcomponent="TelemetryAsyncHub",
    tags=_tags("telemetry", "hub", "trace"),
)

LOG_SERVICE_TELEMETRY_HUB_ERROR = _constant(
    "LOG643",
    "ERROR",
    "Telemetry hub error",
    component="Service",
    subcomponent="TelemetryAsyncHub",
    tags=_tags("telemetry", "hub", "error"),
)

LOG_RENDER_REGULATOR_NOT_STARTED = _constant(
    "LOG630",
    "WARNING",
    "Rendering regulator not started",
    component="Service",
    subcomponent="Rendering",
    tags=_tags("telemetry", "render", "regulator"),
)

LOG_RENDER_DROPPED_FRAME = _constant(
    "LOG631",
    "INFO",
    "Dropped render frame due to throttling",
    component="Service",
    subcomponent="Rendering",
    tags=_tags("telemetry", "render", "throttle"),
)

LOG_DAEMON_START = _constant(
    "LOG650",
    "INFO",
    "Trainer daemon starting",
    component="Service",
    subcomponent="TrainerDaemon",
    tags=_tags("daemon", "lifecycle"),
)

LOG_TRAINER_CLIENT_CONNECTING = _constant(
    "LOG651",
    "INFO",
    "Connecting to trainer daemon",
    component="Service",
    subcomponent="TrainerClient",
    tags=_tags("trainer", "client", "connection", "lifecycle"),
)

LOG_TRAINER_CLIENT_CONNECTED = _constant(
    "LOG652",
    "INFO",
    "Trainer daemon connection established",
    component="Service",
    subcomponent="TrainerClient",
    tags=_tags("trainer", "client", "connection", "lifecycle"),
)

LOG_TRAINER_CLIENT_CONNECTION_TIMEOUT = _constant(
    "LOG653",
    "ERROR",
    "Trainer daemon connection timeout",
    component="Service",
    subcomponent="TrainerClient",
    tags=_tags("trainer", "client", "connection", "timeout", "error"),
)

LOG_TRAINER_CLIENT_LOOP_NONFATAL = _constant(
    "LOG654",
    "DEBUG",
    "Trainer client loop non-fatal condition ignored",
    component="Service",
    subcomponent="TrainerClient",
    tags=_tags("trainer", "client", "loop", "debug"),
)

LOG_TRAINER_CLIENT_LOOP_ERROR = _constant(
    "LOG655",
    "ERROR",
    "Trainer client loop error",
    component="Service",
    subcomponent="TrainerClient",
    tags=_tags("trainer", "client", "loop", "error"),
)

LOG_TRAINER_CLIENT_SHUTDOWN_WARNING = _constant(
    "LOG656",
    "WARNING",
    "Trainer client shutdown cleanup failed",
    component="Service",
    subcomponent="TrainerClient",
    tags=_tags("trainer", "client", "shutdown", "cleanup"),
)

LOG_TRAINER_WORKER_IMPORT_ERROR = _constant(
    "LOG657",
    "ERROR",
    "CleanRL worker module not importable; PYTHONPATH missing vendored packages (MuJoCo/telemetry workers blocked)",
    component="Service",
    subcomponent="TrainerDispatcher",
    tags=_tags("trainer", "worker", "pythonpath", "dependency"),
)

LOG_WORKER_CLEANRL_MODULE_EXECUTION_FAILED = _constant(
    "LOG440",
    "ERROR",
    "CleanRL module execution failed (no main() and subprocess error)",
    component="Worker",
    subcomponent="runtime",
    tags=_tags("worker", "cleanrl", "runtime", "subprocess"),
)

LOG_WORKER_CLEANRL_SUBPROCESS_STDOUT = _constant(
    "LOG441",
    "INFO",
    "CleanRL subprocess stdout",
    component="Worker",
    subcomponent="runtime",
    tags=_tags("worker", "cleanrl", "runtime", "stdout"),
)


LOG_WORKER_TIANSHOU_RUNTIME_STARTED = _constant(
    "LOG450",
    "INFO",
    "Tianshou worker runtime started",
    component="Worker",
    subcomponent="runtime",
    tags=_tags("worker", "tianshou", "runtime", "start"),
)

LOG_WORKER_TIANSHOU_RUNTIME_COMPLETED = _constant(
    "LOG451",
    "INFO",
    "Tianshou worker runtime completed successfully",
    component="Worker",
    subcomponent="runtime",
    tags=_tags("worker", "tianshou", "runtime", "completion"),
)

LOG_WORKER_TIANSHOU_RUNTIME_FAILED = _constant(
    "LOG452",
    "ERROR",
    "Tianshou worker runtime failed",
    component="Worker",
    subcomponent="runtime",
    tags=_tags("worker", "tianshou", "runtime", "error"),
)

LOG_WORKER_TIANSHOU_SUBPROCESS_STDOUT = _constant(
    "LOG453",
    "INFO",
    "Tianshou subprocess stdout",
    component="Worker",
    subcomponent="runtime",
    tags=_tags("worker", "tianshou", "runtime", "stdout"),
)

LOG_WORKER_TIANSHOU_SUBPROCESS_STDERR = _constant(
    "LOG454",
    "ERROR",
    "Tianshou subprocess stderr",
    component="Worker",
    subcomponent="runtime",
    tags=_tags("worker", "tianshou", "runtime", "stderr"),
)


# ---------------------------------------------------------------------------
# MushroomRL Worker (LOG1100-LOG1104)
# ---------------------------------------------------------------------------
LOG_WORKER_MUSHROOMRL_RUNTIME_STARTED = _constant(
    "LOG1100",
    "INFO",
    "MushroomRL worker runtime started",
    component="Worker",
    subcomponent="runtime",
    tags=_tags("worker", "mushroomrl", "runtime", "start"),
)

LOG_WORKER_MUSHROOMRL_RUNTIME_COMPLETED = _constant(
    "LOG1101",
    "INFO",
    "MushroomRL worker runtime completed successfully",
    component="Worker",
    subcomponent="runtime",
    tags=_tags("worker", "mushroomrl", "runtime", "completion"),
)

LOG_WORKER_MUSHROOMRL_RUNTIME_FAILED = _constant(
    "LOG1102",
    "ERROR",
    "MushroomRL worker runtime failed",
    component="Worker",
    subcomponent="runtime",
    tags=_tags("worker", "mushroomrl", "runtime", "error"),
)

LOG_WORKER_MUSHROOMRL_SUBPROCESS_STDOUT = _constant(
    "LOG1103",
    "INFO",
    "MushroomRL subprocess stdout",
    component="Worker",
    subcomponent="runtime",
    tags=_tags("worker", "mushroomrl", "runtime", "stdout"),
)

LOG_WORKER_MUSHROOMRL_SUBPROCESS_STDERR = _constant(
    "LOG1104",
    "ERROR",
    "MushroomRL subprocess stderr",
    component="Worker",
    subcomponent="runtime",
    tags=_tags("worker", "mushroomrl", "runtime", "stderr"),
)


# ---------------------------------------------------------------------------
# OmniSafe Worker (LOG1110-LOG1114)
# ---------------------------------------------------------------------------
LOG_WORKER_OMNISAFE_RUNTIME_STARTED = _constant(
    "LOG1110",
    "INFO",
    "OmniSafe worker runtime started",
    component="Worker",
    subcomponent="runtime",
    tags=_tags("worker", "omnisafe", "runtime", "start"),
)

LOG_WORKER_OMNISAFE_RUNTIME_COMPLETED = _constant(
    "LOG1111",
    "INFO",
    "OmniSafe worker runtime completed successfully",
    component="Worker",
    subcomponent="runtime",
    tags=_tags("worker", "omnisafe", "runtime", "completion"),
)

LOG_WORKER_OMNISAFE_RUNTIME_FAILED = _constant(
    "LOG1112",
    "ERROR",
    "OmniSafe worker runtime failed",
    component="Worker",
    subcomponent="runtime",
    tags=_tags("worker", "omnisafe", "runtime", "error"),
)

LOG_WORKER_OMNISAFE_SUBPROCESS_STDOUT = _constant(
    "LOG1113",
    "INFO",
    "OmniSafe subprocess stdout",
    component="Worker",
    subcomponent="runtime",
    tags=_tags("worker", "omnisafe", "runtime", "stdout"),
)

LOG_WORKER_OMNISAFE_SUBPROCESS_STDERR = _constant(
    "LOG1114",
    "ERROR",
    "OmniSafe subprocess stderr",
    component="Worker",
    subcomponent="runtime",
    tags=_tags("worker", "omnisafe", "runtime", "stderr"),
)


LOG_WORKER_PEARL_RUNTIME_STARTED = _constant(
    "LOG460",
    "INFO",
    "Pearl worker runtime started",
    component="Worker",
    subcomponent="runtime",
    tags=_tags("worker", "pearl", "runtime", "start"),
)

LOG_WORKER_PEARL_RUNTIME_COMPLETED = _constant(
    "LOG461",
    "INFO",
    "Pearl worker runtime completed successfully",
    component="Worker",
    subcomponent="runtime",
    tags=_tags("worker", "pearl", "runtime", "completion"),
)

LOG_WORKER_PEARL_RUNTIME_FAILED = _constant(
    "LOG462",
    "ERROR",
    "Pearl worker runtime failed",
    component="Worker",
    subcomponent="runtime",
    tags=_tags("worker", "pearl", "runtime", "error"),
)

LOG_WORKER_PEARL_SUBPROCESS_STDOUT = _constant(
    "LOG463",
    "INFO",
    "Pearl subprocess stdout",
    component="Worker",
    subcomponent="runtime",
    tags=_tags("worker", "pearl", "runtime", "stdout"),
)

LOG_WORKER_PEARL_SUBPROCESS_STDERR = _constant(
    "LOG464",
    "ERROR",
    "Pearl subprocess stderr",
    component="Worker",
    subcomponent="runtime",
    tags=_tags("worker", "pearl", "runtime", "stderr"),
)


LOG_WORKER_TORCHRL_RUNTIME_STARTED = _constant(
    "LOG470",
    "INFO",
    "TorchRL worker runtime started",
    component="Worker",
    subcomponent="runtime",
    tags=_tags("worker", "torchrl", "runtime", "start"),
)

LOG_WORKER_TORCHRL_RUNTIME_COMPLETED = _constant(
    "LOG471",
    "INFO",
    "TorchRL worker runtime completed successfully",
    component="Worker",
    subcomponent="runtime",
    tags=_tags("worker", "torchrl", "runtime", "completion"),
)

LOG_WORKER_TORCHRL_RUNTIME_FAILED = _constant(
    "LOG472",
    "ERROR",
    "TorchRL worker runtime failed",
    component="Worker",
    subcomponent="runtime",
    tags=_tags("worker", "torchrl", "runtime", "error"),
)

LOG_WORKER_TORCHRL_SUBPROCESS_STDOUT = _constant(
    "LOG473",
    "INFO",
    "TorchRL subprocess stdout",
    component="Worker",
    subcomponent="runtime",
    tags=_tags("worker", "torchrl", "runtime", "stdout"),
)

LOG_WORKER_TORCHRL_SUBPROCESS_STDERR = _constant(
    "LOG474",
    "ERROR",
    "TorchRL subprocess stderr",
    component="Worker",
    subcomponent="runtime",
    tags=_tags("worker", "torchrl", "runtime", "stderr"),
)


LOG_WORKER_SB3_RUNTIME_STARTED = _constant(
    "LOG480",
    "INFO",
    "SB3 worker runtime started",
    component="Worker",
    subcomponent="runtime",
    tags=_tags("worker", "sb3", "runtime", "start"),
)

LOG_WORKER_SB3_RUNTIME_COMPLETED = _constant(
    "LOG481",
    "INFO",
    "SB3 worker runtime completed successfully",
    component="Worker",
    subcomponent="runtime",
    tags=_tags("worker", "sb3", "runtime", "completion"),
)

LOG_WORKER_SB3_RUNTIME_FAILED = _constant(
    "LOG482",
    "ERROR",
    "SB3 worker runtime failed",
    component="Worker",
    subcomponent="runtime",
    tags=_tags("worker", "sb3", "runtime", "error"),
)

LOG_WORKER_SB3_SUBPROCESS_STDOUT = _constant(
    "LOG483",
    "INFO",
    "SB3 subprocess stdout",
    component="Worker",
    subcomponent="runtime",
    tags=_tags("worker", "sb3", "runtime", "stdout"),
)

LOG_WORKER_SB3_SUBPROCESS_STDERR = _constant(
    "LOG484",
    "ERROR",
    "SB3 subprocess stderr",
    component="Worker",
    subcomponent="runtime",
    tags=_tags("worker", "sb3", "runtime", "stderr"),
)


LOG_WORKER_SBX_RUNTIME_STARTED = _constant(
    "LOG490",
    "INFO",
    "SBX worker runtime started",
    component="Worker",
    subcomponent="runtime",
    tags=_tags("worker", "sbx", "runtime", "start"),
)

LOG_WORKER_SBX_RUNTIME_COMPLETED = _constant(
    "LOG491",
    "INFO",
    "SBX worker runtime completed successfully",
    component="Worker",
    subcomponent="runtime",
    tags=_tags("worker", "sbx", "runtime", "completion"),
)

LOG_WORKER_SBX_RUNTIME_FAILED = _constant(
    "LOG492",
    "ERROR",
    "SBX worker runtime failed",
    component="Worker",
    subcomponent="runtime",
    tags=_tags("worker", "sbx", "runtime", "error"),
)

LOG_WORKER_SBX_SUBPROCESS_STDOUT = _constant(
    "LOG493",
    "INFO",
    "SBX subprocess stdout",
    component="Worker",
    subcomponent="runtime",
    tags=_tags("worker", "sbx", "runtime", "stdout"),
)

LOG_WORKER_SBX_SUBPROCESS_STDERR = _constant(
    "LOG494",
    "ERROR",
    "SBX subprocess stderr",
    component="Worker",
    subcomponent="runtime",
    tags=_tags("worker", "sbx", "runtime", "stderr"),
)


LOG_SERVICE_FRAME_INFO = _constant(
    "LOG660",
    "INFO",
    "Frame storage event",
    component="Service",
    subcomponent="FrameStorage",
    tags=_tags("service", "frame", "storage"),
)

LOG_SERVICE_FRAME_WARNING = _constant(
    "LOG661",
    "WARNING",
    "Frame storage warning",
    component="Service",
    subcomponent="FrameStorage",
    tags=_tags("service", "frame", "storage", "warning"),
)

LOG_SERVICE_FRAME_ERROR = _constant(
    "LOG662",
    "ERROR",
    "Frame storage error",
    component="Service",
    subcomponent="FrameStorage",
    tags=_tags("service", "frame", "storage", "error"),
)

LOG_SERVICE_FRAME_DEBUG = _constant(
    "LOG663",
    "DEBUG",
    "Frame storage debug event",
    component="Service",
    subcomponent="FrameStorage",
    tags=_tags("service", "frame", "storage", "debug"),
)

# Session (JSONL recorder / normalization) constants
LOG_SERVICE_SESSION_NUMPY_SCALAR_COERCE_FAILED = _constant(
    "LOG672",
    "DEBUG",
    "Session JSON normalization: numpy scalar coerce failed",
    component="Service",
    subcomponent="Session",
    tags=_tags("session", "json", "numpy", "coerce"),
)

LOG_SERVICE_SESSION_NDARRAY_SUMMARY_FAILED = _constant(
    "LOG673",
    "WARNING",
    "Session JSON normalization: ndarray summary failed; falling back",
    component="Service",
    subcomponent="Session",
    tags=_tags("session", "json", "ndarray", "summary"),
)

LOG_SERVICE_SESSION_LAZYFRAMES_SUMMARY_FAILED = _constant(
    "LOG674",
    "WARNING",
    "Session JSON normalization: LazyFrames summary failed; falling back",
    component="Service",
    subcomponent="Session",
    tags=_tags("session", "json", "lazyframes", "summary"),
)

LOG_SERVICE_SESSION_TOLIST_COERCE_FAILED = _constant(
    "LOG675",
    "DEBUG",
    "Session JSON normalization: tolist() coerce failed",
    component="Service",
    subcomponent="Session",
    tags=_tags("session", "json", "tolist", "coerce"),
)

LOG_SERVICE_SESSION_ITERABLE_COERCE_FAILED = _constant(
    "LOG676",
    "DEBUG",
    "Session JSON normalization: iterable coerce failed",
    component="Service",
    subcomponent="Session",
    tags=_tags("session", "json", "iterable", "coerce"),
)

LOG_SERVICE_VALIDATION_DEBUG = _constant(
    "LOG664",
    "DEBUG",
    "Validation service debug",
    component="Service",
    subcomponent="Validation",
    tags=_tags("service", "validation", "debug"),
)

LOG_SERVICE_VALIDATION_WARNING = _constant(
    "LOG665",
    "WARNING",
    "Validation service warning",
    component="Service",
    subcomponent="Validation",
    tags=_tags("service", "validation", "warning"),
)

LOG_SERVICE_VALIDATION_ERROR = _constant(
    "LOG666",
    "ERROR",
    "Validation service error",
    component="Service",
    subcomponent="Validation",
    tags=_tags("service", "validation", "error"),
)

LOG_SERVICE_ACTOR_SEED_ERROR = _constant(
    "LOG667",
    "ERROR",
    "Actor seeding failed",
    component="Service",
    subcomponent="Actor",
    tags=_tags("service", "actor", "seeding", "error"),
)

LOG_SCHEMA_MISMATCH = _constant(
    "LOG668",
    "WARNING",
    "Telemetry payload failed schema validation",
    component="Service",
    subcomponent="Telemetry",
    tags=_tags("telemetry", "schema", "validation"),
)

LOG_VECTOR_AUTORESET_MODE = _constant(
    "LOG669",
    "WARNING",
    "Unsupported vector autoreset mode encountered",
    component="Service",
    subcomponent="Telemetry",
    tags=_tags("telemetry", "vector", "schema"),
)

LOG_SPACE_DESCRIPTOR_MISSING = _constant(
    "LOG670",
    "WARNING",
    "Space descriptor missing from telemetry payload",
    component="Service",
    subcomponent="Telemetry",
    tags=_tags("telemetry", "schema", "space"),
)

LOG_NORMALIZATION_STATS_DROPPED = _constant(
    "LOG671",
    "INFO",
    "Normalization statistics absent for schema-enabled payload",
    component="Service",
    subcomponent="Telemetry",
    tags=_tags("telemetry", "schema", "normalization"),
)

# ---------------------------------------------------------------------------
# Operator Service constants (LOG685–LOG689)
# ---------------------------------------------------------------------------
LOG_SERVICE_OPERATOR_REGISTERED = _constant(
    "LOG685",
    "INFO",
    "Operator registered with OperatorService",
    component="Service",
    subcomponent="Operator",
    tags=_tags("operator", "service", "registration"),
)

LOG_SERVICE_OPERATOR_ACTIVATED = _constant(
    "LOG686",
    "INFO",
    "Operator activated as current action selector",
    component="Service",
    subcomponent="Operator",
    tags=_tags("operator", "service", "activation"),
)

LOG_SERVICE_OPERATOR_DEACTIVATED = _constant(
    "LOG687",
    "INFO",
    "Operator deactivated",
    component="Service",
    subcomponent="Operator",
    tags=_tags("operator", "service", "deactivation"),
)

LOG_SERVICE_OPERATOR_ACTION_SELECTED = _constant(
    "LOG688",
    "DEBUG",
    "Operator selected action",
    component="Service",
    subcomponent="Operator",
    tags=_tags("operator", "service", "action"),
)

LOG_SERVICE_OPERATOR_ERROR = _constant(
    "LOG689",
    "ERROR",
    "Operator encountered an error",
    component="Service",
    subcomponent="Operator",
    tags=_tags("operator", "service", "error"),
)

# ---------------------------------------------------------------------------
# Operator Launcher Interactive Mode constants (LOG690–LOG697)
# ---------------------------------------------------------------------------
LOG_OPERATOR_INTERACTIVE_LAUNCHED = _constant(
    "LOG690",
    "INFO",
    "Operator launched in interactive mode",
    component="Service",
    subcomponent="OperatorLauncher",
    tags=_tags("operator", "launcher", "interactive", "launch"),
)

LOG_OPERATOR_RESET_COMMAND_SENT = _constant(
    "LOG691",
    "INFO",
    "Reset command sent to interactive operator",
    component="Service",
    subcomponent="OperatorLauncher",
    tags=_tags("operator", "launcher", "interactive", "reset"),
)

LOG_OPERATOR_STEP_COMMAND_SENT = _constant(
    "LOG692",
    "DEBUG",
    "Step command sent to interactive operator",
    component="Service",
    subcomponent="OperatorLauncher",
    tags=_tags("operator", "launcher", "interactive", "step"),
)

LOG_OPERATOR_STOP_COMMAND_SENT = _constant(
    "LOG693",
    "INFO",
    "Stop command sent to interactive operator",
    component="Service",
    subcomponent="OperatorLauncher",
    tags=_tags("operator", "launcher", "interactive", "stop"),
)

LOG_OPERATOR_COMMAND_FAILED = _constant(
    "LOG694",
    "WARNING",
    "Failed to send command to interactive operator",
    component="Service",
    subcomponent="OperatorLauncher",
    tags=_tags("operator", "launcher", "interactive", "error"),
)

LOG_OPERATOR_RESET_ALL_STARTED = _constant(
    "LOG695",
    "INFO",
    "Reset All operators started with shared seed",
    component="UI",
    subcomponent="MainWindow",
    tags=_tags("operator", "interactive", "reset", "seed"),
)

LOG_OPERATOR_STEP_ALL_COMPLETED = _constant(
    "LOG696",
    "DEBUG",
    "Step All operators completed",
    component="UI",
    subcomponent="MainWindow",
    tags=_tags("operator", "interactive", "step"),
)

LOG_OPERATOR_STOP_ALL_COMPLETED = _constant(
    "LOG697",
    "INFO",
    "Stop All operators completed",
    component="UI",
    subcomponent="MainWindow",
    tags=_tags("operator", "interactive", "stop"),
)

# ---------------------------------------------------------------------------
# Script Operator constants (LOG4080–LOG4089)
# ---------------------------------------------------------------------------
LOG_SCRIPT_OPERATOR_LAUNCHED = _constant(
    "LOG4080",
    "INFO",
    "Script operator launched for automated experiment",
    component="Service",
    subcomponent="OperatorLauncher",
    tags=_tags("operator", "script", "launch", "experiment"),
)

LOG_SCRIPT_OPERATOR_BEHAVIOR_SET = _constant(
    "LOG4081",
    "DEBUG",
    "Script operator behavior configured",
    component="Service",
    subcomponent="OperatorLauncher",
    tags=_tags("operator", "script", "config", "behavior"),
)

LOG_SCRIPT_OPERATOR_EPISODE_START = _constant(
    "LOG4082",
    "INFO",
    "Script operator episode started",
    component="Worker",
    subcomponent="OperatorsWorker",
    tags=_tags("operator", "script", "episode", "start"),
)

LOG_SCRIPT_OPERATOR_EPISODE_END = _constant(
    "LOG4083",
    "INFO",
    "Script operator episode completed",
    component="Worker",
    subcomponent="OperatorsWorker",
    tags=_tags("operator", "script", "episode", "end"),
)

LOG_SCRIPT_OPERATOR_TELEMETRY_EMITTED = _constant(
    "LOG4084",
    "DEBUG",
    "Script operator telemetry written to JSONL",
    component="Worker",
    subcomponent="OperatorsWorker",
    tags=_tags("operator", "script", "telemetry", "jsonl"),
)

LOG_SCRIPT_LOADED = _constant(
    "LOG4085",
    "INFO",
    "Operator script loaded from file",
    component="UI",
    subcomponent="ScriptWidget",
    tags=_tags("operator", "script", "load"),
)

LOG_SCRIPT_PARSED = _constant(
    "LOG4086",
    "INFO",
    "Operator script parsed successfully",
    component="UI",
    subcomponent="ScriptWidget",
    tags=_tags("operator", "script", "parse"),
)

LOG_SCRIPT_VALIDATION_FAILED = _constant(
    "LOG4087",
    "WARNING",
    "Operator script validation failed",
    component="UI",
    subcomponent="ScriptWidget",
    tags=_tags("operator", "script", "validation", "error"),
)

LOG_SCRIPT_AUTO_EXECUTION_STARTED = _constant(
    "LOG4088",
    "INFO",
    "Script auto-execution started",
    component="UI",
    subcomponent="OperatorsTab",
    tags=_tags("operator", "script", "auto-execution", "start"),
)

LOG_SCRIPT_AUTO_EXECUTION_COMPLETED = _constant(
    "LOG4089",
    "INFO",
    "Script auto-execution completed",
    component="UI",
    subcomponent="OperatorsTab",
    tags=_tags("operator", "script", "auto-execution", "complete"),
)

# ---------------------------------------------------------------------------
# Multi-Agent Operator constants (LOG698–LOG702)
# ---------------------------------------------------------------------------
LOG_OPERATOR_INIT_AGENT_SENT = _constant(
    "LOG698",
    "INFO",
    "Init agent command sent to interactive operator",
    component="Service",
    subcomponent="OperatorLauncher",
    tags=_tags("operator", "launcher", "multiagent", "init"),
)

LOG_OPERATOR_SELECT_ACTION_SENT = _constant(
    "LOG699",
    "DEBUG",
    "Select action command sent to interactive operator",
    component="Service",
    subcomponent="OperatorLauncher",
    tags=_tags("operator", "launcher", "multiagent", "action"),
)

LOG_OPERATOR_MULTIAGENT_LAUNCHED = _constant(
    "LOG700",
    "INFO",
    "Multi-agent operator launched with player workers",
    component="Service",
    subcomponent="OperatorLauncher",
    tags=_tags("operator", "launcher", "multiagent", "launch"),
)

LOG_OPERATOR_MULTIAGENT_INIT_FAILED = _constant(
    "LOG701",
    "ERROR",
    "Failed to initialize agent for multi-agent operator",
    component="Service",
    subcomponent="OperatorLauncher",
    tags=_tags("operator", "launcher", "multiagent", "error"),
)

LOG_OPERATOR_MULTIAGENT_ACTION_FAILED = _constant(
    "LOG702",
    "ERROR",
    "Failed to send select action to multi-agent operator",
    component="Service",
    subcomponent="OperatorLauncher",
    tags=_tags("operator", "launcher", "multiagent", "error"),
)

# ---------------------------------------------------------------------------
# Operator Environment Preview constants (LOG703–LOG706)
# ---------------------------------------------------------------------------
LOG_OPERATOR_ENV_PREVIEW_STARTED = _constant(
    "LOG703",
    "INFO",
    "Operator environment preview started",
    component="UI",
    subcomponent="MainWindow",
    tags=_tags("operator", "environment", "preview", "start"),
)

LOG_OPERATOR_ENV_PREVIEW_SUCCESS = _constant(
    "LOG704",
    "INFO",
    "Operator environment preview loaded successfully",
    component="UI",
    subcomponent="MainWindow",
    tags=_tags("operator", "environment", "preview", "success"),
)

LOG_OPERATOR_ENV_PREVIEW_IMPORT_ERROR = _constant(
    "LOG705",
    "WARNING",
    "Operator environment preview failed due to missing package",
    component="UI",
    subcomponent="MainWindow",
    tags=_tags("operator", "environment", "preview", "import", "error"),
)

LOG_OPERATOR_ENV_PREVIEW_ERROR = _constant(
    "LOG706",
    "ERROR",
    "Operator environment preview failed",
    component="UI",
    subcomponent="MainWindow",
    tags=_tags("operator", "environment", "preview", "error"),
)

# ---------------------------------------------------------------------------
# Parallel Multi-Agent Operator constants (LOG707–LOG709)
# For simultaneous environments: MultiGrid, MeltingPot, Overcooked
# ---------------------------------------------------------------------------
LOG_OPERATOR_PARALLEL_RESET_STARTED = _constant(
    "LOG707",
    "INFO",
    "Parallel multi-agent operator reset started",
    component="UI",
    subcomponent="MainWindow",
    tags=_tags("operator", "parallel", "multiagent", "reset", "start"),
)

LOG_OPERATOR_PARALLEL_STEP_STARTED = _constant(
    "LOG708",
    "DEBUG",
    "Parallel multi-agent step started, collecting actions",
    component="UI",
    subcomponent="MainWindow",
    tags=_tags("operator", "parallel", "multiagent", "step", "start"),
)

LOG_OPERATOR_PARALLEL_STEP_COMPLETED = _constant(
    "LOG709",
    "DEBUG",
    "Parallel multi-agent step completed",
    component="UI",
    subcomponent="MainWindow",
    tags=_tags("operator", "parallel", "multiagent", "step", "complete"),
)

# ---------------------------------------------------------------------------
# vLLM Server Widget constants (LOG710–LOG719)
# ---------------------------------------------------------------------------
LOG_VLLM_SERVER_COUNT_CHANGED = _constant(
    "LOG710",
    "INFO",
    "vLLM server count changed",
    component="UI",
    subcomponent="VLLMServerWidget",
    tags=_tags("vllm", "server", "config"),
)

LOG_VLLM_SERVER_STARTING = _constant(
    "LOG711",
    "INFO",
    "vLLM server starting",
    component="UI",
    subcomponent="VLLMServerWidget",
    tags=_tags("vllm", "server", "start"),
)

LOG_VLLM_SERVER_RUNNING = _constant(
    "LOG712",
    "INFO",
    "vLLM server is running",
    component="UI",
    subcomponent="VLLMServerWidget",
    tags=_tags("vllm", "server", "running"),
)

LOG_VLLM_SERVER_STOPPING = _constant(
    "LOG713",
    "INFO",
    "vLLM server stopping",
    component="UI",
    subcomponent="VLLMServerWidget",
    tags=_tags("vllm", "server", "stop"),
)

LOG_VLLM_SERVER_START_FAILED = _constant(
    "LOG714",
    "ERROR",
    "vLLM server failed to start",
    component="UI",
    subcomponent="VLLMServerWidget",
    tags=_tags("vllm", "server", "error"),
)

LOG_VLLM_SERVER_PROCESS_EXITED = _constant(
    "LOG715",
    "WARNING",
    "vLLM server process exited unexpectedly",
    component="UI",
    subcomponent="VLLMServerWidget",
    tags=_tags("vllm", "server", "error"),
)

LOG_VLLM_SERVER_NOT_RESPONDING = _constant(
    "LOG716",
    "WARNING",
    "vLLM server stopped responding",
    component="UI",
    subcomponent="VLLMServerWidget",
    tags=_tags("vllm", "server", "error"),
)

LOG_VLLM_ORPHAN_PROCESS_KILLED = _constant(
    "LOG717",
    "INFO",
    "Killed orphan vLLM process",
    component="UI",
    subcomponent="VLLMServerWidget",
    tags=_tags("vllm", "server", "cleanup"),
)

LOG_VLLM_GPU_MEMORY_FREED = _constant(
    "LOG718",
    "INFO",
    "GPU memory verified freed",
    component="UI",
    subcomponent="VLLMServerWidget",
    tags=_tags("vllm", "server", "gpu", "memory"),
)

LOG_VLLM_GPU_MEMORY_NOT_FREED = _constant(
    "LOG719",
    "WARNING",
    "GPU memory may not be fully freed",
    component="UI",
    subcomponent="VLLMServerWidget",
    tags=_tags("vllm", "server", "gpu", "memory", "warning"),
)

# ---------------------------------------------------------------------------
# Runtime/application constants (LOG680–LOG683)
# ---------------------------------------------------------------------------
LOG_RUNTIME_APP_DEBUG = _constant(
    "LOG680",
    "DEBUG",
    "Application runtime debug",
    component="Runtime",
    subcomponent="App",
    tags=_tags("runtime", "app", "debug"),
)

LOG_RUNTIME_APP_INFO = _constant(
    "LOG681",
    "INFO",
    "Application runtime event",
    component="Runtime",
    subcomponent="App",
    tags=_tags("runtime", "app"),
)

LOG_RUNTIME_APP_WARNING = _constant(
    "LOG682",
    "WARNING",
    "Application runtime warning",
    component="Runtime",
    subcomponent="App",
    tags=_tags("runtime", "app", "warning"),
)

LOG_RUNTIME_APP_ERROR = _constant(
    "LOG683",
    "ERROR",
    "Application runtime error",
    component="Runtime",
    subcomponent="App",
    tags=_tags("runtime", "app", "error"),
)


# ---------------------------------------------------------------------------
# UI constants (LOG701–LOG739)
# ---------------------------------------------------------------------------
LOG_UI_MAINWINDOW_TRACE = _constant(
    "LOG701",
    "DEBUG",
    "UI main window trace",
    component="UI",
    subcomponent="MainWindow",
    tags=_tags("ui", "mainwindow", "trace"),
)

LOG_UI_MAINWINDOW_INFO = _constant(
    "LOG702",
    "INFO",
    "UI main window event",
    component="UI",
    subcomponent="MainWindow",
    tags=_tags("ui", "mainwindow"),
)

LOG_UI_MAINWINDOW_WARNING = _constant(
    "LOG703",
    "WARNING",
    "UI main window warning",
    component="UI",
    subcomponent="MainWindow",
    tags=_tags("ui", "mainwindow", "warning"),
)

LOG_UI_MAINWINDOW_ERROR = _constant(
    "LOG704",
    "ERROR",
    "UI main window error",
    component="UI",
    subcomponent="MainWindow",
    tags=_tags("ui", "mainwindow", "error"),
)

LOG_UI_MAINWINDOW_INVALID_CONFIG = _constant(
    "LOG705",
    "ERROR",
    "UI main window invalid training configuration",
    component="UI",
    subcomponent="MainWindow",
    tags=_tags("ui", "mainwindow", "config", "invalid"),
)

LOG_UI_LIVE_TAB_TRACE = _constant(
    "LOG710",
    "DEBUG",
    "Live telemetry tab trace",
    component="UI",
    subcomponent="LiveTelemetryTab",
    tags=_tags("ui", "telemetry", "trace"),
)

LOG_UI_LIVE_TAB_INFO = _constant(
    "LOG711",
    "INFO",
    "Live telemetry tab event",
    component="UI",
    subcomponent="LiveTelemetryTab",
    tags=_tags("ui", "telemetry"),
)

LOG_UI_LIVE_TAB_WARNING = _constant(
    "LOG712",
    "WARNING",
    "Live telemetry tab warning",
    component="UI",
    subcomponent="LiveTelemetryTab",
    tags=_tags("ui", "telemetry", "warning"),
)

LOG_UI_LIVE_TAB_ERROR = _constant(
    "LOG713",
    "ERROR",
    "Live telemetry tab error",
    component="UI",
    subcomponent="LiveTelemetryTab",
    tags=_tags("ui", "telemetry", "error"),
)

LOG_UI_RENDER_TABS_TRACE = _constant(
    "LOG720",
    "DEBUG",
    "Render tabs trace",
    component="UI",
    subcomponent="RenderTabs",
    tags=_tags("ui", "render", "trace"),
)

LOG_UI_RENDER_TABS_INFO = _constant(
    "LOG721",
    "INFO",
    "Render tabs event",
    component="UI",
    subcomponent="RenderTabs",
    tags=_tags("ui", "render"),
)

LOG_UI_RENDER_TABS_WARNING = _constant(
    "LOG722",
    "WARNING",
    "Render tabs warning",
    component="UI",
    subcomponent="RenderTabs",
    tags=_tags("ui", "render", "warning"),
)

LOG_UI_RENDER_TABS_ERROR = _constant(
    "LOG723",
    "ERROR",
    "Render tabs error",
    component="UI",
    subcomponent="RenderTabs",
    tags=_tags("ui", "render", "error"),
)

LOG_UI_RENDER_TABS_TENSORBOARD_STATUS = _constant(
    "LOG724",
    "INFO",
    "TensorBoard log directory status",
    component="UI",
    subcomponent="RenderTabs",
    tags=_tags("ui", "tensorboard", "status"),
)

LOG_UI_RENDER_TABS_TENSORBOARD_WAITING = _constant(
    "LOG725",
    "DEBUG",
    "TensorBoard directory not yet available",
    component="UI",
    subcomponent="RenderTabs",
    tags=_tags("ui", "tensorboard", "waiting"),
)

LOG_UI_RENDER_TABS_WANDB_STATUS = _constant(
    "LOG726",
    "INFO",
    "Weights & Biases run status",
    component="UI",
    subcomponent="RenderTabs",
    tags=_tags("ui", "wandb", "status"),
)

LOG_UI_RENDER_TABS_WANDB_WARNING = _constant(
    "LOG727",
    "WARNING",
    "Weights & Biases run warning",
    component="UI",
    subcomponent="RenderTabs",
    tags=_tags("ui", "wandb", "warning"),
)

LOG_UI_RENDER_TABS_WANDB_ERROR = _constant(
    "LOG728",
    "ERROR",
    "Weights & Biases run error",
    component="UI",
    subcomponent="RenderTabs",
    tags=_tags("ui", "wandb", "error"),
)

LOG_UI_RENDER_TABS_WANDB_PROXY_APPLIED = _constant(
    "LOG729",
    "INFO",
    "Weights & Biases VPN proxy applied",
    component="UI",
    subcomponent="RenderTabs",
    tags=_tags("ui", "wandb", "proxy", "vpn"),
)

LOG_UI_RENDER_TABS_WANDB_PROXY_SKIPPED = _constant(
    "LOG72A",
    "DEBUG",
    "Weights & Biases VPN proxy not applied",
    component="UI",
    subcomponent="RenderTabs",
    tags=_tags("ui", "wandb", "proxy", "vpn"),
)

LOG_UI_RENDER_TABS_ARTIFACTS_MISSING = _constant(
    "LOG72B",
    "ERROR",
    "Run artifacts directory missing",
    component="UI",
    subcomponent="RenderTabs",
    tags=_tags("ui", "render", "runs", "missing"),
)

LOG_UI_RENDER_TABS_DELETE_REQUESTED = _constant(
    "LOG736",
    "INFO",
    "User requested run deletion from tab",
    component="UI",
    subcomponent="RenderTabs",
    tags=_tags("ui", "render", "tab_closure", "delete"),
)

LOG_UI_RENDER_TABS_EVENT_FOR_DELETED_RUN = _constant(
    "LOG737",
    "WARNING",
    "Live tab recreated for run already marked deleted",
    component="UI",
    subcomponent="RenderTabs",
    tags=_tags("ui", "render", "deleted", "diagnostics"),
)

LOG_UI_RENDER_TABS_TAB_ADDED = _constant(
    "LOG738",
    "INFO",
    "Render tabs added dynamic tab",
    component="UI",
    subcomponent="RenderTabs",
    tags=_tags("ui", "render", "tab"),
)

# ---------------------------------------------------------------------------
# Tab Closure Dialog constants (LOG724–LOG729)
# ---------------------------------------------------------------------------
LOG_UI_TAB_CLOSURE_DIALOG_OPENED = _constant(
    "LOG724",
    "INFO",
    "Tab closure dialog opened for run",
    component="UI",
    subcomponent="TabClosureDialog",
    tags=_tags("ui", "tab_closure", "dialog", "opened"),
)

LOG_UI_TAB_CLOSURE_CHOICE_SELECTED = _constant(
    "LOG725",
    "INFO",
    "Tab closure choice selected by user",
    component="UI",
    subcomponent="TabClosureDialog",
    tags=_tags("ui", "tab_closure", "choice"),
)

LOG_UI_TAB_CLOSURE_CHOICE_CANCELLED = _constant(
    "LOG726",
    "INFO",
    "Tab closure dialog cancelled",
    component="UI",
    subcomponent="TabClosureDialog",
    tags=_tags("ui", "tab_closure", "cancelled"),
)

LOG_UI_TRAIN_FORM_TRACE = _constant(
    "LOG730",
    "DEBUG",
    "Train form trace",
    component="UI",
    subcomponent="TrainForm",
    tags=_tags("ui", "train_form", "trace"),
)

LOG_UI_TRAIN_FORM_INFO = _constant(
    "LOG731",
    "INFO",
    "Train form event",
    component="UI",
    subcomponent="TrainForm",
    tags=_tags("ui", "train_form"),
)

LOG_UI_TRAIN_FORM_WARNING = _constant(
    "LOG732",
    "WARNING",
    "Train form warning",
    component="UI",
    subcomponent="TrainForm",
    tags=_tags("ui", "train_form", "warning"),
)

LOG_UI_TRAIN_FORM_ERROR = _constant(
    "LOG733",
    "ERROR",
    "Train form error",
    component="UI",
    subcomponent="TrainForm",
    tags=_tags("ui", "train_form", "error"),
)

LOG_UI_TRAIN_FORM_UI_PATH = _constant(
    "LOG734",
    "INFO",
    "Train form UI-only path configured",
    component="UI",
    subcomponent="TrainForm",
    tags=_tags("ui", "train_form", "ui_only_path"),
)

LOG_UI_TRAIN_FORM_TELEMETRY_PATH = _constant(
    "LOG735",
    "INFO",
    "Train form telemetry durable path configured",
    component="UI",
    subcomponent="TrainForm",
    tags=_tags("ui", "train_form", "telemetry_path"),
)

LOG_UI_POLICY_FORM_TRACE = _constant(
    "LOG736",
    "DEBUG",
    "Policy form trace",
    component="UI",
    subcomponent="PolicyForm",
    tags=_tags("ui", "policy_form", "trace"),
)

LOG_UI_POLICY_FORM_INFO = _constant(
    "LOG737",
    "INFO",
    "Policy form event",
    component="UI",
    subcomponent="PolicyForm",
    tags=_tags("ui", "policy_form"),
)

LOG_UI_POLICY_FORM_ERROR = _constant(
    "LOG738",
    "ERROR",
    "Policy form error",
    component="UI",
    subcomponent="PolicyForm",
    tags=_tags("ui", "policy_form", "error"),
)

LOG_OPERATOR_VIEW_SIZE_CONFIGURED = _constant(
    "LOG739",
    "INFO",
    "Operator view_size configured for MOSAIC environment",
    component="UI",
    subcomponent="OperatorConfig",
    tags=_tags("operator", "mosaic", "multigrid", "view_size", "config"),
)

LOG_UI_WORKER_TABS_TRACE = _constant(
    "LOG740",
    "DEBUG",
    "Worker tabs trace",
    component="UI",
    subcomponent="WorkerTabs",
    tags=_tags("ui", "worker", "trace"),
)

LOG_UI_WORKER_TABS_INFO = _constant(
    "LOG741",
    "INFO",
    "Worker tabs event",
    component="UI",
    subcomponent="WorkerTabs",
    tags=_tags("ui", "worker"),
)

LOG_UI_WORKER_TABS_WARNING = _constant(
    "LOG742",
    "WARNING",
    "Worker tabs warning",
    component="UI",
    subcomponent="WorkerTabs",
    tags=_tags("ui", "worker", "warning"),
)

LOG_UI_WORKER_TABS_ERROR = _constant(
    "LOG743",
    "ERROR",
    "Worker tabs error",
    component="UI",
    subcomponent="WorkerTabs",
    tags=_tags("ui", "worker", "error"),
)

LOG_UI_PRESENTER_SIGNAL_CONNECTION_WARNING = _constant(
    "LOG744",
    "WARNING",
    "UI presenter signal connection failed",
    component="UI",
    subcomponent="Presenter",
    tags=_tags("ui", "presenter", "signal", "warning"),
)

LOG_UI_MAIN_WINDOW_SHUTDOWN_WARNING = _constant(
    "LOG745",
    "WARNING",
    "Main window shutdown cleanup failed",
    component="UI",
    subcomponent="MainWindow",
    tags=_tags("ui", "main_window", "shutdown", "cleanup"),
)

LOG_UI_TENSORBOARD_KILL_WARNING = _constant(
    "LOG746",
    "WARNING",
    "TensorBoard process kill failed",
    component="UI",
    subcomponent="TensorBoardTab",
    tags=_tags("ui", "tensorboard", "process", "kill", "warning"),
)

LOG_ADAPTER_RENDERING_WARNING = _constant(
    "LOG747",
    "WARNING",
    "Environment adapter rendering failed",
    component="Adapter",
    subcomponent="Rendering",
    tags=_tags("adapter", "rendering", "warning"),
)

LOG_TRAINER_LAUNCHER_LOG_FLUSH_WARNING = _constant(
    "LOG748",
    "WARNING",
    "Trainer launcher log file flush failed",
    component="Service",
    subcomponent="TrainerLauncher",
    tags=_tags("trainer", "launcher", "log", "flush", "warning"),
)

LOG_WORKER_CONFIG_READ_WARNING = _constant(
    "LOG749",
    "WARNING",
    "Worker config file read failed",
    component="Worker",
    subcomponent="Presenter",
    tags=_tags("worker", "presenter", "config", "read", "warning"),
)


# ---------------------------------------------------------------------------
# Multi-Agent Tab constants (LOG750–LOG759)
# ---------------------------------------------------------------------------
LOG_UI_MULTI_AGENT_ENV_LOAD_REQUESTED = _constant(
    "LOG750",
    "INFO",
    "Multi-agent environment load requested",
    component="UI",
    subcomponent="MultiAgentTab",
    tags=_tags("ui", "multi_agent", "environment", "load"),
)

LOG_UI_MULTI_AGENT_ENV_LOADED = _constant(
    "LOG751",
    "INFO",
    "Multi-agent environment loaded successfully",
    component="UI",
    subcomponent="MultiAgentTab",
    tags=_tags("ui", "multi_agent", "environment", "loaded"),
)

LOG_UI_MULTI_AGENT_ENV_LOAD_ERROR = _constant(
    "LOG752",
    "ERROR",
    "Multi-agent environment load failed",
    component="UI",
    subcomponent="MultiAgentTab",
    tags=_tags("ui", "multi_agent", "environment", "error"),
)

LOG_UI_MULTI_AGENT_POLICY_LOAD_REQUESTED = _constant(
    "LOG753",
    "INFO",
    "Multi-agent policy load requested",
    component="UI",
    subcomponent="MultiAgentTab",
    tags=_tags("ui", "multi_agent", "policy", "load"),
)

LOG_UI_MULTI_AGENT_GAME_START_REQUESTED = _constant(
    "LOG754",
    "INFO",
    "Multi-agent game start requested",
    component="UI",
    subcomponent="MultiAgentTab",
    tags=_tags("ui", "multi_agent", "game", "start"),
)

LOG_UI_MULTI_AGENT_RESET_REQUESTED = _constant(
    "LOG755",
    "INFO",
    "Multi-agent reset requested",
    component="UI",
    subcomponent="MultiAgentTab",
    tags=_tags("ui", "multi_agent", "reset"),
)

LOG_UI_MULTI_AGENT_ACTION_SUBMITTED = _constant(
    "LOG756",
    "DEBUG",
    "Multi-agent action submitted",
    component="UI",
    subcomponent="MultiAgentTab",
    tags=_tags("ui", "multi_agent", "action", "submit"),
)

LOG_UI_MULTI_AGENT_TRAIN_REQUESTED = _constant(
    "LOG757",
    "INFO",
    "Multi-agent training requested",
    component="UI",
    subcomponent="MultiAgentTab",
    tags=_tags("ui", "multi_agent", "training", "start"),
)

LOG_UI_MULTI_AGENT_EVALUATE_REQUESTED = _constant(
    "LOG758",
    "INFO",
    "Multi-agent evaluation requested",
    component="UI",
    subcomponent="MultiAgentTab",
    tags=_tags("ui", "multi_agent", "evaluation", "start"),
)

LOG_UI_MULTI_AGENT_ENV_NOT_LOADED = _constant(
    "LOG759",
    "WARNING",
    "Multi-agent action attempted but no environment loaded",
    component="UI",
    subcomponent="MultiAgentTab",
    tags=_tags("ui", "multi_agent", "environment", "warning"),
)

LOG_UI_POLICY_ASSIGNMENT_REQUESTED = _constant(
    "LOG760",
    "INFO",
    "Policy assignment evaluation requested",
    component="UI",
    subcomponent="PolicyAssignment",
    tags=_tags("ui", "multi_agent", "policy", "evaluation", "start"),
)

LOG_UI_POLICY_ASSIGNMENT_LOADED = _constant(
    "LOG761",
    "INFO",
    "Policy checkpoint loaded for evaluation",
    component="UI",
    subcomponent="PolicyAssignment",
    tags=_tags("ui", "multi_agent", "policy", "loaded"),
)

LOG_UI_POLICY_DISCOVERY_SCAN = _constant(
    "LOG762",
    "DEBUG",
    "Scanning for policy checkpoints",
    component="UI",
    subcomponent="PolicyDiscovery",
    tags=_tags("ui", "policy", "discovery", "scan"),
)

LOG_UI_POLICY_DISCOVERY_FOUND = _constant(
    "LOG763",
    "INFO",
    "Policy checkpoints discovered",
    component="UI",
    subcomponent="PolicyDiscovery",
    tags=_tags("ui", "policy", "discovery", "found"),
)


# ---------------------------------------------------------------------------
# LLM Chat UI constants (LOG770–LOG789)
# ---------------------------------------------------------------------------
LOG_UI_CHAT_GPU_DETECTION_STARTED = _constant(
    "LOG770",
    "INFO",
    "GPU detection worker started",
    component="UI",
    subcomponent="ChatPanel",
    tags=_tags("ui", "chat", "gpu", "detection", "thread"),
)

LOG_UI_CHAT_GPU_DETECTION_COMPLETED = _constant(
    "LOG771",
    "INFO",
    "GPU detection completed",
    component="UI",
    subcomponent="ChatPanel",
    tags=_tags("ui", "chat", "gpu", "detection"),
)

LOG_UI_CHAT_GPU_DETECTION_ERROR = _constant(
    "LOG772",
    "ERROR",
    "GPU detection failed",
    component="UI",
    subcomponent="ChatPanel",
    tags=_tags("ui", "chat", "gpu", "detection", "error"),
)

LOG_UI_CHAT_HF_TOKEN_SAVE_STARTED = _constant(
    "LOG773",
    "INFO",
    "HuggingFace token save worker started",
    component="UI",
    subcomponent="ChatPanel",
    tags=_tags("ui", "chat", "huggingface", "token", "save", "thread"),
)

LOG_UI_CHAT_HF_TOKEN_SAVED = _constant(
    "LOG774",
    "INFO",
    "HuggingFace token saved successfully",
    component="UI",
    subcomponent="ChatPanel",
    tags=_tags("ui", "chat", "huggingface", "token", "save"),
)

LOG_UI_CHAT_HF_TOKEN_SAVE_ERROR = _constant(
    "LOG775",
    "ERROR",
    "HuggingFace token save failed",
    component="UI",
    subcomponent="ChatPanel",
    tags=_tags("ui", "chat", "huggingface", "token", "save", "error"),
)

LOG_UI_CHAT_HF_TOKEN_VALIDATION_STARTED = _constant(
    "LOG776",
    "INFO",
    "HuggingFace token validation worker started",
    component="UI",
    subcomponent="ChatPanel",
    tags=_tags("ui", "chat", "huggingface", "token", "validation", "thread"),
)

LOG_UI_CHAT_HF_TOKEN_VALIDATED = _constant(
    "LOG777",
    "INFO",
    "HuggingFace token validated successfully",
    component="UI",
    subcomponent="ChatPanel",
    tags=_tags("ui", "chat", "huggingface", "token", "validation"),
)

LOG_UI_CHAT_HF_TOKEN_VALIDATION_ERROR = _constant(
    "LOG778",
    "ERROR",
    "HuggingFace token validation failed",
    component="UI",
    subcomponent="ChatPanel",
    tags=_tags("ui", "chat", "huggingface", "token", "validation", "error"),
)

LOG_UI_CHAT_MODEL_DOWNLOAD_STARTED = _constant(
    "LOG779",
    "INFO",
    "Model download worker started",
    component="UI",
    subcomponent="ChatPanel",
    tags=_tags("ui", "chat", "model", "download", "thread"),
)

LOG_UI_CHAT_MODEL_DOWNLOAD_PROGRESS = _constant(
    "LOG780",
    "DEBUG",
    "Model download progress update",
    component="UI",
    subcomponent="ChatPanel",
    tags=_tags("ui", "chat", "model", "download", "progress"),
)

LOG_UI_CHAT_MODEL_DOWNLOADED = _constant(
    "LOG781",
    "INFO",
    "Model downloaded successfully",
    component="UI",
    subcomponent="ChatPanel",
    tags=_tags("ui", "chat", "model", "download"),
)

LOG_UI_CHAT_MODEL_DOWNLOAD_ERROR = _constant(
    "LOG782",
    "ERROR",
    "Model download failed",
    component="UI",
    subcomponent="ChatPanel",
    tags=_tags("ui", "chat", "model", "download", "error"),
)

LOG_UI_CHAT_REQUEST_STARTED = _constant(
    "LOG783",
    "INFO",
    "Chat completion request worker started",
    component="UI",
    subcomponent="ChatPanel",
    tags=_tags("ui", "chat", "completion", "request", "thread"),
)

LOG_UI_CHAT_REQUEST_COMPLETED = _constant(
    "LOG784",
    "INFO",
    "Chat completion request completed",
    component="UI",
    subcomponent="ChatPanel",
    tags=_tags("ui", "chat", "completion", "request"),
)

LOG_UI_CHAT_REQUEST_ERROR = _constant(
    "LOG785",
    "ERROR",
    "Chat completion request failed",
    component="UI",
    subcomponent="ChatPanel",
    tags=_tags("ui", "chat", "completion", "request", "error"),
)

LOG_UI_CHAT_REQUEST_CANCELLED = _constant(
    "LOG786",
    "WARNING",
    "Chat completion request cancelled by user",
    component="UI",
    subcomponent="ChatPanel",
    tags=_tags("ui", "chat", "completion", "cancelled"),
)

LOG_UI_CHAT_PROXY_ENABLED = _constant(
    "LOG787",
    "INFO",
    "Proxy settings enabled for LLM operations",
    component="UI",
    subcomponent="ChatPanel",
    tags=_tags("ui", "chat", "proxy", "enabled"),
)

LOG_UI_CHAT_PROXY_DISABLED = _constant(
    "LOG788",
    "INFO",
    "Proxy settings disabled for LLM operations",
    component="UI",
    subcomponent="ChatPanel",
    tags=_tags("ui", "chat", "proxy", "disabled"),
)

LOG_UI_CHAT_CLEANUP_WARNING = _constant(
    "LOG789",
    "WARNING",
    "Chat panel cleanup warning",
    component="UI",
    subcomponent="ChatPanel",
    tags=_tags("ui", "chat", "cleanup", "warning"),
)


# ---------------------------------------------------------------------------
# Board Config Dialog constants (LOG790–LOG799)
# ---------------------------------------------------------------------------
LOG_UI_BOARD_CONFIG_DIALOG_OPENED = _constant(
    "LOG790",
    "INFO",
    "Board configuration dialog opened",
    component="UI",
    subcomponent="BoardConfigDialog",
    tags=_tags("ui", "board_config", "dialog", "opened"),
)

LOG_UI_BOARD_CONFIG_STATE_APPLIED = _constant(
    "LOG791",
    "INFO",
    "Custom board state applied",
    component="UI",
    subcomponent="BoardConfigDialog",
    tags=_tags("ui", "board_config", "state", "applied"),
)

LOG_UI_BOARD_CONFIG_VALIDATION_ERROR = _constant(
    "LOG792",
    "WARNING",
    "Board state validation failed",
    component="UI",
    subcomponent="BoardConfigDialog",
    tags=_tags("ui", "board_config", "validation", "error"),
)

LOG_UI_BOARD_CONFIG_PRESET_LOADED = _constant(
    "LOG793",
    "DEBUG",
    "Board preset position loaded",
    component="UI",
    subcomponent="BoardConfigDialog",
    tags=_tags("ui", "board_config", "preset", "loaded"),
)

LOG_UI_BOARD_CONFIG_PIECE_MOVED = _constant(
    "LOG794",
    "DEBUG",
    "Board piece moved",
    component="UI",
    subcomponent="BoardConfigDialog",
    tags=_tags("ui", "board_config", "piece", "moved"),
)

LOG_UI_BOARD_CONFIG_PIECE_REMOVED = _constant(
    "LOG795",
    "DEBUG",
    "Board piece removed",
    component="UI",
    subcomponent="BoardConfigDialog",
    tags=_tags("ui", "board_config", "piece", "removed"),
)

LOG_UI_BOARD_CONFIG_NOTATION_EDITED = _constant(
    "LOG796",
    "DEBUG",
    "Board notation manually edited",
    component="UI",
    subcomponent="BoardConfigDialog",
    tags=_tags("ui", "board_config", "notation", "edited"),
)

LOG_UI_BOARD_CONFIG_FACTORY_CREATE = _constant(
    "LOG797",
    "DEBUG",
    "Board config dialog created via factory",
    component="UI",
    subcomponent="BoardConfigDialogFactory",
    tags=_tags("ui", "board_config", "factory", "create"),
)

LOG_UI_BOARD_CONFIG_UNSUPPORTED_GAME = _constant(
    "LOG798",
    "WARNING",
    "Board configuration not supported for game",
    component="UI",
    subcomponent="BoardConfigDialogFactory",
    tags=_tags("ui", "board_config", "unsupported"),
)

LOG_UI_BOARD_CONFIG_ENV_INIT_CUSTOM = _constant(
    "LOG799",
    "INFO",
    "Environment initialized with custom board state",
    component="UI",
    subcomponent="MainWindow",
    tags=_tags("ui", "board_config", "env", "init"),
)


# ---------------------------------------------------------------------------
# Operators Grid Config Dialog constants (LOG800-LOG819)
# ---------------------------------------------------------------------------
LOG_OP_GRID_CONFIG_DIALOG_OPENED = _constant(
    "LOG800",
    "INFO",
    "Grid configuration dialog opened",
    component="Operators",
    subcomponent="GridConfigDialog",
    tags=_tags("operators", "grid_config", "dialog", "open"),
)

LOG_OP_GRID_CONFIG_STATE_APPLIED = _constant(
    "LOG801",
    "INFO",
    "Grid configuration state applied",
    component="Operators",
    subcomponent="GridConfigDialog",
    tags=_tags("operators", "grid_config", "state", "apply"),
)

LOG_OP_GRID_CONFIG_VALIDATION_ERROR = _constant(
    "LOG802",
    "WARNING",
    "Grid configuration validation error",
    component="Operators",
    subcomponent="GridConfigDialog",
    tags=_tags("operators", "grid_config", "validation", "error"),
)

LOG_OP_GRID_CONFIG_PRESET_LOADED = _constant(
    "LOG803",
    "DEBUG",
    "Grid configuration preset loaded",
    component="Operators",
    subcomponent="GridConfigDialog",
    tags=_tags("operators", "grid_config", "preset", "load"),
)

LOG_OP_GRID_CONFIG_OBJECT_PLACED = _constant(
    "LOG804",
    "DEBUG",
    "Object placed on grid",
    component="Operators",
    subcomponent="GridConfigDialog",
    tags=_tags("operators", "grid_config", "object", "place"),
)

LOG_OP_GRID_CONFIG_OBJECT_REMOVED = _constant(
    "LOG805",
    "DEBUG",
    "Object removed from grid",
    component="Operators",
    subcomponent="GridConfigDialog",
    tags=_tags("operators", "grid_config", "object", "remove"),
)

LOG_OP_GRID_CONFIG_FACTORY_CREATE = _constant(
    "LOG806",
    "DEBUG",
    "Grid config dialog created by factory",
    component="Operators",
    subcomponent="GridConfigDialogFactory",
    tags=_tags("operators", "grid_config", "factory", "create"),
)

LOG_OP_GRID_CONFIG_UNSUPPORTED_ENV = _constant(
    "LOG807",
    "WARNING",
    "Unsupported environment for grid configuration",
    component="Operators",
    subcomponent="GridConfigDialogFactory",
    tags=_tags("operators", "grid_config", "factory", "unsupported"),
)

LOG_OP_GRID_CONFIG_ENV_INIT_CUSTOM = _constant(
    "LOG808",
    "INFO",
    "Environment initialized with custom grid state",
    component="Operators",
    subcomponent="OperatorsTab",
    tags=_tags("operators", "grid_config", "env", "init"),
)

# Agent Linking constants (LOG809-LOG819)
LOG_OP_AGENT_LINK_DIALOG_OPENED = _constant(
    "LOG809",
    "INFO",
    "Agent link dialog opened",
    component="Operators",
    subcomponent="OperatorConfigWidget",
    tags=_tags("operators", "agent_link", "dialog", "open"),
)

LOG_OP_AGENT_LINK_GROUP_CREATED = _constant(
    "LOG810",
    "INFO",
    "Agent link group created",
    component="Operators",
    subcomponent="OperatorConfigWidget",
    tags=_tags("operators", "agent_link", "group", "create"),
)

LOG_OP_AGENT_LINKED = _constant(
    "LOG811",
    "INFO",
    "Agent added to link group",
    component="Operators",
    subcomponent="OperatorConfigWidget",
    tags=_tags("operators", "agent_link", "add"),
)

LOG_OP_AGENT_UNLINKED = _constant(
    "LOG812",
    "INFO",
    "Agent removed from link group",
    component="Operators",
    subcomponent="OperatorConfigWidget",
    tags=_tags("operators", "agent_link", "remove"),
)

LOG_OP_AGENT_LINK_GROUP_DISSOLVED = _constant(
    "LOG813",
    "INFO",
    "Agent link group dissolved",
    component="Operators",
    subcomponent="OperatorConfigWidget",
    tags=_tags("operators", "agent_link", "group", "dissolve"),
)

LOG_OP_AGENT_LINK_VALIDATION_ERROR = _constant(
    "LOG814",
    "WARNING",
    "Agent link validation error",
    component="Operators",
    subcomponent="OperatorConfigWidget",
    tags=_tags("operators", "agent_link", "validation", "error"),
)

LOG_OP_AGENT_LINK_PRIMARY_PROMOTED = _constant(
    "LOG815",
    "INFO",
    "Primary agent promoted in link group",
    component="Operators",
    subcomponent="OperatorConfigWidget",
    tags=_tags("operators", "agent_link", "primary", "promote"),
)

LOG_OP_AGENT_LINK_POLICY_SYNCED = _constant(
    "LOG816",
    "DEBUG",
    "Policy path synced to linked agents",
    component="Operators",
    subcomponent="OperatorConfigWidget",
    tags=_tags("operators", "agent_link", "policy", "sync"),
)

LOG_OP_AGENT_LINK_INCOMPATIBLE_CHANGE = _constant(
    "LOG817",
    "WARNING",
    "Incompatible change triggered automatic unlink",
    component="Operators",
    subcomponent="OperatorConfigWidget",
    tags=_tags("operators", "agent_link", "incompatible", "unlink"),
)


# ---------------------------------------------------------------------------
# Fast Lane / RunBus telemetry constants (LOG950–LOG959)
# ---------------------------------------------------------------------------
LOG_FASTLANE_CONNECTED = _constant(
    "LOG950",
    "INFO",
    "Fast lane consumer connected",
    component="UI",
    subcomponent="FastLane",
    tags=_tags("fastlane", "status", "connected"),
)

LOG_FASTLANE_UNAVAILABLE = _constant(
    "LOG951",
    "WARNING",
    "Fast lane consumer unavailable",
    component="UI",
    subcomponent="FastLane",
    tags=_tags("fastlane", "status", "warning"),
)

LOG_FASTLANE_QUEUE_DEPTH = _constant(
    "LOG952",
    "INFO",
    "Fast lane queue depth snapshot",
    component="UI",
    subcomponent="FastLane",
    tags=_tags("fastlane", "metrics", "queue_depth"),
)

LOG_FASTLANE_READER_LAG = _constant(
    "LOG953",
    "WARNING",
    "Fast lane reader lag detected",
    component="UI",
    subcomponent="FastLane",
    tags=_tags("fastlane", "lag", "warning"),
)

LOG_RUNBUS_UI_QUEUE_DEPTH = _constant(
    "LOG954",
    "INFO",
    "RunBus UI queue depth snapshot",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("runbus", "ui", "queue_depth"),
)

LOG_FASTLANE_HEADER_INVALID = _constant(
    "LOG955",
    "WARNING",
    "Fast lane shared memory header invalid",
    component="UI",
    subcomponent="FastLane",
    tags=_tags("fastlane", "header", "warning"),
)

LOG_FASTLANE_FRAME_READ_ERROR = _constant(
    "LOG956",
    "ERROR",
    "Fast lane frame read failed",
    component="UI",
    subcomponent="FastLane",
    tags=_tags("fastlane", "reader", "error"),
)

LOG_RUNBUS_DB_QUEUE_DEPTH = _constant(
    "LOG957",
    "INFO",
    "RunBus durable queue depth snapshot",
    component="Telemetry",
    subcomponent="DBSink",
    tags=_tags("runbus", "durable", "queue_depth"),
)

LOG_UI_FASTLANE_EVAL_SUMMARY_UPDATE = _constant(
    "LOG958",
    "INFO",
    "Fast lane evaluation summary updated",
    component="UI",
    subcomponent="FastLane",
    tags=_tags("fastlane", "eval", "summary"),
)

LOG_UI_FASTLANE_EVAL_SUMMARY_WARNING = _constant(
    "LOG959",
    "WARNING",
    "Fast lane evaluation summary warning",
    component="UI",
    subcomponent="FastLane",
    tags=_tags("fastlane", "eval", "warning"),
)

# ---------------------------------------------------------------------------
# FastLane Worker-side constants (LOG940–LOG949)
# Shared memory lifecycle: create, close, unlink, leak detection
# ---------------------------------------------------------------------------
LOG_FASTLANE_WRITER_CLOSED = _constant(
    "LOG940",
    "INFO",
    "FastLane shared memory writer closed and unlinked",
    component="Worker",
    subcomponent="FastLane",
    tags=_tags("fastlane", "shm", "cleanup"),
)

LOG_FASTLANE_SHM_STALE_UNLINKED = _constant(
    "LOG941",
    "WARNING",
    "FastLane stale shared memory segment unlinked from previous crashed run",
    component="Worker",
    subcomponent="FastLane",
    tags=_tags("fastlane", "shm", "stale", "cleanup"),
)

LOG_FASTLANE_SHM_UNLINK_FAILED = _constant(
    "LOG942",
    "ERROR",
    "FastLane shared memory unlink failed — orphaned /dev/shm segment may cause "
    "resource_tracker to spawn and leak memory until system OOM",
    component="Worker",
    subcomponent="FastLane",
    tags=_tags("fastlane", "shm", "leak", "error"),
)

LOG_FASTLANE_SHM_LEAK_RISK = _constant(
    "LOG943",
    "ERROR",
    "FastLane writer destroyed without unlink — resource_tracker process will spawn "
    "to clean up /dev/shm, but may spin indefinitely consuming RAM. "
    "This is a known cause of system-wide OOM crashes",
    component="Worker",
    subcomponent="FastLane",
    tags=_tags("fastlane", "shm", "leak", "oom", "critical"),
)


# ---------------------------------------------------------------------------
# SMAC Adapter constants (LOG960–LOG969)
# ---------------------------------------------------------------------------
LOG_SMAC_ENV_CREATED = _constant(
    "LOG960",
    "INFO",
    "SMAC environment created",
    component="Adapter",
    subcomponent="SMAC",
    tags=_tags("smac", "env", "created"),
)

LOG_SMAC_ENV_RESET = _constant(
    "LOG961",
    "INFO",
    "SMAC environment reset",
    component="Adapter",
    subcomponent="SMAC",
    tags=_tags("smac", "env", "reset"),
)

LOG_SMAC_STEP_SUMMARY = _constant(
    "LOG962",
    "DEBUG",
    "SMAC step summary",
    component="Adapter",
    subcomponent="SMAC",
    tags=_tags("smac", "step", "summary"),
)

LOG_SMAC_ENV_CLOSED = _constant(
    "LOG963",
    "INFO",
    "SMAC environment closed",
    component="Adapter",
    subcomponent="SMAC",
    tags=_tags("smac", "env", "closed"),
)

LOG_SMAC_RENDER_ERROR = _constant(
    "LOG964",
    "WARNING",
    "SMAC render failed",
    component="Adapter",
    subcomponent="SMAC",
    tags=_tags("smac", "render", "error"),
)

LOG_SMAC_SC2_PATH_MISSING = _constant(
    "LOG965",
    "ERROR",
    "StarCraft II installation not found (SC2PATH)",
    component="Adapter",
    subcomponent="SMAC",
    tags=_tags("smac", "sc2", "path", "error"),
)

LOG_SMAC_IMPORT_FALLBACK = _constant(
    "LOG966",
    "INFO",
    "SMAC import fallback triggered",
    component="Adapter",
    subcomponent="SMAC",
    tags=_tags("smac", "import", "fallback"),
)

LOG_SMAC_REPLAY_SAVED = _constant(
    "LOG967",
    "INFO",
    "SMAC replay saved",
    component="Adapter",
    subcomponent="SMAC",
    tags=_tags("smac", "replay", "saved"),
)

LOG_SMAC_ACTION_MASK_WARN = _constant(
    "LOG968",
    "WARNING",
    "SMAC action mask unavailable",
    component="Adapter",
    subcomponent="SMAC",
    tags=_tags("smac", "action_mask", "warning"),
)

LOG_SMAC_BATTLE_RESULT = _constant(
    "LOG969",
    "INFO",
    "SMAC battle result",
    component="Adapter",
    subcomponent="SMAC",
    tags=_tags("smac", "battle", "result"),
)


# ---------------------------------------------------------------------------
# RWARE (Robotic Warehouse) Adapter constants (LOG970-LOG975)
# ---------------------------------------------------------------------------
LOG_RWARE_ENV_CREATED = _constant(
    "LOG970",
    "INFO",
    "RWARE environment created",
    component="Adapter",
    subcomponent="RWARE",
    tags=_tags("rware", "environment", "lifecycle"),
)
LOG_RWARE_ENV_RESET = _constant(
    "LOG971",
    "INFO",
    "RWARE environment reset",
    component="Adapter",
    subcomponent="RWARE",
    tags=_tags("rware", "episode", "lifecycle"),
)
LOG_RWARE_STEP_SUMMARY = _constant(
    "LOG972",
    "DEBUG",
    "RWARE step summary",
    component="Adapter",
    subcomponent="RWARE",
    tags=_tags("rware", "step"),
)
LOG_RWARE_ENV_CLOSED = _constant(
    "LOG973",
    "INFO",
    "RWARE environment closed",
    component="Adapter",
    subcomponent="RWARE",
    tags=_tags("rware", "environment", "lifecycle"),
)
LOG_RWARE_RENDER_ERROR = _constant(
    "LOG974",
    "WARNING",
    "RWARE render error",
    component="Adapter",
    subcomponent="RWARE",
    tags=_tags("rware", "render", "error"),
)
LOG_RWARE_DELIVERY = _constant(
    "LOG975",
    "INFO",
    "RWARE shelf delivery event",
    component="Adapter",
    subcomponent="RWARE",
    tags=_tags("rware", "delivery", "reward"),
)


# ---------------------------------------------------------------------------
# Griddly Adapter constants (LOG4090–LOG4094)
# ---------------------------------------------------------------------------
LOG_GRIDDLY_ENV_CREATED = _constant(
    "LOG4090",
    "INFO",
    "Griddly environment created",
    component="Adapter",
    subcomponent="Griddly",
    tags=_tags("griddly", "environment", "lifecycle"),
)
LOG_GRIDDLY_ENV_RESET = _constant(
    "LOG4091",
    "INFO",
    "Griddly environment reset",
    component="Adapter",
    subcomponent="Griddly",
    tags=_tags("griddly", "episode", "lifecycle"),
)
LOG_GRIDDLY_STEP_SUMMARY = _constant(
    "LOG4092",
    "DEBUG",
    "Griddly step summary",
    component="Adapter",
    subcomponent="Griddly",
    tags=_tags("griddly", "step"),
)
LOG_GRIDDLY_ENV_CLOSED = _constant(
    "LOG4093",
    "INFO",
    "Griddly environment closed",
    component="Adapter",
    subcomponent="Griddly",
    tags=_tags("griddly", "environment", "lifecycle"),
)
LOG_GRIDDLY_RENDER_ERROR = _constant(
    "LOG4094",
    "WARNING",
    "Griddly render error",
    component="Adapter",
    subcomponent="Griddly",
    tags=_tags("griddly", "render", "error"),
)


# ---------------------------------------------------------------------------
# PettingZoo Adapter Specific constants (LOG4095–LOG4110)
# ---------------------------------------------------------------------------

# PettingZoo/butterfly Adapter Specific Environment
LOG_PETTINGZOO_BUTTERFLY_INPUT_CONFIGURED = _constant(
    "LOG4095",
    "INFO",
    "PettingZoo Butterfly environment input configured",
    component="Adapter",
    subcomponent="PettingZoo/Butterfly",
    tags=_tags("pettingzoo", "butterfly", "input", "keyboard"),
)

LOG_PISTONBALL_ACTION_RESOLVED = _constant(
    "LOG4096",
    "DEBUG",
    "Pistonball action resolved from keyboard input",
    component="Adapter",
    subcomponent="PettingZoo/Butterfly",
    tags=_tags("pettingzoo", "butterfly", "pistonball", "action"),
)

LOG_KNIGHTS_ARCHERS_ZOMBIES_ACTION_RESOLVED = _constant(
    "LOG4097",
    "DEBUG",
    "Knights Archers Zombies action resolved from keyboard input",
    component="Adapter",
    subcomponent="PettingZoo/Butterfly",
    tags=_tags("pettingzoo", "butterfly", "knights_archers_zombies", "action"),
)

LOG_COOPERATIVE_PONG_ACTION_RESOLVED = _constant(
    "LOG4098",
    "DEBUG",
    "Cooperative Pong action resolved from keyboard input",
    component="Adapter",
    subcomponent="PettingZoo/Butterfly",
    tags=_tags("pettingzoo", "butterfly", "cooperative_pong", "action"),
)

# PettingZoo/classic Adapter Specific Environment
LOG_PETTINGZOO_CLASSIC_INPUT_CONFIGURED = _constant(
    "LOG4099",
    "INFO",
    "PettingZoo Classic environment input configured",
    component="Adapter",
    subcomponent="PettingZoo/Classic",
    tags=_tags("pettingzoo", "classic", "input", "keyboard"),
)

LOG_CHESS_MOVE_INPUT = _constant(
    "LOG4100",
    "DEBUG",
    "Chess move input from keyboard",
    component="Adapter",
    subcomponent="PettingZoo/Classic",
    tags=_tags("pettingzoo", "classic", "chess", "input"),
)

LOG_GO_MOVE_INPUT = _constant(
    "LOG4101",
    "DEBUG",
    "Go move input from keyboard",
    component="Adapter",
    subcomponent="PettingZoo/Classic",
    tags=_tags("pettingzoo", "classic", "go", "input"),
)

LOG_CONNECT_FOUR_MOVE_INPUT = _constant(
    "LOG4102",
    "DEBUG",
    "Connect Four move input from keyboard",
    component="Adapter",
    subcomponent="PettingZoo/Classic",
    tags=_tags("pettingzoo", "classic", "connect_four", "input"),
)

LOG_HANABI_ACTION_INPUT = _constant(
    "LOG4103",
    "DEBUG",
    "Hanabi action input from keyboard",
    component="Adapter",
    subcomponent="PettingZoo/Classic",
    tags=_tags("pettingzoo", "classic", "hanabi", "input"),
)

LOG_LEDUC_HOLDEM_ACTION_INPUT = _constant(
    "LOG4104",
    "DEBUG",
    "Leduc Hold'em action input from keyboard",
    component="Adapter",
    subcomponent="PettingZoo/Classic",
    tags=_tags("pettingzoo", "classic", "leduc_holdem", "input"),
)

LOG_RPS_ACTION_INPUT = _constant(
    "LOG4105",
    "DEBUG",
    "Rock Paper Scissors action input from keyboard",
    component="Adapter",
    subcomponent="PettingZoo/Classic",
    tags=_tags("pettingzoo", "classic", "rps", "input"),
)

LOG_TEXAS_HOLDEM_ACTION_INPUT = _constant(
    "LOG4106",
    "DEBUG",
    "Texas Hold'em action input from keyboard",
    component="Adapter",
    subcomponent="PettingZoo/Classic",
    tags=_tags("pettingzoo", "classic", "texas_holdem", "input"),
)

# HeMAC Adapter Specific Environment
LOG_HEMAC_ENV_CONFIGURED = _constant(
    "LOG4107",
    "INFO",
    "HeMAC environment configured",
    component="Adapter",
    subcomponent="HeMAC",
    tags=_tags("hemac", "heterogeneous", "multi_agent", "config"),
)

LOG_HEMAC_AGENT_ACTION = _constant(
    "LOG4108",
    "DEBUG",
    "HeMAC agent action executed",
    component="Adapter",
    subcomponent="HeMAC",
    tags=_tags("hemac", "agent", "action"),
)

LOG_HEMAC_TARGET_REACHED = _constant(
    "LOG4109",
    "INFO",
    "HeMAC target reached by agent",
    component="Adapter",
    subcomponent="HeMAC",
    tags=_tags("hemac", "target", "success"),
)

LOG_HEMAC_ENERGY_DEPLETED = _constant(
    "LOG4110",
    "WARNING",
    "HeMAC quadcopter energy depleted",
    component="Adapter",
    subcomponent="HeMAC",
    tags=_tags("hemac", "energy", "warning"),
)


# ---------------------------------------------------------------------------
# Google Research Football Adapter constants (LOG4120–LOG4125)
# ---------------------------------------------------------------------------
LOG_GFOOTBALL_ENV_CREATED = _constant(
    "LOG4120",
    "INFO",
    "GFootball environment created",
    component="Adapter",
    subcomponent="GFootball",
    tags=_tags("gfootball", "environment", "lifecycle"),
)
LOG_GFOOTBALL_ENV_RESET = _constant(
    "LOG4121",
    "INFO",
    "GFootball environment reset",
    component="Adapter",
    subcomponent="GFootball",
    tags=_tags("gfootball", "episode", "lifecycle"),
)
LOG_GFOOTBALL_STEP_SUMMARY = _constant(
    "LOG4122",
    "DEBUG",
    "GFootball step summary",
    component="Adapter",
    subcomponent="GFootball",
    tags=_tags("gfootball", "step"),
)
LOG_GFOOTBALL_ENV_CLOSED = _constant(
    "LOG4123",
    "INFO",
    "GFootball environment closed",
    component="Adapter",
    subcomponent="GFootball",
    tags=_tags("gfootball", "environment", "lifecycle"),
)
LOG_GFOOTBALL_RENDER_ERROR = _constant(
    "LOG4124",
    "WARNING",
    "GFootball render error",
    component="Adapter",
    subcomponent="GFootball",
    tags=_tags("gfootball", "render", "error"),
)
LOG_GFOOTBALL_GOAL = _constant(
    "LOG4125",
    "INFO",
    "GFootball goal scored",
    component="Adapter",
    subcomponent="GFootball",
    tags=_tags("gfootball", "goal", "reward"),
)


# ---------------------------------------------------------------------------
# Utility constants (LOG801–LOG809)
# ---------------------------------------------------------------------------
LOG_UTIL_QT_RESEED_SKIPPED = _constant(
    "LOG801",
    "DEBUG",
    "Qt random reseed skipped",
    component="Utility",
    subcomponent="Seeding",
    tags=_tags("utility", "seeding", "qt"),
)

LOG_UTIL_QT_STATE_CAPTURE_FAILED = _constant(
    "LOG802",
    "ERROR",
    "Failed to capture Qt random generator state",
    component="Utility",
    subcomponent="Seeding",
    tags=_tags("utility", "seeding", "qt", "error"),
)

LOG_UTIL_SEED_CALLBACK_FAILED = _constant(
    "LOG803",
    "ERROR",
    "Seed callback failed",
    component="Utility",
    subcomponent="Seeding",
    tags=_tags("utility", "seeding", "callback", "error"),
)

# JSON serialization helpers
LOG_UTIL_JSON_NUMPY_SCALAR_COERCE_FAILED = _constant(
    "LOG804",
    "DEBUG",
    "Safe JSON: numpy scalar coerce failed; using fallback",
    component="Utility",
    subcomponent="Serialization",
    tags=_tags("utility", "json", "numpy", "coerce"),
)


# ---------------------------------------------------------------------------
# Go AI Service constants (LOG850–LOG859)
# ---------------------------------------------------------------------------
LOG_GO_AI_KATAGO_START = _constant(
    "LOG850",
    "INFO",
    "KataGo engine started",
    component="Service",
    subcomponent="GoAI",
    tags=_tags("katago", "start"),
)

LOG_GO_AI_KATAGO_STOP = _constant(
    "LOG851",
    "INFO",
    "KataGo engine stopped",
    component="Service",
    subcomponent="GoAI",
    tags=_tags("katago", "stop"),
)

LOG_GO_AI_KATAGO_UNAVAILABLE = _constant(
    "LOG852",
    "WARNING",
    "KataGo not available, falling back to random AI",
    component="Service",
    subcomponent="GoAI",
    tags=_tags("katago", "fallback"),
)

LOG_GO_AI_GNUGO_START = _constant(
    "LOG853",
    "INFO",
    "GNU Go engine started",
    component="Service",
    subcomponent="GoAI",
    tags=_tags("gnugo", "start"),
)

LOG_GO_AI_GNUGO_STOP = _constant(
    "LOG854",
    "INFO",
    "GNU Go engine stopped",
    component="Service",
    subcomponent="GoAI",
    tags=_tags("gnugo", "stop"),
)

LOG_GO_AI_GNUGO_UNAVAILABLE = _constant(
    "LOG855",
    "WARNING",
    "GNU Go not available",
    component="Service",
    subcomponent="GoAI",
    tags=_tags("gnugo", "fallback"),
)

LOG_GO_AI_GTP_ERROR = _constant(
    "LOG856",
    "ERROR",
    "GTP communication error",
    component="Service",
    subcomponent="GoAI",
    tags=_tags("gtp", "error"),
)

LOG_GO_AI_MOVE_GENERATED = _constant(
    "LOG857",
    "DEBUG",
    "Go AI move generated",
    component="Service",
    subcomponent="GoAI",
    tags=_tags("move", "generated"),
)


# ---------------------------------------------------------------------------
# Worker constants (LOG901–LOG915)
# ---------------------------------------------------------------------------
LOG_WORKER_RUNTIME_EVENT = _constant(
    "LOG901",
    "INFO",
    "Worker runtime event",
    component="Worker",
    subcomponent="Runtime",
    tags=_tags("worker", "runtime"),
)

LOG_WORKER_RUNTIME_WARNING = _constant(
    "LOG902",
    "WARNING",
    "Worker runtime warning",
    component="Worker",
    subcomponent="Runtime",
    tags=_tags("worker", "runtime", "warning"),
)

LOG_WORKER_RUNTIME_ERROR = _constant(
    "LOG903",
    "ERROR",
    "Worker runtime error",
    component="Worker",
    subcomponent="Runtime",
    tags=_tags("worker", "runtime", "error"),
)

LOG_WORKER_RUNTIME_DEBUG = _constant(
    "LOG904",
    "DEBUG",
    "Worker runtime debug",
    component="Worker",
    subcomponent="Runtime",
    tags=_tags("worker", "runtime", "debug"),
)

LOG_WORKER_RUNTIME_JSON_SANITIZED = _constant(
    "LOG916",
    "INFO",
    "Worker sanitized telemetry payload for JSON compatibility",
    component="Worker",
    subcomponent="Runtime",
    tags=_tags("worker", "runtime", "telemetry", "json"),
)

LOG_WORKER_CONFIG_EVENT = _constant(
    "LOG905",
    "INFO",
    "Worker configuration event",
    component="Worker",
    subcomponent="Config",
    tags=_tags("worker", "config"),
)

LOG_WORKER_CONFIG_WARNING = _constant(
    "LOG906",
    "WARNING",
    "Worker configuration warning",
    component="Worker",
    subcomponent="Config",
    tags=_tags("worker", "config", "warning"),
)

LOG_WORKER_CONFIG_UI_PATH = _constant(
    "LOG907",
    "INFO",
    "Worker UI-only path settings applied",
    component="Worker",
    subcomponent="Config",
    tags=_tags("worker", "config", "ui_only_path"),
)

LOG_WORKER_CONFIG_DURABLE_PATH = _constant(
    "LOG908",
    "INFO",
    "Worker telemetry durable path settings applied",
    component="Worker",
    subcomponent="Config",
    tags=_tags("worker", "config", "telemetry_path"),
)

LOG_WORKER_POLICY_EVENT = _constant(
    "LOG909",
    "INFO",
    "Worker policy event",
    component="Worker",
    subcomponent="Policy",
    tags=_tags("worker", "policy"),
)

LOG_WORKER_POLICY_WARNING = _constant(
    "LOG910",
    "WARNING",
    "Worker policy warning",
    component="Worker",
    subcomponent="Policy",
    tags=_tags("worker", "policy", "warning"),
)

LOG_WORKER_POLICY_ERROR = _constant(
    "LOG911",
    "ERROR",
    "Worker policy error",
    component="Worker",
    subcomponent="Policy",
    tags=_tags("worker", "policy", "error"),
)

LOG_WORKER_POLICY_EVAL_STARTED = _constant(
    "LOG918",
    "INFO",
    "Worker policy evaluation started",
    component="Worker",
    subcomponent="Policy",
    tags=_tags("worker", "policy", "eval", "start"),
)

LOG_WORKER_POLICY_EVAL_COMPLETED = _constant(
    "LOG919",
    "INFO",
    "Worker policy evaluation completed",
    component="Worker",
    subcomponent="Policy",
    tags=_tags("worker", "policy", "eval", "complete"),
)

LOG_WORKER_POLICY_EVAL_BATCH_STARTED = _constant(
    "LOG961",
    "INFO",
    "Worker policy evaluation batch started",
    component="Worker",
    subcomponent="Policy",
    tags=_tags("worker", "policy", "eval", "batch", "start"),
)

LOG_WORKER_POLICY_EVAL_BATCH_COMPLETED = _constant(
    "LOG962",
    "INFO",
    "Worker policy evaluation batch completed",
    component="Worker",
    subcomponent="Policy",
    tags=_tags("worker", "policy", "eval", "batch", "complete"),
)

LOG_WORKER_POLICY_LOAD_FAILED = _constant(
    "LOG920",
    "ERROR",
    "Worker policy checkpoint missing",
    component="Worker",
    subcomponent="Policy",
    tags=_tags("worker", "policy", "missing"),
)


# ---------------------------------------------------------------------------
# Episode Counter constants (LOG921–LOG930)
# ---------------------------------------------------------------------------
LOG_COUNTER_INITIALIZED = _constant(
    "LOG921",
    "INFO",
    "Episode counter initialized",
    component="Core",
    subcomponent="EpisodeCounter",
    tags=_tags("counter", "initialization", "info"),
)

LOG_COUNTER_RESUME_SUCCESS = _constant(
    "LOG922",
    "INFO",
    "Episode counter resumed from database",
    component="Core",
    subcomponent="EpisodeCounter",
    tags=_tags("counter", "resume", "persistence", "info"),
)

LOG_COUNTER_RESUME_FAILURE = _constant(
    "LOG923",
    "ERROR",
    "Failed to resume episode counter from database",
    component="Core",
    subcomponent="EpisodeCounter",
    tags=_tags("counter", "resume", "persistence", "error"),
)

LOG_COUNTER_MAX_REACHED = _constant(
    "LOG924",
    "ERROR",
    "Maximum episodes per run limit reached",
    component="Core",
    subcomponent="EpisodeCounter",
    tags=_tags("counter", "bounds", "limit", "error"),
)

LOG_COUNTER_INVALID_STATE = _constant(
    "LOG925",
    "ERROR",
    "Episode counter in invalid state",
    component="Core",
    subcomponent="EpisodeCounter",
    tags=_tags("counter", "state", "error"),
)

LOG_COUNTER_CONCURRENCY_ERROR = _constant(
    "LOG926",
    "ERROR",
    "Episode counter concurrency error",
    component="Core",
    subcomponent="EpisodeCounter",
    tags=_tags("counter", "concurrency", "threading", "error"),
)

LOG_COUNTER_PERSISTENCE_ERROR = _constant(
    "LOG927",
    "ERROR",
    "Episode counter persistence error",
    component="Core",
    subcomponent="EpisodeCounter",
    tags=_tags("counter", "persistence", "database", "error"),
)

LOG_COUNTER_VALIDATION_ERROR = _constant(
    "LOG928",
    "ERROR",
    "Episode counter validation error",
    component="Core",
    subcomponent="EpisodeCounter",
    tags=_tags("counter", "validation", "bounds", "error"),
)

LOG_COUNTER_NEXT_EPISODE = _constant(
    "LOG929",
    "DEBUG",
    "Episode counter allocated next episode index",
    component="Core",
    subcomponent="EpisodeCounter",
    tags=_tags("counter", "allocation", "debug"),
)

LOG_COUNTER_RESET = _constant(
    "LOG930",
    "INFO",
    "Episode counter reset for new run",
    component="Core",
    subcomponent="EpisodeCounter",
    tags=_tags("counter", "reset", "info"),
)


# ---------------------------------------------------------------------------
# Ray Worker constants (LOG970–LOG979)
# ---------------------------------------------------------------------------
LOG_RAY_WORKER_RUNTIME_STARTED = _constant(
    "LOG970",
    "INFO",
    "Ray RLlib worker runtime started",
    component="Worker",
    subcomponent="RayRuntime",
    tags=_tags("ray", "worker", "runtime", "lifecycle"),
)

LOG_RAY_WORKER_RUNTIME_STOPPED = _constant(
    "LOG971",
    "INFO",
    "Ray RLlib worker runtime stopped",
    component="Worker",
    subcomponent="RayRuntime",
    tags=_tags("ray", "worker", "runtime", "lifecycle"),
)

LOG_RAY_WORKER_RUNTIME_ERROR = _constant(
    "LOG972",
    "ERROR",
    "Ray RLlib worker runtime error",
    component="Worker",
    subcomponent="RayRuntime",
    tags=_tags("ray", "worker", "runtime", "error"),
)

LOG_RAY_WORKER_TRAINING_STARTED = _constant(
    "LOG973",
    "INFO",
    "Ray RLlib training started",
    component="Worker",
    subcomponent="RayRuntime",
    tags=_tags("ray", "worker", "training", "start"),
)

LOG_RAY_WORKER_TRAINING_COMPLETED = _constant(
    "LOG974",
    "INFO",
    "Ray RLlib training completed",
    component="Worker",
    subcomponent="RayRuntime",
    tags=_tags("ray", "worker", "training", "complete"),
)

LOG_RAY_WORKER_CHECKPOINT_SAVED = _constant(
    "LOG975",
    "INFO",
    "Ray RLlib checkpoint saved",
    component="Worker",
    subcomponent="RayRuntime",
    tags=_tags("ray", "worker", "checkpoint", "save"),
)

LOG_RAY_WORKER_FASTLANE_ENABLED = _constant(
    "LOG976",
    "INFO",
    "Ray RLlib FastLane streaming enabled",
    component="Worker",
    subcomponent="RayRuntime",
    tags=_tags("ray", "worker", "fastlane", "enabled"),
)

LOG_RAY_WORKER_ENV_WRAPPED = _constant(
    "LOG977",
    "INFO",
    "PettingZoo environment wrapped for Ray RLlib",
    component="Worker",
    subcomponent="RayRuntime",
    tags=_tags("ray", "worker", "environment", "wrapped"),
)

LOG_RAY_WORKER_POLICY_LOADED = _constant(
    "LOG978",
    "INFO",
    "Ray RLlib policy loaded from checkpoint",
    component="Worker",
    subcomponent="RayRuntime",
    tags=_tags("ray", "worker", "policy", "loaded"),
)

LOG_RAY_WORKER_ANALYTICS_WRITTEN = _constant(
    "LOG979",
    "INFO",
    "Ray RLlib analytics manifest written",
    component="Worker",
    subcomponent="RayRuntime",
    tags=_tags("ray", "worker", "analytics", "manifest"),
)

# =========================================================================
# Ray Policy Evaluation Constants
# =========================================================================

LOG_RAY_EVAL_REQUESTED = _constant(
    "LOG980",
    "INFO",
    "Ray policy evaluation requested",
    component="Worker",
    subcomponent="RayEvaluator",
    tags=_tags("ray", "evaluation", "requested"),
)

LOG_RAY_EVAL_SETUP_STARTED = _constant(
    "LOG981",
    "INFO",
    "Ray policy evaluation setup started",
    component="Worker",
    subcomponent="RayEvaluator",
    tags=_tags("ray", "evaluation", "setup", "started"),
)

LOG_RAY_EVAL_SETUP_COMPLETED = _constant(
    "LOG982",
    "INFO",
    "Ray policy evaluation setup completed",
    component="Worker",
    subcomponent="RayEvaluator",
    tags=_tags("ray", "evaluation", "setup", "completed"),
)

LOG_RAY_EVAL_EPISODE_STARTED = _constant(
    "LOG983",
    "DEBUG",
    "Ray policy evaluation episode started",
    component="Worker",
    subcomponent="RayEvaluator",
    tags=_tags("ray", "evaluation", "episode", "started"),
)

LOG_RAY_EVAL_EPISODE_COMPLETED = _constant(
    "LOG984",
    "INFO",
    "Ray policy evaluation episode completed",
    component="Worker",
    subcomponent="RayEvaluator",
    tags=_tags("ray", "evaluation", "episode", "completed"),
)

LOG_RAY_EVAL_RUN_COMPLETED = _constant(
    "LOG985",
    "INFO",
    "Ray policy evaluation run completed",
    component="Worker",
    subcomponent="RayEvaluator",
    tags=_tags("ray", "evaluation", "completed"),
)

LOG_RAY_EVAL_ERROR = _constant(
    "LOG986",
    "ERROR",
    "Ray policy evaluation error",
    component="Worker",
    subcomponent="RayEvaluator",
    tags=_tags("ray", "evaluation", "error"),
)

LOG_RAY_EVAL_FASTLANE_CONNECTED = _constant(
    "LOG987",
    "INFO",
    "Ray evaluation FastLane connected",
    component="Worker",
    subcomponent="RayEvaluator",
    tags=_tags("ray", "evaluation", "fastlane", "connected"),
)

LOG_RAY_EVAL_POLICY_LOADED = _constant(
    "LOG988",
    "INFO",
    "Ray evaluation policy loaded from checkpoint",
    component="Worker",
    subcomponent="RayEvaluator",
    tags=_tags("ray", "evaluation", "policy", "loaded"),
)

LOG_RAY_EVAL_TAB_CREATED = _constant(
    "LOG989",
    "INFO",
    "Ray evaluation FastLane tab created",
    component="UI",
    subcomponent="MainWindow",
    tags=_tags("ray", "evaluation", "tab", "created"),
)


# ---------------------------------------------------------------------------
# XuanCe Worker constants (LOG990–LOG999)
# ---------------------------------------------------------------------------
LOG_XUANCE_WORKER_RUNTIME_STARTED = _constant(
    "LOG990",
    "INFO",
    "XuanCe worker runtime started",
    component="Worker",
    subcomponent="XuanCeRuntime",
    tags=_tags("xuance", "worker", "runtime", "lifecycle"),
)

LOG_XUANCE_WORKER_RUNTIME_STOPPED = _constant(
    "LOG991",
    "INFO",
    "XuanCe worker runtime stopped",
    component="Worker",
    subcomponent="XuanCeRuntime",
    tags=_tags("xuance", "worker", "runtime", "lifecycle"),
)

LOG_XUANCE_WORKER_RUNTIME_ERROR = _constant(
    "LOG992",
    "ERROR",
    "XuanCe worker runtime error",
    component="Worker",
    subcomponent="XuanCeRuntime",
    tags=_tags("xuance", "worker", "runtime", "error"),
)

LOG_XUANCE_WORKER_TRAINING_STARTED = _constant(
    "LOG993",
    "INFO",
    "XuanCe training started",
    component="Worker",
    subcomponent="XuanCeRuntime",
    tags=_tags("xuance", "worker", "training", "start"),
)

LOG_XUANCE_WORKER_TRAINING_COMPLETED = _constant(
    "LOG994",
    "INFO",
    "XuanCe training completed",
    component="Worker",
    subcomponent="XuanCeRuntime",
    tags=_tags("xuance", "worker", "training", "complete"),
)

LOG_XUANCE_WORKER_CHECKPOINT_SAVED = _constant(
    "LOG995",
    "INFO",
    "XuanCe checkpoint saved",
    component="Worker",
    subcomponent="XuanCeRuntime",
    tags=_tags("xuance", "worker", "checkpoint", "save"),
)

LOG_XUANCE_WORKER_RUNNER_CREATED = _constant(
    "LOG996",
    "INFO",
    "XuanCe runner created",
    component="Worker",
    subcomponent="XuanCeRuntime",
    tags=_tags("xuance", "worker", "runner", "created"),
)

LOG_XUANCE_WORKER_CONFIG_LOADED = _constant(
    "LOG997",
    "INFO",
    "XuanCe configuration loaded",
    component="Worker",
    subcomponent="XuanCeRuntime",
    tags=_tags("xuance", "worker", "config", "loaded"),
)

LOG_XUANCE_WORKER_BENCHMARK_STARTED = _constant(
    "LOG998",
    "INFO",
    "XuanCe benchmark mode started",
    component="Worker",
    subcomponent="XuanCeRuntime",
    tags=_tags("xuance", "worker", "benchmark", "start"),
)

LOG_XUANCE_WORKER_DEBUG = _constant(
    "LOG999",
    "DEBUG",
    "XuanCe worker debug event",
    component="Worker",
    subcomponent="XuanCeRuntime",
    tags=_tags("xuance", "worker", "debug"),
)


# ---------------------------------------------------------------------------
# Human Action Debug constants (LOG1000–LOG1009)
# ---------------------------------------------------------------------------
LOG_HUMAN_ACTION_BUTTON_CLICKED = _constant(
    "LOG1000",
    "INFO",
    "Human action button clicked in render container",
    component="UI",
    subcomponent="OperatorRenderContainer",
    tags=_tags("human", "action", "button", "click", "debug"),
)

LOG_HUMAN_ACTION_RECEIVED = _constant(
    "LOG1001",
    "INFO",
    "Human action received in main window handler",
    component="UI",
    subcomponent="MainWindow",
    tags=_tags("human", "action", "received", "debug"),
)

LOG_HUMAN_ACTION_SENT_TO_SUBPROCESS = _constant(
    "LOG1002",
    "INFO",
    "Human action sent to subprocess",
    component="UI",
    subcomponent="MainWindow",
    tags=_tags("human", "action", "subprocess", "sent", "debug"),
)

LOG_HUMAN_ACTION_SIGNAL_EMITTED = _constant(
    "LOG1003",
    "INFO",
    "Human action signal emitted from container",
    component="UI",
    subcomponent="OperatorRenderContainer",
    tags=_tags("human", "action", "signal", "emitted", "debug"),
)


# ---------------------------------------------------------------------------
# CleanRL Worker constants (LOG431–LOG445)
# ---------------------------------------------------------------------------
LOG_WORKER_CLEANRL_RUNTIME_STARTED = _constant(
    "LOG431",
    "INFO",
    "CleanRL worker runtime started",
    component="Worker",
    subcomponent="CleanRLRuntime",
    tags=_tags("cleanrl", "worker", "runtime", "lifecycle"),
)

LOG_WORKER_CLEANRL_RUNTIME_COMPLETED = _constant(
    "LOG432",
    "INFO",
    "CleanRL worker runtime completed",
    component="Worker",
    subcomponent="CleanRLRuntime",
    tags=_tags("cleanrl", "worker", "runtime", "lifecycle"),
)

LOG_WORKER_CLEANRL_RUNTIME_FAILED = _constant(
    "LOG433",
    "ERROR",
    "CleanRL worker runtime failed",
    component="Worker",
    subcomponent="CleanRLRuntime",
    tags=_tags("cleanrl", "worker", "runtime", "error"),
)

LOG_WORKER_CLEANRL_MODULE_RESOLVED = _constant(
    "LOG434",
    "DEBUG",
    "CleanRL algorithm module resolved",
    component="Worker",
    subcomponent="CleanRLRuntime",
    tags=_tags("cleanrl", "worker", "module", "resolution"),
)

LOG_WORKER_CLEANRL_CONFIG_LOADED = _constant(
    "LOG435",
    "INFO",
    "CleanRL configuration loaded",
    component="Worker",
    subcomponent="CleanRLRuntime",
    tags=_tags("cleanrl", "worker", "config"),
)

LOG_WORKER_CLEANRL_TENSORBOARD_ENABLED = _constant(
    "LOG436",
    "INFO",
    "CleanRL TensorBoard logging enabled",
    component="Worker",
    subcomponent="CleanRLRuntime",
    tags=_tags("cleanrl", "worker", "tensorboard", "analytics"),
)

LOG_WORKER_CLEANRL_WANDB_ENABLED = _constant(
    "LOG437",
    "INFO",
    "CleanRL Weights & Biases tracking enabled",
    component="Worker",
    subcomponent="CleanRLRuntime",
    tags=_tags("cleanrl", "worker", "wandb", "analytics"),
)

LOG_WORKER_CLEANRL_HEARTBEAT = _constant(
    "LOG438",
    "DEBUG",
    "CleanRL worker heartbeat",
    component="Worker",
    subcomponent="CleanRLRuntime",
    tags=_tags("cleanrl", "worker", "heartbeat"),
)

LOG_WORKER_CLEANRL_SUBPROCESS_STARTED = _constant(
    "LOG439",
    "INFO",
    "CleanRL subprocess started",
    component="Worker",
    subcomponent="CleanRLRuntime",
    tags=_tags("cleanrl", "worker", "subprocess"),
)

LOG_WORKER_CLEANRL_SUBPROCESS_FAILED = _constant(
    "LOG440",
    "ERROR",
    "CleanRL subprocess failed",
    component="Worker",
    subcomponent="CleanRLRuntime",
    tags=_tags("cleanrl", "worker", "subprocess", "error"),
)

LOG_WORKER_CLEANRL_CHECKPOINT_SAVED = _constant(
    "LOG441",
    "INFO",
    "CleanRL checkpoint saved",
    component="Worker",
    subcomponent="CleanRLRuntime",
    tags=_tags("cleanrl", "worker", "checkpoint", "save"),
)

LOG_WORKER_CLEANRL_CHECKPOINT_LOADED = _constant(
    "LOG442",
    "INFO",
    "CleanRL checkpoint loaded",
    component="Worker",
    subcomponent="CleanRLRuntime",
    tags=_tags("cleanrl", "worker", "checkpoint", "load"),
)

LOG_WORKER_CLEANRL_ANALYTICS_MANIFEST_CREATED = _constant(
    "LOG443",
    "INFO",
    "CleanRL analytics manifest created",
    component="Worker",
    subcomponent="CleanRLRuntime",
    tags=_tags("cleanrl", "worker", "analytics", "manifest"),
)

LOG_WORKER_CLEANRL_EVAL_MODE_STARTED = _constant(
    "LOG444",
    "INFO",
    "CleanRL evaluation mode started",
    component="Worker",
    subcomponent="CleanRLRuntime",
    tags=_tags("cleanrl", "worker", "eval", "mode"),
)

LOG_WORKER_CLEANRL_EVAL_MODE_COMPLETED = _constant(
    "LOG445",
    "INFO",
    "CleanRL evaluation mode completed",
    component="Worker",
    subcomponent="CleanRLRuntime",
    tags=_tags("cleanrl", "worker", "eval", "mode"),
)

LOG_WORKER_CLEANRL_SUBPROCESS_STDERR = _constant(
    "LOG463",
    "WARNING",
    "CleanRL subprocess stderr",
    component="Worker",
    subcomponent="CleanRLRuntime",
    tags=_tags("cleanrl", "worker", "runtime", "stderr"),
)

LOG_WORKER_CLEANRL_MAIN_ENTRYPOINT_MISSING = _constant(
    "LOG464",
    "WARNING",
    "CleanRL module lacks main() entrypoint, falling back to subprocess",
    component="Worker",
    subcomponent="CleanRLRuntime",
    tags=_tags("cleanrl", "worker", "runtime", "fallback"),
)

LOG_WORKER_CLEANRL_TRAINING_STARTED = _constant(
    "LOG465",
    "INFO",
    "CleanRL training started",
    component="Worker",
    subcomponent="CleanRLRuntime",
    tags=_tags("cleanrl", "worker", "training", "start"),
)

LOG_WORKER_CLEANRL_TRAINING_COMPLETED = _constant(
    "LOG466",
    "INFO",
    "CleanRL training completed",
    component="Worker",
    subcomponent="CleanRLRuntime",
    tags=_tags("cleanrl", "worker", "training", "complete"),
)

LOG_WORKER_CLEANRL_MODULE_EXECUTION_FAILED = _constant(
    "LOG467",
    "ERROR",
    "CleanRL module execution failed",
    component="Worker",
    subcomponent="CleanRLRuntime",
    tags=_tags("cleanrl", "worker", "module", "execution", "error"),
)

# ---------------------------------------------------------------------------
# CleanRL Worker subprocess lifecycle diagnostics (LOG468–LOG475)
# These catch silent subprocess deaths (OOM, SIGKILL, etc.)
# ---------------------------------------------------------------------------
LOG_WORKER_CLEANRL_SUBPROCESS_OOM_KILLED = _constant(
    "LOG468",
    "CRITICAL",
    "CleanRL subprocess killed by OOM (exit code -9/137). "
    "The system ran out of memory during training. "
    "Check 'free -h' and close other applications, or increase swap size",
    component="Worker",
    subcomponent="CleanRLRuntime",
    tags=_tags("cleanrl", "worker", "subprocess", "oom", "critical"),
)

LOG_WORKER_CLEANRL_SUBPROCESS_SIGNALED = _constant(
    "LOG469",
    "ERROR",
    "CleanRL subprocess terminated by signal (not OOM). "
    "Process received an external signal before training could complete",
    component="Worker",
    subcomponent="CleanRLRuntime",
    tags=_tags("cleanrl", "worker", "subprocess", "signal", "error"),
)

LOG_WORKER_CLEANRL_SUBPROCESS_NO_OUTPUT = _constant(
    "LOG470",
    "WARNING",
    "CleanRL subprocess produced no stdout output. "
    "The worker started but did not complete a single training update. "
    "Possible causes: OOM kill, GPU memory exhaustion, MuJoCo rendering hang, "
    "or insufficient time for first update (2048 steps with num_envs=1)",
    component="Worker",
    subcomponent="CleanRLRuntime",
    tags=_tags("cleanrl", "worker", "subprocess", "silent", "warning"),
)

LOG_WORKER_CLEANRL_SUBPROCESS_MEMORY_SNAPSHOT = _constant(
    "LOG471",
    "INFO",
    "System memory snapshot at worker subprocess launch",
    component="Worker",
    subcomponent="CleanRLRuntime",
    tags=_tags("cleanrl", "worker", "subprocess", "memory", "diagnostic"),
)

LOG_WORKER_CLEANRL_SUBPROCESS_EXIT_SUMMARY = _constant(
    "LOG472",
    "INFO",
    "CleanRL subprocess exit summary with diagnostics",
    component="Worker",
    subcomponent="CleanRLRuntime",
    tags=_tags("cleanrl", "worker", "subprocess", "exit", "summary"),
)

LOG_WORKER_CLEANRL_TENSORBOARD_PATH_RESOLVED = _constant(
    "LOG473",
    "DEBUG",
    "CleanRL TensorBoard log directory resolved",
    component="Worker",
    subcomponent="CleanRLRuntime",
    tags=_tags("cleanrl", "worker", "tensorboard", "path", "debug"),
)

LOG_WORKER_CLEANRL_SUBPROCESS_SLOW_START = _constant(
    "LOG474",
    "WARNING",
    "CleanRL subprocess has not produced output after 30 seconds. "
    "First training update requires num_steps environment interactions "
    "which may take time for heavy environments like MuJoCo",
    component="Worker",
    subcomponent="CleanRLRuntime",
    tags=_tags("cleanrl", "worker", "subprocess", "slow", "warning"),
)

LOG_WORKER_CLEANRL_SUBPROCESS_STDERR_ONLY = _constant(
    "LOG475",
    "WARNING",
    "CleanRL subprocess wrote to stderr but not stdout. "
    "Check the stderr log for errors, warnings, or resource_tracker leaks",
    component="Worker",
    subcomponent="CleanRLRuntime",
    tags=_tags("cleanrl", "worker", "subprocess", "stderr", "diagnostic"),
)


# ---------------------------------------------------------------------------
# Ray Worker constants (LOG446–LOG460)
# ---------------------------------------------------------------------------
LOG_WORKER_RAY_RUNTIME_STARTED = _constant(
    "LOG446",
    "INFO",
    "Ray worker runtime started",
    component="Worker",
    subcomponent="RayRuntime",
    tags=_tags("ray", "worker", "runtime", "lifecycle"),
)

LOG_WORKER_RAY_RUNTIME_COMPLETED = _constant(
    "LOG447",
    "INFO",
    "Ray worker runtime completed",
    component="Worker",
    subcomponent="RayRuntime",
    tags=_tags("ray", "worker", "runtime", "lifecycle"),
)

LOG_WORKER_RAY_RUNTIME_FAILED = _constant(
    "LOG448",
    "ERROR",
    "Ray worker runtime failed",
    component="Worker",
    subcomponent="RayRuntime",
    tags=_tags("ray", "worker", "runtime", "error"),
)

LOG_WORKER_RAY_CLUSTER_STARTED = _constant(
    "LOG449",
    "INFO",
    "Ray cluster initialized",
    component="Worker",
    subcomponent="RayRuntime",
    tags=_tags("ray", "worker", "cluster"),
)

LOG_WORKER_RAY_CLUSTER_SHUTDOWN = _constant(
    "LOG450",
    "INFO",
    "Ray cluster shutdown",
    component="Worker",
    subcomponent="RayRuntime",
    tags=_tags("ray", "worker", "cluster"),
)

LOG_WORKER_RAY_TUNE_STARTED = _constant(
    "LOG451",
    "INFO",
    "Ray Tune experiment started",
    component="Worker",
    subcomponent="RayRuntime",
    tags=_tags("ray", "worker", "tune", "experiment"),
)

LOG_WORKER_RAY_TUNE_COMPLETED = _constant(
    "LOG452",
    "INFO",
    "Ray Tune experiment completed",
    component="Worker",
    subcomponent="RayRuntime",
    tags=_tags("ray", "worker", "tune", "experiment"),
)

LOG_WORKER_RAY_RLLIB_TRAINING_STARTED = _constant(
    "LOG453",
    "INFO",
    "Ray RLlib training started",
    component="Worker",
    subcomponent="RayRuntime",
    tags=_tags("ray", "worker", "rllib", "training"),
)

LOG_WORKER_RAY_RLLIB_TRAINING_ITERATION = _constant(
    "LOG454",
    "DEBUG",
    "Ray RLlib training iteration",
    component="Worker",
    subcomponent="RayRuntime",
    tags=_tags("ray", "worker", "rllib", "iteration"),
)

LOG_WORKER_RAY_CHECKPOINT_SAVED = _constant(
    "LOG455",
    "INFO",
    "Ray checkpoint saved",
    component="Worker",
    subcomponent="RayRuntime",
    tags=_tags("ray", "worker", "checkpoint", "save"),
)

LOG_WORKER_RAY_CHECKPOINT_LOADED = _constant(
    "LOG456",
    "INFO",
    "Ray checkpoint loaded",
    component="Worker",
    subcomponent="RayRuntime",
    tags=_tags("ray", "worker", "checkpoint", "load"),
)

LOG_WORKER_RAY_TENSORBOARD_ENABLED = _constant(
    "LOG457",
    "INFO",
    "Ray TensorBoard logging enabled",
    component="Worker",
    subcomponent="RayRuntime",
    tags=_tags("ray", "worker", "tensorboard", "analytics"),
)

LOG_WORKER_RAY_WANDB_ENABLED = _constant(
    "LOG458",
    "INFO",
    "Ray Weights & Biases tracking enabled",
    component="Worker",
    subcomponent="RayRuntime",
    tags=_tags("ray", "worker", "wandb", "analytics"),
)

LOG_WORKER_RAY_HEARTBEAT = _constant(
    "LOG459",
    "DEBUG",
    "Ray worker heartbeat",
    component="Worker",
    subcomponent="RayRuntime",
    tags=_tags("ray", "worker", "heartbeat"),
)

LOG_WORKER_RAY_ANALYTICS_MANIFEST_CREATED = _constant(
    "LOG460",
    "INFO",
    "Ray analytics manifest created",
    component="Worker",
    subcomponent="RayRuntime",
    tags=_tags("ray", "worker", "analytics", "manifest"),
)

LOG_WORKER_RAY_LOG_FILE_CREATED = _constant(
    "LOG461",
    "INFO",
    "Ray worker log file handler created",
    component="Worker",
    subcomponent="RayRuntime",
    tags=_tags("ray", "worker", "logging", "file"),
)

LOG_WORKER_RAY_ALGORITHM_BUILT = _constant(
    "LOG462",
    "INFO",
    "Ray RLlib algorithm built successfully",
    component="Worker",
    subcomponent="RayRuntime",
    tags=_tags("ray", "worker", "rllib", "algorithm"),
)


# ---------------------------------------------------------------------------
# BALROG Worker constants (LOG1001–LOG1015)
# ---------------------------------------------------------------------------
LOG_WORKER_BALROG_RUNTIME_STARTED = _constant(
    "LOG1001",
    "INFO",
    "BALROG worker runtime started",
    component="Worker",
    subcomponent="BarlogRuntime",
    tags=_tags("balrog", "worker", "runtime", "lifecycle"),
)

LOG_WORKER_BALROG_RUNTIME_STOPPED = _constant(
    "LOG1002",
    "INFO",
    "BALROG worker runtime stopped",
    component="Worker",
    subcomponent="BarlogRuntime",
    tags=_tags("balrog", "worker", "runtime", "lifecycle"),
)

LOG_WORKER_BALROG_RUNTIME_ERROR = _constant(
    "LOG1003",
    "ERROR",
    "BALROG worker runtime error",
    component="Worker",
    subcomponent="BarlogRuntime",
    tags=_tags("balrog", "worker", "runtime", "error"),
)

LOG_WORKER_BALROG_EPISODE_STARTED = _constant(
    "LOG1004",
    "INFO",
    "BALROG episode started",
    component="Worker",
    subcomponent="BarlogRuntime",
    tags=_tags("balrog", "worker", "episode", "start"),
)

LOG_WORKER_BALROG_EPISODE_COMPLETED = _constant(
    "LOG1005",
    "INFO",
    "BALROG episode completed",
    component="Worker",
    subcomponent="BarlogRuntime",
    tags=_tags("balrog", "worker", "episode", "complete"),
)

LOG_WORKER_BALROG_LLM_REQUEST = _constant(
    "LOG1006",
    "DEBUG",
    "BALROG LLM request sent",
    component="Worker",
    subcomponent="BarlogRuntime",
    tags=_tags("balrog", "worker", "llm", "request"),
)

LOG_WORKER_BALROG_LLM_RESPONSE = _constant(
    "LOG1007",
    "DEBUG",
    "BALROG LLM response received",
    component="Worker",
    subcomponent="BarlogRuntime",
    tags=_tags("balrog", "worker", "llm", "response"),
)

LOG_WORKER_BALROG_LLM_ERROR = _constant(
    "LOG1008",
    "ERROR",
    "BALROG LLM request failed",
    component="Worker",
    subcomponent="BarlogRuntime",
    tags=_tags("balrog", "worker", "llm", "error"),
)

LOG_WORKER_BALROG_ACTION_SELECTED = _constant(
    "LOG1009",
    "DEBUG",
    "BALROG agent selected action",
    component="Worker",
    subcomponent="BarlogRuntime",
    tags=_tags("balrog", "worker", "action"),
)

LOG_WORKER_BALROG_STEP_TELEMETRY = _constant(
    "LOG1010",
    "DEBUG",
    "BALROG step telemetry emitted",
    component="Worker",
    subcomponent="BarlogRuntime",
    tags=_tags("balrog", "worker", "telemetry", "step"),
)

LOG_WORKER_BALROG_EPISODE_TELEMETRY = _constant(
    "LOG1011",
    "INFO",
    "BALROG episode telemetry emitted",
    component="Worker",
    subcomponent="BarlogRuntime",
    tags=_tags("balrog", "worker", "telemetry", "episode"),
)

LOG_WORKER_BALROG_CONFIG_LOADED = _constant(
    "LOG1012",
    "INFO",
    "BALROG configuration loaded",
    component="Worker",
    subcomponent="BarlogRuntime",
    tags=_tags("balrog", "worker", "config", "loaded"),
)

LOG_WORKER_BALROG_ENV_CREATED = _constant(
    "LOG1013",
    "INFO",
    "BALROG environment created",
    component="Worker",
    subcomponent="BarlogRuntime",
    tags=_tags("balrog", "worker", "environment", "created"),
)

LOG_WORKER_BALROG_AGENT_CREATED = _constant(
    "LOG1014",
    "INFO",
    "BALROG agent created",
    component="Worker",
    subcomponent="BarlogRuntime",
    tags=_tags("balrog", "worker", "agent", "created"),
)

LOG_WORKER_BALROG_DEBUG = _constant(
    "LOG1015",
    "DEBUG",
    "BALROG worker debug event",
    component="Worker",
    subcomponent="BarlogRuntime",
    tags=_tags("balrog", "worker", "debug"),
)


# ---------------------------------------------------------------------------
# XuanCe Worker constants (LOG1016–LOG1029)
# ---------------------------------------------------------------------------
LOG_WORKER_XUANCE_RUNTIME_STARTED = _constant(
    "LOG1016",
    "INFO",
    "XuanCe worker runtime started",
    component="Worker",
    subcomponent="XuanCeRuntime",
    tags=_tags("xuance", "worker", "runtime", "lifecycle"),
)

LOG_WORKER_XUANCE_RUNTIME_STOPPED = _constant(
    "LOG1017",
    "INFO",
    "XuanCe worker runtime stopped",
    component="Worker",
    subcomponent="XuanCeRuntime",
    tags=_tags("xuance", "worker", "runtime", "lifecycle"),
)

LOG_WORKER_XUANCE_RUNTIME_ERROR = _constant(
    "LOG1018",
    "ERROR",
    "XuanCe worker runtime error",
    component="Worker",
    subcomponent="XuanCeRuntime",
    tags=_tags("xuance", "worker", "runtime", "error"),
)

LOG_WORKER_XUANCE_EPISODE_STARTED = _constant(
    "LOG1019",
    "INFO",
    "XuanCe episode started",
    component="Worker",
    subcomponent="XuanCeRuntime",
    tags=_tags("xuance", "worker", "episode", "start"),
)

LOG_WORKER_XUANCE_EPISODE_COMPLETED = _constant(
    "LOG1020",
    "INFO",
    "XuanCe episode completed",
    component="Worker",
    subcomponent="XuanCeRuntime",
    tags=_tags("xuance", "worker", "episode", "complete"),
)

LOG_WORKER_XUANCE_TRAINING_STARTED = _constant(
    "LOG1021",
    "INFO",
    "XuanCe training started",
    component="Worker",
    subcomponent="XuanCeRuntime",
    tags=_tags("xuance", "worker", "training", "start"),
)

LOG_WORKER_XUANCE_TRAINING_STEP = _constant(
    "LOG1022",
    "DEBUG",
    "XuanCe training step completed",
    component="Worker",
    subcomponent="XuanCeRuntime",
    tags=_tags("xuance", "worker", "training", "step"),
)

LOG_WORKER_XUANCE_TRAINING_COMPLETED = _constant(
    "LOG1023",
    "INFO",
    "XuanCe training completed",
    component="Worker",
    subcomponent="XuanCeRuntime",
    tags=_tags("xuance", "worker", "training", "complete"),
)

LOG_WORKER_XUANCE_CONFIG_LOADED = _constant(
    "LOG1024",
    "INFO",
    "XuanCe configuration loaded",
    component="Worker",
    subcomponent="XuanCeRuntime",
    tags=_tags("xuance", "worker", "config", "loaded"),
)

LOG_WORKER_XUANCE_ENV_CREATED = _constant(
    "LOG1025",
    "INFO",
    "XuanCe environment created",
    component="Worker",
    subcomponent="XuanCeRuntime",
    tags=_tags("xuance", "worker", "environment", "created"),
)

LOG_WORKER_XUANCE_AGENT_CREATED = _constant(
    "LOG1026",
    "INFO",
    "XuanCe agent created",
    component="Worker",
    subcomponent="XuanCeRuntime",
    tags=_tags("xuance", "worker", "agent", "created"),
)

LOG_WORKER_XUANCE_CHECKPOINT_SAVED = _constant(
    "LOG1027",
    "INFO",
    "XuanCe checkpoint saved",
    component="Worker",
    subcomponent="XuanCeRuntime",
    tags=_tags("xuance", "worker", "checkpoint", "saved"),
)

LOG_WORKER_XUANCE_METRICS_LOGGED = _constant(
    "LOG1028",
    "INFO",
    "XuanCe metrics logged",
    component="Worker",
    subcomponent="XuanCeRuntime",
    tags=_tags("xuance", "worker", "metrics", "logged"),
)

LOG_WORKER_XUANCE_DEBUG = _constant(
    "LOG1029",
    "DEBUG",
    "XuanCe worker debug event",
    component="Worker",
    subcomponent="XuanCeRuntime",
    tags=_tags("xuance", "worker", "debug"),
)


# =========================================================================
# Checkers Board Game (Human Control Mode)
# =========================================================================

LOG_CHECKERS_BOARD_CLICK = _constant(
    "LOG1030",
    "INFO",
    "Checkers board cell clicked",
    component="Rendering",
    subcomponent="CheckersBoardRenderer",
    tags=_tags("checkers", "board", "click", "ui"),
)

LOG_CHECKERS_BOARD_CLICK_IGNORED = _constant(
    "LOG1031",
    "DEBUG",
    "Checkers board click ignored (disabled or game over)",
    component="Rendering",
    subcomponent="CheckersBoardRenderer",
    tags=_tags("checkers", "board", "click", "ignored"),
)

LOG_CHECKERS_CELL_SIGNAL_EMITTED = _constant(
    "LOG1032",
    "DEBUG",
    "Checkers cell_clicked signal emitted",
    component="Rendering",
    subcomponent="CheckersBoardRenderer",
    tags=_tags("checkers", "signal", "emitted"),
)

LOG_CHECKERS_HANDLER_CLICK_RECEIVED = _constant(
    "LOG1033",
    "INFO",
    "Checkers handler received cell click",
    component="UI",
    subcomponent="CheckersHandler",
    tags=_tags("checkers", "handler", "click"),
)

LOG_CHECKERS_HANDLER_GAME_MISMATCH = _constant(
    "LOG1034",
    "DEBUG",
    "Checkers click ignored - game_id mismatch",
    component="UI",
    subcomponent="CheckersHandler",
    tags=_tags("checkers", "handler", "mismatch"),
)

LOG_CHECKERS_PIECE_SELECTED = _constant(
    "LOG1035",
    "INFO",
    "Checkers piece selected",
    component="UI",
    subcomponent="CheckersHandler",
    tags=_tags("checkers", "handler", "selection"),
)

LOG_CHECKERS_MOVE_EXECUTED = _constant(
    "LOG1036",
    "INFO",
    "Checkers move executed",
    component="UI",
    subcomponent="CheckersHandler",
    tags=_tags("checkers", "handler", "move"),
)

LOG_CHECKERS_MOVE_FAILED = _constant(
    "LOG1037",
    "ERROR",
    "Checkers move execution failed",
    component="UI",
    subcomponent="CheckersHandler",
    tags=_tags("checkers", "handler", "error"),
)


# ---------------------------------------------------------------------------
# MOSAIC MultiGrid Extension constants (LOG1038–LOG1043)
# ---------------------------------------------------------------------------
LOG_WORKER_MOSAIC_PROMPT_GENERATED = _constant(
    "LOG1038",
    "DEBUG",
    "MOSAIC MultiGrid instruction prompt generated",
    component="Worker",
    subcomponent="MosaicExtension",
    tags=_tags("mosaic", "multigrid", "prompt", "generated"),
)

LOG_WORKER_MOSAIC_OBSERVATION_EGOCENTRIC = _constant(
    "LOG1039",
    "DEBUG",
    "MOSAIC egocentric observation generated",
    component="Worker",
    subcomponent="MosaicExtension",
    tags=_tags("mosaic", "multigrid", "observation", "egocentric"),
)

LOG_WORKER_MOSAIC_OBSERVATION_TEAMMATES = _constant(
    "LOG1040",
    "DEBUG",
    "MOSAIC teammates observation generated",
    component="Worker",
    subcomponent="MosaicExtension",
    tags=_tags("mosaic", "multigrid", "observation", "teammates"),
)

LOG_WORKER_MOSAIC_ACTION_PARSED = _constant(
    "LOG1041",
    "DEBUG",
    "MOSAIC action parsed from LLM output",
    component="Worker",
    subcomponent="MosaicExtension",
    tags=_tags("mosaic", "multigrid", "action", "parsed"),
)

LOG_WORKER_MOSAIC_ACTION_PARSE_FAILED = _constant(
    "LOG1042",
    "WARNING",
    "MOSAIC action parse failed, defaulting to 'still'",
    component="Worker",
    subcomponent="MosaicExtension",
    tags=_tags("mosaic", "multigrid", "action", "parse_failed"),
)

LOG_WORKER_MOSAIC_RUNTIME_INTEGRATION = _constant(
    "LOG1043",
    "INFO",
    "MOSAIC runtime integration initialized",
    component="Worker",
    subcomponent="BalrogRuntime",
    tags=_tags("mosaic", "multigrid", "runtime", "integration"),
)


# ---------------------------------------------------------------------------
# LLM Worker constants (LOG1044–LOG1058)
# ---------------------------------------------------------------------------
LOG_MOSAIC_WORKER_LLM_RUNTIME_STARTED = _constant(
    "LOG1044",
    "INFO",
    "LLM worker runtime started",
    component="Worker",
    subcomponent="LLMRuntime",
    tags=_tags("llm", "worker", "runtime", "lifecycle"),
)

LOG_MOSAIC_WORKER_LLM_RUNTIME_STOPPED = _constant(
    "LOG1045",
    "INFO",
    "LLM worker runtime stopped",
    component="Worker",
    subcomponent="LLMRuntime",
    tags=_tags("llm", "worker", "runtime", "lifecycle"),
)

LOG_MOSAIC_WORKER_LLM_RUNTIME_ERROR = _constant(
    "LOG1046",
    "ERROR",
    "LLM worker runtime error",
    component="Worker",
    subcomponent="LLMRuntime",
    tags=_tags("llm", "worker", "runtime", "error"),
)

LOG_MOSAIC_WORKER_LLM_EPISODE_STARTED = _constant(
    "LOG1047",
    "INFO",
    "LLM episode started",
    component="Worker",
    subcomponent="LLMRuntime",
    tags=_tags("llm", "worker", "episode", "start"),
)

LOG_MOSAIC_WORKER_LLM_EPISODE_COMPLETED = _constant(
    "LOG1048",
    "INFO",
    "LLM episode completed",
    component="Worker",
    subcomponent="LLMRuntime",
    tags=_tags("llm", "worker", "episode", "complete"),
)

LOG_MOSAIC_WORKER_LLM_REQUEST = _constant(
    "LOG1049",
    "DEBUG",
    "LLM request sent",
    component="Worker",
    subcomponent="LLMRuntime",
    tags=_tags("llm", "worker", "llm", "request"),
)

LOG_MOSAIC_WORKER_LLM_RESPONSE = _constant(
    "LOG1050",
    "DEBUG",
    "LLM response received",
    component="Worker",
    subcomponent="LLMRuntime",
    tags=_tags("llm", "worker", "llm", "response"),
)

LOG_MOSAIC_WORKER_LLM_ERROR = _constant(
    "LOG1051",
    "ERROR",
    "LLM request failed",
    component="Worker",
    subcomponent="LLMRuntime",
    tags=_tags("llm", "worker", "llm", "error"),
)

LOG_MOSAIC_WORKER_LLM_ACTION_SELECTED = _constant(
    "LOG1052",
    "DEBUG",
    "LLM agent selected action",
    component="Worker",
    subcomponent="LLMRuntime",
    tags=_tags("llm", "worker", "action"),
)

LOG_MOSAIC_WORKER_LLM_STEP_TELEMETRY = _constant(
    "LOG1053",
    "DEBUG",
    "LLM step telemetry emitted",
    component="Worker",
    subcomponent="LLMRuntime",
    tags=_tags("llm", "worker", "telemetry", "step"),
)

LOG_MOSAIC_WORKER_LLM_EPISODE_TELEMETRY = _constant(
    "LOG1054",
    "INFO",
    "LLM episode telemetry emitted",
    component="Worker",
    subcomponent="LLMRuntime",
    tags=_tags("llm", "worker", "telemetry", "episode"),
)

LOG_MOSAIC_WORKER_LLM_CONFIG_LOADED = _constant(
    "LOG1055",
    "INFO",
    "LLM configuration loaded",
    component="Worker",
    subcomponent="LLMRuntime",
    tags=_tags("llm", "worker", "config", "loaded"),
)

LOG_MOSAIC_WORKER_LLM_ENV_CREATED = _constant(
    "LOG1056",
    "INFO",
    "LLM environment created",
    component="Worker",
    subcomponent="LLMRuntime",
    tags=_tags("llm", "worker", "environment", "created"),
)

LOG_MOSAIC_WORKER_LLM_AGENT_CREATED = _constant(
    "LOG1057",
    "INFO",
    "LLM agent created",
    component="Worker",
    subcomponent="LLMRuntime",
    tags=_tags("llm", "worker", "agent", "created"),
)

LOG_MOSAIC_WORKER_LLM_DEBUG = _constant(
    "LOG1058",
    "DEBUG",
    "LLM worker debug event",
    component="Worker",
    subcomponent="LLMRuntime",
    tags=_tags("llm", "worker", "debug"),
)

# VLM Worker constants (LOG1071–LOG1087)
LOG_MOSAIC_WORKER_VLM_RUNTIME_STARTED = _constant(
    "LOG1071",
    "INFO",
    "VLM worker runtime started",
    component="Worker",
    subcomponent="VLMRuntime",
    tags=_tags("vlm", "worker", "runtime", "lifecycle"),
)

LOG_MOSAIC_WORKER_VLM_RUNTIME_STOPPED = _constant(
    "LOG1072",
    "INFO",
    "VLM worker runtime stopped",
    component="Worker",
    subcomponent="VLMRuntime",
    tags=_tags("vlm", "worker", "runtime", "lifecycle"),
)

LOG_MOSAIC_WORKER_VLM_RUNTIME_ERROR = _constant(
    "LOG1073",
    "ERROR",
    "VLM worker runtime error",
    component="Worker",
    subcomponent="VLMRuntime",
    tags=_tags("vlm", "worker", "runtime", "error"),
)

LOG_MOSAIC_WORKER_VLM_EPISODE_STARTED = _constant(
    "LOG1074",
    "INFO",
    "VLM episode started",
    component="Worker",
    subcomponent="VLMRuntime",
    tags=_tags("vlm", "worker", "episode", "start"),
)

LOG_MOSAIC_WORKER_VLM_EPISODE_COMPLETED = _constant(
    "LOG1075",
    "INFO",
    "VLM episode completed",
    component="Worker",
    subcomponent="VLMRuntime",
    tags=_tags("vlm", "worker", "episode", "complete"),
)

LOG_MOSAIC_WORKER_VLM_REQUEST = _constant(
    "LOG1076",
    "DEBUG",
    "VLM request sent",
    component="Worker",
    subcomponent="VLMRuntime",
    tags=_tags("vlm", "worker", "llm", "request"),
)

LOG_MOSAIC_WORKER_VLM_RESPONSE = _constant(
    "LOG1077",
    "DEBUG",
    "VLM response received",
    component="Worker",
    subcomponent="VLMRuntime",
    tags=_tags("vlm", "worker", "llm", "response"),
)

LOG_MOSAIC_WORKER_VLM_ERROR = _constant(
    "LOG1078",
    "ERROR",
    "VLM request failed",
    component="Worker",
    subcomponent="VLMRuntime",
    tags=_tags("vlm", "worker", "llm", "error"),
)

LOG_MOSAIC_WORKER_VLM_ACTION_SELECTED = _constant(
    "LOG1079",
    "DEBUG",
    "VLM agent selected action",
    component="Worker",
    subcomponent="VLMRuntime",
    tags=_tags("vlm", "worker", "action"),
)

LOG_MOSAIC_WORKER_VLM_STEP_TELEMETRY = _constant(
    "LOG1080",
    "DEBUG",
    "VLM step telemetry emitted",
    component="Worker",
    subcomponent="VLMRuntime",
    tags=_tags("vlm", "worker", "telemetry", "step"),
)

LOG_MOSAIC_WORKER_VLM_EPISODE_TELEMETRY = _constant(
    "LOG1081",
    "INFO",
    "VLM episode telemetry emitted",
    component="Worker",
    subcomponent="VLMRuntime",
    tags=_tags("vlm", "worker", "telemetry", "episode"),
)

LOG_MOSAIC_WORKER_VLM_CONFIG_LOADED = _constant(
    "LOG1082",
    "INFO",
    "VLM configuration loaded",
    component="Worker",
    subcomponent="VLMRuntime",
    tags=_tags("vlm", "worker", "config", "loaded"),
)

LOG_MOSAIC_WORKER_VLM_ENV_CREATED = _constant(
    "LOG1083",
    "INFO",
    "VLM environment created",
    component="Worker",
    subcomponent="VLMRuntime",
    tags=_tags("vlm", "worker", "environment", "created"),
)

LOG_MOSAIC_WORKER_VLM_AGENT_CREATED = _constant(
    "LOG1084",
    "INFO",
    "VLM agent created",
    component="Worker",
    subcomponent="VLMRuntime",
    tags=_tags("vlm", "worker", "agent", "created"),
)

LOG_MOSAIC_WORKER_VLM_DEBUG = _constant(
    "LOG1085",
    "DEBUG",
    "VLM worker debug event",
    component="Worker",
    subcomponent="VLMRuntime",
    tags=_tags("vlm", "worker", "debug"),
)

LOG_MOSAIC_WORKER_VLM_EPISODE_AUTO_RESET = _constant(
    "LOG1086",
    "INFO",
    "VLM episode auto-reset with seed",
    component="Worker",
    subcomponent="VLMRuntime",
    tags=_tags("vlm", "worker", "episode", "auto-reset", "seed"),
)

LOG_MOSAIC_WORKER_VLM_ACTION_DEFAULTED = _constant(
    "LOG1087",
    "WARNING",
    "VLM action defaulted due to error",
    component="Worker",
    subcomponent="VLMRuntime",
    tags=_tags("vlm", "worker", "llm", "action", "default"),
)

# Auto-reset and action-defaulting events (LOG1059–LOG1062)
LOG_WORKER_BALROG_EPISODE_AUTO_RESET = _constant(
    "LOG1059",
    "INFO",
    "BALROG episode auto-reset with seed",
    component="Worker",
    subcomponent="BarlogRuntime",
    tags=_tags("balrog", "worker", "episode", "auto-reset", "seed"),
)
LOG_WORKER_BALROG_ACTION_DEFAULTED = _constant(
    "LOG1060",
    "WARNING",
    "BALROG LLM action defaulted due to error",
    component="Worker",
    subcomponent="BarlogRuntime",
    tags=_tags("balrog", "worker", "llm", "action", "default"),
)
LOG_MOSAIC_WORKER_LLM_EPISODE_AUTO_RESET = _constant(
    "LOG1061",
    "INFO",
    "MOSAIC LLM episode auto-reset with seed",
    component="Worker",
    subcomponent="LLMRuntime",
    tags=_tags("mosaic", "llm", "worker", "episode", "auto-reset", "seed"),
)
LOG_MOSAIC_WORKER_LLM_ACTION_DEFAULTED = _constant(
    "LOG1062",
    "WARNING",
    "MOSAIC LLM action defaulted due to error",
    component="Worker",
    subcomponent="LLMRuntime",
    tags=_tags("mosaic", "llm", "worker", "action", "default"),
)
LOG_WORKER_CLEANRL_EPISODE_AUTO_RESET = _constant(
    "LOG1063",
    "INFO",
    "CleanRL episode auto-reset with seed",
    component="Worker",
    subcomponent="CleanRLRuntime",
    tags=_tags("cleanrl", "worker", "episode", "auto-reset", "seed"),
)

# ── Passive Worker ──────────────────────────────────────────────────
LOG_WORKER_PASSIVE_RUNTIME_STARTED = _constant(
    "LOG1064",
    "INFO",
    "Passive worker runtime started",
    component="Worker",
    subcomponent="PassiveRuntime",
    tags=_tags("passive", "worker", "runtime", "lifecycle"),
)
LOG_WORKER_PASSIVE_RUNTIME_STOPPED = _constant(
    "LOG1065",
    "INFO",
    "Passive worker runtime stopped",
    component="Worker",
    subcomponent="PassiveRuntime",
    tags=_tags("passive", "worker", "runtime", "lifecycle"),
)
LOG_WORKER_PASSIVE_RUNTIME_ERROR = _constant(
    "LOG1066",
    "ERROR",
    "Passive worker runtime error",
    component="Worker",
    subcomponent="PassiveRuntime",
    tags=_tags("passive", "worker", "runtime", "error"),
)
LOG_WORKER_PASSIVE_AGENT_READY = _constant(
    "LOG1067",
    "INFO",
    "Passive worker agent ready",
    component="Worker",
    subcomponent="PassiveRuntime",
    tags=_tags("passive", "worker", "agent", "lifecycle"),
)
LOG_WORKER_PASSIVE_ENV_CREATED = _constant(
    "LOG1068",
    "INFO",
    "Passive worker environment created",
    component="Worker",
    subcomponent="PassiveRuntime",
    tags=_tags("passive", "worker", "env", "lifecycle"),
)
LOG_WORKER_PASSIVE_ENV_RESET = _constant(
    "LOG1069",
    "INFO",
    "Passive worker environment reset",
    component="Worker",
    subcomponent="PassiveRuntime",
    tags=_tags("passive", "worker", "env", "reset"),
)
LOG_WORKER_PASSIVE_DEBUG = _constant(
    "LOG1070",
    "DEBUG",
    "Passive worker debug",
    component="Worker",
    subcomponent="PassiveRuntime",
    tags=_tags("passive", "worker", "debug"),
)


# =========================================================================
# Helper Functions for Runtime Discovery & Validation
# =========================================================================


def get_constant_by_code(code: str) -> LogConstant | None:
    """Retrieve a log constant by its code identifier.

    Args:
        code: The constant code (e.g., "LOG401", "LOG502").

    Returns:
        The LogConstant instance if found, else None.
    """
    for const in ALL_LOG_CONSTANTS:
        if const.code == code:
            return const
    return None


def list_known_components() -> list[str]:
    """Return a sorted list of all known component names.

    Returns:
        Sorted list of unique component strings (e.g., ["Adapter", "Controller", "Service"]).
    """
    return sorted(set(const.component for const in ALL_LOG_CONSTANTS))


def get_component_snapshot() -> Dict[str, Set[str]]:
    """Build a snapshot of component → subcomponents mapping.

    This is useful for GUI filters to display hierarchical component/subcomponent
    choices without hard-coding them.

    Returns:
        A dict mapping each component name to the set of its known subcomponents.

    Example:
        >>> snapshot = get_component_snapshot()
        >>> snapshot["Controller"]
        {'Input', 'LiveTelemetry', 'Session'}
    """
    snapshot: Dict[str, Set[str]] = {}
    for const in ALL_LOG_CONSTANTS:
        if const.component not in snapshot:
            snapshot[const.component] = set()
        snapshot[const.component].add(const.subcomponent)
    return snapshot


def validate_log_constants() -> list[str]:
    """Validate that all log constants conform to logging standards.

    Checks that:
    - Every constant has a valid logging level (matches logging._nameToLevel).
    - Every constant has non-empty code, message, component, subcomponent fields.
    - No duplicate codes exist.

    Returns:
        A list of error messages. Empty list if all constants are valid.
    """
    errors: list[str] = []
    seen_codes: set[str] = set()

    for const in ALL_LOG_CONSTANTS:
        # Check non-empty fields
        if not const.code:
            errors.append(f"LogConstant has empty code: {const}")
        if not const.message:
            errors.append(f"LogConstant {const.code} has empty message")
        if not const.component:
            errors.append(f"LogConstant {const.code} has empty component")
        if not const.subcomponent:
            errors.append(f"LogConstant {const.code} has empty subcomponent")

        # Check level validity
        if isinstance(const.level, str):
            if const.level not in logging._nameToLevel:
                errors.append(
                    f"LogConstant {const.code} has invalid level: {const.level}. "
                    f"Must be one of {sorted(logging._nameToLevel.keys())}"
                )
        elif not isinstance(const.level, int):
            errors.append(
                f"LogConstant {const.code} has invalid level type: {type(const.level)}. "
                f"Must be str or int."
            )

        # Check for duplicate codes
        if const.code in seen_codes:
            errors.append(f"Duplicate constant code: {const.code}")
        seen_codes.add(const.code)

    return errors


# Aggregate tuple for quick iteration during tests or tooling.
ALL_LOG_CONSTANTS: Tuple[LogConstant, ...] = (
    LOG_SESSION_ADAPTER_LOAD_ERROR,
    LOG_SESSION_STEP_ERROR,
    LOG_SESSION_EPISODE_ERROR,
    LOG_SESSION_TIMER_PRECISION_WARNING,
    LOG_INPUT_CONTROLLER_ERROR,
    LOG_KEYBOARD_DETECTED,
    LOG_KEYBOARD_ASSIGNED,
    LOG_KEYBOARD_DETECTION_ERROR,
    LOG_KEYBOARD_EVDEV_SETUP_START,
    LOG_KEYBOARD_EVDEV_SETUP_SUCCESS,
    LOG_KEYBOARD_EVDEV_SETUP_FAILED,
    LOG_KEY_RESOLVER_INITIALIZED,
    LOG_KEY_RESOLVER_UNAVAILABLE,
    LOG_INPUT_MODE_CONFIGURED,
    LOG_EVDEV_KEY_PRESSED,
    LOG_EVDEV_KEY_RELEASED,
    LOG_HUMAN_CONTROL_WORKER_LAUNCH,
    LOG_HUMAN_CONTROL_WORKER_READY,
    LOG_HUMAN_CONTROL_WORKER_ACTION,
    LOG_HUMAN_CONTROL_WORKER_STEP,
    LOG_HUMAN_CONTROL_WORKER_STOPPED,
    LOG_HUMAN_CONTROL_WORKER_ERROR,
    LOG_HUMAN_CONTROL_BRIDGE_START,
    LOG_HUMAN_CONTROL_BRIDGE_STOP,
    LOG_HUMAN_CONTROL_AUTO_ASSIGN,
    LOG_HUMAN_CONTROL_INPUT_DISABLED,
    LOG_HUMAN_CONTROL_RAW_KEY_NO_HANDLER,
    LOG_LIVE_CONTROLLER_INITIALIZED,
    LOG_LIVE_CONTROLLER_THREAD_STARTED,
    LOG_LIVE_CONTROLLER_THREAD_STOPPED,
    LOG_LIVE_CONTROLLER_THREAD_STOP_TIMEOUT,
    LOG_LIVE_CONTROLLER_ALREADY_RUNNING,
    LOG_LIVE_CONTROLLER_RUN_SUBSCRIBED,
    LOG_LIVE_CONTROLLER_RUN_UNSUBSCRIBED,
    LOG_LIVE_CONTROLLER_RUN_ALREADY_SUBSCRIBED,
    LOG_LIVE_CONTROLLER_RUNBUS_SUBSCRIBED,
    LOG_LIVE_CONTROLLER_RUN_COMPLETED,
    LOG_LIVE_CONTROLLER_QUEUE_OVERFLOW,
    LOG_LIVE_CONTROLLER_BUFFER_STEPS_FLUSHED,
    LOG_LIVE_CONTROLLER_BUFFER_EPISODES_FLUSHED,
    LOG_BUFFER_DROP,
    LOG_TELEMETRY_CONTROLLER_THREAD_ERROR,
    LOG_LIVE_CONTROLLER_SIGNAL_EMIT_FAILED,
    LOG_LIVE_CONTROLLER_TAB_ADD_FAILED,
    LOG_CREDIT_STARVED,
    LOG_CREDIT_RESUMED,
    LOG_LIVE_CONTROLLER_LOOP_EXITED,
    LOG_TELEMETRY_SUBSCRIBE_ERROR,
    LOG_LIVE_CONTROLLER_EVENT_FOR_DELETED_RUN,
    LOG_LIVE_CONTROLLER_TAB_REQUESTED,
    LOG_ADAPTER_INIT_ERROR,
    LOG_ADAPTER_STEP_ERROR,
    LOG_ADAPTER_PAYLOAD_ERROR,
    LOG_ADAPTER_STATE_INVALID,
    LOG_ADAPTER_RENDER_ERROR,
    LOG_ADAPTER_ENV_CREATED,
    LOG_ADAPTER_ENV_RESET,
    LOG_ADAPTER_STEP_SUMMARY,
    LOG_ADAPTER_ENV_CLOSED,
    LOG_ADAPTER_MAP_GENERATION,
    LOG_ADAPTER_HOLE_PLACEMENT,
    LOG_ADAPTER_GOAL_OVERRIDE,
    LOG_ADAPTER_RENDER_PAYLOAD,
    LOG_ENV_MINIGRID_BOOT,
    LOG_ENV_MINIGRID_STEP,
    LOG_ENV_MINIGRID_ERROR,
    LOG_ENV_MINIGRID_RENDER_WARNING,
    LOG_ADAPTER_ALE_NAMESPACE_IMPORT_FAILED,
    LOG_ADAPTER_ALE_METADATA_PROBE_FAILED,
    LOG_MOSAIC_MULTIGRID_GOAL_SCORED,
    LOG_MOSAIC_MULTIGRID_PASS_COMPLETED,
    LOG_MOSAIC_MULTIGRID_STEAL_COMPLETED,
    LOG_MOSAIC_MULTIGRID_VISIBILITY,
    LOG_MOSAIC_MULTIGRID_OBSERVATION,
    LOG_ENV_CRAFTER_BOOT,
    LOG_ENV_CRAFTER_STEP,
    LOG_ENV_CRAFTER_ERROR,
    LOG_ENV_CRAFTER_RENDER_WARNING,
    LOG_ENV_CRAFTER_ACHIEVEMENT,
    LOG_ENV_PROCGEN_BOOT,
    LOG_ENV_PROCGEN_STEP,
    LOG_ENV_PROCGEN_ERROR,
    LOG_ENV_PROCGEN_RENDER_WARNING,
    LOG_ENV_PROCGEN_LEVEL_COMPLETE,
    LOG_SERVICE_TELEMETRY_STEP_REJECTED,
    LOG_SERVICE_TELEMETRY_ASYNC_ERROR,
    LOG_SERVICE_DB_SINK_INITIALIZED,
    LOG_SERVICE_DB_SINK_STARTED,
    LOG_SERVICE_DB_SINK_ALREADY_RUNNING,
    LOG_SERVICE_DB_SINK_STOPPED,
    LOG_SERVICE_DB_SINK_STOP_TIMEOUT,
    LOG_SERVICE_DB_SINK_LOOP_EXITED,
    LOG_SERVICE_DB_SINK_FATAL,
    LOG_SERVICE_DB_SINK_QUEUE_DEPTH,
    LOG_SERVICE_DB_SINK_QUEUE_PRESSURE,
    LOG_SERVICE_DB_SINK_FLUSH_STATS,
    LOG_SERVICE_DB_SINK_FLUSH_LATENCY,
    LOG_SERVICE_SQLITE_DEBUG,
    LOG_SERVICE_SQLITE_INFO,
    LOG_SERVICE_SQLITE_WARNING,
    LOG_SERVICE_SQLITE_WORKER_STARTED,
    LOG_SERVICE_SQLITE_WORKER_STOPPED,
    LOG_SERVICE_SQLITE_WRITE_ERROR,
    LOG_SERVICE_SQLITE_DISK_IO_ERROR,
    LOG_SERVICE_SQLITE_INIT_ERROR,
    LOG_SERVICE_SQLITE_DESERIALIZATION_FAILED,
    LOG_SERVICE_TELEMETRY_BRIDGE_STEP_QUEUED,
    LOG_SERVICE_TELEMETRY_BRIDGE_EPISODE_QUEUED,
    LOG_SERVICE_TELEMETRY_BRIDGE_STEP_DELIVERED,
    LOG_SERVICE_TELEMETRY_BRIDGE_EPISODE_DELIVERED,
    LOG_SERVICE_TELEMETRY_BRIDGE_OVERFLOW,
    LOG_SERVICE_TELEMETRY_BRIDGE_RUN_COMPLETED,
    LOG_SERVICE_TELEMETRY_HUB_STARTED,
    LOG_SERVICE_TELEMETRY_HUB_SUBSCRIBED,
    LOG_SERVICE_TELEMETRY_HUB_TRACE,
    LOG_SERVICE_TELEMETRY_HUB_ERROR,
    LOG_RENDER_REGULATOR_NOT_STARTED,
    LOG_RENDER_DROPPED_FRAME,
    LOG_DAEMON_START,
    LOG_TRAINER_CLIENT_CONNECTING,
    LOG_TRAINER_CLIENT_CONNECTED,
    LOG_TRAINER_CLIENT_CONNECTION_TIMEOUT,
    LOG_TRAINER_CLIENT_LOOP_NONFATAL,
    LOG_TRAINER_CLIENT_LOOP_ERROR,
    LOG_TRAINER_CLIENT_SHUTDOWN_WARNING,
    LOG_SERVICE_FRAME_INFO,
    LOG_SERVICE_FRAME_WARNING,
    LOG_SERVICE_FRAME_ERROR,
    LOG_SERVICE_FRAME_DEBUG,
    LOG_SERVICE_SESSION_NUMPY_SCALAR_COERCE_FAILED,
    LOG_SERVICE_SESSION_NDARRAY_SUMMARY_FAILED,
    LOG_SERVICE_SESSION_LAZYFRAMES_SUMMARY_FAILED,
    LOG_SERVICE_SESSION_TOLIST_COERCE_FAILED,
    LOG_SERVICE_SESSION_ITERABLE_COERCE_FAILED,
    LOG_SERVICE_VALIDATION_DEBUG,
    LOG_SERVICE_VALIDATION_WARNING,
    LOG_SERVICE_VALIDATION_ERROR,
    LOG_SCHEMA_MISMATCH,
    LOG_VECTOR_AUTORESET_MODE,
    LOG_SPACE_DESCRIPTOR_MISSING,
    LOG_NORMALIZATION_STATS_DROPPED,
    LOG_SERVICE_OPERATOR_REGISTERED,
    LOG_SERVICE_OPERATOR_ACTIVATED,
    LOG_SERVICE_OPERATOR_DEACTIVATED,
    LOG_SERVICE_OPERATOR_ACTION_SELECTED,
    LOG_SERVICE_OPERATOR_ERROR,
    LOG_OPERATOR_INTERACTIVE_LAUNCHED,
    LOG_OPERATOR_RESET_COMMAND_SENT,
    LOG_OPERATOR_STEP_COMMAND_SENT,
    LOG_OPERATOR_STOP_COMMAND_SENT,
    LOG_OPERATOR_COMMAND_FAILED,
    LOG_OPERATOR_RESET_ALL_STARTED,
    LOG_OPERATOR_STEP_ALL_COMPLETED,
    LOG_OPERATOR_STOP_ALL_COMPLETED,
    LOG_SCRIPT_OPERATOR_LAUNCHED,
    LOG_SCRIPT_OPERATOR_BEHAVIOR_SET,
    LOG_SCRIPT_OPERATOR_EPISODE_START,
    LOG_SCRIPT_OPERATOR_EPISODE_END,
    LOG_SCRIPT_OPERATOR_TELEMETRY_EMITTED,
    LOG_SCRIPT_LOADED,
    LOG_SCRIPT_PARSED,
    LOG_SCRIPT_VALIDATION_FAILED,
    LOG_SCRIPT_AUTO_EXECUTION_STARTED,
    LOG_SCRIPT_AUTO_EXECUTION_COMPLETED,
    LOG_OPERATOR_INIT_AGENT_SENT,
    LOG_OPERATOR_SELECT_ACTION_SENT,
    LOG_OPERATOR_MULTIAGENT_LAUNCHED,
    LOG_OPERATOR_MULTIAGENT_INIT_FAILED,
    LOG_OPERATOR_MULTIAGENT_ACTION_FAILED,
    LOG_OPERATOR_ENV_PREVIEW_STARTED,
    LOG_OPERATOR_ENV_PREVIEW_SUCCESS,
    LOG_OPERATOR_ENV_PREVIEW_IMPORT_ERROR,
    LOG_OPERATOR_ENV_PREVIEW_ERROR,
    LOG_OPERATOR_PARALLEL_RESET_STARTED,
    LOG_OPERATOR_PARALLEL_STEP_STARTED,
    LOG_OPERATOR_PARALLEL_STEP_COMPLETED,
    LOG_OPERATOR_VIEW_SIZE_CONFIGURED,
    LOG_VLLM_SERVER_COUNT_CHANGED,
    LOG_VLLM_SERVER_STARTING,
    LOG_VLLM_SERVER_RUNNING,
    LOG_VLLM_SERVER_STOPPING,
    LOG_VLLM_SERVER_START_FAILED,
    LOG_VLLM_SERVER_PROCESS_EXITED,
    LOG_VLLM_SERVER_NOT_RESPONDING,
    LOG_VLLM_ORPHAN_PROCESS_KILLED,
    LOG_VLLM_GPU_MEMORY_FREED,
    LOG_VLLM_GPU_MEMORY_NOT_FREED,
    LOG_SERVICE_ACTOR_SEED_ERROR,
    LOG_RUNTIME_APP_DEBUG,
    LOG_RUNTIME_APP_INFO,
    LOG_RUNTIME_APP_WARNING,
    LOG_RUNTIME_APP_ERROR,
    LOG_UI_MAINWINDOW_TRACE,
    LOG_UI_MAINWINDOW_INFO,
    LOG_UI_MAINWINDOW_WARNING,
    LOG_UI_MAINWINDOW_ERROR,
    LOG_UI_LIVE_TAB_TRACE,
    LOG_UI_LIVE_TAB_INFO,
    LOG_UI_LIVE_TAB_WARNING,
    LOG_UI_LIVE_TAB_ERROR,
    LOG_UI_RENDER_TABS_TRACE,
    LOG_UI_RENDER_TABS_INFO,
    LOG_UI_RENDER_TABS_WARNING,
    LOG_UI_RENDER_TABS_ERROR,
    LOG_UI_RENDER_TABS_TENSORBOARD_STATUS,
    LOG_UI_RENDER_TABS_TENSORBOARD_WAITING,
    LOG_UI_RENDER_TABS_WANDB_STATUS,
    LOG_UI_RENDER_TABS_WANDB_WARNING,
    LOG_UI_RENDER_TABS_WANDB_ERROR,
    LOG_UI_RENDER_TABS_WANDB_PROXY_APPLIED,
    LOG_UI_RENDER_TABS_WANDB_PROXY_SKIPPED,
    LOG_UI_RENDER_TABS_ARTIFACTS_MISSING,
    LOG_UI_RENDER_TABS_DELETE_REQUESTED,
    LOG_UI_RENDER_TABS_EVENT_FOR_DELETED_RUN,
    LOG_UI_RENDER_TABS_TAB_ADDED,
    LOG_UI_TAB_CLOSURE_DIALOG_OPENED,
    LOG_UI_TAB_CLOSURE_CHOICE_SELECTED,
    LOG_UI_TAB_CLOSURE_CHOICE_CANCELLED,
    LOG_UI_TRAIN_FORM_TRACE,
    LOG_UI_TRAIN_FORM_INFO,
    LOG_UI_TRAIN_FORM_WARNING,
    LOG_UI_TRAIN_FORM_ERROR,
    LOG_UI_TRAIN_FORM_UI_PATH,
    LOG_UI_TRAIN_FORM_TELEMETRY_PATH,
    LOG_UI_POLICY_FORM_TRACE,
    LOG_UI_POLICY_FORM_INFO,
    LOG_UI_POLICY_FORM_ERROR,
    LOG_UI_WORKER_TABS_TRACE,
    LOG_UI_WORKER_TABS_INFO,
    LOG_UI_WORKER_TABS_WARNING,
    LOG_UI_WORKER_TABS_ERROR,
    LOG_UI_MULTI_AGENT_ENV_LOAD_REQUESTED,
    LOG_UI_MULTI_AGENT_ENV_LOADED,
    LOG_UI_MULTI_AGENT_ENV_LOAD_ERROR,
    LOG_UI_MULTI_AGENT_POLICY_LOAD_REQUESTED,
    LOG_UI_MULTI_AGENT_GAME_START_REQUESTED,
    LOG_UI_MULTI_AGENT_RESET_REQUESTED,
    LOG_UI_MULTI_AGENT_ACTION_SUBMITTED,
    LOG_UI_MULTI_AGENT_TRAIN_REQUESTED,
    LOG_UI_MULTI_AGENT_EVALUATE_REQUESTED,
    LOG_UI_MULTI_AGENT_ENV_NOT_LOADED,
    LOG_UI_POLICY_ASSIGNMENT_REQUESTED,
    LOG_UI_POLICY_ASSIGNMENT_LOADED,
    LOG_UI_POLICY_DISCOVERY_SCAN,
    LOG_UI_POLICY_DISCOVERY_FOUND,
    LOG_UI_PRESENTER_SIGNAL_CONNECTION_WARNING,
    LOG_UI_MAIN_WINDOW_SHUTDOWN_WARNING,
    LOG_UI_TENSORBOARD_KILL_WARNING,
    LOG_ADAPTER_RENDERING_WARNING,
    LOG_TRAINER_LAUNCHER_LOG_FLUSH_WARNING,
    LOG_WORKER_CONFIG_READ_WARNING,
    LOG_FASTLANE_CONNECTED,
    LOG_FASTLANE_UNAVAILABLE,
    LOG_FASTLANE_QUEUE_DEPTH,
    LOG_FASTLANE_READER_LAG,
    LOG_RUNBUS_UI_QUEUE_DEPTH,
    LOG_FASTLANE_HEADER_INVALID,
    LOG_FASTLANE_FRAME_READ_ERROR,
    LOG_FASTLANE_WRITER_CLOSED,
    LOG_FASTLANE_SHM_STALE_UNLINKED,
    LOG_FASTLANE_SHM_UNLINK_FAILED,
    LOG_FASTLANE_SHM_LEAK_RISK,
    LOG_RUNBUS_DB_QUEUE_DEPTH,
    LOG_UI_FASTLANE_EVAL_SUMMARY_UPDATE,
    LOG_UI_FASTLANE_EVAL_SUMMARY_WARNING,
    LOG_UTIL_QT_RESEED_SKIPPED,
    LOG_UTIL_QT_STATE_CAPTURE_FAILED,
    LOG_UTIL_SEED_CALLBACK_FAILED,
    LOG_UTIL_JSON_NUMPY_SCALAR_COERCE_FAILED,
    LOG_WORKER_RUNTIME_EVENT,
    LOG_WORKER_RUNTIME_WARNING,
    LOG_WORKER_RUNTIME_ERROR,
    LOG_WORKER_RUNTIME_DEBUG,
    LOG_WORKER_RUNTIME_JSON_SANITIZED,
    LOG_WORKER_CONFIG_EVENT,
    LOG_WORKER_CONFIG_WARNING,
    LOG_WORKER_CONFIG_UI_PATH,
    LOG_WORKER_CONFIG_DURABLE_PATH,
    LOG_WORKER_POLICY_EVENT,
    LOG_WORKER_POLICY_WARNING,
    LOG_WORKER_POLICY_ERROR,
    LOG_COUNTER_INITIALIZED,
    LOG_COUNTER_RESUME_SUCCESS,
    LOG_COUNTER_RESUME_FAILURE,
    LOG_COUNTER_MAX_REACHED,
    LOG_COUNTER_INVALID_STATE,
    LOG_COUNTER_CONCURRENCY_ERROR,
    LOG_COUNTER_PERSISTENCE_ERROR,
    LOG_COUNTER_VALIDATION_ERROR,
    LOG_COUNTER_NEXT_EPISODE,
    LOG_COUNTER_RESET,
    # Ray Worker
    LOG_RAY_WORKER_RUNTIME_STARTED,
    LOG_RAY_WORKER_RUNTIME_STOPPED,
    LOG_RAY_WORKER_RUNTIME_ERROR,
    LOG_RAY_WORKER_TRAINING_STARTED,
    LOG_RAY_WORKER_TRAINING_COMPLETED,
    LOG_RAY_WORKER_CHECKPOINT_SAVED,
    LOG_RAY_WORKER_FASTLANE_ENABLED,
    LOG_RAY_WORKER_ENV_WRAPPED,
    LOG_RAY_WORKER_POLICY_LOADED,
    LOG_RAY_WORKER_ANALYTICS_WRITTEN,
    # Ray Worker (LOG446–LOG462)
    LOG_WORKER_RAY_RUNTIME_STARTED,
    LOG_WORKER_RAY_RUNTIME_COMPLETED,
    LOG_WORKER_RAY_RUNTIME_FAILED,
    LOG_WORKER_RAY_CLUSTER_STARTED,
    LOG_WORKER_RAY_CLUSTER_SHUTDOWN,
    LOG_WORKER_RAY_TUNE_STARTED,
    LOG_WORKER_RAY_TUNE_COMPLETED,
    LOG_WORKER_RAY_RLLIB_TRAINING_STARTED,
    LOG_WORKER_RAY_RLLIB_TRAINING_ITERATION,
    LOG_WORKER_RAY_CHECKPOINT_SAVED,
    LOG_WORKER_RAY_CHECKPOINT_LOADED,
    LOG_WORKER_RAY_TENSORBOARD_ENABLED,
    LOG_WORKER_RAY_WANDB_ENABLED,
    LOG_WORKER_RAY_HEARTBEAT,
    LOG_WORKER_RAY_ANALYTICS_MANIFEST_CREATED,
    LOG_WORKER_RAY_LOG_FILE_CREATED,
    LOG_WORKER_RAY_ALGORITHM_BUILT,
    # Ray Evaluation
    LOG_RAY_EVAL_REQUESTED,
    LOG_RAY_EVAL_SETUP_STARTED,
    LOG_RAY_EVAL_SETUP_COMPLETED,
    LOG_RAY_EVAL_EPISODE_STARTED,
    LOG_RAY_EVAL_EPISODE_COMPLETED,
    LOG_RAY_EVAL_RUN_COMPLETED,
    LOG_RAY_EVAL_ERROR,
    LOG_RAY_EVAL_FASTLANE_CONNECTED,
    LOG_RAY_EVAL_POLICY_LOADED,
    LOG_RAY_EVAL_TAB_CREATED,
    # XuanCe Worker
    LOG_XUANCE_WORKER_RUNTIME_STARTED,
    LOG_XUANCE_WORKER_RUNTIME_STOPPED,
    LOG_XUANCE_WORKER_RUNTIME_ERROR,
    LOG_XUANCE_WORKER_TRAINING_STARTED,
    LOG_XUANCE_WORKER_TRAINING_COMPLETED,
    LOG_XUANCE_WORKER_CHECKPOINT_SAVED,
    LOG_XUANCE_WORKER_RUNNER_CREATED,
    LOG_XUANCE_WORKER_CONFIG_LOADED,
    LOG_XUANCE_WORKER_BENCHMARK_STARTED,
    LOG_XUANCE_WORKER_DEBUG,
    # BALROG Worker
    LOG_WORKER_BALROG_RUNTIME_STARTED,
    LOG_WORKER_BALROG_RUNTIME_STOPPED,
    LOG_WORKER_BALROG_RUNTIME_ERROR,
    LOG_WORKER_BALROG_EPISODE_STARTED,
    LOG_WORKER_BALROG_EPISODE_COMPLETED,
    LOG_WORKER_BALROG_LLM_REQUEST,
    LOG_WORKER_BALROG_LLM_RESPONSE,
    LOG_WORKER_BALROG_LLM_ERROR,
    LOG_WORKER_BALROG_ACTION_SELECTED,
    LOG_WORKER_BALROG_STEP_TELEMETRY,
    LOG_WORKER_BALROG_EPISODE_TELEMETRY,
    LOG_WORKER_BALROG_CONFIG_LOADED,
    LOG_WORKER_BALROG_ENV_CREATED,
    LOG_WORKER_BALROG_AGENT_CREATED,
    LOG_WORKER_BALROG_DEBUG,
    # XuanCe Worker
    LOG_WORKER_XUANCE_RUNTIME_STARTED,
    LOG_WORKER_XUANCE_RUNTIME_STOPPED,
    LOG_WORKER_XUANCE_RUNTIME_ERROR,
    LOG_WORKER_XUANCE_EPISODE_STARTED,
    LOG_WORKER_XUANCE_EPISODE_COMPLETED,
    LOG_WORKER_XUANCE_TRAINING_STARTED,
    LOG_WORKER_XUANCE_TRAINING_STEP,
    LOG_WORKER_XUANCE_TRAINING_COMPLETED,
    LOG_WORKER_XUANCE_CONFIG_LOADED,
    LOG_WORKER_XUANCE_ENV_CREATED,
    LOG_WORKER_XUANCE_AGENT_CREATED,
    LOG_WORKER_XUANCE_CHECKPOINT_SAVED,
    LOG_WORKER_XUANCE_METRICS_LOGGED,
    LOG_WORKER_XUANCE_DEBUG,
    # LLM Worker
    LOG_MOSAIC_WORKER_LLM_RUNTIME_STARTED,
    LOG_MOSAIC_WORKER_LLM_RUNTIME_STOPPED,
    LOG_MOSAIC_WORKER_LLM_RUNTIME_ERROR,
    LOG_MOSAIC_WORKER_LLM_EPISODE_STARTED,
    LOG_MOSAIC_WORKER_LLM_EPISODE_COMPLETED,
    LOG_MOSAIC_WORKER_LLM_REQUEST,
    LOG_MOSAIC_WORKER_LLM_RESPONSE,
    LOG_MOSAIC_WORKER_LLM_ERROR,
    LOG_MOSAIC_WORKER_LLM_ACTION_SELECTED,
    LOG_MOSAIC_WORKER_LLM_STEP_TELEMETRY,
    LOG_MOSAIC_WORKER_LLM_EPISODE_TELEMETRY,
    LOG_MOSAIC_WORKER_LLM_CONFIG_LOADED,
    LOG_MOSAIC_WORKER_LLM_ENV_CREATED,
    LOG_MOSAIC_WORKER_LLM_AGENT_CREATED,
    LOG_MOSAIC_WORKER_LLM_DEBUG,
    # VLM Worker
    LOG_MOSAIC_WORKER_VLM_RUNTIME_STARTED,
    LOG_MOSAIC_WORKER_VLM_RUNTIME_STOPPED,
    LOG_MOSAIC_WORKER_VLM_RUNTIME_ERROR,
    LOG_MOSAIC_WORKER_VLM_EPISODE_STARTED,
    LOG_MOSAIC_WORKER_VLM_EPISODE_COMPLETED,
    LOG_MOSAIC_WORKER_VLM_REQUEST,
    LOG_MOSAIC_WORKER_VLM_RESPONSE,
    LOG_MOSAIC_WORKER_VLM_ERROR,
    LOG_MOSAIC_WORKER_VLM_ACTION_SELECTED,
    LOG_MOSAIC_WORKER_VLM_STEP_TELEMETRY,
    LOG_MOSAIC_WORKER_VLM_EPISODE_TELEMETRY,
    LOG_MOSAIC_WORKER_VLM_CONFIG_LOADED,
    LOG_MOSAIC_WORKER_VLM_ENV_CREATED,
    LOG_MOSAIC_WORKER_VLM_AGENT_CREATED,
    LOG_MOSAIC_WORKER_VLM_DEBUG,
    LOG_MOSAIC_WORKER_VLM_EPISODE_AUTO_RESET,
    LOG_MOSAIC_WORKER_VLM_ACTION_DEFAULTED,
    # Auto-reset and action-defaulting events
    LOG_WORKER_BALROG_EPISODE_AUTO_RESET,
    LOG_WORKER_BALROG_ACTION_DEFAULTED,
    LOG_MOSAIC_WORKER_LLM_EPISODE_AUTO_RESET,
    LOG_MOSAIC_WORKER_LLM_ACTION_DEFAULTED,
    LOG_WORKER_CLEANRL_EPISODE_AUTO_RESET,
    LOG_WORKER_CLEANRL_SUBPROCESS_OOM_KILLED,
    LOG_WORKER_CLEANRL_SUBPROCESS_SIGNALED,
    LOG_WORKER_CLEANRL_SUBPROCESS_NO_OUTPUT,
    LOG_WORKER_CLEANRL_SUBPROCESS_MEMORY_SNAPSHOT,
    LOG_WORKER_CLEANRL_SUBPROCESS_EXIT_SUMMARY,
    LOG_WORKER_CLEANRL_TENSORBOARD_PATH_RESOLVED,
    LOG_WORKER_CLEANRL_SUBPROCESS_SLOW_START,
    LOG_WORKER_CLEANRL_SUBPROCESS_STDERR_ONLY,
    # Passive Worker
    LOG_WORKER_PASSIVE_RUNTIME_STARTED,
    LOG_WORKER_PASSIVE_RUNTIME_STOPPED,
    LOG_WORKER_PASSIVE_RUNTIME_ERROR,
    LOG_WORKER_PASSIVE_AGENT_READY,
    LOG_WORKER_PASSIVE_ENV_CREATED,
    LOG_WORKER_PASSIVE_ENV_RESET,
    LOG_WORKER_PASSIVE_DEBUG,
    # MushroomRL Worker
    LOG_WORKER_MUSHROOMRL_RUNTIME_STARTED,
    LOG_WORKER_MUSHROOMRL_RUNTIME_COMPLETED,
    LOG_WORKER_MUSHROOMRL_RUNTIME_FAILED,
    LOG_WORKER_MUSHROOMRL_SUBPROCESS_STDOUT,
    LOG_WORKER_MUSHROOMRL_SUBPROCESS_STDERR,
    # OmniSafe Worker
    LOG_WORKER_OMNISAFE_RUNTIME_STARTED,
    LOG_WORKER_OMNISAFE_RUNTIME_COMPLETED,
    LOG_WORKER_OMNISAFE_RUNTIME_FAILED,
    LOG_WORKER_OMNISAFE_SUBPROCESS_STDOUT,
    LOG_WORKER_OMNISAFE_SUBPROCESS_STDERR,
)


__all__ = (
    "LogConstant",
    "ALL_LOG_CONSTANTS",
    # Helper functions for runtime discovery
    "get_constant_by_code",
    "list_known_components",
    "get_component_snapshot",
    "validate_log_constants",
    # Constants
    "LOG_SESSION_ADAPTER_LOAD_ERROR",
    "LOG_SESSION_STEP_ERROR",
    "LOG_SESSION_EPISODE_ERROR",
    "LOG_SESSION_TIMER_PRECISION_WARNING",
    "LOG_INPUT_CONTROLLER_ERROR",
    "LOG_KEYBOARD_DETECTED",
    "LOG_KEYBOARD_ASSIGNED",
    "LOG_KEYBOARD_DETECTION_ERROR",
    "LOG_KEYBOARD_EVDEV_SETUP_START",
    "LOG_KEYBOARD_EVDEV_SETUP_SUCCESS",
    "LOG_KEYBOARD_EVDEV_SETUP_FAILED",
    "LOG_KEY_RESOLVER_INITIALIZED",
    "LOG_KEY_RESOLVER_UNAVAILABLE",
    "LOG_INPUT_MODE_CONFIGURED",
    "LOG_EVDEV_KEY_PRESSED",
    "LOG_EVDEV_KEY_RELEASED",
    "LOG_HUMAN_CONTROL_WORKER_LAUNCH",
    "LOG_HUMAN_CONTROL_WORKER_READY",
    "LOG_HUMAN_CONTROL_WORKER_ACTION",
    "LOG_HUMAN_CONTROL_WORKER_STEP",
    "LOG_HUMAN_CONTROL_WORKER_STOPPED",
    "LOG_HUMAN_CONTROL_WORKER_ERROR",
    "LOG_HUMAN_CONTROL_BRIDGE_START",
    "LOG_HUMAN_CONTROL_BRIDGE_STOP",
    "LOG_HUMAN_CONTROL_AUTO_ASSIGN",
    "LOG_HUMAN_CONTROL_INPUT_DISABLED",
    "LOG_LIVE_CONTROLLER_INITIALIZED",
    "LOG_LIVE_CONTROLLER_THREAD_STARTED",
    "LOG_LIVE_CONTROLLER_THREAD_STOPPED",
    "LOG_LIVE_CONTROLLER_THREAD_STOP_TIMEOUT",
    "LOG_LIVE_CONTROLLER_ALREADY_RUNNING",
    "LOG_LIVE_CONTROLLER_RUN_SUBSCRIBED",
    "LOG_LIVE_CONTROLLER_RUN_UNSUBSCRIBED",
    "LOG_LIVE_CONTROLLER_RUN_ALREADY_SUBSCRIBED",
    "LOG_LIVE_CONTROLLER_RUNBUS_SUBSCRIBED",
    "LOG_LIVE_CONTROLLER_RUN_COMPLETED",
    "LOG_LIVE_CONTROLLER_QUEUE_OVERFLOW",
    "LOG_LIVE_CONTROLLER_BUFFER_STEPS_FLUSHED",
    "LOG_LIVE_CONTROLLER_BUFFER_EPISODES_FLUSHED",
    "LOG_BUFFER_DROP",
    "LOG_TELEMETRY_CONTROLLER_THREAD_ERROR",
    "LOG_LIVE_CONTROLLER_SIGNAL_EMIT_FAILED",
    "LOG_LIVE_CONTROLLER_TAB_ADD_FAILED",
    "LOG_CREDIT_STARVED",
    "LOG_CREDIT_RESUMED",
    "LOG_LIVE_CONTROLLER_LOOP_EXITED",
    "LOG_TELEMETRY_SUBSCRIBE_ERROR",
    "LOG_LIVE_CONTROLLER_EVENT_FOR_DELETED_RUN",
    "LOG_LIVE_CONTROLLER_TAB_REQUESTED",
    "LOG_ADAPTER_INIT_ERROR",
    "LOG_ADAPTER_STEP_ERROR",
    "LOG_ADAPTER_PAYLOAD_ERROR",
    "LOG_ADAPTER_STATE_INVALID",
    "LOG_ADAPTER_RENDER_ERROR",
    "LOG_ADAPTER_ENV_CREATED",
    "LOG_ADAPTER_ENV_RESET",
    "LOG_ADAPTER_STEP_SUMMARY",
    "LOG_ADAPTER_ENV_CLOSED",
    "LOG_ADAPTER_MAP_GENERATION",
    "LOG_ADAPTER_HOLE_PLACEMENT",
    "LOG_ADAPTER_GOAL_OVERRIDE",
    "LOG_ADAPTER_RENDER_PAYLOAD",
    "LOG_ENV_MINIGRID_BOOT",
    "LOG_ENV_MINIGRID_STEP",
    "LOG_ENV_MINIGRID_ERROR",
    "LOG_ENV_MINIGRID_RENDER_WARNING",
    "LOG_ADAPTER_ALE_NAMESPACE_IMPORT_FAILED",
    "LOG_ADAPTER_ALE_METADATA_PROBE_FAILED",
    "LOG_MOSAIC_MULTIGRID_GOAL_SCORED",
    "LOG_MOSAIC_MULTIGRID_PASS_COMPLETED",
    "LOG_MOSAIC_MULTIGRID_STEAL_COMPLETED",
    "LOG_MOSAIC_MULTIGRID_VISIBILITY",
    "LOG_MOSAIC_MULTIGRID_OBSERVATION",
    "LOG_ENV_CRAFTER_BOOT",
    "LOG_ENV_CRAFTER_STEP",
    "LOG_ENV_CRAFTER_ERROR",
    "LOG_ENV_CRAFTER_RENDER_WARNING",
    "LOG_ENV_CRAFTER_ACHIEVEMENT",
    "LOG_ENV_PROCGEN_BOOT",
    "LOG_ENV_PROCGEN_STEP",
    "LOG_ENV_PROCGEN_ERROR",
    "LOG_ENV_PROCGEN_RENDER_WARNING",
    "LOG_ENV_PROCGEN_LEVEL_COMPLETE",
    "LOG_SERVICE_TELEMETRY_STEP_REJECTED",
    "LOG_SERVICE_TELEMETRY_ASYNC_ERROR",
    "LOG_SERVICE_DB_SINK_INITIALIZED",
    "LOG_SERVICE_DB_SINK_STARTED",
    "LOG_SERVICE_DB_SINK_ALREADY_RUNNING",
    "LOG_SERVICE_DB_SINK_STOPPED",
    "LOG_SERVICE_DB_SINK_STOP_TIMEOUT",
    "LOG_SERVICE_DB_SINK_LOOP_EXITED",
    "LOG_SERVICE_DB_SINK_FATAL",
    "LOG_SERVICE_DB_SINK_QUEUE_DEPTH",
    "LOG_SERVICE_DB_SINK_QUEUE_PRESSURE",
    "LOG_SERVICE_DB_SINK_FLUSH_STATS",
    "LOG_SERVICE_DB_SINK_FLUSH_LATENCY",
    "LOG_SERVICE_SQLITE_DEBUG",
    "LOG_SERVICE_SQLITE_INFO",
    "LOG_SERVICE_SQLITE_WARNING",
    "LOG_SERVICE_SQLITE_WORKER_STARTED",
    "LOG_SERVICE_SQLITE_WORKER_STOPPED",
    "LOG_SERVICE_SQLITE_WRITE_ERROR",
    "LOG_SERVICE_SQLITE_DISK_IO_ERROR",
    "LOG_SERVICE_SQLITE_INIT_ERROR",
    "LOG_SERVICE_SQLITE_DESERIALIZATION_FAILED",
    "LOG_SERVICE_TELEMETRY_BRIDGE_STEP_QUEUED",
    "LOG_SERVICE_TELEMETRY_BRIDGE_EPISODE_QUEUED",
    "LOG_SERVICE_TELEMETRY_BRIDGE_STEP_DELIVERED",
    "LOG_SERVICE_TELEMETRY_BRIDGE_EPISODE_DELIVERED",
    "LOG_SERVICE_TELEMETRY_BRIDGE_OVERFLOW",
    "LOG_SERVICE_TELEMETRY_BRIDGE_RUN_COMPLETED",
    "LOG_SERVICE_TELEMETRY_HUB_STARTED",
    "LOG_SERVICE_TELEMETRY_HUB_SUBSCRIBED",
    "LOG_SERVICE_TELEMETRY_HUB_TRACE",
    "LOG_SERVICE_TELEMETRY_HUB_ERROR",
    "LOG_RENDER_REGULATOR_NOT_STARTED",
    "LOG_RENDER_DROPPED_FRAME",
    "LOG_DAEMON_START",
    "LOG_TRAINER_CLIENT_CONNECTING",
    "LOG_TRAINER_CLIENT_CONNECTED",
    "LOG_TRAINER_CLIENT_CONNECTION_TIMEOUT",
    "LOG_TRAINER_CLIENT_LOOP_NONFATAL",
    "LOG_TRAINER_CLIENT_LOOP_ERROR",
    "LOG_TRAINER_CLIENT_SHUTDOWN_WARNING",
    "LOG_SERVICE_FRAME_INFO",
    "LOG_SERVICE_FRAME_WARNING",
    "LOG_SERVICE_FRAME_ERROR",
    "LOG_SERVICE_FRAME_DEBUG",
    "LOG_SERVICE_SESSION_NUMPY_SCALAR_COERCE_FAILED",
    "LOG_SERVICE_SESSION_NDARRAY_SUMMARY_FAILED",
    "LOG_SERVICE_SESSION_LAZYFRAMES_SUMMARY_FAILED",
    "LOG_SERVICE_SESSION_TOLIST_COERCE_FAILED",
    "LOG_SERVICE_SESSION_ITERABLE_COERCE_FAILED",
    "LOG_SERVICE_VALIDATION_DEBUG",
    "LOG_SERVICE_VALIDATION_WARNING",
    "LOG_SERVICE_VALIDATION_ERROR",
    "LOG_SERVICE_OPERATOR_REGISTERED",
    "LOG_SERVICE_OPERATOR_ACTIVATED",
    "LOG_SERVICE_OPERATOR_DEACTIVATED",
    "LOG_SERVICE_OPERATOR_ACTION_SELECTED",
    "LOG_SERVICE_OPERATOR_ERROR",
    "LOG_OPERATOR_INTERACTIVE_LAUNCHED",
    "LOG_OPERATOR_RESET_COMMAND_SENT",
    "LOG_OPERATOR_STEP_COMMAND_SENT",
    "LOG_OPERATOR_STOP_COMMAND_SENT",
    "LOG_OPERATOR_COMMAND_FAILED",
    "LOG_OPERATOR_RESET_ALL_STARTED",
    "LOG_OPERATOR_STEP_ALL_COMPLETED",
    "LOG_OPERATOR_STOP_ALL_COMPLETED",
    "LOG_SCRIPT_OPERATOR_LAUNCHED",
    "LOG_SCRIPT_OPERATOR_BEHAVIOR_SET",
    "LOG_SCRIPT_OPERATOR_EPISODE_START",
    "LOG_SCRIPT_OPERATOR_EPISODE_END",
    "LOG_SCRIPT_OPERATOR_TELEMETRY_EMITTED",
    "LOG_SCRIPT_LOADED",
    "LOG_SCRIPT_PARSED",
    "LOG_SCRIPT_VALIDATION_FAILED",
    "LOG_SCRIPT_AUTO_EXECUTION_STARTED",
    "LOG_SCRIPT_AUTO_EXECUTION_COMPLETED",
    "LOG_OPERATOR_INIT_AGENT_SENT",
    "LOG_OPERATOR_SELECT_ACTION_SENT",
    "LOG_OPERATOR_MULTIAGENT_LAUNCHED",
    "LOG_OPERATOR_MULTIAGENT_INIT_FAILED",
    "LOG_OPERATOR_MULTIAGENT_ACTION_FAILED",
    "LOG_OPERATOR_ENV_PREVIEW_STARTED",
    "LOG_OPERATOR_ENV_PREVIEW_SUCCESS",
    "LOG_OPERATOR_ENV_PREVIEW_IMPORT_ERROR",
    "LOG_OPERATOR_ENV_PREVIEW_ERROR",
    "LOG_OPERATOR_PARALLEL_RESET_STARTED",
    "LOG_OPERATOR_PARALLEL_STEP_STARTED",
    "LOG_OPERATOR_PARALLEL_STEP_COMPLETED",
    "LOG_OPERATOR_VIEW_SIZE_CONFIGURED",
    "LOG_VLLM_SERVER_COUNT_CHANGED",
    "LOG_VLLM_SERVER_STARTING",
    "LOG_VLLM_SERVER_RUNNING",
    "LOG_VLLM_SERVER_STOPPING",
    "LOG_VLLM_SERVER_START_FAILED",
    "LOG_VLLM_SERVER_PROCESS_EXITED",
    "LOG_VLLM_SERVER_NOT_RESPONDING",
    "LOG_VLLM_ORPHAN_PROCESS_KILLED",
    "LOG_VLLM_GPU_MEMORY_FREED",
    "LOG_VLLM_GPU_MEMORY_NOT_FREED",
    "LOG_SERVICE_ACTOR_SEED_ERROR",
    "LOG_RUNTIME_APP_DEBUG",
    "LOG_RUNTIME_APP_INFO",
    "LOG_RUNTIME_APP_WARNING",
    "LOG_RUNTIME_APP_ERROR",
    "LOG_UI_MAINWINDOW_TRACE",
    "LOG_UI_MAINWINDOW_INFO",
    "LOG_UI_MAINWINDOW_WARNING",
    "LOG_UI_MAINWINDOW_ERROR",
    "LOG_UI_MAINWINDOW_INVALID_CONFIG",
    "LOG_UI_LIVE_TAB_TRACE",
    "LOG_UI_LIVE_TAB_INFO",
    "LOG_UI_LIVE_TAB_WARNING",
    "LOG_UI_LIVE_TAB_ERROR",
    "LOG_UI_RENDER_TABS_TRACE",
    "LOG_UI_RENDER_TABS_INFO",
    "LOG_UI_RENDER_TABS_WARNING",
    "LOG_UI_RENDER_TABS_ERROR",
    "LOG_UI_RENDER_TABS_TENSORBOARD_STATUS",
    "LOG_UI_RENDER_TABS_TENSORBOARD_WAITING",
    "LOG_UI_RENDER_TABS_WANDB_STATUS",
    "LOG_UI_RENDER_TABS_WANDB_WARNING",
    "LOG_UI_RENDER_TABS_WANDB_ERROR",
    "LOG_UI_RENDER_TABS_WANDB_PROXY_APPLIED",
    "LOG_UI_RENDER_TABS_WANDB_PROXY_SKIPPED",
    "LOG_UI_RENDER_TABS_ARTIFACTS_MISSING",
    "LOG_UI_RENDER_TABS_DELETE_REQUESTED",
    "LOG_UI_RENDER_TABS_EVENT_FOR_DELETED_RUN",
    "LOG_UI_RENDER_TABS_TAB_ADDED",
    "LOG_UI_TAB_CLOSURE_DIALOG_OPENED",
    "LOG_UI_TAB_CLOSURE_CHOICE_SELECTED",
    "LOG_UI_TAB_CLOSURE_CHOICE_CANCELLED",
    "LOG_UI_TRAIN_FORM_TRACE",
    "LOG_UI_TRAIN_FORM_INFO",
    "LOG_UI_TRAIN_FORM_WARNING",
    "LOG_UI_TRAIN_FORM_ERROR",
    "LOG_UI_TRAIN_FORM_UI_PATH",
    "LOG_UI_TRAIN_FORM_TELEMETRY_PATH",
    "LOG_UI_POLICY_FORM_TRACE",
    "LOG_UI_POLICY_FORM_INFO",
    "LOG_UI_POLICY_FORM_ERROR",
    "LOG_UI_WORKER_TABS_TRACE",
    "LOG_UI_WORKER_TABS_INFO",
    "LOG_UI_WORKER_TABS_WARNING",
    "LOG_UI_WORKER_TABS_ERROR",
    "LOG_UI_MULTI_AGENT_ENV_LOAD_REQUESTED",
    "LOG_UI_MULTI_AGENT_ENV_LOADED",
    "LOG_UI_MULTI_AGENT_ENV_LOAD_ERROR",
    "LOG_UI_MULTI_AGENT_POLICY_LOAD_REQUESTED",
    "LOG_UI_MULTI_AGENT_GAME_START_REQUESTED",
    "LOG_UI_MULTI_AGENT_RESET_REQUESTED",
    "LOG_UI_MULTI_AGENT_ACTION_SUBMITTED",
    "LOG_UI_MULTI_AGENT_TRAIN_REQUESTED",
    "LOG_UI_MULTI_AGENT_EVALUATE_REQUESTED",
    "LOG_UI_MULTI_AGENT_ENV_NOT_LOADED",
    "LOG_UI_POLICY_ASSIGNMENT_REQUESTED",
    "LOG_UI_POLICY_ASSIGNMENT_LOADED",
    "LOG_UI_POLICY_DISCOVERY_SCAN",
    "LOG_UI_POLICY_DISCOVERY_FOUND",
    # LLM Chat UI
    "LOG_UI_CHAT_GPU_DETECTION_STARTED",
    "LOG_UI_CHAT_GPU_DETECTION_COMPLETED",
    "LOG_UI_CHAT_GPU_DETECTION_ERROR",
    "LOG_UI_CHAT_HF_TOKEN_SAVE_STARTED",
    "LOG_UI_CHAT_HF_TOKEN_SAVED",
    "LOG_UI_CHAT_HF_TOKEN_SAVE_ERROR",
    "LOG_UI_CHAT_HF_TOKEN_VALIDATION_STARTED",
    "LOG_UI_CHAT_HF_TOKEN_VALIDATED",
    "LOG_UI_CHAT_HF_TOKEN_VALIDATION_ERROR",
    "LOG_UI_CHAT_MODEL_DOWNLOAD_STARTED",
    "LOG_UI_CHAT_MODEL_DOWNLOAD_PROGRESS",
    "LOG_UI_CHAT_MODEL_DOWNLOADED",
    "LOG_UI_CHAT_MODEL_DOWNLOAD_ERROR",
    "LOG_UI_CHAT_REQUEST_STARTED",
    "LOG_UI_CHAT_REQUEST_COMPLETED",
    "LOG_UI_CHAT_REQUEST_ERROR",
    "LOG_UI_CHAT_REQUEST_CANCELLED",
    "LOG_UI_CHAT_PROXY_ENABLED",
    "LOG_UI_CHAT_PROXY_DISABLED",
    "LOG_UI_CHAT_CLEANUP_WARNING",
    # Board Config Dialog
    "LOG_UI_BOARD_CONFIG_DIALOG_OPENED",
    "LOG_UI_BOARD_CONFIG_STATE_APPLIED",
    "LOG_UI_BOARD_CONFIG_VALIDATION_ERROR",
    "LOG_UI_BOARD_CONFIG_PRESET_LOADED",
    "LOG_UI_BOARD_CONFIG_PIECE_MOVED",
    "LOG_UI_BOARD_CONFIG_PIECE_REMOVED",
    "LOG_UI_BOARD_CONFIG_NOTATION_EDITED",
    "LOG_UI_BOARD_CONFIG_FACTORY_CREATE",
    "LOG_UI_BOARD_CONFIG_UNSUPPORTED_GAME",
    "LOG_UI_BOARD_CONFIG_ENV_INIT_CUSTOM",
    "LOG_OP_GRID_CONFIG_DIALOG_OPENED",
    "LOG_OP_GRID_CONFIG_STATE_APPLIED",
    "LOG_OP_GRID_CONFIG_VALIDATION_ERROR",
    "LOG_OP_GRID_CONFIG_PRESET_LOADED",
    "LOG_OP_GRID_CONFIG_OBJECT_PLACED",
    "LOG_OP_GRID_CONFIG_OBJECT_REMOVED",
    "LOG_OP_GRID_CONFIG_FACTORY_CREATE",
    "LOG_OP_GRID_CONFIG_UNSUPPORTED_ENV",
    "LOG_OP_GRID_CONFIG_ENV_INIT_CUSTOM",
    "LOG_OP_AGENT_LINK_DIALOG_OPENED",
    "LOG_OP_AGENT_LINK_GROUP_CREATED",
    "LOG_OP_AGENT_LINKED",
    "LOG_OP_AGENT_UNLINKED",
    "LOG_OP_AGENT_LINK_GROUP_DISSOLVED",
    "LOG_OP_AGENT_LINK_VALIDATION_ERROR",
    "LOG_OP_AGENT_LINK_PRIMARY_PROMOTED",
    "LOG_OP_AGENT_LINK_POLICY_SYNCED",
    "LOG_OP_AGENT_LINK_INCOMPATIBLE_CHANGE",
    "LOG_UI_PRESENTER_SIGNAL_CONNECTION_WARNING",
    "LOG_UI_MAIN_WINDOW_SHUTDOWN_WARNING",
    "LOG_UI_TENSORBOARD_KILL_WARNING",
    "LOG_ADAPTER_RENDERING_WARNING",
    "LOG_TRAINER_LAUNCHER_LOG_FLUSH_WARNING",
    "LOG_WORKER_CONFIG_READ_WARNING",
    "LOG_FASTLANE_CONNECTED",
    "LOG_FASTLANE_UNAVAILABLE",
    "LOG_FASTLANE_QUEUE_DEPTH",
    "LOG_FASTLANE_READER_LAG",
    "LOG_RUNBUS_UI_QUEUE_DEPTH",
    "LOG_FASTLANE_HEADER_INVALID",
    "LOG_FASTLANE_FRAME_READ_ERROR",
    "LOG_FASTLANE_WRITER_CLOSED",
    "LOG_FASTLANE_SHM_STALE_UNLINKED",
    "LOG_FASTLANE_SHM_UNLINK_FAILED",
    "LOG_FASTLANE_SHM_LEAK_RISK",
    "LOG_RUNBUS_DB_QUEUE_DEPTH",
    "LOG_UI_FASTLANE_EVAL_SUMMARY_UPDATE",
    "LOG_UI_FASTLANE_EVAL_SUMMARY_WARNING",
    "LOG_UTIL_QT_RESEED_SKIPPED",
    "LOG_UTIL_QT_STATE_CAPTURE_FAILED",
    "LOG_UTIL_SEED_CALLBACK_FAILED",
    "LOG_UTIL_JSON_NUMPY_SCALAR_COERCE_FAILED",
    "LOG_WORKER_RUNTIME_EVENT",
    "LOG_WORKER_RUNTIME_WARNING",
    "LOG_WORKER_RUNTIME_ERROR",
    "LOG_WORKER_RUNTIME_DEBUG",
    "LOG_WORKER_RUNTIME_JSON_SANITIZED",
    "LOG_WORKER_CONFIG_EVENT",
    "LOG_WORKER_CONFIG_WARNING",
    "LOG_WORKER_CONFIG_UI_PATH",
    "LOG_WORKER_CONFIG_DURABLE_PATH",
    "LOG_WORKER_POLICY_EVENT",
    "LOG_WORKER_POLICY_WARNING",
    "LOG_WORKER_POLICY_ERROR",
    "LOG_WORKER_POLICY_EVAL_STARTED",
    "LOG_WORKER_POLICY_EVAL_COMPLETED",
    "LOG_WORKER_POLICY_EVAL_BATCH_STARTED",
    "LOG_WORKER_POLICY_EVAL_BATCH_COMPLETED",
    "LOG_WORKER_POLICY_LOAD_FAILED",
    "LOG_COUNTER_INITIALIZED",
    "LOG_COUNTER_RESUME_SUCCESS",
    "LOG_COUNTER_RESUME_FAILURE",
    "LOG_COUNTER_MAX_REACHED",
    "LOG_COUNTER_INVALID_STATE",
    "LOG_COUNTER_CONCURRENCY_ERROR",
    "LOG_COUNTER_PERSISTENCE_ERROR",
    "LOG_COUNTER_VALIDATION_ERROR",
    "LOG_COUNTER_NEXT_EPISODE",
    "LOG_COUNTER_RESET",
    # Ray Worker
    "LOG_RAY_WORKER_RUNTIME_STARTED",
    "LOG_RAY_WORKER_RUNTIME_STOPPED",
    "LOG_RAY_WORKER_RUNTIME_ERROR",
    "LOG_RAY_WORKER_TRAINING_STARTED",
    "LOG_RAY_WORKER_TRAINING_COMPLETED",
    "LOG_RAY_WORKER_CHECKPOINT_SAVED",
    "LOG_RAY_WORKER_FASTLANE_ENABLED",
    "LOG_RAY_WORKER_ENV_WRAPPED",
    "LOG_RAY_WORKER_POLICY_LOADED",
    "LOG_RAY_WORKER_ANALYTICS_WRITTEN",
    # Ray Worker (LOG446–LOG462)
    "LOG_WORKER_RAY_RUNTIME_STARTED",
    "LOG_WORKER_RAY_RUNTIME_COMPLETED",
    "LOG_WORKER_RAY_RUNTIME_FAILED",
    "LOG_WORKER_RAY_CLUSTER_STARTED",
    "LOG_WORKER_RAY_CLUSTER_SHUTDOWN",
    "LOG_WORKER_RAY_TUNE_STARTED",
    "LOG_WORKER_RAY_TUNE_COMPLETED",
    "LOG_WORKER_RAY_RLLIB_TRAINING_STARTED",
    "LOG_WORKER_RAY_RLLIB_TRAINING_ITERATION",
    "LOG_WORKER_RAY_CHECKPOINT_SAVED",
    "LOG_WORKER_RAY_CHECKPOINT_LOADED",
    "LOG_WORKER_RAY_TENSORBOARD_ENABLED",
    "LOG_WORKER_RAY_WANDB_ENABLED",
    "LOG_WORKER_RAY_HEARTBEAT",
    "LOG_WORKER_RAY_ANALYTICS_MANIFEST_CREATED",
    "LOG_WORKER_RAY_LOG_FILE_CREATED",
    "LOG_WORKER_RAY_ALGORITHM_BUILT",
    # Ray Evaluation
    "LOG_RAY_EVAL_REQUESTED",
    "LOG_RAY_EVAL_SETUP_STARTED",
    "LOG_RAY_EVAL_SETUP_COMPLETED",
    "LOG_RAY_EVAL_EPISODE_STARTED",
    "LOG_RAY_EVAL_EPISODE_COMPLETED",
    "LOG_RAY_EVAL_RUN_COMPLETED",
    "LOG_RAY_EVAL_ERROR",
    "LOG_RAY_EVAL_FASTLANE_CONNECTED",
    "LOG_RAY_EVAL_POLICY_LOADED",
    "LOG_RAY_EVAL_TAB_CREATED",
    # XuanCe Worker
    "LOG_XUANCE_WORKER_RUNTIME_STARTED",
    "LOG_XUANCE_WORKER_RUNTIME_STOPPED",
    "LOG_XUANCE_WORKER_RUNTIME_ERROR",
    "LOG_XUANCE_WORKER_TRAINING_STARTED",
    "LOG_XUANCE_WORKER_TRAINING_COMPLETED",
    "LOG_XUANCE_WORKER_CHECKPOINT_SAVED",
    "LOG_XUANCE_WORKER_RUNNER_CREATED",
    "LOG_XUANCE_WORKER_CONFIG_LOADED",
    "LOG_XUANCE_WORKER_BENCHMARK_STARTED",
    "LOG_XUANCE_WORKER_DEBUG",
    # Human Action Debug
    "LOG_HUMAN_ACTION_BUTTON_CLICKED",
    "LOG_HUMAN_ACTION_RECEIVED",
    "LOG_HUMAN_ACTION_SENT_TO_SUBPROCESS",
    "LOG_HUMAN_ACTION_SIGNAL_EMITTED",
    # BALROG Worker
    "LOG_WORKER_BALROG_RUNTIME_STARTED",
    "LOG_WORKER_BALROG_RUNTIME_STOPPED",
    "LOG_WORKER_BALROG_RUNTIME_ERROR",
    "LOG_WORKER_BALROG_EPISODE_STARTED",
    "LOG_WORKER_BALROG_EPISODE_COMPLETED",
    "LOG_WORKER_BALROG_LLM_REQUEST",
    "LOG_WORKER_BALROG_LLM_RESPONSE",
    "LOG_WORKER_BALROG_LLM_ERROR",
    "LOG_WORKER_BALROG_ACTION_SELECTED",
    "LOG_WORKER_BALROG_STEP_TELEMETRY",
    "LOG_WORKER_BALROG_EPISODE_TELEMETRY",
    "LOG_WORKER_BALROG_CONFIG_LOADED",
    "LOG_WORKER_BALROG_ENV_CREATED",
    "LOG_WORKER_BALROG_AGENT_CREATED",
    "LOG_WORKER_BALROG_DEBUG",
    # XuanCe Worker
    "LOG_WORKER_XUANCE_RUNTIME_STARTED",
    "LOG_WORKER_XUANCE_RUNTIME_STOPPED",
    "LOG_WORKER_XUANCE_RUNTIME_ERROR",
    "LOG_WORKER_XUANCE_EPISODE_STARTED",
    "LOG_WORKER_XUANCE_EPISODE_COMPLETED",
    "LOG_WORKER_XUANCE_TRAINING_STARTED",
    "LOG_WORKER_XUANCE_TRAINING_STEP",
    "LOG_WORKER_XUANCE_TRAINING_COMPLETED",
    "LOG_WORKER_XUANCE_CONFIG_LOADED",
    "LOG_WORKER_XUANCE_ENV_CREATED",
    "LOG_WORKER_XUANCE_AGENT_CREATED",
    "LOG_WORKER_XUANCE_CHECKPOINT_SAVED",
    "LOG_WORKER_XUANCE_METRICS_LOGGED",
    "LOG_WORKER_XUANCE_DEBUG",
    # LLM Worker
    "LOG_MOSAIC_WORKER_LLM_RUNTIME_STARTED",
    "LOG_MOSAIC_WORKER_LLM_RUNTIME_STOPPED",
    "LOG_MOSAIC_WORKER_LLM_RUNTIME_ERROR",
    "LOG_MOSAIC_WORKER_LLM_EPISODE_STARTED",
    "LOG_MOSAIC_WORKER_LLM_EPISODE_COMPLETED",
    "LOG_MOSAIC_WORKER_LLM_REQUEST",
    "LOG_MOSAIC_WORKER_LLM_RESPONSE",
    "LOG_MOSAIC_WORKER_LLM_ERROR",
    "LOG_MOSAIC_WORKER_LLM_ACTION_SELECTED",
    "LOG_MOSAIC_WORKER_LLM_STEP_TELEMETRY",
    "LOG_MOSAIC_WORKER_LLM_EPISODE_TELEMETRY",
    "LOG_MOSAIC_WORKER_LLM_CONFIG_LOADED",
    "LOG_MOSAIC_WORKER_LLM_ENV_CREATED",
    "LOG_MOSAIC_WORKER_LLM_AGENT_CREATED",
    "LOG_MOSAIC_WORKER_LLM_DEBUG",
    # VLM Worker
    "LOG_MOSAIC_WORKER_VLM_RUNTIME_STARTED",
    "LOG_MOSAIC_WORKER_VLM_RUNTIME_STOPPED",
    "LOG_MOSAIC_WORKER_VLM_RUNTIME_ERROR",
    "LOG_MOSAIC_WORKER_VLM_EPISODE_STARTED",
    "LOG_MOSAIC_WORKER_VLM_EPISODE_COMPLETED",
    "LOG_MOSAIC_WORKER_VLM_REQUEST",
    "LOG_MOSAIC_WORKER_VLM_RESPONSE",
    "LOG_MOSAIC_WORKER_VLM_ERROR",
    "LOG_MOSAIC_WORKER_VLM_ACTION_SELECTED",
    "LOG_MOSAIC_WORKER_VLM_STEP_TELEMETRY",
    "LOG_MOSAIC_WORKER_VLM_EPISODE_TELEMETRY",
    "LOG_MOSAIC_WORKER_VLM_CONFIG_LOADED",
    "LOG_MOSAIC_WORKER_VLM_ENV_CREATED",
    "LOG_MOSAIC_WORKER_VLM_AGENT_CREATED",
    "LOG_MOSAIC_WORKER_VLM_DEBUG",
    "LOG_MOSAIC_WORKER_VLM_EPISODE_AUTO_RESET",
    "LOG_MOSAIC_WORKER_VLM_ACTION_DEFAULTED",
    # Auto-reset and action-defaulting events
    "LOG_WORKER_BALROG_EPISODE_AUTO_RESET",
    "LOG_WORKER_BALROG_ACTION_DEFAULTED",
    "LOG_MOSAIC_WORKER_LLM_EPISODE_AUTO_RESET",
    "LOG_MOSAIC_WORKER_LLM_ACTION_DEFAULTED",
    "LOG_WORKER_CLEANRL_EPISODE_AUTO_RESET",
    "LOG_WORKER_CLEANRL_SUBPROCESS_OOM_KILLED",
    "LOG_WORKER_CLEANRL_SUBPROCESS_SIGNALED",
    "LOG_WORKER_CLEANRL_SUBPROCESS_NO_OUTPUT",
    "LOG_WORKER_CLEANRL_SUBPROCESS_MEMORY_SNAPSHOT",
    "LOG_WORKER_CLEANRL_SUBPROCESS_EXIT_SUMMARY",
    "LOG_WORKER_CLEANRL_TENSORBOARD_PATH_RESOLVED",
    "LOG_WORKER_CLEANRL_SUBPROCESS_SLOW_START",
    "LOG_WORKER_CLEANRL_SUBPROCESS_STDERR_ONLY",
    # Passive Worker
    "LOG_WORKER_PASSIVE_RUNTIME_STARTED",
    "LOG_WORKER_PASSIVE_RUNTIME_STOPPED",
    "LOG_WORKER_PASSIVE_RUNTIME_ERROR",
    "LOG_WORKER_PASSIVE_AGENT_READY",
    "LOG_WORKER_PASSIVE_ENV_CREATED",
    "LOG_WORKER_PASSIVE_ENV_RESET",
    "LOG_WORKER_PASSIVE_DEBUG",
    # Worker Availability Messages
    "GODOT_NOT_INSTALLED_TITLE",
    "GODOT_NOT_INSTALLED_MSG",
    "GODOT_BINARY_NOT_FOUND_TITLE",
    "GODOT_BINARY_NOT_FOUND_MSG",
    "MJPC_NOT_INSTALLED_TITLE",
    "MJPC_NOT_INSTALLED_MSG",
    "MJPC_NOT_BUILT_TITLE",
    "MJPC_NOT_BUILT_MSG",
)

# ================================================================
# Worker Availability Messages (User Notifications)
# ================================================================

# Godot Worker Messages
GODOT_NOT_INSTALLED_TITLE = "Godot Not Available"
GODOT_NOT_INSTALLED_MSG = (
    "Godot worker is not installed.\n\n"
    "Install with:\n\n"
    "  pip install -e 3rd_party/workers/godot_worker\n\n"
    "Also requires the Godot binary in 3rd_party/workers/godot_worker/bin/"
)

GODOT_BINARY_NOT_FOUND_TITLE = "Godot Not Available"
GODOT_BINARY_NOT_FOUND_MSG = (
    "Godot binary not found.\n\n"
    "Please ensure the Godot binary is installed at:\n"
    "  {godot_binary}\n\n"
    "You can copy the Godot binary from:\n"
    "  Vesna_RL/Godot_v4.5.1-stable_linux.x86_64\n"
    "to:\n"
    "  3rd_party/workers/godot_worker/bin/godot"
)

# MuJoCo MPC Worker Messages
MJPC_NOT_INSTALLED_TITLE = "MJPC Not Available"
MJPC_NOT_INSTALLED_MSG = (
    "MuJoCo MPC worker is not installed.\n\n"
    "Install with:\n\n"
    "  pip install -e 3rd_party/workers/mujoco_mpc_worker\n\n"
    "Also requires building the MJPC agent_server binary.\n"
    "See: 3rd_party/workers/mujoco_mpc_worker/mujoco_mpc/README.md"
)

MJPC_NOT_BUILT_TITLE = "MJPC Not Built"
MJPC_NOT_BUILT_MSG = (
    "MuJoCo MPC needs to be built first.\n\n"
    "Run the following commands:\n\n"
    "  cd 3rd_party/workers/mujoco_mpc_worker/mujoco_mpc/build\n"
    "  cmake .. -DCMAKE_BUILD_TYPE=Release -G Ninja\n"
    "  ninja -j$(nproc)\n\n"
    "Source dir: {source_dir}\n"
    "Source exists: {source_exists}"
)
