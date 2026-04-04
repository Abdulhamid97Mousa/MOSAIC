from __future__ import annotations

"""Qt session controller that bridges Gym adapters to the GUI."""

import hashlib
import logging
import time
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from typing import Any, Dict, Mapping, Optional, cast

import gymnasium.spaces as spaces
import numpy as np
from PyQt6 import QtCore
from PyQt6.QtCore import pyqtSignal  # type: ignore[attr-defined]

from gym_gui.config.game_configs import (
    BipedalWalkerConfig,
    CarRacingConfig,
    CliffWalkingConfig,
    FrozenLakeConfig,
    GameConfig,
    LunarLanderConfig,
    MiniGridConfig,
    TaxiConfig,
)
from gym_gui.config.settings import Settings
from gym_gui.constants import DEFAULT_RENDER_DELAY_MS, format_episode_id
from gym_gui.constants.constants_telemetry import (
    TELEMETRY_KEY_AUTORESET_MODE,
    TELEMETRY_KEY_SPACE_SIGNATURE,
    TELEMETRY_KEY_TIME_STEP,
    TELEMETRY_KEY_VECTOR_METADATA,
)
from gym_gui.constants.constants_vector import SUPPORTED_AUTORESET_MODES
from gym_gui.controllers.interaction import (
    AleInteractionController,
    Box2DInteractionController,
    GriddlyInteractionController,
    InteractionController,
    JumanjiArcadeInteractionController,
    ProcgenInteractionController,
    SMACInteractionController,
    TurnBasedInteractionController,
    ViZDoomInteractionController,
)
from gym_gui.controllers.malmo_interaction import MalmoInteractionController
from gym_gui.core.adapters.base import AdapterContext, AdapterStep, EnvironmentAdapter
from gym_gui.core.data_model import EpisodeRollup, StepRecord
from gym_gui.core.enums import (
    CONTINUOUS_GRIDDLY_GAMES,
    ENVIRONMENT_FAMILY_BY_GAME,
    ControlMode,
    EnvironmentFamily,
    GameId,
)
from gym_gui.core.factories.adapters import create_adapter, get_adapter_cls
from gym_gui.core.run_counter_manager import RunCounterManager
from gym_gui.core.schema import schema_registry
from gym_gui.core.spaces.vector_metadata import extract_vector_step_details
from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_NORMALIZATION_STATS_DROPPED,
    LOG_SCHEMA_MISMATCH,
    LOG_SESSION_ADAPTER_LOAD_ERROR,
    LOG_SESSION_EPISODE_ERROR,
    LOG_SESSION_STEP_ERROR,
    LOG_SESSION_TIMER_PRECISION_WARNING,
    LOG_SPACE_DESCRIPTOR_MISSING,
    LOG_VECTOR_AUTORESET_MODE,
)
from gym_gui.services.actor import ActorService, EpisodeSummary, StepSnapshot
from gym_gui.services.policy_mapping import PolicyMappingService
from gym_gui.services.service_locator import get_service_locator
from gym_gui.services.storage import StorageRecorderService
from gym_gui.services.telemetry import TelemetryService
from gym_gui.utils.fps_counter import FpsCounter
from gym_gui.utils.seeding import SessionSeedManager
from gym_gui.utils.timekeeping import SessionTimers

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class SessionState:
    """Lightweight snapshot of the current episode state."""

    game_id: GameId
    control_mode: ControlMode
    step_index: int
    turn: str
    terminated: bool
    truncated: bool


class SessionController(QtCore.QObject, LogConstantMixin):
    """Manage adapter lifecycle and emit Qt-friendly signals for the UI."""

    session_initialized = pyqtSignal(str, str, object)
    step_processed = pyqtSignal(object, int)
    episode_finished = pyqtSignal(bool)
    status_message = pyqtSignal(str)
    awaiting_human = pyqtSignal(bool, str)
    turn_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    auto_play_state_changed = pyqtSignal(bool)
    seed_applied = pyqtSignal(int)
    fps_updated = pyqtSignal(float)

    def __init__(self, settings: Settings, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._logger = _LOGGER
        self._settings = settings
        self._adapter: EnvironmentAdapter | None = None
        self._game_id: GameId | None = None
        self._control_mode: ControlMode = settings.default_control_mode
        self._step_index: int = -1
        self._turn: str = "human"
        self._game_started: bool = False
        self._game_paused: bool = False
        self._auto_timer = QtCore.QTimer(self)
        self._auto_timer.setInterval(DEFAULT_RENDER_DELAY_MS)
        if hasattr(self._auto_timer, "setTimerType"):
            try:
                self._auto_timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)  # type: ignore[attr-defined]
            except Exception as exc:
                self.log_constant(
                    LOG_SESSION_TIMER_PRECISION_WARNING,
                    exc_info=exc,
                    extra={"timer": "auto"},
                )
        self._auto_timer.timeout.connect(self._auto_step)
        self._idle_timer = QtCore.QTimer(self)
        self._idle_timer.setInterval(DEFAULT_RENDER_DELAY_MS)
        if hasattr(self._idle_timer, "setTimerType"):
            try:
                self._idle_timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)  # type: ignore[attr-defined]
            except Exception as exc:
                self.log_constant(
                    LOG_SESSION_TIMER_PRECISION_WARNING,
                    exc_info=exc,
                    extra={"timer": "idle"},
                )
        self._idle_timer.timeout.connect(self._idle_step)
        self._user_idle_interval: int | None = None
        self._current_idle_interval = self._idle_timer.interval()
        self._awaiting_human = False
        self._schema_alerts_emitted: set[str] = set()
        self._pending_input_label: str | None = None
        self._last_agent_position: tuple[int, int] | None = None
        self._timers = SessionTimers()
        self._fps_counter = FpsCounter(window_s=1.5)
        self._seed_manager = SessionSeedManager()
        self._settings_overrides: dict[str, Any] = {}
        self._effective_settings: Settings = settings
        locator = get_service_locator()
        self._telemetry = locator.resolve(TelemetryService)
        self._slow_lane_enabled: bool = False
        self._actor_service = locator.resolve(ActorService)
        self._policy_mapping = locator.resolve(PolicyMappingService)
        self._storage_service = locator.resolve(StorageRecorderService)
        if self._actor_service is not None:
            self._seed_manager.register_consumer("actor_service", self._actor_service.seed)
        self._seed_manager.register_consumer("session_timers", self._seed_timers)
        self._run_counter_manager: RunCounterManager | None = None  # Set on load_game
        self._episode_id: str | None = None
        self._episode_active = False
        self._episode_reward = 0.0
        self._episode_metadata: dict[str, Any] = {}
        self._last_seed_state: dict[str, Any] | None = None
        self._last_step: AdapterStep | None = None
        self._passive_action: Any | None = None
        self._allow_seed_reuse = settings.allow_seed_reuse
        self._last_seed = 0
        self._next_seed = max(1, settings.default_seed)
        self._current_seed = self._next_seed
        self._game_config: GameConfig | None = None
        self._interaction: InteractionController | None = None
        self._input_controller: Any = None  # HumanInputController, set via set_input_controller

    def set_input_controller(self, controller: Any) -> None:
        """Set the HumanInputController for querying key state during idle ticks.

        Args:
            controller: The HumanInputController instance.
        """
        self._input_controller = controller

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def configure_interval(self, interval_ms: int) -> None:
        interval = max(50, interval_ms)
        self._auto_timer.setInterval(interval)
        self._user_idle_interval = interval
        self._update_idle_timer()

    def set_slow_lane_enabled(self, enabled: bool) -> None:
        self._slow_lane_enabled = bool(enabled)

    def start_game(self) -> None:
        """Mark the game as started and enable input controls."""
        if self._adapter is None:
            self.status_message.emit("⚠ No environment loaded. Click 'Load Environment' first.")
            return

        self._game_started = True
        self._game_paused = False
        # Note: Human input controller is set up in main_window, not here
        self.status_message.emit("Game started! Use controls to play.")
        self._update_idle_timer()

    def pause_game(self) -> None:
        """Pause the game - stops all actions."""
        if not self._game_started:
            self.status_message.emit("⚠ No game to pause.")
            return
        self.stop_auto_play()
        self._stop_idle_tick()  # Stop the idle timer for Box2D games
        self._game_paused = True
        self.status_message.emit("Game paused.")

    def resume_game(self) -> None:
        """Resume the game from paused state."""
        if not self._game_started:
            self.status_message.emit("⚠ No game to resume.")
            return
        self._game_paused = False
        self._update_idle_timer()  # Restart idle timer if needed for Box2D games
        self.status_message.emit("Game resumed.")

    def terminate_game(self) -> None:
        """Stop the game and mark episode as finished."""
        self.stop_auto_play()
        self._game_started = False
        self._game_paused = False
        self._stop_idle_tick()
        if self._awaiting_human:
            self._awaiting_human = False
            self.awaiting_human.emit(False, "Game terminated")
        self._pending_input_label = None
        if self._episode_active and self._last_step is not None:
            self._finalize_episode(self._last_step, aborted=True)
        self.episode_finished.emit(True)
        self.status_message.emit("Game terminated. Click 'Start Game' to play again.")

    def load_environment(
        self,
        game_id: GameId,
        control_mode: ControlMode,
        *,
        seed: int | None = None,
        settings_overrides: dict[str, Any] | None = None,
        game_config: GameConfig | None = None,
    ) -> None:
        self.stop_auto_play()
        self._dispose_adapter()
        if settings_overrides is not None:
            self._settings_overrides = settings_overrides
        if self._settings_overrides:
            merged_settings = asdict(self._settings)
            merged_settings.update(self._settings_overrides)
            effective_settings = Settings(**merged_settings)
        else:
            effective_settings = self._settings
        self._effective_settings = effective_settings
        context = AdapterContext(
            settings=effective_settings,
            control_mode=control_mode,
            logger_factory=logging.getLogger,
        )
        seed_to_use = self._activate_seed(seed, advance_next=False)
        try:
            adapter = create_adapter(game_id, context, game_config=game_config)
            adapter.ensure_control_mode(control_mode)
            adapter.load()
            initial_step = adapter.reset(seed=seed_to_use)
        except Exception as exc:  # pragma: no cover - UI surfaces error
            self.log_constant(
                LOG_SESSION_ADAPTER_LOAD_ERROR,
                exc_info=exc,
                extra={
                    "game_id": game_id.value,
                    "control_mode": control_mode.value,
                },
            )
            self.error_occurred.emit(str(exc))
            return

        self._adapter = adapter
        self._game_id = game_id
        self._control_mode = control_mode
        self._schema_alerts_emitted.clear()
        self._game_config = game_config
        self._step_index = 0
        self._turn = "human"
        self._game_started = False  # Game not started until user clicks "Start Game"
        self._game_paused = False
        self._awaiting_human = False  # Don't await input until game starts
        self._begin_episode(game_id, control_mode)
        self._timers.reset_episode()
        # Create interaction controller BEFORE emitting session_initialized,
        # because UI handlers (e.g. MalmoEnvLoader) inspect _interaction to
        # decide between native vs legacy mouse capture paths.
        try:
            family = ENVIRONMENT_FAMILY_BY_GAME.get(game_id)
        except Exception:
            family = None
        self._interaction = self._create_interaction_controller(family)
        self._passive_action = self._resolve_passive_action()
        self._last_agent_position = self._extract_agent_position(initial_step)
        self._last_step = initial_step
        self.session_initialized.emit(game_id.value, control_mode.value, initial_step)
        self.step_processed.emit(initial_step, self._step_index)
        self._update_status(initial_step, prefix="Environment ready")
        self.awaiting_human.emit(False, "Click 'Start Game' to begin")
        self.turn_changed.emit(self._turn)
        self._pending_input_label = "environment_ready"
        self._record_step(initial_step, action=None, input_source=self._pending_input_label)
        self._update_idle_timer()

    def reset_environment(self, *, seed: int | None = None) -> None:
        if self._adapter is None or self._game_id is None:
            return
        self.stop_auto_play()
        seed_to_use = self._activate_seed(seed, advance_next=False)
        try:
            step = self._adapter.reset(seed=seed_to_use)
        except Exception as exc:  # pragma: no cover - UI surfaces error
            self.log_constant(
                LOG_SESSION_ADAPTER_LOAD_ERROR,
                exc_info=exc,
                extra={
                    "game_id": self._game_id.value if self._game_id else "unknown",
                    "control_mode": self._control_mode.value if self._game_id else "unknown",
                    "context": "reset",
                },
            )
            self.error_occurred.emit(str(exc))
            return
        self._step_index = 0
        self._turn = "human"
        self._game_started = False  # Reset requires "Start Game" again
        self._game_paused = False
        self._awaiting_human = False
        self._schema_alerts_emitted.clear()
        self._begin_episode(self._game_id, self._control_mode)
        self.step_processed.emit(step, self._step_index)
        self._update_status(step, prefix="Environment reset")
        self.awaiting_human.emit(False, "Click 'Start Game' to begin")
        self.turn_changed.emit(self._turn)
        self._last_agent_position = self._extract_agent_position(step)
        self._timers.reset_episode()
        try:
            self._fps_counter.reset()
        except Exception:
            pass
        self._last_step = step
        # Re-select interaction controller (family may change if different env loaded later)
        try:
            family = ENVIRONMENT_FAMILY_BY_GAME.get(self._game_id)
        except Exception:
            family = None
        self._interaction = self._create_interaction_controller(family)
        self._passive_action = self._resolve_passive_action()
        self._pending_input_label = "environment_reset"
        self._record_step(step, action=None, input_source=self._pending_input_label)
        self._update_idle_timer()

    # Jumanji arcade-style games that need continuous simulation (ghosts move, blocks fall)
    _JUMANJI_ARCADE_GAMES = frozenset({
        GameId.JUMANJI_PACMAN,         # Ghosts chase player
        GameId.JUMANJI_SNAKE,          # Snake auto-moves
        GameId.JUMANJI_TETRIS,         # Blocks fall automatically
    })

    def _create_interaction_controller(
        self, family: EnvironmentFamily | None
    ) -> InteractionController:
        """Return the interaction controller best suited for the env family."""

        if family in (EnvironmentFamily.BOX2D, EnvironmentFamily.MUJOCO):
            return Box2DInteractionController(self, target_hz=50)
        if family in (EnvironmentFamily.ATARI, EnvironmentFamily.ALE):
            return AleInteractionController(self, target_hz=60)
        if family == EnvironmentFamily.VIZDOOM:
            # ViZDoom runs at 35 FPS by default - game advances continuously
            # like ALE, enemies move and projectiles fly even without player input
            return ViZDoomInteractionController(self, target_hz=35)
        if family == EnvironmentFamily.PROCGEN:
            # Procgen docs suggest 15 Hz, but 30 FPS feels more responsive in a GUI
            return ProcgenInteractionController(self, target_hz=30)
        # Check for Jumanji arcade-style games (PacMan, Snake, Tetris)
        if family == EnvironmentFamily.JUMANJI and self._game_id in self._JUMANJI_ARCADE_GAMES:
            # 10 FPS for arcade feel - ghosts/snake/blocks move automatically
            return JumanjiArcadeInteractionController(self, target_hz=10)
        if family in (EnvironmentFamily.SMAC, EnvironmentFamily.SMACV2):
            # SMAC is agent-only: auto-step with random valid actions at 5 FPS
            return SMACInteractionController(self, target_hz=5)
        if family == EnvironmentFamily.MALMOENV:
            # MalmoEnv runs against a live Minecraft client; use 60Hz for smooth rendering
            # while input is handled natively via side-channel.
            return MalmoInteractionController(self, target_hz=60)
        if family == EnvironmentFamily.GRIDDLY:
            # Check if this is a continuous Griddly game (moving enemies, spawning objects)
            if self._game_id in CONTINUOUS_GRIDDLY_GAMES:
                # Continuous games need idle ticking with NOOP actions at 30 FPS
                return GriddlyInteractionController(self, target_hz=30)
            else:
                # Turn-based puzzle games don't need idle ticking
                return TurnBasedInteractionController()
        return TurnBasedInteractionController()

    def perform_human_action(self, action: int | list[int], *, key_label: str | None = None) -> None:
        if self._adapter is None:
            return
        if not self._game_started:
            self.status_message.emit("Click 'Start Game' before taking actions")
            return
        if self._game_paused:
            self.status_message.emit("Game is paused. Click 'Continue' to resume.")
            return
        if self._control_mode not in {
            ControlMode.HUMAN_ONLY,
            ControlMode.HYBRID_TURN_BASED,
            ControlMode.HYBRID_HUMAN_AGENT,
        }:
            self.status_message.emit("Current mode ignores human input")
            return
        if self._control_mode == ControlMode.HYBRID_TURN_BASED and self._turn != "human":
            self.status_message.emit("Hybrid mode: waiting for agent turn")
            return
        self._awaiting_human = False
        self.awaiting_human.emit(False, "")
        self._pending_input_label = key_label or "human"
        _LOGGER.debug(
            "Human action received label='%s' action=%s", self._pending_input_label, action
        )
        self._apply_action(action)

    def handle_native_mouse(self, dx: int, dy: int) -> bool:
        """Handle a raw mouse delta event for native input (bypassing the step loop).

        Returns:
            True if the mouse event was handled and should not be processed further.
        """
        interaction = getattr(self, "_interaction", None)
        if interaction and hasattr(interaction, "handle_native_mouse"):
            return interaction.handle_native_mouse(dx, dy)
        return False

    def toggle_auto_play(self, *, enabled: bool) -> None:
        if enabled:
            self.start_auto_play()
        else:
            self.stop_auto_play()

    def start_auto_play(self) -> None:
        if self._adapter is None:
            return
        if not self._game_started:
            self.status_message.emit("Click 'Start Game' before auto-play")
            return
        if self._game_paused:
            self.status_message.emit("Game is paused. Click 'Continue' to resume.")
            return
        if self._auto_timer.isActive():
            return
        self._auto_timer.start()
        self.auto_play_state_changed.emit(True)
        self.status_message.emit("Auto-play started")

    def stop_auto_play(self) -> None:
        if self._auto_timer.isActive():
            self._auto_timer.stop()
            self.auto_play_state_changed.emit(False)
            self.status_message.emit("Auto-play paused")

    def perform_agent_step(self) -> None:
        if self._adapter is None:
            return
        if not self._game_started:
            self.status_message.emit("Click 'Start Game' before agent step")
            return
        if self._game_paused:
            self.status_message.emit("Game is paused. Click 'Continue' to resume.")
            return
        # SMAC: simultaneous multi-agent stepping -- generate actions for all agents
        interaction = getattr(self, "_interaction", None)
        if isinstance(interaction, SMACInteractionController):
            action = interaction.maybe_passive_action()
            if action is not None:
                self._pending_input_label = "agent"
                self._apply_action(action)
            return
        # Get active agent from last step for multi-agent environments
        active_agent = self._get_active_agent()
        action = self._select_agent_action(agent_id=active_agent)
        if action is None:
            self._awaiting_human = True
            self.awaiting_human.emit(True, "Awaiting human action")
            return
        self._pending_input_label = "agent"
        self._apply_action(action)

    def shutdown(self) -> None:
        self.stop_auto_play()
        self._dispose_adapter()

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    @staticmethod
    def supported_control_modes(game_id: GameId) -> tuple[ControlMode, ...]:
        adapter_cls = get_adapter_cls(game_id)
        return adapter_cls.supported_control_modes

    @property
    def action_space(self) -> Any | None:
        if self._adapter is None:
            return None
        return self._adapter.action_space

    @property
    def game_id(self) -> GameId | None:
        return self._game_id

    @property
    def next_seed(self) -> int:
        return self._next_seed

    @property
    def timers(self) -> SessionTimers:
        return self._timers

    @property
    def current_episode_reward(self) -> float:
        """Return the cumulative reward for the active episode."""
        return float(self._episode_reward)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _dispose_adapter(self) -> None:
        if self._episode_active and self._last_step is not None:
            self._finalize_episode(self._last_step, aborted=True, timestamp=datetime.utcnow())
        if self._adapter is not None:
            try:
                self._adapter.close()
            finally:
                self._adapter = None
                self._game_id = None
                self._step_index = -1
        self._episode_active = False
        self._episode_id = None
        self._stop_idle_tick()
        self._passive_action = None
        if self._telemetry is not None:
            self._telemetry.reset()

    def _auto_step(self) -> None:
        if self._adapter is None:
            self.stop_auto_play()
            return
        if self._game_paused:
            # Auto-play should be stopped when paused, but add safety check
            self.stop_auto_play()
            return
        # SMAC: simultaneous multi-agent stepping -- generate actions for all agents
        interaction = getattr(self, "_interaction", None)
        if isinstance(interaction, SMACInteractionController):
            action = interaction.maybe_passive_action()
            if action is not None:
                self._pending_input_label = "auto_play"
                self._apply_action(action)
            return
        # Get active agent from last step for multi-agent environments
        active_agent = self._get_active_agent()
        action = self._select_agent_action(agent_id=active_agent)
        if action is None:
            # Waiting for human input in human or hybrid modes
            self.stop_auto_play()
            self._awaiting_human = True
            self.awaiting_human.emit(True, "Awaiting human action")
            return
        self._pending_input_label = "auto_play"
        self._apply_action(action)

    def _select_agent_action(self, agent_id: str | None = None) -> int | None:
        """Select action for an agent.

        For multi-agent environments, uses PolicyMappingService for per-agent
        policy selection. For single-agent, falls back to ActorService.

        Args:
            agent_id: The agent ID for multi-agent environments (from step.state.active_agent).
                     If None, uses legacy single-agent behavior.

        Returns:
            The action to take, or None to abstain.
        """
        if self._adapter is None:
            return None

        if self._control_mode == ControlMode.HUMAN_ONLY:
            return None

        if self._control_mode == ControlMode.HYBRID_TURN_BASED and self._turn != "agent":
            return None

        if self._last_step is not None:
            snapshot = self._build_step_snapshot(self._last_step)
            try:
                # Use PolicyMappingService for multi-agent, ActorService for single-agent
                if self._policy_mapping is not None and agent_id is not None:
                    selected = self._policy_mapping.select_action(agent_id, snapshot)
                elif self._actor_service is not None:
                    selected = self._actor_service.select_action(snapshot)
                else:
                    selected = None
            except Exception as exc:  # pragma: no cover - defensive
                self.log_constant(
                    LOG_SESSION_STEP_ERROR,
                    exc_info=exc,
                    extra={
                        "stage": "actor_select",
                        "agent_id": agent_id,
                        "game_id": self._game_id.value if self._game_id else "unknown",
                    },
                )
            else:
                if selected is not None:
                    return selected

        try:
            space = self._adapter.action_space
            if hasattr(space, "sample"):
                action = space.sample()
                return int(action) if isinstance(action, (bool, int)) else action
        except Exception as exc:  # pragma: no cover - defensive
            self.log_constant(
                LOG_SESSION_STEP_ERROR,
                exc_info=exc,
                extra={
                    "stage": "sample_action",
                    "game_id": self._game_id.value if self._game_id else "unknown",
                },
            )
            self.error_occurred.emit(str(exc))
        return None

    def _apply_action(self, action: int | list[int]) -> None:
        if self._adapter is None:
            return
        try:
            step = self._adapter.step(action)
        except Exception as exc:  # pragma: no cover - surfaced in UI
            self.log_constant(
                LOG_SESSION_STEP_ERROR,
                exc_info=exc,
                extra={
                    "stage": "adapter_step",
                    "action": action,
                    "game_id": self._game_id.value if self._game_id else "unknown",
                },
            )
            self.error_occurred.emit(str(exc))
            self.stop_auto_play()
            self._pending_input_label = None
            return

        self._step_index += 1
        self._last_step = step
        step_timestamp = datetime.utcnow()
        source_label = self._pending_input_label or "unknown"

        # For recording, use first action from list or the single action
        record_action: int | None
        if isinstance(action, list):
            record_action = action[0] if action else 0
        else:
            record_action = action

        self._record_step(
            step,
            action=record_action,
            timestamp=step_timestamp,
            input_source=source_label,
        )
        self._timers.mark_first_move(when=step_timestamp)
        self.step_processed.emit(step, self._step_index)
        # Update dynamic FPS meter based on actual step cadence
        try:
            fps = self._fps_counter.tick(time.monotonic())
            self.fps_updated.emit(fps)
        except Exception:
            pass
        self._update_status(step)
        self._log_step_outcome(record_action, step)

        # Clear the label so future steps require explicit attribution
        self._pending_input_label = None

        finished = bool(step.terminated or step.truncated)
        if finished:
            self._timers.mark_outcome(when=step_timestamp)
            self._finalize_episode(step, timestamp=step_timestamp)
            self.episode_finished.emit(True)
            self.stop_auto_play()
            self._awaiting_human = False
            self.awaiting_human.emit(False, "Episode finished")
            self._stop_idle_tick()
        else:
            self._advance_turn()
            if self._control_mode == ControlMode.HUMAN_ONLY:
                self._awaiting_human = True
                self.awaiting_human.emit(True, "Awaiting human action")
            self._update_idle_timer()

    def _advance_turn(self) -> None:
        if self._control_mode != ControlMode.HYBRID_TURN_BASED:
            return
        self._turn = "agent" if self._turn == "human" else "human"
        self.turn_changed.emit(self._turn)
        if self._turn == "human":
            self._awaiting_human = True
            self.awaiting_human.emit(True, "Awaiting human action")
        else:
            self._awaiting_human = False
            self.awaiting_human.emit(False, "Agent turn")
            # If auto-play is active, immediately allow agent to act on next timeout
            if not self._auto_timer.isActive():
                self.start_auto_play()

    def set_run_context(
        self,
        run_id: str,
        max_episodes: int | None = None,
        db_conn: Any | None = None,
        *,
        worker_id: str | None = None,
    ) -> None:
        """Set the run context for this session (e.g., when launched from trainer).

        Args:
            run_id: ULID or unique run identifier
            max_episodes: Max episodes per run (default: 1M)
            db_conn: SQLite connection for counter persistence (optional for human mode)
        """
        if db_conn is not None:
            self._run_counter_manager = RunCounterManager(
                db_conn,
                run_id,
                max_episodes=max_episodes,
                worker_id=worker_id,
            )
            try:
                self._run_counter_manager.initialize()
            except Exception as exc:
                _LOGGER.error(f"Failed to initialize RunCounterManager: {exc}")
                self._run_counter_manager = None
        else:
            _LOGGER.debug(f"No DB connection provided; counter will start at 0 for run {run_id}")
            self._run_counter_manager = RunCounterManager(
                None,
                run_id,
                max_episodes=max_episodes,
                worker_id=worker_id,
            )
            self._run_counter_manager._current_index = -1

    def _begin_episode(self, game_id: GameId | None, control_mode: ControlMode) -> None:
        """Begin a new episode with counter-based episode ID.

        Episode ID format:
        - With run context: f"{run_id}-ep{ep_index:06d}"
        - Without run context: f"{game_id.value}-ep{ep_index:06d}"
        """
        if game_id is None:
            return
        if self._episode_active and self._last_step is not None:
            self._finalize_episode(self._last_step, aborted=True, timestamp=datetime.utcnow())

        state_snapshot = self._seed_manager.capture_state()
        self._last_seed_state = state_snapshot

        # Determine episode index and ID
        if self._run_counter_manager is not None:
            try:
                # Use RunCounterManager for bounded, ordered episodes
                with self._run_counter_manager.next_episode() as ep_index:
                    self._episode_id = format_episode_id(
                        self._run_counter_manager._run_id,
                        ep_index,
                        self._run_counter_manager._worker_id,
                    )
                    ep_index_for_metadata = ep_index
            except RuntimeError as exc:
                _LOGGER.error(f"RunCounterManager error: {exc}")
                self.error_occurred.emit(str(exc))
                return
        else:
            # Fallback for human-input mode: use simple counter without SHA256
            # Counter starts at 0 on each load
            ep_index_for_metadata = self._step_index  # Use step index as episode marker
            self._episode_id = format_episode_id(game_id.value, ep_index_for_metadata)

        self._episode_reward = 0.0
        run_context = self._run_counter_manager
        run_id = getattr(run_context, "_run_id", None) if run_context is not None else None
        worker_id = getattr(run_context, "_worker_id", None) if run_context is not None else None
        self._episode_metadata = {
            "game_id": game_id.value,
            "control_mode": control_mode.value,
            "seed": self._current_seed,
            "episode_index": ep_index_for_metadata,
            "rng_state": state_snapshot,
            "run_id": run_id,
            "worker_id": worker_id,
        }
        game_config_snapshot = self._game_config_snapshot()
        if game_config_snapshot is not None:
            self._episode_metadata["game_config"] = game_config_snapshot
        self._episode_active = True
        self._last_step = None
        if self._telemetry is not None:
            self._telemetry.reset()

    def _record_step(
        self,
        step: AdapterStep,
        *,
        action: int | None,
        timestamp: datetime | None = None,
        input_source: str | None = None,
    ) -> StepRecord | None:
        if not self._episode_active or self._episode_id is None:
            return None
        snapshot = self._build_step_snapshot(step)
        self._episode_reward += step.reward

        # Notify the appropriate policy of the step
        # Use PolicyMappingService for multi-agent, ActorService for single-agent
        agent_id = getattr(step.state, "active_agent", None)
        if action is not None:
            try:
                if self._policy_mapping is not None and agent_id is not None:
                    self._policy_mapping.notify_step(agent_id, snapshot)
                elif self._actor_service is not None:
                    self._actor_service.notify_step(snapshot)
            except Exception as exc:  # pragma: no cover - defensive safeguard
                self.log_constant(
                    LOG_SESSION_STEP_ERROR,
                    exc_info=exc,
                    extra={
                        "stage": "actor_notify_step",
                        "agent_id": agent_id,
                        "game_id": self._game_id.value if self._game_id else "unknown",
                    },
                )
        info_payload = dict(snapshot.info)
        if input_source is not None:
            info_payload.setdefault("input_source", input_source)
        game_config_snapshot = self._game_config_snapshot()
        if game_config_snapshot is not None:
            info_payload.setdefault("game_config", game_config_snapshot)

        adapter = self._adapter
        space_signature: Mapping[str, Any] | None = None
        vector_metadata: Mapping[str, Any] | None = None
        time_step_value: int | None = None
        raw_episode_step = cast(Any, snapshot.info.get("episode_step"))
        if raw_episode_step is not None:
            try:
                time_step_value = int(raw_episode_step)
            except (TypeError, ValueError, OverflowError):  # pragma: no cover - defensive
                time_step_value = None
        if adapter is not None:
            raw_signature = adapter.space_signature
            if raw_signature:
                space_signature = dict(raw_signature)
            raw_vector_metadata = adapter.vector_metadata
            if raw_vector_metadata:
                vector_metadata = dict(raw_vector_metadata)
            if time_step_value is None:
                time_step_value = adapter.elapsed_steps()

        step_vector_details = extract_vector_step_details(step.info)
        combined_vector_metadata: Mapping[str, Any] | None = vector_metadata
        if step_vector_details:
            normalized_details: dict[str, Any] = {}
            for key, value in step_vector_details.items():
                if hasattr(value, "tolist"):
                    try:
                        normalized_details[key] = value.tolist()
                        continue
                    except Exception:  # pragma: no cover - defensive conversion
                        pass
                normalized_details[key] = value
            if combined_vector_metadata is not None:
                merged = dict(combined_vector_metadata)
                merged.update(normalized_details)
                combined_vector_metadata = merged
            else:
                combined_vector_metadata = normalized_details

        if time_step_value is None:
            time_step_value = self._step_index

        if space_signature is not None:
            info_payload.setdefault(TELEMETRY_KEY_SPACE_SIGNATURE, space_signature)
        if combined_vector_metadata is not None:
            info_payload.setdefault(TELEMETRY_KEY_VECTOR_METADATA, combined_vector_metadata)
            autoreset_value = combined_vector_metadata.get(TELEMETRY_KEY_AUTORESET_MODE)
            if autoreset_value is not None:
                info_payload.setdefault(TELEMETRY_KEY_AUTORESET_MODE, autoreset_value)
        if time_step_value is not None:
            info_payload.setdefault(TELEMETRY_KEY_TIME_STEP, time_step_value)

        schema_key = None
        if self._game_id is not None:
            schema_key = self._game_id.value
        schema = schema_registry.get(schema_key) or schema_registry.get("default")
        if schema is not None:
            missing_required = [
                field
                for field in schema.required_fields
                if field not in info_payload
            ]
            if missing_required:
                cache_key = f"schema_missing:{','.join(sorted(missing_required))}"
                if cache_key not in self._schema_alerts_emitted:
                    self._schema_alerts_emitted.add(cache_key)
                    self.log_constant(
                        LOG_SCHEMA_MISMATCH,
                        extra={
                            "game_id": schema_key or "unknown",
                            "missing": sorted(missing_required),
                            "schema_id": schema.schema_id,
                        },
                    )
            if space_signature is None:
                cache_key = "space_signature_missing"
                if cache_key not in self._schema_alerts_emitted:
                    self._schema_alerts_emitted.add(cache_key)
                    self.log_constant(
                        LOG_SPACE_DESCRIPTOR_MISSING,
                        extra={
                            "game_id": schema_key or "unknown",
                            "schema_id": schema.schema_id,
                        },
                    )
            if schema.vector_metadata is not None:
                if combined_vector_metadata is None:
                    cache_key = "vector_metadata_missing"
                    if cache_key not in self._schema_alerts_emitted:
                        self._schema_alerts_emitted.add(cache_key)
                        self.log_constant(
                            LOG_SCHEMA_MISMATCH,
                            extra={
                                "game_id": schema_key or "unknown",
                                "missing": [TELEMETRY_KEY_VECTOR_METADATA],
                                "schema_id": schema.schema_id,
                            },
                        )
                else:
                    vector_missing = [
                        key
                        for key in schema.vector_metadata.required_keys
                        if key not in combined_vector_metadata
                    ]
                    if vector_missing:
                        cache_key = f"vector_missing:{','.join(sorted(vector_missing))}"
                        if cache_key not in self._schema_alerts_emitted:
                            self._schema_alerts_emitted.add(cache_key)
                            self.log_constant(
                                LOG_SCHEMA_MISMATCH,
                                extra={
                                    "game_id": schema_key or "unknown",
                                    "missing": sorted(vector_missing),
                                    "schema_id": schema.schema_id,
                                },
                            )
                    autoreset_value = combined_vector_metadata.get(TELEMETRY_KEY_AUTORESET_MODE)
                    if (
                        autoreset_value
                        and autoreset_value not in SUPPORTED_AUTORESET_MODES
                    ):
                        cache_key = f"autoreset:{autoreset_value}"
                        if cache_key not in self._schema_alerts_emitted:
                            self._schema_alerts_emitted.add(cache_key)
                            self.log_constant(
                                LOG_VECTOR_AUTORESET_MODE,
                                extra={
                                    "autoreset_mode": autoreset_value,
                                    "supported": sorted(SUPPORTED_AUTORESET_MODES),
                                },
                            )

        if (
            combined_vector_metadata
            and combined_vector_metadata.get("vectorized")
            and "normalization_stats" not in info_payload
        ):
            cache_key = "normalization_stats_missing"
            if cache_key not in self._schema_alerts_emitted:
                self._schema_alerts_emitted.add(cache_key)
                self.log_constant(
                    LOG_NORMALIZATION_STATS_DROPPED,
                    extra={
                        "game_id": schema_key or "unknown",
                        "reason": "normalization stats missing in vector payload",
                    },
                )

        # Ensure telemetry StepRecord mirrors the augmented snapshot information.
        agent_id = step.agent_id or getattr(step.state, "active_agent", None)
        if agent_id is None and self._actor_service is not None:
            agent_id = self._actor_service.get_active_actor_id()
        render_hint = self._coalesce_render_hint(step)
        frame_ref = step.frame_ref or self._extract_frame_reference(step)
        payload_version = step.payload_version if step.payload_version is not None else 0

        run_context = self._run_counter_manager
        run_id = getattr(run_context, "_run_id", None) if run_context is not None else None
        worker_id = getattr(run_context, "_worker_id", None) if run_context is not None else None

        record = StepRecord(
            episode_id=self._episode_id,
            step_index=self._step_index,
            action=action,
            observation=step.observation,
            reward=step.reward,
            terminated=step.terminated,
            truncated=step.truncated,
            info=info_payload,
            timestamp=timestamp or datetime.utcnow(),
            render_payload=step.render_payload,
            agent_id=agent_id,
            render_hint=render_hint,
            frame_ref=frame_ref,
            payload_version=payload_version,
            run_id=run_id,
            worker_id=worker_id,
            time_step=time_step_value,
            space_signature=space_signature,
            vector_metadata=combined_vector_metadata,
        )
        if self._telemetry is not None and self._slow_lane_enabled:
            self._telemetry.record_step(record)
        return record

    def _should_capture_frames(self) -> bool:
        if self._storage_service is None:
            return False
        return self._storage_service.capture_frames_enabled()

    def _idle_step(self) -> None:
        if not self._should_idle_tick():
            self._stop_idle_tick()
            return
        interaction = getattr(self, "_interaction", None)
        # For ALE, ViZDoom, Procgen, Jumanji arcade, Griddly, and SMAC games, do not gate on awaiting_human
        # These are continuous games where the world should keep moving regardless of player input
        require_awaiting = not isinstance(interaction, (AleInteractionController, ViZDoomInteractionController, ProcgenInteractionController, JumanjiArcadeInteractionController, GriddlyInteractionController, SMACInteractionController))
        if require_awaiting and not self._awaiting_human:
            return

        # Determine idle action - prioritize input controller's key state
        action = None
        input_label = "idle_tick"

        # Check if the input controller has keys pressed (state-based input)
        if self._input_controller is not None:
            # Check if using state-based input and has keys pressed
            if hasattr(self._input_controller, 'is_state_based') and self._input_controller.is_state_based():
                if hasattr(self._input_controller, 'get_current_action'):
                    key_action = self._input_controller.get_current_action()
                    if key_action is not None:
                        action = key_action
                        input_label = "key_combo"

        # Fall back to passive action if no key action
        if action is None:
            if interaction is not None:
                action = interaction.maybe_passive_action()
            if action is None:
                action = self._passive_action

        if action is None:
            _LOGGER.debug("Idle timer active but passive action unavailable")
            self._stop_idle_tick()
            return
        if require_awaiting:
            self._awaiting_human = False
            self.awaiting_human.emit(False, "Passive idle step")
        self._pending_input_label = input_label
        self._apply_action(action)

    def _determine_idle_interval(self) -> int:
        if self._user_idle_interval is not None:
            return self._user_idle_interval
        interaction = getattr(self, "_interaction", None)
        if interaction is not None:
            interval = interaction.idle_interval_ms()
            if interval is not None:
                return interval
        family = None
        if self._game_id is not None:
            try:
                family = ENVIRONMENT_FAMILY_BY_GAME.get(self._game_id)
            except Exception:
                family = None
        if family in (EnvironmentFamily.ATARI, EnvironmentFamily.ALE):
            return 16  # ~60 FPS fallback for Atari/ALE
        return self._compute_idle_interval()

    def _set_idle_interval(self, interval: int) -> None:
        clamped = max(10, int(interval))
        if clamped != self._current_idle_interval:
            self._idle_timer.setInterval(clamped)
            self._current_idle_interval = clamped

    def _compute_idle_interval(self) -> int:
        adapter = self._adapter
        if adapter is None:
            return max(30, self._current_idle_interval)

        env = getattr(adapter, "_env", None)
        metadata = getattr(env, "metadata", None) if env is not None else None
        if isinstance(metadata, dict):
            fps_raw = metadata.get("render_fps") or metadata.get("video.frames_per_second")
            fps_value = 0.0
            if fps_raw is not None:
                try:
                    fps_value = float(fps_raw)
                except (TypeError, ValueError):
                    fps_value = 0.0
            if fps_value > 0:
                return max(30, int(1000 / fps_value))

        try:
            space = adapter.action_space
        except Exception:
            return 150

        if isinstance(space, spaces.Box):
            return 80
        if isinstance(space, (spaces.MultiBinary, spaces.MultiDiscrete, spaces.Discrete)):
            return 120
        if isinstance(space, spaces.Tuple):
            return 120
        return 180

    def _should_idle_tick(self) -> bool:
        if self._adapter is None or self._game_id is None:
            return False
        if not self._game_started or self._game_paused:
            return False
        if self._last_step is not None and (self._last_step.terminated or self._last_step.truncated):
            return False
        # Delegate finer gating to the active interaction controller
        interaction = getattr(self, "_interaction", None)
        if interaction is not None:
            return interaction.should_idle_tick()
        return False

    def _start_idle_tick(self) -> None:
        if not self._idle_timer.isActive():
            self._idle_timer.start()

    def _stop_idle_tick(self) -> None:
        if self._idle_timer.isActive():
            self._idle_timer.stop()

    def _update_idle_timer(self) -> None:
        desired_interval = self._determine_idle_interval()
        self._set_idle_interval(desired_interval)
        if self._should_idle_tick():
            self._start_idle_tick()
        else:
            self._stop_idle_tick()

    def _resolve_passive_action(self) -> Any | None:
        if self._adapter is None or self._game_id is None:
            return None
        # Delegate passive action decision to interaction controller if present
        interaction = getattr(self, "_interaction", None)
        if interaction is not None:
            # Interaction controller uses _passive_action after resolution; fall through for Box2D parity
            pass
        fam = ENVIRONMENT_FAMILY_BY_GAME.get(self._game_id)
        if fam not in (EnvironmentFamily.BOX2D, EnvironmentFamily.ATARI, EnvironmentFamily.ALE, EnvironmentFamily.GRIDDLY):
            return None
        space = self._adapter.action_space
        if isinstance(space, spaces.Discrete):
            return 0
        if isinstance(space, spaces.MultiDiscrete):
            return np.zeros_like(space.nvec)
        if isinstance(space, spaces.MultiBinary):
            return np.zeros(space.n, dtype=int)
        if isinstance(space, spaces.Box):
            dtype = space.dtype if getattr(space, "dtype", None) is not None else np.float32
            return np.zeros(space.shape, dtype=dtype)
        if isinstance(space, spaces.Tuple):
            return tuple(self._resolve_passive_component(subspace) for subspace in space.spaces)
        return None

    def _resolve_passive_component(self, space: spaces.Space) -> Any:
        if isinstance(space, spaces.Discrete):
            return 0
        if isinstance(space, spaces.MultiDiscrete):
            return np.zeros_like(space.nvec)
        if isinstance(space, spaces.MultiBinary):
            return np.zeros(space.n, dtype=int)
        if isinstance(space, spaces.Box):
            dtype = space.dtype if getattr(space, "dtype", None) is not None else np.float32
            return np.zeros(space.shape, dtype=dtype)
        if isinstance(space, spaces.Tuple):
            return tuple(self._resolve_passive_component(sub) for sub in space.spaces)
        return None

    def _get_active_agent(self) -> str | None:
        """Get the active agent ID from the last step.

        For PettingZoo AEC environments, this returns the agent whose turn it is.
        For single-agent environments, returns None.

        Returns:
            The active agent ID, or None for single-agent environments.
        """
        if self._last_step is None:
            return None
        # PettingZoo AEC environments provide active_agent in state
        return getattr(self._last_step.state, "active_agent", None)

    def _build_step_snapshot(self, step: AdapterStep) -> StepSnapshot:
        info_payload = dict(step.info)
        if self._current_seed is not None:
            info_payload.setdefault("seed", self._current_seed)
        if self._last_seed_state is not None and self._step_index == 0:
            info_payload.setdefault("rng_state", self._last_seed_state)
        return StepSnapshot(
            step_index=self._step_index,
            observation=step.observation,
            reward=step.reward,
            terminated=step.terminated,
            truncated=step.truncated,
            seed=self._current_seed,
            info=info_payload,
        )

    def _json_safe(self, value: Any) -> Any:
        """Coerce nested telemetry values into JSON-serialisable structures."""
        if isinstance(value, Mapping):
            return {key: self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._json_safe(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _game_config_snapshot(self) -> dict[str, Any] | None:
        """Return the active game configuration as a JSON-safe mapping."""
        config = self._game_config
        if config is None:
            return None
        if is_dataclass(config):
            snapshot: Any = asdict(config)
        elif isinstance(config, Mapping):
            snapshot = dict(config)
        else:
            snapshot = None
        if not snapshot:
            return None
        safe_snapshot = self._json_safe(snapshot)
        return safe_snapshot if isinstance(safe_snapshot, dict) else None

    def _coalesce_render_hint(self, step: AdapterStep) -> dict[str, Any] | None:
        if step.render_hint:
            return dict(step.render_hint)
        state = step.state
        hint: dict[str, Any] = {}
        if state.active_agent:
            hint["active_agent"] = state.active_agent
        if state.metrics:
            hint["metrics"] = dict(state.metrics)
        if state.environment:
            hint["environment"] = dict(state.environment)
        if state.inventory:
            hint["inventory"] = dict(state.inventory)
        return hint or None

    def _extract_frame_reference(self, step: AdapterStep) -> str | None:
        # First check if frame_ref is already in state or payload
        state_raw = getattr(step.state, "raw", None)
        if isinstance(state_raw, Mapping):
            ref = state_raw.get("frame_ref")
            if isinstance(ref, str):
                return ref
        payload = step.render_payload
        if isinstance(payload, Mapping):
            ref = payload.get("frame_ref")
            if isinstance(ref, str):
                return ref

        # If not found, ask the adapter to generate one
        if self._adapter is not None:
            try:
                return self._adapter.build_frame_reference(step.render_payload, step.state)
            except Exception as e:
                _LOGGER.debug(f"Failed to build frame reference: {e}")

        return None

    def _finalize_episode(
        self,
        step: AdapterStep,
        *,
        aborted: bool = False,
        timestamp: datetime | None = None,
    ) -> None:
        if not self._episode_active or self._episode_id is None:
            return
        # Get episode_index from metadata (set in _begin_episode)
        episode_index = self._episode_metadata.get("episode_index", 0)
        summary = EpisodeSummary(
            episode_index=episode_index,
            total_reward=self._episode_reward,
            steps=self._step_index + 1,
            metadata=dict(self._episode_metadata),
        )

        # Notify all agents of episode end
        # For multi-agent, use PolicyMappingService.notify_all_episode_end
        # For single-agent, use ActorService.notify_episode_end
        agent_id = getattr(step.state, "active_agent", None)
        try:
            if self._policy_mapping is not None and self._policy_mapping.is_multi_agent():
                # Create summary for each agent
                agent_summaries = {
                    aid: EpisodeSummary(
                        episode_index=episode_index,
                        total_reward=self._episode_reward,  # Note: per-agent rewards not yet tracked
                        steps=self._step_index + 1,
                        metadata=dict(self._episode_metadata),
                    )
                    for aid in self._policy_mapping.agent_ids
                }
                self._policy_mapping.notify_all_episode_end(agent_summaries)
            elif self._actor_service is not None:
                self._actor_service.notify_episode_end(summary)
        except Exception as exc:  # pragma: no cover - defensive safeguard
            self.log_constant(
                LOG_SESSION_EPISODE_ERROR,
                exc_info=exc,
                extra={
                    "stage": "actor_notify_episode_end",
                    "agent_id": agent_id,
                    "game_id": summary.metadata.get("game_id")
                    or (self._game_id.value if self._game_id else "unknown"),
                    "episode_id": self._episode_id,
                },
            )
        if self._telemetry is not None:
            run_context = self._run_counter_manager
            run_id = getattr(run_context, "_run_id", None) if run_context is not None else None
            worker_id = getattr(run_context, "_worker_id", None) if run_context is not None else None
            rollup = EpisodeRollup(
                episode_id=self._episode_id,
                total_reward=self._episode_reward,
                steps=self._step_index + 1,
                terminated=step.terminated if not aborted else False,
                truncated=step.truncated if not aborted else True,
                metadata=dict(self._episode_metadata),
                timestamp=timestamp or datetime.utcnow(),
                agent_id=getattr(step.state, "active_agent", None),
                game_id=self._game_id.value if self._game_id else None,
                run_id=run_id,
                worker_id=worker_id,
            )
            self._telemetry.complete_episode(rollup)
        self._episode_active = False
        self._episode_id = None

        if self._current_seed is not None:
            self._register_seed(self._current_seed, advance_next=True, notify=False)

    def _activate_seed(self, seed: int | None, *, advance_next: bool, notify: bool = True) -> int:
        resolved = self._resolve_seed(seed)
        self._seed_manager.apply(resolved)
        self._last_seed_state = self._seed_manager.capture_state()
        self._register_seed(resolved, advance_next=advance_next, notify=notify)
        return resolved

    def _resolve_seed(self, seed: int | None) -> int:
        if seed is None or seed <= 0:
            candidate = self._next_seed
        else:
            candidate = seed
        if self._allow_seed_reuse:
            return candidate
        if candidate <= self._last_seed:
            if seed is not None and seed <= self._last_seed:
                self.status_message.emit(
                    f"Seed {seed} already used. Advancing to {self._last_seed + 1}."
                )
            candidate = self._last_seed + 1
        if candidate < self._next_seed:
            candidate = self._next_seed
        return candidate

    def _register_seed(self, seed: int, *, advance_next: bool = True, notify: bool = True) -> None:
        self._last_seed = max(seed, self._last_seed)
        self._current_seed = seed
        if advance_next:
            self._next_seed = self._last_seed + 1
        else:
            self._next_seed = max(self._next_seed, seed)
        if notify:
            self.seed_applied.emit(seed)

    def _seed_timers(self, seed: int) -> None:
        del seed
        auto_active = self._auto_timer.isActive()
        idle_active = self._idle_timer.isActive()
        self._auto_timer.stop()
        self._idle_timer.stop()
        if auto_active:
            self._auto_timer.start(self._auto_timer.interval())
        if idle_active:
            self._idle_timer.start(self._idle_timer.interval())

    def _extract_agent_position(self, step: AdapterStep) -> tuple[int, int] | None:
        payload = getattr(step, "render_payload", None)
        if isinstance(payload, dict):
            position = payload.get("agent_position")
            if isinstance(position, tuple) and len(position) == 2:
                return position  # type: ignore[return-value]
        return None

    def _log_step_outcome(self, action: int | list[int] | None, step: AdapterStep) -> None:
        input_label = self._pending_input_label or "system"
        self._pending_input_label = None

        prev_position = self._last_agent_position
        new_position = self._extract_agent_position(step)
        self._last_agent_position = new_position

        delta: tuple[int, int] | None = None
        if prev_position is not None and new_position is not None:
            delta = (new_position[0] - prev_position[0], new_position[1] - prev_position[1])

        # Build configuration context string
        config_info = self._build_config_context()

        # Detect slippage for discrete action games (FrozenLake, CliffWalking)
        slippage_info = self._detect_slippage(action, delta)

        # Construct log message with configuration and slippage context
        log_parts = [
            f"Step {self._step_index}",
            f"via input='{input_label}'",
            f"action={action}",
            f"reward={step.reward:.2f}",
            f"terminated={step.terminated}",
            f"truncated={step.truncated}",
            f"position={new_position}",
            f"delta={delta}",
        ]

        if config_info:
            log_parts.append(config_info)

        if slippage_info:
            log_parts.append(slippage_info)

        _LOGGER.debug(" ".join(log_parts))

    def _build_config_context(self) -> str:
        """Build a configuration context string for logging."""
        if self._game_config is None or self._game_id is None:
            return ""

        config_parts = []

        if isinstance(self._game_config, FrozenLakeConfig):
            config_parts.append(f"is_slippery={self._game_config.is_slippery}")
        elif isinstance(self._game_config, CliffWalkingConfig):
            config_parts.append(f"is_slippery={self._game_config.is_slippery}")
        elif isinstance(self._game_config, TaxiConfig):
            config_parts.append(f"is_raining={self._game_config.is_raining}")
            config_parts.append(f"fickle_passenger={self._game_config.fickle_passenger}")
        elif isinstance(self._game_config, LunarLanderConfig):
            config_parts.append(f"continuous={self._game_config.continuous}")
            config_parts.append(f"gravity={self._game_config.gravity:.1f}")
            config_parts.append(f"wind={self._game_config.enable_wind}")
        elif isinstance(self._game_config, CarRacingConfig):
            config_parts.append(f"continuous={self._game_config.continuous}")
            config_parts.append(f"domain_randomize={self._game_config.domain_randomize}")
        elif isinstance(self._game_config, BipedalWalkerConfig):
            config_parts.append(f"hardcore={self._game_config.hardcore}")

        if config_parts:
            return f"config=[{', '.join(config_parts)}]"
        return ""

    def _detect_slippage(self, action: int | list[int] | None, delta: tuple[int, int] | None) -> str:
        """Detect if movement differs from intended action (slippage)."""
        if delta is None or self._game_id is None:
            return ""

        # Only check for discrete action games with directional movement
        if self._game_id not in {GameId.FROZEN_LAKE, GameId.CLIFF_WALKING}:
            return ""

        # Skip slippage detection for multi-agent actions (list) or None
        if not isinstance(action, int):
            return ""

        # Map actions to expected deltas for toy-text games
        # 0=Left, 1=Down, 2=Right, 3=Up
        expected_deltas = {
            0: (0, -1),  # Left
            1: (1, 0),   # Down
            2: (0, 1),   # Right
            3: (-1, 0),  # Up
        }

        expected = expected_deltas.get(action)
        if expected is None:
            return ""

        if delta != expected:
            action_names = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}
            intended = action_names.get(action, f"action{action}")

            # Determine observed direction from delta
            observed_dir = "no_movement"
            if delta == (0, -1):
                observed_dir = "Left"
            elif delta == (1, 0):
                observed_dir = "Down"
            elif delta == (0, 1):
                observed_dir = "Right"
            elif delta == (-1, 0):
                observed_dir = "Up"
            elif delta == (0, 0):
                observed_dir = "stayed_in_place"

            return f"slippage_detected[intended={intended}, observed={observed_dir}]"

        return ""

    def _update_status(self, step: AdapterStep, *, prefix: str | None = None) -> None:
        # Avoid emitting routine per-step status messages to the global status bar.
        # Only emit when an explicit prefix is supplied (for notable events like
        # environment ready or environment reset). Emitting per-step details
        # causes noisy status text such as "reward=... terminated=... truncated=..."
        # to appear in the UI which is undesirable.
        if prefix is None:
            return

        message = (
            f"reward={step.reward:.2f} terminated={step.terminated} truncated={step.truncated}"
        )
        message = f"{prefix}: {message}"
        self.status_message.emit(message)


__all__ = ["SessionController", "SessionState"]
