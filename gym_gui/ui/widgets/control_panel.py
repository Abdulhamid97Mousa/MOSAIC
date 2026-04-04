from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import pyqtSignal  # type: ignore[attr-defined]

from gym_gui.config.game_configs import (
    DEFAULT_FROZEN_LAKE_V2_CONFIG,
    ALEConfig,
    BipedalWalkerConfig,
    CarRacingConfig,
    CliffWalkingConfig,
    FrozenLakeConfig,
    LunarLanderConfig,
    MeltingPotConfig,
    MiniGridConfig,
    MultiGridConfig,
    ProcgenConfig,
    RWAREConfig,
    SMACConfig,
    TaxiConfig,
)
from gym_gui.core.adapters.vizdoom import ViZDoomConfig
from gym_gui.core.enums import (
    ENVIRONMENT_FAMILY_BY_GAME,
    ControlMode,
    EnvironmentFamily,
    GameId,
    get_game_display_name,
)
from gym_gui.services.operator import OperatorConfig, OperatorDescriptor
from gym_gui.telemetry.semconv import (
    TELEMETRY_MODE_DESCRIPTORS,
    TelemetryModeDescriptor,
    TelemetryModes,
)
from gym_gui.ui.config_panels.multi_agent.meltingpot import (
    MELTINGPOT_GAME_IDS,
    build_meltingpot_controls,
)
from gym_gui.ui.config_panels.multi_agent.meltingpot import (
    ControlCallbacks as MeltingPotControlCallbacks,
)
from gym_gui.ui.config_panels.multi_agent.multigrid import (
    MULTIGRID_GAME_IDS,
    build_multigrid_controls,
)
from gym_gui.ui.config_panels.multi_agent.multigrid import (
    ControlCallbacks as MultiGridControlCallbacks,
)
from gym_gui.ui.config_panels.multi_agent.rware import (
    ALL_RWARE_GAME_IDS,
    build_rware_controls,
)
from gym_gui.ui.config_panels.multi_agent.smac import (
    ALL_SMAC_GAME_IDS,
    build_smac_controls,
)
from gym_gui.ui.config_panels.multi_agent.smac import (
    ControlCallbacks as SMACControlCallbacks,
)
from gym_gui.ui.config_panels.single_agent.ale import (
    ALE_GAME_IDS,
    build_ale_controls,
)
from gym_gui.ui.config_panels.single_agent.ale import (
    ControlCallbacks as ALEControlCallbacks,
)
from gym_gui.ui.config_panels.single_agent.gym import (
    build_bipedal_controls,
    build_car_racing_controls,
    build_cliff_controls,
    build_frozenlake_controls,
    build_frozenlake_v2_controls,
    build_lunarlander_controls,
    build_taxi_controls,
)
from gym_gui.ui.config_panels.single_agent.malmo import (
    DEFAULT_MALMO_CONFIG,
    MALMOENV_FAMILY,
    build_malmo_controls,
)
from gym_gui.ui.config_panels.single_agent.minigrid.config_panel import (
    MINIGRID_GAME_IDS,
    build_minigrid_controls,
)
from gym_gui.ui.config_panels.single_agent.minigrid.config_panel import (
    ControlCallbacks as MinigridControlCallbacks,
)
from gym_gui.ui.config_panels.single_agent.minigrid.config_panel import (
    resolve_default_config as resolve_minigrid_default_config,
)
from gym_gui.ui.config_panels.single_agent.procgen import (
    PROCGEN_GAME_IDS,
    build_procgen_controls,
)
from gym_gui.ui.config_panels.single_agent.procgen import (
    ControlCallbacks as ProcgenControlCallbacks,
)
from gym_gui.ui.config_panels.single_agent.vizdoom import (
    VIZDOOM_GAME_IDS,
    build_vizdoom_controls,
)
from gym_gui.ui.config_panels.single_agent.vizdoom import (
    ControlCallbacks as ViZDoomControlCallbacks,
)
from gym_gui.ui.widgets.godot_ge_tab import GodotGETab
from gym_gui.ui.widgets.keyboard_assignment_widget import KeyboardAssignmentWidget
from gym_gui.ui.widgets.mujoco_mpc_tab import MuJoCoMPCTab
from gym_gui.ui.widgets.multi_agent_tab import MultiAgentTab
from gym_gui.ui.widgets.operator_config_widget import OperatorConfigWidget
from gym_gui.ui.widgets.operators_tab import OperatorsTab
from gym_gui.ui.widgets.single_agent_tab import SingleAgentTab
from gym_gui.ui.worker_catalog import WorkerDefinition, get_worker_catalog


@dataclass(frozen=True)
class ControlPanelConfig:
    """Configuration for the control panel with game-specific configs."""

    available_modes: Dict[GameId, Iterable[ControlMode]]
    default_mode: ControlMode
    frozen_lake_config: FrozenLakeConfig
    taxi_config: TaxiConfig
    cliff_walking_config: CliffWalkingConfig
    lunar_lander_config: LunarLanderConfig
    car_racing_config: CarRacingConfig
    bipedal_walker_config: BipedalWalkerConfig
    minigrid_empty_config: MiniGridConfig
    minigrid_doorkey_5x5_config: MiniGridConfig
    minigrid_doorkey_6x6_config: MiniGridConfig
    minigrid_doorkey_8x8_config: MiniGridConfig
    minigrid_doorkey_16x16_config: MiniGridConfig
    minigrid_lavagap_config: MiniGridConfig
    default_seed: int
    allow_seed_reuse: bool
    operators: tuple[OperatorDescriptor, ...]
    # Optional configs for additional MiniGrid variants
    minigrid_redbluedoors_6x6_config: MiniGridConfig | None = None
    minigrid_redbluedoors_8x8_config: MiniGridConfig | None = None
    default_operator_id: Optional[str] = None
    workers: Tuple[WorkerDefinition, ...] = tuple()


class ControlPanelWidget(QtWidgets.QWidget):
    control_mode_changed = pyqtSignal(ControlMode)
    game_changed = pyqtSignal(GameId)
    load_requested = pyqtSignal(GameId, ControlMode, int)
    reset_requested = pyqtSignal(int)
    worker_changed = pyqtSignal(str)
    slippery_toggled = pyqtSignal(bool)
    frozen_v2_config_changed = pyqtSignal(str, object)  # (param_name, value)
    taxi_config_changed = pyqtSignal(str, bool)  # (param_name, value)
    cliff_config_changed = pyqtSignal(str, bool)  # (param_name, value)
    lunar_config_changed = pyqtSignal(str, object)  # (param_name, value)
    car_config_changed = pyqtSignal(str, object)  # (param_name, value)
    bipedal_config_changed = pyqtSignal(str, object)  # (param_name, value)
    vizdoom_config_changed = pyqtSignal(str, object)  # (param_name, value)
    start_game_requested = pyqtSignal()
    pause_game_requested = pyqtSignal()
    continue_game_requested = pyqtSignal()
    terminate_game_requested = pyqtSignal()
    agent_step_requested = pyqtSignal()
    operator_changed = pyqtSignal(str)
    # Multi-Operator Mode signals - scientific execution for fair LLM/RL comparison
    operators_changed = pyqtSignal(list)  # List[OperatorConfig] - when operator configs change
    step_all_requested = pyqtSignal(int)  # Step all operators with seed (lock-step execution)
    step_player_requested = pyqtSignal(str, int)  # Step specific player (player_id, seed)
    reset_all_requested = pyqtSignal(int)  # Reset all operators with seed (fair comparison)
    stop_operators_requested = pyqtSignal()  # Stop all running operators
    initialize_operator_requested = pyqtSignal(str, object, int)  # operator_id, config, seed - preview env
    human_action_requested = pyqtSignal(str, int)  # operator_id, action - human operator action from action buttons
    train_agent_requested = pyqtSignal(str)  # Start fresh headless training
    trained_agent_requested = pyqtSignal(str)  # Load trained policy for evaluation
    resume_training_requested = pyqtSignal(str)  # Resume training from checkpoint
    custom_script_requested = pyqtSignal(str)  # Launch custom bash script training
    # MuJoCo MPC signals - emitted when user launches from sidebar
    mpc_launch_requested = pyqtSignal(str)  # Launch with display mode ("external" or "embedded")
    mpc_stop_all_requested = pyqtSignal()  # Stop all MJPC instances
    # Godot GE signals - emitted when user launches from sidebar
    godot_launch_requested = pyqtSignal(str)  # Launch with display mode ("external" or "embedded")
    godot_editor_requested = pyqtSignal()  # Launch Godot editor
    godot_stop_all_requested = pyqtSignal()  # Stop all Godot instances
    # Multi-Agent Mode signals - emitted from the Multi-Agent tab
    multi_agent_load_requested = pyqtSignal(str, int)  # env_id, seed
    multi_agent_start_requested = pyqtSignal(str, str, int)  # env_id, human_agent, seed
    multi_agent_reset_requested = pyqtSignal(int)  # seed
    policy_evaluate_requested = pyqtSignal(dict)  # Full evaluation config with policies

    def __init__(
        self,
        *,
        config: ControlPanelConfig,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        if not config.workers:
            config = ControlPanelConfig(
                available_modes=config.available_modes,
                default_mode=config.default_mode,
                frozen_lake_config=config.frozen_lake_config,
                taxi_config=config.taxi_config,
                cliff_walking_config=config.cliff_walking_config,
                lunar_lander_config=config.lunar_lander_config,
                car_racing_config=config.car_racing_config,
                bipedal_walker_config=config.bipedal_walker_config,
                minigrid_empty_config=config.minigrid_empty_config,
                minigrid_doorkey_5x5_config=config.minigrid_doorkey_5x5_config,
                minigrid_doorkey_6x6_config=config.minigrid_doorkey_6x6_config,
                minigrid_doorkey_8x8_config=config.minigrid_doorkey_8x8_config,
                minigrid_doorkey_16x16_config=config.minigrid_doorkey_16x16_config,
                minigrid_lavagap_config=config.minigrid_lavagap_config,
                minigrid_redbluedoors_6x6_config=config.minigrid_redbluedoors_6x6_config,
                minigrid_redbluedoors_8x8_config=config.minigrid_redbluedoors_8x8_config,
                default_seed=config.default_seed,
                allow_seed_reuse=config.allow_seed_reuse,
                operators=config.operators,
                default_operator_id=config.default_operator_id,
                workers=get_worker_catalog(),
            )

        self._config = config
        self._available_modes = config.available_modes
        self._default_seed = max(1, config.default_seed)
        self._allow_seed_reuse = config.allow_seed_reuse
        self._worker_definitions: Tuple[WorkerDefinition, ...] = config.workers
        self._minigrid_defaults: Dict[GameId, MiniGridConfig] = {
            GameId.MINIGRID_EMPTY_5x5: config.minigrid_empty_config,
            GameId.MINIGRID_DOORKEY_5x5: config.minigrid_doorkey_5x5_config,
            GameId.MINIGRID_DOORKEY_6x6: config.minigrid_doorkey_6x6_config,
            GameId.MINIGRID_DOORKEY_8x8: config.minigrid_doorkey_8x8_config,
            GameId.MINIGRID_DOORKEY_16x16: config.minigrid_doorkey_16x16_config,
            GameId.MINIGRID_LAVAGAP_S7: config.minigrid_lavagap_config,
            **({GameId.MINIGRID_REDBLUE_DOORS_6x6: config.minigrid_redbluedoors_6x6_config} if config.minigrid_redbluedoors_6x6_config else {}),
            **({GameId.MINIGRID_REDBLUE_DOORS_8x8: config.minigrid_redbluedoors_8x8_config} if config.minigrid_redbluedoors_8x8_config else {}),
        }
        self._game_overrides: Dict[GameId, Dict[str, object]] = {
            GameId.FROZEN_LAKE: {
                "is_slippery": config.frozen_lake_config.is_slippery,
                "success_rate": config.frozen_lake_config.success_rate,
                "reward_schedule": config.frozen_lake_config.reward_schedule,
            },
            GameId.FROZEN_LAKE_V2: {
                "is_slippery": DEFAULT_FROZEN_LAKE_V2_CONFIG.is_slippery,
                "success_rate": DEFAULT_FROZEN_LAKE_V2_CONFIG.success_rate,
                "reward_schedule": DEFAULT_FROZEN_LAKE_V2_CONFIG.reward_schedule,
                "grid_height": DEFAULT_FROZEN_LAKE_V2_CONFIG.grid_height,
                "grid_width": DEFAULT_FROZEN_LAKE_V2_CONFIG.grid_width,
                "start_position": DEFAULT_FROZEN_LAKE_V2_CONFIG.start_position,
                "goal_position": DEFAULT_FROZEN_LAKE_V2_CONFIG.goal_position,
                "hole_count": DEFAULT_FROZEN_LAKE_V2_CONFIG.hole_count,
                "random_holes": DEFAULT_FROZEN_LAKE_V2_CONFIG.random_holes,
            },
            GameId.TAXI: {
                "is_raining": config.taxi_config.is_raining,
                "fickle_passenger": config.taxi_config.fickle_passenger,
            },
            GameId.CLIFF_WALKING: {"is_slippery": config.cliff_walking_config.is_slippery},
            GameId.LUNAR_LANDER: {
                "continuous": config.lunar_lander_config.continuous,
                "gravity": config.lunar_lander_config.gravity,
                "enable_wind": config.lunar_lander_config.enable_wind,
                "wind_power": config.lunar_lander_config.wind_power,
                "turbulence_power": config.lunar_lander_config.turbulence_power,
                "max_episode_steps": config.lunar_lander_config.max_episode_steps,
            },
            GameId.CAR_RACING: {
                "continuous": config.car_racing_config.continuous,
                "domain_randomize": config.car_racing_config.domain_randomize,
                "lap_complete_percent": config.car_racing_config.lap_complete_percent,
                "max_episode_steps": config.car_racing_config.max_episode_steps,
                "max_episode_seconds": config.car_racing_config.max_episode_seconds,
            },
            GameId.BIPEDAL_WALKER: {
                "hardcore": config.bipedal_walker_config.hardcore,
                "max_episode_steps": config.bipedal_walker_config.max_episode_steps,
                "max_episode_seconds": config.bipedal_walker_config.max_episode_seconds,
            },
            GameId.MINIGRID_EMPTY_5x5: {
                "partial_observation": config.minigrid_empty_config.partial_observation,
                "image_observation": config.minigrid_empty_config.image_observation,
                "reward_multiplier": config.minigrid_empty_config.reward_multiplier,
                "agent_view_size": config.minigrid_empty_config.agent_view_size,
                "max_episode_steps": config.minigrid_empty_config.max_episode_steps,
                "seed": config.minigrid_empty_config.seed,
            },
            GameId.MINIGRID_DOORKEY_5x5: {
                "partial_observation": config.minigrid_doorkey_5x5_config.partial_observation,
                "image_observation": config.minigrid_doorkey_5x5_config.image_observation,
                "reward_multiplier": config.minigrid_doorkey_5x5_config.reward_multiplier,
                "agent_view_size": config.minigrid_doorkey_5x5_config.agent_view_size,
                "max_episode_steps": config.minigrid_doorkey_5x5_config.max_episode_steps,
                "seed": config.minigrid_doorkey_5x5_config.seed,
            },
            GameId.MINIGRID_DOORKEY_6x6: {
                "partial_observation": config.minigrid_doorkey_6x6_config.partial_observation,
                "image_observation": config.minigrid_doorkey_6x6_config.image_observation,
                "reward_multiplier": config.minigrid_doorkey_6x6_config.reward_multiplier,
                "agent_view_size": config.minigrid_doorkey_6x6_config.agent_view_size,
                "max_episode_steps": config.minigrid_doorkey_6x6_config.max_episode_steps,
                "seed": config.minigrid_doorkey_6x6_config.seed,
            },
            GameId.MINIGRID_DOORKEY_8x8: {
                "partial_observation": config.minigrid_doorkey_8x8_config.partial_observation,
                "image_observation": config.minigrid_doorkey_8x8_config.image_observation,
                "reward_multiplier": config.minigrid_doorkey_8x8_config.reward_multiplier,
                "agent_view_size": config.minigrid_doorkey_8x8_config.agent_view_size,
                "max_episode_steps": config.minigrid_doorkey_8x8_config.max_episode_steps,
                "seed": config.minigrid_doorkey_8x8_config.seed,
            },
            GameId.MINIGRID_DOORKEY_16x16: {
                "partial_observation": config.minigrid_doorkey_16x16_config.partial_observation,
                "image_observation": config.minigrid_doorkey_16x16_config.image_observation,
                "reward_multiplier": config.minigrid_doorkey_16x16_config.reward_multiplier,
                "agent_view_size": config.minigrid_doorkey_16x16_config.agent_view_size,
                "max_episode_steps": config.minigrid_doorkey_16x16_config.max_episode_steps,
                "seed": config.minigrid_doorkey_16x16_config.seed,
            },
            GameId.MINIGRID_LAVAGAP_S7: {
                "partial_observation": config.minigrid_lavagap_config.partial_observation,
                "image_observation": config.minigrid_lavagap_config.image_observation,
                "reward_multiplier": config.minigrid_lavagap_config.reward_multiplier,
                "agent_view_size": config.minigrid_lavagap_config.agent_view_size,
                "max_episode_steps": config.minigrid_lavagap_config.max_episode_steps,
                "seed": config.minigrid_lavagap_config.seed,
            },
            **({
                GameId.MINIGRID_REDBLUE_DOORS_6x6: {
                    "partial_observation": config.minigrid_redbluedoors_6x6_config.partial_observation,
                    "image_observation": config.minigrid_redbluedoors_6x6_config.image_observation,
                    "reward_multiplier": config.minigrid_redbluedoors_6x6_config.reward_multiplier,
                    "agent_view_size": config.minigrid_redbluedoors_6x6_config.agent_view_size,
                    "max_episode_steps": config.minigrid_redbluedoors_6x6_config.max_episode_steps,
                    "seed": config.minigrid_redbluedoors_6x6_config.seed,
                }
            } if config.minigrid_redbluedoors_6x6_config else {}),
            **({
                GameId.MINIGRID_REDBLUE_DOORS_8x8: {
                    "partial_observation": config.minigrid_redbluedoors_8x8_config.partial_observation,
                    "image_observation": config.minigrid_redbluedoors_8x8_config.image_observation,
                    "reward_multiplier": config.minigrid_redbluedoors_8x8_config.reward_multiplier,
                    "agent_view_size": config.minigrid_redbluedoors_8x8_config.agent_view_size,
                    "max_episode_steps": config.minigrid_redbluedoors_8x8_config.max_episode_steps,
                    "seed": config.minigrid_redbluedoors_8x8_config.seed,
                }
            } if config.minigrid_redbluedoors_8x8_config else {}),
        }
        self._current_telemetry_mode = TelemetryModes.FASTLANE_ONLY

        self._current_game: Optional[GameId] = None
        self._current_mode: ControlMode = self._load_mode_preference(config.default_mode)
        self._awaiting_human: bool = False
        self._auto_running: bool = False
        self._game_started: bool = False
        self._game_paused: bool = False
        self._operator_descriptors: Dict[str, OperatorDescriptor] = {
            descriptor.operator_id: descriptor for descriptor in config.operators
        }
        self._operator_order: tuple[OperatorDescriptor, ...] = config.operators
        default_operator = config.default_operator_id
        if default_operator is None and self._operator_order:
            default_operator = self._operator_order[0].operator_id
        self._active_operator_id: Optional[str] = default_operator

        # Store game configurations
        self._game_configs: Dict[GameId, Dict[str, object]] = {
            GameId.FROZEN_LAKE: {
                "is_slippery": config.frozen_lake_config.is_slippery
            },
            GameId.TAXI: {
                "is_raining": config.taxi_config.is_raining,
                "fickle_passenger": config.taxi_config.fickle_passenger,
            },
            GameId.CLIFF_WALKING: {
                "is_slippery": config.cliff_walking_config.is_slippery,
            },
            GameId.LUNAR_LANDER: {
                "continuous": config.lunar_lander_config.continuous,
                "gravity": config.lunar_lander_config.gravity,
                "enable_wind": config.lunar_lander_config.enable_wind,
                "wind_power": config.lunar_lander_config.wind_power,
                "turbulence_power": config.lunar_lander_config.turbulence_power,
                "max_episode_steps": config.lunar_lander_config.max_episode_steps,
            },
            GameId.CAR_RACING: {
                "continuous": config.car_racing_config.continuous,
                "domain_randomize": config.car_racing_config.domain_randomize,
                "lap_complete_percent": config.car_racing_config.lap_complete_percent,
                "max_episode_steps": config.car_racing_config.max_episode_steps,
                "max_episode_seconds": config.car_racing_config.max_episode_seconds,
            },
            GameId.BIPEDAL_WALKER: {
                "hardcore": config.bipedal_walker_config.hardcore,
                "max_episode_steps": config.bipedal_walker_config.max_episode_steps,
                "max_episode_seconds": config.bipedal_walker_config.max_episode_seconds,
            },
            GameId.MINIGRID_EMPTY_5x5: {
                "partial_observation": config.minigrid_empty_config.partial_observation,
                "image_observation": config.minigrid_empty_config.image_observation,
                "reward_multiplier": config.minigrid_empty_config.reward_multiplier,
                "agent_view_size": config.minigrid_empty_config.agent_view_size,
                "max_episode_steps": config.minigrid_empty_config.max_episode_steps,
                "seed": config.minigrid_empty_config.seed,
            },
            GameId.MINIGRID_DOORKEY_5x5: {
                "partial_observation": config.minigrid_doorkey_5x5_config.partial_observation,
                "image_observation": config.minigrid_doorkey_5x5_config.image_observation,
                "reward_multiplier": config.minigrid_doorkey_5x5_config.reward_multiplier,
                "agent_view_size": config.minigrid_doorkey_5x5_config.agent_view_size,
                "max_episode_steps": config.minigrid_doorkey_5x5_config.max_episode_steps,
                "seed": config.minigrid_doorkey_5x5_config.seed,
            },
            GameId.MINIGRID_DOORKEY_6x6: {
                "partial_observation": config.minigrid_doorkey_6x6_config.partial_observation,
                "image_observation": config.minigrid_doorkey_6x6_config.image_observation,
                "reward_multiplier": config.minigrid_doorkey_6x6_config.reward_multiplier,
                "agent_view_size": config.minigrid_doorkey_6x6_config.agent_view_size,
                "max_episode_steps": config.minigrid_doorkey_6x6_config.max_episode_steps,
                "seed": config.minigrid_doorkey_6x6_config.seed,
            },
            GameId.MINIGRID_DOORKEY_8x8: {
                "partial_observation": config.minigrid_doorkey_8x8_config.partial_observation,
                "image_observation": config.minigrid_doorkey_8x8_config.image_observation,
                "reward_multiplier": config.minigrid_doorkey_8x8_config.reward_multiplier,
                "agent_view_size": config.minigrid_doorkey_8x8_config.agent_view_size,
                "max_episode_steps": config.minigrid_doorkey_8x8_config.max_episode_steps,
                "seed": config.minigrid_doorkey_8x8_config.seed,
            },
            GameId.MINIGRID_DOORKEY_16x16: {
                "partial_observation": config.minigrid_doorkey_16x16_config.partial_observation,
                "image_observation": config.minigrid_doorkey_16x16_config.image_observation,
                "reward_multiplier": config.minigrid_doorkey_16x16_config.reward_multiplier,
                "agent_view_size": config.minigrid_doorkey_16x16_config.agent_view_size,
                "max_episode_steps": config.minigrid_doorkey_16x16_config.max_episode_steps,
                "seed": config.minigrid_doorkey_16x16_config.seed,
            },
            GameId.MINIGRID_LAVAGAP_S7: {
                "partial_observation": config.minigrid_lavagap_config.partial_observation,
                "image_observation": config.minigrid_lavagap_config.image_observation,
                "reward_multiplier": config.minigrid_lavagap_config.reward_multiplier,
                "agent_view_size": config.minigrid_lavagap_config.agent_view_size,
                "max_episode_steps": config.minigrid_lavagap_config.max_episode_steps,
                "seed": config.minigrid_lavagap_config.seed,
            },
            GameId.MINIGRID_REDBLUE_DOORS_6x6: {
                "partial_observation": config.minigrid_redbluedoors_6x6_config.partial_observation if config.minigrid_redbluedoors_6x6_config else True,
                "image_observation": config.minigrid_redbluedoors_6x6_config.image_observation if config.minigrid_redbluedoors_6x6_config else True,
                "reward_multiplier": config.minigrid_redbluedoors_6x6_config.reward_multiplier if config.minigrid_redbluedoors_6x6_config else 10.0,
                "agent_view_size": config.minigrid_redbluedoors_6x6_config.agent_view_size if config.minigrid_redbluedoors_6x6_config else None,
                "max_episode_steps": config.minigrid_redbluedoors_6x6_config.max_episode_steps if config.minigrid_redbluedoors_6x6_config else None,
                "seed": config.minigrid_redbluedoors_6x6_config.seed if config.minigrid_redbluedoors_6x6_config else None,
            },
            GameId.MINIGRID_REDBLUE_DOORS_8x8: {
                "partial_observation": config.minigrid_redbluedoors_8x8_config.partial_observation if config.minigrid_redbluedoors_8x8_config else True,
                "image_observation": config.minigrid_redbluedoors_8x8_config.image_observation if config.minigrid_redbluedoors_8x8_config else True,
                "reward_multiplier": config.minigrid_redbluedoors_8x8_config.reward_multiplier if config.minigrid_redbluedoors_8x8_config else 10.0,
                "agent_view_size": config.minigrid_redbluedoors_8x8_config.agent_view_size if config.minigrid_redbluedoors_8x8_config else None,
                "max_episode_steps": config.minigrid_redbluedoors_8x8_config.max_episode_steps if config.minigrid_redbluedoors_8x8_config else None,
                "seed": config.minigrid_redbluedoors_8x8_config.seed if config.minigrid_redbluedoors_8x8_config else None,
            },
        }

        self._current_worker_id: Optional[str] = None
        self._family_combo: Optional[QtWidgets.QComboBox] = None
        self._family_to_games: dict[EnvironmentFamily, list[tuple[str, GameId]]] = {}
        self._selected_family: Optional[EnvironmentFamily] = None
        self._all_games: tuple[GameId, ...] = ()
        self._build_ui()
        self._apply_current_mode_selection()
        self._connect_signals()
        self._update_control_states()
        self._populate_operator_combo()
        # Note: Worker combo is now populated by SingleAgentTab
        self._sync_worker_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def populate_games(self, games: Iterable[GameId], *, default: Optional[GameId] = None) -> None:
        if self._family_combo is None or self._game_combo is None:
            return
        games_tuple = tuple(games)
        self._all_games = games_tuple
        self._family_to_games = self._build_family_index(games_tuple)

        if not games_tuple:
            self._family_combo.blockSignals(True)
            self._family_combo.clear()
            self._family_combo.blockSignals(False)
            self._game_combo.blockSignals(True)
            self._game_combo.clear()
            self._game_combo.blockSignals(False)
            return

        chosen_game = default if (default is not None and default in games_tuple) else games_tuple[0]
        chosen_family = self._family_for_game(chosen_game)
        self._populate_family_combo(chosen_family)
        self._rebuild_game_combo(chosen_family, chosen_game)

    def current_operator(self) -> Optional[str]:
        return self._active_operator_id

    def current_worker_id(self) -> Optional[str]:
        return self._current_worker_id

    def current_worker(self) -> Optional[WorkerDefinition]:
        return self._current_worker_definition()

    def set_active_operator(self, operator_id: str) -> None:
        if operator_id == self._active_operator_id:
            return
        index = self._operator_combo.findData(operator_id)
        if index < 0:
            return
        self._operator_combo.blockSignals(True)
        self._operator_combo.setCurrentIndex(index)
        self._operator_combo.blockSignals(False)
        self._active_operator_id = operator_id
        self._update_operator_description()

    def update_modes(self, game_id: GameId) -> None:
        supported = tuple(self._available_modes.get(game_id, ()))
        if not supported:
            self._mode_combo.setEnabled(False)
            return

        self._mode_combo.blockSignals(True)
        self._mode_combo.clear()
        for mode in supported:
            label = mode.value.replace("_", " ").title()
            self._mode_combo.addItem(label, mode)
        self._mode_combo.blockSignals(False)
        self._mode_combo.setEnabled(bool(supported))

        if self._current_mode not in supported:
            self._current_mode = supported[0]
            self._persist_mode_preference(self._current_mode)
            index = self._mode_combo.findData(self._current_mode)
            if index >= 0:
                self._mode_combo.blockSignals(True)
                self._mode_combo.setCurrentIndex(index)
                self._mode_combo.blockSignals(False)
            self._emit_mode_changed(self._current_mode)
            return

        index = self._mode_combo.findData(self._current_mode)
        if index >= 0:
            self._mode_combo.blockSignals(True)
            self._mode_combo.setCurrentIndex(index)
            self._mode_combo.blockSignals(False)
        is_human_tab = self._tab_widget is not None and self._tab_widget.currentWidget() is self._human_tab
        self._mode_combo.setEnabled(is_human_tab and bool(supported))
        self._update_control_states()

    def set_status(
        self,
        *,
        step: int,
        reward: float,
        total_reward: float,
        terminated: bool,
        truncated: bool,
        turn: str,
        awaiting_human: bool,
        session_time: str,
        active_time: str,
        episode_duration: str,
        outcome_time: str = "—",
        outcome_wall_clock: str | None = None,
        team_rewards: dict[int, float] | None = None,
    ) -> None:
        self._step_label.setText(str(step))
        self._reward_label.setText(f"{reward:.2f}")
        self._total_reward_label.setText(f"{total_reward:.2f}")
        self._terminated_label.setText(self._format_bool(terminated))
        self._truncated_label.setText(self._format_bool(truncated))
        self._turn_label.setText(turn)
        self.set_awaiting_human(awaiting_human)
        self.set_time_labels(
            session_time,
            active_time,
            outcome_time,
            outcome_timestamp=outcome_wall_clock,
        )
        self._update_team_rewards(team_rewards)

    def _update_team_rewards(self, team_rewards: dict[int, float] | None) -> None:
        """Dynamically create/update per-team reward labels in the status grid."""
        layout: QtWidgets.QGridLayout = self._status_group.layout()  # type: ignore[assignment]

        if team_rewards is None or len(team_rewards) < 2:
            # Hide team labels for single-team or non-team environments
            for title_lbl, value_lbl in self._team_reward_labels.values():
                title_lbl.hide()
                value_lbl.hide()
            return

        for team_idx, cumulative_reward in sorted(team_rewards.items()):
            if team_idx not in self._team_reward_labels:
                # Create new labels for this team
                title_lbl = QtWidgets.QLabel(f"Team {team_idx}", self._status_group)
                title_lbl.setAlignment(
                    QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
                )
                value_lbl = QtWidgets.QLabel("0.00", self._status_group)
                self._team_reward_labels[team_idx] = (title_lbl, value_lbl)
                # Place in right column, below existing fields
                row = layout.rowCount()
                layout.addWidget(title_lbl, row, 2)
                layout.addWidget(value_lbl, row, 3)

            title_lbl, value_lbl = self._team_reward_labels[team_idx]
            value_lbl.setText(f"{cumulative_reward:+.2f}")
            title_lbl.show()
            value_lbl.show()

    def set_turn(self, turn: str) -> None:
        self._turn_label.setText(turn)

    def set_fps(self, fps: float | None) -> None:
        if fps is None or fps <= 0:
            self._fps_label.setText("—")
        else:
            self._fps_label.setText(f"{fps:.1f}")

    def fastlane_only_enabled(self) -> bool:
        return self._current_telemetry_mode == TelemetryModes.FASTLANE_ONLY

    def set_fastlane_only(self, enabled: bool) -> None:
        if enabled:
            self._fastlane_only_checkbox.setChecked(True)
        else:
            self._dual_path_radio.setChecked(True)

    def set_mode(self, mode: ControlMode) -> None:
        index = self._mode_combo.findData(mode)
        if index < 0:
            return
        self._current_mode = mode
        self._mode_combo.blockSignals(True)
        self._mode_combo.setCurrentIndex(index)
        self._mode_combo.blockSignals(False)
        self._emit_mode_changed(mode)

    def set_game(self, game: GameId) -> None:
        family = self._family_for_game(game)
        self._populate_family_combo(family)
        self._rebuild_game_combo(family, game)

    def override_slippery(self, enabled: bool) -> None:
        overrides = self._game_overrides.setdefault(GameId.FROZEN_LAKE, {})
        overrides["is_slippery"] = enabled
        if self._frozen_slippery_checkbox is not None:
            self._frozen_slippery_checkbox.blockSignals(True)
            self._frozen_slippery_checkbox.setChecked(enabled)
            self._frozen_slippery_checkbox.blockSignals(False)

    def current_seed(self) -> int:
        return int(self._seed_spin.value())

    def current_mode(self) -> ControlMode:
        return self._current_mode

    def current_game(self) -> Optional[GameId]:
        return self._current_game

    def cleanrl_environment_id(self) -> Optional[str]:
        """Return the currently selected CleanRL environment id (Agent-only tab)."""
        return None

    def get_overrides(self, game_id: GameId) -> Dict[str, object]:
        return dict(self._game_overrides.get(game_id, {}))

    def get_mujoco_mpc_tab(self) -> MuJoCoMPCTab:
        """Return the MuJoCo MPC tab for status updates."""
        return self._mujoco_mpc_tab

    def get_godot_ge_tab(self) -> GodotGETab:
        """Return the Godot GE tab for status updates."""
        return self._godot_ge_tab

    def set_auto_running(self, running: bool) -> None:
        self._auto_running = running
        self._update_control_states()

    def set_game_started(self, started: bool) -> None:
        """Set whether the game has been started."""
        self._game_started = started
        if not started:
            self._game_paused = False
        self._update_control_states()

    def set_game_paused(self, paused: bool) -> None:
        """Set whether the game is paused."""
        self._game_paused = paused
        self._update_control_states()

    def set_slippery_visible(self, visible: bool) -> None:
        if self._frozen_slippery_checkbox is not None:
            self._frozen_slippery_checkbox.setVisible(visible)

    def set_awaiting_human(self, awaiting: bool) -> None:
        self._awaiting_human = awaiting
        self._awaiting_label.setText("Yes" if awaiting else "No")
        self._update_control_states()

    def set_time_labels(
        self,
        session_time: str,
        active_time: str,
        outcome_time: str,
        *,
        outcome_timestamp: str | None = None,
    ) -> None:
        self._session_time_label.setText(session_time)
        self._active_time_label.setText(active_time)
        self._outcome_time_label.setText(outcome_time)
        tooltip = "Elapsed time between the first move and the recorded outcome."
        if outcome_timestamp and outcome_timestamp != "—":
            tooltip += f"\nOutcome recorded at {outcome_timestamp}."
        elif outcome_timestamp == "—":
            tooltip += "\nOutcome not recorded yet."
        self._outcome_time_label.setToolTip(tooltip)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self._tab_widget = QtWidgets.QTabWidget(self)
        self._tab_widget.setTabPosition(QtWidgets.QTabWidget.TabPosition.North)
        self._tab_widget.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(self._tab_widget)

        self._tab_to_mode = {}

        self._human_tab = QtWidgets.QWidget(self)
        human_layout = QtWidgets.QVBoxLayout(self._human_tab)
        human_layout.setContentsMargins(0, 0, 0, 0)
        human_layout.setSpacing(12)

        environment_group = self._create_environment_group(self._human_tab)
        human_layout.addWidget(environment_group)

        cfg_group = self._create_config_group(self._human_tab)
        self._config_scroll = QtWidgets.QScrollArea(self._human_tab)
        self._config_scroll.setWidgetResizable(True)
        self._config_scroll.setStyleSheet("QScrollArea { border: none; }")
        self._config_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._config_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)  # type: ignore[attr-defined]
        self._config_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)  # type: ignore[attr-defined]
        # Set minimum height so the config panel isn't too small - expanded by default
        self._config_scroll.setMinimumHeight(350)
        sp = self._config_scroll.sizePolicy()
        sp.setVerticalPolicy(QtWidgets.QSizePolicy.Policy.Expanding)
        sp.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.Preferred)
        self._config_scroll.setSizePolicy(sp)
        self._config_scroll.setWidget(cfg_group)
        # Ensure the config group is visible and expanded
        cfg_group.setVisible(True)
        cfg_group.setMinimumHeight(300)
        human_layout.addWidget(self._config_scroll, stretch=3)  # Give more stretch weight

        human_layout.addWidget(self._create_mode_group(self._human_tab))
        self._control_buttons_widget = self._create_control_group(self._human_tab)
        human_layout.addWidget(self._control_buttons_widget)

        # Keyboard Assignment Widget for multi-human gameplay
        self._keyboard_widget = KeyboardAssignmentWidget(
            available_agents=["agent_0", "agent_1"],
            parent=self._human_tab
        )
        self._keyboard_widget.setVisible(False)  # Hidden until multi-agent env loaded
        human_layout.addWidget(self._keyboard_widget)

        human_layout.addWidget(self._create_telemetry_mode_group(self._human_tab))
        human_layout.addWidget(self._create_status_group(self._human_tab))
        # Minimal stretch at end so Game Configuration gets more space
        human_layout.addStretch(0)
        human_index = self._tab_widget.addTab(self._human_tab, "Human Control")
        self._tab_to_mode[human_index] = ControlMode.HUMAN_ONLY

        # Operators Tab - main tab for configuring action-selecting entities
        self._operators_tab = OperatorsTab(self)
        operators_index = self._tab_widget.addTab(self._operators_tab, "Operators")
        self._tab_to_mode[operators_index] = ControlMode.AGENT_ONLY
        # Initialize legacy operator widgets (hidden, for backward compatibility)
        self._init_legacy_operator_widgets()
        # Connect operators signals - scientific execution for fair comparison
        self._operators_tab.operators_changed.connect(self._on_operators_config_changed)
        self._operators_tab.step_all_requested.connect(self._on_step_all_clicked)
        self._operators_tab.step_player_requested.connect(self._on_step_player_clicked)
        self._operators_tab.reset_all_requested.connect(self._on_reset_all_clicked)
        self._operators_tab.stop_operators_requested.connect(self._on_stop_operators_clicked)
        self._operators_tab.initialize_operator_requested.connect(self._on_initialize_operator_requested)
        self._operators_tab.human_action_requested.connect(self._on_human_action_requested)

        # Single-Agent Mode Tab with Workers subtab (training)
        self._single_agent_tab = SingleAgentTab(self)
        single_index = self._tab_widget.addTab(self._single_agent_tab, "Single-Agent Mode")
        self._tab_to_mode[single_index] = ControlMode.AGENT_ONLY
        # Connect single-agent (workers) signals
        self._single_agent_tab.worker_changed.connect(self._on_single_agent_worker_changed)
        self._single_agent_tab.train_requested.connect(self._emit_train_agent_requested)
        self._single_agent_tab.evaluate_requested.connect(self._emit_trained_agent_requested)
        self._single_agent_tab.resume_requested.connect(self._emit_resume_training_requested)
        self._single_agent_tab.script_requested.connect(self._emit_custom_script_requested)

        # Multi-Agent Mode Tab with subtabs
        self._multi_agent_tab = MultiAgentTab(self)
        multi_index = self._tab_widget.addTab(self._multi_agent_tab, "Multi-Agent Mode")
        self._tab_to_mode[multi_index] = ControlMode.MULTI_AGENT_COOP
        # Connect multi-agent signals
        self._multi_agent_tab.train_requested.connect(self._on_multi_agent_train_requested)
        self._multi_agent_tab.evaluate_requested.connect(self._on_multi_agent_evaluate_requested)
        self._multi_agent_tab.resume_requested.connect(self._on_multi_agent_resume_requested)
        self._multi_agent_tab.script_requested.connect(self._on_multi_agent_script_requested)
        self._multi_agent_tab.load_environment_requested.connect(self._on_multi_agent_load_requested)
        self._multi_agent_tab.start_game_requested.connect(self.multi_agent_start_requested)
        self._multi_agent_tab.reset_game_requested.connect(self.multi_agent_reset_requested)

        # MuJoCo MPC Tab - launcher for MPC visualization in Render View
        self._mujoco_mpc_tab = MuJoCoMPCTab(self)
        self._tab_widget.addTab(self._mujoco_mpc_tab, "MuJoCo MPC")
        # Connect signals directly
        self._mujoco_mpc_tab.launch_mpc_requested.connect(self.mpc_launch_requested)
        self._mujoco_mpc_tab.stop_all_requested.connect(self.mpc_stop_all_requested)

        # Godot GE Tab - launcher for Godot game engine (3D environments)
        self._godot_ge_tab = GodotGETab(self)
        self._tab_widget.addTab(self._godot_ge_tab, "Godot GE")
        # Connect signals directly
        self._godot_ge_tab.launch_godot_requested.connect(self.godot_launch_requested)
        self._godot_ge_tab.launch_editor_requested.connect(self.godot_editor_requested)
        self._godot_ge_tab.stop_all_requested.connect(self.godot_stop_all_requested)

        self._on_tab_changed(self._tab_widget.currentIndex())

    def _create_environment_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Environment", parent)
        layout = QtWidgets.QGridLayout(group)
        layout.setColumnStretch(1, 1)

        layout.addWidget(QtWidgets.QLabel("Family", group), 0, 0)
        self._family_combo = QtWidgets.QComboBox(group)
        family_view = QtWidgets.QListView(self._family_combo)
        family_view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        family_view.setUniformItemSizes(True)
        self._family_combo.setView(family_view)
        self._family_combo.setStyleSheet("QComboBox { combobox-popup: 0; }")
        self._family_combo.setMaxVisibleItems(10)
        layout.addWidget(self._family_combo, 0, 1, 1, 2)

        layout.addWidget(QtWidgets.QLabel("Environment", group), 1, 0)
        self._game_combo = QtWidgets.QComboBox(group)
        game_view = QtWidgets.QListView(self._game_combo)
        game_view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        game_view.setUniformItemSizes(True)
        self._game_combo.setView(game_view)
        self._game_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._game_combo.setStyleSheet("QComboBox { combobox-popup: 0; }")
        self._game_combo.setMaxVisibleItems(10)
        layout.addWidget(self._game_combo, 1, 1, 1, 2)

        layout.addWidget(QtWidgets.QLabel("Seed", group), 2, 0)
        self._seed_spin = QtWidgets.QSpinBox(group)
        self._seed_spin.setRange(1, 10_000_000)
        self._seed_spin.setValue(self._default_seed)
        layout.addWidget(self._seed_spin, 2, 1)

        self._seed_reuse_checkbox = QtWidgets.QCheckBox("Allow seed reuse", group)
        self._seed_reuse_checkbox.setChecked(self._allow_seed_reuse)
        layout.addWidget(self._seed_reuse_checkbox, 2, 2)
        if self._allow_seed_reuse:
            self._seed_spin.setToolTip("Seeds auto-increment by default. Adjust before loading to reuse a previous seed.")
        else:
            self._seed_spin.setToolTip("Seed increments automatically after each episode.")
        self._seed_reuse_checkbox.stateChanged.connect(self._on_seed_reuse_changed)

        self._load_button = QtWidgets.QPushButton("Load Environment", group)
        layout.addWidget(self._load_button, 3, 0, 1, 3)
        return group

    def _format_family_label(self, family: EnvironmentFamily) -> str:
        if family == EnvironmentFamily.OTHER:
            return "Other"
        return family.value.replace("_", " ").title()

    def _family_for_game(self, game: Optional[GameId]) -> EnvironmentFamily:
        if game is None:
            return EnvironmentFamily.TOY_TEXT
        return ENVIRONMENT_FAMILY_BY_GAME.get(game, EnvironmentFamily.OTHER)

    def _build_family_index(
        self, games: Iterable[GameId]
    ) -> dict[EnvironmentFamily, list[tuple[str, GameId]]]:
        mapping: dict[EnvironmentFamily, list[tuple[str, GameId]]] = defaultdict(list)
        for game in games:
            family = self._family_for_game(game)
            mapping[family].append((get_game_display_name(game), game))
        for family in mapping:
            mapping[family].sort(key=lambda item: item[0])
        return mapping

    def _populate_family_combo(self, preferred: Optional[EnvironmentFamily]) -> None:
        if self._family_combo is None:
            return
        # Filter out OTHER - it's a fallback category not meant for UI display
        families = sorted(
            (f for f in self._family_to_games.keys() if f != EnvironmentFamily.OTHER),
            key=lambda fam: fam.value
        )
        if not families:
            self._family_combo.blockSignals(True)
            self._family_combo.clear()
            self._family_combo.blockSignals(False)
            return
        target = preferred if preferred in families else families[0]
        self._selected_family = target
        self._family_combo.blockSignals(True)
        self._family_combo.clear()
        for family in families:
            self._family_combo.addItem(self._format_family_label(family), family)
        index = self._family_combo.findData(target)
        if index < 0:
            index = 0
        self._family_combo.setCurrentIndex(index)
        self._family_combo.blockSignals(False)

    def _rebuild_game_combo(
        self, family: EnvironmentFamily, selected: Optional[GameId] = None
    ) -> None:
        if self._game_combo is None:
            return
        games = self._family_to_games.get(family, [])
        if not games:
            self._game_combo.blockSignals(True)
            self._game_combo.clear()
            self._game_combo.blockSignals(False)
            return
        choices = [gid for _, gid in games]
        target = selected if selected in choices else choices[0]
        self._current_game = target
        self._game_combo.blockSignals(True)
        self._game_combo.clear()
        for label, gid in games:
            self._game_combo.addItem(label, gid)
        index = self._game_combo.findData(target)
        if index < 0:
            index = 0
        self._game_combo.setCurrentIndex(index)
        self._game_combo.blockSignals(False)
        # Refresh config UI to show proper options for the selected game
        self._refresh_game_config_ui()

    def _on_family_changed(self, index: int) -> None:
        if self._family_combo is None:
            return
        data = self._family_combo.itemData(index)
        if not isinstance(data, EnvironmentFamily):
            return
        if data == self._selected_family:
            return
        self._selected_family = data
        self._rebuild_game_combo(data, None)
        # _rebuild_game_combo blocks signals on the game combo, so
        # _on_game_changed / _emit_game_changed never fire.  We must
        # call _emit_game_changed manually so that update_modes() runs
        # and _current_mode is corrected for the newly-selected game
        # (e.g. switching to SMAC which has no HUMAN_ONLY support).
        self._emit_game_changed(self._current_game)

    def _create_config_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        self._config_group = QtWidgets.QGroupBox("Game Configuration", parent)
        self._config_layout = QtWidgets.QFormLayout(self._config_group)
        self._frozen_slippery_checkbox = None
        return self._config_group

    def _create_mode_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        self._mode_group = QtWidgets.QGroupBox("Control Mode", parent)
        layout = QtWidgets.QVBoxLayout(self._mode_group)
        self._mode_combo = QtWidgets.QComboBox(self._mode_group)
        self._mode_combo.setEnabled(False)
        for mode in ControlMode:
            label = mode.value.replace("_", " ").title()
            self._mode_combo.addItem(label, mode)
        layout.addWidget(self._mode_combo)
        return self._mode_group

    def _create_telemetry_mode_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        self._telemetry_mode_group = QtWidgets.QGroupBox("Telemetry Mode", parent)
        group = self._telemetry_mode_group
        layout = QtWidgets.QVBoxLayout(group)
        description = QtWidgets.QLabel(
            "Choose which telemetry path to use for this session. Fast Lane streams frames via shared memory; "
            "Dual Path uses RunBus fast path + durable SQLite so replay remains available.",
            group,
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        self._telemetry_button_group = QtWidgets.QButtonGroup(group)

        fastlane_desc = TELEMETRY_MODE_DESCRIPTORS[TelemetryModes.FASTLANE_ONLY]
        self._fastlane_only_checkbox = QtWidgets.QRadioButton(fastlane_desc.label, group)
        self._fastlane_only_checkbox.setToolTip(fastlane_desc.description)
        self._telemetry_button_group.addButton(self._fastlane_only_checkbox)
        layout.addWidget(self._fastlane_only_checkbox)

        dual_path_desc = TELEMETRY_MODE_DESCRIPTORS[TelemetryModes.UI_AND_DB]
        self._dual_path_radio = QtWidgets.QRadioButton(dual_path_desc.label, group)
        self._dual_path_radio.setToolTip(dual_path_desc.description)
        self._telemetry_button_group.addButton(self._dual_path_radio)
        layout.addWidget(self._dual_path_radio)

        self._fastlane_only_checkbox.setChecked(True)
        self._fastlane_only_checkbox.toggled.connect(self._update_telemetry_status_label)
        self._dual_path_radio.toggled.connect(self._update_telemetry_status_label)
        return group

    def _init_legacy_operator_widgets(self) -> None:
        """Initialize legacy operator widgets for backward compatibility.

        These widgets are hidden but still used by other code paths
        (e.g., set_active_operator, _populate_operator_combo).
        """
        # Legacy single-operator combo (hidden, for backward compatibility)
        self._operator_combo = QtWidgets.QComboBox(self)
        self._operator_combo.setEnabled(bool(self._operator_order))
        self._operator_combo.hide()

        self._operator_description = QtWidgets.QLabel("—", self)
        self._operator_description.setWordWrap(True)
        self._operator_description.hide()

    def _on_operators_config_changed(self, configs: list) -> None:
        """Handle operator configuration changes from the multi-operator widget."""
        self.operators_changed.emit(configs)

    def _on_step_all_clicked(self, seed: int) -> None:
        """Handle Step All button click - advance all operators by one step.

        This implements lock-step execution for scientific comparison.
        Each operator's agent selects one action simultaneously.
        """
        self.step_all_requested.emit(seed)

    def _on_human_action_requested(self, operator_id: str, action: int) -> None:
        """Handle human action request from action buttons.

        Args:
            operator_id: The human operator's ID.
            action: The action index selected.
        """
        self.human_action_requested.emit(operator_id, action)

    def _on_step_player_clicked(self, player_id: str, seed: int) -> None:
        """Handle player-specific step button click (PettingZoo mode).

        Args:
            player_id: Which player to step ("player_0" or "player_1").
            seed: Current seed value.
        """
        self.step_player_requested.emit(player_id, seed)

    def _on_reset_all_clicked(self, seed: int) -> None:
        """Handle Reset All button click - reset all operators with shared seed.

        This ensures all environments start with identical initial conditions
        for fair, reproducible side-by-side comparison (BALROG methodology).
        """
        self.reset_all_requested.emit(seed)

    def _on_stop_operators_clicked(self) -> None:
        """Handle Stop All button click."""
        self.stop_operators_requested.emit()

    def _on_initialize_operator_requested(self, operator_id: str, config: OperatorConfig, seed: int) -> None:
        """Handle Initialize button click - preview environment with shared seed."""
        self.initialize_operator_requested.emit(operator_id, config, seed)

    def _on_single_agent_worker_changed(self, worker_id: str) -> None:
        """Handle worker selection change from SingleAgentTab."""
        self._current_worker_id = worker_id
        self.worker_changed.emit(worker_id)

    # ------------------------------------------------------------------
    # Backward-compatible widget accessors (redirect to SingleAgentTab)
    # ------------------------------------------------------------------
    @property
    def _worker_combo(self) -> QtWidgets.QComboBox:
        """Get worker combo from SingleAgentTab."""
        return self._single_agent_tab.worker_combo

    @property
    def _worker_description(self) -> QtWidgets.QLabel:
        """Get worker description from SingleAgentTab."""
        return self._single_agent_tab.worker_description

    @property
    def _train_agent_button(self) -> QtWidgets.QPushButton:
        """Get train agent button from SingleAgentTab."""
        return self._single_agent_tab.train_agent_button

    @property
    def _trained_agent_button(self) -> QtWidgets.QPushButton:
        """Get evaluate policy button from SingleAgentTab."""
        return self._single_agent_tab.evaluate_policy_button

    @property
    def _resume_training_button(self) -> QtWidgets.QPushButton:
        """Get resume training button from SingleAgentTab."""
        return self._single_agent_tab.resume_training_button

    @property
    def _operator_config_widget(self) -> OperatorConfigWidget:
        """Get operator config widget from OperatorsTab."""
        return self._operators_tab.operator_config_widget

    def set_operator_environment_size(
        self, operator_id: str, width: int, height: int, container_size: int | None = None
    ) -> None:
        """Set the environment size for a specific operator.

        Args:
            operator_id: The operator's unique ID
            width: Rendered environment width in pixels (image size)
            height: Rendered environment height in pixels (image size)
            container_size: Optional container display size in pixels
        """
        self._operators_tab.set_operator_environment_size(
            operator_id, width, height, container_size
        )

    def set_turn_indicator(self, player_id: str, visible: bool = True) -> None:
        """Set the turn indicator for PettingZoo multi-agent games.

        Shows which player's turn is next (e.g., "Next: White (player_0)").

        Args:
            player_id: Current player (e.g., "player_0", "player_1").
            visible: Whether to show the indicator.
        """
        self._operators_tab.set_turn_indicator(player_id, visible)

    def set_pettingzoo_mode(self, enabled: bool) -> None:
        """Enable or disable PettingZoo multi-agent mode.

        When enabled, shows player-specific step buttons instead of Step All.

        Args:
            enabled: True to enable PettingZoo mode.
        """
        self._operators_tab.set_pettingzoo_mode(enabled)

    def set_current_player(self, player_id: str) -> None:
        """Set which player's turn it is (enables their step button).

        Args:
            player_id: The player whose turn it is ("player_0" or "player_1").
        """
        self._operators_tab.set_current_player(player_id)

    def set_step_count(self, count: int) -> None:
        """Set the step counter (called after each successful env step).

        Args:
            count: The current step count to display.
        """
        self._operators_tab.set_step_count(count)

    def _create_control_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Game Control Flow", parent)
        layout = QtWidgets.QVBoxLayout(group)

        row1 = QtWidgets.QHBoxLayout()
        self._start_button = QtWidgets.QPushButton("Start Game", group)
        self._pause_button = QtWidgets.QPushButton("Pause Game", group)
        self._continue_button = QtWidgets.QPushButton("Continue Game", group)
        row1.addWidget(self._start_button)
        row1.addWidget(self._pause_button)
        row1.addWidget(self._continue_button)
        layout.addLayout(row1)

        row2 = QtWidgets.QHBoxLayout()
        self._terminate_button = QtWidgets.QPushButton("Terminate Game", group)
        self._step_button = QtWidgets.QPushButton("Agent Step", group)
        self._reset_button = QtWidgets.QPushButton("Reset", group)
        row2.addWidget(self._terminate_button)
        row2.addWidget(self._step_button)
        row2.addWidget(self._reset_button)
        layout.addLayout(row2)
        return group

    def _create_status_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        self._status_group = QtWidgets.QGroupBox("Status", parent)
        layout = QtWidgets.QGridLayout(self._status_group)
        self._step_label = QtWidgets.QLabel("0", self._status_group)
        self._reward_label = QtWidgets.QLabel("0.0", self._status_group)
        self._total_reward_label = QtWidgets.QLabel("0.00", self._status_group)
        self._terminated_label = QtWidgets.QLabel("No", self._status_group)
        self._truncated_label = QtWidgets.QLabel("No", self._status_group)
        self._turn_label = QtWidgets.QLabel("human", self._status_group)
        self._awaiting_label = QtWidgets.QLabel("–", self._status_group)
        self._session_time_label = QtWidgets.QLabel("00:00:00", self._status_group)
        self._active_time_label = QtWidgets.QLabel("—", self._status_group)
        self._outcome_time_label = QtWidgets.QLabel("—", self._status_group)
        self._fps_label = QtWidgets.QLabel("—", self._status_group)
        self._telemetry_status_label = QtWidgets.QLabel("—", self._status_group)

        # Dynamic team reward labels (created on demand by _update_team_rewards)
        self._team_reward_labels: dict[int, tuple[QtWidgets.QLabel, QtWidgets.QLabel]] = {}

        fields = [
            ("Step", self._step_label),
            ("Reward", self._reward_label),
            ("Total Reward", self._total_reward_label),
            ("Episode Finished", self._terminated_label),
            ("Episode Aborted", self._truncated_label),
            ("Turn", self._turn_label),
            ("Awaiting Input", self._awaiting_label),
            ("Session Uptime", self._session_time_label),
            ("Active Play Time", self._active_time_label),
            ("Outcome Time", self._outcome_time_label),
            ("Live FPS", self._fps_label),
            ("Telemetry Mode", self._telemetry_status_label),
        ]

        midpoint = (len(fields) + 1) // 2
        columns = [fields[:midpoint], fields[midpoint:]]
        for col_index, column_fields in enumerate(columns):
            for row_index, (title, value_label) in enumerate(column_fields):
                title_label = QtWidgets.QLabel(title, self._status_group)
                title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
                base_col = col_index * 2
                layout.addWidget(title_label, row_index, base_col)
                layout.addWidget(value_label, row_index, base_col + 1)

        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)
        self._update_telemetry_status_label()
        return self._status_group

    def _on_tab_changed(self, index: int) -> None:
        if not hasattr(self, "_tab_to_mode"):
            return
        mode = self._tab_to_mode.get(index)
        if mode is not None:
            self._apply_mode_from_tab(mode)

        is_human_tab = self._tab_widget.widget(index) is self._human_tab
        if hasattr(self, "_mode_group") and self._mode_group is not None:
            self._mode_group.setVisible(is_human_tab)
        if hasattr(self, "_status_group") and self._status_group is not None:
            self._status_group.setVisible(is_human_tab)
        if hasattr(self, "_control_buttons_widget") and self._control_buttons_widget is not None:
            self._control_buttons_widget.setVisible(is_human_tab)
        if hasattr(self, "_telemetry_mode_group") and self._telemetry_mode_group is not None:
            self._telemetry_mode_group.setVisible(is_human_tab)

    def _apply_mode_from_tab(self, mode: ControlMode) -> None:
        index = self._mode_combo.findData(mode)
        if index < 0:
            current = self._current_game
            supported = tuple(self._available_modes.get(current, ())) if current is not None else ()
            if supported:
                mode = supported[0]
                index = self._mode_combo.findData(mode)
        if index >= 0:
            self._mode_combo.blockSignals(True)
            self._mode_combo.setCurrentIndex(index)
            self._mode_combo.blockSignals(False)
        self._emit_mode_changed(mode)

    def _connect_signals(self) -> None:
        if self._family_combo is not None:
            self._family_combo.currentIndexChanged.connect(self._on_family_changed)
        self._game_combo.currentIndexChanged.connect(self._on_game_changed)
        self._seed_spin.valueChanged.connect(lambda _: self._update_control_states())
        self._wire_mode_combo()
        # Note: Worker combo signals are now handled by SingleAgentTab

        self._load_button.clicked.connect(self._on_load_clicked)
        # Note: Training button signals are now handled by SingleAgentTab
        self._start_button.clicked.connect(self._on_start_clicked)
        self._pause_button.clicked.connect(self._on_pause_clicked)
        self._continue_button.clicked.connect(self._on_continue_clicked)
        self._terminate_button.clicked.connect(self._on_terminate_clicked)
        self._step_button.clicked.connect(self._on_step_clicked)
        self._reset_button.clicked.connect(self._on_reset_clicked)
        self._operator_combo.currentIndexChanged.connect(self._on_operator_selection_changed)

    def _update_telemetry_status_label(self) -> None:
        if not hasattr(self, "_telemetry_status_label"):
            return
        if self._fastlane_only_checkbox.isChecked():
            self._current_telemetry_mode = TelemetryModes.FASTLANE_ONLY
        elif getattr(self, "_dual_path_radio", None) is not None and self._dual_path_radio.isChecked():
            self._current_telemetry_mode = TelemetryModes.UI_AND_DB
        else:
            self._current_telemetry_mode = TelemetryModes.DB_ONLY

        descriptor = TELEMETRY_MODE_DESCRIPTORS.get(
            self._current_telemetry_mode,
            TelemetryModeDescriptor(
                name=self._current_telemetry_mode,
                label=self._current_telemetry_mode.replace("_", " ").title(),
                description="",
            ),
        )
        self._telemetry_status_label.setText(descriptor.label)

    def set_seed_value(self, seed: int) -> None:
        clamped = max(1, min(seed, self._seed_spin.maximum()))
        if self._seed_spin.value() == clamped:
            return
        self._seed_spin.blockSignals(True)
        self._seed_spin.setValue(clamped)
        self._seed_spin.blockSignals(False)
        self._update_control_states()

    # ------------------------------------------------------------------
    # Signal emitters
    # ------------------------------------------------------------------
    def _emit_mode_changed(self, mode: ControlMode) -> None:
        if self._current_mode != mode:
            self._current_mode = mode
            self._persist_mode_preference(mode)
        self.control_mode_changed.emit(mode)
        self._update_control_states()

    def _emit_game_changed(self, game: GameId | None) -> None:
        if game is None:
            return
        if self._current_game != game:
            self._current_game = game
        self.game_changed.emit(game)
        self.update_modes(game)
        self._refresh_game_config_ui()
        self.set_slippery_visible(game == GameId.FROZEN_LAKE)

    # ------------------------------------------------------------------
    # Qt slots
    # ------------------------------------------------------------------
    def _on_game_changed(self, index: int) -> None:
        game = self._game_combo.itemData(index)
        if isinstance(game, GameId):
            self._emit_game_changed(game)

    def _on_seed_reuse_changed(self, state: int) -> None:
        try:
            state_enum = QtCore.Qt.CheckState(state)
        except ValueError:
            state_enum = QtCore.Qt.CheckState.Unchecked
        enabled = state_enum == QtCore.Qt.CheckState.Checked
        self._allow_seed_reuse = enabled
        if enabled:
            hint = "Seeds auto-increment by default. Adjust before loading to reuse a previous seed."
        else:
            hint = "Seed increments automatically after each episode."
        self._seed_spin.setToolTip(hint)

    def _on_load_clicked(self) -> None:
        if self._current_game is None:
            return
        self.load_requested.emit(self._current_game, self._current_mode, self.current_seed())

    def _on_start_clicked(self) -> None:
        self._game_started = True
        self._game_paused = False
        self._update_control_states()
        self.start_game_requested.emit()

    def _on_pause_clicked(self) -> None:
        self._game_paused = True
        self._update_control_states()
        self.pause_game_requested.emit()

    def _on_continue_clicked(self) -> None:
        self._game_paused = False
        self._update_control_states()
        self.continue_game_requested.emit()

    def _on_terminate_clicked(self) -> None:
        self._game_started = False
        self._game_paused = False
        self._update_control_states()
        self.terminate_game_requested.emit()

    def _on_step_clicked(self) -> None:
        self.agent_step_requested.emit()

    def _on_reset_clicked(self) -> None:
        self._game_started = False
        self._game_paused = False
        self._update_control_states()
        self.reset_requested.emit(self.current_seed())

    def _on_slippery_toggled(self, state: int) -> None:
        try:
            state_enum = QtCore.Qt.CheckState(state)
        except ValueError:
            state_enum = QtCore.Qt.CheckState.Unchecked
        enabled = state_enum == QtCore.Qt.CheckState.Checked
        overrides = self._game_overrides.setdefault(GameId.FROZEN_LAKE, {})
        overrides["is_slippery"] = enabled
        self.slippery_toggled.emit(enabled)

    def _on_taxi_config_changed(self, param_name: str, value: bool) -> None:
        """Handle changes to Taxi configuration parameters."""
        overrides = self._game_overrides.setdefault(GameId.TAXI, {})
        overrides[param_name] = value
        self.taxi_config_changed.emit(param_name, value)

    def _on_cliff_config_changed(self, param_name: str, value: bool) -> None:
        """Handle changes to CliffWalking configuration parameters."""
        overrides = self._game_overrides.setdefault(GameId.CLIFF_WALKING, {})
        overrides[param_name] = value
        self.cliff_config_changed.emit(param_name, value)

    def _on_lunar_config_changed(self, param_name: str, value: object) -> None:
        """Handle changes to LunarLander configuration parameters."""
        overrides = self._game_overrides.setdefault(GameId.LUNAR_LANDER, {})
        overrides[param_name] = value
        self.lunar_config_changed.emit(param_name, value)

    def _on_car_config_changed(self, param_name: str, value: object) -> None:
        """Handle changes to CarRacing configuration parameters."""
        overrides = self._game_overrides.setdefault(GameId.CAR_RACING, {})
        overrides[param_name] = value
        self.car_config_changed.emit(param_name, value)

    def _on_bipedal_config_changed(self, param_name: str, value: object) -> None:
        """Handle changes to BipedalWalker configuration parameters."""
        overrides = self._game_overrides.setdefault(GameId.BIPEDAL_WALKER, {})
        overrides[param_name] = value
        self.bipedal_config_changed.emit(param_name, value)

    def _on_frozen_v2_config_changed(self, param_name: str, value: object) -> None:
        """Handle changes to FrozenLake-v2 configuration parameters."""
        overrides = self._game_overrides.setdefault(GameId.FROZEN_LAKE_V2, {})
        overrides[param_name] = value
        self.frozen_v2_config_changed.emit(param_name, value)

    def _resolve_minigrid_defaults(self, game_id: GameId) -> MiniGridConfig:
        stored = self._minigrid_defaults.get(game_id)
        if stored is not None:
            return stored
        return resolve_minigrid_default_config(game_id)

    def _on_minigrid_config_changed(self, param_name: str, value: object) -> None:
        if self._current_game not in MINIGRID_GAME_IDS:
            return
        current_game = self._current_game
        if current_game is None:
            return
        overrides = self._game_overrides.setdefault(current_game, {})
        overrides[param_name] = value
        config_bucket = self._game_configs.setdefault(current_game, {})
        config_bucket[param_name] = value

    # ------------------------------------------------------------------
    # UI state helpers
    # ------------------------------------------------------------------
    def _wire_mode_combo(self) -> None:
        self._mode_combo.currentIndexChanged.connect(self._on_mode_selection_changed)

    def _on_mode_selection_changed(self, index: int) -> None:
        mode = self._mode_combo.itemData(index)
        if not isinstance(mode, ControlMode):
            return
        if mode == self._current_mode:
            return
        self._current_mode = mode
        self._persist_mode_preference(mode)
        self._emit_mode_changed(mode)

    def _update_control_states(self) -> None:
        """Update button states based on game flow."""
        is_human = self._current_mode == ControlMode.HUMAN_ONLY

        # Start button: enabled only if game not started and environment loaded
        self._start_button.setEnabled(not self._game_started)

        # Pause button: enabled only if game started and not paused
        self._pause_button.setEnabled(self._game_started and not self._game_paused)

        # Continue button: enabled only if game paused
        self._continue_button.setEnabled(self._game_paused)

        # Terminate button: enabled only if game started
        self._terminate_button.setEnabled(self._game_started)

        # Agent Step: enabled only if game started, not paused, not human-only, and not auto-running
        self._step_button.setEnabled(
            self._game_started and not self._game_paused and not is_human and not self._auto_running
        )

        # Reset: always enabled (can reset even during active game)
        self._reset_button.setEnabled(True)

        # Enable operator selector only when an agent can participate
        has_agent_component = self._current_mode != ControlMode.HUMAN_ONLY
        agent_only_mode = self._current_mode == ControlMode.AGENT_ONLY
        self._operator_combo.setEnabled(has_agent_component and self._operator_combo.count() > 0)

        worker_def = self._current_worker_definition()
        supports_training = bool(worker_def and worker_def.supports_training)
        supports_policy = bool(worker_def and worker_def.supports_policy_load)
        self._train_agent_button.setEnabled(agent_only_mode and supports_training)
        self._trained_agent_button.setEnabled(agent_only_mode and supports_policy)
        self._resume_training_button.setEnabled(agent_only_mode and supports_training)

        self._update_operator_description()
        self._update_worker_description()

    def _apply_current_mode_selection(self) -> None:
        index = self._mode_combo.findData(self._current_mode)
        if index < 0:
            return
        self._mode_combo.blockSignals(True)
        self._mode_combo.setCurrentIndex(index)
        self._mode_combo.blockSignals(False)

    def _persist_mode_preference(self, mode: ControlMode) -> None:
        """Persist control mode preference to environment variable."""
        os.environ["GYM_CONTROL_MODE"] = mode.name.lower()

    def _load_mode_preference(self, fallback: ControlMode) -> ControlMode:
        """Load control mode preference from environment variable."""
        stored = os.environ.get("GYM_CONTROL_MODE")
        if stored is None:
            return fallback
        stored = stored.strip().upper()
        try:
            return ControlMode[stored]
        except KeyError:
            try:
                return ControlMode(stored.lower())
            except ValueError:
                return fallback

    def _populate_operator_combo(self) -> None:
        self._operator_combo.blockSignals(True)
        self._operator_combo.clear()
        for descriptor in self._operator_order:
            self._operator_combo.addItem(descriptor.display_name, descriptor.operator_id)
        self._operator_combo.blockSignals(False)

        if not self._operator_order:
            self._operator_description.setText("No operators registered")
            return

        default_id = self._active_operator_id or self._operator_order[0].operator_id
        index = self._operator_combo.findData(default_id)
        if index < 0:
            index = 0
        self._operator_combo.setCurrentIndex(index)
        current_data = self._operator_combo.currentData()
        self._active_operator_id = current_data if isinstance(current_data, str) else None
        self._update_operator_description()

    def _sync_worker_state(self) -> None:
        """Sync worker state from SingleAgentTab without repopulating.

        SingleAgentTab's WorkersSubTab handles its own combo population.
        This method just syncs the _current_worker_id state.
        """
        if not self._worker_definitions:
            self._current_worker_id = None
            self._update_control_states()
            return

        # Get current worker from SingleAgentTab
        worker_id = self._single_agent_tab.workers_subtab.current_worker_id
        self._current_worker_id = worker_id
        self._update_control_states()

    def _on_operator_selection_changed(self, index: int) -> None:
        operator_id = self._operator_combo.itemData(index)
        if not isinstance(operator_id, str):
            return
        if operator_id == self._active_operator_id:
            return
        self._active_operator_id = operator_id
        self._update_operator_description()
        self.operator_changed.emit(operator_id)

    def _on_worker_selection_changed(self, index: int) -> None:
        worker_id = self._worker_combo.itemData(index)
        if not isinstance(worker_id, str):
            worker_id = None
        if worker_id == self._current_worker_id:
            return
        self._current_worker_id = worker_id
        self._update_worker_description()
        self._update_control_states()
        if worker_id:
            self.worker_changed.emit(worker_id)

    def _emit_train_agent_requested(self) -> None:
        worker_id = self._current_worker_id
        if worker_id is None:
            return
        self.train_agent_requested.emit(worker_id)

    def _emit_resume_training_requested(self) -> None:
        worker_id = self._current_worker_id
        if worker_id is None:
            return
        self.resume_training_requested.emit(worker_id)

    def _emit_trained_agent_requested(self) -> None:
        worker_id = self._current_worker_id
        if worker_id is None:
            return
        self.trained_agent_requested.emit(worker_id)

    def _emit_custom_script_requested(self) -> None:
        worker_id = self._current_worker_id
        if worker_id is None:
            return
        self.custom_script_requested.emit(worker_id)

    def _update_operator_description(self) -> None:
        if self._active_operator_id is None:
            self._operator_description.setText("—")
            return
        descriptor = self._operator_descriptors.get(self._active_operator_id)
        if descriptor is None or descriptor.description is None:
            self._operator_description.setText("—")
            return
        self._operator_description.setText(descriptor.description)

    def _current_worker_definition(self) -> Optional[WorkerDefinition]:
        if self._current_worker_id is None:
            return None
        for definition in self._worker_definitions:
            if definition.worker_id == self._current_worker_id:
                return definition
        return None

    def _update_worker_description(self) -> None:
        definition = self._current_worker_definition()
        if definition is None:
            self._worker_description.setText(
                "No worker selected. Please choose an integration to enable training."
            )
            return
        capabilities = ", ".join(definition.capabilities()) or "No declared capabilities"
        self._worker_description.setText(f"{definition.description}\nCapabilities: {capabilities}")

    @staticmethod
    def _format_bool(value: bool) -> str:
        return "Yes" if value else "No"

    def _clear_config_layout(self) -> None:
        while self._config_layout.count():
            item = self._config_layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            layout = item.layout()
            if layout is not None:
                layout.deleteLater()

    def _build_input_mode_controls(self) -> None:
        """No-op. Input mode selection removed.

        All keyboard input now goes through HumanKeyboardRuntime worker
        subprocesses. The subprocess automatically handles both continuous
        (tick mode with held-key state) and turn-based (blocking mode)
        based on the environment's InteractionController.
        """
        pass

    def _refresh_game_config_ui(self) -> None:
        self._clear_config_layout()
        if self._frozen_slippery_checkbox is not None:
            try:
                self._frozen_slippery_checkbox.deleteLater()
            except RuntimeError:
                pass
            self._frozen_slippery_checkbox = None

        if self._current_game is None:
            label = QtWidgets.QLabel(
                "Select an environment to view configuration options.",
                self._config_group,
            )
            label.setWordWrap(True)
            self._config_layout.addRow("", label)
            return

        # -------- Input Mode (common to ALL games) --------
        self._build_input_mode_controls()

        # Add a separator line
        separator = QtWidgets.QFrame(self._config_group)
        separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self._config_layout.addRow(separator)

        if self._current_game == GameId.FROZEN_LAKE:
            overrides = self._game_overrides.setdefault(GameId.FROZEN_LAKE, {})
            checkbox = build_frozenlake_controls(
                layout=self._config_layout,
                group=self._config_group,
                overrides=overrides,
                config=self._config.frozen_lake_config,
                on_slippery_toggled=self._on_slippery_toggled,
            )
            self._frozen_slippery_checkbox = checkbox
        elif self._current_game is not None and self._current_game in MINIGRID_GAME_IDS:
            current_game = self._current_game
            overrides = self._game_overrides.setdefault(current_game, {})
            defaults = self._resolve_minigrid_defaults(current_game)
            callbacks = MinigridControlCallbacks(on_change=self._on_minigrid_config_changed)
            build_minigrid_controls(
                parent=self._config_group,
                layout=self._config_layout,
                game_id=current_game,
                overrides=overrides,
                defaults=defaults,
                callbacks=callbacks,
            )
        elif self._current_game is not None and self._current_game in ALE_GAME_IDS:
            current_game = self._current_game
            overrides = self._game_overrides.setdefault(current_game, {})
            callbacks = ALEControlCallbacks(on_change=self._on_ale_config_changed)
            # Provide a lightweight defaults object consistent with current selection
            defaults = ALEConfig(env_id=current_game.value)
            build_ale_controls(
                parent=self._config_group,
                layout=self._config_layout,
                game_id=current_game,
                overrides=overrides,
                defaults=defaults,
                callbacks=callbacks,
            )
        elif self._current_game == GameId.FROZEN_LAKE_V2:
            overrides = self._game_overrides.setdefault(GameId.FROZEN_LAKE_V2, {})
            build_frozenlake_v2_controls(
                layout=self._config_layout,
                group=self._config_group,
                overrides=overrides,
                on_change=self._on_frozen_v2_config_changed,
            )
        elif self._current_game == GameId.LUNAR_LANDER:
            overrides = self._game_overrides.setdefault(GameId.LUNAR_LANDER, {})
            build_lunarlander_controls(
                layout=self._config_layout,
                group=self._config_group,
                overrides=overrides,
                config=self._config.lunar_lander_config,
                on_change=self._on_lunar_config_changed,
            )
        elif self._current_game == GameId.TAXI:
            overrides = self._game_overrides.setdefault(GameId.TAXI, {})
            build_taxi_controls(
                layout=self._config_layout,
                group=self._config_group,
                overrides=overrides,
                config=self._config.taxi_config,
                on_change=self._on_taxi_config_changed,
            )
        elif self._current_game == GameId.CLIFF_WALKING:
            overrides = self._game_overrides.setdefault(GameId.CLIFF_WALKING, {})
            build_cliff_controls(
                layout=self._config_layout,
                group=self._config_group,
                overrides=overrides,
                config=self._config.cliff_walking_config,
                on_change=self._on_cliff_config_changed,
            )
        elif self._current_game == GameId.CAR_RACING:
            overrides = self._game_overrides.setdefault(GameId.CAR_RACING, {})
            build_car_racing_controls(
                layout=self._config_layout,
                group=self._config_group,
                overrides=overrides,
                config=self._config.car_racing_config,
                on_change=self._on_car_config_changed,
            )
        elif self._current_game == GameId.BIPEDAL_WALKER:
            overrides = self._game_overrides.setdefault(GameId.BIPEDAL_WALKER, {})
            build_bipedal_controls(
                layout=self._config_layout,
                group=self._config_group,
                overrides=overrides,
                config=self._config.bipedal_walker_config,
                on_change=self._on_bipedal_config_changed,
            )
        elif self._current_game is not None and self._current_game in VIZDOOM_GAME_IDS:
            current_game = self._current_game
            overrides = self._game_overrides.setdefault(current_game, {})
            callbacks = ViZDoomControlCallbacks(on_change=self._on_vizdoom_config_changed)
            defaults = ViZDoomConfig()
            build_vizdoom_controls(
                parent=self._config_group,
                layout=self._config_layout,
                game_id=current_game,
                overrides=overrides,
                defaults=defaults,
                callbacks=callbacks,
            )
        elif self._current_game is not None and self._current_game in PROCGEN_GAME_IDS:
            current_game = self._current_game
            overrides = self._game_overrides.setdefault(current_game, {})
            callbacks = ProcgenControlCallbacks(on_change=self._on_procgen_config_changed)
            defaults = ProcgenConfig()
            build_procgen_controls(
                parent=self._config_group,
                layout=self._config_layout,
                game_id=current_game,
                overrides=overrides,
                defaults=defaults,
                callbacks=callbacks,
            )
        elif self._current_game is not None and self._current_game in MELTINGPOT_GAME_IDS:
            current_game = self._current_game
            overrides = self._game_overrides.setdefault(current_game, {})
            callbacks = MeltingPotControlCallbacks(on_change=self._on_meltingpot_config_changed)
            defaults = MeltingPotConfig()
            build_meltingpot_controls(
                parent=self._config_group,
                layout=self._config_layout,
                game_id=current_game,
                overrides=overrides,
                defaults=defaults,
                callbacks=callbacks,
            )
        elif self._current_game is not None and self._current_game in MULTIGRID_GAME_IDS:
            current_game = self._current_game
            overrides = self._game_overrides.setdefault(current_game, {})
            callbacks = MultiGridControlCallbacks(on_change=self._on_multigrid_config_changed)
            defaults = MultiGridConfig()
            build_multigrid_controls(
                parent=self._config_group,
                layout=self._config_layout,
                game_id=current_game,
                overrides=overrides,
                defaults=defaults,
                callbacks=callbacks,
            )
        elif self._current_game is not None and self._current_game in ALL_SMAC_GAME_IDS:
            current_game = self._current_game
            overrides = self._game_overrides.setdefault(current_game, {})
            callbacks = SMACControlCallbacks(on_change=self._on_smac_config_changed)
            defaults = SMACConfig()
            build_smac_controls(
                parent=self._config_group,
                layout=self._config_layout,
                game_id=current_game,
                overrides=overrides,
                defaults=defaults,
                callbacks=callbacks,
            )
        elif self._current_game is not None and self._current_game in ALL_RWARE_GAME_IDS:
            current_game = self._current_game
            overrides = self._game_overrides.setdefault(current_game, {})
            build_rware_controls(
                parent=self._config_group,
                layout=self._config_layout,
                game_id=current_game,
                overrides=overrides,
                on_change=self._on_rware_config_changed,
            )
        elif self._current_game is not None and self._current_game in MALMOENV_FAMILY:
            current_game = self._current_game
            overrides = self._game_overrides.setdefault(current_game, {})
            # Set default values
            for key, value in DEFAULT_MALMO_CONFIG.items():
                overrides.setdefault(key, value)
            build_malmo_controls(
                layout=self._config_layout,
                group=self._config_group,
                overrides=overrides,
                on_change=self._on_malmo_config_changed,
            )
        else:
            label = QtWidgets.QLabel(
                "No additional configuration options for this environment.",
                self._config_group,
            )
            label.setWordWrap(True)
            self._config_layout.addRow("", label)

    def _on_ale_config_changed(self, param_name: str, value: object) -> None:
        current_game = self._current_game
        if current_game is None:
            return
        overrides = self._game_overrides.setdefault(current_game, {})
        overrides[param_name] = value

    def _on_vizdoom_config_changed(self, param_name: str, value: object) -> None:
        current_game = self._current_game
        if current_game is None:
            return
        overrides = self._game_overrides.setdefault(current_game, {})
        overrides[param_name] = value
        self.vizdoom_config_changed.emit(param_name, value)

    def _on_procgen_config_changed(self, param_name: str, value: object) -> None:
        current_game = self._current_game
        if current_game is None:
            return
        overrides = self._game_overrides.setdefault(current_game, {})
        overrides[param_name] = value

    def _on_meltingpot_config_changed(self, param_name: str, value: object) -> None:
        current_game = self._current_game
        if current_game is None:
            return
        overrides = self._game_overrides.setdefault(current_game, {})
        overrides[param_name] = value

    def _on_multigrid_config_changed(self, param_name: str, value: object) -> None:
        current_game = self._current_game
        if current_game is None:
            return
        overrides = self._game_overrides.setdefault(current_game, {})
        overrides[param_name] = value

    def _on_smac_config_changed(self, param_name: str, value: object) -> None:
        current_game = self._current_game
        if current_game is None:
            return
        overrides = self._game_overrides.setdefault(current_game, {})
        overrides[param_name] = value

    def _on_rware_config_changed(self, overrides: Dict[str, Any]) -> None:
        """Handle RWARE config changes (overrides dict already mutated in-place)."""
        pass  # overrides dict is shared; mutation already applied by build_rware_controls

    def _on_malmo_config_changed(self, param_name: str, value: object) -> None:
        """Handle MalmoEnv config changes."""
        current_game = self._current_game
        if current_game is None:
            return
        overrides = self._game_overrides.setdefault(current_game, {})
        overrides[param_name] = value

    # ------------------------------------------------------------------
    # Multi-Agent Tab Handlers
    # ------------------------------------------------------------------
    def _on_multi_agent_train_requested(self, worker_id: str) -> None:
        """Handle train request from Multi-Agent Cooperation/Competition tab."""
        self._current_worker_id = worker_id
        self.worker_changed.emit(worker_id)
        self.train_agent_requested.emit(worker_id)

    def _on_multi_agent_evaluate_requested(self, worker_id: str) -> None:
        """Handle evaluate/policy load request from Multi-Agent Cooperation/Competition tab."""
        self._current_worker_id = worker_id
        self.worker_changed.emit(worker_id)
        self.trained_agent_requested.emit(worker_id)

    def _on_multi_agent_resume_requested(self, worker_id: str) -> None:
        """Handle resume training request from Multi-Agent Cooperation/Competition tab."""
        self._current_worker_id = worker_id
        self.worker_changed.emit(worker_id)
        self.resume_training_requested.emit(worker_id)

    def _on_multi_agent_script_requested(self, worker_id: str) -> None:
        """Handle custom script request from Multi-Agent Cooperation/Competition tab."""
        self._current_worker_id = worker_id
        self.worker_changed.emit(worker_id)
        self.custom_script_requested.emit(worker_id)

    def _on_multi_agent_load_requested(self, env_id: str, seed: int) -> None:
        """Handle environment load request from Multi-Agent tab (Human vs Agent mode)."""
        self.multi_agent_load_requested.emit(env_id, seed)

    # ------------------------------------------------------------------
    # Property accessors
    # ------------------------------------------------------------------
    @property
    def operators_tab(self) -> OperatorsTab:
        """Get the Operators tab widget."""
        return self._operators_tab

    @property
    def single_agent_tab(self) -> SingleAgentTab:
        """Get the Single-Agent tab widget."""
        return self._single_agent_tab

    @property
    def multi_agent_tab(self) -> MultiAgentTab:
        """Get the Multi-Agent tab widget."""
        return self._multi_agent_tab
