"""ViZDoom adapters for Doom-based reinforcement learning scenarios."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from gymnasium import spaces

from gym_gui.core.adapters.base import AdapterContext, AdapterStep, EnvironmentAdapter, StepState
from gym_gui.core.enums import ControlMode, GameId, RenderMode

_LOGGER = logging.getLogger(__name__)


def _ensure_vizdoom():
    """Import ViZDoom lazily and raise a helpful error when missing."""

    try:
        import vizdoom as vzd  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "ViZDoom is not installed. Install via 'pip install -r requirements/vizdoom.txt' "
            "or 'pip install -e .[vizdoom]' to enable Doom environments."
        ) from exc
    return vzd


@dataclass(slots=True)
class ViZDoomConfig:
    """Configuration knobs for ViZDoom scenarios."""

    screen_resolution: str = "RES_640X480"
    screen_format: str = "RGB24"
    render_hud: bool = True
    render_weapon: bool = True
    render_crosshair: bool = False
    render_particles: bool = True
    render_decals: bool = True
    episode_timeout: int = 0  # 0 = no timeout (play until death)
    living_reward: float = 0.0
    death_penalty: float = 100.0
    sound_enabled: bool = False
    depth_buffer: bool = False
    labels_buffer: bool = False
    automap_buffer: bool = False
    # Mouse/delta control options
    enable_mouse_delta: bool = True  # Enable delta buttons for FPS-style mouse
    mouse_sensitivity_x: float = 1.0  # Horizontal turn speed multiplier
    mouse_sensitivity_y: float = 1.0  # Vertical look speed multiplier
    freelook: bool = True  # Enable vertical look (requires +freelook 1)


class ViZDoomAdapter(EnvironmentAdapter[np.ndarray, Sequence[int]]):
    """Shared ViZDoom adapter that powers concrete scenario variants."""

    default_render_mode = RenderMode.RGB_ARRAY
    supported_render_modes = (RenderMode.RGB_ARRAY,)
    supported_control_modes = (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
    )

    # Sub-classes must override scenario metadata
    _scenario_file: str = "basic.cfg"
    _available_buttons: Sequence[str] = ()
    _available_game_variables: Sequence[str] = ()

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: ViZDoomConfig | None = None,
    ) -> None:
        super().__init__(context)
        self._config = config or ViZDoomConfig()
        self._game: Any | None = None
        self._step_counter = 0
        self._episode_return = 0.0
        self._last_observation: np.ndarray | None = None
        self._action_space = spaces.MultiBinary(len(self._available_buttons) or 3)
        h, w = self._resolution_to_shape(self._config.screen_resolution)
        self._observation_space = spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8)
        # Delta button indices (set during load if delta buttons are enabled)
        self._turn_delta_index: int | None = None
        self._look_delta_index: int | None = None
        # Pending mouse delta to apply on next step
        self._pending_mouse_delta: tuple[float, float] = (0.0, 0.0)

    # ------------------------------------------------------------------
    # EnvironmentAdapter overrides
    # ------------------------------------------------------------------

    def load(self) -> None:  # type: ignore[override]
        vzd = _ensure_vizdoom()
        self._game = vzd.DoomGame()

        # Write _vizdoom.ini into var/data/vizdoom/ instead of the repo root.
        from gym_gui.config.paths import VAR_DATA_DIR
        vizdoom_dir = VAR_DATA_DIR / "vizdoom"
        vizdoom_dir.mkdir(parents=True, exist_ok=True)
        self._game.set_doom_config_path(str(vizdoom_dir / "_vizdoom.ini"))

        scenario_path = os.path.join(vzd.scenarios_path, self._scenario_file)
        self._game.load_config(scenario_path)

        self._apply_screen_settings(vzd)
        self._apply_render_flags()
        self._apply_reward_shaping()
        self._configure_buffers()
        self._configure_buttons(vzd)
        self._configure_mode(vzd)

        self._game.init()
        # Update action/observation spaces with actual runtime data
        self._action_space = spaces.MultiBinary(self._game.get_available_buttons_size())
        initial_state = self._game.get_state()
        if initial_state is not None and initial_state.screen_buffer is not None:
            obs = np.asarray(initial_state.screen_buffer)
            self._observation_space = spaces.Box(
                low=0,
                high=255,
                shape=obs.shape,
                dtype=obs.dtype,
            )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> AdapterStep[np.ndarray]:  # type: ignore[override]
        del options
        game = self._require_game()
        if seed is not None:
            game.set_seed(int(seed))
        game.new_episode()
        self._step_counter = 0
        self._episode_return = 0.0
        return self._package_step(*self._observe_step(game, reward=0.0, terminated=False))

    def step(self, action: Sequence[int] | int) -> AdapterStep[np.ndarray]:  # type: ignore[override]
        game = self._require_game()
        num_buttons = game.get_available_buttons_size()

        # Handle both int (from Qt shortcuts) and Sequence[int] (from agent/MultiBinary)
        if isinstance(action, int):
            # -1 is a special sentinel for NOOP (no buttons pressed, used by idle tick)
            # Other ints are button indices: action=2 -> [0, 0, 1, 0, ...]
            cmd = [0.0] * num_buttons  # Use floats to support delta button values
            if action >= 0 and action < num_buttons:
                cmd[action] = 1.0
            # If action == -1, cmd stays all zeros (NOOP)
        else:
            cmd = [float(v) for v in self._coerce_action(action, num_buttons)]

        # Apply any pending mouse delta to the delta button slots
        delta_x, delta_y = self._pending_mouse_delta
        if (delta_x != 0.0 or delta_y != 0.0):
            if self._turn_delta_index is not None and self._turn_delta_index < num_buttons:
                cmd[self._turn_delta_index] = delta_x
            if self._look_delta_index is not None and self._look_delta_index < num_buttons:
                cmd[self._look_delta_index] = delta_y
            # Clear pending delta
            self._pending_mouse_delta = (0.0, 0.0)

        reward = float(game.make_action(cmd))

        self._step_counter += 1
        self._episode_return += reward
        terminated = bool(game.is_episode_finished())
        return self._package_step(*self._observe_step(game, reward=reward, terminated=terminated))

    def render(self) -> Mapping[str, Any]:  # type: ignore[override]
        frame = self._last_observation
        if frame is None:
            h, w = self._resolution_to_shape(self._config.screen_resolution)
            frame = np.zeros((h, w, 3), dtype=np.uint8)
        return {
            "mode": RenderMode.RGB_ARRAY.value,
            "rgb": frame,
            "game_id": getattr(self, "id", "vizdoom"),
        }

    def close(self) -> None:  # type: ignore[override]
        if self._game is not None:
            self._game.close()
            self._game = None
        super().close()

    # ------------------------------------------------------------------
    # Adapter-specific helpers
    # ------------------------------------------------------------------

    def build_step_state(self, observation: np.ndarray, info: Mapping[str, Any]) -> StepState:  # type: ignore[override]
        metrics = {
            "step": self._step_counter,
            "episode_return": float(self._episode_return),
            "tic": info.get("tic", 0),
        }
        environment = {
            "scenario": self._scenario_file,
            "health": info.get("health", 0),
            "ammo": info.get("ammo2", 0),
            "kills": info.get("killcount", 0),
        }
        return StepState(metrics=metrics, environment=environment, raw=dict(info))

    @property
    def action_space(self):  # type: ignore[override]
        return self._action_space

    @property
    def observation_space(self):  # type: ignore[override]
        return self._observation_space

    def apply_mouse_delta(self, delta_x: float, delta_y: float) -> None:
        """Queue mouse movement to be applied on the next step.

        Args:
            delta_x: Horizontal movement (positive = turn right, negative = turn left).
                     Value is in degrees of rotation.
            delta_y: Vertical movement (positive = look down, negative = look up).
                     Value is in degrees of rotation.
        """
        # Scale by sensitivity
        scaled_x = delta_x * self._config.mouse_sensitivity_x
        scaled_y = delta_y * self._config.mouse_sensitivity_y
        # Accumulate (in case multiple deltas before next step)
        current_x, current_y = self._pending_mouse_delta
        self._pending_mouse_delta = (current_x + scaled_x, current_y + scaled_y)

    def has_mouse_delta_support(self) -> bool:
        """Return True if delta buttons are configured for mouse control."""
        return self._turn_delta_index is not None

    def get_delta_button_indices(self) -> tuple[int | None, int | None]:
        """Return (turn_delta_index, look_delta_index) for direct access."""
        return self._turn_delta_index, self._look_delta_index

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _require_game(self):
        if self._game is None:
            raise RuntimeError("ViZDoom adapter has not been loaded. Call load() first.")
        return self._game

    def _observe_step(
        self,
        game: Any,
        *,
        reward: float,
        terminated: bool,
    ) -> tuple[np.ndarray, float, bool, bool, Mapping[str, Any]]:
        state = game.get_state()
        if state is not None and state.screen_buffer is not None:
            obs = np.asarray(state.screen_buffer)
        else:
            obs = np.zeros(self._observation_space.shape, dtype=np.uint8)
        self._last_observation = obs
        info = self._build_info(state)
        return obs, reward, terminated, False, info

    def _build_info(self, state: Any) -> Mapping[str, Any]:
        if state is None:
            return {}
        info: dict[str, Any] = {
            "tic": getattr(state, "tic", 0),
            "frame_number": getattr(state, "number", 0),
        }
        if state.game_variables is not None:
            for idx, name in enumerate(self._available_game_variables):
                if idx < len(state.game_variables):
                    info[name.lower()] = state.game_variables[idx]
        if self._context and self._context.control_mode == ControlMode.HUMAN_ONLY:
            info["last_action"] = self._game.get_last_action()
        return info

    def _apply_screen_settings(self, vzd: Any) -> None:
        res_name = self._config.screen_resolution
        fmt_name = self._config.screen_format
        resolution = getattr(vzd.ScreenResolution, res_name, vzd.ScreenResolution.RES_640X480)
        screen_format = getattr(vzd.ScreenFormat, fmt_name, vzd.ScreenFormat.RGB24)
        self._game.set_screen_resolution(resolution)
        self._game.set_screen_format(screen_format)

    def _apply_render_flags(self) -> None:
        self._game.set_render_hud(self._config.render_hud)
        self._game.set_render_weapon(self._config.render_weapon)
        self._game.set_render_crosshair(self._config.render_crosshair)
        self._game.set_render_particles(self._config.render_particles)
        self._game.set_render_decals(self._config.render_decals)

    def _apply_reward_shaping(self) -> None:
        self._game.set_episode_timeout(self._config.episode_timeout)
        self._game.set_living_reward(self._config.living_reward)
        self._game.set_death_penalty(self._config.death_penalty)

    def _configure_buffers(self) -> None:
        self._game.set_depth_buffer_enabled(self._config.depth_buffer)
        self._game.set_labels_buffer_enabled(self._config.labels_buffer)
        self._game.set_automap_buffer_enabled(self._config.automap_buffer)

    def _configure_buttons(self, vzd: Any) -> None:
        if self._available_buttons:
            self._game.clear_available_buttons()
            for button_name in self._available_buttons:
                button = getattr(vzd.Button, button_name)
                self._game.add_available_button(button)

        # Add delta buttons for smooth mouse control in human modes
        if self._config.enable_mouse_delta:
            control_mode = self._context.control_mode if self._context else ControlMode.AGENT_ONLY
            if control_mode in (ControlMode.HUMAN_ONLY, ControlMode.HYBRID_TURN_BASED, ControlMode.HYBRID_HUMAN_AGENT):
                # Track indices for delta buttons (appended after regular buttons)
                base_button_count = self._game.get_available_buttons_size()
                # Add horizontal turn delta
                self._game.add_available_button(vzd.Button.TURN_LEFT_RIGHT_DELTA)
                self._turn_delta_index = base_button_count
                # Add vertical look delta
                self._game.add_available_button(vzd.Button.LOOK_UP_DOWN_DELTA)
                self._look_delta_index = base_button_count + 1
                _LOGGER.debug(
                    "Added delta buttons: turn_delta=%d, look_delta=%d",
                    self._turn_delta_index,
                    self._look_delta_index,
                )

    def _configure_mode(self, vzd: Any) -> None:
        # Always use PLAYER mode - human input comes via Qt shortcuts, not native window
        # SPECTATOR mode opens a native window that captures mouse (problematic)
        self._game.set_mode(vzd.Mode.PLAYER)
        # Always hide native window - we render via gym_gui's RGB viewer
        self._game.set_window_visible(False)
        self._game.set_sound_enabled(self._config.sound_enabled)
        # Enable freelook for vertical mouse look (up/down)
        if self._config.freelook and self._config.enable_mouse_delta:
            self._game.add_game_args("+freelook 1")

    def _coerce_action(self, action: Sequence[int], size: int) -> list[int]:
        if len(action) < size:
            padded = list(action) + [0] * (size - len(action))
            return padded
        if len(action) > size:
            return list(action[:size])
        return list(action)

    @staticmethod
    def _resolution_to_shape(resolution: str) -> tuple[int, int]:
        if "RES_" in resolution:
            _, spec = resolution.split("RES_", maxsplit=1)
        else:
            spec = resolution
        if "X" in spec:
            try:
                width = int(spec.split("X")[0])
                height = int(spec.split("X")[1])
                return height, width
            except ValueError:
                pass
        # Fallback to 480p landscape
        return 480, 640


# Concrete scenario adapters -------------------------------------------------


class ViZDoomBasicAdapter(ViZDoomAdapter):
    id = GameId.VIZDOOM_BASIC.value
    _scenario_file = "basic.cfg"
    _available_buttons = ("ATTACK", "MOVE_LEFT", "MOVE_RIGHT")
    _available_game_variables = ("AMMO2",)


class ViZDoomDeadlyCorridorAdapter(ViZDoomAdapter):
    id = GameId.VIZDOOM_DEADLY_CORRIDOR.value
    _scenario_file = "deadly_corridor.cfg"
    _available_buttons = (
        "ATTACK",
        "MOVE_LEFT",
        "MOVE_RIGHT",
        "MOVE_FORWARD",
        "TURN_LEFT",
        "TURN_RIGHT",
    )
    _available_game_variables = ("HEALTH", "AMMO2")


class ViZDoomDefendTheCenterAdapter(ViZDoomAdapter):
    id = GameId.VIZDOOM_DEFEND_THE_CENTER.value
    _scenario_file = "defend_the_center.cfg"
    _available_buttons = ("ATTACK", "TURN_LEFT", "TURN_RIGHT")
    _available_game_variables = ("HEALTH", "AMMO2")


class ViZDoomDefendTheLineAdapter(ViZDoomAdapter):
    id = GameId.VIZDOOM_DEFEND_THE_LINE.value
    _scenario_file = "defend_the_line.cfg"
    _available_buttons = ("ATTACK", "TURN_LEFT", "TURN_RIGHT")
    _available_game_variables = ("HEALTH", "AMMO2")


class ViZDoomHealthGatheringAdapter(ViZDoomAdapter):
    id = GameId.VIZDOOM_HEALTH_GATHERING.value
    _scenario_file = "health_gathering.cfg"
    _available_buttons = ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT")
    _available_game_variables = ("HEALTH",)


class ViZDoomHealthGatheringSupremeAdapter(ViZDoomAdapter):
    id = GameId.VIZDOOM_HEALTH_GATHERING_SUPREME.value
    _scenario_file = "health_gathering_supreme.cfg"
    _available_buttons = ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT")
    _available_game_variables = ("HEALTH",)


class ViZDoomMyWayHomeAdapter(ViZDoomAdapter):
    id = GameId.VIZDOOM_MY_WAY_HOME.value
    _scenario_file = "my_way_home.cfg"
    _available_buttons = ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT")
    _available_game_variables = ()


class ViZDoomPredictPositionAdapter(ViZDoomAdapter):
    id = GameId.VIZDOOM_PREDICT_POSITION.value
    _scenario_file = "predict_position.cfg"
    _available_buttons = ("ATTACK", "TURN_LEFT", "TURN_RIGHT")
    _available_game_variables = ("AMMO2",)


class ViZDoomTakeCoverAdapter(ViZDoomAdapter):
    id = GameId.VIZDOOM_TAKE_COVER.value
    _scenario_file = "take_cover.cfg"
    _available_buttons = ("MOVE_LEFT", "MOVE_RIGHT")
    _available_game_variables = ("HEALTH",)


class ViZDoomDeathmatchAdapter(ViZDoomAdapter):
    id = GameId.VIZDOOM_DEATHMATCH.value
    _scenario_file = "deathmatch.cfg"
    _available_buttons = (
        "ATTACK",
        "USE",
        "MOVE_FORWARD",
        "MOVE_BACKWARD",
        "MOVE_LEFT",
        "MOVE_RIGHT",
        "TURN_LEFT",
        "TURN_RIGHT",
    )
    _available_game_variables = ("HEALTH", "AMMO2", "ARMOR", "KILLCOUNT")


VIZDOOM_ADAPTERS: dict[GameId, type[ViZDoomAdapter]] = {
    GameId.VIZDOOM_BASIC: ViZDoomBasicAdapter,
    GameId.VIZDOOM_DEADLY_CORRIDOR: ViZDoomDeadlyCorridorAdapter,
    GameId.VIZDOOM_DEFEND_THE_CENTER: ViZDoomDefendTheCenterAdapter,
    GameId.VIZDOOM_DEFEND_THE_LINE: ViZDoomDefendTheLineAdapter,
    GameId.VIZDOOM_HEALTH_GATHERING: ViZDoomHealthGatheringAdapter,
    GameId.VIZDOOM_HEALTH_GATHERING_SUPREME: ViZDoomHealthGatheringSupremeAdapter,
    GameId.VIZDOOM_MY_WAY_HOME: ViZDoomMyWayHomeAdapter,
    GameId.VIZDOOM_PREDICT_POSITION: ViZDoomPredictPositionAdapter,
    GameId.VIZDOOM_TAKE_COVER: ViZDoomTakeCoverAdapter,
    GameId.VIZDOOM_DEATHMATCH: ViZDoomDeathmatchAdapter,
}

__all__ = [
    "ViZDoomAdapter",
    "ViZDoomConfig",
    "VIZDOOM_ADAPTERS",
    "ViZDoomBasicAdapter",
    "ViZDoomDeadlyCorridorAdapter",
    "ViZDoomDefendTheCenterAdapter",
    "ViZDoomDefendTheLineAdapter",
    "ViZDoomHealthGatheringAdapter",
    "ViZDoomHealthGatheringSupremeAdapter",
    "ViZDoomMyWayHomeAdapter",
    "ViZDoomPredictPositionAdapter",
    "ViZDoomTakeCoverAdapter",
    "ViZDoomDeathmatchAdapter",
]
