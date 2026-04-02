"""Jumanji environment adapters for JAX-based logic puzzle environments.

Jumanji is a suite of JAX-based reinforcement learning environments that
provides logic puzzle games like 2048, Minesweeper, Rubik's Cube, Sudoku,
and more. It leverages JAX for hardware acceleration and JIT compilation.

Repository: https://github.com/google-deepmind/jumanji
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np

from gym_gui.config.game_configs import JumanjiConfig
from gym_gui.core.adapters.base import (
    AdapterContext,
    AdapterStep,
    EnvironmentAdapter,
    StepState,
)
from gym_gui.core.enums import ControlMode, GameId, RenderMode

_LOGGER = logging.getLogger(__name__)

# Try to import jumanji_worker's gymnasium adapter
try:
    from jumanji_worker.gymnasium_adapter import make_jumanji_gym_env
    _JUMANJI_AVAILABLE = True
except ImportError:
    _JUMANJI_AVAILABLE = False
    make_jumanji_gym_env = None  # type: ignore[assignment]


_JUMANJI_STEP_LOG_FREQUENCY = 50

# Jumanji Logic environment names
JUMANJI_ENV_NAMES = [
    "Game2048-v1",
    "Minesweeper-v0",
    "RubiksCube-v0",
    "SlidingTilePuzzle-v0",
    "Sudoku-v0",
    "GraphColoring-v1",
]

# Game2048 actions (4 directions)
GAME2048_ACTIONS = ["up", "down", "left", "right"]

# Minesweeper actions are cell indices (varies by grid size)
# RubiksCube actions are face rotations (12 total: 6 faces x 2 directions)
RUBIKS_CUBE_ACTIONS = [
    "R", "R'", "L", "L'", "U", "U'",
    "D", "D'", "F", "F'", "B", "B'",
]


def _ensure_jumanji() -> None:
    """Ensure Jumanji worker is available, raise helpful error if not."""
    if not _JUMANJI_AVAILABLE:
        raise ImportError(
            "Jumanji worker is not available. Install via:\n"
            "  pip install jax jaxlib jumanji\n"
            "or:\n"
            "  pip install -e '3rd_party/workers/jumanji_worker[jax]'\n"
            "Note: Requires JAX with compatible hardware backend."
        )


class JumanjiAdapter(EnvironmentAdapter[np.ndarray, int]):
    """Base adapter for Jumanji JAX-based logic puzzle environments.

    Jumanji environments provide structured observations (often dict/dataclass)
    and have discrete action spaces. The gymnasium_adapter from jumanji_worker
    provides Gymnasium-compatible wrappers.
    """

    default_render_mode = RenderMode.RGB_ARRAY
    supported_render_modes = (RenderMode.RGB_ARRAY,)
    supported_control_modes = (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    )

    # Subclasses override with their specific environment ID
    _env_id: str = "Game2048-v1"

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: JumanjiConfig | None = None,
    ) -> None:
        super().__init__(context)
        if config is None:
            config = JumanjiConfig(env_id=f"jumanji/{self._env_id}")
        self._config = config
        self._env: Any = None
        self._step_counter = 0
        self._episode_return = 0.0
        self._render_warning_emitted = False

    @property
    def id(self) -> str:
        return f"jumanji/{self._env_id}"

    def load(self) -> None:
        """Initialize the Jumanji environment via gymnasium adapter."""
        _ensure_jumanji()

        try:
            # Use jumanji_worker's gymnasium adapter
            env = make_jumanji_gym_env(
                env_id=self._env_id,
                seed=self._config.seed or 0,
                flatten_obs=self._config.flatten_obs,
                backend=self._config.backend,
            )
            self._env = env

            _LOGGER.debug(
                "Jumanji environment loaded: %s (seed=%s, flatten=%s)",
                self._env_id,
                self._config.seed,
                self._config.flatten_obs,
            )
        except Exception as exc:
            _LOGGER.error(
                "Failed to load Jumanji environment %s: %s",
                self._env_id, exc
            )
            raise

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> AdapterStep[np.ndarray]:
        """Reset the environment and return initial observation."""
        env = self._require_env()

        # Reset with optional seed
        obs, info = env.reset(seed=seed)
        processed_obs = self._process_observation(obs)

        self._step_counter = 0
        self._episode_return = 0.0

        _LOGGER.debug(
            "Jumanji environment reset: %s (seed=%s)",
            self._env_id, seed
        )

        return self._package_step(processed_obs, 0.0, False, False, info)

    def step(self, action: int) -> AdapterStep[np.ndarray]:
        """Execute one step in the environment."""
        env = self._require_env()

        obs, reward, terminated, truncated, info = env.step(action)
        processed_obs = self._process_observation(obs)

        self._step_counter += 1
        self._episode_return += float(reward)

        if self._step_counter % _JUMANJI_STEP_LOG_FREQUENCY == 0:
            _LOGGER.debug(
                "Jumanji step %d: action=%d, reward=%.3f, episode_return=%.3f",
                self._step_counter, action, reward, self._episode_return
            )

        return self._package_step(
            processed_obs, float(reward), terminated, truncated, info
        )

    def render(self) -> dict[str, Any]:
        """Render the environment and return RGB array.

        Jumanji's viewer returns RGBA arrays (4 channels) from matplotlib.
        We convert to RGB (3 channels) for consistency with other adapters.

        The viewer is configured with render_mode='rgb_array' at environment
        creation time to avoid matplotlib popup windows.
        """
        env = self._require_env()

        try:
            # Jumanji wrapper ignores the mode arg but we pass it for clarity
            frame = env.render()
            if frame is None:
                # Some Jumanji envs may not support rendering
                return {
                    "mode": RenderMode.RGB_ARRAY.value,
                    "rgb": None,
                    "game_id": self.id,
                }
            array = np.asarray(frame)

            # Convert RGBA to RGB if needed (Jumanji uses matplotlib which outputs RGBA)
            if array.ndim == 3 and array.shape[2] == 4:
                array = array[:, :, :3]  # Drop alpha channel

            return {
                "mode": RenderMode.RGB_ARRAY.value,
                "rgb": array,
                "game_id": self.id,
            }
        except Exception as exc:
            if not self._render_warning_emitted:
                self._render_warning_emitted = True
                _LOGGER.warning(
                    "Jumanji render failed for %s: %s", self._env_id, exc
                )
            return {
                "mode": RenderMode.RGB_ARRAY.value,
                "rgb": None,
                "game_id": self.id,
                "error": str(exc),
            }

    def close(self) -> None:
        """Close the environment."""
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass  # Jumanji wrapper may not have close()
            self._env = None
            _LOGGER.debug("Jumanji environment closed: %s", self._env_id)

    def build_step_state(
        self, observation: np.ndarray, info: Mapping[str, Any]
    ) -> StepState:
        """Build step state for telemetry - subclasses can override."""
        metrics: dict[str, Any] = {
            "step": self._step_counter,
            "episode_return": self._episode_return,
        }

        # Extract common info fields
        if "score" in info:
            metrics["score"] = info["score"]

        environment: dict[str, Any] = {
            "env_id": self.id,
            "action_space_size": self._get_action_space_size(),
        }

        return StepState(
            metrics=metrics,
            environment=environment,
            raw=dict(info) if info else {},
        )

    def _process_observation(self, observation: Any) -> np.ndarray:
        """Process observation to numpy array format.

        Handles nested dict observations (e.g., PacMan's player_locations: {x, y}).
        """
        return self._flatten_observation(observation).astype(np.float32)

    def _flatten_observation(self, obs: Any) -> np.ndarray:
        """Recursively flatten observation to 1D numpy array."""
        if isinstance(obs, np.ndarray):
            return obs.flatten()
        elif isinstance(obs, dict):
            # Recursively flatten nested dicts
            parts = []
            for key in sorted(obs.keys()):
                parts.append(self._flatten_observation(obs[key]))
            return np.concatenate(parts) if parts else np.array([])
        elif isinstance(obs, (int, float, np.integer, np.floating)):
            return np.array([obs])
        elif isinstance(obs, (list, tuple)):
            return np.asarray(obs).flatten()
        else:
            # Try to convert to numpy array
            try:
                return np.asarray(obs).flatten()
            except (TypeError, ValueError):
                _LOGGER.warning("Could not flatten observation of type %s", type(obs))
                return np.array([])

    def _get_action_space_size(self) -> int:
        """Get the size of the discrete action space."""
        if self._env is not None and hasattr(self._env, "action_space"):
            return int(self._env.action_space.n)
        return 0


# =============================================================================
# Per-environment adapter subclasses
# =============================================================================


class JumanjiGame2048Adapter(JumanjiAdapter):
    """2048: Slide tiles to merge and reach 2048.

    Action space: 4 (up, down, left, right)
    Observation: 4x4 board with tile values
    """
    _env_id = "Game2048-v1"

    def build_step_state(
        self, observation: np.ndarray, info: Mapping[str, Any]
    ) -> StepState:
        metrics: dict[str, Any] = {
            "step": self._step_counter,
            "episode_return": self._episode_return,
        }

        # Game2048 specific metrics
        if "score" in info:
            metrics["score"] = info["score"]
        if isinstance(observation, dict) and "board" in observation:
            board = np.asarray(observation["board"])
            metrics["max_tile"] = int(np.max(board))
            metrics["empty_cells"] = int(np.sum(board == 0))

        return StepState(
            metrics=metrics,
            environment={"env_id": self.id, "game_type": "2048"},
            raw=dict(info) if info else {},
        )


class JumanjiMinesweeperAdapter(JumanjiAdapter):
    """Minesweeper: Reveal cells without hitting mines.

    Action space: grid_size * grid_size (cell indices)
    Observation: Grid with revealed/hidden cells and mine counts
    """
    _env_id = "Minesweeper-v0"

    def build_step_state(
        self, observation: np.ndarray, info: Mapping[str, Any]
    ) -> StepState:
        metrics: dict[str, Any] = {
            "step": self._step_counter,
            "episode_return": self._episode_return,
        }

        # Minesweeper specific metrics
        if "mines_remaining" in info:
            metrics["mines_remaining"] = info["mines_remaining"]
        if "cells_revealed" in info:
            metrics["cells_revealed"] = info["cells_revealed"]

        return StepState(
            metrics=metrics,
            environment={"env_id": self.id, "game_type": "minesweeper"},
            hazards=[{"type": "mine", "hidden": True}],
            raw=dict(info) if info else {},
        )


class JumanjiRubiksCubeAdapter(JumanjiAdapter):
    """Rubik's Cube: Solve the 3x3 cube by face rotations.

    Action space: 12 (6 faces x 2 directions)
    Observation: Cube state (colors of all faces)
    """
    _env_id = "RubiksCube-v0"

    def build_step_state(
        self, observation: np.ndarray, info: Mapping[str, Any]
    ) -> StepState:
        metrics: dict[str, Any] = {
            "step": self._step_counter,
            "episode_return": self._episode_return,
        }

        # RubiksCube specific metrics
        if "solved_faces" in info:
            metrics["solved_faces"] = info["solved_faces"]
        if "is_solved" in info:
            metrics["is_solved"] = info["is_solved"]

        return StepState(
            metrics=metrics,
            environment={"env_id": self.id, "game_type": "rubiks_cube"},
            raw=dict(info) if info else {},
        )

    def get_action_name(self, action: int) -> str:
        """Get human-readable action name for Rubik's Cube."""
        if 0 <= action < len(RUBIKS_CUBE_ACTIONS):
            return RUBIKS_CUBE_ACTIONS[action]
        return f"action_{action}"


class JumanjiSlidingPuzzleAdapter(JumanjiAdapter):
    """Sliding Tile Puzzle: Rearrange tiles to reach goal configuration.

    Action space: 4 (move blank up, down, left, right)
    Observation: Grid with numbered tiles and blank space
    """
    _env_id = "SlidingTilePuzzle-v0"

    def build_step_state(
        self, observation: np.ndarray, info: Mapping[str, Any]
    ) -> StepState:
        metrics: dict[str, Any] = {
            "step": self._step_counter,
            "episode_return": self._episode_return,
        }

        # Sliding puzzle specific metrics
        if "tiles_in_place" in info:
            metrics["tiles_in_place"] = info["tiles_in_place"]
        if "manhattan_distance" in info:
            metrics["manhattan_distance"] = info["manhattan_distance"]

        return StepState(
            metrics=metrics,
            environment={"env_id": self.id, "game_type": "sliding_puzzle"},
            raw=dict(info) if info else {},
        )


class JumanjiSudokuAdapter(JumanjiAdapter):
    """Sudoku: Fill 9x9 grid following Sudoku rules.

    Action space: 9*9*9 (cell x digit combinations)
    Observation: 9x9 grid with some cells filled

    For interactive play:
    - Click cell to select
    - Press 1-9 to place digit
    - Action = row * 81 + col * 9 + (digit - 1)
    """
    _env_id = "Sudoku-v0"

    # Track initial puzzle state for fixed cells
    _initial_board: np.ndarray | None = None
    # Track last raw observation for render()
    _last_obs: dict[str, Any] | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> AdapterStep[np.ndarray]:
        """Reset and capture initial puzzle state and observation."""
        env = self._require_env()

        # Reset with optional seed
        obs, info = env.reset(seed=seed)

        # Store raw observation BEFORE _package_step calls render()
        self._last_obs = obs if isinstance(obs, dict) else None

        # Store initial board to identify fixed cells
        if isinstance(obs, dict) and "board" in obs:
            self._initial_board = np.asarray(obs["board"]).copy()

        processed_obs = self._process_observation(obs)

        self._step_counter = 0
        self._episode_return = 0.0

        return self._package_step(processed_obs, 0.0, False, False, info)

    def step(self, action: int) -> AdapterStep[np.ndarray]:
        """Execute one step and update stored observation."""
        env = self._require_env()

        # Jumanji Sudoku expects a 3-element array [row, col, digit] not a
        # flat integer.  Decode from our flat encoding (row*81 + col*9 + digit).
        row, col, digit = self.decode_action(action)
        action_array = np.array([row, col, digit - 1], dtype=np.int32)

        obs, reward, terminated, truncated, info = env.step(action_array)

        # Store raw observation BEFORE _package_step calls render()
        self._last_obs = obs if isinstance(obs, dict) else None

        processed_obs = self._process_observation(obs)

        self._step_counter += 1
        self._episode_return += float(reward)

        return self._package_step(
            processed_obs, float(reward), terminated, truncated, info
        )

    def build_step_state(
        self, observation: np.ndarray, info: Mapping[str, Any]
    ) -> StepState:
        metrics: dict[str, Any] = {
            "step": self._step_counter,
            "episode_return": self._episode_return,
        }

        # Sudoku specific metrics - use _last_obs since observation is processed
        # Jumanji board uses -1 for empty, 0-8 for digits 1-9
        if self._last_obs is not None and isinstance(self._last_obs, dict):
            board = np.asarray(self._last_obs.get("board", []))
            if board.size > 0:
                # Count cells that are not empty (-1)
                filled = np.sum(board != -1)
                metrics["cells_filled"] = int(filled)
                metrics["cells_remaining"] = int(81 - filled)

        if "violations" in info:
            metrics["violations"] = info["violations"]
        if "is_valid" in info:
            metrics["is_valid"] = info["is_valid"]

        return StepState(
            metrics=metrics,
            environment={"env_id": self.id, "game_type": "sudoku"},
            raw=dict(info) if info else {},
        )

    def render(self) -> dict[str, Any]:
        """Render Sudoku board with data for interactive renderer.

        Returns payload with:
        - board: 9x9 grid (0=empty, 1-9=digit)
        - action_mask: 729 bools for valid actions
        - fixed_cells: List of (row, col) tuples for initial clues
        - game_type: "sudoku" for renderer detection
        """
        self._require_env()

        # Get base render (RGB frame)
        base_render = super().render()

        # Extract sudoku-specific data for interactive renderer
        sudoku_data: dict[str, Any] = {
            "game_type": "sudoku",
            "game_id": self.id,
        }

        # Get current observation from last step
        if self._last_obs is not None:
            obs = self._last_obs
            if isinstance(obs, dict):
                # Board state
                if "board" in obs:
                    sudoku_data["board"] = np.asarray(obs["board"]).tolist()

                # Action mask
                if "action_mask" in obs:
                    sudoku_data["action_mask"] = np.asarray(obs["action_mask"]).flatten().tolist()

        # Fixed cells from initial board
        if self._initial_board is not None:
            fixed = []
            for r in range(9):
                for c in range(9):
                    if self._initial_board[r, c] != 0:
                        fixed.append((r, c))
            sudoku_data["fixed_cells"] = fixed

        # Merge with base render
        base_render["sudoku"] = sudoku_data
        return base_render

    @staticmethod
    def compute_action(row: int, col: int, digit: int) -> int:
        """Compute Jumanji action index from row, col, digit.

        Action = row * 81 + col * 9 + (digit - 1)

        Args:
            row: Row index (0-8)
            col: Column index (0-8)
            digit: Digit to place (1-9)

        Returns:
            Action index (0-728)
        """
        return row * 81 + col * 9 + (digit - 1)

    @staticmethod
    def decode_action(action: int) -> tuple[int, int, int]:
        """Decode Jumanji action index to row, col, digit.

        Args:
            action: Action index (0-728)

        Returns:
            (row, col, digit) tuple
        """
        row = action // 81
        remainder = action % 81
        col = remainder // 9
        digit = (remainder % 9) + 1
        return (row, col, digit)


class JumanjiGraphColoringAdapter(JumanjiAdapter):
    """Graph Coloring: Color graph nodes with minimum colors, no adjacent same colors.

    Action space: num_nodes * num_colors
    Observation: Graph adjacency and current coloring
    """
    _env_id = "GraphColoring-v1"

    def build_step_state(
        self, observation: np.ndarray, info: Mapping[str, Any]
    ) -> StepState:
        metrics: dict[str, Any] = {
            "step": self._step_counter,
            "episode_return": self._episode_return,
        }

        # Graph coloring specific metrics
        if "colors_used" in info:
            metrics["colors_used"] = info["colors_used"]
        if "conflicts" in info:
            metrics["conflicts"] = info["conflicts"]
        if "nodes_colored" in info:
            metrics["nodes_colored"] = info["nodes_colored"]

        return StepState(
            metrics=metrics,
            environment={"env_id": self.id, "game_type": "graph_coloring"},
            raw=dict(info) if info else {},
        )


# =============================================================================
# Phase 2: Packing Environments
# =============================================================================


class JumanjiBinPackAdapter(JumanjiAdapter):
    """BinPack: Pack items into minimum number of bins.

    Action space: Place item into bin (varies by num_bins)
    Observation: Item sizes, bin capacities, current allocations
    """
    _env_id = "BinPack-v2"

    def build_step_state(
        self, observation: np.ndarray, info: Mapping[str, Any]
    ) -> StepState:
        metrics: dict[str, Any] = {
            "step": self._step_counter,
            "episode_return": self._episode_return,
        }

        # BinPack specific metrics
        if "bins_used" in info:
            metrics["bins_used"] = info["bins_used"]
        if "items_packed" in info:
            metrics["items_packed"] = info["items_packed"]
        if "utilization" in info:
            metrics["utilization"] = info["utilization"]

        return StepState(
            metrics=metrics,
            environment={"env_id": self.id, "game_type": "binpack"},
            raw=dict(info) if info else {},
        )


class JumanjiFlatPackAdapter(JumanjiAdapter):
    """FlatPack: Pack 2D rectangular items onto a surface.

    Action space: Rotation and placement of items
    Observation: Surface grid, item dimensions, placed items
    """
    _env_id = "FlatPack-v0"

    def build_step_state(
        self, observation: np.ndarray, info: Mapping[str, Any]
    ) -> StepState:
        metrics: dict[str, Any] = {
            "step": self._step_counter,
            "episode_return": self._episode_return,
        }

        # FlatPack specific metrics
        if "items_placed" in info:
            metrics["items_placed"] = info["items_placed"]
        if "area_used" in info:
            metrics["area_used"] = info["area_used"]
        if "wasted_space" in info:
            metrics["wasted_space"] = info["wasted_space"]

        return StepState(
            metrics=metrics,
            environment={"env_id": self.id, "game_type": "flatpack"},
            raw=dict(info) if info else {},
        )


class JumanjiJobShopAdapter(JumanjiAdapter):
    """JobShop: Schedule jobs on machines to minimize makespan.

    Action space: Assign operation to machine/time slot
    Observation: Job operations, machine schedules, precedence constraints
    """
    _env_id = "JobShop-v0"

    def build_step_state(
        self, observation: np.ndarray, info: Mapping[str, Any]
    ) -> StepState:
        metrics: dict[str, Any] = {
            "step": self._step_counter,
            "episode_return": self._episode_return,
        }

        # JobShop specific metrics
        if "makespan" in info:
            metrics["makespan"] = info["makespan"]
        if "operations_scheduled" in info:
            metrics["operations_scheduled"] = info["operations_scheduled"]
        if "machines_utilized" in info:
            metrics["machines_utilized"] = info["machines_utilized"]

        return StepState(
            metrics=metrics,
            environment={"env_id": self.id, "game_type": "jobshop"},
            raw=dict(info) if info else {},
        )


class JumanjiKnapsackAdapter(JumanjiAdapter):
    """Knapsack: Select items to maximize value under capacity constraint.

    Action space: Binary selection of items (take/leave)
    Observation: Item weights, values, remaining capacity
    """
    _env_id = "Knapsack-v1"

    def build_step_state(
        self, observation: np.ndarray, info: Mapping[str, Any]
    ) -> StepState:
        metrics: dict[str, Any] = {
            "step": self._step_counter,
            "episode_return": self._episode_return,
        }

        # Knapsack specific metrics
        if "total_value" in info:
            metrics["total_value"] = info["total_value"]
        if "total_weight" in info:
            metrics["total_weight"] = info["total_weight"]
        if "remaining_capacity" in info:
            metrics["remaining_capacity"] = info["remaining_capacity"]
        if "items_selected" in info:
            metrics["items_selected"] = info["items_selected"]

        return StepState(
            metrics=metrics,
            environment={"env_id": self.id, "game_type": "knapsack"},
            raw=dict(info) if info else {},
        )


class JumanjiTetrisAdapter(JumanjiAdapter):
    """Tetris: Classic falling block puzzle game.

    Action space: Rotate and position falling piece
    Observation: Game board, current piece, next piece
    """
    _env_id = "Tetris-v0"

    def build_step_state(
        self, observation: np.ndarray, info: Mapping[str, Any]
    ) -> StepState:
        metrics: dict[str, Any] = {
            "step": self._step_counter,
            "episode_return": self._episode_return,
        }

        # Tetris specific metrics
        if "score" in info:
            metrics["score"] = info["score"]
        if "lines_cleared" in info:
            metrics["lines_cleared"] = info["lines_cleared"]
        if "level" in info:
            metrics["level"] = info["level"]
        if "pieces_placed" in info:
            metrics["pieces_placed"] = info["pieces_placed"]

        return StepState(
            metrics=metrics,
            environment={"env_id": self.id, "game_type": "tetris"},
            raw=dict(info) if info else {},
        )


# =============================================================================
# Phase 3: Routing Environments
# =============================================================================


class JumanjiCleanerAdapter(JumanjiAdapter):
    """Cleaner: Navigate grid to clean all dirty cells.

    Action space: 4 directions (up, down, left, right)
    Observation: Grid with dirty/clean cells, agent position
    """
    _env_id = "Cleaner-v0"

    def build_step_state(
        self, observation: np.ndarray, info: Mapping[str, Any]
    ) -> StepState:
        metrics: dict[str, Any] = {
            "step": self._step_counter,
            "episode_return": self._episode_return,
        }

        if "cells_cleaned" in info:
            metrics["cells_cleaned"] = info["cells_cleaned"]
        if "cells_remaining" in info:
            metrics["cells_remaining"] = info["cells_remaining"]

        return StepState(
            metrics=metrics,
            environment={"env_id": self.id, "game_type": "cleaner"},
            raw=dict(info) if info else {},
        )


class JumanjiConnectorAdapter(JumanjiAdapter):
    """Connector: Connect pairs of points without crossing paths.

    Action space: Select next cell in path
    Observation: Grid with endpoints and current paths
    """
    _env_id = "Connector-v3"

    def build_step_state(
        self, observation: np.ndarray, info: Mapping[str, Any]
    ) -> StepState:
        metrics: dict[str, Any] = {
            "step": self._step_counter,
            "episode_return": self._episode_return,
        }

        if "pairs_connected" in info:
            metrics["pairs_connected"] = info["pairs_connected"]
        if "pairs_remaining" in info:
            metrics["pairs_remaining"] = info["pairs_remaining"]

        return StepState(
            metrics=metrics,
            environment={"env_id": self.id, "game_type": "connector"},
            raw=dict(info) if info else {},
        )


class JumanjiCVRPAdapter(JumanjiAdapter):
    """CVRP: Capacitated Vehicle Routing Problem.

    Action space: Select next customer to visit
    Observation: Customer locations, demands, vehicle capacity
    """
    _env_id = "CVRP-v1"

    def build_step_state(
        self, observation: np.ndarray, info: Mapping[str, Any]
    ) -> StepState:
        metrics: dict[str, Any] = {
            "step": self._step_counter,
            "episode_return": self._episode_return,
        }

        if "customers_visited" in info:
            metrics["customers_visited"] = info["customers_visited"]
        if "total_distance" in info:
            metrics["total_distance"] = info["total_distance"]
        if "remaining_capacity" in info:
            metrics["remaining_capacity"] = info["remaining_capacity"]

        return StepState(
            metrics=metrics,
            environment={"env_id": self.id, "game_type": "cvrp"},
            raw=dict(info) if info else {},
        )


class JumanjiMazeAdapter(JumanjiAdapter):
    """Maze: Navigate through maze to reach goal.

    Action space: 4 directions (up, down, left, right)
    Observation: Maze walls, agent position, goal position
    """
    _env_id = "Maze-v0"

    def build_step_state(
        self, observation: np.ndarray, info: Mapping[str, Any]
    ) -> StepState:
        metrics: dict[str, Any] = {
            "step": self._step_counter,
            "episode_return": self._episode_return,
        }

        if "distance_to_goal" in info:
            metrics["distance_to_goal"] = info["distance_to_goal"]
        if "reached_goal" in info:
            metrics["reached_goal"] = info["reached_goal"]

        return StepState(
            metrics=metrics,
            environment={"env_id": self.id, "game_type": "maze"},
            raw=dict(info) if info else {},
        )


class JumanjiMMSTAdapter(JumanjiAdapter):
    """MMST: Multi-agent Minimum Spanning Tree.

    Action space: Select edge to add to tree
    Observation: Graph nodes, edges, current tree
    """
    _env_id = "MMST-v0"

    def build_step_state(
        self, observation: np.ndarray, info: Mapping[str, Any]
    ) -> StepState:
        metrics: dict[str, Any] = {
            "step": self._step_counter,
            "episode_return": self._episode_return,
        }

        if "edges_added" in info:
            metrics["edges_added"] = info["edges_added"]
        if "tree_cost" in info:
            metrics["tree_cost"] = info["tree_cost"]

        return StepState(
            metrics=metrics,
            environment={"env_id": self.id, "game_type": "mmst"},
            raw=dict(info) if info else {},
        )


class JumanjiMultiCVRPAdapter(JumanjiAdapter):
    """MultiCVRP: Multi-vehicle Capacitated Vehicle Routing.

    Action space: Select next customer for current vehicle
    Observation: Customer locations, demands, multiple vehicles
    """
    _env_id = "MultiCVRP-v0"

    def build_step_state(
        self, observation: np.ndarray, info: Mapping[str, Any]
    ) -> StepState:
        metrics: dict[str, Any] = {
            "step": self._step_counter,
            "episode_return": self._episode_return,
        }

        if "customers_visited" in info:
            metrics["customers_visited"] = info["customers_visited"]
        if "total_distance" in info:
            metrics["total_distance"] = info["total_distance"]
        if "vehicles_used" in info:
            metrics["vehicles_used"] = info["vehicles_used"]

        return StepState(
            metrics=metrics,
            environment={"env_id": self.id, "game_type": "multi_cvrp"},
            raw=dict(info) if info else {},
        )


class JumanjiPacManAdapter(JumanjiAdapter):
    """PacMan: Classic arcade game - eat pellets, avoid ghosts.

    Action space: 4 directions (up, down, left, right)
    Observation: Maze, pellets, ghost positions, power pellets
    """
    _env_id = "PacMan-v1"

    def build_step_state(
        self, observation: np.ndarray, info: Mapping[str, Any]
    ) -> StepState:
        metrics: dict[str, Any] = {
            "step": self._step_counter,
            "episode_return": self._episode_return,
        }

        if "score" in info:
            metrics["score"] = info["score"]
        if "pellets_eaten" in info:
            metrics["pellets_eaten"] = info["pellets_eaten"]
        if "ghosts_eaten" in info:
            metrics["ghosts_eaten"] = info["ghosts_eaten"]
        if "lives" in info:
            metrics["lives"] = info["lives"]

        return StepState(
            metrics=metrics,
            environment={"env_id": self.id, "game_type": "pacman"},
            hazards=[{"type": "ghost", "avoidable": True}],
            raw=dict(info) if info else {},
        )


class JumanjiRobotWarehouseAdapter(JumanjiAdapter):
    """RobotWarehouse: Coordinate robots in warehouse for item delivery.

    Action space: Move/pickup/drop actions for each robot
    Observation: Warehouse grid, robot positions, item locations
    """
    _env_id = "RobotWarehouse-v0"

    def build_step_state(
        self, observation: np.ndarray, info: Mapping[str, Any]
    ) -> StepState:
        metrics: dict[str, Any] = {
            "step": self._step_counter,
            "episode_return": self._episode_return,
        }

        if "items_delivered" in info:
            metrics["items_delivered"] = info["items_delivered"]
        if "items_remaining" in info:
            metrics["items_remaining"] = info["items_remaining"]
        if "collisions" in info:
            metrics["collisions"] = info["collisions"]

        return StepState(
            metrics=metrics,
            environment={"env_id": self.id, "game_type": "robot_warehouse"},
            raw=dict(info) if info else {},
        )


class JumanjiSnakeAdapter(JumanjiAdapter):
    """Snake: Classic snake game - eat food, grow longer, avoid walls/self.

    Action space: 4 directions (up, down, left, right)
    Observation: Grid with snake body, food position
    """
    _env_id = "Snake-v1"

    def build_step_state(
        self, observation: np.ndarray, info: Mapping[str, Any]
    ) -> StepState:
        metrics: dict[str, Any] = {
            "step": self._step_counter,
            "episode_return": self._episode_return,
        }

        if "score" in info:
            metrics["score"] = info["score"]
        if "length" in info:
            metrics["length"] = info["length"]
        if "food_eaten" in info:
            metrics["food_eaten"] = info["food_eaten"]

        return StepState(
            metrics=metrics,
            environment={"env_id": self.id, "game_type": "snake"},
            raw=dict(info) if info else {},
        )


class JumanjiSokobanAdapter(JumanjiAdapter):
    """Sokoban: Push boxes onto target locations.

    Action space: 4 directions (up, down, left, right)
    Observation: Grid with boxes, targets, walls, agent
    """
    _env_id = "Sokoban-v0"

    def build_step_state(
        self, observation: np.ndarray, info: Mapping[str, Any]
    ) -> StepState:
        metrics: dict[str, Any] = {
            "step": self._step_counter,
            "episode_return": self._episode_return,
        }

        if "boxes_on_target" in info:
            metrics["boxes_on_target"] = info["boxes_on_target"]
        if "total_boxes" in info:
            metrics["total_boxes"] = info["total_boxes"]
        if "pushes" in info:
            metrics["pushes"] = info["pushes"]

        return StepState(
            metrics=metrics,
            environment={"env_id": self.id, "game_type": "sokoban"},
            raw=dict(info) if info else {},
        )


class JumanjiTSPAdapter(JumanjiAdapter):
    """TSP: Traveling Salesman Problem - visit all cities with minimum distance.

    Action space: Select next city to visit
    Observation: City locations, visited cities, current position
    """
    _env_id = "TSP-v1"

    def build_step_state(
        self, observation: np.ndarray, info: Mapping[str, Any]
    ) -> StepState:
        metrics: dict[str, Any] = {
            "step": self._step_counter,
            "episode_return": self._episode_return,
        }

        if "cities_visited" in info:
            metrics["cities_visited"] = info["cities_visited"]
        if "total_distance" in info:
            metrics["total_distance"] = info["total_distance"]
        if "cities_remaining" in info:
            metrics["cities_remaining"] = info["cities_remaining"]

        return StepState(
            metrics=metrics,
            environment={"env_id": self.id, "game_type": "tsp"},
            raw=dict(info) if info else {},
        )


# =============================================================================
# Adapter registry
# =============================================================================


JUMANJI_ADAPTERS: dict[GameId, type[JumanjiAdapter]] = {
    # Phase 1: Logic
    GameId.JUMANJI_GAME2048: JumanjiGame2048Adapter,
    GameId.JUMANJI_MINESWEEPER: JumanjiMinesweeperAdapter,
    GameId.JUMANJI_RUBIKS_CUBE: JumanjiRubiksCubeAdapter,
    GameId.JUMANJI_SLIDING_PUZZLE: JumanjiSlidingPuzzleAdapter,
    GameId.JUMANJI_SUDOKU: JumanjiSudokuAdapter,
    GameId.JUMANJI_GRAPH_COLORING: JumanjiGraphColoringAdapter,
    # Phase 2: Packing
    GameId.JUMANJI_BINPACK: JumanjiBinPackAdapter,
    GameId.JUMANJI_FLATPACK: JumanjiFlatPackAdapter,
    GameId.JUMANJI_JOBSHOP: JumanjiJobShopAdapter,
    GameId.JUMANJI_KNAPSACK: JumanjiKnapsackAdapter,
    GameId.JUMANJI_TETRIS: JumanjiTetrisAdapter,
    # Phase 3: Routing
    GameId.JUMANJI_CLEANER: JumanjiCleanerAdapter,
    GameId.JUMANJI_CONNECTOR: JumanjiConnectorAdapter,
    GameId.JUMANJI_CVRP: JumanjiCVRPAdapter,
    GameId.JUMANJI_MAZE: JumanjiMazeAdapter,
    GameId.JUMANJI_MMST: JumanjiMMSTAdapter,
    GameId.JUMANJI_MULTI_CVRP: JumanjiMultiCVRPAdapter,
    GameId.JUMANJI_PACMAN: JumanjiPacManAdapter,
    GameId.JUMANJI_ROBOT_WAREHOUSE: JumanjiRobotWarehouseAdapter,
    GameId.JUMANJI_SNAKE: JumanjiSnakeAdapter,
    GameId.JUMANJI_SOKOBAN: JumanjiSokobanAdapter,
    GameId.JUMANJI_TSP: JumanjiTSPAdapter,
}


__all__ = [
    "JumanjiAdapter",
    # Phase 1: Logic
    "JumanjiGame2048Adapter",
    "JumanjiMinesweeperAdapter",
    "JumanjiRubiksCubeAdapter",
    "JumanjiSlidingPuzzleAdapter",
    "JumanjiSudokuAdapter",
    "JumanjiGraphColoringAdapter",
    # Phase 2: Packing
    "JumanjiBinPackAdapter",
    "JumanjiFlatPackAdapter",
    "JumanjiJobShopAdapter",
    "JumanjiKnapsackAdapter",
    "JumanjiTetrisAdapter",
    # Phase 3: Routing
    "JumanjiCleanerAdapter",
    "JumanjiConnectorAdapter",
    "JumanjiCVRPAdapter",
    "JumanjiMazeAdapter",
    "JumanjiMMSTAdapter",
    "JumanjiMultiCVRPAdapter",
    "JumanjiPacManAdapter",
    "JumanjiRobotWarehouseAdapter",
    "JumanjiSnakeAdapter",
    "JumanjiSokobanAdapter",
    "JumanjiTSPAdapter",
    # Registry and constants
    "JUMANJI_ADAPTERS",
    "JUMANJI_ENV_NAMES",
    "GAME2048_ACTIONS",
    "RUBIKS_CUBE_ACTIONS",
]
