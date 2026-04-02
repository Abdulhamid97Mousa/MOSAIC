"""OpenSpiel environment adapters for board games via Shimmy wrapper.

OpenSpiel is a collection of games from Google DeepMind for research in
reinforcement learning, search, and game theory. Shimmy provides PettingZoo-
compatible wrappers for OpenSpiel games.

Repository: https://github.com/google-deepmind/open_spiel
Shimmy: https://shimmy.farama.org/environments/open_spiel/

Currently supported games:
- Checkers: Classic 8x8 checkers (English draughts)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from gym_gui.core.adapters.base import (
    AdapterContext,
    AdapterStep,
    AgentSnapshot,
    EnvironmentAdapter,
    StepState,
)
from gym_gui.core.enums import ControlMode, GameId, RenderMode
from gym_gui.logging_config.log_constants import (
    LOG_ADAPTER_ENV_CREATED,
    LOG_ADAPTER_ENV_RESET,
    LOG_ADAPTER_STEP_SUMMARY,
)

_LOG = logging.getLogger(__name__)


# Try to import OpenSpiel and Shimmy
try:
    import pyspiel
    from shimmy.openspiel_compatibility import OpenSpielCompatibilityV0
    _OPENSPIEL_AVAILABLE = True
except ImportError:
    _OPENSPIEL_AVAILABLE = False
    pyspiel = None  # type: ignore[assignment]
    OpenSpielCompatibilityV0 = None  # type: ignore[assignment, misc]


def _ensure_openspiel() -> None:
    """Ensure OpenSpiel and Shimmy are available, raise helpful error if not."""
    if not _OPENSPIEL_AVAILABLE:
        raise ImportError(
            "OpenSpiel and Shimmy are not available. Install via:\n"
            "  pip install open-spiel shimmy[openspiel]\n"
            "or:\n"
            "  pip install -e '.[openspiel]'"
        )


@dataclass(slots=True)
class CheckersRenderPayload:
    """Render payload for Checkers containing board state.

    Checkers board representation:
    - 8x8 grid (only dark squares are used)
    - Values: 0=empty, 1=black piece, 2=black king, 3=white piece, 4=white king
    - Black moves from row 0 towards row 7, white moves from row 7 towards row 0
    """

    board: List[List[int]]  # 8x8 grid
    current_player: str  # "player_0" or "player_1"
    legal_moves: List[int]  # Valid action indices
    last_move: Optional[int] = None
    is_game_over: bool = False
    winner: Optional[str] = None  # "player_0", "player_1", or "draw"
    move_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game_type": "checkers",
            "board": self.board,
            "current_player": self.current_player,
            "legal_moves": self.legal_moves,
            "last_move": self.last_move,
            "is_game_over": self.is_game_over,
            "winner": self.winner,
            "move_count": self.move_count,
        }


class CheckersEnvironmentAdapter(EnvironmentAdapter[Dict[str, Any], int]):
    """Adapter for OpenSpiel Checkers environment via Shimmy.

    This adapter wraps OpenSpiel's checkers game through Shimmy's PettingZoo
    compatibility layer. It provides:
    - Board state representation for Qt rendering
    - Move execution via action index
    - Turn management for human vs human or human vs AI play

    The observation is a dictionary containing:
    - "observation": The raw observation from OpenSpiel
    - "action_mask": Binary mask of legal actions

    The render_payload is a CheckersRenderPayload with board state.
    """

    id = GameId.OPEN_SPIEL_CHECKERS.value
    supported_control_modes = (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    )
    supported_render_modes = (RenderMode.RGB_ARRAY,)
    default_render_mode = RenderMode.RGB_ARRAY

    # Agent mapping - OpenSpiel uses "player_0" and "player_1"
    PLAYER_0 = "player_0"
    PLAYER_1 = "player_1"

    def __init__(self, context: AdapterContext | None = None) -> None:
        super().__init__(context)
        self._aec_env: Any = None
        self._openspiel_game: Any = None
        self._openspiel_state: Any = None
        self._board: List[List[int]] = [[0] * 8 for _ in range(8)]
        self._last_move: Optional[int] = None
        self._move_count: int = 0
        self._current_player: str = self.PLAYER_0

    def load(self) -> None:
        """Load the OpenSpiel Checkers environment via Shimmy."""
        _ensure_openspiel()

        try:
            # Create Shimmy wrapper for PettingZoo compatibility
            # Shimmy's OpenSpielCompatibilityV0 accepts game_name as string
            self._aec_env = OpenSpielCompatibilityV0(
                game_name="checkers",
                render_mode="rgb_array",
            )

            self.log_constant(
                LOG_ADAPTER_ENV_CREATED,
                extra={
                    "env_id": self.id,
                    "render_mode": "rgb_array",
                    "gym_kwargs": "-",
                    "wrapped_class": "OpenSpielCompatibilityV0(checkers)",
                },
            )
        except Exception as e:
            _LOG.error("Failed to load OpenSpiel Checkers: %s", e)
            raise ImportError(
                f"Failed to load OpenSpiel Checkers environment: {e}\n"
                "Install with: pip install open-spiel shimmy[openspiel]"
            ) from e

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> AdapterStep[Dict[str, Any]]:
        """Reset the Checkers environment."""
        if self._aec_env is None:
            raise RuntimeError("Environment not loaded. Call load() first.")

        self._aec_env.reset(seed=seed)
        self._board = [[0] * 8 for _ in range(8)]
        self._last_move = None
        self._move_count = 0
        self._episode_step = 0
        self._episode_return = 0.0
        self._current_player = self.PLAYER_0

        # Initialize board with starting positions
        self._init_board()

        self.log_constant(
            LOG_ADAPTER_ENV_RESET,
            extra={
                "env_id": self.id,
                "seed": seed if seed is not None else "None",
                "has_options": bool(options),
            },
        )

        # Get initial observation
        obs, _, _, _, info = self._aec_env.last()

        # Sync board from observation
        self._sync_board_from_observation(obs)

        return self._package_step(obs, 0.0, False, False, dict(info))

    def _init_board(self) -> None:
        """Initialize board from OpenSpiel game state.

        OpenSpiel checkers board representation (from string):
        - 'o' = player_0 piece (moves first, bottom rows 1-3)
        - '+' = player_1 piece (top rows 6-8)
        - '8' = player_0 king (always the character '8', regardless of rank)
        - '*' = player_1 king
        - '.' = empty

        Our internal representation:
        - 0 = empty
        - 1 = player_0 piece (rendered as dark/red)
        - 2 = player_0 king
        - 3 = player_1 piece (rendered as light/white)
        - 4 = player_1 king
        """
        self._sync_board_from_game_state()

    def _sync_board_from_observation(self, obs: Dict[str, Any]) -> None:
        """Sync internal board state from OpenSpiel observation."""
        self._sync_board_from_game_state()

    def _sync_board_from_game_state(self) -> None:
        """Sync internal board state from OpenSpiel game_state.

        Parses the string representation of the OpenSpiel checkers state
        to extract piece positions.

        OpenSpiel string format:
        - 'o' = player_0 piece
        - '+' = player_1 piece
        - '8' = player_0 king (always the character '8', regardless of rank)
        - '*' = player_1 king
        - '.' = empty square
        """
        if self._aec_env is None or not hasattr(self._aec_env, 'game_state'):
            return

        try:
            state = self._aec_env.game_state
            board_str = str(state)

            # Reset board
            self._board = [[0] * 8 for _ in range(8)]

            # Parse board string
            # Format: "8.+.+.+.+\n7+.+.+.+.\n..." with rank labels
            lines = board_str.strip().split('\n')

            for line in lines:
                if not line or line.startswith(' '):
                    continue  # Skip column labels

                # Extract rank number and pieces
                # Format: "8.+.+.+.+" where first char is rank
                if not line[0].isdigit():
                    continue

                rank = int(line[0])
                row = 8 - rank  # Convert rank to row (rank 8 = row 0)
                pieces = line[1:]  # Rest is piece positions

                for col, char in enumerate(pieces):
                    if col >= 8:
                        break
                    if char == 'o':
                        self._board[row][col] = 1  # player_0 piece
                    elif char == '8':
                        self._board[row][col] = 2  # player_0 king
                    elif char == '+':
                        self._board[row][col] = 3  # player_1 piece
                    elif char == '*':
                        self._board[row][col] = 4  # player_1 king
                    # '.' = empty (already 0)

        except Exception as e:
            _LOG.debug("Could not sync board from game state: %s", e)

    def step(self, action: int) -> AdapterStep[Dict[str, Any]]:
        """Execute an action (move index)."""
        if self._aec_env is None:
            raise RuntimeError("Environment not loaded. Call load() first.")

        # Validate action - action_mask is in info for Shimmy OpenSpiel
        obs, _, _, _, info = self._aec_env.last()
        action_mask = info.get("action_mask", None) if info else None
        if action_mask is not None:
            if action < 0 or action >= len(action_mask) or action_mask[action] == 0:
                raise ValueError(f"Illegal action: {action}")

        self._last_move = action
        self._move_count += 1
        self._episode_step += 1

        # Execute step
        self._aec_env.step(action)

        # Get new observation
        obs, reward, terminated, truncated, info = self._aec_env.last()

        # Update current player
        self._current_player = self._aec_env.agent_selection

        # Sync board from game state after move
        self._sync_board_from_game_state()

        self._episode_return += reward

        self.log_constant(
            LOG_ADAPTER_STEP_SUMMARY,
            extra={
                "env_id": self.id,
                "episode_step": self._episode_step,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
            },
        )

        return self._package_step(obs, reward, terminated, truncated, dict(info))

    def close(self) -> None:
        """Clean up environment resources."""
        if self._aec_env is not None:
            try:
                self._aec_env.close()
            except Exception:
                pass
        self._aec_env = None
        self._openspiel_game = None
        self._openspiel_state = None

    def render(self) -> np.ndarray | None:
        """Get RGB render of current board state."""
        if self._aec_env is None:
            return None
        try:
            return self._aec_env.render()
        except Exception:
            # If Shimmy doesn't provide rendering, generate our own
            return self._generate_board_image()

    def _generate_board_image(self) -> np.ndarray:
        """Generate a simple RGB image of the checkers board."""
        # Create 480x480 image (60px per square)
        img = np.zeros((480, 480, 3), dtype=np.uint8)
        square_size = 60

        for row in range(8):
            for col in range(8):
                y = row * square_size
                x = col * square_size

                # Board color (checkerboard pattern)
                if (row + col) % 2 == 0:
                    color = (240, 217, 181)  # Light (cream)
                else:
                    color = (181, 136, 99)  # Dark (brown)

                img[y:y+square_size, x:x+square_size] = color

                # Draw pieces
                piece = self._board[row][col]
                if piece > 0:
                    center = (x + square_size // 2, y + square_size // 2)
                    radius = square_size // 2 - 5

                    # Draw piece (simple circle approximation)
                    if piece in (1, 2):  # Black pieces
                        piece_color = (50, 50, 50)
                    else:  # White pieces (3, 4)
                        piece_color = (255, 255, 255)

                    # Draw filled circle
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            if dx*dx + dy*dy <= radius*radius:
                                py = center[1] + dy
                                px = center[0] + dx
                                if 0 <= py < 480 and 0 <= px < 480:
                                    img[py, px] = piece_color

                    # Draw king indicator (smaller circle in center)
                    if piece in (2, 4):  # Kings
                        king_radius = radius // 3
                        king_color = (255, 215, 0)  # Gold
                        for dy in range(-king_radius, king_radius + 1):
                            for dx in range(-king_radius, king_radius + 1):
                                if dx*dx + dy*dy <= king_radius*king_radius:
                                    py = center[1] + dy
                                    px = center[0] + dx
                                    if 0 <= py < 480 and 0 <= px < 480:
                                        img[py, px] = king_color

        return img

    def get_checkers_state(self) -> CheckersRenderPayload:
        """Get structured Checkers state for Qt rendering."""
        if self._aec_env is None:
            raise RuntimeError("Environment not loaded.")

        obs, _, terminated, _, info = self._aec_env.last()
        # action_mask is in info for Shimmy OpenSpiel
        action_mask = info.get("action_mask", np.array([])) if info else np.array([])
        legal_moves = [i for i in range(len(action_mask)) if action_mask[i] == 1]

        # Determine winner
        winner = None
        is_game_over = terminated or len(legal_moves) == 0
        if is_game_over:
            winner = self._determine_winner()

        return CheckersRenderPayload(
            board=[row[:] for row in self._board],
            current_player=self._current_player,
            legal_moves=legal_moves,
            last_move=self._last_move,
            is_game_over=is_game_over,
            winner=winner,
            move_count=self._move_count,
        )

    def _determine_winner(self) -> Optional[str]:
        """Determine winner based on game state."""
        if self._aec_env is None:
            return None

        try:
            rewards = self._aec_env.rewards
            if rewards.get(self.PLAYER_0, 0) > 0:
                return self.PLAYER_0
            elif rewards.get(self.PLAYER_1, 0) > 0:
                return self.PLAYER_1
            else:
                return "draw"
        except Exception:
            return None

    def get_legal_moves(self) -> List[int]:
        """Get list of legal move indices."""
        if self._aec_env is None:
            return []
        obs, _, _, _, info = self._aec_env.last()
        # action_mask is in info for Shimmy OpenSpiel
        action_mask = info.get("action_mask", np.array([])) if info else np.array([])
        return [i for i in range(len(action_mask)) if action_mask[i] == 1]

    def is_move_legal(self, action: int) -> bool:
        """Check if a move is legal."""
        if self._aec_env is None:
            return False
        obs, _, _, _, info = self._aec_env.last()
        # action_mask is in info for Shimmy OpenSpiel
        action_mask = info.get("action_mask", np.array([])) if info else np.array([])
        if action < 0 or action >= len(action_mask):
            return False
        return bool(action_mask[action] == 1)

    def get_action_string(self, action: int) -> Optional[str]:
        """Get action string representation (e.g., 'a3b4') for an action index.

        Args:
            action: Action index

        Returns:
            Action string or None if not available.
        """
        if self._aec_env is None or not _OPENSPIEL_AVAILABLE:
            return None
        try:
            # Access the underlying OpenSpiel state through Shimmy
            pyspiel.load_game("checkers")
            # Reconstruct state - this is a simplified approach
            # In practice, we'd need to track the state properly
            return f"action_{action}"
        except Exception:
            return None

    def find_action_for_move(self, from_sq: str, to_sq: str) -> Optional[int]:
        """Find action index for a move specified by source and destination squares.

        Args:
            from_sq: Source square in algebraic notation (e.g., 'a3')
            to_sq: Destination square in algebraic notation (e.g., 'b4')

        Returns:
            Action index if move is legal, None otherwise.
        """
        if self._aec_env is None or not _OPENSPIEL_AVAILABLE:
            return None

        try:
            # Build move string in OpenSpiel format
            move_str = f"{from_sq}{to_sq}"

            # Get legal actions from the underlying Shimmy environment
            # Shimmy stores the OpenSpiel state in game_state attribute
            if hasattr(self._aec_env, 'game_state'):
                state = self._aec_env.game_state
                legal_actions = state.legal_actions()

                for action in legal_actions:
                    action_str = state.action_to_string(state.current_player(), action)
                    # OpenSpiel checkers move format: "a3b4" or "a3-b4" for jumps
                    # Normalize by removing dashes
                    normalized = action_str.replace("-", "")
                    if normalized == move_str:
                        return action

            return None

        except Exception as e:
            _LOG.debug(f"Failed to find action for move {from_sq}->{to_sq}: {e}")
            return None

    def get_legal_move_strings(self) -> List[str]:
        """Get list of legal moves as strings (e.g., ['a3b4', 'c3b4']).

        Returns:
            List of move strings for all legal actions.
        """
        if self._aec_env is None or not _OPENSPIEL_AVAILABLE:
            return []

        try:
            if hasattr(self._aec_env, 'game_state'):
                state = self._aec_env.game_state
                legal_actions = state.legal_actions()
                return [
                    state.action_to_string(state.current_player(), action)
                    for action in legal_actions
                ]
            return []
        except Exception as e:
            _LOG.debug(f"Failed to get legal move strings: {e}")
            return []

    def get_moves_from_square(self, square: str) -> List[str]:
        """Get all legal destination squares for moves from a given square.

        Args:
            square: Source square in algebraic notation (e.g., 'a3')

        Returns:
            List of destination squares (e.g., ['b4']).
        """
        legal_moves = self.get_legal_move_strings()
        destinations = []
        for move in legal_moves:
            if move.startswith(square):
                # Move format is 'a3b4', extract destination
                destinations.append(move[2:])
        return destinations

    def cell_to_algebraic(self, row: int, col: int) -> str:
        """Convert cell coordinates to algebraic notation.

        Args:
            row: Row index (0 = top/row 8, 7 = bottom/row 1)
            col: Column index (0 = a, 7 = h)

        Returns:
            Algebraic notation (e.g., 'a3')
        """
        file_char = chr(ord('a') + col)
        rank_char = str(8 - row)  # Row 0 = rank 8, Row 7 = rank 1
        return f"{file_char}{rank_char}"

    def algebraic_to_cell(self, square: str) -> tuple[int, int]:
        """Convert algebraic notation to cell coordinates.

        Args:
            square: Algebraic notation (e.g., 'a3')

        Returns:
            Tuple of (row, col) indices
        """
        col = ord(square[0]) - ord('a')
        row = 8 - int(square[1])
        return (row, col)

    def current_agent(self) -> str:
        """Get the current agent's turn."""
        return self._current_player

    def _package_step(
        self,
        observation: Dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
    ) -> AdapterStep[Dict[str, Any]]:
        """Package step result with Checkers-specific render payload."""
        # Get game state for rendering
        try:
            game_state = self.get_checkers_state()
            render_payload = game_state.to_dict()
        except Exception:
            render_payload = None

        # Build step state
        agents = [
            AgentSnapshot(name="player_0", role="player"),
            AgentSnapshot(name="player_1", role="player"),
        ]

        state = StepState(
            active_agent=self._current_player,
            agents=agents,
            metrics={
                "move_count": self._move_count,
                "episode_step": self._episode_step,
            },
            environment={
                "game": "checkers",
                "current_player": self._current_player,
            },
        )

        return AdapterStep(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            render_payload=render_payload,
            render_hint={"type": "checkers", "use_qt_widget": True},
            agent_id=self.current_agent(),
            state=state,
        )

    # Required abstract methods from EnvironmentAdapter
    def gym_kwargs(self) -> dict[str, Any]:
        """Return additional kwargs for gym.make (not used for OpenSpiel)."""
        return {}

    def apply_wrappers(self, env: Any) -> Any:
        """Apply wrappers to environment (not used for OpenSpiel)."""
        return env

    def _require_env(self) -> Any:
        """Get underlying environment or raise if not loaded."""
        if self._aec_env is None:
            raise RuntimeError("Environment not loaded. Call load() first.")
        return self._aec_env

    def _set_env(self, env: Any) -> None:
        """Set the underlying environment."""
        self._aec_env = env

    def _resolve_default_render_mode(self) -> RenderMode:
        """Resolve the default render mode."""
        return self.default_render_mode


# Registry of OpenSpiel adapters
OPENSPIEL_ADAPTERS: Dict[GameId, type[EnvironmentAdapter]] = {
    GameId.OPEN_SPIEL_CHECKERS: CheckersEnvironmentAdapter,
}


__all__ = [
    "CheckersEnvironmentAdapter",
    "CheckersRenderPayload",
    "OPENSPIEL_ADAPTERS",
]
