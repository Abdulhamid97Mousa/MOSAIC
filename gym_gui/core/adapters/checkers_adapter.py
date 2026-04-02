"""Checkers-specific adapter for OpenSpiel checkers environment.

This adapter provides a state-based interface for Checkers, enabling:
- Board state for Qt-based rendering (8x8 grid)
- Legal move validation via action masks
- Move execution via action indices
- Turn management

The adapter uses OpenSpiel via Shimmy's PettingZoo compatibility layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from gym_gui.logging_config.log_constants import (
    LOG_UI_MULTI_AGENT_ENV_LOAD_ERROR,
    LOG_UI_MULTI_AGENT_ENV_LOADED,
)

_LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class CheckersState:
    """Structured Checkers game state for UI rendering.

    Board values:
    - 0: Empty
    - 1: Black piece (player_0)
    - 2: Black king
    - 3: White piece (player_1)
    - 4: White king
    """

    board: List[List[int]]  # 8x8 grid
    current_player: str  # "player_0" or "player_1"
    current_agent: str  # Same as current_player
    legal_moves: List[int]  # Valid action indices
    last_move: Optional[int] = None  # Last action index
    is_game_over: bool = False
    winner: Optional[str] = None  # "player_0", "player_1", or "draw"
    move_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for signal emission."""
        return {
            "game_type": "checkers",
            "board": self.board,
            "current_player": self.current_player,
            "current_agent": self.current_agent,
            "legal_moves": self.legal_moves,
            "last_move": self.last_move,
            "is_game_over": self.is_game_over,
            "winner": self.winner,
            "move_count": self.move_count,
        }


class CheckersAdapter:
    """Adapter for OpenSpiel Checkers environment with state-based interface.

    This adapter wraps OpenSpiel's checkers game via Shimmy and provides:
    - State-based interface (board + legal moves) for Qt rendering
    - Move execution via action index
    - Automatic turn management

    Example usage:
        adapter = CheckersAdapter()
        adapter.load(seed=42)

        state = adapter.get_state()
        # state.board = [[0]*8 for _ in range(8)] with pieces
        # state.legal_moves = [list of valid action indices]

        new_state = adapter.make_move(action_idx)
    """

    # Agent mapping - OpenSpiel uses "player_0" and "player_1"
    PLAYER_0 = "player_0"  # Black (goes first)
    PLAYER_1 = "player_1"  # White

    def __init__(self) -> None:
        """Initialize the Checkers adapter."""
        self._aec_env: Any = None
        self._board: List[List[int]] = [[0] * 8 for _ in range(8)]
        self._last_move: Optional[int] = None
        self._move_count: int = 0
        self._current_player: str = self.PLAYER_0
        self._is_loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        """Check if environment is loaded."""
        return self._is_loaded

    def load(self, seed: int = 42, render_mode: str = "rgb_array") -> CheckersState:
        """Load and reset the Checkers environment.

        Args:
            seed: Random seed for environment reset
            render_mode: Render mode (not used for OpenSpiel)

        Returns:
            Initial Checkers state
        """
        try:
            from shimmy.openspiel_compatibility import OpenSpielCompatibilityV0

            # Create Shimmy wrapper for PettingZoo compatibility
            self._aec_env = OpenSpielCompatibilityV0(
                game_name="checkers",
                render_mode=render_mode,
            )

            # Reset environment
            self._aec_env.reset(seed=seed)

            # Reset tracking
            self._board = [[0] * 8 for _ in range(8)]
            self._last_move = None
            self._move_count = 0
            self._current_player = self.PLAYER_0
            self._is_loaded = True

            # Initialize board with starting positions
            self._init_board()

            _LOG.info(
                f"{LOG_UI_MULTI_AGENT_ENV_LOADED.code} "
                f"{LOG_UI_MULTI_AGENT_ENV_LOADED.message} | "
                f"env_id=checkers seed={seed}"
            )

            return self.get_state()

        except Exception as e:
            _LOG.error(
                f"{LOG_UI_MULTI_AGENT_ENV_LOAD_ERROR.code} "
                f"{LOG_UI_MULTI_AGENT_ENV_LOAD_ERROR.message} | "
                f"env_id=checkers error={e}"
            )
            raise

    def _init_board(self) -> None:
        """Initialize board with standard checkers starting position.

        Black pieces (player_0) start at rows 0-2 (top)
        White pieces (player_1) start at rows 5-7 (bottom)
        Only dark squares (where (row + col) is odd) are used
        """
        # Clear board
        self._board = [[0] * 8 for _ in range(8)]

        # Black pieces (value=1) at rows 0-2
        for row in range(3):
            for col in range(8):
                if (row + col) % 2 == 1:
                    self._board[row][col] = 1

        # White pieces (value=3) at rows 5-7
        for row in range(5, 8):
            for col in range(8):
                if (row + col) % 2 == 1:
                    self._board[row][col] = 3

    def get_state(self) -> CheckersState:
        """Get current Checkers game state.

        Returns:
            CheckersState with board, legal moves, game status, etc.
        """
        if not self._is_loaded or self._aec_env is None:
            raise RuntimeError("Checkers environment not loaded. Call load() first.")

        # Get legal moves from action mask
        obs, _, terminated, _, info = self._aec_env.last()

        # In Shimmy OpenSpiel, action_mask is in info dict
        action_mask = info.get("action_mask", np.array([])) if info else np.array([])
        legal_moves = [i for i in range(len(action_mask)) if action_mask[i] == 1]

        # Determine game over and winner
        is_game_over = terminated or len(legal_moves) == 0
        winner = None
        if is_game_over:
            winner = self._determine_winner()

        return CheckersState(
            board=[row[:] for row in self._board],  # Deep copy
            current_player=self._current_player,
            current_agent=self._current_player,
            legal_moves=legal_moves,
            last_move=self._last_move,
            is_game_over=is_game_over,
            winner=winner,
            move_count=self._move_count,
        )

    def make_move(self, action: int) -> CheckersState:
        """Execute a move by action index.

        Args:
            action: Action index from legal_moves

        Returns:
            New Checkers state after the move

        Raises:
            ValueError: If action is invalid
            RuntimeError: If environment not loaded
        """
        if not self._is_loaded or self._aec_env is None:
            raise RuntimeError("Checkers environment not loaded. Call load() first.")

        # Validate action
        obs, _, _, _, info = self._aec_env.last()
        action_mask = info.get("action_mask", np.array([])) if info else np.array([])

        if action < 0 or action >= len(action_mask) or action_mask[action] == 0:
            raise ValueError(f"Invalid action: {action}")

        self._last_move = action
        self._move_count += 1

        # Execute step in environment
        self._aec_env.step(action)

        # Update current player
        self._current_player = self._aec_env.agent_selection

        # Sync board state from observation
        self._sync_board_from_state()

        return self.get_state()

    def _sync_board_from_state(self) -> None:
        """Sync internal board state from OpenSpiel game state.

        OpenSpiel checkers uses a specific encoding for the board.
        We parse the string representation to extract piece positions.
        """
        if self._aec_env is None:
            return

        try:
            # Try to get the game state string representation
            # Shimmy's OpenSpiel wrapper provides access to the underlying state
            if hasattr(self._aec_env, 'game_state') and self._aec_env.game_state is not None:
                state_str = str(self._aec_env.game_state)
                self._parse_board_from_string(state_str)
            elif hasattr(self._aec_env, '_state') and self._aec_env._state is not None:
                state_str = str(self._aec_env._state)
                self._parse_board_from_string(state_str)
        except Exception as e:
            _LOG.debug(f"Could not sync board from state: {e}")

    def _parse_board_from_string(self, state_str: str) -> None:
        """Parse board from OpenSpiel state string representation.

        OpenSpiel checkers state string format:
        - 'o': Black piece (player_0)
        - '8': Black king (player_0 king, always the character '8')
        - '+': White piece (player_1)
        - '*': White king (player_1 king)
        - '.': Empty square

        Each line starts with a rank digit (8-1), followed by 8 piece chars.
        """
        lines = state_str.strip().split('\n')

        # Reset board
        self._board = [[0] * 8 for _ in range(8)]

        for line in lines:
            if not line or not line[0].isdigit():
                continue  # Skip column labels and empty lines

            rank = int(line[0])
            row = 8 - rank  # rank 8 = row 0, rank 1 = row 7
            pieces = line[1:]

            for col, char in enumerate(pieces):
                if col >= 8:
                    break
                if char == 'o':
                    self._board[row][col] = 1  # Black piece
                elif char == '8':
                    self._board[row][col] = 2  # Black king
                elif char == '+':
                    self._board[row][col] = 3  # White piece
                elif char == '*':
                    self._board[row][col] = 4  # White king
                # '.' = empty (already 0)

    def _determine_winner(self) -> Optional[str]:
        """Determine winner based on game state.

        Returns:
            Winner player name, "draw", or None
        """
        if self._aec_env is None:
            return None

        try:
            # Check rewards to determine winner
            rewards = self._aec_env.rewards
            if rewards.get(self.PLAYER_0, 0) > 0:
                return self.PLAYER_0
            elif rewards.get(self.PLAYER_1, 0) > 0:
                return self.PLAYER_1
            elif rewards.get(self.PLAYER_0, 0) < 0:
                return self.PLAYER_1
            elif rewards.get(self.PLAYER_1, 0) < 0:
                return self.PLAYER_0
            else:
                return "draw"
        except Exception:
            return None

    def is_move_legal(self, action: int) -> bool:
        """Check if a move is legal.

        Args:
            action: Action index

        Returns:
            True if the move is legal
        """
        if not self._is_loaded or self._aec_env is None:
            return False

        obs, _, _, _, info = self._aec_env.last()
        action_mask = info.get("action_mask", np.array([])) if info else np.array([])

        if action < 0 or action >= len(action_mask):
            return False

        return bool(action_mask[action] == 1)

    def get_legal_moves(self) -> List[int]:
        """Get all legal move indices.

        Returns:
            List of legal action indices
        """
        if not self._is_loaded or self._aec_env is None:
            return []

        obs, _, _, _, info = self._aec_env.last()
        action_mask = info.get("action_mask", np.array([])) if info else np.array([])
        return [i for i in range(len(action_mask)) if action_mask[i] == 1]

    def render(self) -> Optional[np.ndarray]:
        """Get pixel render of current board state.

        Returns:
            RGB numpy array (H, W, 3) or None if render fails
        """
        if not self._is_loaded or self._aec_env is None:
            return None

        try:
            return self._aec_env.render()
        except Exception as e:
            _LOG.warning(f"Render failed: {e}")
            return None

    def close(self) -> None:
        """Clean up environment resources."""
        if self._aec_env is not None:
            try:
                self._aec_env.close()
            except Exception:
                pass

        self._aec_env = None
        self._is_loaded = False

    def __del__(self) -> None:
        """Ensure cleanup on deletion."""
        self.close()


__all__ = ["CheckersAdapter", "CheckersState"]
