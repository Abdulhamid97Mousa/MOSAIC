"""Go-specific adapter for PettingZoo go_v5 environment.

This adapter provides a state-based interface for Go, enabling:
- Board state for Qt-based rendering (NxN grid)
- Legal move validation via action masks
- Intersection-based move execution
- Pass action support
- Turn management

The adapter uses the AEC API directly for turn-based gameplay.
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
class GoState:
    """Structured Go game state for UI rendering."""

    board: List[List[int]]  # NxN grid, 0=empty, 1=black, 2=white
    board_size: int  # 9, 13, or 19
    current_player: str  # "black_0" or "white_0"
    current_agent: str  # Same as current_player
    legal_moves: List[int]  # Valid action indices (0 to N*N for placements, N*N for pass)
    last_move: Optional[Tuple[int, int]] = None  # (row, col) of last move, None if pass
    last_was_pass: bool = False
    is_game_over: bool = False
    winner: Optional[str] = None  # "black_0", "white_0", or "draw"
    black_captures: int = 0  # Stones captured by black
    white_captures: int = 0  # Stones captured by white
    move_count: int = 0
    komi: float = 7.5
    consecutive_passes: int = 0  # Game ends after 2 consecutive passes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for signal emission."""
        return {
            "game_type": "go",
            "board": self.board,
            "board_size": self.board_size,
            "current_player": self.current_player,
            "current_agent": self.current_agent,
            "legal_moves": self.legal_moves,
            "last_move": self.last_move,
            "last_was_pass": self.last_was_pass,
            "is_game_over": self.is_game_over,
            "winner": self.winner,
            "black_captures": self.black_captures,
            "white_captures": self.white_captures,
            "move_count": self.move_count,
            "komi": self.komi,
            "consecutive_passes": self.consecutive_passes,
        }


class GoAdapter:
    """Adapter for PettingZoo Go environment with state-based interface.

    This adapter wraps the PettingZoo go_v5 environment and provides:
    - State-based interface (board + legal moves) for Qt rendering
    - Move execution via action index or (row, col) coordinates
    - Pass action support
    - Automatic turn management

    Example usage:
        adapter = GoAdapter(board_size=19)
        adapter.load(seed=42)

        state = adapter.get_state()
        # state.board = [[0]*19 for _ in range(19)]
        # state.legal_moves = [0, 1, 2, ..., 361]  (including pass)

        new_state = adapter.make_move(180)  # Place stone at center (9, 9)
        # Or use coordinates:
        new_state = adapter.make_move_at(9, 9)
        # Or pass:
        new_state = adapter.pass_turn()
    """

    # Agent mapping
    BLACK_AGENT = "black_0"  # Black goes first
    WHITE_AGENT = "white_0"

    def __init__(self, board_size: int = 19, komi: float = 7.5) -> None:
        """Initialize the Go adapter.

        Args:
            board_size: Board size (9, 13, or 19)
            komi: Komi compensation for white (default 7.5)
        """
        self._board_size = board_size
        self._komi = komi
        self._aec_env: Any = None
        self._board: List[List[int]] = [[0] * board_size for _ in range(board_size)]
        self._last_move: Optional[Tuple[int, int]] = None
        self._last_was_pass: bool = False
        self._move_count: int = 0
        self._current_player: str = self.BLACK_AGENT
        self._black_captures: int = 0
        self._white_captures: int = 0
        self._consecutive_passes: int = 0
        self._is_loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        """Check if environment is loaded."""
        return self._is_loaded

    @property
    def board_size(self) -> int:
        """Get the board size."""
        return self._board_size

    @property
    def pass_action(self) -> int:
        """Get the pass action index."""
        return self._board_size ** 2

    def load(self, seed: int = 42, render_mode: str = "rgb_array") -> GoState:
        """Load and reset the Go environment.

        Args:
            seed: Random seed for environment reset
            render_mode: Render mode ("rgb_array" for pixel output)

        Returns:
            Initial Go state
        """
        try:
            from pettingzoo.classic import go_v5

            # Create AEC environment
            self._aec_env = go_v5.env(
                board_size=self._board_size,
                komi=self._komi,
                render_mode=render_mode,
            )

            # Reset environment
            self._aec_env.reset(seed=seed)

            # Reset tracking
            self._board = [[0] * self._board_size for _ in range(self._board_size)]
            self._last_move = None
            self._last_was_pass = False
            self._move_count = 0
            self._current_player = self.BLACK_AGENT
            self._black_captures = 0
            self._white_captures = 0
            self._consecutive_passes = 0
            self._is_loaded = True

            _LOG.info(
                f"{LOG_UI_MULTI_AGENT_ENV_LOADED.code} "
                f"{LOG_UI_MULTI_AGENT_ENV_LOADED.message} | "
                f"env_id=go_v5 seed={seed} board_size={self._board_size}"
            )

            return self.get_state()

        except Exception as e:
            _LOG.error(
                f"{LOG_UI_MULTI_AGENT_ENV_LOAD_ERROR.code} "
                f"{LOG_UI_MULTI_AGENT_ENV_LOAD_ERROR.message} | "
                f"env_id=go_v5 error={e}"
            )
            raise

    def get_state(self) -> GoState:
        """Get current Go game state.

        Returns:
            GoState with board, legal moves, game status, etc.
        """
        if not self._is_loaded or self._aec_env is None:
            raise RuntimeError("Go environment not loaded. Call load() first.")

        # Get legal moves from action mask
        obs, _, terminated, _, _ = self._aec_env.last()
        action_mask = obs["action_mask"]
        legal_moves = [i for i in range(len(action_mask)) if action_mask[i] == 1]

        # Determine game over and winner
        is_game_over = terminated or self._consecutive_passes >= 2
        winner = None
        if is_game_over:
            winner = self._determine_winner()

        return GoState(
            board=[row[:] for row in self._board],  # Deep copy
            board_size=self._board_size,
            current_player=self._current_player,
            current_agent=self._current_player,
            legal_moves=legal_moves,
            last_move=self._last_move,
            last_was_pass=self._last_was_pass,
            is_game_over=is_game_over,
            winner=winner,
            black_captures=self._black_captures,
            white_captures=self._white_captures,
            move_count=self._move_count,
            komi=self._komi,
            consecutive_passes=self._consecutive_passes,
        )

    def make_move(self, action: int) -> GoState:
        """Execute a move by action index.

        Args:
            action: Action index (0 to N*N-1 for placements, N*N for pass)

        Returns:
            New Go state after the move

        Raises:
            ValueError: If action is invalid
            RuntimeError: If environment not loaded
        """
        if not self._is_loaded or self._aec_env is None:
            raise RuntimeError("Go environment not loaded. Call load() first.")

        # Validate action
        total_actions = self._board_size ** 2 + 1
        if action < 0 or action >= total_actions:
            raise ValueError(f"Invalid action: {action}. Must be 0-{total_actions - 1}.")

        obs, _, _, _, _ = self._aec_env.last()
        action_mask = obs["action_mask"]

        if action_mask[action] == 0:
            raise ValueError(f"Action {action} is illegal.")

        # Track captures before move
        prev_board = [row[:] for row in self._board]

        # Handle pass vs placement
        is_pass = action == self.pass_action
        if is_pass:
            self._last_move = None
            self._last_was_pass = True
            self._consecutive_passes += 1
        else:
            row = action // self._board_size
            col = action % self._board_size
            player_value = 1 if self._current_player == self.BLACK_AGENT else 2
            self._board[row][col] = player_value
            self._last_move = (row, col)
            self._last_was_pass = False
            self._consecutive_passes = 0
            _LOG.debug(
                f"[MAKE_MOVE] Before sync: player={self._current_player}, "
                f"action={action}, placed value={player_value} at ({row},{col})"
            )

        self._move_count += 1

        # Execute step in environment
        self._aec_env.step(action)

        # Sync board state from observation (captures are handled by env)
        self._sync_board_from_observation()

        # Debug: Log board state after sync
        if not is_pass:
            _LOG.debug(
                f"[MAKE_MOVE] After sync: board[{row}][{col}]={self._board[row][col]}, "
                f"new_agent={self._aec_env.agent_selection}"
            )

        # Count captures
        self._count_captures(prev_board)

        # Update current player
        self._current_player = self._aec_env.agent_selection

        return self.get_state()

    def make_move_at(self, row: int, col: int) -> GoState:
        """Execute a move by placing a stone at the specified coordinates.

        Args:
            row: Row index (0 to board_size-1)
            col: Column index (0 to board_size-1)

        Returns:
            New Go state after the move
        """
        action = row * self._board_size + col
        return self.make_move(action)

    def pass_turn(self) -> GoState:
        """Pass the current turn.

        Returns:
            New Go state after passing
        """
        return self.make_move(self.pass_action)

    def _sync_board_from_observation(self) -> None:
        """Sync internal board state from observation planes.

        Instead of completely overwriting the board from observation,
        we only detect and apply captures. This is more reliable because:
        1. We already set the stone before step() is called
        2. We only need to remove captured stones
        """
        if self._aec_env is None:
            return

        obs, _, _, _, _ = self._aec_env.last()
        observation = obs["observation"]

        # PettingZoo Go observation planes (relative to the observing agent):
        # Plane 0: opponent's stones
        # Plane 1: current agent's (own) stones

        _LOG.debug(
            f"[SYNC] agent_selection={self._aec_env.agent_selection}"
        )

        # Detect captured stones: positions where we have a stone on our board
        # but no stone exists in the observation
        captures_removed = 0

        for row in range(self._board_size):
            for col in range(self._board_size):
                our_stone = self._board[row][col]

                # Check if there's ANY stone in the observation
                has_current_stone = float(observation[row, col, 0]) > 0.5
                has_opponent_stone = float(observation[row, col, 1]) > 0.5
                has_any_stone = has_current_stone or has_opponent_stone

                # If we have a stone but observation shows empty, it was captured
                if our_stone != 0 and not has_any_stone:
                    _LOG.debug(f"[SYNC] Stone captured at ({row},{col}), was value={our_stone}")
                    self._board[row][col] = 0
                    captures_removed += 1

        if captures_removed > 0:
            _LOG.debug(f"[SYNC] Removed {captures_removed} captured stones")

    def _count_captures(self, prev_board: List[List[int]]) -> None:
        """Count stones captured in the last move.

        Args:
            prev_board: Board state before the move
        """
        # Count stones that were on the board but are now gone
        for row in range(self._board_size):
            for col in range(self._board_size):
                prev_piece = prev_board[row][col]
                curr_piece = self._board[row][col]

                if prev_piece != 0 and curr_piece == 0:
                    # Stone was captured
                    if prev_piece == 1:  # Black stone captured
                        self._white_captures += 1
                    else:  # White stone captured
                        self._black_captures += 1

    def _determine_winner(self) -> Optional[str]:
        """Determine winner based on game state.

        Returns:
            Winner agent name, "draw", or None
        """
        if self._aec_env is None:
            return None

        # In PettingZoo Go, winner is determined by score after both pass
        try:
            rewards = self._aec_env.rewards
            if rewards.get(self.BLACK_AGENT, 0) > 0:
                return self.BLACK_AGENT
            elif rewards.get(self.WHITE_AGENT, 0) > 0:
                return self.WHITE_AGENT
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

        total_actions = self._board_size ** 2 + 1
        if action < 0 or action >= total_actions:
            return False

        obs, _, _, _, _ = self._aec_env.last()
        return bool(obs["action_mask"][action] == 1)

    def is_move_legal_at(self, row: int, col: int) -> bool:
        """Check if placing a stone at coordinates is legal.

        Args:
            row: Row index
            col: Column index

        Returns:
            True if the move is legal
        """
        if row < 0 or row >= self._board_size or col < 0 or col >= self._board_size:
            return False
        action = row * self._board_size + col
        return self.is_move_legal(action)

    def get_legal_moves(self) -> List[int]:
        """Get all legal move indices.

        Returns:
            List of legal action indices
        """
        if not self._is_loaded or self._aec_env is None:
            return []

        obs, _, _, _, _ = self._aec_env.last()
        action_mask = obs["action_mask"]
        return [i for i in range(len(action_mask)) if action_mask[i] == 1]

    def get_legal_positions(self) -> List[Tuple[int, int]]:
        """Get all legal board positions (excluding pass).

        Returns:
            List of (row, col) tuples where stones can be placed
        """
        legal_moves = self.get_legal_moves()
        positions = []
        for action in legal_moves:
            if action < self.pass_action:
                row = action // self._board_size
                col = action % self._board_size
                positions.append((row, col))
        return positions

    def action_to_coords(self, action: int) -> Optional[Tuple[int, int]]:
        """Convert action index to board coordinates.

        Args:
            action: Action index

        Returns:
            (row, col) tuple, or None for pass action
        """
        if action < 0 or action >= self.pass_action:
            return None
        row = action // self._board_size
        col = action % self._board_size
        return (row, col)

    def coords_to_action(self, row: int, col: int) -> int:
        """Convert board coordinates to action index.

        Args:
            row: Row index
            col: Column index

        Returns:
            Action index
        """
        return row * self._board_size + col

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


__all__ = ["GoAdapter", "GoState"]
