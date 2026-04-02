"""PettingZoo Classic environment adapters for turn-based games.

These adapters wrap PettingZoo Classic environments (Chess, Go, Connect Four, etc.)
to work with the GUI's EnvironmentAdapter interface.

PettingZoo Classic games are AEC (Agent Environment Cycle) turn-based games
where players take turns. The adapter handles:
- State-based rendering via FEN (Chess) or custom formats
- Legal move validation via action masks
- Turn management between players
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
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


@dataclass(slots=True)
class ChessRenderPayload:
    """Render payload for Chess containing FEN and game state."""

    fen: str
    current_player: str  # "white" or "black"
    legal_moves: List[str]  # UCI notation
    last_move: Optional[str] = None
    is_check: bool = False
    is_checkmate: bool = False
    is_stalemate: bool = False
    is_game_over: bool = False
    winner: Optional[str] = None
    move_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fen": self.fen,
            "current_player": self.current_player,
            "legal_moves": self.legal_moves,
            "last_move": self.last_move,
            "is_check": self.is_check,
            "is_checkmate": self.is_checkmate,
            "is_stalemate": self.is_stalemate,
            "is_game_over": self.is_game_over,
            "winner": self.winner,
            "move_count": self.move_count,
        }


class ChessEnvironmentAdapter(EnvironmentAdapter[Dict[str, Any], int]):
    """Adapter for PettingZoo Chess environment.

    This adapter wraps chess_v6 from PettingZoo Classic and provides:
    - State-based interface (FEN + legal moves) for Qt rendering
    - Move execution via action index or UCI notation
    - Turn management for human vs human play

    The observation is a dictionary containing:
    - "observation": The raw observation array (8x8x111)
    - "action_mask": Binary mask of legal actions

    The render_payload is a ChessRenderPayload with FEN and game state.
    """

    id = GameId.CHESS.value
    supported_control_modes = (ControlMode.HUMAN_ONLY,)
    supported_render_modes = (RenderMode.RGB_ARRAY,)
    default_render_mode = RenderMode.RGB_ARRAY

    # Agent mapping
    WHITE_AGENT = "player_0"
    BLACK_AGENT = "player_1"

    def __init__(self, context: AdapterContext | None = None) -> None:
        super().__init__(context)
        self._aec_env: Any = None
        self._board: Any = None  # chess.Board reference
        self._last_move: Optional[str] = None
        self._move_count: int = 0

    def load(self) -> None:
        """Load the PettingZoo chess environment."""
        try:
            from pettingzoo.classic import chess_v6

            self._aec_env = chess_v6.env(render_mode="rgb_array")
            self.log_constant(
                LOG_ADAPTER_ENV_CREATED,
                extra={
                    "env_id": self.id,
                    "render_mode": "rgb_array",
                    "gym_kwargs": "-",
                    "wrapped_class": "chess_v6.AECEnv",
                },
            )
        except ImportError as e:
            raise ImportError(
                "PettingZoo Classic is required. Install with: pip install 'pettingzoo[classic]'"
            ) from e

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> AdapterStep[Dict[str, Any]]:
        """Reset the chess environment."""
        if self._aec_env is None:
            raise RuntimeError("Environment not loaded. Call load() first.")

        self._aec_env.reset(seed=seed)
        self._board = self._aec_env.unwrapped.board
        self._last_move = None
        self._move_count = 0
        self._episode_step = 0
        self._episode_return = 0.0

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

        return self._package_step(obs, 0.0, False, False, dict(info))

    def step(self, action: int) -> AdapterStep[Dict[str, Any]]:
        """Execute an action (action index from the 4672 action space)."""
        if self._aec_env is None or self._board is None:
            raise RuntimeError("Environment not loaded. Call load() first.")

        # Convert action to UCI for tracking
        from pettingzoo.classic.chess import chess_utils

        player = 0 if self._board.turn else 1
        try:
            move = chess_utils.action_to_move(self._board, action, player)
            self._last_move = str(move)
        except Exception:
            self._last_move = None

        # Execute step
        self._aec_env.step(action)
        self._move_count += 1
        self._episode_step += 1

        # Get new observation
        obs, reward, terminated, truncated, info = self._aec_env.last()

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

    def step_uci(self, uci_move: str) -> AdapterStep[Dict[str, Any]]:
        """Execute a move in UCI notation (e.g., 'e2e4').

        This is a convenience method for human input.
        """
        if self._aec_env is None or self._board is None:
            raise RuntimeError("Environment not loaded. Call load() first.")

        import chess
        from pettingzoo.classic.chess import chess_utils

        # Parse and validate move
        try:
            move = chess.Move.from_uci(uci_move)
        except ValueError as e:
            raise ValueError(f"Invalid UCI move format: {uci_move}") from e

        if move not in self._board.legal_moves:
            raise ValueError(f"Illegal move: {uci_move}")

        # Find action index
        action = self._move_to_action(move)

        return self.step(action)

    def _move_to_action(self, move: Any) -> int:
        """Convert chess.Move to PettingZoo action index."""
        from pettingzoo.classic.chess import chess_utils

        obs, _, _, _, _ = self._aec_env.last()
        action_mask = obs["action_mask"]
        player = 0 if self._board.turn else 1
        legal_indices = np.where(action_mask == 1)[0]
        move_uci = str(move)

        for action_idx in legal_indices:
            decoded_move = chess_utils.action_to_move(self._board, action_idx, player)
            if str(decoded_move) == move_uci:
                return int(action_idx)

        raise ValueError(f"Could not find action index for move: {move}")

    def close(self) -> None:
        """Clean up environment resources."""
        if self._aec_env is not None:
            try:
                self._aec_env.close()
            except Exception:
                pass
        self._aec_env = None
        self._board = None

    def render(self) -> np.ndarray | None:
        """Get RGB render of current board state."""
        if self._aec_env is None:
            return None
        try:
            return self._aec_env.render()
        except Exception:
            return None

    def get_chess_state(self) -> ChessRenderPayload:
        """Get structured chess state for Qt rendering."""
        if self._board is None:
            raise RuntimeError("Environment not loaded.")

        is_white_turn = self._board.turn
        current_player = "white" if is_white_turn else "black"
        legal_moves = [str(m) for m in self._board.legal_moves]

        is_checkmate = self._board.is_checkmate()
        is_stalemate = self._board.is_stalemate()
        is_game_over = self._board.is_game_over()

        winner = None
        if is_checkmate:
            winner = "black" if is_white_turn else "white"
        elif is_stalemate or (is_game_over and not is_checkmate):
            winner = "draw"

        return ChessRenderPayload(
            fen=self._board.fen(),
            current_player=current_player,
            legal_moves=legal_moves,
            last_move=self._last_move,
            is_check=self._board.is_check(),
            is_checkmate=is_checkmate,
            is_stalemate=is_stalemate,
            is_game_over=is_game_over,
            winner=winner,
            move_count=self._move_count,
        )

    def get_legal_moves_from_square(self, square: str) -> List[str]:
        """Get legal destination squares from a specific square."""
        if self._board is None:
            return []

        import chess

        try:
            from_square = chess.parse_square(square)
        except ValueError:
            return []

        destinations = []
        for move in self._board.legal_moves:
            if move.from_square == from_square:
                destinations.append(chess.square_name(move.to_square))

        return destinations

    def is_move_legal(self, uci_move: str) -> bool:
        """Check if a move is legal."""
        if self._board is None:
            return False

        import chess

        try:
            move = chess.Move.from_uci(uci_move)
            return move in self._board.legal_moves
        except ValueError:
            return False

    def current_agent(self) -> str:
        """Get the current agent's turn."""
        if self._board is None:
            return self.WHITE_AGENT
        return self.WHITE_AGENT if self._board.turn else self.BLACK_AGENT

    def _package_step(
        self,
        observation: Dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
    ) -> AdapterStep[Dict[str, Any]]:
        """Package step result with chess-specific render payload."""
        # Get chess state for rendering
        try:
            chess_state = self.get_chess_state()
            render_payload = chess_state.to_dict()
        except Exception:
            render_payload = None

        # Build step state
        current_player = "white" if self._board and self._board.turn else "black"
        agents = [
            AgentSnapshot(name="white", role="player"),
            AgentSnapshot(name="black", role="player"),
        ]

        state = StepState(
            active_agent=current_player,
            agents=agents,
            metrics={
                "move_count": self._move_count,
                "episode_step": self._episode_step,
            },
            environment={
                "game": "chess",
                "current_player": current_player,
            },
        )

        return AdapterStep(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            render_payload=render_payload,
            render_hint={"type": "chess", "use_qt_widget": True},
            agent_id=self.current_agent(),
            state=state,
        )

    # Required abstract methods from EnvironmentAdapter
    def gym_kwargs(self) -> dict[str, Any]:
        """Return additional kwargs for gym.make (not used for PettingZoo)."""
        return {}

    def apply_wrappers(self, env: Any) -> Any:
        """Apply wrappers to environment (not used for PettingZoo)."""
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


@dataclass(slots=True)
class ConnectFourRenderPayload:
    """Render payload for Connect Four containing board state."""

    board: List[List[int]]  # 6x7 grid, 0=empty, 1=player_0, 2=player_1
    current_player: str  # "player_0" or "player_1"
    legal_columns: List[int]  # Columns where a piece can be dropped (0-6)
    last_column: Optional[int] = None  # Last column played
    is_game_over: bool = False
    winner: Optional[str] = None  # "player_0", "player_1", or "draw"
    move_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game_type": "connect_four",
            "board": self.board,
            "current_player": self.current_player,
            "legal_columns": self.legal_columns,
            "last_column": self.last_column,
            "is_game_over": self.is_game_over,
            "winner": self.winner,
            "move_count": self.move_count,
        }


class ConnectFourEnvironmentAdapter(EnvironmentAdapter[Dict[str, Any], int]):
    """Adapter for PettingZoo Connect Four environment.

    This adapter wraps connect_four_v3 from PettingZoo Classic and provides:
    - Board state representation for Qt rendering
    - Move execution via column index (0-6)
    - Turn management for human vs human play

    The observation is a dictionary containing:
    - "observation": The raw observation array (6x7x2)
    - "action_mask": Binary mask of legal columns (7 values)

    The render_payload is a ConnectFourRenderPayload with board state.
    """

    id = GameId.CONNECT_FOUR.value
    supported_control_modes = (ControlMode.HUMAN_ONLY,)
    supported_render_modes = (RenderMode.RGB_ARRAY,)
    default_render_mode = RenderMode.RGB_ARRAY

    # Agent mapping
    PLAYER_0 = "player_0"
    PLAYER_1 = "player_1"

    def __init__(self, context: AdapterContext | None = None) -> None:
        super().__init__(context)
        self._aec_env: Any = None
        self._board: List[List[int]] = [[0] * 7 for _ in range(6)]
        self._last_column: Optional[int] = None
        self._move_count: int = 0
        self._current_player: str = self.PLAYER_0

    def load(self) -> None:
        """Load the PettingZoo Connect Four environment."""
        try:
            from pettingzoo.classic import connect_four_v3

            self._aec_env = connect_four_v3.env(render_mode="rgb_array")
            self.log_constant(
                LOG_ADAPTER_ENV_CREATED,
                extra={
                    "env_id": self.id,
                    "render_mode": "rgb_array",
                    "gym_kwargs": "-",
                    "wrapped_class": "connect_four_v3.AECEnv",
                },
            )
        except ImportError as e:
            raise ImportError(
                "PettingZoo Classic is required. Install with: pip install 'pettingzoo[classic]'"
            ) from e

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> AdapterStep[Dict[str, Any]]:
        """Reset the Connect Four environment."""
        if self._aec_env is None:
            raise RuntimeError("Environment not loaded. Call load() first.")

        self._aec_env.reset(seed=seed)
        self._board = [[0] * 7 for _ in range(6)]
        self._last_column = None
        self._move_count = 0
        self._episode_step = 0
        self._episode_return = 0.0
        self._current_player = self.PLAYER_0

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

        return self._package_step(obs, 0.0, False, False, dict(info))

    def step(self, action: int) -> AdapterStep[Dict[str, Any]]:
        """Execute an action (column index 0-6)."""
        if self._aec_env is None:
            raise RuntimeError("Environment not loaded. Call load() first.")

        # Validate action
        obs, _, _, _, _ = self._aec_env.last()
        action_mask = obs["action_mask"]
        if action < 0 or action >= 7 or action_mask[action] == 0:
            raise ValueError(f"Illegal column: {action}")

        # Update internal board state
        self._drop_piece(action)
        self._last_column = action
        self._move_count += 1
        self._episode_step += 1

        # Execute step
        self._aec_env.step(action)

        # Get new observation
        obs, reward, terminated, truncated, info = self._aec_env.last()

        # Update current player
        self._current_player = self._aec_env.agent_selection

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

    def _drop_piece(self, column: int) -> None:
        """Drop a piece in the specified column."""
        player_value = 1 if self._current_player == self.PLAYER_0 else 2
        # Find the lowest empty row in the column
        for row in range(5, -1, -1):
            if self._board[row][column] == 0:
                self._board[row][column] = player_value
                break

    def close(self) -> None:
        """Clean up environment resources."""
        if self._aec_env is not None:
            try:
                self._aec_env.close()
            except Exception:
                pass
        self._aec_env = None

    def render(self) -> np.ndarray | None:
        """Get RGB render of current board state."""
        if self._aec_env is None:
            return None
        try:
            return self._aec_env.render()
        except Exception:
            return None

    def get_connect_four_state(self) -> ConnectFourRenderPayload:
        """Get structured Connect Four state for Qt rendering."""
        if self._aec_env is None:
            raise RuntimeError("Environment not loaded.")

        obs, _, terminated, _, _ = self._aec_env.last()
        action_mask = obs["action_mask"]
        legal_columns = [i for i in range(7) if action_mask[i] == 1]

        # Determine winner
        winner = None
        is_game_over = terminated or len(legal_columns) == 0
        if is_game_over:
            winner = self._determine_winner()

        return ConnectFourRenderPayload(
            board=[row[:] for row in self._board],  # Deep copy
            current_player=self._current_player,
            legal_columns=legal_columns,
            last_column=self._last_column,
            is_game_over=is_game_over,
            winner=winner,
            move_count=self._move_count,
        )

    def _determine_winner(self) -> Optional[str]:
        """Check if there's a winner by looking for 4 in a row."""
        # Check horizontal, vertical, and diagonal lines
        for row in range(6):
            for col in range(7):
                piece = self._board[row][col]
                if piece == 0:
                    continue
                # Check horizontal
                if col <= 3:
                    if all(self._board[row][col + i] == piece for i in range(4)):
                        return self.PLAYER_0 if piece == 1 else self.PLAYER_1
                # Check vertical
                if row <= 2:
                    if all(self._board[row + i][col] == piece for i in range(4)):
                        return self.PLAYER_0 if piece == 1 else self.PLAYER_1
                # Check diagonal (down-right)
                if row <= 2 and col <= 3:
                    if all(self._board[row + i][col + i] == piece for i in range(4)):
                        return self.PLAYER_0 if piece == 1 else self.PLAYER_1
                # Check diagonal (up-right)
                if row >= 3 and col <= 3:
                    if all(self._board[row - i][col + i] == piece for i in range(4)):
                        return self.PLAYER_0 if piece == 1 else self.PLAYER_1

        # Check for draw (board full)
        if all(self._board[0][col] != 0 for col in range(7)):
            return "draw"

        return None

    def get_legal_columns(self) -> List[int]:
        """Get list of columns where a piece can be dropped."""
        if self._aec_env is None:
            return []
        obs, _, _, _, _ = self._aec_env.last()
        action_mask = obs["action_mask"]
        return [i for i in range(7) if action_mask[i] == 1]

    def is_column_legal(self, column: int) -> bool:
        """Check if dropping a piece in the column is legal."""
        if self._aec_env is None or column < 0 or column >= 7:
            return False
        obs, _, _, _, _ = self._aec_env.last()
        return bool(obs["action_mask"][column] == 1)

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
        """Package step result with Connect Four-specific render payload."""
        # Get game state for rendering
        try:
            game_state = self.get_connect_four_state()
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
                "game": "connect_four",
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
            render_hint={"type": "connect_four", "use_qt_widget": True},
            agent_id=self.current_agent(),
            state=state,
        )

    # Required abstract methods from EnvironmentAdapter
    def gym_kwargs(self) -> dict[str, Any]:
        """Return additional kwargs for gym.make (not used for PettingZoo)."""
        return {}

    def apply_wrappers(self, env: Any) -> Any:
        """Apply wrappers to environment (not used for PettingZoo)."""
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


@dataclass(slots=True)
class GoRenderPayload:
    """Render payload for Go containing board state."""

    board: List[List[int]]  # NxN grid, 0=empty, 1=black, 2=white
    board_size: int  # 9, 13, or 19
    current_player: str  # "black_0" or "white_0"
    legal_moves: List[int]  # Valid action indices including pass
    last_move: Optional[int] = None  # Last action taken (None if pass)
    is_game_over: bool = False
    winner: Optional[str] = None  # "black_0", "white_0", or "draw"
    black_captures: int = 0  # Stones captured by black
    white_captures: int = 0  # Stones captured by white
    move_count: int = 0
    komi: float = 7.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game_type": "go",
            "board": self.board,
            "board_size": self.board_size,
            "current_player": self.current_player,
            "legal_moves": self.legal_moves,
            "last_move": self.last_move,
            "is_game_over": self.is_game_over,
            "winner": self.winner,
            "black_captures": self.black_captures,
            "white_captures": self.white_captures,
            "move_count": self.move_count,
            "komi": self.komi,
        }


class GoEnvironmentAdapter(EnvironmentAdapter[Dict[str, Any], int]):
    """Adapter for PettingZoo Go environment.

    This adapter wraps go_v5 from PettingZoo Classic and provides:
    - Board state representation for Qt rendering
    - Move execution via action index (0 to N*N for placements, N*N for pass)
    - Turn management for human vs human play

    The observation is a dictionary containing:
    - "observation": The raw observation array (N x N x 3)
    - "action_mask": Binary mask of legal actions

    The render_payload is a GoRenderPayload with board state.
    """

    id = GameId.GO.value
    supported_control_modes = (ControlMode.HUMAN_ONLY,)
    supported_render_modes = (RenderMode.RGB_ARRAY,)
    default_render_mode = RenderMode.RGB_ARRAY

    # Agent mapping
    BLACK_AGENT = "black_0"
    WHITE_AGENT = "white_0"

    def __init__(
        self,
        context: AdapterContext | None = None,
        board_size: int = 19,
        komi: float = 7.5,
    ) -> None:
        super().__init__(context)
        self._aec_env: Any = None
        self._board_size: int = board_size
        self._komi: float = komi
        self._board: List[List[int]] = [[0] * board_size for _ in range(board_size)]
        self._last_move: Optional[int] = None
        self._move_count: int = 0
        self._current_player: str = self.BLACK_AGENT
        self._black_captures: int = 0
        self._white_captures: int = 0

    def load(self) -> None:
        """Load the PettingZoo Go environment."""
        try:
            from pettingzoo.classic import go_v5

            self._aec_env = go_v5.env(
                board_size=self._board_size,
                komi=self._komi,
                render_mode="rgb_array",
            )
            self.log_constant(
                LOG_ADAPTER_ENV_CREATED,
                extra={
                    "env_id": self.id,
                    "render_mode": "rgb_array",
                    "gym_kwargs": f"board_size={self._board_size}, komi={self._komi}",
                    "wrapped_class": "go_v5.AECEnv",
                },
            )
        except ImportError as e:
            raise ImportError(
                "PettingZoo Classic is required. Install with: pip install 'pettingzoo[classic]'"
            ) from e

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> AdapterStep[Dict[str, Any]]:
        """Reset the Go environment."""
        if self._aec_env is None:
            raise RuntimeError("Environment not loaded. Call load() first.")

        self._aec_env.reset(seed=seed)
        self._board = [[0] * self._board_size for _ in range(self._board_size)]
        self._last_move = None
        self._move_count = 0
        self._episode_step = 0
        self._episode_return = 0.0
        self._current_player = self.BLACK_AGENT
        self._black_captures = 0
        self._white_captures = 0

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

        return self._package_step(obs, 0.0, False, False, dict(info))

    def step(self, action: int) -> AdapterStep[Dict[str, Any]]:
        """Execute an action (0 to N*N-1 for placement, N*N for pass)."""
        if self._aec_env is None:
            raise RuntimeError("Environment not loaded. Call load() first.")

        # Validate action
        obs, _, _, _, _ = self._aec_env.last()
        action_mask = obs["action_mask"]
        if action < 0 or action > self._board_size ** 2 or action_mask[action] == 0:
            raise ValueError(f"Illegal action: {action}")

        # Update internal board state (if not pass)
        pass_action = self._board_size ** 2
        if action != pass_action:
            row = action // self._board_size
            col = action % self._board_size
            player_value = 1 if self._current_player == self.BLACK_AGENT else 2
            self._board[row][col] = player_value

        self._last_move = action if action != pass_action else None
        self._move_count += 1
        self._episode_step += 1

        # Execute step
        self._aec_env.step(action)

        # Get new observation
        obs, reward, terminated, truncated, info = self._aec_env.last()

        # Update current player
        self._current_player = self._aec_env.agent_selection

        # Sync board state from observation
        self._sync_board_from_observation(obs)

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

    def _sync_board_from_observation(self, obs: Dict[str, Any]) -> None:
        """Sync internal board state from observation planes.

        PettingZoo Go observation planes (relative to the observing agent):
        - Plane 0: opponent's stones
        - Plane 1: current agent's (own) stones
        """
        observation = obs["observation"]
        is_black = self._current_player == self.BLACK_AGENT

        for row in range(self._board_size):
            for col in range(self._board_size):
                opponent_stones = observation[row, col, 0]
                own_stones = observation[row, col, 1]

                if own_stones == 1:
                    # Current player's stone
                    self._board[row][col] = 1 if is_black else 2
                elif opponent_stones == 1:
                    # Opponent's stone
                    self._board[row][col] = 2 if is_black else 1
                else:
                    self._board[row][col] = 0

    def close(self) -> None:
        """Clean up environment resources."""
        if self._aec_env is not None:
            try:
                self._aec_env.close()
            except Exception:
                pass
        self._aec_env = None

    def render(self) -> np.ndarray | None:
        """Get RGB render of current board state."""
        if self._aec_env is None:
            return None
        try:
            return self._aec_env.render()
        except Exception:
            return None

    def get_go_state(self) -> GoRenderPayload:
        """Get structured Go state for Qt rendering."""
        if self._aec_env is None:
            raise RuntimeError("Environment not loaded.")

        obs, _, terminated, _, _ = self._aec_env.last()
        action_mask = obs["action_mask"]
        legal_moves = [i for i in range(len(action_mask)) if action_mask[i] == 1]

        # Determine winner (simplified - Go scoring is complex)
        winner = None
        is_game_over = terminated
        if is_game_over:
            # Check final rewards to determine winner
            # This is a simplification - actual Go uses territory scoring
            winner = self._determine_winner()

        return GoRenderPayload(
            board=[row[:] for row in self._board],
            board_size=self._board_size,
            current_player=self._current_player,
            legal_moves=legal_moves,
            last_move=self._last_move,
            is_game_over=is_game_over,
            winner=winner,
            black_captures=self._black_captures,
            white_captures=self._white_captures,
            move_count=self._move_count,
            komi=self._komi,
        )

    def _determine_winner(self) -> Optional[str]:
        """Determine winner based on game state."""
        # In PettingZoo Go, the winner is determined by score after both pass
        # The rewards are +1 for winner, -1 for loser
        # We can check the last agent's reward
        if self._aec_env is None:
            return None

        # Get rewards from environment
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

    def get_legal_moves(self) -> List[int]:
        """Get list of legal move indices."""
        if self._aec_env is None:
            return []
        obs, _, _, _, _ = self._aec_env.last()
        action_mask = obs["action_mask"]
        return [i for i in range(len(action_mask)) if action_mask[i] == 1]

    def is_move_legal(self, action: int) -> bool:
        """Check if a move is legal."""
        if self._aec_env is None or action < 0 or action > self._board_size ** 2:
            return False
        obs, _, _, _, _ = self._aec_env.last()
        return bool(obs["action_mask"][action] == 1)

    def action_to_coords(self, action: int) -> Optional[tuple[int, int]]:
        """Convert action index to board coordinates."""
        if action < 0 or action >= self._board_size ** 2:
            return None  # Pass action or invalid
        row = action // self._board_size
        col = action % self._board_size
        return (row, col)

    def coords_to_action(self, row: int, col: int) -> int:
        """Convert board coordinates to action index."""
        return row * self._board_size + col

    def get_pass_action(self) -> int:
        """Get the pass action index."""
        return self._board_size ** 2

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
        """Package step result with Go-specific render payload."""
        # Get game state for rendering
        try:
            game_state = self.get_go_state()
            render_payload = game_state.to_dict()
        except Exception:
            render_payload = None

        # Build step state
        agents = [
            AgentSnapshot(name="black", role="player"),
            AgentSnapshot(name="white", role="player"),
        ]

        state = StepState(
            active_agent=self._current_player,
            agents=agents,
            metrics={
                "move_count": self._move_count,
                "episode_step": self._episode_step,
                "black_captures": self._black_captures,
                "white_captures": self._white_captures,
            },
            environment={
                "game": "go",
                "board_size": self._board_size,
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
            render_hint={"type": "go", "use_qt_widget": True},
            agent_id=self.current_agent(),
            state=state,
        )

    # Required abstract methods from EnvironmentAdapter
    def gym_kwargs(self) -> dict[str, Any]:
        """Return additional kwargs for gym.make (not used for PettingZoo)."""
        return {"board_size": self._board_size, "komi": self._komi}

    def apply_wrappers(self, env: Any) -> Any:
        """Apply wrappers to environment (not used for PettingZoo)."""
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


@dataclass(slots=True)
class CardGameRenderPayload:
    """Render payload for card/poker games (Hanabi, Leduc Hold'em, RPS, Texas Hold'em)."""

    game_type: str  # "hanabi", "leduc_holdem", "rps", "texas_holdem"
    current_player: str
    legal_actions: List[int]
    last_action: Optional[int] = None
    is_game_over: bool = False
    winner: Optional[str] = None
    move_count: int = 0
    game_state: Dict[str, Any] = field(default_factory=dict)  # Game-specific state

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game_type": self.game_type,
            "current_player": self.current_player,
            "legal_actions": self.legal_actions,
            "last_action": self.last_action,
            "is_game_over": self.is_game_over,
            "winner": self.winner,
            "move_count": self.move_count,
            "game_state": self.game_state,
        }


class RPSEnvironmentAdapter(EnvironmentAdapter[Dict[str, Any], int]):
    """Adapter for PettingZoo Rock Paper Scissors environment."""

    id = GameId.RPS.value
    supported_control_modes = (ControlMode.HUMAN_ONLY,)
    supported_render_modes = (RenderMode.RGB_ARRAY,)
    default_render_mode = RenderMode.RGB_ARRAY

    def __init__(self, context: AdapterContext | None = None) -> None:
        super().__init__(context)
        self._aec_env: Any = None
        self._last_action: Optional[int] = None
        self._move_count: int = 0
        self._current_player: str = "player_0"

    def load(self) -> None:
        try:
            from pettingzoo.classic import rps_v2
            self._aec_env = rps_v2.env(render_mode="rgb_array")
            self.log_constant(LOG_ADAPTER_ENV_CREATED, extra={"env_id": self.id, "render_mode": "rgb_array"})
        except ImportError as e:
            raise ImportError("PettingZoo Classic required: pip install 'pettingzoo[classic]'") from e

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> AdapterStep[Dict[str, Any]]:
        if self._aec_env is None:
            raise RuntimeError("Environment not loaded. Call load() first.")
        self._aec_env.reset(seed=seed)
        self._last_action = None
        self._move_count = 0
        self._episode_step = 0
        self._episode_return = 0.0
        self._current_player = "player_0"
        self.log_constant(LOG_ADAPTER_ENV_RESET, extra={"env_id": self.id, "seed": seed if seed is not None else "None"})
        obs, _, _, _, info = self._aec_env.last()
        return self._package_step(obs, 0.0, False, False, dict(info))

    def step(self, action: int) -> AdapterStep[Dict[str, Any]]:
        if self._aec_env is None:
            raise RuntimeError("Environment not loaded. Call load() first.")
        self._last_action = action
        self._move_count += 1
        self._episode_step += 1
        self._aec_env.step(action)
        obs, reward, terminated, truncated, info = self._aec_env.last()
        self._current_player = self._aec_env.agent_selection
        self._episode_return += reward
        self.log_constant(LOG_ADAPTER_STEP_SUMMARY, extra={"env_id": self.id, "episode_step": self._episode_step, "reward": reward})
        return self._package_step(obs, reward, terminated, truncated, dict(info))

    def close(self) -> None:
        if self._aec_env is not None:
            try:
                self._aec_env.close()
            except Exception:
                pass
        self._aec_env = None

    def render(self) -> np.ndarray | None:
        if self._aec_env is None:
            return None
        try:
            return self._aec_env.render()
        except Exception:
            return None

    def _package_step(self, observation: Dict[str, Any], reward: float, terminated: bool, truncated: bool, info: Mapping[str, Any]) -> AdapterStep[Dict[str, Any]]:
        observation.get("observation", observation)
        action_mask = observation.get("action_mask", [1, 1, 1])
        legal_actions = [i for i in range(len(action_mask)) if action_mask[i] == 1]

        render_payload = CardGameRenderPayload(
            game_type="rps",
            current_player=self._current_player,
            legal_actions=legal_actions,
            last_action=self._last_action,
            is_game_over=terminated,
            move_count=self._move_count,
        ).to_dict()

        state = StepState(
            active_agent=self._current_player,
            agents=[AgentSnapshot(name="player_0", role="player"), AgentSnapshot(name="player_1", role="player")],
            metrics={"move_count": self._move_count},
            environment={"game": "rps"},
        )

        return AdapterStep(observation=observation, reward=reward, terminated=terminated, truncated=truncated, info=info,
                          render_payload=render_payload, render_hint={"type": "card_game", "use_qt_widget": True},
                          agent_id=self._current_player, state=state)

    def gym_kwargs(self) -> dict[str, Any]:
        return {}

    def apply_wrappers(self, env: Any) -> Any:
        return env

    def _require_env(self) -> Any:
        if self._aec_env is None:
            raise RuntimeError("Environment not loaded.")
        return self._aec_env

    def _set_env(self, env: Any) -> None:
        self._aec_env = env

    def _resolve_default_render_mode(self) -> RenderMode:
        return self.default_render_mode


class PokerEnvironmentAdapter(EnvironmentAdapter[Dict[str, Any], int]):
    """Adapter for PettingZoo poker games (Leduc Hold'em, Texas Hold'em)."""

    def __init__(self, context: AdapterContext | None = None, game_type: str = "leduc_holdem") -> None:
        super().__init__(context)
        self._aec_env: Any = None
        self._game_type = game_type
        self._last_action: Optional[int] = None
        self._move_count: int = 0
        self._current_player: str = "player_0"

    @property
    def id(self) -> str:
        return f"{self._game_type}_v4"

    supported_control_modes = (ControlMode.HUMAN_ONLY,)
    supported_render_modes = (RenderMode.RGB_ARRAY,)
    default_render_mode = RenderMode.RGB_ARRAY

    def load(self) -> None:
        try:
            if self._game_type == "leduc_holdem":
                from pettingzoo.classic import leduc_holdem_v4
                self._aec_env = leduc_holdem_v4.env(render_mode="rgb_array")
            else:  # texas_holdem
                from pettingzoo.classic import texas_holdem_v4
                self._aec_env = texas_holdem_v4.env(render_mode="rgb_array")
            self.log_constant(LOG_ADAPTER_ENV_CREATED, extra={"env_id": self.id, "render_mode": "rgb_array"})
        except ImportError as e:
            raise ImportError("PettingZoo Classic required: pip install 'pettingzoo[classic]'") from e

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> AdapterStep[Dict[str, Any]]:
        if self._aec_env is None:
            raise RuntimeError("Environment not loaded. Call load() first.")
        self._aec_env.reset(seed=seed)
        self._last_action = None
        self._move_count = 0
        self._episode_step = 0
        self._episode_return = 0.0
        self._current_player = "player_0"
        self.log_constant(LOG_ADAPTER_ENV_RESET, extra={"env_id": self.id, "seed": seed if seed is not None else "None"})
        obs, _, _, _, info = self._aec_env.last()
        return self._package_step(obs, 0.0, False, False, dict(info))

    def step(self, action: int) -> AdapterStep[Dict[str, Any]]:
        if self._aec_env is None:
            raise RuntimeError("Environment not loaded. Call load() first.")
        self._last_action = action
        self._move_count += 1
        self._episode_step += 1
        self._aec_env.step(action)
        obs, reward, terminated, truncated, info = self._aec_env.last()
        self._current_player = self._aec_env.agent_selection
        self._episode_return += reward
        self.log_constant(LOG_ADAPTER_STEP_SUMMARY, extra={"env_id": self.id, "episode_step": self._episode_step, "reward": reward})
        return self._package_step(obs, reward, terminated, truncated, dict(info))

    def close(self) -> None:
        if self._aec_env is not None:
            try:
                self._aec_env.close()
            except Exception:
                pass
        self._aec_env = None

    def render(self) -> np.ndarray | None:
        if self._aec_env is None:
            return None
        try:
            return self._aec_env.render()
        except Exception:
            return None

    def _package_step(self, observation: Dict[str, Any], reward: float, terminated: bool, truncated: bool, info: Mapping[str, Any]) -> AdapterStep[Dict[str, Any]]:
        observation.get("observation", observation)
        action_mask = observation.get("action_mask", [1, 1, 1, 1])
        legal_actions = [i for i in range(len(action_mask)) if action_mask[i] == 1]

        render_payload = CardGameRenderPayload(
            game_type=self._game_type,
            current_player=self._current_player,
            legal_actions=legal_actions,
            last_action=self._last_action,
            is_game_over=terminated,
            move_count=self._move_count,
        ).to_dict()

        state = StepState(
            active_agent=self._current_player,
            agents=[AgentSnapshot(name=f"player_{i}", role="player") for i in range(2)],
            metrics={"move_count": self._move_count},
            environment={"game": self._game_type},
        )

        return AdapterStep(observation=observation, reward=reward, terminated=terminated, truncated=truncated, info=info,
                          render_payload=render_payload, render_hint={"type": "rgb", "use_rgb": True},
                          agent_id=self._current_player, state=state)

    def gym_kwargs(self) -> dict[str, Any]:
        return {}

    def apply_wrappers(self, env: Any) -> Any:
        return env

    def _require_env(self) -> Any:
        if self._aec_env is None:
            raise RuntimeError("Environment not loaded.")
        return self._aec_env

    def _set_env(self, env: Any) -> None:
        self._aec_env = env

    def _resolve_default_render_mode(self) -> RenderMode:
        return self.default_render_mode


class LeducHoldemEnvironmentAdapter(PokerEnvironmentAdapter):
    """Adapter for Leduc Hold'em."""
    id = GameId.LEDUC_HOLDEM.value

    def __init__(self, context: AdapterContext | None = None) -> None:
        super().__init__(context, game_type="leduc_holdem")


class TexasHoldemEnvironmentAdapter(PokerEnvironmentAdapter):
    """Adapter for Texas Hold'em."""
    id = GameId.TEXAS_HOLDEM.value

    def __init__(self, context: AdapterContext | None = None) -> None:
        super().__init__(context, game_type="texas_holdem")


class HanabiEnvironmentAdapter(EnvironmentAdapter[Dict[str, Any], int]):
    """Adapter for PettingZoo Hanabi environment."""

    id = GameId.HANABI.value
    supported_control_modes = (ControlMode.HUMAN_ONLY,)
    supported_render_modes = (RenderMode.RGB_ARRAY,)
    default_render_mode = RenderMode.RGB_ARRAY

    def __init__(self, context: AdapterContext | None = None) -> None:
        super().__init__(context)
        self._aec_env: Any = None
        self._last_action: Optional[int] = None
        self._move_count: int = 0
        self._current_player: str = "player_0"

    def load(self) -> None:
        try:
            from pettingzoo.classic import hanabi_v5
            self._aec_env = hanabi_v5.env(render_mode="rgb_array")
            self.log_constant(LOG_ADAPTER_ENV_CREATED, extra={"env_id": self.id, "render_mode": "rgb_array"})
        except ImportError as e:
            raise ImportError("PettingZoo Classic required: pip install 'pettingzoo[classic]'") from e

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> AdapterStep[Dict[str, Any]]:
        if self._aec_env is None:
            raise RuntimeError("Environment not loaded. Call load() first.")
        self._aec_env.reset(seed=seed)
        self._last_action = None
        self._move_count = 0
        self._episode_step = 0
        self._episode_return = 0.0
        self._current_player = "player_0"
        self.log_constant(LOG_ADAPTER_ENV_RESET, extra={"env_id": self.id, "seed": seed if seed is not None else "None"})
        obs, _, _, _, info = self._aec_env.last()
        return self._package_step(obs, 0.0, False, False, dict(info))

    def step(self, action: int) -> AdapterStep[Dict[str, Any]]:
        if self._aec_env is None:
            raise RuntimeError("Environment not loaded. Call load() first.")
        self._last_action = action
        self._move_count += 1
        self._episode_step += 1
        self._aec_env.step(action)
        obs, reward, terminated, truncated, info = self._aec_env.last()
        self._current_player = self._aec_env.agent_selection
        self._episode_return += reward
        self.log_constant(LOG_ADAPTER_STEP_SUMMARY, extra={"env_id": self.id, "episode_step": self._episode_step, "reward": reward})
        return self._package_step(obs, reward, terminated, truncated, dict(info))

    def close(self) -> None:
        if self._aec_env is not None:
            try:
                self._aec_env.close()
            except Exception:
                pass
        self._aec_env = None

    def render(self) -> np.ndarray | None:
        if self._aec_env is None:
            return None
        try:
            return self._aec_env.render()
        except Exception:
            return None

    def _package_step(self, observation: Dict[str, Any], reward: float, terminated: bool, truncated: bool, info: Mapping[str, Any]) -> AdapterStep[Dict[str, Any]]:
        observation.get("observation", observation)
        action_mask = observation.get("action_mask", [])
        legal_actions = [i for i in range(len(action_mask)) if action_mask[i] == 1]

        render_payload = CardGameRenderPayload(
            game_type="hanabi",
            current_player=self._current_player,
            legal_actions=legal_actions,
            last_action=self._last_action,
            is_game_over=terminated,
            move_count=self._move_count,
        ).to_dict()

        state = StepState(
            active_agent=self._current_player,
            agents=[AgentSnapshot(name=f"player_{i}", role="player") for i in range(2)],
            metrics={"move_count": self._move_count},
            environment={"game": "hanabi"},
        )

        return AdapterStep(observation=observation, reward=reward, terminated=terminated, truncated=truncated, info=info,
                          render_payload=render_payload, render_hint={"type": "rgb", "use_rgb": True},
                          agent_id=self._current_player, state=state)

    def gym_kwargs(self) -> dict[str, Any]:
        return {}

    def apply_wrappers(self, env: Any) -> Any:
        return env

    def _require_env(self) -> Any:
        if self._aec_env is None:
            raise RuntimeError("Environment not loaded.")
        return self._aec_env

    def _set_env(self, env: Any) -> None:
        self._aec_env = env

    def _resolve_default_render_mode(self) -> RenderMode:
        return self.default_render_mode


# Registry of PettingZoo Classic adapters
PETTINGZOO_CLASSIC_ADAPTERS: Dict[GameId, type[EnvironmentAdapter]] = {
    GameId.CHESS: ChessEnvironmentAdapter,
    GameId.CONNECT_FOUR: ConnectFourEnvironmentAdapter,
    GameId.GO: GoEnvironmentAdapter,
    GameId.RPS: RPSEnvironmentAdapter,
    GameId.LEDUC_HOLDEM: LeducHoldemEnvironmentAdapter,
    GameId.TEXAS_HOLDEM: TexasHoldemEnvironmentAdapter,
    GameId.HANABI: HanabiEnvironmentAdapter,
}


__all__ = [
    "ChessEnvironmentAdapter",
    "ChessRenderPayload",
    "ConnectFourEnvironmentAdapter",
    "ConnectFourRenderPayload",
    "GoEnvironmentAdapter",
    "GoRenderPayload",
    "CardGameRenderPayload",
    "RPSEnvironmentAdapter",
    "PokerEnvironmentAdapter",
    "LeducHoldemEnvironmentAdapter",
    "TexasHoldemEnvironmentAdapter",
    "HanabiEnvironmentAdapter",
    "PETTINGZOO_CLASSIC_ADAPTERS",
]
