"""Handler classes for MainWindow using composition pattern.

These handlers extract related functionality from MainWindow to improve
maintainability and testability. Each handler is a standalone class that
receives its dependencies via constructor injection.

Subdirectory structure:
- game_moves/: Board game move handlers for Human Control Mode
  - ChessHandler, GoHandler, ConnectFourHandler
- features/: Cross-cutting feature handlers
  - GameConfigHandler, MPCHandler, LogHandler, HumanVsAgentHandler
- env_loaders/: Environment-specific loaders and lifecycle managers
  - ChessEnvLoader, VizdoomEnvLoader

Usage:
    # In MainWindow.__init__:
    self._game_config_handler = GameConfigHandler(
        control_panel=self._control_panel,
        session=self._session,
        status_bar=self._status_bar,
    )

    # For board games, connect signals from BoardGameRendererStrategy:
    self._render_tabs.chess_move_made.connect(self._chess_handler.on_chess_move)
    self._render_tabs.connect_four_column_clicked.connect(
        self._connect_four_handler.on_column_clicked
    )
    self._render_tabs.go_intersection_clicked.connect(
        self._go_handler.on_intersection_clicked
    )

    # For Human vs Agent mode:
    self._human_vs_agent_handler = HumanVsAgentHandler(
        status_bar=self._status_bar,
    )

    # For environment-specific loaders:
    self._chess_env_loader = ChessEnvLoader(
        render_tabs=self._render_tabs,
        control_panel=self._control_panel,
        status_bar=self._status_bar,
    )
"""

from __future__ import annotations

# Environment loaders
from gym_gui.ui.handlers.env_loaders import (
    CheckersEnvLoader,
    ChessEnvLoader,
    ConnectFourEnvLoader,
    GoEnvLoader,
    JumanjiGridClickLoader,
    MalmoEnvLoader,
    SmacCameraLoader,
    TicTacToeEnvLoader,
    VizdoomEnvLoader,
)

# Feature handlers (cross-cutting concerns)
from gym_gui.ui.handlers.features import (
    FastLaneTabHandler,
    GameConfigHandler,
    GodotHandler,
    HumanVsAgentHandler,
    LogHandler,
    MPCHandler,
    MultiAgentGameHandler,
    PolicyEvaluationHandler,
    TrainingFormHandler,
    TrainingMonitorHandler,
)

# Game move handlers (Human Control Mode)
from gym_gui.ui.handlers.game_moves import (
    CheckersHandler,
    ChessHandler,
    ConnectFourHandler,
    GoHandler,
    SudokuHandler,
)

__all__ = [
    # Game move handlers
    "ChessHandler",
    "GoHandler",
    "ConnectFourHandler",
    "SudokuHandler",
    "CheckersHandler",
    # Feature handlers
    "GameConfigHandler",
    "MPCHandler",
    "GodotHandler",
    "LogHandler",
    "HumanVsAgentHandler",
    "PolicyEvaluationHandler",
    "FastLaneTabHandler",
    "TrainingMonitorHandler",
    "TrainingFormHandler",
    "MultiAgentGameHandler",
    # Environment loaders
    "ChessEnvLoader",
    "ConnectFourEnvLoader",
    "GoEnvLoader",
    "TicTacToeEnvLoader",
    "JumanjiGridClickLoader",
    "MalmoEnvLoader",
    "SmacCameraLoader",
    "VizdoomEnvLoader",
    "CheckersEnvLoader",
]
