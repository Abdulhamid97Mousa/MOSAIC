"""Environment loaders for loading and initializing environments.

These loaders handle environment-specific setup including:
- Widget creation
- Signal connections
- Resource initialization
- Lifecycle management

Available loaders:
- CheckersEnvLoader: Human vs Agent checkers game loading
- ChessEnvLoader: Human vs Agent chess game loading
- ConnectFourEnvLoader: Human vs Agent Connect Four game loading
- GoEnvLoader: Human vs Agent Go game loading
- TicTacToeEnvLoader: Human vs Agent Tic-Tac-Toe game loading
- VizdoomEnvLoader: ViZDoom environment loading with mouse capture
- MalmoEnvLoader: Malmo/Minecraft FPS-style mouse look capture
- JumanjiGridClickLoader: Grid-click mouse input for Jumanji games (Tetris, Minesweeper)
- SmacCameraLoader: Mouse-driven 3D camera panning for SMAC/SMACv2 environments
"""

from gym_gui.ui.handlers.env_loaders.checkers import CheckersEnvLoader
from gym_gui.ui.handlers.env_loaders.chess import ChessEnvLoader
from gym_gui.ui.handlers.env_loaders.connect_four import ConnectFourEnvLoader
from gym_gui.ui.handlers.env_loaders.go import GoEnvLoader
from gym_gui.ui.handlers.env_loaders.jumanji_grid import JumanjiGridClickLoader
from gym_gui.ui.handlers.env_loaders.malmo import MalmoEnvLoader
from gym_gui.ui.handlers.env_loaders.smac_camera import SmacCameraLoader
from gym_gui.ui.handlers.env_loaders.tictactoe import TicTacToeEnvLoader
from gym_gui.ui.handlers.env_loaders.vizdoom import VizdoomEnvLoader

__all__ = [
    "CheckersEnvLoader",
    "ChessEnvLoader",
    "ConnectFourEnvLoader",
    "GoEnvLoader",
    "JumanjiGridClickLoader",
    "MalmoEnvLoader",
    "SmacCameraLoader",
    "TicTacToeEnvLoader",
    "VizdoomEnvLoader",
]
