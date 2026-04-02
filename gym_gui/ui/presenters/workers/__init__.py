"""Worker presenters package for UI orchestration.

This package provides presenter implementations for each supported worker type.
Presenters handle:
1. Building training configurations from form data
2. Creating worker-specific UI tabs
3. Extracting metadata for API contracts

Included presenters:
- ChessWorkerPresenter: LLM-based chess player using llm_chess prompting style
- CleanRlWorkerPresenter: Placeholder for analytics-first CleanRL worker
- HumanWorkerPresenter: Human-in-the-loop action selection via GUI clicks
- MctxWorkerPresenter: GPU-accelerated MCTS (AlphaZero/MuZero) training
- RayWorkerPresenter: Ray RLlib distributed training
- XuanCeWorkerPresenter: XuanCe 46+ algorithm RL library

The registry is auto-populated at module load to support service discovery.
"""

from .chess_worker_presenter import ChessWorkerPresenter
from .cleanrl_worker_presenter import CleanRlWorkerPresenter
from .human_worker_presenter import HumanWorkerPresenter
from .mctx_worker_presenter import MctxWorkerPresenter
from .ray_worker_presenter import RayWorkerPresenter
from .registry import WorkerPresenter, WorkerPresenterRegistry
from .xuance_worker_presenter import XuanCeWorkerPresenter
from .tianshou_worker_presenter import TianshouWorkerPresenter
from .jumanji_worker_presenter import JumanjiWorkerPresenter

# Create and auto-register default presenters
_registry = WorkerPresenterRegistry()

# Discover workers via setuptools entry points
_registry.discover_workers()

# Manual presenter registration (backwards compatibility)
_registry.register("chess_worker", ChessWorkerPresenter())
_registry.register("cleanrl_worker", CleanRlWorkerPresenter())
_registry.register("human_worker", HumanWorkerPresenter())
_registry.register("mctx_worker", MctxWorkerPresenter())
_registry.register("ray_worker", RayWorkerPresenter())
_registry.register("xuance_worker", XuanCeWorkerPresenter())
_registry.register("tianshou_worker", TianshouWorkerPresenter())
_registry.register("jumanji_worker", JumanjiWorkerPresenter())


def get_worker_presenter_registry() -> WorkerPresenterRegistry:
    """Get the global worker presenter registry.

    Returns:
        WorkerPresenterRegistry: Singleton registry of available workers
    """
    return _registry


__all__ = [
    "WorkerPresenter",
    "WorkerPresenterRegistry",
    "ChessWorkerPresenter",
    "CleanRlWorkerPresenter",
    "HumanWorkerPresenter",
    "JumanjiWorkerPresenter",
    "MctxWorkerPresenter",
    "RayWorkerPresenter",
    "XuanCeWorkerPresenter",
    "TianshouWorkerPresenter",
    "get_worker_presenter_registry",
]
