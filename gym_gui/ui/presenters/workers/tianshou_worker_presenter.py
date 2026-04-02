"""Presenter for Tianshou worker analytics lane."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from ulid import ULID

from gym_gui.ui.widgets.fastlane_tab import FastLaneTab

_LOGGER = logging.getLogger(__name__)


class TianshouWorkerPresenter:
    """Presenter for the Tianshou worker.

    Handles creation of analytics tabs (FastLane) and policy evaluation requests.
    """

    @property
    def id(self) -> str:
        return "tianshou_worker"

    def build_train_request(self, policy_path: Any, current_game: Optional[Any]) -> dict:
        """Construct a training request for policy evaluation.
        
        This allows the "Load Policy" dialog to launch a Tianshou evaluation run.
        """
        # We can implement a basic eval request here
        # Assuming policy_path is a Path object or string
        import pathlib
        path = pathlib.Path(policy_path)
        
        # Infer algo from path if possible, default to ppo
        algo = "ppo"
        if "dqn" in path.name.lower():
            algo = "dqn"
            
        ulid = str(ULID())
        run_id = f"tianshou-eval-{path.stem}-{ulid}"
        
        # Default environment if not known (user can change in form, but here we construct direct request)
        # Actually, build_train_request usually returns a CONFIG dict that is passed to the worker launcher
        # via the main window's run_worker method.
        
        return {
            "run_id": run_id,
            "worker_id": "tianshou_worker",
            "algo": algo,
            "env_id": current_game.value if current_game else "CartPole-v1",
            "total_timesteps": 1,
            "extras": {
                "policy_path": str(path),
                "eval_only": True,
                "eval_episodes": 10,
                "fastlane_enabled": True,
                "video_mode": "single",
            }
        }

    def create_tabs(self, run_id: str, agent_id: str, first_payload: dict, parent: Any) -> List[Any]:
        """Create analytics tabs for a running worker."""
        tabs = []
        
        # Check if FastLane is enabled in the config
        # The payload structure usually has 'config' key
        config = first_payload.get("config", {})
        extras = config.get("extras", {})
        
        if extras.get("fastlane_enabled"):
            try:
                # FastLane tab for live video
                tab = FastLaneTab(
                    run_id=run_id,
                    agent_id=agent_id,
                    parent=parent,
                )
                tabs.append(tab)
            except Exception as e:
                _LOGGER.error(f"Failed to create FastLane tab for {run_id}: {e}")
                
        return tabs


__all__ = ["TianshouWorkerPresenter"]
