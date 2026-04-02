"""Presenter for Jumanji worker analytics lane."""

from __future__ import annotations

import logging
from typing import Any, Optional

from ulid import ULID

_LOGGER = logging.getLogger(__name__)


class JumanjiWorkerPresenter:
    """Presenter for the Jumanji (JAX) worker.

    Handles creation of analytics tabs (FastLane) and policy evaluation requests.
    """

    @property
    def id(self) -> str:
        return "jumanji_worker"

    def build_train_request(self, policy_path: Any, current_game: Optional[Any]) -> dict:
        """Construct a training request for policy evaluation."""
        import pathlib

        path = pathlib.Path(policy_path)
        ulid = str(ULID())
        run_id = f"jumanji-eval-{path.stem}-{ulid}"

        return {
            "run_id": run_id,
            "worker_id": "jumanji_worker",
            "agent": "a2c",
            "env_id": current_game.value if current_game else "Game2048-v1",
            "extras": {
                "policy_path": str(path),
                "eval_only": True,
                "fastlane_enabled": True,
                "video_mode": "single",
            },
        }
