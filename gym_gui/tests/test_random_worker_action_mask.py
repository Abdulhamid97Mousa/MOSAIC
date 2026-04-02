"""Tests for random worker action_mask handling and logging.

Covers the fixes:
1. Random worker samples exclusively from legal actions when action_mask is
   provided, regardless of the fallback action_space size.
2. ``_emit`` no longer produces a duplicate "Action selected" log line —
   only ``handle_select_action`` logs it (with the step counter).
3. When action_mask is all-zeros (no legal actions), the worker falls back to
   a full action_space sample rather than crashing.
"""

from __future__ import annotations

import logging
from typing import List
from unittest.mock import patch

import numpy as np
import pytest
from random_worker.config import RandomWorkerConfig
from random_worker.runtime import RandomWorkerRuntime

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_worker(seed: int = 42, task: str = "chess_v6") -> RandomWorkerRuntime:
    """Return an initialised RandomWorkerRuntime in interactive (action-selector) mode."""
    config = RandomWorkerConfig(run_id=f"test_{seed}", seed=seed, task=task)
    rt = RandomWorkerRuntime(config)
    rt.handle_init_agent({"game_name": task, "player_id": "player_0"})
    return rt


def _chess_action_mask(legal_indices: List[int], size: int = 4672) -> List[int]:
    """Build a chess-style action_mask with 1s at the given indices."""
    mask = [0] * size
    for i in legal_indices:
        mask[i] = 1
    return mask


# ---------------------------------------------------------------------------
# TestRandomWorkerActionMask
# ---------------------------------------------------------------------------

class TestRandomWorkerActionMask:
    """Random worker must sample from legal actions defined by action_mask."""

    def test_action_within_legal_set(self):
        """Selected action must be one of the legal indices in the mask."""
        rt = _make_worker()
        legal = [100, 200, 300, 400, 500]
        mask = _chess_action_mask(legal)

        result = rt.handle_select_action({"player_id": "player_0", "action_mask": mask})

        assert result["type"] == "action_selected"
        assert result["action"] in legal, (
            f"action {result['action']} is not in legal set {legal}"
        )

    def test_all_legal_actions_reachable(self):
        """Over many samples every legal action must appear at least once."""
        rt = _make_worker(seed=0)
        legal = [10, 20, 30]
        mask = _chess_action_mask(legal)

        seen = set()
        for _ in range(300):
            res = rt.handle_select_action({"player_id": "player_0", "action_mask": mask})
            seen.add(res["action"])

        assert seen == set(legal), (
            f"Expected all of {legal} to be sampled; got {seen}"
        )

    def test_action_mask_overrides_fallback_action_space(self):
        """Even when action_space is tiny (Discrete(7)), action_mask takes precedence.

        This is the chess_v6 scenario: the worker can't create the env, so it
        defaults to Discrete(7), but must still sample from the 4672-entry mask.
        """
        rt = _make_worker(task="chess_v6")
        # chess_v6 falls back to Discrete(7) — but legal action is index 3563
        legal = [3563, 3564, 3565]
        mask = _chess_action_mask(legal)

        for _ in range(20):
            res = rt.handle_select_action({"player_id": "player_0", "action_mask": mask})
            assert res["action"] in legal, (
                f"action {res['action']} is outside legal set {legal}; "
                "action_mask must override Discrete(7) fallback"
            )

    def test_no_action_mask_falls_back_to_action_space(self):
        """Without action_mask the worker samples from its action_space (Discrete(7))."""
        rt = _make_worker(task="chess_v6")
        actions = set()
        for _ in range(100):
            res = rt.handle_select_action({"player_id": "player_0"})
            actions.add(res["action"])

        # All actions must be within Discrete(7)
        assert all(0 <= a < 7 for a in actions), (
            f"Expected only 0-6 without action_mask; got {actions}"
        )

    def test_empty_action_mask_falls_back_to_action_space(self):
        """All-zeros action_mask (no legal moves) falls back to action_space.sample()."""
        rt = _make_worker(task="chess_v6")
        all_zero_mask = [0] * 4672
        res = rt.handle_select_action({"player_id": "player_0", "action_mask": all_zero_mask})
        assert res["type"] == "action_selected"
        assert 0 <= res["action"] < 7, (
            "Empty mask should fall back to Discrete(7) sample"
        )

    def test_step_count_increments_per_call(self):
        """handle_select_action increments the internal step counter each call."""
        rt = _make_worker()
        mask = _chess_action_mask([0, 1, 2])

        for expected in range(1, 6):
            rt.handle_select_action({"player_id": "player_0", "action_mask": mask})
            assert rt._step_count == expected

    def test_action_selected_response_contains_required_fields(self):
        """Response must include type, action, action_str, player_id, run_id."""
        rt = _make_worker()
        mask = _chess_action_mask([42])
        res = rt.handle_select_action({"player_id": "player_0", "action_mask": mask})

        assert res["type"] == "action_selected"
        assert res["action"] == 42
        assert res["action_str"] == "42"
        assert res["player_id"] == "player_0"
        assert "run_id" in res


# ---------------------------------------------------------------------------
# TestRandomWorkerEmitNoDuplicateLog
# ---------------------------------------------------------------------------

class TestRandomWorkerEmitNoDuplicateLog:
    """_emit must not log a duplicate 'Action selected' line.

    Before the fix, _emit contained an extra logger.info("Action selected …")
    call that produced a second log entry (without step count) after
    handle_select_action had already logged once (with step count).
    """

    def test_emit_action_selected_produces_no_log(self, caplog):
        """_emit for 'action_selected' type must produce zero log records."""
        rt = _make_worker()

        with caplog.at_level(logging.DEBUG, logger="random_worker.runtime"):
            # Directly call _emit with an action_selected payload — bypass the
            # handle_select_action logger to isolate the _emit logger.
            with patch("builtins.print"):  # suppress stdout JSON
                rt._emit({
                    "type": "action_selected",
                    "player_id": "player_0",
                    "action": 99,
                    "action_str": "99",
                    "run_id": "test",
                })

        # _emit must NOT have added any log record for action_selected
        action_logs = [r for r in caplog.records if "Action selected" in r.message]
        assert action_logs == [], (
            f"_emit produced duplicate log records: {[r.message for r in action_logs]}"
        )

    def test_handle_select_action_logs_exactly_once(self, caplog):
        """handle_select_action logs 'Action selected' exactly once per call."""
        rt = _make_worker()
        mask = _chess_action_mask([10, 11])

        with caplog.at_level(logging.INFO, logger="random_worker.runtime"):
            with patch("builtins.print"):
                rt.handle_select_action({"player_id": "player_0", "action_mask": mask})

        action_logs = [r for r in caplog.records if "Action selected" in r.message]
        assert len(action_logs) == 1, (
            f"Expected exactly 1 'Action selected' log; got {len(action_logs)}: "
            f"{[r.message for r in action_logs]}"
        )

    def test_logged_message_contains_step_count(self, caplog):
        """The single 'Action selected' log must include the step counter."""
        rt = _make_worker()
        mask = _chess_action_mask([5])

        with caplog.at_level(logging.INFO, logger="random_worker.runtime"):
            with patch("builtins.print"):
                rt.handle_select_action({"player_id": "player_0", "action_mask": mask})

        action_logs = [r for r in caplog.records if "Action selected" in r.message]
        assert len(action_logs) == 1
        assert "(step 1)" in action_logs[0].message, (
            f"Step count missing from log: '{action_logs[0].message}'"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
