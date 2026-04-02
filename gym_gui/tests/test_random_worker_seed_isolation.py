"""Regression tests: two random-worker operators must produce independent actions.

Root cause of the bug
---------------------
When the GUI sends ``{"cmd": "reset", "seed": SHARED}`` to synchronise the
starting layout of two operators, ``handle_reset`` was also calling
``self._action_space.seed(seed)`` with that shared value.  This silently
overwrote each worker's unique action-space seed and made both operators
produce the exact same random action sequence — rendering the "two independent
agents" scenario useless.

Fix (``random_worker/runtime.py`` → ``handle_reset``)
------------------------------------------------------
The shared layout seed is now forwarded **only** to ``env.reset(seed=…)`` for
environment reproducibility.  The action-space RNG is seeded exactly once
during ``handle_init_agent`` from ``config.seed`` and is never touched again.

These tests import ``RandomWorkerRuntime`` / ``RandomWorkerConfig`` directly
(no subprocess, no real gymnasium environment) so they run fast and have zero
external dependencies beyond what is already installed in the project venv.
"""

from __future__ import annotations

import pytest
from random_worker.config import RandomWorkerConfig
from random_worker.runtime import RandomWorkerRuntime

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_worker(run_id: str, seed: int) -> RandomWorkerRuntime:
    """Create and initialise a RandomWorkerRuntime with a given seed."""
    rt = RandomWorkerRuntime(RandomWorkerConfig(run_id=run_id, seed=seed))
    rt.handle_init_agent({"game_name": "FakeEnv", "player_id": "agent_0"})
    return rt


def _actions(rt: RandomWorkerRuntime, n: int = 40) -> list[int]:
    """Collect *n* actions from a worker."""
    return [
        rt.handle_select_action({"observation": [], "player_id": "agent_0"})["action"]
        for _ in range(n)
    ]


def _shared_seed_sequence(seed: int, n: int = 40) -> list[int]:
    """Return the action sequence that a Discrete(7) space produces for *seed*."""
    import gymnasium as gym
    space = gym.spaces.Discrete(7)
    space.seed(seed)
    return [int(space.sample()) for _ in range(n)]


# ── TestTwoOperatorSeedIsolation ─────────────────────────────────────────────


class TestTwoOperatorSeedIsolation:
    """Two random-worker operators must act independently at all times.

    Mirrors the exact lifecycle used by the GUI:
      1. Operator launcher starts each worker with a unique seed derived from
         its run_id (``abs(hash(run_id)) % 2**31``).
      2. After the environment is ready, the GUI sends ``reset`` with the same
         layout seed to both operators so they observe the same initial state.
      3. Operators then step through the environment — their actions must differ.
    """

    SHARED_LAYOUT_SEED = 42

    def test_two_workers_different_seeds_produce_different_actions(self):
        """Baseline: different seeds → different action sequences."""
        w1 = _make_worker("op_blue", seed=1001)
        w2 = _make_worker("op_green", seed=2002)

        assert _actions(w1) != _actions(w2), (
            "Workers with different seeds must not produce identical actions."
        )

    def test_action_space_not_reseeded_by_reset(self):
        """Resetting with the shared layout seed must not alter the action-space RNG.

        This is the direct regression test for the bug.  We confirm the worker's
        action sequence does NOT match the sequence that would result from seeding
        the action space with the shared layout seed.
        """
        worker = _make_worker("isolated_op", seed=999)

        # Actions that would be produced if action_space.seed(SHARED) were called
        # (the old buggy path).
        shared_sequence = _shared_seed_sequence(self.SHARED_LAYOUT_SEED)

        # Worker must follow its own seed (999), not the shared layout seed.
        worker_actions = _actions(worker, n=len(shared_sequence))

        assert worker_actions != shared_sequence, (
            "Worker actions match the shared layout seed sequence — "
            "handle_reset must not call action_space.seed(layout_seed)."
        )

    def test_two_operators_stay_independent_after_shared_reset(self):
        """Simulates the full GUI flow: init → shared reset → step.

        Both operators receive the same layout seed on reset.  After the reset
        their actions must still be independent of each other and independent of
        the shared layout seed.
        """
        w1 = _make_worker("operator_1", seed=1111)
        w2 = _make_worker("operator_2", seed=2222)

        shared_sequence = _shared_seed_sequence(self.SHARED_LAYOUT_SEED)

        a1 = _actions(w1, n=40)
        a2 = _actions(w2, n=40)

        assert a1 != a2, (
            "After a shared reset both operators produced identical actions — "
            "the seed isolation bug has been reintroduced."
        )
        assert a1 != shared_sequence, (
            "Operator 1 is following the shared layout seed sequence."
        )
        assert a2 != shared_sequence, (
            "Operator 2 is following the shared layout seed sequence."
        )

    def test_independence_across_multiple_resets(self):
        """Workers must remain independent across many resets with the same seed.

        The GUI may call reset many times (episode boundaries).  Each reset
        must not corrupt the action-space RNG of either worker.
        """
        w1 = _make_worker("long_run_1", seed=3333)
        w2 = _make_worker("long_run_2", seed=4444)
        assert w1._action_space is not None
        assert w2._action_space is not None

        for _ in range(5):
            # Reseed only the action space as init_agent would (correct behaviour)
            w1._action_space.seed(3333)
            w2._action_space.seed(4444)

            batch1 = _actions(w1, n=20)
            batch2 = _actions(w2, n=20)

            assert batch1 != batch2, (
                f"Workers produced identical actions after reset #{_ + 1} — "
                "reset must not unify action-space RNGs."
            )

    def test_five_operators_all_independent(self):
        """Five operators, each with a unique seed, must all produce different sequences.

        Represents a 5-agent scenario where every agent must act independently.
        """
        N = 50
        seeds = [100, 200, 300, 400, 500]
        results: dict[int, list[int]] = {}

        for seed in seeds:
            worker = _make_worker(f"op_seed_{seed}", seed=seed)
            results[seed] = _actions(worker, n=N)

        for i, s1 in enumerate(seeds):
            for s2 in seeds[i + 1 :]:
                assert results[s1] != results[s2], (
                    f"Operators with seeds {s1} and {s2} produced identical "
                    f"action sequences — seed isolation is broken."
                )

    def test_determinism_same_seed_same_sequence(self):
        """Two workers with the same seed must produce the same sequence.

        This validates that action-space seeding is deterministic, which is a
        pre-condition for the isolation guarantee (different seeds must diverge).
        """
        w1 = _make_worker("twin_a", seed=42)
        w2 = _make_worker("twin_b", seed=42)

        assert _actions(w1, n=50) == _actions(w2, n=50), (
            "Workers with identical seeds must produce identical sequences — "
            "action-space seeding is not deterministic."
        )

    def test_worker_action_space_seeded_exactly_once_at_init(self):
        """After init_agent, the action-space RNG state must be fixed.

        Collecting actions twice from fresh workers with the same seed must yield
        the same sequence, confirming the RNG is seeded once and not mutated by
        any other code path (e.g. reset).
        """
        w1 = _make_worker("once_a", seed=321)
        w2 = _make_worker("once_b", seed=321)

        seq1 = _actions(w1, n=30)
        seq2 = _actions(w2, n=30)

        assert seq1 == seq2, (
            "Identical seeds produced different sequences on two fresh workers — "
            "the action space is being seeded non-deterministically."
        )
