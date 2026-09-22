"""Tests for jaxmarl_worker._make_env routing logic.

These tests cover the env_id -> (env, sport) dispatch table without
requiring a GPU or actual JAX JIT compilation.  Each SocialJax env name
(bare and prefixed) must produce sport="socialjax"; unknown IDs must raise.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---- helpers ----------------------------------------------------------------

def _make_env(env_id: str, view_size: int = 7):
    """Import inside each test so we can patch cleanly."""
    from jaxmarl_worker.runtime import _make_env as _fn
    return _fn(env_id, view_size)


_SJ_ENVS = [
    "coin_game",
    "harvest_common_open",
    "coop_mining",
    "territory_open",
    "pd_arena",
    "mushrooms",
    "gift",
    "lb_foraging",
]


# ---- SocialJax routing ------------------------------------------------------

class TestMakeEnvSocialJax:

    def _mock_wrapper(self, env_id: str):
        """Return a mock SocialJaxGenericWrapper with the required attributes."""
        m = MagicMock()
        m.num_agents = 2
        m.action_dim = 8
        m._obs_dim = 10
        m.agents = ["agent_0", "agent_1"]
        return m

    @pytest.mark.parametrize("task", _SJ_ENVS)
    def test_bare_task_returns_socialjax_sport(self, task: str) -> None:
        """Bare task name (no prefix) must route to sport='socialjax'."""
        mock_wrapper = self._mock_wrapper(task)
        with patch(
            "jaxmarl_worker.environments.socialjax.generic.SocialJaxGenericWrapper",
            return_value=mock_wrapper,
        ):
            env, sport = _make_env(task)
        assert sport == "socialjax"
        assert env is mock_wrapper

    @pytest.mark.parametrize("task", _SJ_ENVS)
    def test_prefixed_task_returns_socialjax_sport(self, task: str) -> None:
        """'socialjax/<task>' prefix must also route to sport='socialjax'."""
        env_id = f"socialjax/{task}"
        mock_wrapper = self._mock_wrapper(task)
        with patch(
            "jaxmarl_worker.environments.socialjax.generic.SocialJaxGenericWrapper",
            return_value=mock_wrapper,
        ):
            env, sport = _make_env(env_id)
        assert sport == "socialjax"

    def test_coop_mining_sport_is_socialjax(self) -> None:
        """The task used in GUI integration (coop_mining) must give sport='socialjax'."""
        mock_wrapper = MagicMock()
        mock_wrapper.num_agents = 6
        mock_wrapper.action_dim = 8
        mock_wrapper._obs_dim = 1452
        mock_wrapper.agents = [f"agent_{i}" for i in range(6)]
        with patch(
            "jaxmarl_worker.environments.socialjax.generic.SocialJaxGenericWrapper",
            return_value=mock_wrapper,
        ):
            _, sport = _make_env("socialjax/coop_mining")
        assert sport == "socialjax"

    def test_unknown_env_raises_value_error(self) -> None:
        """An env_id that matches no known family must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown env_id"):
            _make_env("not_a_real_env_xyz")

    def test_unknown_prefix_raises_value_error(self) -> None:
        """A correctly-prefixed but non-SJ name must still raise ValueError."""
        with pytest.raises(ValueError, match="Unknown env_id"):
            _make_env("socialjax/not_a_real_task")


class TestMakeEnvMosaicV7:
    """v7 sports factories require their canonical goal-row geometry."""

    @pytest.mark.parametrize(
        "env_id,factory_path,expected_sport",
        [
            (
                "MosaicMultiGrid-S-G-2v0-IndAgObs-v1",
                "jaxmarl_worker.environments.soccer_jax.make_soccer_jax",
                "soccer",
            ),
            (
                "MosaicMultiGrid-BB-G-2v0-IndAgObs-v1",
                "jaxmarl_worker.environments.basketball_jax.make_bb_jax",
                "bb",
            ),
            (
                "MosaicMultiGrid-AF-G-2v0-IndAgObs-v1",
                "jaxmarl_worker.environments.american_football_jax.make_af_jax",
                "af",
            ),
        ],
    )
    def test_factory_receives_goal_rows(
        self, env_id: str, factory_path: str, expected_sport: str
    ) -> None:
        mock_env = MagicMock()
        with patch(factory_path, return_value=mock_env) as factory:
            env, sport = _make_env(env_id)

        assert env is mock_env
        assert sport == expected_sport
        assert factory.call_args.kwargs["view_size"] == 7
        assert factory.call_args.kwargs["goal_rows"]
