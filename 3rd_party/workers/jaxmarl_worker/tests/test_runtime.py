"""Tests for jaxmarl_worker.runtime.InteractiveRuntime protocol.

InteractiveRuntime reads JSON commands from stdin and writes JSON responses
to stdout.  These tests mock the JAX env and actor so no GPU is needed.
They validate:
  - reset emits {"type": "ready", ...}
  - step emits {"type": "step", ...} for non-terminal steps
  - step emits {"type": "episode_done", ...} when done=True
  - step before reset emits {"type": "error", ...}
  - stop emits {"type": "stopped"} and then calls sys.exit
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---- fixtures ---------------------------------------------------------------

def _make_mock_env(num_agents: int = 2, obs_dim: int = 8, action_dim: int = 4,
                   done: bool = False):
    """Return a mock env that satisfies the jaxmarl_worker env interface."""
    env = MagicMock()
    env.num_agents = num_agents
    env.action_dim = action_dim
    env._obs_dim = obs_dim
    env.agents = [f"agent_{i}" for i in range(num_agents)]

    obs = {f"agent_{i}": np.zeros(obs_dim) for i in range(num_agents)}
    state = MagicMock(name="state")

    env.reset.return_value = (obs, state)

    rewards = {f"agent_{i}": 0.5 for i in range(num_agents)}
    done_scalar = bool(done)
    dones = {f"agent_{i}": done_scalar for i in range(num_agents)}
    dones["__all__"] = done_scalar

    env.step_env.return_value = (obs, state, rewards, dones, {})
    return env


def _build_runtime(env, sport="socialjax"):
    """Construct InteractiveRuntime with mocked env and actor -- no GPU."""
    from jaxmarl_worker.runtime import InteractiveRuntime

    mock_params = {"params": {}}

    with (
        patch("jaxmarl_worker.runtime._make_env", return_value=(env, sport)),
        patch("jaxmarl_worker.runtime._load_actor_params", return_value=mock_params),
    ):
        runtime = InteractiveRuntime.__new__(InteractiveRuntime)
        runtime._env_id      = "socialjax/coop_mining"
        runtime._view_size   = 7
        runtime._episode_idx = 0
        runtime._episode_rew = 0.0
        runtime._step_idx    = 0
        runtime._env         = env
        runtime._sport       = sport
        runtime._n_agents    = env.num_agents
        runtime._obs_dim     = env._obs_dim
        runtime._action_dim  = env.action_dim
        runtime._agents      = env.agents
        runtime._obs         = None
        runtime._state       = None

        # Mirror the attributes InteractiveRuntime.__init__ sets:
        # _algo selects the action-sampling path in _select_actions, and
        # _agent_colors=None routes rendering to the rgb fallback path.
        runtime._algo         = "mappo"
        runtime._agent_colors = None
        runtime._max_inner_t  = 0
        runtime._obs_view     = None

        import jax
        runtime._key = jax.random.PRNGKey(0)

        # Stub actor: returns uniform logits
        import jax.numpy as jnp
        def _stub_apply(params, obs):
            return jnp.zeros((obs.shape[0], env.action_dim))
        runtime._actor_params = mock_params
        runtime._apply_fn     = _stub_apply

    return runtime


# ---- reset ------------------------------------------------------------------

class TestHandleReset:

    def test_reset_emits_ready(self) -> None:
        env = _make_mock_env()
        runtime = _build_runtime(env)

        lines = []
        with patch.object(runtime, "_emit", side_effect=lambda obj: lines.append(obj)):
            with patch("jaxmarl_worker.runtime._frame_to_render_payload", return_value=None):
                runtime._handle_reset({"cmd": "reset", "seed": 42})

        assert len(lines) == 1
        resp = lines[0]
        assert resp["type"] == "ready"
        assert resp["seed"] == 42
        assert "observation_shape" in resp
        assert resp["step_index"] == 0

    def test_reset_sets_obs(self) -> None:
        env = _make_mock_env()
        runtime = _build_runtime(env)

        with patch.object(runtime, "_emit"):
            with patch("jaxmarl_worker.runtime._frame_to_render_payload", return_value=None):
                runtime._handle_reset({"seed": 0})

        assert runtime._obs is not None
        assert "agent_0" in runtime._obs

    def test_reset_increments_episode_index(self) -> None:
        env = _make_mock_env()
        runtime = _build_runtime(env)

        with patch.object(runtime, "_emit"):
            with patch("jaxmarl_worker.runtime._frame_to_render_payload", return_value=None):
                runtime._handle_reset({"seed": 0})
                runtime._handle_reset({"seed": 1})

        assert runtime._episode_idx == 2

    def test_reset_uses_seed_zero_as_default(self) -> None:
        env = _make_mock_env()
        runtime = _build_runtime(env)

        lines = []
        with patch.object(runtime, "_emit", side_effect=lambda obj: lines.append(obj)):
            with patch("jaxmarl_worker.runtime._frame_to_render_payload", return_value=None):
                runtime._handle_reset({})  # no "seed" key

        assert lines[0]["seed"] == 0


# ---- step -------------------------------------------------------------------

class TestHandleStep:

    def _reset(self, runtime):
        with patch.object(runtime, "_emit"):
            with patch("jaxmarl_worker.runtime._frame_to_render_payload", return_value=None):
                runtime._handle_reset({"seed": 0})

    def test_step_before_reset_emits_error(self) -> None:
        env = _make_mock_env()
        runtime = _build_runtime(env)

        lines = []
        with patch.object(runtime, "_emit", side_effect=lambda obj: lines.append(obj)):
            runtime._handle_step()

        assert lines[0]["type"] == "error"

    def test_step_emits_step_type(self) -> None:
        env = _make_mock_env(done=False)
        runtime = _build_runtime(env)
        self._reset(runtime)

        lines = []
        with patch.object(runtime, "_emit", side_effect=lambda obj: lines.append(obj)):
            with patch("jaxmarl_worker.runtime._frame_to_render_payload", return_value=None):
                runtime._handle_step()

        assert lines[0]["type"] == "step"
        assert "reward" in lines[0]
        assert "step_index" in lines[0]
        assert lines[0]["terminated"] is False

    def test_step_emits_episode_done_when_done(self) -> None:
        env = _make_mock_env(done=True)
        runtime = _build_runtime(env)
        self._reset(runtime)

        lines = []
        with patch.object(runtime, "_emit", side_effect=lambda obj: lines.append(obj)):
            with patch("jaxmarl_worker.runtime._frame_to_render_payload", return_value=None):
                runtime._handle_step()

        assert lines[0]["type"] == "episode_done"
        assert lines[0]["terminated"] is True
        assert "total_reward" in lines[0]

    def test_step_increments_step_index(self) -> None:
        env = _make_mock_env(done=False)
        runtime = _build_runtime(env)
        self._reset(runtime)

        with patch.object(runtime, "_emit"):
            with patch("jaxmarl_worker.runtime._frame_to_render_payload", return_value=None):
                runtime._handle_step()
                runtime._handle_step()

        assert runtime._step_idx == 2

    def test_step_accumulates_reward(self) -> None:
        env = _make_mock_env(done=False)
        runtime = _build_runtime(env)
        self._reset(runtime)

        with patch.object(runtime, "_emit"):
            with patch("jaxmarl_worker.runtime._frame_to_render_payload", return_value=None):
                runtime._handle_step()
                runtime._handle_step()

        assert runtime._episode_rew > 0.0


# ---- stop -------------------------------------------------------------------

class TestHandleStop:

    def test_stop_emits_stopped_then_exits(self) -> None:
        env = _make_mock_env()
        runtime = _build_runtime(env)

        lines = []
        with patch.object(runtime, "_emit", side_effect=lambda obj: lines.append(obj)):
            with pytest.raises(SystemExit):
                runtime._handle_stop()

        assert lines[0]["type"] == "stopped"


# ---- render failure tolerance -----------------------------------------------

class TestRenderFailureTolerance:

    def test_reset_survives_render_failure(self) -> None:
        """A render exception must not crash the session -- type=ready still emitted."""
        env = _make_mock_env()
        runtime = _build_runtime(env)

        lines = []
        with patch.object(runtime, "_emit", side_effect=lambda obj: lines.append(obj)):
            with patch.object(runtime._env, "_socialjax", create=True) as mock_sj:
                mock_sj.render.side_effect = RuntimeError("render boom")
                runtime._handle_reset({"seed": 0})

        assert lines[0]["type"] == "ready"
        assert lines[0].get("render_payload") is None

    def test_step_survives_render_failure(self) -> None:
        """A render exception on step must not crash -- type=step still emitted."""
        env = _make_mock_env(done=False)
        runtime = _build_runtime(env)

        with patch.object(runtime, "_emit"):
            with patch("jaxmarl_worker.runtime._frame_to_render_payload", return_value=None):
                runtime._handle_reset({"seed": 0})

        lines = []
        with patch.object(runtime, "_emit", side_effect=lambda obj: lines.append(obj)):
            with patch.object(runtime._env, "_socialjax", create=True) as mock_sj:
                mock_sj.render.side_effect = RuntimeError("render boom")
                runtime._handle_step()

        assert lines[0]["type"] == "step"
        assert lines[0].get("render_payload") is None
