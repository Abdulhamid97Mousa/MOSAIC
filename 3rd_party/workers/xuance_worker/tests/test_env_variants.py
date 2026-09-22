# 3rd_party/workers/xuance_worker/tests/test_env_variants.py
"""Tests for all MosaicMultiGrid environment variants in the XuanCe wrapper.

Verifies:
  - _get_env_class resolves every supported v7 short env_id
  - MultiGrid_Env wrapper produces correct obs dims for IndAgObs
  - Gymnasium → XuanCe ID mapping + config resolution works end-to-end
  - reset() and step() produce valid observations and rewards
"""

from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from xuance_worker.environments.multigrid.multigrid import (
    _get_env_class,
    get_available_environments,
    MultiGrid_Env,
)
from xuance_worker.runtime import (
    _gymnasium_to_xuance_env_id,
    _resolve_custom_config_path,
)


# ---------------------------------------------------------------------------
# Expected properties for each environment variant
# ---------------------------------------------------------------------------
ENV_SPECS = {
    # env_id:              (num_agents, obs_dim, is_teamobs, n_actions)
    "soccer_1v1_indagobs": (2, 27, False, 8),
    "soccer_2v2_indagobs": (4, 27, False, 8),
    "soccer_3v3_indagobs": (6, 27, False, 8),
    "collect_1vs1":        (2, 27, False, 8),
    "collect_2vs2_indagobs":   (4, 27, False, 8),
    "basketball_3v3_indagobs": (6, 27, False, 8),
}

# Full gymnasium ID → XuanCe short name mapping
GYMNASIUM_MAPPINGS = {
    "MosaicMultiGrid-S-1v1-IndAgObs-v1": "soccer_1v1_indagobs",
    "MosaicMultiGrid-S-2v2-IndAgObs-v1": "soccer_2v2_indagobs",
    "MosaicMultiGrid-S-3v3-IndAgObs-v1": "soccer_3v3_indagobs",
    "MosaicMultiGrid-BB-3v3-IndAgObs-v1": "basketball_3v3_indagobs",
    "MosaicMultiGrid-BB-G-2v0-IndAgObs-v1": "basketball_g_2v0_indagobs",
    "MosaicMultiGrid-C-1v1-IndAgObs-v1": "collect_1vs1",
    "MosaicMultiGrid-C-2v2-IndAgObs-v1": "collect_2vs2_indagobs",
}


# ---------------------------------------------------------------------------
# Environment class resolution
# ---------------------------------------------------------------------------
class TestEnvClassResolution:
    """Verify _get_env_class finds the correct class for every registered ID."""

    @pytest.mark.parametrize("env_id", list(ENV_SPECS.keys()))
    def test_get_env_class(self, env_id: str):
        cls = _get_env_class(env_id)
        assert cls is not None, f"_get_env_class('{env_id}') returned None"

    def test_available_environments_includes_all(self):
        available = get_available_environments()
        for env_id in ENV_SPECS:
            assert env_id in available, f"'{env_id}' missing from get_available_environments()"

# ---------------------------------------------------------------------------
# Gymnasium → XuanCe ID mapping
# ---------------------------------------------------------------------------
class TestGymnasiumMapping:
    """Verify gymnasium ID → XuanCe short env_id mapping."""

    @pytest.mark.parametrize(
        "gym_id,expected_short",
        list(GYMNASIUM_MAPPINGS.items()),
    )
    def test_mapping(self, gym_id: str, expected_short: str):
        result = _gymnasium_to_xuance_env_id(gym_id)
        assert result == expected_short, f"{gym_id} → {result}, expected {expected_short}"

    def test_unknown_returns_none(self):
        assert _gymnasium_to_xuance_env_id("NonExistent-v99") is None


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------
class TestConfigResolution:
    """Verify YAML config files are found for all registered environments."""

    @pytest.mark.parametrize("env_id", [
        eid for eid in ENV_SPECS
    ])
    @pytest.mark.parametrize("method", ["ippo", "mappo"])
    def test_config_exists(self, env_id: str, method: str):
        import os
        cfg = _resolve_custom_config_path(method, "multigrid", env_id, None, None)
        assert cfg is not None, f"No config for {method}/{env_id}"
        assert os.path.exists(cfg), f"Config file does not exist: {cfg}"

    @pytest.mark.parametrize("gym_id", [
        "MosaicMultiGrid-S-2v2-IndAgObs-v1",
        "MosaicMultiGrid-BB-3v3-IndAgObs-v1",
        "MosaicMultiGrid-C-1v1-IndAgObs-v1",
    ])
    def test_config_via_gymnasium_id(self, gym_id: str):
        """Config resolution works when given a full gymnasium ID (the GUI path)."""
        import os
        cfg = _resolve_custom_config_path("ippo", "multigrid", gym_id, None, None)
        assert cfg is not None, f"No config for gymnasium ID {gym_id}"
        assert os.path.exists(cfg), f"Config file does not exist: {cfg}"


# ---------------------------------------------------------------------------
# Wrapper: reset and step
# ---------------------------------------------------------------------------
class TestMultiGridEnvWrapper:
    """Verify MultiGrid_Env wrapper produces correct obs/action dimensions."""

    @pytest.mark.parametrize("env_id", list(ENV_SPECS.keys()))
    def test_reset_obs_shape(self, env_id: str):
        n_agents, obs_dim, is_teamobs, n_actions = ENV_SPECS[env_id]
        cfg = SimpleNamespace(
            env_name="multigrid", env_id=env_id,
            env_seed=42, training_mode="competitive",
        )
        env = MultiGrid_Env(cfg)
        try:
            obs, info = env.reset()
            assert env.num_agents == n_agents
            assert env._is_teamobs == is_teamobs
            for agent_id in env.agents:
                assert obs[agent_id].shape == (obs_dim,), (
                    f"{env_id}/{agent_id}: obs shape {obs[agent_id].shape} != ({obs_dim},)"
                )
                assert obs[agent_id].dtype == np.float32
        finally:
            env.close()

    @pytest.mark.parametrize("env_id", list(ENV_SPECS.keys()))
    def test_step_obs_shape(self, env_id: str):
        n_agents, obs_dim, is_teamobs, n_actions = ENV_SPECS[env_id]
        cfg = SimpleNamespace(
            env_name="multigrid", env_id=env_id,
            env_seed=42, training_mode="competitive",
        )
        env = MultiGrid_Env(cfg)
        try:
            env.reset()
            actions = {a: env.action_space[a].sample() for a in env.agents}
            obs, rewards, terminated, truncated, info = env.step(actions)
            for agent_id in env.agents:
                assert obs[agent_id].shape == (obs_dim,)
                assert isinstance(rewards[agent_id], float)
        finally:
            env.close()

    @pytest.mark.parametrize("env_id", list(ENV_SPECS.keys()))
    def test_action_space(self, env_id: str):
        _, _, _, n_actions = ENV_SPECS[env_id]
        cfg = SimpleNamespace(
            env_name="multigrid", env_id=env_id,
            env_seed=42, training_mode="competitive",
        )
        env = MultiGrid_Env(cfg)
        try:
            for agent_id in env.agents:
                assert env.action_space[agent_id].n == n_actions
        finally:
            env.close()

    @pytest.mark.parametrize("env_id", list(ENV_SPECS.keys()))
    def test_state_space(self, env_id: str):
        n_agents, obs_dim, _, _ = ENV_SPECS[env_id]
        cfg = SimpleNamespace(
            env_name="multigrid", env_id=env_id,
            env_seed=42, training_mode="competitive",
        )
        env = MultiGrid_Env(cfg)
        try:
            env.reset()
            state = env.state()
            assert state.shape == (n_agents * obs_dim,)
            assert state.dtype == np.float32
        finally:
            env.close()
