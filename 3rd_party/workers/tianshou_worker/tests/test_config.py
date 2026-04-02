"""Tests for TianshouWorkerConfig."""

import json
from tianshou_worker.config import TianshouWorkerConfig

def test_config_from_dict():
    data = {
        "run_id": "test-run",
        "algo": "ppo",
        "env_id": "CartPole-v1",
        "total_timesteps": 1000,
        "seed": 42,
        "extras": {"lr": 0.001}
    }
    config = TianshouWorkerConfig.from_dict(data)
    
    assert config.run_id == "test-run"
    assert config.algo == "ppo"
    assert config.env_id == "CartPole-v1"
    assert config.total_timesteps == 1000
    assert config.seed == 42
    assert config.extras["lr"] == 0.001

def test_config_to_dict():
    config = TianshouWorkerConfig(
        run_id="test-run",
        algo="dqn",
        env_id="CartPole-v1",
        total_timesteps=5000,
        extras={"buffer_size": 10000}
    )
    data = config.to_dict()
    
    assert data["run_id"] == "test-run"
    assert data["algo"] == "dqn"
    assert data["extras"]["buffer_size"] == 10000

def test_config_with_overrides():
    config = TianshouWorkerConfig(
        run_id="base",
        algo="ppo",
        env_id="CartPole-v1",
        total_timesteps=100
    )
    
    new_config = config.with_overrides(
        algo="dqn",
        total_timesteps=200,
        extras={"new_param": 1}
    )
    
    assert new_config.algo == "dqn"
    assert new_config.total_timesteps == 200
    assert new_config.env_id == "CartPole-v1"  # Unchanged
    assert new_config.extras["new_param"] == 1

def test_config_validation():
    try:
        TianshouWorkerConfig(
            run_id="",
            algo="ppo",
            env_id="CartPole-v1",
            total_timesteps=100
        )
        assert False, "Should have raised ValueError for empty run_id"
    except ValueError:
        pass
