"""Tests for TianshouWorkerRuntime."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from tianshou_worker.config import TianshouWorkerConfig
from tianshou_worker.runtime import TianshouWorkerRuntime

@pytest.fixture
def mock_popen():
    with patch("subprocess.Popen") as mock:
        process = MagicMock()
        process.stdout.readline.side_effect = ["output line 1", "output line 2", ""]
        process.stderr.read.return_value = ""
        process.poll.return_value = 0
        mock.return_value = process
        yield mock

def test_runtime_execute_basic(mock_popen, tmp_path):
    runtime = TianshouWorkerRuntime(work_dir=tmp_path)
    
    config = TianshouWorkerConfig(
        run_id="test-run",
        algo="ppo",
        env_id="CartPole-v1",
        total_timesteps=1000,
        seed=42
    )
    
    runtime.execute(config)
    
    # Verify Popen called
    assert mock_popen.called
    args, kwargs = mock_popen.call_args
    cmd = args[0]
    
    # Check command structure
    assert cmd[0] == sys.executable
    assert cmd[2] == "tianshou_worker.launcher"
    assert "--run-id" in cmd
    assert "test-run" in cmd
    assert "--algo" in cmd
    assert "ppo" in cmd
    assert "--env-id" in cmd
    assert "CartPole-v1" in cmd
    assert "--seed" in cmd
    assert "42" in cmd
    
    # Check env vars (PYTHONPATH)
    env = kwargs["env"]
    # assert "PYTHONPATH" in env  # We don't set PYTHONPATH explicitly anymore if installed
    # assert str(Path(__file__).resolve().parents[4]) in env["PYTHONPATH"] # repo root

def test_runtime_execute_fastlane(mock_popen, tmp_path):
    runtime = TianshouWorkerRuntime(work_dir=tmp_path)
    
    config = TianshouWorkerConfig(
        run_id="test-fastlane",
        algo="dqn",
        env_id="CartPole-v1",
        total_timesteps=1000,
        extras={"fastlane_enabled": True}
    )
    
    runtime.execute(config)
    
    args, kwargs = mock_popen.call_args
    env = kwargs["env"]
    
    # Check FastLane env vars
    assert "GYM_GUI_FASTLANE_VIDEO_MODE" in env
    assert env["GYM_GUI_FASTLANE_VIDEO_MODE"] == "single"

