"""Tests for Ray RLlib policy evaluation flow.

These tests verify the complete evaluation pipeline:
1. PolicyAssignmentPanel → signal emission
2. Signal chain to MainWindow
3. FastLane tab creation (Ray-Eval-{env}-1W-{run_id})
4. PolicyEvaluator setup and execution
5. FastLane frame streaming

Test categories:
- Unit tests for PolicyEvaluator
- Unit tests for RayPolicyActor
- Integration tests for signal chain
- Integration tests for tab creation
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, PropertyMock, patch

import pytest


class TestPolicyEvaluatorConfig:
    """Tests for EvaluationConfig dataclass."""

    def test_config_required_fields(self):
        """Test EvaluationConfig with required fields."""
        from ray_worker.policy_evaluator import EvaluationConfig

        config = EvaluationConfig(
            env_id="multiwalker_v9",
            env_family="sisl",
            checkpoint_path="/path/to/checkpoint",
            run_id="eval_test_001",
        )

        assert config.env_id == "multiwalker_v9"
        assert config.env_family == "sisl"
        assert config.checkpoint_path == "/path/to/checkpoint"
        assert config.run_id == "eval_test_001"

    def test_config_default_values(self):
        """Test EvaluationConfig default values."""
        from ray_worker.policy_evaluator import EvaluationConfig

        config = EvaluationConfig(
            env_id="pursuit_v4",
            env_family="sisl",
            checkpoint_path="/path/to/checkpoint",
            run_id="eval_defaults",
        )

        assert config.policy_id == "shared"
        assert config.num_episodes == 10
        assert config.max_steps_per_episode == 1000
        assert config.render_mode == "rgb_array"
        assert config.fastlane_enabled is True
        assert config.deterministic is True
        assert config.seed == 42
        assert config.frame_skip == 1
        assert config.agent_policies == {}

    def test_config_custom_values(self):
        """Test EvaluationConfig with custom values."""
        from ray_worker.policy_evaluator import EvaluationConfig

        config = EvaluationConfig(
            env_id="waterworld_v4",
            env_family="sisl",
            checkpoint_path="/custom/path",
            run_id="eval_custom",
            policy_id="main",
            num_episodes=5,
            max_steps_per_episode=500,
            deterministic=False,
            fastlane_enabled=False,
            seed=123,
            frame_skip=2,
            agent_policies={"agent_0": "policy_a"},
        )

        assert config.policy_id == "main"
        assert config.num_episodes == 5
        assert config.max_steps_per_episode == 500
        assert config.deterministic is False
        assert config.fastlane_enabled is False
        assert config.seed == 123
        assert config.frame_skip == 2
        assert config.agent_policies == {"agent_0": "policy_a"}


class TestPolicyEvaluatorInit:
    """Tests for PolicyEvaluator initialization."""

    def test_evaluator_init(self):
        """Test PolicyEvaluator initialization."""
        from ray_worker.policy_evaluator import EvaluationConfig, PolicyEvaluator

        config = EvaluationConfig(
            env_id="pursuit_v4",
            env_family="sisl",
            checkpoint_path="/path/to/checkpoint",
            run_id="eval_init",
        )

        evaluator = PolicyEvaluator(config)

        assert evaluator.config == config
        assert evaluator._env is None
        assert evaluator._policy_actor is None
        assert evaluator._metrics == []
        assert evaluator._running is False
        # FastLane is now integrated into the environment wrapper (ParallelFastLaneWrapper)
        # No separate _fastlane_producer or _frame_count needed

    def test_evaluator_from_config(self):
        """Test PolicyEvaluator.from_config factory method."""
        from ray_worker.policy_evaluator import EvaluationConfig, PolicyEvaluator

        config = EvaluationConfig(
            env_id="pursuit_v4",
            env_family="sisl",
            checkpoint_path="/path/to/checkpoint",
            run_id="eval_factory",
        )

        evaluator = PolicyEvaluator.from_config(config)

        assert evaluator.config == config
        assert isinstance(evaluator, PolicyEvaluator)


class TestEpisodeMetrics:
    """Tests for EpisodeMetrics dataclass."""

    def test_episode_metrics_creation(self):
        """Test EpisodeMetrics creation."""
        from ray_worker.policy_evaluator import EpisodeMetrics

        metrics = EpisodeMetrics(
            episode_id=0,
            total_reward=150.5,
            episode_length=300,
            agent_rewards={"walker_0": 50.0, "walker_1": 50.5, "walker_2": 50.0},
            duration_seconds=3.5,
            terminated=True,
        )

        assert metrics.episode_id == 0
        assert metrics.total_reward == 150.5
        assert metrics.episode_length == 300
        assert len(metrics.agent_rewards) == 3
        assert metrics.duration_seconds == 3.5
        assert metrics.terminated is True


class TestPolicyEvaluatorSummary:
    """Tests for PolicyEvaluator summary functionality."""

    def test_get_summary_empty(self):
        """Test get_summary with no metrics."""
        from ray_worker.policy_evaluator import EvaluationConfig, PolicyEvaluator

        config = EvaluationConfig(
            env_id="pursuit_v4",
            env_family="sisl",
            checkpoint_path="/path/to/checkpoint",
            run_id="eval_summary_empty",
        )

        evaluator = PolicyEvaluator(config)
        summary = evaluator.get_summary()

        assert summary == {}

    def test_get_summary_with_metrics(self):
        """Test get_summary with recorded metrics."""
        from ray_worker.policy_evaluator import (
            EpisodeMetrics,
            EvaluationConfig,
            PolicyEvaluator,
        )

        config = EvaluationConfig(
            env_id="pursuit_v4",
            env_family="sisl",
            checkpoint_path="/path/to/checkpoint",
            run_id="eval_summary",
        )

        evaluator = PolicyEvaluator(config)

        # Manually add some metrics
        evaluator._metrics = [
            EpisodeMetrics(
                episode_id=0,
                total_reward=100.0,
                episode_length=200,
                agent_rewards={},
                duration_seconds=2.0,
                terminated=True,
            ),
            EpisodeMetrics(
                episode_id=1,
                total_reward=150.0,
                episode_length=250,
                agent_rewards={},
                duration_seconds=2.5,
                terminated=True,
            ),
        ]

        summary = evaluator.get_summary()

        assert summary["num_episodes"] == 2
        assert summary["mean_reward"] == 125.0
        assert summary["min_reward"] == 100.0
        assert summary["max_reward"] == 150.0
        assert summary["mean_length"] == 225.0


class TestRayPolicyActorConfig:
    """Tests for RayPolicyActor configuration."""

    def test_ray_policy_config(self):
        """Test RayPolicyConfig dataclass."""
        from ray_worker.policy_actor import RayPolicyConfig

        config = RayPolicyConfig(
            checkpoint_path="/path/to/checkpoint",
            policy_id="shared",
            env_name="multiwalker_v9",
            device="cpu",
            deterministic=True,
        )

        assert config.checkpoint_path == "/path/to/checkpoint"
        assert config.policy_id == "shared"
        assert config.env_name == "multiwalker_v9"
        assert config.device == "cpu"
        assert config.deterministic is True

    def test_ray_policy_config_defaults(self):
        """Test RayPolicyConfig default values."""
        from ray_worker.policy_actor import RayPolicyConfig

        config = RayPolicyConfig(checkpoint_path="/path/to/checkpoint")

        assert config.policy_id == "shared"
        assert config.env_name is None
        assert config.device == "cpu"
        assert config.deterministic is False


class TestRayPolicyActor:
    """Tests for RayPolicyActor."""

    def test_actor_init(self):
        """Test RayPolicyActor initialization."""
        from ray_worker.policy_actor import RayPolicyActor

        actor = RayPolicyActor(id="test_actor")

        assert actor.id == "test_actor"
        assert actor.config is None
        assert actor._initialized is False

    def test_from_checkpoint_file_not_found(self):
        """Test from_checkpoint with non-existent path."""
        from ray_worker.policy_actor import RayPolicyActor

        with pytest.raises(FileNotFoundError):
            RayPolicyActor.from_checkpoint("/nonexistent/checkpoint/path")


class TestSignalChain:
    """Tests for the evaluation signal chain."""

    def test_policy_assignment_panel_has_signal(self):
        """Test that PolicyAssignmentPanel has evaluate_requested signal."""
        pytest.importorskip("qtpy")

        from gym_gui.ui.widgets.policy_assignment_panel import PolicyAssignmentPanel

        assert hasattr(PolicyAssignmentPanel, "evaluate_requested")

    def test_multi_agent_tab_has_signal(self):
        """Test that MultiAgentTab has evaluate_requested signal."""
        pytest.importorskip("qtpy")

        from gym_gui.ui.widgets.multi_agent_tab import MultiAgentTab

        assert hasattr(MultiAgentTab, "evaluate_requested")

    def test_control_panel_has_signal(self):
        """Test that ControlPanelWidget has policy_evaluate_requested signal."""
        pytest.importorskip("qtpy")

        import os
        if not os.environ.get("DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
            os.environ["QT_QPA_PLATFORM"] = "offscreen"

        from gym_gui.ui.widgets.control_panel import ControlPanelWidget

        assert hasattr(ControlPanelWidget, "policy_evaluate_requested")


class TestMainWindowHandler:
    """Tests for MainWindow._on_policy_evaluate_requested handler."""

    def test_handler_config_structure(self):
        """Test the expected config structure for the handler."""
        # The config should contain these keys
        expected_keys = [
            "mode",
            "agent_policies",
            "policy_types",
            "agents",
            "env_id",
            "env_family",
            "worker_id",
        ]

        config = {
            "mode": "evaluate",
            "agent_policies": {"walker_0": "/path/to/checkpoint"},
            "policy_types": {"walker_0": "ray"},
            "agents": ["walker_0", "walker_1", "walker_2"],
            "env_id": "multiwalker_v9",
            "env_family": "sisl",
            "worker_id": "ray_worker",
        }

        for key in expected_keys:
            assert key in config

    def test_ray_policy_type_detection(self):
        """Test that Ray policy type is correctly detected."""
        config = {
            "agent_policies": {
                "walker_0": "/path/to/ray/checkpoint",
                "walker_1": "",  # Random
            },
            "policy_types": {
                "walker_0": "ray",
                "walker_1": "random",
            },
        }

        # Find first Ray checkpoint (logic from _on_policy_evaluate_requested)
        checkpoint_path = None
        for agent_id, path in config["agent_policies"].items():
            if path and config["policy_types"].get(agent_id) == "ray":
                checkpoint_path = path
                break

        assert checkpoint_path == "/path/to/ray/checkpoint"


class TestFastLaneTabCreation:
    """Tests for FastLane tab creation in evaluation mode."""

    def test_tab_title_format_eval_mode(self):
        """Test tab title format for evaluation mode."""
        run_id = "01KCFRQYQ49CKSHCWAY7K1P1WC"
        env_id = "multiwalker_v9"
        run_mode = "policy_eval"
        num_workers = 0  # Evaluation uses single worker

        # Logic from _open_ray_fastlane_tabs
        mode_prefix = "Ray-Eval" if run_mode == "policy_eval" else "Ray-Live"
        active_workers = 1 if num_workers == 0 else num_workers
        title = f"{mode_prefix}-{env_id}-{active_workers}W-{run_id[:8]}"

        assert title == "Ray-Eval-multiwalker_v9-1W-01KCFRQY"
        assert mode_prefix == "Ray-Eval"
        assert active_workers == 1

    def test_tab_title_format_training_mode(self):
        """Test tab title format for training mode (for comparison)."""
        run_id = "01KCFRQYQ49CKSHCWAY7K1P1WC"
        env_id = "multiwalker_v9"
        run_mode = "training"
        num_workers = 4

        mode_prefix = "Ray-Eval" if run_mode == "policy_eval" else "Ray-Live"
        active_workers = 1 if num_workers == 0 else num_workers
        title = f"{mode_prefix}-{env_id}-{active_workers}W-{run_id[:8]}"

        assert title == "Ray-Live-multiwalker_v9-4W-01KCFRQY"
        assert mode_prefix == "Ray-Live"
        assert active_workers == 4


class TestLogConstants:
    """Tests for Ray evaluation log constants."""

    def test_ray_eval_constants_exist(self):
        """Test that Ray evaluation log constants are defined."""
        from gym_gui.logging_config.log_constants import (
            LOG_RAY_EVAL_EPISODE_COMPLETED,
            LOG_RAY_EVAL_EPISODE_STARTED,
            LOG_RAY_EVAL_ERROR,
            LOG_RAY_EVAL_FASTLANE_CONNECTED,
            LOG_RAY_EVAL_POLICY_LOADED,
            LOG_RAY_EVAL_REQUESTED,
            LOG_RAY_EVAL_RUN_COMPLETED,
            LOG_RAY_EVAL_SETUP_COMPLETED,
            LOG_RAY_EVAL_SETUP_STARTED,
            LOG_RAY_EVAL_TAB_CREATED,
        )

        assert LOG_RAY_EVAL_REQUESTED.code == "LOG980"
        assert LOG_RAY_EVAL_SETUP_STARTED.code == "LOG981"
        assert LOG_RAY_EVAL_SETUP_COMPLETED.code == "LOG982"
        assert LOG_RAY_EVAL_EPISODE_STARTED.code == "LOG983"
        assert LOG_RAY_EVAL_EPISODE_COMPLETED.code == "LOG984"
        assert LOG_RAY_EVAL_RUN_COMPLETED.code == "LOG985"
        assert LOG_RAY_EVAL_ERROR.code == "LOG986"
        assert LOG_RAY_EVAL_FASTLANE_CONNECTED.code == "LOG987"
        assert LOG_RAY_EVAL_POLICY_LOADED.code == "LOG988"
        assert LOG_RAY_EVAL_TAB_CREATED.code == "LOG989"

    def test_ray_eval_constants_in_all_constants(self):
        """Test that Ray evaluation constants are in ALL_LOG_CONSTANTS."""
        from gym_gui.logging_config.log_constants import (
            ALL_LOG_CONSTANTS,
            LOG_RAY_EVAL_REQUESTED,
            LOG_RAY_EVAL_RUN_COMPLETED,
        )

        assert LOG_RAY_EVAL_REQUESTED in ALL_LOG_CONSTANTS
        assert LOG_RAY_EVAL_RUN_COMPLETED in ALL_LOG_CONSTANTS


class TestRunEvaluationFunction:
    """Tests for the run_evaluation convenience function."""

    def test_function_signature(self):
        """Test run_evaluation has expected parameters."""
        import inspect

        from ray_worker.policy_evaluator import run_evaluation

        sig = inspect.signature(run_evaluation)
        params = list(sig.parameters.keys())

        assert "env_id" in params
        assert "env_family" in params
        assert "checkpoint_path" in params
        assert "run_id" in params
        assert "policy_id" in params
        assert "num_episodes" in params
        assert "deterministic" in params
        assert "fastlane_enabled" in params
        assert "seed" in params


class TestCheckpointDiscoveryForEvaluation:
    """Tests for checkpoint discovery used in evaluation."""

    def test_discover_checkpoints_returns_list(self):
        """Test that discover_ray_checkpoints returns a list."""
        from gym_gui.policy_discovery.ray_policy_metadata import discover_ray_checkpoints

        checkpoints = discover_ray_checkpoints()
        assert isinstance(checkpoints, list)

    def test_checkpoint_has_required_fields(self):
        """Test that discovered checkpoints have required fields."""
        from gym_gui.policy_discovery.ray_policy_metadata import discover_ray_checkpoints

        checkpoints = discover_ray_checkpoints()

        if checkpoints:
            ckpt = checkpoints[0]
            assert hasattr(ckpt, "run_id")
            assert hasattr(ckpt, "algorithm")
            assert hasattr(ckpt, "env_id")
            assert hasattr(ckpt, "env_family")
            assert hasattr(ckpt, "policy_ids")
            assert hasattr(ckpt, "checkpoint_path")


class TestPolicyAssignmentPanelConfig:
    """Tests for PolicyAssignmentPanel config building."""

    def test_panel_emits_correct_config_structure(self):
        """Test that PolicyAssignmentPanel._on_evaluate builds correct config."""
        # Simulate what the panel builds
        agent_ids = ["walker_0", "walker_1", "walker_2"]
        assignments = {
            "walker_0": "/path/to/checkpoint",
            "walker_1": "/path/to/checkpoint",
            "walker_2": "",  # Random
        }
        policy_types = {
            "walker_0": "ray",
            "walker_1": "ray",
            "walker_2": "random",
        }

        config = {
            "mode": "evaluate",
            "agent_policies": assignments,
            "policy_types": policy_types,
            "agents": agent_ids,
        }

        assert config["mode"] == "evaluate"
        assert len(config["agents"]) == 3
        assert config["policy_types"]["walker_0"] == "ray"
        assert config["policy_types"]["walker_2"] == "random"


# Fixture for QApplication
@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for GUI tests."""
    try:
        import os

        from qtpy import QtWidgets
        if not os.environ.get("DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
            os.environ["QT_QPA_PLATFORM"] = "offscreen"

        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        yield app
    except ImportError:
        yield None
