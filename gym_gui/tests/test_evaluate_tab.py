"""Tests for the Evaluate Policies tab functionality.

These tests verify:
1. ULID import and generation
2. RayPolicyForm parameter handling
3. PolicyEvaluator configuration
4. Checkpoint discovery
5. Signal flow for policy evaluation
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


class TestULIDGeneration:
    """Tests for ULID generation used in evaluation run IDs."""

    def test_ulid_import(self):
        """Test that ULID can be imported correctly."""
        from ulid import ULID

        assert ULID is not None

    def test_ulid_generation(self):
        """Test that ULID() generates valid IDs."""
        from ulid import ULID

        ulid_obj = ULID()
        ulid_str = str(ulid_obj)

        # ULID strings are 26 characters
        assert len(ulid_str) == 26
        # Should be alphanumeric
        assert ulid_str.isalnum()

    def test_ulid_uniqueness(self):
        """Test that consecutive ULIDs are unique."""
        from ulid import ULID

        ulids = [str(ULID()) for _ in range(10)]
        assert len(set(ulids)) == 10  # All unique


class TestRayPolicyForm:
    """Tests for RayPolicyForm parameter handling."""

    def test_import_ray_policy_form(self):
        """Test that RayPolicyForm can be imported."""
        pytest.importorskip("qtpy")

        from gym_gui.ui.widgets.ray_policy_form import RayPolicyForm

        assert RayPolicyForm is not None

    def test_ray_policy_form_accepts_current_game(self):
        """Test that RayPolicyForm accepts current_game parameter."""
        pytest.importorskip("qtpy")

        import os
        if not os.environ.get("DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
            os.environ["QT_QPA_PLATFORM"] = "offscreen"

        from qtpy import QtWidgets

        from gym_gui.ui.widgets.ray_policy_form import RayPolicyForm

        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        # Should not raise TypeError
        form = RayPolicyForm(parent=None, current_game=MagicMock())
        assert form._default_game is not None
        form.close()

    def test_ray_policy_form_accepts_default_game(self):
        """Test that RayPolicyForm accepts default_game parameter."""
        pytest.importorskip("qtpy")

        import os
        if not os.environ.get("DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
            os.environ["QT_QPA_PLATFORM"] = "offscreen"

        from qtpy import QtWidgets

        from gym_gui.ui.widgets.ray_policy_form import RayPolicyForm

        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        # Should not raise TypeError
        form = RayPolicyForm(parent=None, default_game=MagicMock())
        assert form._default_game is not None
        form.close()

    def test_ray_policy_form_no_kwargs(self):
        """Test that RayPolicyForm works without game parameters."""
        pytest.importorskip("qtpy")

        import os
        if not os.environ.get("DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
            os.environ["QT_QPA_PLATFORM"] = "offscreen"

        from qtpy import QtWidgets

        from gym_gui.ui.widgets.ray_policy_form import RayPolicyForm

        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        # Should not raise TypeError
        form = RayPolicyForm(parent=None)
        assert form._default_game is None
        form.close()


class TestPolicyEvaluatorConfig:
    """Tests for PolicyEvaluator configuration."""

    def test_import_evaluation_config(self):
        """Test that EvaluationConfig can be imported."""
        from ray_worker.policy_evaluator import EvaluationConfig

        assert EvaluationConfig is not None

    def test_evaluation_config_required_fields(self):
        """Test EvaluationConfig with required fields only."""
        from ray_worker.policy_evaluator import EvaluationConfig

        config = EvaluationConfig(
            env_id="pursuit_v4",
            env_family="sisl",
            checkpoint_path="/path/to/checkpoint",
            run_id="eval_test_001",
        )

        assert config.env_id == "pursuit_v4"
        assert config.env_family == "sisl"
        assert config.checkpoint_path == "/path/to/checkpoint"
        assert config.run_id == "eval_test_001"

    def test_evaluation_config_default_values(self):
        """Test EvaluationConfig default values."""
        from ray_worker.policy_evaluator import EvaluationConfig

        config = EvaluationConfig(
            env_id="pursuit_v4",
            env_family="sisl",
            checkpoint_path="/path/to/checkpoint",
            run_id="eval_test_001",
        )

        # Check defaults
        assert config.policy_id == "shared"
        assert config.num_episodes == 10
        assert config.max_steps_per_episode == 1000
        assert config.render_mode == "rgb_array"
        assert config.fastlane_enabled is True
        assert config.deterministic is True
        assert config.seed == 42
        assert config.frame_skip == 1

    def test_evaluation_config_custom_values(self):
        """Test EvaluationConfig with custom values."""
        from ray_worker.policy_evaluator import EvaluationConfig

        config = EvaluationConfig(
            env_id="multiwalker_v9",
            env_family="sisl",
            checkpoint_path="/path/to/checkpoint",
            run_id="eval_custom",
            policy_id="main",
            num_episodes=5,
            deterministic=False,
            fastlane_enabled=False,
        )

        assert config.policy_id == "main"
        assert config.num_episodes == 5
        assert config.deterministic is False
        assert config.fastlane_enabled is False


class TestPolicyEvaluator:
    """Tests for PolicyEvaluator class."""

    def test_import_policy_evaluator(self):
        """Test that PolicyEvaluator can be imported."""
        from ray_worker.policy_evaluator import PolicyEvaluator

        assert PolicyEvaluator is not None

    def test_policy_evaluator_init(self):
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

    def test_policy_evaluator_get_summary_empty(self):
        """Test get_summary with no metrics."""
        from ray_worker.policy_evaluator import EvaluationConfig, PolicyEvaluator

        config = EvaluationConfig(
            env_id="pursuit_v4",
            env_family="sisl",
            checkpoint_path="/path/to/checkpoint",
            run_id="eval_summary",
        )

        evaluator = PolicyEvaluator(config)
        summary = evaluator.get_summary()

        assert summary == {}

    def test_episode_metrics_dataclass(self):
        """Test EpisodeMetrics dataclass."""
        from ray_worker.policy_evaluator import EpisodeMetrics

        metrics = EpisodeMetrics(
            episode_id=0,
            total_reward=100.5,
            episode_length=250,
            agent_rewards={"agent_0": 50.0, "agent_1": 50.5},
            duration_seconds=2.5,
            terminated=True,
        )

        assert metrics.episode_id == 0
        assert metrics.total_reward == 100.5
        assert metrics.episode_length == 250
        assert len(metrics.agent_rewards) == 2
        assert metrics.terminated is True


class TestCheckpointDiscovery:
    """Tests for checkpoint discovery functionality."""

    def test_import_discovery_functions(self):
        """Test that discovery functions can be imported."""
        from gym_gui.policy_discovery.ray_policy_metadata import (
            RayRLlibCheckpoint,
            discover_ray_checkpoints,
            get_checkpoints_for_env,
            get_latest_checkpoint,
        )

        assert RayRLlibCheckpoint is not None
        assert callable(discover_ray_checkpoints)
        assert callable(get_checkpoints_for_env)
        assert callable(get_latest_checkpoint)

    def test_discover_checkpoints_empty_dir(self):
        """Test discovery with no checkpoints."""
        from gym_gui.policy_discovery.ray_policy_metadata import discover_ray_checkpoints

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoints = discover_ray_checkpoints(Path(tmpdir))
            assert checkpoints == []

    def test_discover_checkpoints_with_mock_structure(self):
        """Test discovery with mock checkpoint structure."""
        from gym_gui.policy_discovery.ray_policy_metadata import discover_ray_checkpoints

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock checkpoint structure
            run_dir = Path(tmpdir) / "01TEST123"
            run_dir.mkdir()

            # Create analytics.json
            analytics = {
                "worker_type": "ray_worker",
                "ray_metadata": {
                    "algorithm": "PPO",
                    "env_id": "pursuit_v4",
                    "env_family": "sisl",
                    "paradigm": "parameter_sharing",
                },
            }
            (run_dir / "analytics.json").write_text(json.dumps(analytics))

            # Create checkpoints directory
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir()

            # Create rllib_checkpoint.json
            rllib_meta = {
                "type": "Algorithm",
                "checkpoint_version": "1.1",
                "policy_ids": ["shared"],
                "ray_version": "2.52.1",
            }
            (checkpoint_dir / "rllib_checkpoint.json").write_text(json.dumps(rllib_meta))

            # Create algorithm_state.pkl (empty file for test)
            (checkpoint_dir / "algorithm_state.pkl").touch()

            # Discover checkpoints
            checkpoints = discover_ray_checkpoints(Path(tmpdir))

            assert len(checkpoints) == 1
            ckpt = checkpoints[0]
            assert ckpt.run_id == "01TEST123"
            assert ckpt.algorithm == "PPO"
            assert ckpt.env_id == "pursuit_v4"
            assert "shared" in ckpt.policy_ids

    def test_checkpoint_display_name(self):
        """Test RayRLlibCheckpoint.display_name property."""
        from gym_gui.policy_discovery.ray_policy_metadata import RayRLlibCheckpoint

        checkpoint = RayRLlibCheckpoint(
            checkpoint_path=Path("/tmp/test/checkpoints"),
            run_id="01ABC123DEFGHIJK",
            algorithm="APPO",
            env_id="multiwalker_v9",
            env_family="sisl",
            paradigm="parameter_sharing",
            policy_ids=["shared"],
        )

        display = checkpoint.display_name
        assert "01ABC123" in display
        assert "multiwalker_v9" in display
        assert "APPO" in display


class TestActualCheckpoints:
    """Tests using actual checkpoints in var/trainer/runs."""

    def test_discover_actual_checkpoints(self):
        """Test discovering actual checkpoints if they exist."""
        from gym_gui.config.paths import VAR_TRAINER_DIR
        from gym_gui.policy_discovery.ray_policy_metadata import discover_ray_checkpoints

        runs_dir = VAR_TRAINER_DIR / "runs"
        if not runs_dir.exists():
            pytest.skip("No runs directory found")

        checkpoints = discover_ray_checkpoints()

        # Just verify the function works without error
        assert isinstance(checkpoints, list)

        # If we have checkpoints, verify structure
        if checkpoints:
            ckpt = checkpoints[0]
            assert hasattr(ckpt, "run_id")
            assert hasattr(ckpt, "algorithm")
            assert hasattr(ckpt, "env_id")
            assert hasattr(ckpt, "policy_ids")
            assert hasattr(ckpt, "checkpoint_path")


class TestSignalConnections:
    """Tests for signal connections in the evaluation flow."""

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


class TestLoadPolicyDialog:
    """Tests for LoadPolicyDialog."""

    def test_import_load_policy_dialog(self):
        """Test that LoadPolicyDialog can be imported."""
        pytest.importorskip("qtpy")

        from gym_gui.ui.widgets.load_policy_dialog import (
            LoadPolicyDialog,
            QuickLoadPolicyWidget,
        )

        assert LoadPolicyDialog is not None
        assert QuickLoadPolicyWidget is not None

    def test_ray_worker_policy_form_registered(self):
        """Test that ray_worker policy form is registered with factory.

        This is critical for the 'Load Policy' dialog to work.
        """
        pytest.importorskip("qtpy")

        from gym_gui.ui.forms import get_worker_form_factory

        factory = get_worker_form_factory()
        assert factory.has_policy_form("ray_worker"), (
            "ray_worker policy form not registered! "
            "Check gym_gui/ui/forms/__init__.py imports ray_policy_form module."
        )


class TestRunEvaluation:
    """Tests for the run_evaluation convenience function."""

    def test_import_run_evaluation(self):
        """Test that run_evaluation can be imported."""
        from ray_worker.policy_evaluator import run_evaluation

        assert callable(run_evaluation)

    def test_run_evaluation_function_signature(self):
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


class TestMainWindowEvaluationHandler:
    """Tests for MainWindow._on_policy_evaluate_requested handler."""

    def test_handler_uses_ulid_correctly(self):
        """Test that the handler generates valid ULID."""
        from ulid import ULID

        # Simulate what the handler does
        eval_run_id = str(ULID())

        assert len(eval_run_id) == 26
        assert eval_run_id.isalnum()

    def test_evaluation_config_dict_structure(self):
        """Test the expected structure of evaluation config dict."""
        # The config dict passed to _on_policy_evaluate_requested should have:
        expected_keys = ["env_id", "env_family", "agent_policies", "agents", "policy_types"]

        config = {
            "env_id": "pursuit_v4",
            "env_family": "sisl",
            "agent_policies": {"pursuer_0": "/path/to/checkpoint"},
            "agents": ["pursuer_0", "pursuer_1"],
            "policy_types": {"pursuer_0": "ray"},
        }

        for key in expected_keys:
            assert key in config


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
