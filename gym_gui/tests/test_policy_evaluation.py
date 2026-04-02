"""Tests for Ray RLlib policy evaluation and checkpoint discovery.

These tests verify:
1. Ray policy metadata discovery (ray_policy_metadata.py)
2. PolicyAssignmentPanel checkpoint scanning
3. Signal connections for policy_evaluate_requested
4. PolicyEvaluator configuration and setup
5. Log constants for policy evaluation
6. LoadPolicyDialog functionality
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestRayPolicyMetadata:
    """Tests for Ray RLlib checkpoint discovery."""

    def test_import_ray_policy_metadata(self):
        """Test that ray_policy_metadata module can be imported."""
        from gym_gui.policy_discovery.ray_policy_metadata import (
            RayRLlibCheckpoint,
            RayRLlibPolicy,
            discover_ray_checkpoints,
            discover_ray_policies,
            get_checkpoints_for_env,
            get_latest_checkpoint,
            load_checkpoint_metadata,
        )

        # Verify all exports are available
        assert RayRLlibCheckpoint is not None
        assert RayRLlibPolicy is not None
        assert callable(discover_ray_checkpoints)
        assert callable(discover_ray_policies)
        assert callable(get_checkpoints_for_env)
        assert callable(get_latest_checkpoint)
        assert callable(load_checkpoint_metadata)

    def test_workers_package_exports(self):
        """Test that workers package exports ray_policy_metadata."""
        from gym_gui.policy_discovery import (
            RayRLlibCheckpoint,
            RayRLlibPolicy,
            discover_ray_checkpoints,
            discover_ray_policies,
            get_checkpoints_for_env,
            get_latest_checkpoint,
            load_ray_checkpoint_metadata,
        )

        assert RayRLlibCheckpoint is not None
        assert callable(discover_ray_checkpoints)

    def test_checkpoint_dataclass_properties(self):
        """Test RayRLlibCheckpoint dataclass properties."""
        from gym_gui.policy_discovery.ray_policy_metadata import RayRLlibCheckpoint

        checkpoint = RayRLlibCheckpoint(
            checkpoint_path=Path("/tmp/test/checkpoints"),
            run_id="01ABC123",
            algorithm="PPO",
            env_id="pursuit_v4",
            env_family="sisl",
            paradigm="parameter_sharing",
            policy_ids=["shared"],
            ray_version="2.52.1",
            checkpoint_version="1.1",
            config_path=None,
        )

        # Test display_name property
        assert "01ABC123" in checkpoint.display_name
        assert "pursuit_v4" in checkpoint.display_name
        assert "PPO" in checkpoint.display_name

        # Test policies_dir property
        assert checkpoint.policies_dir == Path("/tmp/test/checkpoints/policies")

    def test_checkpoint_get_policy_path(self):
        """Test get_policy_path method."""
        from gym_gui.policy_discovery.ray_policy_metadata import RayRLlibCheckpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            checkpoint_dir.mkdir()
            policies_dir = checkpoint_dir / "policies" / "shared"
            policies_dir.mkdir(parents=True)

            checkpoint = RayRLlibCheckpoint(
                checkpoint_path=checkpoint_dir,
                run_id="test_run",
                algorithm="PPO",
                env_id="pursuit_v4",
                env_family="sisl",
                paradigm="parameter_sharing",
                policy_ids=["shared"],
            )

            # Should find existing policy
            policy_path = checkpoint.get_policy_path("shared")
            assert policy_path is not None
            assert policy_path.exists()

            # Should return None for non-existent policy
            assert checkpoint.get_policy_path("nonexistent") is None

    def test_discover_ray_checkpoints_empty(self):
        """Test discovery with no checkpoints."""
        from gym_gui.policy_discovery.ray_policy_metadata import discover_ray_checkpoints

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoints = discover_ray_checkpoints(Path(tmpdir))
            assert checkpoints == []

    def test_discover_ray_checkpoints_with_mock_structure(self):
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
                    "algorithm": "APPO",
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
            assert ckpt.algorithm == "APPO"
            assert ckpt.env_id == "pursuit_v4"
            assert ckpt.paradigm == "parameter_sharing"
            assert "shared" in ckpt.policy_ids

    def test_get_checkpoints_for_env(self):
        """Test filtering checkpoints by environment."""
        from gym_gui.policy_discovery.ray_policy_metadata import get_checkpoints_for_env

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two mock checkpoints for different envs
            for run_id, env_id in [("01RUN001", "pursuit_v4"), ("01RUN002", "waterworld_v4")]:
                run_dir = Path(tmpdir) / run_id
                run_dir.mkdir()

                analytics = {
                    "worker_type": "ray_worker",
                    "ray_metadata": {
                        "algorithm": "PPO",
                        "env_id": env_id,
                        "env_family": "sisl",
                        "paradigm": "parameter_sharing",
                    },
                }
                (run_dir / "analytics.json").write_text(json.dumps(analytics))

                checkpoint_dir = run_dir / "checkpoints"
                checkpoint_dir.mkdir()
                (checkpoint_dir / "rllib_checkpoint.json").write_text(
                    json.dumps({"policy_ids": ["shared"], "ray_version": "2.52.1"})
                )
                (checkpoint_dir / "algorithm_state.pkl").touch()

            # Filter by env
            pursuit_ckpts = get_checkpoints_for_env("pursuit_v4", Path(tmpdir))
            assert len(pursuit_ckpts) == 1
            assert pursuit_ckpts[0].env_id == "pursuit_v4"

            waterworld_ckpts = get_checkpoints_for_env("waterworld_v4", Path(tmpdir))
            assert len(waterworld_ckpts) == 1
            assert waterworld_ckpts[0].env_id == "waterworld_v4"


class TestPolicyAssignmentPanel:
    """Tests for PolicyAssignmentPanel checkpoint scanning."""

    def test_import_policy_assignment_panel(self):
        """Test that PolicyAssignmentPanel can be imported."""
        # Skip if Qt not available
        pytest.importorskip("qtpy")

        from gym_gui.ui.widgets.policy_assignment_panel import (
            PolicyAssignmentDialog,
            PolicyAssignmentPanel,
        )

        assert PolicyAssignmentPanel is not None
        assert PolicyAssignmentDialog is not None

    def test_panel_uses_ray_and_cleanrl_discovery(self):
        """Test that panel imports both discovery functions."""
        pytest.importorskip("qtpy")

        # Check imports are present in the module
        from gym_gui.ui.widgets import policy_assignment_panel

        # Verify the imports are used
        assert hasattr(policy_assignment_panel, "discover_ray_checkpoints")
        assert hasattr(policy_assignment_panel, "discover_cleanrl_policies")


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


@pytest.mark.skipif(
    not importlib.util.find_spec("ray_worker") or not importlib.util.find_spec("ray"),
    reason="ray_worker or ray not installed",
)
class TestPolicyEvaluator:
    """Tests for PolicyEvaluator."""

    def test_import_policy_evaluator(self):
        """Test that PolicyEvaluator can be imported."""
        from ray_worker.policy_evaluator import (
            EpisodeMetrics,
            EvaluationConfig,
            PolicyEvaluator,
            run_evaluation,
        )

        assert EvaluationConfig is not None
        assert EpisodeMetrics is not None
        assert PolicyEvaluator is not None
        assert callable(run_evaluation)

    def test_ray_worker_exports_evaluator(self):
        """Test that ray_worker exports evaluator classes."""
        from ray_worker import (
            EpisodeMetrics,
            EvaluationConfig,
            PolicyEvaluator,
            run_evaluation,
        )

        assert EvaluationConfig is not None
        assert callable(run_evaluation)

    def test_evaluation_config_defaults(self):
        """Test EvaluationConfig default values."""
        from ray_worker.policy_evaluator import EvaluationConfig

        config = EvaluationConfig(
            env_id="pursuit_v4",
            env_family="sisl",
            checkpoint_path="/path/to/checkpoint",
            run_id="eval_001",
        )

        assert config.policy_id == "shared"
        assert config.num_episodes == 10
        assert config.max_steps_per_episode == 1000
        assert config.render_mode == "rgb_array"
        assert config.fastlane_enabled is True
        assert config.deterministic is True
        assert config.seed == 42
        assert config.frame_skip == 1

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
        assert metrics.terminated is True

    def test_policy_evaluator_init(self):
        """Test PolicyEvaluator initialization."""
        from ray_worker.policy_evaluator import EvaluationConfig, PolicyEvaluator

        config = EvaluationConfig(
            env_id="pursuit_v4",
            env_family="sisl",
            checkpoint_path="/path/to/checkpoint",
            run_id="eval_001",
        )

        evaluator = PolicyEvaluator(config)

        assert evaluator.config == config
        assert evaluator._env is None
        assert evaluator._policy_actor is None
        assert evaluator._metrics == []

    def test_policy_evaluator_get_summary_empty(self):
        """Test get_summary with no metrics."""
        from ray_worker.policy_evaluator import EvaluationConfig, PolicyEvaluator

        config = EvaluationConfig(
            env_id="pursuit_v4",
            env_family="sisl",
            checkpoint_path="/path/to/checkpoint",
            run_id="eval_001",
        )

        evaluator = PolicyEvaluator(config)
        summary = evaluator.get_summary()

        assert summary == {}


class TestLogConstants:
    """Tests for policy evaluation log constants."""

    def test_policy_assignment_log_constants(self):
        """Test policy assignment log constants are defined."""
        from gym_gui.logging_config.log_constants import (
            LOG_UI_POLICY_ASSIGNMENT_LOADED,
            LOG_UI_POLICY_ASSIGNMENT_REQUESTED,
            LOG_UI_POLICY_DISCOVERY_FOUND,
            LOG_UI_POLICY_DISCOVERY_SCAN,
        )

        assert LOG_UI_POLICY_ASSIGNMENT_REQUESTED.code == "LOG760"
        assert LOG_UI_POLICY_ASSIGNMENT_LOADED.code == "LOG761"
        assert LOG_UI_POLICY_DISCOVERY_SCAN.code == "LOG762"
        assert LOG_UI_POLICY_DISCOVERY_FOUND.code == "LOG763"

    def test_log_constants_in_all_list(self):
        """Test new constants are exported in __all__."""
        from gym_gui.logging_config import log_constants

        assert "LOG_UI_POLICY_ASSIGNMENT_REQUESTED" in log_constants.__all__
        assert "LOG_UI_POLICY_ASSIGNMENT_LOADED" in log_constants.__all__
        assert "LOG_UI_POLICY_DISCOVERY_SCAN" in log_constants.__all__
        assert "LOG_UI_POLICY_DISCOVERY_FOUND" in log_constants.__all__


class TestSignalConnections:
    """Tests for signal connections in the evaluation flow."""

    def test_control_panel_has_policy_evaluate_signal(self):
        """Test that ControlPanelWidget has policy_evaluate_requested signal."""
        pytest.importorskip("qtpy")

        import os
        if not os.environ.get("DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
            os.environ["QT_QPA_PLATFORM"] = "offscreen"

        from qtpy import QtWidgets

        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        # Import and check signal exists
        from gym_gui.ui.widgets.control_panel import ControlPanelWidget

        # Check the signal is defined as a class attribute
        assert hasattr(ControlPanelWidget, "policy_evaluate_requested")

    def test_multi_agent_tab_has_policy_evaluate_signal(self):
        """Test that MultiAgentTab has evaluate_requested signal."""
        pytest.importorskip("qtpy")

        from gym_gui.ui.widgets.multi_agent_tab import MultiAgentTab

        # Check the signal is defined
        assert hasattr(MultiAgentTab, "evaluate_requested")


class TestIntegrationWithActualCheckpoints:
    """Integration tests using actual checkpoints in var/trainer/runs."""

    def test_discover_actual_checkpoints(self):
        """Test discovering actual checkpoints in the project."""
        from gym_gui.config.paths import VAR_TRAINER_DIR
        from gym_gui.policy_discovery.ray_policy_metadata import discover_ray_checkpoints

        runs_dir = VAR_TRAINER_DIR / "runs"
        if not runs_dir.exists():
            pytest.skip("No runs directory found")

        checkpoints = discover_ray_checkpoints()

        # Just verify the function works without error
        # The actual count depends on what's been trained
        assert isinstance(checkpoints, list)

        if checkpoints:
            # If we have checkpoints, verify structure
            ckpt = checkpoints[0]
            assert hasattr(ckpt, "run_id")
            assert hasattr(ckpt, "algorithm")
            assert hasattr(ckpt, "env_id")
            assert hasattr(ckpt, "policy_ids")

    def test_policy_actor_import(self):
        """Test that RayPolicyActor can be imported."""
        pytest.importorskip("ray")

        from ray_worker.policy_actor import (
            RayPolicyActor,
            RayPolicyConfig,
            RayPolicyController,
            create_ray_actor,
            list_checkpoint_policies,
        )

        assert RayPolicyActor is not None
        assert callable(create_ray_actor)


class TestCleanRLPolicyMetadata:
    """Tests for CleanRL policy metadata (existing functionality)."""

    def test_cleanrl_metadata_import(self):
        """Test CleanRL metadata can still be imported."""
        from gym_gui.policy_discovery.cleanrl_policy_metadata import (
            CleanRlCheckpoint,
            discover_policies,
            load_metadata_for_policy,
        )

        assert CleanRlCheckpoint is not None
        assert callable(discover_policies)

    def test_workers_exports_cleanrl(self):
        """Test workers package exports CleanRL metadata."""
        from gym_gui.policy_discovery import (
            CleanRlCheckpoint,
            discover_cleanrl_policies,
            load_cleanrl_metadata,
        )

        assert CleanRlCheckpoint is not None
        assert callable(discover_cleanrl_policies)


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
