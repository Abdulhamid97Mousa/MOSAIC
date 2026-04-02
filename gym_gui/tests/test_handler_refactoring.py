"""Tests for handler refactoring from main_window.py.

These tests verify that the extracted handlers:
1. Can be imported successfully
2. Have correct class interfaces
3. Can be instantiated with mock dependencies
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


class TestHandlerImports:
    """Tests that all handlers can be imported."""

    def test_import_policy_evaluation_handler(self):
        """Test PolicyEvaluationHandler can be imported."""
        from gym_gui.ui.handlers import PolicyEvaluationHandler

        assert PolicyEvaluationHandler is not None

    def test_import_fastlane_tab_handler(self):
        """Test FastLaneTabHandler can be imported."""
        from gym_gui.ui.handlers import FastLaneTabHandler

        assert FastLaneTabHandler is not None

    def test_import_training_monitor_handler(self):
        """Test TrainingMonitorHandler can be imported."""
        from gym_gui.ui.handlers import TrainingMonitorHandler

        assert TrainingMonitorHandler is not None

    def test_import_training_form_handler(self):
        """Test TrainingFormHandler can be imported."""
        from gym_gui.ui.handlers import TrainingFormHandler

        assert TrainingFormHandler is not None

    def test_import_multi_agent_game_handler(self):
        """Test MultiAgentGameHandler can be imported."""
        from gym_gui.ui.handlers import MultiAgentGameHandler

        assert MultiAgentGameHandler is not None

    def test_handlers_in_all_list(self):
        """Test all new handlers are in __all__ list."""
        from gym_gui.ui.handlers import __all__

        expected_handlers = [
            "PolicyEvaluationHandler",
            "FastLaneTabHandler",
            "TrainingMonitorHandler",
            "TrainingFormHandler",
            "MultiAgentGameHandler",
        ]

        for handler in expected_handlers:
            assert handler in __all__, f"{handler} not in __all__"


class TestPolicyEvaluationHandler:
    """Tests for PolicyEvaluationHandler."""

    def test_handler_init(self):
        """Test handler can be instantiated."""
        from gym_gui.ui.handlers import PolicyEvaluationHandler

        mock_parent = MagicMock()
        mock_status_bar = MagicMock()
        mock_open_tabs = MagicMock()

        handler = PolicyEvaluationHandler(
            parent=mock_parent,
            status_bar=mock_status_bar,
            open_ray_fastlane_tabs=mock_open_tabs,
        )

        assert handler is not None
        assert handler._parent == mock_parent
        assert handler._status_bar == mock_status_bar

    def test_handler_has_handle_method(self):
        """Test handler has handle_evaluate_request method."""
        from gym_gui.ui.handlers import PolicyEvaluationHandler

        handler = PolicyEvaluationHandler(
            parent=MagicMock(),
            status_bar=MagicMock(),
            open_ray_fastlane_tabs=MagicMock(),
        )

        assert hasattr(handler, "handle_evaluate_request")
        assert callable(handler.handle_evaluate_request)


class TestFastLaneTabHandler:
    """Tests for FastLaneTabHandler."""

    def test_handler_init(self):
        """Test handler can be instantiated."""
        from gym_gui.ui.handlers import FastLaneTabHandler

        mock_render_tabs = MagicMock()

        handler = FastLaneTabHandler(
            render_tabs=mock_render_tabs,
            log_callback=None,
        )

        assert handler is not None
        assert handler._render_tabs == mock_render_tabs

    def test_metadata_extraction_methods(self):
        """Test metadata extraction methods exist."""
        from gym_gui.ui.handlers import FastLaneTabHandler

        handler = FastLaneTabHandler(
            render_tabs=MagicMock(),
        )

        # Test metadata extraction methods
        assert hasattr(handler, "get_num_workers")
        assert hasattr(handler, "get_canonical_agent_id")
        assert hasattr(handler, "get_worker_id")
        assert hasattr(handler, "get_env_id")
        assert hasattr(handler, "get_run_mode")
        assert hasattr(handler, "metadata_supports_fastlane")

    def test_get_num_workers_from_metadata(self):
        """Test extracting num_workers from metadata."""
        from gym_gui.ui.handlers import FastLaneTabHandler

        handler = FastLaneTabHandler(render_tabs=MagicMock())

        # Test with valid metadata
        metadata = {
            "worker": {
                "config": {
                    "resources": {
                        "num_workers": 4
                    }
                }
            }
        }
        assert handler.get_num_workers(metadata) == 4

        # Test with missing data
        assert handler.get_num_workers({}) == 0
        assert handler.get_num_workers({"worker": {}}) == 0

    def test_metadata_supports_fastlane(self):
        """Test FastLane support detection."""
        from gym_gui.ui.handlers import FastLaneTabHandler

        handler = FastLaneTabHandler(render_tabs=MagicMock())

        # Test CleanRL worker detection
        metadata = {
            "worker": {
                "worker_id": "cleanrl_worker"
            }
        }
        assert handler.metadata_supports_fastlane(metadata) is True

        # Test fastlane_only flag
        metadata = {
            "ui": {
                "fastlane_only": True
            }
        }
        assert handler.metadata_supports_fastlane(metadata) is True

        # Test no FastLane support
        metadata = {
            "worker": {
                "worker_id": "some_other_worker"
            }
        }
        assert handler.metadata_supports_fastlane(metadata) is False


class TestTrainingFormHandler:
    """Tests for TrainingFormHandler."""

    def test_handler_init(self):
        """Test handler can be instantiated."""
        from gym_gui.ui.handlers import TrainingFormHandler

        handler = TrainingFormHandler(
            parent=MagicMock(),
            get_form_factory=MagicMock(),
            get_current_game=MagicMock(return_value=None),
            get_cleanrl_env_id=MagicMock(return_value=None),
            submit_config=MagicMock(),
            build_policy_config=MagicMock(return_value=None),
        )

        assert handler is not None

    def test_handler_has_form_methods(self):
        """Test handler has all form handling methods."""
        from gym_gui.ui.handlers import TrainingFormHandler

        handler = TrainingFormHandler(
            parent=MagicMock(),
            get_form_factory=MagicMock(),
            get_current_game=MagicMock(return_value=None),
            get_cleanrl_env_id=MagicMock(return_value=None),
            submit_config=MagicMock(),
            build_policy_config=MagicMock(return_value=None),
        )

        assert hasattr(handler, "on_trained_agent_requested")
        assert hasattr(handler, "on_train_agent_requested")
        assert hasattr(handler, "on_resume_training_requested")


class TestMultiAgentGameHandler:
    """Tests for MultiAgentGameHandler."""

    def test_handler_init(self):
        """Test handler can be instantiated."""
        from gym_gui.ui.handlers import MultiAgentGameHandler

        handler = MultiAgentGameHandler(
            status_bar=MagicMock(),
            chess_loader=MagicMock(),
            connect_four_loader=MagicMock(),
            go_loader=MagicMock(),
            tictactoe_loader=MagicMock(),
            checkers_loader=MagicMock(),
            set_game_info=MagicMock(),
            get_game_info=MagicMock(return_value=None),
        )

        assert handler is not None

    def test_handler_has_routing_methods(self):
        """Test handler has all routing methods."""
        from gym_gui.ui.handlers import MultiAgentGameHandler

        handler = MultiAgentGameHandler(
            status_bar=MagicMock(),
            chess_loader=MagicMock(),
            connect_four_loader=MagicMock(),
            go_loader=MagicMock(),
            tictactoe_loader=MagicMock(),
            checkers_loader=MagicMock(),
            set_game_info=MagicMock(),
            get_game_info=MagicMock(return_value=None),
        )

        assert hasattr(handler, "on_load_requested")
        assert hasattr(handler, "on_start_requested")
        assert hasattr(handler, "on_reset_requested")
        assert hasattr(handler, "on_ai_opponent_changed")

    def test_load_routes_to_chess(self):
        """Test loading chess routes to chess loader."""
        from gym_gui.ui.handlers import MultiAgentGameHandler

        mock_chess_loader = MagicMock()
        mock_get_game_info = MagicMock(return_value="<html>Chess</html>")

        handler = MultiAgentGameHandler(
            status_bar=MagicMock(),
            chess_loader=mock_chess_loader,
            connect_four_loader=MagicMock(),
            go_loader=MagicMock(),
            tictactoe_loader=MagicMock(),
            checkers_loader=MagicMock(),
            set_game_info=MagicMock(),
            get_game_info=mock_get_game_info,
        )

        handler.on_load_requested("chess_v6", seed=42)

        mock_chess_loader.load.assert_called_once()

    def test_load_routes_to_connect_four(self):
        """Test loading connect_four routes to connect_four loader."""
        from gym_gui.ui.handlers import MultiAgentGameHandler

        mock_connect_four_loader = MagicMock()
        mock_get_game_info = MagicMock(return_value="<html>Connect Four</html>")

        handler = MultiAgentGameHandler(
            status_bar=MagicMock(),
            chess_loader=MagicMock(),
            connect_four_loader=mock_connect_four_loader,
            go_loader=MagicMock(),
            tictactoe_loader=MagicMock(),
            checkers_loader=MagicMock(),
            set_game_info=MagicMock(),
            get_game_info=mock_get_game_info,
        )

        handler.on_load_requested("connect_four_v3", seed=42)

        mock_connect_four_loader.load.assert_called_once()


class TestTrainingMonitorHandler:
    """Tests for TrainingMonitorHandler (requires Qt)."""

    @pytest.fixture(scope="class")
    def qt_app(self):
        """Create QApplication for Qt-dependent tests."""
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

        from qtpy import QtWidgets
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        yield app

    def test_handler_init(self, qt_app):
        """Test handler can be instantiated."""
        from gym_gui.ui.handlers import TrainingMonitorHandler

        mock_live_controller = MagicMock()
        mock_analytics_tabs = MagicMock()
        mock_render_tabs = MagicMock()
        mock_run_metadata: Dict[tuple, Dict[str, Any]] = {}

        handler = TrainingMonitorHandler(
            parent=None,  # type: ignore[arg-type]
            live_controller=mock_live_controller,
            analytics_tabs=mock_analytics_tabs,
            render_tabs=mock_render_tabs,
            run_metadata=mock_run_metadata,
            trainer_dir=Path("/tmp/trainer"),
        )

        assert handler is not None

    def test_handler_has_monitoring_methods(self, qt_app):
        """Test handler has monitoring methods."""
        from gym_gui.ui.handlers import TrainingMonitorHandler

        handler = TrainingMonitorHandler(
            parent=None,  # type: ignore[arg-type]
            live_controller=MagicMock(),
            analytics_tabs=MagicMock(),
            render_tabs=MagicMock(),
            run_metadata={},
            trainer_dir=Path("/tmp/trainer"),
        )

        assert hasattr(handler, "poll_for_new_runs")
        assert hasattr(handler, "start_run_watch")
        assert hasattr(handler, "shutdown_run_watch")
        assert hasattr(handler, "on_run_completed")
        assert hasattr(handler, "backfill_run_metadata_from_disk")


class TestEvaluationWorker:
    """Tests for EvaluationWorker QThread."""

    @pytest.fixture(scope="class")
    def qt_app(self):
        """Create QApplication for Qt-dependent tests."""
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

        from qtpy import QtWidgets
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        yield app

    def test_worker_class_exists(self, qt_app):
        """Test EvaluationWorker class can be imported."""
        from gym_gui.ui.handlers.features.policy_evaluation_handler import EvaluationWorker

        assert EvaluationWorker is not None

    def test_worker_has_signals(self, qt_app):
        """Test EvaluationWorker has required signals."""
        from gym_gui.ui.handlers.features.policy_evaluation_handler import EvaluationWorker

        worker = EvaluationWorker(
            eval_run_id="test-run",
            env_id="pursuit_v4",
            env_family="sisl",
            checkpoint_path="/tmp/checkpoint",
        )

        assert hasattr(worker, "finished_signal")
        assert hasattr(worker, "error_signal")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
