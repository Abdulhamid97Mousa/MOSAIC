
import logging
import sys
import unittest
from unittest.mock import MagicMock
import os
from pathlib import Path

# Mock Qt modules to avoid GUI requirement in headless environment
# We must mock these BEFORE importing any gym_gui.ui modules

# Define mock classes that are compatible with LogConstantMixin
class MockQDialog:
    def __init__(self, parent=None):
        pass
    def setWindowTitle(self, title): pass
    def setMinimumWidth(self, w): pass
    def setMinimumHeight(self, h): pass
    def resize(self, w, h): pass
    def setLayout(self, l): pass
    def accept(self): pass
    def reject(self): pass

class MockQWidget:
    def __init__(self, parent=None):
        pass

# Create a mock module structure
qt_mock = MagicMock()
qt_mock.QtWidgets.QDialog = MockQDialog
qt_mock.QtWidgets.QWidget = MockQWidget
qt_mock.QtCore.Qt = MagicMock()
qt_mock.QtCore.QThread = MagicMock()
qt_mock.QtCore.Signal = MagicMock()

# Setup sys.modules
sys.modules["qtpy"] = qt_mock
sys.modules["qtpy.QtWidgets"] = qt_mock.QtWidgets
sys.modules["qtpy.QtCore"] = qt_mock.QtCore
sys.modules["qtpy.QtGui"] = MagicMock()
sys.modules["PyQt6"] = MagicMock()
sys.modules["PyQt6.QtCore"] = MagicMock()
sys.modules["PyQt6.QtGui"] = MagicMock()
sys.modules["PyQt6.QtQml"] = MagicMock()
sys.modules["PyQt6.QtQuick"] = MagicMock()
sys.modules["PySide6"] = MagicMock()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_tianshou_ui_flows")

class TestTianshouUIFlows(unittest.TestCase):
    def test_imports(self):
        """Verify that UI modules can be imported without error."""
        try:
            from gym_gui.ui.widgets import tianshou_train_form
            from gym_gui.ui.widgets import tianshou_script_form
            from gym_gui.ui.widgets import tianshou_resume_form
            from gym_gui.ui.widgets import tianshou_policy_form
            from gym_gui.ui.presenters.workers import tianshou_worker_presenter
            
            logger.info("All Tianshou UI modules imported successfully.")
        except ImportError as e:
            self.fail(f"ImportError: {e}")

    def test_presenter_train_request(self):
        """Verify presenter builds correct train request with ULID."""
        from gym_gui.ui.presenters.workers import tianshou_worker_presenter
        
        presenter = tianshou_worker_presenter.TianshouWorkerPresenter()
        
        # Test build_train_request logic
        # Pass string path to avoid pathlib mock issues
        path_str = "/tmp/test_policy.pt"
        
        request = presenter.build_train_request(path_str, None)
        
        self.assertEqual(request["worker_id"], "tianshou_worker")
        self.assertEqual(request["algo"], "ppo") # Defaults to ppo if not in name
        
        run_id = request["run_id"]
        logger.info(f"Generated Run ID: {run_id}")
        
        # Verify format: tianshou-eval-{stem}-{ulid}
        parts = run_id.split("-")
        self.assertEqual(parts[0], "tianshou")
        self.assertEqual(parts[1], "eval")
        self.assertEqual(parts[2], "test_policy")
        self.assertEqual(len(parts[3]), 26) # ULID length
        
        # Verify extras
        self.assertTrue(request["extras"]["eval_only"])
        self.assertTrue(request["extras"]["fastlane_enabled"])
        self.assertEqual(request["extras"]["video_mode"], "single")

    def test_dqn_algo_detection(self):
        """Verify algo detection from filename."""
        from gym_gui.ui.presenters.workers import tianshou_worker_presenter
        presenter = tianshou_worker_presenter.TianshouWorkerPresenter()
        
        path_str = "/tmp/my_dqn_agent.pt"
        request = presenter.build_train_request(path_str, None)
        
        self.assertEqual(request["algo"], "dqn")
        self.assertIn("my_dqn_agent", request["run_id"])

if __name__ == "__main__":
    unittest.main()
