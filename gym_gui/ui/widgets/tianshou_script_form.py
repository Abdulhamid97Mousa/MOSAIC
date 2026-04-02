"""Tianshou Custom Script form."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from qtpy import QtCore, QtWidgets
from qtpy.QtWidgets import QFileDialog
from ulid import ULID

from gym_gui.config.paths import TIANSHOU_SCRIPTS_DIR
from gym_gui.telemetry.semconv import VideoModes

# Import config from worker package
try:
    from tianshou_worker.config import TianshouWorkerConfig
except ImportError:
    @dataclass(frozen=True)
    class TianshouWorkerConfig:
        run_id: str
        algo: str
        env_id: str
        total_timesteps: int
        seed: Optional[int] = None
        extras: Dict[str, Any] = None
        worker_id: Optional[str] = None
        
        def to_dict(self):
            return self.__dict__

_LOGGER = logging.getLogger("gym_gui.ui.tianshou_script_form")


class TianshouScriptForm(QtWidgets.QDialog):
    """Configuration dialog for Tianshou script execution."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Tianshou Script Execution")
        self.setMinimumWidth(500)
        self.setMinimumHeight(300)
        
        self._setup_ui()
        self._populate_scripts()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # --- Script Selection ---
        group_script = QtWidgets.QGroupBox("Script Selection")
        form_script = QtWidgets.QFormLayout(group_script)
        
        self._script_combo = QtWidgets.QComboBox()
        form_script.addRow("Script:", self._script_combo)
        
        self._browse_btn = QtWidgets.QPushButton("Browse...")
        self._browse_btn.clicked.connect(self._on_browse_clicked)
        form_script.addRow("", self._browse_btn)
        
        layout.addWidget(group_script)
        
        # --- General Settings ---
        group_general = QtWidgets.QGroupBox("Settings")
        form_general = QtWidgets.QFormLayout(group_general)
        
        self._run_id_edit = QtWidgets.QLineEdit()
        self._run_id_edit.setReadOnly(True)
        form_general.addRow("Run ID:", self._run_id_edit)
        
        self._env_id_edit = QtWidgets.QLineEdit("CartPole-v1") # Default placeholder
        form_general.addRow("Env ID:", self._env_id_edit)
        
        layout.addWidget(group_general)
        
        # --- FastLane ---
        group_video = QtWidgets.QGroupBox("Rendering")
        form_video = QtWidgets.QFormLayout(group_video)
        self._fastlane_check = QtWidgets.QCheckBox("Enable FastLane")
        self._fastlane_check.setChecked(True)
        form_video.addRow(self._fastlane_check)
        layout.addWidget(group_video)
        
        # --- Buttons ---
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self._update_run_id()

    def _update_run_id(self):
        ulid = str(ULID())
        script_name = self._script_combo.currentText() or "script"
        slug = Path(script_name).stem
        self._run_id_edit.setText(f"tianshou-{slug}-{ulid}")

    def _populate_scripts(self):
        self._script_combo.clear()
        if TIANSHOU_SCRIPTS_DIR.exists():
            scripts = sorted(TIANSHOU_SCRIPTS_DIR.glob("*.py"))
            self._script_combo.addItems([s.name for s in scripts])
        
        if self._script_combo.count() == 0:
            self._script_combo.addItem("No scripts found")
            self._script_combo.setEnabled(False)

    def _on_browse_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Python Script", str(TIANSHOU_SCRIPTS_DIR), "Python Files (*.py)"
        )
        if path:
            # TODO: Add logic to handle external scripts if needed
            pass

    def get_config(self) -> TianshouWorkerConfig:
        run_id = self._run_id_edit.text()
        algo = "script" # Special algo type for scripts
        env_id = self._env_id_edit.text()
        
        extras = {
            "script_path": str(TIANSHOU_SCRIPTS_DIR / self._script_combo.currentText())
        }
        
        if self._fastlane_check.isChecked():
            extras["fastlane_enabled"] = True
            extras["video_mode"] = VideoModes.SINGLE.value

        return TianshouWorkerConfig(
            run_id=run_id,
            algo=algo,
            env_id=env_id,
            total_timesteps=1, # Dummy value for scripts
            extras=extras
        )

from gym_gui.ui.forms.factory import get_worker_form_factory

_factory = get_worker_form_factory()
if not _factory.has_script_form("tianshou_worker"):
    _factory.register_script_form(
        "tianshou_worker",
        lambda parent=None, **kwargs: TianshouScriptForm(parent=parent, **kwargs),
    )
