"""Dialog for evaluating a Tianshou policy."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from qtpy import QtCore, QtWidgets
from qtpy.QtWidgets import QFileDialog
from ulid import ULID

from gym_gui.config.paths import VAR_TRAINER_DIR
from gym_gui.telemetry.semconv import VideoModes

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

_LOGGER = logging.getLogger("gym_gui.ui.tianshou_policy_form")


class TianshouPolicyForm(QtWidgets.QDialog):
    """Dialog for evaluating a Tianshou policy."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Evaluate Tianshou Policy")
        self.resize(600, 400)
        
        self._setup_ui()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # --- Policy Selection ---
        group_policy = QtWidgets.QGroupBox("Policy Selection")
        form_policy = QtWidgets.QFormLayout(group_policy)
        
        self._policy_path_edit = QtWidgets.QLineEdit()
        self._policy_btn = QtWidgets.QPushButton("Browse...")
        self._policy_btn.clicked.connect(self._on_browse_policy)
        
        policy_layout = QtWidgets.QHBoxLayout()
        policy_layout.addWidget(self._policy_path_edit)
        policy_layout.addWidget(self._policy_btn)
        
        form_policy.addRow("Policy File:", policy_layout)
        
        layout.addWidget(group_policy)
        
        # --- Config Override ---
        group_config = QtWidgets.QGroupBox("Configuration")
        form_config = QtWidgets.QFormLayout(group_config)
        
        self._algo_edit = QtWidgets.QLineEdit()
        self._algo_edit.setPlaceholderText("e.g. ppo")
        form_config.addRow("Algorithm:", self._algo_edit)
        
        self._env_edit = QtWidgets.QLineEdit()
        self._env_edit.setPlaceholderText("e.g. CartPole-v1")
        form_config.addRow("Environment:", self._env_edit)
        
        self._episodes_spin = QtWidgets.QSpinBox()
        self._episodes_spin.setRange(1, 1000)
        self._episodes_spin.setValue(10)
        form_config.addRow("Evaluation Episodes:", self._episodes_spin)
        
        layout.addWidget(group_config)
        
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

    def _on_browse_policy(self):
        start_dir = str(VAR_TRAINER_DIR) if VAR_TRAINER_DIR.exists() else "."
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Policy", start_dir, "Policy Files (*.pth *.pt);;All Files (*)"
        )
        if path:
            self._policy_path_edit.setText(path)
            # Try to infer details
            if "ppo" in path.lower():
                self._algo_edit.setText("ppo")
            elif "dqn" in path.lower():
                self._algo_edit.setText("dqn")

    def get_config(self) -> TianshouWorkerConfig:
        policy_path = self._policy_path_edit.text()
        algo = self._algo_edit.text() or "ppo"
        env_id = self._env_edit.text() or "CartPole-v1"
        episodes = self._episodes_spin.value()
        
        extras = {
            "policy_path": policy_path,
            "eval_only": True,
            "eval_episodes": episodes
        }
        
        if self._fastlane_check.isChecked():
            extras["fastlane_enabled"] = True
            extras["video_mode"] = VideoModes.SINGLE.value

        ulid = str(ULID())
        run_id = f"tianshou-eval-{Path(policy_path).stem}-{ulid}"

        # Note: total_timesteps is reused as episodes count for eval mode in some workers,
        # or handled by setting it to 1 and using extras.
        # CleanRL worker uses total_timesteps for total frames even in eval.
        # Here we pass 1 to satisfy required field, actual eval loop uses extras["eval_episodes"]
        
        return TianshouWorkerConfig(
            run_id=run_id,
            algo=algo,
            env_id=env_id,
            total_timesteps=1, 
            extras=extras
        )


from gym_gui.ui.forms.factory import get_worker_form_factory

_factory = get_worker_form_factory()
if not _factory.has_policy_form("tianshou_worker"):
    _factory.register_policy_form(
        "tianshou_worker",
        lambda parent=None, **kwargs: TianshouPolicyForm(parent=parent, **kwargs),
    )
