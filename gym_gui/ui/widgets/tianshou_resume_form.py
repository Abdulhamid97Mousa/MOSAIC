"""Dialog for resuming Tianshou training from a checkpoint."""

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

_LOGGER = logging.getLogger("gym_gui.ui.tianshou_resume_form")


class TianshouResumeForm(QtWidgets.QDialog):
    """Dialog for resuming training from a Tianshou checkpoint."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Resume Tianshou Training")
        self.resize(600, 400)
        
        self._setup_ui()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # --- Checkpoint Selection ---
        group_ckpt = QtWidgets.QGroupBox("Checkpoint Selection")
        form_ckpt = QtWidgets.QFormLayout(group_ckpt)
        
        self._ckpt_path_edit = QtWidgets.QLineEdit()
        self._ckpt_btn = QtWidgets.QPushButton("Browse...")
        self._ckpt_btn.clicked.connect(self._on_browse_ckpt)
        
        ckpt_layout = QtWidgets.QHBoxLayout()
        ckpt_layout.addWidget(self._ckpt_path_edit)
        ckpt_layout.addWidget(self._ckpt_btn)
        
        form_ckpt.addRow("Checkpoint File:", ckpt_layout)
        
        layout.addWidget(group_ckpt)
        
        # --- Config Override ---
        group_config = QtWidgets.QGroupBox("Configuration")
        form_config = QtWidgets.QFormLayout(group_config)
        
        self._algo_edit = QtWidgets.QLineEdit()
        self._algo_edit.setPlaceholderText("e.g. ppo")
        form_config.addRow("Algorithm:", self._algo_edit)
        
        self._env_edit = QtWidgets.QLineEdit()
        self._env_edit.setPlaceholderText("e.g. CartPole-v1")
        form_config.addRow("Environment:", self._env_edit)
        
        self._timesteps_spin = QtWidgets.QSpinBox()
        self._timesteps_spin.setRange(1000, 1_000_000_000)
        self._timesteps_spin.setValue(100_000)
        self._timesteps_spin.setSingleStep(10_000)
        self._timesteps_spin.setSuffix(" additional steps")
        form_config.addRow("Additional Steps:", self._timesteps_spin)
        
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

    def _on_browse_ckpt(self):
        start_dir = str(VAR_TRAINER_DIR) if VAR_TRAINER_DIR.exists() else "."
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Checkpoint", start_dir, "Checkpoint Files (*.pth *.pt);;All Files (*)"
        )
        if path:
            self._ckpt_path_edit.setText(path)
            self._try_load_config(Path(path).parent)

    def _try_load_config(self, run_dir: Path):
        # Try to find config-*.json in the directory
        configs = list(run_dir.glob("config-*.json"))
        if configs:
            try:
                data = json.loads(configs[0].read_text())
                if "algo" in data:
                    self._algo_edit.setText(data["algo"])
                if "env_id" in data:
                    self._env_edit.setText(data["env_id"])
            except Exception as e:
                _LOGGER.warning(f"Failed to load config from {configs[0]}: {e}")

    def get_config(self) -> TianshouWorkerConfig:
        ckpt_path = self._ckpt_path_edit.text()
        algo = self._algo_edit.text() or "ppo"
        env_id = self._env_edit.text() or "CartPole-v1"
        additional_steps = self._timesteps_spin.value()
        
        extras = {
            "resume_from": ckpt_path,
            "resume": True
        }
        
        if self._fastlane_check.isChecked():
            extras["fastlane_enabled"] = True
            extras["video_mode"] = VideoModes.SINGLE.value

        ulid = str(ULID())
        run_id = f"tianshou-resume-{Path(ckpt_path).stem}-{ulid}"

        return TianshouWorkerConfig(
            run_id=run_id,
            algo=algo,
            env_id=env_id,
            total_timesteps=additional_steps,
            extras=extras
        )

from gym_gui.ui.forms.factory import get_worker_form_factory

_factory = get_worker_form_factory()
if not _factory.has_resume_form("tianshou_worker"):
    _factory.register_resume_form(
        "tianshou_worker",
        lambda parent=None, **kwargs: TianshouResumeForm(parent=parent, **kwargs),
    )
