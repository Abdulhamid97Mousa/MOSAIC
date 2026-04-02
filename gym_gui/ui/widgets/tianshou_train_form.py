"""Tianshou worker training form."""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from qtpy import QtCore, QtGui, QtWidgets

from gym_gui.core.enums import (
    ENVIRONMENT_FAMILY_BY_GAME,
    EnvironmentFamily,
    GameId,
)
from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_UI_TRAIN_FORM_ERROR,
    LOG_UI_TRAIN_FORM_INFO,
    LOG_UI_TRAIN_FORM_TRACE,
    LOG_UI_TRAIN_FORM_WARNING,
)
from gym_gui.fastlane.worker_helpers import apply_fastlane_environment
from gym_gui.telemetry.semconv import VideoModes
from ulid import ULID

# Import config from worker package
try:
    from tianshou_worker.config import TianshouWorkerConfig
except ImportError:
    # Fallback/mock if worker package not installed in GUI env
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

_LOGGER = logging.getLogger("gym_gui.ui.tianshou_train_form")


def _generate_run_id(prefix: str, algo: str) -> str:
    """Generate a canonical run ID using ULID."""
    ulid = str(ULID())
    if algo and algo != "auto":
        return f"{prefix}-{algo}-{ulid}"
    return f"{prefix}-{ulid}"


@dataclass(frozen=True)
class _AlgoParamSpec:
    key: str
    label: str
    default: Any
    field_type: type
    tooltip: str | None = None


_ALGO_PARAM_SPECS: dict[str, tuple[_AlgoParamSpec, ...]] = {
    "ppo": (
        _AlgoParamSpec("lr", "Learning Rate", 3e-4, float, "Adam optimizer learning rate"),
        _AlgoParamSpec("hidden_sizes", "Hidden Sizes", [64, 64], list, "Network architecture"),
        _AlgoParamSpec("num_envs", "Parallel Envs", 8, int, "Number of parallel environments"),
        _AlgoParamSpec("step_per_collect", "Steps/Collect", 2048, int, "Steps per collection phase"),
        _AlgoParamSpec("batch_size", "Batch Size", 64, int, "Optimization batch size"),
        _AlgoParamSpec("epoch", "Epochs", 10, int, "Training epochs"),
    ),
    "dqn": (
        _AlgoParamSpec("lr", "Learning Rate", 1e-3, float, "Adam optimizer learning rate"),
        _AlgoParamSpec("hidden_sizes", "Hidden Sizes", [128, 128], list, "Network architecture"),
        _AlgoParamSpec("buffer_size", "Buffer Size", 20000, int, "Replay buffer capacity"),
        _AlgoParamSpec("batch_size", "Batch Size", 64, int, "Optimization batch size"),
        _AlgoParamSpec("epoch", "Epochs", 10, int, "Training epochs"),
        _AlgoParamSpec("eps_train", "Epsilon Train", 0.1, float, "Exploration rate during training"),
        _AlgoParamSpec("eps_test", "Epsilon Test", 0.05, float, "Exploration rate during testing"),
    ),
}

_SUPPORTED_FAMILIES: set[EnvironmentFamily] = {
    EnvironmentFamily.CLASSIC_CONTROL,
    EnvironmentFamily.BOX2D,
    EnvironmentFamily.MUJOCO,
    EnvironmentFamily.TOY_TEXT,
    EnvironmentFamily.MINIGRID,
}


class TianshouTrainForm(QtWidgets.QDialog, LogConstantMixin):
    """Configuration dialog for Tianshou training runs."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Tianshou Training Configuration")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)
        
        self._algo_widgets: dict[str, QtWidgets.QWidget] = {}
        
        self._setup_ui()
        self._populate_families()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # --- General Settings ---
        group_general = QtWidgets.QGroupBox("General Settings")
        form_general = QtWidgets.QFormLayout(group_general)
        
        self._run_id_edit = QtWidgets.QLineEdit()
        self._run_id_edit.setPlaceholderText("auto-generated")
        self._run_id_edit.setReadOnly(True)
        form_general.addRow("Run ID:", self._run_id_edit)
        
        self._algo_combo = QtWidgets.QComboBox()
        self._algo_combo.addItems(sorted(_ALGO_PARAM_SPECS.keys()))
        self._algo_combo.currentTextChanged.connect(self._on_algo_changed)
        form_general.addRow("Algorithm:", self._algo_combo)
        
        layout.addWidget(group_general)
        
        # --- Environment ---
        group_env = QtWidgets.QGroupBox("Environment")
        form_env = QtWidgets.QFormLayout(group_env)
        
        self._family_combo = QtWidgets.QComboBox()
        self._family_combo.currentTextChanged.connect(self._on_family_changed)
        form_env.addRow("Family:", self._family_combo)
        
        self._env_combo = QtWidgets.QComboBox()
        form_env.addRow("Environment:", self._env_combo)
        
        self._timesteps_spin = QtWidgets.QSpinBox()
        self._timesteps_spin.setRange(1000, 1_000_000_000)
        self._timesteps_spin.setValue(100_000)
        self._timesteps_spin.setSingleStep(10_000)
        self._timesteps_spin.setSuffix(" steps")
        form_env.addRow("Total Timesteps:", self._timesteps_spin)
        
        self._seed_spin = QtWidgets.QSpinBox()
        self._seed_spin.setRange(0, 999999)
        self._seed_spin.setValue(1)
        form_env.addRow("Seed:", self._seed_spin)
        
        layout.addWidget(group_env)
        
        # --- Algorithm Parameters ---
        self._group_params = QtWidgets.QGroupBox("Algorithm Hyperparameters")
        self._layout_params = QtWidgets.QFormLayout(self._group_params)
        layout.addWidget(self._group_params)
        
        # --- FastLane / Video ---
        group_video = QtWidgets.QGroupBox("Rendering & Video")
        form_video = QtWidgets.QFormLayout(group_video)
        
        self._fastlane_check = QtWidgets.QCheckBox("Enable FastLane (Live Rendering)")
        self._fastlane_check.setChecked(True)
        self._fastlane_check.setToolTip("Stream frames via shared memory for GUI visualization")
        form_video.addRow(self._fastlane_check)
        
        layout.addWidget(group_video)
        
        # --- Buttons ---
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        # Initial populate
        self._update_run_id()
        self._on_algo_changed(self._algo_combo.currentText())

    def _update_run_id(self):
        algo = self._algo_combo.currentText()
        self._run_id_edit.setText(_generate_run_id("tianshou", algo))

    def _populate_families(self):
        families = sorted([f.value for f in _SUPPORTED_FAMILIES])
        self._family_combo.addItems(families)
        
        # Default to Classic Control
        idx = self._family_combo.findText(EnvironmentFamily.CLASSIC_CONTROL.value)
        if idx >= 0:
            self._family_combo.setCurrentIndex(idx)

    def _on_family_changed(self, family_name: str):
        self._env_combo.clear()
        try:
            family = EnvironmentFamily(family_name)
            games = ENVIRONMENT_FAMILY_BY_GAME.get(family, [])
            # Filter games that are string IDs or simple enums
            env_ids = sorted([g.value if hasattr(g, "value") else str(g) for g in games])
            self._env_combo.addItems(env_ids)
        except ValueError:
            pass

    def _on_algo_changed(self, algo_name: str):
        # Clear existing params
        while self._layout_params.rowCount() > 0:
            self._layout_params.removeRow(0)
        self._algo_widgets.clear()
            
        specs = _ALGO_PARAM_SPECS.get(algo_name, ())
        for spec in specs:
            widget = None
            if spec.field_type == int:
                widget = QtWidgets.QSpinBox()
                widget.setRange(1, 1_000_000)
                widget.setValue(int(spec.default))
            elif spec.field_type == float:
                widget = QtWidgets.QDoubleSpinBox()
                widget.setRange(0.0001, 1000.0)
                widget.setDecimals(5)
                widget.setValue(float(spec.default))
            elif spec.field_type == bool:
                widget = QtWidgets.QCheckBox()
                widget.setChecked(bool(spec.default))
            elif spec.field_type == list:
                widget = QtWidgets.QLineEdit()
                widget.setText(json.dumps(spec.default))
                widget.setToolTip("JSON list, e.g. [64, 64]")
            
            if widget:
                if spec.tooltip:
                    widget.setToolTip(spec.tooltip)
                self._layout_params.addRow(spec.label + ":", widget)
                self._algo_widgets[spec.key] = widget

    def get_config(self) -> TianshouWorkerConfig:
        """Construct the worker configuration from form state."""
        run_id = self._run_id_edit.text()
        algo = self._algo_combo.currentText()
        env_id = self._env_combo.currentText()
        total_timesteps = self._timesteps_spin.value()
        seed = self._seed_spin.value()
        
        extras = {}
        for key, widget in self._algo_widgets.items():
            if isinstance(widget, QtWidgets.QSpinBox):
                extras[key] = widget.value()
            elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                extras[key] = widget.value()
            elif isinstance(widget, QtWidgets.QCheckBox):
                extras[key] = widget.isChecked()
            elif isinstance(widget, QtWidgets.QLineEdit):
                try:
                    extras[key] = json.loads(widget.text())
                except json.JSONDecodeError:
                    _LOGGER.warning(f"Failed to parse JSON for {key}, using text")
                    extras[key] = widget.text()

        # Handle FastLane
        if self._fastlane_check.isChecked():
            # Tianshou launcher needs to support this via extras or wrapper
            # For now, we just pass it in extras, launcher can ignore or use it
            extras["fastlane_enabled"] = True
            extras["video_mode"] = VideoModes.SINGLE.value  # Default to single for now
        
        return TianshouWorkerConfig(
            run_id=run_id,
            algo=algo,
            env_id=env_id,
            total_timesteps=total_timesteps,
            seed=seed,
            extras=extras
        )
