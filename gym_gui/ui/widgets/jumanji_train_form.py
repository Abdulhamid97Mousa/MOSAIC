"""Jumanji worker training form.

Provides a configuration dialog for launching Jumanji training runs.
Supports two environment backends:
  - Jumanji (native JAX): Game2048, Sudoku, TSP, etc. with A2C
  - gymnax (JAX CartPole/Pendulum): Fully JIT-compiled PPO training loop
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from qtpy import QtCore, QtWidgets

from gym_gui.core.enums import EnvironmentFamily, GameId
from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.fastlane.worker_helpers import apply_fastlane_environment
from gym_gui.telemetry.semconv import VideoModes
from ulid import ULID

try:
    from jumanji_worker.config import JumanjiWorkerConfig
except ImportError:

    @dataclass(frozen=True)
    class JumanjiWorkerConfig:
        run_id: str
        env_id: str
        agent: str = "a2c"
        seed: Optional[int] = None
        num_epochs: int = 100
        device: str = "cpu"
        extras: Dict[str, Any] = None

        def to_dict(self):
            return self.__dict__


_LOGGER = logging.getLogger("gym_gui.ui.jumanji_train_form")


def _generate_run_id(prefix: str, backend: str) -> str:
    ulid = str(ULID())
    return f"{prefix}-{backend}-{ulid}"


# Jumanji native environments (JAX-based, use A2C)
_JUMANJI_ENVS = [
    "Game2048-v1",
    "GraphColoring-v1",
    "Minesweeper-v0",
    "RubiksCube-v0",
    "SlidingTilePuzzle-v0",
    "Sudoku-v0",
    "Snake-v1",
    "Tetris-v0",
    "Maze-v0",
    "PacMan-v1",
    "TSP-v1",
    "Knapsack-v1",
    "BinPack-v2",
    "Connector-v3",
    "CVRP-v1",
    "Sokoban-v0",
    "Cleaner-v0",
    "RobotWarehouse-v0",
]

# gymnax environments (fully JIT-compiled, use PPO)
_GYMNAX_ENVS = [
    "CartPole-v1",
    "Pendulum-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
    "Acrobot-v1",
]


class JumanjiTrainForm(QtWidgets.QDialog, LogConstantMixin):
    """Configuration dialog for Jumanji/gymnax training runs."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Jumanji (JAX) Training Configuration")
        self.setMinimumWidth(620)
        self.setMinimumHeight(520)

        self._setup_ui()
        self._on_backend_changed(self._backend_combo.currentText())

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # --- Backend Selection ---
        group_backend = QtWidgets.QGroupBox("JAX Backend")
        form_backend = QtWidgets.QFormLayout(group_backend)

        self._run_id_edit = QtWidgets.QLineEdit()
        self._run_id_edit.setPlaceholderText("auto-generated")
        self._run_id_edit.setReadOnly(True)
        form_backend.addRow("Run ID:", self._run_id_edit)

        self._backend_combo = QtWidgets.QComboBox()
        self._backend_combo.addItems(["gymnax (JIT-compiled)", "jumanji (native JAX)"])
        self._backend_combo.currentTextChanged.connect(self._on_backend_changed)
        self._backend_combo.setToolTip(
            "gymnax: Fully JIT-compiled env+agent loop (CartPole, Pendulum).\n"
            "jumanji: JAX environments with A2C (Game2048, TSP, etc.)."
        )
        form_backend.addRow("Environment Backend:", self._backend_combo)

        self._device_combo = QtWidgets.QComboBox()
        self._device_combo.addItems(["cpu", "gpu", "tpu"])
        self._device_combo.setToolTip("JAX device backend (XLA compiler target)")
        form_backend.addRow("Device:", self._device_combo)

        layout.addWidget(group_backend)

        # --- Environment ---
        group_env = QtWidgets.QGroupBox("Environment")
        form_env = QtWidgets.QFormLayout(group_env)

        self._env_combo = QtWidgets.QComboBox()
        form_env.addRow("Environment:", self._env_combo)

        self._seed_spin = QtWidgets.QSpinBox()
        self._seed_spin.setRange(0, 999999)
        self._seed_spin.setValue(42)
        form_env.addRow("Seed:", self._seed_spin)

        layout.addWidget(group_env)

        # --- Training Parameters ---
        self._group_params = QtWidgets.QGroupBox("Training Parameters")
        self._layout_params = QtWidgets.QFormLayout(self._group_params)

        # gymnax PPO params
        self._timesteps_spin = QtWidgets.QSpinBox()
        self._timesteps_spin.setRange(1000, 100_000_000)
        self._timesteps_spin.setValue(100_000)
        self._timesteps_spin.setSingleStep(10_000)
        self._timesteps_spin.setSuffix(" steps")
        self._layout_params.addRow("Total Timesteps:", self._timesteps_spin)

        self._lr_spin = QtWidgets.QDoubleSpinBox()
        self._lr_spin.setRange(0.00001, 1.0)
        self._lr_spin.setDecimals(5)
        self._lr_spin.setValue(2.5e-4)
        self._layout_params.addRow("Learning Rate:", self._lr_spin)

        self._num_envs_spin = QtWidgets.QSpinBox()
        self._num_envs_spin.setRange(1, 8192)
        self._num_envs_spin.setValue(4)
        self._num_envs_spin.setToolTip(
            "For gymnax: vectorized via jax.vmap (can be very large).\n"
            "For jumanji: batch size for A2C agent."
        )
        self._layout_params.addRow("Parallel Envs:", self._num_envs_spin)

        # jumanji A2C params
        self._epochs_spin = QtWidgets.QSpinBox()
        self._epochs_spin.setRange(1, 100_000)
        self._epochs_spin.setValue(100)
        self._layout_params.addRow("Num Epochs:", self._epochs_spin)

        self._n_steps_spin = QtWidgets.QSpinBox()
        self._n_steps_spin.setRange(1, 10_000)
        self._n_steps_spin.setValue(20)
        self._layout_params.addRow("Rollout Steps:", self._n_steps_spin)

        self._learner_steps_spin = QtWidgets.QSpinBox()
        self._learner_steps_spin.setRange(1, 1024)
        self._learner_steps_spin.setValue(64)
        self._layout_params.addRow("Learner Steps/Epoch:", self._learner_steps_spin)

        layout.addWidget(self._group_params)

        # --- FastLane ---
        group_video = QtWidgets.QGroupBox("Rendering & Video")
        form_video = QtWidgets.QFormLayout(group_video)
        self._fastlane_check = QtWidgets.QCheckBox("Enable FastLane (Live Rendering)")
        self._fastlane_check.setChecked(True)
        form_video.addRow(self._fastlane_check)
        layout.addWidget(group_video)

        # --- Buttons ---
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self._update_run_id()

    def _update_run_id(self):
        backend = "gymnax" if "gymnax" in self._backend_combo.currentText() else "jumanji"
        self._run_id_edit.setText(_generate_run_id("jumanji", backend))

    def _on_backend_changed(self, text: str):
        self._env_combo.clear()
        is_gymnax = "gymnax" in text

        if is_gymnax:
            self._env_combo.addItems(_GYMNAX_ENVS)
            # Show PPO params, hide A2C-specific
            self._timesteps_spin.setVisible(True)
            self._lr_spin.setVisible(True)
            self._epochs_spin.setVisible(False)
            self._n_steps_spin.setVisible(False)
            self._learner_steps_spin.setVisible(False)
            self._num_envs_spin.setToolTip(
                "Number of parallel environments via jax.vmap.\n"
                "Fully JIT-compiled: can scale to thousands."
            )
        else:
            self._env_combo.addItems(_JUMANJI_ENVS)
            # Show A2C params, hide PPO-specific
            self._timesteps_spin.setVisible(False)
            self._lr_spin.setVisible(True)
            self._epochs_spin.setVisible(True)
            self._n_steps_spin.setVisible(True)
            self._learner_steps_spin.setVisible(True)
            self._num_envs_spin.setToolTip(
                "Total batch size for A2C agent.\n"
                "Distributed across devices via pmap."
            )

        self._update_run_id()

    def get_config(self) -> JumanjiWorkerConfig:
        """Construct the worker configuration from form state."""
        is_gymnax = "gymnax" in self._backend_combo.currentText()
        env_id = self._env_combo.currentText()
        device = self._device_combo.currentText()

        extras = {
            "backend": "gymnax" if is_gymnax else "jumanji",
            "device": device,
            "num_envs": self._num_envs_spin.value(),
            "learning_rate": self._lr_spin.value(),
        }

        if is_gymnax:
            extras["total_timesteps"] = self._timesteps_spin.value()
            extras["algo"] = "ppo"
        else:
            extras["num_epochs"] = self._epochs_spin.value()
            extras["n_steps"] = self._n_steps_spin.value()
            extras["num_learner_steps_per_epoch"] = self._learner_steps_spin.value()
            extras["algo"] = "a2c"

        if self._fastlane_check.isChecked():
            extras["fastlane_enabled"] = True
            extras["video_mode"] = VideoModes.SINGLE.value

        return JumanjiWorkerConfig(
            run_id=self._run_id_edit.text(),
            env_id=env_id,
            agent="a2c" if not is_gymnax else "ppo",
            seed=self._seed_spin.value(),
            num_epochs=extras.get("num_epochs", 100),
            device=device,
            extras=extras,
        )
