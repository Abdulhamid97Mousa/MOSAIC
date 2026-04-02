"""Tests for CleanRL training form."""

from __future__ import annotations

import os
from typing import Any, Dict

import pytest
from qtpy import QtWidgets

from gym_gui.core.enums import GameId
from gym_gui.telemetry.semconv import VideoModes
from gym_gui.ui.widgets.cleanrl_train_form import CleanRlTrainForm

# Ensure Qt renders offscreen in CI environments
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qt_app():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


def _base_form(qt_app) -> CleanRlTrainForm:
    form = CleanRlTrainForm(default_game=GameId.FROZEN_LAKE_V2)
    index = form._algo_combo.findText("ppo")
    if index >= 0:
        form._algo_combo.setCurrentIndex(index)
    env_index = form._env_combo.findData("CartPole-v1")
    if env_index < 0:
        env_index = 0
    form._env_combo.setCurrentIndex(env_index)
    form._test_selected_env = form._env_combo.currentData()  # type: ignore[attr-defined]
    form._timesteps_spin.setValue(4096)
    form._seed_spin.setValue(123)
    form._agent_id_input.setText("agent-cleanrl-test")
    form._worker_id_input.setText("cleanrl-test-worker")
    form._tensorboard_checkbox.setChecked(True)
    form._track_wandb_checkbox.setChecked(False)
    form._notes_edit.setPlainText("integration-test")
    return form


def test_get_config_includes_worker_metadata(qt_app) -> None:
    form = _base_form(qt_app)
    config = form.get_config()

    assert isinstance(config, dict)
    metadata = config.get("metadata", {})
    worker_meta: Dict[str, Any] = metadata.get("worker", {})
    assert worker_meta.get("module") == "cleanrl_worker.cli"
    assert worker_meta.get("worker_id") == "cleanrl-test-worker"
    assert worker_meta.get("config", {}).get("algo") == "ppo"
    assert worker_meta.get("config", {}).get("env_id") == form._test_selected_env  # type: ignore[attr-defined]
    assert worker_meta.get("config", {}).get("seed") == 123
    # arguments key is no longer present (removed dry-run checkbox)
    assert "arguments" not in worker_meta

    extras = worker_meta.get("config", {}).get("extras", {})
    assert extras.get("cuda") is True
    assert extras.get("tensorboard_dir") == "tensorboard"
    assert extras.get("notes") == "integration-test"
    assert "algo_params" in extras

    artifacts = metadata.get("artifacts", {})
    tensorboard_meta = artifacts.get("tensorboard", {})
    assert tensorboard_meta.get("enabled") is True
    assert tensorboard_meta.get("relative_path").endswith("tensorboard")
    wandb_meta = artifacts.get("wandb", {})
    assert wandb_meta.get("enabled") is False

    form.deleteLater()


def test_wandb_fields_populate_extras_and_environment(qt_app) -> None:
    form = _base_form(qt_app)
    form._track_wandb_checkbox.setChecked(True)
    form._wandb_project_input.setText("MOSAIC")
    form._wandb_entity_input.setText("abdulhamid97mousa")
    form._wandb_run_name_input.setText("demo-run")
    form._wandb_api_key_input.setText("test-key-123")

    config = form.get_config()
    metadata = config["metadata"]
    worker_meta = metadata["worker"]
    extras = worker_meta["config"].get("extras", {})
    assert extras.get("track_wandb") is True
    assert extras.get("wandb_project_name") == "MOSAIC"
    assert extras.get("wandb_entity") == "abdulhamid97mousa"
    assert extras.get("wandb_run_name") == "demo-run"
    env_overrides = config["environment"]
    assert env_overrides.get("WANDB_API_KEY") == "test-key-123"

    form.deleteLater()


def test_disable_gpu_sets_cuda_false(qt_app) -> None:
    form = _base_form(qt_app)
    form._use_gpu_checkbox.setChecked(False)

    config = form.get_config()
    extras = config["metadata"]["worker"]["config"].get("extras", {})
    assert extras.get("cuda") is False

    form.deleteLater()


def _num_env_spin(form: CleanRlTrainForm) -> QtWidgets.QSpinBox:
    widget = form._algo_param_inputs.get("num_envs")
    assert isinstance(widget, QtWidgets.QSpinBox)
    return widget


def test_video_mode_syncs_to_single_env(qt_app) -> None:
    form = _base_form(qt_app)
    num_envs = _num_env_spin(form)
    num_envs.setValue(1)

    assert form._video_mode_combo.currentData() == VideoModes.SINGLE
    assert form._grid_limit_spin.value() == 1
    assert form._fastlane_slot_spin.value() == 0

    form.deleteLater()


def test_video_mode_syncs_to_grid_when_multi_env(qt_app) -> None:
    form = _base_form(qt_app)
    num_envs = _num_env_spin(form)
    num_envs.setValue(4)

    assert form._video_mode_combo.currentData() == VideoModes.GRID
    assert form._grid_limit_spin.value() == 4
    assert 0 <= form._fastlane_slot_spin.value() <= 3

    form.deleteLater()


# =============================================================================
# Standard Training Config Tests
# =============================================================================


def test_standard_training_produces_python_entry_point(qt_app) -> None:
    """Test that standard training produces cleanrl_worker.cli config."""
    form = _base_form(qt_app)

    config = form.get_config()

    # Should use Python as entry point with cleanrl_worker.cli
    assert config["entry_point"].endswith("python") or "python" in config["entry_point"]
    assert config["arguments"] == ["-m", "cleanrl_worker.cli"]

    # Should not have MOSAIC_CONFIG_FILE in environment
    env = config.get("environment", {})
    assert "MOSAIC_CONFIG_FILE" not in env

    form.deleteLater()


def test_standard_training_uses_module_not_script(qt_app) -> None:
    """Test that standard training mode uses metadata.worker.module."""
    form = _base_form(qt_app)

    config = form.get_config()
    metadata = config.get("metadata", {})
    worker_meta = metadata.get("worker", {})

    # In standard mode, module should be set
    assert worker_meta.get("module") == "cleanrl_worker.cli"
    # script should NOT be present
    assert "script" not in worker_meta or worker_meta.get("script") is None

    form.deleteLater()


def test_standard_training_sets_cleanrl_num_envs(qt_app) -> None:
    """Test that standard training sets CLEANRL_NUM_ENVS in environment."""
    form = _base_form(qt_app)

    config = form.get_config()
    env = config.get("environment", {})

    assert "CLEANRL_NUM_ENVS" in env, (
        "CLEANRL_NUM_ENVS SHOULD be set in standard training mode"
    )

    form.deleteLater()
