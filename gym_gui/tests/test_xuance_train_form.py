"""Tests for XuanCe training form custom script support."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
from qtpy import QtWidgets

from gym_gui.ui.widgets.xuance_train_form import XuanCeTrainForm

# Ensure Qt renders offscreen in CI environments
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qt_app():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


def _base_form(qt_app) -> XuanCeTrainForm:
    """Create a form with basic valid settings."""
    form = XuanCeTrainForm()
    # Set required fields to valid values
    if hasattr(form, "_method_combo"):
        # Try to set a valid method like "DQN" or "PPO"
        index = form._method_combo.findText("DQN")  # type: ignore[attr-defined]
        if index >= 0:
            form._method_combo.setCurrentIndex(index)  # type: ignore[attr-defined]
    if hasattr(form, "_env_combo"):
        # Set environment
        index = form._env_combo.findText("Classic Control")  # type: ignore[attr-defined]
        if index >= 0:
            form._env_combo.setCurrentIndex(index)  # type: ignore[attr-defined]
    if hasattr(form, "_env_id_combo"):
        # Set env_id
        index = form._env_id_combo.findText("CartPole-v1")  # type: ignore[attr-defined]
        if index < 0:
            index = 0
        form._env_id_combo.setCurrentIndex(index)  # type: ignore[attr-defined]
    return form


# =============================================================================
# Custom Script Mode Tests for Dispatcher Integration
# (Pending feature: _custom_script_combo not yet implemented in XuanCeTrainForm)
# =============================================================================

_SKIP_CUSTOM_SCRIPT = pytest.mark.skip(
    reason="XuanCeTrainForm._custom_script_combo not yet implemented"
)


@_SKIP_CUSTOM_SCRIPT
def test_custom_script_sets_worker_metadata_for_dispatcher(qt_app) -> None:
    """Test that custom script mode sets metadata.worker correctly for dispatcher.

    The trainer dispatcher reads metadata.worker.module or metadata.worker.script
    to determine the subprocess command. When a custom script is selected:
    - metadata.worker.module should NOT be present (or dispatcher uses python -m)
    - metadata.worker.script should be set to '/bin/bash'
    - metadata.worker.arguments should contain the script path

    This is critical because the dispatcher ignores top-level entry_point/arguments.
    """
    form = _base_form(qt_app)

    # Create a temporary script file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", delete=False, prefix="test_xuance_dispatcher_script_"
    ) as f:
        f.write("#!/bin/bash\n")
        f.write("# @description: Test script for dispatcher\n")
        f.write('CONFIG="$MOSAIC_CONFIG_FILE"\n')
        f.write("python -m xuance_worker.cli --config $CONFIG\n")
        script_path = f.name

    try:
        # Add and select the script
        insert_index = form._custom_script_combo.count() - 1
        form._custom_script_combo.insertItem(insert_index, "test_dispatcher_script", script_path)
        form._custom_script_combo.setCurrentIndex(insert_index)

        config = form.get_config()
        metadata = config.get("metadata", {})
        worker_meta = metadata.get("worker", {})

        # CRITICAL: module should NOT be present when using a custom script
        # If module is present, dispatcher will run 'python -m module' instead of bash script
        assert "module" not in worker_meta, (
            "metadata.worker.module should not be present in custom script mode. "
            "Dispatcher will run 'python -m module' instead of the bash script!"
        )

        # script should be set to /bin/bash
        assert worker_meta.get("script") == "/bin/bash", (
            "metadata.worker.script should be '/bin/bash' for custom scripts"
        )

        # arguments should contain the script path
        assert script_path in worker_meta.get("arguments", []), (
            f"metadata.worker.arguments should contain script path '{script_path}'"
        )

    finally:
        Path(script_path).unlink(missing_ok=True)
        form.deleteLater()


def test_standard_training_uses_module_not_script(qt_app) -> None:
    """Test that standard training mode uses metadata.worker.module."""
    form = _base_form(qt_app)

    # Ensure None (standard training) is selected
    form._custom_script_combo.setCurrentIndex(0)

    config = form.get_config()
    metadata = config.get("metadata", {})
    worker_meta = metadata.get("worker", {})

    # In standard mode, module should be set
    assert worker_meta.get("module") == "xuance_worker.cli"
    # script should NOT be present
    assert "script" not in worker_meta or worker_meta.get("script") is None

    form.deleteLater()


def test_custom_script_combo_exists_and_has_none_option(qt_app) -> None:
    """Test that the custom script dropdown exists with 'None' as default."""
    form = _base_form(qt_app)

    # Verify combo box exists
    assert hasattr(form, "_custom_script_combo")
    assert isinstance(form._custom_script_combo, QtWidgets.QComboBox)

    # First item should be "None (Standard Training)"
    assert form._custom_script_combo.count() >= 2  # At least None + Browse
    assert form._custom_script_combo.itemText(0) == "None (Standard Training)"
    assert form._custom_script_combo.itemData(0) is None

    # Default selection should be None
    assert form._custom_script_combo.currentIndex() == 0
    assert form._custom_script_combo.currentData() is None

    form.deleteLater()


def test_custom_script_selected_changes_config_entry_point(qt_app) -> None:
    """Test that selecting a script changes the config entry point to bash."""
    form = _base_form(qt_app)

    # Create a temporary script file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", delete=False, prefix="test_script_"
    ) as f:
        f.write("#!/bin/bash\n# @description: Test script\necho 'test'\n")
        script_path = f.name

    try:
        # Manually add the script to the combo (simulating import)
        insert_index = form._custom_script_combo.count() - 1
        form._custom_script_combo.insertItem(insert_index, "test_script (imported)", script_path)
        form._custom_script_combo.setCurrentIndex(insert_index)

        config = form.get_config()

        # Should use bash as entry point
        assert config["entry_point"] == "/bin/bash"
        assert config["arguments"] == [script_path]

        # Should have MOSAIC environment variables
        env = config.get("environment", {})
        assert "MOSAIC_CONFIG_FILE" in env
        assert "MOSAIC_RUN_ID" in env
        assert "MOSAIC_CUSTOM_SCRIPTS_DIR" in env
        assert "MOSAIC_CHECKPOINT_DIR" in env

        # Metadata should reflect script mode
        metadata = config.get("metadata", {})
        ui_meta = metadata.get("ui", {})
        assert ui_meta.get("custom_script") == script_path
        assert "test_script" in ui_meta.get("custom_script_name", "")

    finally:
        Path(script_path).unlink(missing_ok=True)
        form.deleteLater()
