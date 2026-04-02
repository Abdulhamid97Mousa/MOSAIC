"""Integration tests for Ray RLlib worker with MOSAIC GUI.

These tests verify:
1. Train form registration and creation
2. Policy form registration and creation
3. Worker catalog integration
4. Worker presenter integration
5. PettingZoo adapter SISL support
6. Configuration building from forms
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# Skip all tests if Ray is not available
pytest.importorskip("ray")


class TestRayWorkerFormRegistration:
    """Tests for Ray worker form registration with WorkerFormFactory."""

    def test_train_form_registered(self):
        """Test that ray_worker train form is registered."""
        from gym_gui.ui.forms import ensure_all_forms_registered, get_worker_form_factory

        ensure_all_forms_registered()
        factory = get_worker_form_factory()
        assert factory.has_train_form("ray_worker")

    def test_policy_form_registered(self):
        """Test that ray_worker policy form is registered."""
        from gym_gui.ui.forms import ensure_all_forms_registered, get_worker_form_factory

        ensure_all_forms_registered()
        factory = get_worker_form_factory()
        assert factory.has_policy_form("ray_worker")

    def test_train_form_can_be_created(self):
        """Test that train form can be instantiated (headless)."""
        pytest.importorskip("qtpy")

        # Skip if no display available
        import os
        if not os.environ.get("DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
            os.environ["QT_QPA_PLATFORM"] = "offscreen"

        from qtpy import QtWidgets

        # Create app if needed
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        from gym_gui.ui.forms import ensure_all_forms_registered, get_worker_form_factory

        ensure_all_forms_registered()
        factory = get_worker_form_factory()
        form = factory.create_train_form("ray_worker", parent=None)

        assert form is not None
        assert hasattr(form, "get_config")
        form.close()

    def test_policy_form_can_be_created(self):
        """Test that policy form can be instantiated (headless)."""
        pytest.importorskip("qtpy")

        import os
        if not os.environ.get("DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
            os.environ["QT_QPA_PLATFORM"] = "offscreen"

        from qtpy import QtWidgets

        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        from gym_gui.ui.forms import ensure_all_forms_registered, get_worker_form_factory

        ensure_all_forms_registered()
        factory = get_worker_form_factory()
        form = factory.create_policy_form("ray_worker", parent=None)

        assert form is not None
        assert hasattr(form, "get_config")
        form.close()


class TestRayWorkerCatalog:
    """Tests for Ray worker in worker catalog."""

    def test_ray_worker_in_catalog(self):
        """Test that ray_worker is listed in worker catalog."""
        from gym_gui.ui.worker_catalog import get_worker_catalog

        catalog = get_worker_catalog()
        worker_ids = [w.worker_id for w in catalog]

        assert "ray_worker" in worker_ids

    def test_ray_worker_definition(self):
        """Test ray_worker definition properties."""
        from gym_gui.ui.worker_catalog import get_worker_catalog

        catalog = get_worker_catalog()
        ray_worker = next((w for w in catalog if w.worker_id == "ray_worker"), None)

        assert ray_worker is not None
        assert ray_worker.display_name == "Ray RLlib Worker"
        assert ray_worker.supports_training is True
        assert ray_worker.supports_policy_load is True


class TestRayWorkerPresenter:
    """Tests for Ray worker presenter."""

    def test_presenter_registered(self):
        """Test that ray_worker presenter is registered."""
        from gym_gui.ui.presenters.workers import get_worker_presenter_registry

        registry = get_worker_presenter_registry()
        presenter = registry.get("ray_worker")

        assert presenter is not None
        assert presenter.id == "ray_worker"

    def test_presenter_paradigms(self):
        """Test presenter returns available paradigms."""
        from gym_gui.ui.presenters.workers import get_worker_presenter_registry

        registry = get_worker_presenter_registry()
        presenter = registry.get("ray_worker")
        assert presenter is not None

        paradigms = presenter.get_available_paradigms()  # type: ignore[attr-defined]
        assert len(paradigms) == 4

        paradigm_ids = [p["id"] for p in paradigms]
        assert "parameter_sharing" in paradigm_ids
        assert "independent" in paradigm_ids
        assert "self_play" in paradigm_ids
        assert "shared_value" in paradigm_ids

    def test_presenter_build_train_config(self):
        """Test presenter builds training configuration."""
        from gym_gui.ui.presenters.workers import get_worker_presenter_registry

        registry = get_worker_presenter_registry()
        presenter = registry.get("ray_worker")
        assert presenter is not None

        config = presenter.build_train_config(  # type: ignore[attr-defined]
            env_id="waterworld_v4",
            env_family="sisl",
            paradigm="parameter_sharing",
            total_timesteps=10000,
        )

        assert config is not None
        assert config["run_id"].startswith("ray_")
        assert config["environment"]["env_id"] == "waterworld_v4"
        assert config["environment"]["family"] == "sisl"
        assert config["paradigm"] == "parameter_sharing"


class TestPettingZooAdapterSISL:
    """Tests for PettingZoo adapter SISL support."""

    def test_sisl_enums_defined(self):
        """Test SISL environments are defined in enums."""
        from gym_gui.core.pettingzoo_enums import PettingZooEnvId, PettingZooFamily

        # Test family
        assert PettingZooFamily.SISL.value == "sisl"

        # Test envs
        assert PettingZooEnvId.WATERWORLD.value == "waterworld_v4"
        assert PettingZooEnvId.MULTIWALKER.value == "multiwalker_v9"
        assert PettingZooEnvId.PURSUIT.value == "pursuit_v4"

    def test_sisl_envs_in_metadata(self):
        """Test SISL environments have metadata."""
        from gym_gui.core.pettingzoo_enums import (
            PETTINGZOO_ENV_METADATA,
            PettingZooEnvId,
            PettingZooFamily,
        )

        for env_id in [
            PettingZooEnvId.WATERWORLD,
            PettingZooEnvId.MULTIWALKER,
            PettingZooEnvId.PURSUIT,
        ]:
            assert env_id in PETTINGZOO_ENV_METADATA
            metadata = PETTINGZOO_ENV_METADATA[env_id]
            assert metadata[0] == PettingZooFamily.SISL

    def test_get_envs_by_sisl_family(self):
        """Test getting SISL environments by family."""
        from gym_gui.core.pettingzoo_enums import (
            PettingZooEnvId,
            PettingZooFamily,
            get_envs_by_family,
        )

        sisl_envs = get_envs_by_family(PettingZooFamily.SISL)

        assert PettingZooEnvId.WATERWORLD in sisl_envs
        assert PettingZooEnvId.MULTIWALKER in sisl_envs
        assert PettingZooEnvId.PURSUIT in sisl_envs


class TestRayWorkerConfigIntegration:
    """Tests for ray_worker config integration with gym_gui paths."""

    def test_config_uses_var_trainer_dir(self):
        """Test config resolves to var/trainer directory."""
        from ray_worker.config import EnvironmentConfig, RayWorkerConfig

        config = RayWorkerConfig(
            run_id="test_integration_001",
            environment=EnvironmentConfig(
                family="sisl",
                env_id="waterworld_v4",
            ),
        )

        run_dir = str(config.run_dir)
        assert "var/trainer/runs" in run_dir or "var\\trainer\\runs" in run_dir

    def test_config_tensorboard_path(self):
        """Test tensorboard path is correctly resolved."""
        from ray_worker.config import EnvironmentConfig, RayWorkerConfig

        config = RayWorkerConfig(
            run_id="test_tb_001",
            environment=EnvironmentConfig(
                family="sisl",
                env_id="waterworld_v4",
            ),
            tensorboard=True,
        )

        tb_dir = config.tensorboard_log_dir
        assert tb_dir is not None
        assert "tensorboard" in str(tb_dir)


class TestLogConstants:
    """Tests for Ray worker log constants."""

    def test_ray_log_constants_exist(self):
        """Test Ray worker log constants are defined."""
        from gym_gui.logging_config.log_constants import (
            LOG_RAY_WORKER_CHECKPOINT_SAVED,
            LOG_RAY_WORKER_RUNTIME_STARTED,
            LOG_RAY_WORKER_RUNTIME_STOPPED,
            LOG_RAY_WORKER_TRAINING_COMPLETED,
            LOG_RAY_WORKER_TRAINING_STARTED,
        )

        assert LOG_RAY_WORKER_RUNTIME_STARTED.code == "LOG970"
        assert LOG_RAY_WORKER_RUNTIME_STOPPED.code == "LOG971"
        assert LOG_RAY_WORKER_TRAINING_STARTED.code == "LOG973"
        assert LOG_RAY_WORKER_TRAINING_COMPLETED.code == "LOG974"
        assert LOG_RAY_WORKER_CHECKPOINT_SAVED.code == "LOG975"


class TestTrainFormConfiguration:
    """Tests for train form configuration building."""

    def test_form_builds_valid_config(self):
        """Test that form builds a valid configuration dict."""
        pytest.importorskip("qtpy")

        import os
        if not os.environ.get("DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
            os.environ["QT_QPA_PLATFORM"] = "offscreen"

        from qtpy import QtWidgets

        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        from gym_gui.ui.widgets.ray_train_form import RayRLlibTrainForm

        form = RayRLlibTrainForm(parent=None)

        # Build config without accepting dialog
        config = form._build_config()

        # Verify structure - run_name is at top level, run_id is in worker config
        assert "run_name" in config
        assert "metadata" in config
        assert "worker" in config["metadata"]
        assert "config" in config["metadata"]["worker"]

        worker_config = config["metadata"]["worker"]["config"]
        assert "environment" in worker_config
        assert "paradigm" in worker_config
        assert "training" in worker_config

        form.close()


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
