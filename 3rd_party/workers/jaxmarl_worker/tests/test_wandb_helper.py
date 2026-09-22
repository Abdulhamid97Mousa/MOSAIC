"""Tests for jaxmarl_worker.wandb_helper.

The helper must:
  - Be a safe no-op when JAXMARL_WANDB_ENABLED != '1' (Gap 12 acceptance)
  - Be a safe no-op when wandb package is not installed
  - Correctly read env vars when enabled and wandb is installed
  - Never raise on log/finish calls if init was skipped
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_wandb_state():
    """Reset the module-level singleton before and after each test."""
    from jaxmarl_worker import wandb_helper
    wandb_helper._reset_for_tests()
    yield
    wandb_helper._reset_for_tests()


@pytest.fixture
def _clean_env(monkeypatch):
    """Clear all JAXMARL_WANDB_* env vars for isolation."""
    for key in list(os.environ):
        if key.startswith("JAXMARL_WANDB_") or key in ("WANDB_API_KEY",):
            monkeypatch.delenv(key, raising=False)
    yield monkeypatch


class TestIsWandbEnabled:

    def test_returns_false_when_env_flag_missing(self, _clean_env):
        from jaxmarl_worker.wandb_helper import is_wandb_enabled
        assert is_wandb_enabled() is False

    def test_returns_false_when_env_flag_zero(self, _clean_env):
        _clean_env.setenv("JAXMARL_WANDB_ENABLED", "0")
        from jaxmarl_worker.wandb_helper import is_wandb_enabled
        assert is_wandb_enabled() is False

    def test_returns_false_when_env_flag_other_truthy(self, _clean_env):
        """Only '1' should enable, not 'true' or 'yes' — strict opt-in."""
        _clean_env.setenv("JAXMARL_WANDB_ENABLED", "true")
        from jaxmarl_worker.wandb_helper import is_wandb_enabled
        assert is_wandb_enabled() is False

    def test_returns_false_when_wandb_not_importable(self, _clean_env):
        _clean_env.setenv("JAXMARL_WANDB_ENABLED", "1")
        with patch.dict(sys.modules, {"wandb": None}):
            from jaxmarl_worker.wandb_helper import is_wandb_enabled
            assert is_wandb_enabled() is False

    def test_returns_true_when_enabled_and_wandb_importable(self, _clean_env):
        _clean_env.setenv("JAXMARL_WANDB_ENABLED", "1")
        with patch.dict(sys.modules, {"wandb": MagicMock()}):
            from jaxmarl_worker.wandb_helper import is_wandb_enabled
            assert is_wandb_enabled() is True


class TestInitWandbDisabled:
    """When wandb is disabled or unavailable, init_wandb must be a safe no-op."""

    def test_init_returns_false_when_env_flag_missing(self, _clean_env):
        from jaxmarl_worker.wandb_helper import init_wandb
        assert init_wandb(run_id="test") is False

    def test_init_returns_false_when_wandb_not_installed(self, _clean_env):
        _clean_env.setenv("JAXMARL_WANDB_ENABLED", "1")
        with patch.dict(sys.modules, {"wandb": None}):
            from jaxmarl_worker.wandb_helper import init_wandb
            assert init_wandb(run_id="test") is False

    def test_init_does_not_raise_on_missing_config(self, _clean_env):
        from jaxmarl_worker.wandb_helper import init_wandb
        init_wandb(run_id="test", config=None)  # should not raise


class TestInitWandbEnabled:
    """When enabled, init_wandb should call wandb.init with the right args."""

    def test_reads_env_vars_and_calls_wandb_init(self, _clean_env):
        _clean_env.setenv("JAXMARL_WANDB_ENABLED", "1")
        _clean_env.setenv("JAXMARL_WANDB_PROJECT", "my-project")
        _clean_env.setenv("JAXMARL_WANDB_ENTITY", "my-team")
        _clean_env.setenv("JAXMARL_WANDB_RUN_NAME", "custom-run-name")

        mock_wandb = MagicMock()
        mock_wandb.init.return_value = MagicMock()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            from jaxmarl_worker.wandb_helper import init_wandb
            ok = init_wandb(run_id="fallback-run-id", config={"lr": 3e-4}, tags=["mappo"])

        assert ok is True
        mock_wandb.init.assert_called_once()
        _, kwargs = mock_wandb.init.call_args
        assert kwargs["project"] == "my-project"
        assert kwargs["entity"] == "my-team"
        assert kwargs["name"] == "custom-run-name"
        assert kwargs["config"] == {"lr": 3e-4}
        assert kwargs["tags"] == ["mappo"]

    def test_defaults_when_env_vars_absent(self, _clean_env):
        _clean_env.setenv("JAXMARL_WANDB_ENABLED", "1")
        # No PROJECT/ENTITY/RUN_NAME set

        mock_wandb = MagicMock()
        mock_wandb.init.return_value = MagicMock()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            from jaxmarl_worker.wandb_helper import init_wandb
            init_wandb(run_id="my-run-id")

        _, kwargs = mock_wandb.init.call_args
        assert kwargs["project"] == "jaxmarl"       # default project
        assert kwargs["entity"] is None              # no entity
        assert kwargs["name"] == "my-run-id"         # falls back to run_id


class TestLogWandb:

    def test_log_is_noop_when_not_initialized(self, _clean_env):
        # No init called → _wandb_run is None → log should silently return
        from jaxmarl_worker.wandb_helper import log_wandb
        log_wandb({"reward": 0.5}, step=10)  # must not raise

    def test_log_forwards_metrics_after_init(self, _clean_env):
        _clean_env.setenv("JAXMARL_WANDB_ENABLED", "1")
        mock_wandb = MagicMock()
        mock_wandb.init.return_value = MagicMock()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            from jaxmarl_worker.wandb_helper import init_wandb, log_wandb
            init_wandb(run_id="r")
            log_wandb({"reward": 1.5}, step=42)

        mock_wandb.log.assert_called_once_with({"reward": 1.5}, step=42)

    def test_log_without_step(self, _clean_env):
        _clean_env.setenv("JAXMARL_WANDB_ENABLED", "1")
        mock_wandb = MagicMock()
        mock_wandb.init.return_value = MagicMock()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            from jaxmarl_worker.wandb_helper import init_wandb, log_wandb
            init_wandb(run_id="r")
            log_wandb({"loss": 0.1})

        mock_wandb.log.assert_called_once_with({"loss": 0.1})


class TestFinishWandb:

    def test_finish_is_noop_when_not_initialized(self, _clean_env):
        from jaxmarl_worker.wandb_helper import finish_wandb
        finish_wandb()  # must not raise

    def test_finish_calls_wandb_finish_after_init(self, _clean_env):
        _clean_env.setenv("JAXMARL_WANDB_ENABLED", "1")
        mock_wandb = MagicMock()
        mock_wandb.init.return_value = MagicMock()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            from jaxmarl_worker.wandb_helper import init_wandb, finish_wandb
            init_wandb(run_id="r")
            finish_wandb()

        mock_wandb.finish.assert_called_once()

    def test_finish_resets_state_so_subsequent_log_is_noop(self, _clean_env):
        _clean_env.setenv("JAXMARL_WANDB_ENABLED", "1")
        mock_wandb = MagicMock()
        mock_wandb.init.return_value = MagicMock()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            from jaxmarl_worker.wandb_helper import init_wandb, log_wandb, finish_wandb
            init_wandb(run_id="r")
            finish_wandb()
            log_wandb({"reward": 2.0})  # should NOT call wandb.log again

        mock_wandb.log.assert_not_called()


class TestFormToEnvVarPlumbing:
    """Verify the jaxmarl_train_form config-builder plumbs wandb settings into
    the trainer 'environment' dict correctly. This is the frontend↔backend
    contract for Gap 12."""

    def _import_form(self):
        # Instantiate offscreen so QApplication exists for widget construction.
        import os as _os
        _os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from qtpy import QtWidgets
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        from gym_gui.ui.widgets.jaxmarl_train_form import JaxMARLTrainForm
        return JaxMARLTrainForm(), app

    def test_wandb_env_vars_not_set_when_disabled(self):
        form, _app = self._import_form()
        form._wandb_enable_check.setChecked(False)
        cfg = form.get_config()
        assert cfg is not None
        env = cfg.get("environment", {})
        assert "JAXMARL_WANDB_ENABLED" not in env
        assert "WANDB_API_KEY" not in env

    def test_wandb_env_vars_set_when_enabled(self):
        form, _app = self._import_form()
        form._wandb_enable_check.setChecked(True)
        form._wandb_project_edit.setText("test-proj")
        form._wandb_entity_edit.setText("test-team")
        form._wandb_run_label_edit.setText("custom-name")
        form._wandb_api_key_edit.setText("secret-key")
        form._wandb_http_proxy_edit.setText("http://proxy:7890")
        form._wandb_https_proxy_edit.setText("http://proxy:7890")

        cfg = form.get_config()
        assert cfg is not None
        env = cfg.get("environment", {})
        assert env.get("JAXMARL_WANDB_ENABLED") == "1"
        assert env.get("JAXMARL_WANDB_PROJECT") == "test-proj"
        assert env.get("JAXMARL_WANDB_ENTITY") == "test-team"
        assert env.get("JAXMARL_WANDB_RUN_NAME") == "custom-name"
        assert env.get("WANDB_API_KEY") == "secret-key"
        assert env.get("HTTP_PROXY") == "http://proxy:7890"
        assert env.get("HTTPS_PROXY") == "http://proxy:7890"

    def test_wandb_optional_fields_omitted_when_empty(self):
        """Only fields the user actually filled in should end up in env dict."""
        form, _app = self._import_form()
        form._wandb_enable_check.setChecked(True)
        form._wandb_project_edit.setText("only-project")
        # entity, run_name, api_key, proxies all left blank

        cfg = form.get_config()
        env = cfg.get("environment", {})
        assert env.get("JAXMARL_WANDB_ENABLED") == "1"
        assert env.get("JAXMARL_WANDB_PROJECT") == "only-project"
        assert "JAXMARL_WANDB_ENTITY" not in env
        assert "JAXMARL_WANDB_RUN_NAME" not in env
        assert "WANDB_API_KEY" not in env
        assert "HTTP_PROXY" not in env
        assert "HTTPS_PROXY" not in env
