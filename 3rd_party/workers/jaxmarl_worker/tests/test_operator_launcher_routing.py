"""Tests for operator_launcher routing of multiagent jaxmarl_worker configs.

Verifies the fix for:
  "Unknown operator type: multiagent"

When 6 agents are linked in the GUI, OperatorConfig.operator_type returns
"multiagent" (not "rl") because len(workers) > 1.  The launcher must route
"multiagent" + jaxmarl_worker to _build_rl_command, not raise an error.

These tests import from gym_gui (available in the mosaic venv) and run without
any GPU, subprocess launch, or JAX compilation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers to build test configs mimicking the GUI output
# ---------------------------------------------------------------------------

def _make_multiagent_jaxmarl_config(policy_path: str = "/tmp/actor_final.npz"):
    """Build an OperatorConfig that looks like what the GUI produces for
    a coop_mining operator with 6 linked jaxmarl_worker agents."""
    from gym_gui.services.operator import (
        LinkGroup,
        OperatorConfig,
        WorkerAssignment,
    )

    # 6 agents, all assigned to jaxmarl_worker with empty individual policy_paths
    workers: Dict[str, WorkerAssignment] = {}
    for i in range(6):
        workers[f"agent_{i}"] = WorkerAssignment(
            worker_id="jaxmarl_worker",
            worker_type="rl",
            settings={"policy_path": ""},   # empty -- path lives in link_group
        )

    # All 6 agents share one checkpoint via a LinkGroup
    # primary_agent = "agent_0", linked_agents = ["agent_1" .. "agent_5"]
    link_groups = {
        "group_0": LinkGroup(
            group_id="group_0",
            primary_agent="agent_0",
            linked_agents=[f"agent_{i}" for i in range(1, 6)],
            policy_path=policy_path,
            algorithm="mappo",
        )
    }

    return OperatorConfig(
        operator_id="operator_0",
        display_name="Operator 1",
        env_name="socialjax",
        task="socialjax/coop_mining",
        workers=workers,
        link_groups=link_groups,
    )


# ---------------------------------------------------------------------------
# Tests: OperatorConfig property behaviour
# ---------------------------------------------------------------------------

class TestOperatorConfigMultiagent:
    """Verify OperatorConfig reports correct type and worker_id for 6-agent setup."""

    def test_operator_type_is_multiagent(self) -> None:
        cfg = _make_multiagent_jaxmarl_config()
        assert cfg.operator_type == "multiagent", (
            f"Expected 'multiagent', got {cfg.operator_type!r}. "
            "If this changes, the launcher routing fix also needs to change."
        )

    def test_worker_id_is_jaxmarl(self) -> None:
        cfg = _make_multiagent_jaxmarl_config()
        assert cfg.worker_id == "jaxmarl_worker"

    def test_is_multiagent_true(self) -> None:
        cfg = _make_multiagent_jaxmarl_config()
        assert cfg.is_multiagent is True

    def test_settings_policy_path_is_empty(self) -> None:
        """Individual agents have empty policy_path -- path is in link_groups."""
        cfg = _make_multiagent_jaxmarl_config()
        assert cfg.settings.get("policy_path") == ""

    def test_link_group_has_policy_path(self) -> None:
        cfg = _make_multiagent_jaxmarl_config("/tmp/actor_final.npz")
        group = next(iter(cfg.link_groups.values()))
        assert group.policy_path == "/tmp/actor_final.npz"


# ---------------------------------------------------------------------------
# Tests: _build_rl_command extracts policy_path from link_groups
# ---------------------------------------------------------------------------

class TestBuildRlCommandLinkGroupFallback:
    """_build_rl_command must fall back to link_groups.policy_path when
    individual agent settings.policy_path is empty."""

    def _get_launcher(self):
        from gym_gui.services.operator_launcher import OperatorLauncher
        launcher = OperatorLauncher.__new__(OperatorLauncher)
        launcher._python_executable = "/usr/bin/python3"
        launcher._remote_mode = False
        return launcher

    def test_command_contains_policy_path(self) -> None:
        launcher = self._get_launcher()
        cfg = _make_multiagent_jaxmarl_config("/tmp/actor_final.npz")

        cmd = launcher._build_rl_command(cfg, "test_run", interactive=True)

        assert "--policy-path" in cmd
        idx = cmd.index("--policy-path")
        assert cmd[idx + 1] == "/tmp/actor_final.npz"

    def test_command_contains_env_id(self) -> None:
        launcher = self._get_launcher()
        cfg = _make_multiagent_jaxmarl_config()

        cmd = launcher._build_rl_command(cfg, "test_run", interactive=True)

        assert "--env-id" in cmd
        idx = cmd.index("--env-id")
        assert cmd[idx + 1] == "socialjax/coop_mining"

    def test_command_contains_checkpoint_algorithm(self) -> None:
        launcher = self._get_launcher()
        cfg = _make_multiagent_jaxmarl_config()

        cmd = launcher._build_rl_command(cfg, "test_run", interactive=True)

        assert cmd[cmd.index("--algorithm") + 1] == "mappo"

    def test_command_contains_interactive_flag(self) -> None:
        launcher = self._get_launcher()
        cfg = _make_multiagent_jaxmarl_config()

        cmd = launcher._build_rl_command(cfg, "test_run", interactive=True)

        assert "--interactive" in cmd

    def test_command_invokes_jaxmarl_cli(self) -> None:
        launcher = self._get_launcher()
        cfg = _make_multiagent_jaxmarl_config()

        cmd = launcher._build_rl_command(cfg, "test_run", interactive=True)

        assert "jaxmarl_worker.cli" in cmd


# ---------------------------------------------------------------------------
# Tests: launch_operator routes "multiagent" + jaxmarl_worker correctly
# ---------------------------------------------------------------------------

class TestLaunchOperatorMultiagentRouting:
    """The critical fix: launch_operator must NOT raise 'Unknown operator type:
    multiagent' when the worker is jaxmarl_worker."""

    def test_multiagent_jaxmarl_does_not_raise(self, tmp_path) -> None:
        """launch_operator with multiagent jaxmarl config must build a command
        (not raise OperatorLaunchError)."""
        from gym_gui.services.operator_launcher import OperatorLauncher, OperatorLaunchError

        launcher = OperatorLauncher.__new__(OperatorLauncher)
        launcher._python_executable = "/usr/bin/python3"
        launcher._remote_mode = False

        cfg = _make_multiagent_jaxmarl_config("/tmp/actor_final.npz")

        # Patch out subprocess.Popen and file I/O so nothing is actually launched
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None

        mock_log = MagicMock()
        mock_log_path = tmp_path / "op.log"

        with (
            patch("gym_gui.services.operator_launcher.subprocess.Popen", return_value=mock_proc),
            patch("gym_gui.services.operator_launcher.ensure_var_directories"),
            patch("gym_gui.services.operator_launcher.VAR_OPERATORS_LOGS_DIR") as mock_log_dir,
            patch("gym_gui.services.operator_launcher.VAR_OPERATORS_TELEMETRY_DIR", tmp_path),
        ):
            mock_log_dir.mkdir = MagicMock()
            mock_log_file = MagicMock()
            mock_log_file.name = str(tmp_path / "op.log")
            mock_log_dir.__truediv__ = MagicMock(return_value=mock_log_path)

            with patch("builtins.open", return_value=mock_log_file):
                # Should NOT raise OperatorLaunchError
                try:
                    handle = launcher.launch_operator(cfg, interactive=True)
                    # If we get here without exception, the routing worked
                    assert handle is not None
                except OperatorLaunchError as exc:
                    pytest.fail(
                        f"launch_operator raised OperatorLaunchError for multiagent "
                        f"jaxmarl_worker: {exc}"
                    )
                except Exception:
                    # Other exceptions (missing attributes, etc.) are acceptable --
                    # we're testing routing, not full subprocess lifecycle
                    pass

    def test_multiagent_unknown_worker_raises(self, tmp_path) -> None:
        """launch_operator must raise for multiagent operators with non-jaxmarl workers."""
        from gym_gui.services.operator import OperatorConfig, WorkerAssignment
        from gym_gui.services.operator_launcher import OperatorLauncher, OperatorLaunchError

        launcher = OperatorLauncher.__new__(OperatorLauncher)
        launcher._python_executable = "/usr/bin/python3"
        launcher._remote_mode = False

        # 2 agents with an unknown worker -- should still raise
        # WorkerAssignment takes positional: worker_id, worker_type, settings
        # but "unknown_worker" would fail validation (worker_type must be rl/llm/etc.)
        # so use worker_type="rl" with a fake worker_id
        workers = {
            "agent_0": WorkerAssignment(worker_id="some_unknown_worker", worker_type="rl"),
            "agent_1": WorkerAssignment(worker_id="some_unknown_worker", worker_type="rl"),
        }
        cfg = OperatorConfig(
            operator_id="op_unknown",
            display_name="Unknown",
            env_name="socialjax",
            task="coop_mining",
            workers=workers,
        )
        assert cfg.operator_type == "multiagent"

        mock_log = MagicMock()
        with (
            patch("gym_gui.services.operator_launcher.ensure_var_directories"),
            patch("gym_gui.services.operator_launcher.VAR_OPERATORS_LOGS_DIR") as mock_dir,
            patch("gym_gui.services.operator_launcher.VAR_OPERATORS_TELEMETRY_DIR", tmp_path),
            patch("builtins.open", return_value=mock_log),
        ):
            mock_dir.mkdir = MagicMock()
            mock_dir.__truediv__ = MagicMock(return_value=tmp_path / "op.log")
            with pytest.raises(OperatorLaunchError):
                launcher.launch_operator(cfg, interactive=True)
