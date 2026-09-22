"""Tests for jaxmarl_worker CLI argument parsing and dispatch.

Tests cover:
  - parse_args: required flags, defaults, values
  - main(): --interactive dispatches to InteractiveRuntime
  - main() strips --interactive before passing remaining args to runtime
  - _resolve_scan_module: sport/algo → module-path routing (Gap 11)
  - _run_from_config: dispatches to correct per-sport scan module (Gap 11)
  - _dry_run_from_config: validates config without launching training (Gap 7)
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestParseArgs:

    def test_requires_env_id(self) -> None:
        from jaxmarl_worker.runtime import parse_args
        with pytest.raises(SystemExit):
            parse_args([])  # missing --env-id and --policy-path

    def test_requires_policy_path(self) -> None:
        from jaxmarl_worker.runtime import parse_args
        with pytest.raises(SystemExit):
            parse_args(["--env-id", "coop_mining"])  # missing --policy-path

    def test_parse_minimum_required(self) -> None:
        from jaxmarl_worker.runtime import parse_args
        args = parse_args(["--env-id", "coop_mining", "--policy-path", "/tmp/fake.npz"])
        assert args.env_id == "coop_mining"
        assert args.policy_path == "/tmp/fake.npz"

    def test_parse_view_size(self) -> None:
        from jaxmarl_worker.runtime import parse_args
        args = parse_args([
            "--env-id", "coop_mining",
            "--policy-path", "/tmp/fake.npz",
            "--view-size", "5",
        ])
        assert args.view_size == 5

    def test_default_view_size(self) -> None:
        from jaxmarl_worker.runtime import parse_args
        args = parse_args(["--env-id", "coop_mining", "--policy-path", "/tmp/fake.npz"])
        assert args.view_size == 7

    def test_socialjax_prefixed_env_id(self) -> None:
        from jaxmarl_worker.runtime import parse_args
        args = parse_args([
            "--env-id", "socialjax/coop_mining",
            "--policy-path", "/tmp/fake.npz",
        ])
        assert args.env_id == "socialjax/coop_mining"


class TestCliDispatch:

    def test_interactive_flag_routes_to_runtime(self) -> None:
        """--interactive in sys.argv must call runtime_main(), not mappo_main()."""
        import sys
        import jaxmarl_worker.cli as cli_mod

        original_argv = sys.argv[:]
        mock_runtime_main = MagicMock()
        try:
            sys.argv = [
                "jaxmarl_worker.cli",
                "--interactive",
                "--env-id", "coop_mining",
                "--policy-path", "/tmp/fake.npz",
            ]
            with patch("jaxmarl_worker.runtime.main", mock_runtime_main):
                cli_mod.main()
        finally:
            sys.argv = original_argv

        mock_runtime_main.assert_called_once()

    def test_interactive_stripped_from_argv_before_runtime(self) -> None:
        """Runtime's parse_args must NOT see '--interactive' in argv."""
        import sys
        import jaxmarl_worker.cli as cli_mod

        received_argv = []

        def capture_runtime():
            received_argv.extend(sys.argv[:])

        original_argv = sys.argv[:]
        try:
            sys.argv = [
                "jaxmarl_worker.cli",
                "--interactive",
                "--env-id", "coop_mining",
                "--policy-path", "/tmp/fake.npz",
            ]
            with patch("jaxmarl_worker.runtime.main", side_effect=capture_runtime):
                cli_mod.main()
        finally:
            sys.argv = original_argv

        assert "--interactive" not in received_argv


# ---------------------------------------------------------------------------
# Gap 11: Sport/algo → scan module resolution
# ---------------------------------------------------------------------------

class TestResolveScanModule:
    """Verify _resolve_scan_module returns the correct per-sport module path."""

    @pytest.mark.parametrize("env_family,expected_dir", [
        ("soccer", "S"),
        ("af",     "AF"),
        ("bb",     "BB"),
    ])
    def test_mappo_routes_per_sport(self, env_family, expected_dir):
        from jaxmarl_worker.cli import _resolve_scan_module
        got = _resolve_scan_module("mappo", env_family)
        assert got == f"jaxmarl_worker.algorithms.mosaic_multigrid.{expected_dir}.mappo_scan"

    @pytest.mark.parametrize("env_family,expected_dir", [
        ("soccer", "S"),
        ("af",     "AF"),
        ("bb",     "BB"),
    ])
    def test_ippo_routes_per_sport(self, env_family, expected_dir):
        from jaxmarl_worker.cli import _resolve_scan_module
        got = _resolve_scan_module("ippo", env_family)
        assert got == f"jaxmarl_worker.algorithms.mosaic_multigrid.{expected_dir}.ippo_scan"

    def test_case_insensitive_algo(self):
        from jaxmarl_worker.cli import _resolve_scan_module
        assert _resolve_scan_module("MAPPO", "soccer").endswith(".mappo_scan")
        assert _resolve_scan_module("IPPO",  "bb").endswith(".ippo_scan")

    def test_unknown_env_family_raises(self):
        from jaxmarl_worker.cli import _resolve_scan_module
        with pytest.raises(ValueError, match="Unknown env_family"):
            _resolve_scan_module("mappo", "hockey")

    def test_unknown_algo_raises(self):
        from jaxmarl_worker.cli import _resolve_scan_module
        with pytest.raises(ValueError, match="Unknown algo"):
            _resolve_scan_module("random", "soccer")

    def test_resolved_modules_actually_import(self):
        """Every (sport, algo) combination must resolve to a real, importable module."""
        import importlib.util
        from jaxmarl_worker.cli import _resolve_scan_module, _SPORT_TO_DIR
        for env_family in _SPORT_TO_DIR:
            for algo in ("mappo", "ippo"):
                path = _resolve_scan_module(algo, env_family)
                # Just check the file exists — actual import brings in JAX and is slow.
                rel = path.replace(".", "/") + ".py"
                spec = importlib.util.find_spec("jaxmarl_worker")
                assert spec is not None and spec.origin is not None
                pkg_root = Path(spec.origin).parent.parent
                assert (pkg_root / rel).is_file(), f"{path} does not exist at {pkg_root / rel}"


class TestRunFromConfigDispatch:
    """Verify _run_from_config dispatches to the correct scan module per (sport, algo)."""

    def _write_config(self, tmp_path, **overrides):
        cfg = {
            "run_id":     "test_dispatch",
            "algo":       overrides.get("algo", "mappo"),
            "env_family": overrides.get("env_family", "soccer"),
            "variant":    overrides.get("variant", "1v1"),
        }
        cfg.update({k: v for k, v in overrides.items() if k not in cfg})
        p = tmp_path / "config.json"
        p.write_text(json.dumps(cfg))
        return str(p)

    @pytest.mark.parametrize("env_family,expected_dir", [
        ("soccer", "S"),
        ("af",     "AF"),
        ("bb",     "BB"),
    ])
    def test_mappo_dispatches_to_correct_sport_module(self, tmp_path, env_family, expected_dir):
        from jaxmarl_worker.cli import _run_from_config
        cfg_path = self._write_config(tmp_path, algo="mappo", env_family=env_family)

        expected_module = f"jaxmarl_worker.algorithms.mosaic_multigrid.{expected_dir}.mappo_scan"
        fake_module = MagicMock()
        with patch("importlib.import_module", return_value=fake_module) as m:
            _run_from_config(cfg_path)
        m.assert_called_once_with(expected_module)
        fake_module.main.assert_called_once()

    @pytest.mark.parametrize("env_family,expected_dir", [
        ("soccer", "S"),
        ("af",     "AF"),
        ("bb",     "BB"),
    ])
    def test_ippo_dispatches_to_correct_sport_module(self, tmp_path, env_family, expected_dir):
        from jaxmarl_worker.cli import _run_from_config
        cfg_path = self._write_config(tmp_path, algo="ippo", env_family=env_family)

        expected_module = f"jaxmarl_worker.algorithms.mosaic_multigrid.{expected_dir}.ippo_scan"
        fake_module = MagicMock()
        with patch("importlib.import_module", return_value=fake_module) as m:
            _run_from_config(cfg_path)
        m.assert_called_once_with(expected_module)
        fake_module.main.assert_called_once()

    def test_unknown_algo_raises_before_importing(self, tmp_path):
        from jaxmarl_worker.cli import _run_from_config
        cfg_path = self._write_config(tmp_path, algo="COMA", env_family="soccer")
        with pytest.raises(ValueError, match="Unknown algo"):
            _run_from_config(cfg_path)

    def test_unknown_sport_raises_before_importing(self, tmp_path):
        from jaxmarl_worker.cli import _run_from_config
        cfg_path = self._write_config(tmp_path, algo="mappo", env_family="cricket")
        with pytest.raises(ValueError, match="Unknown env_family"):
            _run_from_config(cfg_path)

    def test_missing_main_raises_import_error(self, tmp_path):
        """If the scan module exists but has no main(), we should raise ImportError."""
        from jaxmarl_worker.cli import _run_from_config
        cfg_path = self._write_config(tmp_path, algo="mappo", env_family="soccer")
        fake_module = MagicMock(spec=[])  # no attributes at all → no 'main'
        with patch("importlib.import_module", return_value=fake_module):
            with pytest.raises(ImportError, match="has no main"):
                _run_from_config(cfg_path)


# ---------------------------------------------------------------------------
# Gap 7 + Gap 11: dry-run validation
# ---------------------------------------------------------------------------

class TestDryRunFromConfig:
    """Verify _dry_run_from_config validates without launching training."""

    def _write_config(self, tmp_path, **overrides):
        cfg = {
            "run_id":     "dry_run_test",
            "algo":       "mappo",
            "env_family": "soccer",
            "variant":    "G-1v0",
        }
        cfg.update(overrides)
        p = tmp_path / "config.json"
        p.write_text(json.dumps(cfg))
        return str(p)

    def test_valid_config_exits_zero(self, tmp_path):
        from jaxmarl_worker.cli import _dry_run_from_config
        cfg_path = self._write_config(tmp_path)
        assert _dry_run_from_config(cfg_path) == 0

    def test_missing_file_exits_nonzero(self, tmp_path):
        from jaxmarl_worker.cli import _dry_run_from_config
        assert _dry_run_from_config(str(tmp_path / "does_not_exist.json")) == 1

    def test_malformed_json_exits_nonzero(self, tmp_path):
        from jaxmarl_worker.cli import _dry_run_from_config
        p = tmp_path / "bad.json"
        p.write_text("not valid json {")
        assert _dry_run_from_config(str(p)) == 1

    @pytest.mark.parametrize("field", ["run_id", "algo", "env_family"])
    def test_missing_required_field(self, tmp_path, field):
        """Variant is not required (socialjax doesn't have variants)."""
        from jaxmarl_worker.cli import _dry_run_from_config
        cfg = {"run_id": "x", "algo": "mappo", "env_family": "soccer", "variant": "G-1v0"}
        del cfg[field]
        p = tmp_path / "config.json"
        p.write_text(json.dumps(cfg))
        assert _dry_run_from_config(str(p)) == 1

    def test_variant_not_required(self, tmp_path):
        """Variant may be omitted (e.g., socialjax runs)."""
        from jaxmarl_worker.cli import _dry_run_from_config
        cfg = {"run_id": "x", "algo": "mappo", "env_family": "socialjax", "environment": "cleanup"}
        p = tmp_path / "config.json"
        p.write_text(json.dumps(cfg))
        assert _dry_run_from_config(str(p)) == 0


    def test_invalid_algo(self, tmp_path):
        from jaxmarl_worker.cli import _dry_run_from_config
        cfg_path = self._write_config(tmp_path, algo="ALPHAZERO")
        assert _dry_run_from_config(cfg_path) == 1

    def test_invalid_env_family(self, tmp_path):
        from jaxmarl_worker.cli import _dry_run_from_config
        cfg_path = self._write_config(tmp_path, env_family="curling")
        assert _dry_run_from_config(cfg_path) == 1

    def test_invalid_variant(self, tmp_path):
        from jaxmarl_worker.cli import _dry_run_from_config
        cfg_path = self._write_config(tmp_path, variant="X-9v9")
        assert _dry_run_from_config(cfg_path) == 1

    def test_numeric_out_of_range(self, tmp_path):
        from jaxmarl_worker.cli import _dry_run_from_config
        cfg_path = self._write_config(tmp_path, gamma=42.0)  # > 1.0
        assert _dry_run_from_config(cfg_path) == 1

    def test_all_sports_all_algos_pass(self, tmp_path):
        """Every valid (sport, algo, variant) combo should dry-run OK."""
        from jaxmarl_worker.cli import _dry_run_from_config
        for algo in ("mappo", "ippo"):
            for env_family in ("soccer", "af", "bb"):
                for variant in ("1v1", "2v2", "3v3"):
                    cfg_path = self._write_config(
                        tmp_path,
                        algo=algo,
                        env_family=env_family,
                        variant=variant,
                    )
                    result = _dry_run_from_config(cfg_path)
                    assert result == 0, f"({env_family}, {algo}, {variant}) failed dry-run"

    def test_emit_summary_writes_json(self, tmp_path, capsys):
        from jaxmarl_worker.cli import _dry_run_from_config
        cfg_path = self._write_config(tmp_path)
        _dry_run_from_config(cfg_path, emit_summary=True)
        captured = capsys.readouterr().out
        # The last stdout line should be parseable JSON with our summary fields.
        summary_line = [
            line for line in captured.strip().splitlines()
            if line.startswith("{") and line.endswith("}")
        ]
        assert summary_line, f"No JSON summary in output:\n{captured}"
        summary = json.loads(summary_line[-1])
        assert summary["status"] == "ok"
        assert summary["algo"] == "mappo"
        # env_family is normalized to "mosaic_multigrid" (soccer is a legacy value promoted)
        assert summary["env_family"] == "mosaic_multigrid"
        assert summary["environment"] == "soccer"
        assert summary["variant"] == "G-1v0"


# ---------------------------------------------------------------------------
# Gap 11 (extended): socialjax family routing
# ---------------------------------------------------------------------------

class TestSocialJaxDispatch:
    """Verify socialjax family routing works end-to-end."""

    def _write_config(self, tmp_path, **overrides):
        cfg = {
            "run_id":      "socialjax_test",
            "algo":        "mappo",
            "env_family":  "socialjax",
            "environment": "cleanup",
        }
        cfg.update(overrides)
        p = tmp_path / "config.json"
        p.write_text(json.dumps(cfg))
        return str(p)

    @pytest.mark.parametrize("environment,algo,expected_basename", [
        ("cleanup",     "mappo",   "mappo_cnn_scan_socialjax"),
        ("cleanup",     "ippo",    "ippo_cnn_scan_socialjax"),
        ("coins",       "mappo",   "mappo_cnn_scan_socialjax"),
        ("coins",       "ippo",    "ippo_cnn_scan_socialjax"),
        ("coins",       "mat",     "mat_cnn_scan_socialjax"),
        ("coins",       "commnet", "commnet_cnn_scan_socialjax"),
        ("coop_mining", "mappo",   "mappo_scan"),
        ("coop_mining", "ippo",    "ippo_scan"),
        ("coop_mining", "coma",    "coma_cnn_4_agents_scan_socialjax"),
    ])
    def test_socialjax_routes(self, environment, algo, expected_basename):
        from jaxmarl_worker.cli import _resolve_scan_module
        got = _resolve_scan_module(algo, "socialjax", environment)
        assert got == f"jaxmarl_worker.algorithms.socialjax.{environment}.{expected_basename}"

    def test_socialjax_dry_run_valid(self, tmp_path):
        from jaxmarl_worker.cli import _dry_run_from_config
        cfg_path = self._write_config(tmp_path)
        assert _dry_run_from_config(cfg_path) == 0

    def test_socialjax_invalid_environment(self, tmp_path):
        from jaxmarl_worker.cli import _dry_run_from_config
        cfg_path = self._write_config(tmp_path, environment="not_a_task")
        assert _dry_run_from_config(cfg_path) == 1

    def test_socialjax_unknown_algo_env_pair(self):
        from jaxmarl_worker.cli import _resolve_scan_module
        with pytest.raises(ValueError, match="No socialjax scan file"):
            _resolve_scan_module("mat", "socialjax", "cleanup")  # mat not defined for cleanup

    def test_socialjax_dispatch_calls_correct_module(self, tmp_path):
        from jaxmarl_worker.cli import _run_from_config
        cfg_path = self._write_config(tmp_path, algo="mat", environment="coins")
        expected = "jaxmarl_worker.algorithms.socialjax.coins.mat_cnn_scan_socialjax"
        fake_module = MagicMock()
        with patch("importlib.import_module", return_value=fake_module) as m:
            _run_from_config(cfg_path)
        m.assert_called_once_with(expected)
        fake_module.main.assert_called_once()
