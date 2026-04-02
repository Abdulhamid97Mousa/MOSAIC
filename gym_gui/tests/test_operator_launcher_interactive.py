"""Tests for OperatorLauncher interactive mode support."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch


from typing import TYPE_CHECKING


def _has_pyqt6() -> bool:
    """Check if PyQt6 is available."""
    try:
        from PyQt6 import QtCore  # noqa: F401
        return True
    except ImportError:
        return False


# Import real classes when PyQt6 is available; use TYPE_CHECKING for pyright.
if TYPE_CHECKING:
    from gym_gui.services.operator import OperatorConfig
    from gym_gui.services.operator_launcher import (
        OperatorLauncher,
        OperatorLaunchError,
        OperatorProcessHandle,
    )
elif _has_pyqt6():
    from gym_gui.services.operator import OperatorConfig
    from gym_gui.services.operator_launcher import (
        OperatorLauncher,
        OperatorLaunchError,
        OperatorProcessHandle,
    )


@unittest.skipUnless(_has_pyqt6(), "Requires PyQt6")
class TestOperatorProcessHandleInteractive(unittest.TestCase):
    """Test OperatorProcessHandle interactive methods."""

    def _make_mock_handle(
        self,
        interactive: bool = True,
        is_running: bool = True,
        has_stdin: bool = True,
    ) -> OperatorProcessHandle:
        """Create a mock OperatorProcessHandle for testing."""
        mock_process = MagicMock(spec=subprocess.Popen)
        mock_process.poll.return_value = None if is_running else 0

        if has_stdin:
            mock_process.stdin = MagicMock()
            mock_process.stdin.write = MagicMock()
            mock_process.stdin.flush = MagicMock()
        else:
            mock_process.stdin = None

        mock_config = MagicMock(spec=OperatorConfig)
        mock_config.max_steps = None

        return OperatorProcessHandle(
            operator_id="test_op",
            run_id="test_run_123",
            process=mock_process,
            log_path=Path("/tmp/test.log"),
            config=mock_config,
            interactive=interactive,
        )

    def test_send_command_success(self) -> None:
        """Should send JSON command to stdin."""
        handle = self._make_mock_handle(interactive=True, is_running=True)

        result = handle.send_command({"cmd": "step"})

        self.assertTrue(result)
        assert handle.process.stdin is not None
        handle.process.stdin.write.assert_called_once()  # type: ignore[union-attr]
        handle.process.stdin.flush.assert_called_once()  # type: ignore[union-attr]

        # Verify JSON format
        written_data = handle.process.stdin.write.call_args[0][0]  # type: ignore[union-attr]
        parsed = json.loads(written_data.strip())
        self.assertEqual(parsed["cmd"], "step")

    def test_send_command_not_interactive(self) -> None:
        """Should return False for non-interactive operator."""
        handle = self._make_mock_handle(interactive=False)

        result = handle.send_command({"cmd": "step"})

        self.assertFalse(result)
        assert handle.process.stdin is not None
        handle.process.stdin.write.assert_not_called()  # type: ignore[union-attr]

    def test_send_command_not_running(self) -> None:
        """Should return False if process not running."""
        handle = self._make_mock_handle(interactive=True, is_running=False)

        result = handle.send_command({"cmd": "step"})

        self.assertFalse(result)

    def test_send_command_no_stdin(self) -> None:
        """Should return False if no stdin pipe."""
        handle = self._make_mock_handle(interactive=True, has_stdin=False)

        result = handle.send_command({"cmd": "step"})

        self.assertFalse(result)

    def test_send_reset_with_seed(self) -> None:
        """Should send reset command with seed."""
        handle = self._make_mock_handle(interactive=True)

        result = handle.send_reset(seed=42)

        self.assertTrue(result)
        assert handle.process.stdin is not None
        written_data = handle.process.stdin.write.call_args[0][0]  # type: ignore[union-attr]
        parsed = json.loads(written_data.strip())
        self.assertEqual(parsed["cmd"], "reset")
        self.assertEqual(parsed["seed"], 42)

    def test_send_reset_without_seed(self) -> None:
        """Should send reset command without seed."""
        handle = self._make_mock_handle(interactive=True)

        result = handle.send_reset()

        self.assertTrue(result)
        assert handle.process.stdin is not None
        written_data = handle.process.stdin.write.call_args[0][0]  # type: ignore[union-attr]
        parsed = json.loads(written_data.strip())
        self.assertEqual(parsed["cmd"], "reset")
        self.assertNotIn("seed", parsed)

    def test_send_step(self) -> None:
        """Should send step command."""
        handle = self._make_mock_handle(interactive=True)

        result = handle.send_step()

        self.assertTrue(result)
        assert handle.process.stdin is not None
        written_data = handle.process.stdin.write.call_args[0][0]  # type: ignore[union-attr]
        parsed = json.loads(written_data.strip())
        self.assertEqual(parsed["cmd"], "step")

    def test_send_stop(self) -> None:
        """Should send stop command."""
        handle = self._make_mock_handle(interactive=True)

        result = handle.send_stop()

        self.assertTrue(result)
        assert handle.process.stdin is not None
        written_data = handle.process.stdin.write.call_args[0][0]  # type: ignore[union-attr]
        parsed = json.loads(written_data.strip())
        self.assertEqual(parsed["cmd"], "stop")

    def test_send_command_write_error(self) -> None:
        """Should handle write errors gracefully."""
        handle = self._make_mock_handle(interactive=True)
        assert handle.process.stdin is not None
        handle.process.stdin.write.side_effect = IOError("Broken pipe")  # type: ignore[union-attr]

        result = handle.send_command({"cmd": "step"})

        self.assertFalse(result)


@unittest.skipUnless(_has_pyqt6(), "Requires PyQt6")
class TestOperatorLauncherInteractive(unittest.TestCase):
    """Test OperatorLauncher interactive mode."""

    def _make_config(self) -> OperatorConfig:
        """Create a test operator config."""
        return OperatorConfig.single_agent(
            operator_id="test_operator",
            worker_id="balrog_worker",
            worker_type="llm",
            display_name="Test Operator",
            env_name="babyai",
            task="BabyAI-GoToRedBall-v0",
            settings={
                "client_name": "vllm",
                "model_id": "test-model",
                "base_url": "http://localhost:8000/v1",
            },
        )

    @patch("gym_gui.services.operator_launcher.validated_popen")
    @patch("gym_gui.services.operator_launcher.VAR_OPERATORS_LOGS_DIR", Path("/tmp/operators/logs"))
    @patch("gym_gui.services.operator_launcher.VAR_OPERATORS_TELEMETRY_DIR", Path("/tmp/operators/telemetry"))
    @patch("gym_gui.services.operator_launcher.ensure_var_directories")
    def test_launch_interactive_uses_pipes(
        self,
        mock_ensure_dirs: MagicMock,
        mock_popen: MagicMock,
    ) -> None:
        """Should use stdin/stdout pipes in interactive mode."""
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("gym_gui.services.operator_launcher.VAR_OPERATORS_LOGS_DIR", Path(tmpdir)):
                launcher = OperatorLauncher()
                config = self._make_config()

                handle = launcher.launch_operator(config, interactive=True)

        # Verify popen was called with stdin=PIPE
        call_kwargs = mock_popen.call_args[1]
        self.assertEqual(call_kwargs["stdin"], subprocess.PIPE)
        self.assertEqual(call_kwargs["stdout"], subprocess.PIPE)

        # Verify handle is marked as interactive
        self.assertTrue(handle.interactive)

    @patch("gym_gui.services.operator_launcher.validated_popen")
    @patch("gym_gui.services.operator_launcher.VAR_OPERATORS_LOGS_DIR", Path("/tmp/operators/logs"))
    @patch("gym_gui.services.operator_launcher.VAR_OPERATORS_TELEMETRY_DIR", Path("/tmp/operators/telemetry"))
    @patch("gym_gui.services.operator_launcher.ensure_var_directories")
    def test_launch_non_interactive_uses_log_file(
        self,
        mock_ensure_dirs: MagicMock,
        mock_popen: MagicMock,
    ) -> None:
        """Should use log file for stdout in non-interactive mode."""
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("gym_gui.services.operator_launcher.VAR_OPERATORS_LOGS_DIR", Path(tmpdir)):
                launcher = OperatorLauncher()
                config = self._make_config()

                handle = launcher.launch_operator(config, interactive=False)

        # Verify popen was called with file for stdout (not PIPE)
        call_kwargs = mock_popen.call_args[1]
        self.assertNotEqual(call_kwargs.get("stdin"), subprocess.PIPE)
        self.assertNotEqual(call_kwargs["stdout"], subprocess.PIPE)

        # Verify handle is not marked as interactive
        self.assertFalse(handle.interactive)

    def test_build_llm_command_interactive_flag(self) -> None:
        """Should include --interactive flag when interactive=True."""
        launcher = OperatorLauncher()
        config = self._make_config()

        cmd_interactive = launcher._build_llm_command(config, "run123", interactive=True)
        cmd_normal = launcher._build_llm_command(config, "run123", interactive=False)

        self.assertIn("--interactive", cmd_interactive)
        self.assertNotIn("--interactive", cmd_normal)

    def test_build_llm_command_has_required_args(self) -> None:
        """Should include all required CLI arguments."""
        launcher = OperatorLauncher()
        config = self._make_config()

        cmd = launcher._build_llm_command(config, "run123", interactive=True)

        # Check required args
        self.assertIn("--run-id", cmd)
        self.assertIn("run123", cmd)
        self.assertIn("--env", cmd)
        self.assertIn("babyai", cmd)
        self.assertIn("--task", cmd)
        self.assertIn("BabyAI-GoToRedBall-v0", cmd)
        self.assertIn("--client", cmd)
        self.assertIn("vllm", cmd)
        self.assertIn("--model", cmd)
        self.assertIn("test-model", cmd)
        self.assertIn("--base-url", cmd)
        self.assertIn("http://localhost:8000/v1", cmd)


@unittest.skipUnless(_has_pyqt6(), "Requires PyQt6")
class TestOperatorProcessHandleDataclass(unittest.TestCase):
    """Test OperatorProcessHandle dataclass fields."""

    def test_interactive_field_default(self) -> None:
        """Interactive field should default to False."""
        mock_process = MagicMock(spec=subprocess.Popen)
        mock_config = MagicMock(spec=OperatorConfig)

        handle = OperatorProcessHandle(
            operator_id="test",
            run_id="run",
            process=mock_process,
            log_path=Path("/tmp/test.log"),
            config=mock_config,
        )

        self.assertFalse(handle.interactive)

    def test_interactive_field_explicit(self) -> None:
        """Interactive field should be settable."""
        mock_process = MagicMock(spec=subprocess.Popen)
        mock_config = MagicMock(spec=OperatorConfig)

        handle = OperatorProcessHandle(
            operator_id="test",
            run_id="run",
            process=mock_process,
            log_path=Path("/tmp/test.log"),
            config=mock_config,
            interactive=True,
        )

        self.assertTrue(handle.interactive)


@unittest.skipUnless(_has_pyqt6(), "Requires PyQt6")
class TestOperatorProcessHandleStdoutReader(unittest.TestCase):
    """Test OperatorProcessHandle stdout reading methods."""

    def _make_mock_handle(
        self,
        interactive: bool = True,
        is_running: bool = True,
        stdout_data: str = "",
    ) -> OperatorProcessHandle:
        """Create a mock OperatorProcessHandle with configurable stdout.

        Args:
            interactive: Whether the handle is in interactive mode.
            is_running: Whether the process is running.
            stdout_data: Data to return from stdout.readline().
        """
        import io

        mock_process = MagicMock(spec=subprocess.Popen)
        mock_process.poll.return_value = None if is_running else 0

        # Create a real StringIO for stdout so select() can work with it
        if stdout_data:
            mock_stdout = io.StringIO(stdout_data)
        else:
            mock_stdout = io.StringIO("")
        mock_process.stdout = mock_stdout

        mock_config = MagicMock(spec=OperatorConfig)

        return OperatorProcessHandle(
            operator_id="test_op",
            run_id="test_run",
            process=mock_process,
            log_path=Path("/tmp/test.log"),
            config=mock_config,
            interactive=interactive,
        )

    def test_try_read_response_not_interactive(self) -> None:
        """Should return None for non-interactive operator."""
        handle = self._make_mock_handle(interactive=False)

        result = handle.try_read_response(timeout=0.0)

        self.assertIsNone(result)

    def test_try_read_response_no_stdout(self) -> None:
        """Should return None if stdout is None."""
        handle = self._make_mock_handle(interactive=True)
        handle.process.stdout = None

        result = handle.try_read_response(timeout=0.0)

        self.assertIsNone(result)

    def test_try_read_response_process_not_running_no_stdout(self) -> None:
        """Should return None if process not running and no stdout."""
        handle = self._make_mock_handle(interactive=True, is_running=False)
        handle.process.stdout = None

        result = handle.try_read_response(timeout=0.0)

        self.assertIsNone(result)

    @patch("select.select")
    def test_try_read_response_parses_json(self, mock_select: MagicMock) -> None:
        """Should parse JSON response from stdout."""
        json_line = '{"type": "step", "step_index": 5, "reward": 0.5}\n'
        handle = self._make_mock_handle(interactive=True, stdout_data=json_line)

        # Mock select to indicate data is available
        mock_select.return_value = ([handle.process.stdout], [], [])

        result = handle.try_read_response(timeout=0.1)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["type"], "step")
        self.assertEqual(result["step_index"], 5)
        self.assertEqual(result["reward"], 0.5)

    @patch("select.select")
    def test_try_read_response_no_data_available(self, mock_select: MagicMock) -> None:
        """Should return None when no data available."""
        handle = self._make_mock_handle(interactive=True, stdout_data="")

        # Mock select to indicate no data
        mock_select.return_value = ([], [], [])

        result = handle.try_read_response(timeout=0.0)

        self.assertIsNone(result)

    @patch("select.select")
    def test_try_read_response_invalid_json(self, mock_select: MagicMock) -> None:
        """Should return None and not crash on invalid JSON."""
        handle = self._make_mock_handle(interactive=True, stdout_data="not valid json\n")

        mock_select.return_value = ([handle.process.stdout], [], [])

        result = handle.try_read_response(timeout=0.1)

        self.assertIsNone(result)

    @patch("select.select")
    def test_poll_responses_reads_multiple(self, mock_select: MagicMock) -> None:
        """Should read all available responses."""
        json_lines = '{"type": "step", "step_index": 0}\n{"type": "step", "step_index": 1}\n'
        handle = self._make_mock_handle(interactive=True, stdout_data=json_lines)

        # First two calls have data, third doesn't
        call_count = [0]
        def select_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                return ([handle.process.stdout], [], [])
            return ([], [], [])

        mock_select.side_effect = select_side_effect

        results = handle.poll_responses(max_responses=10)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["step_index"], 0)
        self.assertEqual(results[1]["step_index"], 1)

    @patch("select.select")
    def test_poll_responses_respects_max(self, mock_select: MagicMock) -> None:
        """Should stop at max_responses even if more data available."""
        # Lots of JSON lines
        json_lines = ''.join([f'{{"type": "step", "step_index": {i}}}\n' for i in range(20)])
        handle = self._make_mock_handle(interactive=True, stdout_data=json_lines)

        mock_select.return_value = ([handle.process.stdout], [], [])

        results = handle.poll_responses(max_responses=3)

        self.assertEqual(len(results), 3)

    def test_read_response_calls_try_read_with_timeout(self) -> None:
        """read_response should delegate to try_read_response with timeout."""
        handle = self._make_mock_handle(interactive=True)

        with patch.object(handle, 'try_read_response', return_value={"type": "test"}) as mock_try:
            result = handle.read_response(timeout=15.0)

            mock_try.assert_called_once_with(timeout=15.0)
            assert result is not None
            self.assertEqual(result["type"], "test")


@unittest.skipUnless(_has_pyqt6(), "Requires PyQt6")
class TestOperatorStdoutReaderIntegration(unittest.TestCase):
    """Integration tests for stdout reading with real subprocess."""

    def test_read_from_echo_subprocess(self) -> None:
        """Should read JSON from a real subprocess stdout."""
        import sys

        # Create a subprocess that echoes JSON to stdout
        echo_script = '''
import sys
import json
print(json.dumps({"type": "step", "step_index": 42, "reward": 1.0}))
sys.stdout.flush()
'''
        process = subprocess.Popen(
            [sys.executable, "-c", echo_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        mock_config = MagicMock(spec=OperatorConfig)
        handle = OperatorProcessHandle(
            operator_id="test_echo",
            run_id="test_run",
            process=process,
            log_path=Path("/tmp/test.log"),
            config=mock_config,
            interactive=True,
        )

        try:
            # Wait for subprocess to produce output
            response = handle.read_response(timeout=5.0)

            self.assertIsNotNone(response)
            assert response is not None
            self.assertEqual(response["type"], "step")
            self.assertEqual(response["step_index"], 42)
            self.assertEqual(response["reward"], 1.0)
        finally:
            process.terminate()
            process.wait(timeout=2.0)

    def test_bidirectional_communication(self) -> None:
        """Should send command and receive response."""
        import sys

        # Interactive echo subprocess
        echo_script = '''
import sys
import json
for line in sys.stdin:
    try:
        cmd = json.loads(line.strip())
        if cmd.get("cmd") == "step":
            response = {"type": "step", "step_index": 0, "action": "forward"}
            print(json.dumps(response))
            sys.stdout.flush()
        elif cmd.get("cmd") == "stop":
            print(json.dumps({"type": "stopped"}))
            sys.stdout.flush()
            break
    except:
        pass
'''
        process = subprocess.Popen(
            [sys.executable, "-c", echo_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        mock_config = MagicMock(spec=OperatorConfig)
        handle = OperatorProcessHandle(
            operator_id="test_bidir",
            run_id="test_run",
            process=process,
            log_path=Path("/tmp/test.log"),
            config=mock_config,
            interactive=True,
        )

        try:
            # Send step command
            result = handle.send_step()
            self.assertTrue(result)

            # Read response
            response = handle.read_response(timeout=5.0)
            self.assertIsNotNone(response)
            assert response is not None
            self.assertEqual(response["type"], "step")
            self.assertEqual(response["action"], "forward")

            # Send stop
            handle.send_stop()
            stop_response = handle.read_response(timeout=5.0)
            self.assertIsNotNone(stop_response)
            assert stop_response is not None
            self.assertEqual(stop_response["type"], "stopped")
        finally:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()


@unittest.skipUnless(_has_pyqt6(), "Requires PyQt6")
class TestBuildRandomCommandSeed(unittest.TestCase):
    """Tests for _build_random_command seed derivation logic."""

    def _make_random_config(self, settings: dict | None = None) -> OperatorConfig:
        return OperatorConfig.single_agent(
            operator_id="rnd_op",
            worker_id="random_worker",
            worker_type="random",
            display_name="Random Agent",
            env_name="babyai",
            task="BabyAI-GoToRedBall-v0",
            settings=settings or {},
        )

    def test_seed_derived_from_run_id_when_not_set(self) -> None:
        """When no seed in settings, seed is derived from run_id."""
        launcher = OperatorLauncher()
        config = self._make_random_config()
        cmd = launcher._build_random_command(config, "run-abc", interactive=False)
        self.assertIn("--seed", cmd)
        idx = cmd.index("--seed")
        seed_value = int(cmd[idx + 1])
        self.assertEqual(seed_value, abs(hash("run-abc")) % (2**31))

    def test_explicit_seed_in_settings_is_used(self) -> None:
        """When seed is set in settings, it overrides the derived seed."""
        launcher = OperatorLauncher()
        config = self._make_random_config(settings={"seed": 42})
        cmd = launcher._build_random_command(config, "run-abc", interactive=False)
        idx = cmd.index("--seed")
        self.assertEqual(cmd[idx + 1], "42")

    def test_two_run_ids_produce_different_seeds(self) -> None:
        """Different run_ids must produce different seeds."""
        launcher = OperatorLauncher()
        config = self._make_random_config()
        cmd1 = launcher._build_random_command(config, "operator-1", interactive=False)
        cmd2 = launcher._build_random_command(config, "operator-2", interactive=False)
        seed1 = cmd1[cmd1.index("--seed") + 1]
        seed2 = cmd2[cmd2.index("--seed") + 1]
        self.assertNotEqual(seed1, seed2)

    def test_same_run_id_produces_same_seed(self) -> None:
        """Same run_id must always produce the same deterministic seed."""
        launcher = OperatorLauncher()
        config = self._make_random_config()
        cmd1 = launcher._build_random_command(config, "stable-run", interactive=False)
        cmd2 = launcher._build_random_command(config, "stable-run", interactive=False)
        seed1 = cmd1[cmd1.index("--seed") + 1]
        seed2 = cmd2[cmd2.index("--seed") + 1]
        self.assertEqual(seed1, seed2)

    def test_seed_is_valid_numpy_range(self) -> None:
        """Derived seed must be in [0, 2**31) for NumPy compatibility."""
        launcher = OperatorLauncher()
        config = self._make_random_config()
        cmd = launcher._build_random_command(config, "any-run-id", interactive=False)
        seed = int(cmd[cmd.index("--seed") + 1])
        self.assertGreaterEqual(seed, 0)
        self.assertLess(seed, 2**31)


@unittest.skipUnless(_has_pyqt6(), "Requires PyQt6")
class TestBuildPassiveCommandSeed(unittest.TestCase):
    """Tests for _build_passive_command seed derivation logic."""

    def _make_passive_config(self, settings: dict | None = None) -> OperatorConfig:
        return OperatorConfig.single_agent(
            operator_id="pass_op",
            worker_id="passive_worker",
            worker_type="passive",
            display_name="Passive Agent",
            env_name="babyai",
            task="BabyAI-GoToRedBall-v0",
            settings=settings or {},
        )

    def test_seed_derived_from_run_id_when_not_set(self) -> None:
        """When no seed in settings, seed is derived from run_id."""
        launcher = OperatorLauncher()
        config = self._make_passive_config()
        cmd = launcher._build_passive_command(config, "run-xyz", interactive=False)
        self.assertIn("--seed", cmd)
        idx = cmd.index("--seed")
        seed_value = int(cmd[idx + 1])
        self.assertEqual(seed_value, abs(hash("run-xyz")) % (2**31))

    def test_explicit_seed_in_settings_is_used(self) -> None:
        """When seed is set in settings, it overrides the derived seed."""
        launcher = OperatorLauncher()
        config = self._make_passive_config(settings={"seed": 99})
        cmd = launcher._build_passive_command(config, "run-xyz", interactive=False)
        idx = cmd.index("--seed")
        self.assertEqual(cmd[idx + 1], "99")

    def test_two_run_ids_produce_different_seeds(self) -> None:
        """Different run_ids must produce different seeds."""
        launcher = OperatorLauncher()
        config = self._make_passive_config()
        cmd1 = launcher._build_passive_command(config, "passive-1", interactive=False)
        cmd2 = launcher._build_passive_command(config, "passive-2", interactive=False)
        seed1 = cmd1[cmd1.index("--seed") + 1]
        seed2 = cmd2[cmd2.index("--seed") + 1]
        self.assertNotEqual(seed1, seed2)


if __name__ == "__main__":
    unittest.main()
