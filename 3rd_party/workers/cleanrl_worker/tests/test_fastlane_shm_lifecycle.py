"""Tests for FastLane shared memory lifecycle — create, close, unlink.

These tests verify that the FastLaneTelemetryWrapper properly cleans up
shared memory segments to prevent orphaned resource_tracker processes
that consume unbounded RAM and cause system-wide OOM crashes.

Root cause (fixed):
  _close_writer() called writer.close() but never writer.unlink().
  This left /dev/shm/mosaic.fastlane.* segments behind.  Python's
  multiprocessing.resource_tracker spawned to clean them up, but
  inherited heavy imports (PyTorch, numpy, gymnasium) via sitecustomize.py,
  causing each tracker to consume ~1GB RAM with 95 threads.  Multiple
  trackers would spin indefinitely until the OOM killer nuked the GUI,
  Cursor IDE, and everything else on the system.
"""

from __future__ import annotations

import os
import glob
from multiprocessing import shared_memory
from unittest.mock import MagicMock, patch

import gymnasium as gym
import numpy as np
import pytest

from cleanrl_worker.fastlane import (
    FastLaneTelemetryWrapper,
    _FastLaneConfig,
    reset_slot_counter,
    reload_fastlane_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(run_id: str = "test-run", **overrides) -> _FastLaneConfig:
    defaults = dict(
        enabled=True, slot=0, run_id=run_id, agent_id="a",
        worker_id="w", seed=0, total_envs=1,
        video_mode="single", grid_limit=1,
    )
    defaults.update(overrides)
    return _FastLaneConfig(**defaults)


def _make_wrapped_env(config: _FastLaneConfig, slot: int = 0) -> FastLaneTelemetryWrapper:
    """Create a wrapper around a real lightweight gym env."""
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    return FastLaneTelemetryWrapper(env, config, slot_index=slot)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_shm():
    """Remove any stale mosaic fastlane shm segments before/after each test."""
    def _cleanup():
        for path in glob.glob("/dev/shm/mosaic.fastlane.*"):
            try:
                name = os.path.basename(path)
                shm = shared_memory.SharedMemory(name=name, create=False)
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
    _cleanup()
    yield
    _cleanup()


@pytest.fixture
def fastlane_env():
    """Set environment variables for fastlane-enabled test runs."""
    env_vars = {
        "GYM_GUI_FASTLANE_ONLY": "1",
        "GYM_GUI_FASTLANE_SLOT": "0",
        "CLEANRL_NUM_ENVS": "1",
        "CLEANRL_RUN_ID": "test-shm-lifecycle",
        "GYM_GUI_FASTLANE_VIDEO_MODE": "single",
    }
    with patch.dict(os.environ, env_vars):
        yield env_vars


# ---------------------------------------------------------------------------
# Tests: _close_writer must call both close() and unlink()
# ---------------------------------------------------------------------------

class TestCloseWriterUnlinks:
    """Verify _close_writer calls both close() and unlink()."""

    def test_close_writer_calls_unlink(self):
        """The writer must be unlinked on close to prevent resource_tracker leaks."""
        config = _make_config(run_id="test-unlink")
        wrapper = _make_wrapped_env(config)

        mock_writer = MagicMock()
        wrapper._writer = mock_writer

        wrapper._close_writer()

        mock_writer.close.assert_called_once()
        mock_writer.unlink.assert_called_once()
        assert wrapper._writer is None
        wrapper.env.close()

    def test_close_writer_handles_already_unlinked(self):
        """If unlink raises FileNotFoundError, it should not crash."""
        config = _make_config(run_id="test-double-unlink")
        wrapper = _make_wrapped_env(config)

        mock_writer = MagicMock()
        mock_writer.unlink.side_effect = FileNotFoundError("already gone")
        wrapper._writer = mock_writer

        # Should not raise
        wrapper._close_writer()
        assert wrapper._writer is None
        wrapper.env.close()

    def test_close_writer_noop_when_no_writer(self):
        """_close_writer with no active writer should be a safe no-op."""
        config = _make_config(run_id="test-noop")
        wrapper = _make_wrapped_env(config)
        assert wrapper._writer is None

        # Should not raise
        wrapper._close_writer()
        assert wrapper._writer is None
        wrapper.env.close()


# ---------------------------------------------------------------------------
# Tests: _create_writer cleans stale shm on FileExistsError
# ---------------------------------------------------------------------------

class TestCreateWriterCleansStaleShm:
    """Verify _create_writer unlinks stale segments on FileExistsError."""

    def test_create_writer_unlinks_stale_then_recreates(self):
        """On FileExistsError, stale shm should be unlinked before retry."""
        config = _make_config(run_id="test-stale")
        wrapper = _make_wrapped_env(config)

        mock_fl_config = MagicMock()
        mock_writer = MagicMock()

        with patch("cleanrl_worker.fastlane.FastLaneWriter") as MockFLW, \
             patch("cleanrl_worker.fastlane.create_fastlane_name", return_value="mosaic.fastlane.test-stale"):
            MockFLW.create.side_effect = [FileExistsError, mock_writer]

            with patch("cleanrl_worker.fastlane.shared_memory.SharedMemory") as MockSHM:
                mock_stale = MagicMock()
                MockSHM.return_value = mock_stale

                result = wrapper._create_writer(mock_fl_config)

                # Stale segment should have been closed and unlinked
                mock_stale.close.assert_called_once()
                mock_stale.unlink.assert_called_once()
                # Then create should have been retried
                assert MockFLW.create.call_count == 2
                assert result is mock_writer

        wrapper.env.close()


# ---------------------------------------------------------------------------
# Tests: env.close() triggers full shm cleanup
# ---------------------------------------------------------------------------

class TestEnvCloseUnlinksShm:
    """Verify that env.close() triggers full shm cleanup."""

    def test_gym_wrapper_close_triggers_unlink(self):
        """When the gym env is closed, the shm writer must be unlinked."""
        config = _make_config(run_id="test-env-close")
        wrapper = _make_wrapped_env(config)

        mock_writer = MagicMock()
        wrapper._writer = mock_writer

        wrapper.close()

        mock_writer.close.assert_called_once()
        mock_writer.unlink.assert_called_once()
        assert wrapper._writer is None


# ---------------------------------------------------------------------------
# Tests: no /dev/shm leak after stepping and closing
# ---------------------------------------------------------------------------

class TestNoShmLeakAfterSteps:
    """Integration-style test: step the env and verify no /dev/shm leak."""

    def test_no_shm_segments_after_close(self):
        """After creating and closing a wrapper, no mosaic.fastlane.* should remain."""
        shm_before = set(glob.glob("/dev/shm/mosaic.fastlane.*"))

        config = _make_config(run_id="test-shm-lifecycle")
        wrapper = _make_wrapped_env(config)

        # No actual FastLane write happens (no real writer), just verify
        # the wrapper creates/closes cleanly
        wrapper.close()

        shm_after = set(glob.glob("/dev/shm/mosaic.fastlane.*"))
        leaked = shm_after - shm_before
        assert not leaked, (
            f"Leaked shared memory segments after close: {leaked}. "
            f"These will spawn resource_tracker processes that consume ~1GB RAM each "
            f"and eventually OOM-kill the system."
        )


# ---------------------------------------------------------------------------
# Tests: real SharedMemory create → unlink lifecycle
# ---------------------------------------------------------------------------

class TestResourceTrackerNotSpawned:
    """Verify the fix prevents resource_tracker from being needed."""

    def test_writer_unlink_prevents_tracker(self):
        """If unlink is called, resource_tracker should not need to intervene."""
        config = _make_config(run_id="test-no-tracker")
        wrapper = _make_wrapped_env(config)

        # Create a real SharedMemory segment
        shm = shared_memory.SharedMemory(
            name="mosaic.fastlane.test-no-tracker", create=True, size=1024,
        )

        # Wire up a mock writer that delegates to real shm operations
        mock_writer = MagicMock()
        mock_writer.close = MagicMock(side_effect=lambda: shm.close())
        mock_writer.unlink = MagicMock(side_effect=lambda: shm.unlink())
        wrapper._writer = mock_writer

        wrapper._close_writer()

        # The segment should be gone
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(
                name="mosaic.fastlane.test-no-tracker", create=False,
            )

        wrapper.env.close()


# ---------------------------------------------------------------------------
# Tests: slot counter and config reload
# ---------------------------------------------------------------------------

class TestSlotCounterReset:
    """Verify reset_slot_counter works for curriculum learning."""

    def test_reset_slot_counter(self):
        import cleanrl_worker.fastlane as fl

        # Consume some slots
        next(fl._ENV_SLOT_COUNTER)
        next(fl._ENV_SLOT_COUNTER)

        reset_slot_counter()

        # After reset, should start from 0 again
        assert next(fl._ENV_SLOT_COUNTER) == 0


class TestReloadConfig:
    """Verify reload_fastlane_config picks up new env vars."""

    def test_reload_changes_run_id(self):
        import cleanrl_worker.fastlane as fl

        with patch.dict(os.environ, {"CLEANRL_RUN_ID": "reloaded-run"}):
            new_config = reload_fastlane_config()

        assert new_config.run_id == "reloaded-run"
        assert fl._CONFIG is new_config
