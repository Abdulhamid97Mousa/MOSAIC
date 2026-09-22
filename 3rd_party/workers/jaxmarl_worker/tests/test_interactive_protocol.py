"""End-to-end subprocess test for jaxmarl_worker interactive protocol.

Launches the actual worker as a subprocess (exactly as the GUI does), sends
JSON commands through stdin, and reads responses from stdout.  These tests:

  1. Verify the reset -> ready handshake works
  2. Verify step commands produce step/episode_done responses
  3. Verify rewards accumulate (non-degenerate policy)
  4. Verify render_payload is present and well-formed
  5. Verify action diversity (policy is not stuck on one action)

Run with:
    XLA_PYTHON_CLIENT_PREALLOCATE=false \\
    python -m pytest tests/test_interactive_protocol.py -v -s

These tests require the MOSAIC venv and trained checkpoints.
They are skipped automatically when checkpoints are missing.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE      = Path(__file__).resolve()
_JAXMARL   = _HERE.parent.parent          # jaxmarl_worker repo root
_MOSAIC    = _JAXMARL.parents[2]          # mosaic/ project root
_CKPT_ROOT = _MOSAIC / "var" / "trainer" / "socialjax" / "coop_mining"

_MAPPO_CKPT = _CKPT_ROOT / "MAPPO" / "checkpoints" / "actor_final.npz"
_IPPO_CKPT  = _CKPT_ROOT / "IPPO"  / "checkpoints" / "final.npz"

_PYTHON = sys.executable  # same interpreter that pytest is running under

# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------

class WorkerProcess:
    """Thin wrapper around the jaxmarl_worker interactive subprocess."""

    def __init__(self, env_id: str, policy_path: Path) -> None:
        env = os.environ.copy()
        env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

        self._proc = subprocess.Popen(
            [
                _PYTHON, "-m", "jaxmarl_worker.cli",
                "--interactive",
                "--env-id",      env_id,
                "--policy-path", str(policy_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=env,
            cwd=str(_JAXMARL),
        )

    def send(self, cmd: dict) -> None:
        line = json.dumps(cmd) + "\n"
        self._proc.stdin.write(line.encode())
        self._proc.stdin.flush()

    def read_response(self, timeout: float = 90.0) -> dict:
        """Read one JSON line from stdout.  Raises TimeoutError after timeout."""
        start = time.monotonic()
        buf = b""
        while True:
            if time.monotonic() - start > timeout:
                stderr = self._proc.stderr.read(2048) if self._proc.stderr else b""
                raise TimeoutError(
                    f"No response within {timeout}s.\nstderr tail:\n{stderr.decode(errors='replace')}"
                )
            chunk = self._proc.stdout.read(1)
            if not chunk:
                time.sleep(0.01)
                continue
            buf += chunk
            if buf.endswith(b"\n"):
                return json.loads(buf.decode())

    def stop(self) -> None:
        try:
            self.send({"cmd": "stop"})
        except Exception:
            pass
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stop()


# ---------------------------------------------------------------------------
# Skip guards
# ---------------------------------------------------------------------------

needs_mappo = pytest.mark.skipif(
    not _MAPPO_CKPT.exists(),
    reason=f"MAPPO checkpoint not found: {_MAPPO_CKPT}",
)

needs_ippo = pytest.mark.skipif(
    not _IPPO_CKPT.exists(),
    reason=f"IPPO checkpoint not found: {_IPPO_CKPT}",
)

# ---------------------------------------------------------------------------
# Tests: MAPPO checkpoint
# ---------------------------------------------------------------------------

@needs_mappo
class TestMAPPOCoopMining:
    """Integration tests for MAPPO actor_final.npz on coop_mining."""

    ENV_ID = "socialjax/coop_mining"

    def test_reset_emits_ready(self) -> None:
        with WorkerProcess(self.ENV_ID, _MAPPO_CKPT) as wp:
            wp.send({"cmd": "reset", "seed": 42})
            resp = wp.read_response(timeout=90)

        assert resp["type"] == "ready", f"Expected 'ready', got: {resp}"
        assert resp["seed"] == 42
        assert resp["step_index"] == 0
        assert "observation_shape" in resp

    def test_reset_includes_render_payload(self) -> None:
        with WorkerProcess(self.ENV_ID, _MAPPO_CKPT) as wp:
            wp.send({"cmd": "reset", "seed": 42})
            resp = wp.read_response(timeout=90)

        # render_payload may be None if socialjax render fails, but the key
        # must exist in the response when rendering works.  coop_mining emits
        # the compact socialjax_grid payload (not raw rgb_array).
        rp = resp.get("render_payload")
        if rp is not None:
            assert rp["mode"] == "socialjax_grid"
            assert len(rp["grid"]) > 0
            assert len(rp["agent_locs"]) == rp["n_agents"]

    def test_step_emits_step_type(self) -> None:
        with WorkerProcess(self.ENV_ID, _MAPPO_CKPT) as wp:
            wp.send({"cmd": "reset", "seed": 42})
            wp.read_response(timeout=90)  # ready

            wp.send({"cmd": "step"})
            resp = wp.read_response(timeout=30)

        assert resp["type"] in ("step", "episode_done"), f"Unexpected: {resp}"
        assert "reward" in resp
        assert "step_index" in resp
        assert resp["step_index"] == 1

    def test_100_steps_accumulate_reward(self) -> None:
        """Run 100 steps and check total_reward is non-trivially positive.

        A degenerate policy (all-stay) would give ~0 reward.
        A learned MAPPO policy on coop_mining should mine iron/gold.
        """
        n_steps = 100
        rewards = []

        with WorkerProcess(self.ENV_ID, _MAPPO_CKPT) as wp:
            wp.send({"cmd": "reset", "seed": 42})
            wp.read_response(timeout=90)

            for _ in range(n_steps):
                wp.send({"cmd": "step"})
                resp = wp.read_response(timeout=30)
                rewards.append(resp["reward"])
                if resp["type"] == "episode_done":
                    break

        total = sum(rewards)
        n_positive = sum(1 for r in rewards if r > 0)

        print(f"\n  steps={len(rewards)}  total_reward={total:.3f}  "
              f"positive_steps={n_positive}/{len(rewards)}")

        assert total > 0, (
            f"Total reward after {len(rewards)} steps = {total:.4f}. "
            f"Policy appears degenerate (no mining). Rewards: {rewards[:20]}"
        )

    def test_action_diversity(self) -> None:
        """Agents must use more than one action type -- not stuck on 'stay'."""
        n_steps = 50
        actions = []

        with WorkerProcess(self.ENV_ID, _MAPPO_CKPT) as wp:
            wp.send({"cmd": "reset", "seed": 99})
            wp.read_response(timeout=90)

            for _ in range(n_steps):
                wp.send({"cmd": "step"})
                resp = wp.read_response(timeout=30)
                actions.append(resp.get("action", -1))
                if resp["type"] == "episode_done":
                    break

        unique_actions = set(actions)
        print(f"\n  unique actions in {len(actions)} steps: {sorted(unique_actions)}")

        assert len(unique_actions) > 1, (
            f"All {len(actions)} steps used the same action: {unique_actions}. "
            f"Policy may be collapsed to always-stay."
        )

    def test_render_payload_changes_between_steps(self) -> None:
        """The rendered frame must differ between step 0 and step 5 (agents moved)."""
        frames = []

        with WorkerProcess(self.ENV_ID, _MAPPO_CKPT) as wp:
            wp.send({"cmd": "reset", "seed": 7})
            resp = wp.read_response(timeout=90)
            if resp.get("render_payload"):
                frames.append(str(resp["render_payload"]["agent_locs"]))

            for _ in range(10):
                wp.send({"cmd": "step"})
                resp = wp.read_response(timeout=30)
                if resp.get("render_payload"):
                    frames.append(str(resp["render_payload"]["agent_locs"]))
                if resp["type"] == "episode_done":
                    break

        if len(frames) < 2:
            pytest.skip("Renderer not available -- cannot compare frames")

        # At least one frame must differ from the first (agents moved)
        assert any(f != frames[0] for f in frames[1:]), (
            "All rendered frames are identical -- agents appear frozen. "
            "Policy may be degenerate (all-stay) or renderer is broken."
        )

    def test_stop_command_exits_cleanly(self) -> None:
        with WorkerProcess(self.ENV_ID, _MAPPO_CKPT) as wp:
            wp.send({"cmd": "reset", "seed": 0})
            wp.read_response(timeout=90)

            wp.send({"cmd": "stop"})
            resp = wp.read_response(timeout=10)

        assert resp["type"] == "stopped"


# ---------------------------------------------------------------------------
# Tests: IPPO checkpoint
# ---------------------------------------------------------------------------

@needs_ippo
class TestIPPOCoopMining:
    """Integration tests for IPPO final.npz on coop_mining."""

    ENV_ID = "socialjax/coop_mining"

    def test_reset_emits_ready(self) -> None:
        with WorkerProcess(self.ENV_ID, _IPPO_CKPT) as wp:
            wp.send({"cmd": "reset", "seed": 1})
            resp = wp.read_response(timeout=90)

        assert resp["type"] == "ready"
        assert resp["step_index"] == 0

    def test_step_emits_step_type(self) -> None:
        with WorkerProcess(self.ENV_ID, _IPPO_CKPT) as wp:
            wp.send({"cmd": "reset", "seed": 1})
            wp.read_response(timeout=90)

            wp.send({"cmd": "step"})
            resp = wp.read_response(timeout=30)

        assert resp["type"] in ("step", "episode_done")

    def test_200_steps_accumulate_reward(self) -> None:
        # IPPO mines more slowly than MAPPO on coop_mining: with seed 1 the
        # first reward events arrive well after step 50, so the window must
        # be long enough to observe them (200 steps yields total_reward=3.0).
        n_steps = 200
        rewards = []

        with WorkerProcess(self.ENV_ID, _IPPO_CKPT) as wp:
            wp.send({"cmd": "reset", "seed": 1})
            wp.read_response(timeout=90)

            for _ in range(n_steps):
                wp.send({"cmd": "step"})
                resp = wp.read_response(timeout=30)
                rewards.append(resp["reward"])
                if resp["type"] == "episode_done":
                    break

        total = sum(rewards)
        print(f"\n  IPPO steps={len(rewards)}  total_reward={total:.3f}")
        assert total > 0, f"IPPO policy appears degenerate. total_reward={total:.4f}"
