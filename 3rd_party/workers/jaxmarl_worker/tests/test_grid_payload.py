"""Tests for the socialjax_grid payload format and end-to-end render path.

Two test classes:
  TestGridPayloadStructure  -- pure-Python checks on payload shape/content.
                               No JAX, no GPU, no checkpoint required.
  TestGridPayloadSubprocess -- subprocess integration tests.
                               Require a real coop_mining MAPPO checkpoint.
                               Check: payload is small, agent_locs change across steps.

Run all:
    XLA_PYTHON_CLIENT_PREALLOCATE=false \\
    python -m pytest tests/test_grid_payload.py -v -s

Run only unit tests (no GPU needed):
    python -m pytest tests/test_grid_payload.py -k unit -v
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE    = Path(__file__).resolve()
_JAXMARL = _HERE.parent.parent
_MOSAIC  = _JAXMARL.parents[2]
_CKPT    = _MOSAIC / "var" / "trainer" / "socialjax" / "coop_mining" / "MAPPO" / "checkpoints" / "actor_final.npz"
_PYTHON  = sys.executable

needs_ckpt = pytest.mark.skipif(
    not _CKPT.exists(),
    reason=f"MAPPO checkpoint not found: {_CKPT}",
)

# ---------------------------------------------------------------------------
# Shared helper: a fake coop_mining-like state for unit tests
# ---------------------------------------------------------------------------

def _make_fake_payload(rows=27, cols=27, n_agents=6, inner_t=50, max_inner_t=1000):
    """Construct a payload that matches _state_to_grid_payload output."""
    import colorsys

    tile_colors = {
        "0": [220, 220, 220],
        "1": [127, 127, 127],
        "2": [200, 200, 170],
        "3": [180, 180, 250],
        "4": [139,  69,  19],
        "5": [180, 180,  40],
        "6": [190, 190,  80],
    }

    agent_colors = []
    for i in range(n_agents):
        r, g, b = colorsys.hsv_to_rgb(i / max(n_agents, 1), 0.8, 0.8)
        agent_colors.append([int(r * 255), int(g * 255), int(b * 255)])

    # Minimal grid: all empty (0) except a ring of walls (1)
    grid = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                grid[r][c] = 1  # wall

    # Place 6 agents at deterministic interior positions
    agent_locs = [[2 + i * 3, 2, i % 4] for i in range(n_agents)]

    return {
        "mode":         "socialjax_grid",
        "grid":         grid,
        "agent_locs":   agent_locs,
        "n_agents":     n_agents,
        "inner_t":      inner_t,
        "max_inner_t":  max_inner_t,
        "tile_colors":  tile_colors,
        "agent_colors": agent_colors,
    }


# ---------------------------------------------------------------------------
# Unit tests -- no JAX, no checkpoint
# ---------------------------------------------------------------------------

class TestGridPayloadStructure:
    """Pure-Python checks on payload content.  Mark with 'unit' for fast runs."""

    pytestmark = pytest.mark.unit

    def test_mode_field(self) -> None:
        p = _make_fake_payload()
        assert p["mode"] == "socialjax_grid"

    def test_grid_dimensions(self) -> None:
        p = _make_fake_payload(rows=27, cols=27)
        grid = p["grid"]
        assert len(grid) == 27, f"Expected 27 rows, got {len(grid)}"
        assert all(len(row) == 27 for row in grid), "Not all rows have 27 cols"

    def test_agent_locs_count(self) -> None:
        p = _make_fake_payload(n_agents=6)
        assert len(p["agent_locs"]) == 6
        assert p["n_agents"] == 6

    def test_agent_locs_shape(self) -> None:
        p = _make_fake_payload(n_agents=6)
        for i, loc in enumerate(p["agent_locs"]):
            assert len(loc) == 3, f"agent_locs[{i}] has {len(loc)} values, expected [row, col, orient]"

    def test_tile_colors_present(self) -> None:
        p = _make_fake_payload()
        tc = p["tile_colors"]
        assert len(tc) >= 7, f"Expected at least 7 tile color entries, got {len(tc)}"
        for code in range(7):
            assert str(code) in tc, f"tile_colors missing key '{code}'"
            rgb = tc[str(code)]
            assert len(rgb) == 3 and all(0 <= v <= 255 for v in rgb), f"Bad RGB for code {code}: {rgb}"

    def test_agent_colors_count(self) -> None:
        p = _make_fake_payload(n_agents=6)
        assert len(p["agent_colors"]) == 6

    def test_time_fields(self) -> None:
        p = _make_fake_payload(inner_t=42, max_inner_t=1000)
        assert p["inner_t"] == 42
        assert p["max_inner_t"] == 1000

    def test_payload_serializes_to_json(self) -> None:
        p = _make_fake_payload()
        serialized = json.dumps(p)
        size_kb = len(serialized) / 1024
        assert size_kb < 50, f"Payload too large: {size_kb:.1f}KB (expected < 50KB for 27x27 grid)"

    def test_grid_contains_valid_tile_codes(self) -> None:
        p = _make_fake_payload()
        tc = p["tile_colors"]
        all_codes = {str(v) for row in p["grid"] for v in row}
        for code in all_codes:
            assert code in tc, f"Grid contains tile code {code} not in tile_colors"

    def test_inner_t_module_import(self) -> None:
        """_state_to_grid_payload is importable without JAX."""
        sys.path.insert(0, str(_JAXMARL))
        try:
            from jaxmarl_worker.runtime import _state_to_grid_payload, _COOP_MINING_TILE_COLORS
            assert isinstance(_COOP_MINING_TILE_COLORS, dict)
            assert len(_COOP_MINING_TILE_COLORS) >= 7
        finally:
            sys.path.pop(0)

    def test_detect_render_mode_socialjax_grid(self) -> None:
        """_detect_render_mode must return SOCIALJAX_GRID for a socialjax_grid payload.

        Also verifies the fix: the payload has a 'grid' key, but the 'mode' field
        must take precedence over key-presence checks so it is NOT misrouted to
        the generic GRID renderer.
        """
        from gym_gui.ui.widgets.operator_render_container import OperatorRenderContainer
        from gym_gui.services.operator import OperatorConfig, WorkerAssignment
        from gym_gui.core.enums import RenderMode

        cfg = OperatorConfig(
            operator_id="op_test",
            display_name="Test",
            env_name="socialjax",
            task="socialjax/coop_mining",
            workers={"agent_0": WorkerAssignment(worker_id="jaxmarl_worker", worker_type="rl")},
        )
        # Bypass Qt widget construction
        container = OperatorRenderContainer.__new__(OperatorRenderContainer)
        container._config = cfg

        payload = _make_fake_payload()
        mode = container._detect_render_mode(payload)
        assert mode == RenderMode.SOCIALJAX_GRID, (
            f"Expected SOCIALJAX_GRID, got {mode!r}. "
            "The 'mode' field check must come BEFORE 'grid' in payload check."
        )

    def test_detect_render_mode_fallback_for_generic_grid(self) -> None:
        """Generic grid payloads (no 'mode' field) still route to GRID renderer."""
        from gym_gui.ui.widgets.operator_render_container import OperatorRenderContainer
        from gym_gui.services.operator import OperatorConfig, WorkerAssignment
        from gym_gui.core.enums import RenderMode

        cfg = OperatorConfig(
            operator_id="op_test",
            display_name="Test",
            env_name="frozen_lake",
            task="FrozenLake-v1",
            workers={"agent_0": WorkerAssignment(worker_id="some_worker", worker_type="rl")},
        )
        container = OperatorRenderContainer.__new__(OperatorRenderContainer)
        container._config = cfg

        payload = {"grid": [[0, 1], [2, 3]]}   # no "mode" field
        mode = container._detect_render_mode(payload)
        assert mode == RenderMode.GRID, f"Expected GRID fallback, got {mode!r}"

    def test_socialjax_grid_strategy_supports(self) -> None:
        """SocialJaxGridStrategy.supports() must discriminate by 'mode' field."""
        from gym_gui.rendering.strategies.socialjax_grid import SocialJaxGridStrategy
        strategy = SocialJaxGridStrategy.__new__(SocialJaxGridStrategy)

        assert strategy.supports({"mode": "socialjax_grid", "grid": [], "agent_locs": []}) is True
        assert strategy.supports({"mode": "rgb_array", "rgb": []}) is False
        assert strategy.supports({"grid": [[0, 1]]}) is False  # no mode field


# ---------------------------------------------------------------------------
# Subprocess integration tests -- require real checkpoint
# ---------------------------------------------------------------------------

class _WorkerProcess:
    """Thin subprocess wrapper (same as test_interactive_protocol.py)."""

    def __init__(self, env_id: str, policy_path: Path) -> None:
        import subprocess
        env = os.environ.copy()
        env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        self._proc = subprocess.Popen(
            [_PYTHON, "-m", "jaxmarl_worker.cli",
             "--interactive", "--env-id", env_id, "--policy-path", str(policy_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0, env=env, cwd=str(_JAXMARL),
        )

    def send(self, cmd: dict) -> None:
        self._proc.stdin.write((json.dumps(cmd) + "\n").encode())
        self._proc.stdin.flush()

    def read_response(self, timeout: float = 90.0) -> dict:
        start = time.monotonic()
        buf = b""
        while True:
            if time.monotonic() - start > timeout:
                raise TimeoutError(f"No response within {timeout}s")
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


@needs_ckpt
class TestGridPayloadSubprocess:
    """Integration tests: verify the worker emits compact grid payloads."""

    ENV_ID = "socialjax/coop_mining"

    def test_reset_emits_socialjax_grid_payload(self) -> None:
        """After reset, render_payload.mode must be 'socialjax_grid'."""
        with _WorkerProcess(self.ENV_ID, _CKPT) as wp:
            wp.send({"cmd": "reset", "seed": 0})
            resp = wp.read_response(timeout=90)

        assert resp["type"] == "ready"
        rp = resp.get("render_payload")
        assert rp is not None, "render_payload missing from ready response"
        assert rp["mode"] == "socialjax_grid", (
            f"Expected mode='socialjax_grid', got {rp.get('mode')!r}. "
            "Worker may still be sending the old rgb_array payload."
        )

    def test_grid_payload_is_small(self) -> None:
        """Payload JSON must be well under 100KB (not 12MB as with raw RGB)."""
        with _WorkerProcess(self.ENV_ID, _CKPT) as wp:
            wp.send({"cmd": "reset", "seed": 1})
            resp = wp.read_response(timeout=90)

        rp = resp.get("render_payload", {})
        size_kb = len(json.dumps(rp)) / 1024
        print(f"\n  Grid payload size: {size_kb:.1f}KB")
        assert size_kb < 100, (
            f"Grid payload is {size_kb:.1f}KB -- expected < 100KB. "
            "The rgb.tolist() bottleneck may still be active."
        )

    def test_agent_locs_change_across_steps(self) -> None:
        """Agent positions must differ between step 0 and some later step.

        This is the core regression check: if agents stand idle the locs will be
        identical for every step.
        """
        all_locs = []

        with _WorkerProcess(self.ENV_ID, _CKPT) as wp:
            wp.send({"cmd": "reset", "seed": 7})
            resp = wp.read_response(timeout=90)
            rp = resp.get("render_payload")
            if rp and rp.get("mode") == "socialjax_grid":
                all_locs.append(str(rp["agent_locs"]))

            for _ in range(20):
                wp.send({"cmd": "step"})
                resp = wp.read_response(timeout=30)
                rp = resp.get("render_payload")
                if rp and rp.get("mode") == "socialjax_grid":
                    all_locs.append(str(rp["agent_locs"]))
                if resp["type"] == "episode_done":
                    break

        if len(all_locs) < 2:
            pytest.skip("Not enough frames to compare")

        assert any(locs != all_locs[0] for locs in all_locs[1:]), (
            "All 20 steps returned identical agent_locs -- agents appear frozen. "
            "Policy may be degenerate or agent_locs is not updating in the payload."
        )

    def test_grid_contains_valid_tile_codes(self) -> None:
        """All tile codes in the grid must have a colour entry in tile_colors."""
        with _WorkerProcess(self.ENV_ID, _CKPT) as wp:
            wp.send({"cmd": "reset", "seed": 2})
            resp = wp.read_response(timeout=90)

        rp = resp.get("render_payload", {})
        grid = rp.get("grid", [])
        tile_colors = rp.get("tile_colors", {})

        unknown = set()
        for row in grid:
            for code in row:
                if str(code) not in tile_colors:
                    unknown.add(code)

        assert not unknown, f"Grid contains tile codes with no colour mapping: {unknown}"

    def test_grid_dimensions_match_n_agents(self) -> None:
        """Grid must be non-empty and n_agents must match agent_locs count."""
        with _WorkerProcess(self.ENV_ID, _CKPT) as wp:
            wp.send({"cmd": "reset", "seed": 3})
            resp = wp.read_response(timeout=90)

        rp = resp.get("render_payload", {})
        grid = rp.get("grid", [])
        agent_locs = rp.get("agent_locs", [])
        n_agents = rp.get("n_agents", -1)

        assert len(grid) > 0, "Grid is empty"
        assert len(grid[0]) > 0, "Grid rows are empty"
        assert len(agent_locs) == n_agents, (
            f"agent_locs length {len(agent_locs)} != n_agents {n_agents}"
        )
