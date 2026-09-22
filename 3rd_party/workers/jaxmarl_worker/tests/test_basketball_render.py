"""Regression tests for native v7 sports rendering in live JaxMARL runs."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


def _state():
    return SimpleNamespace(
        agent_pos=np.array([[8, 3], [8, 7]], dtype=np.int32),
        agent_dir=np.array([0, 2], dtype=np.int32),
        agent_team=np.array([0, 0], dtype=np.int32),
        agent_carrying=np.array([-1, -1], dtype=np.int32),
        ball_pos=np.array([[9, 5]], dtype=np.int32),
        ball_carried_by=np.array([-1], dtype=np.int32),
        scores=np.array([0, 0], dtype=np.int32),
        step=np.int32(0),
    )


def test_live_basketball_uses_native_19_by_11_geometry(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")

    from jaxmarl_worker.runtime import _render_state

    frame = _render_state(_state(), sport="bb", view_size=7)

    assert frame.shape == (11 * 32, 19 * 32, 3)
    assert frame.dtype == np.uint8


def test_live_basketball_has_single_cell_hoops(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")

    from jaxmarl_worker.runtime import _render_state

    frame = _render_state(_state(), sport="bb", view_size=7)

    # Hoop cells are (1, 5) and (17, 5). Adjacent baseline cells must retain
    # the native paint/floor color instead of becoming five-cell goal columns.
    left_hoop_center = frame[5 * 32 + 16, 1 * 32 + 16]
    left_adjacent_center = frame[3 * 32 + 16, 1 * 32 + 16]
    right_hoop_center = frame[5 * 32 + 16, 17 * 32 + 16]
    right_adjacent_center = frame[3 * 32 + 16, 17 * 32 + 16]

    assert tuple(left_hoop_center) != tuple(left_adjacent_center)
    assert tuple(right_hoop_center) != tuple(right_adjacent_center)
    assert left_hoop_center[1] > left_hoop_center[0]
    assert right_hoop_center[2] > right_hoop_center[0]


@pytest.mark.parametrize(
    ("sport", "goal_rows", "expected_shape"),
    [
        ("soccer", [4, 5, 6], (11 * 32, 16 * 32, 3)),
        ("af", list(range(1, 10)), (11 * 32, 16 * 32, 3)),
        ("bb", [5], (11 * 32, 19 * 32, 3)),
    ],
)
def test_all_live_sports_use_native_v7_geometry(
    monkeypatch, sport, goal_rows, expected_shape
):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")

    from jaxmarl_worker.runtime import _render_state

    jax_env = SimpleNamespace(_goal_rows=np.asarray(goal_rows, dtype=np.int32))
    frame = _render_state(_state(), sport=sport, view_size=7, jax_env=jax_env)

    assert frame.shape == expected_shape
    assert frame.dtype == np.uint8


def test_live_soccer_uses_running_environment_goal_rows(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")

    from jaxmarl_worker.runtime import _render_state

    # This deliberately differs from the default [4, 5, 6]. It proves the
    # adapter reads the running environment instead of embedding goal cells.
    jax_env = SimpleNamespace(_goal_rows=np.asarray([5], dtype=np.int32))
    frame = _render_state(_state(), "soccer", 7, jax_env)

    goal_center = frame[5 * 32 + 16, 1 * 32 + 16]
    adjacent_center = frame[4 * 32 + 16, 1 * 32 + 16]
    assert tuple(goal_center) != tuple(adjacent_center)


def test_live_football_uses_full_native_endzones(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")

    from jaxmarl_worker.runtime import _render_state

    jax_env = SimpleNamespace(
        _goal_rows=np.asarray(list(range(1, 10)), dtype=np.int32)
    )
    frame = _render_state(_state(), "af", 7, jax_env)

    left_top = frame[1 * 32 + 16, 1 * 32 + 16]
    left_bottom = frame[9 * 32 + 16, 1 * 32 + 16]
    right_top = frame[1 * 32 + 16, 14 * 32 + 16]
    assert left_top[1] > left_top[2]
    assert left_bottom[1] > left_bottom[2]
    assert right_top[2] > right_top[1]
