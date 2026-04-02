import pytest

from gym_gui.controllers.interaction import (
    AleInteractionController,
    Box2DInteractionController,
    TurnBasedInteractionController,
)
from gym_gui.controllers.malmo_interaction import MalmoInteractionController


class _OwnerStub:
    def __init__(self) -> None:
        class _CM:
            name = "HUMAN_ONLY"
        self._control_mode = _CM()
        self._game_paused = False
        self._passive_action = 0
        self._game_id = object()
        self._game_started = True
        self._last_step = None

        # Minimal adapter stub with no action_space (e.g. Malmo before env load)
        class _AdapterStub:
            action_space = None
        self._adapter = _AdapterStub()


def test_turn_based_controller_behavior():
    tb = TurnBasedInteractionController()
    assert tb.idle_interval_ms() is None
    assert tb.should_idle_tick() is False
    assert tb.maybe_passive_action() is None
    assert tb.step_dt() == 0.0


def test_box2d_controller_defaults():
    owner = _OwnerStub()
    ctrl = Box2DInteractionController(owner, target_hz=50)
    assert ctrl.idle_interval_ms() == 20  # 1000/50 ms
    assert ctrl.should_idle_tick() is True
    assert ctrl.step_dt() == pytest.approx(1.0 / 50.0, rel=1e-3)


def test_ale_controller_noop_and_interval():
    owner = _OwnerStub()
    ctrl = AleInteractionController(owner, target_hz=60)
    assert ctrl.idle_interval_ms() in (16, 17)  # ~16.6ms
    assert ctrl.should_idle_tick() is True
    assert ctrl.maybe_passive_action() == 0


def test_malmo_controller_interval_and_passive_action_fallback():
    owner = _OwnerStub()
    ctrl = MalmoInteractionController(owner, target_hz=20)
    assert ctrl.idle_interval_ms() == 50
    assert ctrl.should_idle_tick() is True
    # With no action_space, falls back to action 0
    assert ctrl.maybe_passive_action() == 0


essential_dummy = object()  # keep module non-empty for pytest collectors


def test_session_components_constructs_without_runtime_imports():
    from gym_gui.core.session_components import SessionComponents

    class DummyAdapter:  # minimal stub
        pass

    class DummyRenderer:  # minimal stub
        pass

    comp = SessionComponents(
        adapter=DummyAdapter(),  # type: ignore[arg-type]
        renderer=DummyRenderer(),  # type: ignore[arg-type]
        interaction=TurnBasedInteractionController(),
    )
    assert comp.adapter is not None and comp.renderer is not None and comp.interaction is not None
