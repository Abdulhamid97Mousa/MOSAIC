import pytest
from PyQt6 import QtCore

from gym_gui.config.settings import Settings
from gym_gui.controllers.interaction import (
    AleInteractionController,
    Box2DInteractionController,
    TurnBasedInteractionController,
)
from gym_gui.controllers.malmo_interaction import MalmoInteractionController
from gym_gui.controllers.session import SessionController
from gym_gui.core.adapters.base import AdapterStep
from gym_gui.core.enums import ControlMode, EnvironmentFamily, GameId


@pytest.fixture(scope="module", autouse=True)
def _qt_core_app():
    app = QtCore.QCoreApplication.instance()
    if app is None:
        app = QtCore.QCoreApplication([])
    yield app


def _make_session() -> SessionController:
    return SessionController(Settings())


def test_session_selects_ale_interaction_for_ale_family():
    session = _make_session()
    controller = session._create_interaction_controller(EnvironmentFamily.ALE)
    assert isinstance(controller, AleInteractionController)


def test_session_selects_box2d_controller_for_box2d_family():
    session = _make_session()
    controller = session._create_interaction_controller(EnvironmentFamily.BOX2D)
    assert isinstance(controller, Box2DInteractionController)


def test_session_defaults_to_turn_based_controller():
    session = _make_session()
    controller = session._create_interaction_controller(EnvironmentFamily.TOY_TEXT)
    assert isinstance(controller, TurnBasedInteractionController)


def test_session_selects_malmo_controller_for_malmo_family():
    session = _make_session()
    controller = session._create_interaction_controller(EnvironmentFamily.MALMOENV)
    assert isinstance(controller, MalmoInteractionController)


def test_idle_tick_requires_game_started():
    session = _make_session()
    session._adapter = object()  # type: ignore[assignment]
    session._game_id = GameId.ALE_ASSAULT_V5
    session._control_mode = ControlMode.HUMAN_ONLY
    session._passive_action = 0
    session._last_step = AdapterStep(
        observation=None,
        reward=0.0,
        terminated=False,
        truncated=False,
        info={},
    )
    session._interaction = session._create_interaction_controller(EnvironmentFamily.ALE)

    assert session._should_idle_tick() is False
    session._game_started = True
    assert session._should_idle_tick() is True
