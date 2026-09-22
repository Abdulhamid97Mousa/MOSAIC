"""Scenario specifications for ViZDoom environments.

Each entry maps an xuance env_id (as listed in xuance_train_form.py:338)
to the concrete ViZDoom engine configuration needed to load it: which
.cfg file under vizdoom.scenarios_path, which buttons the agent controls,
and which screen buffer format to render.

Deathmatch is intentionally omitted; it is multi-agent and requires
REGISTRY_MULTI_AGENT_ENV registration, out of scope for this plan.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ViZDoomScenarioSpec:
    scenario_cfg: str
    available_buttons: tuple[str, ...]
    screen_format: str = "RGB24"
    screen_resolution: str = "RES_320X240"
    frame_repeat: int = 4


VIZDOOM_SCENARIOS: dict[str, ViZDoomScenarioSpec] = {
    "ViZDoom-Basic-v0": ViZDoomScenarioSpec(
        scenario_cfg="basic.cfg",
        available_buttons=("MOVE_LEFT", "MOVE_RIGHT", "ATTACK"),
    ),
    "ViZDoom-DeadlyCorridor-v0": ViZDoomScenarioSpec(
        scenario_cfg="deadly_corridor.cfg",
        available_buttons=(
            "MOVE_LEFT", "MOVE_RIGHT", "ATTACK",
            "MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT",
        ),
    ),
    "ViZDoom-DefendTheCenter-v0": ViZDoomScenarioSpec(
        scenario_cfg="defend_the_center.cfg",
        available_buttons=("TURN_LEFT", "TURN_RIGHT", "ATTACK"),
    ),
    "ViZDoom-DefendTheLine-v0": ViZDoomScenarioSpec(
        scenario_cfg="defend_the_line.cfg",
        available_buttons=("TURN_LEFT", "TURN_RIGHT", "ATTACK"),
    ),
    "ViZDoom-HealthGathering-v0": ViZDoomScenarioSpec(
        scenario_cfg="health_gathering.cfg",
        available_buttons=("TURN_LEFT", "TURN_RIGHT", "MOVE_FORWARD"),
    ),
    "ViZDoom-HealthGatheringSupreme-v0": ViZDoomScenarioSpec(
        scenario_cfg="health_gathering_supreme.cfg",
        available_buttons=("TURN_LEFT", "TURN_RIGHT", "MOVE_FORWARD"),
    ),
    "ViZDoom-MyWayHome-v0": ViZDoomScenarioSpec(
        scenario_cfg="my_way_home.cfg",
        available_buttons=("TURN_LEFT", "TURN_RIGHT", "MOVE_FORWARD"),
    ),
    "ViZDoom-PredictPosition-v0": ViZDoomScenarioSpec(
        scenario_cfg="predict_position.cfg",
        available_buttons=("TURN_LEFT", "TURN_RIGHT", "ATTACK"),
    ),
    "ViZDoom-TakeCover-v0": ViZDoomScenarioSpec(
        scenario_cfg="take_cover.cfg",
        available_buttons=("MOVE_LEFT", "MOVE_RIGHT"),
    ),
}
