"""Tests for Malmo adapter info parsing and HUD payload mapping."""

from __future__ import annotations

import numpy as np

from gym_gui.core.adapters.mosaic_malmo import MalmoEnvAdapter


def test_parse_info_payload_from_json_string() -> None:
    info = '{"Life":20,"Food":18,"XPos":4.5}'
    parsed = MalmoEnvAdapter._parse_info_payload(info)
    assert parsed["Life"] == 20
    assert parsed["Food"] == 18
    assert parsed["XPos"] == 4.5


def test_parse_info_payload_invalid_string_keeps_raw() -> None:
    parsed = MalmoEnvAdapter._parse_info_payload("not-json")
    assert parsed["malmo_info_raw"] == "not-json"


def test_normalize_info_maps_hud_fields() -> None:
    info = (
        '{"Life":20.0,"Food":19,"Air":300,"XP":7,"Score":5,"IsAlive":true,'
        '"XPos":4.5,"YPos":55.0,"ZPos":1.25,"Yaw":-90.0,"Pitch":10.0,"TotalTime":12345}'
    )
    normalized = MalmoEnvAdapter._normalize_info(
        info,
        reward=1.5,
        episode_step=3,
        episode_score=2.5,
    )

    assert normalized["episode_step"] == 3
    assert normalized["episode_score"] == 2.5
    assert normalized["reward"] == 1.5
    assert normalized["agent_pos"] == [4.5, 55.0, 1.25]
    assert normalized["agent_yaw"] == -90.0
    assert normalized["agent_pitch"] == 10.0
    assert normalized["tick"] == 12345
    assert normalized["ms_per_tick"] == 50
    assert normalized["life"] == 20.0
    assert normalized["food"] == 19
    assert normalized["air"] == 300
    assert normalized["xp"] == 7
    assert normalized["score"] == 5
    assert normalized["is_alive"] is True


def test_render_includes_hud_fields_from_last_info() -> None:
    adapter = object.__new__(MalmoEnvAdapter)
    adapter.mission_xml = "attic.xml"
    adapter._last_obs = np.zeros((2, 2, 3), dtype=np.uint8)
    adapter._malmo_env = None
    adapter._last_info = {
        "life": 20,
        "food": 18,
        "air": 300,
        "xp": 4,
        "score": 2,
        "is_alive": True,
        "agent_pos": [1.0, 2.0, 3.0],
        "agent_yaw": 90.0,
        "agent_pitch": 0.0,
        "tick": 42,
        "reward": 1.0,
        "ms_per_tick": 50,
    }

    payload = MalmoEnvAdapter.render(adapter)
    assert payload["mode"] == "mosaic_malmo"
    assert payload["mission"] == "attic"
    assert payload["life"] == 20
    assert payload["food"] == 18
    assert payload["air"] == 300
    assert payload["xp"] == 4
    assert payload["score"] == 2
    assert payload["is_alive"] is True
