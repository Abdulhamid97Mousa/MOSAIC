from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cleanrl_worker.runtime import DEFAULT_ALGO_REGISTRY

SCHEMA_PATH = REPO_ROOT / "metadata" / "cleanrl" / "0.1.0" / "schemas.json"


def test_registry_contains_extended_algorithms() -> None:
    required = {
        "ppo",
        "ppo_atari_envpool",
        "td3_continuous_action",
        "sac_continuous_action",
        "rainbow_atari",
    }
    assert required.issubset(DEFAULT_ALGO_REGISTRY.keys())


def test_schema_includes_ppo_learning_rate_field() -> None:
    assert SCHEMA_PATH.exists(), f"Missing schema file: {SCHEMA_PATH}"
    data = json.loads(SCHEMA_PATH.read_text())
    algorithms = data.get("algorithms", {})
    ppo = algorithms.get("ppo")
    assert ppo is not None, "ppo entry missing from schema"
    field_names = {field.get("name") for field in ppo.get("fields", [])}
    assert "learning_rate" in field_names, "learning_rate field missing for ppo"
