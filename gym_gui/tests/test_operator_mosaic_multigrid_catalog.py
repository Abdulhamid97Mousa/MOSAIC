"""Operator environment catalog parity tests for mosaic_multigrid v7."""

from __future__ import annotations

import pytest

from gym_gui.services.operator_environment_catalog import (
    MOSAIC_MULTIGRID_ENVIRONMENTS,
    get_mosaic_multigrid_environments,
)


def test_v7_catalog_uses_current_namespace_and_has_full_inventory() -> None:
    assert len(MOSAIC_MULTIGRID_ENVIRONMENTS) == 81
    assert all(
        env_id.startswith("MosaicMultiGrid-")
        for env_id in MOSAIC_MULTIGRID_ENVIRONMENTS
    )
    assert not any("TeamObs" in env_id for env_id in MOSAIC_MULTIGRID_ENVIRONMENTS)
    assert "MosaicMultiGrid-S-4v4-IndAgObs-v1" in MOSAIC_MULTIGRID_ENVIRONMENTS
    assert "MosaicMultiGrid-BB-1v2-IndAgObs-v1" in MOSAIC_MULTIGRID_ENVIRONMENTS
    assert "MosaicMultiGrid-AF-G-6v0-IndAgObs-v1" in MOSAIC_MULTIGRID_ENVIRONMENTS


def test_operator_catalog_matches_installed_gymnasium_registry() -> None:
    gymnasium = pytest.importorskip("gymnasium")
    pytest.importorskip("mosaic_multigrid.envs")

    registered = {
        env_id
        for env_id in gymnasium.registry
        if env_id.startswith("MosaicMultiGrid-")
    }

    assert registered
    assert set(get_mosaic_multigrid_environments()) == registered
