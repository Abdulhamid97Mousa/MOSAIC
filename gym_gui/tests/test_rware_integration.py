"""Test RWARE (Robotic Warehouse) integration with MOSAIC.

Tests that RWARE environments:
1. Have GameIds and EnvironmentFamily registered in MOSAIC enums
2. Have game documentation registered
3. Have adapter classes wired in the factory
4. Support the full load -> reset -> step -> render -> close lifecycle
5. Have config panel and keyboard input support
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Test 1: Enum registration
# ---------------------------------------------------------------------------
class TestRWAREEnumRegistration:
    """Test that RWARE GameIds and families are registered in enums."""

    def test_rware_family_exists(self) -> None:
        from gym_gui.core.enums import EnvironmentFamily

        assert hasattr(EnvironmentFamily, "RWARE")
        assert EnvironmentFamily.RWARE.value == "rware"

    def test_rware_game_ids_exist(self) -> None:
        from gym_gui.core.enums import GameId

        expected = [
            "RWARE_TINY_2AG",
            "RWARE_TINY_4AG",
            "RWARE_SMALL_2AG",
            "RWARE_SMALL_4AG",
            "RWARE_MEDIUM_2AG",
            "RWARE_MEDIUM_4AG",
            "RWARE_MEDIUM_4AG_EASY",
            "RWARE_MEDIUM_4AG_HARD",
            "RWARE_LARGE_4AG",
            "RWARE_LARGE_4AG_HARD",
            "RWARE_LARGE_8AG",
            "RWARE_LARGE_8AG_HARD",
        ]
        for name in expected:
            assert hasattr(GameId, name), f"GameId.{name} not found"

    def test_rware_in_family_mapping(self) -> None:
        from gym_gui.core.enums import (
            ENVIRONMENT_FAMILY_BY_GAME,
            EnvironmentFamily,
            GameId,
        )

        rware_ids = [
            GameId.RWARE_TINY_2AG,
            GameId.RWARE_TINY_4AG,
            GameId.RWARE_SMALL_2AG,
            GameId.RWARE_SMALL_4AG,
            GameId.RWARE_MEDIUM_2AG,
            GameId.RWARE_MEDIUM_4AG,
            GameId.RWARE_MEDIUM_4AG_EASY,
            GameId.RWARE_MEDIUM_4AG_HARD,
            GameId.RWARE_LARGE_4AG,
            GameId.RWARE_LARGE_4AG_HARD,
            GameId.RWARE_LARGE_8AG,
            GameId.RWARE_LARGE_8AG_HARD,
        ]
        for gid in rware_ids:
            assert gid in ENVIRONMENT_FAMILY_BY_GAME
            assert ENVIRONMENT_FAMILY_BY_GAME[gid] == EnvironmentFamily.RWARE

    def test_rware_stepping_paradigm(self) -> None:
        from gym_gui.core.enums import (
            DEFAULT_PARADIGM_BY_FAMILY,
            EnvironmentFamily,
            SteppingParadigm,
        )

        assert EnvironmentFamily.RWARE in DEFAULT_PARADIGM_BY_FAMILY
        assert (
            DEFAULT_PARADIGM_BY_FAMILY[EnvironmentFamily.RWARE]
            == SteppingParadigm.SIMULTANEOUS
        )


# ---------------------------------------------------------------------------
# Test 2: Game documentation
# ---------------------------------------------------------------------------
class TestRWAREDocumentation:
    """Test RWARE game documentation is registered."""

    def test_rware_game_info_registered(self) -> None:
        from gym_gui.core.enums import GameId
        from gym_gui.game_docs import get_game_info

        for gid in (GameId.RWARE_TINY_2AG, GameId.RWARE_LARGE_8AG):
            doc = get_game_info(gid)
            assert isinstance(doc, str)
            assert len(doc) > 0
            assert "Documentation unavailable" not in doc

    def test_documentation_has_warehouse_content(self) -> None:
        from gym_gui.core.enums import GameId
        from gym_gui.game_docs import get_game_info

        doc = get_game_info(GameId.RWARE_TINY_2AG)
        # Should mention warehouse or RWARE concepts
        lower = doc.lower()
        assert "warehouse" in lower or "rware" in lower or "robotic" in lower


# ---------------------------------------------------------------------------
# Test 3: Config dataclass
# ---------------------------------------------------------------------------
class TestRWAREConfig:
    """Test RWAREConfig dataclass has expected defaults."""

    def test_config_defaults(self) -> None:
        from gym_gui.config.game_configs import RWAREConfig

        cfg = RWAREConfig()
        assert cfg.observation_type == "flattened"
        assert cfg.sensor_range == 1
        assert cfg.reward_type == "individual"
        assert cfg.msg_bits == 0
        assert cfg.max_steps == 500
        assert cfg.seed is None
        assert cfg.render_mode == "rgb_array"


# ---------------------------------------------------------------------------
# Test 4: Adapter factory wiring
# ---------------------------------------------------------------------------
class TestRWAREAdapterFactory:
    """Test RWARE adapters are registered in the factory."""

    def test_adapter_registered(self) -> None:
        from gym_gui.core.enums import GameId
        from gym_gui.core.factories.adapters import create_adapter

        # Should not raise for a registered RWARE game
        adapter = create_adapter(GameId.RWARE_TINY_2AG)
        assert adapter is not None

    def test_all_rware_adapters_registered(self) -> None:
        from gym_gui.core.factories.adapters import create_adapter
        from gym_gui.ui.config_panels.multi_agent.rware import ALL_RWARE_GAME_IDS

        for gid in ALL_RWARE_GAME_IDS:
            adapter = create_adapter(gid)
            assert adapter is not None, f"No adapter for {gid}"

    def test_ensure_control_mode_exists(self) -> None:
        """Adapter inherits ensure_control_mode from EnvironmentAdapter."""
        from gym_gui.core.enums import ControlMode, GameId
        from gym_gui.core.factories.adapters import create_adapter

        adapter = create_adapter(GameId.RWARE_TINY_2AG)
        # Should not raise for supported modes
        adapter.ensure_control_mode(ControlMode.AGENT_ONLY)
        adapter.ensure_control_mode(ControlMode.HUMAN_ONLY)

    def test_create_adapter_with_context(self) -> None:
        """Adapter works when created via factory with context (the runtime path)."""
        from gym_gui.core.adapters.base import AdapterContext
        from gym_gui.core.enums import ControlMode, GameId
        from gym_gui.core.factories.adapters import create_adapter

        context = AdapterContext(settings=None, control_mode=ControlMode.AGENT_ONLY)
        adapter = create_adapter(GameId.RWARE_LARGE_4AG_HARD, context)
        assert adapter is not None
        assert hasattr(adapter, "ensure_control_mode")


# ---------------------------------------------------------------------------
# Test 5: Full lifecycle (requires rware installed)
# ---------------------------------------------------------------------------
class TestRWARELifecycle:
    """Test full load -> reset -> step -> render -> close cycle."""

    @pytest.fixture
    def adapter(self):
        """Create a tiny-2ag adapter for lifecycle testing."""
        pytest.importorskip("rware", reason="rware not installed")
        from gym_gui.core.enums import GameId
        from gym_gui.core.factories.adapters import create_adapter

        adpt = create_adapter(GameId.RWARE_TINY_2AG)
        yield adpt
        try:
            adpt.close()
        except Exception:
            pass

    def test_load_and_reset(self, adapter) -> None:
        from gym_gui.core.adapters.base import AdapterStep

        adapter.load()
        result = adapter.reset()
        assert isinstance(result, AdapterStep)
        assert result.observation is not None

    def test_step_returns_adapter_step(self, adapter) -> None:
        from gym_gui.core.adapters.base import AdapterStep

        adapter.load()
        adapter.reset()
        # RWARE tiny-2ag has 2 agents with 5 actions each
        # Action space is Tuple(Discrete(5), Discrete(5))
        action = adapter.action_space.sample()
        result = adapter.step(action)
        assert isinstance(result, AdapterStep)
        assert result.observation is not None
        assert isinstance(result.reward, (int, float))
        assert isinstance(result.terminated, bool)
        assert isinstance(result.truncated, bool)

    def test_render_returns_payload_dict(self, adapter) -> None:
        """Render returns a dict with 'rgb' key for MOSAIC's Render View."""
        adapter.load()
        adapter.reset()
        payload = adapter.render()
        assert payload is not None
        assert isinstance(payload, dict), "render_payload must be a dict, not raw array"
        assert "rgb" in payload, "dict must contain 'rgb' key"
        assert "mode" in payload
        frame = payload["rgb"]
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        assert frame.shape[2] == 3  # RGB

    def test_close_succeeds(self, adapter) -> None:
        adapter.load()
        adapter.reset()
        adapter.close()  # should not raise


# ---------------------------------------------------------------------------
# Test 7: Config panel
# ---------------------------------------------------------------------------
class TestRWAREConfigPanel:
    """Test RWARE config panel imports and exports are correct."""

    def test_config_panel_imports(self) -> None:
        from gym_gui.ui.config_panels.multi_agent.rware import (
            ALL_RWARE_GAME_IDS,
            build_rware_controls,
        )

        assert len(ALL_RWARE_GAME_IDS) == 12
        assert callable(build_rware_controls)

    def test_all_game_ids_consistent_with_enums(self) -> None:
        from gym_gui.core.enums import ENVIRONMENT_FAMILY_BY_GAME, EnvironmentFamily, GameId
        from gym_gui.ui.config_panels.multi_agent.rware import ALL_RWARE_GAME_IDS

        for gid in ALL_RWARE_GAME_IDS:
            assert isinstance(gid, GameId)
            assert ENVIRONMENT_FAMILY_BY_GAME[gid] == EnvironmentFamily.RWARE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
