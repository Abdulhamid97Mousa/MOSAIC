"""Custom environment wrappers for XuanCe training in MOSAIC.

This module provides XuanCe-compatible environment wrappers that extend
RawMultiAgentEnv for environments not natively supported by XuanCe.

These wrappers enable MOSAIC to train multi-agent RL policies on additional
environment families while maintaining compatibility with XuanCe's training
infrastructure (runners, agents, etc.).

Usage:
    # Register environments with XuanCe at startup
    from xuance_worker.environments import register_mosaic_environments
    register_mosaic_environments()

    # Or import specific environments
    from xuance_worker.environments import MultiGrid_Env
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

_logger = logging.getLogger(__name__)

# Lazy imports to avoid circular dependencies
if TYPE_CHECKING:
    from xuance_worker.environments.multigrid.multigrid import MultiGrid_Env

_REGISTERED = False


def register_mosaic_environments() -> None:
    """Register MOSAIC custom environments with XuanCe's registry.

    This function adds MOSAIC's environment wrappers to XuanCe's
    REGISTRY_MULTI_AGENT_ENV, enabling them to be used with XuanCe's
    training infrastructure.

    This should be called once at startup, typically in xuance_worker's
    runtime initialization.

    Example:
        >>> from xuance_worker.environments import register_mosaic_environments
        >>> register_mosaic_environments()
        >>> # Now 'multigrid' is available as an env_name in XuanCe configs
    """
    global _REGISTERED

    if _REGISTERED:
        _logger.debug("MOSAIC environments already registered with XuanCe")
        return

    try:
        from xuance.environment.multi_agent_env import REGISTRY_MULTI_AGENT_ENV
        from xuance_worker.environments.multigrid.multigrid import MultiGrid_Env

        # Register MultiGrid (multi-agent)
        if "multigrid" not in REGISTRY_MULTI_AGENT_ENV:
            REGISTRY_MULTI_AGENT_ENV["multigrid"] = MultiGrid_Env
            REGISTRY_MULTI_AGENT_ENV["MultiGrid"] = MultiGrid_Env
            _logger.info("Registered MultiGrid_Env with XuanCe registry")

        # Register Solo MultiGrid (single-agent PPO)
        try:
            from xuance.environment.single_agent_env import REGISTRY_ENV
            from xuance_worker.environments.multigrid.multigrid import SoloMultiGrid_Env

            if "multigrid_sports" not in REGISTRY_ENV:
                REGISTRY_ENV["multigrid_sports"] = SoloMultiGrid_Env
                _logger.info("Registered SoloMultiGrid_Env with XuanCe single-agent registry")
        except ImportError as e:
            _logger.warning(f"Could not register SoloMultiGrid_Env: {e}")

        # Register MeltingPot PD (multi-agent)
        try:
            from xuance_worker.environments.meltingpot.pd import MeltingPot_PD_Env

            if "meltingpot" not in REGISTRY_MULTI_AGENT_ENV:
                REGISTRY_MULTI_AGENT_ENV["meltingpot"] = MeltingPot_PD_Env
                _logger.info("Registered MeltingPot_PD_Env with XuanCe registry")
        except ImportError as e:
            _logger.warning(f"Could not register MeltingPot_PD_Env: {e}")

        # Register MeltingPot Paintball Capture-the-Flag (multi-agent, 4v4).
        # Registry key mirrors the substrate name so each substrate that
        # needs bespoke handling (shaping kwargs, CNN vs MLP path, etc.)
        # can live under its own "meltingpot_<substrate>" entry without
        # colliding with the generic "meltingpot" PD entry above.
        try:
            from xuance_worker.environments.meltingpot.paintball_ctf import (
                MeltingPot_PaintballCaptureTheFlag_Env,
            )

            key = "meltingpot_paintball__capture_the_flag"
            if key not in REGISTRY_MULTI_AGENT_ENV:
                REGISTRY_MULTI_AGENT_ENV[key] = MeltingPot_PaintballCaptureTheFlag_Env
                _logger.info(
                    "Registered MeltingPot_PaintballCaptureTheFlag_Env "
                    "with XuanCe registry under key %s",
                    key,
                )
        except ImportError as e:
            _logger.warning(
                f"Could not register MeltingPot_PaintballCaptureTheFlag_Env: {e}"
            )

        _REGISTERED = True
        _logger.info("MOSAIC environments registered with XuanCe successfully")

    except ImportError as e:
        _logger.warning(f"Could not register MOSAIC environments with XuanCe: {e}")
    except Exception as e:
        _logger.error(f"Error registering MOSAIC environments: {e}")

    # Independent block: register FastLane-instrumented GRF wrapper. Kept
    # outside the main try/except so that a failure elsewhere in that block
    # (e.g. missing multigrid module) does not short-circuit the
    # Football override.
    #
    # The wrapper swaps in for xuance's upstream 'Football' registry entry
    # so the training-form factory picks it up when users select
    # Multi-Agent -> Football. When MOSAIC_FASTLANE_ENABLED is NOT set in
    # the subprocess environment, the wrapper's __init__ short-circuits to
    # a pure pass-through, so this override is safe for non-FastLane runs
    # too.
    try:
        from xuance.environment.multi_agent_env import REGISTRY_MULTI_AGENT_ENV as _REG
        # New family-per-directory layout: environments/gfootball/gfootball.py
        # (was environments/football_fastlane.py before the 2026-08-13 reorg).
        from xuance_worker.environments.gfootball import (
            GFootballFastLane_Env,
        )
        if "Football" in _REG and not isinstance(_REG["Football"], str):
            # Override xuance's original registry entry (capital "Football")
            _REG["Football"] = GFootballFastLane_Env
            # Also expose under the upstream Python package name
            # (`import gfootball`). This is the key the xuance training form
            # uses as `env_family="gfootball"`; xuance's runtime looks up
            # config.env_name in this dict, so both keys must resolve.
            _REG["gfootball"] = GFootballFastLane_Env
            _logger.info(
                "Registered GFootballFastLane_Env under registry keys "
                "'Football' (xuance legacy) and 'gfootball' (upstream package name)"
            )
        else:
            _logger.debug(
                "Skipping GRF FastLane override: xuance 'Football' entry missing "
                "or unavailable (%r)",
                _REG.get("Football"),
            )
    except ImportError as e:
        _logger.warning(f"Could not register GFootballFastLane_Env: {e}")
    except Exception as e:
        _logger.error(f"Error registering GFootballFastLane_Env: {e}")

    # Independent block: register FastLane-instrumented SMAC wrapper.
    # Overrides xuance's upstream 'StarCraft2' registry entry. When
    # MOSAIC_FASTLANE_ENABLED is NOT set, the wrapper's __init__
    # short-circuits to identical behaviour as xuance's original
    # StarCraft2_Env, so this override is safe for non-FastLane runs.
    try:
        from xuance.environment.multi_agent_env import REGISTRY_MULTI_AGENT_ENV as _REG
        from xuance_worker.environments.smac import StarCraft2FastLane_Env

        if "StarCraft2" in _REG and not isinstance(_REG["StarCraft2"], str):
            _REG["StarCraft2"] = StarCraft2FastLane_Env
            _logger.info(
                "Registered StarCraft2FastLane_Env under registry key 'StarCraft2'"
            )
        else:
            _logger.debug(
                "Skipping SMAC FastLane override: xuance 'StarCraft2' entry "
                "missing or unavailable (%r)",
                _REG.get("StarCraft2"),
            )
    except ImportError as e:
        _logger.warning(f"Could not register StarCraft2FastLane_Env: {e}")
    except Exception as e:
        _logger.error(f"Error registering StarCraft2FastLane_Env: {e}")

    # Independent block: register SMACv2 FastLane wrapper under 'StarCraft2v2'.
    # SMACv2 uses StarCraftCapabilityEnvWrapper (procedural unit injection via
    # SC2 debug API) and cannot share StarCraft2FastLane_Env (SMAC v1 reads
    # units pre-placed in .SC2Map files). A fresh registry key keeps them
    # fully independent; XuanCe's RunnerStarCraft2 is reused via YAML field
    # runner: "RunnerStarCraft2".
    try:
        from xuance.environment.multi_agent_env import REGISTRY_MULTI_AGENT_ENV as _REG
        from xuance_worker.environments.smacv2 import StarCraft2v2FastLane_Env

        _REG["StarCraft2v2"] = StarCraft2v2FastLane_Env
        _logger.info(
            "Registered StarCraft2v2FastLane_Env under registry key 'StarCraft2v2'"
        )
    except ImportError as e:
        _logger.warning(f"Could not register StarCraft2v2FastLane_Env: {e}")
    except Exception as e:
        _logger.error(f"Error registering StarCraft2v2FastLane_Env: {e}")

    # Independent block: register ViZDoom single-agent wrapper. Kept
    # outside prior try/except blocks so a failure elsewhere does not
    # short-circuit the ViZDoom registration. Registered in REGISTRY_ENV
    # (single-agent) because 9 of 10 ViZDoom scenarios are single-agent;
    # Deathmatch (multi-agent) is out of scope for this registration.
    try:
        from xuance.environment.single_agent_env import REGISTRY_ENV as _REG_SA
        from xuance_worker.environments.vizdoom import ViZDoom_Env

        if "vizdoom" not in _REG_SA:
            _REG_SA["vizdoom"] = ViZDoom_Env
            _logger.info(
                "Registered ViZDoom_Env under xuance single-agent registry key 'vizdoom'"
            )
    except ImportError as e:
        _logger.warning(f"Could not register ViZDoom_Env: {e}")
    except Exception as e:
        _logger.error(f"Error registering ViZDoom_Env: {e}")


def get_registered_environments() -> list[str]:
    """Return list of MOSAIC environments registered with XuanCe.

    Returns:
        List of environment names that have been registered.
    """
    registered = []
    try:
        from xuance.environment.multi_agent_env import REGISTRY_MULTI_AGENT_ENV
        for name in ["multigrid", "MultiGrid"]:
            if name in REGISTRY_MULTI_AGENT_ENV:
                registered.append(name)
    except ImportError:
        pass
    return registered


# Export for direct import
def _get_multigrid_env():
    """Lazy import of MultiGrid_Env."""
    from xuance_worker.environments.multigrid.multigrid import MultiGrid_Env
    return MultiGrid_Env


__all__ = [
    "register_mosaic_environments",
    "get_registered_environments",
]
