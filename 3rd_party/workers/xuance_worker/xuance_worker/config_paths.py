"""Resolve YAML config paths that live in xuance_worker instead of the
vendored xuance tree.

Motivation
==========
xuance's `get_arguments(algo, env, env_id, ...)` looks for YAMLs under
`xuance/configs/{algo}/{env}/{env_id}.yaml`. That directory ships only a
minimal subset of scenarios per env family. For gfootball, xuance ships
`1v1.yaml` and `3v1.yaml` under `mappo/football/` and `ippo/football/`.
The wrapper's `GFOOTBALL_ENV_ID` shorthand table has 18 scenarios though.

To expose the extra scenarios in the training form without modifying
vendored xuance, we ship our own YAMLs under
`xuance_worker/configs/{algo}/{env}/{env_id}.yaml` and pass their path
into `get_runner(..., config_path=<our_path>)`, bypassing xuance's default
lookup for that scenario. When our directory has no matching YAML, we
return `None` and let xuance fall back to its own defaults.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

# Directory holding YAMLs we ship in the worker (not the vendored xuance).
# Layout mirrors xuance's config directory:
#   xuance_worker/configs/{algo}/{env}/{env_id}.yaml
_WORKER_CONFIGS_ROOT: Path = Path(__file__).resolve().parent / "configs"


def resolve_worker_config_path(algo: str, env: str, env_id: str) -> Optional[str]:
    """Return path to xuance_worker's YAML for (algo, env, env_id), or None.

    Args:
        algo: Algorithm shorthand (e.g. "mappo", "ippo"). Case-insensitive.
        env: Environment family (e.g. "football"). Should match xuance's
             directory naming (not the form's display name — the form's
             boundary translation converts "gfootball" to "football" before
             calling this).
        env_id: Scenario shorthand (e.g. "5v5", "corner").

    Returns:
        Absolute path string if we ship a matching YAML, else None. When
        None is returned, the caller should let xuance's `get_arguments()`
        use its default lookup.
    """
    if not algo or not env or not env_id:
        return None
    candidate = _WORKER_CONFIGS_ROOT / algo.lower() / env / f"{env_id}.yaml"
    return str(candidate) if candidate.is_file() else None


def list_worker_scenarios(algo: str, env: str) -> list[str]:
    """Return the sorted list of env_id shorthands we ship YAMLs for.

    Used by tests to assert form's static tables stay in sync with the
    YAMLs we actually ship on disk.
    """
    if not algo or not env:
        return []
    directory = _WORKER_CONFIGS_ROOT / algo.lower() / env
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.yaml"))


__all__ = ["resolve_worker_config_path", "list_worker_scenarios"]
