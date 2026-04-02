"""MOSAIC MultiGrid Extension for LLM Worker.

Multi-agent coordination strategies and Theory of Mind observations for LLMs.

This extension enables research on:
1. Multi-agent LLM coordination (3 levels: Emergent, Basic Hints, Role-Based)
2. Theory of Mind in multi-agent scenarios (Egocentric vs Visible Teammates)
3. Role-based team strategies (Forward/Defender/Quarterback/Receiver/Point Guard/Center)

Environments supported:
- Soccer (2v2): Drop ball at opponent's goal
- Collect (3-agent FFA, 2v1, 2v2): Race to collect balls
- Basketball (3v3): Drop ball at baseline goal
- American Football (1v1, 2v2, 3v3): Walk into end zone (touchdown!)

Observation Types:
- IndAgObs: Individual agent observations (decentralized)
- TeamObs: SMAC-style teammate awareness (positions, directions, has_ball)

Example usage:
    from llm_worker.environments.multigrid import prompts, observations

    # Level 1: Emergent coordination
    prompt = prompts.get_instruction_prompt_level1(agent_id=0, team=0, env_id="MosaicMultiGrid-Soccer-2vs2-IndAgObs-v0")

    # Egocentric observation
    obs_text = observations.describe_observation_egocentric(obs, agent_direction=0, carrying=None)

"""

from .prompts import (
    # Action spaces
    MULTIGRID_ACTION_SPACE,
    LEGACY_MULTIGRID_ACTION_SPACE,
    INI_MULTIGRID_ACTION_SPACE,
    MULTIGRID_ACTIONS,
    # Environment detection
    is_ini_multigrid,
    is_teamobs_env,
    get_game_type,
    get_team_size,
    get_action_space,
    # Action parsing
    parse_action,
    # Prompt generators
    get_instruction_prompt_level1,
    get_instruction_prompt_level2,
    get_instruction_prompt_level3,
)

from .observations import (
    describe_observation_egocentric,
    describe_observation_with_teammates,
    extract_visible_teammates,
)

__all__ = [
    # Action spaces
    "MULTIGRID_ACTION_SPACE",
    "LEGACY_MULTIGRID_ACTION_SPACE",
    "INI_MULTIGRID_ACTION_SPACE",
    "MULTIGRID_ACTIONS",
    # Environment detection
    "is_ini_multigrid",
    "is_teamobs_env",
    "get_game_type",
    "get_team_size",
    "get_action_space",
    # Action parsing
    "parse_action",
    # Prompt generators
    "get_instruction_prompt_level1",
    "get_instruction_prompt_level2",
    "get_instruction_prompt_level3",
    # Observations
    "describe_observation_egocentric",
    "describe_observation_with_teammates",
    "extract_visible_teammates",
]
