"""MOSAIC MultiGrid Instruction Prompts - 3 Coordination Levels.

This module implements MOSAIC's research on multi-agent LLM coordination strategies.
Three levels are provided to study how explicit coordination guidance affects performance:

- Level 1 (Emergent): Minimal guidance, test emergent coordination
- Level 2 (Basic Hints): Add cooperation tips, balance emergence and guidance
- Level 3 (Role-Based): Explicit roles with detailed strategies

Supported Games:
- Soccer (2v2): Drop ball at opponent's goal to score
- Collect (3-agent FFA, 2v2, 1v1): Race to collect balls
- Basketball (3v3): Drop ball at baseline goal to score
- American Football (1v1, 2v2, 3v3): Walk into end zone while carrying (touchdown!)

Observation Types:
- IndAgObs: Individual agent observations (decentralized)
- TeamObs: SMAC-style teammate awareness (positions, directions, has_ball)

Research Questions:
- RQ1: How do coordination levels affect multi-agent performance?
- RQ2: Can LLMs coordinate effectively without explicit guidance?
- RQ3: Do role-based strategies improve team-based game outcomes?
- RQ4: Does TeamObs (teammate awareness) improve coordination?
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Try to import structured logging from gym_gui (optional)
# When gym_gui is not available, we fall back to standard logging
_HAS_STRUCTURED_LOGGING = False
log_constant = None  # type: ignore[assignment]
LOG_WORKER_MOSAIC_PROMPT_GENERATED = None  # type: ignore[assignment]
LOG_WORKER_MOSAIC_ACTION_PARSED = None  # type: ignore[assignment]
LOG_WORKER_MOSAIC_ACTION_PARSE_FAILED = None  # type: ignore[assignment]

try:
    from gym_gui.logging_config.helpers import log_constant
    from gym_gui.logging_config.log_constants import (
        LOG_WORKER_MOSAIC_PROMPT_GENERATED,
        LOG_WORKER_MOSAIC_ACTION_PARSED,
        LOG_WORKER_MOSAIC_ACTION_PARSE_FAILED,
    )
    _HAS_STRUCTURED_LOGGING = True
except ImportError:
    pass  # Keep fallback values defined above

# Legacy MultiGrid action space (8 actions) - Soccer, Collect
# Note: Legacy MultiGrid extends MiniGrid by adding STILL action at index 0
LEGACY_MULTIGRID_ACTION_SPACE = [
    "still",    # 0 - Legacy only: do nothing
    "left",     # 1
    "right",    # 2
    "forward",  # 3
    "pickup",   # 4
    "drop",     # 5
    "toggle",   # 6
    "done",     # 7
]

# INI MultiGrid action space (7 actions) - Empty, BlockedUnlockPickup, LockedHallway, etc.
# Note: INI MultiGrid has NO STILL action, LEFT is at index 0
INI_MULTIGRID_ACTION_SPACE = [
    "left",     # 0
    "right",    # 1
    "forward",  # 2
    "pickup",   # 3
    "drop",     # 4
    "toggle",   # 5
    "done",     # 6 - Also used as idle action in INI
]

# Default to Legacy for backwards compatibility
MULTIGRID_ACTION_SPACE = LEGACY_MULTIGRID_ACTION_SPACE

# Action descriptions for LLM instruction prompts
MULTIGRID_ACTIONS = {
    "still": "do nothing (wait in place) [Legacy only]",
    "left": "turn left 90° counter-clockwise",
    "right": "turn right 90° clockwise",
    "forward": "move one step in facing direction",
    "pickup": "pick up object (ball, key, box) or steal from agent",
    "drop": "drop held object (scores if at goal, or pass to teammate)",
    "toggle": "interact with object in front (open door, unlock with key)",
    "done": "signal completion / idle action",
}

# Environment detection helpers
def is_ini_multigrid(env_id: str) -> bool:
    """Check if environment uses INI MultiGrid (7 actions) vs Legacy (8 actions)."""
    ini_patterns = ["Empty", "BlockedUnlockPickup", "LockedHallway", "RedBlueDoors", "Playground"]
    return any(pattern in env_id for pattern in ini_patterns)

def is_teamobs_env(env_id: str) -> bool:
    """Check if environment uses TeamObs (SMAC-style teammate awareness)."""
    return "TeamObs" in env_id

def get_game_type(env_id: str) -> str:
    """Detect the game type from environment ID."""
    if "Soccer" in env_id:
        return "soccer"
    elif "Basketball" in env_id:
        return "basketball"
    elif "AmericanFootball" in env_id:
        return "american_football"
    elif "Collect" in env_id:
        return "collect"
    else:
        return "unknown"

def get_team_size(env_id: str) -> int:
    """Get team size from environment ID."""
    if "3vs3" in env_id or "3v3" in env_id:
        return 3
    elif "2vs2" in env_id or "2v2" in env_id:
        return 2
    elif "1vs1" in env_id or "1v1" in env_id:
        return 1
    elif "Solo" in env_id:
        return 1
    return 2  # Default

def get_action_space(env_id: str) -> list[str]:
    """Get the appropriate action space for the environment."""
    return INI_MULTIGRID_ACTION_SPACE if is_ini_multigrid(env_id) else LEGACY_MULTIGRID_ACTION_SPACE


def parse_action(llm_output: str, env_id: str | None = None) -> int:
    """Parse LLM text output to extract action index.

    Args:
        llm_output: Raw text output from LLM containing action name.
        env_id: Environment identifier to determine action space (Legacy vs INI).
                If None, defaults to Legacy action space.

    Returns:
        Action index for the appropriate action space.
        - Legacy (Soccer, Collect): 0-7 with STILL=0
        - INI (BlockedUnlockPickup, etc.): 0-6 with LEFT=0
        Defaults to idle action (STILL=0 for Legacy, DONE=6 for INI) if parsing fails.
    """
    llm_lower = llm_output.lower().strip()

    # Determine which action space to use
    use_ini = env_id is not None and is_ini_multigrid(env_id)
    action_space = INI_MULTIGRID_ACTION_SPACE if use_ini else LEGACY_MULTIGRID_ACTION_SPACE
    idle_action = 6 if use_ini else 0  # DONE for INI, STILL for Legacy

    # Try exact match first
    for i, action in enumerate(action_space):
        if action in llm_lower:
            if _HAS_STRUCTURED_LOGGING and log_constant:
                log_constant(
                    logger,
                    LOG_WORKER_MOSAIC_ACTION_PARSED,  # type: ignore[arg-type]
                    extra={
                        "action_name": action,
                        "action_index": i,
                        "llm_output": llm_output[:50],
                        "env_id": env_id,
                        "action_space": "INI" if use_ini else "Legacy",
                    },
                )
            else:
                logger.debug(f"MOSAIC: Parsed action '{action}' (index {i}) from LLM output: {llm_output[:50]}")
            return i

    # Try common variations - map to action names then look up index
    def _get_action_index(action_name: str) -> int:
        """Get index for action name in current action space."""
        try:
            return action_space.index(action_name)
        except ValueError:
            return idle_action  # Action not in this space, return idle

    def _log_action(action_name: str, action_index: int):
        """Helper to log action parsing with structured logging if available."""
        if _HAS_STRUCTURED_LOGGING and log_constant:
            log_constant(
                logger,
                LOG_WORKER_MOSAIC_ACTION_PARSED,  # type: ignore[arg-type]
                extra={
                    "action_name": action_name,
                    "action_index": action_index,
                    "llm_output": llm_output[:50],
                    "env_id": env_id,
                    "action_space": "INI" if use_ini else "Legacy",
                },
            )
        else:
            logger.debug(f"MOSAIC: Parsed action '{action_name}' (index {action_index}) from LLM output: {llm_output[:50]}")

    if "turn left" in llm_lower or "rotate left" in llm_lower:
        idx = _get_action_index("left")
        _log_action("left", idx)
        return idx
    if "turn right" in llm_lower or "rotate right" in llm_lower:
        idx = _get_action_index("right")
        _log_action("right", idx)
        return idx
    if "move forward" in llm_lower or "go forward" in llm_lower or "move ahead" in llm_lower:
        idx = _get_action_index("forward")
        _log_action("forward", idx)
        return idx
    if "pick up" in llm_lower or "pick-up" in llm_lower or "grab" in llm_lower:
        idx = _get_action_index("pickup")
        _log_action("pickup", idx)
        return idx
    if "wait" in llm_lower or "stay" in llm_lower or "nothing" in llm_lower:
        # For INI, use "done" as idle; for Legacy, use "still"
        idle_name = "done" if use_ini else "still"
        _log_action(idle_name, idle_action)
        return idle_action

    # Default to idle action if no action found
    idle_name = "done" if use_ini else "still"
    if _HAS_STRUCTURED_LOGGING and log_constant:
        log_constant(
            logger,
            LOG_WORKER_MOSAIC_ACTION_PARSE_FAILED,  # type: ignore[arg-type]
            extra={
                "llm_output": llm_output[:100],
                "env_id": env_id,
                "default_action": idle_name,
                "default_index": idle_action,
            },
        )
    else:
        logger.warning(f"MOSAIC: Failed to parse action from LLM output, defaulting to '{idle_name}': {llm_output[:100]}")
    return idle_action


def get_instruction_prompt_level1(agent_id: int, team: int, env_id: str) -> str:
    """Generate Level 1 (Emergent) instruction prompt.

    Minimal guidance - let LLMs discover coordination naturally.
    Tests emergent multi-agent behavior without explicit cooperation strategies.

    Args:
        agent_id: Agent index (0-3 for Soccer, 0-2 for Collect).
        team: Team index (0 or 1 for Soccer).
        env_id: Environment identifier.

    Returns:
        Level 1 instruction prompt.
    """
    if _HAS_STRUCTURED_LOGGING and log_constant:
        log_constant(
            logger,
            LOG_WORKER_MOSAIC_PROMPT_GENERATED,  # type: ignore[arg-type]
            extra={
                "coordination_level": 1,
                "agent_id": agent_id,
                "team": team,
                "env_id": env_id,
            },
        )
    else:
        logger.debug(
            f"MOSAIC: Generating Level 1 prompt | agent={agent_id} team={team} env={env_id}"
        )

    action_strings = ",\n".join(
        f"  {action}: {description}"
        for action, description in MULTIGRID_ACTIONS.items()
    )

    game_type = get_game_type(env_id)
    team_size = get_team_size(env_id)

    # American Football - simplified scoring (walk into end zone)
    if game_type == "american_football":
        team_name = "Green" if team == 0 else "Blue"
        opponent_name = "Blue" if team == 0 else "Green"

        return f"""
You are Agent {agent_id} playing American Football ({team_size}v{team_size}).

Your team: {team_name}
Opponent team: {opponent_name}

Actions:
{action_strings}

Goal: Score TOUCHDOWNS by walking into the opponent's END ZONE while carrying the ball!

Game Mechanics:
- Pick up the ball by walking to it and using "pickup"
- You can STEAL the ball from opponents using "pickup"
- Score by WALKING into the opponent's end zone while carrying (no drop needed!)
- Pass to teammate by using "drop" (ball teleports to nearest teammate)
- Green's end zone is on the LEFT (column 1), Blue scores there
- Blue's end zone is on the RIGHT (column 14), Green scores there

PLAY!
""".strip()

    # Basketball - drop ball at baseline goal
    elif game_type == "basketball":
        team_name = "Green" if team == 0 else "Blue"
        opponent_name = "Blue" if team == 0 else "Green"

        return f"""
You are Agent {agent_id} playing Basketball (3v3).

Your team: {team_name}
Opponents: {opponent_name}

Actions:
{action_strings}

Goal: Score by dropping the ball at the opponent's baseline GOAL!

Game Mechanics:
- Pick up the ball by walking to it and using "pickup"
- You can STEAL the ball from opponents
- Score by using "drop" at the opponent's goal (on their baseline)
- Pass to teammate using "drop" (ball teleports to them)
- Green's goal is on the LEFT, Blue's goal is on the RIGHT

PLAY!
""".strip()

    # Soccer - drop ball at goal
    elif game_type == "soccer":
        team_name = "Red" if team == 0 else "Green"
        opponent_name = "Green" if team == 0 else "Red"

        return f"""
You are Agent {agent_id} playing a {team_size}v{team_size} soccer game.

Your team: {team_name} (Agents {team * 2}, {team * 2 + 1})
Opponent team: {opponent_name}

Actions:
{action_strings}

Goal: Score by carrying the ball to the opponent's goal.

Game Mechanics:
- Pick up the ball by walking to it and using "pickup"
- You can steal the ball from opponents
- Score by dropping the ball at the opponent's goal
- Pass to teammate by dropping the ball near them

PLAY!
""".strip()

    # Collect - race to collect balls
    elif game_type == "collect":
        colors = ["Red", "Green", "Blue"]
        color = colors[agent_id] if agent_id < 3 else "Unknown"

        return f"""
You are Agent {agent_id} ({color}) playing a ball collection game.

Your color: {color}
Opponents: {', '.join(c for i, c in enumerate(colors) if i != agent_id)}

Actions:
{action_strings}

Goal: Collect as many balls as possible before your opponents.

Game Mechanics:
- Walk to balls and use "pickup" to collect them
- Balls disappear once collected
- Race against opponents

PLAY!
""".strip()

    # Unknown game type
    else:
        return f"""
You are Agent {agent_id} in a multi-agent grid world.

Actions:
{action_strings}

PLAY!
""".strip()


def get_instruction_prompt_level2(agent_id: int, team: int, env_id: str) -> str:
    """Generate Level 2 (Basic Hints) instruction prompt.

    Add cooperation tips to guide LLMs toward teamwork without being prescriptive.
    Balances emergent behavior with helpful coordination guidance.

    Args:
        agent_id: Agent index.
        team: Team index.
        env_id: Environment identifier.

    Returns:
        Level 2 instruction prompt.
    """
    if _HAS_STRUCTURED_LOGGING and log_constant:
        log_constant(
            logger,
            LOG_WORKER_MOSAIC_PROMPT_GENERATED,  # type: ignore[arg-type]
            extra={
                "coordination_level": 2,
                "agent_id": agent_id,
                "team": team,
                "env_id": env_id,
            },
        )
    else:
        logger.debug(
            f"MOSAIC: Generating Level 2 prompt | agent={agent_id} team={team} env={env_id}"
        )

    action_strings = ",\n".join(
        f"  {action}: {description}"
        for action, description in MULTIGRID_ACTIONS.items()
    )

    game_type = get_game_type(env_id)
    team_size = get_team_size(env_id)

    # American Football - simplified scoring (walk into end zone)
    if game_type == "american_football":
        team_name = "Green" if team == 0 else "Blue"
        opponent_name = "Blue" if team == 0 else "Green"

        return f"""
You are Agent {agent_id} playing American Football ({team_size}v{team_size}).

Your team: {team_name}
Opponent team: {opponent_name}

Actions:
{action_strings}

Goal: Score TOUCHDOWNS by walking into the opponent's END ZONE while carrying the ball!

Game Mechanics:
- Pick up the ball by walking to it and using "pickup"
- You can STEAL the ball from opponents using "pickup"
- Score by WALKING into the opponent's end zone while carrying (no drop needed!)
- Pass to teammate by using "drop" (ball teleports to nearest teammate)
- Green's end zone is on the LEFT (column 1), Blue scores there
- Blue's end zone is on the RIGHT (column 14), Green scores there

Tips for Coordination:
- Spread out to create passing lanes
- If you have the ball, advance toward the opponent's end zone
- If a teammate has the ball, get open for a pass or block defenders
- If opponents have the ball, swarm the ball carrier to steal

PLAY!
""".strip()

    # Basketball - drop ball at baseline goal
    elif game_type == "basketball":
        team_name = "Green" if team == 0 else "Blue"
        opponent_name = "Blue" if team == 0 else "Green"

        return f"""
You are Agent {agent_id} playing Basketball (3v3).

Your team: {team_name}
Opponent team: {opponent_name}

Actions:
{action_strings}

Goal: Score by dropping the ball at the opponent's baseline GOAL!

Game Mechanics:
- Pick up the ball by walking to it and using "pickup"
- You can STEAL the ball from opponents
- Score by using "drop" at the opponent's goal (on their baseline)
- Pass to teammate using "drop" (ball teleports to them)
- Green's goal is on the LEFT, Blue's goal is on the RIGHT

Tips for Coordination:
- Create spacing - don't cluster together
- Pass the ball to open teammates for better shots
- If you don't have the ball, get open or set screens
- On defense, stay between your opponent and the goal

PLAY!
""".strip()

    # Soccer - drop ball at goal
    elif game_type == "soccer":
        team_name = "Red" if team == 0 else "Green"
        opponent_name = "Green" if team == 0 else "Red"

        return f"""
You are Agent {agent_id} playing a {team_size}v{team_size} soccer game.

Your team: {team_name} (Agents {team * 2}, {team * 2 + 1})
Opponent team: {opponent_name}

Actions:
{action_strings}

Goal: Score by carrying the ball to the opponent's goal.

Game Mechanics:
- Pick up the ball by walking to it and using "pickup"
- You can steal the ball from opponents
- Score by dropping the ball at the opponent's goal
- Pass to teammate by dropping the ball near them

Tips for Coordination:
- Spread out to cover more area - don't cluster together
- If your teammate has the ball, position yourself to receive a pass
- If an opponent has the ball, try to intercept or block their path
- One player should focus on attacking, the other on supporting
- Don't all chase the ball - maintain team positioning

PLAY!
""".strip()

    elif game_type == "collect":
        colors = ["Red", "Green", "Blue"]
        color = colors[agent_id] if agent_id < 3 else "Unknown"

        return f"""
You are Agent {agent_id} ({color}) playing a ball collection game.

Your color: {color}
Opponents: {', '.join(c for i, c in enumerate(colors) if i != agent_id)}

Actions:
{action_strings}

Goal: Collect as many balls as possible before your opponents.

Game Mechanics:
- Walk to balls and use "pickup" to collect them
- Balls disappear once collected
- Race against opponents

Tips for Competition:
- Target the nearest uncollected ball
- Watch opponent movements and predict their targets
- If an opponent is closer to a ball, switch to a different target
- Move quickly and decisively

PLAY!
""".strip()

    else:
        return f"""
You are Agent {agent_id} in a multi-agent grid world.

Actions:
{action_strings}

Tips:
- Coordinate with visible teammates when possible
- Avoid repeating the same action if observation doesn't change

PLAY!
""".strip()


def get_instruction_prompt_level3(agent_id: int, team: int, role: str, env_id: str) -> str:
    """Generate Level 3 (Role-Based) instruction prompt.

    Assign explicit roles (Forward/Defender) with detailed role-specific strategies.
    Tests whether explicit role division improves team coordination and performance.

    Args:
        agent_id: Agent index.
        team: Team index.
        role: Assigned role ("forward" or "defender").
        env_id: Environment identifier.

    Returns:
        Level 3 instruction prompt with role-specific strategies.
    """
    if _HAS_STRUCTURED_LOGGING and log_constant:
        log_constant(
            logger,
            LOG_WORKER_MOSAIC_PROMPT_GENERATED,  # type: ignore[arg-type]
            extra={
                "coordination_level": 3,
                "agent_id": agent_id,
                "team": team,
                "role": role,
                "env_id": env_id,
            },
        )
    else:
        logger.debug(
            f"MOSAIC: Generating Level 3 prompt | agent={agent_id} team={team} role={role} env={env_id}"
        )

    action_strings = ",\n".join(
        f"  {action}: {description}"
        for action, description in MULTIGRID_ACTIONS.items()
    )

    game_type = get_game_type(env_id)
    team_size = get_team_size(env_id)

    # American Football with roles (Quarterback/Receiver)
    if game_type == "american_football":
        team_name = "Green" if team == 0 else "Blue"
        opponent_name = "Blue" if team == 0 else "Green"

        # Roles: Quarterback (ball handler) vs Receiver (get open)
        if role.lower() == "forward":
            role_strategy = """
Your Role: QUARTERBACK (Ball Handler)
- Priority: Advance the ball, score touchdowns
- If you have ball → run toward opponent's end zone
- If you have ball and defenders nearby → pass to open teammate
- If teammate has ball → provide blocking support
- Take calculated risks - protect the ball above all"""
        else:  # defender role maps to receiver
            role_strategy = """
Your Role: RECEIVER (Get Open)
- Priority: Get open for passes, create scoring opportunities
- If teammate has ball → get open downfield for a pass
- If you receive the ball → run to the end zone!
- If opponent has ball → pursue and attempt to steal
- Create separation from defenders"""

        return f"""
You are Agent {agent_id} playing American Football ({team_size}v{team_size}).

Your team: {team_name}
Opponent team: {opponent_name}

{role_strategy}

Actions:
{action_strings}

Goal: Score TOUCHDOWNS by walking into the opponent's END ZONE while carrying!

Game Mechanics:
- Pick up ball with "pickup", STEAL from opponents with "pickup"
- Score by WALKING into end zone while carrying (no drop needed!)
- Pass with "drop" (teleports to nearest teammate)
- Green scores at x=14 (Blue's end zone), Blue scores at x=1 (Green's end zone)

Coordination Protocol:
- Trust your teammates to handle their roles
- Communicate through positioning
- Protect the ball carrier

PLAY!
""".strip()

    # Basketball with roles (Point Guard/Center)
    elif game_type == "basketball":
        team_name = "Green" if team == 0 else "Blue"
        opponent_name = "Blue" if team == 0 else "Green"

        if role.lower() == "forward":
            role_strategy = """
Your Role: POINT GUARD (Playmaker)
- Priority: Create scoring opportunities, handle the ball
- If you have ball → drive toward opponent's goal or pass
- If teammate is open → pass with "drop" (teleports to them)
- If opponent has ball → pressure the ball handler
- Control the tempo of the game"""
        else:
            role_strategy = """
Your Role: CENTER (Interior Presence)
- Priority: Score inside, protect the basket
- Position yourself near the opponent's goal for easy scores
- If you get the ball → drop it at the goal immediately
- On defense, stay near your goal to block opponents
- Create space for teammates to operate"""

        return f"""
You are Agent {agent_id} playing Basketball (3v3).

Your team: {team_name}
Opponent team: {opponent_name}

{role_strategy}

Actions:
{action_strings}

Goal: Score by dropping the ball at the opponent's baseline GOAL!

Game Mechanics:
- Pick up ball with "pickup", STEAL from opponents
- Score with "drop" at opponent's goal (on their baseline)
- Pass with "drop" (teleports to nearest teammate)
- Green's goal is on the LEFT, Blue's is on the RIGHT

PLAY!
""".strip()

    # Soccer with roles (Forward/Defender)
    elif game_type == "soccer":
        team_name = "Red" if team == 0 else "Green"
        opponent_name = "Green" if team == 0 else "Red"
        teammate_id = 1 if agent_id == 0 else 0 if agent_id in [0, 1] else 3 if agent_id == 2 else 2
        teammate_role = "DEFENDER" if role.lower() == "forward" else "FORWARD"

        if role.lower() == "forward":
            role_strategy = """
Your Role: FORWARD (Offensive)
- Priority: Score goals, pressure opponents
- If you have ball → advance toward opponent goal
- If teammate has ball → move to open space ahead for pass
- If opponent has ball → pressure the ball carrier
- Stay in attacking half when possible
- Take risks to create scoring opportunities"""
        else:  # defender
            role_strategy = """
Your Role: DEFENDER (Defensive)
- Priority: Protect goal, support attacks
- If you have ball → pass forward or advance carefully
- If teammate has ball → cover defensive position
- If opponent has ball → fall back, block goal path
- Stay between ball and your goal
- Be conservative - don't leave goal undefended"""

        return f"""
You are Agent {agent_id} playing a {team_size}v{team_size} soccer game.

Your team: {team_name} (Agents {team * 2}, {team * 2 + 1})
Opponent team: {opponent_name}

{role_strategy}

Actions:
{action_strings}

Goal: Score by carrying the ball to the opponent's goal.

Game Mechanics:
- Pick up the ball by walking to it and using "pickup"
- You can steal the ball from opponents
- Score by dropping the ball at the opponent's goal
- Pass to teammate by dropping the ball near them

Coordination Protocol:
- You are the {role.upper()}, your teammate (Agent {teammate_id}) is the {teammate_role}
- Trust your teammate to handle their role
- Adjust your position based on visible teammate location
- Don't overlap roles - maintain positional discipline

PLAY!
""".strip()

    # Other games - fall back to level 2
    else:
        return get_instruction_prompt_level2(agent_id, team, env_id)


__all__ = [
    # Action spaces
    "MULTIGRID_ACTION_SPACE",           # Default (Legacy) for backwards compatibility
    "LEGACY_MULTIGRID_ACTION_SPACE",    # 8 actions: Soccer, Collect, Basketball, American Football
    "INI_MULTIGRID_ACTION_SPACE",       # 7 actions: Empty, BlockedUnlockPickup, etc.
    "MULTIGRID_ACTIONS",                # Action descriptions
    # Environment detection
    "is_ini_multigrid",                 # Check if env uses INI action space
    "is_teamobs_env",                   # Check if env uses TeamObs
    "get_game_type",                    # Detect game type (soccer, basketball, etc.)
    "get_team_size",                    # Get team size from env ID
    "get_action_space",                 # Get appropriate action space for env
    # Action parsing
    "parse_action",
    # Prompt generators
    "get_instruction_prompt_level1",
    "get_instruction_prompt_level2",
    "get_instruction_prompt_level3",
]
