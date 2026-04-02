"""MOSAIC MultiGrid Observation Text Conversion - Theory of Mind Study.

This module implements MOSAIC's research on Theory of Mind in multi-agent LLMs.
Two observation modes are provided to study how social information affects coordination:

- Egocentric Only: Agent sees only its own view (decentralized)
- Visible Teammates: Include visible teammates (Theory of Mind)

Research Questions:
- RQ1: Does Theory of Mind observation improve multi-agent coordination?
- RQ2: Can LLMs reason about teammate states and intentions?
- RQ3: What's the tradeoff between context size and coordination quality?

Based on:
- Egocentric: https://arxiv.org/html/2402.01680v2 (decentralized control)
- Teammates: https://openreview.net/forum?id=cfL8zApofK (ToM reasoning)
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import structured logging from gym_gui (optional)
try:
    from gym_gui.logging_config.helpers import log_constant
    from gym_gui.logging_config.log_constants import (
        LOG_WORKER_MOSAIC_OBSERVATION_EGOCENTRIC,
        LOG_WORKER_MOSAIC_OBSERVATION_TEAMMATES,
    )
    _HAS_STRUCTURED_LOGGING = True
except ImportError:
    log_constant = None
    _HAS_STRUCTURED_LOGGING = False

# Object type encoding from MiniGrid/MultiGrid
# Note: mosaic_multigrid adds ENDZONE (10) for American Football
OBJECT_TYPES = {
    0: "empty",
    1: "wall",
    2: "floor",
    3: "door",
    4: "key",
    5: "ball",
    6: "box",
    7: "goal",
    8: "lava",
    9: "agent",
    10: "endzone",  # American Football end zones
}

# Color encoding
COLORS = {
    0: "red",
    1: "green",
    2: "blue",
    3: "purple",
    4: "yellow",
    5: "grey",
}

# Direction encoding
DIRECTIONS = {
    0: "EAST",  # right
    1: "SOUTH",  # down
    2: "WEST",  # left
    3: "NORTH",  # up
}


def _get_relative_position(dx: int, dy: int) -> str:
    """Convert relative grid offset to natural language.

    Args:
        dx: Horizontal offset (-left, +right).
        dy: Vertical offset (-up, +down).

    Returns:
        Natural language like "2 steps to your north-east".
    """
    distance = abs(dx) + abs(dy)
    if distance == 0:
        return "at your position"
    elif distance == 1:
        if dx == 0 and dy == -1:
            return "1 step to your front"
        elif dx == 0 and dy == 1:
            return "1 step to your back"
        elif dx == -1 and dy == 0:
            return "1 step to your left"
        elif dx == 1 and dy == 0:
            return "1 step to your right"

    # General directional description
    vertical = "north" if dy < 0 else "south" if dy > 0 else ""
    horizontal = "west" if dx < 0 else "east" if dx > 0 else ""

    direction = f"{vertical}-{horizontal}" if vertical and horizontal else (vertical or horizontal)
    return f"{distance} steps to your {direction}"


def _parse_grid_observation(obs: np.ndarray) -> List[Tuple[int, int, Dict[str, Any]]]:
    """Parse encoded grid observation.

    Args:
        obs: Encoded grid observation (view_size, view_size, 6).

    Returns:
        List of (dx, dy, obj_info) tuples for visible objects.
    """
    view_size = obs.shape[0]
    center = view_size // 2  # Agent at center

    objects = []
    for y in range(view_size):
        for x in range(view_size):
            obj_type = obs[y, x, 0]
            if obj_type == 0:  # empty
                continue

            obj_info = {
                "type": OBJECT_TYPES.get(obj_type, f"unknown_{obj_type}"),
                "color": COLORS.get(obs[y, x, 1], f"color_{obs[y, x, 1]}"),
                "state": obs[y, x, 2],
                "carrying_type": OBJECT_TYPES.get(obs[y, x, 3], None) if obs[y, x, 3] > 0 else None,
                "carrying_color": COLORS.get(obs[y, x, 4], None) if obs[y, x, 4] > 0 else None,
                "direction": DIRECTIONS.get(obs[y, x, 5], "UNKNOWN") if obj_type == 9 else None,
            }

            dx = x - center
            dy = y - center
            objects.append((dx, dy, obj_info))

    return objects


def describe_observation_egocentric(
    obs: np.ndarray,
    agent_direction: int = 0,
    carrying: Optional[str] = None,
    env_id: Optional[str] = None,
) -> str:
    """Convert observation to text (Mode 1: Egocentric Only).

    Decentralized control - agent sees only its own view.
    Tests whether LLMs can coordinate without shared information.

    Args:
        obs: Encoded grid observation (view_size, view_size, 6).
        agent_direction: Agent's facing direction (0-3).
        carrying: What the agent is carrying.
        env_id: Environment identifier for game-specific descriptions.

    Returns:
        Natural language description of agent's view.
    """
    if _HAS_STRUCTURED_LOGGING and log_constant:
        log_constant(
            logger,
            LOG_WORKER_MOSAIC_OBSERVATION_EGOCENTRIC,
            extra={
                "agent_direction": agent_direction,
                "carrying": carrying,
            },
        )
    else:
        logger.debug(f"MOSAIC: Generating egocentric observation | direction={agent_direction} carrying={carrying}")

    objects = _parse_grid_observation(obs)

    # Categorize objects
    balls = []
    goals = []
    walls = []
    agents = []
    endzones = []

    for dx, dy, obj in objects:
        if dx == 0 and dy == 0:
            continue  # Skip own position

        pos_desc = _get_relative_position(dx, dy)
        obj_type = obj["type"]
        obj_color = obj["color"]

        if obj_type == "ball":
            balls.append(f"{obj_color} ball {pos_desc}")
        elif obj_type == "goal":
            goals.append(f"{obj_color} goal {pos_desc}")
        elif obj_type == "endzone":
            endzones.append(f"{obj_color} end zone {pos_desc}")
        elif obj_type == "wall":
            if dx == 0 and dy == -1:  # Only immediate wall
                walls.append("wall at your front")
        elif obj_type == "agent":
            agent_desc = f"agent ({obj_color}) {pos_desc}"
            if obj["carrying_type"]:
                agent_desc += f", carrying {obj['carrying_color']} {obj['carrying_type']}"
            agents.append(agent_desc)

    # Build description
    lines = ["You see:"]

    if balls:
        for ball in balls[:3]:
            lines.append(f"- {ball}")
    if goals:
        for goal in goals:
            lines.append(f"- {goal}")
    if endzones:
        for endzone in endzones:
            lines.append(f"- {endzone}")
    if agents:
        for agent in agents[:2]:
            lines.append(f"- {agent}")
    if walls:
        lines.append(f"- {walls[0]}")

    if len(lines) == 1:
        lines.append("- clear area")

    lines.append("")
    lines.append(f"You are facing: {DIRECTIONS.get(agent_direction, 'UNKNOWN')}")
    lines.append(f"You are carrying: {carrying if carrying else 'nothing'}")

    return "\n".join(lines)


def describe_observation_with_teammates(
    obs: np.ndarray,
    agent_id: int,
    visible_teammates: List[Dict[str, Any]],
    agent_direction: int = 0,
    carrying: Optional[str] = None,
) -> str:
    """Convert observation to text (Mode 2: Visible Teammates).

    Theory of Mind - include visible teammates to enable reasoning about others.
    Tests whether LLMs can improve coordination with social information.

    Args:
        obs: Encoded grid observation.
        agent_id: Current agent index.
        visible_teammates: List of teammate info dicts with keys:
            - id: Teammate agent ID
            - position: (x, y) position
            - direction: Facing direction
            - carrying: What teammate is carrying
            - color: Teammate color
        agent_direction: Agent's facing direction.
        carrying: What agent is carrying.

    Returns:
        Natural language description including teammates.
    """
    if _HAS_STRUCTURED_LOGGING and log_constant:
        log_constant(
            logger,
            LOG_WORKER_MOSAIC_OBSERVATION_TEAMMATES,
            extra={
                "agent_id": agent_id,
                "num_teammates": len(visible_teammates),
                "agent_direction": agent_direction,
            },
        )
    else:
        logger.debug(
            f"MOSAIC: Generating teammates observation | agent={agent_id} teammates={len(visible_teammates)} direction={agent_direction}"
        )

    # Start with egocentric description
    base_description = describe_observation_egocentric(obs, agent_direction, carrying)

    # Add teammate information
    if not visible_teammates:
        teammate_section = "\n\nVisible Teammates: none in view"
    else:
        teammate_lines = ["\n\nVisible Teammates:"]
        for tm in visible_teammates[:2]:  # Limit to 2 closest
            tm_id = tm.get("id", "?")
            tm_dir = DIRECTIONS.get(tm.get("direction", 0), "UNKNOWN")
            tm_carrying = tm.get("carrying", None)
            tm_color = tm.get("color", "unknown")

            tm_desc = f"- Teammate Agent {tm_id} ({tm_color}), facing {tm_dir}"
            if tm_carrying:
                tm_desc += f", carrying: {tm_carrying}"
            else:
                tm_desc += ", carrying: nothing"

            teammate_lines.append(tm_desc)

        teammate_section = "\n".join(teammate_lines)

    return base_description + teammate_section


def extract_visible_teammates(
    env: Any,
    agent_id: int,
    agent_team: int,
) -> List[Dict[str, Any]]:
    """Extract visible teammate information from environment.

    Args:
        env: MultiGrid environment instance.
        agent_id: Current agent index.
        agent_team: Current agent's team.

    Returns:
        List of visible teammate dictionaries.
    """
    visible_teammates = []

    if not hasattr(env, 'agents'):
        return visible_teammates

    for i, agent in enumerate(env.agents):
        if i == agent_id:
            continue  # Skip self

        # Check if teammate (same team in Soccer)
        if hasattr(agent, 'index'):
            other_team = agent.index // 2
            if other_team != agent_team:
                continue  # Opponent

        # Add teammate info
        if hasattr(agent, 'pos'):
            teammate_info = {
                "id": i,
                "position": tuple(agent.pos) if agent.pos is not None else None,
                "direction": agent.dir if hasattr(agent, 'dir') else 0,
                "carrying": str(agent.carrying) if hasattr(agent, 'carrying') and agent.carrying else None,
                "color": agent.color if hasattr(agent, 'color') else "unknown",
            }
            visible_teammates.append(teammate_info)

    return visible_teammates


__all__ = [
    "describe_observation_egocentric",
    "describe_observation_with_teammates",
    "extract_visible_teammates",
]
