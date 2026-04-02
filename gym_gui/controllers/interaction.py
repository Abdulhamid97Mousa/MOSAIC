from __future__ import annotations

import socket
import struct
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import numpy as np
from qtpy.QtCore import Qt


class InteractionController(ABC):
    """Family-specific stepping/idle/passive-action policy."""

    @abstractmethod
    def idle_interval_ms(self) -> Optional[int]:
        """Desired idle tick interval in milliseconds, or None to disable idle ticking."""
        raise NotImplementedError

    @abstractmethod
    def should_idle_tick(self) -> bool:
        """Whether an idle tick should be performed now (timer gating)."""
        raise NotImplementedError

    @abstractmethod
    def maybe_passive_action(self) -> Optional[Any]:
        """Return a passive action to apply on idle, or None if not applicable."""
        raise NotImplementedError

    @abstractmethod
    def step_dt(self) -> float:
        """Fixed simulation dt if applicable (physics), else 0.0."""
        raise NotImplementedError

    def handle_native_key(self, key_code: int, pressed: bool) -> bool:
        """Handle a raw key event. Return True if handled."""
        return False

    def handle_native_mouse(self, dx: int, dy: int) -> bool:
        """Handle a raw mouse delta event. Return True if handled."""
        return False


class Box2DInteractionController(InteractionController):
    def __init__(self, owner, target_hz: int = 50):
        # owner is SessionController; used to reuse existing passive action logic safely
        self._owner = owner
        self._dt = 1.0 / float(target_hz)
        self._interval_ms = int(1000 / float(target_hz))

    def idle_interval_ms(self) -> Optional[int]:
        return self._interval_ms

    def should_idle_tick(self) -> bool:
        # Delegate to owner's existing gating to keep parity
        return self._owner._control_mode.name == "HUMAN_ONLY" and not self._owner._game_paused and self._owner._passive_action is not None and self._owner._adapter is not None and self._owner._game_id is not None

    def maybe_passive_action(self) -> Optional[Any]:
        # Reuse owner's resolved passive action (kept in sync by owner)
        return self._owner._passive_action

    def step_dt(self) -> float:
        return self._dt


class TurnBasedInteractionController(InteractionController):
    def idle_interval_ms(self) -> Optional[int]:
        return None

    def should_idle_tick(self) -> bool:
        return False

    def maybe_passive_action(self) -> Optional[Any]:
        return None

    def step_dt(self) -> float:
        return 0.0


def _get_qt_key(name: str) -> int:
    """Get Qt key constant by name, handling Qt5/Qt6 differences."""
    key_enum = getattr(Qt, "Key", None)
    if key_enum is not None and hasattr(key_enum, name):
        return int(getattr(key_enum, name))
    legacy = getattr(Qt, name, None)
    if legacy is not None:
        return int(legacy)
    # Return 0 or specific marker if not found, to avoid crashing.
    # But usually we want to know if it's missing.
    # For this mapping table, we can skip missing keys.
    return 0


class AleInteractionController(InteractionController):
    """Idle controller for Atari/ALE: step continuously with NOOP when idle."""

    def __init__(self, owner, target_hz: int = 60):
        self._owner = owner
        self._interval_ms = max(1, int(1000 / float(target_hz)))  # ~16ms

    def idle_interval_ms(self) -> Optional[int]:
        return self._interval_ms

    def should_idle_tick(self) -> bool:
        # Human-only, game started, not paused, adapter and game present, episode not finished
        o = self._owner
        if o._adapter is None or o._game_id is None:
            return False
        if not getattr(o, "_game_started", False):
            return False
        if o._game_paused:
            return False
        if getattr(o._control_mode, "name", "") != "HUMAN_ONLY":
            return False
        if o._last_step is not None and (o._last_step.terminated or o._last_step.truncated):
            return False
        return True

    def maybe_passive_action(self) -> Optional[Any]:
        # ALE NOOP is action 0 in minimal action set
        return 0

    def step_dt(self) -> float:
        return 0.0


class ViZDoomInteractionController(InteractionController):
    """Idle controller for ViZDoom: step continuously with NOOP when idle.

    ViZDoom games run at ~35 FPS (configurable). In Human Control mode,
    the game should continue running (enemies move, projectiles fly, etc.)
    even when the player doesn't provide input - just like a real FPS game.

    This mirrors the ALE behavior where the game world keeps advancing.
    """

    def __init__(self, owner, target_hz: int = 35):
        """Initialize ViZDoom interaction controller.

        Args:
            owner: SessionController instance.
            target_hz: Target frame rate (default 35 FPS, ViZDoom's default ticrate).
        """
        self._owner = owner
        self._interval_ms = max(1, int(1000 / float(target_hz)))  # ~28ms for 35 FPS

    def idle_interval_ms(self) -> Optional[int]:
        return self._interval_ms

    def should_idle_tick(self) -> bool:
        """Check if we should advance the game this tick.

        Returns True when:
        - Adapter and game are loaded
        - Game is started
        - Game is not paused
        - Control mode is HUMAN_ONLY
        - Episode is not finished
        """
        o = self._owner
        if o._adapter is None or o._game_id is None:
            return False
        if not getattr(o, "_game_started", False):
            return False
        if o._game_paused:
            return False
        if getattr(o._control_mode, "name", "") != "HUMAN_ONLY":
            return False
        if o._last_step is not None and (o._last_step.terminated or o._last_step.truncated):
            return False
        return True

    def maybe_passive_action(self) -> Optional[Any]:
        """Return NOOP action for ViZDoom.

        ViZDoom uses MultiBinary action space - NOOP is all zeros.
        We return -1 as a special sentinel meaning "no buttons pressed".
        The adapter's step() will recognize this and use all zeros.
        """
        # Return -1 as sentinel for NOOP (no buttons pressed)
        # The adapter's step() handles this by keeping cmd as all zeros
        return -1

    def step_dt(self) -> float:
        return 0.0


class ProcgenInteractionController(InteractionController):
    """Idle controller for Procgen: step continuously with NOOP when idle.

    Procgen games are real-time arcade-style games (like Atari) where the
    world should continue advancing even without player input. Enemies move,
    projectiles fly, and timers count down regardless of human action.

    Default 30 FPS gives responsive gameplay in a GUI environment.
    """

    def __init__(self, owner, target_hz: int = 30):
        """Initialize Procgen interaction controller.

        Args:
            owner: SessionController instance.
            target_hz: Target frame rate (default 30 FPS for responsive play).
        """
        self._owner = owner
        self._interval_ms = max(1, int(1000 / float(target_hz)))  # ~33ms for 30 FPS

    def idle_interval_ms(self) -> Optional[int]:
        return self._interval_ms

    def should_idle_tick(self) -> bool:
        """Check if we should advance the game this tick."""
        o = self._owner
        if o._adapter is None or o._game_id is None:
            return False
        if not getattr(o, "_game_started", False):
            return False
        if o._game_paused:
            return False
        if getattr(o._control_mode, "name", "") != "HUMAN_ONLY":
            return False
        if o._last_step is not None and (o._last_step.terminated or o._last_step.truncated):
            return False
        return True

    def maybe_passive_action(self) -> Optional[Any]:
        """Return NOOP action for Procgen.

        Procgen uses Discrete(15) action space. Action 4 is NOOP
        (no movement, no action). Action 0 is actually DOWN_LEFT!
        """
        return 4

    def step_dt(self) -> float:
        return 0.0


class SMACInteractionController(InteractionController):
    """Idle controller for SMAC/SMACv2: step continuously with random valid actions.

    SMAC environments are multi-agent cooperative benchmarks where all agents
    act simultaneously each timestep.  In AGENT_ONLY mode (the only supported
    mode for SMAC), the game should auto-step with random valid actions so the
    user can observe the battle unfolding after clicking "Start Game".

    Each ``maybe_passive_action()`` call returns a ``List[int]`` -- one random
    valid action per agent, respecting the action masks that SMAC exposes via
    ``get_avail_agent_actions()``.  Dead agents can only perform NO-OP (0).
    """

    def __init__(self, owner: Any, target_hz: int = 5) -> None:
        self._owner = owner
        self._interval_ms = max(1, int(1000 / float(target_hz)))  # 200ms for 5 FPS

    def idle_interval_ms(self) -> Optional[int]:
        return self._interval_ms

    def should_idle_tick(self) -> bool:
        o = self._owner
        if o._adapter is None or o._game_id is None:
            return False
        if not getattr(o, "_game_started", False):
            return False
        if o._game_paused:
            return False
        # SMAC supports AGENT_ONLY and MULTI_AGENT_COOP -- both need auto-stepping
        mode_name = getattr(o._control_mode, "name", "")
        if mode_name not in ("AGENT_ONLY", "MULTI_AGENT_COOP"):
            return False
        if o._last_step is not None and (o._last_step.terminated or o._last_step.truncated):
            return False
        return True

    def maybe_passive_action(self) -> Optional[List[int]]:
        """Sample random valid actions for all SMAC agents using action masks."""
        adapter = getattr(self._owner, "_adapter", None)
        if adapter is None:
            return None
        # Access the adapter's get_avail_actions() for mask-based sampling
        if not hasattr(adapter, "get_avail_actions"):
            return None
        try:
            avail = adapter.get_avail_actions()
            if not avail:
                return None
            actions: List[int] = []
            for mask in avail:
                valid_indices = [i for i, v in enumerate(mask) if v == 1]
                if valid_indices:
                    actions.append(int(np.random.choice(valid_indices)))
                else:
                    actions.append(0)  # NO-OP fallback for dead agents
            return actions
        except Exception:
            return None

    def step_dt(self) -> float:
        return 0.0


class JumanjiArcadeInteractionController(InteractionController):
    """Idle controller for arcade-style Jumanji games (PacMan, Snake, Tetris).

    These games have continuous simulation where enemies/pieces move even when
    the player doesn't provide input. Ghosts chase PacMan, the snake auto-moves,
    and Tetris blocks fall automatically.

    Default 10 FPS gives arcade-like gameplay without being too hectic.
    Jumanji games are JAX-based and step quickly.
    """

    def __init__(self, owner, target_hz: int = 10):
        """Initialize Jumanji arcade interaction controller.

        Args:
            owner: SessionController instance.
            target_hz: Target frame rate (default 10 FPS for arcade feel).
        """
        self._owner = owner
        self._interval_ms = max(1, int(1000 / float(target_hz)))  # 100ms for 10 FPS

    def idle_interval_ms(self) -> Optional[int]:
        return self._interval_ms

    def should_idle_tick(self) -> bool:
        """Check if we should advance the game this tick."""
        o = self._owner
        if o._adapter is None or o._game_id is None:
            return False
        if not getattr(o, "_game_started", False):
            return False
        if o._game_paused:
            return False
        if getattr(o._control_mode, "name", "") != "HUMAN_ONLY":
            return False
        if o._last_step is not None and (o._last_step.terminated or o._last_step.truncated):
            return False
        return True

    def maybe_passive_action(self) -> Optional[Any]:
        """Return NOOP/stay action for Jumanji arcade games.

        Most Jumanji games use Discrete action space where:
        - PacMan: action 0 is "no-op/stay"
        - Snake: action 0 is "no-op/continue"
        - Tetris: action 0 is "no-op" (block falls)
        """
        return 0

    def step_dt(self) -> float:
        return 0.0


class GriddlyInteractionController(InteractionController):
    """Idle controller for Griddly: step continuously with NOOP when idle.

    Griddly games are grid-based environments with state-based real-time input.
    The game should continuously step with NOOP actions when no keys are pressed,
    allowing the game world to advance (enemies move, timers count, etc.).

    Default 30 FPS provides responsive gameplay for grid-based environments.
    """

    def __init__(self, owner, target_hz: int = 30):
        """Initialize Griddly interaction controller.

        Args:
            owner: SessionController instance.
            target_hz: Target frame rate (default 30 FPS for responsive play).
        """
        self._owner = owner
        self._interval_ms = max(1, int(1000 / float(target_hz)))  # ~33ms for 30 FPS

    def idle_interval_ms(self) -> Optional[int]:
        return self._interval_ms

    def should_idle_tick(self) -> bool:
        """Check if we should advance the game this tick."""
        o = self._owner
        if o._adapter is None or o._game_id is None:
            return False
        if not getattr(o, "_game_started", False):
            return False
        if o._game_paused:
            return False
        if getattr(o._control_mode, "name", "") != "HUMAN_ONLY":
            return False
        if o._last_step is not None and (o._last_step.terminated or o._last_step.truncated):
            return False
        return True

    def maybe_passive_action(self) -> Optional[Any]:
        """Return NOOP action for Griddly.

        Griddly uses Discrete(5) action space where action 0 is NOOP.
        """
        return 0

    def step_dt(self) -> float:
        return 0.0
