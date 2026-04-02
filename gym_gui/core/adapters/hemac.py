"""HeMAC (Heterogeneous Multi-Agent Challenge) environment adapter.

HeMAC is a PettingZoo-based benchmark for Heterogeneous Multi-Agent RL (HeMARL)
with three agent types: Quadcopters, Observers, and Provisioners.

Published at ECAI 2025 by ThalesGroup.
Repository: https://github.com/ThalesGroup/hemac
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from gym_gui.core.adapters.base import (
    AdapterContext,
    AdapterStep,
    AgentSnapshot,
    EnvironmentAdapter,
    StepState,
)
from gym_gui.core.enums import ControlMode, GameId, RenderMode
from gym_gui.logging_config.log_constants import (
    LOG_ADAPTER_ENV_CREATED,
    LOG_ADAPTER_ENV_RESET,
    LOG_ADAPTER_STEP_SUMMARY,
)

_LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class HeMACRenderPayload:
    """Render payload for HeMAC containing heterogeneous agent state."""

    scenario: str  # e.g., "simple-fleet-3q1o", "fleet-10q3o", "complex-fleet-3q1o1p"
    agents: Dict[str, Dict[str, Any]]  # agent_name -> agent_state
    targets: List[Dict[str, Any]]  # List of target states
    obstacles: List[Dict[str, Any]]  # List of obstacle states
    current_agent: str
    move_count: int = 0
    is_game_over: bool = False
    team_reward: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": RenderMode.RGB_ARRAY.value,
            "scenario": self.scenario,
            "agents": self.agents,
            "targets": self.targets,
            "obstacles": self.obstacles,
            "current_agent": self.current_agent,
            "move_count": self.move_count,
            "is_game_over": self.is_game_over,
            "team_reward": self.team_reward,
        }


class HeMACEnvironmentAdapter(EnvironmentAdapter[Dict[str, Any], Any]):
    """Adapter for HeMAC heterogeneous multi-agent environments.

    HeMAC features three agent types with different capabilities:
    - Quadcopters: Agile, low-altitude, limited energy
    - Observers: High-altitude scouts with broad views
    - Provisioners: Ground vehicles on road networks

    Supports three challenge levels:
    - Simple Fleet: Quadcopters + Observers
    - Fleet: Energy constraints, obstacles, limited communication
    - Complex Fleet: All three agent types with maximum heterogeneity
    """

    supported_control_modes = (ControlMode.HUMAN_ONLY,)
    supported_render_modes = (RenderMode.RGB_ARRAY,)
    default_render_mode = RenderMode.RGB_ARRAY

    def __init__(
        self,
        context: AdapterContext | None = None,
        scenario: str = "simple-fleet-3q1o",
        n_drones: int = 3,
        n_observers: int = 1,
        n_provisioners: int = 0,
        max_cycles: int = 300,
    ) -> None:
        super().__init__(context)
        self._aec_env: Any = None
        self._scenario = scenario
        self._n_drones = n_drones
        self._n_observers = n_observers
        self._n_provisioners = n_provisioners
        self._max_cycles = max_cycles
        self._move_count: int = 0
        self._current_agent: str = ""
        self._team_reward: float = 0.0

    @property
    def id(self) -> str:
        return f"hemac-{self._scenario}-v0"

    def load(self) -> None:
        """Load the HeMAC environment."""
        try:
            from hemac import HeMAC_v0

            self._aec_env = HeMAC_v0.env(
                n_drones=self._n_drones,
                n_observers=self._n_observers,
                n_provisioners=self._n_provisioners,
                max_cycles=self._max_cycles,
                render_mode="rgb_array",
            )
            self.log_constant(
                LOG_ADAPTER_ENV_CREATED,
                extra={
                    "env_id": self.id,
                    "render_mode": "rgb_array",
                    "gym_kwargs": f"n_drones={self._n_drones}, n_observers={self._n_observers}, n_provisioners={self._n_provisioners}",
                    "wrapped_class": "HeMAC_v0.AECEnv",
                },
            )
        except ImportError as e:
            raise ImportError(
                "HeMAC is required. Install with: pip install -e 3rd_party/environments/hemac"
            ) from e

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> AdapterStep[Dict[str, Any]]:
        """Reset the HeMAC environment."""
        if self._aec_env is None:
            raise RuntimeError("Environment not loaded. Call load() first.")

        self._aec_env.reset(seed=seed)
        self._move_count = 0
        self._episode_step = 0
        self._episode_return = 0.0
        self._team_reward = 0.0
        self._current_agent = self._aec_env.agent_selection

        self.log_constant(
            LOG_ADAPTER_ENV_RESET,
            extra={
                "env_id": self.id,
                "seed": seed if seed is not None else "None",
                "has_options": bool(options),
            },
        )

        # Get initial observation
        obs, _, _, _, info = self._aec_env.last()

        return self._package_step(obs, 0.0, False, False, dict(info))

    def step(self, action: Any) -> AdapterStep[Dict[str, Any]]:
        """Execute an action in the HeMAC environment."""
        if self._aec_env is None:
            raise RuntimeError("Environment not loaded. Call load() first.")

        self._move_count += 1
        self._episode_step += 1

        # Execute step
        self._aec_env.step(action)

        # Get new observation
        obs, reward, terminated, truncated, info = self._aec_env.last()

        # Update current agent
        self._current_agent = self._aec_env.agent_selection

        # Update team reward (cooperative)
        self._team_reward += reward
        self._episode_return += reward

        self.log_constant(
            LOG_ADAPTER_STEP_SUMMARY,
            extra={
                "env_id": self.id,
                "episode_step": self._episode_step,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
            },
        )

        return self._package_step(obs, reward, terminated, truncated, dict(info))

    def close(self) -> None:
        """Clean up environment resources."""
        if self._aec_env is not None:
            try:
                self._aec_env.close()
            except Exception:
                pass
        self._aec_env = None

    def render(self) -> np.ndarray | None:
        """Get RGB render of current environment state."""
        if self._aec_env is None:
            return None
        try:
            return self._aec_env.render()
        except Exception:
            return None

    def _package_step(
        self,
        observation: Dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
    ) -> AdapterStep[Dict[str, Any]]:
        """Package step result with HeMAC-specific render payload."""
        # Extract agent states
        agents_state = {}
        if hasattr(self._aec_env, "agents"):
            for agent_name in self._aec_env.agents:
                agents_state[agent_name] = {
                    "type": self._get_agent_type(agent_name),
                    "active": agent_name == self._current_agent,
                }

        # Create render payload
        render_payload = HeMACRenderPayload(
            scenario=self._scenario,
            agents=agents_state,
            targets=[],  # Would need to extract from environment state
            obstacles=[],  # Would need to extract from environment state
            current_agent=self._current_agent,
            move_count=self._move_count,
            is_game_over=terminated or truncated,
            team_reward=self._team_reward,
        ).to_dict()

        # Build agent snapshots
        agent_snapshots = [
            AgentSnapshot(
                name=agent_name,
                role=self._get_agent_type(agent_name),
            )
            for agent_name in agents_state.keys()
        ]

        state = StepState(
            active_agent=self._current_agent,
            agents=agent_snapshots,
            metrics={
                "move_count": self._move_count,
                "episode_step": self._episode_step,
                "team_reward": self._team_reward,
            },
            environment={
                "game": "hemac",
                "scenario": self._scenario,
                "n_drones": self._n_drones,
                "n_observers": self._n_observers,
                "n_provisioners": self._n_provisioners,
            },
        )

        return AdapterStep(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            render_payload=render_payload,
            render_hint={"type": "rgb", "use_rgb": True},
            agent_id=self._current_agent,
            state=state,
        )

    def _get_agent_type(self, agent_name: str) -> str:
        """Determine agent type from agent name."""
        if "drone" in agent_name.lower() or "quadcopter" in agent_name.lower():
            return "quadcopter"
        elif "observer" in agent_name.lower():
            return "observer"
        elif "provisioner" in agent_name.lower():
            return "provisioner"
        return "unknown"

    # Required abstract methods from EnvironmentAdapter
    def gym_kwargs(self) -> dict[str, Any]:
        """Return additional kwargs for environment creation."""
        return {
            "n_drones": self._n_drones,
            "n_observers": self._n_observers,
            "n_provisioners": self._n_provisioners,
            "max_cycles": self._max_cycles,
        }

    def apply_wrappers(self, env: Any) -> Any:
        """Apply wrappers to environment (not used for HeMAC)."""
        return env

    def _require_env(self) -> Any:
        """Get underlying environment or raise if not loaded."""
        if self._aec_env is None:
            raise RuntimeError("Environment not loaded. Call load() first.")
        return self._aec_env

    def _set_env(self, env: Any) -> None:
        """Set the underlying environment."""
        self._aec_env = env

    def _resolve_default_render_mode(self) -> RenderMode:
        """Resolve the default render mode."""
        return self.default_render_mode


# Registry of HeMAC adapters
HEMAC_ADAPTERS: Dict[GameId, type[EnvironmentAdapter]] = {
    GameId.HEMAC_SIMPLE_FLEET_1Q1O: HeMACEnvironmentAdapter,
    GameId.HEMAC_SIMPLE_FLEET_3Q1O: HeMACEnvironmentAdapter,
    GameId.HEMAC_SIMPLE_FLEET_5Q2O: HeMACEnvironmentAdapter,
    GameId.HEMAC_FLEET_3Q1O: HeMACEnvironmentAdapter,
    GameId.HEMAC_FLEET_10Q3O: HeMACEnvironmentAdapter,
    GameId.HEMAC_FLEET_20Q5O: HeMACEnvironmentAdapter,
    GameId.HEMAC_COMPLEX_FLEET_3Q1O1P: HeMACEnvironmentAdapter,
    GameId.HEMAC_COMPLEX_FLEET_5Q2O1P: HeMACEnvironmentAdapter,
}


__all__ = [
    "HeMACEnvironmentAdapter",
    "HeMACRenderPayload",
    "HEMAC_ADAPTERS",
]
