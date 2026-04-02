"""MalmoEnv adapters — Microsoft Malmo (Java-based Minecraft) via the MalmoEnv Python package.

MalmoEnv connects directly to a running Minecraft instance (launched via launchClient.sh)
over a TCP socket on port 9000.  Each adapter wraps a specific mission XML file bundled
with the MalmoEnv package.

Before using any of these adapters, start Minecraft:
    cd 3rd_party/environments/malmo/Minecraft
    bash launch_malmo_java8.sh -port 9000 -env

    OR with explicit display:
    DISPLAY=:1 bash launch_malmo_java8.sh -port 9000 -env

Repository: 3rd_party/environments/malmo/
"""

from __future__ import annotations

import json
import re
import uuid as _uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from gym_gui.core.adapters.base import (
    AdapterStep,
    EnvironmentAdapter,
)
from gym_gui.core.enums import ControlMode, GameId, RenderMode, SteppingParadigm
from gym_gui.logging_config.log_constants import (
    LOG_ADAPTER_ENV_CLOSED,
    LOG_ADAPTER_ENV_CREATED,
    LOG_ADAPTER_ENV_RESET,
    LOG_ADAPTER_STEP_SUMMARY,
)

try:
    import malmoenv
    _MALMOENV_AVAILABLE = True
except ImportError:
    malmoenv = None
    _MALMOENV_AVAILABLE = False

# Path to bundled MalmoEnv mission XMLs
_MISSIONS_DIR = (
    Path(__file__).parents[3]
    / "3rd_party"
    / "environments"
    / "malmo"
    / "MalmoEnv"
    / "missions"
)

# Default MalmoEnv command ordering from source parser when using the built-in
# action_filter {"move", "turn", "use", "attack"}.
# NOTE: Actual action spaces are mission-specific (allow/deny lists can remove entries).
MALMOENV_ACTIONS = [
    "move 1",
    "move -1",
    "turn 1",
    "turn -1",
    "attack 1",
    "attack 0",
    "use 1",
    "use 0",
]

_DEFAULT_PORT = 9000
_DEFAULT_SERVER = "localhost"

# Default mission overrides (0 = no timeout, for exploration mode)
_DEFAULT_MISSION_OVERRIDES = {
    "time_limit_ms": 0,  # 0 = unlimited (exploration mode)
    "ms_per_tick": 50,
    "video_width": 320,
    "video_height": 240,
    "reward_goal": 1000,
    "reward_timeout": -1000,
    "reward_death": -10000,
}


def _apply_mission_overrides(xml_content: str, overrides: dict[str, Any]) -> str:
    """Apply mission parameter overrides to XML content.

    Modifies the mission XML in-place using regex replacements:
    - time_limit_ms: ServerQuitFromTimeUp timeLimitMs attribute (0 = remove timeout)
    - ms_per_tick: MsPerTick element
    - video_width/video_height: VideoProducer dimensions
    - reward_goal/reward_timeout/reward_death: RewardForMissionEnd values
    """
    # Apply MsPerTick override
    ms_per_tick = overrides.get("ms_per_tick")
    if ms_per_tick is not None:
        xml_content = re.sub(
            r"<MsPerTick>\d+</MsPerTick>",
            f"<MsPerTick>{int(ms_per_tick)}</MsPerTick>",
            xml_content,
        )

    # Apply time limit override
    time_limit_ms = overrides.get("time_limit_ms", 0)
    if time_limit_ms == 0:
        # Remove ServerQuitFromTimeUp element entirely for unlimited time
        xml_content = re.sub(
            r"<ServerQuitFromTimeUp[^>]*/>",
            "<!-- ServerQuitFromTimeUp removed (unlimited time) -->",
            xml_content,
        )
        xml_content = re.sub(
            r"<ServerQuitFromTimeUp[^>]*>.*?</ServerQuitFromTimeUp>",
            "<!-- ServerQuitFromTimeUp removed (unlimited time) -->",
            xml_content,
            flags=re.DOTALL,
        )
    else:
        # Update timeLimitMs attribute
        xml_content = re.sub(
            r'timeLimitMs="\d+"',
            f'timeLimitMs="{int(time_limit_ms)}"',
            xml_content,
        )

    # Apply video resolution overrides
    video_width = overrides.get("video_width")
    video_height = overrides.get("video_height")
    if video_width is not None:
        xml_content = re.sub(
            r"<Width>\d+</Width>",
            f"<Width>{int(video_width)}</Width>",
            xml_content,
        )
    if video_height is not None:
        xml_content = re.sub(
            r"<Height>\d+</Height>",
            f"<Height>{int(video_height)}</Height>",
            xml_content,
        )

    # Apply reward overrides
    reward_goal = overrides.get("reward_goal")
    if reward_goal is not None:
        xml_content = re.sub(
            r'rewardForDeath="[^"]*"',
            f'rewardForDeath="{int(reward_goal)}"',
            xml_content,
        )

    # Remove command quota for multi-agent (idle ticks count as commands)
    command_quota = overrides.get("command_quota")
    if command_quota == 0:
        # Remove AgentQuitFromReachingCommandQuota entirely
        xml_content = re.sub(
            r"<AgentQuitFromReachingCommandQuota[^>]*/>",
            "<!-- AgentQuitFromReachingCommandQuota removed (unlimited) -->",
            xml_content,
        )

    return xml_content


class MalmoEnvAdapter(EnvironmentAdapter[np.ndarray, int]):
    """Base adapter for MalmoEnv (Microsoft Malmo) missions.

    Overrides the full lifecycle because MalmoEnv does not use gym.make() —
    it requires ``env.init(xml, port)`` and connects to a running Minecraft
    server.  The observation is an RGB numpy array; ``step()`` returns the
    old 4-tuple gym format which we translate to a standard AdapterStep.

    Mission overrides can be passed via gym_kwargs to modify:
    - time_limit_ms: Mission timeout (0 = unlimited for exploration)
    - ms_per_tick: Tick rate (default 50ms = 20 tps)
    - video_width/video_height: Resolution
    - port/server: Connection settings
    """

    mission_xml: str = "mobchase_single_agent.xml"  # override in subclasses
    malmo_port: int = _DEFAULT_PORT
    malmo_server: str = _DEFAULT_SERVER

    default_render_mode = RenderMode.MOSAIC_MALMO
    supported_render_modes = (RenderMode.MOSAIC_MALMO, RenderMode.RGB_ARRAY)
    supported_control_modes = (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    )

    _malmo_env: Any = None
    _last_obs: np.ndarray | None = None
    _last_info: dict[str, Any] = {}
    _mission_overrides: dict[str, Any] = {}

    @staticmethod
    def _as_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _as_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_info_payload(info: Any) -> dict[str, Any]:
        if isinstance(info, Mapping):
            return dict(info)
        if isinstance(info, str):
            raw = info.strip()
            if not raw:
                return {}
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return {"malmo_info_raw": raw}
            if isinstance(parsed, Mapping):
                return dict(parsed)
            return {"malmo_info_raw": raw}
        return {}

    @classmethod
    def _normalize_info(
        cls,
        info: Any,
        *,
        reward: float,
        episode_step: int,
        episode_score: float,
    ) -> dict[str, Any]:
        info_dict = cls._parse_info_payload(info)
        info_dict["episode_step"] = episode_step
        info_dict["episode_score"] = episode_score
        info_dict["reward"] = reward

        x = cls._as_float(info_dict.get("XPos"))
        y = cls._as_float(info_dict.get("YPos"))
        z = cls._as_float(info_dict.get("ZPos"))
        if x is not None and y is not None and z is not None:
            info_dict["agent_pos"] = [x, y, z]

        yaw = cls._as_float(info_dict.get("Yaw"))
        if yaw is not None:
            info_dict["agent_yaw"] = yaw

        pitch = cls._as_float(info_dict.get("Pitch"))
        if pitch is not None:
            info_dict["agent_pitch"] = pitch

        tick = cls._as_int(info_dict.get("TotalTime"))
        if tick is None:
            tick = cls._as_int(info_dict.get("WorldTime"))
        if tick is not None:
            info_dict["tick"] = tick

        if "ms_per_tick" not in info_dict:
            info_dict["ms_per_tick"] = 50

        for source, target in (
            ("Life", "life"),
            ("Food", "food"),
            ("Air", "air"),
            ("XP", "xp"),
            ("Score", "score"),
            ("IsAlive", "is_alive"),
        ):
            if source in info_dict and target not in info_dict:
                info_dict[target] = info_dict[source]

        return info_dict

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        if not _MALMOENV_AVAILABLE or malmoenv is None:
            raise RuntimeError(
                "malmoenv package is not installed. "
                "Run: pip install -e 3rd_party/environments/malmo/MalmoEnv/"
            )
        xml_path = _MISSIONS_DIR / self.mission_xml
        if not xml_path.exists():
            raise FileNotFoundError(f"Mission XML not found: {xml_path}")
        xml_content = xml_path.read_text()

        # Apply mission overrides from config
        merged_overrides = dict(_DEFAULT_MISSION_OVERRIDES)
        merged_overrides.update(self._mission_overrides)
        xml_content = _apply_mission_overrides(xml_content, merged_overrides)

        # Get port/server from overrides or use class defaults
        port = merged_overrides.get("port", self.malmo_port)
        server = merged_overrides.get("server", self.malmo_server)

        env = malmoenv.Env()
        try:
            env.init(xml_content, port, server=server, reshape=True)
        except (ConnectionRefusedError, OSError) as exc:
            raise RuntimeError(
                f"Cannot connect to Minecraft on {server}:{port}.\n\n"
                f"Start Minecraft first:\n"
                f"  ./run_malmo.sh\n\n"
                f"Wait for 'CLIENT enter state: DORMANT' then try again."
            ) from exc

        # Inject "move 0" (NOOP) into the action space for passive ticking
        # This allows the GUI to step the environment without causing movement
        try:
            if env.action_space and hasattr(env.action_space, "actions"):
                actions = list(env.action_space.actions)
                if "noop 1" not in actions:
                    actions.append("noop 1")
                    env.action_space = malmoenv.ActionSpace(actions)
        except Exception:
            pass

        self._malmo_env = env
        self.log_constant(
            LOG_ADAPTER_ENV_CREATED,
            extra={
                "env_id": self.id,
                "render_mode": RenderMode.MOSAIC_MALMO.value,
                "gym_kwargs": f"port={port},server={server},overrides={merged_overrides}",
                "wrapped_class": "malmoenv.Env",
            },
        )

    def set_mission_overrides(self, overrides: dict[str, Any]) -> None:
        """Set mission parameter overrides before loading.

        Should be called before load() to apply custom mission settings.
        """
        self._mission_overrides = dict(overrides)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> AdapterStep[np.ndarray]:
        if self._malmo_env is None:
            raise RuntimeError(f"Adapter '{self.id}' has not been loaded.")
        self._episode_step = 0
        self._episode_return = 0.0
        obs = self._malmo_env.reset()
        if obs is None or len(obs) == 0:
            h = getattr(self._malmo_env, "height", 84)
            w = getattr(self._malmo_env, "width", 84)
            obs = np.zeros((h, w, 3), dtype=np.uint8)
        obs = np.flipud(np.asarray(obs, dtype=np.uint8))
        self._last_obs = obs
        self._last_info = {"episode_step": 0}
        self.log_constant(
            LOG_ADAPTER_ENV_RESET,
            extra={
                "env_id": self.id,
                "seed": seed if seed is not None else "None",
                "has_options": bool(options),
            },
        )
        return self._package_step(obs, 0.0, False, False, self._last_info)

    def step(self, action: int) -> AdapterStep[np.ndarray]:
        if self._malmo_env is None:
            raise RuntimeError(f"Adapter '{self.id}' has not been loaded.")
        # MalmoEnv returns old gym 4-tuple: (obs, reward, done, info)
        obs, reward, done, info = self._malmo_env.step(action)
        if obs is None or len(obs) == 0:
            h = getattr(self._malmo_env, "height", 84)
            w = getattr(self._malmo_env, "width", 84)
            obs = (
                self._last_obs
                if self._last_obs is not None
                else np.zeros((h, w, 3), dtype=np.uint8)
            )
        obs = np.flipud(np.asarray(obs, dtype=np.uint8))
        r = float(reward) if reward is not None else 0.0
        self._episode_step += 1
        self._episode_return += r
        info_dict = self._normalize_info(
            info,
            reward=r,
            episode_step=self._episode_step,
            episode_score=self._episode_return,
        )
        self._last_obs = obs
        self._last_info = info_dict
        self.log_constant(
            LOG_ADAPTER_STEP_SUMMARY,
            extra={
                "env_id": self.id,
                "action": repr(action),
                "reward": r,
                "terminated": done,
                "truncated": False,
            },
        )
        return self._package_step(obs, r, bool(done), False, info_dict)

    def close(self) -> None:
        if self._malmo_env is not None:
            self.log_constant(LOG_ADAPTER_ENV_CLOSED, extra={"env_id": self.id})
            try:
                self._malmo_env.close()
            except Exception:
                pass
            self._malmo_env = None

    @property
    def action_space(self):  # type: ignore[override]
        if self._malmo_env is not None:
            return self._malmo_env.action_space
        return None

    @property
    def observation_space(self):  # type: ignore[override]
        if self._malmo_env is not None:
            return self._malmo_env.observation_space
        return None

    def render(self) -> dict[str, Any]:
        """Return an ``{"rgb": ndarray}`` payload for MosaicMalmoRendererStrategy."""
        frame = self._last_obs
        if frame is None:
            h = getattr(self._malmo_env, "height", 84) if self._malmo_env else 84
            w = getattr(self._malmo_env, "width", 84) if self._malmo_env else 84
            frame = np.zeros((h, w, 3), dtype=np.uint8)
        payload: dict[str, Any] = {
            "mode": RenderMode.MOSAIC_MALMO.value,
            "rgb": np.asarray(frame),
            "mission": self.mission_xml.replace(".xml", ""),
        }
        for key in (
            "ms_per_tick",
            "agent_pos",
            "agent_yaw",
            "agent_pitch",
            "tick",
            "reward",
            "life",
            "food",
            "air",
            "xp",
            "score",
            "is_alive",
        ):
            if key in self._last_info:
                payload[key] = self._last_info[key]
        return payload


# ---------------------------------------------------------------------------
# Mission-specific subclasses
# ---------------------------------------------------------------------------

class MalmoEnvMobChaseAdapter(MalmoEnvAdapter):
    id = GameId.MALMOENV_MOBCHASE.value
    mission_xml = "mobchase_single_agent.xml"


class MalmoEnvMazeRunnerAdapter(MalmoEnvAdapter):
    id = GameId.MALMOENV_MAZERUNNER.value
    mission_xml = "mazerunner.xml"


class MalmoEnvVerticalAdapter(MalmoEnvAdapter):
    id = GameId.MALMOENV_VERTICAL.value
    mission_xml = "vertical.xml"


class MalmoEnvCliffWalkingAdapter(MalmoEnvAdapter):
    id = GameId.MALMOENV_CLIFFWALKING.value
    mission_xml = "cliffwalking.xml"


class MalmoEnvCatchTheMobAdapter(MalmoEnvAdapter):
    id = GameId.MALMOENV_CATCHTHEMOB.value
    mission_xml = "catchthemob.xml"


class MalmoEnvFindTheGoalAdapter(MalmoEnvAdapter):
    id = GameId.MALMOENV_FINDTHEGOAL.value
    mission_xml = "findthegoal.xml"


class MalmoEnvAtticAdapter(MalmoEnvAdapter):
    id = GameId.MALMOENV_ATTIC.value
    mission_xml = "attic.xml"


class MalmoEnvDefaultFlatWorldAdapter(MalmoEnvAdapter):
    id = GameId.MALMOENV_DEFAULTFLATWORLD.value
    mission_xml = "defaultflatworld.xml"


class MalmoEnvDefaultWorldAdapter(MalmoEnvAdapter):
    id = GameId.MALMOENV_DEFAULTWORLD.value
    mission_xml = "defaultworld.xml"


class MalmoEnvEatingAdapter(MalmoEnvAdapter):
    id = GameId.MALMOENV_EATING.value
    mission_xml = "eating.xml"


class MalmoEnvObstaclesAdapter(MalmoEnvAdapter):
    id = GameId.MALMOENV_OBSTACLES.value
    mission_xml = "obstacles.xml"


class MalmoEnvTrickyArenaAdapter(MalmoEnvAdapter):
    id = GameId.MALMOENV_TRICKYARENA.value
    mission_xml = "trickyarena.xml"


class MalmoEnvTreasureHuntAdapter(MalmoEnvAdapter):
    id = GameId.MALMOENV_TREASUREHUNT.value
    mission_xml = "treasurehunt.xml"


# ---------------------------------------------------------------------------
# Multi-agent adapter
# ---------------------------------------------------------------------------


class MalmoEnvMultiAgentAdapter(MalmoEnvAdapter):
    """Adapter managing N agents in a single MalmoEnv mission.

    Creates one ``malmoenv.Env`` per role (0 … agent_count-1), all sharing
    the same ``exp_uid`` so they join the same server-side mission.  The
    render payload includes per-agent frames under an ``"agents"`` key so
    the renderer can display a split view.

    TreasureHunt is turn-based, so ``step`` advances each agent sequentially
    (role 0 first, then role 1, etc.).
    """

    # Subclasses set these
    mission_xml: str = "treasurehunt.xml"
    expected_agent_count: int = 2

    _agents: list[Any] = []
    _agent_obs: list[np.ndarray | None] = []
    _agent_info: list[dict[str, Any]] = []

    def load(self) -> None:
        if not _MALMOENV_AVAILABLE or malmoenv is None:
            raise RuntimeError(
                "malmoenv package is not installed. "
                "Run: pip install -e 3rd_party/environments/malmo/MalmoEnv/"
            )
        xml_path = _MISSIONS_DIR / self.mission_xml
        if not xml_path.exists():
            raise FileNotFoundError(f"Mission XML not found: {xml_path}")
        xml_content = xml_path.read_text()

        merged_overrides = dict(_DEFAULT_MISSION_OVERRIDES)
        # Multi-agent: remove command quota (idle ticks count as commands
        # and would exhaust the quota in seconds)
        merged_overrides["command_quota"] = 0
        merged_overrides.update(self._mission_overrides)
        xml_content = _apply_mission_overrides(xml_content, merged_overrides)

        port = merged_overrides.get("port", self.malmo_port)
        server = merged_overrides.get("server", self.malmo_server)

        # MalmoEnv uses consecutive ports: agent0=port, agent1=port+1, etc.
        # NativeInputHandler is now on malmoPort+1000 (no conflict).
        # Generate a shared experiment UID so all agents join the same mission.
        exp_uid = str(_uuid.uuid4())

        self._agents = []
        self._agent_obs = []
        self._agent_info = []

        expected_ports = [port + r for r in range(self.expected_agent_count)]

        for role in range(self.expected_agent_count):
            env = malmoenv.Env()
            try:
                env.init(
                    xml_content,
                    port,                   # coordinator port (always agent 0)
                    server=server,
                    port2=(port + role),     # this agent's MalmoEnv port
                    role=role,
                    exp_uid=exp_uid,
                    reshape=True,
                )
            except (ConnectionRefusedError, OSError) as exc:
                # Clean up any already-connected agents
                for prev_env in self._agents:
                    try:
                        prev_env.close()
                    except Exception:
                        pass
                self._agents = []
                raise RuntimeError(
                    f"Cannot connect agent role={role} to Minecraft on "
                    f"{server}:{port + role}.\n\n"
                    f"For multi-agent missions (like TreasureHunt), you need "
                    f"{self.expected_agent_count} Minecraft clients.\n\n"
                    f"Start them with:\n"
                    f"  ./run_malmo.sh --agents {self.expected_agent_count}\n\n"
                    f"This launches clients on ports {expected_ports}.\n"
                    f"Wait for ALL to show 'DORMANT', then try again."
                ) from exc

            # Inject NOOP into each agent's action space
            try:
                if env.action_space and hasattr(env.action_space, "actions"):
                    actions = list(env.action_space.actions)
                    if "noop 1" not in actions:
                        actions.append("noop 1")
                        env.action_space = malmoenv.ActionSpace(actions)
            except Exception:
                pass

            self._agents.append(env)
            self._agent_obs.append(None)
            self._agent_info.append({})

        # Expose agent 0's spaces as the primary adapter spaces
        self._malmo_env = self._agents[0]

        self.log_constant(
            LOG_ADAPTER_ENV_CREATED,
            extra={
                "env_id": self.id,
                "render_mode": RenderMode.MOSAIC_MALMO.value,
                "gym_kwargs": (
                    f"port={port},server={server},"
                    f"agents={self.expected_agent_count},"
                    f"exp_uid={exp_uid}"
                ),
                "wrapped_class": "malmoenv.Env (multi-agent)",
            },
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> AdapterStep[np.ndarray]:
        if not self._agents:
            raise RuntimeError(f"Adapter '{self.id}' has not been loaded.")

        self._episode_step = 0
        self._episode_return = 0.0

        # Reset agents CONCURRENTLY in threads.  MalmoEnv's turn-based
        # protocol requires all agents to call reset() at the same time;
        # sequential reset deadlocks because agent 0 blocks waiting for
        # agent 1 to join the mission.
        import threading

        errors: list[Exception | None] = [None] * len(self._agents)

        def _reset_one(role: int, env: Any) -> None:
            try:
                obs = env.reset()
                if obs is None or len(obs) == 0:
                    h = getattr(env, "height", 84)
                    w = getattr(env, "width", 84)
                    obs = np.zeros((h, w, 3), dtype=np.uint8)
                obs = np.flipud(np.asarray(obs, dtype=np.uint8))
                self._agent_obs[role] = obs
                self._agent_info[role] = {"episode_step": 0}
            except Exception as exc:
                errors[role] = exc

        threads = [
            threading.Thread(target=_reset_one, args=(role, env))
            for role, env in enumerate(self._agents)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)

        # Check for errors
        for role, err in enumerate(errors):
            if err is not None:
                raise RuntimeError(
                    f"Agent role={role} failed to reset: {err}"
                ) from err

        obs_0 = self._agent_obs[0]
        assert obs_0 is not None, "Agent 0 obs must be set after reset"
        self._last_obs = obs_0
        # Include multi-agent metadata so the session/keyboard widget knows
        agent_names = [f"agent_{r}" for r in range(len(self._agents))]
        reset_info = dict(self._agent_info[0])
        reset_info["num_agents"] = len(self._agents)
        reset_info["agents"] = agent_names
        self._last_info = reset_info
        self.log_constant(
            LOG_ADAPTER_ENV_RESET,
            extra={
                "env_id": self.id,
                "seed": seed if seed is not None else "None",
                "has_options": bool(options),
                "agent_count": len(self._agents),
            },
        )
        return self._package_step(
            obs_0, 0.0, False, False, reset_info
        )

    def step(self, action: Any) -> AdapterStep[np.ndarray]:
        """Step agents with the given action(s).

        Accepts:
          - int: steps role 0 with that action, NOOPs for others
          - list/tuple: steps each role with the corresponding action
          - dict: passed directly to step_multi
        """
        if isinstance(action, dict):
            return self.step_multi(action)
        if isinstance(action, (list, tuple)):
            actions = {i: int(a) for i, a in enumerate(action)}
            for role in range(len(action), len(self._agents)):
                actions[role] = self._resolve_noop(role)
            return self.step_multi(actions)
        # Single int: role 0 acts, others NOOP
        actions = {0: int(action)}
        for role in range(1, len(self._agents)):
            actions[role] = self._resolve_noop(role)
        return self.step_multi(actions)

    def step_agent(self, role: int, action: int) -> AdapterStep[np.ndarray]:
        """Step a single agent while sending NOOP to others."""
        actions = {}
        for r in range(len(self._agents)):
            actions[r] = action if r == role else self._resolve_noop(r)
        return self.step_multi(actions)

    def step_multi(self, actions: dict[int, int]) -> AdapterStep[np.ndarray]:
        """Step each agent with its own action.  Turn-based: role 0 first."""
        if not self._agents:
            raise RuntimeError(f"Adapter '{self.id}' has not been loaded.")

        total_reward = 0.0
        any_done = False

        for role, env in enumerate(self._agents):
            act = actions.get(role, self._resolve_noop(role))
            obs, reward, done, info = env.step(act)

            if obs is None or len(obs) == 0:
                h = getattr(env, "height", 84)
                w = getattr(env, "width", 84)
                obs = (
                    self._agent_obs[role]
                    if self._agent_obs[role] is not None
                    else np.zeros((h, w, 3), dtype=np.uint8)
                )
            obs = np.flipud(np.asarray(obs, dtype=np.uint8))
            r = float(reward) if reward is not None else 0.0
            total_reward += r
            any_done = any_done or bool(done)

            info_dict = self._normalize_info(
                info,
                reward=r,
                episode_step=self._episode_step + 1,
                episode_score=self._episode_return + total_reward,
            )
            self._agent_obs[role] = obs
            self._agent_info[role] = info_dict

        self._episode_step += 1
        self._episode_return += total_reward
        obs_0 = self._agent_obs[0]
        assert obs_0 is not None, "Agent 0 obs must be set after step"
        self._last_obs = obs_0
        self._last_info = self._agent_info[0]

        self.log_constant(
            LOG_ADAPTER_STEP_SUMMARY,
            extra={
                "env_id": self.id,
                "action": repr(actions),
                "reward": total_reward,
                "terminated": any_done,
                "truncated": False,
            },
        )
        return self._package_step(
            obs_0, total_reward, any_done, False, self._agent_info[0]
        )

    def render(self) -> dict[str, Any]:
        """Return multi-agent render payload with per-agent frames."""
        # Build per-agent payloads
        agent_payloads: list[dict[str, Any]] = []
        for role in range(len(self._agents)):
            frame = self._agent_obs[role] if role < len(self._agent_obs) else None
            if frame is None:
                env = self._agents[role] if role < len(self._agents) else None
                h = getattr(env, "height", 84) if env else 84
                w = getattr(env, "width", 84) if env else 84
                frame = np.zeros((h, w, 3), dtype=np.uint8)
            agent_payload: dict[str, Any] = {
                "role": role,
                "rgb": np.asarray(frame),
            }
            info = self._agent_info[role] if role < len(self._agent_info) else {}
            for key in (
                "ms_per_tick", "agent_pos", "agent_yaw", "agent_pitch",
                "tick", "reward", "life", "food", "air", "xp", "score", "is_alive",
            ):
                if key in info:
                    agent_payload[key] = info[key]
            agent_payloads.append(agent_payload)

        # Primary payload uses agent 0's frame for backward compat
        payload = dict(super().render())
        payload["agents"] = agent_payloads
        payload["agent_count"] = len(self._agents)
        payload["multi_agent"] = True
        return payload

    def close(self) -> None:
        self.log_constant(LOG_ADAPTER_ENV_CLOSED, extra={"env_id": self.id})
        for env in self._agents:
            try:
                env.close()
            except Exception:
                pass
        self._agents = []
        self._agent_obs = []
        self._agent_info = []
        self._malmo_env = None

    @property
    def agent_count(self) -> int:
        return len(self._agents)

    def get_agent_action_space(self, role: int) -> Any:
        """Return the action space for a specific agent role."""
        if role < len(self._agents):
            return self._agents[role].action_space
        return None

    def _resolve_noop(self, role: int) -> int:
        """Find NOOP action index for the given agent role."""
        if role >= len(self._agents):
            return 0
        env = self._agents[role]
        actions = getattr(env.action_space, "actions", [])
        for noop_cmd in ("noop 1", "move 0", "turn 0"):
            try:
                return actions.index(noop_cmd)
            except ValueError:
                continue
        return 0


class MalmoEnvTreasureHuntMultiAgentAdapter(MalmoEnvMultiAgentAdapter):
    """TreasureHunt — 2-agent cooperative mission (turn-based, discrete)."""
    id = GameId.MALMOENV_TREASUREHUNT.value
    mission_xml = "treasurehunt.xml"
    expected_agent_count = 2
    stepping_paradigm = SteppingParadigm.SEQUENTIAL


# Adapter registry — imported by adapters/__init__.py and factories/adapters.py
MALMOENV_ADAPTERS = {
    GameId.MALMOENV_MOBCHASE: MalmoEnvMobChaseAdapter,
    GameId.MALMOENV_MAZERUNNER: MalmoEnvMazeRunnerAdapter,
    GameId.MALMOENV_VERTICAL: MalmoEnvVerticalAdapter,
    GameId.MALMOENV_CLIFFWALKING: MalmoEnvCliffWalkingAdapter,
    GameId.MALMOENV_CATCHTHEMOB: MalmoEnvCatchTheMobAdapter,
    GameId.MALMOENV_FINDTHEGOAL: MalmoEnvFindTheGoalAdapter,
    GameId.MALMOENV_ATTIC: MalmoEnvAtticAdapter,
    GameId.MALMOENV_DEFAULTFLATWORLD: MalmoEnvDefaultFlatWorldAdapter,
    GameId.MALMOENV_DEFAULTWORLD: MalmoEnvDefaultWorldAdapter,
    GameId.MALMOENV_EATING: MalmoEnvEatingAdapter,
    GameId.MALMOENV_OBSTACLES: MalmoEnvObstaclesAdapter,
    GameId.MALMOENV_TRICKYARENA: MalmoEnvTrickyArenaAdapter,
    # TreasureHunt requires 2 agents — always use the multi-agent adapter
    GameId.MALMOENV_TREASUREHUNT: MalmoEnvTreasureHuntMultiAgentAdapter,
}

# Multi-agent adapter registry — missions that require >1 agent.
# When multi_agent=True is requested, these override the single-agent registry.
MALMOENV_MULTI_AGENT_ADAPTERS = {
    GameId.MALMOENV_TREASUREHUNT: MalmoEnvTreasureHuntMultiAgentAdapter,
}

# Legacy aliases so any existing imports of the old names still resolve
MOSAIC_MALMO_ADAPTERS = MALMOENV_ADAPTERS
MOSAIC_MALMO_ACTIONS = MALMOENV_ACTIONS
