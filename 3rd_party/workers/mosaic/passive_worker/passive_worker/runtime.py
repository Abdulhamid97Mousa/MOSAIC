"""Runtime for Passive Worker — NOOP / STILL only.

The passive worker always selects the environment's "do nothing" action:

1. **NOOP** — action 0 when the environment maps index 0 to a passive
   meaning such as ``"still"``, ``"noop"``, ``"idle"``, or ``"wait"``.
2. **STILL** — if action 0 is *not* passive (e.g. MiniGrid where 0 is
   "Turn Left"), the worker scans the action-meaning list for the first
   entry whose name matches a passive keyword and uses that index instead.

Supports two execution modes:

**Interactive mode** (``--interactive``):
    Action-selector protocol — GUI owns the environment, worker always
    selects the resolved passive action.

    IN  → {"cmd": "init_agent", "game_name": "...", "player_id": "agent_0"}
    OUT ← {"type": "agent_ready", "run_id": "...", ...}

    IN  → {"cmd": "select_action", "observation": [...], "player_id": "agent_0"}
    OUT ← {"type": "action_selected", "action": 0, ...}

**Autonomous mode** (default, no ``--interactive``):
    Env-owning protocol — worker creates and owns the environment, GUI sends
    ``reset``/``step``/``stop`` commands.

    IN  → {"cmd": "reset", "seed": 42}
    OUT ← {"type": "ready", "run_id": "...", "seed": 42, "render_payload": {...}}

    IN  → {"cmd": "step"}
    OUT ← {"type": "step", "action": 0, "reward": 0.0, ...}
    OUT ← {"type": "episode_end", ...}   (when terminated or truncated)

Both modes:
    IN  → {"cmd": "stop"}
    OUT ← {"type": "stopped"}
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np

from passive_worker.config import PassiveWorkerConfig

logger = logging.getLogger(__name__)

_PASSIVE_KEYWORDS = frozenset({"still", "noop", "idle", "wait", "done"})


def _ensure_env_registered(task: str) -> None:
    """Import the package that registers the gymnasium environment."""
    try:
        if "MosaicMultiGrid" in task or "MultiGrid" in task:
            import mosaic_multigrid.envs  # noqa: F401
        elif "MiniGrid" in task or "BabyAI" in task:
            from minigrid import register_minigrid_envs
            register_minigrid_envs()
    except ImportError:
        pass


class PassiveWorkerRuntime:
    """Lightweight subprocess that always selects the passive (NOOP / STILL) action.

    Serves as a passive baseline for comparison against RL, LLM, and human
    decision-makers.  Useful for measuring environment dynamics under a
    do-nothing policy and for fault-isolation testing.
    """

    NOOP_ACTION = 0

    def __init__(self, config: PassiveWorkerConfig) -> None:
        self.config = config
        self._action_space: Optional[gym.spaces.Space] = None
        self._passive_action: int = self.NOOP_ACTION
        self._step_count = 0

        # Autonomous mode state
        self._env: Optional[gym.Env] = None
        self._raw_action_space: Optional[gym.spaces.Space] = None
        self._agent_keys: Optional[list] = None
        self._episode_index = 0
        self._episode_reward = 0.0

    def _emit(self, data: Dict[str, Any]) -> None:
        """Emit JSON response to stdout and also log to stderr for debugging."""
        json_str = json.dumps(data, default=_json_default)
        print(json_str, flush=True)
        # Log to stderr for debugging (appears in operator log file)
        if data.get("type") == "action_selected":
            logger.info(
                "Action selected: player=%s action=%s (passive)",
                data.get("player_id"), data.get("action"),
            )
        elif data.get("type") == "step":
            logger.info(
                "Step: action=%s reward=%.2f (passive)",
                data.get("action"), data.get("reward"),
            )

    def _resolve_action_space(self, game_name: str) -> gym.spaces.Space:
        """Determine the action space from the game/task name."""
        task = self.config.task or game_name

        try:
            _ensure_env_registered(task)
            env = gym.make(task, render_mode=None, disable_env_checker=True)
            action_space = env.action_space
            env.close()

            if hasattr(action_space, "spaces"):
                first_key = next(iter(action_space.spaces))
                action_space = action_space.spaces[first_key]

            logger.info("Resolved action space from %s: %s", task, action_space)
            return action_space
        except Exception as exc:
            logger.warning(
                "Could not create env %s: %s — defaulting to Discrete(7)",
                task, exc,
            )

        return gym.spaces.Discrete(7)

    @staticmethod
    def _resolve_passive_action(
        action_space: gym.spaces.Space,
        action_meanings: Optional[List[str]] = None,
    ) -> int:
        """Return the best passive (NOOP / STILL) action index.

        Strategy:
        1. If *action_meanings* is provided, scan for the first entry whose
           lower-cased name is in ``_PASSIVE_KEYWORDS``.
        2. Otherwise fall back to action 0 (the NOOP default).
        """
        if action_meanings:
            for idx, name in enumerate(action_meanings):
                if name.strip().lower() in _PASSIVE_KEYWORDS:
                    return idx
        return PassiveWorkerRuntime.NOOP_ACTION

    # ── Render Helpers ───────────────────────────────────────────────

    def _render_payload(self) -> Optional[Dict[str, Any]]:
        """Capture an RGB render frame from the environment."""
        if self._env is None:
            return None
        try:
            frame = self._env.render()
            if frame is not None and isinstance(frame, np.ndarray):
                h, w = frame.shape[:2]
                return {
                    "mode": "rgb",
                    "rgb": frame.tolist(),
                    "width": int(w),
                    "height": int(h),
                }
        except Exception as exc:
            logger.debug("Render failed: %s", exc)
        return None

    # ── Interactive Mode Handlers ────────────────────────────────────

    def handle_init_agent(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        """Handle init_agent: resolve action space and passive action."""
        game_name = cmd.get("game_name", "")
        player_id = cmd.get("player_id", "agent")

        self._action_space = self._resolve_action_space(game_name)

        meanings: Optional[List[str]] = cmd.get("action_meanings")
        self._passive_action = self._resolve_passive_action(
            self._action_space, meanings,
        )
        self._step_count = 0

        behavior = "noop" if self._passive_action == self.NOOP_ACTION else "still"
        logger.info(
            "Agent ready: player=%s game=%s action_space=%s behavior=%s passive_action=%d",
            player_id, game_name, self._action_space, behavior, self._passive_action,
        )

        return {
            "type": "agent_ready",
            "run_id": self.config.run_id,
            "game_name": game_name,
            "player_id": player_id,
            "mode": "action_selector",
            "behavior": behavior,
            "passive_action": self._passive_action,
        }

    def handle_select_action(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        """Handle select_action: return the resolved passive action (NOOP or STILL)."""
        if self._action_space is None:
            return {
                "type": "error",
                "message": "Action space not initialized. Send init_agent first.",
            }

        player_id = cmd.get("player_id", "unknown")
        self._step_count += 1

        return {
            "type": "action_selected",
            "run_id": self.config.run_id,
            "player_id": player_id,
            "action": self._passive_action,
            "action_str": str(self._passive_action),
        }

    # ── Autonomous Mode Handlers ─────────────────────────────────────

    def _create_env(self) -> gym.Env:
        """Create the gymnasium environment for autonomous mode."""
        task = self.config.task
        if not task:
            raise ValueError("--task is required for autonomous (non-interactive) mode")

        kwargs: Dict[str, Any] = {"render_mode": "rgb_array", "disable_env_checker": True}
        _ensure_env_registered(task)

        logger.info("Creating env: %s", task)
        return gym.make(task, **kwargs)

    def handle_reset(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        """Handle reset: reset or create env, resolve passive action, return ready."""
        seed = cmd.get("seed")

        if self._env is None:
            self._env = self._create_env()
            self._raw_action_space = self._env.action_space
            self._action_space = self._raw_action_space
            if hasattr(self._raw_action_space, "spaces"):
                self._agent_keys = list(self._raw_action_space.spaces.keys())
                first_key = self._agent_keys[0]
                self._action_space = self._raw_action_space.spaces[first_key]

            meanings: Optional[List[str]] = None
            unwrapped = self._env.unwrapped
            if hasattr(unwrapped, "get_action_meanings"):
                meanings = unwrapped.get_action_meanings()
            elif hasattr(unwrapped, "ACTION_MEANINGS"):
                meanings = list(unwrapped.ACTION_MEANINGS)
            self._passive_action = self._resolve_passive_action(
                self._action_space, meanings,
            )

        self._step_count = 0
        self._episode_reward = 0.0

        reset_kwargs: Dict[str, Any] = {}
        if seed is not None:
            reset_kwargs["seed"] = seed
        obs, info = self._env.reset(**reset_kwargs)

        obs_shape: Optional[List[int]] = None
        if isinstance(obs, np.ndarray):
            obs_shape = list(obs.shape)
        elif isinstance(obs, dict):
            first_val = next(iter(obs.values()))
            if isinstance(first_val, np.ndarray):
                obs_shape = list(first_val.shape)

        payload = self._render_payload()

        logger.info(
            "Env reset: task=%s seed=%s action_space=%s episode=%d",
            self.config.task, seed, self._action_space, self._episode_index,
        )

        return {
            "type": "ready",
            "run_id": self.config.run_id,
            "env_id": self.config.task,
            "seed": seed,
            "observation_shape": obs_shape,
            "step_index": 0,
            "episode_index": self._episode_index,
            "episode_reward": 0.0,
            "render_payload": payload,
        }

    def _build_env_action(self) -> Any:
        """Build passive action for env.step().

        For single-agent envs, returns the resolved passive action index.
        For multi-agent envs (Dict action space), returns a dict mapping
        each agent key to the passive action.
        """
        if self._agent_keys is not None:
            return {key: self._passive_action for key in self._agent_keys}
        return self._passive_action

    def handle_step(self, cmd: Dict[str, Any]) -> None:
        """Handle step: pick passive action, step env, emit step + possibly episode_end."""
        if self._env is None or self._action_space is None:
            self._emit({
                "type": "error",
                "message": "Environment not initialized. Send reset first.",
            })
            return

        env_action = self._build_env_action()
        report_action = self._passive_action

        obs, reward, terminated, truncated, info = self._env.step(env_action)

        if isinstance(reward, dict):
            reward = sum(reward.values())
        if isinstance(terminated, dict):
            terminated = any(terminated.values())
        if isinstance(truncated, dict):
            truncated = any(truncated.values())

        self._episode_reward += float(reward)
        payload = self._render_payload()

        self._emit({
            "type": "step",
            "run_id": self.config.run_id,
            "step_index": self._step_count,
            "episode_index": self._episode_index,
            "action": report_action,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "episode_reward": self._episode_reward,
            "render_payload": payload,
        })

        self._step_count += 1

        if terminated or truncated:
            self._emit({
                "type": "episode_end",
                "run_id": self.config.run_id,
                "episode_index": self._episode_index,
                "episode_steps": self._step_count,
                "episode_return": self._episode_reward,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "render_payload": payload,
            })
            self._episode_index += 1

    # ── Main Loops ───────────────────────────────────────────────────

    def run(self) -> None:
        """Interactive mode: read JSON commands from stdin (action-selector protocol)."""
        self._emit({
            "type": "init",
            "run_id": self.config.run_id,
            "worker": "passive_worker",
            "behavior": "noop",
        })

        logger.info(
            "Passive worker started [interactive] (run_id=%s)",
            self.config.run_id,
        )

        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue

                try:
                    cmd = json.loads(line)
                except json.JSONDecodeError as exc:
                    self._emit({"type": "error", "message": f"Invalid JSON: {exc}"})
                    continue

                cmd_type = cmd.get("cmd", "").lower()

                if cmd_type == "init_agent":
                    self._emit(self.handle_init_agent(cmd))
                elif cmd_type == "select_action":
                    self._emit(self.handle_select_action(cmd))
                elif cmd_type == "reset":
                    # Also handle env-owning reset/step for single-agent manual mode
                    try:
                        self._emit(self.handle_reset(cmd))
                    except Exception as exc:
                        self._emit({"type": "error", "message": f"Reset failed: {exc}"})
                elif cmd_type == "step":
                    try:
                        self.handle_step(cmd)
                    except Exception as exc:
                        self._emit({"type": "error", "message": f"Step failed: {exc}"})
                elif cmd_type == "stop":
                    self._emit({"type": "stopped"})
                    break
                elif cmd_type == "ping":
                    self._emit({"type": "pong"})
                else:
                    self._emit({
                        "type": "error",
                        "message": f"Unknown command: {cmd_type}",
                    })

        except KeyboardInterrupt:
            logger.info("Interrupted")
        finally:
            if self._env is not None:
                try:
                    self._env.close()
                except Exception:
                    pass
            logger.info("Passive worker stopped")

    def run_autonomous(self) -> None:
        """Autonomous mode: own the env, handle reset/step/stop commands."""
        self._emit({
            "type": "init",
            "run_id": self.config.run_id,
            "worker": "passive_worker",
            "behavior": "noop",
            "mode": "autonomous",
        })

        logger.info(
            "Passive worker started [autonomous] (run_id=%s, task=%s)",
            self.config.run_id, self.config.task,
        )

        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue

                try:
                    cmd = json.loads(line)
                except json.JSONDecodeError as exc:
                    self._emit({"type": "error", "message": f"Invalid JSON: {exc}"})
                    continue

                cmd_type = cmd.get("cmd", "").lower()

                if cmd_type == "reset":
                    try:
                        self._emit(self.handle_reset(cmd))
                    except Exception as exc:
                        self._emit({"type": "error", "message": f"Reset failed: {exc}"})
                elif cmd_type == "step":
                    try:
                        self.handle_step(cmd)
                    except Exception as exc:
                        self._emit({"type": "error", "message": f"Step failed: {exc}"})
                elif cmd_type == "stop":
                    self._emit({"type": "stopped"})
                    break
                elif cmd_type == "ping":
                    self._emit({"type": "pong"})
                else:
                    self._emit({
                        "type": "error",
                        "message": f"Unknown command: {cmd_type}",
                    })

        except KeyboardInterrupt:
            logger.info("Interrupted")
        finally:
            if self._env is not None:
                try:
                    self._env.close()
                except Exception:
                    pass
            logger.info("Passive worker stopped")


def _json_default(obj: Any) -> Any:
    """JSON serialization helper for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
