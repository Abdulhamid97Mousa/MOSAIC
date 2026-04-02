"""Runtime for Random Worker.

Supports two execution modes:

**Interactive mode** (``--interactive``):
    Action-selector protocol — GUI owns the environment, worker just picks actions.

    IN  → {"cmd": "init_agent", "game_name": "...", "player_id": "agent_0"}
    OUT ← {"type": "agent_ready", "run_id": "...", ...}

    IN  → {"cmd": "select_action", "observation": [...], "player_id": "agent_0"}
    OUT ← {"type": "action_selected", "action": 3, ...}

**Autonomous mode** (default, no ``--interactive``):
    Env-owning protocol — worker creates and owns the environment, GUI sends
    ``reset``/``step``/``stop`` commands.  Used by Script Mode experiments.

    IN  → {"cmd": "reset", "seed": 42}
    OUT ← {"type": "ready", "run_id": "...", "seed": 42, "render_payload": {...}}

    IN  → {"cmd": "step"}
    OUT ← {"type": "step", "action": 3, "reward": 0.0, ...}
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

from random_worker.config import RandomWorkerConfig

logger = logging.getLogger(__name__)


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


class RandomWorkerRuntime:
    """Lightweight subprocess that selects random actions.

    Supports both interactive (action-selector) and autonomous (env-owning)
    execution modes.
    """

    def __init__(self, config: RandomWorkerConfig) -> None:
        self.config = config
        self._action_space: Optional[gym.spaces.Space] = None
        self._step_count = 0

        # Autonomous mode state
        self._env: Optional[gym.Env] = None
        self._raw_action_space: Optional[gym.spaces.Space] = None
        self._agent_keys: Optional[list] = None
        self._episode_index = 0
        self._episode_reward = 0.0

    def _emit(self, data: Dict[str, Any]) -> None:
        """Emit JSON response to stdout."""
        json_str = json.dumps(data, default=_json_default)
        print(json_str, flush=True)
        if data.get("type") == "step":
            logger.info(
                "Step: action=%s reward=%.2f",
                data.get("action"), data.get("reward"),
            )

    def _resolve_action_space(self, game_name: str) -> gym.spaces.Space:
        """Determine the action space from the game/task name."""
        task = self.config.task or game_name

        # Special handling for Crafter (uses old gym API, not gymnasium)
        if "Crafter" in task:
            try:
                import crafter
                env = crafter.Env()
                action_space = gym.spaces.Discrete(env.action_space.n)
                env.close()
                logger.info("Resolved Crafter action space: %s", action_space)
                return action_space
            except ImportError:
                logger.warning("Crafter not installed, defaulting to Discrete(17)")
                return gym.spaces.Discrete(17)

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

    def _select_action(self, action_mask: Optional[List] = None) -> int:
        """Select a uniformly random action, respecting an optional legal-action mask.

        Args:
            action_mask: A list/array of 0/1 values (length = action space size).
                         When provided, only indices where the value is 1 are sampled.
                         This is critical for environments like chess_v6 where sampling
                         from the full Discrete(4672) space almost always yields an
                         illegal move (only ~20 of 4672 actions are legal per turn).
        """
        if self._action_space is None:
            return 0
        if action_mask is not None:
            legal_indices = [i for i, v in enumerate(action_mask) if v]
            if legal_indices:
                return int(np.random.choice(legal_indices))
        return int(self._action_space.sample())

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
        """Handle init_agent: resolve action space, seed RNG."""
        game_name = cmd.get("game_name", "")
        player_id = cmd.get("player_id", "agent")

        self._action_space = self._resolve_action_space(game_name)

        if self.config.seed is not None:
            self._action_space.seed(self.config.seed)

        self._step_count = 0

        logger.info(
            "Agent ready: player=%s game=%s action_space=%s",
            player_id, game_name, self._action_space,
        )

        return {
            "type": "agent_ready",
            "run_id": self.config.run_id,
            "game_name": game_name,
            "player_id": player_id,
            "mode": "action_selector",
            "behavior": "random",
        }

    def handle_select_action(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        """Handle select_action: sample a random legal action and return it."""
        if self._action_space is None:
            return {
                "type": "error",
                "message": "Action space not initialized. Send init_agent first.",
            }

        player_id = cmd.get("player_id", "unknown")
        action_mask = cmd.get("action_mask")  # list of 0/1, length = action space size
        action = self._select_action(action_mask=action_mask)
        self._step_count += 1

        logger.info(
            "Action selected: player=%s action=%s (step %d)",
            player_id, action, self._step_count,
        )

        return {
            "type": "action_selected",
            "run_id": self.config.run_id,
            "player_id": player_id,
            "action": action,
            "action_str": str(action),
        }

    # ── Autonomous Mode Handlers ─────────────────────────────────────

    def _create_env(self) -> gym.Env:
        """Create the gymnasium environment for autonomous mode."""
        task = self.config.task
        if not task:
            raise ValueError("--task is required for autonomous (non-interactive) mode")

        # Special handling for Crafter (uses old gym API, not gymnasium)
        if "Crafter" in task:
            try:
                import crafter
                from gymnasium import Env as GymEnv, spaces

                class CrafterGymnasiumWrapper(GymEnv):
                    """Wrap crafter.Env for gymnasium compatibility."""

                    def __init__(self, crafter_env):
                        self._env = crafter_env
                        self.action_space = spaces.Discrete(crafter_env.action_space.n)
                        self.observation_space = spaces.Box(
                            low=0, high=255,
                            shape=crafter_env.observation_space.shape,
                            dtype=crafter_env.observation_space.dtype
                        )
                        self.action_names = crafter_env.action_names
                        self._render_mode = "rgb_array"

                    @property
                    def render_mode(self):
                        return self._render_mode

                    def reset(self, seed=None, options=None):
                        if seed is not None:
                            import numpy as np
                            np.random.seed(seed)
                        obs = self._env.reset()
                        return obs, {}

                    def step(self, action):
                        obs, reward, done, info = self._env.step(action)
                        return obs, reward, done, False, info

                    def render(self):
                        return self._env.render()

                    def close(self):
                        return self._env.close()

                # Use same defaults as CrafterConfig for high-quality rendering
                # area=(64, 64), view=(9, 9), size=(512, 512), reward=True, length=10000
                env = crafter.Env(
                    area=(64, 64),
                    view=(9, 9),
                    size=(512, 512),
                    reward=True,
                    length=10000,
                )
                logger.info("Creating Crafter env with gymnasium wrapper: %s (size=512x512)", task)
                return CrafterGymnasiumWrapper(env)
            except ImportError as e:
                logger.warning("Crafter not installed: %s", e)
                raise ValueError(f"Crafter package not installed: {e}")

        kwargs: Dict[str, Any] = {"render_mode": "rgb_array", "disable_env_checker": True}
        _ensure_env_registered(task)

        logger.info("Creating env: %s", task)
        return gym.make(task, **kwargs)

    def handle_reset(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        """Handle reset: reset or create env, return ready response."""
        seed = cmd.get("seed")

        if self._env is None:
            self._env = self._create_env()
            self._raw_action_space = self._env.action_space
            self._action_space = self._raw_action_space
            if hasattr(self._raw_action_space, "spaces"):
                self._agent_keys = list(self._raw_action_space.spaces.keys())
                first_key = self._agent_keys[0]
                self._action_space = self._raw_action_space.spaces[first_key]

            # Seed the action space RNG with config.seed (derived from run_id in launcher).
            # This ensures each operator has independent random actions.
            # The env.reset(seed=...) only seeds the environment layout, not the action space.
            if self.config.seed is not None:
                self._action_space.seed(self.config.seed)
                logger.info(
                    "Seeded action space with config.seed=%s (derived from run_id)",
                    self.config.seed,
                )

        self._step_count = 0
        self._episode_reward = 0.0

        # Only seed the env layout — do NOT re-seed the action space here.
        # Both operators receive the same layout seed (for reproducibility), but
        # the action space uses config.seed for independent random sequences.
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
        """Build the action to send to env.step().

        For single-agent envs, returns a single int.
        For multi-agent envs (Dict action space), returns a dict mapping
        each agent key to a sampled action.
        """
        if self._agent_keys is not None:
            return {key: self._select_action() for key in self._agent_keys}
        return self._select_action()

    def handle_step(self, cmd: Dict[str, Any]) -> None:
        """Handle step: pick action, step env, emit step + possibly episode_end."""
        if self._env is None or self._action_space is None:
            self._emit({
                "type": "error",
                "message": "Environment not initialized. Send reset first.",
            })
            return

        env_action = self._build_env_action()
        if isinstance(env_action, dict):
            report_action = next(iter(env_action.values()))
        else:
            report_action = env_action

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
            "worker": "random_worker",
            "behavior": "random",
        })

        logger.info(
            "Random worker started [interactive] (run_id=%s)",
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
            logger.info("Random worker stopped")

    def run_autonomous(self) -> None:
        """Autonomous mode: own the env, handle reset/step/stop commands."""
        self._emit({
            "type": "init",
            "run_id": self.config.run_id,
            "worker": "random_worker",
            "behavior": "random",
            "mode": "autonomous",
        })

        logger.info(
            "Random worker started [autonomous] (run_id=%s, task=%s)",
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
            logger.info("Random worker stopped")


def _json_default(obj: Any) -> Any:
    """JSON serialization helper for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
