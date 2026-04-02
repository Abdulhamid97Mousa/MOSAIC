"""Human Worker Runtime - Owns gymnasium environment, waits for human input.

This worker implements human-in-the-loop control for gymnasium environments:
1. Worker owns the gymnasium environment
2. GUI sends "reset" command → worker creates/resets env, returns RGB frame
3. GUI displays frame, shows action buttons
4. User clicks action button → GUI sends "step" command with action
5. Worker steps env, returns RGB frame + reward
6. Repeat until episode ends

Protocol:
    GUI -> Worker: {"cmd": "reset", "seed": 42, "env_name": "minigrid", "task": "MiniGrid-Empty-8x8-v0"}
    Worker -> GUI: {"type": "ready", "action_labels": [...], "render_payload": {...}}

    GUI -> Worker: {"cmd": "step", "action": 1}
    Worker -> GUI: {"type": "step", "reward": 0.0, "render_payload": {...}}

    GUI -> Worker: {"cmd": "stop"}
    Worker -> GUI: {"type": "stopped"}
"""

import json
import logging
import select
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np

from .config import HumanWorkerConfig, get_action_labels

logger = logging.getLogger("human_worker")


class HumanInteractiveRuntime:
    """Interactive runtime for human operators - owns gymnasium environment.

    Reads JSON commands from stdin, executes actions, emits results to stdout.
    The environment lives in this subprocess, not in the GUI process.
    """

    def __init__(self, config: HumanWorkerConfig):
        """Initialize the human worker runtime.

        Args:
            config: Worker configuration.
        """
        self.config = config
        self._env: Optional[gym.Env] = None
        self._action_space_n: int = 0
        self._action_labels: List[str] = []

        # Episode state (start at -1 so first reset makes it 0)
        self._step_index: int = 0
        self._episode_index: int = -1
        self._total_reward: float = 0.0
        self._episode_start_time: Optional[datetime] = None

        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    def _emit(self, data: Dict[str, Any]) -> None:
        """Emit JSON response to stdout for GUI consumption."""
        print(json.dumps(data), flush=True)

    def _create_env(self, env_name: str, task: str, seed: int) -> gym.Env:
        """Create gymnasium environment.

        Args:
            env_name: Environment family (minigrid, babyai, etc.)
            task: Specific environment task (MiniGrid-Empty-8x8-v0, etc.)
            seed: Random seed

        Returns:
            Created gymnasium environment
        """
        # Register environment families that need registration
        if env_name in ("minigrid", "babyai"):
            try:
                import minigrid
                # Only register if not already in registry
                if "MiniGrid-Empty-5x5-v0" not in gym.envs.registry:
                    minigrid.register_minigrid_envs()
            except ImportError:
                logger.warning("minigrid not installed, environment creation may fail")

        # MultiGrid uses old gym API (not gymnasium)
        if env_name == "multigrid":
            try:
                import gym as old_gym
                from mosaic_multigrid.envs import SoccerGame4HEnv10x15N2, CollectGame4HEnv10x10N2

                logger.info(f"Creating MultiGrid environment: {task}")

                if task == "MultiGrid-Soccer-v0":
                    env = SoccerGame4HEnv10x15N2()
                elif task == "MultiGrid-Collect-v0":
                    env = CollectGame4HEnv10x10N2()
                else:
                    env = old_gym.make(task, render_mode="rgb_array")

                env.reset()
                return env
            except ImportError as e:
                logger.warning(f"mosaic_multigrid not installed: {e}")
                raise

        # Crafter uses its own API, not gym.make()
        if env_name == "crafter":
            try:
                import crafter

                logger.info(f"Creating Crafter environment: {task}")

                # Determine reward mode from task name
                reward = "dense" if "Reward" in task else "sparse"
                # Use config.game_resolution for render size (default 512x512)
                size = self.config.game_resolution
                logger.info(f"Crafter render size: {size[0]}x{size[1]}")
                env = crafter.Env(reward=reward == "dense", size=size)

                # Wrap with gymnasium compatibility
                from gymnasium.wrappers import TransformReward
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

                return CrafterGymnasiumWrapper(env)
            except ImportError as e:
                logger.warning(f"crafter not installed: {e}")
                raise

        # Determine the environment ID to use with gym.make
        # For most envs, task IS the gym ID (MiniGrid-Empty-8x8-v0)
        # For some, we need to construct it
        env_id = task

        logger.info(f"Creating environment: {env_id} (family: {env_name})")

        env = gym.make(env_id, render_mode=self.config.render_mode)
        return env

    def _get_render_payload(self) -> Dict[str, Any]:
        """Get render payload from current environment state.

        Returns:
            Dict with rgb frame data for GUI display
        """
        if self._env is None:
            return {}

        try:
            frame = self._env.render()
            if frame is None:
                return {}

            if isinstance(frame, np.ndarray):
                h, w = int(frame.shape[0]), int(frame.shape[1])
                return {
                    "mode": "rgb",
                    "rgb": frame.tolist(),
                    "width": w,
                    "height": h,
                }
            return {}
        except Exception as e:
            logger.warning(f"Failed to render: {e}")
            return {}

    def _apply_minigrid_custom_state(self, state_json: str) -> bool:
        """Apply custom grid state to a MiniGrid environment.

        Args:
            state_json: JSON string containing the custom grid state

        Returns:
            True if state was applied successfully
        """
        if self._env is None:
            return False

        try:
            state_dict = json.loads(state_json)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid MiniGrid state JSON: {e}")
            return False

        try:
            # Import MiniGrid object types
            from minigrid.core.world_object import (
                Wall, Goal, Key, Door, Ball, Box, Lava
            )

            # Map our object types to MiniGrid classes
            obj_type_map = {
                "wall": lambda color: Wall(),
                "goal": lambda color: Goal(),
                "lava": lambda color: Lava(),
                "key": lambda color: Key(color=color if color != "none" else "yellow"),
                "door": lambda color: Door(color=color if color != "none" else "yellow"),
                "ball": lambda color: Ball(color=color if color != "none" else "blue"),
                "box": lambda color: Box(color=color if color != "none" else "red"),
            }

            unwrapped = self._env.unwrapped
            grid = unwrapped.grid
            rows = state_dict.get("rows", unwrapped.height)
            cols = state_dict.get("cols", unwrapped.width)

            # Clear the interior of the grid (keep walls on border)
            for x in range(1, cols - 1):
                for y in range(1, rows - 1):
                    grid.set(x, y, None)

            # Place objects from state
            for cell_data in state_dict.get("cells", []):
                row = cell_data.get("row", 0)
                col = cell_data.get("col", 0)
                # Convert (row, col) to MiniGrid's (x, y) where x=col, y=row
                x, y = col, row

                for obj_data in cell_data.get("objects", []):
                    obj_type = obj_data.get("type", "empty")
                    color = obj_data.get("color", "none")

                    if obj_type in obj_type_map:
                        obj = obj_type_map[obj_type](color)
                        grid.set(x, y, obj)

            # Set agent position and direction
            agent_pos = state_dict.get("agent_pos")
            if agent_pos:
                # Convert (row, col) to (x, y)
                agent_row, agent_col = agent_pos
                unwrapped.agent_pos = (agent_col, agent_row)

            agent_dir = state_dict.get("agent_dir", 0)
            unwrapped.agent_dir = agent_dir

            logger.info(
                f"Applied MiniGrid state: {rows}x{cols}, "
                f"agent_pos={unwrapped.agent_pos}, agent_dir={agent_dir}"
            )
            return True

        except Exception as e:
            logger.warning(f"Failed to apply MiniGrid state: {e}")
            return False

    def _handle_reset(self, cmd: Dict[str, Any]) -> None:
        """Handle reset command - create/reset environment.

        Args:
            cmd: Command dict with optional seed, env_name, task, settings
        """
        try:
            # Get parameters from command or config
            seed = cmd.get("seed", self.config.seed)
            env_name = cmd.get("env_name", self.config.env_name)
            task = cmd.get("task", self.config.task)
            settings = cmd.get("settings", {})

            # Update config
            self.config.env_name = env_name
            self.config.task = task
            self.config.seed = seed

            # Close existing environment
            if self._env is not None:
                try:
                    self._env.close()
                except Exception:
                    pass

            # Create new environment
            self._env = self._create_env(env_name, task, seed)

            # Get action space info (convert to int for JSON serialization)
            if hasattr(self._env.action_space, 'n'):
                self._action_space_n = int(self._env.action_space.n)
            else:
                self._action_space_n = 0

            # Get action labels
            self._action_labels = get_action_labels(env_name, task, self._action_space_n)

            # Reset environment
            obs, info = self._env.reset(seed=seed)

            # Apply custom initial state if provided (for MiniGrid, etc.)
            initial_state = settings.get("initial_state") if settings else None
            if initial_state and env_name in ("minigrid", "babyai"):
                if self._apply_minigrid_custom_state(initial_state):
                    logger.info("Applied custom MiniGrid initial state")

            # Reset episode state
            self._step_index = 0
            self._episode_index += 1
            self._total_reward = 0.0
            self._episode_start_time = datetime.utcnow()

            # Get render payload
            render_payload = self._get_render_payload()

            # Emit ready response
            self._emit({
                "type": "ready",
                "run_id": self.config.run_id,
                "env_name": env_name,
                "task": task,
                "seed": seed,
                "action_space": self._action_space_n,
                "action_labels": self._action_labels,
                "render_payload": render_payload,
                "episode_index": self._episode_index - 1,
                "step_index": 0,
            })

            logger.info(
                f"Environment reset: {task}, seed={seed}, "
                f"actions={self._action_space_n}"
            )

        except Exception as e:
            logger.exception(f"Reset failed: {e}")
            self._emit({"type": "error", "message": str(e)})

    def _handle_step(self, cmd: Dict[str, Any]) -> None:
        """Handle step command - execute action on environment.

        Args:
            cmd: Command dict with action index
        """
        if self._env is None:
            self._emit({
                "type": "error",
                "message": "Environment not initialized. Send reset first."
            })
            return

        try:
            action = cmd.get("action", 0)

            # Validate action
            if not (0 <= action < self._action_space_n):
                self._emit({
                    "type": "error",
                    "message": f"Invalid action {action}. Valid: 0-{self._action_space_n - 1}"
                })
                return

            # Step environment
            obs, reward, terminated, truncated, info = self._env.step(action)

            # Update state
            self._step_index += 1
            self._total_reward += reward

            # Get render payload
            render_payload = self._get_render_payload()

            # Get action label for logging
            action_label = self._action_labels[action] if action < len(self._action_labels) else f"Action {action}"

            # Emit step response
            step_response = {
                "type": "step",
                "run_id": self.config.run_id,
                "step_index": self._step_index - 1,
                "action": action,
                "action_label": action_label,
                "reward": float(reward),
                "terminated": terminated,
                "truncated": truncated,
                "episode_index": self._episode_index - 1,
                "episode_reward": self._total_reward,
                "render_payload": render_payload,
                "timestamp": datetime.utcnow().isoformat(),
            }
            self._emit(step_response)

            logger.info(
                f"Step {self._step_index}: action={action_label}, "
                f"reward={reward:.2f}, done={terminated or truncated}"
            )

            # Check for episode end
            if terminated or truncated:
                self._emit_episode_done(terminated, truncated, info)

        except Exception as e:
            logger.exception(f"Step failed: {e}")
            self._emit({"type": "error", "message": str(e)})

    def _emit_episode_done(
        self, terminated: bool, truncated: bool, info: Dict[str, Any]
    ) -> None:
        """Emit episode completion event.

        Args:
            terminated: Whether episode terminated naturally
            truncated: Whether episode was truncated
            info: Info dict from environment
        """
        end_time = datetime.utcnow()
        duration = 0.0
        if self._episode_start_time:
            duration = (end_time - self._episode_start_time).total_seconds()

        success = info.get("success", terminated and self._total_reward > 0)

        self._emit({
            "type": "episode_done",
            "run_id": self.config.run_id,
            "episode_index": self._episode_index - 1,
            "total_reward": self._total_reward,
            "num_steps": self._step_index,
            "terminated": terminated,
            "truncated": truncated,
            "success": bool(success),
            "duration_seconds": duration,
        })

        logger.info(
            f"Episode {self._episode_index} done: "
            f"reward={self._total_reward:.2f}, steps={self._step_index}, "
            f"success={success}"
        )

    def run(self) -> None:
        """Main loop - read commands from stdin, execute, respond."""
        logger.info(
            f"Human Interactive Runtime started. "
            f"Waiting for commands on stdin..."
        )

        # Emit init message
        self._emit({
            "type": "init",
            "run_id": self.config.run_id,
            "player_name": self.config.player_name,
            "version": "2.0",  # New interactive architecture
        })

        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue

                try:
                    cmd = json.loads(line)
                except json.JSONDecodeError as e:
                    self._emit({"type": "error", "message": f"Invalid JSON: {e}"})
                    continue

                cmd_type = cmd.get("cmd", "").lower()

                if cmd_type == "reset":
                    self._handle_reset(cmd)
                elif cmd_type == "step":
                    self._handle_step(cmd)
                elif cmd_type == "stop":
                    logger.info("Stop command received")
                    self._emit({"type": "stopped", "run_id": self.config.run_id})
                    break
                elif cmd_type == "ping":
                    self._emit({"type": "pong", "run_id": self.config.run_id})
                else:
                    self._emit({
                        "type": "error",
                        "message": f"Unknown command: {cmd_type}"
                    })

        except KeyboardInterrupt:
            logger.info("Interrupted")
        finally:
            # Cleanup
            if self._env is not None:
                try:
                    self._env.close()
                except Exception:
                    pass
            logger.info("Human Interactive Runtime stopped")


# Legacy runtime for board game move selection (backwards compatibility)
class HumanWorkerRuntime:
    """Legacy Human Worker Runtime - waits for human input via GUI.

    This class is for board game move selection where the GUI owns
    the environment. For single-agent environments (MiniGrid, etc.),
    use HumanInteractiveRuntime instead.
    """

    def __init__(self, config: HumanWorkerConfig):
        """Initialize the human worker.

        Args:
            config: Worker configuration.
        """
        self.config = config
        self._player_id: str = ""
        self._game_name: str = ""
        self._waiting_for_input: bool = False
        self._current_legal_moves: List[str] = []

        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    def init_agent(self, game_name: str, player_id: str) -> None:
        """Initialize the agent for a game.

        Args:
            game_name: Name of the game (e.g., "chess_v6").
            player_id: Player identifier (e.g., "player_0").
        """
        self._game_name = game_name
        self._player_id = player_id
        self._waiting_for_input = False
        self._current_legal_moves = []

        logger.info(
            f"Human agent initialized for {game_name} as {player_id} "
            f"(player: {self.config.player_name})"
        )

    def request_human_input(
        self,
        observation: str,
        legal_moves: List[str],
        board_str: str,
    ) -> None:
        """Signal that we're waiting for human input.

        Args:
            observation: Current game observation string.
            legal_moves: List of legal moves (UCI for chess).
            board_str: String representation of the board.
        """
        self._waiting_for_input = True
        self._current_legal_moves = legal_moves

        # Emit signal to GUI
        self._emit({
            "type": "waiting_for_human",
            "run_id": self.config.run_id,
            "player_id": self._player_id,
            "player_name": self.config.player_name,
            "legal_moves": legal_moves,
            "show_legal_moves": self.config.show_legal_moves,
            "confirm_moves": self.config.confirm_moves,
            "timestamp": datetime.utcnow().isoformat(),
        })

        logger.info(f"Waiting for human input ({len(legal_moves)} legal moves)")

    def process_human_input(self, move: str) -> Dict[str, Any]:
        """Process human input and validate the move.

        Args:
            move: The move selected by the human (e.g., "e7e5").

        Returns:
            Dict with action_str, success, and optional error message.
        """
        if not self._waiting_for_input:
            logger.warning("Received human input but not waiting for input")
            return {
                "action_str": move,
                "success": False,
                "error": "Not waiting for input",
            }

        # Validate move against legal moves
        if move not in self._current_legal_moves:
            logger.warning(f"Invalid move '{move}' - not in legal moves")
            return {
                "action_str": move,
                "success": False,
                "error": f"Invalid move '{move}'. Legal moves: {', '.join(self._current_legal_moves[:10])}...",
            }

        self._waiting_for_input = False
        logger.info(f"Human selected move: {move}")

        return {
            "action_str": move,
            "success": True,
        }

    def run_interactive(self) -> None:
        """Run in interactive mode, reading commands from stdin.

        Protocol:
            - init_agent: Initialize for a game/player
            - select_action: Start waiting for human input
            - human_input: Receive the human's move selection
            - stop: Terminate gracefully
        """
        logger.info(
            f"Human Worker started for {self.config.player_name}. "
            "Waiting for commands on stdin..."
        )

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                cmd = json.loads(line)
                cmd_type = cmd.get("cmd") or cmd.get("type", "")

                if cmd_type == "init_agent":
                    game_name = cmd.get("game_name", "chess_v6")
                    player_id = cmd.get("player_id", "player_0")
                    self.init_agent(game_name, player_id)
                    self._emit({
                        "type": "agent_initialized",
                        "run_id": self.config.run_id,
                        "game_name": game_name,
                        "player_id": player_id,
                        "player_name": self.config.player_name,
                    })

                elif cmd_type == "select_action":
                    # GUI is asking us to select an action
                    # We emit 'waiting_for_human' and wait for 'human_input'
                    observation = cmd.get("observation", "")
                    info = cmd.get("info", {})
                    legal_moves = info.get("legal_moves", [])

                    self.request_human_input(observation, legal_moves, observation)
                    # Don't emit action_selected yet - wait for human_input

                elif cmd_type == "human_input":
                    # Human has made their selection via GUI
                    move = cmd.get("move", "")
                    player_id = cmd.get("player_id", self._player_id)

                    result = self.process_human_input(move)

                    self._emit({
                        "type": "action_selected",
                        "run_id": self.config.run_id,
                        "player_id": player_id,
                        "action": result["action_str"],
                        "action_str": result["action_str"],
                        "success": result["success"],
                        "error": result.get("error", ""),
                        "source": "human",
                        "player_name": self.config.player_name,
                        "timestamp": datetime.utcnow().isoformat(),
                    })

                elif cmd_type == "cancel_input":
                    # GUI cancelled the human input request
                    self._waiting_for_input = False
                    logger.info("Human input cancelled")
                    self._emit({
                        "type": "input_cancelled",
                        "run_id": self.config.run_id,
                        "player_id": self._player_id,
                    })

                elif cmd_type == "stop":
                    logger.info("Stop command received, exiting")
                    self._emit({"type": "stopped", "run_id": self.config.run_id})
                    break

                else:
                    logger.warning(f"Unknown command type: {cmd_type}")

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
            except Exception as e:
                logger.exception(f"Command processing error: {e}")
                self._emit({"type": "error", "message": str(e)})

    def _emit(self, data: Dict[str, Any]) -> None:
        """Emit a response to stdout."""
        print(json.dumps(data), flush=True)


class HumanKeyboardRuntime:
    """Keyboard-driven human worker that reads a physical USB keyboard.

    Runs in a subprocess, owns a single ``/dev/input/eventX`` device,
    and responds to ``select_action`` commands by waiting for the human
    to press a recognized key combination on the assigned keyboard.

    This keeps keyboard I/O and key-to-action resolution entirely off
    the Qt main thread, solving the UI-freeze problem that occurs when
    ``adapter.step()`` blocks the event loop.

    Protocol (JSON over stdin/stdout)::

        GUI -> Worker: {"cmd": "init_agent", "game_name": "MultiGrid-Soccer-v0",
                        "player_id": "agent_0"}
        Worker -> GUI: {"type": "agent_initialized", ...}

        GUI -> Worker: {"cmd": "select_action", "player_id": "agent_0"}
        (Worker blocks reading keyboard until a key is pressed)
        Worker -> GUI: {"type": "action_selected", "action": 3, ...}

        GUI -> Worker: {"cmd": "stop"}
        Worker -> GUI: {"type": "stopped"}
    """

    def __init__(self, config: HumanWorkerConfig, device_path: str, mouse_path: Optional[str] = None) -> None:
        self.config = config
        self._device_path = device_path
        self._mouse_path = mouse_path
        self._player_id: str = ""
        self._game_name: str = ""
        self._resolver = None
        self._reader = None
        self._mouse_reader = None
        self._setup_logging()

    def _setup_logging(self) -> None:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    def _emit(self, data: Dict[str, Any]) -> None:
        """Emit JSON response to stdout."""
        print(json.dumps(data), flush=True)

    def _handle_init_agent(self, cmd: Dict[str, Any]) -> None:
        """Handle init_agent: open keyboard device and create resolver."""
        from .evdev_input import EvdevDeviceReader, create_resolver

        self._game_name = cmd.get("game_name", "")
        self._player_id = cmd.get("player_id", "")
        env_name = self.config.env_name or "multigrid"

        # Create the keycode resolver.  Priority:
        # 1. key_action_map: dynamic mapping from GUI (works for ALL envs)
        # 2. action_list: Malmo's mission-specific action names
        # 3. env_name: fallback to hardcoded per-environment resolver
        key_action_map = cmd.get("key_action_map")
        action_list = cmd.get("action_list")
        self._resolver = create_resolver(
            env_name,
            action_list=action_list,
            key_action_map=key_action_map,
        )

        # Open the evdev device
        self._reader = EvdevDeviceReader(self._device_path)
        try:
            self._reader.open()
        except PermissionError:
            msg = (
                f"Permission denied: {self._device_path}. "
                f"Run: sudo usermod -a -G input $USER"
            )
            logger.error(msg)
            self._emit({"type": "error", "message": msg})
            return
        except OSError as exc:
            msg = f"Failed to open {self._device_path}: {exc}"
            logger.error(msg)
            self._emit({"type": "error", "message": msg})
            return

        # Open optional mouse device
        if self._mouse_path:
            from .evdev_input import MouseDeviceReader
            self._mouse_reader = MouseDeviceReader(self._mouse_path)
            try:
                self._mouse_reader.open()
                logger.info("Mouse device opened: %s", self._mouse_path)
            except (PermissionError, OSError) as exc:
                logger.warning(
                    "Failed to open mouse %s: %s (continuing without mouse)",
                    self._mouse_path, exc,
                )
                self._mouse_reader = None

        logger.info(
            "Keyboard agent initialized: %s as %s (device: %s, mouse: %s, env: %s)",
            self._game_name, self._player_id, self._device_path,
            self._mouse_path or "none", env_name,
        )
        self._emit({
            "type": "agent_initialized",
            "run_id": self.config.run_id,
            "game_name": self._game_name,
            "player_id": self._player_id,
            "device_path": self._device_path,
            "mouse_path": self._mouse_path or "",
        })

    def _handle_select_action(self, cmd: Dict[str, Any]) -> None:
        """Handle select_action: read keyboard input, return action or NOOP.

        In tick mode (``tick_timeout > 0``), returns NOOP if no key is
        pressed within the timeout window. This allows multi-agent games
        to step continuously without agents blocking each other.

        In blocking mode (``tick_timeout <= 0``, default for single-agent),
        waits indefinitely until the human presses a recognized key.
        """
        if self._reader is None or self._resolver is None:
            self._emit({
                "type": "error",
                "message": "Keyboard not initialized. Send init_agent first.",
            })
            return

        player_id = cmd.get("player_id", self._player_id)
        noop_action = cmd.get("noop_action", 0)
        tick_timeout = cmd.get("tick_timeout", 0.0)  # 0 = blocking (single-agent)
        use_tick = tick_timeout > 0

        action = None
        mouse_dx, mouse_dy = 0, 0
        raw_keys: List[Dict[str, Any]] = []  # [{keycode, pressed}] for native forwarding
        deadline = time.monotonic() + tick_timeout if use_tick else 0

        while action is None:
            # Check stdin for stop/cancel (non-blocking)
            stdin_readable, _, _ = select.select([sys.stdin], [], [], 0.0)
            if stdin_readable:
                line = sys.stdin.readline().strip()
                if line:
                    try:
                        interrupt_cmd = json.loads(line)
                        if interrupt_cmd.get("cmd") in ("stop", "cancel_input"):
                            logger.info("Keyboard input cancelled by GUI")
                            if interrupt_cmd.get("cmd") == "stop":
                                self._emit({
                                    "type": "stopped",
                                    "run_id": self.config.run_id,
                                })
                                return
                            self._emit({
                                "type": "input_cancelled",
                                "run_id": self.config.run_id,
                                "player_id": player_id,
                            })
                            return
                    except json.JSONDecodeError:
                        pass

            # Build fd list: keyboard fd + optional mouse fd
            fds = []
            kbd_fd = getattr(self._reader, '_fd', None)
            mouse_fd = None
            if self._mouse_reader is not None and self._mouse_reader.is_open:
                mouse_fd = self._mouse_reader.fd

            if kbd_fd is not None:
                fds.append(kbd_fd)
            if mouse_fd is not None:
                fds.append(mouse_fd)

            if not fds:
                break

            # In tick mode, limit how long we wait
            if use_tick:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    action = noop_action  # Timeout: return NOOP
                    break
                wait = min(remaining, 0.05)
            else:
                wait = 0.1

            readable, _, _ = select.select(fds, [], [], wait)

            # Drain mouse deltas if mouse fd is ready (accumulate across loop)
            if mouse_fd is not None and mouse_fd in readable:
                mdx, mdy = self._mouse_reader.drain_deltas()
                mouse_dx += mdx
                mouse_dy += mdy

            # Read keyboard: drain ALL key events (press + release) for native
            # forwarding, and resolve press events to game actions.
            if kbd_fd is not None and kbd_fd in readable:
                events = self._reader.drain_key_events()
                for keycode, pressed in events:
                    raw_keys.append({"keycode": keycode, "pressed": pressed})
                    if pressed and action is None:
                        action = self._resolver.resolve({keycode})

        response: Dict[str, Any] = {
            "type": "action_selected",
            "run_id": self.config.run_id,
            "player_id": player_id,
            "action": action,
            "source": "keyboard" if action != noop_action else "noop",
            "device_path": self._device_path,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if mouse_dx != 0 or mouse_dy != 0:
            response["mouse_dx"] = mouse_dx
            response["mouse_dy"] = mouse_dy
        if raw_keys:
            response["raw_keys"] = raw_keys

        if action != noop_action:
            logger.info(
                "Keyboard action: %s -> action %d (mouse_dx=%d, mouse_dy=%d, raw_keys=%d)",
                player_id, action, mouse_dx, mouse_dy, len(raw_keys),
            )
        self._emit(response)

    def run(self) -> None:
        """Main loop: read commands from stdin, handle keyboard input."""
        logger.info(
            "Human Keyboard Runtime started (device: %s). "
            "Waiting for commands on stdin...",
            self._device_path,
        )

        self._emit({
            "type": "init",
            "run_id": self.config.run_id,
            "player_name": self.config.player_name,
            "version": "2.0",
            "mode": "keyboard",
            "device_path": self._device_path,
            "mouse_path": self._mouse_path or "",
        })

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
                    self._handle_init_agent(cmd)
                elif cmd_type == "select_action":
                    self._handle_select_action(cmd)
                elif cmd_type == "stop":
                    logger.info("Stop command received")
                    self._emit({"type": "stopped", "run_id": self.config.run_id})
                    break
                elif cmd_type == "ping":
                    self._emit({"type": "pong", "run_id": self.config.run_id})
                else:
                    self._emit({
                        "type": "error",
                        "message": f"Unknown command: {cmd_type}",
                    })
        except KeyboardInterrupt:
            logger.info("Interrupted")
        finally:
            if self._reader is not None:
                self._reader.close()
            if self._mouse_reader is not None:
                self._mouse_reader.close()
            logger.info("Human Keyboard Runtime stopped")


__all__ = [
    "HumanInteractiveRuntime",
    "HumanKeyboardRuntime",
    "HumanWorkerRuntime",
]
