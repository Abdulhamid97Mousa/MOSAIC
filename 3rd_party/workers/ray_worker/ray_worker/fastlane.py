"""FastLane telemetry for Ray RLlib multi-agent training.

This module provides live visualization of multi-agent training by:
1. Wrapping PettingZoo AEC environments
2. Collecting frames from all agents
3. Adding per-agent metrics overlay to each frame
4. Compositing into a tiled grid
5. Publishing to FastLane shared memory for UI display

The composite view shows each agent's perspective with their individual
metrics (reward, steps) overlaid, plus aggregate metrics at the bottom.

Example:
    ┌─────────────────────┐  ┌─────────────────────┐
    │ pursuer_0           │  │ pursuer_1           │
    │ [agent view]        │  │ [agent view]        │
    │ R: -12.3 | S: 245   │  │ R: 8.7 | S: 245     │
    └─────────────────────┘  └─────────────────────┘
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_LOGGER = logging.getLogger(__name__)

# Try to import PIL for text overlay
try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# Try to import FastLane components from gym_gui
try:
    from gym_gui.fastlane import FastLaneWriter, FastLaneConfig, FastLaneMetrics
    from gym_gui.fastlane.buffer import create_fastlane_name
    from gym_gui.fastlane.tiling import tile_frames
    _FASTLANE_AVAILABLE = True
except ImportError:
    FastLaneWriter = None
    FastLaneConfig = None
    FastLaneMetrics = None
    create_fastlane_name = None
    _FASTLANE_AVAILABLE = False

    def tile_frames(frames: List[np.ndarray]) -> np.ndarray:
        """Fallback tile_frames if gym_gui not available."""
        if not frames:
            raise ValueError("tile_frames requires at least one frame")
        if len(frames) == 1:
            return frames[0]

        n = len(frames)
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))

        h, w, c = frames[0].shape
        total = rows * cols

        # Pad with black frames if needed
        while len(frames) < total:
            frames.append(np.zeros_like(frames[0]))

        # Stack and reshape
        stacked = np.stack(frames, axis=0)
        reshaped = stacked.reshape(rows, cols, h, w, c)
        transposed = reshaped.swapaxes(1, 2)
        return transposed.reshape(rows * h, cols * w, c)


# Environment variable keys
ENV_FASTLANE_ENABLED = "RAY_FASTLANE_ENABLED"
ENV_FASTLANE_RUN_ID = "RAY_FASTLANE_RUN_ID"
ENV_FASTLANE_ENV_NAME = "RAY_FASTLANE_ENV_NAME"
ENV_FASTLANE_THROTTLE_MS = "RAY_FASTLANE_THROTTLE_MS"
ENV_FASTLANE_WORKER_INDEX = "RAY_FASTLANE_WORKER_INDEX"


def is_fastlane_enabled() -> bool:
    """Check if FastLane is enabled via environment variable."""
    val = os.getenv(ENV_FASTLANE_ENABLED, "").lower()
    return val in {"1", "true", "yes", "on"}


@dataclass
class AgentMetrics:
    """Per-agent metrics for display overlay."""
    agent_id: str
    reward: float = 0.0
    cumulative_reward: float = 0.0
    steps: int = 0
    done: bool = False


@dataclass
class MultiAgentMetrics:
    """Aggregate metrics for all agents."""
    agents: Dict[str, AgentMetrics] = field(default_factory=dict)
    episode: int = 0
    total_timesteps: int = 0
    combined_reward: float = 0.0
    step_rate_hz: float = 0.0


@dataclass
class FastLaneRayConfig:
    """Configuration for Ray FastLane streaming.

    Each Ray rollout worker gets its own FastLane stream, identified by
    worker_index. This allows the UI to tile multiple workers' views together.

    Stream naming: {run_id}-w{worker_index}
    """
    enabled: bool
    run_id: str
    env_name: str
    worker_index: int = 0  # Ray worker index (0=local, 1+=remote workers)
    throttle_interval_ms: int = 33  # ~30 FPS default

    @property
    def stream_id(self) -> str:
        """Get the unique stream ID for this worker.

        Format: {run_id}-w{worker_index}

        All workers use the same naming pattern for consistency.
        Worker naming: w0, w1, w2, ... (worker-0 is reserved for coordination)
        """
        return f"{self.run_id}-w{self.worker_index}"

    @classmethod
    def from_env(cls, worker_index: Optional[int] = None) -> "FastLaneRayConfig":
        """Load config from environment variables.

        Args:
            worker_index: Ray worker index. If None, reads from env var.
        """
        enabled = is_fastlane_enabled()
        run_id = os.getenv(ENV_FASTLANE_RUN_ID, "ray-run")
        env_name = os.getenv(ENV_FASTLANE_ENV_NAME, "MultiAgent")

        # Default to 33ms (~30 FPS) to avoid excessive frame publishing
        try:
            throttle_ms = int(os.getenv(ENV_FASTLANE_THROTTLE_MS, "33"))
        except ValueError:
            throttle_ms = 33

        # Get worker index from param or env var
        if worker_index is None:
            try:
                worker_index = int(os.getenv(ENV_FASTLANE_WORKER_INDEX, "0"))
            except ValueError:
                worker_index = 0

        return cls(
            enabled=enabled,
            run_id=run_id,
            env_name=env_name,
            worker_index=worker_index,
            throttle_interval_ms=throttle_ms,
        )


class TextOverlay:
    """Renders text overlays on frames using PIL."""

    # Colors (RGB)
    COLOR_WHITE = (255, 255, 255)
    COLOR_BLACK = (0, 0, 0)
    COLOR_GREEN = (0, 255, 0)
    COLOR_RED = (255, 0, 0)
    COLOR_YELLOW = (255, 255, 0)

    # Font settings
    FONT_SIZE_SMALL = 12
    FONT_SIZE_MEDIUM = 14
    FONT_SIZE_LARGE = 16

    def __init__(self):
        self._font_small: Optional[ImageFont.FreeTypeFont] = None
        self._font_medium: Optional[ImageFont.FreeTypeFont] = None
        self._font_large: Optional[ImageFont.FreeTypeFont] = None
        self._init_fonts()

    def _init_fonts(self) -> None:
        """Initialize fonts (falls back to default if custom not available)."""
        if not _PIL_AVAILABLE:
            return

        try:
            # Try to load a monospace font
            self._font_small = ImageFont.truetype("DejaVuSansMono.ttf", self.FONT_SIZE_SMALL)
            self._font_medium = ImageFont.truetype("DejaVuSansMono.ttf", self.FONT_SIZE_MEDIUM)
            self._font_large = ImageFont.truetype("DejaVuSansMono.ttf", self.FONT_SIZE_LARGE)
        except (OSError, IOError):
            # Fall back to default font
            try:
                self._font_small = ImageFont.load_default()
                self._font_medium = ImageFont.load_default()
                self._font_large = ImageFont.load_default()
            except Exception:
                pass

    def add_agent_overlay(
        self,
        frame: np.ndarray,
        agent_id: str,
        metrics: AgentMetrics,
    ) -> np.ndarray:
        """Add agent name and metrics overlay to a frame.

        Args:
            frame: RGB numpy array (H, W, C)
            agent_id: Agent identifier
            metrics: Agent's current metrics

        Returns:
            Frame with text overlay
        """
        if not _PIL_AVAILABLE or frame is None:
            return frame

        # Convert to PIL Image
        img = Image.fromarray(frame.astype(np.uint8))
        draw = ImageDraw.Draw(img)

        h, w = frame.shape[:2]
        padding = 4

        # Agent name (top-left, with background)
        agent_text = agent_id
        font = self._font_medium or ImageFont.load_default()

        # Draw background rectangle for agent name
        bbox = draw.textbbox((0, 0), agent_text, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle(
            [padding - 2, padding - 2, padding + text_w + 4, padding + text_h + 4],
            fill=(0, 0, 0, 180)
        )
        draw.text((padding, padding), agent_text, fill=self.COLOR_WHITE, font=font)

        # Metrics (bottom, with background)
        reward_color = self.COLOR_GREEN if metrics.reward >= 0 else self.COLOR_RED
        metrics_text = f"R: {metrics.cumulative_reward:+.1f} | S: {metrics.steps}"

        font_small = self._font_small or ImageFont.load_default()
        bbox = draw.textbbox((0, 0), metrics_text, font=font_small)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        y_pos = h - text_h - padding - 4
        draw.rectangle(
            [padding - 2, y_pos - 2, padding + text_w + 4, y_pos + text_h + 4],
            fill=(0, 0, 0, 180)
        )
        draw.text((padding, y_pos), metrics_text, fill=self.COLOR_YELLOW, font=font_small)

        return np.array(img)


class MultiAgentFastLaneWrapper:
    """Wraps PettingZoo AEC environment for FastLane streaming.

    This wrapper:
    1. Intercepts step() calls to collect per-agent frames
    2. Tracks per-agent metrics (reward, steps)
    3. Composites all agent views into a tiled grid
    4. Publishes composite frame to FastLane shared memory
    5. For board games (chess), includes structured game state metadata

    For AEC (turn-based) environments, frames are collected as each agent acts
    and the composite is published after all agents have taken their turn.
    """

    # Metadata size for JSON board game state (FEN + legal_moves)
    METADATA_SIZE = 4096  # 4KB should be plenty for chess state

    def __init__(self, env: Any, config: Optional[FastLaneRayConfig] = None):
        """Initialize the wrapper.

        Args:
            env: PettingZoo AEC environment to wrap
            config: FastLane configuration (loads from env if None)
        """
        self._env = env
        self._config = config or FastLaneRayConfig.from_env()

        # Agent tracking
        self._agent_ids: List[str] = []
        self._agent_frames: Dict[str, np.ndarray] = {}
        self._agent_metrics: Dict[str, AgentMetrics] = {}
        self._agents_acted_this_round: set = set()

        # Aggregate metrics
        self._episode = 0
        self._total_timesteps = 0
        self._episode_rewards: Dict[str, float] = {}

        # FastLane writer
        self._writer: Optional[FastLaneWriter] = None
        self._overlay = TextOverlay()

        # Throttling
        self._last_emit_ns = 0
        self._throttle_ns = self._config.throttle_interval_ms * 1_000_000

        # Metrics timing
        self._last_metrics_ts: Optional[float] = None

        # Debug
        self._debug_counter = 0

        # Board game detection (chess, go, etc.)
        self._is_chess = self._detect_chess_env()
        self._chess_board: Any = None
        if self._is_chess:
            self._log_debug("Chess environment detected - will send board state metadata")

        self._log_debug(f"MultiAgentFastLane initialized: stream={self._config.stream_id}, worker={self._config.worker_index}")

    def _detect_chess_env(self) -> bool:
        """Check if the wrapped environment is a chess environment."""
        try:
            unwrapped = self._env.unwrapped if hasattr(self._env, 'unwrapped') else self._env
            if hasattr(unwrapped, 'board'):
                # Check if it's a chess.Board
                board = unwrapped.board
                if hasattr(board, 'fen') and hasattr(board, 'legal_moves'):
                    return True
        except Exception:
            pass
        return False

    def _get_chess_board(self) -> Any:
        """Get the underlying chess.Board object."""
        if self._chess_board is not None:
            return self._chess_board
        try:
            unwrapped = self._env.unwrapped if hasattr(self._env, 'unwrapped') else self._env
            if hasattr(unwrapped, 'board'):
                self._chess_board = unwrapped.board
                return self._chess_board
        except Exception:
            pass
        return None

    def _get_board_game_metadata(self) -> Optional[str]:
        """Get board game state as JSON string for metadata.

        Returns JSON with:
        - game_type: "chess"
        - fen: FEN string
        - legal_moves: list of UCI move strings
        - current_player: "white" or "black"
        - is_check: bool
        - is_game_over: bool
        """
        if not self._is_chess:
            return None

        try:
            import json
            board = self._get_chess_board()
            if board is None:
                return None

            # Get current player
            is_white_turn = board.turn  # True = white, False = black
            current_player = "white" if is_white_turn else "black"
            current_agent = "player_0" if is_white_turn else "player_1"

            # Get legal moves in UCI notation
            legal_moves = [str(move) for move in board.legal_moves]

            metadata = {
                "game_type": "chess",
                "fen": board.fen(),
                "legal_moves": legal_moves,
                "current_player": current_player,
                "current_agent": current_agent,
                "is_check": board.is_check(),
                "is_checkmate": board.is_checkmate(),
                "is_stalemate": board.is_stalemate(),
                "is_game_over": board.is_game_over(),
            }
            return json.dumps(metadata)
        except Exception as e:
            self._log_debug(f"Failed to get chess metadata: {e}", limit=5)
            return None

    # ------------------------------------------------------------------
    # Environment API passthrough
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to wrapped environment."""
        return getattr(self._env, name)

    def reset(self, *args, **kwargs) -> Any:
        """Reset environment and clear agent tracking."""
        result = self._env.reset(*args, **kwargs)

        # Initialize agent tracking
        if hasattr(self._env, 'possible_agents'):
            self._agent_ids = list(self._env.possible_agents)
        elif hasattr(self._env, 'agents'):
            self._agent_ids = list(self._env.agents)

        # Reset metrics
        self._agent_frames.clear()
        self._agents_acted_this_round.clear()
        self._agent_metrics = {
            aid: AgentMetrics(agent_id=aid) for aid in self._agent_ids
        }
        self._episode_rewards = {aid: 0.0 for aid in self._agent_ids}
        self._episode += 1

        self._log_debug(f"Reset: agents={self._agent_ids}, episode={self._episode}")

        return result

    def step(self, action: Any) -> Any:
        """Execute step and capture frame for current agent."""
        # Get current agent before step
        current_agent = None
        if hasattr(self._env, 'agent_selection'):
            current_agent = self._env.agent_selection

        # Execute the actual step
        self._env.step(action)

        # Update timestep count
        self._total_timesteps += 1

        if current_agent and self._config.enabled:
            self._capture_agent_frame(current_agent)
            self._update_agent_metrics(current_agent)
            self._agents_acted_this_round.add(current_agent)

            # Check if all agents have acted
            if self._all_agents_acted():
                self._publish_composite_frame()
                self._agents_acted_this_round.clear()

    def observe(self, agent: str) -> Any:
        """Get observation for specified agent."""
        return self._env.observe(agent)

    def render(self) -> Any:
        """Render the environment."""
        return self._env.render()

    def close(self) -> None:
        """Close environment and cleanup FastLane resources."""
        self._close_writer()
        if hasattr(self._env, 'close'):
            self._env.close()

    # ------------------------------------------------------------------
    # FastLane frame capture and publishing
    # ------------------------------------------------------------------

    def _capture_agent_frame(self, agent_id: str) -> None:
        """Capture rendered frame for an agent."""
        try:
            frame = self._env.render()
            if frame is not None:
                frame = self._ensure_numpy(frame)
                if frame is not None:
                    self._agent_frames[agent_id] = frame
        except Exception as e:
            self._log_debug(f"Frame capture failed for {agent_id}: {e}", limit=5)

    def _update_agent_metrics(self, agent_id: str) -> None:
        """Update metrics for an agent after their step."""
        metrics = self._agent_metrics.get(agent_id)
        if metrics is None:
            metrics = AgentMetrics(agent_id=agent_id)
            self._agent_metrics[agent_id] = metrics

        # Get reward from environment
        reward = 0.0
        if hasattr(self._env, 'rewards') and agent_id in self._env.rewards:
            reward = self._env.rewards[agent_id]
        elif hasattr(self._env, '_cumulative_rewards') and agent_id in self._env._cumulative_rewards:
            reward = self._env._cumulative_rewards[agent_id]

        # Update metrics
        metrics.reward = reward
        self._episode_rewards[agent_id] = self._episode_rewards.get(agent_id, 0.0) + reward
        metrics.cumulative_reward = self._episode_rewards[agent_id]
        metrics.steps += 1

        # Check if done
        if hasattr(self._env, 'terminations') and agent_id in self._env.terminations:
            metrics.done = self._env.terminations[agent_id]
        elif hasattr(self._env, 'truncations') and agent_id in self._env.truncations:
            metrics.done = metrics.done or self._env.truncations[agent_id]

    def _all_agents_acted(self) -> bool:
        """Check if all agents have acted this round."""
        # For AEC, check if we've cycled through all agents
        active_agents = set()
        if hasattr(self._env, 'agents'):
            active_agents = set(self._env.agents)
        else:
            active_agents = set(self._agent_ids)

        # Consider round complete when all active agents have acted
        return self._agents_acted_this_round >= active_agents

    def _publish_composite_frame(self) -> None:
        """Create composite frame from all agent frames and publish."""
        if not self._config.enabled or not _FASTLANE_AVAILABLE:
            return

        if not self._agent_frames:
            return

        # Check throttling
        if self._should_throttle():
            return

        # Build ordered list of frames with overlays
        frames_with_overlay = []
        for agent_id in self._agent_ids:
            frame = self._agent_frames.get(agent_id)
            if frame is None:
                # Create placeholder for missing agent
                if frames_with_overlay:
                    frame = np.zeros_like(frames_with_overlay[0])
                else:
                    continue

            # Add per-agent metrics overlay
            metrics = self._agent_metrics.get(agent_id, AgentMetrics(agent_id=agent_id))
            frame_with_overlay = self._overlay.add_agent_overlay(frame, agent_id, metrics)
            frames_with_overlay.append(frame_with_overlay)

        if not frames_with_overlay:
            return

        # Tile frames into composite
        try:
            composite = tile_frames(frames_with_overlay)
        except Exception as e:
            self._log_debug(f"Tiling failed: {e}", limit=5)
            return

        # Publish to FastLane
        self._publish_frame(composite)

    def _publish_frame(self, frame: np.ndarray) -> None:
        """Publish composite frame to FastLane shared memory."""
        if FastLaneWriter is None or FastLaneConfig is None or FastLaneMetrics is None:
            return

        height, width, channels = frame.shape
        frame_bytes = frame.astype(np.uint8).tobytes()

        # Create writer if needed (with metadata support for board games)
        if self._writer is None:
            metadata_size = self.METADATA_SIZE if self._is_chess else 0
            config = FastLaneConfig(
                width=width,
                height=height,
                channels=channels,
                pixel_format="RGB" if channels == 3 else "RGBA",
                metadata_size=metadata_size,
            )
            self._writer = self._create_writer(config)
            if self._writer is None:
                return

        # Calculate step rate
        now = perf_counter()
        if self._last_metrics_ts is None:
            step_rate = 0.0
        else:
            delta = now - self._last_metrics_ts
            step_rate = 1.0 / delta if delta > 0 else 0.0
        self._last_metrics_ts = now

        # Calculate combined reward
        combined_reward = sum(self._episode_rewards.values())

        # Create metrics
        metrics = FastLaneMetrics(
            last_reward=combined_reward,
            rolling_return=combined_reward,
            step_rate_hz=step_rate,
        )

        # Get board game metadata if applicable
        metadata = self._get_board_game_metadata()

        try:
            self._writer.publish(frame_bytes, metrics=metrics, metadata=metadata)
            self._log_debug("Frame published", limit=20)
        except Exception as e:
            self._log_debug(f"Publish failed: {e}", limit=5)
            self._close_writer()

    def _create_writer(self, config: Any) -> Optional[FastLaneWriter]:
        """Create FastLane writer for shared memory."""
        try:
            return FastLaneWriter.create(self._config.stream_id, config)
        except FileExistsError:
            # Try to connect to existing shared memory
            if create_fastlane_name is None:
                return None
            try:
                name = create_fastlane_name(self._config.stream_id)
                shm = shared_memory.SharedMemory(name=name, create=False)
                return FastLaneWriter(shm, config)
            except Exception:
                return None
        except Exception as e:
            self._log_debug(f"Writer creation failed: {e}", limit=3)
            return None

    def _close_writer(self) -> None:
        """Close FastLane writer and unlink shared memory."""
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
            
            try:
                self._writer.unlink()
                _LOGGER.debug("FastLane writer closed and unlinked for stream_id=%s", self._config.stream_id)
            except FileNotFoundError:
                pass  # Already unlinked
            except Exception as exc:
                _LOGGER.debug("Error closing FastLane writer: %s", exc)
            finally:
                self._writer = None

    def _should_throttle(self) -> bool:
        """Check if we should skip this frame due to throttling."""
        if self._throttle_ns <= 0:
            return False

        now_ns = time.time_ns()
        if self._last_emit_ns and (now_ns - self._last_emit_ns) < self._throttle_ns:
            return True

        self._last_emit_ns = now_ns
        return False

    def _ensure_numpy(self, frame: Any) -> Optional[np.ndarray]:
        """Convert frame to numpy array if needed."""
        if isinstance(frame, np.ndarray):
            arr = frame
        elif hasattr(frame, '__array__'):
            arr = np.asarray(frame)
        else:
            return None

        # Ensure 3D (H, W, C)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)
        if arr.ndim != 3:
            return None

        return np.ascontiguousarray(arr.astype(np.uint8, copy=False))

    def _log_debug(self, message: str, *, limit: Optional[int] = None) -> None:
        """Log debug message using standard logging."""
        if limit is not None and self._debug_counter >= limit:
            return
        self._debug_counter += 1
        _LOGGER.debug("[FastLane-AEC W%d] %s", self._config.worker_index, message)


class ParallelFastLaneWrapper:
    """Wraps PettingZoo Parallel environment for FastLane streaming.

    This wrapper is designed for Parallel API environments where all agents
    act simultaneously. Unlike AEC environments, there's no agent_selection -
    all agents provide actions in a single step() call.

    Frame capture strategy:
    - After each step(), capture a single frame representing all agents
    - Create composite view showing all agent perspectives
    - Publish composite frame immediately (no waiting for rounds)

    This is more efficient than AEC as we don't need to track which agents
    have acted - every step() is a complete timestep for all agents.

    NOTE: This wrapper exposes all PettingZoo ParallelEnv attributes directly
    to ensure Ray's ParallelPettingZooEnv can properly interact with it.
    """

    def __init__(self, env: Any, config: Optional[FastLaneRayConfig] = None):
        """Initialize the wrapper.

        Args:
            env: PettingZoo Parallel environment to wrap
            config: FastLane configuration (loads from env if None)
        """
        self._env = env
        self._config = config or FastLaneRayConfig.from_env()

        # Agent tracking
        self._agent_ids: List[str] = []
        self._agent_metrics: Dict[str, AgentMetrics] = {}

        # Aggregate metrics
        self._episode = 0
        self._total_timesteps = 0
        self._episode_rewards: Dict[str, float] = {}

        # FastLane writer
        self._writer: Optional[FastLaneWriter] = None
        self._overlay = TextOverlay()

        # Throttling
        self._last_emit_ns = 0
        self._throttle_ns = self._config.throttle_interval_ms * 1_000_000

        # Metrics timing
        self._last_metrics_ts: Optional[float] = None

        # Debug
        self._debug_counter = 0

        # Log initialization at INFO level for visibility
        _LOGGER.info(
            "[FastLane-Parallel W%d] Initialized: enabled=%s, stream=%s, fastlane_available=%s",
            self._config.worker_index,
            self._config.enabled,
            self._config.stream_id,
            _FASTLANE_AVAILABLE,
        )

    # ------------------------------------------------------------------
    # Environment API passthrough - explicit properties for Ray compatibility
    # ------------------------------------------------------------------

    @property
    def unwrapped(self) -> Any:
        """Return the unwrapped environment."""
        return getattr(self._env, 'unwrapped', self._env)

    @property
    def possible_agents(self) -> List[str]:
        """List of all possible agents."""
        return self._env.possible_agents

    @property
    def agents(self) -> List[str]:
        """List of currently active agents."""
        return self._env.agents

    @property
    def observation_spaces(self) -> Dict[str, Any]:
        """Observation spaces for all agents."""
        return self._env.observation_spaces

    @property
    def action_spaces(self) -> Dict[str, Any]:
        """Action spaces for all agents."""
        return self._env.action_spaces

    @property
    def max_num_agents(self) -> int:
        """Maximum number of agents."""
        return getattr(self._env, 'max_num_agents', len(self._env.possible_agents))

    @property
    def num_agents(self) -> int:
        """Current number of agents."""
        return getattr(self._env, 'num_agents', len(self._env.agents))

    def observation_space(self, agent: str) -> Any:
        """Get observation space for an agent."""
        return self._env.observation_space(agent)

    def action_space(self, agent: str) -> Any:
        """Get action space for an agent."""
        return self._env.action_space(agent)

    def state(self) -> Any:
        """Get global state if available."""
        if hasattr(self._env, 'state'):
            return self._env.state()
        return None

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to wrapped environment."""
        return getattr(self._env, name)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment and clear agent tracking.

        Args:
            seed: Optional seed for reproducibility
            options: Optional options dict

        Returns:
            Tuple of (observations_dict, infos_dict)
        """
        result = self._env.reset(seed=seed, options=options)

        # Initialize agent tracking from possible_agents
        if hasattr(self._env, 'possible_agents'):
            self._agent_ids = list(self._env.possible_agents)
        elif hasattr(self._env, 'agents'):
            self._agent_ids = list(self._env.agents)

        # Reset metrics
        self._agent_metrics = {
            aid: AgentMetrics(agent_id=aid) for aid in self._agent_ids
        }
        self._episode_rewards = {aid: 0.0 for aid in self._agent_ids}
        self._episode += 1

        _LOGGER.info(
            "[FastLane-Parallel W%d] Reset: episode=%d, agents=%d, timesteps=%d",
            self._config.worker_index,
            self._episode,
            len(self._agent_ids),
            self._total_timesteps,
        )

        # Capture initial frame (note: timesteps=0 at this point)
        if self._config.enabled:
            self._capture_and_publish_frame()

        return result

    def step(
        self, actions: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """Execute step for all agents simultaneously.

        Args:
            actions: Dict mapping agent_id to action

        Returns:
            Tuple of (observations, rewards, terminations, truncations, infos)
        """
        # Execute the actual step - all agents act simultaneously
        observations, rewards, terminations, truncations, infos = self._env.step(actions)

        # Update timestep count
        self._total_timesteps += 1

        # Log every 100 steps to confirm wrapper step() is being called
        if self._total_timesteps % 100 == 0:
            _LOGGER.debug(
                "FastLane wrapper step %d (worker=%d, episode=%d)",
                self._total_timesteps,
                self._config.worker_index,
                self._episode,
            )

        if self._config.enabled:
            # Update metrics for all agents
            for agent_id in self._agent_ids:
                if agent_id in rewards:
                    self._update_agent_metrics(
                        agent_id,
                        reward=rewards.get(agent_id, 0.0),
                        terminated=terminations.get(agent_id, False),
                        truncated=truncations.get(agent_id, False),
                    )

            # Capture and publish frame immediately - all agents acted
            self._capture_and_publish_frame()

        return observations, rewards, terminations, truncations, infos

    def render(self) -> Any:
        """Render the environment."""
        return self._env.render()

    def close(self) -> None:
        """Close environment and cleanup FastLane resources."""
        self._close_writer()
        if hasattr(self._env, 'close'):
            self._env.close()

    # ------------------------------------------------------------------
    # FastLane frame capture and publishing
    # ------------------------------------------------------------------

    def _update_agent_metrics(
        self,
        agent_id: str,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Update metrics for an agent after step."""
        metrics = self._agent_metrics.get(agent_id)
        if metrics is None:
            metrics = AgentMetrics(agent_id=agent_id)
            self._agent_metrics[agent_id] = metrics

        # Update metrics
        metrics.reward = reward
        self._episode_rewards[agent_id] = self._episode_rewards.get(agent_id, 0.0) + reward
        metrics.cumulative_reward = self._episode_rewards[agent_id]
        metrics.steps += 1
        metrics.done = terminated or truncated

    def _capture_and_publish_frame(self) -> None:
        """Capture frame and publish to FastLane.

        Each Ray worker has its own FastLane stream, so we publish a single
        frame showing this worker's environment state with aggregate metrics.
        The UI tiles multiple worker streams together for the composite view.
        """
        if not self._config.enabled:
            if self._total_timesteps <= 5:
                _LOGGER.debug("[FastLane W%d] Skipping frame: config.enabled=False", self._config.worker_index)
            return

        if not _FASTLANE_AVAILABLE:
            if self._total_timesteps <= 5:
                _LOGGER.warning("[FastLane W%d] Skipping frame: FastLane not available", self._config.worker_index)
            return

        # Check throttling FIRST - skip all work if throttled
        if self._should_throttle():
            # Log occasionally when throttling
            if self._total_timesteps % 500 == 0:
                _LOGGER.debug(
                    "[FastLane W%d] Frame throttled at step %d",
                    self._config.worker_index,
                    self._total_timesteps,
                )
            return

        # Render the environment to get frame
        try:
            frame = self._env.render()
            if frame is None:
                return
            frame = self._ensure_numpy(frame)
            if frame is None:
                return
        except Exception as e:
            self._log_debug(f"Render failed: {e}", limit=5)
            return

        # Add aggregate metrics overlay showing worker info and combined stats
        frame_with_overlay = self._add_aggregate_overlay(frame)

        # Publish to FastLane
        self._publish_frame(frame_with_overlay)

    def _add_aggregate_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add aggregate metrics overlay to frame.

        Shows worker index, episode, agent count, reward, and timesteps.
        Each worker has its own FastLane stream - UI tiles them together.
        """
        if not _PIL_AVAILABLE or frame is None:
            return frame

        # Convert to PIL Image
        img = Image.fromarray(frame.astype(np.uint8))
        draw = ImageDraw.Draw(img)

        h, w = frame.shape[:2]
        padding = 4

        # Calculate aggregate stats
        total_reward = sum(self._episode_rewards.values())
        active_agents = len([m for m in self._agent_metrics.values() if not m.done])
        total_agents = len(self._agent_ids)

        # Create summary text with worker index
        summary_text = f"W{self._config.worker_index} | Ep: {self._episode} | Agents: {active_agents}/{total_agents} | R: {total_reward:+.1f} | T: {self._total_timesteps}"

        font = self._overlay._font_small or ImageFont.load_default()
        bbox = draw.textbbox((0, 0), summary_text, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # Position at bottom center
        x_pos = (w - text_w) // 2
        y_pos = h - text_h - padding - 4

        # Draw background rectangle
        draw.rectangle(
            [x_pos - 4, y_pos - 2, x_pos + text_w + 4, y_pos + text_h + 4],
            fill=(0, 0, 0, 200)
        )
        draw.text((x_pos, y_pos), summary_text, fill=(255, 255, 0), font=font)

        return np.array(img)

    def _publish_frame(self, frame: np.ndarray) -> None:
        """Publish frame to FastLane shared memory."""
        if FastLaneWriter is None or FastLaneConfig is None or FastLaneMetrics is None:
            return

        height, width, channels = frame.shape
        frame_bytes = frame.astype(np.uint8).tobytes()

        # Create writer if needed
        if self._writer is None:
            config = FastLaneConfig(
                width=width,
                height=height,
                channels=channels,
                pixel_format="RGB" if channels == 3 else "RGBA",
            )
            self._writer = self._create_writer(config)
            if self._writer is None:
                return

        # Calculate step rate
        now = perf_counter()
        if self._last_metrics_ts is None:
            step_rate = 0.0
        else:
            delta = now - self._last_metrics_ts
            step_rate = 1.0 / delta if delta > 0 else 0.0
        self._last_metrics_ts = now

        # Calculate combined reward
        combined_reward = sum(self._episode_rewards.values())

        # Create metrics
        metrics = FastLaneMetrics(
            last_reward=combined_reward,
            rolling_return=combined_reward,
            step_rate_hz=step_rate,
        )

        try:
            self._writer.publish(frame_bytes, metrics=metrics)
            # Log frame publish with timestep info (first 10 and then every 100)
            if self._total_timesteps <= 10 or self._total_timesteps % 100 == 0:
                _LOGGER.info(
                    "[FastLane W%d] Frame published: T=%d, Ep=%d, rate=%.1f Hz",
                    self._config.worker_index,
                    self._total_timesteps,
                    self._episode,
                    step_rate,
                )
        except Exception as e:
            _LOGGER.warning("[FastLane W%d] Publish failed: %s", self._config.worker_index, e)
            self._close_writer()

    def _create_writer(self, config: Any) -> Optional[FastLaneWriter]:
        """Create FastLane writer for shared memory."""
        try:
            writer = FastLaneWriter.create(self._config.stream_id, config)
            _LOGGER.info(
                "[FastLane W%d] Writer created successfully for stream: %s",
                self._config.worker_index,
                self._config.stream_id,
            )
            return writer
        except FileExistsError:
            # Try to connect to existing shared memory
            if create_fastlane_name is None:
                _LOGGER.warning("[FastLane W%d] SharedMemory exists but create_fastlane_name not available", self._config.worker_index)
                return None
            try:
                name = create_fastlane_name(self._config.stream_id)
                shm = shared_memory.SharedMemory(name=name, create=False)
                writer = FastLaneWriter(shm, config)
                _LOGGER.info(
                    "[FastLane W%d] Connected to existing shared memory: %s",
                    self._config.worker_index,
                    name,
                )
                return writer
            except Exception as e:
                _LOGGER.warning("[FastLane W%d] Failed to connect to existing shm: %s", self._config.worker_index, e)
                return None
        except Exception as e:
            _LOGGER.warning("[FastLane W%d] Writer creation failed: %s", self._config.worker_index, e)
            return None

    def _close_writer(self) -> None:
        """Close FastLane writer and unlink shared memory."""
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
            
            try:
                self._writer.unlink()
                _LOGGER.debug("FastLane writer closed and unlinked for stream_id=%s", self._config.stream_id)
            except FileNotFoundError:
                pass  # Already unlinked
            except Exception as exc:
                _LOGGER.debug("Error closing FastLane writer: %s", exc)
            finally:
                self._writer = None

    def _should_throttle(self) -> bool:
        """Check if we should skip this frame due to throttling."""
        if self._throttle_ns <= 0:
            return False

        now_ns = time.time_ns()
        if self._last_emit_ns and (now_ns - self._last_emit_ns) < self._throttle_ns:
            return True

        self._last_emit_ns = now_ns
        return False

    def _ensure_numpy(self, frame: Any) -> Optional[np.ndarray]:
        """Convert frame to numpy array if needed."""
        if isinstance(frame, np.ndarray):
            arr = frame
        elif hasattr(frame, '__array__'):
            arr = np.asarray(frame)
        else:
            return None

        # Ensure 3D (H, W, C)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)
        if arr.ndim != 3:
            return None

        return np.ascontiguousarray(arr.astype(np.uint8, copy=False))

    def _log_debug(self, message: str, *, limit: Optional[int] = None) -> None:
        """Log debug message using standard logging."""
        if limit is not None and self._debug_counter >= limit:
            return
        self._debug_counter += 1
        _LOGGER.debug("[FastLane-Parallel W%d] %s", self._config.worker_index, message)


def maybe_wrap_env(env: Any, worker_index: int = 0) -> Any:
    """Wrap AEC environment with FastLane if enabled.

    Args:
        env: PettingZoo AEC environment
        worker_index: Ray worker index for unique stream identification

    Returns:
        Wrapped environment if FastLane enabled, otherwise original
    """
    if not is_fastlane_enabled():
        return env

    config = FastLaneRayConfig.from_env(worker_index=worker_index)
    return MultiAgentFastLaneWrapper(env, config)


def maybe_wrap_parallel_env(env: Any, worker_index: int = 0) -> Any:
    """Wrap Parallel environment with FastLane if enabled.

    Args:
        env: PettingZoo Parallel environment
        worker_index: Ray worker index for unique stream identification

    Returns:
        Wrapped environment if FastLane enabled, otherwise original
    """
    enabled = is_fastlane_enabled()
    _LOGGER.info(
        "[FastLane] maybe_wrap_parallel_env called: worker_index=%d, enabled=%s",
        worker_index,
        enabled,
    )

    if not enabled:
        return env

    config = FastLaneRayConfig.from_env(worker_index=worker_index)
    _LOGGER.info(
        "[FastLane] Creating ParallelFastLaneWrapper: stream_id=%s",
        config.stream_id,
    )
    return ParallelFastLaneWrapper(env, config)


def create_fastlane_config(
    run_id: str,
    env_name: str,
    *,
    throttle_ms: int = 0,
) -> FastLaneRayConfig:
    """Create FastLane configuration programmatically.

    Args:
        run_id: Unique run identifier
        env_name: Environment name for tab title
        throttle_ms: Minimum ms between frame publishes

    Returns:
        FastLane configuration
    """
    return FastLaneRayConfig(
        enabled=True,
        run_id=run_id,
        env_name=env_name,
        throttle_interval_ms=throttle_ms,
    )


def set_fastlane_env_vars(
    run_id: str,
    env_name: str,
    *,
    enabled: bool = True,
    throttle_ms: int = 0,
) -> None:
    """Set environment variables for FastLane configuration.

    Call this before creating the environment to enable FastLane.

    Args:
        run_id: Unique run identifier
        env_name: Environment name for tab title
        enabled: Whether FastLane is enabled
        throttle_ms: Minimum ms between frame publishes
    """
    os.environ[ENV_FASTLANE_ENABLED] = "1" if enabled else "0"
    os.environ[ENV_FASTLANE_RUN_ID] = run_id
    os.environ[ENV_FASTLANE_ENV_NAME] = env_name
    os.environ[ENV_FASTLANE_THROTTLE_MS] = str(throttle_ms)


__all__ = [
    "is_fastlane_enabled",
    "maybe_wrap_env",
    "maybe_wrap_parallel_env",
    "create_fastlane_config",
    "set_fastlane_env_vars",
    "MultiAgentFastLaneWrapper",
    "ParallelFastLaneWrapper",
    "FastLaneRayConfig",
    "AgentMetrics",
    "MultiAgentMetrics",
    "TextOverlay",
]
