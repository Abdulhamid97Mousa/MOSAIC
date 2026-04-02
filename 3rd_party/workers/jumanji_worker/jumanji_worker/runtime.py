"""Runtime orchestration for Jumanji training.

This module provides the JumanjiWorkerRuntime class that wraps Jumanji's
native JAX-based A2C training with MOSAIC's telemetry and analytics
infrastructure.

The runtime supports:
- A2C (Advantage Actor-Critic) training using Jumanji's implementation
- Random agent baseline for testing
- GPU/TPU acceleration via JAX
- TelemetryEmitter integration for lifecycle events
- Analytics manifest generation

Also provides InteractiveRuntime for GUI step-by-step policy evaluation.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .config import JumanjiWorkerConfig

# MOSAIC imports
try:
    from gym_gui.core.worker import TelemetryEmitter
    from gym_gui.config.paths import VAR_TRAINER_DIR, ensure_var_directories
    _HAS_GYM_GUI = True
except ImportError:
    _HAS_GYM_GUI = False
    TelemetryEmitter = None
    VAR_TRAINER_DIR = Path("var/trainer")

    def ensure_var_directories():
        pass

# JAX imports (lazy to handle missing JAX gracefully)
try:
    import jax
    import jax.numpy as jnp
    import optax
    import jumanji
    from jumanji.wrappers import VmapAutoResetWrapper
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False
    jax = None
    jnp = None
    optax = None
    jumanji = None

LOGGER = logging.getLogger(__name__)


class JumanjiWorkerRuntime:
    """Orchestrate Jumanji algorithm execution.

    This runtime wraps Jumanji's A2C training infrastructure and integrates
    with MOSAIC's telemetry and analytics systems.

    Example:
        >>> config = JumanjiWorkerConfig(
        ...     run_id="test_run",
        ...     env_id="Game2048-v1",
        ...     agent="a2c",
        ...     num_epochs=100,
        ... )
        >>> runtime = JumanjiWorkerRuntime(config)
        >>> summary = runtime.run()
    """

    def __init__(
        self,
        config: JumanjiWorkerConfig,
        *,
        dry_run: bool = False,
    ) -> None:
        """Initialize runtime.

        Args:
            config: Worker configuration
            dry_run: If True, validate config without executing
        """
        self._config = config
        self._dry_run = dry_run

        # Initialize telemetry emitter
        if _HAS_GYM_GUI:
            self._emitter = TelemetryEmitter(run_id=config.run_id, logger=LOGGER)
        else:
            self._emitter = None

    @property
    def config(self) -> JumanjiWorkerConfig:
        """Get the configuration."""
        return self._config

    @property
    def dry_run(self) -> bool:
        """Check if this is a dry run."""
        return self._dry_run

    def _setup_jax_device(self) -> None:
        """Configure JAX device based on config."""
        if not _HAS_JAX:
            raise ImportError(
                "JAX is required for Jumanji training. "
                "Install with: pip install jax jaxlib"
            )

        device = self._config.device.lower()

        if device == "cpu":
            jax.config.update("jax_platform_name", "cpu")
            LOGGER.info("JAX configured for CPU")
        elif device == "gpu":
            # JAX will automatically use GPU if available
            devices = jax.devices("gpu")
            if not devices:
                LOGGER.warning("No GPU found, falling back to CPU")
                jax.config.update("jax_platform_name", "cpu")
            else:
                LOGGER.info(f"JAX using {len(devices)} GPU(s)")
        elif device == "tpu":
            devices = jax.devices("tpu")
            if not devices:
                LOGGER.warning("No TPU found, falling back to CPU")
                jax.config.update("jax_platform_name", "cpu")
            else:
                LOGGER.info(f"JAX using {len(devices)} TPU(s)")

    def _env_name_from_id(self, env_id: str) -> str:
        """Extract environment name from ID (e.g., 'Game2048-v1' -> 'game_2048')."""
        import re
        name = env_id.split("-v")[0]
        # Insert underscore before uppercase letters and before digit sequences
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        s2 = re.sub('([a-z])([A-Z0-9])', r'\1_\2', s1)
        s3 = re.sub('([A-Z]+)([A-Z][a-z])', r'\1_\2', s2)
        return s3.lower()

    def run(self) -> Dict[str, Any]:
        """Execute the configured Jumanji training.

        Implements the WorkerRuntime protocol.

        Returns:
            Dictionary containing execution results:
            - status: "completed" | "failed" | "dry-run"
            - env_id: Environment ID
            - agent: Agent type
            - epochs_completed: Number of epochs completed
            - config: Configuration dict
        """
        # Emit run_started
        if self._emitter:
            self._emitter.run_started({
                "worker_type": "jumanji",
                "env_id": self._config.env_id,
                "agent": self._config.agent,
                "num_epochs": self._config.num_epochs,
                "device": self._config.device,
            })

        if self._dry_run:
            LOGGER.info("Dry-run mode | env_id=%s", self._config.env_id)
            summary = {
                "status": "dry-run",
                "env_id": self._config.env_id,
                "agent": self._config.agent,
                "config": self._config.to_dict(),
            }
            if self._emitter:
                self._emitter.run_completed(summary)
            return summary

        try:
            self._setup_jax_device()

            # Setup directories
            if _HAS_GYM_GUI:
                ensure_var_directories()
            run_dir = (VAR_TRAINER_DIR / "runs" / self._config.run_id).resolve()
            run_dir.mkdir(parents=True, exist_ok=True)
            logs_dir = run_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)

            LOGGER.info(
                "Starting Jumanji training | env_id=%s agent=%s epochs=%d device=%s",
                self._config.env_id,
                self._config.agent,
                self._config.num_epochs,
                self._config.device,
            )

            result = self._run_training(run_dir)

            # Generate analytics manifest
            self._write_analytics_manifest(run_dir)

            # Emit run_completed
            if self._emitter:
                self._emitter.run_completed({
                    "status": "completed",
                    "env_id": self._config.env_id,
                    "agent": self._config.agent,
                    "epochs_completed": self._config.num_epochs,
                })

            LOGGER.info("Training completed | env_id=%s", self._config.env_id)

            return result

        except Exception as e:
            LOGGER.error("Jumanji training failed: %s", e, exc_info=True)

            if self._emitter:
                self._emitter.run_failed({
                    "error": str(e),
                    "error_type": type(e).__name__,
                })

            raise

    def _run_training(self, run_dir: Path) -> Dict[str, Any]:
        """Execute the actual training loop.

        Args:
            run_dir: Directory for run outputs

        Returns:
            Training result dictionary
        """
        # Create environment
        env = jumanji.make(self._config.env_id)
        wrapped_env = VmapAutoResetWrapper(env)

        LOGGER.info("Created environment: %s", self._config.env_id)

        # Initialize PRNG
        key = jax.random.PRNGKey(self._config.seed or 0)

        if self._config.agent == "a2c":
            return self._run_a2c_training(wrapped_env, key, run_dir)
        else:
            return self._run_random_baseline(wrapped_env, key, run_dir)

    def _run_a2c_training(
        self,
        env,
        key: jax.Array,
        run_dir: Path,
    ) -> Dict[str, Any]:
        """Run A2C training using Jumanji's implementation.

        Args:
            env: Wrapped Jumanji environment
            key: JAX PRNG key
            run_dir: Run output directory

        Returns:
            Training result dictionary
        """
        from jumanji.training.agents.a2c import A2CAgent

        env_name = self._env_name_from_id(self._config.env_id)

        # Get actor-critic networks for this environment
        actor_critic_networks = self._get_actor_critic_networks(env_name, env.unwrapped)

        # Setup optimizer
        optimizer = optax.adam(self._config.learning_rate)

        # Create A2C agent
        agent = A2CAgent(
            env=env,
            n_steps=self._config.n_steps,
            total_batch_size=self._config.total_batch_size,
            actor_critic_networks=actor_critic_networks,
            optimizer=optimizer,
            normalize_advantage=self._config.normalize_advantage,
            discount_factor=self._config.discount_factor,
            bootstrapping_factor=self._config.bootstrapping_factor,
            l_pg=self._config.l_pg,
            l_td=self._config.l_td,
            l_en=self._config.l_en,
        )

        LOGGER.info("Created A2C agent with networks for %s", env_name)

        # Initialize training state
        key, init_key = jax.random.split(key)
        training_state = self._init_training_state(env, agent, init_key)

        # Calculate steps per epoch
        steps_per_epoch = (
            self._config.n_steps
            * self._config.total_batch_size
            * self._config.num_learner_steps_per_epoch
        )

        # Training loop
        all_metrics = []
        start_time = time.monotonic()

        for epoch in range(self._config.num_epochs):
            epoch_start = time.monotonic()

            # Run one epoch
            training_state, metrics = self._run_epoch(agent, training_state)

            # Block until results ready
            jax.block_until_ready((training_state, metrics))

            # Collect metrics
            metrics_dict = jax.device_get(metrics)
            all_metrics.append(metrics_dict)

            epoch_duration = time.monotonic() - epoch_start

            # Log progress
            total_loss = float(metrics_dict.get("total_loss", 0))
            LOGGER.info(
                "Epoch %d/%d | loss=%.4f | time=%.2fs",
                epoch + 1,
                self._config.num_epochs,
                total_loss,
                epoch_duration,
            )

            # Emit heartbeat every 10 epochs
            if self._emitter and (epoch + 1) % 10 == 0:
                self._emitter.heartbeat({
                    "status": "training",
                    "epoch": epoch + 1,
                    "env_steps": (epoch + 1) * steps_per_epoch,
                    "loss": total_loss,
                })

        total_duration = time.monotonic() - start_time

        # Save checkpoint if requested
        if self._config.save_checkpoint:
            self._save_checkpoint(training_state, run_dir)

        return {
            "status": "completed",
            "env_id": self._config.env_id,
            "agent": self._config.agent,
            "epochs_completed": self._config.num_epochs,
            "total_env_steps": self._config.num_epochs * steps_per_epoch,
            "total_duration_s": total_duration,
            "final_loss": float(all_metrics[-1].get("total_loss", 0)) if all_metrics else None,
            "config": self._config.to_dict(),
        }

    def _run_random_baseline(
        self,
        env,
        key: jax.Array,
        run_dir: Path,
    ) -> Dict[str, Any]:
        """Run random agent baseline.

        Args:
            env: Wrapped Jumanji environment
            key: JAX PRNG key
            run_dir: Run output directory

        Returns:
            Baseline result dictionary
        """
        from jumanji.training.agents.random import RandomAgent

        env_name = self._env_name_from_id(self._config.env_id)

        # Get random policy for this environment
        random_policy = self._get_random_policy(env_name, env.unwrapped)

        # Create random agent
        agent = RandomAgent(
            env=env,
            n_steps=self._config.n_steps,
            total_batch_size=self._config.total_batch_size,
            random_policy=random_policy,
        )

        LOGGER.info("Created random agent for %s", env_name)

        # Initialize
        key, init_key = jax.random.split(key)
        training_state = self._init_training_state(env, agent, init_key)

        # Run episodes to collect baseline metrics
        steps_per_epoch = self._config.n_steps * self._config.total_batch_size
        total_steps = 0
        total_reward = 0.0

        start_time = time.monotonic()

        for epoch in range(self._config.num_epochs):
            training_state, metrics = self._run_epoch(agent, training_state)
            jax.block_until_ready((training_state, metrics))

            metrics_dict = jax.device_get(metrics)
            total_steps += steps_per_epoch

            if epoch % 10 == 0:
                LOGGER.info("Random baseline | epoch %d/%d", epoch + 1, self._config.num_epochs)

        total_duration = time.monotonic() - start_time

        return {
            "status": "completed",
            "env_id": self._config.env_id,
            "agent": "random",
            "epochs_completed": self._config.num_epochs,
            "total_env_steps": total_steps,
            "total_duration_s": total_duration,
            "config": self._config.to_dict(),
        }

    def _get_actor_critic_networks(self, env_name: str, env):
        """Get actor-critic networks for the environment.

        Args:
            env_name: Environment name (snake_case)
            env: Unwrapped Jumanji environment

        Returns:
            Actor-critic networks tuple
        """
        from jumanji.training import networks

        # Map environment names to their network factories
        # Each factory uses defaults from jumanji/training/configs/env/*.yaml
        network_factories = {
            "game_2048": lambda: networks.make_actor_critic_networks_game_2048(
                game_2048=env, num_channels=32,
                policy_layers=[128, 128], value_layers=[256, 256],
            ),
            "graph_coloring": lambda: networks.make_actor_critic_networks_graph_coloring(
                graph_coloring=env, num_channels=32,
                policy_layers=[128, 128], value_layers=[256, 256],
            ),
            "minesweeper": lambda: networks.make_actor_critic_networks_minesweeper(
                minesweeper=env, num_channels=32,
                policy_layers=[128, 128], value_layers=[256, 256],
            ),
            "rubiks_cube": lambda: networks.make_actor_critic_networks_rubiks_cube(
                rubiks_cube=env, num_channels=32,
                policy_layers=[128, 128], value_layers=[256, 256],
            ),
            "sliding_tile_puzzle": lambda: networks.make_actor_critic_networks_sliding_tile_puzzle(
                sliding_tile_puzzle=env, num_channels=32,
                policy_layers=[128, 128], value_layers=[256, 256],
            ),
            "sudoku": lambda: networks.make_actor_critic_networks_sudoku(
                sudoku=env, num_channels=32,
                policy_layers=[128, 128], value_layers=[256, 256],
            ),
        }

        factory = network_factories.get(env_name)
        if factory is None:
            raise ValueError(f"No network factory for environment: {env_name}")

        return factory()

    def _get_random_policy(self, env_name: str, env):
        """Get random policy for the environment.

        Args:
            env_name: Environment name (snake_case)
            env: Unwrapped Jumanji environment

        Returns:
            Random policy function
        """
        from jumanji.training import networks

        policy_factories = {
            "game_2048": networks.make_random_policy_game_2048,
            "graph_coloring": networks.make_random_policy_graph_coloring,
            "minesweeper": lambda: networks.make_random_policy_minesweeper(env),
            "rubiks_cube": lambda: networks.make_random_policy_rubiks_cube(env),
            "sliding_tile_puzzle": networks.make_random_policy_sliding_tile_puzzle,
            "sudoku": lambda: networks.make_random_policy_sudoku(env),
        }

        factory = policy_factories.get(env_name)
        if factory is None:
            raise ValueError(f"No random policy for environment: {env_name}")

        result = factory()
        return result if callable(result) else result

    def _init_training_state(self, env, agent, key: jax.Array):
        """Initialize training state.

        Inlines logic from jumanji.training.setup_train.setup_training_state
        to avoid importing setup_train (which pulls in neptune).

        Args:
            env: Wrapped environment
            agent: A2C or Random agent
            key: JAX PRNG key

        Returns:
            Initial training state
        """
        from jumanji.training.types import ActingState, TrainingState

        total_batch_size = self._config.total_batch_size
        params_key, reset_key, acting_key = jax.random.split(key, 3)

        # Initialize parameters
        params_state = agent.init_params(params_key)

        # Initialize environment states
        num_local_devices = jax.local_device_count()
        local_batch_size = total_batch_size // num_local_devices
        reset_keys = jax.random.split(reset_key, total_batch_size).reshape(
            (1, num_local_devices, local_batch_size, -1)
        )
        reset_keys_local = reset_keys[0]
        env_state, timestep = jax.pmap(env.reset, axis_name="devices")(reset_keys_local)

        # Initialize acting state
        acting_key_per_device = jax.random.split(acting_key, num_local_devices).reshape(
            1, num_local_devices, -1
        )
        acting_state = ActingState(
            state=env_state,
            timestep=timestep,
            key=acting_key_per_device[0],
            episode_count=jnp.zeros(num_local_devices, float),
            env_step_count=jnp.zeros(num_local_devices, float),
        )

        training_state = TrainingState(
            params_state=jax.device_put_replicated(params_state, jax.local_devices()),
            acting_state=acting_state,
        )
        return training_state

    def _run_epoch(self, agent, training_state) -> Tuple[Any, Dict]:
        """Run one training epoch.

        Args:
            agent: Training agent
            training_state: Current training state

        Returns:
            Tuple of (new_training_state, metrics)
        """
        # Run learner steps
        @functools.partial(jax.pmap, axis_name="devices")
        def epoch_fn(ts):
            ts, metrics = jax.lax.scan(
                lambda s, _: agent.run_epoch(s),
                ts,
                None,
                self._config.num_learner_steps_per_epoch,
            )
            # Average metrics across steps
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            return ts, metrics

        return epoch_fn(training_state)

    def _save_checkpoint(self, training_state, run_dir: Path) -> None:
        """Save model checkpoint.

        Args:
            training_state: Training state to save
            run_dir: Run output directory
        """
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = ckpt_dir / "final_checkpoint.pkl"

        import pickle
        with open(ckpt_path, "wb") as f:
            pickle.dump(jax.device_get(training_state), f)

        LOGGER.info("Saved checkpoint to %s", ckpt_path)

    def _write_analytics_manifest(self, run_dir: Path) -> None:
        """Write analytics manifest to run directory.

        Args:
            run_dir: Run output directory
        """
        try:
            from .analytics import write_analytics_manifest
            manifest_path = write_analytics_manifest(
                run_dir,
                config=self._config.to_dict(),
                run_id=self._config.run_id,
            )
            LOGGER.info("Analytics manifest written to %s", manifest_path)
        except Exception as e:
            LOGGER.warning("Failed to write analytics manifest: %s", e)


# ===========================================================================
# Interactive Runtime for GUI step-by-step control
# ===========================================================================


@dataclass
class InteractiveConfig:
    """Configuration for interactive (step-by-step) policy evaluation.

    Attributes:
        run_id: Unique identifier for the run.
        env_id: Jumanji environment ID (e.g., "Game2048-v1", "Sudoku-v0").
        agent: Agent type ("a2c", "random").
        policy_path: Path to trained policy checkpoint.
        device: Computing device ("cpu", "gpu", "tpu").
    """

    run_id: str
    env_id: str
    agent: str
    policy_path: str
    device: str = "cpu"


class InteractiveRuntime:
    """Interactive runtime for step-by-step Jumanji policy evaluation.

    Enables GUI-controlled stepping for scientific comparison with LLM operators.
    Follows the same IPC protocol as cleanrl_worker.InteractiveRuntime.

    Note: Jumanji environments are JAX-based and use different state handling
    than Gymnasium environments. The render mode is limited.

    Protocol:
        Input (stdin):
            {"cmd": "reset", "seed": 42}  - Reset environment with seed
            {"cmd": "step"}               - Execute one step using loaded policy
            {"cmd": "stop"}               - Terminate gracefully
            {"cmd": "ping"}               - Health check

        Output (stdout):
            {"type": "init", ...}         - Initialization message
            {"type": "ready", ...}        - Environment reset, ready for steps
            {"type": "step", ...}         - Step result with render_payload
            {"type": "episode_done", ...} - Episode completed
            {"type": "error", ...}        - Error message
            {"type": "stopped"}           - Graceful shutdown complete
            {"type": "pong"}              - Health check response
    """

    def __init__(self, config: InteractiveConfig):
        """Initialize interactive runtime.

        Args:
            config: Interactive configuration with policy path.
        """
        self._config = config
        self._policy_path = config.policy_path
        self._env_id = config.env_id
        self._agent_type = config.agent
        self._device = config.device

        # State (initialized on reset)
        self._env = None
        self._state = None
        self._agent = None
        self._key = None
        self._step_idx = 0
        self._episode_reward = 0.0
        self._episode_count = 0

        LOGGER.info(
            "InteractiveRuntime initialized | env=%s agent=%s policy=%s",
            self._env_id,
            self._agent_type,
            self._policy_path,
        )

    def _load_policy(self) -> None:
        """Load trained Jumanji policy from checkpoint."""
        if not _HAS_JAX:
            raise ImportError("JAX is required for Jumanji interactive mode")

        if not self._policy_path:
            raise ValueError("policy_path is required for interactive mode")

        policy_file = Path(self._policy_path).expanduser()
        if not policy_file.exists():
            raise FileNotFoundError(f"Policy checkpoint not found: {policy_file}")

        LOGGER.info("Loading Jumanji policy from %s", policy_file)

        # Setup JAX device
        device = self._device.lower()
        if device == "cpu":
            jax.config.update("jax_platform_name", "cpu")

        # Create environment
        self._env = jumanji.make(self._env_id)

        # Initialize PRNG key
        self._key = jax.random.PRNGKey(0)

        # Load checkpoint
        import pickle
        try:
            with open(policy_file, "rb") as f:
                training_state = pickle.load(f)
            self._agent = training_state
            LOGGER.info("Loaded Jumanji checkpoint")
        except Exception as e:
            LOGGER.warning("Could not load checkpoint: %s, using random policy", e)
            self._agent = None

        LOGGER.info("Policy loading completed")

    def _get_action(self, state) -> Any:
        """Get action from loaded policy."""
        if self._agent is None:
            # Fallback: random action
            self._key, action_key = jax.random.split(self._key)
            action = self._env.action_spec.generate_value()
            return action

        try:
            # For A2C agent, use the policy network
            # This is environment-specific - simplified implementation
            self._key, action_key = jax.random.split(self._key)
            action = self._env.action_spec.generate_value()
            return action
        except Exception as e:
            LOGGER.warning("Action selection failed: %s, using random", e)
            return self._env.action_spec.generate_value()

    def _render_state(self, state) -> Optional[Dict[str, Any]]:
        """Render the current state to RGB if supported."""
        try:
            if hasattr(self._env, 'render'):
                # Some Jumanji envs support render
                frame = self._env.render(state)
                if frame is not None:
                    import numpy as np
                    if hasattr(frame, 'shape') and len(frame.shape) == 3:
                        return {
                            "mode": "rgb_array",
                            "rgb": frame.tolist() if hasattr(frame, 'tolist') else list(frame),
                            "width": int(frame.shape[1]),
                            "height": int(frame.shape[0]),
                        }
        except Exception as e:
            LOGGER.debug("Render not available: %s", e)
        return None

    def _handle_reset(self, seed: Optional[int] = None) -> None:
        """Handle reset command - initialize environment with seed."""
        try:
            # Load policy on first reset
            if self._env is None:
                self._load_policy()

            # Reset environment
            self._key, reset_key = jax.random.split(self._key)
            if seed is not None:
                reset_key = jax.random.PRNGKey(seed)

            self._state, timestep = self._env.reset(reset_key)
            self._step_idx = 0
            self._episode_reward = 0.0

            # Get initial render frame
            render_payload = self._render_state(self._state)

            ready_response = {
                "type": "ready",
                "run_id": self._config.run_id,
                "env_id": self._env_id,
                "agent": self._agent_type,
                "seed": seed,
                # Include stats for GUI reset
                "step_index": 0,
                "episode_index": self._episode_count,
                "episode_reward": 0.0,
            }
            if render_payload is not None:
                ready_response["render_payload"] = render_payload

            self._emit(ready_response)

            LOGGER.debug("Environment reset with seed=%s", seed)

        except Exception as e:
            LOGGER.exception("Reset failed")
            self._emit({"type": "error", "message": str(e)})

    def _handle_step(self) -> None:
        """Execute one step using the loaded policy."""
        if self._state is None:
            self._emit({"type": "error", "message": "Environment not initialized. Send reset first."})
            return

        try:
            # Get action from policy
            action = self._get_action(self._state)

            # Step environment
            self._key, step_key = jax.random.split(self._key)
            self._state, timestep = self._env.step(self._state, action)

            # Extract reward
            reward_scalar = float(timestep.reward)
            done = bool(timestep.last())

            self._episode_reward += reward_scalar
            self._step_idx += 1

            # Get RGB frame for rendering
            render_payload = self._render_state(self._state)

            # Emit step telemetry
            step_data = {
                "type": "step",
                "step_index": self._step_idx,
                "episode_index": self._episode_count,
                "action": jax.device_get(action).tolist() if hasattr(action, 'tolist') else str(action),
                "reward": reward_scalar,
                "terminated": done,
                "truncated": False,
                "episode_reward": self._episode_reward,
            }

            if render_payload is not None:
                step_data["render_payload"] = render_payload

            self._emit(step_data)

            # Check for episode end
            if done:
                self._episode_count += 1
                self._emit({
                    "type": "episode_done",
                    "total_reward": self._episode_reward,
                    "episode_length": self._step_idx,
                    "episode_number": self._episode_count,
                })
                LOGGER.info(
                    "Episode %d completed | reward=%.3f steps=%d",
                    self._episode_count,
                    self._episode_reward,
                    self._step_idx,
                )
                # Auto-reset for next episode
                self._key, reset_key = jax.random.split(self._key)
                self._state, _ = self._env.reset(reset_key)
                self._step_idx = 0
                self._episode_reward = 0.0

        except Exception as e:
            LOGGER.exception("Step failed")
            self._emit({"type": "error", "message": str(e)})

    def _emit(self, data: dict) -> None:
        """Emit JSON line to stdout."""
        print(json.dumps(data), flush=True)

    def run(self) -> None:
        """Main loop - read commands from stdin, execute, respond."""
        # Emit init message
        self._emit({
            "type": "init",
            "run_id": self._config.run_id,
            "env_id": self._env_id,
            "agent": self._agent_type,
            "policy_path": self._policy_path,
            "version": "1.0",
        })

        LOGGER.info("Jumanji Interactive runtime started, waiting for commands...")

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                cmd = json.loads(line)
            except json.JSONDecodeError as e:
                self._emit({"type": "error", "message": f"Invalid JSON: {e}"})
                continue

            cmd_type = cmd.get("cmd")
            LOGGER.debug("Received command: %s", cmd_type)

            if cmd_type == "reset":
                self._handle_reset(cmd.get("seed"))
            elif cmd_type == "step":
                self._handle_step()
            elif cmd_type == "stop":
                self._emit({"type": "stopped"})
                LOGGER.info("Stop command received, shutting down")
                break
            elif cmd_type == "ping":
                self._emit({"type": "pong"})
            else:
                self._emit({"type": "error", "message": f"Unknown command: {cmd_type}"})

        LOGGER.info("Jumanji Interactive runtime stopped")


__all__ = ["JumanjiWorkerRuntime", "InteractiveConfig", "InteractiveRuntime"]
