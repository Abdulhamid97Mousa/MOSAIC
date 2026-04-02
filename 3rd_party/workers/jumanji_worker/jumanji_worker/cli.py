"""CLI entry point for Jumanji worker.

This module provides the command-line interface for running Jumanji
training jobs, either directly or as a subprocess launched by MOSAIC.

Usage:
    # Config file mode
    jumanji-worker --config /path/to/config.json

    # Direct parameter mode
    jumanji-worker --env-id Game2048-v1 --agent a2c --num-epochs 100

    # GPU training
    jumanji-worker --env-id RubiksCube-v0 --device gpu --num-epochs 500

    # Dry-run mode (validate without executing)
    jumanji-worker --env-id Sudoku-v0 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Optional

from .config import JumanjiWorkerConfig, LOGIC_ENVIRONMENTS


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        args: Command-line arguments (None for sys.argv).

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Jumanji Worker - JAX-based RL environments for MOSAIC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file mode
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to JSON configuration file",
    )

    # Environment selection
    parser.add_argument(
        "--env-id",
        type=str,
        default="Game2048-v1",
        choices=sorted(LOGIC_ENVIRONMENTS),
        help="Jumanji environment ID",
    )

    # Agent selection
    parser.add_argument(
        "--agent",
        type=str,
        default="a2c",
        choices=["a2c", "random"],
        help="Agent type (a2c for training, random for baseline)",
    )

    # Training parameters
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=20,
        help="Rollout length per update",
    )
    parser.add_argument(
        "--total-batch-size",
        type=int,
        default=128,
        help="Total batch size across devices",
    )
    parser.add_argument(
        "--num-learner-steps-per-epoch",
        type=int,
        default=64,
        help="Learning steps per epoch",
    )

    # A2C hyperparameters
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Optimizer learning rate",
    )
    parser.add_argument(
        "--discount-factor",
        type=float,
        default=0.99,
        help="Reward discounting (gamma)",
    )
    parser.add_argument(
        "--bootstrapping-factor",
        type=float,
        default=0.95,
        help="GAE lambda for bootstrapping",
    )
    parser.add_argument(
        "--l-pg",
        type=float,
        default=1.0,
        help="Policy gradient loss coefficient",
    )
    parser.add_argument(
        "--l-td",
        type=float,
        default=0.5,
        help="TD loss coefficient",
    )
    parser.add_argument(
        "--l-en",
        type=float,
        default=0.01,
        help="Entropy loss coefficient",
    )

    # Device configuration
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu", "tpu"],
        help="JAX device backend",
    )

    # Seed
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    # Logging and checkpointing
    parser.add_argument(
        "--logger-type",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "terminal"],
        help="Logger type for training metrics",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpoint saving",
    )

    # Run configuration
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Unique run identifier (auto-generated if not provided)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration without executing",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    # MOSAIC integration arguments (passed by trainer dispatcher)
    parser.add_argument(
        "--grpc",
        action="store_true",
        help="Enable gRPC telemetry (passed by MOSAIC dispatcher)",
    )
    parser.add_argument(
        "--grpc-target",
        type=str,
        default="127.0.0.1:50055",
        help="gRPC telemetry server address",
    )
    parser.add_argument(
        "--worker-id",
        type=str,
        default=None,
        help="Worker identifier (for multi-worker training)",
    )

    # Interactive mode for Multi-Operator view (step-by-step policy evaluation)
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode for GUI step-by-step policy evaluation",
    )
    parser.add_argument(
        "--policy-path",
        type=str,
        default=None,
        help="Path to trained policy checkpoint (required for interactive mode)",
    )

    return parser.parse_args(args)


def main(args: Optional[list[str]] = None) -> int:
    """Main entry point for jumanji-worker CLI.

    Args:
        args: Command-line arguments (None for sys.argv).

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    parsed = parse_args(args)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, parsed.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("jumanji_worker")

    # Build configuration
    if parsed.config is not None:
        # Load from JSON file
        if not parsed.config.exists():
            logger.error("Config file not found: %s", parsed.config)
            return 1
        try:
            from .config import load_worker_config
            config = load_worker_config(str(parsed.config))
            # Apply CLI overrides
            config = config.with_overrides(
                env_id=parsed.env_id if parsed.env_id != "Game2048-v1" else None,
                agent=parsed.agent if parsed.agent != "a2c" else None,
                seed=parsed.seed,
                num_epochs=parsed.num_epochs if parsed.num_epochs != 100 else None,
                device=parsed.device if parsed.device != "cpu" else None,
            )
            logger.info("Loaded config from: %s", parsed.config)
        except Exception as e:
            logger.error("Failed to load config: %s", e)
            return 1
    else:
        # Build from CLI arguments
        config = JumanjiWorkerConfig(
            run_id=parsed.run_id or f"jumanji_{uuid.uuid4().hex[:8]}",
            env_id=parsed.env_id,
            agent=parsed.agent,
            seed=parsed.seed,
            num_epochs=parsed.num_epochs,
            n_steps=parsed.n_steps,
            total_batch_size=parsed.total_batch_size,
            num_learner_steps_per_epoch=parsed.num_learner_steps_per_epoch,
            learning_rate=parsed.learning_rate,
            discount_factor=parsed.discount_factor,
            bootstrapping_factor=parsed.bootstrapping_factor,
            normalize_advantage=True,
            l_pg=parsed.l_pg,
            l_td=parsed.l_td,
            l_en=parsed.l_en,
            device=parsed.device,
            logger_type=parsed.logger_type,
            save_checkpoint=not parsed.no_checkpoint,
            worker_id=parsed.worker_id,
        )

    # Log gRPC settings if enabled
    if parsed.grpc:
        logger.info("gRPC telemetry enabled, target: %s", parsed.grpc_target)

    logger.info(
        "Jumanji Worker Config: env_id=%s agent=%s epochs=%d device=%s seed=%s",
        config.env_id,
        config.agent,
        config.num_epochs,
        config.device,
        config.seed,
    )

    # Dry-run mode: print config and exit
    if parsed.dry_run:
        print(json.dumps(config.to_dict(), indent=2))
        return 0

    # Interactive mode: run step-by-step policy evaluation
    if parsed.interactive:
        from .runtime import InteractiveRuntime, InteractiveConfig

        if not parsed.policy_path:
            logger.error("--policy-path is required for interactive mode")
            return 1

        interactive_config = InteractiveConfig(
            run_id=config.run_id,
            env_id=config.env_id,
            agent=config.agent,
            policy_path=parsed.policy_path,
            device=config.device,
        )
        runtime = InteractiveRuntime(interactive_config)
        runtime.run()
        return 0

    # Import runtime (lazy to avoid JAX import until needed)
    try:
        from .runtime import JumanjiWorkerRuntime
    except ImportError as e:
        logger.error(
            "Failed to import runtime. JAX/Jumanji may not be installed: %s", e
        )
        return 1

    # Create runtime and execute
    runtime = JumanjiWorkerRuntime(config, dry_run=False)

    try:
        summary = runtime.run()

        logger.info(
            "Run completed: status=%s env=%s",
            summary.get("status", "unknown") if isinstance(summary, dict) else getattr(summary, "status", "unknown"),
            config.env_id,
        )
        return 0

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130  # Standard Unix SIGINT exit code

    except Exception as e:
        logger.exception("Jumanji worker failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
