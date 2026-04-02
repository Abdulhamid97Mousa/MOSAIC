"""CLI entrypoint for Tianshou worker."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import load_worker_config
from .runtime import TianshouWorkerRuntime

LOGGER = logging.getLogger(__name__)


def main() -> int:
    """Main entry point for the Tianshou worker CLI."""
    parser = argparse.ArgumentParser(description="Tianshou Worker CLI")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to worker configuration JSON file",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("."),
        help="Working directory for artifacts",
    )

    args = parser.parse_args()

    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        LOGGER.info(f"Loading configuration from {args.config}")
        config = load_worker_config(args.config)
        
        runtime = TianshouWorkerRuntime(work_dir=args.work_dir)
        runtime.execute(config)
        
        return 0
    except Exception as e:
        LOGGER.exception("Worker execution failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
