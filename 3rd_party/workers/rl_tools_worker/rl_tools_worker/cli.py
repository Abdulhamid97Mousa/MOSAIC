"""CLI entry point for RLtools worker."""

import argparse
import logging
import sys
from pathlib import Path

from .config import load_worker_config
from .runtime import RLToolsWorkerRuntime


def main() -> int:
    parser = argparse.ArgumentParser(description="RLtools MOSAIC Worker")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    try:
        config = load_worker_config(str(args.config))
        runtime = RLToolsWorkerRuntime(work_dir=args.work_dir)
        runtime.execute(config)
        return 0
    except Exception as e:
        print(f"RLtools worker failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
