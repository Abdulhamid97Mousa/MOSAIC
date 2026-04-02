"""Runtime for RLtools worker."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from .config import RLToolsWorkerConfig


class RLToolsWorkerRuntime:
    """Spawns the RLtools launcher as a subprocess."""

    def __init__(self, work_dir: Path):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def execute(self, config: RLToolsWorkerConfig) -> None:
        log_dir = self.work_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        config_path = self.work_dir / f"config-{config.run_id}.json"
        with open(config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)

        cmd = [
            sys.executable, "-m", "rl_tools_worker.launcher",
            "--run-id", config.run_id,
            "--algo", config.algo,
            "--env-id", config.env_id,
            "--total-timesteps", str(config.total_timesteps),
            "--seed", str(config.seed or 0),
            "--config-file", str(config_path),
            "--work-dir", str(self.work_dir),
            "--log-dir", str(log_dir),
        ]

        env = os.environ.copy()
        env["WANDB_MODE"] = "disabled"

        result = subprocess.run(cmd, capture_output=True, text=True, env=env)

        if result.stdout:
            for line in result.stdout.splitlines():
                print(f"  [rltools] {line}")

        if result.returncode != 0:
            raise RuntimeError(
                f"RLtools worker failed (exit {result.returncode}): "
                f"{result.stderr[-500:] if result.stderr else 'no stderr'}"
            )
