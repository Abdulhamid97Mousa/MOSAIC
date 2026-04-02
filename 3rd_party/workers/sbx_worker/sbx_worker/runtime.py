"""Runtime orchestration helpers for launching SBX (Stable-Baselines Jax) algorithms."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from gym_gui.config.paths import VAR_TRAINER_DIR
except ImportError:
    VAR_TRAINER_DIR = Path("var/trainer")

try:
    from gym_gui.fastlane.worker_helpers import apply_fastlane_environment
except ImportError:
    apply_fastlane_environment = None

try:
    from gym_gui.telemetry.semconv import VideoModes
except ImportError:
    VideoModes = None

try:
    from gym_gui.logging_config.helpers import log_constant
    from gym_gui.logging_config.log_constants import (
        LOG_WORKER_SBX_RUNTIME_STARTED,
        LOG_WORKER_SBX_RUNTIME_COMPLETED,
        LOG_WORKER_SBX_RUNTIME_FAILED,
        LOG_WORKER_SBX_SUBPROCESS_STDOUT,
        LOG_WORKER_SBX_SUBPROCESS_STDERR,
    )
    _HAS_LOG_CONSTANTS = True
except ImportError:
    _HAS_LOG_CONSTANTS = False

from .config import SBXWorkerConfig

LOGGER = logging.getLogger(__name__)


class SBXWorkerRuntime:
    """Orchestrates SBX training jobs."""

    def __init__(self, work_dir: Path | str) -> None:
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def execute(self, config: SBXWorkerConfig) -> None:
        """Execute an SBX training run based on configuration.

        Args:
            config: SBXWorkerConfig instance
        """
        if _HAS_LOG_CONSTANTS:
            log_constant(
                LOGGER,
                LOG_WORKER_SBX_RUNTIME_STARTED,
                extra={"run_id": config.run_id, "algo": config.algo, "env_id": config.env_id},
            )
        else:
            LOGGER.info(
                "SBX worker runtime started: run_id=%s algo=%s env_id=%s",
                config.run_id, config.algo, config.env_id,
            )

        # Ensure tensorboard directory exists
        tb_log_dir = VAR_TRAINER_DIR / "runs" / config.run_id
        tb_log_dir.mkdir(parents=True, exist_ok=True)

        # Prepare the command to run the launcher
        cmd = [
            sys.executable,
            "-m",
            "sbx_worker.launcher",
            "--run-id",
            config.run_id,
            "--algo",
            config.algo,
            "--env-id",
            config.env_id,
            "--total-timesteps",
            str(config.total_timesteps),
        ]

        if config.seed is not None:
            cmd.extend(["--seed", str(config.seed)])

        if config.worker_id:
            cmd.extend(["--worker-id", config.worker_id])

        # Save full config to a file so launcher can read it
        config_path = self.work_dir / f"config-{config.run_id}.json"
        config_path.write_text(json.dumps(config.to_dict(), indent=2))
        cmd.extend(["--config-file", str(config_path)])

        # Add work_dir to cmd
        cmd.extend(["--work-dir", str(self.work_dir)])

        # Add log_dir to cmd
        cmd.extend(["--log-dir", str(tb_log_dir)])

        env = os.environ.copy()

        # Configure FastLane
        if config.extras.get("fastlane_enabled") and apply_fastlane_environment is not None:
            try:
                video_mode = config.extras.get("video_mode")
                if VideoModes is not None:
                    video_mode = video_mode or VideoModes.SINGLE
                apply_fastlane_environment(
                    env,
                    fastlane_only=config.extras.get("eval_only", False),
                    fastlane_slot=0,
                    video_mode=video_mode,
                    grid_limit=4,
                )
            except Exception as e:
                LOGGER.warning("Failed to configure FastLane: %s", e)

        LOGGER.info("Executing command: %s", " ".join(cmd))

        try:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            while True:
                output = process.stdout.readline()
                if output == "" and process.poll() is not None:
                    break
                if output:
                    line = output.strip()
                    print(line)
                    if _HAS_LOG_CONSTANTS:
                        log_constant(
                            LOGGER,
                            LOG_WORKER_SBX_SUBPROCESS_STDOUT,
                            extra={"output": line, "run_id": config.run_id},
                        )

            # Check for stderr after loop
            stderr_output = process.stderr.read()
            if stderr_output:
                print(stderr_output, file=sys.stderr)
                if _HAS_LOG_CONSTANTS:
                    log_constant(
                        LOGGER,
                        LOG_WORKER_SBX_SUBPROCESS_STDERR,
                        extra={"output": stderr_output, "run_id": config.run_id},
                    )

            return_code = process.poll()

            if return_code != 0:
                raise RuntimeError(f"SBX worker process failed with exit code {return_code}")

            if _HAS_LOG_CONSTANTS:
                log_constant(
                    LOGGER,
                    LOG_WORKER_SBX_RUNTIME_COMPLETED,
                    extra={"run_id": config.run_id},
                )
            else:
                LOGGER.info("SBX worker runtime completed: run_id=%s", config.run_id)

        except Exception as e:
            if _HAS_LOG_CONSTANTS:
                log_constant(
                    LOGGER,
                    LOG_WORKER_SBX_RUNTIME_FAILED,
                    exc_info=e,
                    extra={"run_id": config.run_id},
                )
            else:
                LOGGER.exception("SBX worker runtime failed: run_id=%s", config.run_id)
            raise
