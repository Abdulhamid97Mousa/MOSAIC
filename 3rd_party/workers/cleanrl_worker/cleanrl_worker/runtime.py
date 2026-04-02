

"""Runtime orchestration helpers for launching CleanRL algorithms."""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Iterator

import grpc
from gym_gui.config.paths import VAR_TRAINER_DIR, ensure_var_directories
from gym_gui.telemetry.semconv import TelemetryEnv, VideoModes
from . import fastlane as fastlane_module
from .eval_registry import get_eval_entry
from .eval import EvalBatchSummary, EvalRunResult, run_batched_evaluation
from gym_gui.services.trainer.proto import trainer_pb2, trainer_pb2_grpc
from gym_gui.logging_config.helpers import log_constant
from gym_gui.logging_config.log_constants import (
    LOG_WORKER_POLICY_EVAL_STARTED,
    LOG_WORKER_POLICY_EVAL_COMPLETED,
    LOG_WORKER_POLICY_EVAL_BATCH_STARTED,
    LOG_WORKER_POLICY_EVAL_BATCH_COMPLETED,
    LOG_WORKER_CLEANRL_RUNTIME_STARTED,
    LOG_WORKER_CLEANRL_RUNTIME_COMPLETED,
    LOG_WORKER_CLEANRL_RUNTIME_FAILED,
    LOG_WORKER_CLEANRL_HEARTBEAT,
    LOG_WORKER_CLEANRL_MODULE_RESOLVED,
    LOG_WORKER_CLEANRL_CONFIG_LOADED,
    LOG_WORKER_CLEANRL_TENSORBOARD_ENABLED,
    LOG_WORKER_CLEANRL_WANDB_ENABLED,
    LOG_WORKER_CLEANRL_SUBPROCESS_STARTED,
    LOG_WORKER_CLEANRL_SUBPROCESS_FAILED,
    LOG_WORKER_CLEANRL_ANALYTICS_MANIFEST_CREATED,
    LOG_WORKER_CLEANRL_EPISODE_AUTO_RESET,
    LOG_WORKER_CLEANRL_SUBPROCESS_OOM_KILLED,
    LOG_WORKER_CLEANRL_SUBPROCESS_SIGNALED,
    LOG_WORKER_CLEANRL_SUBPROCESS_NO_OUTPUT,
    LOG_WORKER_CLEANRL_SUBPROCESS_MEMORY_SNAPSHOT,
    LOG_WORKER_CLEANRL_SUBPROCESS_EXIT_SUMMARY,
    LOG_WORKER_CLEANRL_TENSORBOARD_PATH_RESOLVED,
    LOG_WORKER_CLEANRL_SUBPROCESS_SLOW_START,
    LOG_WORKER_CLEANRL_SUBPROCESS_STDERR_ONLY,
)

from .analytics import build_manifest
from .config import CleanRLWorkerConfig, WorkerConfig  # WorkerConfig is backwards compat alias
from gym_gui.core.worker import TelemetryEmitter, WorkerAnalyticsManifest
from contextlib import contextmanager, redirect_stdout, redirect_stderr

REPO_ROOT = Path(__file__).resolve().parents[4]
_MODULE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.]+$")
_CMD_COMPONENT_PATTERN = re.compile(r"^[^\n\r\x00]*$")
_SHM_DIR = Path("/dev/shm")


def _cleanup_orphaned_shm(*, run_id: Optional[str] = None) -> int:
    """Remove orphaned shared-memory segments left by killed subprocesses.

    Python's multiprocessing.shared_memory creates ``psm_*`` tracker files
    in /dev/shm.  When a subprocess is SIGKILL'd these are never cleaned up
    and accumulate over time.

    Also cleans up any FastLane segments (``mosaic.fastlane.*``) whose
    owning process no longer exists.

    Returns the number of segments removed.
    """
    if not _SHM_DIR.is_dir():
        return 0
    removed = 0
    uid = os.getuid()
    for entry in _SHM_DIR.iterdir():
        try:
            # Only touch files owned by us
            if entry.stat().st_uid != uid:
                continue
            name = entry.name
            # Orphaned Python shared-memory tracker semaphores
            if name.startswith("psm_"):
                entry.unlink()
                removed += 1
            # Stale FastLane segments from previous runs
            elif name.startswith("mosaic.fastlane."):
                if run_id and run_id in name:
                    # Current run's segment — leave it
                    continue
                entry.unlink()
                removed += 1
        except (PermissionError, FileNotFoundError, OSError):
            continue
    if removed:
        LOGGER.info(
            "Cleaned up %d orphaned /dev/shm segments", removed,
        )
    return removed


LOGGER = logging.getLogger(__name__)


class _EvalStepLoggingWrapper:
    """Lightweight wrapper that logs every evaluation step for debugging."""

    def __init__(self, env, run_id: str, slot: int):
        self.env = env
        self._run_id = run_id
        self._slot = slot
        self._step = 0

    def reset(self, *args, **kwargs):  # pragma: no cover - env specific
        self._step = 0
        LOGGER.debug(
            "Policy eval env reset | run_id=%s slot=%s",
            self._run_id,
            self._slot,
        )
        return self.env.reset(*args, **kwargs)

    def step(self, action):  # pragma: no cover - env specific
        result = self.env.step(action)
        self._step += 1
        obs, reward, terminated, truncated, info = result
        try:
            reward_value = float(reward[0] if hasattr(reward, "__getitem__") else reward)
        except Exception:  # pragma: no cover - defensive conversion
            reward_value = 0.0
        terminated_flag = bool(terminated[0] if hasattr(terminated, "__getitem__") else terminated)
        truncated_flag = bool(truncated[0] if hasattr(truncated, "__getitem__") else truncated)
        LOGGER.debug(
            "Policy eval env step | run_id=%s slot=%s step=%s reward=%.4f done=%s truncated=%s",
            self._run_id,
            self._slot,
            self._step,
            reward_value,
            terminated_flag,
            truncated_flag,
        )
        if terminated_flag or truncated_flag:
            LOGGER.info(
                "Policy eval episode finished | run_id=%s slot=%s steps=%s done=%s truncated=%s",
                self._run_id,
                self._slot,
                self._step,
                terminated_flag,
                truncated_flag,
            )
        return result

    def __getattr__(self, name):  # pragma: no cover - proxy helpers
        return getattr(self.env, name)

try:  # pragma: no cover - optional dependency
    from torch.utils.tensorboard import SummaryWriter as _TensorBoardWriter
except Exception:  # pragma: no cover - optional dependency
    _TensorBoardWriter = None

AlgoRegistry = Mapping[str, str]


DEFAULT_ALGO_REGISTRY: AlgoRegistry = {
    # --------------------------------------------------------------------------
    # MOSAIC custom algorithms (cleanrl_worker.algorithms.*)
    # These have MiniGrid/BabyAI support, custom agents, curriculum learning
    # --------------------------------------------------------------------------
    "ppo": "cleanrl_worker.algorithms.ppo",
    "ppo_gru": "cleanrl_worker.algorithms.ppo_gru",
    # --------------------------------------------------------------------------
    # Upstream CleanRL algorithms (cleanrl.*)
    # These use the original gym/gymnasium API that CleanRL was built with.
    # We do NOT use our cleanrl_worker.algorithms.* copies for these because
    # the upstream code was written for the old gym API and works correctly
    # with the vendored cleanrl package.
    # --------------------------------------------------------------------------
    # PPO family
    "ppo_continuous_action": "cleanrl_worker.algorithms.ppo_continuous_action",
    "ppo_atari": "cleanrl.ppo_atari",
    "ppo_atari_multigpu": "cleanrl.ppo_atari_multigpu",
    "ppo_atari_lstm": "cleanrl.ppo_atari_lstm",
    "ppo_atari_envpool": "cleanrl.ppo_atari_envpool",
    "ppo_atari_envpool_xla_jax": "cleanrl.ppo_atari_envpool_xla_jax",
    "ppo_atari_envpool_xla_jax_scan": "cleanrl.ppo_atari_envpool_xla_jax_scan",
    "ppo_procgen": "cleanrl.ppo_procgen",
    "ppo_rnd_envpool": "cleanrl.ppo_rnd_envpool",
    "ppo_pettingzoo_ma_atari": "cleanrl.ppo_pettingzoo_ma_atari",
    # Policy optimization variants
    "ppg_procgen": "cleanrl.ppg_procgen",
    "pqn": "cleanrl.pqn",
    "pqn_atari_envpool": "cleanrl.pqn_atari_envpool",
    "pqn_atari_envpool_lstm": "cleanrl.pqn_atari_envpool_lstm",
    "rpo_continuous_action": "cleanrl.rpo_continuous_action",
    # Q-learning family
    "dqn": "cleanrl.dqn",
    "dqn_atari": "cleanrl.dqn_atari",
    "dqn_atari_jax": "cleanrl.dqn_atari_jax",
    "dqn_jax": "cleanrl.dqn_jax",
    "rainbow_atari": "cleanrl.rainbow_atari",
    "qdagger_dqn_atari_impalacnn": "cleanrl.qdagger_dqn_atari_impalacnn",
    "qdagger_dqn_atari_jax_impalacnn": "cleanrl.qdagger_dqn_atari_jax_impalacnn",
    "c51": "cleanrl.c51",
    "c51_jax": "cleanrl.c51_jax",
    "c51_atari": "cleanrl.c51_atari",
    "c51_atari_jax": "cleanrl.c51_atari_jax",
    # Continuous control (MuJoCo, etc.)
    "ddpg_continuous_action": "cleanrl.ddpg_continuous_action",
    "ddpg_continuous_action_jax": "cleanrl.ddpg_continuous_action_jax",
    "td3_continuous_action": "cleanrl.td3_continuous_action",
    "td3_continuous_action_jax": "cleanrl.td3_continuous_action_jax",
    "sac_continuous_action": "cleanrl.sac_continuous_action",
    "sac_atari": "cleanrl.sac_atari",
}


_RESERVED_EXTRA_KEYS: frozenset[str] = frozenset(
    {
        "tensorboard_dir",
        "track_wandb",
        "algo_params",
        "notes",
        "cuda",
        "use_cuda",
        "gpu",
        "wandb_run_name",
        "wandb_email",
        "wandb_api_key",
        "wandb_http_proxy",
        "wandb_https_proxy",
        "wandb_use_vpn_proxy",
        "agent_id",
        "fastlane_only",
        "fastlane_slot",
        "fastlane_video_mode",
        "fastlane_grid_limit",
        "fastlane_enabled",
        "mode",
        "policy_path",
        "checkpoint_path",
        "checkpoint_dir",
        "curriculum_schedule",
        "eval_capture_video",
        "eval_episodes",
        "eval_gamma",
        "eval_max_episode_steps",
        "eval_max_episode_seconds",
        "eval_batch_size",
        "eval_repeat",
    }
)


@dataclass(frozen=True)
class RuntimeSummary:
    """Lightweight summary returned from runtime dry-runs."""

    status: str
    module: str
    callable: str
    config: Dict[str, Any]
    extras: Dict[str, Any]


class CleanRLWorkerRuntime:
    """Resolve CleanRL algorithms and coordinate launch parameters."""

    def __init__(
        self,
        config: WorkerConfig,
        *,
        use_grpc: bool,
        grpc_target: str,
        dry_run: bool = False,
        algo_registry: AlgoRegistry = DEFAULT_ALGO_REGISTRY,
    ) -> None:
        self._config = config
        self._use_grpc = use_grpc
        self._grpc_target = grpc_target
        self._dry_run = dry_run
        self._algo_registry = dict(algo_registry)
        self._session_token: Optional[str] = None
        self._emitter = TelemetryEmitter(run_id=config.run_id, logger=LOGGER)

    @property
    def config(self) -> WorkerConfig:
        return self._config

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    def _allowed_entry_modules(self) -> set[str]:
        canonical = {str(value) for value in self._algo_registry.values()}
        aliased = {f"cleanrl_worker.{value}" for value in canonical}
        return canonical.union(aliased)

    def _assert_whitelisted_module(self, module_name: str) -> None:
        if not _MODULE_NAME_PATTERN.fullmatch(module_name):
            raise ValueError(f"Refusing to import unexpected module name '{module_name}'")
        if module_name not in self._allowed_entry_modules():
            raise ValueError(f"Module '{module_name}' is not registered in the CleanRL registry")

    def resolve_entrypoint(self) -> tuple[str, str]:
        """Resolve the module and callable implementing the requested algorithm."""

        canonical_module = self._algo_registry.get(self._config.algo)
        if canonical_module is None:
            raise ValueError(f"Algorithm '{self._config.algo}' is not registered")

        candidates = [canonical_module]
        # Fallback: try the other namespace if primary is missing
        if canonical_module.startswith("cleanrl_worker.algorithms."):
            candidates.append(canonical_module.replace("cleanrl_worker.algorithms.", "cleanrl."))
        elif canonical_module.startswith("cleanrl."):
            candidates.append(canonical_module.replace("cleanrl.", "cleanrl_worker.algorithms."))

        resolved_name: Optional[str] = None
        for candidate in candidates:
            try:
                spec = importlib.util.find_spec(candidate)
            except (ImportError, AttributeError):
                spec = None

            if spec is not None:
                resolved_name = candidate
                break

        if resolved_name is None:
            raise ModuleNotFoundError(
                f"Unable to locate CleanRL module for algorithm '{self._config.algo}' "
                f"(tried {', '.join(candidates)})"
            )

        self._assert_whitelisted_module(resolved_name)

        return resolved_name, f"{resolved_name}.main"

    def _import_entrypoint(self, module_name: str):
        """Import module and fetch its main entrypoint."""

        self._assert_whitelisted_module(module_name)
        module = importlib.import_module(module_name)
        entrypoint = getattr(module, "main", None)
        if entrypoint is None or not callable(entrypoint):
            LOGGER.warning(
                "Module %s does not expose a main() entrypoint; relying on CLI execution",
                module_name,
            )
            return None
        return entrypoint

    def build_cleanrl_args(self) -> list[str]:
        """Construct CLI arguments for the underlying CleanRL script."""

        args = [
            f"--env-id={self._config.env_id}",
            f"--total-timesteps={self._config.total_timesteps}",
        ]
        if self._config.seed is not None:
            args.append(f"--seed={self._config.seed}")

        extras: Mapping[str, Any] = self._config.extras
        wandb = extras.get("track_wandb")
        if isinstance(wandb, bool) and wandb:
            args.append("--track")

        cuda_flag = extras.get("cuda") or extras.get("use_cuda") or extras.get("gpu")
        if isinstance(cuda_flag, bool) and cuda_flag:
            args.append("--cuda")

        # Keys that appear in algo_params but must NOT be forwarded as CLI
        # flags (they are handled via environment variables or other means).
        _ALGO_PARAMS_SKIP = frozenset({"procedural_generation"})

        algo_params = extras.get("algo_params")
        if isinstance(algo_params, Mapping):
            for key, value in algo_params.items():
                if key in _ALGO_PARAMS_SKIP:
                    continue
                flag = f"--{key.replace('_', '-')}"
                if isinstance(value, bool):
                    if value:
                        args.append(flag)
                else:
                    args.append(f"{flag}={value}")

        for key, value in extras.items():
            if key in _RESERVED_EXTRA_KEYS:
                continue
            args.extend(self._format_cli_override(key, value))

        return args

    def _format_cli_override(self, key: str, value: Any) -> list[str]:
        flag = f"--{key.replace('_', '-')}"
        if value is None:
            return []
        if isinstance(value, bool):
            # tyro uses --flag / --no-flag (not --flag=true/false)
            return [flag if value else f"--no-{key.replace('_', '-')}"]
        if isinstance(value, (int, float)):
            return [f"{flag}={value}"]
        if isinstance(value, str):
            if not value.strip():
                return []
            return [f"{flag}={value}"]
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            formatted: list[str] = []
            for item in value:
                if isinstance(item, bool):
                    formatted.append(flag if item else f"--no-{key.replace('_', '-')}")
                elif isinstance(item, (int, float, str)):
                    formatted.append(f"{flag}={item}")
            return formatted
        return []

    def _prepare_summary(self, module_name: str) -> RuntimeSummary:
        return RuntimeSummary(
            status="dry-run" if self._dry_run else "pending",
            module=module_name,
            callable=f"{module_name}.main",
            config=asdict(self._config),
            extras=dict(self._config.extras),
        )

    def _register_with_trainer(self) -> None:
        """Perform RegisterWorker handshake directly with the trainer daemon."""

        if not self._use_grpc:
            return

        options = (
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
        )

        worker_id = self._config.worker_id or f"cleanrl-worker-{self._config.run_id[:8]}"
        schema_id = self._config.extras.get("schema_id") or f"cleanrl.{self._config.algo}"
        supports_pause = bool(self._config.extras.get("supports_pause", False))
        supports_checkpoint = bool(self._config.extras.get("supports_checkpoint", False))
        schema_version_raw = self._config.extras.get("schema_version", 1)
        try:
            schema_version = int(schema_version_raw)
        except (TypeError, ValueError):
            schema_version = 1

        # gRPC core only supports HTTP proxies (CONNECT). Some environments export
        # HTTPS_PROXY/https_proxy with an https:// URI, which leads to noisy
        # 'https scheme not supported in proxy URI' errors even for local targets.
        # Suppress HTTPS proxy variables during the handshake and ensure localhost
        # is in no_proxy to avoid proxy resolution altogether for the trainer.
        with _suppress_https_proxy_for_grpc():
            with grpc.insecure_channel(self._grpc_target, options=options) as channel:
                stub = trainer_pb2_grpc.TrainerServiceStub(channel)
                request = trainer_pb2.RegisterWorkerRequest(  # type: ignore[attr-defined]
                    run_id=self._config.run_id,
                    worker_id=worker_id,
                    worker_kind="cleanrl",
                    proto_version="MOSAIC/1.0",
                    schema_id=str(schema_id),
                    schema_version=schema_version,
                    supports_pause=supports_pause,
                    supports_checkpoint=supports_checkpoint,
                )
                try:
                    response = stub.RegisterWorker(request)
                except grpc.RpcError as exc:  # pragma: no cover - defensive logging
                    detail = exc.details() if hasattr(exc, "details") else str(exc)
                    code = exc.code().name if hasattr(exc, "code") else "UNKNOWN"
                    raise RuntimeError(
                        f"RegisterWorker handshake failed ({code}): {detail}"
                    ) from exc

        self._session_token = getattr(response, "session_token", None)

    def run(self) -> Dict[str, Any]:
        """Execute (or dry-run) the configured CleanRL algorithm.

        Implements the WorkerRuntime protocol.

        Returns:
            Dictionary containing execution results:
            - status: "completed" | "failed" | "cancelled"
            - module: Module name
            - callable: Callable name
            - config: Configuration dict
            - extras: Extra parameters
        """

        module_name, _ = self.resolve_entrypoint()
        if self._dry_run:
            summary = self._prepare_summary(module_name)
            return {
                "status": summary.status,
                "module": summary.module,
                "callable": summary.callable,
                "config": summary.config,
                "extras": summary.extras,
            }

        ensure_var_directories()
        # Custom scripts set MOSAIC_RUN_DIR to custom_scripts/{ULID}/.
        # Without this check, logs/tensorboard/checkpoints scatter to runs/
        # even though the script writes everything to custom_scripts/.
        mosaic_run_dir = os.environ.get("MOSAIC_RUN_DIR")
        if mosaic_run_dir:
            run_dir = Path(mosaic_run_dir).resolve()
        else:
            mode = str(dict(self._config.extras).get("mode") or "train")
            subdir = "evals" if mode == "policy_eval" else "runs"
            run_dir = (VAR_TRAINER_DIR / subdir / self._config.run_id).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Emit run_started lifecycle event
        self._emitter.run_started(
            {
                "worker_type": "cleanrl",
                "algo": self._config.algo,
                "env_id": self._config.env_id,
                "seed": self._config.seed,
                "total_timesteps": self._config.total_timesteps,
                "module": module_name,
            },
            constant=LOG_WORKER_CLEANRL_RUNTIME_STARTED,
        )

        try:
            self._import_entrypoint(module_name)
        except ModuleNotFoundError:
            if module_name.startswith(("cleanrl_worker.algorithms.", "cleanrl.")):
                LOGGER.debug(
                    "Deferring import of %s to launcher bootstrap (vendored CleanRL expected)",
                    module_name,
                )
            else:
                raise

        self._register_with_trainer()

        extras: Dict[str, Any] = dict(self._config.extras)
        mode = str(extras.get("mode") or "train")
        if mode == "policy_eval":
            return self._run_policy_eval(module_name, run_dir, extras)

        args = self.build_cleanrl_args()
        cmd = _sanitize_launch_command(
            [
                sys.executable,
                "-m",
                "cleanrl_worker.launcher",
                module_name,
                *args,
            ]
        )

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")

        pythonpath_entries = []
        # Use the parent of the cleanrl_worker package (not the package dir itself)
        # to avoid Python auto-importing sitecustomize.py in every forked process
        # (SyncVectorEnv workers). The launcher explicitly imports sitecustomize.
        package_parent = Path(__file__).resolve().parent.parent
        pythonpath_entries.append(str(package_parent))
        repo_path = str(REPO_ROOT)
        pythonpath_entries.append(repo_path)
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            pythonpath_entries.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
        tb_path: Optional[Path] = None
        tensorboard_dir = extras.get("tensorboard_dir")
        # Preserve the original relative dirname for analytics manifest
        tensorboard_dirname_relative: Optional[str] = None
        if isinstance(tensorboard_dir, str) and tensorboard_dir:
            tensorboard_dirname_relative = tensorboard_dir.strip()
            tb_path = (run_dir / tensorboard_dir).resolve()
            tb_path.mkdir(parents=True, exist_ok=True)
            extras["tensorboard_dir"] = str(tb_path)
            log_constant(
                LOGGER,
                LOG_WORKER_CLEANRL_TENSORBOARD_ENABLED,
                extra={
                    "run_id": self._config.run_id,
                    "tensorboard_dir": tensorboard_dirname_relative,
                },
            )
        else:
            extras.pop("tensorboard_dir", None)

        wandb_enabled = bool(extras.get("track_wandb"))
        wandb_root: Optional[Path] = None
        if wandb_enabled:
            wandb_root = (run_dir / "wandb").resolve()
            (wandb_root / "cache").mkdir(parents=True, exist_ok=True)
            (wandb_root / "config").mkdir(parents=True, exist_ok=True)
            (wandb_root / "logs").mkdir(parents=True, exist_ok=True)
            log_constant(
                LOGGER,
                LOG_WORKER_CLEANRL_WANDB_ENABLED,
                extra={
                    "run_id": self._config.run_id,
                    "wandb_root": str(wandb_root),
                },
            )

        stdout_path = logs_dir / "cleanrl.stdout.log"
        stderr_path = logs_dir / "cleanrl.stderr.log"

        capture_flag = extras.get("algo_params", {}).get("capture_video") if extras.get("algo_params") else None
        if capture_flag:
            # Only the CleanRL CLI flag controls recording; the global MOSAIC_CAPTURE_VIDEO
            # shim stays disabled to avoid duplicate video artifacts. We still ensure the
            # videos directory exists so downstream analytics can surface files written by
            # CleanRL (which uses cwd/videos/<run_name>/...).
            (run_dir / "videos").mkdir(parents=True, exist_ok=True)
        else:
            env.pop("MOSAIC_CAPTURE_VIDEO", None)
            env.pop("MOSAIC_VIDEOS_DIR", None)

        # FastLane environment variables for live video streaming during training
        # These must be set BEFORE the subprocess starts since fastlane.py reads them at import time
        #
        # IMPORTANT: Use setdefault() to preserve existing values from dispatcher/custom scripts.
        # Default to ENABLED ("1") so FastLane works out-of-the-box for standard training.
        algo_params = extras.get("algo_params", {}) if isinstance(extras, dict) else {}
        num_envs_from_config = int(algo_params.get("num_envs", 1))
        num_envs = max(1, int(env.get("CLEANRL_NUM_ENVS", num_envs_from_config)))
        grid_limit = max(1, min(int(extras.get("fastlane_grid_limit", num_envs)), num_envs))
        # Default FastLane to ENABLED - only disable if explicitly set to False
        env.setdefault(TelemetryEnv.FASTLANE_ONLY, "1")
        env.setdefault(TelemetryEnv.FASTLANE_SLOT, str(extras.get("fastlane_slot", 0)))
        env.setdefault(TelemetryEnv.FASTLANE_VIDEO_MODE, extras.get("fastlane_video_mode", VideoModes.SINGLE))
        env.setdefault(TelemetryEnv.FASTLANE_GRID_LIMIT, str(grid_limit))
        env.setdefault("CLEANRL_NUM_ENVS", str(num_envs))
        # CRITICAL: Preserve dispatcher's CLEANRL_RUN_ID for FastLane buffer name matching
        env.setdefault("CLEANRL_RUN_ID", self._config.run_id)
        if self._config.seed is not None:
            env.setdefault("CLEANRL_SEED", str(self._config.seed))

        if tensorboard_dir and isinstance(tensorboard_dir, str):
            env["CLEANRL_TENSORBOARD_DIR"] = str(tb_path)
        def _ensure_no_proxy(var: str) -> None:
            current = env.get(var, "")
            entries = [entry.strip() for entry in current.split(",") if entry.strip()]
            for target in ("127.0.0.1", "localhost"):
                if target not in entries:
                    entries.append(target)
            env[var] = ",".join(entries)

        _ensure_no_proxy("no_proxy")
        _ensure_no_proxy("NO_PROXY")

        if wandb_enabled and wandb_root is not None:
            env.setdefault("WANDB_DIR", str(wandb_root))
            env.setdefault("WANDB_CACHE_DIR", str(wandb_root / "cache"))
            env.setdefault("WANDB_CONFIG_DIR", str(wandb_root / "config"))
            env.setdefault("WANDB_NETRC_PATH", str(wandb_root / "netrc"))
            env.setdefault("WANDB_START_METHOD", "thread")
            env.setdefault("WANDB__SERVICE", "disabled")
            env.setdefault("WANDB_SKIP_SERVICE", "1")
            env.setdefault("WANDB_DISABLE_SERVICE", "true")
            env.setdefault("WANDB_DISABLE_GYM", "true")
            env.setdefault("WANDB_MODE", "online")

        # Resolve WANDB proxies, preferring VPN vars if present, then extras and generic fallbacks.
        http_proxy, https_proxy = _resolve_wandb_proxies(extras, os.environ)
        _apply_proxy_env(env, http_proxy, https_proxy)

        wandb_run_name = extras.get("wandb_run_name")
        if isinstance(wandb_run_name, str) and wandb_run_name:
            env.setdefault("WANDB_NAME", wandb_run_name)
            env.setdefault("WANDB_RUN_ID", wandb_run_name)

        wandb_email = extras.get("wandb_email")
        if isinstance(wandb_email, str) and wandb_email:
            env.setdefault("WANDB_EMAIL", wandb_email)

        wandb_entity = extras.get("wandb_entity")
        if isinstance(wandb_entity, str) and wandb_entity:
            env.setdefault("WANDB_ENTITY", wandb_entity)

        wandb_project = extras.get("wandb_project_name") or extras.get("wandb_project")
        if isinstance(wandb_project, str) and wandb_project:
            env.setdefault("WANDB_PROJECT", wandb_project)

        wandb_api_key = extras.get("wandb_api_key")
        if isinstance(wandb_api_key, str) and wandb_api_key:
            env.setdefault("WANDB_API_KEY", wandb_api_key)

        if wandb_enabled:
            if not _preflight_wandb_login(env):
                LOGGER.warning(
                    "WANDB preflight login failed; forcing offline mode for this run",
                )
                env.setdefault("WANDB_MODE", "offline")
        else:
            env.setdefault("WANDB_MODE", "offline")

        # Clean up orphaned /dev/shm segments from previous crashed runs
        _cleanup_orphaned_shm(run_id=self._config.run_id)

        with stdout_path.open("w", encoding="utf-8", buffering=1) as out, stderr_path.open(
            "w", encoding="utf-8", buffering=1
        ) as err:
            proc = subprocess.Popen(
                cmd,
                cwd=run_dir,
                stdout=out,
                stderr=err,
                env=env,
            )

            # Memory snapshot at launch time for OOM diagnosis
            _mem_snapshot = _get_memory_snapshot()
            log_constant(
                LOGGER,
                LOG_WORKER_CLEANRL_SUBPROCESS_MEMORY_SNAPSHOT,
                extra={
                    "run_id": self._config.run_id,
                    "pid": getattr(proc, "pid", None),
                    **_mem_snapshot,
                },
            )

            log_constant(
                LOGGER,
                LOG_WORKER_CLEANRL_SUBPROCESS_STARTED,
                extra={
                    "run_id": self._config.run_id,
                    "pid": getattr(proc, "pid", None),
                    "command": " ".join(cmd),
                },
            )

            last_heartbeat = time.monotonic()
            start_time = time.monotonic()
            heartbeat_interval = 30.0
            slow_start_warned = False
            while True:
                rc = proc.poll()
                if rc is not None:
                    return_code = rc
                    break
                now = time.monotonic()
                # Warn if subprocess hasn't produced output after 30s
                if not slow_start_warned and (now - start_time) >= 30.0:
                    if stdout_path.stat().st_size == 0:
                        log_constant(
                            LOGGER,
                            LOG_WORKER_CLEANRL_SUBPROCESS_SLOW_START,
                            extra={
                                "run_id": self._config.run_id,
                                "elapsed_seconds": int(now - start_time),
                                "algo": self._config.algo,
                                "env_id": self._config.env_id,
                            },
                        )
                        slow_start_warned = True
                if now - last_heartbeat >= heartbeat_interval:
                    self._emitter.heartbeat(
                        {
                            "status": "running",
                            "algo": self._config.algo,
                            "env_id": self._config.env_id,
                        },
                        constant=LOG_WORKER_CLEANRL_HEARTBEAT,
                    )
                    last_heartbeat = now
                time.sleep(1.0)

        # --- Subprocess exit diagnostics ---
        elapsed = time.monotonic() - start_time
        stdout_size = stdout_path.stat().st_size if stdout_path.exists() else 0
        stderr_size = stderr_path.stat().st_size if stderr_path.exists() else 0
        _exit_mem = _get_memory_snapshot()

        log_constant(
            LOGGER,
            LOG_WORKER_CLEANRL_SUBPROCESS_EXIT_SUMMARY,
            extra={
                "run_id": self._config.run_id,
                "return_code": return_code,
                "elapsed_seconds": round(elapsed, 1),
                "stdout_bytes": stdout_size,
                "stderr_bytes": stderr_size,
                "signal": -return_code if return_code < 0 else None,
                **_exit_mem,
            },
        )

        # Clean up orphaned shm after any abnormal exit
        if return_code != 0:
            _cleanup_orphaned_shm(run_id=self._config.run_id)

        # Detect OOM kill (SIGKILL = signal 9, exit code = -9 or 137)
        if return_code in (-9, 137):
            log_constant(
                LOGGER,
                LOG_WORKER_CLEANRL_SUBPROCESS_OOM_KILLED,
                extra={
                    "run_id": self._config.run_id,
                    "return_code": return_code,
                    "elapsed_seconds": round(elapsed, 1),
                    **_exit_mem,
                },
            )
        elif return_code < 0:
            import signal as _signal
            sig_name = "unknown"
            try:
                sig_name = _signal.Signals(-return_code).name
            except (ValueError, AttributeError):
                sig_name = f"signal({-return_code})"
            log_constant(
                LOGGER,
                LOG_WORKER_CLEANRL_SUBPROCESS_SIGNALED,
                extra={
                    "run_id": self._config.run_id,
                    "return_code": return_code,
                    "signal_name": sig_name,
                    "elapsed_seconds": round(elapsed, 1),
                },
            )

        # Detect empty stdout (training never produced output)
        if stdout_size == 0 and return_code != 0:
            log_constant(
                LOGGER,
                LOG_WORKER_CLEANRL_SUBPROCESS_NO_OUTPUT,
                extra={
                    "run_id": self._config.run_id,
                    "return_code": return_code,
                    "stderr_bytes": stderr_size,
                    "elapsed_seconds": round(elapsed, 1),
                },
            )

        # Detect stderr-only output (resource_tracker warnings, etc.)
        if stdout_size == 0 and stderr_size > 0 and return_code == 0:
            log_constant(
                LOGGER,
                LOG_WORKER_CLEANRL_SUBPROCESS_STDERR_ONLY,
                extra={
                    "run_id": self._config.run_id,
                    "stderr_bytes": stderr_size,
                },
            )

        if return_code != 0:
            # Read the last 30 lines of stderr to surface the actual error
            stderr_tail = ""
            try:
                raw = stderr_path.read_text(encoding="utf-8", errors="replace")
                lines = [ln for ln in raw.splitlines() if ln.strip()]
                # Skip noisy warnings, keep only lines that aren't pydantic/pygame/pkg_resources
                meaningful = [
                    ln for ln in lines
                    if not any(skip in ln for skip in (
                        "UnsupportedFieldAttributeWarning",
                        "warnings.warn(",
                        "pkg_resources is deprecated",
                        "resource_stream, resource_exists",
                        "resource_tracker:",
                    ))
                ]
                tail_lines = meaningful[-20:] if meaningful else lines[-20:]
                stderr_tail = "\n".join(tail_lines)
            except Exception:
                stderr_tail = f"(could not read {stderr_path})"

            error_msg = (
                f"CleanRL process exited with code {return_code}\n"
                f"--- stderr tail ---\n{stderr_tail}\n---"
            )
            LOGGER.error(error_msg)
            self._emitter.run_failed(
                {
                    "error": error_msg,
                    "error_type": "CalledProcessError",
                    "return_code": return_code,
                    "command": " ".join(cmd),
                    "stderr_log": str(stderr_path),
                },
                constant=LOG_WORKER_CLEANRL_RUNTIME_FAILED,
            )
            raise subprocess.CalledProcessError(return_code, cmd)

        # Build manifest with relative paths for portability
        manifest_extras = dict(extras)
        if tensorboard_dirname_relative:
            manifest_extras["tensorboard_dir"] = tensorboard_dirname_relative
        manifest = build_manifest(
            run_dir, extras=manifest_extras, run_id=self._config.run_id
        )
        manifest_path = run_dir / "analytics.json"
        manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")

        log_constant(
            LOGGER,
            LOG_WORKER_CLEANRL_ANALYTICS_MANIFEST_CREATED,
            extra={
                "run_id": self._config.run_id,
                "manifest_path": str(manifest_path),
            },
        )

        # Clean up any shm segments leaked by the subprocess
        _cleanup_orphaned_shm(run_id=self._config.run_id)

        # Emit run_completed lifecycle event
        self._emitter.run_completed(
            {
                "status": "completed",
                "module": module_name,
                "algo": self._config.algo,
                "env_id": self._config.env_id,
            },
            constant=LOG_WORKER_CLEANRL_RUNTIME_COMPLETED,
        )

        return {
            "status": "completed",
            "module": module_name,
            "callable": f"{module_name}.main",
            "config": asdict(self._config),
            "extras": extras,
        }

    # ---------------------------------------------------------------------------
    # Local helpers
    # ---------------------------------------------------------------------------
    def _run_policy_eval(
        self,
        module_name: str,
        run_dir: Path,
        extras: Dict[str, Any],
    ) -> Dict[str, Any]:
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        policy_path = extras.get("policy_path")
        if not policy_path:
            raise ValueError("Policy evaluation requested without policy_path extra")
        policy_file = Path(policy_path).expanduser()
        if not policy_file.exists():
            LOGGER.warning("Policy checkpoint missing: %s", policy_file)
            raise FileNotFoundError(policy_file)

        stdout_path = logs_dir / "cleanrl.stdout.log"
        stderr_path = logs_dir / "cleanrl.stderr.log"
        videos_dir = run_dir / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        tensorboard_dirname = "tensorboard"
        tensorboard_log_dir = run_dir / tensorboard_dirname
        # Create tensorboard directory early so TensorBoard tab can start watching it
        tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
        summary_path = run_dir / "eval_summary.json"
        manifest_path = run_dir / "analytics.json"
        collected_returns: list[float] = []

        entry = get_eval_entry(self._config.algo)
        
        if entry is not None:
            agent_cls = entry.agent_cls
            make_env = entry.make_env
            evaluate_fn = entry.evaluate
            accepts_gamma = entry.accepts_gamma
        else:
            module = importlib.import_module(module_name)
            evaluate_fn = self._resolve_eval_helper()
            agent_cls = getattr(module, "Agent", None)
            make_env = getattr(module, "make_env", None)
            accepts_gamma = True
            if evaluate_fn is None or agent_cls is None or make_env is None:
                raise RuntimeError(
                    f"Algorithm '{self._config.algo}' does not expose evaluation helpers"
                )

        capture_video = bool(extras.get("eval_capture_video"))
        eval_episodes = extras.get("eval_episodes", 5)
        try:
            eval_episodes = max(1, int(eval_episodes))
        except (TypeError, ValueError):
            eval_episodes = 5
        episodes_per_batch = extras.get("eval_batch_size", eval_episodes)
        try:
            episodes_per_batch = max(1, int(episodes_per_batch))
        except (TypeError, ValueError):
            episodes_per_batch = eval_episodes
        repeat_eval = bool(extras.get("eval_repeat"))
        device_name = "cuda" if extras.get("cuda") else "cpu"
        gamma = extras.get("algo_params", {}).get("gamma", 0.99)
        try:
            gamma = float(gamma)
        except (TypeError, ValueError):
            gamma = 0.99
        eval_gamma = extras.get("eval_gamma")
        if isinstance(eval_gamma, (int, float)):
            gamma = float(eval_gamma)

        LOGGER.info(
            "Policy evaluation starting | run_id=%s env=%s batch_episodes=%s repeat=%s",
            self._config.run_id,
            self._config.env_id,
            episodes_per_batch,
            repeat_eval,
        )

        try:
            import torch

            device = torch.device("cuda") if device_name == "cuda" else torch.device("cpu")
        except Exception:  # pragma: no cover - torch optional during tests
            device = device_name

        algo_params = extras.get("algo_params", {}) if isinstance(extras, dict) else {}

        def _coerce_positive(value, fallback):
            try:
                return max(1, int(value))
            except (TypeError, ValueError):
                return fallback

        num_envs = _coerce_positive(algo_params.get("num_envs"), 1)
        grid_limit_value = extras.get("fastlane_grid_limit", num_envs)
        grid_limit = _coerce_positive(grid_limit_value, num_envs)
        fastlane_env = {
            TelemetryEnv.FASTLANE_ONLY: "1" if extras.get("fastlane_only") else "0",
            TelemetryEnv.FASTLANE_SLOT: str(extras.get("fastlane_slot", 0)),
            TelemetryEnv.FASTLANE_VIDEO_MODE: extras.get("fastlane_video_mode", VideoModes.SINGLE),
            TelemetryEnv.FASTLANE_GRID_LIMIT: str(grid_limit),
        }
        core_env = {
            "CLEANRL_RUN_ID": self._config.run_id,
            "CLEANRL_AGENT_ID": extras.get("agent_id", "cleanrl_eval"),
            "CLEANRL_NUM_ENVS": str(num_envs),
        }
        env_updates = {
            "MOSAIC_VIDEOS_DIR": str(videos_dir),
            "MOSAIC_CAPTURE_VIDEO": "1" if capture_video else "0",
            **fastlane_env,
            **core_env,
        }
        max_steps_override = extras.get("eval_max_episode_steps")
        max_seconds_override = extras.get("eval_max_episode_seconds")

        def _write_manifest_payload(returns_snapshot: Sequence[float]) -> None:
            manifest = build_manifest(
                run_dir,
                extras={
                    "mode": "policy_eval",
                    "returns": list(returns_snapshot),
                    "tensorboard_dir": tensorboard_dirname,
                },
                run_id=self._config.run_id,
            )
            manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")

        _write_manifest_payload([])
        log_constant(
            LOGGER,
            LOG_WORKER_POLICY_EVAL_STARTED,
            extra={
                "run_id": self._config.run_id,
                "algo": self._config.algo,
                "env_id": self._config.env_id,
                "policy_path": str(policy_file),
                "batch_size": episodes_per_batch,
                "repeat": repeat_eval,
                "max_steps": max_steps_override,
                "max_seconds": max_seconds_override,
            },
        )

        def _record_batch(summary: EvalBatchSummary) -> None:
            collected_returns.extend(summary.returns)
            log_constant(
                LOGGER,
                LOG_WORKER_POLICY_EVAL_BATCH_COMPLETED,
                extra={
                    "run_id": self._config.run_id,
                    "batch_index": summary.index,
                    "episodes": summary.episodes,
                    "avg_return": summary.avg_return,
                    "min_return": summary.min_return,
                    "max_return": summary.max_return,
                    "duration_sec": summary.duration_sec,
                },
            )
            try:
                _write_manifest_payload(collected_returns)
            except Exception:  # pragma: no cover - manifest best effort
                pass
            self._emitter.heartbeat(
                {
                    "status": "policy_eval_batch",
                    "algo": self._config.algo,
                    "env_id": self._config.env_id,
                    "mode": "policy_eval",
                    "batch_index": summary.index,
                    "episodes": summary.episodes,
                    "avg_return": summary.avg_return,
                    "min_return": summary.min_return,
                    "max_return": summary.max_return,
                    "median_return": summary.median_return,
                    "std_return": summary.std_return,
                    "duration_sec": summary.duration_sec,
                },
                constant=LOG_WORKER_CLEANRL_HEARTBEAT,
            )

        def _record_batch_start(batch_index: int) -> None:
            log_constant(
                LOGGER,
                LOG_WORKER_POLICY_EVAL_BATCH_STARTED,
                extra={
                    "run_id": self._config.run_id,
                    "batch_index": batch_index,
                    "episodes": episodes_per_batch,
                },
            )

        returns: list[float] = []
        with stdout_path.open("w", encoding="utf-8", buffering=1) as out, stderr_path.open(
            "w", encoding="utf-8", buffering=1
        ) as err, redirect_stdout(out), redirect_stderr(err):
            with _temporary_environ(env_updates), _working_directory(run_dir):
                fastlane_module.reload_fastlane_config()
                try:
                    import cleanrl_worker.sitecustomize as sc  # noqa: WPS433

                    importlib.reload(sc)
                except Exception:  # pragma: no cover - defensive reload
                    pass

                target_gamma = gamma
                if accepts_gamma:
                    base_factory = make_env
                else:

                    def base_factory(env_id, idx, capture_video, run_name, _gamma):
                        return make_env(env_id, idx, capture_video, run_name)

                def adapted_make_env(env_id, idx, capture_video, run_name, _gamma_unused):
                    # Return a factory function that creates the wrapped environment.
                    # CleanRL's ppo_eval.py passes this to gym.vector.SyncVectorEnv which
                    # expects a callable that returns an env, not an env directly.
                    def env_factory():
                        # base_factory returns a factory function, so we call it twice:
                        # base_factory(...) -> returns factory function
                        # base_factory(...)() -> calls factory to create actual env
                        env = base_factory(env_id, idx, capture_video, run_name, target_gamma)()
                        if max_seconds_override is not None:
                            from gym_gui.core.wrappers.time_limits import EpisodeTimeLimitSeconds as TimeLimitSeconds

                            try:
                                env = TimeLimitSeconds(env, float(max_seconds_override))
                            except Exception:
                                pass
                        if max_steps_override is not None:
                            from gym_gui.core.wrappers.time_limits import configure_step_limit

                            try:
                                env = configure_step_limit(env, int(max_steps_override))
                            except Exception:
                                pass
                        env = _EvalStepLoggingWrapper(env, self._config.run_id, idx)
                        return env
                    return env_factory

                result: Optional[EvalRunResult] = run_batched_evaluation(
                    evaluate_fn,  # type: ignore[arg-type]
                    policy_path=str(policy_file),
                    make_env=adapted_make_env,
                    env_id=self._config.env_id,
                    agent_cls=agent_cls,
                    device=device,
                    capture_video=capture_video,
                    gamma=gamma,
                    episodes_per_batch=episodes_per_batch,
                    repeat=repeat_eval,
                    log_dir=tensorboard_log_dir,
                    summary_path=summary_path,
                    run_name_prefix=self._config.run_id,
                    on_batch=_record_batch,
                    on_batch_start=_record_batch_start,
                )
                if result is not None:
                    returns = self._coerce_return_values(result.returns)
                else:
                    returns = self._coerce_return_values(collected_returns)
        fastlane_module.reload_fastlane_config()

        avg = sum(returns) / len(returns) if returns else 0.0
        log_constant(
            LOGGER,
            LOG_WORKER_POLICY_EVAL_COMPLETED,
            extra={
                "run_id": self._config.run_id,
                "episodes": len(returns),
                "avg_return": avg,
            },
        )
        _write_manifest_payload(returns)

        self._emitter.run_completed(
            {
                "status": "completed",
                "algo": self._config.algo,
                "env_id": self._config.env_id,
                "mode": "policy_eval",
                "avg_return": sum(returns) / len(returns) if returns else 0.0,
                "total_episodes": len(returns),
            },
            constant=LOG_WORKER_CLEANRL_RUNTIME_COMPLETED,
        )

        return {
            "status": "completed",
            "module": module_name,
            "callable": "policy_eval",
            "config": asdict(self._config),
            "extras": {
                **extras,
                "evaluation_returns": returns,
                "tensorboard_dir": tensorboard_dirname,
                "eval_batch_size": episodes_per_batch,
                "eval_repeat": repeat_eval,
            },
        }

    def _resolve_eval_helper(self):
        algo = self._config.algo
        if algo not in self._algo_registry:
            return None
        module_candidates = [
            f"cleanrl_worker.cleanrl_utils.evals.{algo}_eval",
            f"cleanrl_utils.evals.{algo}_eval",
        ]
        if algo.startswith("ppo"):
            module_candidates.append("cleanrl_worker.cleanrl_utils.evals.ppo_eval")
            module_candidates.append("cleanrl_utils.evals.ppo_eval")
        allowed_candidates = {name for name in module_candidates if _MODULE_NAME_PATTERN.fullmatch(name)}
        for module_name in module_candidates:
            if module_name not in allowed_candidates:
                continue
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            evaluate = getattr(module, "evaluate", None)
            if callable(evaluate):
                return evaluate
        return None

    def _coerce_return_values(self, values: Any) -> list[float]:
        if isinstance(values, (int, float)):
            return [float(values)]
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
            coerced: list[float] = []
            for item in values:
                try:
                    coerced.append(float(item))
                except (TypeError, ValueError):
                    LOGGER.debug("Skipping non-numeric evaluation return: %s", item)
            return coerced
        return []

    def _write_eval_tensorboard(self, log_dir: Path, returns: Sequence[float], avg: float) -> None:
        if not returns or _TensorBoardWriter is None:
            return
        log_dir.mkdir(parents=True, exist_ok=True)
        try:
            writer = _TensorBoardWriter(log_dir=str(log_dir))
        except Exception:
            return
        with writer:
            for idx, value in enumerate(returns):
                writer.add_scalar("eval/episode_return", value, idx)
            writer.add_scalar("eval/avg_return", avg, len(returns))
            writer.flush()


@contextmanager
def _temporary_environ(overrides: Dict[str, str]) -> Iterator[None]:
    saved: dict[str, Optional[str]] = {k: os.environ.get(k) for k in overrides}
    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    current = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(current)


@contextmanager
def _suppress_https_proxy_for_grpc() -> Iterator[None]:
    """Temporarily remove HTTPS proxy env vars to prevent gRPC errors.

    gRPC only respects HTTP CONNECT proxies. If an https:// proxy is present in
    HTTPS_PROXY/https_proxy, gRPC will emit errors like:
    'https scheme not supported in proxy URI'. We temporarily remove these
    variables and ensure localhost is listed in NO_PROXY while establishing the
    channel/performing the handshake. Variables are restored afterwards.
    """

    keys = ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy")
    saved: dict[str, Optional[str]] = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            if k in os.environ:
                os.environ.pop(k, None)

        def _append_no_proxy(var: str) -> None:
            current = os.environ.get(var, "")
            entries = [e.strip() for e in current.split(",") if e.strip()]
            for host in ("localhost", "127.0.0.1", "::1"):
                if host not in entries:
                    entries.append(host)
            os.environ[var] = ",".join(entries)

        _append_no_proxy("NO_PROXY")
        _append_no_proxy("no_proxy")
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _resolve_wandb_proxies(
    extras: Mapping[str, Any], base_env: Mapping[str, str]
) -> tuple[Optional[str], Optional[str]]:
    """Resolve WANDB HTTP/HTTPS proxy values with sensible precedence.

    Precedence (first non-empty string wins):
    - Extras overrides: wandb_http_proxy / wandb_https_proxy
    - VPN-specific env vars: WANDB_VPN_HTTP_PROXY / WANDB_VPN_HTTPS_PROXY
    - WANDB-specific env vars: WANDB_HTTP_PROXY / WANDB_HTTPS_PROXY
    - Generic env vars: HTTP_PROXY/http_proxy and HTTPS_PROXY/https_proxy
    """

    def _pick_str(value: Any) -> Optional[str]:
        return value if isinstance(value, str) and value.strip() else None

    http_proxy = _pick_str(extras.get("wandb_http_proxy")) or _pick_str(
        base_env.get("WANDB_VPN_HTTP_PROXY")
    ) or _pick_str(base_env.get("WANDB_HTTP_PROXY")) or _pick_str(base_env.get("HTTP_PROXY")) or _pick_str(
        base_env.get("http_proxy")
    )

    https_proxy = _pick_str(extras.get("wandb_https_proxy")) or _pick_str(
        base_env.get("WANDB_VPN_HTTPS_PROXY")
    ) or _pick_str(base_env.get("WANDB_HTTPS_PROXY")) or _pick_str(base_env.get("HTTPS_PROXY")) or _pick_str(
        base_env.get("https_proxy")
    )

    return http_proxy, https_proxy


def _apply_proxy_env(env: Dict[str, str], http_proxy: Optional[str], https_proxy: Optional[str]) -> None:
    """Apply resolved proxies to child environment variables consistently."""
    if http_proxy:
        for key in ("WANDB_HTTP_PROXY", "HTTP_PROXY", "http_proxy"):
            env[key] = http_proxy
    if https_proxy:
        for key in ("WANDB_HTTPS_PROXY", "HTTPS_PROXY", "https_proxy"):
            env[key] = https_proxy


def _preflight_wandb_login(env: Mapping[str, str]) -> bool:
    """Attempt to authenticate with Weights & Biases using provided environment."""

    try:
        import wandb
    except Exception as exc:  # pragma: no cover - wandb optional
        LOGGER.warning("WANDB SDK unavailable during preflight: %s", exc)
        return False

    api_key = env.get("WANDB_API_KEY") or os.environ.get("WANDB_API_KEY")

    overlay = {k: v for k, v in env.items() if isinstance(v, str)}
    with _temporary_env(overlay):
        try:
            if api_key:
                result = wandb.login(key=api_key, relogin=True)
            else:
                result = wandb.login(relogin=True)
        except Exception as exc:  # pragma: no cover - network/proxy issues
            LOGGER.warning("WANDB login failed during preflight: %s", exc)
            return False
        finally:
            try:
                wandb.finish()
            except Exception:
                pass

    return bool(result)


@contextmanager
def _temporary_env(overrides: Mapping[str, str]) -> Iterator[None]:
    """Temporarily overlay environment variables for the duration of a block."""

    saved: dict[str, Optional[str]] = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            if isinstance(value, str):
                os.environ[key] = value
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _get_memory_snapshot() -> dict[str, Any]:
    """Capture system memory state for OOM diagnostics."""
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        with open("/proc/meminfo") as f:
            meminfo = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    meminfo[parts[0].rstrip(":")] = int(parts[1])
        ram_total_mb = meminfo.get("MemTotal", 0) // 1024
        ram_available_mb = meminfo.get("MemAvailable", 0) // 1024
        swap_total_mb = meminfo.get("SwapTotal", 0) // 1024
        swap_free_mb = meminfo.get("SwapFree", 0) // 1024
        return {
            "ram_total_mb": ram_total_mb,
            "ram_available_mb": ram_available_mb,
            "ram_used_pct": round(100 * (1 - ram_available_mb / max(ram_total_mb, 1)), 1),
            "swap_total_mb": swap_total_mb,
            "swap_free_mb": swap_free_mb,
            "swap_used_pct": round(100 * (1 - swap_free_mb / max(swap_total_mb, 1)), 1),
        }
    except Exception:
        return {"memory_snapshot": "unavailable"}


def _sanitize_launch_command(cmd: Sequence[Any]) -> list[str]:
    """Ensure CLI components are strings without control characters."""

    sanitized: list[str] = []
    for index, component in enumerate(cmd):
        if component is None:
            raise ValueError(f"Command component at index {index} is None")
        if not isinstance(component, str):
            component = str(component)
        if not _CMD_COMPONENT_PATTERN.fullmatch(component):
            raise ValueError(
                f"Command component at index {index} contains unsupported characters"
            )
        sanitized.append(component)
    return sanitized


# ===========================================================================
# Interactive Runtime for GUI step-by-step control
# ===========================================================================


class InteractiveRuntime:
    """Interactive runtime for step-by-step RL policy evaluation.

    Enables GUI-controlled stepping for scientific comparison with LLM operators.
    Follows the same IPC protocol as balrog_worker.InteractiveRuntime.

    Protocol:
        Input (stdin):
            {"cmd": "reset", "seed": 42}  - Reset environment with seed
            {"cmd": "step"}               - Execute one step using loaded policy
            {"cmd": "stop"}               - Terminate gracefully

        Output (stdout):
            {"type": "ready", "run_id": "...", "env_id": "...", "seed": 42}
            {"type": "step", "step_index": 0, "action": 2, "reward": 0.5, ...}
            {"type": "episode_done", "total_reward": 0.95, "episode_length": 15}
            {"type": "error", "message": "..."}
            {"type": "stopped"}
    """

    def __init__(self, config: CleanRLWorkerConfig):
        """Initialize interactive runtime.

        Args:
            config: Worker configuration with policy_path in extras.
        """
        self._config = config
        self._policy_path = config.extras.get("policy_path")
        self._env_id = config.env_id
        self._algo = config.algo

        # State (initialized on reset)
        self._env = None
        self._envs = None  # Vector env wrapper for CleanRL compatibility
        self._selector = None
        self._obs = None
        self._step_idx = 0
        self._episode_reward = 0.0
        self._episode_count = 0
        self._device = "cpu"
        self._reset_seed = None

        LOGGER.info(
            "InteractiveRuntime initialized | env=%s algo=%s policy=%s",
            self._env_id,
            self._algo,
            self._policy_path,
        )

    def _load_policy(self) -> None:
        """Load trained policy from checkpoint."""
        import torch
        import gymnasium as gym
        import numpy as np

        if not self._policy_path:
            raise ValueError("policy_path is required for interactive mode")

        policy_file = Path(self._policy_path).expanduser()
        if not policy_file.exists():
            raise FileNotFoundError(f"Policy checkpoint not found: {policy_file}")

        LOGGER.info("Loading policy from %s", policy_file)

        # Auto-import environment packages to register their environments
        if self._env_id.startswith("MiniGrid") or self._env_id.startswith("BabyAI"):
            try:
                import minigrid  # noqa: F401 - registers MiniGrid/BabyAI envs
                LOGGER.debug("Imported minigrid to register environments")
            except ImportError:
                LOGGER.warning("minigrid package not installed")

        # Create environment
        self._env = gym.make(self._env_id)

        # Apply MiniGrid observation wrappers if needed (Dict → image Box)
        # NOTE: do NOT apply FlattenObservation here — the MinigridAgent CNN
        # expects raw (7, 7, 3) image observations, not a flattened vector.
        if self._env_id.startswith("MiniGrid") or self._env_id.startswith("BabyAI"):
            try:
                from minigrid.wrappers import ImgObsWrapper
                self._env = ImgObsWrapper(self._env)
                LOGGER.debug("Applied MiniGrid ImgObsWrapper")
            except ImportError:
                pass

        self._env = gym.wrappers.RecordEpisodeStatistics(self._env)

        # Detect architecture from checkpoint BEFORE creating the vector env,
        # because the MLP agent needs FlattenObservation while CNN does not.
        import torch as _torch
        _ckpt = _torch.load(str(policy_file), map_location="cpu", weights_only=True)
        _state = _ckpt.get("model_state_dict", _ckpt) if isinstance(_ckpt, dict) else _ckpt
        _has_cnn = any(k.startswith("cnn.") for k in _state.keys()) if hasattr(_state, "keys") else False

        # Wrap in vector env for compatibility with CleanRL models
        env_id = self._env_id
        is_minigrid = env_id.startswith("MiniGrid") or env_id.startswith("BabyAI")
        needs_flatten = is_minigrid and not _has_cnn

        def make_env():
            env = gym.make(env_id, render_mode="rgb_array")  # Enable RGB rendering
            # Apply MiniGrid wrappers if needed
            if is_minigrid:
                try:
                    from minigrid.wrappers import ImgObsWrapper
                    env = ImgObsWrapper(env)
                except ImportError:
                    pass
                # MLP-based checkpoints were trained on flattened (7,7,3) → 147-dim
                if needs_flatten:
                    env = gym.wrappers.FlattenObservation(env)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            return env

        self._envs = gym.vector.SyncVectorEnv([make_env])

        # Get eval entry for agent class, and adapter for action selection
        from .unified_eval.registry import get_adapter
        eval_entry = get_eval_entry(self._algo)
        if eval_entry is None:
            raise ValueError(f"No evaluation entry for algorithm: {self._algo}")

        adapter = get_adapter(self._algo)
        if adapter is None:
            raise ValueError(f"No unified adapter for algorithm: {self._algo}")

        # Select agent class matching the checkpoint architecture
        agent_cls = eval_entry.agent_cls
        if is_minigrid and _has_cnn:
            from .agents.minigrid import MinigridAgent
            agent_cls = MinigridAgent
            LOGGER.info("Detected CNN-based MinigridAgent checkpoint")
        elif is_minigrid:
            LOGGER.info("Detected MLP-based checkpoint for MiniGrid env (flattened obs)")

        # Load the model using adapter
        adapter.load(
            str(policy_file),
            self._envs,
            self._device,
            agent_cls=agent_cls,
        )
        self._selector = adapter

        LOGGER.info("Policy loaded successfully")

    def _handle_reset(self, seed: Optional[int] = None) -> None:
        """Handle reset command - initialize environment with seed."""
        try:
            # Load policy on first reset
            if self._selector is None:
                self._load_policy()

            # Reset environment
            assert self._envs is not None, "Environment not initialized"
            self._obs, info = self._envs.reset(seed=seed)
            self._reset_seed = seed  # Store for deterministic auto-reset
            self._step_idx = 0
            self._episode_reward = 0.0

            # Get initial render frame
            render_payload = None
            try:
                frame = self._envs.call("render")
                if frame is not None and len(frame) > 0:
                    rgb_frame = frame[0]
                    if rgb_frame is not None and hasattr(rgb_frame, 'shape'):
                        render_payload = {
                            "mode": "rgb_array",
                            "rgb": rgb_frame.tolist() if hasattr(rgb_frame, 'tolist') else rgb_frame,
                            "width": int(rgb_frame.shape[1]),
                            "height": int(rgb_frame.shape[0]),
                        }
            except Exception:
                pass

            ready_response = {
                "type": "ready",
                "run_id": self._config.run_id,
                "env_id": self._env_id,
                "algo": self._algo,
                "seed": seed,
                "observation_shape": list(self._obs.shape) if hasattr(self._obs, 'shape') else None,
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
        if self._selector is None or self._obs is None:
            self._emit({"type": "error", "message": "Environment not initialized. Send reset first."})
            return

        try:
            import numpy as np

            # Get action from policy
            action = self._selector.select_action(self._obs)

            # Unwrap from vector env format if needed
            if hasattr(action, '__len__') and len(action) == 1:
                action_scalar = int(action[0])
            else:
                action_scalar = int(action)

            # Step environment
            assert self._envs is not None, "Environment not initialized"
            obs_new, reward, terminated, truncated, info = self._envs.step(action)

            # Handle reward (may be array from vector env)
            if hasattr(reward, '__len__'):
                reward_scalar = float(reward[0])
            else:
                reward_scalar = float(reward)

            # Handle terminated/truncated
            if hasattr(terminated, '__len__'):
                term = bool(terminated[0])
                trunc = bool(truncated[0])
            else:
                term = bool(terminated)
                trunc = bool(truncated)

            done = term or trunc

            self._episode_reward += reward_scalar
            self._step_idx += 1

            # Save telemetry values BEFORE episode-end handling modifies them
            tel_step = self._step_idx
            tel_episode = self._episode_count
            tel_reward = self._episode_reward

            # Handle episode end BEFORE render capture so the seeded re-reset
            # is reflected in the render frame (SyncVectorEnv auto-resets
            # without a seed, which randomises MiniGrid layouts).
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
                # SyncVectorEnv auto-resets without a seed.  For MiniGrid/BabyAI
                # envs the seed determines the grid layout, so we must re-reset
                # with the original seed to preserve deterministic layouts.
                is_minigrid = (
                    self._env_id.startswith("MiniGrid")
                    or self._env_id.startswith("BabyAI")
                )
                if is_minigrid and self._reset_seed is not None and self._envs is not None:
                    obs_new, _ = self._envs.reset(seed=self._reset_seed)
                    log_constant(
                        LOGGER,
                        LOG_WORKER_CLEANRL_EPISODE_AUTO_RESET,
                        extra={
                            "episode_index": self._episode_count,
                            "seed": self._reset_seed,
                            "env_id": self._env_id,
                        },
                    )
                # Reset counters for next episode
                self._step_idx = 0
                self._episode_reward = 0.0

            # Capture render AFTER potential seeded re-reset so the frame
            # shows the correctly-seeded layout (not the random auto-reset).
            render_payload = None
            try:
                frame = self._envs.call("render") if self._envs is not None else None
                if frame is not None and len(frame) > 0:
                    rgb_frame = frame[0]
                    if rgb_frame is not None and hasattr(rgb_frame, 'shape'):
                        render_payload = {
                            "mode": "rgb_array",
                            "rgb": rgb_frame.tolist() if hasattr(rgb_frame, 'tolist') else rgb_frame,
                            "width": int(rgb_frame.shape[1]),
                            "height": int(rgb_frame.shape[0]),
                        }
            except Exception:
                pass  # Rendering not available

            # Emit step telemetry using saved values (pre-reset counters)
            step_data = {
                "type": "step",
                "step_index": tel_step,
                "episode_index": tel_episode,
                "action": action_scalar,
                "reward": reward_scalar,
                "terminated": term,
                "truncated": trunc,
                "episode_reward": tel_reward,
            }

            if render_payload is not None:
                step_data["render_payload"] = render_payload

            self._emit(step_data)

            self._obs = obs_new

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
            "algo": self._algo,
            "policy_path": self._policy_path,
            "version": "1.0",
        })

        LOGGER.info("Interactive runtime started, waiting for commands...")

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

        # Cleanup
        if self._envs is not None:
            try:
                self._envs.close()
            except Exception:
                pass

        LOGGER.info("Interactive runtime stopped")
