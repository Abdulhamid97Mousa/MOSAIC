from __future__ import annotations

"""Application entry-point helpers for manual smoke-testing."""

# CRITICAL: Set Qt API BEFORE any other imports that might use Qt
import os

os.environ.setdefault("QT_API", "PyQt6")

# Suppress noisy deprecation warnings from 3rd party dependencies
import warnings

warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", message=".*Matplotlib is not installed.*")
# Suppress mosaic_multigrid deprecation warnings (old gym API)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym.utils.seeding")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym.core")

import asyncio
import contextlib
import errno
import json
import logging
import sys
from dataclasses import replace
from functools import partial
from typing import Any

from gym_gui.config.settings import Settings, get_settings
from gym_gui.logging_config.helpers import log_constant
from gym_gui.logging_config.log_constants import (
    LOG_RUNTIME_APP_DEBUG,
    LOG_RUNTIME_APP_INFO,
    LOG_RUNTIME_APP_WARNING,
)
from gym_gui.logging_config.logger import configure_logging

# NOTE: bootstrap_default_services and TrainerDaemonHandle are imported inside main()
# to ensure Qt API is set before any Qt imports happen


LOGGER = logging.getLogger("gym_gui.app")
_log = partial(log_constant, LOGGER)



def _format_settings(settings: Settings) -> str:
    """Format settings for display, including system information."""
    import torch

    # Core settings
    payload: dict[str, Any] = {
        "qt_api": settings.qt_api,
        "log_level": settings.log_level,
        "default_control_mode": settings.default_control_mode.value,
        "default_seed": settings.default_seed,
        "allow_seed_reuse": settings.allow_seed_reuse,
    }

    # UI settings
    payload["ui"] = {
        "chat_panel_collapsed": settings.chat_panel_collapsed,
    }

    # vLLM settings
    payload["vllm"] = {
        "max_servers": settings.vllm_max_servers,
        "gpu_memory_utilization": settings.vllm_gpu_memory_utilization,
    }

    # System information (CPU, RAM)
    payload["system"] = _get_system_info()

    # CUDA/GPU information
    cuda_info: dict[str, Any] = {
        "available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        cuda_info["device_count"] = torch.cuda.device_count()
        cuda_info["current_device"] = torch.cuda.current_device()
        cuda_info["device_name"] = torch.cuda.get_device_name(0)
        # Get GPU memory info
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            cuda_info["memory_total_gb"] = round(int(info.total) / (1024**3), 1)
            cuda_info["memory_free_gb"] = round(int(info.free) / (1024**3), 1)
            cuda_info["memory_used_gb"] = round(int(info.used) / (1024**3), 1)
            pynvml.nvmlShutdown()
        except Exception:
            pass
    payload["cuda"] = cuda_info

    # Display / Qt platform information
    display_info: dict[str, Any] = {
        "DISPLAY": os.getenv("DISPLAY", "not set"),
        "WAYLAND_DISPLAY": os.getenv("WAYLAND_DISPLAY", "not set"),
        "XDG_RUNTIME_DIR": os.getenv("XDG_RUNTIME_DIR", "not set"),
        "QT_QPA_PLATFORM": os.getenv("QT_QPA_PLATFORM", "auto"),
    }
    # Detect WSLg
    display_info["wslg_available"] = os.path.exists("/mnt/wslg/.X11-unix")
    # Check X11 socket
    display_info["x11_socket_exists"] = os.path.exists("/tmp/.X11-unix/X0")
    payload["display"] = display_info

    # LLM API configuration
    def _mask_key(key: str | None) -> str:
        if not key:
            return "not set"
        return f"{key[:12]}...{key[-4:]}" if len(key) > 20 else "***"

    payload["llm_apis"] = {
        "openrouter_api_key": _mask_key(os.getenv("OPENROUTER_API_KEY")),
        "vllm_base_url": os.getenv("VLLM_BASE_URL", "not set"),
        "hf_token": _mask_key(os.getenv("HF_TOKEN")),
    }

    # Environment variables (important ones)
    payload["env"] = {
        "MUJOCO_GL": os.getenv("MUJOCO_GL", "not set"),
        "QT_DEBUG_PLUGINS": os.getenv("QT_DEBUG_PLUGINS", "not set"),
        "PLATFORM": os.getenv("PLATFORM", "not set"),
        "MPI4PY_RC_INITIALIZE": os.getenv("MPI4PY_RC_INITIALIZE", "not set"),
    }

    # Optional dependencies status
    payload["optional_deps"] = _detect_optional_dependencies()

    # Protobuf/gRPC status
    payload["protobuf"] = _check_protobuf_status()

    return json.dumps(payload, indent=2)


def _get_system_info() -> dict[str, Any]:
    """Get system information (CPU, RAM)."""
    import platform

    import psutil

    info: dict[str, Any] = {}

    # CPU information
    try:
        info["cpu_model"] = platform.processor() or "Unknown"
        info["cpu_cores_physical"] = psutil.cpu_count(logical=False)
        info["cpu_cores_logical"] = psutil.cpu_count(logical=True)
        info["cpu_freq_max_mhz"] = round(psutil.cpu_freq().max) if psutil.cpu_freq() else "Unknown"
    except Exception:
        info["cpu_model"] = "Unknown"

    # RAM information
    try:
        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / (1024**3), 1)
        info["ram_available_gb"] = round(mem.available / (1024**3), 1)
        info["ram_used_gb"] = round(mem.used / (1024**3), 1)
        info["ram_percent_used"] = round(mem.percent, 1)
    except Exception:
        info["ram_total_gb"] = "Unknown"

    return info


def _detect_optional_dependencies() -> dict[str, bool]:
    """Detect which optional dependency groups are installed.

    Uses ``importlib.util.find_spec()`` to probe package availability
    **without** actually importing them.  This is critical because some
    packages execute blocking code at import time:

    * **xuance** -- ``xuance/common/statistic_tools.py`` does
      ``from mpi4py import MPI`` at module level.  OpenMPI's ``MPI_Init()``
      blocks forever when not launched via ``mpirun``.
      **Solution:** Set ``MPI4PY_RC_INITIALIZE=0`` in ``.env`` (already
      configured) and use ``find_spec()`` here so the module is never
      loaded just for availability checks.

    * **ray** -- Imports tensorflow, pydantic, wandb at module level which
      adds several seconds of startup time.

    If you add a new worker/dependency here and it hangs on import, use
    ``find_spec()`` (not ``__import__()``) -- it only checks whether the
    package *can* be found on ``sys.path`` without executing any code.

    Note: Some packages have multiple versions (e.g., mosaic_multigrid vs
    multigrid-ini).  This detection shows which core packages are available.
    """
    import importlib.util

    deps: dict[str, bool] = {}

    # Check for key packages that indicate installed optional deps.
    # IMPORTANT: We use find_spec() to avoid executing module code.
    # Never replace this with __import__() -- it will block on packages
    # like xuance (mpi4py MPI_Init hang) or add multi-second delays.
    checks = {
        # Chat/LLM support (OpenRouter + vLLM)
        "chat": "openai",  # Note: Uses OpenAI-compatible API for OpenRouter/vLLM

        # Environment families
        "minigrid": "minigrid",
        "babyai": "minigrid.envs.babyai",  # BabyAI (bundled in minigrid>=2)
        "mosaic_multigrid": "mosaic_multigrid",  # Modern fork (Gymnasium API)
        "malmoenv": "malmoenv",  # MalmoEnv: Microsoft Malmo Java-based Minecraft (requires running Minecraft server)
        "multigrid_ini": "multigrid",  # Original INI version (cooperative exploration)
        "pettingzoo": "pettingzoo",
        "mujoco": "mujoco",
        "atari": "ale_py",  # ALE = Arcade Learning Environment
        "vizdoom": "vizdoom",
        "crafter": "crafter",
        "nethack": "nle",  # NLE = NetHack Learning Environment (different from ALE!)
        "procgen": "procgen",  # Procgen procedurally generated benchmark
        "textworld": "textworld",  # TextWorld text-based game environment
        "babaisai": "baba_is_ai",  # BabaIsAI rule manipulation puzzle benchmark
        "griddly": "griddly",  # Griddly C++ backend grid world platform
        "jumanji": "jumanji",  # Jumanji JAX-based RL environments
        ## "pybullet_drones": "gym_pybullet_drones",  # PyBullet Drones quadcopter environments
        "openspiel": "pyspiel",  # OpenSpiel board games via Shimmy
        "meltingpot": "meltingpot",  # Melting Pot multi-agent social scenarios
        ## "neuralmmo": "nmmo",  # Neural MMO massively multiagent game environment
        ## "hemac": "hemac",  # HeMAC Heterogeneous Multi-Agent Challenge

        # Overcooked versions
        "overcooked_ai": "overcooked_ai_py",  # Original UC Berkeley version

        # SMAC: StarCraft Multi-Agent Challenge
        "smac": "smac",           # SMAC v1: hand-designed cooperative micromanagement maps
        "smacv2": "smacv2",       # SMACv2: procedural unit generation
        "rware": "rware",         # RWARE: Robotic Warehouse multi-agent cooperative

        # RL training frameworks
        "ray_worker": "ray",
        "cleanrl_worker": "cleanrl",
        "xuance_worker": "xuance",
        "tianshou_worker": "tianshou",
        "sb3_worker": "stable_baselines3",
        "sbx_worker": "sbx.version",
        "mushroomrl_worker": "mushroom_rl",
        "pearl_worker": "pearl",
        "torchrl_worker": "torchrl",
        "omnisafe_worker": "omnisafe",
        "mctx_worker": "mctx",
        "marllib_worker": "marllib",
        "chess_worker": "chess_worker",

        # MOSAIC native workers
        "llm_worker": "llm_worker",
        "vlm_worker": "vlm_worker",
        "human_worker": "human_worker",
        "passive_worker": "passive_worker",
        "random_worker": "random_worker",
    }

    for dep_name, package_name in checks.items():
        found = importlib.util.find_spec(package_name) is not None
        deps[dep_name] = found
        if not found:
            LOGGER.debug(
                "optional dependency '%s' (package '%s') not found",
                dep_name,
                package_name,
            )

    _log(
        LOG_RUNTIME_APP_DEBUG,
        message="optional_deps_detected",
        extra={
            "method": "find_spec",
            "found": sum(deps.values()),
            "total": len(deps),
        },
    )

    return deps


def _check_protobuf_status() -> dict[str, Any]:
    """Check if protobuf files are compiled and up-to-date."""
    from pathlib import Path
    project_root = Path(__file__).parent.parent

    # Protobuf files are in gym_gui/services/trainer/proto/
    trainer_proto_dir = project_root / "gym_gui" / "services" / "trainer" / "proto"

    status: dict[str, Any] = {
        "compiled": False,
        "location": str(trainer_proto_dir.relative_to(project_root)) if trainer_proto_dir.exists() else "not found",
    }

    if trainer_proto_dir.exists():
        # Check if compiled _pb2.py files exist
        pb2_files = list(trainer_proto_dir.glob("*_pb2.py"))
        pb2_grpc_files = list(trainer_proto_dir.glob("*_pb2_grpc.py"))

        status["compiled"] = len(pb2_files) > 0
        status["pb2_files_count"] = len(pb2_files)
        status["grpc_files_count"] = len(pb2_grpc_files)

        if not status["compiled"]:
            status["note"] = "Run tools/generate_protos.sh to compile protobuf files"
    else:
        status["note"] = "Trainer proto directory missing"

    return status



@contextlib.contextmanager
def _suppress_annoying_warnings():
    """Suppress known noisy output from 3rd party libraries."""
    class StreamFilter:
        def __init__(self, stream):
            self.stream = stream

        def write(self, data):
            # Suppress empty lines (from Qt and other library initialization)
            if data.strip() == "":
                return len(data)  # Pretend we wrote it
            # Suppress TF-Keras import failure messages (from tensorflow_probability)
            if "Failed to import TF-Keras" in data:
                return
            if "Please note that TF-Keras is not installed" in data:
                return
            if "tensorflow-probability" in data and "extra" in data:
                return
            # Suppress Gym legacy warnings (if printed directly to stderr)
            if "Gym has been unmaintained" in data:
                return
            # Suppress specific Gym warnings about replacing 'gym' with 'gymnasium'
            if "replace 'import gym' with 'import gymnasium as gym'" in data:
                return
            self.stream.write(data)

        def flush(self):
            self.stream.flush()

        def __getattr__(self, name):
            return getattr(self.stream, name)

    # Suppress logging from noisy libraries
    import logging
    loggers_to_silence = ["tensorflow", "absl"]
    old_levels = {}
    for name in loggers_to_silence:
        logger = logging.getLogger(name)
        old_levels[name] = logger.level
        logger.setLevel(logging.ERROR)

    original_stderr = sys.stderr
    original_stdout = sys.stdout
    sys.stderr = StreamFilter(original_stderr)
    sys.stdout = StreamFilter(original_stdout)

    try:
        yield
    finally:
        sys.stderr = original_stderr
        sys.stdout = original_stdout
        for name, level in old_levels.items():
            logging.getLogger(name).setLevel(level)


def main() -> int:
    """Print the currently loaded settings and, if Qt is installed, show a stub window."""

    os.environ.setdefault("QT_DEBUG_PLUGINS", "0")

    settings = replace(get_settings(), default_seed=1)
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    configure_logging(level=log_level, stream=False, log_to_file=True)
    _log(
        LOG_RUNTIME_APP_DEBUG,
        message="settings_loaded",
        extra={"qt_api": settings.qt_api, "default_env": settings.gym_default_env},
    )
    print("[gym_gui] Loaded settings:\n" + _format_settings(settings))

    # Warn if MPI auto-initialisation is enabled -- this will cause the
    # XuanCe worker to hang on import (OpenMPI's MPI_Init blocks forever
    # outside an mpirun context).
    mpi_init_val = os.getenv("MPI4PY_RC_INITIALIZE", "0")
    if mpi_init_val != "0":
        _log(
            LOG_RUNTIME_APP_WARNING,
            message="mpi4py_auto_init_enabled",
            extra={"MPI4PY_RC_INITIALIZE": mpi_init_val},
        )
        print(
            "[gym_gui] WARNING: MPI4PY_RC_INITIALIZE is set to "
            f"'{mpi_init_val}' (expected '0').\n"
            "  XuanCe worker may hang on import because OpenMPI's MPI_Init()\n"
            "  blocks forever outside an MPI launch context (mpirun/mpiexec).\n"
            "  Fix: set MPI4PY_RC_INITIALIZE=0 in .env"
        )

    try:
        with _suppress_annoying_warnings():
            from qtpy.QtWidgets import QApplication, QMessageBox

            from gym_gui.ui.main_window import MainWindow
    except ImportError as exc:  # pragma: no cover - optional dependency
        print("[gym_gui] Qt bindings not available. Install qtpy and a Qt backend (PyQt5/PyQt6/PySide2/PySide6):", exc)
        return 0
    except SyntaxError as exc:  # pragma: no cover - syntax errors in imported modules
        import traceback
        print("[gym_gui] Syntax error detected while loading modules:")
        print(f"  File: {exc.filename}")
        print(f"  Line {exc.lineno}: {exc.text.strip() if exc.text else 'N/A'}")
        print(f"  Error: {exc.msg}")
        print("\nThis is likely a malformed string in a BabyAI or MiniGrid documentation file.")
        print("Check for missing quotes, unclosed strings, or invalid characters.")
        print("\nFull traceback:")
        traceback.print_exc()
        return 0
    except Exception as exc:  # pragma: no cover - optional dependency
        import traceback
        print("[gym_gui] Unexpected error loading modules:", exc)
        print("\nFull traceback:")
        traceback.print_exc()
        return 0

    # Share OpenGL contexts across QQuickWidget instances (must be set
    # *before* QApplication is created).  On WSLg the default QRhiGles2
    # backend may fail to create a context; QSG_RHI_BACKEND=software in
    # .env sidesteps this, but the attribute is still required for any
    # GL-based fallback to work.
    from PyQt6.QtCore import Qt
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

    app = QApplication(sys.argv)
    app.setApplicationName("Gym GUI")

    # Set the MOSAIC logo as the taskbar icon (read via _NET_WM_ICON on X11).
    from pathlib import Path

    from PyQt6.QtGui import QIcon
    _logo = Path(__file__).parent / "assets" / "mosaic_logo.png"
    if _logo.exists():
        app.setWindowIcon(QIcon(str(_logo)))

    print(f"[gym_gui] Qt platform plugin: {app.platformName()}")

    # Setup Qt-compatible asyncio event loop using qasync
    _setup_qasync_event_loop(app)

    # Import bootstrap AFTER Qt is initialized
    from gym_gui.services.trainer.launcher import TrainerDaemonHandle, TrainerDaemonLaunchError
    try:
        with _suppress_annoying_warnings():
            from gym_gui.services.bootstrap import bootstrap_default_services

            locator = bootstrap_default_services()
    except TrainerDaemonLaunchError as exc:
        QMessageBox.critical(None, "Trainer Daemon Error", str(exc))
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        QMessageBox.critical(None, "Bootstrap Error", str(exc))
        return 1

    daemon_handle: TrainerDaemonHandle | None = locator.resolve(TrainerDaemonHandle) if locator else None

    window = MainWindow(settings)
    window.show()

    if daemon_handle:
        status_bar = window.statusBar()
        if status_bar is not None:
            if daemon_handle.reused:
                status_bar.showMessage("Connected to existing trainer daemon", 5000)
            else:
                status_bar.showMessage("Trainer daemon started automatically", 5000)

        def _stop_daemon() -> None:
            daemon_handle.stop()

        app.aboutToQuit.connect(_stop_daemon)

    return app.exec()


def _setup_qasync_event_loop(app: Any) -> None:
    """Setup Qt-compatible asyncio event loop using qasync.

    This allows asyncio and Qt to share the same event loop, preventing
    conflicts between the two event loop systems.
    """
    try:
        from qasync import QEventLoop
    except ImportError:
        _log(
            LOG_RUNTIME_APP_WARNING,
            message="qasync_missing_fallback",
        )
        _install_asyncio_exception_handler()
        return

    # Create Qt-compatible event loop
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    # Install exception handler
    def _handler(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        exc = context.get("exception")
        if isinstance(exc, BlockingIOError) and getattr(exc, "errno", None) in {errno.EAGAIN, errno.EWOULDBLOCK}:
            _log(
                LOG_RUNTIME_APP_DEBUG,
                message="grpc_blocking_io_ignored",
                extra={"source": "qasync", "errno": getattr(exc, "errno", None)},
            )
            return
        loop.default_exception_handler(context)

    loop.set_exception_handler(_handler)
    _log(
        LOG_RUNTIME_APP_INFO,
        message="qasync_event_loop_initialized",
    )


def _install_asyncio_exception_handler() -> None:
    """Fallback: Install exception handler for separate asyncio loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    def _handler(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        exc = context.get("exception")
        if isinstance(exc, BlockingIOError) and getattr(exc, "errno", None) in {errno.EAGAIN, errno.EWOULDBLOCK}:
            _log(
                LOG_RUNTIME_APP_DEBUG,
                message="grpc_blocking_io_ignored",
                extra={"source": "asyncio", "errno": getattr(exc, "errno", None)},
            )
            return
        loop.default_exception_handler(context)

    loop.set_exception_handler(_handler)


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    raise SystemExit(main())
