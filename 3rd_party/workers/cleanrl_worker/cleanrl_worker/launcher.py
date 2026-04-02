"""Launch CleanRL modules with vendored dependency bootstrapping."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path


def _bootstrap() -> None:
    """Ensure cleanrl, cleanrl_utils and MOSAIC site customizations are loaded.

    The sitecustomize module patches gymnasium wrappers (TransformObservation),
    TensorBoard log redirection, wandb defaults, and checkpoint resume support.
    It must be loaded before any cleanrl module is imported.
    """
    try:
        import cleanrl_worker.sitecustomize  # noqa: F401
    except Exception:
        pass


def _run_subprocess(module_name: str, module_args: list[str]) -> int:
    cmd = [sys.executable, "-m", module_name, *module_args]
    env = None
    # Ensure the child process auto-loads sitecustomize.py at startup.
    # Python loads sitecustomize.py from directories on PYTHONPATH.
    site_dir = str(Path(__file__).resolve().parent)
    current = os.environ.get("PYTHONPATH", "")
    if site_dir not in current.split(os.pathsep):
        env = os.environ.copy()
        env["PYTHONPATH"] = site_dir + os.pathsep + current if current else site_dir
    return subprocess.call(cmd, env=env)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m cleanrl_worker.launcher <module> [args...]")

    module_name = sys.argv[1]
    module_args = sys.argv[2:]

    _bootstrap()

    # Prefer an explicit main() if the module exposes one.
    # We differentiate between "module/main not found" (fallback to subprocess)
    # and "main() crashed" (let exception propagate).
    try:
        module = importlib.import_module(module_name)
        entry = getattr(module, "main", None)
    except ImportError:
        # Module not importable - fallback to subprocess
        entry = None
    except Exception:
        # Other errors during import (e.g. syntax error in module) - fallback
        entry = None

    if callable(entry):
        # Found main() - run it directly.
        # Do NOT catch exceptions here; if main() crashes, we want the stack trace.
        sys.argv = [module_name, *module_args]
        entry()
        return

    # Fallback for scripts without main() or import failures
    raise SystemExit(_run_subprocess(module_name, module_args))


if __name__ == "__main__":  # pragma: no cover
    main()
