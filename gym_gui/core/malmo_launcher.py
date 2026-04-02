"""Auto-launch Minecraft client processes for MalmoEnv environments.

When the adapter's ``load()`` method detects that Minecraft is not running on the
required ports, it calls ``ensure_minecraft_running()`` which launches the
appropriate number of headless Minecraft clients and waits for them to become
ready (listening on their assigned ports).

Multi-agent missions (e.g. TreasureHunt) require one Minecraft client per agent,
each on a consecutive port starting from the base port (default 9000).

Example:
    # For a 2-agent mission on ports 9000, 9001:
    processes = ensure_minecraft_running(
        agent_count=2,
        base_port=9000,
        progress_callback=lambda msg: print(msg),
    )
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Callable

_LOGGER = logging.getLogger(__name__)

# Path to the headless launch script
_ROOT_DIR = Path(__file__).parents[1].parent
_HEADLESS_SCRIPT = _ROOT_DIR / "3rd_party" / "environments" / "mosaic_malmo" / "launch_malmo_headless.sh"
_UPSTREAM_MINECRAFT_DIR = _ROOT_DIR / "3rd_party" / "environments" / "malmo_updated" / "Minecraft"

# How long to wait for each client to become ready
_DEFAULT_TIMEOUT_S = 300  # 5 minutes
_POLL_INTERVAL_S = 3


def _port_has_listener(port: int) -> bool:
    """Check if something is listening on the given TCP port."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        result = sock.connect_ex(("127.0.0.1", port))
        sock.close()
        return result == 0
    except Exception:
        return False


# Port spacing: each Minecraft client uses one MalmoEnv port.
# NativeInputHandler now uses malmoPort + 1000 (no conflict).
# Clients use consecutive ports: agent0=9000, agent1=9001, agent2=9002.
PORTS_PER_CLIENT = 1


def _find_java8() -> str | None:
    """Find Java 8 JDK path, required for Gradle 2.14 / Minecraft 1.11.2."""
    candidates = [
        "/usr/lib/jvm/java-8-openjdk-amd64",
        "/usr/lib/jvm/java-1.8.0-openjdk-amd64",
        "/usr/lib/jvm/java-8-openjdk",
    ]
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, "bin", "java")):
            return candidate
    return None


def _launch_one_client(
    port: int,
    *,
    env_override: dict[str, str] | None = None,
) -> subprocess.Popen:
    """Launch a single Minecraft client in the background on the given port.

    Uses the headless launch script (xvfb-run wrapper) or falls back to
    direct xvfb-run + launchClient.sh.  The ``launchClient.sh`` script
    writes the port config then runs ``./gradlew build && ./gradlew runClient``.

    Important: only ONE launch at a time from the same Minecraft directory.
    The caller must wait for each client to be ready before launching the next
    to avoid Gradle lock contention.
    """
    env = dict(os.environ)
    java8 = _find_java8()
    if java8:
        env["JAVA_HOME"] = java8
        env["PATH"] = f"{java8}/bin:{env.get('PATH', '')}"
    if env_override:
        env.update(env_override)

    if _HEADLESS_SCRIPT.is_file():
        cmd = ["bash", str(_HEADLESS_SCRIPT), "-port", str(port), "-env"]
    else:
        launch_sh = _UPSTREAM_MINECRAFT_DIR / "launchClient.sh"
        if not launch_sh.is_file():
            raise FileNotFoundError(
                f"Cannot find Minecraft launch script at:\n"
                f"  {_HEADLESS_SCRIPT}\n"
                f"  {launch_sh}\n\n"
                f"Run ./setup_malmo.sh first."
            )
        xvfb_args = env.get(
            "MALMO_XVFB_ARGS",
            "-screen 0 1280x720x24 -ac +extension GLX +render -noreset",
        )
        cmd = [
            "xvfb-run",
            "--auto-servernum",
            f"--server-args={xvfb_args}",
            "bash",
            str(launch_sh),
            "-port",
            str(port),
            "-env",
        ]

    _LOGGER.info("Launching Minecraft client on port %d: %s", port, " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        env=env,
        start_new_session=True,
    )
    return proc


# Track launched processes so we can clean them up
_managed_processes: list[subprocess.Popen] = []


def agent_port(base_port: int, role: int) -> int:
    """Return the MalmoEnv service port for a given agent role.

    Each Minecraft client occupies 2 ports:
      - port N:   MalmoEnv service (TCP)
      - port N+1: NativeInputHandler (hardcoded in Java as malmoPort + 1)

    So clients are spaced ``PORTS_PER_CLIENT`` apart::

        Agent 0 → MalmoEnv 9000, NativeInput 9001
        Agent 1 → MalmoEnv 9002, NativeInput 9003
        Agent 2 → MalmoEnv 9004, NativeInput 9005
    """
    return base_port + role * PORTS_PER_CLIENT


def ensure_minecraft_running(
    agent_count: int = 1,
    base_port: int = 9000,
    timeout_s: int = _DEFAULT_TIMEOUT_S,
    progress_callback: Callable[[str], None] | None = None,
) -> list[subprocess.Popen]:
    """Ensure Minecraft clients are running on the required ports.

    For each agent (0 .. agent_count-1), checks if the MalmoEnv service is
    listening on its port.  If not, launches a headless Minecraft client.
    Then polls until all ports are ready or timeout is reached.

    Port assignment uses ``PORTS_PER_CLIENT`` spacing (default 2) to avoid
    conflict between the MalmoEnv service and the NativeInputHandler which
    is hardcoded to malmoPort+1 in the Java mod.

    Args:
        agent_count: Number of agents (and therefore Minecraft clients).
        base_port: The MalmoEnv base port (agent 0).
        timeout_s: Maximum seconds to wait for all clients to be ready.
        progress_callback: Optional function called with status messages.

    Returns:
        List of launched ``Popen`` processes (may be empty if all were
        already running).

    Raises:
        RuntimeError: If clients fail to start within the timeout.
    """

    def _report(msg: str) -> None:
        _LOGGER.info(msg)
        if progress_callback:
            progress_callback(msg)

    ports = [agent_port(base_port, role) for role in range(agent_count)]
    launched: list[subprocess.Popen] = []

    _report(
        f"Multi-agent Minecraft: {agent_count} client(s) needed, "
        f"ports: {ports}"
    )

    # Launch clients SEQUENTIALLY — each must be fully ready before the next
    # starts, because launchClient.sh runs ./gradlew build which takes a
    # Gradle lock on the shared Minecraft directory.
    for port in ports:
        if _port_has_listener(port):
            _report(f"Port {port}: Minecraft already running")
            continue

        _report(f"Port {port}: Launching Minecraft client...")
        proc = _launch_one_client(port)
        launched.append(proc)
        _managed_processes.append(proc)

        # Wait for THIS client to be ready before launching the next
        _report(f"Waiting for port {port} to become ready (timeout: {timeout_s}s)...")
        start = time.monotonic()
        ready = False
        while (time.monotonic() - start) < timeout_s:
            time.sleep(_POLL_INTERVAL_S)
            if _port_has_listener(port):
                _report(f"Port {port}: Minecraft is ready")
                ready = True
                break
            elapsed = int(time.monotonic() - start)
            if elapsed % 15 < _POLL_INTERVAL_S:
                _report(f"Still waiting for port {port}... ({elapsed}s / {timeout_s}s)")

        if not ready:
            # Cleanup all launched processes
            for p in launched:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                except Exception:
                    pass
            raise RuntimeError(
                f"Minecraft client failed to start on port {port} "
                f"within {timeout_s}s.\n\n"
                f"Check that Java 8 is installed and the Malmo mod is built:\n"
                f"  ./setup_malmo.sh"
            )

    _report(f"All {agent_count} Minecraft client(s) are ready")
    return launched


def shutdown_managed_processes() -> None:
    """Terminate all Minecraft processes launched by this module."""
    global _managed_processes
    for proc in _managed_processes:
        if proc.poll() is None:  # Still running
            _LOGGER.info("Terminating Minecraft process PID %d", proc.pid)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass
    _managed_processes = []
