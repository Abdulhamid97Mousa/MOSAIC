#!/usr/bin/env bash
# ==============================================================================
# MOSAIC — Single-Machine Launch
# Starts the server (trainer daemon) and client (GUI) on the same machine.
#
# For split client-server deployments use:
#   Server (Hamid@a1-R84228-G11):  bash run_server.sh
#   Client (hamid@HamidOnUbuntu):  MOSAIC_DAEMON_TARGET=<ip>:50055 bash run_client.sh
#
# Usage:
#   ./run.sh
#   PYTHON_BIN=python3.12 ./run.sh
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi

mkdir -p var/logs var/trainer

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
DAEMON_ADDR="127.0.0.1:50055"

# Warn if the user has a remote target set — run.sh always uses a local server
if [[ -n "${MOSAIC_DAEMON_TARGET:-}" && "${MOSAIC_DAEMON_TARGET}" != "127.0.0.1:50055" ]]; then
    echo "NOTE: MOSAIC_DAEMON_TARGET=${MOSAIC_DAEMON_TARGET} is set but run.sh always starts a local server."
    echo "      For a remote server use: MOSAIC_DAEMON_TARGET=${MOSAIC_DAEMON_TARGET} bash run_client.sh"
fi

# ------------------------------------------------------------------------------
# Kill any existing trainer daemon
# ------------------------------------------------------------------------------
echo "Checking for existing trainer processes..."
EXISTING_PIDS="$(pgrep -f 'gym_gui.services.trainer_daemon' || true)"
if [ -n "$EXISTING_PIDS" ]; then
    echo "Stopping existing daemon..."
    for pid in $EXISTING_PIDS; do
        kill "$pid" 2>/dev/null || true
    done
    sleep 1
    REMAINING="$(pgrep -f 'gym_gui.services.trainer_daemon' || true)"
    [ -n "$REMAINING" ] && kill -9 $REMAINING 2>/dev/null || true
fi
rm -f var/trainer/trainer.pid 2>/dev/null || true

# ------------------------------------------------------------------------------
# Start server in background
# ------------------------------------------------------------------------------
echo "Starting MOSAIC server..."
QT_DEBUG_PLUGINS=0 \
    "$PYTHON_BIN" -m gym_gui.services.trainer_daemon \
    > var/logs/trainer_daemon.log 2>&1 &
DAEMON_PID=$!

# ------------------------------------------------------------------------------
# Wait for server to be ready (up to 10 s)
# ------------------------------------------------------------------------------
echo "Waiting for server..."
for attempt in {1..10}; do
    if "$PYTHON_BIN" -c "
import asyncio, grpc
async def check():
    ch = grpc.aio.insecure_channel('${DAEMON_ADDR}')
    try:
        await asyncio.wait_for(ch.channel_ready(), timeout=2.0)
    finally:
        await ch.close()
asyncio.run(check())
" 2>/dev/null; then
        echo "Server ready."
        break
    fi
    if [ "$attempt" -eq 10 ]; then
        echo "ERROR: Server failed to start. Check var/logs/trainer_daemon.log" >&2
        echo "Fix: bash tools/generate_protos.sh && ./run.sh" >&2
        exit 1
    fi
    sleep 1
done

# ------------------------------------------------------------------------------
# Launch client (foreground — server is killed when this exits)
# ------------------------------------------------------------------------------
echo "Launching MOSAIC..."
MOSAIC_DAEMON_TARGET="${DAEMON_ADDR}" \
QT_API=PyQt6 QT_DEBUG_PLUGINS=0 \
    "$PYTHON_BIN" -m gym_gui.app

# Shut down server when GUI exits
kill "$DAEMON_PID" 2>/dev/null || true
