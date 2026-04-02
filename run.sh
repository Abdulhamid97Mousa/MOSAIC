#!/usr/bin/env bash
# ==============================================================================
# MOSAIC - Launch Script
# Multi-Agent Orchestration System with Adaptive Intelligent Control
# ==============================================================================
#
# Usage:
#   ./run.sh              # Launch MOSAIC with trainer daemon
#   PYTHON_BIN=python3.11 ./run.sh   # Use specific Python version
#
# Prerequisites:
#   pip install -e .                                # Install MOSAIC
#   pip install -e 3rd_party/workers/cleanrl_worker # Install CleanRL worker (optional)
#
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# ------------------------------------------------------------------------------
# Load environment variables
# ------------------------------------------------------------------------------
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# ------------------------------------------------------------------------------
# Activate virtual environment if available
# ------------------------------------------------------------------------------
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

# ------------------------------------------------------------------------------
# Setup directories
# ------------------------------------------------------------------------------
mkdir -p var/logs var/trainer

# ------------------------------------------------------------------------------
# Cleanup handler — kills daemon, worker subprocesses, and orphan trackers
# ------------------------------------------------------------------------------
# cleanup() {
#     echo ""
#     # 1. Kill trainer daemon
#     if ps -p "${DAEMON_PID:-0}" >/dev/null 2>&1; then
#         echo "Shutting down trainer daemon (PID: $DAEMON_PID)..."
#         kill "$DAEMON_PID" 2>/dev/null || true
#         wait "$DAEMON_PID" 2>/dev/null || true
#     fi

#     # 2. Kill any orphaned worker subprocesses spawned by the trainer
#     local worker_pids
#     worker_pids="$(pgrep -f 'cleanrl_worker\.(launcher|train)' 2>/dev/null || true)"
#     if [ -n "$worker_pids" ]; then
#         echo "Cleaning up orphaned worker processes..."
#         kill $worker_pids 2>/dev/null || true
#         sleep 0.5
#         kill -9 $worker_pids 2>/dev/null || true
#     fi

#     # 3. Kill runaway resource_tracker processes spawned from our venv
#     local tracker_pids
#     tracker_pids="$(pgrep -f 'resource_tracker.*main' 2>/dev/null || true)"
#     if [ -n "$tracker_pids" ]; then
#         echo "Cleaning up orphaned resource_tracker processes..."
#         kill -9 $tracker_pids 2>/dev/null || true
#     fi

#     # 4. Clean up stale shared memory segments left by FastLane
#     for shm in /dev/shm/mosaic.fastlane.*; do
#         [ -e "$shm" ] && rm -f "$shm" 2>/dev/null && echo "Removed stale shm: $shm"
#     done
# }
# trap cleanup EXIT

# ------------------------------------------------------------------------------
# Python binary selection
# ------------------------------------------------------------------------------
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

# ------------------------------------------------------------------------------
# Kill any existing trainer daemon processes
# ------------------------------------------------------------------------------
echo "Checking for existing trainer processes..."
EXISTING_PIDS="$(pgrep -f 'gym_gui.services.trainer_daemon' || true)"
if [ -n "$EXISTING_PIDS" ]; then
    echo "Killing existing trainer daemon processes..."
    for pid in $EXISTING_PIDS; do
        echo "  Terminating PID: $pid"
        kill "$pid" 2>/dev/null || true
    done
    # Give processes time to terminate gracefully
    sleep 1
    # Force kill any remaining
    REMAINING_PIDS="$(pgrep -f 'gym_gui.services.trainer_daemon' || true)"
    if [ -n "$REMAINING_PIDS" ]; then
        echo "Force killing remaining processes..."
        for pid in $REMAINING_PIDS; do
            kill -9 "$pid" 2>/dev/null || true
        done
        sleep 0.5
    fi
    echo "Existing trainer processes terminated."
fi

rm -f var/trainer/trainer.pid 2>/dev/null || true

# ------------------------------------------------------------------------------
# Start trainer daemon
# ------------------------------------------------------------------------------
echo "Starting trainer daemon..."
QT_DEBUG_PLUGINS=0 \
    "$PYTHON_BIN" -m gym_gui.services.trainer_daemon > var/logs/trainer_daemon.log 2>&1 &
DAEMON_PID=$!

# ------------------------------------------------------------------------------
# Wait for daemon to be ready
# ------------------------------------------------------------------------------
echo "Waiting for trainer daemon..."
for attempt in {1..10}; do
    if "$PYTHON_BIN" - <<'PY'
import asyncio
import grpc

async def check():
    channel = grpc.aio.insecure_channel("127.0.0.1:50055")
    try:
        await asyncio.wait_for(channel.channel_ready(), timeout=2.0)
        return True
    except Exception:
        return False
    finally:
        await channel.close()

if not asyncio.run(check()):
    raise SystemExit(1)
PY
    then
        echo "Trainer daemon ready."
        break
    fi
    if [ "$attempt" -eq 10 ]; then
        echo "ERROR: Trainer daemon failed to start. Check var/logs/trainer_daemon.log" >&2
        echo "" >&2
        echo "Common fix: Regenerate protobuf stubs to match your environment:" >&2
        echo "  source .venv/bin/activate" >&2
        echo "  bash tools/generate_protos.sh" >&2
        echo "  ./run.sh" >&2
        exit 1
    fi
    sleep 1
done

# ------------------------------------------------------------------------------
# Launch MOSAIC
# ------------------------------------------------------------------------------
echo "Launching MOSAIC..."
QT_API=PyQt6 QT_DEBUG_PLUGINS=0 "$PYTHON_BIN" -m gym_gui.app
