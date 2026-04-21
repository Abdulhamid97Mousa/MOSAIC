#!/usr/bin/env bash
# ==============================================================================
# MOSAIC — Server
# Run this on the compute / GPU node (Hamid@a1-R84228-G11).
# The GUI client connects from another machine:
#   MOSAIC_DAEMON_TARGET=<server-ip>:50055 ./run_client.sh
#
# Usage:
#   ./run_server.sh                                         # all interfaces, port 50055
#   DAEMON_PORT=50056 ./run_server.sh                       # custom port
#   DAEMON_HOST=127.0.0.1 ./run_server.sh                   # loopback only (local test)
#   MOSAIC_START_VLLM=1 MOSAIC_VLLM_MODEL=<path> ./run_server.sh   # with vLLM
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi

mkdir -p var/logs var/trainer

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
DAEMON_HOST="${DAEMON_HOST:-0.0.0.0}"
DAEMON_PORT="${DAEMON_PORT:-50055}"
LISTEN="${DAEMON_HOST}:${DAEMON_PORT}"

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

echo "Starting MOSAIC server on ${LISTEN} ..."

# ------------------------------------------------------------------------------
# Optional: launch vLLM inference server alongside the daemon
# Set MOSAIC_START_VLLM=1 and MOSAIC_VLLM_MODEL=<model_path> to enable.
# ------------------------------------------------------------------------------
if [[ "${MOSAIC_START_VLLM:-0}" == "1" ]]; then
    VLLM_MODEL="${MOSAIC_VLLM_MODEL:-}"
    VLLM_PORT="${MOSAIC_VLLM_PORT:-8000}"
    if [ -z "$VLLM_MODEL" ]; then
        echo "WARNING: MOSAIC_START_VLLM=1 but MOSAIC_VLLM_MODEL is not set — skipping vLLM startup."
    else
        echo "Starting vLLM on 0.0.0.0:${VLLM_PORT} with model: ${VLLM_MODEL}"
        nohup vllm serve "$VLLM_MODEL" \
            --host 0.0.0.0 \
            --port "$VLLM_PORT" \
            --gpu-memory-utilization "${MOSAIC_VLLM_GPU_UTIL:-0.85}" \
            --enforce-eager \
            > var/logs/vllm_server_${VLLM_PORT}.log 2>&1 &
        echo "vLLM started (PID: $!), logs: var/logs/vllm_server_${VLLM_PORT}.log"
    fi
fi

# ------------------------------------------------------------------------------
# Start trainer daemon (foreground — use screen/tmux for persistence)
# ------------------------------------------------------------------------------
"$PYTHON_BIN" -m gym_gui.services.trainer_daemon \
    --listen "$LISTEN" \
    2>&1 | tee var/logs/trainer_daemon.log
