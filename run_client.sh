#!/usr/bin/env bash
# ==============================================================================
# MOSAIC — Client
# Run this on any machine with a display (hamid@HamidOnUbuntu).
# Set MOSAIC_DAEMON_TARGET to point at the server (default: 127.0.0.1:50055).
#
# Usage:
#   ./run_client.sh                                          # connect to local server
#   MOSAIC_DAEMON_TARGET=192.168.0.4:50055 ./run_client.sh  # connect to remote server
#
# The server must be running before launching the client:
#   Local server:   ./run_server.sh  (on this machine)
#   Remote server:  ssh Hamid@a1-R84228-G11 'cd ~/projects/mosaic && bash run_server.sh'
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi

mkdir -p var/logs

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
DAEMON_TARGET="${MOSAIC_DAEMON_TARGET:-127.0.0.1:50055}"
DAEMON_HOST="${DAEMON_TARGET%%:*}"

echo "Connecting to MOSAIC server at ${DAEMON_TARGET} ..."

# ------------------------------------------------------------------------------
# Verify the server is reachable before opening the GUI
# ------------------------------------------------------------------------------
if ! "$PYTHON_BIN" -c "
import asyncio, grpc, sys
async def check():
    ch = grpc.aio.insecure_channel('${DAEMON_TARGET}')
    try:
        await asyncio.wait_for(ch.channel_ready(), timeout=5.0)
        print('Server reachable at ${DAEMON_TARGET}')
    except Exception as e:
        print(f'ERROR: Cannot reach server at ${DAEMON_TARGET}: {e}', file=sys.stderr)
        sys.exit(1)
    finally:
        await ch.close()
asyncio.run(check())
"; then
    if [[ "$DAEMON_HOST" == "127.0.0.1" || "$DAEMON_HOST" == "localhost" || "$DAEMON_HOST" == "::1" ]]; then
        echo "Hint: start the local server first:" >&2
        echo "  bash run_server.sh" >&2
        echo "Common fix if proto errors: bash tools/generate_protos.sh && bash run_server.sh" >&2
    else
        echo "Hint: start the remote server on ${DAEMON_HOST}:" >&2
        echo "  ssh Hamid@${DAEMON_HOST} 'cd ~/projects/mosaic && bash run_server.sh'" >&2
    fi
    exit 1
fi

# ------------------------------------------------------------------------------
# Launch MOSAIC GUI
# ------------------------------------------------------------------------------
echo "Launching MOSAIC..."
QT_API=PyQt6 QT_DEBUG_PLUGINS=0 \
    MOSAIC_DAEMON_TARGET="${DAEMON_TARGET}" \
    "$PYTHON_BIN" -m gym_gui.app
