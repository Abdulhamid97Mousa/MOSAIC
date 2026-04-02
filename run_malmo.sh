#!/usr/bin/env bash
# ==============================================================================
# MOSAIC Malmo — Launch Minecraft headless for MalmoEnv environments
# ==============================================================================
#
# Usage:
#   ./run_malmo.sh                       # Single agent (port 9000)
#   ./run_malmo.sh --agents 2            # Two agents (ports 9000 + 9002)
#   ./run_malmo.sh --visible             # Visible window (debugging)
#
# Multi-agent missions (e.g. TreasureHunt) require one Minecraft client per
# agent.  Each client uses 2 ports (MalmoEnv + NativeInputHandler), spaced 2
# apart:
#   Agent 0 → port 9000 (MalmoEnv) + 9001 (NativeInput)
#   Agent 1 → port 9002 (MalmoEnv) + 9003 (NativeInput)
#
# Wait for ALL clients to show "CLIENT enter state: DORMANT", then:
#   ./run.sh
#
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MALMO_UPDATED_DIR="$ROOT_DIR/3rd_party/environments/malmo_updated"
MINECRAFT_DIR="$MALMO_UPDATED_DIR/Minecraft"
HEADLESS_SCRIPT="$ROOT_DIR/3rd_party/environments/mosaic_malmo/launch_malmo_headless.sh"

# Each client uses one MalmoEnv port (consecutive).
# NativeInputHandler now uses malmoPort + 1000 (no conflict).
PORTS_PER_CLIENT=1

# Parse arguments
PORT=9000
VISIBLE=0
AGENTS=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        -port)       PORT="$2"; shift 2 ;;
        --agents|-agents) AGENTS="$2"; shift 2 ;;
        --visible)   VISIBLE=1; shift ;;
        -h|--help)
            echo "Usage: $0 [-port PORT] [--agents N] [--visible]"
            echo ""
            echo "  -port PORT    Base MalmoEnv TCP port (default: 9000)"
            echo "  --agents N    Number of agents/clients (default: 1)"
            echo "  --visible     Show Minecraft window (default: headless)"
            echo ""
            echo "Multi-agent example (TreasureHunt):"
            echo "  ./run_malmo.sh --agents 2"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --------------------------------------------------------------------------
# Kill ALL existing Minecraft/Java/Gradle processes on relevant ports
# --------------------------------------------------------------------------
echo "Cleaning up existing Minecraft processes..."

# Kill by port
for (( i=0; i<AGENTS; i++ )); do
    malmo_port=$((PORT + i * PORTS_PER_CLIENT))
    native_port=$((malmo_port + 1000))
    for check_port in "$malmo_port" "$native_port"; do
        PIDS="$(lsof -ti ":$check_port" 2>/dev/null || true)"
        if [ -n "$PIDS" ]; then
            echo "  Killing processes on port $check_port: $PIDS"
            kill -9 $PIDS 2>/dev/null || true
        fi
    done
done

# Also kill any stray GradleStart/Malmo Java processes
STRAY_PIDS="$(pgrep -f 'GradleStart|malmomod' 2>/dev/null || true)"
if [ -n "$STRAY_PIDS" ]; then
    echo "  Killing stray Minecraft/Gradle processes: $STRAY_PIDS"
    kill -9 $STRAY_PIDS 2>/dev/null || true
fi

sleep 2
echo "Ports are clean."

# --------------------------------------------------------------------------
# Check prerequisites
# --------------------------------------------------------------------------
if [ ! -f "$MINECRAFT_DIR/build/libs/MalmoMod-0.37.0.jar" ]; then
    echo "ERROR: Malmo mod not built. Run: ./setup_malmo.sh"
    exit 1
fi

# --------------------------------------------------------------------------
# Find Java 8
# --------------------------------------------------------------------------
JAVA8_HOME=""
for candidate in /usr/lib/jvm/java-8-openjdk-amd64 /usr/lib/jvm/java-1.8.0-openjdk-amd64 /usr/lib/jvm/java-8-openjdk; do
    if [ -x "$candidate/bin/java" ]; then
        JAVA8_HOME="$candidate"
        break
    fi
done
if [ -z "$JAVA8_HOME" ]; then
    echo "ERROR: Java 8 not found. Run: sudo apt install openjdk-8-jdk"
    exit 1
fi
export JAVA_HOME="$JAVA8_HOME"
export PATH="$JAVA8_HOME/bin:$PATH"

# --------------------------------------------------------------------------
# Set MALMO_XSD_PATH
# --------------------------------------------------------------------------
if [ -z "${MALMO_XSD_PATH:-}" ]; then
    for schema_dir in "$ROOT_DIR/3rd_party/environments/malmo/Schemas" "$MALMO_UPDATED_DIR/Schemas"; do
        if [ -d "$schema_dir" ]; then
            export MALMO_XSD_PATH="$schema_dir"
            break
        fi
    done
fi

# --------------------------------------------------------------------------
# Compute ports
# --------------------------------------------------------------------------
AGENT_PORTS=()
for (( i=0; i<AGENTS; i++ )); do
    AGENT_PORTS+=($((PORT + i * PORTS_PER_CLIENT)))
done

echo "=========================================="
echo "  MOSAIC Malmo - Minecraft Launcher"
echo "=========================================="
echo "  Agents:    $AGENTS"
echo "  Ports:     ${AGENT_PORTS[*]}"
echo "  Mode:      $([ $VISIBLE -eq 1 ] && echo "Visible" || echo "Headless (Xvfb)")"
echo "  Java:      $JAVA8_HOME"
echo "=========================================="

# --------------------------------------------------------------------------
# Launch function — waits for port to be ready before returning
# --------------------------------------------------------------------------
wait_for_port() {
    local port="$1"
    local timeout=300
    local elapsed=0
    echo "  Waiting for port $port to become ready (timeout: ${timeout}s)..."
    while [ $elapsed -lt $timeout ]; do
        if lsof -ti ":$port" >/dev/null 2>&1; then
            echo "  Port $port: READY"
            return 0
        fi
        sleep 3
        elapsed=$((elapsed + 3))
        if [ $((elapsed % 15)) -eq 0 ]; then
            echo "  Still waiting for port $port... (${elapsed}s)"
        fi
    done
    echo "  ERROR: Port $port not ready after ${timeout}s"
    return 1
}

launch_client() {
    local agent_port="$1"
    local agent_index="$2"

    echo ""
    echo "--- Launching Minecraft client $agent_index on port $agent_port ---"

    if [ $VISIBLE -eq 1 ]; then
        cd "$MINECRAFT_DIR"
        bash launchClient.sh -port "$agent_port" -env &
    elif [ -x "$HEADLESS_SCRIPT" ]; then
        bash "$HEADLESS_SCRIPT" -port "$agent_port" -env &
    else
        if ! command -v xvfb-run >/dev/null 2>&1; then
            echo "ERROR: xvfb-run not found. Install: sudo apt install xvfb"
            exit 1
        fi
        xvfb_args="${MALMO_XVFB_ARGS:--screen 0 1280x720x24 -ac +extension GLX +render -noreset}"
        cd "$MINECRAFT_DIR"
        xvfb-run --auto-servernum --server-args="$xvfb_args" \
            bash launchClient.sh -port "$agent_port" -env &
    fi

    CHILD_PIDS+=($!)
}

# --------------------------------------------------------------------------
# Launch clients SEQUENTIALLY — each must be DORMANT before the next starts
# (Gradle lock + shared run/config directory)
# --------------------------------------------------------------------------
CHILD_PIDS=()

for (( i=0; i<AGENTS; i++ )); do
    agent_port=${AGENT_PORTS[$i]}

    launch_client "$agent_port" "$i"

    # Wait for this client to be ready before launching the next
    if ! wait_for_port "$agent_port"; then
        echo "FATAL: Minecraft client $i failed to start on port $agent_port"
        echo "Killing all launched clients..."
        for pid in "${CHILD_PIDS[@]}"; do
            kill -9 "$pid" 2>/dev/null || true
        done
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "  All $AGENTS Minecraft client(s) ready!"
echo "=========================================="
echo ""
echo "  PIDs:  ${CHILD_PIDS[*]}"
echo "  Ports: ${AGENT_PORTS[*]}"
echo ""
echo "  Now in another terminal:"
echo "    ./run.sh"
echo ""
echo "  Press Ctrl+C to stop all clients."
echo "=========================================="

# Wait and cleanup on Ctrl+C
cleanup() {
    echo ""
    echo "Stopping all Minecraft clients..."
    for pid in "${CHILD_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    sleep 2
    for pid in "${CHILD_PIDS[@]}"; do
        kill -9 "$pid" 2>/dev/null || true
    done
    echo "Done."
}

trap cleanup INT TERM
wait
