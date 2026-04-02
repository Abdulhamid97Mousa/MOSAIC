#!/usr/bin/env bash
# ==============================================================================
# MOSAIC Malmo Setup — One-time setup for Minecraft/MalmoEnv integration
# ==============================================================================
#
# Usage:
#   ./setup_malmo.sh
#
# What this does:
#   1. Checks Java 8 JDK is installed
#   2. Checks xvfb is installed (for headless Minecraft)
#   3. Sets MALMO_XSD_PATH environment variable
#   4. Creates version.properties if missing
#   5. Downloads Minecraft 1.11.2 assets (HTTPS bypass for Mojang CDN)
#   6. Builds the Malmo Forge mod (./gradlew build)
#   7. Installs MalmoEnv Python package
#
# After setup, launch with:
#   ./run_malmo.sh        (starts Minecraft headless)
#   ./run.sh              (starts MOSAIC GUI, in a second terminal)
#
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MALMO_DIR="$ROOT_DIR/3rd_party/environments/malmo"
MALMO_UPDATED_DIR="$ROOT_DIR/3rd_party/environments/malmo_updated"
MINECRAFT_DIR="$MALMO_UPDATED_DIR/Minecraft"

echo "=========================================="
echo "  MOSAIC Malmo Setup"
echo "=========================================="
echo ""

# --------------------------------------------------------------------------
# Step 1: Check Java 8
# --------------------------------------------------------------------------
echo "[1/7] Checking Java 8 JDK..."

JAVA8_HOME=""
for candidate in /usr/lib/jvm/java-8-openjdk-amd64 /usr/lib/jvm/java-1.8.0-openjdk-amd64 /usr/lib/jvm/java-8-openjdk; do
    if [ -x "$candidate/bin/java" ] && [ -f "$candidate/lib/tools.jar" ]; then
        JAVA8_HOME="$candidate"
        break
    fi
done

if [ -z "$JAVA8_HOME" ]; then
    echo "ERROR: Java 8 JDK not found."
    echo ""
    echo "Install with:"
    echo "  sudo apt install openjdk-8-jdk"
    echo ""
    echo "Then re-run this script."
    exit 1
fi

echo "  Found: $JAVA8_HOME"
export JAVA_HOME="$JAVA8_HOME"
export PATH="$JAVA8_HOME/bin:$PATH"

# --------------------------------------------------------------------------
# Step 2: Check xvfb
# --------------------------------------------------------------------------
echo "[2/7] Checking xvfb (headless display)..."

if ! command -v xvfb-run >/dev/null 2>&1; then
    echo "ERROR: xvfb-run not found."
    echo ""
    echo "Install with:"
    echo "  sudo apt install xvfb"
    echo ""
    echo "Then re-run this script."
    exit 1
fi

echo "  Found: $(command -v xvfb-run)"

# --------------------------------------------------------------------------
# Step 3: Set MALMO_XSD_PATH
# --------------------------------------------------------------------------
echo "[3/7] Setting MALMO_XSD_PATH..."

SCHEMA_DIR="$MALMO_DIR/Schemas"
if [ ! -d "$SCHEMA_DIR" ]; then
    SCHEMA_DIR="$MALMO_UPDATED_DIR/Schemas"
fi

if [ ! -d "$SCHEMA_DIR" ]; then
    echo "  WARNING: Schemas directory not found. MalmoEnv may fail XML validation."
else
    export MALMO_XSD_PATH="$SCHEMA_DIR"
    echo "  MALMO_XSD_PATH=$SCHEMA_DIR"
    echo ""
    echo "  To make this permanent, add to ~/.bashrc:"
    echo "    echo 'export MALMO_XSD_PATH=$SCHEMA_DIR' >> ~/.bashrc"
fi

# --------------------------------------------------------------------------
# Step 4: Create version.properties
# --------------------------------------------------------------------------
echo "[4/7] Checking version.properties..."

VERSION_FILE="$MINECRAFT_DIR/src/main/resources/version.properties"
if [ -f "$VERSION_FILE" ]; then
    echo "  Already exists: $VERSION_FILE"
else
    mkdir -p "$(dirname "$VERSION_FILE")"
    echo "malmomod.version=0.37.0" > "$VERSION_FILE"
    echo "  Created: $VERSION_FILE"
fi

# --------------------------------------------------------------------------
# Step 5: Download Minecraft assets
# --------------------------------------------------------------------------
echo "[5/7] Downloading Minecraft 1.11.2 assets (HTTPS)..."
echo "  This may take a few minutes on first run..."

python3 - <<'PYEOF'
import hashlib, json, os, sys, time, urllib.request
from pathlib import Path

ASSETS_DIR = Path.home() / ".gradle/caches/minecraft/assets/objects"
INDEX_URL = (
    "https://launchermeta.mojang.com/v1/packages/"
    "d5a285bdf7b0c8f1dd6e57f36564da0ea17e3c87/1.11.json"
)
CDN = "https://resources.download.minecraft.net"

print("  Fetching asset index...")
try:
    with urllib.request.urlopen(INDEX_URL, timeout=30) as r:
        index = json.load(r)
except Exception as exc:
    print(f"  WARNING: Could not fetch asset index: {exc}")
    print("  Skipping asset download (Minecraft may still work with cached assets).")
    sys.exit(0)

objects = index["objects"]
total = len(objects)
downloaded = skipped = failed = 0

for i, (name, info) in enumerate(objects.items(), 1):
    h = info["hash"]
    prefix = h[:2]
    dest = ASSETS_DIR / prefix / h
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == info["size"]:
        skipped += 1
        continue
    url = f"{CDN}/{prefix}/{h}"
    for attempt in range(1, 4):
        try:
            urllib.request.urlretrieve(url, dest)
            downloaded += 1
            break
        except Exception:
            if attempt == 3:
                failed += 1
            else:
                time.sleep(1)

print(f"  Done. Downloaded: {downloaded}  Skipped: {skipped}  Failed: {failed}  Total: {total}")
if failed > 0:
    print(f"  ({failed} assets failed — usually just nether2.ogg which is harmless)")
PYEOF

# --------------------------------------------------------------------------
# Step 6: Build Malmo Forge mod
# --------------------------------------------------------------------------
echo "[6/7] Building Malmo Forge mod..."
echo "  This may take a few minutes on first build..."

cd "$MINECRAFT_DIR"
./gradlew build

echo "  Build successful."

# --------------------------------------------------------------------------
# Step 7: Install MalmoEnv Python package
# --------------------------------------------------------------------------
echo "[7/7] Installing MalmoEnv Python package..."

MALMOENV_DIR="$MALMO_DIR/MalmoEnv"
if [ ! -d "$MALMOENV_DIR" ]; then
    MALMOENV_DIR="$MALMO_UPDATED_DIR/MalmoEnv"
fi

if [ -d "$MALMOENV_DIR" ]; then
    cd "$ROOT_DIR"
    pip install --no-build-isolation -e "$MALMOENV_DIR"
    echo "  Installed: malmoenv"
else
    echo "  WARNING: MalmoEnv directory not found at $MALMOENV_DIR"
    echo "  You may need to initialize git submodules:"
    echo "    git submodule update --init 3rd_party/environments/malmo"
fi

# --------------------------------------------------------------------------
# Done
# --------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "To launch Minecraft (headless):"
echo "  ./run_malmo.sh"
echo ""
echo "Then in another terminal, launch MOSAIC:"
echo "  ./run.sh"
echo ""
echo "Select any MalmoEnv environment from the dropdown."
echo "=========================================="
