#!/usr/bin/env bash
set -euo pipefail

# Proto generation script for gym_gui trainer stubs.
# Usage:
#   source .venv/bin/activate  # ensure grpcio-tools installed
#   bash tools/generate_protos.sh [--java]
#
# Options:
#   --java    Also generate Java stubs for MalmoMod integration
#
# This regenerates trainer proto stubs.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GENERATE_JAVA=false

# Parse arguments
for arg in "$@"; do
  case $arg in
    --java)
      GENERATE_JAVA=true
      shift
      ;;
  esac
done

echo "[protos] Root: $ROOT_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[protos] Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

if ! "$PYTHON_BIN" -c "import grpc_tools" 2>/dev/null; then
  echo "[protos] grpcio-tools not installed. Run: pip install grpcio-tools" >&2
  exit 1
fi

TRAINER_DIR="gym_gui/services/trainer/proto"

cd "$ROOT_DIR"

echo "[protos] Generating Python trainer stubs"
"$PYTHON_BIN" -m grpc_tools.protoc \
  -I "$TRAINER_DIR" \
  --python_out="$TRAINER_DIR" \
  --grpc_python_out="$TRAINER_DIR" \
  --pyi_out="$TRAINER_DIR" \
  "$TRAINER_DIR/trainer.proto"

# Generate Java stubs for MalmoMod if requested
if [ "$GENERATE_JAVA" = true ]; then
  # Check for Java protoc plugins
  if ! command -v protoc-gen-java >/dev/null 2>&1; then
    echo "[protos] protoc-gen-java not found. Install with:" >&2
    echo "  Ubuntu/Debian: sudo apt install protobuf-compiler" >&2
    echo "  Or download from: https://github.com/protocolbuffers/protobuf/releases" >&2
    exit 1
  fi

  if ! command -v protoc-gen-grpc-java >/dev/null 2>&1; then
    echo "[protos] protoc-gen-grpc-java not found. Install gRPC Java plugin:" >&2
    echo "  Download from: https://repo1.maven.org/maven2/io/grpc/protoc-gen-grpc-java/" >&2
    echo "  Then: chmod +x protoc-gen-grpc-java-*.exe && sudo mv protoc-gen-grpc-java-*.exe /usr/local/bin/protoc-gen-grpc-java" >&2
    exit 1
  fi

  JAVA_OUT_DIR="3rd_party/environments/mosaic_malmo/java/proto"
  mkdir -p "$JAVA_OUT_DIR"

  echo "[protos] Generating Java trainer stubs for MalmoMod"
  protoc \
    -I "$TRAINER_DIR" \
    --java_out="$JAVA_OUT_DIR" \
    --grpc_java_out="$JAVA_OUT_DIR" \
    "$TRAINER_DIR/trainer.proto"

  echo "[protos] Java stubs written to: $JAVA_OUT_DIR"
fi

echo "[protos] Done"
