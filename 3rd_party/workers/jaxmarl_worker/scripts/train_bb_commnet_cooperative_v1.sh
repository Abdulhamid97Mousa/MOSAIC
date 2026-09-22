#!/bin/bash
# BB CommNet COOPERATIVE — V1 campaign
# Variants: G-2v0, B-0v2 — needed for ghost-inertness table.
# Usage: GPU=0 bash .../train_bb_commnet_cooperative_v1.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
RUNS_ROOT="$PROJECT_ROOT/var/trainer/multigrid_sports/BB_m256_20k/CommNet/VIEWSIZE7/cooperative/V1"
GPU="${GPU:-0}"
VARIANTS="${VARIANTS:-G-2v0 B-0v2}"

cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_DETERMINISTIC_OPS=1
export TF_CUDNN_DETERMINISTIC=1
export TF_ENABLE_ONEDNN_OPTS=0

echo "========================================================"
echo "  BB CommNet COOPERATIVE V1  (G-2v0, B-0v2)"
echo "  GPU=$GPU  variants=$VARIANTS"
echo "  output: $RUNS_ROOT"
echo "========================================================"

for VARIANT in $VARIANTS; do
    RUN_DIR="$RUNS_ROOT/$VARIANT"
    LOG="$PROJECT_ROOT/var/logs/bb-commnet-cooperative-v1-$VARIANT.log"
    if [ -f "$RUN_DIR/checkpoints/bb-$VARIANT/final.npz" ]; then
        echo "[SKIP] $VARIANT — checkpoint already exists"; continue
    fi
    mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/tensorboard" "$(dirname "$LOG")"
    echo "[START] $(date '+%H:%M:%S') $VARIANT"
    "$PYTHON" -m jaxmarl_worker.algorithms.commnet_scan \
        --env-family bb --variant "$VARIANT" \
        --run-dir "$RUN_DIR" \
        --n-envs 256 --n-steps 256 \
        --total-updates 20000 \
        --n-epochs 4 --n-minibatches 4 \
        --ent-coef 0.005 \
        --ball-approach-coef 0.01 \
        --max-steps 256 \
        --view-size 7 --goal-rows 5 \
        --cooperative \
        --tensorboard \
        > "$LOG" 2>&1
    echo "[DONE]  $(date '+%H:%M:%S') $VARIANT → $RUN_DIR/checkpoints/bb-$VARIANT/final.npz"
done
echo; echo "========================================================"; echo "  All variants complete."; echo "========================================================"
