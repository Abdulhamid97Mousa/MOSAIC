#!/bin/bash
# AF IC3Net ADVERSARIAL — V1 campaign
# IC3Net: CommNet + per-agent sigmoid gate (Singh et al. 2019, arXiv:1812.09755)
# Gate: g_i = sigmoid(Linear(h_i)); Comm: c_i = mean_{j≠i}(g_j * h_j)
#
# Path: var/trainer/multigrid_sports/AF_m256_20k/IC3Net/VIEWSIZE7/adversarial/V1/<VARIANT>/checkpoints/af-<variant>/final.npz
#
# Usage: GPU=0 bash 3rd_party/workers/jaxmarl_worker/scripts/train_af_ic3net_adversarial_v1.sh
# Single variant: VARIANTS="1v1" GPU=1 bash ...

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
RUNS_ROOT="$PROJECT_ROOT/var/trainer/multigrid_sports/AF_m256_20k/IC3Net/VIEWSIZE7/adversarial/V1"
GPU="${GPU:-0}"
VARIANTS="${VARIANTS:-2v2}"

cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_DETERMINISTIC_OPS=1
export TF_CUDNN_DETERMINISTIC=1
export TF_ENABLE_ONEDNN_OPTS=0

echo "========================================================"
echo "  AF IC3Net ADVERSARIAL V1 (gated comm)"
echo "  GPU=$GPU  variants=$VARIANTS"
echo "  output: $RUNS_ROOT"
echo "========================================================"

for VARIANT in $VARIANTS; do
    RUN_DIR="$RUNS_ROOT/$VARIANT"
    LOG="$PROJECT_ROOT/var/logs/af-ic3net-adversarial-v1-$VARIANT.log"
    if [ -f "$RUN_DIR/checkpoints/af-$VARIANT/final.npz" ]; then
        echo "[SKIP] $VARIANT — checkpoint already exists"; continue
    fi
    mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/tensorboard" "$(dirname "$LOG")"
    echo "[START] $(date '+%H:%M:%S') $VARIANT"
    "$PYTHON" -m jaxmarl_worker.algorithms.ic3net_scan \
        --env-family af --variant "$VARIANT" \
        --run-dir "$RUN_DIR" \
        --n-envs 256 --n-steps 256 \
        --total-updates 20000 \
        --n-epochs 4 --n-minibatches 4 \
        --ent-coef 0.005 \
        --ball-approach-coef 0.01 \
        --max-steps 256 \
        --view-size 7 --goal-rows 1 2 3 4 5 6 7 8 9 \
        --no-cooperative \
        --tensorboard \
        > "$LOG" 2>&1
    echo "[DONE]  $(date '+%H:%M:%S') $VARIANT → $RUN_DIR/checkpoints/af-$VARIANT/final.npz"
done
echo; echo "========================================================"; echo "  All variants complete."; echo "========================================================"
