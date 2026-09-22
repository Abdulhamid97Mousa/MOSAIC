#!/bin/bash
# Soccer cooperative training — IPPO, MAPPO, HAPPO, QMIX, VDPPO
# Variants: G-2v0 and G-3v0
# Output: var/trainer/mosaic_multigrid/S/cooperative/{2v0,3v0}/{ALG}/
#
# IPPO/MAPPO/HAPPO/VDPPO are Hydra-driven (Phase 1 refactor, 2026-09-18):
# sport is env=s (not an S.<alg>_scan module path), flags are key=value.
# QMIX remains the pre-refactor argparse script (Phase 2, not yet migrated).
#
# Usage:
#   GPU=0 bash scripts/train_s_cooperative.sh
#   GPU=0 VARIANTS="G-2v0" bash scripts/train_s_cooperative.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PYTHON="$PROJECT_ROOT/.venv/bin/python"

GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export PYTHONPATH="3rd_party/workers/jaxmarl_worker"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1
export TF_DETERMINISTIC_OPS=1
export TF_CUDNN_DETERMINISTIC=1
export TF_ENABLE_ONEDNN_OPTS=0

cd "$PROJECT_ROOT"

RUNS_ROOT="$PROJECT_ROOT/var/trainer/mosaic_multigrid/S/cooperative"
VARIANTS="${VARIANTS:-G-2v0 G-3v0}"

# Shared PPO hyperparams (Hydra key=value overrides; env=s supplies
# view_size/ball_approach_coef/goal_rows/max_steps defaults from conf/env/s.yaml)
PPO_ARGS=(
    env=s
    training_reward=cooperative_no_opponent
    total_updates=20000
    ppo.n_envs=256
    ppo.n_steps=256
    ppo.n_epochs=4
    ppo.n_minibatches=4
    ppo.hidden_dim=256
    ppo.lr=3e-4
    ppo.gamma=0.99
    ppo.gae_lambda=0.95
    ppo.clip_eps=0.2
    ppo.vf_coef=0.5
    ppo.ent_coef=0.005
    tensorboard=true
)

QMIX_ARGS=(
    --training-reward cooperative-no-opponent
    --n-envs 256
    --n-steps 256
    --max-steps 256
    --total-updates 20000
    --hidden-dim 256
    --mixer-embed-dim 64
    --mixer-hypernet-dim 256
    --lr 3e-4
    --gamma 0.99
    --eps-start 1.0
    --eps-finish 0.05
    --eps-decay 0.1
    --buffer-size 12800
    --buffer-batch-size 32
    --learning-starts 10000
    --n-epochs 8
    --ball-approach-coef 0.01
    --view-size 7
    --goal-rows 4 5 6
    --tensorboard
)

run_alg() {
    local VARIANT="$1" ALG="$2" MODULE="$3"
    shift 3
    local VSHORT="${VARIANT##*-}"
    local RUN_DIR="$RUNS_ROOT/$VSHORT/$ALG"
    local CKPT="$RUN_DIR/checkpoints/soccer-$VARIANT/final.npz"
    if [ -f "$CKPT" ]; then
        echo "[SKIP] $ALG $VARIANT — checkpoint exists"
        return
    fi
    mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/tensorboard"
    echo "[START] $(date '+%H:%M:%S')  $ALG  $VARIANT"
    "$PYTHON" -m "$MODULE" env.variant="$VARIANT" run_dir="$RUN_DIR" "$@"
    echo "[DONE]  $(date '+%H:%M:%S')  $ALG  $VARIANT → $CKPT"
}

# QMIX is Phase 2 (not yet migrated to Hydra) — still uses the argparse CLI.
run_alg_qmix() {
    local VARIANT="$1" ALG="$2" MODULE="$3"
    shift 3
    local VSHORT="${VARIANT##*-}"
    local RUN_DIR="$RUNS_ROOT/$VSHORT/$ALG"
    local CKPT="$RUN_DIR/checkpoints/soccer-$VARIANT/final.npz"
    if [ -f "$CKPT" ]; then
        echo "[SKIP] $ALG $VARIANT — checkpoint exists"
        return
    fi
    mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/tensorboard"
    echo "[START] $(date '+%H:%M:%S')  $ALG  $VARIANT"
    "$PYTHON" -m "$MODULE" --variant "$VARIANT" --run-dir "$RUN_DIR" "$@"
    echo "[DONE]  $(date '+%H:%M:%S')  $ALG  $VARIANT → $CKPT"
}

echo "========================================================"
echo "  Soccer cooperative training  GPU=$GPU"
echo "  variants: $VARIANTS"
echo "  output:   $RUNS_ROOT"
echo "========================================================"

for VARIANT in $VARIANTS; do
    echo ""
    echo "────────── $VARIANT ──────────"

    [ -z "${SKIP_IPPO:-}" ]  && run_alg "$VARIANT" IPPO  jaxmarl_worker.algorithms.mosaic_multigrid.ippo_scan  "${PPO_ARGS[@]}"
    [ -z "${SKIP_MAPPO:-}" ] && run_alg "$VARIANT" MAPPO jaxmarl_worker.algorithms.mosaic_multigrid.mappo_scan "${PPO_ARGS[@]}"
    [ -z "${SKIP_HAPPO:-}" ] && run_alg "$VARIANT" HAPPO jaxmarl_worker.algorithms.mosaic_multigrid.happo_scan "${PPO_ARGS[@]}"
    [ -z "${SKIP_VDPPO:-}" ] && run_alg "$VARIANT" VDPPO jaxmarl_worker.algorithms.mosaic_multigrid.vdppo_scan "${PPO_ARGS[@]}"
    [ -z "${SKIP_QMIX:-}" ]  && run_alg_qmix "$VARIANT" QMIX jaxmarl_worker.algorithms.mosaic_multigrid.S.qmix_scan "${QMIX_ARGS[@]}"
done

echo ""
echo "========================================================"
echo "  Done.  tensorboard --logdir $RUNS_ROOT"
echo "========================================================"
