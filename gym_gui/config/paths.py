from __future__ import annotations

"""Centralized filesystem paths used across the Gym GUI application."""

from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PACKAGE_ROOT.parent
_3RD_PARTY_ROOT = _REPO_ROOT / "3rd_party" / "workers"

# Writable runtime artifacts (shared with worker stack). Prefer repo-level var/.
VAR_ROOT = (_REPO_ROOT / "var").resolve()
VAR_REPLAY_DIR = VAR_ROOT / "replay"  # HDF5 replay storage (frames, observations)
VAR_TELEMETRY_DIR = VAR_ROOT / "telemetry"
VAR_CACHE_DIR = VAR_ROOT / "cache"
VAR_TMP_DIR = VAR_ROOT / "tmp"
VAR_LOGS_DIR = VAR_ROOT / "logs"
VAR_TRAINER_DIR = VAR_ROOT / "trainer"
VAR_TENSORBOARD_DIR = VAR_TRAINER_DIR / "runs"
VAR_WANDB_DIR = VAR_TRAINER_DIR / "runs"  # WANDB manifests stored per-run like TensorBoard
VAR_TRAINER_DB = VAR_TRAINER_DIR / "trainer.sqlite"
VAR_EVALS_DIR = VAR_TRAINER_DIR / "evals"  # Evaluation run artifacts (separate from training runs)
VAR_CUSTOM_SCRIPTS_DIR = VAR_TRAINER_DIR / "custom_scripts"  # Custom script phase configs and checkpoints
VAR_DATA_DIR = VAR_ROOT / "data"
VAR_MODELS_DIR = VAR_ROOT / "models"  # LLM models for vLLM serving
VAR_MODELS_HF_CACHE = VAR_MODELS_DIR / "huggingface"  # HuggingFace cache
VAR_MODELS_KATAGO_DIR = VAR_MODELS_DIR / "katago"  # KataGo neural network models
VAR_MODELS_GO_AI_DIR = VAR_MODELS_DIR / "go_ai"  # Go AI engines config/models
VAR_OPERATORS_DIR = VAR_ROOT / "operators"  # Operator subprocess data
VAR_OPERATORS_LOGS_DIR = VAR_OPERATORS_DIR / "logs"  # Operator subprocess logs
VAR_OPERATORS_TELEMETRY_DIR = VAR_OPERATORS_DIR / "telemetry"  # Operator telemetry (steps, episodes)
VAR_HUMAN_CONTROL_DIR = VAR_ROOT / "human_control"  # Human Control subprocess data
VAR_HUMAN_CONTROL_LOGS_DIR = VAR_HUMAN_CONTROL_DIR / "logs"  # Keyboard worker subprocess logs
VAR_HUMAN_CONTROL_TELEMETRY_DIR = VAR_HUMAN_CONTROL_DIR / "telemetry"  # Human play telemetry
VAR_EVALUATOR_DIR = VAR_ROOT / "evaluator"  # LLM evaluator episode logs (CSV, JSON, images)
VAR_VLLM_DIR = VAR_ROOT / "vllm"  # vLLM server logs and state
VAR_BIN_DIR = VAR_ROOT / "bin"  # Project-local binaries (KataGo, etc.)

# StarCraft II / SMAC directories
# SC2 unzips into a StarCraftII/ subdirectory, so SC2PATH = var/data/StarCraftII
VAR_SC2_ROOT = VAR_ROOT / "data"  # Parent dir containing the unzipped SC2 archive
VAR_SC2_DIR = VAR_SC2_ROOT / "StarCraftII"  # Actual SC2 installation (= SC2PATH)
VAR_SMAC_MAPS_DIR = VAR_SC2_DIR / "Maps" / "SMAC_Maps"  # SMAC v1 map files
VAR_SMACV2_MAPS_DIR = VAR_SC2_DIR / "Maps" / "SMACv2_Maps"  # SMACv2 map files
VAR_SMAC_REPLAYS_DIR = VAR_ROOT / "replay" / "smac"  # SMAC replay files (.SC2Replay)

# Neural MMO generated maps (procedural terrain)
VAR_NMMO_MAPS_DIR = VAR_DATA_DIR / "nmmo"

# Custom training scripts directories
CLEANRL_SCRIPTS_DIR = _3RD_PARTY_ROOT / "cleanrl_worker" / "cleanrl_worker" / "scripts"
XUANCE_SCRIPTS_DIR = _3RD_PARTY_ROOT / "xuance_worker" / "xuance_worker" / "scripts"
TIANSHOU_SCRIPTS_DIR = _3RD_PARTY_ROOT / "tianshou_worker" / "tianshou_worker" / "scripts"

# Human worker package root (for evdev_input, runtime, etc.)
_MOSAIC_WORKERS_ROOT = _REPO_ROOT / "3rd_party" / "workers" / "mosaic"
HUMAN_WORKER_PKG_DIR = _MOSAIC_WORKERS_ROOT / "human_worker"


def ensure_var_directories() -> None:
    """Create the writable directory structure if it does not exist."""

    for path in (
        VAR_ROOT,
        VAR_REPLAY_DIR,
        VAR_TELEMETRY_DIR,
        VAR_CACHE_DIR,
        VAR_TMP_DIR,
        VAR_LOGS_DIR,
        VAR_TRAINER_DIR,
        VAR_TENSORBOARD_DIR,
        VAR_WANDB_DIR,
        VAR_DATA_DIR,
        VAR_MODELS_DIR,
        VAR_MODELS_HF_CACHE,
        VAR_MODELS_KATAGO_DIR,
        VAR_MODELS_GO_AI_DIR,
        VAR_OPERATORS_DIR,
        VAR_OPERATORS_LOGS_DIR,
        VAR_HUMAN_CONTROL_DIR,
        VAR_HUMAN_CONTROL_LOGS_DIR,
        VAR_HUMAN_CONTROL_TELEMETRY_DIR,
        VAR_OPERATORS_TELEMETRY_DIR,
        VAR_EVALUATOR_DIR,
        VAR_VLLM_DIR,
        VAR_BIN_DIR,
        VAR_EVALS_DIR,
        VAR_CUSTOM_SCRIPTS_DIR,
        VAR_NMMO_MAPS_DIR,
        # VAR_SC2_ROOT is VAR_DATA_DIR (already created above); no extra mkdir needed.
        VAR_SMAC_REPLAYS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


__all__ = [
    "VAR_ROOT",
    "VAR_REPLAY_DIR",
    "VAR_TELEMETRY_DIR",
    "VAR_CACHE_DIR",
    "VAR_TMP_DIR",
    "VAR_LOGS_DIR",
    "VAR_TRAINER_DIR",
    "VAR_TRAINER_DB",
    "VAR_TENSORBOARD_DIR",
    "VAR_WANDB_DIR",
    "VAR_DATA_DIR",
    "VAR_MODELS_DIR",
    "VAR_MODELS_HF_CACHE",
    "VAR_MODELS_KATAGO_DIR",
    "VAR_MODELS_GO_AI_DIR",
    "VAR_OPERATORS_DIR",
    "VAR_OPERATORS_LOGS_DIR",
    "VAR_OPERATORS_TELEMETRY_DIR",
    "VAR_HUMAN_CONTROL_DIR",
    "VAR_HUMAN_CONTROL_LOGS_DIR",
    "VAR_HUMAN_CONTROL_TELEMETRY_DIR",
    "VAR_EVALUATOR_DIR",
    "VAR_VLLM_DIR",
    "VAR_BIN_DIR",
    "VAR_SC2_ROOT",
    "VAR_SC2_DIR",
    "VAR_SMAC_MAPS_DIR",
    "VAR_SMACV2_MAPS_DIR",
    "VAR_SMAC_REPLAYS_DIR",
    "VAR_EVALS_DIR",
    "VAR_CUSTOM_SCRIPTS_DIR",
    "VAR_NMMO_MAPS_DIR",
    "CLEANRL_SCRIPTS_DIR",
    "XUANCE_SCRIPTS_DIR",
    "TIANSHOU_SCRIPTS_DIR",
    "ensure_var_directories",
]
