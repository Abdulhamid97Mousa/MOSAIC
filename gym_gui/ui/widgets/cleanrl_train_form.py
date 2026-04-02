"""CleanRL worker training form."""

from __future__ import annotations

import copy
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from qtpy import QtCore, QtGui, QtWidgets

from gym_gui.Algo_docs.cleanrl_worker import get_algo_doc
from gym_gui.core.enums import (
    ENVIRONMENT_FAMILY_BY_GAME,
    EnvironmentFamily,
    GameId,
)
from gym_gui.fastlane.worker_helpers import apply_fastlane_environment
from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_UI_TRAIN_FORM_ERROR,
    LOG_UI_TRAIN_FORM_INFO,
    LOG_UI_TRAIN_FORM_TELEMETRY_PATH,
    LOG_UI_TRAIN_FORM_TRACE,
    LOG_UI_TRAIN_FORM_UI_PATH,
    LOG_UI_TRAIN_FORM_WARNING,
)
from gym_gui.telemetry.semconv import VIDEO_MODE_DESCRIPTORS, VideoModes
from gym_gui.validations.validation_cleanrl_worker_form import run_cleanrl_dry_run

REPO_ROOT = Path(__file__).resolve().parents[3]
_LEGACY_ALGOS: tuple[str, ...] = (
    "ppo",
    "ppo_continuous_action",
    "ppo_atari",
    "ppo_atari_multigpu",
    "ppo_atari_lstm",
    "ppo_atari_envpool",
    "ppo_atari_envpool_xla_jax",
    "ppo_atari_envpool_xla_jax_scan",
    "ppo_pettingzoo_ma_atari",
    "pqn",
    "pqn_atari_envpool",
    "pqn_atari_envpool_lstm",
    "rpo_continuous_action",
    "dqn",
    "dqn_atari",
    "dqn_atari_jax",
    "dqn_jax",
    "rainbow_atari",
    "c51",
    "c51_atari",
    "c51_atari_jax",
    "c51_jax",
    "ddpg_continuous_action",
    "ddpg_continuous_action_jax",
    "td3_continuous_action",
    "td3_continuous_action_jax",
    "sac_continuous_action",
    "sac_atari",
    "qdagger_dqn_atari_impalacnn",
    "qdagger_dqn_atari_jax_impalacnn",
)


def _load_cleanrl_schemas() -> tuple[dict[str, Any], Optional[str]]:
    schema_root = REPO_ROOT / "metadata" / "cleanrl"
    if not schema_root.exists():
        return {}, None

    candidates: list[tuple[str, Path]] = []
    for entry in schema_root.iterdir():
        if not entry.is_dir():
            continue
        schema_file = entry / "schemas.json"
        if schema_file.exists():
            candidates.append((entry.name, schema_file))

    candidates.sort(key=lambda item: item[0], reverse=True)
    if not candidates:
        fallback = schema_root / "schemas.json"
        if fallback.exists():
            candidates.append(("latest", fallback))

    for _, schema_file in candidates:
        try:
            data = json.loads(schema_file.read_text())
        except Exception:
            continue
        return data.get("algorithms", {}), data.get("cleanrl_version")

    return {}, None


_CLEANRL_SCHEMAS, _CLEANRL_SCHEMA_VERSION = _load_cleanrl_schemas()

if _CLEANRL_SCHEMAS:
    _DEFAULT_ALGOS = tuple(sorted(set(_LEGACY_ALGOS) | set(_CLEANRL_SCHEMAS.keys())))
else:
    _DEFAULT_ALGOS = _LEGACY_ALGOS


@dataclass(frozen=True)
class _AlgoParamSpec:
    key: str
    label: str
    default: Any
    field_type: type
    tooltip: str | None = None


_ALGO_PARAM_SPECS: dict[str, tuple[_AlgoParamSpec, ...]] = {
    "ppo": (
        _AlgoParamSpec("learning_rate", "Learning Rate", 2.5e-4, float, "Adam optimizer learning rate"),
        _AlgoParamSpec("num_envs", "Parallel Envs", 16, int, "Number of vectorized environments"),
        _AlgoParamSpec("num_steps", "Steps per Update", 2048, int, "Rollout length before each update"),
        _AlgoParamSpec("capture_video", "Capture Video", False, bool, "Record episode videos (first environment only)"),
    ),
    "ppo_atari": (
        _AlgoParamSpec("learning_rate", "Learning Rate", 2.5e-4, float, "Adam optimizer learning rate"),
        _AlgoParamSpec("total_frames", "Total Frames", 10_000_000, int, "Total Atari frames to collect"),
        _AlgoParamSpec("capture_video", "Capture Video", False, bool, "Record gameplay videos when supported"),
    ),
    "dqn": (
        _AlgoParamSpec("learning_rate", "Learning Rate", 1e-4, float, "Adam optimizer learning rate"),
        _AlgoParamSpec("batch_size", "Batch Size", 128, int, "Mini-batch size from replay buffer"),
        _AlgoParamSpec("buffer_size", "Replay Buffer", 100_000, int, "Maximum replay buffer size"),
        _AlgoParamSpec("capture_video", "Capture Video", False, bool, "Record evaluation videos (first environment only)"),
    ),
}

_SCHEMA_EXCLUDED_FIELDS: set[str] = {
    "exp_name",
    "track",
    "seed",
    "total_timesteps",
    "cuda",
    "wandb_project_name",
    "wandb_project",
    "wandb_entity",
    "wandb_entity_name",
    "wandb_api_key",
    "wandb_email",
    "wandb_mode",
    "wandb_run_name",
    "batch_size",
    "minibatch_size",
    "num_iterations",
}


_SUPPORTED_FAMILIES: set[EnvironmentFamily] = {
    # Basic Gymnasium environments
    EnvironmentFamily.CLASSIC_CONTROL,
    EnvironmentFamily.BOX2D,
    EnvironmentFamily.MUJOCO,
    EnvironmentFamily.TOY_TEXT,
    # Grid-based environments
    EnvironmentFamily.MINIGRID,
    EnvironmentFamily.BABYAI,
    # Atari environments
    EnvironmentFamily.ATARI,
    EnvironmentFamily.ALE,
    # Advanced RL benchmarks
    EnvironmentFamily.VIZDOOM,
    EnvironmentFamily.MINIHACK,
    EnvironmentFamily.NETHACK,
    EnvironmentFamily.CRAFTER,
    EnvironmentFamily.PROCGEN,
    # Multi-agent environments
    EnvironmentFamily.PETTINGZOO_CLASSIC,
    EnvironmentFamily.MOSAIC_MULTIGRID,
    EnvironmentFamily.INI_MULTIGRID,
    EnvironmentFamily.MELTINGPOT,
    EnvironmentFamily.OVERCOOKED,
}

_ADDITIONAL_SUPPORTED_GAMES: set[GameId] = {
    GameId.PONG_NO_FRAMESKIP,
    GameId.BREAKOUT_NO_FRAMESKIP,
}

_PREFERRED_GAME_ORDER: Sequence[GameId] = (
    GameId.FROZEN_LAKE,
    GameId.FROZEN_LAKE_V2,
    GameId.CLIFF_WALKING,
    GameId.TAXI,
    GameId.BLACKJACK,
    GameId.CART_POLE,
    GameId.ACROBOT,
    GameId.MOUNTAIN_CAR,
    GameId.LUNAR_LANDER,
    GameId.CAR_RACING,
    GameId.BIPEDAL_WALKER,
    GameId.ANT,
    GameId.HALF_CHEETAH,
    GameId.HOPPER,
    GameId.WALKER2D,
    GameId.HUMANOID,
    GameId.HUMANOID_STANDUP,
    GameId.INVERTED_PENDULUM,
    GameId.INVERTED_DOUBLE_PENDULUM,
    GameId.REACHER,
    GameId.PUSHER,
    GameId.SWIMMER,
    GameId.PONG_NO_FRAMESKIP,
    GameId.BREAKOUT_NO_FRAMESKIP,
)


def _format_cleanrl_family_label(family: EnvironmentFamily | None) -> str:
    if family is None:
        return "General"
    return family.value.replace("_", " ").title()


def _build_environment_choices() -> tuple[tuple[str, str], ...]:
    supported: list[GameId] = []
    for game in GameId:
        family = ENVIRONMENT_FAMILY_BY_GAME.get(game)
        if family in _SUPPORTED_FAMILIES or game in _ADDITIONAL_SUPPORTED_GAMES:
            supported.append(game)

    ordered: list[GameId] = []
    for game in _PREFERRED_GAME_ORDER:
        if game in supported and game not in ordered:
            ordered.append(game)
    for game in sorted(supported, key=lambda g: g.value):
        if game not in ordered:
            ordered.append(game)

    choices: list[tuple[str, str]] = []
    for game in ordered:
        family = ENVIRONMENT_FAMILY_BY_GAME.get(game)
        label = f"{game.value} ({_format_cleanrl_family_label(family)})"
        # Special-case for Atari where get_game_display_name already includes prefix
        if family == EnvironmentFamily.ATARI:
            label = f"{game.value} (Atari)"
        choices.append((label, game.value))
    return tuple(choices)


CLEANRL_ENVIRONMENT_CHOICES: tuple[tuple[str, str], ...] = _build_environment_choices()


def _env_family_from_env_id(env_id: str) -> EnvironmentFamily | None:
    try:
        game = GameId(env_id)
    except ValueError:
        return None
    return ENVIRONMENT_FAMILY_BY_GAME.get(game)


def _build_cleanrl_family_index() -> dict[EnvironmentFamily | None, list[tuple[str, str]]]:
    mapping: dict[EnvironmentFamily | None, list[tuple[str, str]]] = defaultdict(list)
    for label, env_id in CLEANRL_ENVIRONMENT_CHOICES:
        family = _env_family_from_env_id(env_id)
        mapping[family].append((label, env_id))
    for family in mapping:
        mapping[family].sort(key=lambda item: item[0])
    return mapping


CLEANRL_ENVIRONMENT_FAMILY_INDEX = _build_cleanrl_family_index()


def get_cleanrl_environment_choices() -> tuple[tuple[str, str], ...]:
    """Return the list of (label, env_id) tuples supported by CleanRL."""

    return CLEANRL_ENVIRONMENT_CHOICES


def get_cleanrl_environment_family_index() -> dict[EnvironmentFamily | None, list[tuple[str, str]]]:
    """Return the CleanRL env choices grouped by EnvironmentFamily."""

    return {family: list(options) for family, options in CLEANRL_ENVIRONMENT_FAMILY_INDEX.items()}


def _generate_run_id(prefix: str, algo: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    slug = algo.replace("_", "-")
    return f"{prefix}-{slug}-{timestamp}"


@dataclass(frozen=True)
class _FormState:
    algo: str
    env_id: str
    total_timesteps: int
    seed: Optional[int]
    agent_id: Optional[str]
    worker_id: Optional[str]
    use_gpu: bool
    capture_video: bool
    track_tensorboard: bool
    track_wandb: bool
    wandb_project: Optional[str]
    wandb_entity: Optional[str]
    wandb_run_name: Optional[str]
    wandb_api_key: Optional[str]
    wandb_http_proxy: Optional[str]
    wandb_https_proxy: Optional[str]
    use_wandb_vpn: bool
    notes: Optional[str]
    algo_params: Dict[str, Any]
    fastlane_only: bool
    fastlane_slot: int
    fastlane_video_mode: str
    fastlane_grid_limit: int


_LOGGER = logging.getLogger("gym_gui.ui.cleanrl_train_form")


class CleanRlTrainForm(QtWidgets.QDialog, LogConstantMixin):
    """Minimal training configuration dialog for CleanRL worker."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        default_game: Optional[GameId] = None,
        default_env_id: Optional[str] = None,
    ) -> None:
        super().__init__(parent)
        self._logger = _LOGGER
        self.setWindowTitle("CleanRL Agent Train Form")
        self.setModal(True)
        self.resize(720, 420)

        self._last_config: Optional[Dict[str, Any]] = None
        self._last_validation_output: str = ""

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        layout.addWidget(scroll, 1)

        form_panel = QtWidgets.QWidget(scroll)
        form_layout = QtWidgets.QVBoxLayout(form_panel)
        form_layout.setSpacing(12)
        form_layout.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(form_panel)

        intro = QtWidgets.QLabel(
            "Configure a CleanRL training run. The table below keeps core inputs aligned so you "
            "can see how algorithm, environment, and validation settings relate at a glance."
        )
        intro.setWordWrap(True)
        form_layout.addWidget(intro)

        table_widget = QtWidgets.QWidget(self)
        table_layout = QtWidgets.QGridLayout(table_widget)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.setHorizontalSpacing(12)
        table_layout.setVerticalSpacing(8)

        def _inline_field(title: str, body: QtWidgets.QWidget) -> QtWidgets.QWidget:
            container = QtWidgets.QWidget(self)
            h_layout = QtWidgets.QHBoxLayout(container)
            h_layout.setContentsMargins(0, 0, 0, 0)
            h_layout.setSpacing(6)
            label = QtWidgets.QLabel(f"{title}:", container)
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            label.setMinimumWidth(110)
            h_layout.addWidget(label)
            h_layout.addWidget(body, 1)
            return container

        self._algo_combo = QtWidgets.QComboBox(self)
        algo_view = QtWidgets.QListView(self._algo_combo)
        algo_view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        algo_view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        algo_view.setUniformItemSizes(True)
        self._algo_combo.setView(algo_view)
        self._algo_combo.setMaxVisibleItems(10)
        for algo in _DEFAULT_ALGOS:
            self._algo_combo.addItem(algo)
        self._algo_combo.setToolTip("Select the CleanRL algorithm entry point to invoke.")

        self._env_family_combo = QtWidgets.QComboBox(self)
        self._env_family_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._env_family_combo.setMaxVisibleItems(10)
        self._env_family_combo.setStyleSheet("QComboBox { combobox-popup: 0; }")

        self._env_combo = QtWidgets.QComboBox(self)
        self._env_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        env_view = QtWidgets.QListView(self._env_combo)
        env_view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        env_view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        env_view.setUniformItemSizes(True)
        self._env_combo.setView(env_view)
        self._env_combo.setStyleSheet("QComboBox { combobox-popup: 0; }")
        self._env_combo.setMaxVisibleItems(10)

        self._cleanrl_family_index = CLEANRL_ENVIRONMENT_FAMILY_INDEX
        self._selected_cleanrl_family: Optional[EnvironmentFamily | None] = None
        preferred_env = default_env_id or (default_game.value if default_game is not None else None)
        self._populate_cleanrl_environment_selectors(preferred_env)

        table_layout.addWidget(_inline_field("Algorithm", self._algo_combo), 0, 0)
        table_layout.addWidget(_inline_field("Env Family", self._env_family_combo), 0, 1)
        table_layout.addWidget(_inline_field("Environment", self._env_combo), 0, 2)

        self._seed_spin = QtWidgets.QSpinBox(self)
        self._seed_spin.setRange(0, 1_000_000_000)
        self._seed_spin.setValue(1)
        self._seed_spin.setToolTip("Algorithm seed forwarded to CleanRL (use 0 to leave the seed unspecified).")
        self._agent_id_input = QtWidgets.QLineEdit(self)
        self._agent_id_input.setPlaceholderText("Optional agent identifier")
        self._agent_id_input.setToolTip("Label used in analytics manifests and WANDB/TensorBoard tabs.")
        self._worker_id_input = QtWidgets.QLineEdit(self)
        self._worker_id_input.setPlaceholderText("Optional worker override (e.g. cleanrl-gpu-01)")
        self._worker_id_input.setToolTip("Override the worker id reported to the trainer daemon.")

        table_layout.addWidget(_inline_field("Seed (optional)", self._seed_spin), 1, 0)
        table_layout.addWidget(_inline_field("Agent ID", self._agent_id_input), 1, 1)
        table_layout.addWidget(_inline_field("Worker ID", self._worker_id_input), 1, 2)

        self._use_gpu_checkbox = QtWidgets.QCheckBox("Enable CUDA (GPU)", self)
        self._use_gpu_checkbox.setChecked(True)
        self._use_gpu_checkbox.setToolTip("Toggle CleanRL's --cuda flag; disable if the host lacks a GPU.")

        self._capture_video_slot = QtWidgets.QVBoxLayout()
        self._capture_video_slot.setContentsMargins(0, 0, 0, 0)
        self._capture_video_slot.setSpacing(4)
        self._capture_video_placeholder_label = QtWidgets.QLabel(
            "Capture video toggle becomes available once the algorithm loads.", self
        )
        self._capture_video_placeholder_label.setWordWrap(True)
        self._capture_video_slot.addWidget(self._capture_video_placeholder_label)
        capture_video_container = QtWidgets.QWidget(self)
        capture_video_layout = QtWidgets.QVBoxLayout(capture_video_container)
        capture_video_layout.setContentsMargins(0, 0, 0, 0)
        capture_video_layout.setSpacing(4)
        capture_video_layout.addLayout(self._capture_video_slot)
        capture_video_layout.addStretch(1)

        self._save_model_checkbox = QtWidgets.QCheckBox("Save model after training", self)
        self._save_model_checkbox.setChecked(True)
        self._save_model_checkbox.setToolTip(
            "Save the trained model checkpoint when training completes.\n"
            "The model will be saved in the run directory."
        )

        self._procedural_generation_checkbox = QtWidgets.QCheckBox("Procedural generation (randomize levels)", self)
        self._procedural_generation_checkbox.setChecked(True)
        self._procedural_generation_checkbox.setToolTip(
            "Enable procedural generation: each episode uses a different random level layout (standard RL training).\n"
            "Disable for fixed generation: all episodes use the same level layout (for debugging/memorization testing)."
        )

        table_layout.addWidget(_inline_field("GPU", self._use_gpu_checkbox), 2, 0)
        table_layout.addWidget(_inline_field("Capture Video", capture_video_container), 2, 1)
        table_layout.addWidget(_inline_field("Save Model", self._save_model_checkbox), 2, 2)

        self._fastlane_checkbox = QtWidgets.QCheckBox("Fast Lane Only (skip telemetry persistence)", self)
        self._fastlane_checkbox.setChecked(True)
        self._fastlane_checkbox.setToolTip("Disables the slow lane (gRPC/SQLite) so only the shared-memory fast lane runs.")
        self._fastlane_checkbox.toggled.connect(self._on_fastlane_toggled)
        self._video_mode_combo = QtWidgets.QComboBox(self)
        for descriptor in VIDEO_MODE_DESCRIPTORS.values():
            self._video_mode_combo.addItem(descriptor.label, descriptor.name)
        self._video_mode_combo.setCurrentIndex(
            self._video_mode_combo.findData(VideoModes.SINGLE)
        )
        self._video_mode_combo.setToolTip("Choose how vectorized envs are rendered to Fast Lane.")
        self._video_mode_combo.currentIndexChanged.connect(lambda _: self._update_video_mode_controls())
        self._grid_limit_spin = QtWidgets.QSpinBox(self)
        self._grid_limit_spin.setRange(1, 16)
        self._grid_limit_spin.setValue(4)
        self._grid_limit_spin.setToolTip("When using grid mode, stream this many env slots (starting from index 0).")
        self._fastlane_slot_spin = QtWidgets.QSpinBox(self)
        self._fastlane_slot_spin.setRange(0, 64)
        self._fastlane_slot_spin.setValue(0)
        self._fastlane_slot_spin.setToolTip(
            "Select which vectorized env index serves as the detailed probe when video mode is Single."
        )
        table_layout.addWidget(_inline_field("Telemetry Mode", self._fastlane_checkbox), 3, 0)
        table_layout.addWidget(_inline_field("Video Mode", self._video_mode_combo), 3, 1)
        table_layout.addWidget(_inline_field("Grid Env Limit", self._grid_limit_spin), 3, 2)
        table_layout.addWidget(_inline_field("Procedural Gen", self._procedural_generation_checkbox), 4, 0)
        table_layout.addWidget(_inline_field("Probe Env (index)", self._fastlane_slot_spin), 4, 1)

        table_layout.setColumnStretch(0, 1)
        table_layout.setColumnStretch(1, 1)
        table_layout.setColumnStretch(2, 1)

        form_layout.addWidget(table_widget)

        content_widget = QtWidgets.QWidget(self)
        content_layout = QtWidgets.QHBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(16)
        left_panel = QtWidgets.QWidget(content_widget)
        left_column = QtWidgets.QVBoxLayout(left_panel)
        left_column.setContentsMargins(0, 0, 0, 0)
        left_column.setSpacing(12)
        right_panel = QtWidgets.QWidget(content_widget)
        right_column = QtWidgets.QVBoxLayout(right_panel)
        right_column.setContentsMargins(0, 0, 0, 0)
        right_column.setSpacing(12)
        content_layout.addWidget(left_panel, 1)
        content_layout.addWidget(right_panel, 1)

        self._timesteps_spin = QtWidgets.QSpinBox(self)
        self._timesteps_spin.setRange(1_024, 1_000_000_000)
        self._timesteps_spin.setSingleStep(1_024)
        self._timesteps_spin.setValue(2048)
        self._timesteps_spin.setToolTip("Total timesteps (or frames) CleanRL will train before exiting.")
        left_column.addWidget(self._labeled("Total Timesteps", self._timesteps_spin))

        help_box = QtWidgets.QGroupBox("Algorithm Notes", self)
        help_layout = QtWidgets.QVBoxLayout(help_box)
        help_layout.setContentsMargins(8, 8, 8, 8)

        self._algo_help_text = QtWidgets.QTextEdit(help_box)
        self._algo_help_text.setReadOnly(True)
        self._algo_help_text.setMinimumHeight(160)
        self._algo_help_text.setWordWrapMode(QtGui.QTextOption.WrapMode.WordWrap)
        help_layout.addWidget(self._algo_help_text)
        left_column.addWidget(help_box)

        analytics_group = QtWidgets.QGroupBox("Analytics Tracking", self)
        analytics_layout = QtWidgets.QVBoxLayout(analytics_group)
        analytics_layout.setContentsMargins(8, 8, 8, 8)
        analytics_layout.setSpacing(6)
        self._analytics_hint_label = QtWidgets.QLabel(
            "Select analytics to export after the run completes (fast training only).",
            analytics_group,
        )
        self._analytics_hint_label.setStyleSheet("color: #777777; font-size: 11px;")
        self._analytics_hint_label.setWordWrap(True)
        analytics_layout.addWidget(self._analytics_hint_label)

        self._tensorboard_checkbox = QtWidgets.QCheckBox("Export TensorBoard artifacts", self)
        self._tensorboard_checkbox.setToolTip(
            "Write TensorBoard event files to var/trainer/runs/<run_id>/tensorboard."
        )
        analytics_layout.addWidget(self._tensorboard_checkbox)

        self._track_wandb_checkbox = QtWidgets.QCheckBox("Export WANDB artifacts", self)
        self._track_wandb_checkbox.setToolTip("Requires wandb login on the trainer host.")
        self._track_wandb_checkbox.toggled.connect(self._on_track_wandb_toggled)
        analytics_layout.addWidget(self._track_wandb_checkbox)

        right_column.addWidget(analytics_group)

        wandb_container = QtWidgets.QWidget(self)
        wandb_layout = QtWidgets.QFormLayout(wandb_container)
        wandb_layout.setContentsMargins(0, 0, 0, 0)
        self._wandb_project_input = QtWidgets.QLineEdit(self)
        self._wandb_project_input.setPlaceholderText("e.g. MOSAIC")
        self._wandb_project_input.setToolTip("Project name inside wandb.ai where runs will be grouped.")
        wandb_layout.addRow("WANDB Project", self._wandb_project_input)
        self._wandb_entity_input = QtWidgets.QLineEdit(self)
        self._wandb_entity_input.setPlaceholderText("e.g. abdulhamid97mousa")
        self._wandb_entity_input.setToolTip("WANDB entity (team or user namespace) to publish to.")
        wandb_layout.addRow("WANDB Entity", self._wandb_entity_input)
        self._wandb_run_name_input = QtWidgets.QLineEdit(self)
        self._wandb_run_name_input.setPlaceholderText("Optional custom run name")
        wandb_layout.addRow("WANDB Run Name", self._wandb_run_name_input)
        self._wandb_api_key_input = QtWidgets.QLineEdit(self)
        self._wandb_api_key_input.setPlaceholderText("Optional API key override")
        self._wandb_api_key_input.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        wandb_layout.addRow("WANDB API Key", self._wandb_api_key_input)
        self._wandb_use_vpn_checkbox = QtWidgets.QCheckBox("Route WANDB traffic through VPN proxy", self)
        self._wandb_use_vpn_checkbox.toggled.connect(self._on_wandb_vpn_toggled)
        wandb_layout.addRow("Use WANDB VPN", self._wandb_use_vpn_checkbox)
        self._wandb_http_proxy_input = QtWidgets.QLineEdit(self)
        self._wandb_http_proxy_input.setPlaceholderText("Optional HTTP proxy (e.g. http://127.0.0.1:7890)")
        wandb_layout.addRow("WANDB HTTP Proxy", self._wandb_http_proxy_input)
        self._wandb_https_proxy_input = QtWidgets.QLineEdit(self)
        self._wandb_https_proxy_input.setPlaceholderText("Optional HTTPS proxy (e.g. https://127.0.0.1:7890)")
        wandb_layout.addRow("WANDB HTTPS Proxy", self._wandb_https_proxy_input)
        right_column.addWidget(self._labeled("WANDB", wandb_container))

        self._notes_edit = QtWidgets.QPlainTextEdit(self)
        self._notes_edit.setPlaceholderText("Optional notes for analytics manifests and logs.")
        self._notes_edit.setToolTip("Notes appear alongside analytics artifacts for later reference.")
        right_column.addWidget(self._labeled("Notes", self._notes_edit))
        right_column.addStretch(1)

        self._update_wandb_controls()
        self._update_video_mode_controls()

        # Trace form open event for analytics with structured code
        self.log_constant(
            LOG_UI_TRAIN_FORM_INFO,
            message="CleanRlTrainForm opened",
            extra={"default_game": getattr(default_game, "value", None)},
        )

        self._algo_param_group = QtWidgets.QGroupBox("Algorithm Parameters", self)
        self._algo_param_layout = QtWidgets.QGridLayout(self._algo_param_group)
        self._algo_param_layout.setContentsMargins(12, 12, 12, 12)
        self._algo_param_layout.setHorizontalSpacing(12)
        self._algo_param_layout.setVerticalSpacing(10)
        self._algo_env_widget: Optional[QtWidgets.QLineEdit] = None
        left_column.addWidget(self._algo_param_group)
        left_column.addStretch(1)
        form_layout.addWidget(content_widget)
        form_layout.addStretch(1)
        self._algo_param_inputs: Dict[str, QtWidgets.QWidget] = {}
        self._num_env_widget: Optional[QtWidgets.QSpinBox] = None
        self._env_family_combo.currentIndexChanged.connect(self._on_env_family_changed)
        self._env_combo.currentIndexChanged.connect(self._sync_env_param)
        self._algo_combo.currentTextChanged.connect(self._on_algorithm_changed)
        self._on_algorithm_changed(self._algo_combo.currentText())

        self._validation_status_label = QtWidgets.QLabel(
            "Dry-run validation has not been executed yet."
        )
        self._validation_status_label.setStyleSheet("color: #666666;")
        layout.addWidget(self._validation_status_label)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            QtCore.Qt.Orientation.Horizontal,
            self,
        )
        buttons.accepted.connect(self._handle_accept)
        buttons.rejected.connect(self.reject)
        validate_btn = buttons.addButton("Validate", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
        if validate_btn is not None:
            validate_btn.clicked.connect(self._on_validate_clicked)
        layout.addWidget(buttons)

    def _labeled(self, label: str, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget(self)
        form_layout = QtWidgets.QFormLayout(container)
        form_layout.addRow(label, widget)
        return container

    def _clear_layout(self, layout: QtWidgets.QLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            if item is None:
                # Defensive: takeAt can return None; skip to satisfy static analysis.
                continue
            child_layout = item.layout()
            if child_layout is not None:
                self._clear_layout(child_layout)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

    def _reset_capture_video_slot(self, widget: Optional[QtWidgets.QWidget]) -> None:
        if not hasattr(self, "_capture_video_slot"):
            return
        self._clear_layout(self._capture_video_slot)
        if widget is not None:
            self._capture_video_slot.addWidget(widget)
        elif hasattr(self, "_capture_video_placeholder_label"):
            self._capture_video_slot.addWidget(self._capture_video_placeholder_label)

    @staticmethod
    def _format_label(name: str) -> str:
        return name.replace("_", " ").title()

    def _wrap_with_label(self, label: str, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        title = QtWidgets.QLabel(label, container)
        title.setStyleSheet("font-weight: 600;")
        layout.addWidget(title)
        layout.addWidget(widget)
        return container

    def _create_widget_from_schema_field(self, spec: Dict[str, Any]) -> Optional[QtWidgets.QWidget]:
        name = spec.get("name") or spec.get("key") or spec.get("id")
        field_type = spec.get("type")
        default = spec.get("default")
        tooltip = spec.get("help") or ""

        if field_type == "bool":
            checkbox = QtWidgets.QCheckBox(self)
            checkbox.setChecked(bool(default))
            if tooltip:
                checkbox.setToolTip(tooltip)
            return checkbox

        if field_type == "int":
            spin = QtWidgets.QSpinBox(self)
            spin.setRange(-1_000_000_000, 1_000_000_000)
            if isinstance(default, int):
                spin.setValue(default)
            if tooltip:
                spin.setToolTip(tooltip)
            return spin

        if field_type == "float":
            spin = QtWidgets.QDoubleSpinBox(self)
            spin.setDecimals(6)
            spin.setRange(-1e9, 1e9)
            if isinstance(default, (int, float)):
                spin.setValue(float(default))
            if tooltip:
                spin.setToolTip(tooltip)
            return spin

        if field_type == "str":
            line = QtWidgets.QLineEdit(self)
            if isinstance(default, str):
                line.setText(default)
            if tooltip:
                line.setToolTip(tooltip)
            if name == "env_id":
                line.setReadOnly(True)
                line.setPlaceholderText("Syncs with Environment selector")
                self._algo_env_widget = line
                self._sync_env_param()
            return line

        return None

    def _populate_params_from_schema(self, fields: Sequence[Dict[str, Any]]) -> bool:
        added = False
        columns = 2
        row = col = 0

        for field in fields:
            name = field.get("name")
            if not isinstance(name, str):
                continue
            if field.get("runtime_only") or name in _SCHEMA_EXCLUDED_FIELDS:
                continue

            if name == "capture_video":
                checkbox = QtWidgets.QCheckBox(self)
                checkbox.setChecked(bool(field.get("default")))
                checkbox.setToolTip(field.get("help") or "")
                self._algo_param_inputs[name] = checkbox
                self._capture_video_checkbox = checkbox
                self._reset_capture_video_slot(checkbox)
                continue

            widget = self._create_widget_from_schema_field(field)
            if widget is None:
                continue

            self._algo_param_inputs[name] = widget
            if name == "num_envs":
                self._register_num_env_widget(widget)
            container = self._wrap_with_label(self._format_label(name), widget)
            self._algo_param_layout.addWidget(container, row, col)
            col += 1
            if col >= columns:
                col = 0
                row += 1
            added = True

        if not added:
            self._reset_capture_video_slot(None)

        return added

    def _sync_env_param(self) -> None:
        if self._algo_env_widget is None:
            return
        env_data = self._env_combo.currentData()
        value = str(env_data) if env_data is not None else ""
        blocked = self._algo_env_widget.blockSignals(True)
        self._algo_env_widget.setText(value)
        self._algo_env_widget.blockSignals(blocked)

    def _apply_schema_defaults(self, algo: str) -> None:
        schema_entry = _CLEANRL_SCHEMAS.get(algo)
        if not schema_entry:
            return
        defaults = {field.get("name"): field.get("default") for field in schema_entry.get("fields", [])}
        seed_default = defaults.get("seed")
        if isinstance(seed_default, int) and seed_default >= 0:
            self._seed_spin.setValue(seed_default)
        timesteps_default = defaults.get("total_timesteps")
        if isinstance(timesteps_default, int) and timesteps_default > 0:
            self._timesteps_spin.setValue(timesteps_default)

    def _update_wandb_controls(self) -> None:
        track_enabled = self._track_wandb_checkbox.isChecked()
        base_fields = (
            self._wandb_project_input,
            self._wandb_entity_input,
            self._wandb_run_name_input,
            self._wandb_api_key_input,
        )
        for field in base_fields:
            field.setEnabled(track_enabled)
        self._wandb_use_vpn_checkbox.setEnabled(track_enabled)
        if not track_enabled:
            self._wandb_use_vpn_checkbox.setChecked(False)
        vpn_enabled = track_enabled and self._wandb_use_vpn_checkbox.isChecked()
        for field in (self._wandb_http_proxy_input, self._wandb_https_proxy_input):
            field.setEnabled(vpn_enabled)

    def _on_track_wandb_toggled(self, checked: bool) -> None:
        _ = checked
        self._update_wandb_controls()

    def _on_wandb_vpn_toggled(self, checked: bool) -> None:
        _ = checked
        self._update_wandb_controls()

    def _on_fastlane_toggled(self, checked: bool) -> None:
        """Handle telemetry mode toggle - disable video controls when fastlane is off."""
        _ = checked
        self._update_video_mode_controls()

    def _update_video_mode_controls(self) -> None:
        # Check if fastlane (telemetry mode) is enabled
        fastlane_enabled = self._fastlane_checkbox.isChecked() if hasattr(self, "_fastlane_checkbox") else True

        # If fastlane is disabled, disable all video-related controls
        if not fastlane_enabled:
            self._video_mode_combo.setEnabled(False)
            self._grid_limit_spin.setEnabled(False)
            self._fastlane_slot_spin.setEnabled(False)
            return

        # Fastlane is enabled - apply normal mode-based logic
        self._video_mode_combo.setEnabled(True)
        mode_data = self._video_mode_combo.currentData() if hasattr(self, "_video_mode_combo") else None
        mode = mode_data if isinstance(mode_data, str) else VideoModes.SINGLE
        show_grid = mode == VideoModes.GRID
        show_probe = mode != VideoModes.GRID
        self._grid_limit_spin.setEnabled(show_grid)
        self._fastlane_slot_spin.setEnabled(show_probe)

    def _register_num_env_widget(self, widget: QtWidgets.QWidget) -> None:
        if isinstance(widget, QtWidgets.QSpinBox):
            self._num_env_widget = widget
            widget.valueChanged.connect(self._handle_num_envs_changed)
            self._handle_num_envs_changed(widget.value())
        else:
            self._num_env_widget = None

    def _set_video_mode(self, mode: str) -> None:
        if not isinstance(mode, str):
            return
        index = self._video_mode_combo.findData(mode)
        if index < 0 or self._video_mode_combo.currentIndex() == index:
            self._update_video_mode_controls()
            return
        blocked = self._video_mode_combo.blockSignals(True)
        self._video_mode_combo.setCurrentIndex(index)
        self._video_mode_combo.blockSignals(blocked)
        self._update_video_mode_controls()

    def _set_grid_limit(self, limit: int) -> None:
        bounded = max(1, min(int(limit), self._grid_limit_spin.maximum()))
        blocked = self._grid_limit_spin.blockSignals(True)
        self._grid_limit_spin.setValue(bounded)
        self._grid_limit_spin.blockSignals(blocked)

    def _set_probe_index(self, index: int) -> None:
        bounded = max(self._fastlane_slot_spin.minimum(), min(int(index), self._fastlane_slot_spin.maximum()))
        blocked = self._fastlane_slot_spin.blockSignals(True)
        self._fastlane_slot_spin.setValue(bounded)
        self._fastlane_slot_spin.blockSignals(blocked)

    def _handle_num_envs_changed(self, value: int) -> None:
        count = max(1, int(value))
        if count <= 1:
            self._set_grid_limit(1)
            self._set_video_mode(VideoModes.SINGLE)
            self._set_probe_index(0)
            return

        self._set_grid_limit(count)
        current_slot = min(self._fastlane_slot_spin.value(), count - 1)
        self._set_probe_index(current_slot)
        self._set_video_mode(VideoModes.GRID)

    def _populate_cleanrl_environment_selectors(self, preferred_env: Optional[str]) -> None:
        family = self._family_for_env_id(preferred_env)
        self._populate_cleanrl_family_combo(family)
        self._rebuild_cleanrl_env_combo(family, preferred_env)

    def _family_for_env_id(self, env_id: Optional[str]) -> EnvironmentFamily | None:
        if env_id:
            inferred = _env_family_from_env_id(env_id)
            if inferred is not None:
                return inferred
        families = list(self._cleanrl_family_index.keys())
        return families[0] if families else None

    def _cleanrl_family_sort_key(self, family: EnvironmentFamily | None) -> tuple[str, str]:
        if family is None:
            return ("", "")
        return (family.value, family.value)

    def _populate_cleanrl_family_combo(self, preferred: Optional[EnvironmentFamily | None]) -> None:
        if self._env_family_combo is None:
            return
        families = sorted(self._cleanrl_family_index.keys(), key=self._cleanrl_family_sort_key)
        if not families:
            self._env_family_combo.blockSignals(True)
            self._env_family_combo.clear()
            self._env_family_combo.blockSignals(False)
            return
        target = preferred if preferred in families else families[0]
        self._selected_cleanrl_family = target
        self._env_family_combo.blockSignals(True)
        self._env_family_combo.clear()
        for family in families:
            self._env_family_combo.addItem(_format_cleanrl_family_label(family), family)
        index = self._env_family_combo.findData(target)
        if index < 0:
            index = 0
        self._env_family_combo.setCurrentIndex(index)
        self._env_family_combo.blockSignals(False)

    def _rebuild_cleanrl_env_combo(
        self, family: EnvironmentFamily | None, preferred_env: Optional[str]
    ) -> None:
        options = self._cleanrl_family_index.get(family, [])
        if not options:
            self._env_combo.blockSignals(True)
            self._env_combo.clear()
            self._env_combo.blockSignals(False)
            return
        env_ids = [env_id for _, env_id in options]
        target = preferred_env if preferred_env in env_ids else env_ids[0]
        self._env_combo.blockSignals(True)
        self._env_combo.clear()
        for label, env_id in options:
            self._env_combo.addItem(label, env_id)
        self._env_combo.blockSignals(False)
        index = self._env_combo.findData(target)
        if index < 0:
            index = 0
        self._env_combo.setCurrentIndex(index)

    def _on_env_family_changed(self, index: int) -> None:
        data = self._env_family_combo.itemData(index)
        family = data if isinstance(data, EnvironmentFamily) else None
        if family == self._selected_cleanrl_family:
            return
        self._selected_cleanrl_family = family
        self._rebuild_cleanrl_env_combo(family, None)

    def _collect_state(self) -> _FormState:
        algo = self._algo_combo.currentText().strip()
        env_data = self._env_combo.currentData()
        env_id = str(env_data) if env_data is not None else ""
        seed_value = int(self._seed_spin.value())
        selected_seed: Optional[int] = seed_value if seed_value > 0 else None
        notes = self._notes_edit.toPlainText().strip() or None
        worker_id_value = self._worker_id_input.text().strip() or None
        agent_id_value = self._agent_id_input.text().strip() or None
        wandb_project = self._wandb_project_input.text().strip() or None
        wandb_entity = self._wandb_entity_input.text().strip() or None
        wandb_run_name = self._wandb_run_name_input.text().strip() or None
        wandb_api_key = self._wandb_api_key_input.text().strip() or None
        use_wandb_vpn = self._wandb_use_vpn_checkbox.isChecked()
        raw_http_proxy = self._wandb_http_proxy_input.text().strip()
        raw_https_proxy = self._wandb_https_proxy_input.text().strip()
        wandb_http_proxy = None
        wandb_https_proxy = None
        if use_wandb_vpn:
            fallback_http = os.environ.get("WANDB_VPN_HTTP_PROXY", "").strip()
            fallback_https = os.environ.get("WANDB_VPN_HTTPS_PROXY", "").strip()
            wandb_http_proxy = raw_http_proxy or fallback_http or None
            wandb_https_proxy = raw_https_proxy or fallback_https or None

        algo_params: Dict[str, Any] = {}
        for key, widget in self._algo_param_inputs.items():
            if isinstance(widget, QtWidgets.QSpinBox):
                algo_params[key] = int(widget.value())
            elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                algo_params[key] = float(widget.value())
            elif isinstance(widget, QtWidgets.QCheckBox):
                algo_params[key] = widget.isChecked()
            elif isinstance(widget, QtWidgets.QLineEdit):
                algo_params[key] = widget.text().strip()

        # Save model based on checkbox
        algo_params["save_model"] = self._save_model_checkbox.isChecked()

        # NOTE: procedural_generation is communicated via the
        # CLEANRL_PROCEDURAL_GENERATION env-var (see _build_environment()),
        # NOT as an algo_param — tyro would reject the unknown flag.

        capture_video_widget = getattr(self, "_capture_video_checkbox", None)
        capture_video_enabled = (
            bool(capture_video_widget.isChecked())
            if isinstance(capture_video_widget, QtWidgets.QCheckBox)
            else False
        )
        fastlane_only = bool(self._fastlane_checkbox.isChecked())
        fastlane_slot = int(self._fastlane_slot_spin.value())
        video_mode_data = self._video_mode_combo.currentData()
        video_mode = video_mode_data if isinstance(video_mode_data, str) else VideoModes.SINGLE
        grid_limit = int(self._grid_limit_spin.value())

        return _FormState(
            algo=algo,
            env_id=env_id,
            total_timesteps=int(self._timesteps_spin.value()),
            seed=selected_seed,
            agent_id=agent_id_value or None,
            worker_id=worker_id_value,
            use_gpu=self._use_gpu_checkbox.isChecked(),
            capture_video=capture_video_enabled,
            track_tensorboard=self._tensorboard_checkbox.isChecked(),
            track_wandb=self._track_wandb_checkbox.isChecked(),
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_run_name=wandb_run_name,
            wandb_api_key=wandb_api_key,
            wandb_http_proxy=wandb_http_proxy,
            wandb_https_proxy=wandb_https_proxy,
            use_wandb_vpn=use_wandb_vpn,
            notes=notes,
            algo_params=algo_params,
            fastlane_only=fastlane_only,
            fastlane_slot=fastlane_slot,
            fastlane_video_mode=video_mode,
            fastlane_grid_limit=grid_limit,
        )

    def _build_config(self, state: _FormState, *, run_id: Optional[str] = None) -> Dict[str, Any]:
        run_id = run_id or _generate_run_id("cleanrl", state.algo)

        extras: Dict[str, Any] = {"cuda": state.use_gpu}
        if state.track_tensorboard:
            extras["tensorboard_dir"] = "tensorboard"
        if state.track_wandb:
            extras["track_wandb"] = True
        if state.wandb_project:
            extras["wandb_project_name"] = state.wandb_project
        if state.wandb_entity:
            extras["wandb_entity"] = state.wandb_entity
        if state.wandb_run_name:
            extras["wandb_run_name"] = state.wandb_run_name
        if state.use_wandb_vpn:
            extras["wandb_use_vpn_proxy"] = True
        if state.wandb_api_key:
            extras["wandb_api_key"] = state.wandb_api_key
        if state.use_wandb_vpn and state.wandb_http_proxy:
            extras["wandb_http_proxy"] = state.wandb_http_proxy
        if state.use_wandb_vpn and state.wandb_https_proxy:
            extras["wandb_https_proxy"] = state.wandb_https_proxy
        if state.notes:
            extras["notes"] = state.notes
        if state.algo_params:
            extras["algo_params"] = state.algo_params
        extras["fastlane_only"] = bool(state.fastlane_only)
        extras["fastlane_slot"] = int(state.fastlane_slot)
        extras["fastlane_video_mode"] = state.fastlane_video_mode
        extras["fastlane_grid_limit"] = int(state.fastlane_grid_limit)

        worker_config: Dict[str, Any] = {
            "run_id": run_id,
            "algo": state.algo,
            "env_id": state.env_id,
            "total_timesteps": state.total_timesteps,
            "extras": extras,
        }
        if state.seed is not None:
            worker_config["seed"] = state.seed
        if state.worker_id:
            worker_config["worker_id"] = state.worker_id
        if state.agent_id:
            worker_config["agent_id"] = state.agent_id
            worker_config.setdefault("extras", extras).setdefault("agent_id", state.agent_id)

        metadata = {
            "ui": {
                "worker_id": state.worker_id or "cleanrl_worker",
                "agent_id": state.agent_id or "cleanrl_agent",
                "algo": state.algo,
                "env_id": state.env_id,
                "fastlane_only": state.fastlane_only,
                "fastlane_slot": int(state.fastlane_slot),
                "fastlane_video_mode": state.fastlane_video_mode,
                "fastlane_grid_limit": int(state.fastlane_grid_limit),
            },
            "worker": {
                "worker_id": state.worker_id or "cleanrl_worker",
                "module": "cleanrl_worker.cli",
                "use_grpc": True,
                "grpc_target": "127.0.0.1:50055",
                "config": worker_config,
            },
        }

        tensorboard_relpath = None
        if state.track_tensorboard:
            tensorboard_relpath = f"var/trainer/runs/{run_id}/tensorboard"

        metadata["artifacts"] = {
            "tensorboard": {
                "enabled": state.track_tensorboard,
                "relative_path": tensorboard_relpath,
            },
            "wandb": {
                "enabled": state.track_wandb,
                "run_path": None,
                "use_vpn_proxy": state.use_wandb_vpn,
                "http_proxy": state.wandb_http_proxy,
                "https_proxy": state.wandb_https_proxy,
            },
            "notes": state.notes,
        }

        num_envs_value = state.algo_params.get("num_envs")
        num_envs_int = 1
        if isinstance(num_envs_value, (int, float)):
            try:
                candidate = int(num_envs_value)
                if candidate > 0:
                    num_envs_int = candidate
            except (TypeError, ValueError):
                num_envs_int = 1
        num_envs_str = str(num_envs_int)

        # Get procedural generation setting from algo_params
        procedural_gen = state.algo_params.get("procedural_generation", True)

        environment: Dict[str, Any] = {
            "CLEANRL_RUN_ID": run_id,
            "CLEANRL_AGENT_ID": state.agent_id or "cleanrl_agent",
            "TRACK_TENSORBOARD": "1" if state.track_tensorboard else "0",
            "TRACK_WANDB": "1" if state.track_wandb else "0",
            "WANDB_MODE": "online" if state.track_wandb else "offline",
            "WANDB_DISABLE_GYM": "true",
            "CLEANRL_NUM_ENVS": num_envs_str,
            "CLEANRL_PROCEDURAL_GENERATION": "1" if procedural_gen else "0",
        }
        # Add seed to environment if specified
        if state.seed is not None:
            environment["CLEANRL_SEED"] = str(state.seed)
        apply_fastlane_environment(
            environment,
            fastlane_only=state.fastlane_only,
            fastlane_slot=int(state.fastlane_slot),
            video_mode=state.fastlane_video_mode,
            grid_limit=int(state.fastlane_grid_limit),
        )
        if state.wandb_api_key:
            environment["WANDB_API_KEY"] = state.wandb_api_key
        if state.use_wandb_vpn and state.wandb_http_proxy:
            environment.update(
                {
                    "WANDB_HTTP_PROXY": state.wandb_http_proxy,
                    "HTTP_PROXY": state.wandb_http_proxy,
                    "http_proxy": state.wandb_http_proxy,
                }
            )
        if state.use_wandb_vpn and state.wandb_https_proxy:
            environment.update(
                {
                    "WANDB_HTTPS_PROXY": state.wandb_https_proxy,
                    "HTTPS_PROXY": state.wandb_https_proxy,
                    "https_proxy": state.wandb_https_proxy,
                }
            )

        entry_point = sys.executable
        arguments = ["-m", "cleanrl_worker.cli"]

        config: Dict[str, Any] = {
            "run_name": run_id,
            "entry_point": entry_point,
            "arguments": arguments,
            "environment": environment,
            "resources": {
                "cpus": 4,
                "memory_mb": 4096,
                "gpus": {"requested": 0, "mandatory": False},
            },
            "metadata": metadata,
            "artifacts": {
                "output_prefix": f"runs/{run_id}",
                "persist_logs": True,
                "keep_checkpoints": False,
            },
        }

        # Emit UI/telemetry path summaries using structured log constants so callers can
        # filter by code (e.g., LOG734 / LOG735) when inspecting logs.
        self.log_constant(
            LOG_UI_TRAIN_FORM_UI_PATH,
            extra={
                "run_name": run_id,
                "cuda": state.use_gpu,
                "track_tensorboard": state.track_tensorboard,
                "track_wandb": state.track_wandb,
                "agent_id": state.agent_id or "cleanrl_agent",
            },
        )
        self.log_constant(
            LOG_UI_TRAIN_FORM_TELEMETRY_PATH,
            extra={
                "run_name": run_id,
                "env_id": state.env_id,
                "total_timesteps": state.total_timesteps,
                "has_algo_params": bool(state.algo_params),
            },
        )

        return config

    def _run_validation(
        self,
        state: _FormState,
        *,
        run_id: str,
        persist_config: bool,
    ) -> tuple[bool, Optional[Dict[str, Any]]]:
        config = self._build_config(state, run_id=run_id)

        self._validation_status_label.setText("Running CleanRL dry-run validation…")
        # Structured trace before invoking validator
        self.log_constant(
            LOG_UI_TRAIN_FORM_TRACE,
            message="Starting CleanRL dry-run",
            extra={"run_id": run_id, "algo": state.algo, "env_id": state.env_id},
        )
        self._validation_status_label.setStyleSheet("color: #1565c0;")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            success, output = run_cleanrl_dry_run(config)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        self._set_validation_result(success, output)
        self._append_validation_notes(success, output)

        if persist_config and success:
            self._last_config = copy.deepcopy(config)
        elif not persist_config:
            self._last_config = None

        return success, (copy.deepcopy(config) if success else None)

    def _set_validation_result(self, success: bool, output: str) -> None:
        self._last_validation_output = output or ""
        if success:
            self._validation_status_label.setText("Dry-run validation succeeded.")
            self._validation_status_label.setStyleSheet("color: #2e7d32;")
            # Emit success info for downstream listeners that filter by code
            self.log_constant(
                LOG_UI_TRAIN_FORM_INFO,
                message="CleanRL dry-run validation succeeded",
            )
        else:
            self._validation_status_label.setText(
                "Dry-run validation failed. Check the details returned by cleanrl_worker."
            )
            self._validation_status_label.setStyleSheet("color: #c62828;")
            # Emit error with structured code to enable catch-by-code workflows
            snippet = (output or "").strip()
            self.log_constant(
                LOG_UI_TRAIN_FORM_ERROR,
                message="CleanRL dry-run validation failed",
                extra={"output": snippet[:1000]},
            )

    def _append_validation_notes(self, success: bool, output: str) -> None:
        status = "SUCCESS" if success else "FAILED"
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        details = output.strip() if output else ("Dry-run completed." if success else "Dry-run failed without output.")
        entry = (
            f"[Dry-Run {status} — {timestamp}]\n"
            f"{details}\n"
            f"{'-' * 40}\n"
        )
        self._notes_edit.appendPlainText(entry)

    def _on_algorithm_changed(self, algo: str) -> None:
        self._rebuild_algo_params(algo)
        self._apply_schema_defaults(algo)
        self._algo_help_text.setHtml(get_algo_doc(algo))

    def _handle_accept(self) -> None:
        state = self._collect_state()

        if not state.algo:
            self.log_constant(
                LOG_UI_TRAIN_FORM_WARNING,
                message="Accept rejected: algorithm not selected",
            )
            QtWidgets.QMessageBox.warning(
                self,
                "Algorithm Required",
                "Select a CleanRL algorithm before launching.",
            )
            return
        if not state.env_id:
            self.log_constant(
                LOG_UI_TRAIN_FORM_WARNING,
                message="Accept rejected: environment not selected",
            )
            QtWidgets.QMessageBox.warning(
                self,
                "Environment Required",
                "Specify a Gymnasium environment id (e.g. CartPole-v1).",
            )
            return
        run_id = _generate_run_id("cleanrl", state.algo)
        self.log_constant(
            LOG_UI_TRAIN_FORM_INFO,
            message="Accepting training config",
            extra={"algo": state.algo, "env_id": state.env_id, "run_id": run_id},
        )

        success, config = self._run_validation(state, run_id=run_id, persist_config=True)
        if not success:
            self._last_config = None
            return

        self._last_config = config
        self.accept()

    def _on_validate_clicked(self) -> None:
        state = self._collect_state()

        if not state.algo or not state.env_id:
            self.log_constant(
                LOG_UI_TRAIN_FORM_WARNING,
                message="Validation rejected: incomplete configuration",
                extra={"has_algo": bool(state.algo), "has_env_id": bool(state.env_id)},
            )
            QtWidgets.QMessageBox.warning(
                self,
                "Incomplete Configuration",
                "Select an algorithm and environment before running validation.",
            )
            return
        run_id = _generate_run_id("cleanrl", state.algo)

        self._run_validation(state, run_id=run_id, persist_config=False)

    def get_config(self) -> Dict[str, Any]:
        """Return the trainer payload generated by the form."""

        if self._last_config is not None:
            return copy.deepcopy(self._last_config)

        state = self._collect_state()
        return self._build_config(state)

    def _rebuild_algo_params(self, algo: str) -> None:
        self._clear_layout(self._algo_param_layout)
        self._algo_param_inputs.clear()
        self._num_env_widget = None
        self._algo_env_widget = None
        schema_entry = _CLEANRL_SCHEMAS.get(algo)
        if schema_entry and self._populate_params_from_schema(schema_entry.get("fields", [])):
            self._algo_param_group.setVisible(bool(self._algo_param_inputs))
            return

        specs = _ALGO_PARAM_SPECS.get(algo, ())
        capture_widget: Optional[QtWidgets.QWidget] = None
        if not specs:
            self._algo_param_group.setVisible(False)
            self._reset_capture_video_slot(None)
            return

        self._algo_param_group.setVisible(True)

        entries: list[tuple[str, QtWidgets.QWidget, bool]] = []
        for spec in specs:
            widget: QtWidgets.QWidget
            if spec.field_type is int:
                spin = QtWidgets.QSpinBox(self)
                spin.setRange(-1_000_000_000, 1_000_000_000)
                spin.setValue(int(spec.default))
                if spec.tooltip:
                    spin.setToolTip(spec.tooltip)
                widget = spin
            elif spec.field_type is float:
                spin = QtWidgets.QDoubleSpinBox(self)
                spin.setDecimals(6)
                spin.setRange(-1e9, 1e9)
                spin.setSingleStep(
                    abs(spec.default) / 10 if isinstance(spec.default, (int, float)) and spec.default else 0.1
                )
                spin.setValue(float(spec.default))
                if spec.tooltip:
                    spin.setToolTip(spec.tooltip)
                widget = spin
            elif spec.field_type is bool:
                checkbox = QtWidgets.QCheckBox(spec.label, self)
                checkbox.setChecked(bool(spec.default))
                if spec.tooltip:
                    checkbox.setToolTip(spec.tooltip)
                widget = checkbox
            else:
                line = QtWidgets.QLineEdit(self)
                line.setText(str(spec.default))
                if spec.tooltip:
                    line.setToolTip(spec.tooltip)
                widget = line

            self._algo_param_inputs[spec.key] = widget
            if spec.key == "num_envs":
                self._register_num_env_widget(widget)
            if spec.key == "capture_video":
                capture_widget = widget
                continue

            entries.append((spec.label, widget, isinstance(widget, QtWidgets.QCheckBox)))

        columns = 2
        for idx, (label, widget, is_checkbox) in enumerate(entries):
            cell = QtWidgets.QWidget(self._algo_param_group)
            cell_layout = QtWidgets.QVBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(2)
            if not is_checkbox:
                label_widget = QtWidgets.QLabel(label, cell)
                label_widget.setStyleSheet("font-weight: 600;")
                cell_layout.addWidget(label_widget)
            cell_layout.addWidget(widget)
            row = idx // columns
            col = idx % columns
            self._algo_param_layout.addWidget(cell, row, col)

        for col_idx in range(columns):
            self._algo_param_layout.setColumnStretch(col_idx, 1)

        self._capture_video_checkbox = capture_widget
        self._reset_capture_video_slot(capture_widget)


__all__ = [
    "CleanRlTrainForm",
    "get_cleanrl_environment_choices",
    "get_cleanrl_environment_family_index",
]


# Import from factory module directly — importing gym_gui.ui.forms would
# trigger eager loading of ALL worker forms (including ray_train_form which
# pulls in ray -> rllib -> wandb, causing noisy Pydantic warnings).
from gym_gui.ui.forms.factory import get_worker_form_factory

_factory = get_worker_form_factory()
if not _factory.has_train_form("cleanrl_worker"):
    _factory.register_train_form(
        "cleanrl_worker",
        lambda parent=None, **kwargs: CleanRlTrainForm(parent=parent, **kwargs),
    )
