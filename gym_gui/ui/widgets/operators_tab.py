"""Operators tab widget for configuring action-selecting entities.

This module provides the OperatorsTab widget that allows configuring and
running multiple operators (LLM agents, RL policies) for scientific comparison.

Operators are the action-selecting entities in MOSAIC - they represent
who or what controls the agent in an environment.
"""

from __future__ import annotations

import logging
import random
from functools import partial
from typing import Optional

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import pyqtSignal

from gym_gui.logging_config.helpers import log_constant
from gym_gui.logging_config.log_constants import (
    LOG_OP_GRID_CONFIG_DIALOG_OPENED,
    LOG_OP_GRID_CONFIG_STATE_APPLIED,
    LOG_SCRIPT_AUTO_EXECUTION_COMPLETED,
    LOG_SCRIPT_AUTO_EXECUTION_STARTED,
    LOG_SCRIPT_PARSED,
    LOG_UI_BOARD_CONFIG_DIALOG_OPENED,
    LOG_UI_BOARD_CONFIG_STATE_APPLIED,
)
from gym_gui.services.operator import OperatorConfig
from gym_gui.ui.widgets.keyboard_assignment_widget import KeyboardAssignmentWidget
from gym_gui.ui.widgets.operator_config_widget import OperatorConfigWidget, VLLMServerInfo
from gym_gui.ui.widgets.script_experiment_widget import ScriptExperimentWidget
from gym_gui.ui.widgets.vllm_server_widget import VLLMServerWidget

# Use operators namespace for dedicated operators.log routing
_LOGGER = logging.getLogger("gym_gui.operators.operators_tab")
_log = partial(log_constant, _LOGGER)


class OperatorsTab(QtWidgets.QWidget):
    """Main tab for configuring Operators.

    Operators are the action-selecting entities in MOSAIC - they represent
    who or what controls the agent in an environment.

    Scientific Execution Model (inspired by BALROG):
    - Shared seed ensures all operators start with identical initial conditions
    - Step All advances all operators by exactly one step (lock-step execution)
    - Reset All resets all operators to the same seed for fair comparison
    - No arbitrary timing delays - steps are explicitly controlled
    """

    # Signals
    operators_changed = pyqtSignal(list)  # List[OperatorConfig]
    step_all_requested = pyqtSignal(int)  # Emit with current seed
    reset_all_requested = pyqtSignal(object)  # Emit with current seed (int or None)
    stop_operators_requested = pyqtSignal()
    initialize_operator_requested = pyqtSignal(str, object, object)  # operator_id, config, seed (int or None)
    step_player_requested = pyqtSignal(str, int)  # player_id, seed
    human_step_completed = pyqtSignal(str)  # operator_id - emitted when human completes their step
    human_action_requested = pyqtSignal(str, int)  # operator_id, action_index - request to step human operator
    human_step_parallel_requested = pyqtSignal()  # Request a parallel multi-agent step cycle (Human Step in parallel mode)
    keyboard_assignment_changed = pyqtSignal(str, object)  # (device_path, agent_id) - relayed from embedded KeyboardAssignmentWidget

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        # Allow this widget to expand vertically
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        self._step_count = 0
        self._is_running = False
        self._pettingzoo_mode = False
        self._current_player: str = ""
        # Human operator tracking
        self._has_human_operator = False
        self._human_step_completed = False
        self._human_operator_ids: list[str] = []  # IDs of human operators
        self._human_available_actions: list[int] = []  # Available action indices
        self._human_action_labels: list[str] = []  # Human-readable action labels

        # Turn-based execution tracking
        self._operator_order: list[str] = []  # Ordered list of operator IDs
        self._operator_types: dict[str, str] = {}  # operator_id -> "human", "llm", "rl", etc.
        self._operator_states: dict[str, str] = {}  # operator_id -> "stopped", "running"
        self._current_agent_index: int = 0  # Which agent's turn it is (0-based)

        # Parallel multi-agent mode flag (MultiGrid, MeltingPot, Overcooked)
        self._parallel_mode = False

        # Script-based execution is now handled by OperatorScriptExecutionManager
        # (no state needed in OperatorsTab)

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the UI layout."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Explanation section (collapsible, collapsed by default)
        self._explanation_toggle = QtWidgets.QToolButton(self)
        self._explanation_toggle.setText("What are Operators?")
        self._explanation_toggle.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self._explanation_toggle.setCheckable(True)
        self._explanation_toggle.setChecked(False)
        self._explanation_toggle.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self._explanation_toggle.toggled.connect(self._on_explanation_toggled)
        layout.addWidget(self._explanation_toggle)

        self._explanation_content = QtWidgets.QLabel(
            "<p><b>Operators</b> are the evaluation-time interface for controlling agents in environments. "
            "Each operator wraps a decision-making worker and exposes a unified "
            "<code>select_action(observation) → action</code> protocol. Supported worker types:</p>"
            "<ul>"
            "<li><b>LLM Agents</b> - GPT-4, Claude, Gemini with multi-agent coordination and Theory of Mind</li>"
            "<li><b>VLM Agents</b> - Vision-Language Models that reason over image observations</li>"
            "<li><b>RL Policies</b> - Trained neural network policies (XuanCe, CleanRL, Ray RLlib)</li>"
            "<li><b>Human Players</b> - Interactive human-in-the-loop control</li>"
            "<li><b>Random</b> - Uniform random action sampling baseline</li>"
            "<li><b>Passive</b> - Environment-aware no-op (do-nothing) agent</li>"
            "</ul>"
            "<p>Operators enable <b>heterogeneous evaluation</b>: mix different paradigms "
            "(e.g. LLM vs RL vs Human) in the same multi-agent environment. "
            "Add multiple operators below to compare their performance side-by-side.</p>"
            "<p><b>Policy Mappings for Multi-Agent RL:</b> When deploying RL policies in multi-agent scenarios, "
            "MOSAIC supports flexible policy-to-agent mappings:</p>"
            "<ul>"
            "<li><b>One-to-one Mapping</b> - Each agent has its own independent policy checkpoint (default)</li>"
            "<li><b>One-to-many Mapping</b> - Multiple agents share a single policy checkpoint via link groups</li>"
            "</ul>"
            "<p>Link groups are essential for multi-agent RL evaluation because MAPPO/IPPO checkpoints store "
            "all agents' policies in a single file. Linking agents prevents manual copy-paste errors, "
            "ensures automatic policy path updates, and enables complex team configurations "
            "(e.g., two independent teams with different policies).</p>",
            self
        )
        self._explanation_content.setWordWrap(True)
        self._explanation_content.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._explanation_content.setContentsMargins(20, 0, 0, 0)
        self._explanation_content.setVisible(False)
        layout.addWidget(self._explanation_content)

        # vLLM Servers section (collapsible, collapsed by default)
        self._vllm_toggle = QtWidgets.QToolButton(self)
        self._vllm_toggle.setText("vLLM Servers (Local Inference)")
        self._vllm_toggle.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self._vllm_toggle.setCheckable(True)
        self._vllm_toggle.setChecked(False)
        self._vllm_toggle.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self._vllm_toggle.toggled.connect(self._on_vllm_toggled)
        layout.addWidget(self._vllm_toggle)

        self._vllm_server_widget = VLLMServerWidget(max_servers=2, parent=self)
        self._vllm_server_widget.server_status_changed.connect(self._on_vllm_server_status_changed)
        self._vllm_server_widget.setContentsMargins(20, 0, 0, 0)
        self._vllm_server_widget.setVisible(False)
        layout.addWidget(self._vllm_server_widget)

        # Configure Operators section (always visible, takes remaining space)
        config_group = QtWidgets.QGroupBox("Configure Operators", self)
        # No custom styling - let Qt handle dark/light mode automatically
        config_group.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        config_layout = QtWidgets.QVBoxLayout(config_group)
        config_layout.setContentsMargins(8, 12, 8, 8)

        # Tab widget for Manual vs Script mode
        self._config_tabs = QtWidgets.QTabWidget(config_group)
        self._config_tabs.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )

        # Tab 1: Manual Mode (existing widget)
        self._operator_config_widget = OperatorConfigWidget(max_operators=8, parent=self._config_tabs)
        self._operator_config_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        self._operator_config_widget.operators_changed.connect(self._on_operators_config_changed)
        self._operator_config_widget.initialize_requested.connect(self._on_initialize_requested)
        self._operator_config_widget.configure_requested.connect(self._on_configure_requested)
        self._operator_config_widget.vllm_refresh_requested.connect(self.refresh_vllm_servers)
        self._config_tabs.addTab(self._operator_config_widget, "Manual Mode")

        # Tab 2: Script Experiments (clean one-click execution)
        # Script execution logic is handled by OperatorScriptExecutionManager (not OperatorsTab)
        self._script_experiment_widget = ScriptExperimentWidget(parent=self._config_tabs)
        self._script_experiment_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        # MainWindow will connect to the execution manager's signals directly
        self._config_tabs.addTab(self._script_experiment_widget, "Script Experiments")

        config_layout.addWidget(self._config_tabs, 1)  # Stretch factor for expansion

        # Add with stretch=1 so config_group expands to fill available vertical space
        layout.addWidget(config_group, 1)

        # Keyboard Assignment for multi-human operators (hidden by default).
        # Shown when 2+ human operators are configured, allowing each USB keyboard
        # to be assigned to a different human agent via the Linux evdev subsystem.
        self._keyboard_assignment_widget = KeyboardAssignmentWidget(
            available_agents=[],
            parent=self,
        )
        self._keyboard_assignment_widget.setVisible(False)
        self._keyboard_assignment_widget.assignment_changed.connect(
            self.keyboard_assignment_changed.emit
        )
        layout.addWidget(self._keyboard_assignment_widget)

        # Scientific Execution Controls (inspired by BALROG methodology)
        exec_group = QtWidgets.QGroupBox("Execution Controls (Scientific Comparison)", self)
        # No custom styling - let Qt handle dark/light mode automatically
        exec_layout = QtWidgets.QVBoxLayout(exec_group)
        exec_layout.setContentsMargins(12, 12, 12, 12)

        # Seed row - for reproducibility like BALROG
        seed_row = QtWidgets.QHBoxLayout()
        seed_row.setSpacing(8)

        seed_label = QtWidgets.QLabel("Shared Seed:", exec_group)
        seed_label.setToolTip(
            "All operators use this seed for identical initial conditions.\n"
            "This ensures fair, reproducible side-by-side comparison."
        )
        seed_row.addWidget(seed_label)

        self._seed_spin = QtWidgets.QSpinBox(exec_group)
        self._seed_spin.setRange(0, 2147483647)  # Max int32
        self._seed_spin.setValue(42)  # Default seed like BALROG
        self._seed_spin.setToolTip(
            "Seed for random number generators.\n"
            "Same seed = same initial environment state for all operators."
        )
        seed_row.addWidget(self._seed_spin, 1)

        self._random_seed_button = QtWidgets.QPushButton("Random", exec_group)
        self._random_seed_button.setToolTip("Generate a new random seed")
        self._random_seed_button.clicked.connect(self._on_random_seed_clicked)
        seed_row.addWidget(self._random_seed_button)

        self._use_shared_seed_checkbox = QtWidgets.QCheckBox("Use Shared Seed", exec_group)
        self._use_shared_seed_checkbox.setChecked(True)
        self._use_shared_seed_checkbox.setToolTip(
            "When checked, all operators use the same seed for identical layouts.\n"
            "When unchecked, each operator gets a random layout."
        )
        seed_row.addWidget(self._use_shared_seed_checkbox)

        exec_layout.addLayout(seed_row)

        # Control buttons row
        button_row = QtWidgets.QHBoxLayout()
        button_row.setSpacing(8)

        self._reset_all_button = QtWidgets.QPushButton("Reset All", exec_group)
        self._reset_all_button.setToolTip(
            "Reset all operators to initial state with the shared seed.\n"
            "All environments will have identical starting conditions."
        )
        self._reset_all_button.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #FF9800; color: white; }"
        )
        self._reset_all_button.setEnabled(False)
        self._reset_all_button.clicked.connect(self._on_reset_all_clicked)
        button_row.addWidget(self._reset_all_button)

        self._step_all_button = QtWidgets.QPushButton("Step All", exec_group)
        self._step_all_button.setToolTip(
            "Step all AI operators (RL, LLM, VLM) until the next Human operator.\n"
            "Disabled when waiting for a Human to take their turn."
        )
        self._step_all_button.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #4CAF50; color: white; }"
        )
        self._step_all_button.setEnabled(False)
        self._step_all_button.clicked.connect(self._on_step_all_clicked)
        button_row.addWidget(self._step_all_button)

        # Human Step button - for human operators to take their turn
        self._human_step_button = QtWidgets.QPushButton("Human Step", exec_group)
        self._human_step_button.setToolTip(
            "Take a step as the current Human operator.\n"
            "Use keyboard (WASD/Arrows) or click action buttons below.\n"
            "Enabled only when it's a Human operator's turn."
        )
        self._human_step_button.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #FF5722; color: white; }"
        )
        self._human_step_button.setEnabled(False)
        self._human_step_button.setVisible(False)  # Hidden until human operator is configured
        self._human_step_button.clicked.connect(self._on_human_step_button_clicked)
        button_row.addWidget(self._human_step_button)

        # PettingZoo player-specific step buttons (hidden by default)
        self._step_player_0_btn = QtWidgets.QPushButton("Step White", exec_group)
        self._step_player_0_btn.setToolTip("Step player_0 (White) - their turn to move")
        self._step_player_0_btn.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #4CAF50; color: white; }"
        )
        self._step_player_0_btn.setEnabled(False)
        self._step_player_0_btn.setVisible(False)  # Hidden by default
        self._step_player_0_btn.clicked.connect(lambda: self._on_step_player_clicked("player_0"))
        button_row.addWidget(self._step_player_0_btn)

        self._step_player_1_btn = QtWidgets.QPushButton("Step Black", exec_group)
        self._step_player_1_btn.setToolTip("Step player_1 (Black) - their turn to move")
        self._step_player_1_btn.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #4CAF50; color: white; }"
        )
        self._step_player_1_btn.setEnabled(False)
        self._step_player_1_btn.setVisible(False)  # Hidden by default
        self._step_player_1_btn.clicked.connect(lambda: self._on_step_player_clicked("player_1"))
        button_row.addWidget(self._step_player_1_btn)

        self._stop_operators_button = QtWidgets.QPushButton("Stop All", exec_group)
        self._stop_operators_button.setToolTip("Stop all running operators and release resources")
        self._stop_operators_button.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #F44336; color: white; }"
        )
        self._stop_operators_button.setEnabled(False)
        self._stop_operators_button.clicked.connect(self._on_stop_operators_clicked)
        button_row.addWidget(self._stop_operators_button)

        exec_layout.addLayout(button_row)

        # Step counter and status row
        status_row = QtWidgets.QHBoxLayout()
        status_row.setSpacing(12)

        self._step_count_label = QtWidgets.QLabel("Steps: 0", exec_group)
        self._step_count_label.setStyleSheet("QLabel { font-weight: bold; }")
        status_row.addWidget(self._step_count_label)

        self._status_label = QtWidgets.QLabel("Ready", exec_group)
        # No custom styling - let Qt handle dark/light mode
        status_row.addWidget(self._status_label)

        # Turn indicator for PettingZoo multi-agent games
        self._turn_indicator_label = QtWidgets.QLabel("", exec_group)
        self._turn_indicator_label.setStyleSheet("QLabel { font-weight: bold; }")
        self._turn_indicator_label.setVisible(False)  # Hidden by default
        status_row.addWidget(self._turn_indicator_label)

        status_row.addStretch(1)

        exec_layout.addLayout(status_row)

        layout.addWidget(exec_group)

        # Human Actions Panel (visible only when human operator is waiting)
        self._human_actions_group = QtWidgets.QGroupBox("Human Actions", self)
        # Keep orange border for attention, but no background color for dark mode compatibility
        self._human_actions_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 2px solid #FF5722; }"
        )
        self._human_actions_group.setVisible(False)  # Hidden by default

        human_actions_layout = QtWidgets.QVBoxLayout(self._human_actions_group)
        human_actions_layout.setContentsMargins(8, 12, 8, 8)
        human_actions_layout.setSpacing(8)

        # Instructions label
        self._human_instructions_label = QtWidgets.QLabel(
            "Your turn! Click an action button or use keyboard shortcuts:",
            self._human_actions_group
        )
        self._human_instructions_label.setWordWrap(True)
        human_actions_layout.addWidget(self._human_instructions_label)

        # Action buttons container
        self._human_action_buttons_widget = QtWidgets.QWidget(self._human_actions_group)
        self._human_action_buttons_layout = QtWidgets.QGridLayout(self._human_action_buttons_widget)
        self._human_action_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self._human_action_buttons_layout.setSpacing(4)
        human_actions_layout.addWidget(self._human_action_buttons_widget)

        # Keyboard shortcut hint
        self._keyboard_hint_label = QtWidgets.QLabel(
            "Keyboard: WASD/Arrows for movement, Space/G for pickup, E for toggle",
            self._human_actions_group
        )
        self._keyboard_hint_label.setWordWrap(True)
        human_actions_layout.addWidget(self._keyboard_hint_label)

        layout.addWidget(self._human_actions_group)

        # No stretch at end - let config_group (with stretch=1) fill available space

    def _on_operators_config_changed(self, configs: list) -> None:
        """Handle operator configuration changes."""
        self.operators_changed.emit(configs)
        has_operators = len(configs) > 0
        self._reset_all_button.setEnabled(has_operators)

        # Build operator order, types, and initial states for turn-based execution
        self._operator_order = [cfg.operator_id for cfg in configs]
        self._operator_types = {}
        # Initialize all operators as "stopped" (will be updated to "running" when launched)
        for cfg in configs:
            # Determine operator type based on worker_id
            if cfg.worker_id == "human_worker":
                self._operator_types[cfg.operator_id] = "human"
            elif cfg.worker_id in ("cleanrl_worker", "rl_worker", "sb3_worker"):
                self._operator_types[cfg.operator_id] = "rl"
            else:
                # LLM, VLM, etc.
                self._operator_types[cfg.operator_id] = "llm"

            # Initialize state as "stopped" if not already tracked
            if cfg.operator_id not in self._operator_states:
                self._operator_states[cfg.operator_id] = "stopped"

        # Detect human operators (worker_id == "human_worker")
        self._human_operator_ids = [
            cfg.operator_id for cfg in configs
            if cfg.worker_id == "human_worker"
        ]
        self._has_human_operator = len(self._human_operator_ids) > 0
        self._human_step_completed = False  # Reset on config change
        self._current_agent_index = 0  # Reset to first agent

        # Show/hide Human Step button based on whether there are human operators
        self._human_step_button.setVisible(self._has_human_operator)

        # Show keyboard assignment widget when 2+ human agents are configured.
        # Single human agent doesn't need multi-keyboard — just use action buttons.
        # Note: _human_operator_ids are operator-level IDs, but for multi-agent envs
        # we need the per-agent human IDs (e.g., agent_0, agent_1) from within each config.
        human_agent_ids: list[str] = []
        for cfg in configs:
            human_agent_ids.extend(cfg.get_human_agents())
        if len(human_agent_ids) >= 2:
            self._keyboard_assignment_widget.set_available_agents(human_agent_ids)
            self._keyboard_assignment_widget.setVisible(True)
        else:
            self._keyboard_assignment_widget.setVisible(False)

        # Step All only enabled after Reset All is done
        if not has_operators:
            self._step_all_button.setEnabled(False)
            self._human_step_button.setEnabled(False)

    # Script execution methods removed - now handled by OperatorScriptExecutionManager
    # (see gym_gui/services/operator_script_execution_manager.py)

    @property
    def script_execution_manager(self):
        """Get the script execution manager for MainWindow to wire up signals."""
        return self._script_experiment_widget.execution_manager

    @property
    def keyboard_assignment_widget(self) -> KeyboardAssignmentWidget:
        """Access the keyboard assignment widget for evdev integration."""
        return self._keyboard_assignment_widget

    def _on_random_seed_clicked(self) -> None:
        """Generate a random seed."""
        new_seed = random.randint(0, 2147483647)
        self._seed_spin.setValue(new_seed)

    def _on_reset_all_clicked(self) -> None:
        """Handle Reset All button click.

        Always sends the spinbox seed so that workers produce the same layout
        shown by "Load Environment" preview.  The "Use Shared Seed" checkbox
        is a UI hint for the user; the actual seed is always deterministic.
        """
        seed = self._seed_spin.value()
        self._step_count = 0
        self._step_count_label.setText("Steps: 0")
        self._status_label.setText(f"Resetting with seed {seed}...")
        self._is_running = True
        self._human_step_completed = False  # Reset human step state
        self.reset_all_requested.emit(seed)

        # Enable stop button
        self._stop_operators_button.setEnabled(True)

        # Step All: enabled only if no human operators, or after human steps.
        # In parallel mode, Step All is always enabled (it collects AI actions
        # and shows the human action panel for the remaining human agents).
        if self._parallel_mode or not self._has_human_operator:
            self._step_all_button.setEnabled(True)
            self._status_label.setText(f"Running (seed={seed})")
        else:
            self._step_all_button.setEnabled(False)
            self._status_label.setText(f"Waiting for Human... (seed={seed})")

    def _on_step_all_clicked(self) -> None:
        """Handle Step All button click.

        For Human operators: After AI operators step, Step All is disabled
        until the human makes their next move.
        """
        if not self._is_running:
            return
        seed = self._seed_spin.value()
        self._step_count += 1
        self._step_count_label.setText(f"Steps: {self._step_count}")
        self.step_all_requested.emit(seed)

        # If there are human operators, disable Step All until human steps again.
        # In parallel mode, Step All stays enabled — the step cycle handles
        # collecting human actions via the embedded MultiAgentActionPanel.
        if self._has_human_operator and not self._parallel_mode:
            self._human_step_completed = False
            self._step_all_button.setEnabled(False)
            self._status_label.setText(f"Waiting for Human... (step {self._step_count})")

    def _on_stop_operators_clicked(self) -> None:
        """Handle Stop All button click."""
        self.stop_operators_requested.emit()
        self._is_running = False
        self._step_all_button.setEnabled(False)
        self._stop_operators_button.setEnabled(False)
        self._status_label.setText("Stopped")

        # Mark all operators as stopped
        for operator_id in self._operator_states:
            self._operator_states[operator_id] = "stopped"

    def _on_initialize_requested(self, operator_id: str, config: OperatorConfig) -> None:
        """Handle initialize request from an operator row.

        Always uses the spinbox seed so the preview matches the layout
        that the worker subprocess will produce on Reset All.
        """
        seed = self.get_current_seed()
        self.initialize_operator_requested.emit(operator_id, config, seed)

    def _on_configure_requested(self, operator_id: str, config: OperatorConfig) -> None:
        """Handle configure request from an operator row.

        Opens an environment-specific configuration dialog to set up custom
        starting positions/states. Supports both:
        - Board games (chess, Go, checkers) via BoardConfigDialogFactory
        - Grid environments (MiniGrid, BabyAI) via GridConfigDialogFactory
        """
        from gym_gui.ui.widgets.operators_board_config_form import BoardConfigDialogFactory
        from gym_gui.ui.widgets.operators_grid_config_form import GridConfigDialogFactory

        env_id = config.task  # e.g., "chess_v6" or "MiniGrid-Empty-8x8-v0"

        # Get current state if any (from worker settings)
        current_state = None
        if config.workers:
            first_worker_id = next(iter(config.workers.keys()))
            current_state = config.workers[first_worker_id].settings.get("initial_state")

        # Try board game factory first (chess, Go, checkers, etc.)
        if BoardConfigDialogFactory.supports(env_id):
            _log(
                LOG_UI_BOARD_CONFIG_DIALOG_OPENED,
                extra={"operator_id": operator_id, "game_id": env_id},
            )

            dialog = BoardConfigDialogFactory.create(env_id, current_state, self)
            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                custom_state = dialog.get_state()
                _log(
                    LOG_UI_BOARD_CONFIG_STATE_APPLIED,
                    extra={
                        "operator_id": operator_id,
                        "game_id": env_id,
                        "state_preview": custom_state[:50] if custom_state else "",
                    },
                )
                self._operator_config_widget.set_operator_initial_state(operator_id, custom_state)
            return

        # Try grid environment factory (MiniGrid, BabyAI, etc.)
        if GridConfigDialogFactory.supports(env_id):
            _log(
                LOG_OP_GRID_CONFIG_DIALOG_OPENED,
                extra={"operator_id": operator_id, "env_id": env_id},
            )

            # For grid environments, current_state is a dict (JSON serializable)
            import json
            initial_dict = None
            if current_state:
                try:
                    initial_dict = json.loads(current_state) if isinstance(current_state, str) else current_state
                except json.JSONDecodeError:
                    _LOGGER.warning(f"Could not parse initial state as JSON: {current_state[:50]}")

            dialog = GridConfigDialogFactory.create(env_id, initial_dict, self)
            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                custom_state_dict = dialog.get_state()
                # Store as JSON string for consistency with board games
                custom_state = json.dumps(custom_state_dict)
                _log(
                    LOG_OP_GRID_CONFIG_STATE_APPLIED,
                    extra={
                        "operator_id": operator_id,
                        "env_id": env_id,
                        "grid_size": f"{custom_state_dict.get('rows', '?')}x{custom_state_dict.get('cols', '?')}",
                    },
                )
                self._operator_config_widget.set_operator_initial_state(operator_id, custom_state)
            return

        # No factory supports this environment
        _LOGGER.warning(f"No configuration dialog for environment: {env_id}")

    def set_step_count(self, count: int) -> None:
        """Set the step count (called externally when steps complete)."""
        self._step_count = count
        self._step_count_label.setText(f"Steps: {count}")

    def set_status(self, status: str) -> None:
        """Set the status label text."""
        self._status_label.setText(status)

    def set_turn_indicator(self, player_id: str, visible: bool = True) -> None:
        """Set the turn indicator for PettingZoo multi-agent games.

        Args:
            player_id: Current player (e.g., "player_0", "player_1").
            visible: Whether to show the indicator.
        """
        if visible and player_id:
            # Map player_id to friendly name
            player_names = {
                "player_0": "White",
                "player_1": "Black",
            }
            friendly_name = player_names.get(player_id, player_id)
            self._turn_indicator_label.setText(f"Next: {friendly_name} ({player_id})")
            self._turn_indicator_label.setVisible(True)
        else:
            self._turn_indicator_label.setVisible(False)

    def set_pettingzoo_mode(self, enabled: bool) -> None:
        """Enable or disable PettingZoo multi-agent mode.

        When enabled:
        - Hides "Step All" button
        - Shows player-specific step buttons (Step White, Step Black)

        Args:
            enabled: True to enable PettingZoo mode, False for normal mode.
        """
        _LOGGER.debug("set_pettingzoo_mode: enabled=%s", enabled)
        self._pettingzoo_mode = enabled
        self._step_all_button.setVisible(not enabled)
        self._step_player_0_btn.setVisible(enabled)
        self._step_player_1_btn.setVisible(enabled)
        _LOGGER.debug(
            "set_pettingzoo_mode: step_all visible=%s, player btns visible=%s",
            not enabled, enabled,
        )

        if not enabled:
            # Reset to normal mode
            self._step_player_0_btn.setEnabled(False)
            self._step_player_1_btn.setEnabled(False)
            self._current_player = ""

    def set_current_player(self, player_id: str) -> None:
        """Set which player's turn it is (enables their button, disables the other).

        Args:
            player_id: The player whose turn it is ("player_0" or "player_1").
        """
        _LOGGER.debug(
            "set_current_player: player_id=%s",
            player_id,
            extra={"agent_id": player_id},
        )
        self._current_player = player_id

        if player_id == "player_0":
            self._step_player_0_btn.setEnabled(True)
            self._step_player_1_btn.setEnabled(False)
            _LOGGER.debug("Enabled player_0 button, disabled player_1 button")
        elif player_id == "player_1":
            self._step_player_0_btn.setEnabled(False)
            self._step_player_1_btn.setEnabled(True)
            _LOGGER.debug("Disabled player_0 button, enabled player_1 button")
        else:
            # Unknown player or game over
            self._step_player_0_btn.setEnabled(False)
            self._step_player_1_btn.setEnabled(False)
            _LOGGER.debug("Disabled both buttons (game over or unknown player)")

    def _on_step_player_clicked(self, player_id: str) -> None:
        """Handle click on a player-specific step button.

        Args:
            player_id: Which player's button was clicked.
        """
        seed = self._seed_spin.value()
        # Disable button immediately to prevent double-clicks
        if player_id == "player_0":
            self._step_player_0_btn.setEnabled(False)
        else:
            self._step_player_1_btn.setEnabled(False)
        # Note: _step_count is incremented only after the step actually completes
        # (via set_step_count), to keep it in sync with the render container's Step counter.
        self.step_player_requested.emit(player_id, seed)

    def get_current_seed(self) -> int:
        """Get the current seed value."""
        return self._seed_spin.value()

    def on_human_action_received(self, operator_id: str) -> None:
        """Called when a human operator completes their action via the UI.

        This enables the Step All button so AI operators can take their turn.

        Args:
            operator_id: The ID of the human operator that acted.
        """
        if not self._has_human_operator:
            return

        if operator_id not in self._human_operator_ids:
            _LOGGER.warning(f"Unknown human operator: {operator_id}")
            return

        _LOGGER.info(f"Human operator {operator_id} completed action")
        self._human_step_completed = True
        self._step_all_button.setEnabled(True)
        self._status_label.setText(f"Human acted - Ready to Step All (step {self._step_count})")

        # Hide the human actions panel until next step
        self._human_actions_group.setVisible(False)

        # Emit signal for any listeners
        self.human_step_completed.emit(operator_id)

    @property
    def has_human_operator(self) -> bool:
        """Check if there are any human operators configured."""
        return self._has_human_operator

    @property
    def human_operator_ids(self) -> list[str]:
        """Get the list of human operator IDs."""
        return self._human_operator_ids.copy()

    def _on_explanation_toggled(self, checked: bool) -> None:
        """Toggle explanation content visibility."""
        self._explanation_content.setVisible(checked)
        self._explanation_toggle.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if checked else QtCore.Qt.ArrowType.RightArrow
        )

    def _on_vllm_toggled(self, checked: bool) -> None:
        """Toggle vLLM server widget visibility."""
        self._vllm_server_widget.setVisible(checked)
        self._vllm_toggle.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if checked else QtCore.Qt.ArrowType.RightArrow
        )

    def _on_vllm_server_status_changed(self, server_id: int, status: str, base_url: str) -> None:
        """Handle vLLM server status changes.

        This allows the UI to update operator dropdowns when servers start/stop.
        Collects all server states and passes them to the operator config widget.
        """
        _LOGGER.info(f"vLLM Server {server_id} status: {status}, URL: {base_url}")

        # Collect all server info from the vLLM server widget
        servers: list[VLLMServerInfo] = []
        for sid, state in self._vllm_server_widget._server_states.items():
            if state.model_id:  # Only include servers with a model selected
                server_info = VLLMServerInfo(
                    server_id=sid,
                    port=state.port,
                    model_id=state.model_id,
                    base_url=f"http://127.0.0.1:{state.port}/v1",
                    status=state.status,
                )
                servers.append(server_info)

        # Update operator config widget with available servers
        self._operator_config_widget.set_vllm_servers(servers)

    def refresh_vllm_servers(self) -> None:
        """Manually refresh the vLLM server list for operator dropdowns.

        Call this to sync operator config with current vLLM server state.
        Useful when servers were started before operators were configured.
        """
        servers: list[VLLMServerInfo] = []
        for sid, state in self._vllm_server_widget._server_states.items():
            if state.model_id:  # Only include servers with a model selected
                server_info = VLLMServerInfo(
                    server_id=sid,
                    port=state.port,
                    model_id=state.model_id,
                    base_url=f"http://127.0.0.1:{state.port}/v1",
                    status=state.status,
                )
                servers.append(server_info)

        _LOGGER.debug(f"Refreshing vLLM servers: {len(servers)} available")
        self._operator_config_widget.set_vllm_servers(servers)

    def set_operator_environment_size(
        self, operator_id: str, width: int, height: int, container_size: int | None = None
    ) -> None:
        """Set the environment size for a specific operator.

        Args:
            operator_id: The operator's unique ID
            width: Rendered environment width in pixels (image size)
            height: Rendered environment height in pixels (image size)
            container_size: Optional container display size in pixels
        """
        self._operator_config_widget.set_operator_environment_size(
            operator_id, width, height, container_size
        )

    @property
    def vllm_server_widget(self) -> VLLMServerWidget:
        """Get the vLLM server management widget."""
        return self._vllm_server_widget

    @property
    def operator_config_widget(self) -> OperatorConfigWidget:
        """Get the operator configuration widget."""
        return self._operator_config_widget

    # --- Human Action Panel Methods ---

    def set_human_available_actions(
        self, actions: list[int], labels: list[str], show_panel: bool = True
    ) -> None:
        """Set the available actions for human operators and populate buttons.

        Args:
            actions: List of valid action indices.
            labels: Human-readable labels for each action.
            show_panel: Whether to show the action panel.
        """
        self._human_available_actions = actions
        self._human_action_labels = labels
        self._populate_human_action_buttons(actions, labels)

        if show_panel and self._has_human_operator:
            self._human_actions_group.setVisible(True)
            self._status_label.setText(f"Waiting for Human... (step {self._step_count})")

    def _populate_human_action_buttons(
        self, actions: list[int], labels: list[str]
    ) -> None:
        """Populate the human action buttons grid.

        Args:
            actions: List of action indices.
            labels: Human-readable labels for each action.
        """
        # Clear existing buttons
        while self._human_action_buttons_layout.count():
            item = self._human_action_buttons_layout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

        # Create buttons in a grid (4 columns max)
        cols = 4
        for i, (action_idx, label) in enumerate(zip(actions, labels)):
            row = i // cols
            col = i % cols

            btn = QtWidgets.QPushButton(f"{label}", self._human_action_buttons_widget)
            btn.setToolTip(f"Action {action_idx}: {label}")
            btn.setStyleSheet(
                "QPushButton { font-weight: bold; background-color: #FF5722; color: white; }"
            )
            btn.clicked.connect(lambda checked, a=action_idx: self._on_human_action_button_clicked(a))
            self._human_action_buttons_layout.addWidget(btn, row, col)

    def _on_human_action_button_clicked(self, action: int) -> None:
        """Handle click on a human action button.

        Args:
            action: The action index that was clicked.
        """
        if not self._human_operator_ids:
            _LOGGER.warning("Human action clicked but no human operators configured")
            return

        # Get the first human operator (for now, support single human)
        operator_id = self._human_operator_ids[0]
        _LOGGER.info(f"Human action button clicked: operator={operator_id}, action={action}")

        # Emit signal to main_window to process the action
        self.human_action_requested.emit(operator_id, action)

    def _on_human_step_button_clicked(self) -> None:
        """Handle Human Step button click.

        In parallel multi-agent mode, this triggers a full step cycle
        (collect AI actions → show human action panel → step env).

        In subprocess mode, this shows the single-agent action panel
        so the human can act via keyboard/buttons.
        """
        if not self._human_operator_ids:
            _LOGGER.warning("Human Step clicked but no human operators configured")
            return

        if self._parallel_mode:
            _LOGGER.info("Human Step (parallel mode) — emitting human_step_parallel_requested")
            self.human_step_parallel_requested.emit()
            return

        _LOGGER.info("Human Step button clicked - activating controller for human operator")

        # Show the action panel for discrete action environments
        self.show_human_actions_panel(show=True)

        # Update status
        self._status_label.setText("Human's Turn - Use keyboard/mouse or click action button")

    def show_human_actions_panel(self, show: bool = True) -> None:
        """Show or hide the human actions panel.

        Args:
            show: True to show, False to hide.
        """
        if self._has_human_operator:
            self._human_actions_group.setVisible(show)
            if show:
                self._status_label.setText(f"Waiting for Human... (step {self._step_count})")

    def hide_human_actions_panel(self) -> None:
        """Hide the human actions panel after action is taken."""
        self._human_actions_group.setVisible(False)

    # --- Parallel Multi-Agent Mode Panel Methods ---

    def set_parallel_mode(self, enabled: bool) -> None:
        """Enable or disable parallel multi-agent mode.

        When enabled:
        - Step All stays enabled even with human operators
        - Human Step emits human_step_parallel_requested instead of showing
          the single-agent action panel

        Args:
            enabled: True for parallel mode, False for subprocess mode.
        """
        self._parallel_mode = enabled
        _LOGGER.debug("set_parallel_mode: enabled=%s", enabled)

__all__ = ["OperatorsTab"]
