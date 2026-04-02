"""Multi-operator configuration widget for side-by-side agent comparison.

This module provides UI widgets for configuring N operators (LLM or RL workers)
that can run in parallel, each with its own render container.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import pyqtSignal  # type: ignore[attr-defined]
from qtpy import QtCore, QtGui, QtWidgets

from gym_gui.config.paths import VAR_MODELS_HF_CACHE
from gym_gui.constants.constants_operator import (
    BALROG_DEFAULT_TASK,
    BALROG_SUPPORTED_ENVS,
)
from gym_gui.logging_config.helpers import log_constant
from gym_gui.logging_config.log_constants import LOG_OPERATOR_VIEW_SIZE_CONFIGURED
from gym_gui.services.operator import LinkGroup, OperatorConfig, WorkerAssignment
from gym_gui.ui.widgets.multi_agent_action_panel import (
    COLOR_PALETTE,
    DEFAULT_AGENT_COLOR_NAMES,
)
from gym_gui.ui.worker_catalog.catalog import WorkerDefinition, get_worker_catalog


@dataclass
class VLLMServerInfo:
    """Information about a running vLLM server for operator assignment."""

    server_id: int  # 1-indexed (Server 1, Server 2, etc.)
    port: int
    model_id: str
    base_url: str
    status: str  # "running", "stopped", etc.

    @property
    def display_name(self) -> str:
        """Display name for dropdown: 'Server 1 - ModelName @ :8000'."""
        model_short = self.model_id.split("/")[-1] if "/" in self.model_id else self.model_id
        return f"Server {self.server_id} - {model_short} @ :{self.port}"

_LOGGER = logging.getLogger(__name__)


def scan_local_models() -> List[Tuple[str, str]]:
    """Scan var/models/huggingface for installed local models.

    Returns:
        List of (model_id, display_name) tuples for locally installed models.
    """
    models = []
    hf_cache = VAR_MODELS_HF_CACHE

    if not hf_cache.exists():
        _LOGGER.debug("HuggingFace model cache not found: %s", hf_cache)
        return models

    for model_dir in hf_cache.iterdir():
        if not model_dir.is_dir():
            continue
        if model_dir.name.startswith("."):
            continue

        # Skip non-model directories
        if model_dir.name in ("hub", "xet", "datasets"):
            continue

        # Convert directory name to HuggingFace model ID format
        # HuggingFace cache format: "models--Org--ModelName" -> "Org/ModelName"
        # Simple format: "Org--ModelName" -> "Org/ModelName"
        dir_name = model_dir.name

        # Handle HuggingFace cache format: models--Org--ModelName
        if dir_name.startswith("models--"):
            # Strip "models--" prefix and split remaining by first "--"
            remainder = dir_name[len("models--"):]
            if "--" in remainder:
                org, model_name = remainder.split("--", 1)
                model_id = f"{org}/{model_name}"
                display_name = model_name
            else:
                # Malformed, skip
                _LOGGER.debug("Skipping malformed cache dir: %s", dir_name)
                continue
        elif "--" in dir_name:
            # Simple format: "Org--ModelName"
            org, model_name = dir_name.split("--", 1)
            model_id = f"{org}/{model_name}"
            display_name = model_name
        elif dir_name.startswith("Llama"):
            # Meta Llama models often stored without org prefix
            model_id = f"meta-llama/{dir_name}"
            display_name = dir_name
        else:
            # Use as-is
            model_id = dir_name
            display_name = dir_name

        models.append((model_id, display_name))
        _LOGGER.debug("Found local model: %s", model_id)

    # Sort by display name
    models.sort(key=lambda x: x[1])
    return models


def get_vllm_models() -> List[Tuple[str, str]]:
    """Get available vLLM models (scanned from local cache).

    Returns:
        List of (model_id, display_name) tuples.
        Returns default suggestions if no local models found.
    """
    local_models = scan_local_models()
    if local_models:
        return local_models

    # Fallback if no models installed
    return [
        ("(No local models found)", "(Install models to var/models/huggingface)"),
    ]


# Maximum number of operators allowed
MAX_OPERATORS = 8

# PettingZoo Classic Games - turn-based multi-agent environments
# Each game has player IDs and human-readable labels
PETTINGZOO_GAMES: Dict[str, Dict[str, Any]] = {
    # Two-player board games
    "chess_v6": {
        "family": "classic",
        "players": ["player_0", "player_1"],
        "player_labels": {"player_0": "White", "player_1": "Black"},
    },
    "connect_four_v3": {
        "family": "classic",
        "players": ["player_0", "player_1"],
        "player_labels": {"player_0": "Yellow", "player_1": "Red"},
    },
    "go_v5": {
        "family": "classic",
        "players": ["black_0", "white_0"],
        "player_labels": {"black_0": "Black", "white_0": "White"},
    },
    "tictactoe_v3": {
        "family": "classic",
        "players": ["player_1", "player_2"],
        "player_labels": {"player_1": "X", "player_2": "O"},
    },
    "backgammon_v3": {
        "family": "classic",
        "players": ["player_0", "player_1"],
        "player_labels": {"player_0": "White", "player_1": "Black"},
    },
    "gin_rummy_v4": {
        "family": "classic",
        "players": ["player_0", "player_1"],
        "player_labels": {"player_0": "Player 1", "player_1": "Player 2"},
    },
    "texas_holdem_v4": {
        "family": "classic",
        "players": ["player_0", "player_1"],
        "player_labels": {"player_0": "Player 1", "player_1": "Player 2"},
    },
}

# All environment families with their environments
# Used by both RL and LLM operators
ENV_FAMILIES: Dict[str, Tuple[str, ...]] = {
    "babyai": (),  # Loaded dynamically from gymnasium
    "minigrid": (),  # Loaded dynamically from gymnasium
    "minihack": (),  # Loaded dynamically from gymnasium
    "crafter": ("CrafterReward-v1", "CrafterNoReward-v1"),
    "nle": (),  # Loaded dynamically from gymnasium
    "textworld": ("treasure_hunter", "the_cooking_game", "coin_collector"),
    "toytext": (
        "FrozenLake-v1",
        "Taxi-v3",
        "CliffWalking-v1",
        "Blackjack-v1",
    ),
    # PettingZoo Classic board games (Chess, Go, Connect Four, TicTacToe)
    "pettingzoo_classic": (
        "chess_v6", "connect_four_v3", "go_v5", "tictactoe_v3",
        "backgammon_v3", "gin_rummy_v4", "texas_holdem_v4",
    ),
    # OpenSpiel board games (Google DeepMind) + custom draughts variants
    # - open_spiel/* : Original OpenSpiel implementations via Shimmy
    # - draughts/*   : Custom implementations with proper rule variants
    "open_spiel": (
        "open_spiel/checkers",           # Original OpenSpiel (American rules only)
        "draughts/american_checkers",    # Custom: 8x8, no backward captures
        "draughts/russian_checkers",     # Custom: 8x8, men capture backward, flying kings
        "draughts/international_draughts",  # Custom: 10x10, backward captures, flying kings
    ),
    # mosaic_multigrid: competitive team sports (view_size=3, simultaneous stepping)
    # PyPI: https://pypi.org/project/mosaic-multigrid/
    "mosaic_multigrid": (
        # Original (Deprecated)
        "MosaicMultiGrid-Soccer-v0",
        "MosaicMultiGrid-Collect-v0",
        "MosaicMultiGrid-Collect-2vs2-v0",
        "MosaicMultiGrid-Collect-1vs1-v0",
        # IndAgObs v4.0.0 (Recommended for RL training)
        "MosaicMultiGrid-Soccer-2vs2-IndAgObs-v0",
        "MosaicMultiGrid-Soccer-1vs1-IndAgObs-v0",
        "MosaicMultiGrid-Collect-IndAgObs-v0",
        "MosaicMultiGrid-Collect-2vs2-IndAgObs-v0",
        "MosaicMultiGrid-Collect-1vs1-IndAgObs-v0",
        "MosaicMultiGrid-Basketball-3vs3-IndAgObs-v0",
        # TeamObs v4.0.0 (SMAC-style teammate awareness)
        "MosaicMultiGrid-Soccer-2vs2-TeamObs-v0",
        "MosaicMultiGrid-Collect-2vs2-TeamObs-v0",
        "MosaicMultiGrid-Basketball-3vs3-TeamObs-v0",
    ),
    # ini_multigrid: cooperative exploration environments (view_size=7, simultaneous stepping)
    # GitHub: https://github.com/ini/multigrid
    "ini_multigrid": (
        "MultiGrid-Empty-5x5-v0",
        "MultiGrid-Empty-Random-5x5-v0",
        "MultiGrid-Empty-6x6-v0",
        "MultiGrid-Empty-Random-6x6-v0",
        "MultiGrid-Empty-8x8-v0",
        "MultiGrid-Empty-16x16-v0",
        "MultiGrid-RedBlueDoors-6x6-v0",
        "MultiGrid-RedBlueDoors-8x8-v0",
        "MultiGrid-LockedHallway-2Rooms-v0",
        "MultiGrid-LockedHallway-4Rooms-v0",
        "MultiGrid-LockedHallway-6Rooms-v0",
        "MultiGrid-BlockedUnlockPickup-v0",
        "MultiGrid-Playground-v0",
    ),
    # Melting Pot multi-agent social scenarios (DeepMind)
    "meltingpot": (),  # Loaded dynamically from available substrates
    # Overcooked-AI cooperative cooking (UC Berkeley CHAI)
    "overcooked": (
        "overcooked/cramped_room",
        "overcooked/asymmetric_advantages",
        "overcooked/coordination_ring",
        "overcooked/forced_coordination",
        "overcooked/counter_circuit",
    ),
}


def _auto_detect_agent_count(env_family: str, env_id: str) -> int:
    """Auto-detect the number of agents in a multi-agent environment.

    Args:
        env_family: Environment family (e.g., "pettingzoo", "mosaic_multigrid")
        env_id: Environment ID (e.g., "chess_v6", "MosaicMultiGrid-Soccer-v0")

    Returns:
        Number of agents, or 0 if detection fails or single-agent
    """
    try:
        if env_family in ("pettingzoo", "pettingzoo_classic"):
            # PettingZoo / PettingZoo Classic: use predefined game info
            if env_id in PETTINGZOO_GAMES:
                return len(PETTINGZOO_GAMES[env_id]["players"])
            # PettingZoo Classic board games: always 2 players
            if env_family == "pettingzoo_classic":
                return 2
            return 0

        elif env_family in ("mosaic_multigrid", "ini_multigrid"):
            # MultiGrid: instantiate environment and query agent count
            from gym_gui.core.enums import GameId
            from gym_gui.core.factories.adapters import create_adapter

            # Map UI env_id to GameId enum
            try:
                game_id = GameId(env_id)
            except ValueError:
                # Fallback: unknown MultiGrid variant
                return 0

            adapter = create_adapter(game_id)
            adapter.load()
            num_agents = getattr(adapter, 'num_agents', 0)
            adapter.close()
            return num_agents

        elif env_family == "meltingpot":
            # Melting Pot: variable agent count (2-16)
            # Instantiate adapter to query
            from gym_gui.core.enums import GameId
            from gym_gui.core.factories.adapters import create_adapter

            try:
                game_id = GameId(env_id)
                adapter = create_adapter(game_id)
                adapter.load()
                num_agents = getattr(adapter, 'num_agents', 0)
                adapter.close()
                return num_agents
            except (ValueError, Exception):
                return 0

        elif env_family == "overcooked":
            # Overcooked-AI: always 2 agents
            return 2

        elif env_family == "open_spiel":
            # OpenSpiel board games: always 2 players
            return 2

        # Unknown multi-agent family
        return 0

    except Exception as e:
        # Log error but don't crash
        import logging
        logging.getLogger(__name__).warning(
            f"Failed to auto-detect agent count for {env_family}/{env_id}: {e}"
        )
        return 0


def _get_execution_mode(env_family: str) -> str:
    """Get the default execution mode for an environment family.

    Args:
        env_family: Environment family (e.g., "pettingzoo", "mosaic_multigrid")

    Returns:
        "aec" for turn-based, "parallel" for simultaneous
    """
    if env_family in ("pettingzoo", "pettingzoo_classic", "open_spiel"):
        return "aec"  # Turn-based by default
    elif env_family in ("mosaic_multigrid", "meltingpot"):
        return "parallel"  # Simultaneous by default — but AEC is also supported
    elif env_family in ("ini_multigrid", "overcooked"):
        return "parallel"  # Simultaneous only (no NOOP action — AEC not available)
    return "aec"  # Default


# RL has access to ALL environment families
RL_ENV_FAMILIES = tuple(ENV_FAMILIES.keys())

# LLM has access to all environment families (same as RL)
LLM_ENV_FAMILIES = tuple(ENV_FAMILIES.keys())

# LLM Client configurations
# Maps client_name -> (display_name, requires_api_key, default_base_url)
LLM_CLIENTS: Dict[str, Tuple[str, bool, Optional[str]]] = {
    "openrouter": ("OpenRouter", True, "https://openrouter.ai/api/v1"),
    "vllm": ("vLLM (Local)", False, "http://localhost:8000/v1"),
    "openai": ("OpenAI (Direct)", True, None),
    "anthropic": ("Anthropic (Direct)", True, None),
    "google": ("Google (Direct)", True, None),
}

# =============================================================================
# VLM Models (Vision Language Models) - Support image input
# Verified via OpenRouter API: models with 'image' in input_modalities
# =============================================================================
VLM_CLIENT_MODELS: Dict[str, List[Tuple[str, str]]] = {
    "openrouter": [
        # OpenAI VLMs (verified: image modality)
        # GPT-5 Series (Latest - Vision capable)
        ("openai/gpt-5.2", "GPT-5.2"),
        ("openai/gpt-5.1", "GPT-5.1"),
        ("openai/gpt-5", "GPT-5"),
        ("openai/gpt-5-mini", "GPT-5 Mini"),
        ("openai/gpt-5-image", "GPT-5 Image"),
        # GPT-4.1 Series (Vision capable)
        ("openai/gpt-4.1", "GPT-4.1"),
        ("openai/gpt-4.1-mini", "GPT-4.1 Mini"),
        ("openai/gpt-4.1-nano", "GPT-4.1 Nano"),
        # GPT-4o Series (Vision capable)
        ("openai/gpt-4o", "GPT-4o"),
        ("openai/gpt-4o-mini", "GPT-4o Mini"),
        ("openai/gpt-4-turbo", "GPT-4 Turbo"),
        # Anthropic VLMs (verified: image modality)
        ("anthropic/claude-3.5-sonnet", "Claude 3.5 Sonnet"),
        ("anthropic/claude-3.5-haiku", "Claude 3.5 Haiku"),
        ("anthropic/claude-3-opus", "Claude 3 Opus"),
        ("anthropic/claude-3-haiku", "Claude 3 Haiku"),
        # Google Gemini VLMs (verified: image modality)
        # Gemini 3 (Latest - Vision capable)
        ("google/gemini-3-flash-preview", "Gemini 3 Flash Preview"),
        ("google/gemini-3-pro-preview", "Gemini 3 Pro Preview"),
        ("google/gemini-3-pro-image-preview", "Gemini 3 Pro Image Preview"),
        # Gemini 2.5 (Vision capable)
        ("google/gemini-2.5-pro-preview-05-06", "Gemini 2.5 Pro Preview"),
        ("google/gemini-2.5-pro-preview-03-25", "Gemini 2.5 Pro (March)"),
        ("google/gemini-2.5-flash-preview", "Gemini 2.5 Flash Preview"),
        ("google/gemini-2.5-flash-preview-05-20", "Gemini 2.5 Flash (May)"),
        ("google/gemini-2.5-flash-lite-preview-06-17", "Gemini 2.5 Flash Lite"),
        ("google/gemini-2.5-flash-preview-image-05-20", "Gemini 2.5 Flash Image"),
        # Gemini 2.0 (Vision capable)
        ("google/gemini-2.0-flash-001", "Gemini 2.0 Flash"),
        ("google/gemini-2.0-flash-lite-001", "Gemini 2.0 Flash Lite"),
        ("google/gemini-2.0-flash-exp:free", "Gemini 2.0 Flash Exp (Free)"),
        # Meta Llama VLMs (verified: image modality)
        # Llama 4 (Latest)
        ("meta-llama/llama-4-maverick", "Llama 4 Maverick"),
        ("meta-llama/llama-4-scout", "Llama 4 Scout"),
        # Llama 3.2 Vision
        ("meta-llama/llama-3.2-90b-vision-instruct", "Llama 3.2 90B Vision"),
        ("meta-llama/llama-3.2-11b-vision-instruct", "Llama 3.2 11B Vision"),
        # Qwen VLMs (verified: image modality)
        # Qwen 3 VL (Latest)
        ("qwen/qwen3-vl-235b-a22b-instruct", "Qwen 3 VL 235B"),
        ("qwen/qwen3-vl-32b-instruct", "Qwen 3 VL 32B"),
        ("qwen/qwen3-vl-8b-instruct", "Qwen 3 VL 8B"),
        # Qwen 2.5 VL
        ("qwen/qwen2.5-vl-72b-instruct", "Qwen 2.5 VL 72B"),
        ("qwen/qwen2.5-vl-32b-instruct", "Qwen 2.5 VL 32B"),
        ("qwen/qwen-2.5-vl-7b-instruct:free", "Qwen 2.5 VL 7B (Free)"),
        ("qwen/qwen-2.5-vl-7b-instruct", "Qwen 2.5 VL 7B"),
        # Qwen VL Legacy
        ("qwen/qwen-vl-max", "Qwen VL Max"),
        ("qwen/qwen-vl-plus", "Qwen VL Plus"),
        # Google Gemma VLMs (verified: image modality - Gemma 3 has vision)
        ("google/gemma-3-27b-it", "Gemma 3 27B"),
        ("google/gemma-3-27b-it:free", "Gemma 3 27B (Free)"),
        ("google/gemma-3-12b-it", "Gemma 3 12B"),
        ("google/gemma-3-12b-it:free", "Gemma 3 12B (Free)"),
        ("google/gemma-3-4b-it", "Gemma 3 4B"),
        ("google/gemma-3-4b-it:free", "Gemma 3 4B (Free)"),
        # Mistral VLMs (verified: image modality)
        # Pixtral (Vision-first models)
        ("mistralai/pixtral-large-2411", "Pixtral Large 2411"),
        ("mistralai/pixtral-12b", "Pixtral 12B"),
        # Mistral Small 3.x (Vision capable)
        ("mistralai/mistral-small-3.2-24b-instruct-2506", "Mistral Small 3.2 24B"),
        ("mistralai/mistral-small-3.1-24b-instruct-2503", "Mistral Small 3.1 24B"),
        ("mistralai/mistral-small-3.1-24b-instruct:free", "Mistral Small 3.1 24B (Free)"),
        # Ministral 3 (Vision capable)
        ("mistralai/ministral-14b-2512", "Ministral 3 14B"),
        ("mistralai/ministral-8b-2512", "Ministral 3 8B"),
        ("mistralai/ministral-3b-2512", "Ministral 3 3B"),
    ],
    "vllm": [],  # vLLM servers are dynamically scanned
    "openai": [
        ("gpt-4o", "GPT-4o"),
        ("gpt-4o-mini", "GPT-4o Mini"),
        ("gpt-4-turbo", "GPT-4 Turbo"),
    ],
    "anthropic": [
        ("claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet"),
        ("claude-3-5-haiku-20241022", "Claude 3.5 Haiku"),
        ("claude-3-opus-20240229", "Claude 3 Opus"),
        ("claude-3-haiku-20240307", "Claude 3 Haiku"),
    ],
    "google": [
        ("gemini-2.0-flash-exp", "Gemini 2.0 Flash"),
        ("gemini-1.5-pro", "Gemini 1.5 Pro"),
    ],
}

# =============================================================================
# LLM Models (Text-Only Language Models) - NO image support
# Verified via OpenRouter API: models with only 'text' in input_modalities
# =============================================================================
LLM_CLIENT_MODELS: Dict[str, List[Tuple[str, str]]] = {
    "openrouter": [
        # Meta Llama text-only models (verified: text-only modality)
        # Llama 3.3
        ("meta-llama/llama-3.3-70b-instruct:free", "Llama 3.3 70B (Free)"),
        ("meta-llama/llama-3.3-70b-instruct", "Llama 3.3 70B"),
        # Llama 3.2
        ("meta-llama/llama-3.2-3b-instruct:free", "Llama 3.2 3B (Free)"),
        ("meta-llama/llama-3.2-3b-instruct", "Llama 3.2 3B"),
        ("meta-llama/llama-3.2-1b-instruct", "Llama 3.2 1B"),
        # Llama 3.1
        ("meta-llama/llama-3.1-405b-instruct:free", "Llama 3.1 405B (Free)"),
        ("meta-llama/llama-3.1-405b-instruct", "Llama 3.1 405B"),
        ("meta-llama/llama-3.1-70b-instruct", "Llama 3.1 70B"),
        ("meta-llama/llama-3.1-8b-instruct", "Llama 3.1 8B"),
        # Llama 3.0
        ("meta-llama/llama-3-70b-instruct", "Llama 3 70B"),
        ("meta-llama/llama-3-8b-instruct", "Llama 3 8B"),
        # Mistral text-only models (verified: text-only modality)
        # Mistral Large Series
        ("mistralai/mistral-large-2512", "Mistral Large 3 2512"),
        ("mistralai/mistral-large-2411", "Mistral Large 2411"),
        ("mistralai/mistral-large-2407", "Mistral Large 2407"),
        # Mistral Medium Series
        ("mistralai/mistral-medium-3.1", "Mistral Medium 3.1"),
        ("mistralai/mistral-medium-3", "Mistral Medium 3"),
        # Codestral (Code-focused)
        ("mistralai/codestral-2508", "Codestral 2508"),
        ("mistralai/codestral-mamba", "Codestral Mamba"),
        # Devstral (Agentic coding)
        ("mistralai/devstral-2512", "Devstral 2 2512"),
        ("mistralai/devstral-2512:free", "Devstral 2 2512 (Free)"),
        # Mixtral (MoE)
        ("mistralai/mixtral-8x22b-instruct", "Mixtral 8x22B"),
        ("mistralai/mixtral-8x7b-instruct", "Mixtral 8x7B"),
        # Mistral Small/7B
        ("mistralai/mistral-7b-instruct:free", "Mistral 7B (Free)"),
        ("mistralai/mistral-7b-instruct", "Mistral 7B"),
        # DeepSeek text-only models (verified: text-only modality)
        # DeepSeek V3.x Series
        ("deepseek/deepseek-v3.2", "DeepSeek V3.2"),
        ("deepseek/deepseek-v3.2-speciale", "DeepSeek V3.2 Speciale"),
        ("deepseek/deepseek-chat-v3.1", "DeepSeek V3.1"),
        ("deepseek/deepseek-v3.1-terminus", "DeepSeek V3.1 Terminus"),
        ("deepseek/deepseek-chat", "DeepSeek V3"),
        ("deepseek/deepseek-chat-v3-0324", "DeepSeek V3 0324"),
        # DeepSeek R1 (Reasoning)
        ("deepseek/deepseek-r1-0528", "DeepSeek R1 0528"),
        ("deepseek/deepseek-r1-0528:free", "DeepSeek R1 0528 (Free)"),
        ("deepseek/deepseek-r1", "DeepSeek R1"),
        # DeepSeek R1 Distill (Smaller/Faster)
        ("deepseek/deepseek-r1-distill-llama-70b", "R1 Distill Llama 70B"),
        ("deepseek/deepseek-r1-distill-qwen-32b", "R1 Distill Qwen 32B"),
        ("deepseek/deepseek-r1-distill-qwen-14b", "R1 Distill Qwen 14B"),
        ("deepseek/deepseek-r1-distill-llama-8b", "R1 Distill Llama 8B"),
        ("deepseek/deepseek-r1-distill-qwen-7b", "R1 Distill Qwen 7B"),
        # DeepSeek Prover (Math)
        ("deepseek/deepseek-prover-v2", "DeepSeek Prover V2"),
        # Qwen text-only models (verified: text-only modality)
        # Qwen 3
        ("qwen/qwen3-235b-a22b", "Qwen 3 235B"),
        ("qwen/qwen3-32b", "Qwen 3 32B"),
        ("qwen/qwen3-14b", "Qwen 3 14B"),
        ("qwen/qwen3-8b", "Qwen 3 8B"),
        ("qwen/qwen3-4b:free", "Qwen 3 4B (Free)"),
        # Qwen 3 Coder
        ("qwen/qwen3-coder:free", "Qwen 3 Coder (Free)"),
        ("qwen/qwen3-coder", "Qwen 3 Coder"),
        ("qwen/qwen3-coder-plus", "Qwen 3 Coder Plus"),
        # Qwen 2.5
        ("qwen/qwen-2.5-72b-instruct", "Qwen 2.5 72B"),
        ("qwen/qwen-2.5-7b-instruct", "Qwen 2.5 7B"),
        ("qwen/qwen-2.5-coder-32b-instruct", "Qwen 2.5 Coder 32B"),
        # QwQ (Reasoning)
        ("qwen/qwq-32b", "QwQ 32B"),
        # Qwen API Models
        ("qwen/qwen-max", "Qwen Max"),
        ("qwen/qwen-plus", "Qwen Plus"),
        ("qwen/qwen-turbo", "Qwen Turbo"),
        # Google Gemma text-only models (verified: text-only modality)
        # Gemma 3n (Text-only, smaller efficient models)
        ("google/gemma-3n-e4b-it:free", "Gemma 3n 4B (Free)"),
        ("google/gemma-3n-e4b-it", "Gemma 3n 4B"),
        ("google/gemma-3n-e2b-it:free", "Gemma 3n 2B (Free)"),
        # Gemma 2 (Text-only)
        ("google/gemma-2-27b-it", "Gemma 2 27B"),
        ("google/gemma-2-9b-it", "Gemma 2 9B"),
        ("google/gemma-2-9b-it:free", "Gemma 2 9B (Free)"),
        # Paid models (reliable, no rate-limit issues with credits)
        ("google/gemini-2.0-flash-001", "Gemini 2.0 Flash ($0.10/M)"),
        ("mistralai/mistral-small-24b-instruct-2501", "Mistral Small 24B ($0.05/M)"),
        # NVIDIA text-only models (verified: text-only modality)
        # Nemotron 3 Nano (30B MoE, 3B active — designed for agentic AI, ~1.6s)
        ("nvidia/nemotron-3-nano-30b-a3b:free", "Nemotron 3 Nano 30B (Free)"),
        # Arcee AI text-only models (verified: text-only modality)
        # Trinity Large (400B MoE, 13B active — reasoning, ~2.4s)
        ("arcee-ai/trinity-large-preview:free", "Arcee Trinity Large 400B (Free)"),
        # Trinity Mini (fast lightweight reasoning, ~2.7s)
        ("arcee-ai/trinity-mini:free", "Arcee Trinity Mini (Free)"),
        # Upstage text-only models (verified: text-only modality)
        # Solar Pro 3 (multilingual reasoning, ~4s)
        ("upstage/solar-pro-3:free", "Solar Pro 3 (Free)"),
    ],
    "vllm": [],  # vLLM servers are dynamically scanned
    "openai": [
        ("gpt-3.5-turbo", "GPT-3.5 Turbo"),
    ],
    "anthropic": [],  # All Claude 3+ models support vision
    "google": [
        ("gemini-1.5-flash", "Gemini 1.5 Flash"),
    ],
}


def _get_llm_workers() -> List[WorkerDefinition]:
    """Get LLM workers from catalog (supports_training=False)."""
    return [w for w in get_worker_catalog() if not w.supports_training]


def _get_rl_workers() -> List[WorkerDefinition]:
    """Get RL workers from catalog (supports_training=True)."""
    return [w for w in get_worker_catalog() if w.supports_training]


def _get_rl_evaluation_workers() -> List[WorkerDefinition]:
    """Get RL workers that support policy loading for evaluation."""
    return [w for w in get_worker_catalog() if w.supports_training and w.supports_policy_load]


def _get_registered_envs(prefix: str) -> List[str]:
    """Get registered gymnasium environments matching a prefix.

    Args:
        prefix: Environment name prefix (e.g., "MiniGrid-", "BabyAI-", "MiniHack-", "NetHack").

    Returns:
        Sorted list of environment IDs matching the prefix.
    """
    try:
        import gymnasium

        # Import the package that registers environments with gymnasium
        # MiniGrid and BabyAI environments are registered by the minigrid package
        if prefix in ("MiniGrid-", "BabyAI-"):
            try:
                import minigrid
                # Only register if not already registered (avoid duplicate registration warnings)
                if "MiniGrid-Empty-5x5-v0" not in gymnasium.registry:
                    minigrid.register_minigrid_envs()  # Required in minigrid 2.3.1+
            except ImportError:
                _LOGGER.debug("minigrid package not installed")
                return []

        # MiniHack environments are registered by the minihack package
        if prefix == "MiniHack-":
            try:
                import minihack  # noqa: F401 - registers envs on import
            except ImportError:
                _LOGGER.debug("minihack package not installed")
                return []

        # NetHack environments are registered by the nle package
        if prefix == "NetHack":
            try:
                import nle  # noqa: F401 - registers envs on import
            except ImportError:
                _LOGGER.debug("nle package not installed")
                return []

        envs = [
            env_id for env_id in gymnasium.registry.keys()
            if env_id.startswith(prefix)
        ]
        return sorted(envs)
    except Exception as e:
        _LOGGER.warning(f"Could not load {prefix} environments: {e}")
        return []


class LinkAgentDialog(QtWidgets.QDialog):
    """Dialog for selecting agents to link together.

    Allows users to select which agents should share the same policy checkpoint.
    """

    def __init__(
        self,
        primary_agent: str,
        compatible_agents: list[str],
        existing_group: Optional[LinkGroup] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        """Initialize the link agent dialog.

        Args:
            primary_agent: The agent that initiated the linking.
            compatible_agents: List of compatible agents that can be linked.
            existing_group: Existing link group if agent is already linked.
            parent: Parent widget.
        """
        super().__init__(parent)
        self._primary_agent = primary_agent
        self._compatible_agents = compatible_agents
        self._existing_group = existing_group
        self._checkboxes: Dict[str, QtWidgets.QCheckBox] = {}

        self.setWindowTitle("Link Agents")
        self.setModal(True)
        self.setMinimumWidth(400)

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        # Title label
        title = QtWidgets.QLabel(
            f"Select agents to link with <b>{self._primary_agent}</b>:",
            self,
        )
        title.setWordWrap(True)
        layout.addWidget(title)

        # Info label
        info = QtWidgets.QLabel(
            "Linked agents will share the same policy checkpoint path.\n"
            "When the primary agent's policy path changes, all linked agents will update automatically.",
            self,
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: gray; font-size: 10pt;")
        layout.addWidget(info)

        layout.addSpacing(10)

        # Scroll area for agent checkboxes
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(200)

        scroll_widget = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_widget)

        # Add checkbox for each compatible agent
        for agent_id in self._compatible_agents:
            checkbox = QtWidgets.QCheckBox(agent_id, scroll_widget)

            # Check if agent is already linked
            if self._existing_group and agent_id in self._existing_group.linked_agents:
                checkbox.setChecked(True)

            scroll_layout.addWidget(checkbox)
            self._checkboxes[agent_id] = checkbox

        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        # Buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_selected_agents(self) -> list[str]:
        """Get list of selected agents.

        Returns:
            List of agent IDs that were selected.
        """
        return [
            agent_id
            for agent_id, checkbox in self._checkboxes.items()
            if checkbox.isChecked()
        ]


class LinkingIndicatorWidget(QtWidgets.QWidget):
    """Visual indicator showing agents are linked with clickable text to unlink."""

    unlink_requested = pyqtSignal(str, str)  # Emitted when user clicks to unlink (group_id, agent_id)

    def __init__(self, group_id: str, agent_id: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        """Initialize the linking indicator.

        Args:
            group_id: The link group ID.
            agent_id: The agent ID this indicator represents.
            parent: Parent widget.
        """
        super().__init__(parent)
        self._group_id = group_id
        self._agent_id = agent_id
        self._color = "#4CAF50"  # Default green color
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 0, 0)
        layout.setSpacing(8)

        # Vertical line on the left (bracket connector)
        self._left_line = QtWidgets.QFrame(self)
        self._left_line.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        self._left_line.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
        self._left_line.setStyleSheet(f"background-color: {self._color}; min-width: 3px; max-width: 3px;")
        self._left_line.setMinimumHeight(20)
        layout.addWidget(self._left_line)

        # Clickable "Linked" label with chain icon
        self._linked_label = QtWidgets.QPushButton("🔗 Linked", self)
        self._linked_label.setFlat(True)
        self._update_label_style()
        self._linked_label.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._linked_label.setToolTip("Click to unlink this agent from the group")
        self._linked_label.clicked.connect(self._on_clicked)
        layout.addWidget(self._linked_label, 0)

        layout.addStretch()

    def set_color(self, color: str) -> None:
        """Set the color of the vertical line and label.

        Args:
            color: Hex color string (e.g., "#4CAF50")
        """
        self._color = color
        self._left_line.setStyleSheet(f"background-color: {color}; min-width: 3px; max-width: 3px;")
        self._update_label_style()

    def _update_label_style(self) -> None:
        """Update the label style with the current color."""
        self._linked_label.setStyleSheet(
            f"QPushButton {{"
            f"  background-color: {self._color};"
            f"  color: white;"
            f"  border: none;"
            f"  border-radius: 8px;"
            f"  padding: 2px 10px;"
            f"  font-size: 10px;"
            f"  font-weight: bold;"
            f"}}"
            f"QPushButton:hover {{ background-color: {self._color}; opacity: 0.8; }}"
        )

    def _on_clicked(self) -> None:
        """Handle click on the Linked label."""
        self.unlink_requested.emit(self._group_id, self._agent_id)


class LinkGroupDialog(QtWidgets.QDialog):
    """Dialog for creating a link group with multiple agents sharing a policy.

    This dialog allows users to manually create link groups for multi-agent RL,
    where multiple agents share the same policy checkpoint from the primary agent.
    """

    def __init__(
        self,
        primary_agent: str,
        primary_agent_label: str,
        available_agents: Dict[str, str],  # agent_id -> agent_label
        currently_linked: List[str] | None = None,  # Currently linked agent IDs
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        """Initialize the link group dialog.

        Args:
            primary_agent: The primary agent ID (e.g., "agent_0").
            primary_agent_label: Human-readable label for primary agent.
            available_agents: Dict of available agents to link with (agent_id -> label).
            currently_linked: List of agent IDs currently linked to primary (for editing).
            parent: Parent widget.
        """
        super().__init__(parent)
        self._primary_agent = primary_agent
        self._primary_agent_label = primary_agent_label
        self._available_agents = available_agents
        self._currently_linked = currently_linked or []
        self._agent_checkboxes: Dict[str, QtWidgets.QCheckBox] = {}

        self.setWindowTitle("Link Agents" if not currently_linked else "Edit Link Group")
        self.setMinimumWidth(450)
        self._build_ui()

    def _build_ui(self) -> None:
        """Build the dialog UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        # Header
        header = QtWidgets.QLabel(
            "Link agents to share the same policy checkpoint.\n"
            "All linked agents will use the primary agent's policy path and algorithm.",
            self
        )
        header.setWordWrap(True)
        header.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(header)

        # Primary agent (read-only)
        primary_group = QtWidgets.QGroupBox("Primary Agent", self)
        primary_layout = QtWidgets.QVBoxLayout(primary_group)
        primary_label = QtWidgets.QLabel(f"{self._primary_agent_label} ({self._primary_agent})", self)
        primary_label.setStyleSheet("font-weight: bold;")
        primary_layout.addWidget(primary_label)
        layout.addWidget(primary_group)

        # Agent selection
        if self._available_agents:
            agents_group = QtWidgets.QGroupBox("Select Agents to Link", self)
            agents_layout = QtWidgets.QVBoxLayout(agents_group)

            for agent_id, agent_label in self._available_agents.items():
                checkbox = QtWidgets.QCheckBox(f"{agent_label} ({agent_id})", self)
                # Pre-check if this agent is currently linked
                if agent_id in self._currently_linked:
                    checkbox.setChecked(True)
                self._agent_checkboxes[agent_id] = checkbox
                agents_layout.addWidget(checkbox)

            layout.addWidget(agents_group)
        else:
            no_agents_label = QtWidgets.QLabel("No other RL agents available to link.", self)
            no_agents_label.setStyleSheet("color: #999; font-style: italic;")
            layout.addWidget(no_agents_label)

        # Buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _on_accept(self) -> None:
        """Validate and accept the dialog."""
        # Check if at least one agent is selected
        selected_agents = self.get_selected_agents()
        if not selected_agents:
            QtWidgets.QMessageBox.warning(
                self,
                "No Agents Selected",
                "Please select at least one agent to link with the primary agent."
            )
            return

        self.accept()

    def get_selected_agents(self) -> List[str]:
        """Get list of selected agent IDs.

        Returns:
            List of agent IDs that were checked.
        """
        return [
            agent_id for agent_id, checkbox in self._agent_checkboxes.items()
            if checkbox.isChecked()
        ]



class PlayerAssignmentRow(QtWidgets.QWidget):
    """Single player assignment row within a multi-agent operator.

    Each row allows configuring a worker for one player in a multi-agent game.
    Displays: Player label, Worker dropdown, Provider dropdown, Model dropdown, API Key.
    """

    assignment_changed = pyqtSignal()  # Emitted when any field changes
    policy_path_changed = pyqtSignal(str, str)  # Emitted when policy path changes (player_id, new_path)
    link_agents_requested = pyqtSignal(str)  # Emitted when "Link Agents" button is clicked (player_id)
    unlink_agents_requested = pyqtSignal(str)  # Emitted when "Unlink Agents" button is clicked (player_id)

    def __init__(
        self,
        player_id: str,
        player_label: str,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        """Initialize a player assignment row.

        Args:
            player_id: The internal player ID (e.g., "player_0", "black_0").
            player_label: Human-readable label (e.g., "White", "Black").
            parent: Parent widget.
        """
        super().__init__(parent)
        self._player_id = player_id
        self._player_label = player_label
        self._updating = False
        self._vllm_servers: List[VLLMServerInfo] = []

        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        # Main vertical layout with two rows
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(4, 2, 4, 2)
        main_layout.setSpacing(2)

        # Row 1: Player label, Type selector, Worker dropdown, Provider (LLM only)
        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(6)

        # Player label with ID (will be updated to show worker name)
        label_text = f"{self._player_label} ({self._player_id})"
        self._player_label_widget = QtWidgets.QLabel(label_text, self)
        self._player_label_widget.setMinimumWidth(200)
        self._player_label_widget.setStyleSheet("font-weight: bold;")
        row1.addWidget(self._player_label_widget)

        # Type selector (LLM / RL / Human)
        row1.addWidget(QtWidgets.QLabel("Type:", self))
        self._type_combo = QtWidgets.QComboBox(self)
        self._type_combo.setFixedWidth(80)
        self._type_combo.addItems(["LLM", "RL", "Human", "Random", "Passive"])
        row1.addWidget(self._type_combo)

        # Agent color selector
        row1.addWidget(QtWidgets.QLabel("Color:", self))
        self._color_combo = QtWidgets.QComboBox(self)
        self._color_combo.setFixedWidth(100)
        self._color_combo.addItem("Auto", "auto")
        for color_name, (primary_hex, _) in COLOR_PALETTE.items():
            display_name = color_name.replace("_", " ").title()
            self._color_combo.addItem(display_name, color_name)
            idx = self._color_combo.count() - 1
            self._color_combo.setItemData(
                idx, QtGui.QColor(primary_hex), QtCore.Qt.ItemDataRole.DecorationRole,
            )
        # Default selection matches the agent's traditional colour
        default_color = DEFAULT_AGENT_COLOR_NAMES.get(self._player_id, "auto")
        cidx = self._color_combo.findData(default_color)
        if cidx >= 0:
            self._color_combo.setCurrentIndex(cidx)
        row1.addWidget(self._color_combo)

        # Worker dropdown
        row1.addWidget(QtWidgets.QLabel("Worker:", self))
        self._worker_combo = QtWidgets.QComboBox(self)
        self._worker_combo.setMinimumWidth(140)
        # Populated dynamically by _update_worker_dropdown()
        row1.addWidget(self._worker_combo)

        # Provider dropdown (LLM only)
        self._provider_label = QtWidgets.QLabel("Provider:", self)
        row1.addWidget(self._provider_label)
        self._client_combo = QtWidgets.QComboBox(self)
        self._client_combo.setMinimumWidth(100)
        for client_name, (display_name, _, _) in LLM_CLIENTS.items():
            self._client_combo.addItem(display_name, client_name)
        row1.addWidget(self._client_combo)

        row1.addStretch()
        main_layout.addLayout(row1)

        # Row 2: LLM settings container (Model/Server, API Key)
        self._llm_row = QtWidgets.QWidget(self)
        llm_layout = QtWidgets.QHBoxLayout(self._llm_row)
        llm_layout.setContentsMargins(0, 0, 0, 0)
        llm_layout.setSpacing(6)

        # Spacer to align with row1 (same width as player label)
        spacer = QtWidgets.QWidget(self._llm_row)
        spacer.setFixedWidth(160)
        llm_layout.addWidget(spacer)

        # Model dropdown
        self._model_label = QtWidgets.QLabel("Model:", self._llm_row)
        llm_layout.addWidget(self._model_label)
        self._model_combo = QtWidgets.QComboBox(self._llm_row)
        self._model_combo.setMinimumWidth(160)
        self._model_combo.setMaxVisibleItems(20)  # Limit dropdown height with scrollbar
        self._model_combo.setStyleSheet("QComboBox { combobox-popup: 0; }")
        model_view = self._model_combo.view()
        if model_view is not None:
            model_view.setVerticalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )
        llm_layout.addWidget(self._model_combo)

        # API Key field
        self._api_key_label = QtWidgets.QLabel("API Key:", self._llm_row)
        llm_layout.addWidget(self._api_key_label)
        self._api_key_edit = QtWidgets.QLineEdit(self._llm_row)
        self._api_key_edit.setPlaceholderText("API key")
        self._api_key_edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self._api_key_edit.setMinimumWidth(120)
        llm_layout.addWidget(self._api_key_edit)

        llm_layout.addStretch()
        main_layout.addWidget(self._llm_row)

        # Row 3: RL settings container (Policy path)
        self._rl_row = QtWidgets.QWidget(self)
        rl_layout = QtWidgets.QHBoxLayout(self._rl_row)
        rl_layout.setContentsMargins(0, 0, 0, 0)
        rl_layout.setSpacing(6)

        # Spacer to align with row1 (same width as player label)
        rl_spacer = QtWidgets.QWidget(self._rl_row)
        rl_spacer.setFixedWidth(120)
        rl_layout.addWidget(rl_spacer)

        # Policy path field
        rl_layout.addWidget(QtWidgets.QLabel("Policy:", self._rl_row))
        self._policy_path_edit = QtWidgets.QLineEdit(self._rl_row)
        self._policy_path_edit.setPlaceholderText("Path to trained policy/checkpoint")
        self._policy_path_edit.setMinimumWidth(200)
        rl_layout.addWidget(self._policy_path_edit)

        # Browse button
        self._browse_btn = QtWidgets.QPushButton("Browse...", self._rl_row)
        self._browse_btn.setFixedWidth(70)
        self._browse_btn.clicked.connect(self._on_browse_policy)
        rl_layout.addWidget(self._browse_btn)

        # Algorithm dropdown (xuance_worker: ippo, mappo; others: ppo, dqn, etc.)
        rl_layout.addWidget(QtWidgets.QLabel("Algorithm:", self._rl_row))
        self._algorithm_combo = QtWidgets.QComboBox(self._rl_row)
        self._algorithm_combo.setFixedWidth(90)
        self._algorithm_combo.addItem("IPPO", "ippo")
        self._algorithm_combo.addItem("MAPPO", "mappo")
        self._algorithm_combo.addItem("PPO", "ppo")
        self._algorithm_combo.addItem("QMIX", "qmix")
        self._algorithm_combo.setToolTip(
            "Algorithm that produced the checkpoint.\n"
            "IPPO: per-agent networks, no one-hot, portable across team compositions.\n"
            "MAPPO: shared network + agent one-hot, requires fixed n_agents.\n"
            "Must match the --method used during training."
        )
        self._algorithm_combo.currentIndexChanged.connect(self._on_changed)
        rl_layout.addWidget(self._algorithm_combo)

        # Link Agents button (for creating multi-agent policy groups)
        self._link_agents_btn = QtWidgets.QPushButton("Link Agents", self._rl_row)
        self._link_agents_btn.setFixedWidth(100)
        self._link_agents_btn.setToolTip(
            "Create a link group with other agents to share the same policy.\n"
            "Useful for multi-agent RL (MAPPO/IPPO) where multiple agents\n"
            "use the same checkpoint file."
        )
        self._link_agents_btn.clicked.connect(self._on_link_agents_clicked)
        rl_layout.addWidget(self._link_agents_btn)

        # Unlink Agents button (for removing agent from link group)
        self._unlink_agents_btn = QtWidgets.QPushButton("Unlink Agents", self._rl_row)
        self._unlink_agents_btn.setFixedWidth(100)
        self._unlink_agents_btn.setToolTip(
            "Remove this agent from its link group.\n"
            "The agent will use its own policy instead of the shared one."
        )
        self._unlink_agents_btn.clicked.connect(self._on_unlink_agents_clicked)
        self._unlink_agents_btn.hide()  # Hidden by default, shown only for linked agents
        rl_layout.addWidget(self._unlink_agents_btn)

        rl_layout.addStretch()
        main_layout.addWidget(self._rl_row)
        self._rl_row.hide()  # Hidden by default (LLM is default type)

        # Initialize dropdowns and visibility
        self._update_worker_dropdown()
        self._update_model_dropdown()
        self._update_api_key_visibility()
        self._update_type_visibility()

    def _connect_signals(self) -> None:
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        self._color_combo.currentIndexChanged.connect(self._on_changed)
        self._worker_combo.currentIndexChanged.connect(self._on_worker_changed)
        self._client_combo.currentIndexChanged.connect(self._on_client_changed)
        self._model_combo.currentIndexChanged.connect(self._on_changed)
        self._api_key_edit.textChanged.connect(self._on_changed)
        self._policy_path_edit.textChanged.connect(self._on_changed)
        self._policy_path_edit.textChanged.connect(self._on_policy_path_changed)

    def _on_changed(self) -> None:
        if not self._updating:
            self.assignment_changed.emit()

    def _on_policy_path_changed(self) -> None:
        """Handle policy path change - emit signal for parent to handle."""
        if not self._updating:
            new_path = self._policy_path_edit.text()
            self.policy_path_changed.emit(self._player_id, new_path)

    def _on_link_agents_clicked(self) -> None:
        """Handle Link Agents button click - emit signal for parent to handle."""
        self.link_agents_requested.emit(self._player_id)

    def _on_unlink_agents_clicked(self) -> None:
        """Handle Unlink Agents button click - emit signal for parent to handle."""
        self.unlink_agents_requested.emit(self._player_id)

    def _on_client_changed(self) -> None:
        if self._updating:
            return
        self._update_model_dropdown()
        self._update_api_key_visibility()
        self.assignment_changed.emit()

    def _on_type_changed(self) -> None:
        """Handle worker type change (LLM <-> RL)."""
        if self._updating:
            return
        self._update_worker_dropdown()
        self._update_type_visibility()
        self._update_player_label()
        self.assignment_changed.emit()

    def _on_worker_changed(self) -> None:
        """Handle worker selection change."""
        if not self._updating:
            self._update_player_label()
            self.assignment_changed.emit()

    def _update_player_label(self) -> None:
        """Update player label to show agent info and current worker."""
        worker_name = self._worker_combo.currentText()
        worker_type = self._type_combo.currentText()

        # Build label: "Agent Name (agent_id) - Worker Name"
        if worker_name and worker_type not in ("Human", "Random"):
            label_text = f"{self._player_label} ({self._player_id}) - {worker_name}"
        else:
            # For Human/Random or when no worker selected, just show agent info
            label_text = f"{self._player_label} ({self._player_id}) - {worker_type}"

        self._player_label_widget.setText(label_text)

    def _update_worker_dropdown(self) -> None:
        """Update worker dropdown based on selected type (LLM, RL, or Human)."""
        self._updating = True
        current_worker = self._worker_combo.currentData()
        self._worker_combo.clear()

        worker_type = self._type_combo.currentText().lower()
        if worker_type == "llm":
            workers = _get_llm_workers()
            for worker in workers:
                self._worker_combo.addItem(worker.display_name, worker.worker_id)
        elif worker_type == "rl":
            # RL: only workers that support policy loading for evaluation
            workers = _get_rl_evaluation_workers()
            for worker in workers:
                self._worker_combo.addItem(worker.display_name, worker.worker_id)
        elif worker_type == "human":
            self._worker_combo.addItem("MOSAIC Human Worker", "human_worker")
        elif worker_type == "passive":
            self._worker_combo.addItem("MOSAIC Passive Worker", "passive_worker")
        else:
            # Random
            self._worker_combo.addItem("MOSAIC Random Worker", "random_worker")

        # Restore selection if possible
        if current_worker:
            idx = self._worker_combo.findData(current_worker)
            if idx >= 0:
                self._worker_combo.setCurrentIndex(idx)

        self._updating = False
        self._update_player_label()

    def _update_type_visibility(self) -> None:
        """Show/hide LLM, RL, or Human settings based on selected type."""
        worker_type = self._type_combo.currentText().lower()
        is_llm = worker_type == "llm"
        is_rl = worker_type == "rl"

        # LLM row visibility
        self._llm_row.setVisible(is_llm)
        self._provider_label.setVisible(is_llm)
        self._client_combo.setVisible(is_llm)
        # RL row visibility
        self._rl_row.setVisible(is_rl)
        # Worker dropdown: always visible to show which MOSAIC worker is active
        self._worker_combo.setVisible(True)

    def _on_browse_policy(self) -> None:
        """Open file dialog to browse for policy file."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Policy File",
            "",
            "Policy Files (*.pt *.pth *.zip *.pkl *.ckpt);;All Files (*)"
        )
        if file_path:
            self._policy_path_edit.setText(file_path)

    def set_rl_row_visible(self, visible: bool) -> None:
        """Set RL row visibility based on link state.

        Args:
            visible: True to show RL row, False to hide it.
        """
        if self._type_combo.currentText().lower() == "rl":
            self._rl_row.setVisible(visible)

    def set_policy_fields_visible(self, visible: bool) -> None:
        """Show or hide policy-related fields (for linked agents).

        When an agent is linked, we hide the policy fields but keep the Unlink button visible.
        This allows linked agents to show only the "Unlink Agents" button.

        Args:
            visible: True to show policy fields, False to hide them.
        """
        # Hide/show policy-related widgets
        self._policy_path_edit.setVisible(visible)
        self._browse_btn.setVisible(visible)
        self._algorithm_combo.setVisible(visible)

        # Find and hide/show the labels
        layout = self._rl_row.layout()
        if layout is None:
            return
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                if isinstance(widget, QtWidgets.QLabel):
                    if widget.text() in ("Policy:", "Algorithm:"):
                        widget.setVisible(visible)

    def _update_model_dropdown(self) -> None:
        """Update model dropdown based on selected provider."""
        self._updating = True
        current_data = self._model_combo.currentData()
        self._model_combo.clear()

        client_name = self._client_combo.currentData()
        if client_name:
            if client_name == "vllm":
                self._model_label.setText("Server:")
                running_servers = [s for s in self._vllm_servers if s.status == "running"]
                if running_servers:
                    for server in running_servers:
                        self._model_combo.addItem(server.display_name, server)
                else:
                    self._model_combo.addItem("(No running servers)", None)
            else:
                self._model_label.setText("Model:")
                models = LLM_CLIENT_MODELS.get(client_name, [])
                for model_id, display_name in models:
                    self._model_combo.addItem(display_name, model_id)

        # Restore selection if possible
        if current_data:
            if client_name == "vllm" and isinstance(current_data, VLLMServerInfo):
                for i in range(self._model_combo.count()):
                    item_data = self._model_combo.itemData(i)
                    if isinstance(item_data, VLLMServerInfo) and item_data.server_id == current_data.server_id:
                        self._model_combo.setCurrentIndex(i)
                        break
            else:
                idx = self._model_combo.findData(current_data)
                if idx >= 0:
                    self._model_combo.setCurrentIndex(idx)

        self._updating = False

    def _update_api_key_visibility(self) -> None:
        """Show/hide API key field based on selected provider."""
        client_name = self._client_combo.currentData()
        if client_name and client_name in LLM_CLIENTS:
            _, requires_api_key, _ = LLM_CLIENTS[client_name]
            self._api_key_label.setVisible(requires_api_key)
            self._api_key_edit.setVisible(requires_api_key)

    def set_vllm_servers(self, servers: List[VLLMServerInfo]) -> None:
        """Update available vLLM servers."""
        self._vllm_servers = servers
        if self._client_combo.currentData() == "vllm":
            self._update_model_dropdown()

    def get_assignment(self) -> WorkerAssignment:
        """Get the WorkerAssignment for this player.

        Returns:
            WorkerAssignment with worker_id, worker_type, and settings.
        """
        worker_type = self._type_combo.currentText().lower()
        worker_id = self._worker_combo.currentData() or ""

        settings: Dict[str, Any] = {}

        if worker_type == "human":
            # Human worker: minimal settings, no model/policy needed
            worker_id = "human_worker"
            settings["player_name"] = self._player_label
            settings["player_id"] = self._player_id
        elif worker_type == "random":
            # Random baseline: uses random_worker subprocess
            worker_id = "random_worker"
        elif worker_type == "passive":
            # Passive baseline: uses passive_worker subprocess
            worker_id = "passive_worker"
        elif worker_type == "rl":
            # RL worker settings: policy path and algorithm
            policy_path = self._policy_path_edit.text().strip()
            if policy_path:
                settings["policy_path"] = policy_path
            algorithm = self._algorithm_combo.currentData()
            if algorithm:
                settings["algorithm"] = algorithm
            # Default worker if none selected
            if not worker_id:
                worker_id = "xuance_worker"
        else:
            # LLM worker settings: client, model, API key
            client_name = self._client_combo.currentData() or "openrouter"
            api_key = self._api_key_edit.text().strip()

            settings["client_name"] = client_name

            if api_key:
                settings["api_key"] = api_key

            # Handle vLLM vs other providers
            if client_name == "vllm":
                server_info = self._model_combo.currentData()
                if isinstance(server_info, VLLMServerInfo):
                    settings["model_id"] = server_info.model_id
                    settings["base_url"] = server_info.base_url
                else:
                    settings["model_id"] = ""
                    settings["base_url"] = "http://localhost:8000/v1"
            else:
                settings["model_id"] = self._model_combo.currentData() or ""
                if client_name in LLM_CLIENTS:
                    _, _, default_base_url = LLM_CLIENTS[client_name]
                    if default_base_url:
                        settings["base_url"] = default_base_url

            # Default worker if none selected
            if not worker_id:
                worker_id = "balrog_worker"

        # Agent color preference (applies to all worker types)
        color_name = self._color_combo.currentData()
        if color_name and color_name != "auto":
            settings["agent_color"] = color_name

        return WorkerAssignment(
            worker_id=worker_id,
            worker_type=worker_type,
            settings=settings,
        )

    @property
    def player_id(self) -> str:
        return self._player_id


class PlayerAssignmentPanel(QtWidgets.QWidget):
    """Panel showing all agent/player assignments for a multi-agent environment.

    Contains a PlayerAssignmentRow for each agent/player in the game.
    Supports both PettingZoo (turn-based) and MultiGrid (simultaneous) environments.
    """

    assignments_changed = pyqtSignal()  # Emitted when any player assignment changes

    def __init__(
        self,
        env_family: str,
        env_id: str,
        num_agents: int,
        agent_ids: Optional[List[str]] = None,
        agent_labels: Optional[Dict[str, str]] = None,
        operator_id: str = "",
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        """Initialize the player assignment panel.

        Args:
            env_family: Environment family ("pettingzoo", "mosaic_multigrid", etc.)
            env_id: Environment ID (e.g., "chess_v6", "MosaicMultiGrid-Soccer-v0")
            num_agents: Number of agents in the environment
            agent_ids: Optional list of agent IDs (e.g., ["player_0", "player_1"])
                      If None, auto-generates ["agent_0", "agent_1", ...]
            agent_labels: Optional dict mapping agent_id to display label
                         If None, uses agent_id as label
            operator_id: Parent operator ID for scoping link group IDs
                        (e.g., "operator_0" → link groups become "operator_0_link_0").
            parent: Parent widget.
        """
        super().__init__(parent)
        self._env_family = env_family
        self._env_id = env_id
        self._num_agents = num_agents
        self._operator_id = operator_id
        self._rows: Dict[str, PlayerAssignmentRow] = {}

        # Generate agent IDs if not provided
        if agent_ids is None:
            agent_ids = [f"agent_{i}" for i in range(num_agents)]
        self._agent_ids = agent_ids

        # Use provided labels or default to agent_id
        if agent_labels is None:
            agent_labels = {aid: aid for aid in agent_ids}
        self._agent_labels = agent_labels

        # Agent linking for multi-agent RL (MAPPO/IPPO)
        self._link_groups: Dict[str, LinkGroup] = {}  # group_id -> LinkGroup
        self._next_link_group_index: int = 0  # Counter for generating group IDs

        # Environments that support agent linking
        self._linking_supported_envs = {
            "mosaic_multigrid",
            "multigrid_ini",
            "meltingpot",
            "overcooked",
        }

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(2)

        # Header
        header = QtWidgets.QLabel(f"Agent Assignments ({self._num_agents} agents):", self)
        header.setStyleSheet("font-weight: bold; color: #555;")
        layout.addWidget(header)

        # Store linking indicator widgets
        self._linking_indicators: Dict[str, LinkingIndicatorWidget] = {}

        # Create row for each agent/player
        for i, agent_id in enumerate(self._agent_ids):
            agent_label = self._agent_labels.get(agent_id, agent_id)
            row = PlayerAssignmentRow(agent_id, agent_label, self)
            row.assignment_changed.connect(self.assignments_changed)
            row.assignment_changed.connect(self._on_assignment_changed)
            row.policy_path_changed.connect(self._on_policy_path_changed)
            row.link_agents_requested.connect(self._on_link_agents_requested)
            row.unlink_agents_requested.connect(self._on_unlink_agents_requested)
            layout.addWidget(row)
            self._rows[agent_id] = row

            # Add linking indicator between agents (except after the last agent)
            if i < len(self._agent_ids) - 1:
                next_agent_id = self._agent_ids[i + 1]
                # The indicator represents the agent below it (next_agent_id)
                indicator = LinkingIndicatorWidget("", next_agent_id, self)
                indicator.unlink_requested.connect(self._on_unlink_requested)
                indicator.hide()  # Hidden by default, shown when agents are linked
                layout.addWidget(indicator)
                # Store indicator with a key based on the current and next agent
                self._linking_indicators[f"{agent_id}_{next_agent_id}"] = indicator

        # Note: Link groups are now created manually via "Link Agents" button
        # Automatic grouping has been disabled to support multiple independent groups

    def set_vllm_servers(self, servers: List[VLLMServerInfo]) -> None:
        """Update all rows with available vLLM servers."""
        for row in self._rows.values():
            row.set_vllm_servers(servers)

    def get_worker_assignments(self) -> Dict[str, WorkerAssignment]:
        """Get all player -> worker assignments.

        For agents in link groups, the policy_path and algorithm are overridden
        with the group's values to ensure all agents in the group use the same policy.

        Returns:
            Dict mapping player_id to WorkerAssignment.
        """
        assignments = {}
        for player_id, row in self._rows.items():
            assignment = row.get_assignment()

            # Check if this agent is in a link group
            group = self._find_link_group_by_agent(player_id)
            if group and assignment.worker_type == "rl":
                # Override policy_path and algorithm with group's values
                settings = assignment.settings.copy()
                settings["policy_path"] = group.policy_path
                settings["algorithm"] = group.algorithm

                assignment = WorkerAssignment(
                    worker_id=assignment.worker_id,
                    worker_type=assignment.worker_type,
                    settings=settings,
                )

            assignments[player_id] = assignment

        return assignments

    @property
    def game_name(self) -> str:
        """Return environment ID for backwards compatibility."""
        return self._env_id

    @property
    def env_family(self) -> str:
        """Return environment family."""
        return self._env_family

    def has_llm_agent(self) -> bool:
        """Return True if any agent row has type LLM."""
        return any(
            row._type_combo.currentText().lower() == "llm"
            for row in self._rows.values()
        )

    def _supports_linking(self) -> bool:
        """Check if the current environment supports agent linking.

        Returns:
            True if the environment supports agent linking (MAPPO/IPPO).
        """
        return self._env_family in self._linking_supported_envs

    def _create_default_link_groups(self) -> None:
        """Create default link groups for all RL agents.

        By default, all RL agents with the same algorithm are linked together.
        This is called after the UI is built.
        """
        if not self._supports_linking():
            return

        # Group RL agents by algorithm
        rl_agents_by_algo: Dict[str, List[str]] = {}
        for agent_id, row in self._rows.items():
            if row._type_combo.currentText().lower() == "rl":
                algorithm = row._algorithm_combo.currentText()
                if algorithm not in rl_agents_by_algo:
                    rl_agents_by_algo[algorithm] = []
                rl_agents_by_algo[algorithm].append(agent_id)

        # Create link groups for each algorithm (if 2+ agents)
        for algorithm, agents in rl_agents_by_algo.items():
            if len(agents) >= 2:
                primary_agent = agents[0]
                linked_agents = agents[1:]
                self.create_link_group(primary_agent, linked_agents)

    def _on_assignment_changed(self) -> None:
        """Handle assignment change - update visual indicators only.

        This is called whenever an agent's worker type or algorithm changes.
        Link groups are now managed manually by the user, so we only update
        the visual indicators to reflect the current state.
        """
        if not self._supports_linking():
            return

        # Update visual indicators to reflect current state
        # Note: Link groups are now created manually via "Link Agents" button
        self._update_link_visual_indicators()

    def _on_unlink_requested(self, group_id: str, agent_id: str) -> None:
        """Handle unlink request from linking indicator.

        Args:
            group_id: The group to modify.
            agent_id: The specific agent to remove from the group.
        """
        from gym_gui.logging_config.helpers import log_constant
        from gym_gui.logging_config.log_constants import LOG_OP_AGENT_UNLINKED

        group = self._link_groups.get(group_id)
        if not group:
            return

        # Log the unlinking
        log_constant(
            _LOGGER,
            LOG_OP_AGENT_UNLINKED,
            extra={
                "agent_name": agent_id,
                "group_id": group_id,
                "primary_agent": group.primary_agent,
            }
        )

        # If the agent being removed is the primary agent, dissolve the entire group
        if agent_id == group.primary_agent:
            self.remove_link_group(group_id)
            return

        # Remove the agent from the linked_agents list
        if agent_id in group.linked_agents:
            group.linked_agents.remove(agent_id)

        # If only one agent remains in linked_agents, dissolve the group
        # (a link group needs at least 2 agents: primary + 1 linked)
        if len(group.linked_agents) == 0:
            self.remove_link_group(group_id)
            return

        # Update visual indicators to reflect the change
        self._update_link_visual_indicators()

    # -------------------------------------------------------------------------
    # Agent Linking Methods
    # -------------------------------------------------------------------------

    def _get_compatible_agents(self, player_id: str) -> list[str]:
        """Get list of agents compatible for linking with the given agent.

        Args:
            player_id: The agent to find compatible agents for.

        Returns:
            List of compatible agent IDs (excluding the given agent).
        """
        row = self._rows.get(player_id)
        if not row:
            return []

        # Only RL workers can be linked
        if row._type_combo.currentText().lower() != "rl":
            return []

        worker_type = row._type_combo.currentText()
        algorithm = row._algorithm_combo.currentText()

        compatible = []
        for agent_id, agent_row in self._rows.items():
            if agent_id == player_id:
                continue

            # Check if agent has same worker type and algorithm
            if (agent_row._type_combo.currentText() == worker_type and
                agent_row._algorithm_combo.currentText() == algorithm):
                compatible.append(agent_id)

        return compatible

    def create_link_group(
        self,
        primary_agent: str,
        linked_agents: list[str],
    ) -> str:
        """Create a new link group that shares the primary agent's policy.

        Args:
            primary_agent: The primary agent (leader).
            linked_agents: List of agents to link to the primary.

        Returns:
            The group_id of the created group.
        """
        from gym_gui.logging_config.helpers import log_constant
        from gym_gui.logging_config.log_constants import (
            LOG_OP_AGENT_LINK_GROUP_CREATED,
            LOG_OP_AGENT_LINK_VALIDATION_ERROR,
        )

        # Validate that all agents exist
        all_agents = [primary_agent] + linked_agents
        for agent in all_agents:
            if agent not in self._rows:
                log_constant(
                    _LOGGER,
                    LOG_OP_AGENT_LINK_VALIDATION_ERROR,
                    extra={
                        "primary_agent": primary_agent,
                        "linked_agents": linked_agents,
                        "reason": f"Agent {agent} not found",
                    }
                )
                return ""

        # Get primary agent's row
        primary_row = self._rows[primary_agent]
        worker_type = primary_row._type_combo.currentText()

        # Read policy path and algorithm from primary agent's row
        policy_path = primary_row._policy_path_edit.text()
        algorithm = primary_row._algorithm_combo.currentText().lower()

        # Generate group ID
        prefix = f"{self._operator_id}_" if self._operator_id else ""
        group_id = f"{prefix}link_{self._next_link_group_index}"
        self._next_link_group_index += 1

        # Get primary agent's color for the group background
        color_name = primary_row._color_combo.currentData()
        if color_name and color_name != "auto":
            # Use the primary agent's color
            color = COLOR_PALETTE.get(color_name, ("#E3F2FD", ""))[0]
        else:
            # Default to light blue if auto
            color = "#E3F2FD"

        # Create link group
        group = LinkGroup(
            group_id=group_id,
            primary_agent=primary_agent,
            linked_agents=linked_agents.copy(),
            policy_path=policy_path,
            algorithm=algorithm,
            worker_type=worker_type,
            color=color,
        )

        self._link_groups[group_id] = group

        # Log group creation
        log_constant(
            _LOGGER,
            LOG_OP_AGENT_LINK_GROUP_CREATED,
            extra={
                "group_id": group_id,
                "primary_agent": primary_agent,
                "linked_agents": linked_agents,
                "policy_path": policy_path,
                "algorithm": algorithm,
            }
        )

        # Sync policy path from primary to linked agents
        self._sync_policy_path(group_id)

        # Update visual indicators
        self._update_link_visual_indicators()

        return group_id

    def _update_link_group(self, group_id: str, primary_agent: str, new_linked_agents: list[str]) -> None:
        """Update an existing link group with new linked agents.

        Args:
            group_id: The group to update.
            primary_agent: The primary agent.
            new_linked_agents: New list of linked agents.
        """
        from gym_gui.logging_config.helpers import log_constant
        from gym_gui.logging_config.log_constants import (
            LOG_OP_AGENT_LINKED,
            LOG_OP_AGENT_UNLINKED,
        )

        group = self._link_groups.get(group_id)
        if not group:
            return

        old_linked = set(group.linked_agents)
        new_linked = set(new_linked_agents)

        # Find agents that were added
        added = new_linked - old_linked
        for agent in added:
            log_constant(
                _LOGGER,
                LOG_OP_AGENT_LINKED,
                extra={
                    "agent_name": agent,
                    "group_id": group_id,
                    "primary_agent": primary_agent,
                }
            )

        # Find agents that were removed
        removed = old_linked - new_linked
        for agent in removed:
            log_constant(
                _LOGGER,
                LOG_OP_AGENT_UNLINKED,
                extra={
                    "agent_name": agent,
                    "group_id": group_id,
                    "primary_agent": primary_agent,
                }
            )

        # Update group
        group.linked_agents = new_linked_agents.copy()

        # If no linked agents remain, dissolve the group
        if not group.linked_agents:
            self.remove_link_group(group_id)
        else:
            # Update visual indicators and sync policy paths
            self._update_link_visual_indicators()
            self._sync_policy_path(group_id)

    def remove_link_group(self, group_id: str) -> None:
        """Remove a link group and unlink all agents.

        Args:
            group_id: The group to remove.
        """
        from gym_gui.logging_config.helpers import log_constant
        from gym_gui.logging_config.log_constants import LOG_OP_AGENT_LINK_GROUP_DISSOLVED

        group = self._link_groups.get(group_id)
        if not group:
            return

        # Log group dissolution
        log_constant(
            _LOGGER,
            LOG_OP_AGENT_LINK_GROUP_DISSOLVED,
            extra={
                "group_id": group_id,
                "primary_agent": group.primary_agent,
                "linked_agents": group.linked_agents,
            }
        )

        # Remove group
        del self._link_groups[group_id]

        # Update visual indicators
        self._update_link_visual_indicators()

    def _update_link_visual_indicators(self) -> None:
        """Update background colors, RL row visibility, and linking indicators for all linked agents."""
        # Reset all backgrounds and show all RL rows first
        for row in self._rows.values():
            row.setStyleSheet("")
            row.set_rl_row_visible(True)
            row.set_policy_fields_visible(True)
            # Reset button visibility: show "Link Agents", hide "Unlink Agents"
            if hasattr(row, '_link_agents_btn') and hasattr(row, '_unlink_agents_btn'):
                row._link_agents_btn.show()
                row._unlink_agents_btn.hide()

        # Hide all linking indicators first
        for indicator in self._linking_indicators.values():
            indicator.hide()

        # Apply background colors and update visibility for linked agents
        for group in self._link_groups.values():
            # Apply background color to all agents in the group
            # Use QWidget-specific styling to constrain the background to the widget bounds
            for agent in group.all_agents():
                row = self._rows.get(agent)
                if row:
                    row.setStyleSheet(f"PlayerAssignmentRow {{ background-color: {group.color}; }}")

            # Update button visibility for agents in the group
            # Primary agent: show "Link Agents", hide "Unlink Agents", show all policy fields
            primary_row = self._rows.get(group.primary_agent)
            if primary_row:
                if hasattr(primary_row, '_link_agents_btn') and hasattr(primary_row, '_unlink_agents_btn'):
                    primary_row._link_agents_btn.show()
                    primary_row._unlink_agents_btn.hide()
                primary_row.set_policy_fields_visible(True)

            # Linked agents (not primary): hide "Link Agents", show "Unlink Agents", hide policy fields
            for agent in group.linked_agents:
                row = self._rows.get(agent)
                if row:
                    if hasattr(row, '_link_agents_btn') and hasattr(row, '_unlink_agents_btn'):
                        row._link_agents_btn.hide()
                        row._unlink_agents_btn.show()
                    # Keep RL row visible but hide policy fields (keep Unlink button visible)
                    row.set_policy_fields_visible(False)

            # Show linking indicators between consecutive agents in the group
            all_agents = group.all_agents()

            # Get primary agent's color
            primary_row = self._rows.get(group.primary_agent)
            hex_color = "#4CAF50"  # Default green
            if primary_row:
                color_name = primary_row._color_combo.currentData()
                if color_name and color_name != "auto":
                    hex_color = COLOR_PALETTE.get(color_name, ("#4CAF50", ""))[0]

            for i in range(len(all_agents) - 1):
                current_agent = all_agents[i]
                next_agent = all_agents[i + 1]

                # Check if these agents are consecutive in the agent list
                try:
                    current_idx = self._agent_ids.index(current_agent)
                    next_idx = self._agent_ids.index(next_agent)

                    # Only show indicator if agents are consecutive
                    if next_idx == current_idx + 1:
                        indicator_key = f"{current_agent}_{next_agent}"
                        indicator = self._linking_indicators.get(indicator_key)
                        if indicator:
                            indicator._group_id = group.group_id
                            indicator._agent_id = next_agent  # Update agent_id to match the agent below
                            indicator.set_color(hex_color)
                            indicator.show()
                except ValueError:
                    # Agent not found in list, skip
                    pass

    def _sync_policy_path(self, group_id: str) -> None:
        """Sync policy path from primary agent to all linked agents.

        Args:
            group_id: The group to sync.
        """
        from gym_gui.logging_config.helpers import log_constant
        from gym_gui.logging_config.log_constants import LOG_OP_AGENT_LINK_POLICY_SYNCED

        group = self._link_groups.get(group_id)
        if not group:
            return

        # Get primary agent's policy path
        primary_row = self._rows.get(group.primary_agent)
        if not primary_row:
            return

        policy_path = primary_row._policy_path_edit.text()

        # Update group's policy path
        group.policy_path = policy_path

        # Sync to all linked agents
        for agent in group.linked_agents:
            row = self._rows.get(agent)
            if row:
                row._policy_path_edit.setText(policy_path)

        # Log sync
        log_constant(
            _LOGGER,
            LOG_OP_AGENT_LINK_POLICY_SYNCED,
            extra={
                "group_id": group_id,
                "primary_agent": group.primary_agent,
                "linked_agents": group.linked_agents,
                "policy_path": policy_path,
            }
        )

    def _find_link_group_by_agent(self, agent_name: str) -> Optional[LinkGroup]:
        """Find the link group that contains the given agent.

        Args:
            agent_name: The agent to search for.

        Returns:
            The LinkGroup containing the agent, or None if not found.
        """
        for group in self._link_groups.values():
            if group.contains_agent(agent_name):
                return group
        return None

    def _on_policy_path_changed(self, player_id: str, new_path: str) -> None:
        """Handle policy path change - sync to linked agents if primary.

        Args:
            player_id: The agent whose policy path changed.
            new_path: The new policy path.
        """
        # Check if this agent is a primary agent in a link group
        group = self._find_link_group_by_agent(player_id)
        if not group:
            return

        # Only sync if this is the primary agent
        if group.primary_agent != player_id:
            return

        # Update group's policy path
        group.policy_path = new_path

        # Sync to all linked agents
        for agent in group.linked_agents:
            row = self._rows.get(agent)
            if row:
                # Temporarily disable updating to prevent signal loops
                row._updating = True
                row._policy_path_edit.setText(new_path)
                row._updating = False

    def _on_link_agents_requested(self, player_id: str) -> None:
        """Handle Link Agents button click - open dialog to create/edit link group.

        Args:
            player_id: The agent requesting to create a link group.
        """
        # Get the primary agent's row
        primary_row = self._rows.get(player_id)
        if not primary_row:
            return

        # Check if agent is RL type
        if primary_row._type_combo.currentText().lower() != "rl":
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Agent Type",
                "Only RL agents can be linked. Please set the agent type to RL first."
            )
            return

        # Check if this agent is already a primary agent in a link group
        existing_group = None
        currently_linked = []
        for group in self._link_groups.values():
            if group.primary_agent == player_id:
                existing_group = group
                currently_linked = group.linked_agents.copy()
                break

        # Get available RL agents to link with
        available_agents: Dict[str, str] = {}
        for agent_id, row in self._rows.items():
            if agent_id == player_id:
                continue  # Skip primary agent

            # Only include RL agents
            if row._type_combo.currentText().lower() != "rl":
                continue

            # If editing existing group, include currently linked agents
            if existing_group and agent_id in currently_linked:
                agent_label = self._agent_labels.get(agent_id, agent_id)
                available_agents[agent_id] = agent_label
                continue

            # Skip agents already in OTHER link groups
            agent_group = self._find_link_group_by_agent(agent_id)
            if agent_group and agent_group.group_id != (existing_group.group_id if existing_group else None):
                continue

            # Add to available agents
            agent_label = self._agent_labels.get(agent_id, agent_id)
            available_agents[agent_id] = agent_label

        # Get primary agent label
        primary_label = self._agent_labels.get(player_id, player_id)

        # Open dialog
        dialog = LinkGroupDialog(
            primary_agent=player_id,
            primary_agent_label=primary_label,
            available_agents=available_agents,
            currently_linked=currently_linked,
            parent=self
        )

        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            # Get selected agents
            linked_agents = dialog.get_selected_agents()

            if existing_group:
                # Update existing link group
                existing_group.linked_agents = linked_agents
                self._sync_policy_path(existing_group.group_id)
                self._update_link_visual_indicators()

                QtWidgets.QMessageBox.information(
                    self,
                    "Link Group Updated",
                    f"Successfully updated link group '{existing_group.group_id}' with {len(linked_agents) + 1} agents."
                )
            else:
                # Create new link group
                group_id = self.create_link_group(
                    primary_agent=player_id,
                    linked_agents=linked_agents,
                )

                if group_id:
                    QtWidgets.QMessageBox.information(
                        self,
                        "Link Group Created",
                        f"Successfully created link group '{group_id}' with {len(linked_agents) + 1} agents."
                    )

    def _on_unlink_agents_requested(self, player_id: str) -> None:
        """Handle Unlink Agents button click - remove agent from its link group.

        Args:
            player_id: The agent to unlink.
        """
        # Find which link group this agent belongs to
        group = self._find_link_group_by_agent(player_id)
        if not group:
            QtWidgets.QMessageBox.warning(
                self,
                "Not Linked",
                f"Agent '{player_id}' is not part of any link group."
            )
            return

        # Confirm unlinking
        agent_label = self._agent_labels.get(player_id, player_id)
        reply = QtWidgets.QMessageBox.question(
            self,
            "Unlink Agent",
            f"Remove '{agent_label}' from link group '{group.group_id}'?\n\n"
            f"The agent will use its own policy instead of the shared one.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No
        )

        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            # Call the existing unlink handler
            self._on_unlink_requested(group.group_id, player_id)



class OperatorConfigRow(QtWidgets.QWidget):
    """Single row in the operator configuration list.

    Each row represents one operator with:
    - Row 1: Display name, Type selector, Worker dropdown, Remove button
    - Row 2 (LLM): Client selector, Model selector, API Key, Environment, Task
    - Row 2 (RL): Environment, Task, Policy Path with Browse
    """

    config_changed = pyqtSignal(str, object)  # operator_id, new_config
    remove_requested = pyqtSignal(str)  # operator_id
    initialize_requested = pyqtSignal(str)  # operator_id - request to preview env
    configure_requested = pyqtSignal(str)  # operator_id - request to configure board game
    vllm_refresh_requested = pyqtSignal()  # request parent to refresh vLLM servers

    def __init__(
        self,
        operator_id: str,
        initial_config: Optional[OperatorConfig] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._operator_id = operator_id
        self._updating = False  # Prevent signal loops
        self._vllm_servers: List[VLLMServerInfo] = []  # Available vLLM servers
        self._player_panel: Optional[PlayerAssignmentPanel] = None  # Multi-agent panel
        self._initial_state: Optional[str] = None  # Custom board/grid state (FEN, JSON, etc.)

        self._build_ui()
        self._connect_signals()

        if initial_config:
            self._load_config(initial_config)

    def _build_ui(self) -> None:
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # ============================================================
        # Row 1: Identity - Index, Name, Type, Worker, Remove button
        # ============================================================
        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(8)

        # Operator index label (1-indexed for UI, updated by _update_ui_state)
        try:
            idx = int(self._operator_id.split("_")[-1]) + 1
        except (ValueError, IndexError):
            idx = 1
        self._index_label = QtWidgets.QLabel(f"#{idx}", self)
        self._index_label.setFixedWidth(20)
        self._index_label.setStyleSheet("font-weight: bold; color: #666;")
        row1.addWidget(self._index_label)

        # Display name - shows default "Operator N" when empty
        self._name_edit = QtWidgets.QLineEdit(self)
        self._name_edit.setPlaceholderText(f"Operator {idx}")
        self._name_edit.setFixedWidth(120)
        row1.addWidget(self._name_edit)

        # Decision-Maker selector (LLM / VLM / RL / Human / Random)
        row1.addWidget(QtWidgets.QLabel("Decision-Maker:", self))
        self._type_combo = QtWidgets.QComboBox(self)
        self._type_combo.addItems(["LLM", "VLM", "RL", "Human", "Random", "Passive"])
        self._type_combo.setFixedWidth(80)
        row1.addWidget(self._type_combo)

        # Worker dropdown
        row1.addWidget(QtWidgets.QLabel("Worker:", self))
        self._worker_combo = QtWidgets.QComboBox(self)
        self._worker_combo.setMinimumWidth(160)
        row1.addWidget(self._worker_combo)

        row1.addStretch()

        # Remove button (red X)
        self._remove_btn = QtWidgets.QPushButton("✕", self)
        self._remove_btn.setFixedSize(24, 24)
        self._remove_btn.setToolTip("Remove this operator")
        self._remove_btn.setStyleSheet(
            "QPushButton { color: #c00; font-weight: bold; border: none; }"
            "QPushButton:hover { color: #f00; background-color: #fee; border-radius: 3px; }"
        )
        row1.addWidget(self._remove_btn)

        main_layout.addLayout(row1)

        # ============================================================
        # Row 2: Two-column layout for environment and display settings
        # ============================================================
        row2 = QtWidgets.QHBoxLayout()
        row2.setSpacing(16)

        # --- Left Column: Environment Selection + Load Button ---
        left_col = QtWidgets.QVBoxLayout()
        left_col.setSpacing(6)

        left_form = QtWidgets.QFormLayout()
        left_form.setSpacing(6)
        left_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        self._env_combo = QtWidgets.QComboBox(self)
        self._env_combo.setMinimumWidth(200)
        left_form.addRow("Env Family:", self._env_combo)

        self._task_combo = QtWidgets.QComboBox(self)
        self._task_combo.setMinimumWidth(200)
        self._task_combo.setMaxVisibleItems(20)  # Limit dropdown height with scrollbar
        self._task_combo.setStyleSheet("QComboBox { combobox-popup: 0; }")  # Force native popup limiting
        task_view = self._task_combo.view()
        if task_view is not None:
            task_view.setVerticalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )
        left_form.addRow("Environment:", self._task_combo)

        left_col.addLayout(left_form)

        # Load button and size label under environment selection
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(8)

        # Preview label
        preview_label = QtWidgets.QLabel("Preview:", self)
        preview_label.setStyleSheet("font-weight: bold; color: #333;")
        btn_row.addWidget(preview_label)

        self._init_btn = QtWidgets.QPushButton("Load Environment", self)
        self._init_btn.setToolTip("Load and initialize this environment for the operator")
        btn_row.addWidget(self._init_btn)

        # Map label and Configure button - for board games and custom environments
        self._map_label = QtWidgets.QLabel("Map:", self)
        self._map_label.setStyleSheet("font-weight: bold; color: #333;")
        self._map_label.hide()  # Hidden by default
        btn_row.addWidget(self._map_label)

        self._configure_btn = QtWidgets.QPushButton("Configure Environment", self)
        self._configure_btn.setToolTip("Configure custom starting position or environment layout")
        self._configure_btn.hide()  # Hidden by default, shown only for supported games
        btn_row.addWidget(self._configure_btn)

        self._size_label = QtWidgets.QLabel("", self)
        self._size_label.setStyleSheet("color: #666; font-size: 10px;")
        self._size_label.setToolTip("Actual displayed dimensions after scaling (width × height)")
        self._size_label.hide()
        btn_row.addWidget(self._size_label)

        btn_row.addStretch()
        left_col.addLayout(btn_row)

        row2.addLayout(left_col)

        # --- Right Column: Display Settings ---
        right_col = QtWidgets.QFormLayout()
        right_col.setSpacing(6)
        right_col.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        # Container size dropdown (operator card boundary)
        self._container_size_combo = QtWidgets.QComboBox(self)
        self._container_size_combo.setToolTip(
            "Operator container size.\n"
            "Controls the size of the operator card boundary."
        )
        self._container_size_combo.addItem("Auto", 0)
        self._container_size_combo.addItem("300px", 300)
        self._container_size_combo.addItem("350px", 350)
        self._container_size_combo.addItem("400px", 400)
        self._container_size_combo.addItem("512px", 512)
        self._container_size_combo.addItem("600px", 600)
        self._container_size_combo.addItem("700px", 700)
        self._container_size_combo.addItem("768px", 768)
        self._container_size_combo.addItem("800px", 800)
        self._container_size_combo.addItem("1024px", 1024)
        self._container_size_combo.addItem("1280px", 1280)
        self._container_size_combo.addItem("1440px", 1440)
        self._container_size_combo.addItem("1600px", 1600)
        self._container_size_combo.addItem("1920px", 1920)
        self._container_size_combo.setCurrentIndex(7)  # Default to 800px
        self._container_size_combo.setFixedWidth(100)
        right_col.addRow("Container:", self._container_size_combo)

        # Image scale dropdown (RGB image scaling)
        self._image_scale_combo = QtWidgets.QComboBox(self)
        self._image_scale_combo.setToolTip(
            "Image scaling resolution.\n"
            "The RGB frame will be scaled to this size before display."
        )
        self._image_scale_combo.addItem("Native", 0)
        self._image_scale_combo.addItem("256px", 256)
        self._image_scale_combo.addItem("384px", 384)
        self._image_scale_combo.addItem("512px", 512)
        self._image_scale_combo.addItem("768px", 768)
        self._image_scale_combo.addItem("1024px", 1024)
        self._image_scale_combo.addItem("1280px", 1280)
        self._image_scale_combo.addItem("1440px", 1440)
        self._image_scale_combo.setCurrentIndex(0)  # Default to Native
        self._image_scale_combo.setFixedWidth(100)
        right_col.addRow("Image:", self._image_scale_combo)

        # Game resolution dropdown (for Crafter - controls native render size)
        self._game_resolution_label = QtWidgets.QLabel("Resolution:", self)
        self._game_resolution_label.setStyleSheet("font-weight: bold; color: #333;")
        self._game_resolution_label.hide()  # Hidden by default, shown only for Crafter

        self._game_resolution_combo = QtWidgets.QComboBox(self)
        self._game_resolution_combo.setToolTip(
            "Game render resolution (for Crafter).\n"
            "Higher resolution = sharper image but larger data transfer.\n"
            "64x64 is native, 512x512 matches Human Control mode."
        )
        self._game_resolution_combo.addItem("64x64 (Native)", (64, 64))
        self._game_resolution_combo.addItem("128x128", (128, 128))
        self._game_resolution_combo.addItem("256x256", (256, 256))
        self._game_resolution_combo.addItem("512x512 (Recommended)", (512, 512))
        self._game_resolution_combo.setCurrentIndex(3)  # Default to 512x512
        self._game_resolution_combo.setFixedWidth(150)
        self._game_resolution_combo.hide()  # Hidden by default
        right_col.addRow(self._game_resolution_label, self._game_resolution_combo)

        # Square size dropdown (for board games - Chess, Go, Connect Four, Checkers)
        self._square_size_label = QtWidgets.QLabel("Square:", self)
        self._square_size_label.setStyleSheet("font-weight: bold; color: #333;")
        self._square_size_label.hide()  # Hidden by default, shown only for board games

        self._square_size_combo = QtWidgets.QComboBox(self)
        self._square_size_combo.setToolTip(
            "Tile/square size in pixels.\n"
            "For board games: size of each square on the board.\n"
            "For multigrid: size of each grid cell (default 32px)."
        )
        self._square_size_combo.addItem("Small (30px)", 30)
        self._square_size_combo.addItem("Medium (50px)", 50)
        self._square_size_combo.addItem("Default (70px)", 70)
        self._square_size_combo.addItem("Large (90px)", 90)
        self._square_size_combo.addItem("XL (110px)", 110)
        self._square_size_combo.setCurrentIndex(2)  # Default to 70px
        self._square_size_combo.setFixedWidth(150)
        self._square_size_combo.hide()  # Hidden by default
        right_col.addRow(self._square_size_label, self._square_size_combo)

        row2.addLayout(right_col)

        row2.addStretch()

        main_layout.addLayout(row2)

        # ============================================================
        # Row 3: Type-specific settings (LLM or RL)
        # ============================================================

        # === LLM-specific widgets ===
        self._llm_container = QtWidgets.QWidget(self)
        llm_layout = QtWidgets.QHBoxLayout(self._llm_container)
        llm_layout.setContentsMargins(0, 0, 0, 0)
        llm_layout.setSpacing(8)

        # LLM Client selector
        llm_layout.addWidget(QtWidgets.QLabel("Provider:", self._llm_container))
        self._client_combo = QtWidgets.QComboBox(self._llm_container)
        self._client_combo.setMinimumWidth(110)
        for client_name, (display_name, _, _) in LLM_CLIENTS.items():
            self._client_combo.addItem(display_name, client_name)
        llm_layout.addWidget(self._client_combo)

        # LLM Model/Server selector (label changes based on provider)
        self._model_label = QtWidgets.QLabel("Model:", self._llm_container)
        llm_layout.addWidget(self._model_label)
        self._model_combo = QtWidgets.QComboBox(self._llm_container)
        self._model_combo.setMinimumWidth(200)  # Wider for server names
        self._model_combo.setMaxVisibleItems(20)  # Limit dropdown height with scrollbar
        self._model_combo.setStyleSheet("QComboBox { combobox-popup: 0; }")
        model_view = self._model_combo.view()
        if model_view is not None:
            model_view.setVerticalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )
        llm_layout.addWidget(self._model_combo)

        llm_layout.addStretch()

        # API Key field
        self._api_key_label = QtWidgets.QLabel("API Key:", self._llm_container)
        llm_layout.addWidget(self._api_key_label)
        self._api_key_edit = QtWidgets.QLineEdit(self._llm_container)
        self._api_key_edit.setPlaceholderText("Set env var or enter key")
        self._api_key_edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self._api_key_edit.setMinimumWidth(180)
        llm_layout.addWidget(self._api_key_edit)

        # Show/hide API key toggle
        self._show_key_btn = QtWidgets.QPushButton("👁", self._llm_container)
        self._show_key_btn.setFixedWidth(28)
        self._show_key_btn.setToolTip("Show/hide API key")
        self._show_key_btn.setCheckable(True)
        self._show_key_btn.clicked.connect(self._toggle_api_key_visibility)
        llm_layout.addWidget(self._show_key_btn)

        main_layout.addWidget(self._llm_container)

        # === RL-specific widgets ===
        self._rl_container = QtWidgets.QWidget(self)
        rl_layout = QtWidgets.QHBoxLayout(self._rl_container)
        rl_layout.setContentsMargins(0, 0, 0, 0)
        rl_layout.setSpacing(8)

        # Policy Path field
        rl_layout.addWidget(QtWidgets.QLabel("Policy:", self._rl_container))
        self._policy_path_edit = QtWidgets.QLineEdit(self._rl_container)
        self._policy_path_edit.setPlaceholderText("Path to trained policy/checkpoint")
        rl_layout.addWidget(self._policy_path_edit)

        # Browse button
        self._browse_btn = QtWidgets.QPushButton("Browse...", self._rl_container)
        self._browse_btn.setFixedWidth(70)
        self._browse_btn.clicked.connect(self._on_browse_policy)
        rl_layout.addWidget(self._browse_btn)

        main_layout.addWidget(self._rl_container)

        # === Execution Mode Selector (for multi-agent environments) ===
        self._execution_mode_container = QtWidgets.QWidget(self)
        exec_mode_layout = QtWidgets.QHBoxLayout(self._execution_mode_container)
        exec_mode_layout.setContentsMargins(0, 4, 0, 4)
        exec_mode_layout.setSpacing(8)

        exec_mode_label = QtWidgets.QLabel("Execution Mode:", self._execution_mode_container)
        exec_mode_label.setStyleSheet("font-weight: bold; color: #555;")
        exec_mode_layout.addWidget(exec_mode_label)

        self._execution_mode_combo = QtWidgets.QComboBox(self._execution_mode_container)
        self._execution_mode_combo.addItem("AEC (True Sequential Physics)", "aec")
        self._execution_mode_combo.addItem("Parallel (Simultaneous)", "parallel")
        self._execution_mode_combo.setToolTip(
            "Execution paradigm for multi-agent environments:\n"
            "\n"
            "Parallel: All agents observe S(t) simultaneously, act, then\n"
            "env.step([A_0, A_1]) fires once. Neither agent sees what the\n"
            "other did before deciding.\n"
            "\n"
            "AEC (True Sequential Physics): Each agent's action is applied\n"
            "to the physics immediately via env.step([A_i, NOOP]). The next\n"
            "agent observes the intermediate state S(t+0.5) — it sees the\n"
            "result of the previous agent's action before deciding.\n"
            "Requires NOOP=0 in the action space (mosaic_multigrid v5+,\n"
            "MeltingPot). Not available for environments without NOOP."
        )
        self._execution_mode_combo.setMinimumWidth(200)
        self._execution_mode_combo.currentIndexChanged.connect(self._on_config_changed)
        exec_mode_layout.addWidget(self._execution_mode_combo)

        exec_mode_layout.addStretch()
        self._execution_mode_container.hide()  # Hidden until multi-agent env selected
        main_layout.addWidget(self._execution_mode_container)

        # === MultiGrid Settings (for MultiGrid multi-agent environments) ===
        self._multigrid_settings_container = QtWidgets.QWidget(self)
        multigrid_layout = QtWidgets.QVBoxLayout(self._multigrid_settings_container)
        multigrid_layout.setContentsMargins(0, 4, 0, 4)
        multigrid_layout.setSpacing(8)

        # Observation Mode Selector
        obs_mode_row = QtWidgets.QHBoxLayout()
        obs_mode_label = QtWidgets.QLabel("Observation Mode:", self._multigrid_settings_container)
        obs_mode_label.setStyleSheet("font-weight: bold; color: #555;")
        obs_mode_row.addWidget(obs_mode_label)

        self._observation_mode_combo = QtWidgets.QComboBox(self._multigrid_settings_container)
        self._observation_mode_combo.addItem("Egocentric Only", "egocentric")
        self._observation_mode_combo.addItem("Visible Teammates (Recommended)", "visible_teammates")
        self._observation_mode_combo.setCurrentIndex(1)  # Default: visible_teammates
        self._observation_mode_combo.setToolTip(
            "Observation mode for LLM agents:\n"
            "- Egocentric Only: Agent sees only its own view (decentralized, realistic)\n"
            "- Visible Teammates: Include visible teammates in observation (better for games)"
        )
        self._observation_mode_combo.setMinimumWidth(250)
        self._observation_mode_combo.currentIndexChanged.connect(self._on_config_changed)
        obs_mode_row.addWidget(self._observation_mode_combo)
        obs_mode_row.addStretch()
        multigrid_layout.addLayout(obs_mode_row)

        # View Size Selector (MOSAIC only — controls agent NxN partial observation window)
        view_size_row = QtWidgets.QHBoxLayout()
        view_size_label = QtWidgets.QLabel("View Size:", self._multigrid_settings_container)
        view_size_label.setStyleSheet("font-weight: bold; color: #555;")
        view_size_row.addWidget(view_size_label)

        self._view_size_spin = QtWidgets.QSpinBox(self._multigrid_settings_container)
        self._view_size_spin.setRange(3, 15)
        self._view_size_spin.setSingleStep(2)  # Odd values preferred (3, 5, 7, 9, ...)
        self._view_size_spin.setSpecialValueText("3 (default)")
        self._view_size_spin.setValue(3)
        self._view_size_spin.setToolTip(
            "Agent view size (NxN partial observation window).\n"
            "Default: 3 (9 cells visible, 7.1% of 16x11 grid).\n"
            "Larger values give agents more information:\n"
            "  5 = 25 cells (19.8%)\n"
            "  7 = 49 cells (38.9%)\n"
            "  9 = 81 cells (64.3%)\n"
            "Must be odd for symmetric view. Even values are rounded up."
        )
        self._view_size_spin.setMinimumWidth(100)
        self._view_size_spin.valueChanged.connect(self._on_config_changed)
        view_size_row.addWidget(self._view_size_spin)
        view_size_row.addStretch()
        multigrid_layout.addLayout(view_size_row)

        # Coordination Level Selector (only visible when an LLM agent is present)
        self._coordination_container = QtWidgets.QWidget(self._multigrid_settings_container)
        coord_container_layout = QtWidgets.QVBoxLayout(self._coordination_container)
        coord_container_layout.setContentsMargins(0, 0, 0, 0)
        coord_container_layout.setSpacing(4)

        coord_level_row = QtWidgets.QHBoxLayout()
        coord_level_label = QtWidgets.QLabel("Coordination Strategy (LLM only):", self._coordination_container)
        coord_level_label.setStyleSheet("font-weight: bold; color: #555;")
        coord_level_row.addWidget(coord_level_label)

        self._coordination_level_combo = QtWidgets.QComboBox(self._coordination_container)
        self._coordination_level_combo.addItem("Level 1: Emergent (Minimal)", 1)
        self._coordination_level_combo.addItem("Level 2: Basic Hints", 2)
        self._coordination_level_combo.addItem("Level 3: Role-Based", 3)
        self._coordination_level_combo.setCurrentIndex(0)  # Default: Level 1
        self._coordination_level_combo.setToolTip(
            "Coordination strategy for LLM agents:\n"
            "- Level 1 (Emergent): Let LLMs figure out coordination naturally\n"
            "- Level 2 (Basic Hints): Add cooperation tips in system prompt\n"
            "- Level 3 (Role-Based): Assign explicit roles (forward/defender)"
        )
        self._coordination_level_combo.setMinimumWidth(250)
        self._coordination_level_combo.currentIndexChanged.connect(self._on_coordination_level_changed)
        coord_level_row.addWidget(self._coordination_level_combo)
        coord_level_row.addStretch()
        coord_container_layout.addLayout(coord_level_row)

        # Role Assignment Panel (shown only for Level 3)
        self._role_assignment_container = QtWidgets.QWidget(self._coordination_container)
        role_layout = QtWidgets.QVBoxLayout(self._role_assignment_container)
        role_layout.setContentsMargins(20, 4, 0, 4)
        role_layout.setSpacing(4)

        role_info_label = QtWidgets.QLabel("Assign roles to agents:", self._role_assignment_container)
        role_info_label.setStyleSheet("color: #666; font-style: italic;")
        role_layout.addWidget(role_info_label)

        # Role selectors will be created dynamically based on number of agents
        self._role_selectors: Dict[str, QtWidgets.QComboBox] = {}
        self._role_selectors_layout = QtWidgets.QVBoxLayout()
        role_layout.addLayout(self._role_selectors_layout)

        self._role_assignment_container.hide()  # Hidden by default
        coord_container_layout.addWidget(self._role_assignment_container)

        self._coordination_container.hide()  # Hidden until an LLM agent is present
        multigrid_layout.addWidget(self._coordination_container)

        self._multigrid_settings_container.hide()  # Hidden until MultiGrid selected
        main_layout.addWidget(self._multigrid_settings_container)

        # === Multi-agent player assignment container ===
        # Created dynamically when multi-agent environment is selected
        self._player_panel_container = QtWidgets.QWidget(self)
        self._player_panel_layout = QtWidgets.QVBoxLayout(self._player_panel_container)
        self._player_panel_layout.setContentsMargins(0, 0, 0, 0)
        self._player_panel_container.hide()  # Hidden until multi-agent env selected
        main_layout.addWidget(self._player_panel_container)

        # Store main_layout reference for dynamic updates
        self._main_layout = main_layout

        # Initialize dropdowns
        self._update_worker_dropdown()
        self._update_env_dropdown()
        self._update_task_dropdown()
        self._update_model_dropdown()
        self._update_type_specific_visibility()

    def _connect_signals(self) -> None:
        self._name_edit.textChanged.connect(self._on_config_changed)
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        self._worker_combo.currentIndexChanged.connect(self._on_config_changed)
        self._env_combo.currentIndexChanged.connect(self._on_env_changed)
        self._task_combo.currentIndexChanged.connect(self._on_task_changed)
        self._remove_btn.clicked.connect(lambda: self.remove_requested.emit(self._operator_id))
        self._init_btn.clicked.connect(lambda: self.initialize_requested.emit(self._operator_id))
        self._configure_btn.clicked.connect(lambda: self.configure_requested.emit(self._operator_id))

        # LLM-specific signals
        self._client_combo.currentIndexChanged.connect(self._on_client_changed)
        self._model_combo.currentIndexChanged.connect(self._on_config_changed)
        self._api_key_edit.textChanged.connect(self._on_config_changed)

        # RL-specific signals
        self._policy_path_edit.textChanged.connect(self._on_config_changed)

    def _on_type_changed(self) -> None:
        """Handle operator type change (LLM <-> VLM <-> RL)."""
        if self._updating:
            return
        self._update_worker_dropdown()
        self._update_env_dropdown()
        self._update_task_dropdown()
        self._update_model_dropdown()  # Update models based on VLM vs LLM
        self._update_type_specific_visibility()
        self._on_config_changed()

    def _on_env_changed(self) -> None:
        """Handle environment family change."""
        if self._updating:
            return
        self._update_task_dropdown()
        self._update_multiagent_panel()
        self._update_type_specific_visibility()
        self._on_config_changed()

    def _on_task_changed(self) -> None:
        """Handle task/game change within an environment family."""
        if self._updating:
            return
        # For multi-agent envs, different games have different agent counts/players
        if self._is_multiagent_env_selected():
            self._update_multiagent_panel()
        # Update Configure button visibility based on game support
        self._update_configure_button_visibility()
        self._on_config_changed()

    def _update_configure_button_visibility(self) -> None:
        """Show/hide Configure button based on whether game supports configuration."""
        try:
            from gym_gui.ui.widgets.operators_board_config_form import BoardConfigDialogFactory
        except ImportError:
            BoardConfigDialogFactory = None
        try:
            from gym_gui.ui.widgets.operators_grid_config_form import GridConfigDialogFactory
        except ImportError:
            GridConfigDialogFactory = None

        env_family = self._env_combo.currentText()
        task = self._task_combo.currentText()

        # Show Configure button for:
        # 1. Turn-based board games (Chess, Go, Checkers) - via BoardConfigDialogFactory
        # 2. Grid environments (MiniGrid, BabyAI, MultiGrid, MeltingPot) - via GridConfigDialogFactory
        is_board_game = BoardConfigDialogFactory.supports(task) if BoardConfigDialogFactory else False
        is_grid_env = GridConfigDialogFactory.supports(task) if GridConfigDialogFactory else False

        is_configurable = is_board_game or is_grid_env
        self._configure_btn.setVisible(is_configurable)
        self._map_label.setVisible(is_configurable)

        # Show square size dropdown for board games and grid environments
        # Board games: controls square size on the board
        # Grid envs (minigrid, babyai, multigrid): controls tile_size (grid cell pixel size, default 32px)
        is_square_size_env = (
            env_family in (
                "pettingzoo_classic", "open_spiel",
                "babyai", "minigrid",
                "mosaic_multigrid", "ini_multigrid",
            )
            or is_board_game
        )
        self._square_size_combo.setVisible(is_square_size_env)
        self._square_size_label.setVisible(is_square_size_env)

        # Show game resolution dropdown only for Crafter (controls native render size)
        is_crafter = env_family == "crafter"
        self._game_resolution_combo.setVisible(is_crafter)
        self._game_resolution_label.setVisible(is_crafter)

    def _is_pettingzoo_selected(self) -> bool:
        """Check if pettingzoo environment family is selected."""
        return self._env_combo.currentText() == "pettingzoo"

    def _is_multiagent_env_selected(self) -> bool:
        """Check if a multi-agent environment family is selected.

        Multi-agent environments include:
        - pettingzoo: Turn-based games (Chess, Go, Connect Four)
        - pettingzoo_classic: Classic board games (Chess, Go, Connect Four, TicTacToe)
        - open_spiel: OpenSpiel board games (Checkers, etc.)
        - mosaic_multigrid: Competitive team sports (Soccer, Collect, Basketball)
        - ini_multigrid: Cooperative exploration (Empty, LockedHallway, etc.)
        - meltingpot: DeepMind social scenarios
        - overcooked: Cooperative cooking
        """
        env_family = self._env_combo.currentText()
        return env_family in ("pettingzoo", "pettingzoo_classic", "open_spiel", "mosaic_multigrid", "ini_multigrid", "meltingpot", "overcooked")

    def _update_multiagent_panel(self) -> None:
        """Update the multi-agent player assignment panel based on selected game."""
        # Remove existing player panel
        if self._player_panel is not None:
            self._player_panel_layout.removeWidget(self._player_panel)
            self._player_panel.deleteLater()
            self._player_panel = None

        # Get current environment selection
        env_family = self._env_combo.currentText()
        env_id = self._task_combo.currentText()

        # Only show for multi-agent environments
        if not self._is_multiagent_env_selected():
            self._player_panel_container.hide()
            self._execution_mode_container.hide()
            return

        if not env_id:
            self._player_panel_container.hide()
            self._execution_mode_container.hide()
            return

        # Auto-detect number of agents
        num_agents = _auto_detect_agent_count(env_family, env_id)
        if num_agents == 0:
            self._player_panel_container.hide()
            self._execution_mode_container.hide()
            return

        # Set default execution mode based on environment family
        default_mode = _get_execution_mode(env_family)

        # Disable/enable AEC option based on environment capabilities.
        #
        # AEC (GymnasiumMultiAgentAECWrapper) requires NOOP=0 in the
        # action space so that non-acting agents do nothing while the current
        # agent's physics step runs.
        #
        # AEC-capable (NOOP=0 available):
        #   mosaic_multigrid  — action 0 = NOOP (v5.0.0+)
        #   meltingpot        — action 0 = NOOP (dm_env convention)
        #   pettingzoo / open_spiel — native AEC
        #
        # Simultaneous-only (no NOOP action — AEC not available):
        #   ini_multigrid     — action 0 = LEFT (no NOOP)
        #   overcooked        — no NOOP action
        simultaneous_only_envs = ("overcooked", "ini_multigrid")

        from PyQt6.QtGui import QStandardItemModel
        model = self._execution_mode_combo.model()

        if env_family in simultaneous_only_envs:
            # Disable AEC option — env has no NOOP action
            if isinstance(model, QStandardItemModel):
                item = model.item(0)  # AEC is at index 0
                if item is not None:
                    item.setEnabled(False)
                    item.setToolTip(
                        "AEC not available: this environment has no NOOP action.\n"
                        "Non-acting agents cannot stay idle during per-agent physics steps."
                    )
            # Force selection to Parallel
            self._execution_mode_combo.setCurrentIndex(1)  # Parallel
        else:
            # Enable AEC option — NOOP=0 is available, AEC works
            if isinstance(model, QStandardItemModel):
                item = model.item(0)  # AEC is at index 0
                if item is not None:
                    item.setEnabled(True)
                    item.setToolTip("")
            # Set based on default mode
            if default_mode == "aec":
                self._execution_mode_combo.setCurrentIndex(0)  # AEC
            else:
                self._execution_mode_combo.setCurrentIndex(1)  # Parallel

        # Show execution mode selector
        self._execution_mode_container.show()

        # Show MultiGrid/MeltingPot settings (observation mode, coordination strategy)
        if env_family in ("mosaic_multigrid", "ini_multigrid", "meltingpot"):
            self._multigrid_settings_container.show()
            # Update role selectors if Level 3 is selected
            if self._coordination_level_combo.currentData() == 3:
                self._update_role_selectors()
                self._role_assignment_container.show()
            else:
                self._role_assignment_container.hide()
        else:
            self._multigrid_settings_container.hide()

        # Get agent IDs and labels based on environment type
        agent_ids: Optional[List[str]] = None
        agent_labels: Optional[Dict[str, str]] = None

        if env_family in ("pettingzoo", "pettingzoo_classic") and env_id in PETTINGZOO_GAMES:
            # PettingZoo / PettingZoo Classic: use predefined player info
            game_info = PETTINGZOO_GAMES[env_id]
            agent_ids = game_info.get("players", None)
            agent_labels = game_info.get("player_labels", None)
        elif env_family in ("mosaic_multigrid", "ini_multigrid", "meltingpot", "overcooked"):
            # Simultaneous multi-agent: auto-generate agent_0, agent_1, etc.
            agent_ids = [f"agent_{i}" for i in range(num_agents)]
            agent_labels = {aid: f"Agent {i}" for i, aid in enumerate(agent_ids)}

        # Create new player panel
        self._player_panel = PlayerAssignmentPanel(
            env_family=env_family,
            env_id=env_id,
            num_agents=num_agents,
            agent_ids=agent_ids,
            agent_labels=agent_labels,
            operator_id=self._operator_id,
            parent=self._player_panel_container,
        )
        self._player_panel.assignments_changed.connect(self._on_config_changed)
        self._player_panel.set_vllm_servers(self._vllm_servers)
        self._player_panel_layout.addWidget(self._player_panel)
        self._player_panel_container.show()

    def _on_client_changed(self) -> None:
        """Handle LLM client change."""
        if self._updating:
            return
        # If switching to vLLM, request parent to refresh server list
        if self._client_combo.currentData() == "vllm":
            self.vllm_refresh_requested.emit()
        self._update_model_dropdown()
        self._update_api_key_visibility()
        self._on_config_changed()

    def _on_config_changed(self) -> None:
        """Emit config_changed signal with current configuration."""
        if self._updating:
            return
        self._update_coordination_visibility()
        config = self.get_config()
        self.config_changed.emit(self._operator_id, config)

    def _update_coordination_visibility(self) -> None:
        """Show coordination strategy only when at least one agent is LLM."""
        if self._player_panel is not None and self._player_panel.has_llm_agent():
            self._coordination_container.show()
        else:
            self._coordination_container.hide()

    def _on_coordination_level_changed(self) -> None:
        """Handle coordination level change - show/hide role assignment panel."""
        if self._updating:
            return

        # Show role assignment panel only for Level 3 (Role-Based)
        level = self._coordination_level_combo.currentData()
        if level == 3:
            self._update_role_selectors()
            self._role_assignment_container.show()
        else:
            self._role_assignment_container.hide()

        # Emit config changed
        self._on_config_changed()

    def _update_role_selectors(self) -> None:
        """Create or update role assignment selectors based on current environment."""
        # Clear existing role selectors
        for combo in self._role_selectors.values():
            combo.deleteLater()
        self._role_selectors.clear()

        # Clear layout
        while self._role_selectors_layout.count():
            item = self._role_selectors_layout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

        # Get current environment
        env_family = self._env_combo.currentText()
        env_id = self._task_combo.currentText()

        # Only create role selectors for MultiGrid Soccer
        if env_family != "mosaic_multigrid" or "Soccer" not in env_id:
            return

        # Get number of agents
        num_agents = _auto_detect_agent_count(env_family, env_id)
        if num_agents == 0:
            return

        # Create role selector for each agent (Soccer: 4 agents, 2 per team)
        for i in range(num_agents):
            agent_row = QtWidgets.QHBoxLayout()
            agent_label = QtWidgets.QLabel(f"Agent {i}:")
            agent_label.setMinimumWidth(80)
            agent_row.addWidget(agent_label)

            role_combo = QtWidgets.QComboBox()
            role_combo.addItem("Forward", "forward")
            role_combo.addItem("Defender", "defender")
            # Default: Agent 0, 2 = Forward; Agent 1, 3 = Defender
            if i % 2 == 0:
                role_combo.setCurrentIndex(0)  # Forward
            else:
                role_combo.setCurrentIndex(1)  # Defender
            role_combo.currentIndexChanged.connect(self._on_config_changed)
            agent_row.addWidget(role_combo)
            agent_row.addStretch()

            self._role_selectors_layout.addLayout(agent_row)
            self._role_selectors[f"agent_{i}"] = role_combo

    def _toggle_api_key_visibility(self) -> None:
        """Toggle API key visibility."""
        if self._show_key_btn.isChecked():
            self._api_key_edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Normal)
            self._show_key_btn.setText("Hide")
        else:
            self._api_key_edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
            self._show_key_btn.setText("Show")

    def _on_browse_policy(self) -> None:
        """Open file dialog to browse for policy file."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Policy File",
            "",
            "Policy Files (*.pt *.pth *.zip *.pkl *.ckpt);;All Files (*)"
        )
        if file_path:
            self._policy_path_edit.setText(file_path)

    def _update_type_specific_visibility(self) -> None:
        """Show/hide LLM/VLM, RL, or Human specific widgets based on operator type and env.

        For single-agent environments: Show single worker row (LLM, RL, or Human).
        For multi-agent environments: Hide single worker row, show player panel.
            - pettingzoo: Turn-based games (Chess, Go)
            - mosaic_multigrid, ini_multigrid, meltingpot, overcooked: Simultaneous multi-agent
        For Human type: Hide both LLM and RL containers (no configuration needed).
        """
        operator_type = self._type_combo.currentText().lower()
        is_llm_or_vlm = operator_type in ("llm", "vlm")
        is_human = operator_type == "human"
        is_random = operator_type == "random"
        is_multiagent = self._is_multiagent_env_selected()

        # For multi-agent environments: hide single-worker UI, workers are per-agent
        if is_multiagent:
            self._llm_container.hide()
            self._rl_container.hide()
            self._worker_combo.hide()
            # Type dropdown should also be hidden - each agent has its own type in player panel
            self._type_combo.hide()
        else:
            # Single-agent mode: show appropriate container
            self._type_combo.show()

            if is_human or is_random:
                # Human and Random operators: hide all config (no LLM/RL settings needed)
                self._llm_container.hide()
                self._rl_container.hide()
                self._worker_combo.hide()  # No worker selection for human/random
            else:
                self._worker_combo.show()
                self._llm_container.setVisible(is_llm_or_vlm)
                self._rl_container.setVisible(not is_llm_or_vlm)

                if is_llm_or_vlm:
                    self._update_api_key_visibility()

    def _update_api_key_visibility(self) -> None:
        """Show/hide API key field based on selected client."""
        client_name = self._client_combo.currentData()
        if client_name and client_name in LLM_CLIENTS:
            _, requires_api_key, _ = LLM_CLIENTS[client_name]
            self._api_key_label.setVisible(requires_api_key)
            self._api_key_edit.setVisible(requires_api_key)
            self._show_key_btn.setVisible(requires_api_key)

            # Update placeholder based on client
            if requires_api_key:
                env_var_map = {
                    "openai": "OPENAI_API_KEY",
                    "anthropic": "ANTHROPIC_API_KEY",
                    "google": "GOOGLE_API_KEY",
                }
                env_var = env_var_map.get(client_name, "")
                if env_var and os.environ.get(env_var):
                    self._api_key_edit.setPlaceholderText(f"Using {env_var} from env")
                else:
                    self._api_key_edit.setPlaceholderText(f"Enter key or set {env_var}")

    def _update_model_dropdown(self) -> None:
        """Update model/server dropdown based on selected LLM client and operator type.

        For vLLM: Shows running servers from vLLM Server widget
        For VLM type: Shows VLM_CLIENT_MODELS (vision-capable models)
        For LLM type: Shows LLM_CLIENT_MODELS (text-only models)
        """
        self._updating = True
        current_data = self._model_combo.currentData()
        self._model_combo.clear()

        client_name = self._client_combo.currentData()
        operator_type = self._type_combo.currentText().lower()

        if client_name:
            if client_name == "vllm":
                # Show running vLLM servers instead of models
                self._model_label.setText("Server:")
                running_servers = [s for s in self._vllm_servers if s.status == "running"]
                if running_servers:
                    for server in running_servers:
                        # Store full server info for base_url lookup
                        self._model_combo.addItem(server.display_name, server)
                else:
                    # No running servers - show placeholder
                    self._model_combo.addItem("(No running servers)", None)
            else:
                # Other providers: show models based on operator type (VLM vs LLM)
                self._model_label.setText("Model:")
                # Use VLM models for VLM type, LLM models for LLM type
                if operator_type == "vlm":
                    model_dict = VLM_CLIENT_MODELS
                else:
                    model_dict = LLM_CLIENT_MODELS

                if client_name in model_dict:
                    models = model_dict[client_name]
                else:
                    models = []
                for model_id, display_name in models:
                    self._model_combo.addItem(display_name, model_id)

        # Restore selection if possible
        if current_data:
            # For vLLM, match by server_id; for others, match by model_id
            if client_name == "vllm" and isinstance(current_data, VLLMServerInfo):
                for i in range(self._model_combo.count()):
                    item_data = self._model_combo.itemData(i)
                    if isinstance(item_data, VLLMServerInfo) and item_data.server_id == current_data.server_id:
                        self._model_combo.setCurrentIndex(i)
                        break
            else:
                idx = self._model_combo.findData(current_data)
                if idx >= 0:
                    self._model_combo.setCurrentIndex(idx)

        self._updating = False

    def set_vllm_servers(self, servers: List[VLLMServerInfo]) -> None:
        """Update the list of available vLLM servers.

        Called by parent widget when vLLM server status changes.

        Args:
            servers: List of VLLMServerInfo from the vLLM server widget.
        """
        self._vllm_servers = servers
        # Refresh dropdown if vLLM is currently selected
        if self._client_combo.currentData() == "vllm":
            self._update_model_dropdown()
        # Also update player panel if it exists (for multi-agent mode)
        if self._player_panel is not None:
            self._player_panel.set_vllm_servers(servers)

    def set_initial_state(self, initial_state: Optional[str]) -> None:
        """Set the custom initial state for this operator.

        Stores the board/grid configuration (FEN for chess, JSON for MiniGrid, etc.)
        that will be applied when the environment is loaded.

        Args:
            initial_state: State notation string (FEN, JSON, etc.) or None to clear.
        """
        self._initial_state = initial_state
        _LOGGER.debug(f"Set initial state for {self._operator_id}: {initial_state[:50] if initial_state else 'None'}...")

    def _update_worker_dropdown(self) -> None:
        """Update worker dropdown based on selected type."""
        self._updating = True
        current_worker = self._worker_combo.currentData()
        self._worker_combo.clear()

        operator_type = self._type_combo.currentText().lower()
        # LLM and VLM both use LLM workers (same BALROG worker, different image settings)
        if operator_type in ("llm", "vlm"):
            workers = _get_llm_workers()
            for worker in workers:
                self._worker_combo.addItem(worker.display_name, worker.worker_id)
        elif operator_type == "random":
            # Random: uses random_worker subprocess
            self._worker_combo.addItem("MOSAIC Random Worker", "random_worker")
        elif operator_type == "passive":
            # Passive: uses passive_worker subprocess
            self._worker_combo.addItem("MOSAIC Passive Worker", "passive_worker")
        elif operator_type == "human":
            self._worker_combo.addItem("MOSAIC Human Worker", "human_worker")
        else:
            # RL workers
            workers = _get_rl_workers()
            for worker in workers:
                self._worker_combo.addItem(worker.display_name, worker.worker_id)

        # Restore selection if possible
        if current_worker:
            idx = self._worker_combo.findData(current_worker)
            if idx >= 0:
                self._worker_combo.setCurrentIndex(idx)

        self._updating = False

    def _update_env_dropdown(self) -> None:
        """Update environment family dropdown based on selected type."""
        self._updating = True
        current_env = self._env_combo.currentText()
        self._env_combo.clear()

        operator_type = self._type_combo.currentText().lower()
        # LLM and VLM use same env families (text-based reasoning environments)
        if operator_type in ("llm", "vlm"):
            envs = LLM_ENV_FAMILIES
        elif operator_type == "human" or operator_type == "random":
            # Human and Random can play any environment (same as RL)
            envs = RL_ENV_FAMILIES
        else:
            # RL: all environment families
            envs = RL_ENV_FAMILIES

        self._env_combo.addItems(envs)

        # Restore selection if possible
        if current_env:
            idx = self._env_combo.findText(current_env)
            if idx >= 0:
                self._env_combo.setCurrentIndex(idx)

        self._updating = False

    def _update_task_dropdown(self) -> None:
        """Update environment dropdown based on selected environment family."""
        self._updating = True
        current_task = self._task_combo.currentText()
        self._task_combo.clear()

        env_family = self._env_combo.currentText()

        # Get environments for this family
        # Some families load dynamically from gymnasium, others use static lists
        envs: List[str] = []

        if env_family == "babyai":
            # Dynamically load all BabyAI environments from gymnasium registry
            envs = _get_registered_envs("BabyAI-")
            if not envs:
                envs = ["BabyAI-GoToRedBall-v0", "BabyAI-GoToObj-v0", "BabyAI-GoToLocal-v0"]
        elif env_family == "minigrid":
            # Dynamically load all MiniGrid environments from gymnasium registry
            envs = _get_registered_envs("MiniGrid-")
            if not envs:
                envs = ["MiniGrid-Empty-5x5-v0", "MiniGrid-DoorKey-5x5-v0"]
        elif env_family == "minihack":
            # Dynamically load all MiniHack environments from gymnasium registry
            envs = _get_registered_envs("MiniHack-")
            if not envs:
                envs = ["MiniHack-Room-5x5-v0", "MiniHack-Corridor-R5-v0"]
        elif env_family == "nle":
            # Dynamically load all NetHack environments from gymnasium registry
            envs = _get_registered_envs("NetHack")
            if not envs:
                envs = ["NetHackScore-v0", "NetHackStaircase-v0", "NetHackEat-v0"]
        elif env_family == "meltingpot":
            # Dynamically load available Melting Pot substrates
            try:
                from gym_gui.core.factories.adapters import available_games
                envs = [g for g in available_games() if g.startswith("meltingpot/")]
                if not envs:
                    # Fallback to common substrates
                    envs = ["meltingpot/clean_up", "meltingpot/prisoners_dilemma_in_the_matrix__arena"]
            except Exception:
                envs = ["meltingpot/clean_up", "meltingpot/prisoners_dilemma_in_the_matrix__arena"]
        elif env_family in ENV_FAMILIES:
            # Use static list from ENV_FAMILIES
            envs = list(ENV_FAMILIES[env_family])
        else:
            envs = [BALROG_DEFAULT_TASK]

        self._task_combo.addItems(envs if envs else ["default"])

        # Restore selection if possible
        if current_task:
            idx = self._task_combo.findText(current_task)
            if idx >= 0:
                self._task_combo.setCurrentIndex(idx)

        # Always show environment dropdown
        self._task_combo.setVisible(True)

        self._updating = False

        # Update Configure button visibility for board games
        self._update_configure_button_visibility()

    def _load_config(self, config: OperatorConfig) -> None:
        """Load configuration into UI elements."""
        self._updating = True

        self._name_edit.setText(config.display_name)

        # Determine type: check if it's VLM based on max_image_history setting
        operator_type = config.operator_type
        if operator_type in ("llm", "vlm") and config.settings:
            # If max_image_history > 0, it's VLM mode
            max_image_history = config.settings.get("max_image_history", 0)
            if max_image_history > 0:
                operator_type = "vlm"
            else:
                operator_type = "llm"

        # Set type: LLM=0, VLM=1, RL=2, Human=3
        type_map = {"llm": 0, "vlm": 1, "rl": 2, "human": 3}
        type_idx = type_map.get(operator_type, 0)
        self._type_combo.setCurrentIndex(type_idx)

        # Update dropdowns for type
        self._update_worker_dropdown()
        self._update_env_dropdown()

        # Set worker
        worker_idx = self._worker_combo.findData(config.worker_id)
        if worker_idx >= 0:
            self._worker_combo.setCurrentIndex(worker_idx)

        # Set environment
        env_idx = self._env_combo.findText(config.env_name)
        if env_idx >= 0:
            self._env_combo.setCurrentIndex(env_idx)

        # Update and set task
        self._update_task_dropdown()
        task_idx = self._task_combo.findText(config.task)
        if task_idx >= 0:
            self._task_combo.setCurrentIndex(task_idx)

        # Load LLM/VLM-specific settings
        if operator_type in ("llm", "vlm") and config.settings:
            # Set client
            client_name = config.settings.get("client_name", "openai")
            client_idx = self._client_combo.findData(client_name)
            if client_idx >= 0:
                self._client_combo.setCurrentIndex(client_idx)

            # Update and set model
            self._update_model_dropdown()
            model_id = config.settings.get("model_id")
            if model_id:
                model_idx = self._model_combo.findData(model_id)
                if model_idx >= 0:
                    self._model_combo.setCurrentIndex(model_idx)

            # Set API key
            api_key = config.settings.get("api_key", "")
            self._api_key_edit.setText(api_key)

        # Load RL-specific settings
        elif operator_type == "rl" and config.settings:
            policy_path = config.settings.get("policy_path", "")
            self._policy_path_edit.setText(policy_path)

        # Update visibility
        self._update_type_specific_visibility()

        self._updating = False

        # Update Configure button visibility for board games
        self._update_configure_button_visibility()

    def get_config(self) -> OperatorConfig:
        """Get current configuration from UI elements.

        Returns:
            OperatorConfig with single_agent or multi_agent configuration
            depending on whether pettingzoo environment is selected.
        """
        # Extract index from operator_id (e.g., "operator_0" -> 0) for display name
        try:
            idx = int(self._operator_id.split("_")[-1]) + 1
        except (ValueError, IndexError):
            idx = 1
        display_name = self._name_edit.text() or f"Operator {idx}"
        env_name = self._env_combo.currentText() or "babyai"
        task = self._task_combo.currentText() or BALROG_DEFAULT_TASK

        # Multi-agent mode: multi-agent environment with player assignments
        if self._is_multiagent_env_selected() and self._player_panel is not None:
            player_workers = self._player_panel.get_worker_assignments()
            # Get execution mode from dropdown
            execution_mode = self._execution_mode_combo.currentData() or "aec"
            # Get MultiGrid settings
            observation_mode = self._observation_mode_combo.currentData() or "visible_teammates"
            coordination_level = self._coordination_level_combo.currentData() or 1
            # Add container_size and image_scale to first worker's settings
            # (accessed via config.settings property)
            if player_workers:
                first_player = next(iter(player_workers.keys()))
                container_size = self._container_size_combo.currentData()
                if container_size and container_size > 0:
                    player_workers[first_player].settings["container_size"] = container_size
                image_scale = self._image_scale_combo.currentData()
                if image_scale and image_scale > 0:
                    player_workers[first_player].settings["image_scale"] = image_scale
                square_size = self._square_size_combo.currentData()
                if square_size and square_size > 0:
                    player_workers[first_player].settings["square_size"] = square_size
                game_resolution = self._game_resolution_combo.currentData()
                if game_resolution:
                    player_workers[first_player].settings["game_resolution"] = game_resolution

                # Add role assignments for Level 3 (Role-Based)
                if env_name in ("mosaic_multigrid", "ini_multigrid") and coordination_level == 3:
                    for player_id, worker in player_workers.items():
                        if player_id in self._role_selectors:
                            role = self._role_selectors[player_id].currentData()
                            worker.settings["role"] = role

                # Include custom initial state (board/grid config) if set
                if self._initial_state:
                    player_workers[first_player].settings["initial_state"] = self._initial_state

            # Get view_size (MOSAIC only, None = default of 3)
            view_size_val = self._view_size_spin.value()
            view_size = view_size_val if view_size_val != 3 else None
            if view_size is not None:
                _logger = logging.getLogger(__name__)
                log_constant(
                    _logger,
                    LOG_OPERATOR_VIEW_SIZE_CONFIGURED,
                    extra={
                        "operator_id": self._operator_id,
                        "view_size": view_size,
                        "env_name": env_name,
                        "task": task,
                    },
                )

            return OperatorConfig.multi_agent(
                operator_id=self._operator_id,
                display_name=display_name,
                env_name=env_name,
                task=task,
                player_workers=player_workers,
                execution_mode=execution_mode,
                observation_mode=observation_mode,
                coordination_level=coordination_level,
                view_size=view_size,
                link_groups=self._player_panel._link_groups if self._player_panel else None,
            )

        # Single-agent mode: standard LLM/VLM/RL/Human configuration
        operator_type = self._type_combo.currentText().lower()
        worker_id = self._worker_combo.currentData() or ""

        # Build settings based on operator type
        settings: Dict[str, Any] = {}

        if operator_type in ("llm", "vlm"):
            # LLM/VLM-specific settings
            client_name = self._client_combo.currentData() or "openai"
            api_key = self._api_key_edit.text().strip()

            settings["client_name"] = client_name

            # VLM mode: 0 = text-only (LLM), 1 = vision mode (VLM)
            # The type dropdown now directly controls this
            settings["max_image_history"] = 1 if operator_type == "vlm" else 0

            # Only include API key if provided
            if api_key:
                settings["api_key"] = api_key

            # Handle vLLM server selection vs model selection
            if client_name == "vllm":
                # vLLM: get model_id and base_url from selected server
                server_info = self._model_combo.currentData()
                if isinstance(server_info, VLLMServerInfo):
                    settings["model_id"] = server_info.model_id
                    settings["base_url"] = server_info.base_url
                else:
                    # No server selected - use empty values
                    settings["model_id"] = ""
                    settings["base_url"] = "http://localhost:8000/v1"
            else:
                # Other providers: get model_id from dropdown
                settings["model_id"] = self._model_combo.currentData() or ""
                # Include default base_url if available
                if client_name in LLM_CLIENTS:
                    _, _, default_base_url = LLM_CLIENTS[client_name]
                    if default_base_url:
                        settings["base_url"] = default_base_url

        elif operator_type == "rl":
            # RL-specific settings
            policy_path = self._policy_path_edit.text().strip()
            if policy_path:
                settings["policy_path"] = policy_path

        elif operator_type == "human":
            # Human operators: no subprocess worker, environment lives in GUI
            # No special settings needed - action selection happens via UI
            worker_id = "human_worker"  # Special marker for human operators

        elif operator_type == "random":
            # Random baseline: uses random_worker subprocess
            worker_id = "random_worker"

        elif operator_type == "passive":
            # Passive baseline: uses passive_worker subprocess
            worker_id = "passive_worker"

        # Common settings - container size, image scale, and square size
        container_size = self._container_size_combo.currentData()
        if container_size and container_size > 0:
            settings["container_size"] = container_size
        image_scale = self._image_scale_combo.currentData()
        if image_scale and image_scale > 0:
            settings["image_scale"] = image_scale
        square_size = self._square_size_combo.currentData()
        if square_size and square_size > 0:
            settings["square_size"] = square_size
        game_resolution = self._game_resolution_combo.currentData()
        if game_resolution:
            settings["game_resolution"] = game_resolution

        # Include custom initial state (board/grid config) if set
        if self._initial_state:
            settings["initial_state"] = self._initial_state
            _LOGGER.debug(f"get_config: Including initial_state for {self._operator_id}")
        else:
            _LOGGER.debug(f"get_config: No initial_state for {self._operator_id}")

        return OperatorConfig.single_agent(
            operator_id=self._operator_id,
            display_name=display_name,
            worker_id=worker_id,
            worker_type=operator_type,
            env_name=env_name,
            task=task,
            settings=settings,
        )

    def set_environment_size(
        self, width: int, height: int, container_size: Optional[int] = None
    ) -> None:
        """Set the environment size label after loading.

        Args:
            width: Rendered environment width in pixels (image size)
            height: Rendered environment height in pixels (image size)
            container_size: Optional container display size in pixels
        """
        if container_size:
            self._size_label.setText(f"Image: {width}×{height} | Container: {container_size}px")
        else:
            self._size_label.setText(f"Image: {width}×{height}")
        self._size_label.show()

    def clear_environment_size(self) -> None:
        """Clear the environment size label."""
        self._size_label.setText("")
        self._size_label.hide()

    @property
    def operator_id(self) -> str:
        return self._operator_id


class OperatorConfigWidget(QtWidgets.QWidget):
    """Widget for managing N operator configurations.

    Provides:
    - List of OperatorConfigRow widgets
    - Add Operator button
    - Operator count limit (default MAX_OPERATORS)
    - Signals for operator list changes
    """

    operators_changed = pyqtSignal(list)  # List[OperatorConfig]
    initialize_requested = pyqtSignal(str, object)  # operator_id, config
    configure_requested = pyqtSignal(str, object)  # operator_id, config - for board game config
    vllm_refresh_requested = pyqtSignal()  # request to refresh vLLM server list

    def __init__(
        self,
        max_operators: int = MAX_OPERATORS,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._max_operators = max_operators
        self._rows: Dict[str, OperatorConfigRow] = {}
        self._next_index = 0
        self._vllm_servers: List[VLLMServerInfo] = []  # Cache for new rows

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Header
        header = QtWidgets.QLabel("Configure Operators", self)
        header.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(header)

        # Scroll area for operator rows - expands to fill available space
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setMinimumHeight(300)
        # No maximum height - let it fill available space
        scroll.setMinimumWidth(450)
        scroll.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )

        self._rows_container = QtWidgets.QWidget(scroll)
        self._rows_layout = QtWidgets.QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(2)
        self._rows_layout.addStretch()

        scroll.setWidget(self._rows_container)
        layout.addWidget(scroll, 1)  # Stretch factor 1 to fill available space

        # Add button
        self._add_btn = QtWidgets.QPushButton("+ Add Operator", self)
        self._add_btn.clicked.connect(self.add_operator)
        layout.addWidget(self._add_btn)

        # Info label
        self._info_label = QtWidgets.QLabel(f"0 / {self._max_operators} operators", self)
        self._info_label.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(self._info_label)

    def add_operator(self, config: Optional[OperatorConfig] = None) -> Optional[str]:
        """Add a new operator row.

        Args:
            config: Optional initial configuration.

        Returns:
            The operator_id of the new row, or None if max reached.
        """
        if len(self._rows) >= self._max_operators:
            QtWidgets.QMessageBox.warning(
                self,
                "Maximum Operators",
                f"Maximum of {self._max_operators} operators allowed."
            )
            return None

        operator_id = f"operator_{self._next_index}"
        self._next_index += 1

        # Create default config if not provided
        if config is None:
            config = OperatorConfig.single_agent(
                operator_id=operator_id,
                display_name=f"Operator {len(self._rows) + 1}",
                worker_id="balrog_worker",
                worker_type="llm",
                env_name="babyai",
                task="BabyAI-GoToRedBall-v0",
                settings={
                    "client_name": "openai",
                    "model_id": "gpt-4o-mini",
                },
            )

        # Create row widget
        row = OperatorConfigRow(operator_id, config, self._rows_container)
        row.config_changed.connect(self._on_row_config_changed)
        row.remove_requested.connect(self.remove_operator)
        row.initialize_requested.connect(self._on_initialize_requested)
        row.configure_requested.connect(self._on_configure_requested)
        row.vllm_refresh_requested.connect(self.vllm_refresh_requested)  # Propagate to parent

        # Pass cached vLLM server list to new row
        if self._vllm_servers:
            row.set_vllm_servers(self._vllm_servers)

        # Insert before stretch
        self._rows_layout.insertWidget(self._rows_layout.count() - 1, row)
        self._rows[operator_id] = row

        self._update_ui_state()
        self._emit_operators_changed()

        return operator_id

    def remove_operator(self, operator_id: str) -> None:
        """Remove an operator row.

        Args:
            operator_id: ID of the operator to remove.
        """
        if operator_id not in self._rows:
            return

        row = self._rows.pop(operator_id)
        self._rows_layout.removeWidget(row)
        row.deleteLater()

        self._update_ui_state()
        self._emit_operators_changed()

    def get_operators(self) -> List[OperatorConfig]:
        """Get all current operator configurations."""
        return [row.get_config() for row in self._rows.values()]

    def set_operators(self, configs: List[OperatorConfig]) -> None:
        """Set operator configurations, replacing any existing ones."""
        # Clear existing
        for operator_id in list(self._rows.keys()):
            self.remove_operator(operator_id)

        # Add new
        for config in configs:
            self.add_operator(config)

    def clear(self) -> None:
        """Remove all operators."""
        for operator_id in list(self._rows.keys()):
            self.remove_operator(operator_id)

    def _on_row_config_changed(self, operator_id: str, config: OperatorConfig) -> None:
        """Handle config change from a row."""
        self._emit_operators_changed()

    def _on_initialize_requested(self, operator_id: str) -> None:
        """Handle initialize request from a row."""
        if operator_id not in self._rows:
            return
        config = self._rows[operator_id].get_config()
        _LOGGER.info(f"Initialize requested for operator {operator_id}: {config.env_name}/{config.task}")
        self.initialize_requested.emit(operator_id, config)

    def _on_configure_requested(self, operator_id: str) -> None:
        """Handle configure request from a row (board game configuration)."""
        if operator_id not in self._rows:
            return
        config = self._rows[operator_id].get_config()
        _LOGGER.info(f"Configure requested for operator {operator_id}: {config.env_name}/{config.task}")
        self.configure_requested.emit(operator_id, config)

    def _emit_operators_changed(self) -> None:
        """Emit the operators_changed signal with current configs."""
        configs = self.get_operators()
        self.operators_changed.emit(configs)
        _LOGGER.debug(f"Operators changed: {len(configs)} operators")

    def _update_ui_state(self) -> None:
        """Update UI state based on current operator count."""
        count = len(self._rows)
        self._info_label.setText(f"{count} / {self._max_operators} operators")
        self._add_btn.setEnabled(count < self._max_operators)

        # Update index labels
        for i, (operator_id, row) in enumerate(self._rows.items()):
            row._index_label.setText(f"#{i + 1}")

    @property
    def operator_count(self) -> int:
        """Get the number of configured operators."""
        return len(self._rows)

    def set_vllm_servers(self, servers: List[VLLMServerInfo]) -> None:
        """Update all operator rows with available vLLM servers.

        Called when vLLM server status changes. Propagates the server list
        to all OperatorConfigRow widgets so they can update their dropdowns.

        Args:
            servers: List of VLLMServerInfo from the vLLM server widget.
        """
        # Cache for new rows that will be added later
        self._vllm_servers = servers
        _LOGGER.debug(f"Updating operator rows with {len(servers)} vLLM servers")
        for row in self._rows.values():
            row.set_vllm_servers(servers)

    def set_operator_initial_state(self, operator_id: str, initial_state: str) -> None:
        """Set the initial state (board position) for a specific operator.

        Called after the board configuration dialog is accepted to store
        the custom starting position in the operator's config.

        Args:
            operator_id: The operator's unique ID
            initial_state: Board state notation (FEN for chess, SGF for Go, etc.)
        """
        if operator_id not in self._rows:
            _LOGGER.warning(f"Cannot set initial state: operator {operator_id} not found")
            return

        row = self._rows[operator_id]
        # Store initial_state in the row widget itself so it persists across get_config() calls
        row.set_initial_state(initial_state)
        _LOGGER.info(f"Set initial state for {operator_id}: {initial_state[:50]}...")

        # Emit config changed to propagate the update
        self._emit_operators_changed()

    def set_operator_environment_size(
        self, operator_id: str, width: int, height: int, container_size: Optional[int] = None
    ) -> None:
        """Set the environment size for a specific operator.

        Called after an environment is successfully loaded to display
        the rendered dimensions in the operator config row.

        Args:
            operator_id: The operator's unique ID
            width: Rendered environment width in pixels (image size)
            height: Rendered environment height in pixels (image size)
            container_size: Optional container display size in pixels
        """
        if operator_id in self._rows:
            self._rows[operator_id].set_environment_size(width, height, container_size)
            _LOGGER.debug(f"Set environment size for {operator_id}: {width}×{height}, container: {container_size}")


__all__ = [
    "OperatorConfigRow",
    "OperatorConfigWidget",
    "PlayerAssignmentRow",
    "PlayerAssignmentPanel",
    "VLLMServerInfo",
    "MAX_OPERATORS",
    "LLM_CLIENTS",
    "LLM_CLIENT_MODELS",
    "VLM_CLIENT_MODELS",
    "PETTINGZOO_GAMES",
]
