"""Worker catalog for training and analytics integrations.

This module centralises metadata about available worker integrations so the UI can
present consistent choices. Each worker definition describes how the control panel
should expose forms, whether evaluation (policy load) is supported, and which
capabilities (telemetry vs. analytics) the worker provides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple


@dataclass(frozen=True)
class WorkerDefinition:
    """Describe a worker integration exposed through the GUI."""

    worker_id: str
    display_name: str
    description: str
    supports_training: bool = True
    supports_policy_load: bool = False
    requires_live_telemetry: bool = True
    provides_fast_analytics: bool = False
    supports_multi_agent: bool = True

    def capabilities(self) -> Tuple[str, ...]:
        """Return a human-readable tuple of capability labels."""
        labels: list[str] = []
        if self.requires_live_telemetry:
            labels.append("Live telemetry")
        if self.provides_fast_analytics:
            labels.append("Analytics artifacts")
        if self.supports_training:
            labels.append("Training")
        if self.supports_policy_load:
            labels.append("Policy evaluation")
        if self.supports_multi_agent:
            labels.append("Multi-agent")
        return tuple(labels)


def get_worker_catalog() -> Tuple[WorkerDefinition, ...]:
    """Return the catalog of worker integrations recognised by the UI.

    Workers available:
    - BALROG: LLM-based agents using BALROG benchmark framework
    - CleanRL: Single/multi-agent RL with clean implementations (PPO, DQN, SAC, TD3, etc.)
    - XuanCe: Comprehensive deep RL library with 50+ algorithms for single/multi-agent
    - Ray RLlib: Multi-agent distributed RL with various paradigms

    Note: PettingZoo is an environment library (not algorithms) and is supported
    BY the workers above, not as a separate worker.
    """
    return (
        WorkerDefinition(
            worker_id="mosaic_llm_worker",
            display_name="MOSAIC LLM Worker",
            description=(
                "Native MOSAIC multi-agent LLM worker with Theory of Mind. "
                "Multi-agent coordination: Independent, Shared Memory, Full Communication. "
                "Supports OpenRouter, OpenAI, Anthropic, Google, and local vLLM backends. "
                "Works with BabyAI, MiniGrid, MiniHack, Crafter, TextWorld, BabaIsAI, MultiGrid, MeltingPot. "
                "Agent strategies: Naive, Chain-of-Thought, Robust, Few-Shot, Custom."
            ),
            supports_training=False,  # LLM inference only, no RL training
            supports_policy_load=True,  # Can load LLM configuration
            requires_live_telemetry=True,  # Emits step/episode telemetry
            provides_fast_analytics=False,  # No pre-computed analytics
            supports_multi_agent=True,  # Multi-agent coordination
        ),
        WorkerDefinition(
            worker_id="balrog_worker",
            display_name="BALROG LLM Worker",
            description=(
                "LLM-based agents using the BALROG benchmark framework. "
                "Supports OpenAI, Anthropic Claude, Google Gemini, and local vLLM backends. "
                "Works with BabyAI/MiniGrid, MiniHack, Crafter, and TextWorld environments. "
                "Agent reasoning strategies: naive, chain-of-thought (cot), few-shot."
            ),
            supports_training=False,  # LLM inference only, no RL training
            supports_policy_load=True,  # Can load LLM configuration
            requires_live_telemetry=True,  # Emits step/episode telemetry
            provides_fast_analytics=False,  # No pre-computed analytics
            supports_multi_agent=False,  # Single-agent LLM interaction
        ),
        WorkerDefinition(
            worker_id="cleanrl_worker",
            display_name="CleanRL Worker",
            description=(
                "Reinforcement learning using CleanRL implementations. "
                "Clean, single-file implementations of popular algorithms: PPO, DQN, SAC, TD3, DDPG. "
                "Supports single-agent Gymnasium environments and multi-agent PettingZoo (MA-Atari). "
                "FastLane integration for live training visualization."
            ),
            supports_training=True,
            supports_policy_load=True,
            requires_live_telemetry=True,
            provides_fast_analytics=True,
            supports_multi_agent=True,
        ),
        WorkerDefinition(
            worker_id="xuance_worker",
            display_name="XuanCe Worker",
            description=(
                "Comprehensive deep RL library with 50+ algorithms. "
                "Single-agent: DQN, PPO, SAC, TD3, DreamerV3. Multi-agent: MAPPO, QMIX, MADDPG, VDN, COMA. "
                "Works with Gymnasium, PettingZoo, SMAC, and Google Football environments. "
                "PyTorch backend with TensorBoard/WandB logging."
            ),
            supports_training=True,
            supports_policy_load=True,  # InteractiveRuntime: init_agent + select_action IPC
            requires_live_telemetry=False,  # Phase 1: No FastLane yet
            provides_fast_analytics=False,
            supports_multi_agent=True,
        ),
        WorkerDefinition(
            worker_id="ray_worker",
            display_name="Ray RLlib Worker",
            description=(
                "Distributed multi-agent reinforcement learning using Ray RLlib. "
                "Supports multiple training paradigms: Parameter Sharing, Independent Learning, "
                "Self-Play, and Shared Value Function (CTDE). "
                "Works with PettingZoo environments (SISL, Classic, Butterfly, MPE). "
                "FastLane integration for live training visualization with per-worker grid."
            ),
            supports_training=True,
            supports_policy_load=True,
            requires_live_telemetry=False,
            provides_fast_analytics=True,
            supports_multi_agent=True,
        ),
        WorkerDefinition(
            worker_id="chess_worker",
            display_name="Chess LLM Worker",
            description=(
                "LLM-based chess player using llm_chess prompting style. "
                "Multi-turn conversation: LLM can request board state, legal moves, or make moves. "
                "Supports OpenAI, Anthropic, and local vLLM backends. "
                "Works with PettingZoo chess_v6 environment. Regex validation with retry on invalid moves."
            ),
            supports_training=False,  # LLM inference only
            supports_policy_load=True,  # Can load LLM configuration
            requires_live_telemetry=True,  # Emits action/move telemetry
            provides_fast_analytics=False,
            supports_multi_agent=True,  # PettingZoo AEC turn-based
        ),
        WorkerDefinition(
            worker_id="human_worker",
            display_name="MOSAIC Human Worker",
            description=(
                "Native MOSAIC human-in-the-loop action selection. "
                "Keyboard shortcuts and click-to-select for all supported environments. "
                "Works with MiniGrid, BabyAI, Crafter, MiniHack, NetHack, PettingZoo games. "
                "Enables human vs LLM and human vs RL policy comparisons."
            ),
            supports_training=False,  # Human doesn't train
            supports_policy_load=False,  # No policy to load
            requires_live_telemetry=True,  # Emits action telemetry
            provides_fast_analytics=False,
            supports_multi_agent=True,  # Multi-agent turn-based games
        ),
        WorkerDefinition(
            worker_id="mctx_worker",
            display_name="MCTX GPU Worker",
            description=(
                "GPU-accelerated MCTS training using mctx + Pgx. "
                "Implements AlphaZero, MuZero, and Gumbel MuZero algorithms. "
                "Runs thousands of parallel games on GPU via JAX for massive speedup. "
                "Supports Chess, Go (9x9/19x19), Shogi, Connect Four, Othello, Backgammon, Hex."
            ),
            supports_training=True,  # Self-play training
            supports_policy_load=True,  # Can load trained policies
            requires_live_telemetry=False,  # GPU batch training
            provides_fast_analytics=True,  # TensorBoard logging
            supports_multi_agent=True,  # Two-player games
        ),
        WorkerDefinition(
            worker_id="tianshou_worker",
            display_name="Tianshou Worker",
            description=(
                "Deep Reinforcement Learning platform based on PyTorch. "
                "Supports 20+ algorithms including PPO, DQN, SAC, TD3, A2C. "
                "Modular design with building blocks for policies, networks, and collectors. "
                "FastLane integration for live training visualization."
            ),
            supports_training=True,
            supports_policy_load=True,
            requires_live_telemetry=True,
            provides_fast_analytics=True,
            supports_multi_agent=True,
        ),
        WorkerDefinition(
            worker_id="marllib_worker",
            display_name="MARLlib Worker",
            description=(
                "Multi-Agent RLlib (MARLlib) with 18 algorithms across 3 paradigms. "
                "Independent Learning (IL): IA2C, IDDPG, ITRPO, IPPO. "
                "Centralized Critic (CC): MAA2C, MADDPG, MAPPO, MATRPO, HAPPO, HATRPO, COMA. "
                "Value Decomposition (VD): VDA2C, VDPPO, FACMAC, IQL, VDN, QMIX. "
                "Supports 19 environments: MPE, SMAC, Football, MAMuJoCo, Hanabi, and more. "
                "Built on Ray/RLlib with configurable policy sharing and network architectures."
            ),
            supports_training=True,
            supports_policy_load=False,
            requires_live_telemetry=False,
            provides_fast_analytics=False,
            supports_multi_agent=True,
        ),
        WorkerDefinition(
            worker_id="sb3_worker",
            display_name="Stable-Baselines3 Worker",
            description=(
                "Reliable RL implementations using Stable-Baselines3. "
                "PPO, A2C, DQN, SAC, TD3, HER. PyTorch backend. "
                "Works with single-agent Gymnasium environments."
            ),
            supports_training=True,
            supports_policy_load=False,
            requires_live_telemetry=False,
            provides_fast_analytics=False,
            supports_multi_agent=False,
        ),
        WorkerDefinition(
            worker_id="sbx_worker",
            display_name="SBX (JAX) Worker",
            description=(
                "JAX-accelerated RL using SBX (Stable-Baselines Jax). "
                "Drop-in replacement for SB3 with JAX backends for faster training. "
                "PPO, DQN, SAC, TQC, CrossQ. Works with Gymnasium environments."
            ),
            supports_training=True,
            supports_policy_load=False,
            requires_live_telemetry=False,
            provides_fast_analytics=False,
            supports_multi_agent=False,
        ),
        WorkerDefinition(
            worker_id="mushroomrl_worker",
            display_name="MushroomRL Worker",
            description=(
                "Model-free and model-based RL using MushroomRL. "
                "PPO, TRPO, SAC, TD3, DDPG. Continuous control focus. "
                "Works with Gymnasium and MuJoCo environments."
            ),
            supports_training=True,
            supports_policy_load=False,
            requires_live_telemetry=False,
            provides_fast_analytics=False,
            supports_multi_agent=False,
        ),
        WorkerDefinition(
            worker_id="pearl_worker",
            display_name="Pearl Worker",
            description=(
                "Production-ready RL using Meta's Pearl library. "
                "PPO, DQN, SAC, bandits. Modular agent design with "
                "dynamic action spaces, safety modules, and offline RL support."
            ),
            supports_training=True,
            supports_policy_load=False,
            requires_live_telemetry=False,
            provides_fast_analytics=False,
            supports_multi_agent=False,
        ),
        WorkerDefinition(
            worker_id="torchrl_worker",
            display_name="TorchRL Worker",
            description=(
                "PyTorch official RL library (TorchRL). "
                "PPO with modular collector/loss/advantage design. "
                "Works with Gymnasium environments via GymEnv wrapper."
            ),
            supports_training=True,
            supports_policy_load=False,
            requires_live_telemetry=False,
            provides_fast_analytics=False,
            supports_multi_agent=False,
        ),
        WorkerDefinition(
            worker_id="omnisafe_worker",
            display_name="OmniSafe Worker",
            description=(
                "Safe RL using OmniSafe (PKU-Alignment). "
                "PPOLag, CPO, FOCOPS, CUP and 30+ constrained RL algorithms. "
                "Works with Safety-Gymnasium environments for safe policy learning."
            ),
            supports_training=True,
            supports_policy_load=False,
            requires_live_telemetry=False,
            provides_fast_analytics=False,
            supports_multi_agent=False,
        ),
        WorkerDefinition(
            worker_id="jumanji_worker",
            display_name="Jumanji Worker (JAX)",
            description=(
                "JAX-based RL using Jumanji environments (DeepMind, ICLR 2024). "
                "A2C with environment-specific networks on Game2048, Sudoku, TSP, etc. "
                "Supports gymnax for fully JIT-compiled CartPole/Pendulum training. "
                "Hardware-accelerated on CPU, GPU, and TPU via XLA."
            ),
            supports_training=True,
            supports_policy_load=False,
            requires_live_telemetry=True,
            provides_fast_analytics=False,
            supports_multi_agent=False,
        ),
        # NOTE: PettingZoo is an environment library, not an algorithm provider.
        # PettingZoo environments are supported BY other workers (CleanRL, XuanCe, Ray RLlib, Chess, Human, MCTX).
    )


__all__ = ["WorkerDefinition", "get_worker_catalog"]
