from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore


@dataclass
class EnvConfig:
    family: str = "???"          # "bb" | "af" | "soccer"
    variant: str = "2v2"
    view_size: int = 7
    ball_approach_coef: float = 0.01
    max_steps: int = 256
    goal_rows: Optional[list] = None


@dataclass
class PPOConfig:
    n_envs: int = 256
    n_steps: int = 256
    n_epochs: int = 4
    n_minibatches: int = 4
    hidden_dim: int = 256
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.005


@dataclass
class VDPPOConfig:
    n_envs: int = 256
    n_steps: int = 256
    n_epochs: int = 4
    n_minibatches: int = 4
    hidden_dim: int = 256
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.005
    mixer: str = "QMIX"
    mix_hidden: int = 32
    hyper_hidden: int = 32


@dataclass
class AgentIDConfig:
    enabled: bool = False


@dataclass
class TrainConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    agent_id: AgentIDConfig = field(default_factory=AgentIDConfig)
    training_reward: str = "cooperative-no-opponent"
    total_updates: int = 20000
    seed: int = 1
    log_every: int = 50
    save_every: int = 500        # unused by the training loop today; kept for schema completeness
    tensorboard: bool = True
    run_dir: str = "???"


@dataclass
class VDPPOTrainConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: VDPPOConfig = field(default_factory=VDPPOConfig)
    agent_id: AgentIDConfig = field(default_factory=AgentIDConfig)
    training_reward: str = "cooperative-no-opponent"
    total_updates: int = 20000
    seed: int = 1
    log_every: int = 50
    save_every: int = 500
    tensorboard: bool = True
    run_dir: str = "???"


@dataclass
class HyperMARLConfig:
    n_envs: int = 256
    n_steps: int = 256
    n_epochs: int = 4
    n_minibatches: int = 4
    actor_hidden: int = 64
    critic_hidden: int = 64
    hypernet_hidden: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.005
    use_learned_embeddings: bool = False
    embedding_dim: int = 4


@dataclass
class HyperMARLTrainConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    hypermarl: HyperMARLConfig = field(default_factory=HyperMARLConfig)
    agent_id: AgentIDConfig = field(default_factory=AgentIDConfig)
    training_reward: str = "cooperative-no-opponent"
    total_updates: int = 20000
    seed: int = 1
    log_every: int = 50
    save_every: int = 500
    tensorboard: bool = True
    run_dir: str = "???"


def register() -> None:
    cs = ConfigStore.instance()
    cs.store(group="env", name="base_env", node=EnvConfig)
    cs.store(group="ppo", name="base_ppo", node=PPOConfig)
    cs.store(group="vdppo_ppo", name="base_vdppo_ppo", node=VDPPOConfig)
    cs.store(group="hypermarl", name="base_hypermarl", node=HyperMARLConfig)
    cs.store(group="agent_id", name="base_agent_id", node=AgentIDConfig)
    cs.store(name="base_train", node=TrainConfig)
    cs.store(name="base_vdppo_train", node=VDPPOTrainConfig)
    cs.store(name="base_hypermarl_train", node=HyperMARLTrainConfig)


register()
