"""Launcher for RLtools algorithms.

RLtools JIT-compiles C++ at runtime via the rltools Python wrapper.
This gives near-native C++ performance with a Python API.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict

import gymnasium as gym
import rltools

from .config import RLToolsWorkerConfig

LOGGER = logging.getLogger(__name__)


def _make_env_factory(env_id: str):
    """Create an env factory with RescaleAction for continuous envs."""
    def factory():
        env = gym.make(env_id)
        if isinstance(env.action_space, gym.spaces.Box):
            env = gym.wrappers.RescaleAction(env, -1.0, 1.0)
        return env
    return factory


def run_sac(config: RLToolsWorkerConfig, work_dir: Path, log_dir: Path):
    """Run SAC using RLtools JIT-compiled C++ backend."""
    extras = config.extras
    seed = config.seed if config.seed is not None else 0
    env_factory = _make_env_factory(config.env_id)

    module = rltools.SAC(
        env_factory,
        STEP_LIMIT=config.total_timesteps,
        verbose=False,
    )
    module.set_environment_factory(env_factory)
    state = module.State(seed)

    start = time.perf_counter()
    finished = False
    while not finished:
        finished = state.step()
    elapsed = time.perf_counter() - start

    print(f"RLtools SAC training finished in {elapsed:.2f}s")


def run_td3(config: RLToolsWorkerConfig, work_dir: Path, log_dir: Path):
    """Run TD3 using RLtools JIT-compiled C++ backend."""
    seed = config.seed if config.seed is not None else 0
    env_factory = _make_env_factory(config.env_id)

    module = rltools.TD3(
        env_factory,
        STEP_LIMIT=config.total_timesteps,
        verbose=False,
    )
    module.set_environment_factory(env_factory)
    state = module.State(seed)

    start = time.perf_counter()
    finished = False
    while not finished:
        finished = state.step()
    elapsed = time.perf_counter() - start

    print(f"RLtools TD3 training finished in {elapsed:.2f}s")


ALGO_MAP: Dict[str, Callable] = {
    "sac": run_sac,
    "td3": run_td3,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--algo", type=str, required=True)
    parser.add_argument("--env-id", type=str, required=True)
    parser.add_argument("--total-timesteps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worker-id", type=str)
    parser.add_argument("--config-file", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--log-dir", type=Path)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.config_file and args.config_file.exists():
        with open(args.config_file) as f:
            config_dict = json.load(f)
            config = RLToolsWorkerConfig.from_dict(config_dict)
    else:
        config = RLToolsWorkerConfig(
            run_id=args.run_id,
            algo=args.algo,
            env_id=args.env_id,
            total_timesteps=args.total_timesteps,
            seed=args.seed,
            worker_id=args.worker_id,
        )

    runner = ALGO_MAP.get(config.algo)
    if not runner:
        raise ValueError(f"Algorithm {config.algo} not supported. Available: {list(ALGO_MAP.keys())}")

    work_dir = args.work_dir or Path(os.getcwd())
    log_dir = args.log_dir or work_dir / "log"

    runner(config, work_dir, log_dir)


if __name__ == "__main__":
    main()
