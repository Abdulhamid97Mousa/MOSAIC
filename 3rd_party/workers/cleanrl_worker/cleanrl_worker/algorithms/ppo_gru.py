# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
# Extended to support GRU-based recurrent policies for partially observable environments
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

# Import algorithm-agnostic components
from cleanrl_worker.agents import GRUAgent
from cleanrl_worker.fastlane import maybe_wrap_env
from cleanrl_worker.wrappers.minigrid import is_minigrid_env, make_env as make_minigrid_env
from cleanrl_worker.wrappers.mosaic_multigrid import is_mosaic_env, make_env as make_mosaic_env

# Setup logger for this module
logger = logging.getLogger(__name__)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """if toggled, save the final agent checkpoint and run an evaluation pass"""
    upload_model: bool = False
    """if toggled, upload the saved checkpoint to Hugging Face Hub"""
    hf_entity: str = ""
    """huggingface.co entity (username or org) used when uploading models"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    procedural_generation: bool = True
    """if toggled, each episode will use different random level layouts (standard RL training). If False, all episodes use the same fixed layout."""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""

    # GRU specific arguments
    gru_hidden_size: int = 128
    """hidden size of the GRU"""
    gru_num_layers: int = 1
    """number of GRU layers"""

    # MiniGrid/BabyAI/MOSAIC specific arguments
    max_episode_steps: int = 200
    """maximum steps per episode for MiniGrid/BabyAI/MOSAIC environments"""
    reward_scale: float = 1.0
    """reward scaling factor (BabyAI paper uses 20.0)"""
    view_size: Optional[int] = None
    """agent view size for MOSAIC MultiGrid (3, 5, 7, etc.). None = use environment default"""

    # Action masking arguments
    mask_invalid_actions: bool = False
    """if toggled, mask invalid actions during training (speeds up learning)"""
    invalid_actions: Optional[list[int]] = None
    """list of action indices to mask (e.g., [0, 6, 7] for noop, toggle, done). If None and mask_invalid_actions=True, defaults to [0, 6, 7]"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # Respect MOSAIC_RUN_DIR for custom_scripts compatibility
    mosaic_run_dir = os.environ.get("MOSAIC_RUN_DIR")

    # Set CleanRL environment variables for FastLane integration
    if not os.environ.get("CLEANRL_NUM_ENVS"):
        os.environ["CLEANRL_NUM_ENVS"] = str(args.num_envs)
    if not os.environ.get("CLEANRL_RUN_ID"):
        os.environ["CLEANRL_RUN_ID"] = os.environ.get("MOSAIC_RUN_ID", run_name)
    if not os.environ.get("CLEANRL_AGENT_ID"):
        os.environ["CLEANRL_AGENT_ID"] = "ppo-gru-agent"
    logger.info(f"FastLane config: RUN_ID={os.environ.get('CLEANRL_RUN_ID')}, NUM_ENVS={os.environ.get('CLEANRL_NUM_ENVS')}")

    if args.track:
        import wandb

        # Configure wandb to use MOSAIC_RUN_DIR if available
        wandb_kwargs = {
            "project": args.wandb_project_name,
            "entity": args.wandb_entity,
            "sync_tensorboard": True,
            "config": vars(args),
            "name": run_name,
            "monitor_gym": True,
            "save_code": True,
        }

        if mosaic_run_dir:
            wandb_kwargs["dir"] = mosaic_run_dir
            logger.info(f"Using MOSAIC_RUN_DIR for wandb: {mosaic_run_dir}")

        wandb.init(**wandb_kwargs)

    if mosaic_run_dir:
        writer = SummaryWriter(f"{mosaic_run_dir}/tensorboard")
        logger.info(f"Using MOSAIC_RUN_DIR for tensorboard: {mosaic_run_dir}/tensorboard")
    else:
        writer = SummaryWriter(f"runs/{run_name}")

    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Set MOSAIC_VIEW_SIZE environment variable if specified
    if args.view_size is not None:
        os.environ["MOSAIC_VIEW_SIZE"] = str(args.view_size)
        logger.info(f"Set MOSAIC_VIEW_SIZE={args.view_size}")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_mosaic_env(
                args.env_id,
                i,
                args.capture_video,
                run_name,
                args.max_episode_steps,
                args.reward_scale,
                args.view_size,
            ) if is_mosaic_env(args.env_id) else make_minigrid_env(
                args.env_id,
                i,
                args.capture_video,
                run_name,
                args.max_episode_steps,
                args.reward_scale,
                args.procedural_generation,
            )
            for i in range(args.num_envs)
        ],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = GRUAgent(envs, gru_hidden_size=args.gru_hidden_size, gru_num_layers=args.gru_num_layers).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Create action mask if enabled (for invalid action masking)
    if args.mask_invalid_actions:
        invalid_actions = args.invalid_actions if args.invalid_actions is not None else [0, 6, 7]
        action_mask = torch.zeros(envs.single_action_space.n, device=device)
        for invalid_action in invalid_actions:
            action_mask[invalid_action] = float('-inf')
        logger.info(f"Action masking enabled. Masking actions: {invalid_actions}")
    else:
        action_mask = None

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    next_gru_state = torch.zeros(agent.gru.num_layers, args.num_envs, agent.gru.hidden_size).to(device)

    for iteration in range(1, args.num_iterations + 1):
        initial_gru_state = next_gru_state.clone()

        # Annealing the rate if instructed to do so
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, next_gru_state = agent.get_action_and_value(
                    next_obs, next_gru_state, next_done, action_mask=action_mask
                )
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "episode" in infos:
                # Batched episode stats style (gymnasium SyncVectorEnv)
                episode_mask = infos.get("_episode", [True] * len(infos["episode"]["r"]))
                for has_ep, r, l in zip(episode_mask, infos["episode"]["r"], infos["episode"]["l"]):
                    if has_ep:
                        logger.info(f"global_step={global_step}, episodic_return={r}")
                        writer.add_scalar("charts/episodic_return", r, global_step)
                        writer.add_scalar("charts/episodic_length", l, global_step)
            elif "final_info" in infos:
                # Per-env info dict style (newer gymnasium versions)
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        logger.info(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs, next_gru_state, next_done).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds],
                    initial_gru_state[:, mbenvinds],
                    b_dones[mb_inds],
                    b_actions.long()[mb_inds],
                    action_mask=action_mask,
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        logger.debug(f"SPS: {int(global_step / (time.time() - start_time))}")
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # Save model checkpoint every 1M steps if MOSAIC_RUN_DIR is set
        if mosaic_run_dir and global_step % 1000000 == 0:
            checkpoint_dir = Path(mosaic_run_dir) / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model_path = checkpoint_dir / f"step_{global_step}.pth"
            torch.save({
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step,
                'args': vars(args),
            }, model_path)
            logger.info(f"Model checkpoint saved to {model_path}")

    # Save final model
    if args.save_model or mosaic_run_dir:
        if mosaic_run_dir:
            checkpoint_dir = Path(mosaic_run_dir) / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model_path = checkpoint_dir / "final_train_model.pth"
        else:
            model_path = f"runs/{run_name}/final_model.pth"

        torch.save({
            'model_state_dict': agent.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
            'args': vars(args),
        }, model_path)
        logger.info(f"Final model saved to {model_path}")

    envs.close()
    writer.close()
