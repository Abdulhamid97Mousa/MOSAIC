"""
PyTorch MAGIC training script for the NumPy Predator-Prey environment.

Adapts MAGIC/trainer.py directly (no imports from that path).
Run from: /usrhome/Hamid/projects/mosaic/3rd_party/workers/jaxmarl_worker/
Command:  .venv/bin/python benchmark/train_magic_pytorch.py

Saves results to /tmp/magic_comparison/pytorch_results.npz:
  update_rewards : (num_epochs,)  mean team reward per epoch (matches compare_results.py)
  wall_times     : (num_epochs,)  cumulative wall-clock seconds
  steps_per_sec  : scalar         average environment steps/second
"""

import sys
import os
import time
import argparse
from collections import namedtuple
from types import SimpleNamespace

import numpy as np
import torch
from torch import optim

# ── float64 MUST come before MAGIC is imported/instantiated ──────────────────
torch.set_default_dtype(torch.float64)
torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled  = True

# ── add MAGIC source to path ──────────────────────────────────────────────────
MAGIC_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../environments/MAGIC")
)
if MAGIC_DIR not in sys.path:
    sys.path.insert(0, MAGIC_DIR)

from magic import MAGIC  # noqa: E402 (after sys.path modification)

# ── local env ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from pp_env_numpy import PredatorPreyEnvNumpy  # noqa: E402


# =============================================================================
# Transition namedtuple (matches trainer.py)
# =============================================================================
Transition = namedtuple(
    "Transition",
    ("state", "action", "action_out", "value", "episode_mask",
     "episode_mini_mask", "next_state", "reward", "misc"),
)


# =============================================================================
# Utility functions (inline copies from MAGIC/utils.py and action_utils.py)
# =============================================================================

def merge_stat(src: dict, dest: dict):
    for k, v in src.items():
        if k not in dest:
            dest[k] = v
        elif isinstance(v, (int, float)):
            dest[k] = dest.get(k, 0) + v
        elif isinstance(v, np.ndarray):
            dest[k] = dest.get(k, 0) + v
        else:
            if isinstance(dest[k], list) and isinstance(v, list):
                dest[k].extend(v)
            elif isinstance(dest[k], list):
                dest[k].append(v)
            else:
                dest[k] = [dest[k], v]


def multinomials_log_density(actions, log_probs):
    """Sum log-probs of chosen actions across all action heads."""
    log_prob = 0
    for i in range(len(log_probs)):
        log_prob = log_prob + log_probs[i].gather(
            1, actions[:, i].long().unsqueeze(1)
        )
    return log_prob


def multinomials_log_densities(actions, log_probs):
    """Per-head log-prob (for advantages_per_action)."""
    log_prob = []
    for i in range(len(log_probs)):
        log_prob.append(
            log_probs[i].gather(1, actions[:, i].long().unsqueeze(1))
        )
    return torch.cat(log_prob, dim=-1)


def select_action(args, action_out):
    """Sample discrete action from each action head's log-softmax distribution."""
    log_p_a = action_out
    # p_a: list of lists  (dim_actions × batch) of tensors (n, num_actions)
    p_a = [[z.exp() for z in x] for x in log_p_a]
    ret = torch.stack(
        [torch.stack([torch.multinomial(x, 1).detach() for x in p]) for p in p_a]
    )
    return ret


def translate_action(args, env, action):
    """Convert sampled action tensor to numpy list for env.step()."""
    action_list = [x.squeeze().data.numpy() for x in action]
    actual = action_list
    return action_list, actual


# =============================================================================
# Trainer  (adapted from MAGIC/trainer.py)
# =============================================================================

class Trainer:
    def __init__(self, args, policy_net, env):
        self.args        = args
        self.policy_net  = policy_net
        self.env         = env
        self.display     = False
        self.last_step   = False
        self.optimizer   = optim.RMSprop(
            policy_net.parameters(),
            lr=args.lrate, alpha=0.97, eps=1e-6,
        )
        self.params = list(policy_net.parameters())

    # ------------------------------------------------------------------
    def get_episode(self, epoch: int):
        """Collect one episode and return (episode, stat)."""
        episode = []
        stat    = {}
        info    = {}

        # reset — use reset_epoch if available (our env has it)
        if hasattr(self.env, "reset_epoch"):
            state_np = self.env.reset_epoch(epoch)
        else:
            state_np = self.env.reset()

        # State shape from env: (n_agents, obs_size)
        # MAGIC expects:        (batch=1, n_agents, obs_size)
        state = torch.from_numpy(state_np).unsqueeze(0)  # (1, 3, 261)

        prev_hid = self.policy_net.init_hidden(batch_size=state.shape[0])
        #   → (n_agents, hid_size) = (3, 64)

        for t in range(self.args.max_steps):
            misc = {}

            x = [state, prev_hid]
            action_out, value, prev_hid = self.policy_net(x, info)

            # Detach hidden state periodically to truncate BPTT
            if (t + 1) % self.args.detach_gap == 0:
                prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())

            action        = select_action(self.args, action_out)
            action, actual = translate_action(self.args, self.env, action)

            next_state_np, reward, done, info = self.env.step(actual)

            # alive_mask — PP env doesn't have it, default to ones
            if "alive_mask" in info:
                misc["alive_mask"] = info["alive_mask"].reshape(reward.shape)
            else:
                misc["alive_mask"] = np.ones_like(reward)

            stat["reward"] = stat.get("reward", 0) + reward[: self.args.nfriendly]

            done = done or (t == self.args.max_steps - 1)

            episode_mask      = np.ones(reward.shape)
            episode_mini_mask = np.ones(reward.shape)
            if done:
                episode_mask = np.zeros(reward.shape)

            next_state = torch.from_numpy(next_state_np).unsqueeze(0)

            trans = Transition(
                state, action, action_out, value,
                episode_mask, episode_mini_mask,
                next_state, reward, misc,
            )
            episode.append(trans)
            state = next_state
            if done:
                break

        stat["num_steps"]   = t + 1
        stat["steps_taken"] = stat["num_steps"]

        # Terminal reward (zeros for PP)
        if hasattr(self.env, "reward_terminal"):
            term_reward = self.env.reward_terminal()
            last = episode[-1]
            episode[-1] = last._replace(reward=last.reward + term_reward)
            stat["reward"] = stat.get("reward", 0) + term_reward[: self.args.nfriendly]

        if hasattr(self.env, "get_stat"):
            merge_stat(self.env.get_stat(), stat)

        return episode, stat

    # ------------------------------------------------------------------
    def compute_grad(self, batch) -> dict:
        """
        Compute loss and backprop gradients.
        Matches trainer.py compute_grad exactly.
        """
        stat = {}
        num_actions = self.args.num_actions  # e.g. [5]
        dim_actions = self.args.dim_actions  # 1
        n           = self.args.nagents
        batch_size  = len(batch.state)

        rewards            = torch.Tensor(batch.reward)        # (T, n)
        episode_masks      = torch.Tensor(batch.episode_mask)  # (T, n)
        episode_mini_masks = torch.Tensor(batch.episode_mini_mask)  # (T, n)

        # actions: (T, 1, 1, n, 1) → transpose → view → (T, n, dim_actions)
        actions = torch.Tensor(batch.action)
        actions = actions.transpose(1, 2).view(-1, n, dim_actions)

        values     = torch.cat(batch.value, dim=0)          # (T*n, 1) or (T, n, 1)
        values     = values.view(batch_size, n)

        # action_out: list of T items, each a list of tensors per head
        # After zip: tuple of dim_actions lists, each of length T
        # After cat: (T, 1, n, num_actions) per head, then view (T*n, num_actions)
        action_out_zip = list(zip(*batch.action_out))
        action_out     = [torch.cat(a, dim=0) for a in action_out_zip]

        alive_masks = torch.Tensor(
            np.concatenate([item["alive_mask"] for item in batch.misc])
        ).view(-1)

        coop_returns  = torch.zeros(batch_size, n)
        ncoop_returns = torch.zeros(batch_size, n)
        returns       = torch.zeros(batch_size, n)
        advantages    = torch.zeros(batch_size, n)

        prev_coop_return  = 0
        prev_ncoop_return = 0

        # Reverse scan over timesteps (matches trainer.py)
        for i in reversed(range(rewards.size(0))):
            coop_returns[i]  = (rewards[i]
                                + self.args.gamma * prev_coop_return
                                * episode_masks[i])
            ncoop_returns[i] = (rewards[i]
                                + self.args.gamma * prev_ncoop_return
                                * episode_masks[i]
                                * episode_mini_masks[i])

            prev_coop_return  = coop_returns[i].clone()
            prev_ncoop_return = ncoop_returns[i].clone()

            returns[i] = (
                self.args.mean_ratio * coop_returns[i].mean()
                + (1 - self.args.mean_ratio) * ncoop_returns[i]
            )

        for i in reversed(range(rewards.size(0))):
            advantages[i] = returns[i] - values.data[i]

        if self.args.normalize_rewards:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Log-prob per step × agent  (shape after cat: (T*n, num_actions[i]))
        log_p_a = [
            action_out[i].view(-1, num_actions[i])
            for i in range(dim_actions)
        ]
        actions_flat = actions.contiguous().view(-1, dim_actions)  # (T*n, 1)

        if self.args.advantages_per_action:
            log_prob    = multinomials_log_densities(actions_flat, log_p_a)
            action_loss = -advantages.view(-1).unsqueeze(-1) * log_prob
            action_loss *= alive_masks.unsqueeze(-1)
        else:
            log_prob    = multinomials_log_density(actions_flat, log_p_a)
            action_loss = -advantages.view(-1) * log_prob.squeeze()
            action_loss *= alive_masks

        action_loss = action_loss.sum()
        stat["action_loss"] = action_loss.item()

        # Value loss
        value_loss = (values - returns).pow(2).view(-1)
        value_loss = value_loss * alive_masks
        value_loss = value_loss.sum()
        stat["value_loss"] = value_loss.item()

        loss = action_loss + self.args.value_coeff * value_loss

        # Entropy regularisation
        entropy = 0
        for lp in log_p_a:
            entropy = entropy - (lp * lp.exp()).sum()
        stat["entropy"] = entropy.item() if isinstance(entropy, torch.Tensor) else entropy
        if self.args.entr > 0:
            loss = loss - self.args.entr * entropy

        loss.backward()
        return stat

    # ------------------------------------------------------------------
    def run_batch(self, epoch: int):
        batch  = []
        stats  = {"num_episodes": 0}
        while len(batch) < self.args.batch_size:
            episode, episode_stat = self.get_episode(epoch)
            merge_stat(episode_stat, stats)
            stats["num_episodes"] += 1
            batch += episode

        stats["num_steps"] = len(batch)
        batch = Transition(*zip(*batch))
        return batch, stats

    # ------------------------------------------------------------------
    def train_batch(self, epoch: int) -> dict:
        batch, stat = self.run_batch(epoch)
        self.optimizer.zero_grad()

        s = self.compute_grad(batch)
        merge_stat(s, stat)

        # Normalise gradients by number of steps (matches trainer.py)
        for p in self.params:
            if p._grad is not None:
                p._grad.data /= stat["num_steps"]

        self.optimizer.step()
        return stat


# =============================================================================
# Args namespace
# =============================================================================

def build_args() -> SimpleNamespace:
    args = SimpleNamespace(
        # agents & network
        nagents           = 3,
        nfriendly         = 3,       # same as nagents for pure-predator run
        hid_size          = 64,
        gat_hid_size      = 32,
        gat_num_heads     = 4,
        gat_num_heads_out = 1,
        directed          = True,
        learn_second_graph= True,
        use_gat_encoder   = False,
        first_graph_complete  = False,
        second_graph_complete = False,
        self_loop_type1   = 2,
        self_loop_type2   = 2,
        message_encoder   = False,
        message_decoder   = False,
        first_gat_normalize  = False,
        second_gat_normalize = False,
        comm_init         = "uniform",
        comm_mask_zero    = False,
        advantages_per_action = False,
        # action heads — set after env is built
        naction_heads     = [5],
        num_actions       = [5],
        dim_actions       = 1,
        continuous        = False,
        # env
        obs_size          = 261,     # n_pred * vocab * (2v+1)^2 → 261 per agent
        batch_size        = 500,
        max_steps         = 20,
        # optimisation
        gamma             = 1.0,
        lrate             = 0.001,
        value_coeff       = 0.01,
        entr              = 0.01,
        detach_gap        = 10,
        normalize_rewards = False,
        mean_ratio        = 1.0,     # fully cooperative
        # training loop
        epoch_size        = 10,
        num_epochs        = 200,
        # misc
        enemy_comm        = False,
    )
    return args


# =============================================================================
# Main
# =============================================================================

def main():
    args = build_args()

    # Create output directory
    out_dir = "/tmp/magic_comparison"
    os.makedirs(out_dir, exist_ok=True)

    # Build env
    env = PredatorPreyEnvNumpy(
        n_predators=3, n_prey=1, dim=5, vision=1,
        max_steps=20, mode="cooperative", no_stay=False,
    )

    # Quick sanity check
    obs = env.reset()
    assert obs.shape == (3, 261), f"Bad obs shape: {obs.shape}"
    assert obs.dtype == np.float64, f"Bad obs dtype: {obs.dtype}"
    obs2, rew, done, info = env.step([4, 4, 4])   # all STAY
    assert rew.shape == (3,), f"Bad reward shape: {rew.shape}"
    # Rewards should be in the expected range (-0.05 to 0.15)
    assert np.all(rew >= -0.05) and np.all(rew <= 0.15), f"Reward out of range: {rew}"
    print(f"[sanity] obs={obs.shape} dtype={obs.dtype}, reward={rew}, done={done}")

    # Build MAGIC policy
    policy_net = MAGIC(args)
    print(policy_net)
    print(f"\nTotal parameters: "
          f"{sum(p.numel() for p in policy_net.parameters()):,}")

    trainer = Trainer(args, policy_net, env)

    epoch_rewards  = np.zeros(args.num_epochs)
    wall_times     = np.zeros(args.num_epochs)
    total_steps    = 0
    t_start        = time.time()

    print("\n" + "="*60)
    print(f"{'Epoch':>6}  {'MeanReward':>12}  {'Steps':>8}  {'WallTime':>10}")
    print("="*60)

    for ep in range(args.num_epochs):
        ep_stat = {}
        ep_start = time.time()

        for _ in range(args.epoch_size):
            stat = trainer.train_batch(ep)
            merge_stat(stat, ep_stat)

        ep_elapsed = time.time() - ep_start
        total_steps += ep_stat.get("num_steps", 0)

        # reward is accumulated over all agents × episodes → divide by episodes.
        # Convention: we report the sum-over-agents reward averaged per episode.
        # The JAX MAGIC script should use the same convention for a fair comparison.
        n_eps    = ep_stat.get("num_episodes", 1)
        raw_rew  = ep_stat.get("reward", 0)
        if isinstance(raw_rew, np.ndarray):
            # sum across agents, mean across episodes
            mean_rew = raw_rew.sum() / n_eps
        else:
            mean_rew = float(raw_rew) / n_eps

        epoch_rewards[ep] = mean_rew
        wall_times[ep]    = time.time() - t_start

        print(
            f"{ep+1:>6}  {mean_rew:>12.4f}  "
            f"{ep_stat.get('num_steps', 0):>8}  "
            f"{wall_times[ep]:>10.2f}s"
        )

    total_time    = time.time() - t_start
    steps_per_sec = total_steps / total_time if total_time > 0 else 0.0

    print("="*60)
    print(f"\nDone. Total steps: {total_steps:,}  "
          f"Steps/sec: {steps_per_sec:.1f}  "
          f"Total time: {total_time:.1f}s")

    # Save results
    # Key names match what compare_results.py expects: update_rewards, wall_times, steps_per_sec
    out_path = os.path.join(out_dir, "pytorch_results.npz")
    np.savez(
        out_path,
        update_rewards = epoch_rewards,   # compare_results.py reads "update_rewards"
        wall_times     = wall_times,
        steps_per_sec  = np.array(steps_per_sec),
    )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
