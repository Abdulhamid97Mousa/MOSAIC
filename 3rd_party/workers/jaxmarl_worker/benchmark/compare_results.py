"""Compare PyTorch MAGIC vs JAX MAGIC on Predator-Prey.

Reads:
  /tmp/magic_comparison/pytorch_results.npz
  /tmp/magic_comparison/jax_results.npz

Both files must contain:
  update_rewards  : (n_updates,) float   — per-update mean team reward
  wall_times      : (n_updates,) float   — cumulative wall-clock seconds
  steps_per_sec   : scalar               — overall throughput

Prints:
  - Summary comparison table
  - Final performance (last 10% of training)
  - Wall-clock time comparison
  - Speedup ratio
  - ASCII learning curve plots
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RESULTS_DIR   = Path("/tmp/magic_comparison")
PYTORCH_FILE  = RESULTS_DIR / "pytorch_results.npz"
JAX_FILE      = RESULTS_DIR / "jax_results.npz"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_results(path: Path) -> dict:
    if not path.exists():
        print(f"[compare] File not found: {path}")
        print("  Run the corresponding training script first.")
        sys.exit(1)
    d = np.load(str(path))
    return {k: d[k] for k in d.files}


# ---------------------------------------------------------------------------
# ASCII plot
# ---------------------------------------------------------------------------

def ascii_plot(series: np.ndarray, label: str, width: int = 70, height: int = 15):
    """Simple ASCII line plot of a 1-D series."""
    n = len(series)
    lo, hi = series.min(), series.max()
    if lo == hi:
        hi = lo + 1e-6

    # Downsample to `width` columns
    idx = np.round(np.linspace(0, n - 1, width)).astype(int)
    sampled = series[idx]

    print(f"\n  {label}  (n={n}, min={lo:+.4f}, max={hi:+.4f})")
    print("  " + "─" * (width + 4))

    for row in range(height - 1, -1, -1):
        threshold = lo + (row / (height - 1)) * (hi - lo)
        line = ""
        for val in sampled:
            line += "█" if val >= threshold else " "
        y_label = f"{threshold:+.3f}"
        print(f"  {y_label} |{line}|")

    print("  " + " " * 8 + "└" + "─" * width + "┘")
    print("  " + " " * 9 + f"update 0{' ' * (width // 2 - 4)}update {n}")


# ---------------------------------------------------------------------------
# Interpolate wall-time to compare at same elapsed time
# ---------------------------------------------------------------------------

def reward_at_time(wall_times: np.ndarray, rewards: np.ndarray,
                   target_time: float) -> float:
    """Return mean reward for all updates up to `target_time` seconds."""
    mask = wall_times <= target_time
    if mask.sum() == 0:
        return float("nan")
    return float(rewards[mask].mean())


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 72)
    print("  MAGIC Algorithm Comparison: PyTorch vs JAX  (Predator-Prey)")
    print("=" * 72)

    pytorch = load_results(PYTORCH_FILE)
    jax     = load_results(JAX_FILE)

    pt_rewards  = pytorch["update_rewards"]
    jax_rewards = jax["update_rewards"]
    pt_times    = pytorch["wall_times"]
    jax_times   = jax["wall_times"]
    pt_sps      = float(pytorch["steps_per_sec"])
    jax_sps     = float(jax["steps_per_sec"])

    pt_n  = len(pt_rewards)
    jax_n = len(jax_rewards)

    # Final performance (last 10%)
    pt_tail  = pt_rewards[int(0.9 * pt_n):]
    jax_tail = jax_rewards[int(0.9 * jax_n):]

    pt_final  = float(pt_tail.mean())
    jax_final = float(jax_tail.mean())

    # Total wall-clock time
    pt_total_time  = float(pt_times[-1])
    jax_total_time = float(jax_times[-1])

    # Speedup
    speedup = jax_sps / pt_sps if pt_sps > 0 else float("nan")

    # Wall-clock comparison at the shorter of the two total times
    common_time = min(pt_total_time, jax_total_time)
    pt_at_common  = reward_at_time(pt_times,  pt_rewards,  common_time)
    jax_at_common = reward_at_time(jax_times, jax_rewards, common_time)

    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│  Metric                         PyTorch MAGIC    JAX MAGIC      │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print(f"│  Total updates                  {pt_n:<16d} {jax_n:<16d}│")
    print(f"│  Total wall-clock time (s)      {pt_total_time:<16.1f} {jax_total_time:<16.1f}│")
    print(f"│  Steps / sec                    {pt_sps:<16,.0f} {jax_sps:<16,.0f}│")
    print(f"│  Final 10% mean_team_reward     {pt_final:<+16.4f} {jax_final:<+16.4f}│")
    print(f"│  Reward at t={common_time:.0f}s              {pt_at_common:<+16.4f} {jax_at_common:<+16.4f}│")
    print(f"│  JAX speedup vs PyTorch         {'—':<16s} {speedup:<.2f}×{' ' * 13}│")
    print("└─────────────────────────────────────────────────────────────────┘")

    # Performance comparison
    delta = jax_final - pt_final
    winner = "JAX" if delta > 0 else "PyTorch"
    print(f"\n  Final performance delta: {delta:+.4f}  ({winner} performs better)")

    if speedup > 1:
        print(f"  JAX is {speedup:.2f}× FASTER than PyTorch")
    else:
        print(f"  PyTorch is {1/speedup:.2f}× faster than JAX")

    # ASCII learning curves
    ascii_plot(pt_rewards,  "PyTorch MAGIC — mean_team_reward per update")
    ascii_plot(jax_rewards, "JAX MAGIC     — mean_team_reward per update")

    print()


if __name__ == "__main__":
    main()
