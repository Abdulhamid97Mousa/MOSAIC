"""Permute actor output layer from JAX v1 action ordering to gym v2.

JAX v1:  left=0  right=1  forward=2  pickup=3  drop=4  noop=5,6,7
gym v2:  noop=0  left=1   right=2    forward=3  pickup=4  drop=5  toggle=6  done=7

Supported checkpoint formats (auto-detected by shape):
  arr_*     : MAPPO, IPPO, IPPO_AGENTID, MAPPO_AGENTID, MAPPO_TEAMOBS
  actor_*   : HAPPO, VDPPO
  agent_q_* : QMIX  (mixer_* keys are untouched; they have no action dim)
  commnet_* : CommNet
  ic3net_*  : IC3Net  (ic3net_0/1 are the 2-class gate, shape (2,)/(256,2) -- auto-excluded)

Detection: finds the unique (8,) bias key and (*, 8) kernel key. Aborts if
ambiguous (multiple pairs or zero pair). Prints a before/after sanity table.

Usage (from mosaic/ root):
  python 3rd_party/workers/jaxmarl_worker/jaxmarl_worker/tools/permute_actor_checkpoint.py \\
      var/trainer/mosaic_multigrid/BB_m256_20k/cooperative/2v0/MAPPO_GLOBAL_INDAGOBS/checkpoints/bb-G-2v0/final.npz

  Overwrites src in-place. A .v1_backup must already exist (script checks).

  To write to a different destination:
  python ... src.npz dst.npz
"""
from __future__ import annotations

import sys
import numpy as np
from pathlib import Path

PERM     = [5, 0, 1, 2, 3, 4, 6, 7]
V1_NAMES = ["left", "right", "forward", "pickup", "drop",   "noop",   "noop",   "noop"]
V2_NAMES = ["noop", "left",  "right",   "forward","pickup", "drop",   "toggle", "done"]

N_ACTIONS = 8


def _detect_action_layers(ck: dict[str, np.ndarray]) -> tuple[str, str]:
    """Return (bias_key, kernel_key) for the action-output layer.

    Looks for a unique pair: bias shape == (8,), kernel shape == (*, 8).
    Raises SystemExit with a diagnostic if detection is ambiguous.
    """
    bias_keys   = [k for k, v in ck.items() if v.shape == (N_ACTIONS,)]
    kernel_keys = [k for k, v in ck.items()
                   if len(v.shape) == 2 and v.shape[1] == N_ACTIONS]

    if not bias_keys or not kernel_keys:
        print("ERROR: no arrays with action-dim shape found.")
        print(f"  bias candidates   (shape=({N_ACTIONS},))      : {bias_keys}")
        print(f"  kernel candidates (shape=(*,{N_ACTIONS})): {kernel_keys}")
        print("\nAll keys:")
        for k, v in sorted(ck.items()):
            print(f"  {k}: {v.shape}")
        sys.exit(1)

    if len(bias_keys) == 1 and len(kernel_keys) == 1:
        return bias_keys[0], kernel_keys[0]

    # Multiple candidates -- try to pair by numeric suffix proximity
    # e.g., arr_4 (bias) + arr_5 (kernel), or agent_q_2 + agent_q_3
    # The kernel key should be immediately after the bias key alphabetically.
    def _suffix(k: str) -> int:
        parts = k.rsplit("_", 1)
        try:
            return int(parts[-1])
        except ValueError:
            return -1

    for bk in bias_keys:
        bsuf = _suffix(bk)
        prefix = bk.rsplit("_", 1)[0]
        # kernel key: same prefix, suffix == bsuf+1 OR same prefix substring
        matched = [kk for kk in kernel_keys
                   if _suffix(kk) == bsuf + 1 and kk.rsplit("_", 1)[0] == prefix]
        if len(matched) == 1:
            return bk, matched[0]

    print("ERROR: could not auto-pair bias/kernel keys. Manual inspection needed.")
    print(f"  bias candidates   : {bias_keys}")
    print(f"  kernel candidates : {kernel_keys}")
    print("\nAll keys:")
    for k, v in sorted(ck.items()):
        print(f"  {k}: {v.shape}")
    sys.exit(1)


def permute(src: Path, dst: Path) -> None:
    if dst == src:
        backup = Path(str(src) + ".v1_backup")
        if not backup.exists():
            print(f"ERROR: backup not found at {backup}")
            print(f"Run first:  cp {src} {backup}")
            sys.exit(1)

    ck = dict(np.load(src))
    keys = sorted(ck.keys())
    print(f"Keys: {keys}")

    bias_key, kernel_key = _detect_action_layers(ck)
    print(f"\nDetected action layers:")
    print(f"  bias   key: {bias_key}   shape: {ck[bias_key].shape}")
    print(f"  kernel key: {kernel_key} shape: {ck[kernel_key].shape}")

    bias_before   = ck[bias_key].copy()
    kernel_before = ck[kernel_key].copy()

    print("\nBefore permutation (col index -> action name):")
    for i, name in enumerate(V1_NAMES):
        print(f"  col {i}: {name:10s}  bias={bias_before[i]:.6f}")

    ck[bias_key]   = bias_before[PERM]
    ck[kernel_key] = kernel_before[:, PERM]

    print("\nAfter permutation (col index -> action name):")
    for i, name in enumerate(V2_NAMES):
        print(f"  col {i}: {name:10s}  bias={ck[bias_key][i]:.6f}")

    # Sanity: noop bias (new col 0) should equal old col 5
    assert ck[bias_key][0] == bias_before[5], "noop bias mismatch after permutation"
    assert ck[bias_key][1] == bias_before[0], "left bias mismatch after permutation"

    np.savez(dst, **ck)
    print(f"\nWritten: {dst}")
    print(f"Permutation: {PERM}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("Usage: permute_actor_checkpoint.py <src.npz> [dst.npz]")
        sys.exit(1)
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src
    permute(src, dst)
