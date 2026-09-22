"""Tests for jaxmarl_worker.runtime._load_actor_params.

These tests exercise checkpoint loading without JAX GPU allocation by using
numpy arrays saved to a tmp .npz file.  The function must:
  - accept >= 6 arrays and build the correct Flax param tree
  - raise ValueError when the checkpoint has < 6 arrays
  - raise ValueError when obs_dim mismatches the Dense_0 kernel's first dim
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _save_fake_checkpoint(path: Path, obs_dim: int = 16, hidden_dim: int = 32,
                          action_dim: int = 8, n_arrays: int = 6) -> None:
    """Write a minimal .npz that satisfies _load_actor_params shape expectations."""
    arrays = {}
    # Layout: arr_0=D0/bias, arr_1=D0/kernel, arr_2=D1/bias, arr_3=D1/kernel,
    #         arr_4=D2/bias, arr_5=D2/kernel  (actor head only)
    shapes = [
        (hidden_dim,),           # arr_0: Dense_0/bias
        (obs_dim, hidden_dim),   # arr_1: Dense_0/kernel  <-- checked against obs_dim
        (hidden_dim,),           # arr_2: Dense_1/bias
        (hidden_dim, hidden_dim),# arr_3: Dense_1/kernel
        (action_dim,),           # arr_4: Dense_2/bias
        (hidden_dim, action_dim),# arr_5: Dense_2/kernel
    ]
    for i, shape in enumerate(shapes[:n_arrays]):
        arrays[f"arr_{i}"] = np.zeros(shape, dtype=np.float32)
    np.savez(str(path), **arrays)


class TestLoadActorParams:

    def test_success_returns_param_tree(self, tmp_path: Path) -> None:
        """Valid 6-array checkpoint returns a nested params dict."""
        ckpt = tmp_path / "actor.npz"
        obs_dim, hidden_dim, action_dim = 16, 32, 8
        _save_fake_checkpoint(ckpt, obs_dim=obs_dim, hidden_dim=hidden_dim,
                              action_dim=action_dim)

        from jaxmarl_worker.runtime import _load_actor_params
        params = _load_actor_params(ckpt, obs_dim=obs_dim, hidden_dim=hidden_dim,
                                    action_dim=action_dim)

        assert "params" in params
        assert "Dense_0" in params["params"]
        assert "Dense_1" in params["params"]
        assert "Dense_2" in params["params"]
        assert params["params"]["Dense_0"]["kernel"].shape == (obs_dim, hidden_dim)
        assert params["params"]["Dense_2"]["bias"].shape == (action_dim,)

    def test_too_few_arrays_raises(self, tmp_path: Path) -> None:
        """Fewer than 6 arrays in checkpoint must raise ValueError."""
        ckpt = tmp_path / "short.npz"
        _save_fake_checkpoint(ckpt, n_arrays=4)

        from jaxmarl_worker.runtime import _load_actor_params
        with pytest.raises(ValueError, match="Expected"):
            _load_actor_params(ckpt, obs_dim=16)

    def test_obs_dim_mismatch_raises(self, tmp_path: Path) -> None:
        """Passing wrong obs_dim must raise ValueError (kernel shape check)."""
        ckpt = tmp_path / "wrong_dim.npz"
        _save_fake_checkpoint(ckpt, obs_dim=16)

        from jaxmarl_worker.runtime import _load_actor_params
        with pytest.raises(ValueError, match="obs_dim mismatch"):
            _load_actor_params(ckpt, obs_dim=999)

    def test_accepts_more_than_six_arrays(self, tmp_path: Path) -> None:
        """IPPO saves ActorCritic (8 arrays); loader must take only the first 6."""
        ckpt = tmp_path / "ippo.npz"
        obs_dim, hidden_dim = 16, 32
        # Simulate ActorCritic: first 6 = actor, arr_6/arr_7 = value head
        _save_fake_checkpoint(ckpt, obs_dim=obs_dim, hidden_dim=hidden_dim, n_arrays=6)
        # Manually add value head arrays
        data = dict(np.load(str(ckpt)))
        data["arr_6"] = np.zeros((hidden_dim,), dtype=np.float32)   # Dense_3/bias
        data["arr_7"] = np.zeros((hidden_dim, 1), dtype=np.float32) # Dense_3/kernel
        np.savez(str(ckpt), **data)

        from jaxmarl_worker.runtime import _load_actor_params
        params = _load_actor_params(ckpt, obs_dim=obs_dim, hidden_dim=hidden_dim)
        # Only 3 Dense layers expected in the actor
        assert set(params["params"].keys()) == {"Dense_0", "Dense_1", "Dense_2"}
