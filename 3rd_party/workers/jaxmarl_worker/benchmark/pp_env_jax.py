"""JAX functional Predator-Prey environment.

Matches the NumPy PredatorPreyEnv from MAGIC/ic3net-envs exactly:
  n_predators=3, n_prey=1, dim=5, vision=1, max_steps=20, mode='cooperative'

Actions: 0=UP(row-1), 1=RIGHT(col+1), 2=DOWN(row+1), 3=LEFT(col-1), 4=STAY

Vocab / observation encoding (identical to the NumPy version):
  BASE          = dim * dim = 25
  OUTSIDE_CLASS = 1 + BASE  = 26   (padding cells)
  PREY_CLASS    = 2 + BASE  = 27
  PREDATOR_CLASS= 3 + BASE  = 28
  vocab_size    = 1 + 1 + BASE + 1 + 1 = 29

Observation per predator:
  padded grid (dim+2*vision, dim+2*vision) = (7, 7)
  1-hot encoding shape: (7, 7, 29)
  extract (3, 3, 29) patch centred on predator
  flatten → 261 dims

Reward (cooperative, per predator):
  - TIMESTEP_PENALTY = -0.05  (every predator every step)
  - on-prey predators get  +0.05 * n_on_prey   (replaces the -0.05 base)
  - episode done when ALL predators have reached prey OR step >= max_steps

API:
  class PredatorPreyJAX:
      num_agents = 3
      _obs_dim   = 261
      def reset(key)             → (obs_dict, state)
      def step(key, state, actions) → (obs_dict, state, reward_dict, done_dict, info)

All methods are pure-functional and vmap-compatible.
"""

from __future__ import annotations
from typing import NamedTuple, Dict, Tuple

import jax
import jax.numpy as jnp
import chex

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_PREDATORS = 3
N_PREY      = 1
DIM         = 5
VISION      = 1
MAX_STEPS   = 20
N_ACTIONS   = 5

BASE           = DIM * DIM                      # 25
OUTSIDE_CLASS  = 1 + BASE                       # 26  (padding class)
PREY_CLASS     = 2 + BASE                       # 27
PREDATOR_CLASS = 3 + BASE                       # 28
VOCAB_SIZE     = 1 + 1 + BASE + 1 + 1          # 29

PATCH          = 2 * VISION + 1                 # 3
OBS_DIM        = VOCAB_SIZE * PATCH * PATCH     # 261

PADDED_DIM     = DIM + 2 * VISION              # 7

TIMESTEP_PENALTY = -0.05
POS_PREY_REWARD  =  0.05

# Action deltas: (delta_row, delta_col)
# 0=UP(-1,0) 1=RIGHT(0,+1) 2=DOWN(+1,0) 3=LEFT(0,-1) 4=STAY(0,0)
ACTION_DROW = jnp.array([-1,  0,  1,  0, 0], dtype=jnp.int32)
ACTION_DCOL = jnp.array([ 0,  1,  0, -1, 0], dtype=jnp.int32)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class PPState(NamedTuple):
    predator_locs: chex.Array   # (N_PREDATORS, 2)  int32  [row, col]
    prey_locs:     chex.Array   # (N_PREY,      2)  int32
    reached_prey:  chex.Array   # (N_PREDATORS,)    bool
    step:          chex.Array   # ()                int32


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def _build_base_grid() -> chex.Array:
    """Build static padded base grid with cell IDs in interior and OUTSIDE_CLASS in border.

    Returns array of shape (PADDED_DIM, PADDED_DIM) int32.
    Interior cell (r, c) (0-indexed, both in [0, DIM)) has value r*DIM+c.
    Border cells have value OUTSIDE_CLASS.
    """
    interior = jnp.arange(BASE, dtype=jnp.int32).reshape(DIM, DIM)
    padded   = jnp.pad(interior, VISION, constant_values=OUTSIDE_CLASS)
    return padded  # (PADDED_DIM, PADDED_DIM)


BASE_GRID = _build_base_grid()


def _make_onehot_grid(base_grid: chex.Array) -> chex.Array:
    """1-hot encode a (H, W) int32 grid into (H, W, VOCAB_SIZE) float32."""
    return jax.nn.one_hot(base_grid, VOCAB_SIZE, dtype=jnp.float32)  # (H, W, V)


# Pre-computed static 1-hot grid (interior + outside border, no agents yet)
EMPTY_ONEHOT_GRID: chex.Array = _make_onehot_grid(BASE_GRID)


def _scatter_agent(grid: chex.Array, row: chex.Array, col: chex.Array,
                   cls: int) -> chex.Array:
    """Set 1-hot class `cls` at padded position (row+VISION, col+VISION)."""
    pr = row + VISION
    pc = col + VISION
    return grid.at[pr, pc, cls].set(1.0)


def _get_obs_for_predator(grid_with_agents: chex.Array, pred_row: chex.Array,
                           pred_col: chex.Array) -> chex.Array:
    """Extract (PATCH, PATCH, VOCAB_SIZE) patch and flatten to OBS_DIM.

    Uses jax.lax.dynamic_slice for vmap-compatibility.
    """
    # Predator is at (pred_row+VISION, pred_col+VISION) in the padded grid.
    # The PATCH is centred there, so start_indices are (pred_row, pred_col, 0).
    patch = jax.lax.dynamic_slice(
        grid_with_agents,
        (pred_row, pred_col, jnp.int32(0)),
        (PATCH, PATCH, VOCAB_SIZE),
    )  # (3, 3, 29)
    return patch.reshape(-1)  # (261,)


def _get_obs(state: PPState) -> Dict[str, chex.Array]:
    """Build observation dict for all predators (pure functional)."""
    grid = EMPTY_ONEHOT_GRID

    # Scatter predators
    def _scatter_pred(g, i):
        r = state.predator_locs[i, 0]
        c = state.predator_locs[i, 1]
        return _scatter_agent(g, r, c, PREDATOR_CLASS)

    grid = _scatter_pred(grid, 0)
    grid = _scatter_pred(grid, 1)
    grid = _scatter_pred(grid, 2)

    # Scatter prey
    grid = _scatter_agent(grid, state.prey_locs[0, 0], state.prey_locs[0, 1], PREY_CLASS)

    # Extract per-predator patch
    def _extract(i):
        r = state.predator_locs[i, 0]
        c = state.predator_locs[i, 1]
        return _get_obs_for_predator(grid, r, c)

    obs = {f"agent_{i}": _extract(i) for i in range(N_PREDATORS)}
    return obs


# ---------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------

def _move_predator(loc: chex.Array, action: chex.Array,
                   reached: chex.Array) -> chex.Array:
    """Move one predator given action, freeze if already reached prey."""
    dr = ACTION_DROW[action]
    dc = ACTION_DCOL[action]
    new_row = jnp.clip(loc[0] + dr, 0, DIM - 1)
    new_col = jnp.clip(loc[1] + dc, 0, DIM - 1)
    new_loc = jnp.array([new_row, new_col], dtype=jnp.int32)
    # If already reached prey, freeze
    return jnp.where(reached, loc, new_loc)


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

def _compute_rewards(new_state: PPState) -> chex.Array:
    """Compute cooperative rewards: shape (N_PREDATORS,) float32."""
    prey_row = new_state.prey_locs[0, 0]
    prey_col = new_state.prey_locs[0, 1]

    def _on_prey(i):
        return (new_state.predator_locs[i, 0] == prey_row) & \
               (new_state.predator_locs[i, 1] == prey_col)

    on_prey = jnp.stack([_on_prey(i) for i in range(N_PREDATORS)])  # (3,)
    n_on_prey = on_prey.sum().astype(jnp.float32)

    # Cooperative mode: on-prey gets +0.05*n_on_prey, others get -0.05
    rewards = jnp.where(
        on_prey,
        POS_PREY_REWARD * n_on_prey,
        jnp.float32(TIMESTEP_PENALTY),
    )
    return rewards


# ---------------------------------------------------------------------------
# Main environment class
# ---------------------------------------------------------------------------

class PredatorPreyJAX:
    """JAX functional Predator-Prey environment.

    Pure functional — all state is in the PPState NamedTuple.
    Fully vmap-compatible.
    """

    num_agents = N_PREDATORS
    _obs_dim   = OBS_DIM       # 261

    # ------------------------------------------------------------------
    def _init_state(self, key: chex.PRNGKey) -> PPState:
        """Randomly place n_predators+n_prey agents without replacement."""
        n_total = N_PREDATORS + N_PREY
        idx = jax.random.choice(key, BASE, (n_total,), replace=False)
        rows = (idx // DIM).astype(jnp.int32)
        cols = (idx % DIM).astype(jnp.int32)
        locs = jnp.stack([rows, cols], axis=1)  # (n_total, 2)
        predator_locs = locs[:N_PREDATORS]       # (3, 2)
        prey_locs     = locs[N_PREDATORS:]       # (1, 2)
        return PPState(
            predator_locs = predator_locs,
            prey_locs     = prey_locs,
            reached_prey  = jnp.zeros(N_PREDATORS, dtype=jnp.bool_),
            step          = jnp.int32(0),
        )

    # ------------------------------------------------------------------
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], PPState]:
        """Reset environment.

        Returns:
            obs_dict: {"agent_0": ..., "agent_1": ..., "agent_2": ...}
            state: PPState
        """
        state = self._init_state(key)
        obs   = _get_obs(state)
        return obs, state

    # ------------------------------------------------------------------
    def step(
        self,
        key: chex.PRNGKey,
        state: PPState,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], PPState, Dict[str, chex.Array], Dict[str, chex.Array], Dict]:
        """Step environment.

        Args:
            key:     PRNGKey (used for auto-reset on episode end)
            state:   current PPState
            actions: {"agent_0": int32, "agent_1": int32, "agent_2": int32}

        Returns:
            obs_dict, next_state, reward_dict, done_dict, info
        """
        action_arr = jnp.stack(
            [actions[f"agent_{i}"] for i in range(N_PREDATORS)], axis=0
        ).astype(jnp.int32)

        # Move each predator (freeze if already reached prey)
        new_locs = jnp.stack([
            _move_predator(
                state.predator_locs[i], action_arr[i], state.reached_prey[i]
            )
            for i in range(N_PREDATORS)
        ], axis=0)  # (3, 2)

        prey_row = state.prey_locs[0, 0]
        prey_col = state.prey_locs[0, 1]

        # Update reached_prey: once reached, stays reached
        newly_on_prey = jnp.stack([
            (new_locs[i, 0] == prey_row) & (new_locs[i, 1] == prey_col)
            for i in range(N_PREDATORS)
        ], axis=0)  # (3,)
        new_reached = state.reached_prey | newly_on_prey

        new_step = state.step + jnp.int32(1)

        new_state = PPState(
            predator_locs = new_locs,
            prey_locs     = state.prey_locs,
            reached_prey  = new_reached,
            step          = new_step,
        )

        # Compute rewards
        rewards_arr = _compute_rewards(new_state)

        # Done condition
        all_reached = jnp.all(new_reached)
        timed_out   = new_step >= MAX_STEPS
        done        = all_reached | timed_out

        # Auto-reset: if done, reset state for the next episode.
        # Use tree_map + where so it is vmap-compatible in all JAX versions.
        key, reset_key = jax.random.split(key)
        reset_state = self._init_state(reset_key)
        final_state = jax.tree_util.tree_map(
            lambda r, n: jnp.where(done, r, n),
            reset_state,
            new_state,
        )

        obs = _get_obs(final_state)

        reward_dict = {f"agent_{i}": rewards_arr[i] for i in range(N_PREDATORS)}
        done_dict   = {f"agent_{i}": done for i in range(N_PREDATORS)}
        done_dict["__all__"] = done

        return obs, final_state, reward_dict, done_dict, {}


# ---------------------------------------------------------------------------
# Convenience singleton
# ---------------------------------------------------------------------------

def make_predator_prey_jax() -> PredatorPreyJAX:
    return PredatorPreyJAX()
