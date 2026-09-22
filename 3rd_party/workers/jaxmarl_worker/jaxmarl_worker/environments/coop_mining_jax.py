"""JAX-native coop_mining environment for jaxmarl_worker Anakin training.

Reimplements MeltingPot's coop_mining substrate in pure JAX so training can
use GPU-vectorised rollouts (jax.vmap + jax.lax.scan) instead of the CPU
dmlab2d env workers used by Mava Sebulba.

Observation layout (23233 floats, identical to MeltingPot Sebulba output):
  obs[0]    READY_TO_SHOOT  1.0 if cooldown==0, else 0.0
  obs[1:]   RGB             88x88x3 egocentric pixel view, normalised [0,1]

View geometry matches MeltingPot exactly: forward=9, backward=1, left=5,
right=5, centered=False. Agent appears at pixel (76, 44) in final 88x88 image.

Game rules (faithful to MeltingPot coop_mining):
  Grid: 27x27 fixed ASCII map with internal walls
  Iron ore (+1): mined by single agent firing beam within BEAM_LEN=3 cells
  Gold ore (+8): requires 2 different agents fire within GOLD_WINDOW=3 steps
  Actions (8): 0=noop 1=fwd 2=bck 3=strafe-L 4=strafe-R 5=turn-L 6=turn-R 7=fire
  Ore regrow: stochastic per step (IRON_REGROW=2e-4, GOLD_REGROW=8e-5)
  Episode length: MAX_STEPS steps
"""
from __future__ import annotations

import os as _os
import sys
from functools import partial
from typing import Dict, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

_JAXMARL_ROOT = _os.path.abspath(
    _os.path.join(_os.path.dirname(__file__), '../../JaxMARL')
)
if _JAXMARL_ROOT not in sys.path:
    sys.path.insert(0, _JAXMARL_ROOT)

from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.spaces import Discrete

# ================================================================
# Map data (from MeltingPot ASCII_MAP, W=wall O=ore P=spawn)
# ================================================================

_WALL_MAP_DATA = (
    (True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True),
    (True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True),
    (True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True),
    (True, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True),
    (True, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True),
    (True, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True),
    (True, False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, False, False, False, False, True, False, False, False, False, False, True),
    (True, False, False, False, True, True, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True),
    (True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True),
    (True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, True, True, False, False, False, False, False, False, False, True),
    (True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True),
    (True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True),
    (True, False, False, False, False, False, True, True, True, False, False, False, False, False, False, True, True, True, True, True, True, True, True, False, False, False, True),
    (True, False, False, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True),
    (True, False, False, False, False, False, True, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True),
    (True, False, False, False, False, False, True, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True),
    (True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, True, False, False, False, False, True),
    (True, False, False, False, False, False, False, True, False, False, False, False, False, True, True, True, True, False, False, False, False, True, False, False, False, False, True),
    (True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, True),
    (True, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True),
    (True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, True),
    (True, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, True),
    (True, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, False, False, False, True),
    (True, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, True),
    (True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True),
    (True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True),
    (True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True),
)

_SPAWN_POS_DATA = (
    (2, 2), (2, 12), (2, 18), (2, 24),
    (6, 24), (7, 2), (8, 13), (11, 24),
    (12, 2), (14, 17), (15, 24), (18, 2),
    (19, 13), (19, 24), (24, 2), (24, 9),
    (24, 17), (24, 24),
)

# ================================================================
# Constants
# ================================================================

_H, _W       = 27, 27
NUM_AGENTS   = 6
NUM_ACTIONS  = 8

# Ore slot counts and active caps (matches MeltingPot MAX_TOKENS_PER_TYPE=6)
NUM_IRON_SLOTS = 15
NUM_GOLD_SLOTS = 10
MAX_IRON       = 6
MAX_GOLD       = 6

BEAM_COOL    = 3        # steps before re-fire (MeltingPot cooldownTime=3)
BEAM_LEN     = 3        # max cells ahead beam reaches (MeltingPot beamLength=3)
GOLD_WINDOW  = 3        # cooperative-fire window in steps
IRON_REGROW  = 2e-4    # per-slot per-step regrow probability (liveRates[0])
GOLD_REGROW  = 8e-5    # per-slot per-step regrow probability (liveRates[1])
IRON_REWARD  = 1.0
GOLD_REWARD  = 8.0
MAX_STEPS    = 1000

# Pixel rendering
CELL_SIZE    = 8
IMG_SIZE     = 88       # 11 cells x 8 px = 88
PAD_PIX      = 72       # 9-cell forward view needs 9*8=72 px padding
PIX_H        = _H * CELL_SIZE   # 216
PIX_W        = _W * CELL_SIZE   # 216
OBS_DIM      = 1 + IMG_SIZE * IMG_SIZE * 3  # 23233

# Egocentric crop offsets: for direction d, extract crop at
#   start_r = r * CELL_SIZE + _CROP_R_OFFSETS[d]
#   start_c = c * CELL_SIZE + _CROP_C_OFFSETS[d]
# then rotate by rot90(k=d). Derived so agent lands at pixel (76,44) post-rotation,
# matching MeltingPot's forward=9 backward=1 left=5 right=5 centered=False view.
_CROP_R_OFFSETS = jnp.array([ 0, 32, 65, 33], dtype=jnp.int32)
_CROP_C_OFFSETS = jnp.array([32, 65, 33,  0], dtype=jnp.int32)

# Direction: N=0 E=1 S=2 W=3
_DR     = jnp.array([-1,  0,  1,  0], dtype=jnp.int32)
_DC     = jnp.array([ 0,  1,  0, -1], dtype=jnp.int32)
_TURN_L = jnp.array([ 3,  0,  1,  2], dtype=jnp.int32)  # CCW 90 deg
_TURN_R = jnp.array([ 1,  2,  3,  0], dtype=jnp.int32)  # CW 90 deg

# Render palette
_WALL_COLOR       = jnp.array([ 50,  50,  50], dtype=jnp.uint8)
_GROUND_COLOR     = jnp.array([200, 180, 140], dtype=jnp.uint8)
_IRON_COLOR       = jnp.array([120, 120, 120], dtype=jnp.uint8)
_GOLD_COLOR       = jnp.array([220, 180,  20], dtype=jnp.uint8)
_GOLD_FLASH_COLOR = jnp.array([255, 220,  80], dtype=jnp.uint8)
_AGENT_COLORS     = jnp.array([
    [220,  50,  50], [50,  50, 220], [ 50, 200,  50],
    [220, 150,  50], [150, 50, 220], [ 50, 200, 200],
], dtype=jnp.uint8)

# JAX arrays derived from data tuples (computed once at import time)
_WALL_MAP  = jnp.array(_WALL_MAP_DATA, dtype=jnp.bool_)   # (27, 27)
_SPAWN_POS = jnp.array(_SPAWN_POS_DATA, dtype=jnp.int32)  # (18, 2)

_SPAWN_POS_SET = frozenset(_SPAWN_POS_DATA)
_ORE_POS = jnp.array(
    [(r, c) for r in range(_H) for c in range(_W)
     if not _WALL_MAP_DATA[r][c] and (r, c) not in _SPAWN_POS_SET],
    dtype=jnp.int32,
)  # (538, 2) valid ore spawn positions


# ================================================================
# State
# ================================================================

@struct.dataclass
class CoopMiningState:
    agent_pos:        chex.Array  # (N, 2) int32
    agent_dir:        chex.Array  # (N,) int32
    beam_cooldown:    chex.Array  # (N,) int32
    iron_pos:         chex.Array  # (NUM_IRON_SLOTS, 2) int32 - FIXED for episode
    iron_active:      chex.Array  # (NUM_IRON_SLOTS,) bool
    gold_pos:         chex.Array  # (NUM_GOLD_SLOTS, 2) int32 - FIXED for episode
    gold_active:      chex.Array  # (NUM_GOLD_SLOTS,) bool
    gold_first_miner: chex.Array  # (NUM_GOLD_SLOTS,) int32, -1 = no first firer
    gold_first_step:  chex.Array  # (NUM_GOLD_SLOTS,) int32, step of first fire
    step:             int
    done:             bool


# ================================================================
# Pixel renderer
# ================================================================

def _render_global_map(state: CoopMiningState) -> chex.Array:
    """Build (H, W, 3) uint8 color map from game state."""
    wall_tile   = jnp.tile(_WALL_COLOR[None, None, :],   (_H, _W, 1))
    ground_tile = jnp.tile(_GROUND_COLOR[None, None, :], (_H, _W, 1))
    cmap = jnp.where(_WALL_MAP[:, :, None], wall_tile, ground_tile)

    for k in range(NUM_IRON_SLOTS):
        r = state.iron_pos[k, 0]
        c = state.iron_pos[k, 1]
        cur  = jax.lax.dynamic_slice(cmap, [r, c, 0], [1, 1, 3])
        newc = jnp.where(state.iron_active[k], _IRON_COLOR[None, None, :], cur)
        cmap = jax.lax.dynamic_update_slice(cmap, newc, [r, c, 0])

    for k in range(NUM_GOLD_SLOTS):
        r         = state.gold_pos[k, 0]
        c         = state.gold_pos[k, 1]
        is_flash  = state.gold_first_miner[k] >= 0
        ore_color = jnp.where(is_flash, _GOLD_FLASH_COLOR, _GOLD_COLOR)
        cur  = jax.lax.dynamic_slice(cmap, [r, c, 0], [1, 1, 3])
        newc = jnp.where(state.gold_active[k], ore_color[None, None, :], cur)
        cmap = jax.lax.dynamic_update_slice(cmap, newc, [r, c, 0])

    for i in range(NUM_AGENTS):
        r    = state.agent_pos[i, 0]
        c    = state.agent_pos[i, 1]
        cmap = jax.lax.dynamic_update_slice(
            cmap, _AGENT_COLORS[i][None, None, :], [r, c, 0]
        )
    return cmap  # (27, 27, 3) uint8


def _agent_view(pmap_padded: chex.Array, r: chex.Array, c: chex.Array,
                d: chex.Array) -> chex.Array:
    """Extract and rotate 88x88 egocentric view for agent at (r,c) facing d.

    Non-centered: forward=9 cells (backward=1), matching MeltingPot layout.
    Agent appears at pixel (76, 44) in final image for all directions.
    """
    start_r = r * CELL_SIZE + _CROP_R_OFFSETS[d]
    start_c = c * CELL_SIZE + _CROP_C_OFFSETS[d]
    crop = jax.lax.dynamic_slice(pmap_padded, [start_r, start_c, 0], [IMG_SIZE, IMG_SIZE, 3])
    crop = jax.lax.switch(d, [
        lambda x: x,
        lambda x: jnp.rot90(x, 1),
        lambda x: jnp.rot90(x, 2),
        lambda x: jnp.rot90(x, 3),
    ], crop)
    return crop  # (88, 88, 3) uint8


def _get_obs(state: CoopMiningState) -> Dict[str, chex.Array]:
    """Build {agent_i: (23233,) float32} matching MeltingPot obs layout."""
    cmap        = _render_global_map(state)
    pmap        = jnp.repeat(jnp.repeat(cmap, CELL_SIZE, axis=0), CELL_SIZE, axis=1)
    pmap_padded = jnp.pad(pmap, ((PAD_PIX, PAD_PIX), (PAD_PIX, PAD_PIX), (0, 0)))

    def _obs_i(i: int) -> chex.Array:
        view  = _agent_view(pmap_padded, state.agent_pos[i, 0], state.agent_pos[i, 1], state.agent_dir[i])
        rgb   = view.reshape(-1).astype(jnp.float32) / 255.0
        ready = (state.beam_cooldown[i] == 0).astype(jnp.float32)
        return jnp.concatenate([jnp.array([ready]), rgb])

    return {f"agent_{i}": _obs_i(i) for i in range(NUM_AGENTS)}


# ================================================================
# Movement (wall-aware)
# ================================================================

def _apply_movement(state: CoopMiningState, actions: chex.Array) -> CoopMiningState:
    new_dirs = jnp.where(
        actions == 5, _TURN_L[state.agent_dir],
        jnp.where(actions == 6, _TURN_R[state.agent_dir], state.agent_dir),
    )

    def _proposed(i: int) -> chex.Array:
        d  = new_dirs[i]
        a  = actions[i]
        sl = (d + 3) % 4
        sr = (d + 1) % 4
        dr = jnp.where(a == 1,  _DR[d],
             jnp.where(a == 2, -_DR[d],
             jnp.where(a == 3,  _DR[sl],
             jnp.where(a == 4,  _DR[sr], jnp.int32(0)))))
        dc = jnp.where(a == 1,  _DC[d],
             jnp.where(a == 2, -_DC[d],
             jnp.where(a == 3,  _DC[sl],
             jnp.where(a == 4,  _DC[sr], jnp.int32(0)))))
        new_r = jnp.clip(state.agent_pos[i, 0] + dr, jnp.int32(0), jnp.int32(_H - 1))
        new_c = jnp.clip(state.agent_pos[i, 1] + dc, jnp.int32(0), jnp.int32(_W - 1))
        # Stay put if target is a wall
        is_wall = _WALL_MAP[new_r, new_c]
        final_r = jnp.where(is_wall, state.agent_pos[i, 0], new_r)
        final_c = jnp.where(is_wall, state.agent_pos[i, 1], new_c)
        return jnp.array([final_r, final_c], dtype=jnp.int32)

    proposed = jnp.stack([_proposed(i) for i in range(NUM_AGENTS)])

    def _collides(i: int) -> chex.Array:
        same = jnp.all(proposed == proposed[i], axis=-1)
        same = same.at[i].set(False)
        return jnp.any(same)

    collisions = jnp.array([_collides(i) for i in range(NUM_AGENTS)])
    new_pos    = jnp.where(collisions[:, None], state.agent_pos, proposed)
    return state.replace(agent_pos=new_pos, agent_dir=new_dirs)


# ================================================================
# Fire / mining (beam length 3, wall-blocked, stochastic regrow)
# ================================================================

def _apply_fire(
    state: CoopMiningState,
    actions: chex.Array,
    key: chex.PRNGKey,
) -> Tuple[CoopMiningState, chex.Array, chex.PRNGKey]:
    rewards = jnp.zeros(NUM_AGENTS, dtype=jnp.float32)
    fires   = (actions == 7) & (state.beam_cooldown == 0)  # (N,)

    # Beam positions for each agent at distances 1..BEAM_LEN ahead
    beam_dr = _DR[state.agent_dir]  # (N,)
    beam_dc = _DC[state.agent_dir]
    beam_pos = jnp.stack([
        jnp.stack([
            jnp.clip(state.agent_pos[:, 0] + (k + 1) * beam_dr, 0, _H - 1),
            jnp.clip(state.agent_pos[:, 1] + (k + 1) * beam_dc, 0, _W - 1),
        ], axis=1)
        for k in range(BEAM_LEN)
    ], axis=1)  # (N, BEAM_LEN, 2)

    # Wall check: beam_is_wall[i, k] = True if beam cell k+1 of agent i is a wall
    beam_is_wall = _WALL_MAP[beam_pos[:, :, 0], beam_pos[:, :, 1]]  # (N, BEAM_LEN)

    # can_reach[i, k]: beam reaches cell k+1 (blocked by walls at earlier cells)
    still_going = fires
    can_reach_list = []
    for k in range(BEAM_LEN):
        can_reach_list.append(still_going)
        still_going = still_going & ~beam_is_wall[:, k]
    can_reach = jnp.stack(can_reach_list, axis=1)  # (N, BEAM_LEN)

    # ---- Iron mining ----
    iron_active = state.iron_active
    for j in range(NUM_IRON_SLOTS):
        pos_j        = state.iron_pos[j]
        beam_matches = jnp.all(beam_pos == pos_j[None, None, :], axis=-1)  # (N, BEAM_LEN)
        hits_matrix  = can_reach & beam_matches & iron_active[j]
        hits_j       = jnp.any(hits_matrix, axis=1)  # (N,)
        mined        = jnp.any(hits_j)
        rewards      = rewards + hits_j.astype(jnp.float32) * IRON_REWARD * mined.astype(jnp.float32)
        iron_active  = iron_active.at[j].set(iron_active[j] & ~mined)

    # ---- Gold mining ----
    gold_active      = state.gold_active
    gold_first_miner = state.gold_first_miner
    gold_first_step  = state.gold_first_step
    step             = state.step

    # Expire stale cooperative windows
    expired          = (gold_first_miner >= 0) & (step >= gold_first_step + GOLD_WINDOW)
    gold_first_miner = jnp.where(expired, jnp.full(NUM_GOLD_SLOTS, -1, jnp.int32), gold_first_miner)
    gold_first_step  = jnp.where(expired, jnp.zeros(NUM_GOLD_SLOTS, jnp.int32), gold_first_step)

    for j in range(NUM_GOLD_SLOTS):
        pos_j        = state.gold_pos[j]
        beam_matches = jnp.all(beam_pos == pos_j[None, None, :], axis=-1)
        hits_matrix  = can_reach & beam_matches & gold_active[j]
        hits_j       = jnp.any(hits_matrix, axis=1)  # (N,)
        hit_count    = jnp.sum(hits_j.astype(jnp.int32))
        any_hit      = hit_count > 0

        has_prior = gold_first_miner[j] >= 0
        prior     = gold_first_miner[j]
        firer     = jnp.argmax(hits_j).astype(jnp.int32)

        mine_A = (hit_count >= 2) & gold_active[j]
        mine_B = (
            any_hit & has_prior
            & (firer != prior)
            & (step < gold_first_step[j] + GOLD_WINDOW)
            & gold_active[j]
        )
        any_mine = mine_A | mine_B

        reward_A   = hits_j.astype(jnp.float32) * mine_A.astype(jnp.float32) * GOLD_REWARD
        safe_prior = jnp.maximum(prior, jnp.int32(0))
        safe_firer = jnp.clip(firer, jnp.int32(0), jnp.int32(NUM_AGENTS - 1))
        reward_B_v = jnp.zeros(NUM_AGENTS, dtype=jnp.float32)
        reward_B_v = reward_B_v.at[safe_prior].add(GOLD_REWARD)
        reward_B_v = reward_B_v.at[safe_firer].add(GOLD_REWARD)
        reward_B   = reward_B_v * mine_B.astype(jnp.float32)
        rewards    = rewards + reward_A + reward_B

        orig_active  = gold_active[j]
        gold_active  = gold_active.at[j].set(orig_active & ~any_mine)
        set_first    = any_hit & (~has_prior) & orig_active & (~any_mine)
        gold_first_miner = gold_first_miner.at[j].set(
            jnp.where(any_mine, jnp.int32(-1),
            jnp.where(set_first, firer, gold_first_miner[j]))
        )
        gold_first_step = gold_first_step.at[j].set(
            jnp.where(any_mine, jnp.int32(0),
            jnp.where(set_first, step, gold_first_step[j]))
        )

    # ---- Stochastic ore regrow ----
    key, iron_rkey, gold_rkey = jax.random.split(key, 3)
    n_iron     = jnp.sum(iron_active.astype(jnp.int32))
    n_gold     = jnp.sum(gold_active.astype(jnp.int32))
    iron_coins = jax.random.bernoulli(iron_rkey, IRON_REGROW, shape=(NUM_IRON_SLOTS,))
    gold_coins = jax.random.bernoulli(gold_rkey, GOLD_REGROW, shape=(NUM_GOLD_SLOTS,))
    iron_active = iron_active | (iron_coins & ~iron_active & (n_iron < MAX_IRON))
    gold_active = gold_active | (gold_coins & ~gold_active & (n_gold < MAX_GOLD))

    # ---- Beam cooldown ----
    new_cooldown = jnp.where(
        fires,
        jnp.full(NUM_AGENTS, BEAM_COOL, dtype=jnp.int32),
        jnp.maximum(jnp.int32(0), state.beam_cooldown - 1),
    )

    new_state = state.replace(
        beam_cooldown    = new_cooldown,
        iron_active      = iron_active,
        gold_active      = gold_active,
        gold_first_miner = gold_first_miner,
        gold_first_step  = gold_first_step,
    )
    return new_state, rewards, key


# ================================================================
# Environment
# ================================================================

class CoopMiningJAX(MultiAgentEnv):
    """JAX-native cooperative mining environment for jaxmarl_worker.

    Pixel obs (23233 dims) match MeltingPot/Sebulba layout in format,
    with identical egocentric view geometry (forward=9, backward=1,
    left=5, right=5, centered=False).

    Compatible with IPPO (ippo_scan.py) and MAPPO (mappo_indagobs_scan.py).
    """

    def __init__(
        self,
        num_iron_slots: int = NUM_IRON_SLOTS,
        num_gold_slots: int = NUM_GOLD_SLOTS,
        max_steps:      int = MAX_STEPS,
    ) -> None:
        self._num_iron_slots = num_iron_slots
        self._num_gold_slots = num_gold_slots
        self._max_steps      = max_steps

        super().__init__(num_agents=NUM_AGENTS)
        self._obs_dim = OBS_DIM
        self.agents   = [f"agent_{i}" for i in range(NUM_AGENTS)]
        for a in self.agents:
            self.observation_spaces[a] = Discrete(OBS_DIM)
            self.action_spaces[a]      = Discrete(NUM_ACTIONS)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], CoopMiningState]:
        key, spawn_key, ore_key, dir_key, init_key = jax.random.split(key, 5)

        # Sample agent start positions from 18 spawn points (no replacement)
        spawn_perm = jax.random.permutation(spawn_key, 18)
        agent_pos  = _SPAWN_POS[spawn_perm[:NUM_AGENTS]]  # (N, 2)

        # Sample ore slot positions from 538 ore cells (no replacement)
        n_ore_total = self._num_iron_slots + self._num_gold_slots
        ore_perm    = jax.random.permutation(ore_key, len(_ORE_POS))
        iron_pos    = _ORE_POS[ore_perm[:self._num_iron_slots]]       # (I, 2)
        gold_pos    = _ORE_POS[ore_perm[self._num_iron_slots:n_ore_total]]  # (G, 2)

        agent_dir = jax.random.randint(dir_key, (NUM_AGENTS,), 0, 4, dtype=jnp.int32)

        # Initially activate exactly MAX_IRON / MAX_GOLD slots
        # Slot order is already random (from permuted ore_perm), so first MAX_* are fair
        iron_init_perm = jax.random.permutation(init_key, self._num_iron_slots)
        gold_init_perm = jax.random.permutation(init_key, self._num_gold_slots)
        iron_active = iron_init_perm < MAX_IRON   # (I,) bool: first MAX_IRON active
        gold_active = gold_init_perm < MAX_GOLD   # (G,) bool

        state = CoopMiningState(
            agent_pos        = agent_pos,
            agent_dir        = agent_dir,
            beam_cooldown    = jnp.zeros(NUM_AGENTS, dtype=jnp.int32),
            iron_pos         = iron_pos,
            iron_active      = iron_active,
            gold_pos         = gold_pos,
            gold_active      = gold_active,
            gold_first_miner = jnp.full(self._num_gold_slots, -1, dtype=jnp.int32),
            gold_first_step  = jnp.zeros(self._num_gold_slots, dtype=jnp.int32),
            step             = jnp.int32(0),
            done             = jnp.bool_(False),
        )
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key:     chex.PRNGKey,
        state:   CoopMiningState,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict, CoopMiningState, Dict, Dict, Dict]:
        actions_arr = jnp.array(
            [actions[f"agent_{i}"] for i in range(NUM_AGENTS)],
            dtype=jnp.int32,
        )

        state            = _apply_movement(state, actions_arr)
        state, rewards, _ = _apply_fire(state, actions_arr, key)

        new_step = state.step + 1
        done     = new_step >= self._max_steps
        state    = state.replace(step=new_step, done=done)

        obs    = self.get_obs(state)
        rew_d  = {f"agent_{i}": rewards[i] for i in range(NUM_AGENTS)}
        done_d = {f"agent_{i}": done       for i in range(NUM_AGENTS)}
        done_d["__all__"] = done

        return obs, state, rew_d, done_d, {}

    def get_obs(self, state: CoopMiningState) -> Dict[str, chex.Array]:
        return _get_obs(state)


# ================================================================
# Factory
# ================================================================

def make_coop_mining_jax(**kwargs) -> CoopMiningJAX:
    return CoopMiningJAX(**kwargs)
