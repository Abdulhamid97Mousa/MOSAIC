"""JAX-native American Football environment — JaxMARL MultiAgentEnv compatible.

Faithfully replicates multigrid_sports AmericanFootball game mechanics in pure JAX
so training can use jax.vmap (parallel envs) + jax.lax.scan (rollout on GPU).

Grid layout (16 × 11, 0-indexed):
  - Walls on all borders (x=0,15 and y=0,10)
  - Green goal: x=1, y∈[4,5,6]   (green scores by bringing ball to blue goal)
  - Blue  goal: x=14, y∈[4,5,6]  (blue  scores by bringing ball to green goal)
  - Ball spawns at (7,5)
  - Agents spawn at fixed positions per variant

Actions (8, matching multigrid_sports):
  0: left   1: right   2: forward
  3: pickup       4: drop         5,6,7: noop

Directions: 0=right (+x), 1=down (+y), 2=left (-x), 3=up (-y)
"""

from __future__ import annotations
from functools import partial
from typing import Dict, Tuple
import sys, os as _os

# --- JaxMARL source path (cloned repo, no pip install needed) ---
_JAXMARL_ROOT = _os.path.abspath(
    _os.path.join(_os.path.dirname(__file__), '../../../../3rd_party/JaxMARL')
)
if _JAXMARL_ROOT not in sys.path:
    sys.path.insert(0, _JAXMARL_ROOT)

import chex
import jax
import jax.numpy as jnp
from flax import struct
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.spaces import Discrete

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WIDTH, HEIGHT = 16, 11
MAX_STEPS     = 300
N_ACTIONS     = 8

DIR_DX = jnp.array([ 1,  0, -1,  0], dtype=jnp.int32)
DIR_DY = jnp.array([ 0,  1,  0, -1], dtype=jnp.int32)

GREEN_GOAL_X = jnp.int32(1)
BLUE_GOAL_X  = jnp.int32(14)
DEFAULT_GOAL_ROWS = list(range(1, HEIGHT - 1))  # full column rows 1-9 (matches reference env)

# ---------------------------------------------------------------------------
# Observation encoding constants  (raw values; normalised by /10.0 in get_obs)
# ---------------------------------------------------------------------------
_OBJ_FLOOR      = 1.0
_OBJ_WALL       = 2.0
_OBJ_GREEN_GOAL = 5.0
_OBJ_BLUE_GOAL  = 6.0
_OBJ_BALL       = 7.0
_OBJ_AGENT      = 10.0

_ALL_OBJ_VALS = (_OBJ_FLOOR, _OBJ_WALL, _OBJ_GREEN_GOAL, _OBJ_BLUE_GOAL, _OBJ_BALL, _OBJ_AGENT)
assert len(_ALL_OBJ_VALS) == len(set(_ALL_OBJ_VALS)), \
    "Observation encoding collision: all _OBJ_* values must be distinct"
assert _OBJ_GREEN_GOAL != _OBJ_FLOOR, "Invisible goals bug: green goal encoded as floor"
assert _OBJ_BLUE_GOAL  != _OBJ_FLOOR, "Invisible goals bug: blue goal encoded as floor"
assert _OBJ_BALL != _OBJ_GREEN_GOAL and _OBJ_BALL != _OBJ_BLUE_GOAL, \
    "Ball encoded same as goal cell"
assert _OBJ_AGENT != _OBJ_FLOOR and _OBJ_AGENT != _OBJ_GREEN_GOAL, \
    "Agent invisible or encoded same as goal"

# Midfield boundaries for random ball respawn (excludes walls and end zones)
MIDFIELD_X_MIN = jnp.int32(2)
MIDFIELD_X_MAX = jnp.int32(13)   # inclusive
MIDFIELD_Y_MIN = jnp.int32(1)
MIDFIELD_Y_MAX = jnp.int32(9)    # inclusive

# Passable mask — True = agent can enter
def _make_passable() -> chex.Array:
    p = jnp.ones((HEIGHT, WIDTH), dtype=jnp.bool_)
    p = p.at[0, :].set(False).at[HEIGHT-1, :].set(False)
    p = p.at[:, 0].set(False).at[:, WIDTH-1].set(False)
    return p

PASSABLE = _make_passable()

# ---------------------------------------------------------------------------
# State dataclass (all jnp arrays — vmappable, scannable)
# ---------------------------------------------------------------------------

@struct.dataclass
class AFState:
    agent_pos:               chex.Array  # (N, 2) int32  [x, y]
    agent_dir:               chex.Array  # (N,)   int32
    agent_team:              chex.Array  # (N,)   int32  0=green 1=blue
    agent_carrying:          chex.Array  # (N,)   int32  ball_idx or -1
    ball_pos:                chex.Array  # (B, 2) int32
    ball_carried_by:         chex.Array  # (B,)   int32  agent_idx or -1
    ball_last_carrier_team:  chex.Array  # (B,)   int32  -1/0/1
    scores:                  chex.Array  # (2,)   int32  [green, blue]
    step:                    int
    done:                    bool

# ---------------------------------------------------------------------------
# Team assignments per variant (positions are randomised at runtime)
# ---------------------------------------------------------------------------

_TEAMS: Dict[str, chex.Array] = {
    'G-1v0': jnp.array([0],            dtype=jnp.int32),
    'B-0v1': jnp.array([1],            dtype=jnp.int32),
    '1v1':   jnp.array([0, 1],         dtype=jnp.int32),
    '2v2':   jnp.array([0, 0, 1, 1],   dtype=jnp.int32),
    '3v3':   jnp.array([0,0,0,1,1,1],  dtype=jnp.int32),
    'G-2v0': jnp.array([0, 0],         dtype=jnp.int32),
    'G-3v0': jnp.array([0, 0, 0],      dtype=jnp.int32),
    'B-0v2': jnp.array([1, 1],         dtype=jnp.int32),
    'B-0v3': jnp.array([1, 1, 1],      dtype=jnp.int32),
}

# ---------------------------------------------------------------------------
# Action application (pure JAX)
# ---------------------------------------------------------------------------

def _forward_pos(pos, direction):
    return pos + jnp.stack([DIR_DX[direction], DIR_DY[direction]])


def _apply_action(state: AFState, agent_idx: int, action: chex.Array) -> AFState:
    ax = state.agent_pos[agent_idx, 0]
    ay = state.agent_pos[agent_idx, 1]
    adir = state.agent_dir[agent_idx]

    # --- turn left ---
    state = jax.lax.cond(
        action == 0,
        lambda s: s.replace(agent_dir=s.agent_dir.at[agent_idx].set((adir - 1) % 4)),
        lambda s: s, state,
    )

    # --- turn right ---
    state = jax.lax.cond(
        action == 1,
        lambda s: s.replace(agent_dir=s.agent_dir.at[agent_idx].set((adir + 1) % 4)),
        lambda s: s, state,
    )

    # --- forward ---
    def _do_forward(s):
        nx = ax + DIR_DX[adir]
        ny = ay + DIR_DY[adir]
        in_bounds = (nx >= 0) & (nx < WIDTH) & (ny >= 0) & (ny < HEIGHT)
        passable  = jnp.where(in_bounds, PASSABLE[ny, nx], jnp.bool_(False))
        agents    = jnp.arange(s.agent_pos.shape[0])
        occupied  = jnp.any((agents != agent_idx) & (s.agent_pos[:, 0] == nx) & (s.agent_pos[:, 1] == ny))
        can_move  = passable & ~occupied
        new_x     = jnp.where(can_move, nx, ax)
        new_y     = jnp.where(can_move, ny, ay)
        new_pos   = s.agent_pos.at[agent_idx].set(jnp.array([new_x, new_y]))
        ball_idx  = s.agent_carrying[agent_idx]
        new_bp    = jax.lax.cond(
            (ball_idx >= 0) & can_move,
            lambda bp: bp.at[ball_idx].set(jnp.array([new_x, new_y])),
            lambda bp: bp,
            s.ball_pos,
        )
        return s.replace(agent_pos=new_pos, ball_pos=new_bp)

    state = jax.lax.cond(action == 2, _do_forward, lambda s: s, state)

    # --- pickup ---
    def _do_pickup(s):
        already = s.agent_carrying[agent_idx] >= 0
        fwd     = jnp.array([ax + DIR_DX[adir], ay + DIR_DY[adir]])
        here    = jnp.array([ax, ay])

        def _ball_at(pos):
            match = (s.ball_pos[:, 0] == pos[0]) & (s.ball_pos[:, 1] == pos[1]) & (s.ball_carried_by == -1)
            return jnp.any(match), jnp.argmax(match)

        found_f, bidx_f = _ball_at(fwd)
        found_h, bidx_h = _ball_at(here)
        found   = found_f | found_h
        bidx    = jnp.where(found_f, bidx_f, bidx_h)
        team    = s.agent_team[agent_idx]
        allowed = s.ball_last_carrier_team[bidx] != team
        can     = found & ~already & allowed

        return jax.lax.cond(
            can,
            lambda ss: ss.replace(
                agent_carrying         = ss.agent_carrying.at[agent_idx].set(bidx),
                ball_carried_by        = ss.ball_carried_by.at[bidx].set(jnp.int32(agent_idx)),
                ball_last_carrier_team = ss.ball_last_carrier_team.at[bidx].set(team),
            ),
            lambda ss: ss,
            s,
        )

    state = jax.lax.cond(action == 3, _do_pickup, lambda s: s, state)

    # --- drop ---
    def _do_drop(s):
        bidx = s.agent_carrying[agent_idx]
        carrying = bidx >= 0
        return jax.lax.cond(
            carrying,
            lambda ss: ss.replace(
                agent_carrying  = ss.agent_carrying.at[agent_idx].set(jnp.int32(-1)),
                ball_carried_by = ss.ball_carried_by.at[bidx].set(jnp.int32(-1)),
            ),
            lambda ss: ss,
            s,
        )

    state = jax.lax.cond(action == 4, _do_drop, lambda s: s, state)

    return state  # actions 5,6,7 = noop

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _check_scoring(state: AFState, goal_rows):
    """Score when an agent carrying the ball is in the opponent's end zone.

    Matches Gymnasium: agent must be carrying AND standing in end zone that
    belongs to the OTHER team.  End zones cover full column height (rows 1..H-2).
    """
    scored_green = jnp.bool_(False)
    scored_blue  = jnp.bool_(False)

    for i in range(state.agent_pos.shape[0]):
        ax_i = state.agent_pos[i, 0]
        ay_i = state.agent_pos[i, 1]
        carrying = state.agent_carrying[i] >= 0
        in_rows  = jnp.any(jnp.equal(ay_i, goal_rows))

        # Green (team 0) scores at Blue goal (x=14) — opponent's zone
        at_blue  = (ax_i == BLUE_GOAL_X) & in_rows & carrying & (state.agent_team[i] == 0)
        # Blue (team 1) scores at Green goal (x=1) — opponent's zone
        at_green = (ax_i == GREEN_GOAL_X) & in_rows & carrying & (state.agent_team[i] == 1)

        scored_green = scored_green | at_blue
        scored_blue  = scored_blue  | at_green

    return scored_green, scored_blue


# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------

def _compute_rewards(state: AFState, new_state: AFState,
                     scored_green: chex.Array, scored_blue: chex.Array,
                     timed_out: chex.Array, n_agents: int,
                     goal_y: float = float(HEIGHT // 2), ball_coef: float = 0.01,
                     cooperative: bool = False) -> chex.Array:
    rewards = jnp.zeros(n_agents, dtype=jnp.float32)

    # Time-efficiency bonus: faster scoring = more reward
    time_bonus = (jnp.float32(MAX_STEPS) - new_state.step.astype(jnp.float32)) * jnp.float32(0.05)

    for i in range(n_agents):
        team = state.agent_team[i]

        # ±(1.0 + time_bonus) scoring (matches Gymnasium environment)
        scored_for  = jnp.where(team == 0, scored_green, scored_blue)
        scored_against = jnp.where(team == 0, scored_blue, scored_green)
        # Cooperative (non-zero-sum): own scoring only, no penalty for opponent scoring.
        if cooperative:
            score_rew = jnp.where(scored_for, 1.0 + time_bonus, 0.0)
        else:
            score_rew = jnp.where(scored_for, 1.0 + time_bonus,
                        jnp.where(scored_against, -(1.0 + time_bonus), 0.0))
        rewards = rewards.at[i].add(score_rew)

        # +0.3 steal: ball was carried by an opponent, now carried by this agent
        prev_not_carrying = state.agent_carrying[i] < 0
        now_carrying_idx  = new_state.agent_carrying[i]
        picked_up = prev_not_carrying & (now_carrying_idx >= 0)
        safe_bidx = jnp.maximum(now_carrying_idx, 0)
        prev_carrier     = state.ball_carried_by[safe_bidx]
        prev_carrier_team = jnp.where(prev_carrier >= 0, state.agent_team[prev_carrier], jnp.int32(-1))
        is_steal = picked_up & (prev_carrier >= 0) & (prev_carrier_team != team)
        # Steal is a take-from-opponent incentive (residually zero-sum); drop in cooperative mode.
        if not cooperative:
            rewards = rewards.at[i].add(jnp.where(is_steal, 0.3, 0.0))

        # +0.01 × Δdist toward next objective (signed 2D Manhattan — no clipping)
        # carrying     → end-zone centre (team 0 → BLUE_GOAL_X=14, team 1 → GREEN_GOAL_X=1, y=HEIGHT//2)
        # not carrying → loose ball
        carrying_now = now_carrying_idx >= 0
        goal_x = jnp.where(team == 0, BLUE_GOAL_X.astype(jnp.float32),
                                       GREEN_GOAL_X.astype(jnp.float32))
        goal_y = jnp.float32(goal_y)
        old_goal_dist = (jnp.abs(state.agent_pos[i, 0].astype(jnp.float32) - goal_x) +
                         jnp.abs(state.agent_pos[i, 1].astype(jnp.float32) - goal_y))
        new_goal_dist = (jnp.abs(new_state.agent_pos[i, 0].astype(jnp.float32) - goal_x) +
                         jnp.abs(new_state.agent_pos[i, 1].astype(jnp.float32) - goal_y))
        goal_delta = old_goal_dist - new_goal_dist
        rewards = rewards.at[i].add(jnp.where(carrying_now, 0.01 * goal_delta, 0.0))

        ball_x = new_state.ball_pos[0, 0].astype(jnp.float32)
        ball_y = new_state.ball_pos[0, 1].astype(jnp.float32)
        ball_is_loose = new_state.ball_carried_by[0] < 0
        old_ball_dist = (jnp.abs(state.agent_pos[i, 0].astype(jnp.float32) - ball_x) +
                         jnp.abs(state.agent_pos[i, 1].astype(jnp.float32) - ball_y))
        new_ball_dist = (jnp.abs(new_state.agent_pos[i, 0].astype(jnp.float32) - ball_x) +
                         jnp.abs(new_state.agent_pos[i, 1].astype(jnp.float32) - ball_y))
        ball_delta = old_ball_dist - new_ball_dist
        rewards = rewards.at[i].add(
            jnp.where(~carrying_now & ball_is_loose, ball_coef * ball_delta, 0.0)
        )

        # -1.0 timeout
        rewards = rewards.at[i].add(jnp.where(timed_out, -1.0, 0.0))

    return rewards

# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

# Static encoding (walls + goals) — (H, W) int32
def _make_static_grid() -> chex.Array:
    g = jnp.zeros((HEIGHT, WIDTH), dtype=jnp.int32)
    g = g.at[0, :].set(1).at[HEIGHT-1, :].set(1).at[:, 0].set(1).at[:, WIDTH-1].set(1)
    for y in DEFAULT_GOAL_ROWS:      # full column rows 1-9 — matches scoring and reward-shaping
        g = g.at[y, 1].set(5).at[y, 14].set(6)
    return g

STATIC_GRID = _make_static_grid()


def _obs_for_agent(state: AFState, agent_idx: int, n_agents: int, view_size: int = 7) -> chex.Array:
    """3-channel MiniGrid-compatible observation matching multigrid_sports format.

    Each cell → [object_type/10, color/5, state/100] (float32, all in [0,1]).
    Output shape: (view_size * view_size * 3,)  e.g. 147 for view_size=7.

    Exact multigrid_sports encoding (verified against live env):
      object: empty=1, wall=2, ball=6, agent=10
      color:  empty=0, wall/ball=5(grey), team-0=0(red), team-1=1(green)
      state:  0=not carrying, 100=carrying  (agent cells only; direction omitted)
    """
    ax = state.agent_pos[agent_idx, 0]
    ay = state.agent_pos[agent_idx, 1]

    r = view_size // 2

    # Pre-allocate output array: (view_size * view_size * 3,)
    n_cells = view_size * view_size
    obs = jnp.zeros((n_cells, 3), dtype=jnp.float32)

    # Iterate over view window and fill observation array (pure functional, no mutations)
    cell_idx = 0
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            gx = jnp.clip(ax + dx, 0, WIDTH  - 1)
            gy = jnp.clip(ay + dy, 0, HEIGHT - 1)

            base    = STATIC_GRID[gy, gx]
            is_wall = (base == 1)
            # wall: type=2, color=5(grey); green_goal: type=5; blue_goal: type=6; empty: type=1
            obj   = jnp.where(is_wall, _OBJ_WALL, jnp.where(base == 5, _OBJ_GREEN_GOAL, jnp.where(base == 6, _OBJ_BLUE_GOAL, _OBJ_FLOOR)))
            color = jnp.where(is_wall, 5.0, 0.0)
            st    = jnp.zeros((), dtype=jnp.float32)

            # Agent overlay: type=10, color=team_index(0=red,1=green)
            # STATE = direction (0-3) + 100 if carrying ball (matches gym env spec)
            hits      = (state.agent_pos[:, 0] == gx) & (state.agent_pos[:, 1] == gy)
            has_agent = jnp.any(hits)
            best_idx  = jnp.argmax(hits)
            a_team    = state.agent_team[best_idx]
            a_dir     = state.agent_dir[best_idx].astype(jnp.float32)
            a_carry   = (state.agent_carrying[best_idx] >= 0).astype(jnp.float32) * 100.0
            obj   = jnp.where(has_agent, _OBJ_AGENT, obj)
            color = jnp.where(has_agent, a_team.astype(jnp.float32), color)
            st    = jnp.where(has_agent, a_dir + a_carry, st)

            # Ball overlay (loose only): type=6, color=5(grey)
            ball_here = jnp.any(
                (state.ball_pos[:, 0] == gx) & (state.ball_pos[:, 1] == gy) &
                (state.ball_carried_by == -1)
            )
            obj   = jnp.where(ball_here & ~has_agent, _OBJ_BALL, obj)
            color = jnp.where(ball_here & ~has_agent, 5.0, color)

            # Store normalized channels using JAX array update operations
            obs = obs.at[cell_idx, 0].set(obj / 10.0)
            obs = obs.at[cell_idx, 1].set(color / 5.0)
            obs = obs.at[cell_idx, 2].set(st / 100.0)
            cell_idx += 1

    return obs.reshape(-1)  # Flatten to (view_size * view_size * 3,)


# ---------------------------------------------------------------------------
# JaxMARL env class
# ---------------------------------------------------------------------------

class AmericanFootballJAX(MultiAgentEnv):
    """JAX-native American Football — compatible with JaxMARL MAPPO/IPPO."""

    def __init__(self, variant: str = '2v2', max_steps: int = MAX_STEPS, goals_to_win: int = 2, view_size: int = 7, ball_coef: float = 0.01, goal_rows: list | None = None, cooperative: bool = False):
        self._cooperative  = cooperative
        self._spawn_teams  = _TEAMS[variant]
        self._n_agents     = len(self._spawn_teams)
        self._n_balls      = 1
        self._max_steps    = max_steps
        self._goals_to_win = goals_to_win
        self._view_size    = view_size
        self._ball_coef    = ball_coef
        assert goal_rows is not None, (
            "\n[AmericanFootballJAX] goal_rows must be explicitly provided — do not rely on defaults.\n"
            "  Expected for AF: goal_rows=list(range(1, 10))  (full column)\n"
            "  In training scripts: --goal-rows 1 2 3 4 5 6 7 8 9\n"
            "  Use DEFAULT_GOAL_ROWS if you want the canonical value."
        )
        rows = list(goal_rows)
        assert len(rows) > 0, f"[AmericanFootballJAX] goal_rows must be non-empty, got {rows!r}"
        assert len(rows) == len(set(rows)), \
            f"[AmericanFootballJAX] goal_rows must not contain duplicates: {rows}"
        assert all(1 <= r <= HEIGHT - 2 for r in rows), \
            f"[AmericanFootballJAX] goal_rows values must be in [1, {HEIGHT - 2}] (playable rows), got {rows}"
        self._goal_rows    = jnp.array(rows, dtype=jnp.int32)
        self._goal_y       = float(sum(rows) / len(rows))
        self._obs_dim      = view_size * view_size * 3

        super().__init__(num_agents=self._n_agents)
        self.agents = [f"agent_{i}" for i in range(self._n_agents)]
        for a in self.agents:
            self.observation_spaces[a] = Discrete(self._obs_dim)
            self.action_spaces[a]      = Discrete(N_ACTIONS)

    def _random_state(self, key: chex.PRNGKey) -> Tuple[chex.PRNGKey, chex.Array, chex.Array, chex.Array]:
        """Returns (key, agent_pos (N,2), agent_dir (N,), ball_pos (1,2)) all random in midfield."""
        key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
        pos_x   = jax.random.randint(k1, (self._n_agents,), MIDFIELD_X_MIN, MIDFIELD_X_MAX + 1, dtype=jnp.int32)
        pos_y   = jax.random.randint(k2, (self._n_agents,), MIDFIELD_Y_MIN, MIDFIELD_Y_MAX + 1, dtype=jnp.int32)
        dirs    = jax.random.randint(k3, (self._n_agents,), 0, 4, dtype=jnp.int32)
        ball_x  = jax.random.randint(k4, (1,), MIDFIELD_X_MIN, MIDFIELD_X_MAX + 1, dtype=jnp.int32)
        ball_y  = jax.random.randint(k5, (1,), MIDFIELD_Y_MIN, MIDFIELD_Y_MAX + 1, dtype=jnp.int32)
        agent_pos = jnp.stack([pos_x, pos_y], axis=1)
        ball_pos  = jnp.concatenate([ball_x, ball_y])[None, :]
        return key, agent_pos, dirs, ball_pos

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict, AFState]:
        key, agent_pos, dirs, ball_pos = self._random_state(key)
        state = AFState(
            agent_pos              = agent_pos,
            agent_dir              = dirs,
            agent_team             = self._spawn_teams,
            agent_carrying         = jnp.full(self._n_agents, -1, dtype=jnp.int32),
            ball_pos               = ball_pos,
            ball_carried_by        = jnp.full(self._n_balls,  -1, dtype=jnp.int32),
            ball_last_carrier_team = jnp.full(self._n_balls,  -1, dtype=jnp.int32),
            scores                 = jnp.zeros(2, dtype=jnp.int32),
            step                   = jnp.int32(0),
            done                   = jnp.bool_(False),
        )
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: AFState,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict, AFState, Dict, Dict, Dict]:
        action_arr = jnp.array([actions[f"agent_{i}"] for i in range(self._n_agents)], dtype=jnp.int32)

        new_state = state
        for i in range(self._n_agents):
            new_state = _apply_action(new_state, i, action_arr[i])

        scored_green, scored_blue = _check_scoring(new_state, self._goal_rows)
        any_scored = scored_green | scored_blue
        timed_out  = (state.step + 1) >= self._max_steps

        new_scores = jnp.where(scored_green, new_state.scores.at[0].add(1), new_state.scores)
        new_scores = jnp.where(scored_blue,  new_scores.at[1].add(1),       new_scores)

        # After goal: ball respawns randomly in midfield; agents keep position/direction
        key, k1, k2 = jax.random.split(key, 3)
        rand_bx      = jax.random.randint(k1, (1,), MIDFIELD_X_MIN, MIDFIELD_X_MAX + 1, dtype=jnp.int32)
        rand_by      = jax.random.randint(k2, (1,), MIDFIELD_Y_MIN, MIDFIELD_Y_MAX + 1, dtype=jnp.int32)
        rand_ball_pos = jnp.concatenate([rand_bx, rand_by])[None, :]
        new_ball_pos  = jnp.where(any_scored, rand_ball_pos, new_state.ball_pos)
        new_carrying  = jnp.where(any_scored, jnp.full(self._n_agents, -1, dtype=jnp.int32), new_state.agent_carrying)
        new_ball_cb   = jnp.where(any_scored, jnp.full(self._n_balls,  -1, dtype=jnp.int32), new_state.ball_carried_by)
        new_ball_lct  = jnp.where(any_scored, jnp.full(self._n_balls,  -1, dtype=jnp.int32), new_state.ball_last_carrier_team)

        done = timed_out | (new_scores[0] >= self._goals_to_win) | (new_scores[1] >= self._goals_to_win)

        new_state = new_state.replace(
            agent_carrying         = new_carrying,
            ball_pos               = new_ball_pos,
            ball_carried_by        = new_ball_cb,
            ball_last_carrier_team = new_ball_lct,
            scores                 = new_scores,
            step                   = state.step + 1,
            done                   = done,
        )

        rewards_arr = _compute_rewards(state, new_state, scored_green, scored_blue, timed_out, self._n_agents, goal_y=self._goal_y, ball_coef=self._ball_coef, cooperative=self._cooperative)
        obs         = self.get_obs(new_state)
        rewards     = {f"agent_{i}": rewards_arr[i] for i in range(self._n_agents)}
        dones       = {f"agent_{i}": done           for i in range(self._n_agents)}
        dones["__all__"] = done

        return obs, new_state, rewards, dones, {}

    def get_obs(self, state: AFState) -> Dict[str, chex.Array]:
        return {
            f"agent_{i}": _obs_for_agent(state, i, self._n_agents, self._view_size)
            for i in range(self._n_agents)
        }


# ---------------------------------------------------------------------------
# Factory — maps Gymnasium env_id to JAX variant
# ---------------------------------------------------------------------------

VARIANT_MAP = {
    'MosaicMultiGrid-AF-G-1v0-v1':          'G-1v0',
    'MosaicMultiGrid-AF-B-0v1-v1':          'B-0v1',
    'MosaicMultiGrid-AF-1v1-IndAgObs-v1':   '1v1',
    'MosaicMultiGrid-AF-2v2-IndAgObs-v1':   '2v2',
    'MosaicMultiGrid-AF-3v3-IndAgObs-v1':   '3v3',
    'MosaicMultiGrid-AF-G-2v0-IndAgObs-v1': 'G-2v0',
    'MosaicMultiGrid-AF-G-3v0-IndAgObs-v1': 'G-3v0',
    'MosaicMultiGrid-AF-B-0v2-IndAgObs-v1': 'B-0v2',
    'MosaicMultiGrid-AF-B-0v3-IndAgObs-v1': 'B-0v3',
}


def make_af_jax(env_id: str, **kwargs) -> AmericanFootballJAX:
    return AmericanFootballJAX(variant=VARIANT_MAP[env_id], **kwargs)
