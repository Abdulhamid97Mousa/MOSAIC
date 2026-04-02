"""Visual demonstration: Parallel vs AEC — information timing.

Run with:
    .venv/bin/python gym_gui/tests/demo_aec_vs_parallel_timing.py

The central question
--------------------
Agent 0 moves right and picks up the ball.
Does Agent 1 see "Agent 0 has the ball" when it decides its action?

  Parallel   →  NO  (Agent 1 still sees the old state)
  AEC        →  YES (Agent 1 sees Agent 0 already holding the ball)
  Wrapper    →  YES (same as AEC, via GymnasiumMultiAgentAECWrapper)

Environment
-----------
A 5-cell 1D soccer field:

    [ 0 ] [ 1 ] [ 2 ] [ 3 ] [ 4=GOAL ]

Start:
    Agent_0 at cell 0
    Ball    at cell 1
    Agent_1 at cell 2

Actions: 0 = STILL/NOOP,  1 = MOVE_RIGHT
Rule:    Moving onto the ball's cell picks it up.

The demonstration steps one round only:
    Agent_0 → MOVE_RIGHT  (lands on ball → picks it up)
    Agent_1 → MOVE_RIGHT  (or any action — we care what it OBSERVES, not what it does)

Demos
-----
  DEMO 1 — Parallel           (manual env.step([A0, A1]))
  DEMO 2 — AEC (manual)       (manual env.step([A0, STILL]) then env.step([STILL, A1]))
  DEMO 3 — AEC Wrapper   (GymnasiumMultiAgentAECWrapper.step(action))
  DEMO 4 — mosaic_multigrid   (real env with GymnasiumMultiAgentAECWrapper, if installed)
"""

import sys

import gymnasium.spaces as spaces
import numpy as np

sys.path.insert(0, ".")

from gym_gui.services.aec_wrapper import GymnasiumMultiAgentAECWrapper

# ============================================================================
# Deterministic 1D soccer environment  (mirrors mosaic_multigrid interface)
# ============================================================================

class Soccer1DEnv:
    """
    Grid layout (5 cells):
        Cell 0  Cell 1  Cell 2  Cell 3  Cell 4
        [    ]  [ ⚽ ]  [    ]  [    ]  [GOAL]

    Initial positions:
        Agent_0 → cell 0
        Ball    → cell 1
        Agent_1 → cell 2

    action_space / observation_space match mosaic_multigrid:
        action_space:      Dict({0: Discrete(2), 1: Discrete(2)})
        observation_space: Dict({0: Box(4,), 1: Box(4,)})

    step(action_list: list) mirrors mosaic_multigrid's list-based call.

    Observation for agent i = [my_cell, ball_cell, i_have_ball, other_has_ball]
        ball_cell = -1 when the ball is being carried (not free on the grid)
    """

    STILL      = 0
    MOVE_RIGHT = 1
    GRID_SIZE  = 5

    def __init__(self):
        self.action_space = spaces.Dict(
            {i: spaces.Discrete(2) for i in range(2)}  # type: ignore[arg-type]
        )
        self.observation_space = spaces.Dict(
            {i: spaces.Box(-1.0, 5.0, shape=(4,), dtype=np.float32)  # type: ignore[arg-type]
             for i in range(2)}
        )
        self.metadata    = {"render_modes": []}
        self.render_mode = None
        self.agents      = [0, 1]

        self._pos      = [0, 2]
        self._ball_pos = 1
        self._has_ball = [False, False]

    def reset(self, seed=None, options=None):
        self._pos      = [0, 2]
        self._ball_pos = 1
        self._has_ball = [False, False]
        return self._obs_dict(), {}

    def step(self, action_list):
        """Apply actions for all agents simultaneously (parallel step)."""
        for i, action in enumerate(action_list):
            if action == self.MOVE_RIGHT:
                self._pos[i] = min(self._pos[i] + 1, self.GRID_SIZE - 1)
                if (not any(self._has_ball)
                        and self._pos[i] == self._ball_pos):
                    self._has_ball[i] = True
            if self._has_ball[i]:
                self._ball_pos = self._pos[i]

        obs    = self._obs_dict()
        rews   = {0: 0.0, 1: 0.0}
        terms  = {0: False, 1: False}
        truncs = {0: False, 1: False}
        infos  = {0: {}, 1: {}}
        return obs, rews, terms, truncs, infos

    def render(self):
        return None

    def close(self):
        pass

    def _obs_dict(self):
        ball_cell = self._ball_pos if not any(self._has_ball) else -1
        return {
            i: np.array(
                [self._pos[i], ball_cell,
                 float(self._has_ball[i]),
                 float(self._has_ball[1 - i])],
                dtype=np.float32,
            )
            for i in range(2)
        }

    def _render_grid(self) -> str:
        cells = ["   " for _ in range(self.GRID_SIZE)]
        cells[4] = "GOL"
        if not any(self._has_ball):
            cells[self._ball_pos] = " ⚽"
        for i in range(2):
            tag = f"A{i}⚽" if self._has_ball[i] else f" A{i}"
            cells[self._pos[i]] = tag
        row = "  ".join(f"[{c}]" for c in cells)
        idx = "   ".join(f" {i} " for i in range(self.GRID_SIZE))
        return f"  {row}\n  {idx}"


# ============================================================================
# Pretty-print helpers
# ============================================================================

SEP  = "=" * 70
SEP2 = "-" * 70

def fmt_obs(obs_array: np.ndarray, label: str = "") -> str:
    my_cell, ball_cell, i_have_ball, other_has_ball = obs_array
    ball_str = f"cell {int(ball_cell)}" if ball_cell >= 0 else "being carried"
    i_str    = "YES — I have it"          if i_have_ball   else "no"
    oth_str  = "YES — other agent has it" if other_has_ball else "no"
    return (
        f"  {label}\n"
        f"    my position     : cell {int(my_cell)}\n"
        f"    ball position   : {ball_str}\n"
        f"    I have ball     : {i_str}\n"
        f"    other has ball  : {oth_str}"
    )


def print_grid(env: Soccer1DEnv, header: str = ""):
    if header:
        print(f"\n  {header}")
    print(env._render_grid())


# ============================================================================
# DEMO 1: Parallel
# ============================================================================

def demo_parallel():
    print()
    print(SEP)
    print("  DEMO 1: Parallel")
    print(SEP)
    print("""
  env.step([action_0, action_1])
  Both actions are collected FIRST, then physics runs ONCE.
  Agent_1 decides based on the state BEFORE Agent_0 acted.
""")

    env = Soccer1DEnv()
    env.reset(seed=0)

    print_grid(env, "INITIAL STATE:")

    # ---- Agent_0's observation and action ----
    print()
    print(SEP2)
    obs = env._obs_dict()
    print(fmt_obs(obs[0], "Agent_0 observes:"))
    print("""
  Agent_0 decides: MOVE_RIGHT  (will land on ball → picks it up)
""")

    # ---- Agent_1's observation (before physics fires) ----
    print(SEP2)
    print(fmt_obs(obs[1], "Agent_1 observes (physics has NOT fired yet):"))
    print()
    if obs[1][3] == 0.0 and obs[1][1] >= 0:
        print("  [!!] Agent_1 still sees the ball as FREE at its original cell.")
        print("       other_has_ball = 0  <-- STALE — Agent_0's move is unknown")

    print("""
  Agent_1 decides: MOVE_RIGHT

  env.step([MOVE_RIGHT, MOVE_RIGHT])  ← physics runs NOW with BOTH actions
""")

    env.step([Soccer1DEnv.MOVE_RIGHT, Soccer1DEnv.MOVE_RIGHT])

    print(SEP2)
    print("  STATE AFTER env.step([MOVE_RIGHT, MOVE_RIGHT]):")
    print_grid(env)
    obs_after = env._obs_dict()
    print()
    print(fmt_obs(obs_after[0], "Agent_0 observes (for next round):"))
    print()
    print(fmt_obs(obs_after[1], "Agent_1 observes (for next round):"))
    print()
    print("  VERDICT: Agent_1 decided WITHOUT seeing Agent_0's pickup.")
    print("           Information timing: SIMULTANEOUS")
    print()


# ============================================================================
# DEMO 2: AEC
# ============================================================================

def demo_aec():
    print()
    print(SEP)
    print("  DEMO 2: AEC")
    print(SEP)
    print("""
  Agent_0's turn:  env.step([MOVE_RIGHT, STILL])   ← physics fires immediately
  Agent_1's turn:  env.step([STILL, MOVE_RIGHT])   ← physics fires immediately
  Agent_1 observes the result of Agent_0's step BEFORE deciding.
""")

    env = Soccer1DEnv()
    env.reset(seed=0)

    print_grid(env, "INITIAL STATE:")

    # ---- Agent_0's turn ----
    print()
    print(SEP2)
    obs = env._obs_dict()
    print(fmt_obs(obs[0], "Agent_0 observes:"))
    print("""
  Agent_0 decides: MOVE_RIGHT

  env.step([MOVE_RIGHT, STILL])  ← physics fires NOW for Agent_0 only
""")
    env.step([Soccer1DEnv.MOVE_RIGHT, Soccer1DEnv.STILL])
    print_grid(env, "STATE after env.step([MOVE_RIGHT, STILL]):")

    # ---- Agent_1's turn ----
    print()
    print(SEP2)
    obs_1 = env._obs_dict()[1]
    print(fmt_obs(obs_1, "Agent_1 observes (Agent_0 has ALREADY acted):"))
    print()
    if obs_1[3] == 1.0:
        print("  [OK] other_has_ball = 1")
        print("       Agent_1 KNOWS Agent_0 already has the ball.")
        print("       Agent_1 can move toward goal instead of chasing the ball.")

    print("""
  Agent_1 decides: MOVE_RIGHT  (toward goal)

  env.step([STILL, MOVE_RIGHT])  ← physics fires NOW for Agent_1 only
""")
    env.step([Soccer1DEnv.STILL, Soccer1DEnv.MOVE_RIGHT])

    print(SEP2)
    print("  FINAL STATE after env.step([STILL, MOVE_RIGHT]):")
    print_grid(env)
    print()
    print("  VERDICT: Agent_1 decided WITH full knowledge of Agent_0's action.")
    print("           Information timing: SEQUENTIAL")
    print()


# ============================================================================
# DEMO 3: AEC via GymnasiumMultiAgentAECWrapper (Option B)
# ============================================================================

def demo_aec_wrapper():
    print()
    print(SEP)
    print("  DEMO 3: AEC via GymnasiumMultiAgentAECWrapper (Option B)")
    print(SEP)
    print("""
  Same information-timing advantage as DEMO 2, but via the clean wrapper
  interface that is used with real mosaic_multigrid environments.

  aec_env.step(action)  — one agent acts; env.step([action, NOOP]) fires NOW
  The NEXT agent's observe() returns the updated intermediate state.
""")

    raw_env = Soccer1DEnv()
    aec_env = GymnasiumMultiAgentAECWrapper(raw_env)
    aec_env.reset(seed=0)

    print_grid(raw_env, "INITIAL STATE:")

    # ---- Agent_0's turn ----
    print()
    print(SEP2)
    agent = aec_env.agent_selection            # "agent_0"
    obs_0 = aec_env.observe(agent)
    print(fmt_obs(obs_0, f"aec_env.observe('{agent}')  [before acting]:"))
    print(f"""
  {agent} decides: MOVE_RIGHT

  aec_env.step(MOVE_RIGHT)
    → internally calls env.step([MOVE_RIGHT, NOOP])
    → physics fires NOW for {agent} only
""")
    aec_env.step(Soccer1DEnv.MOVE_RIGHT)      # NOOP inserted for agent_1 automatically
    print_grid(raw_env, "STATE after aec_env.step(MOVE_RIGHT) for agent_0:")

    # ---- Agent_1's turn ----
    print()
    print(SEP2)
    agent = aec_env.agent_selection            # "agent_1"
    obs_1 = aec_env.observe(agent)
    print(fmt_obs(obs_1, f"aec_env.observe('{agent}')  [sees Agent_0's result]:"))
    print()
    if obs_1[3] == 1.0:
        print("  [OK] other_has_ball = 1")
        print("       Agent_1 KNOWS Agent_0 already has the ball.")
        print("       GymnasiumMultiAgentAECWrapper correctly exposes S(t+0.5).")

    print(f"""
  {agent} decides: MOVE_RIGHT

  aec_env.step(MOVE_RIGHT)
    → internally calls env.step([NOOP, MOVE_RIGHT])
    → physics fires NOW for {agent} only
""")
    aec_env.step(Soccer1DEnv.MOVE_RIGHT)

    print(SEP2)
    print("  FINAL STATE after aec_env.step(MOVE_RIGHT) for agent_1:")
    print_grid(raw_env)
    print()
    print("  VERDICT: Agent_1 decided WITH full knowledge of Agent_0's action.")
    print("           Information timing: SEQUENTIAL (via wrapper)")
    print()


# ============================================================================
# DEMO 4: Real mosaic_multigrid environment (optional — requires package)
# ============================================================================

def demo_mosaic_multigrid():
    print()
    print(SEP)
    print("  DEMO 4: Real mosaic_multigrid env with GymnasiumMultiAgentAECWrapper")
    print(SEP)

    try:
        import gymnasium
        import mosaic_multigrid.envs  # noqa: F401 — registers environments
    except ImportError as exc:
        print(f"\n  [SKIP] mosaic_multigrid not installed: {exc}")
        print("         Install with:  pip install mosaic-multigrid==5.0.0")
        print()
        return

    ENV_ID = "MosaicMultiGrid-Soccer-1vs1-IndAgObs-v0"
    print(f"\n  Environment : {ENV_ID}")
    print( "  Package     : mosaic-multigrid 5.0.0")
    print( "  NOOP action : 0  (v5.0.0+ — safe non-acting placeholder)")
    print()

    raw_env = gymnasium.make(ENV_ID, render_mode=None)
    aec_env = GymnasiumMultiAgentAECWrapper(raw_env)

    obs_dict, info_dict = aec_env.reset(seed=42)

    print(f"  Possible agents : {aec_env.possible_agents}")
    print(f"  Obs shapes      : { {a: obs_dict[a].shape for a in obs_dict} }")
    print()

    # One full round: each agent takes one random action
    round_log = []
    while not aec_env.done:
        agent = aec_env.agent_selection
        obs_before = aec_env.observe(agent)
        action = aec_env.action_space(agent).sample()
        aec_env.step(action)
        obs_after = aec_env.observe(agent)   # NOTE: this is the post-step obs seen
                                              # by the NEXT agent, not this one

        round_log.append((agent, action, obs_before, obs_after))

        # Stop after one full round (both agents acted) for the demo
        if aec_env.agent_selection == aec_env.possible_agents[0]:
            break

    for agent, action, obs_before, obs_after in round_log:
        print(f"  {agent}  action={action}")
        print(f"    obs before step : {np.round(obs_before[:6], 2)}")
        print(f"    obs after  step : {np.round(obs_after[:6], 2)}")
        print()

    print("  [OK] GymnasiumMultiAgentAECWrapper works with real mosaic_multigrid env.")
    print("       Each aec_env.step(action) fires exactly one env.step().")
    print("       Subsequent agents observe the intermediate state S(t+0.5).")
    print()
    aec_env.close()


# ============================================================================
# DEMO 5: Side-by-side summary
# ============================================================================

def demo_summary():
    print()
    print(SEP)
    print("  SUMMARY: What each agent observes when deciding")
    print(SEP)
    print("""
  SETUP (both demos):
    Initial grid:  [A0] [⚽] [A1] [  ] [GOAL]
    Agent_0 acts:  MOVE_RIGHT  →  moves to cell 1  →  picks up ball
    Agent_1 then decides.

  ┌──────────────────────┬───────────────────────────────────────────────────┐
  │         Mode         │  What Agent_1 sees when it decides               │
  ├──────────────────────┼───────────────────────────────────────────────────┤
  │  Parallel (DEMO 1)   │  ball=cell 1   other_has_ball=NO  (stale S(t))   │
  │                      │  env.step([a0, a1]) fires ONCE after both decide  │
  ├──────────────────────┼───────────────────────────────────────────────────┤
  │  AEC manual (DEMO 2) │  ball=-1       other_has_ball=YES (fresh S(t+.5)) │
  │                      │  env.step([a0, NOOP]) fires, THEN Agent_1 sees    │
  ├──────────────────────┼───────────────────────────────────────────────────┤
  │  AEC wrapper (DEMO 3)│  ball=-1       other_has_ball=YES (fresh S(t+.5)) │
  │                      │  aec_env.step(a0) → env.step([a0, NOOP]) inside   │
  │                      │  Same information timing as manual AEC             │
  └──────────────────────┴───────────────────────────────────────────────────┘

  CODE DIFFERENCE:

    Parallel:
        env.step([action_0, action_1])          # one call, simultaneous

    AEC (manual):
        env.step([action_0, NOOP])              # Agent_0's turn
        env.step([NOOP,     action_1])          # Agent_1's turn (sees result above)

    AEC (GymnasiumMultiAgentAECWrapper):
        aec_env.step(action_0)                  # Agent_0's turn  → env.step([a0, NOOP])
        aec_env.step(action_1)                  # Agent_1's turn  → env.step([NOOP, a1])
""")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    demo_parallel()
    demo_aec()
    demo_aec_wrapper()
    demo_mosaic_multigrid()
    demo_summary()
