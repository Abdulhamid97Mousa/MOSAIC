"""MOSAIC MultiGrid game documentation module.

MOSAIC MultiGrid is our custom multi-agent grid-world package built on
Gymnasium.  It provides competitive team-based environments (Soccer, Collect,
Basketball) with two execution modes, partial observability, and multi-keyboard
support.

Package location: 3rd_party/mosaic_multigrid/
PyPI: mosaic-multigrid (v6.0.0)
API: Gymnasium (env.reset(seed=N), 5-tuple step returns)

Action space (v6.0.0):
    Discrete(8): 0=NOOP, 1=LEFT, 2=RIGHT, 3=FORWARD, 4=PICKUP, 5=DROP, 6=TOGGLE, 7=DONE
    NOOP (0) is a genuine no-op — the agent stays in place and does nothing.
    This enables AEC (sequential physics) mode via GymnasiumMultiAgentAECWrapper.

Execution modes:
    Parallel (default):
        Both agents observe S(t) and act simultaneously.
        env.step([A_0, A_1]) fires once — one physics update per round.

    AEC (sequential physics):
        Agent_0 acts; physics fires immediately → S(t+0.5).
        Agent_1 observes the intermediate state before deciding.
        env.step([A_0, NOOP]) → S(t+0.5) → env.step([NOOP, A_1]) → S(t+1)
        Requires GymnasiumMultiAgentAECWrapper (gym_gui/services/aec_wrapper.py).
        Only available because NOOP=0 is a genuine no-op (v5.0.0+).

Environments:
    Deprecated (v1.0.2):
        - MosaicMultiGrid-Soccer-v0: 2v2 soccer, 15x10 grid
        - MosaicMultiGrid-Collect-v0: 3-agent individual competition
        - MosaicMultiGrid-Collect-2vs2-v0: 2v2 team collection
        - MosaicMultiGrid-Collect-1vs1-v0: 1v1 collection

    IndAgObs (v4.0.0+) -- RECOMMENDED:
        - MosaicMultiGrid-Soccer-2vs2-IndAgObs-v0: 2v2 soccer, 16x11 FIFA grid
        - MosaicMultiGrid-Soccer-1vs1-IndAgObs-v0: 1v1 soccer, same grid
        - MosaicMultiGrid-Collect-IndAgObs-v0: 3-agent with natural termination
        - MosaicMultiGrid-Collect-2vs2-IndAgObs-v0: 2v2 with natural termination
        - MosaicMultiGrid-Collect-1vs1-IndAgObs-v0: 1v1 with natural termination
        - MosaicMultiGrid-Basketball-3vs3-IndAgObs-v0: 3v3 basketball, court rendering

    TeamObs (v4.0.0+) -- SMAC-style teammate awareness:
        - MosaicMultiGrid-Soccer-2vs2-TeamObs-v0: IndAgObs + teammate features
        - MosaicMultiGrid-Collect-2vs2-TeamObs-v0: IndAgObs + teammate features
        - MosaicMultiGrid-Basketball-3vs3-TeamObs-v0: IndAgObs + teammate features

    Solo (v6.0.0) -- Single-agent, no opponent, for curriculum pre-training:
        - MosaicMultiGrid-Soccer-Solo-Green-IndAgObs-v0: 1 Green agent, scores right
        - MosaicMultiGrid-Soccer-Solo-Blue-IndAgObs-v0: 1 Blue agent, scores left
        - MosaicMultiGrid-Basketball-Solo-Green-IndAgObs-v0: 1 Green agent, scores right
        - MosaicMultiGrid-Basketball-Solo-Blue-IndAgObs-v0: 1 Blue agent, scores left
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Soccer
# ---------------------------------------------------------------------------

def _get_soccer_base_html() -> str:
    """Soccer v0 (deprecated) documentation."""
    return """
<h2>MosaicMultiGrid-Soccer-v0</h2>

<p style="background-color: #fff3e0; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>Status:</strong> Deprecated -- use <code>MosaicMultiGrid-Soccer-2vs2-IndAgObs-v0</code> instead.
</p>

<p>
A 4-player (2v2) soccer game where two teams compete to score goals.
Agents pick up the ball, navigate to the opponent's goal, and DROP to score.
</p>

<h4>Environment Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Grid Size</td><td style="border: 1px solid #ddd; padding: 8px;">15 x 10 (13 x 8 playable)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Agents</td><td style="border: 1px solid #ddd; padding: 8px;">4 (2 per team)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Team 1</td><td style="border: 1px solid #ddd; padding: 8px;">Agents 0, 1</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Team 2</td><td style="border: 1px solid #ddd; padding: 8px;">Agents 2, 3</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Ball</td><td style="border: 1px solid #ddd; padding: 8px;">1 wildcard ball</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Goals</td><td style="border: 1px solid #ddd; padding: 8px;">(1,5) and (13,5)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">View Size</td><td style="border: 1px solid #ddd; padding: 8px;">3 x 3 (9% of grid)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Max Steps</td><td style="border: 1px solid #ddd; padding: 8px;">10,000</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Termination</td><td style="border: 1px solid #ddd; padding: 8px;">None (runs until max_steps)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Zero-Sum</td><td style="border: 1px solid #ddd; padding: 8px;">Yes</td></tr>
</table>

<h4>Known Issues (Fixed in IndAgObs)</h4>
<ul>
    <li>Ball disappears after goal -- no respawn</li>
    <li>No natural termination -- episode always runs 10,000 steps</li>
    <li>Adjacent-only passing -- DROP passes to the 1-cell in front only</li>
    <li>No steal cooldown -- agents can ping-pong steal indefinitely</li>
</ul>
"""


def _get_soccer_enhanced_html() -> str:
    """Soccer IndAgObs 2vs2 (recommended) documentation."""
    return """
<h2>MosaicMultiGrid-Soccer-2vs2-IndAgObs-v0</h2>

<p style="background-color: #e8f5e9; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>RECOMMENDED</strong> for RL training and human play.
Gymnasium API: <code>env.reset(seed=N)</code> returns <code>(obs, info)</code>.
Supports <strong>Parallel</strong> (default) and <strong>AEC</strong> execution modes.
</p>

<p>
Enhanced 2v2 soccer with <strong>teleport passing</strong>, ball respawn,
first-to-2-goals termination, and dual steal cooldown.
FIFA-ratio grid (16x11) with meaningful partial observability (3x3 view).
</p>

<h4>Environment Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Grid Size</td><td style="border: 1px solid #ddd; padding: 8px;">16 x 11 (14 x 9 playable, FIFA 1.54 ratio)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Agents</td><td style="border: 1px solid #ddd; padding: 8px;">4 (2 per team)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Team 1</td><td style="border: 1px solid #ddd; padding: 8px;">Agents 0, 1</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Team 2</td><td style="border: 1px solid #ddd; padding: 8px;">Agents 2, 3</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Ball</td><td style="border: 1px solid #ddd; padding: 8px;">1 wildcard ball (respawns after goal)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Goals</td><td style="border: 1px solid #ddd; padding: 8px;">(1,5) and (14,5) -- vertical center</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">View Size</td><td style="border: 1px solid #ddd; padding: 8px;">3 x 3 (9 cells, meaningful fog-of-war)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Max Steps</td><td style="border: 1px solid #ddd; padding: 8px;">200</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Win Condition</td><td style="border: 1px solid #ddd; padding: 8px;">First to 2 goals</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Steal Cooldown</td><td style="border: 1px solid #ddd; padding: 8px;">10 steps (both stealer and victim)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Zero-Sum</td><td style="border: 1px solid #ddd; padding: 8px;">Yes (+1 scoring team, -1 opponents)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Execution Mode</td><td style="border: 1px solid #ddd; padding: 8px;">Parallel (default) or AEC</td></tr>
</table>

<h4>Observation Space</h4>
<p>
Each agent receives a partial observation:
<code>Box(low=0, high=255, shape=(3, 3, 3), dtype=uint8)</code>
</p>
<p>The 3 channels encode: object type, color, state.</p>

<h4>Action Space</h4>
<p><code>Discrete(8)</code> -- same for all agents:</p>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">ID</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Action</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Soccer Use</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">0</td><td style="border: 1px solid #ddd; padding: 8px;">NOOP</td><td style="border: 1px solid #ddd; padding: 8px;">Idle — AEC no-op (inspired by MeltingPot)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">1</td><td style="border: 1px solid #ddd; padding: 8px;">LEFT</td><td style="border: 1px solid #ddd; padding: 8px;">Rotate counter-clockwise</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">2</td><td style="border: 1px solid #ddd; padding: 8px;">RIGHT</td><td style="border: 1px solid #ddd; padding: 8px;">Rotate clockwise</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">3</td><td style="border: 1px solid #ddd; padding: 8px;">FORWARD</td><td style="border: 1px solid #ddd; padding: 8px;">Move in facing direction</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">4</td><td style="border: 1px solid #ddd; padding: 8px;">PICKUP</td><td style="border: 1px solid #ddd; padding: 8px;">Grab ball / steal from opponent</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">5</td><td style="border: 1px solid #ddd; padding: 8px;">DROP</td><td style="border: 1px solid #ddd; padding: 8px;">Score at goal / teleport pass / drop</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">6</td><td style="border: 1px solid #ddd; padding: 8px;">TOGGLE</td><td style="border: 1px solid #ddd; padding: 8px;">Toggle object</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">7</td><td style="border: 1px solid #ddd; padding: 8px;">DONE</td><td style="border: 1px solid #ddd; padding: 8px;">Signal completion</td></tr>
</table>

<h4>Game Mechanics</h4>

<h5>Ball Pickup &amp; Stealing</h5>
<ul>
    <li><strong>Pickup</strong>: Face ball on ground and use PICKUP</li>
    <li><strong>Steal</strong>: Face an opponent carrying the ball and PICKUP to steal it</li>
    <li><strong>Cooldown</strong>: After stealing, both stealer and victim cannot steal for 10 steps
        (prevents ping-pong stealing exploits)</li>
</ul>

<h5>DROP Action -- Priority Chain</h5>
<p>When carrying the ball, DROP follows this priority:</p>
<ol>
    <li><strong>Score</strong>: If facing the opponent's goal, the ball scores (+1 team reward).
        Ball respawns at a random location.</li>
    <li><strong>Teleport Pass</strong>: Ball <em>instantly teleports</em> to a random eligible teammate
        anywhere on the grid (teammate must not already be carrying). This is a "blind pass"
        under partial observability -- the passer cannot see where their teammate is.</li>
    <li><strong>Ground Drop</strong>: If no teammate is available, the ball drops onto the empty cell
        in front (fallback).</li>
</ol>

<p style="background-color: #e3f2fd; padding: 8px; border-radius: 4px; margin-top: 10px;">
<strong>Why teleport passing?</strong> With 3x3 view on a 16x11 grid, agents can only see 9 of 126 playable cells.
Adjacent passing (old behavior) was nearly useless -- teammates had to be in the same 3x3 window.
Teleport passing creates meaningful attack/defense dynamics: pass to redistribute the ball
while opponents must mark both players. Combined with stealing, this creates a non-transitive
strategy space (pass beats solo-carry, marking beats pass, solo-carry beats marking).
</p>

<h5>Scoring &amp; Termination</h5>
<ul>
    <li>DROP ball at opponent's goal: +1 to scoring team, -1 to opponents (zero-sum)</li>
    <li>Ball respawns at random position after each goal</li>
    <li>First team to score <strong>2 goals wins</strong> -- all agents terminated</li>
    <li>If no team reaches 2 goals by step 200, episode truncates</li>
</ul>

<h4>Rewards</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Event</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Scoring Team</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Opponent Team</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;">Goal scored</td>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>+1.0</strong> (both teammates)</td>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>-1.0</strong> (both opponents)</td>
    </tr>
</table>

<h4>Keyboard Controls</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #fff3e0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Key</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Action</th>
        <th style="border: 1px solid #ddd; padding: 8px;">ID</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Soccer Use</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><em>(no key)</em></td><td style="border: 1px solid #ddd; padding: 8px;">NOOP</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>0</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Idle — AEC no-op</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">A or Left</td><td style="border: 1px solid #ddd; padding: 8px;">LEFT</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>1</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Rotate counter-clockwise</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">D or Right</td><td style="border: 1px solid #ddd; padding: 8px;">RIGHT</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>2</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Rotate clockwise</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">W or Up</td><td style="border: 1px solid #ddd; padding: 8px;">FORWARD</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>3</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Move in facing direction</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Space or G</td><td style="border: 1px solid #ddd; padding: 8px;">PICKUP</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>4</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Grab ball / steal from opponent</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">H</td><td style="border: 1px solid #ddd; padding: 8px;">DROP</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>5</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Score / teleport pass / drop</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">E or Enter</td><td style="border: 1px solid #ddd; padding: 8px;">TOGGLE</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>6</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Toggle object</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Q</td><td style="border: 1px solid #ddd; padding: 8px;">DONE</td><td style="border: 1px solid #ddd; padding: 8px;"><strong>7</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Signal completion</td></tr>
</table>
<p style="background-color: #ffecb3; padding: 8px; border-radius: 4px; margin-top: 10px;">
<strong>Note:</strong> MOSAIC MultiGrid uses <code>NOOP (0)</code> as the idle action when no key is pressed.
This is different from INI MultiGrid which uses <code>DONE (6)</code>.
NOOP=0 is a genuine no-op (v5.0.0+, current v6.0.0) and enables AEC mode via <code>GymnasiumMultiAgentAECWrapper</code>.
</p>

<h4>Improvements over Deprecated Soccer-v0</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Feature</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Soccer-v0 (Deprecated)</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Soccer-2vs2-IndAgObs-v0</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Grid</td><td style="border: 1px solid #ddd; padding: 8px;">15x10</td><td style="border: 1px solid #ddd; padding: 8px;">16x11 (FIFA ratio)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Passing</td><td style="border: 1px solid #ddd; padding: 8px;">1-cell adjacency (useless)</td><td style="border: 1px solid #ddd; padding: 8px;">Teleport to teammate</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Ball after goal</td><td style="border: 1px solid #ddd; padding: 8px;">Disappears (bug)</td><td style="border: 1px solid #ddd; padding: 8px;">Respawns at random</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Termination</td><td style="border: 1px solid #ddd; padding: 8px;">Never (10k steps)</td><td style="border: 1px solid #ddd; padding: 8px;">First to 2 goals</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Steal cooldown</td><td style="border: 1px solid #ddd; padding: 8px;">None</td><td style="border: 1px solid #ddd; padding: 8px;">10 steps (dual)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Avg episode</td><td style="border: 1px solid #ddd; padding: 8px;">10,000 steps</td><td style="border: 1px solid #ddd; padding: 8px;">~200 steps</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Training speed</td><td style="border: 1px solid #ddd; padding: 8px;">Baseline</td><td style="border: 1px solid #ddd; padding: 8px;">~50x faster</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Execution modes</td><td style="border: 1px solid #ddd; padding: 8px;">Parallel only</td><td style="border: 1px solid #ddd; padding: 8px;">Parallel + AEC</td></tr>
</table>

<h4>References</h4>
<ul>
    <li>Source: <code>3rd_party/mosaic_multigrid/</code></li>
    <li>Class: <code>SoccerGame4HIndAgObsEnv16x11N2</code></li>
    <li>See: <code>SOCCER_IMPROVEMENTS.md</code> for full technical details</li>
</ul>
"""


def _get_soccer_1vs1_html() -> str:
    """Soccer IndAgObs 1vs1 documentation."""
    return """
<h2>MosaicMultiGrid-Soccer-1vs1-IndAgObs-v0</h2>

<p style="background-color: #e8f5e9; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>RECOMMENDED</strong> for RL training and hybrid RL+LLM experiments.
Gymnasium API: <code>env.reset(seed=N)</code> returns <code>(obs, info)</code>.
Supports <strong>Parallel</strong> (default) and <strong>AEC</strong> execution modes.
</p>

<p>
1v1 soccer on the same 16x11 FIFA-ratio grid as the 2vs2 variant.
Reduced to 2 agents — ideal for IPPO training and ablation between Parallel vs AEC modes.
Each agent is the only member of their team; teleport pass has no effect (no teammate).
</p>

<h4>Environment Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Grid Size</td><td style="border: 1px solid #ddd; padding: 8px;">16 x 11 (14 x 9 playable, FIFA 1.54 ratio)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Agents</td><td style="border: 1px solid #ddd; padding: 8px;">2 (1 per team)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Team 1</td><td style="border: 1px solid #ddd; padding: 8px;">Agent 0</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Team 2</td><td style="border: 1px solid #ddd; padding: 8px;">Agent 1</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Ball</td><td style="border: 1px solid #ddd; padding: 8px;">1 wildcard ball (respawns after goal)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">View Size</td><td style="border: 1px solid #ddd; padding: 8px;">3 x 3</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Max Steps</td><td style="border: 1px solid #ddd; padding: 8px;">200</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Win Condition</td><td style="border: 1px solid #ddd; padding: 8px;">First to 2 goals</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Zero-Sum</td><td style="border: 1px solid #ddd; padding: 8px;">Yes (+1 scorer, -1 opponent)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Execution Mode</td><td style="border: 1px solid #ddd; padding: 8px;">Parallel (default) or AEC</td></tr>
</table>

<h4>Why 1vs1?</h4>
<ul>
    <li><strong>IPPO training baseline</strong>: Simplest competitive setting -- one policy per agent, no team coordination required</li>
    <li><strong>AEC ablation</strong>: Only 2 agents means 1 intermediate state S(t+0.5) per round -- clearest setting to study AEC effect</li>
    <li><strong>Hybrid RL+LLM</strong>: Freeze <code>pi_agent_0</code> (RL), pair with LLM-controlled <code>agent_1</code> -- no dimension mismatch from team size</li>
    <li><strong>Curriculum start</strong>: Train here first before scaling to 2vs2</li>
</ul>

<h4>Action Space</h4>
<p><code>Discrete(8)</code> -- same as 2vs2 Soccer (NOOP=0 enables AEC mode).</p>

<h4>References</h4>
<ul>
    <li>Source: <code>3rd_party/mosaic_multigrid/</code></li>
    <li>AEC wrapper: <code>gym_gui/services/aec_wrapper.py</code></li>
    <li>See: <code>SOCCER_IMPROVEMENTS.md</code> for shared mechanics</li>
</ul>
"""


# ---------------------------------------------------------------------------
# Collect
# ---------------------------------------------------------------------------

def _get_collect_base_html() -> str:
    """Collect v0 (deprecated 3-agent) documentation."""
    return """
<h2>MosaicMultiGrid-Collect-v0</h2>

<p style="background-color: #fff3e0; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>Status:</strong> Deprecated -- use <code>MosaicMultiGrid-Collect-IndAgObs-v0</code> instead.
</p>

<p>
A 3-player ball collection game where each agent is their own team.
Agents compete to pick up wildcard balls. Ball is consumed on pickup (not carried).
</p>

<h4>Environment Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Grid Size</td><td style="border: 1px solid #ddd; padding: 8px;">10 x 10</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Agents</td><td style="border: 1px solid #ddd; padding: 8px;">3 (each on their own team)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Balls</td><td style="border: 1px solid #ddd; padding: 8px;">5 wildcard (index=0)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">View Size</td><td style="border: 1px solid #ddd; padding: 8px;">3 x 3</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Max Steps</td><td style="border: 1px solid #ddd; padding: 8px;">10,000</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Termination</td><td style="border: 1px solid #ddd; padding: 8px;">None (bug: runs until max_steps)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Zero-Sum</td><td style="border: 1px solid #ddd; padding: 8px;">Yes</td></tr>
</table>

<h4>Known Issue</h4>
<p>
Episode does <strong>not terminate</strong> when all balls are collected.
Agents wander aimlessly for ~9,500 remaining steps with zero reward.
Fixed in IndAgObs variant.
</p>
"""


def _get_collect_enhanced_html() -> str:
    """Collect IndAgObs (recommended) documentation."""
    return """
<h2>MosaicMultiGrid-Collect-IndAgObs-v0</h2>

<p style="background-color: #e8f5e9; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>RECOMMENDED</strong> for RL training.
Gymnasium API: <code>env.reset(seed=N)</code> returns <code>(obs, info)</code>.
Supports <strong>Parallel</strong> (default) and <strong>AEC</strong> execution modes.
</p>

<p>
Enhanced 3-agent individual competition with <strong>natural termination</strong>
when all balls are collected. 35x faster training than the deprecated variant.
</p>

<h4>Environment Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Grid Size</td><td style="border: 1px solid #ddd; padding: 8px;">10 x 10 (8 x 8 playable)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Agents</td><td style="border: 1px solid #ddd; padding: 8px;">3 (each on their own team)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Balls</td><td style="border: 1px solid #ddd; padding: 8px;">5 wildcard (index=0, any agent can collect)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">View Size</td><td style="border: 1px solid #ddd; padding: 8px;">3 x 3 (9% of grid)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Max Steps</td><td style="border: 1px solid #ddd; padding: 8px;">300</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Termination</td><td style="border: 1px solid #ddd; padding: 8px;">When all 5 balls collected</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Zero-Sum</td><td style="border: 1px solid #ddd; padding: 8px;">Yes (+1 collector, -1 others)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Execution Mode</td><td style="border: 1px solid #ddd; padding: 8px;">Parallel (default) or AEC</td></tr>
</table>

<h4>Game Mechanics</h4>
<ul>
    <li><strong>Pickup</strong>: Face a ball and use PICKUP -- ball is consumed (not carried)</li>
    <li><strong>Team matching</strong>: Wildcard balls (index=0) can be collected by anyone;
        team-indexed balls can only be collected by matching team</li>
    <li><strong>Zero-sum reward</strong>: Collector gets +1, all other agents get -1</li>
    <li><strong>Termination</strong>: Episode ends when all balls are picked up</li>
</ul>

<h4>Rewards</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Event</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Collecting Agent</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Other Agents</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;">Ball pickup</td>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>+1.0</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>-1.0</strong> each</td>
    </tr>
</table>

<h4>References</h4>
<ul>
    <li>Source: <code>3rd_party/mosaic_multigrid/</code></li>
    <li>Class: <code>CollectGame3HIndAgObsEnv10x10N3</code></li>
    <li>See: <code>COLLECT_IMPROVEMENTS.md</code> for full technical details</li>
</ul>
"""


def _get_collect2vs2_base_html() -> str:
    """Collect2vs2 v0 (deprecated) documentation."""
    return """
<h2>MosaicMultiGrid-Collect-2vs2-v0</h2>

<p style="background-color: #fff3e0; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>Status:</strong> Deprecated -- use <code>MosaicMultiGrid-Collect-2vs2-IndAgObs-v0</code> instead.
</p>

<p>
A 4-agent (2v2) team ball collection game. Green team (Agents 0, 1) vs Red team (Agents 2, 3).
7 wildcard balls (odd number prevents draws).
</p>

<h4>Environment Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Grid Size</td><td style="border: 1px solid #ddd; padding: 8px;">10 x 10</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Agents</td><td style="border: 1px solid #ddd; padding: 8px;">4 (2v2 teams)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Team 1 (Green)</td><td style="border: 1px solid #ddd; padding: 8px;">Agents 0, 1</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Team 2 (Red)</td><td style="border: 1px solid #ddd; padding: 8px;">Agents 2, 3</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Balls</td><td style="border: 1px solid #ddd; padding: 8px;">7 wildcard (odd = no draws)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Max Steps</td><td style="border: 1px solid #ddd; padding: 8px;">10,000</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Termination</td><td style="border: 1px solid #ddd; padding: 8px;">None (runs until max_steps)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Zero-Sum</td><td style="border: 1px solid #ddd; padding: 8px;">Yes</td></tr>
</table>
"""


def _get_collect2vs2_enhanced_html() -> str:
    """Collect2vs2 IndAgObs (recommended) documentation."""
    return """
<h2>MosaicMultiGrid-Collect-2vs2-IndAgObs-v0</h2>

<p style="background-color: #e8f5e9; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>RECOMMENDED</strong> for RL training.
Gymnasium API: <code>env.reset(seed=N)</code> returns <code>(obs, info)</code>.
Supports <strong>Parallel</strong> (default) and <strong>AEC</strong> execution modes.
</p>

<p>
Enhanced 2v2 team ball collection with <strong>natural termination</strong>.
Green team (Agents 0, 1) vs Red team (Agents 2, 3) compete to collect
7 wildcard balls. Odd number guarantees a winner (no draws).
</p>

<h4>Environment Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Grid Size</td><td style="border: 1px solid #ddd; padding: 8px;">10 x 10 (8 x 8 playable)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Agents</td><td style="border: 1px solid #ddd; padding: 8px;">4 (2v2 teams)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Team 1 (Green)</td><td style="border: 1px solid #ddd; padding: 8px;">Agents 0, 1</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Team 2 (Red)</td><td style="border: 1px solid #ddd; padding: 8px;">Agents 2, 3</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Balls</td><td style="border: 1px solid #ddd; padding: 8px;">7 wildcard (odd = guaranteed winner)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">View Size</td><td style="border: 1px solid #ddd; padding: 8px;">3 x 3</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Max Steps</td><td style="border: 1px solid #ddd; padding: 8px;">400</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Termination</td><td style="border: 1px solid #ddd; padding: 8px;">When all 7 balls collected</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Zero-Sum</td><td style="border: 1px solid #ddd; padding: 8px;">Yes (+1 team, -1 opponents)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Execution Mode</td><td style="border: 1px solid #ddd; padding: 8px;">Parallel (default) or AEC</td></tr>
</table>

<h4>Game Mechanics</h4>
<ul>
    <li><strong>Pickup</strong>: Face a ball and use PICKUP -- ball consumed on pickup</li>
    <li><strong>Team reward</strong>: Both teammates get +1 when either collects; both opponents get -1</li>
    <li><strong>Guaranteed winner</strong>: 7 balls means one team gets at least 4, other at most 3</li>
    <li><strong>Termination</strong>: Episode ends immediately when all 7 balls collected</li>
</ul>

<h4>Strategy</h4>
<ul>
    <li><strong>Role splitting</strong>: One agent collects, one blocks opponents</li>
    <li><strong>Map coverage</strong>: Split search areas (left/right) to avoid chasing same ball</li>
    <li><strong>MAPPO</strong>: Learns team coordination ("don't both chase same ball")</li>
</ul>

<h4>References</h4>
<ul>
    <li>Source: <code>3rd_party/mosaic_multigrid/</code></li>
    <li>Class: <code>CollectGame4HIndAgObsEnv10x10N2</code></li>
    <li>See: <code>COLLECT_IMPROVEMENTS.md</code> for full technical details</li>
</ul>
"""


# ---------------------------------------------------------------------------
# Basketball
# ---------------------------------------------------------------------------

def _get_basketball_html() -> str:
    """Basketball IndAgObs 3vs3 documentation."""
    return """
<h2>MosaicMultiGrid-Basketball-3vs3-IndAgObs-v0</h2>

<p style="background-color: #e8f5e9; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>NEW</strong> in v4.0.0. 3v3 basketball with court rendering.
Gymnasium API: <code>env.reset(seed=N)</code> returns <code>(obs, info)</code>.
Supports <strong>Parallel</strong> (default) and <strong>AEC</strong> execution modes.
</p>

<p>
6-agent (3v3) basketball game on a court-rendered grid.
Two teams compete to score baskets. Court rendering provides visual
court markings (three-point line, key, half-court line).
</p>

<h4>Environment Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Agents</td><td style="border: 1px solid #ddd; padding: 8px;">6 (3 per team)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Team 1</td><td style="border: 1px solid #ddd; padding: 8px;">Agents 0, 1, 2</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Team 2</td><td style="border: 1px solid #ddd; padding: 8px;">Agents 3, 4, 5</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">View Size</td><td style="border: 1px solid #ddd; padding: 8px;">3 x 3</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Rendering</td><td style="border: 1px solid #ddd; padding: 8px;">Court markings (three-point line, key, half-court)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Execution Mode</td><td style="border: 1px solid #ddd; padding: 8px;">Parallel (default) or AEC</td></tr>
</table>

<h4>Action Space</h4>
<p><code>Discrete(8)</code> -- same for all agents (NOOP=0 enables AEC mode):</p>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">ID</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Action</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Basketball Use</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">0</td><td style="border: 1px solid #ddd; padding: 8px;">NOOP</td><td style="border: 1px solid #ddd; padding: 8px;">Idle — AEC no-op</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">1</td><td style="border: 1px solid #ddd; padding: 8px;">LEFT</td><td style="border: 1px solid #ddd; padding: 8px;">Rotate counter-clockwise</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">2</td><td style="border: 1px solid #ddd; padding: 8px;">RIGHT</td><td style="border: 1px solid #ddd; padding: 8px;">Rotate clockwise</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">3</td><td style="border: 1px solid #ddd; padding: 8px;">FORWARD</td><td style="border: 1px solid #ddd; padding: 8px;">Move in facing direction / dribble</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">4</td><td style="border: 1px solid #ddd; padding: 8px;">PICKUP</td><td style="border: 1px solid #ddd; padding: 8px;">Grab ball / steal from opponent</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">5</td><td style="border: 1px solid #ddd; padding: 8px;">DROP</td><td style="border: 1px solid #ddd; padding: 8px;">Shoot / pass to teammate</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">6</td><td style="border: 1px solid #ddd; padding: 8px;">TOGGLE</td><td style="border: 1px solid #ddd; padding: 8px;">Toggle object</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">7</td><td style="border: 1px solid #ddd; padding: 8px;">DONE</td><td style="border: 1px solid #ddd; padding: 8px;">Signal completion</td></tr>
</table>

<h4>AEC Mode Note</h4>
<p style="background-color: #e3f2fd; padding: 8px; border-radius: 4px;">
With 6 agents in AEC mode, each round requires 6 <code>env.step()</code> calls.
Agent ordering: 0 → 1 → 2 → 3 → 4 → 5 → repeat.
Each agent observes the intermediate state produced by all preceding agents
in the current round before choosing its action.
</p>

<h4>TeamObs Variant</h4>
<p>
<code>MosaicMultiGrid-Basketball-3vs3-TeamObs-v0</code> extends IndAgObs
with SMAC-style teammate awareness: each agent's observation includes
relative positions and states of all teammates.
</p>

<h4>References</h4>
<ul>
    <li>Source: <code>3rd_party/mosaic_multigrid/</code></li>
    <li>AEC wrapper: <code>gym_gui/services/aec_wrapper.py</code></li>
</ul>
"""


# ---------------------------------------------------------------------------
# American Football
# ---------------------------------------------------------------------------

def _get_american_football_html() -> str:
    """American Football documentation."""
    return """
<h2>MosaicMultiGrid-AmericanFootball (1v1, 2v2, 3v3)</h2>

<p style="background-color: #e8f5e9; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>NEW</strong> in v6.3.0. American Football with brown field rendering, end zones,
touchdown scoring, and ball stealing mechanics.
Gymnasium API: <code>env.reset(seed=N)</code> returns <code>(obs, info)</code>.
Supports <strong>Parallel</strong> (default) and <strong>AEC</strong> execution modes.
</p>

<p>
Multi-agent American Football game on a 16x11 brown field with end zones.
Two teams compete to score touchdowns by advancing the ball into the opponent's end zone.
Available in 1v1 (2 agents), 2v2 (4 agents), and 3v3 (6 agents) configurations.
</p>

<h4>Environment Variants</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Environment</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Agents</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Features</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><code>MosaicMultiGrid-AmericanFootball-1v1-v0</code></td><td style="border: 1px solid #ddd; padding: 8px;">2 (1v1)</td><td style="border: 1px solid #ddd; padding: 8px;">Individual play</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><code>MosaicMultiGrid-AmericanFootball-2v2-v0</code></td><td style="border: 1px solid #ddd; padding: 8px;">4 (2v2)</td><td style="border: 1px solid #ddd; padding: 8px;">Team coordination</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><code>MosaicMultiGrid-AmericanFootball-3v3-v0</code></td><td style="border: 1px solid #ddd; padding: 8px;">6 (3v3)</td><td style="border: 1px solid #ddd; padding: 8px;">Full team play</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><code>MosaicMultiGrid-AmericanFootball-2v2-TeamObs-v0</code></td><td style="border: 1px solid #ddd; padding: 8px;">4 (2v2)</td><td style="border: 1px solid #ddd; padding: 8px;">+ teammate awareness</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><code>MosaicMultiGrid-AmericanFootball-3v3-TeamObs-v0</code></td><td style="border: 1px solid #ddd; padding: 8px;">6 (3v3)</td><td style="border: 1px solid #ddd; padding: 8px;">+ teammate awareness</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><code>MosaicMultiGrid-AmericanFootball-Solo-Green-v0</code></td><td style="border: 1px solid #ddd; padding: 8px;">1</td><td style="border: 1px solid #ddd; padding: 8px;">Solo training (Green)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><code>MosaicMultiGrid-AmericanFootball-Solo-Blue-v0</code></td><td style="border: 1px solid #ddd; padding: 8px;">1</td><td style="border: 1px solid #ddd; padding: 8px;">Solo training (Blue)</td></tr>
</table>

<h4>Environment Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Grid Size</td><td style="border: 1px solid #ddd; padding: 8px;">16 x 11 (14 x 9 playable)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">View Size</td><td style="border: 1px solid #ddd; padding: 8px;">3 x 3 (partial observability)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Field Rendering</td><td style="border: 1px solid #ddd; padding: 8px;">Brown field with end zones</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Scoring</td><td style="border: 1px solid #ddd; padding: 8px;">Touchdown in opponent's end zone</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Execution Mode</td><td style="border: 1px solid #ddd; padding: 8px;">Parallel (default) or AEC</td></tr>
</table>

<h4>Action Space</h4>
<p><code>Discrete(8)</code> -- same for all agents (NOOP=0 enables AEC mode):</p>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">ID</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Action</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Football Use</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">0</td><td style="border: 1px solid #ddd; padding: 8px;">NOOP</td><td style="border: 1px solid #ddd; padding: 8px;">Idle — AEC no-op</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">1</td><td style="border: 1px solid #ddd; padding: 8px;">LEFT</td><td style="border: 1px solid #ddd; padding: 8px;">Rotate counter-clockwise</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">2</td><td style="border: 1px solid #ddd; padding: 8px;">RIGHT</td><td style="border: 1px solid #ddd; padding: 8px;">Rotate clockwise</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">3</td><td style="border: 1px solid #ddd; padding: 8px;">FORWARD</td><td style="border: 1px solid #ddd; padding: 8px;">Move in facing direction / advance ball</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">4</td><td style="border: 1px solid #ddd; padding: 8px;">PICKUP</td><td style="border: 1px solid #ddd; padding: 8px;">Grab ball / steal from opponent</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">5</td><td style="border: 1px solid #ddd; padding: 8px;">DROP</td><td style="border: 1px solid #ddd; padding: 8px;">Score touchdown / pass to teammate</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">6</td><td style="border: 1px solid #ddd; padding: 8px;">TOGGLE</td><td style="border: 1px solid #ddd; padding: 8px;">Toggle object</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">7</td><td style="border: 1px solid #ddd; padding: 8px;">DONE</td><td style="border: 1px solid #ddd; padding: 8px;">Signal completion</td></tr>
</table>

<h4>Mechanics</h4>
<ul>
    <li><strong>Ball Carrying:</strong> Agents can pick up and carry the ball while moving</li>
    <li><strong>Stealing:</strong> Opponents can steal the ball using the PICKUP action</li>
    <li><strong>Passing:</strong> DROP action passes to teammates or scores in end zone</li>
    <li><strong>Touchdown Scoring:</strong> Reach opponent's end zone with the ball and DROP to score</li>
    <li><strong>Team Rewards:</strong> Shared positive rewards for team members on touchdown</li>
</ul>

<h4>TeamObs Variant</h4>
<p>
<code>MosaicMultiGrid-AmericanFootball-*-TeamObs-v0</code> variants extend IndAgObs
with SMAC-style teammate awareness: each agent's observation includes
relative positions, directions, and ball-carrying status of all teammates.
</p>

<h4>References</h4>
<ul>
    <li>Source: <code>3rd_party/mosaic_multigrid/</code></li>
    <li>AEC wrapper: <code>gym_gui/services/aec_wrapper.py</code></li>
    <li>Field rendering: Brown field with end zone markings</li>
</ul>
"""


# ---------------------------------------------------------------------------
# Solo
# ---------------------------------------------------------------------------

def _get_solo_html(env_id: str) -> str:
    """Solo environment (v6.0.0) documentation."""
    if "Soccer" in env_id:
        sport = "Soccer"
        grid = "16 x 11 (14 x 9 playable, FIFA 1.54 ratio)"
    elif "Basketball" in env_id:
        sport = "Basketball"
        grid = "19 x 11 (17 x 9 playable)"
    elif "AmericanFootball" in env_id:
        sport = "American Football"
        grid = "16 x 11 (14 x 9 playable)"
    else:
        sport = "Unknown"
        grid = "Unknown"

    if "Green" in env_id:
        color = "Green"
        team = "1"
        direction = "right"
    else:
        color = "Blue"
        team = "2"
        direction = "left"

    return f"""
<h2>{env_id}</h2>

<p style="background-color: #e8f5e9; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>NEW</strong> in v6.0.0. Single-agent solo training -- no opponent on the field.
Gymnasium API: <code>env.reset(seed=N)</code> returns <code>(obs, info)</code>.
</p>

<p>
A single <strong>{color}</strong> agent (team {team}) on an empty {sport.lower()} field.
The agent must learn to pick up the ball, navigate to the goal, and score
without any opponent interference. Designed for <strong>curriculum pre-training</strong>
before transferring to competitive multi-agent environments.
</p>

<h4>Environment Details</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Grid Size</td><td style="border: 1px solid #ddd; padding: 8px;">{grid}</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Agents</td><td style="border: 1px solid #ddd; padding: 8px;">1 ({color}, team {team})</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Opponent</td><td style="border: 1px solid #ddd; padding: 8px;">None</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Ball</td><td style="border: 1px solid #ddd; padding: 8px;">1 wildcard ball (respawns after goal)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Scoring Direction</td><td style="border: 1px solid #ddd; padding: 8px;">Scores {direction}</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">View Size</td><td style="border: 1px solid #ddd; padding: 8px;">3 x 3 (default, override with view_size=7)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Max Steps</td><td style="border: 1px solid #ddd; padding: 8px;">200</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Win Condition</td><td style="border: 1px solid #ddd; padding: 8px;">First to 2 goals</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Reward</td><td style="border: 1px solid #ddd; padding: 8px;">+1 per goal scored</td></tr>
</table>

<h4>Why Solo?</h4>
<ul>
    <li><strong>Curriculum pre-training</strong>: Learn ball pickup + scoring without opponent interference</li>
    <li><strong>Solves sparse reward</strong>: Random agents score 0 goals in 100 episodes with opponents; solo removes opponent blocking</li>
    <li><strong>Team-specific checkpoints</strong>: Green scores right, Blue scores left -- each learns its own goal direction</li>
    <li><strong>Transfer to 1v1/2v2</strong>: Pre-trained solo checkpoint initializes competitive training</li>
</ul>

<h4>Action Space</h4>
<p><code>Discrete(8)</code> -- same as all MOSAIC MultiGrid environments (NOOP=0).</p>

<h4>References</h4>
<ul>
    <li>Source: <code>3rd_party/mosaic_multigrid/</code></li>
    <li>See: <code>SOCCER_IMPROVEMENTS.md</code> for curriculum strategy</li>
</ul>
"""


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

def _get_overview_html() -> str:
    """Return overview HTML for MOSAIC MultiGrid family."""
    return """
<h2>MOSAIC MultiGrid</h2>

<p>
<strong>MOSAIC MultiGrid</strong> is a competitive multi-agent grid-world package
built on Gymnasium. It provides team-based environments (Soccer, Collect, Basketball)
with partial observability (3x3 view) and multi-keyboard support.
</p>

<h4>Execution Modes</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Mode</th>
        <th style="border: 1px solid #ddd; padding: 8px;">How it works</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Game model</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Implementation</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Parallel</strong> (default)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">All agents observe S(t) and act simultaneously.<br><code>env.step([A_0, A_1, ...])</code> fires once per round.</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Normal-form (Markov Game)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Direct Gymnasium API</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>AEC</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Agent_0 acts → physics fires → S(t+0.5).<br>Agent_1 observes intermediate state before deciding.<br><code>env.step([A_i, NOOP, ...])</code> × N per round.</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Extensive-form (sequential)</td>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>GymnasiumMultiAgentAECWrapper</code><br>(gym_gui/services/aec_wrapper.py)</td>
    </tr>
</table>
<p style="background-color: #ffecb3; padding: 8px; border-radius: 4px;">
<strong>AEC requires NOOP=0</strong> (mosaic_multigrid v5.0.0+, current v6.0.0). Non-acting agents receive NOOP
so only the active agent's action affects the physics. AEC is <em>not available</em> for
<code>ini_multigrid</code> (action 0 = LEFT) or <code>overcooked</code> (no NOOP).
</p>

<h4>Available Environments</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Environment</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Agents</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Type</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Status</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MosaicMultiGrid-Soccer-2vs2-IndAgObs-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">4 (2v2)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Zero-sum soccer</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Recommended</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MosaicMultiGrid-Soccer-1vs1-IndAgObs-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">2 (1v1)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Zero-sum soccer</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Recommended</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MosaicMultiGrid-Collect-IndAgObs-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">3</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Individual competition</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Recommended</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MosaicMultiGrid-Collect-2vs2-IndAgObs-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">4 (2v2)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Team collection</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Recommended</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MosaicMultiGrid-Collect-1vs1-IndAgObs-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">2 (1v1)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Team collection</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Recommended</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MosaicMultiGrid-Basketball-3vs3-IndAgObs-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">6 (3v3)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Basketball (court rendering)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Recommended</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MosaicMultiGrid-Soccer-2vs2-TeamObs-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">4 (2v2)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Soccer + teammate obs</td>
        <td style="border: 1px solid #ddd; padding: 8px;">SMAC-style</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MosaicMultiGrid-Collect-2vs2-TeamObs-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">4 (2v2)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Collection + teammate obs</td>
        <td style="border: 1px solid #ddd; padding: 8px;">SMAC-style</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MosaicMultiGrid-Basketball-3vs3-TeamObs-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">6 (3v3)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Basketball + teammate obs</td>
        <td style="border: 1px solid #ddd; padding: 8px;">SMAC-style</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MosaicMultiGrid-Soccer-Solo-Green-IndAgObs-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">1</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Solo soccer (Green, scores right)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">v6.0.0</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MosaicMultiGrid-Soccer-Solo-Blue-IndAgObs-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">1</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Solo soccer (Blue, scores left)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">v6.0.0</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MosaicMultiGrid-Basketball-Solo-Green-IndAgObs-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">1</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Solo basketball (Green, scores right)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">v6.0.0</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MosaicMultiGrid-Basketball-Solo-Blue-IndAgObs-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">1</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Solo basketball (Blue, scores left)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">v6.0.0</td>
    </tr>
</table>

<h4>API</h4>
<p>
Uses <strong>Gymnasium</strong> API:
<code>env.reset(seed=42)</code> returns <code>(obs, info)</code>.
<code>env.step(actions)</code> returns <code>(obs, rewards, terminated, truncated, info)</code>.
Action space v6.0.0: <code>Discrete(8)</code> — 0=NOOP, 1=LEFT, 2=RIGHT, 3=FORWARD, 4=PICKUP, 5=DROP, 6=TOGGLE, 7=DONE.
</p>

<h4>Source</h4>
<p><code>3rd_party/mosaic_multigrid/</code> &nbsp;|&nbsp; PyPI: <code>mosaic-multigrid</code></p>
"""


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def get_mosaic_multigrid_html(env_id: str) -> str:
    """Generate MOSAIC MultiGrid HTML documentation for a specific variant.

    Args:
        env_id: Environment identifier (e.g., "MosaicMultiGrid-Soccer-2vs2-IndAgObs-v0")

    Returns:
        HTML string containing environment documentation.
    """
    _is_modern = ("IndAgObs" in env_id or "TeamObs" in env_id
                   or "Enhanced" in env_id)
    if "Solo" in env_id:
        return _get_solo_html(env_id)
    elif "Basketball" in env_id:
        return _get_basketball_html()
    elif "Soccer" in env_id:
        if "1vs1" in env_id:
            return _get_soccer_1vs1_html()
        if _is_modern:
            return _get_soccer_enhanced_html()
        return _get_soccer_base_html()
    elif "Collect-2vs2" in env_id or "Collect2vs2" in env_id:
        if _is_modern:
            return _get_collect2vs2_enhanced_html()
        return _get_collect2vs2_base_html()
    elif "Collect-1vs1" in env_id:
        return _get_collect2vs2_enhanced_html()  # 1vs1 shares collect team docs
    elif "Collect" in env_id:
        if _is_modern:
            return _get_collect_enhanced_html()
        return _get_collect_base_html()
    elif "AmericanFootball" in env_id:
        return _get_american_football_html()
    else:
        return _get_overview_html()


# Pre-generated HTML constants for convenience
MOSAIC_SOCCER_HTML = _get_soccer_enhanced_html()
MOSAIC_SOCCER_BASE_HTML = _get_soccer_base_html()
MOSAIC_SOCCER_1VS1_HTML = _get_soccer_1vs1_html()
MOSAIC_COLLECT_HTML = _get_collect_enhanced_html()
MOSAIC_COLLECT_BASE_HTML = _get_collect_base_html()
MOSAIC_COLLECT2VS2_HTML = _get_collect2vs2_enhanced_html()
MOSAIC_COLLECT2VS2_BASE_HTML = _get_collect2vs2_base_html()
MOSAIC_BASKETBALL_HTML = _get_basketball_html()
MOSAIC_AMERICAN_FOOTBALL_HTML = _get_american_football_html()
MOSAIC_OVERVIEW_HTML = _get_overview_html()
MOSAIC_SOLO_SOCCER_GREEN_HTML = _get_solo_html("MosaicMultiGrid-Soccer-Solo-Green-IndAgObs-v0")
MOSAIC_SOLO_SOCCER_BLUE_HTML = _get_solo_html("MosaicMultiGrid-Soccer-Solo-Blue-IndAgObs-v0")
MOSAIC_SOLO_BASKETBALL_GREEN_HTML = _get_solo_html("MosaicMultiGrid-Basketball-Solo-Green-IndAgObs-v0")
MOSAIC_SOLO_BASKETBALL_BLUE_HTML = _get_solo_html("MosaicMultiGrid-Basketball-Solo-Blue-IndAgObs-v0")
MOSAIC_SOLO_AMERICAN_FOOTBALL_GREEN_HTML = _get_solo_html("MosaicMultiGrid-AmericanFootball-Solo-Green-v0")
MOSAIC_SOLO_AMERICAN_FOOTBALL_BLUE_HTML = _get_solo_html("MosaicMultiGrid-AmericanFootball-Solo-Blue-v0")

__all__ = [
    "get_mosaic_multigrid_html",
    "MOSAIC_SOCCER_HTML",
    "MOSAIC_SOCCER_BASE_HTML",
    "MOSAIC_SOCCER_1VS1_HTML",
    "MOSAIC_COLLECT_HTML",
    "MOSAIC_COLLECT_BASE_HTML",
    "MOSAIC_COLLECT2VS2_HTML",
    "MOSAIC_COLLECT2VS2_BASE_HTML",
    "MOSAIC_BASKETBALL_HTML",
    "MOSAIC_AMERICAN_FOOTBALL_HTML",
    "MOSAIC_OVERVIEW_HTML",
    "MOSAIC_SOLO_SOCCER_GREEN_HTML",
    "MOSAIC_SOLO_SOCCER_BLUE_HTML",
    "MOSAIC_SOLO_BASKETBALL_GREEN_HTML",
    "MOSAIC_SOLO_BASKETBALL_BLUE_HTML",
    "MOSAIC_SOLO_AMERICAN_FOOTBALL_GREEN_HTML",
    "MOSAIC_SOLO_AMERICAN_FOOTBALL_BLUE_HTML",
]
