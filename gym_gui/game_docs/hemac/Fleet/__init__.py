"""Documentation for HeMAC Fleet environments."""
from __future__ import annotations

_FLEET_SCENARIOS = {
    "3q1o": {"quadcopters": 3, "observers": 1, "provisioners": 0},
    "10q3o": {"quadcopters": 10, "observers": 3, "provisioners": 0},
    "20q5o": {"quadcopters": 20, "observers": 5, "provisioners": 0},
}


def get_fleet_html(env_id: str) -> str:
    """Generate Fleet HTML documentation for a specific variant."""
    for key, comp in _FLEET_SCENARIOS.items():
        if key in env_id:
            composition = comp
            break
    else:
        composition = {"quadcopters": "?", "observers": "?", "provisioners": 0}

    total = composition["quadcopters"] + composition["observers"]

    return f"""
<h2>{env_id}</h2>

<p style="background-color: #fff3e0; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>API:</strong> PettingZoo AEC -- Heterogeneous multi-agent drone fleet.
<a href="https://github.com/ThalesGroup/HeMAC" target="_blank">GitHub</a>
</p>

<p>Fleet challenge: medium-scale heterogeneous drone swarm. Larger teams
of quadcopters and observers coordinate to search and intercept POIs
across a continuous area. Scales multi-agent credit assignment and
communication challenges. {total} agents ({composition['quadcopters']}Q + {composition['observers']}O).</p>

<h4>Available Variants</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Environment ID</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Quadcopters</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Observers</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Total</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>hemac-fleet-3q1o-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">3</td>
        <td style="border: 1px solid #ddd; padding: 8px;">1</td>
        <td style="border: 1px solid #ddd; padding: 8px;">4</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>hemac-fleet-10q3o-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">10</td>
        <td style="border: 1px solid #ddd; padding: 8px;">3</td>
        <td style="border: 1px solid #ddd; padding: 8px;">13</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>hemac-fleet-20q5o-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">20</td>
        <td style="border: 1px solid #ddd; padding: 8px;">5</td>
        <td style="border: 1px solid #ddd; padding: 8px;">25</td>
    </tr>
</table>

<h4>Agent Types & Observation Spaces</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Agent Type</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Obs Space</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Action Space</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Description</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Quadcopter</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Box(-10000, 10000, (14,))</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Box(-1, 1, (2,))</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Low-altitude interceptor. Continuous thrust/heading. Max speed 14–16 m/s.</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Observer</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Box(-10000, 10000, (11,))</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Discrete(5)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">High-altitude surveillance (100m). Discrete heading control. Comm range 150m.</td>
    </tr>
</table>

<h4>Rewards & Episode End</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Condition</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">POI interception</td><td style="border: 1px solid #ddd; padding: 8px;">+reward (distance-based)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Termination</td><td style="border: 1px solid #ddd; padding: 8px;">All POIs captured or all agents destroyed</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Truncation</td><td style="border: 1px solid #ddd; padding: 8px;">max_cycles (default 300 steps)</td></tr>
</table>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/ThalesGroup/HeMAC" target="_blank">HeMAC GitHub (ThalesGroup)</a></li>
</ul>
"""


HEMAC_FLEET_HTML = get_fleet_html("hemac-fleet-3q1o-v0")

__all__ = ["HEMAC_FLEET_HTML", "get_fleet_html"]
