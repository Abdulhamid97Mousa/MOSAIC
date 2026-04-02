"""Documentation for HeMAC Simple Fleet environments."""
from __future__ import annotations

_SIMPLE_FLEET_SCENARIOS = {
    "1q1o": {"quadcopters": 1, "observers": 1, "provisioners": 0},
    "3q1o": {"quadcopters": 3, "observers": 1, "provisioners": 0},
    "5q2o": {"quadcopters": 5, "observers": 2, "provisioners": 0},
}


def get_simple_fleet_html(env_id: str) -> str:
    """Generate Simple Fleet HTML documentation for a specific variant."""
    for key, comp in _SIMPLE_FLEET_SCENARIOS.items():
        if key in env_id:
            composition = comp
            break
    else:
        composition = {"quadcopters": "?", "observers": "?", "provisioners": 0}

    total = composition["quadcopters"] + composition["observers"]

    return f"""
<h2>{env_id}</h2>

<p style="background-color: #fff3e0; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>API:</strong> PettingZoo AEC -- Heterogeneous multi-agent drone search.
<a href="https://github.com/ThalesGroup/HeMAC" target="_blank">GitHub</a>
</p>

<p>Simple Fleet challenge: quadcopter drones and high-altitude observers
cooperate to locate and intercept Points of Interest (POIs) in a 2D
continuous area. No provisioner (ground vehicle) — focuses on aerial
coordination. {total} agents ({composition['quadcopters']}Q + {composition['observers']}O).</p>

<h4>Available Variants</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Environment ID</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Quadcopters</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Observers</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Total</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>hemac-simple-fleet-1q1o-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">1</td>
        <td style="border: 1px solid #ddd; padding: 8px;">1</td>
        <td style="border: 1px solid #ddd; padding: 8px;">2</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>hemac-simple-fleet-3q1o-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">3</td>
        <td style="border: 1px solid #ddd; padding: 8px;">1</td>
        <td style="border: 1px solid #ddd; padding: 8px;">4</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>hemac-simple-fleet-5q2o-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">5</td>
        <td style="border: 1px solid #ddd; padding: 8px;">2</td>
        <td style="border: 1px solid #ddd; padding: 8px;">7</td>
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
        <td style="border: 1px solid #ddd; padding: 8px;">Low-altitude interceptor. Continuous thrust/heading control. Sensors: DownwardCamera, RoundCamera, IMU, UWB.</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Observer</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Box(-10000, 10000, (11,))</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Discrete(5)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">High-altitude surveillance (100m). Discrete heading: turn_right, turn_left, straight×3. ForwardFacingCamera.</td>
    </tr>
</table>

<h4>Rewards & Episode End</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Condition</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">POI interception</td><td style="border: 1px solid #ddd; padding: 8px;">+reward (distance-based)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Energy penalty</td><td style="border: 1px solid #ddd; padding: 8px;">Implicit via dynamics</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Termination</td><td style="border: 1px solid #ddd; padding: 8px;">All POIs captured or all agents destroyed</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Truncation</td><td style="border: 1px solid #ddd; padding: 8px;">max_cycles (default 300 steps)</td></tr>
</table>

<h4>Key Parameters</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Parameter</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Default</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Area size</td><td style="border: 1px solid #ddd; padding: 8px;">1000 × 1000</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Obstacles</td><td style="border: 1px solid #ddd; padding: 8px;">2–3 random per episode</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Time factor</td><td style="border: 1px solid #ddd; padding: 8px;">0.8–1.0</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Render</td><td style="border: 1px solid #ddd; padding: 8px;">PyGame ('human' or None)</td></tr>
</table>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/ThalesGroup/HeMAC" target="_blank">HeMAC GitHub (ThalesGroup)</a></li>
    <li>ECAI 2025 Paper: Heterogeneous Multi-Agent Challenge</li>
</ul>
"""


HEMAC_SIMPLE_FLEET_HTML = get_simple_fleet_html("hemac-simple-fleet-1q1o-v0")

__all__ = ["HEMAC_SIMPLE_FLEET_HTML", "get_simple_fleet_html"]
