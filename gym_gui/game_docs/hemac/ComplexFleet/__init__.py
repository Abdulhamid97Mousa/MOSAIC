"""Documentation for HeMAC Complex Fleet environments."""
from __future__ import annotations

_COMPLEX_FLEET_SCENARIOS = {
    "3q1o1p": {"quadcopters": 3, "observers": 1, "provisioners": 1},
    "5q2o1p": {"quadcopters": 5, "observers": 2, "provisioners": 1},
}


def get_complex_fleet_html(env_id: str) -> str:
    """Generate Complex Fleet HTML documentation for a specific variant."""
    for key, comp in _COMPLEX_FLEET_SCENARIOS.items():
        if key in env_id:
            composition = comp
            break
    else:
        composition = {"quadcopters": "?", "observers": "?", "provisioners": "?"}

    total = composition["quadcopters"] + composition["observers"] + composition["provisioners"]

    return f"""
<h2>{env_id}</h2>

<p style="background-color: #fff3e0; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>API:</strong> PettingZoo AEC -- Heterogeneous multi-agent fleet with ground vehicles.
<a href="https://github.com/ThalesGroup/HeMAC" target="_blank">GitHub</a>
</p>

<p>Complex Fleet challenge: the full heterogeneous benchmark with all three
agent types — quadcopters (aerial interceptors), observers (high-altitude
surveillance), and provisioners (road-constrained ground vehicles).
Provisioners relay observer intel and support drones, adding a
communication-routing layer. {total} agents
({composition['quadcopters']}Q + {composition['observers']}O + {composition['provisioners']}P).</p>

<h4>Available Variants</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Environment ID</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Quadcopters</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Observers</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Provisioners</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Total</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>hemac-complex-fleet-3q1o1p-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">3</td>
        <td style="border: 1px solid #ddd; padding: 8px;">1</td>
        <td style="border: 1px solid #ddd; padding: 8px;">1</td>
        <td style="border: 1px solid #ddd; padding: 8px;">5</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>hemac-complex-fleet-5q2o1p-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">5</td>
        <td style="border: 1px solid #ddd; padding: 8px;">2</td>
        <td style="border: 1px solid #ddd; padding: 8px;">1</td>
        <td style="border: 1px solid #ddd; padding: 8px;">8</td>
    </tr>
</table>

<h4>Agent Types & Spaces</h4>
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
        <td style="border: 1px solid #ddd; padding: 8px;">Low-altitude interceptor. Continuous thrust/heading. Speed 14–16 m/s, alt 30m.</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Observer</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Box(-10000, 10000, (11,))</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Discrete(5)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">High-altitude surveillance (100m). Discrete heading. Comm range 150m.</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Provisioner</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Box(-10000, 10000, (8,))</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Discrete(5)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Ground vehicle. Road-constrained: stay, east, north, west, south. Speed 8 m/s.</td>
    </tr>
</table>

<h4>Provisioner Action Space (Discrete(5))</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #e8f5e9;">
        <th style="border: 1px solid #ddd; padding: 8px;">ID</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Action</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Description</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>0</strong></td><td style="border: 1px solid #ddd; padding: 8px;">STAY</td><td style="border: 1px solid #ddd; padding: 8px;">Remain at current position</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>1</strong></td><td style="border: 1px solid #ddd; padding: 8px;">DRIVE_EAST</td><td style="border: 1px solid #ddd; padding: 8px;">Drive east (if road exists)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>2</strong></td><td style="border: 1px solid #ddd; padding: 8px;">DRIVE_NORTH</td><td style="border: 1px solid #ddd; padding: 8px;">Drive north (if road exists)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>3</strong></td><td style="border: 1px solid #ddd; padding: 8px;">DRIVE_WEST</td><td style="border: 1px solid #ddd; padding: 8px;">Drive west (if road exists)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>4</strong></td><td style="border: 1px solid #ddd; padding: 8px;">DRIVE_SOUTH</td><td style="border: 1px solid #ddd; padding: 8px;">Drive south (if road exists)</td></tr>
</table>

<h4>Rewards & Episode End</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Condition</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">POI interception</td><td style="border: 1px solid #ddd; padding: 8px;">+reward (distance-based, global)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Termination</td><td style="border: 1px solid #ddd; padding: 8px;">All POIs captured or all agents destroyed</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Truncation</td><td style="border: 1px solid #ddd; padding: 8px;">max_cycles (default 300 steps)</td></tr>
</table>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/ThalesGroup/HeMAC" target="_blank">HeMAC GitHub (ThalesGroup)</a></li>
    <li>ECAI 2025: Heterogeneous Multi-Agent Challenge</li>
</ul>
"""


HEMAC_COMPLEX_FLEET_HTML = get_complex_fleet_html("hemac-complex-fleet-3q1o1p-v0")

__all__ = ["HEMAC_COMPLEX_FLEET_HTML", "get_complex_fleet_html"]
