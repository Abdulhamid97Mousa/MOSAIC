"""Documentation for MosaicMalmo Navigate environments."""
from __future__ import annotations


def get_malmo_navigate_html(env_id: str) -> str:
    """Generate MosaicMalmo Navigate HTML documentation."""
    return f"""
<h2>{env_id}</h2>

<p style="background-color: #e8f5e9; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>API:</strong> Gymnasium (single-agent) -- 3D Minecraft navigation via Go-based Malmo.
</p>

<p>Navigate a Minecraft world to reach a goal location. Uses a Go-based Malmo
backend (replacing the original Java client) for faster, lightweight 3D
environment interaction. The agent receives an 84×84 RGB egocentric view and
must learn spatial reasoning to move through block-based terrain.</p>

<h4>Available Variants</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Environment ID</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Task</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Agents</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>MosaicMalmo-Navigate-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Navigate to goal</td>
        <td style="border: 1px solid #ddd; padding: 8px;">1</td>
    </tr>
</table>

<h4>Observation Space</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Component</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Space</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Description</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>pixels</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Box(0, 255, (84, 84, 3), uint8)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">84×84 RGB egocentric view of the Minecraft world</td>
    </tr>
</table>

<h4>Action Space (Discrete(10))</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #e8f5e9;">
        <th style="border: 1px solid #ddd; padding: 8px;">ID</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Action</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Description</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>0</strong></td><td style="border: 1px solid #ddd; padding: 8px;">NOOP</td><td style="border: 1px solid #ddd; padding: 8px;">Do nothing</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>1</strong></td><td style="border: 1px solid #ddd; padding: 8px;">MOVE_FORWARD</td><td style="border: 1px solid #ddd; padding: 8px;">Move forward one block</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>2</strong></td><td style="border: 1px solid #ddd; padding: 8px;">MOVE_BACKWARD</td><td style="border: 1px solid #ddd; padding: 8px;">Move backward one block</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>3</strong></td><td style="border: 1px solid #ddd; padding: 8px;">MOVE_LEFT</td><td style="border: 1px solid #ddd; padding: 8px;">Strafe left</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>4</strong></td><td style="border: 1px solid #ddd; padding: 8px;">MOVE_RIGHT</td><td style="border: 1px solid #ddd; padding: 8px;">Strafe right</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>5</strong></td><td style="border: 1px solid #ddd; padding: 8px;">TURN_LEFT</td><td style="border: 1px solid #ddd; padding: 8px;">Rotate camera left</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>6</strong></td><td style="border: 1px solid #ddd; padding: 8px;">TURN_RIGHT</td><td style="border: 1px solid #ddd; padding: 8px;">Rotate camera right</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>7</strong></td><td style="border: 1px solid #ddd; padding: 8px;">JUMP</td><td style="border: 1px solid #ddd; padding: 8px;">Jump</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>8</strong></td><td style="border: 1px solid #ddd; padding: 8px;">ATTACK</td><td style="border: 1px solid #ddd; padding: 8px;">Attack / break block</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>9</strong></td><td style="border: 1px solid #ddd; padding: 8px;">USE</td><td style="border: 1px solid #ddd; padding: 8px;">Use / place block</td></tr>
</table>

<h4>Rewards & Episode End</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Condition</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Reach goal</td><td style="border: 1px solid #ddd; padding: 8px;">+1.0</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Step penalty</td><td style="border: 1px solid #ddd; padding: 8px;">0</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Termination</td><td style="border: 1px solid #ddd; padding: 8px;">Agent reaches goal location</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Truncation</td><td style="border: 1px solid #ddd; padding: 8px;">Max episode steps timeout</td></tr>
</table>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/Microsoft/malmo" target="_blank">Project Malmo (Microsoft)</a></li>
</ul>
"""


MOSAIC_MALMO_NAVIGATE_HTML = get_malmo_navigate_html("MosaicMalmo-Navigate-v0")

__all__ = ["MOSAIC_MALMO_NAVIGATE_HTML", "get_malmo_navigate_html"]
