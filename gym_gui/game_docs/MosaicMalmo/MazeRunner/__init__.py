"""Documentation for MOSAIC MalmoEnv MazeRunner environment."""
from __future__ import annotations


def get_mazerunner_html(env_id: str) -> str:
    """Generate MazeRunner HTML documentation."""
    return f"""
<h2>{env_id}</h2>

<p style="background-color: #e3f2fd; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>API:</strong> MalmoEnv (single-agent) &mdash; Minecraft mission via MOSAIC MalmoEnv.
<a href="https://github.com/microsoft/malmo" target="_blank">Project Malmo</a> |
<em>Microsoft Research</em>
</p>

<p>Navigate through a procedurally generated maze in Minecraft. Find the exit while avoiding walls. Attack commands allow breaking obstacles.</p>

<h4>Observation Space</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Component</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Space</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Description</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>pixels</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Box(0, 255, (240, 320, 3), uint8)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">240x320 RGB egocentric Minecraft view</td>
    </tr>
</table>

<h4>Action Space (Discrete(8))</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #e3f2fd;">
        <th style="border: 1px solid #ddd; padding: 8px;">ID</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Command</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Description</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">0</td><td style="border: 1px solid #ddd; padding: 8px;"><code>move 1</code></td><td style="border: 1px solid #ddd; padding: 8px;">Move Forward</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">1</td><td style="border: 1px solid #ddd; padding: 8px;"><code>move -1</code></td><td style="border: 1px solid #ddd; padding: 8px;">Move Backward</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">2</td><td style="border: 1px solid #ddd; padding: 8px;"><code>turn 1</code></td><td style="border: 1px solid #ddd; padding: 8px;">Turn Right</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">3</td><td style="border: 1px solid #ddd; padding: 8px;"><code>turn -1</code></td><td style="border: 1px solid #ddd; padding: 8px;">Turn Left</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">4</td><td style="border: 1px solid #ddd; padding: 8px;"><code>attack 1</code></td><td style="border: 1px solid #ddd; padding: 8px;">Attack/Break Block</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">5</td><td style="border: 1px solid #ddd; padding: 8px;"><code>attack 0</code></td><td style="border: 1px solid #ddd; padding: 8px;">Stop Attack</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">6</td><td style="border: 1px solid #ddd; padding: 8px;"><code>use 1</code></td><td style="border: 1px solid #ddd; padding: 8px;">Use/Place</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">7</td><td style="border: 1px solid #ddd; padding: 8px;"><code>use 0</code></td><td style="border: 1px solid #ddd; padding: 8px;">Stop Use</td></tr>
</table>

<h4>Keyboard Controls</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f5f5f5;">
        <th style="border: 1px solid #ddd; padding: 6px;">Key</th>
        <th style="border: 1px solid #ddd; padding: 6px;">Action</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;"><kbd>W</kbd></td><td style="border: 1px solid #ddd; padding: 6px;">Move Forward (0)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;"><kbd>S</kbd></td><td style="border: 1px solid #ddd; padding: 6px;">Move Backward (1)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;"><kbd>A</kbd> / Mouse Left</td><td style="border: 1px solid #ddd; padding: 6px;">Turn Left (3)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;"><kbd>D</kbd> / Mouse Right</td><td style="border: 1px solid #ddd; padding: 6px;">Turn Right (2)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;"><kbd>Space</kbd></td><td style="border: 1px solid #ddd; padding: 6px;">Attack (4)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;"><kbd>E</kbd></td><td style="border: 1px solid #ddd; padding: 6px;">Use (6)</td></tr>
</table>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/microsoft/malmo" target="_blank">Project Malmo (Microsoft Research)</a></li>
    <li><a href="https://github.com/microsoft/malmo/tree/master/MalmoEnv" target="_blank">MalmoEnv Python package</a></li>
</ul>
"""


MALMOENV_MAZERUNNER_HTML = get_mazerunner_html("MalmoEnv-MazeRunner-v0")

__all__ = ["MALMOENV_MAZERUNNER_HTML", "get_mazerunner_html"]
