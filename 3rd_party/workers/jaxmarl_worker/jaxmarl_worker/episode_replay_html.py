"""RL episode debug viewer -- jaxmarl_worker edition.

Generates a self-contained dark-theme HTML file from the _debug.jsonl schema
produced by smoke_rl_debug.py and smoke_rl_mappo_debug.py.

Per-step fields used from the JSONL:
  episode, step, rewards, dones, truncated
  agents.<id>.obs_long_term   -- decoded obs text
  agents.<id>.image_base64    -- PNG frame (base64)
  agents.<id>.parsed_action   -- action name string
  agents.<id>.reasoning       -- entropy / action-prob table

Episode summary fields used from episode.json:
  episode_return, num_steps, num_episodes, num_agents, task, ckpt, role, variant
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """\
:root {
  --bg: #0d1117; --bg2: #161b22; --bg3: #21262d;
  --border: #30363d; --text: #e6edf3; --text-dim: #8b949e;
  --green: #3fb950; --red: #f85149; --blue: #58a6ff;
  --yellow: #d29922; --purple: #bc8cff;
  --agent0: #58a6ff; --agent1: #3fb950; --agent2: #d29922; --agent3: #bc8cff;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: 'SF Mono', 'Fira Code', monospace;
       font-size: 13px; line-height: 1.5; }
#header { background: var(--bg2); border-bottom: 1px solid var(--border);
          padding: 10px 16px; display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
#title { font-size: 15px; font-weight: 600; color: var(--text); }
#metaInfo { font-size: 11px; color: var(--text-dim); }
#summaryBar { display: flex; flex-wrap: wrap; gap: 12px; padding: 10px 16px;
              background: var(--bg2); border-bottom: 1px solid var(--border); }
.stat { display: flex; flex-direction: column; }
.stat-label { font-size: 10px; color: var(--text-dim); }
.stat-value { font-size: 13px; font-weight: 600; color: var(--text); }
#controls { display: flex; align-items: center; gap: 12px; padding: 8px 16px;
            background: var(--bg3); border-bottom: 1px solid var(--border); flex-wrap: wrap;
            position: sticky; top: 0; z-index: 100; }
.ctrl-group { display: flex; align-items: center; gap: 8px; }
label { display: flex; align-items: center; gap: 4px; cursor: pointer; color: var(--text-dim); font-size: 12px; }
input[type=checkbox] { accent-color: var(--blue); }
input[type=range] { accent-color: var(--blue); width: 180px; }
#stepLabel { font-size: 12px; color: var(--text-dim); min-width: 60px; }
.ep-row { display: flex; align-items: center; gap: 6px; }
.ep-select { background: var(--bg3); border: 1px solid var(--border); color: var(--text);
             border-radius: 4px; padding: 2px 6px; font-size: 12px; font-family: inherit; }
.ep-badge-timeout    { display: inline-block; font-size: 10px; padding: 1px 5px; border-radius: 3px;
                       background: rgba(248,81,73,0.2); color: var(--red); font-weight: 600; }
.ep-badge-truncation { display: inline-block; font-size: 10px; padding: 1px 5px; border-radius: 3px;
                       background: rgba(63,185,80,0.25); color: var(--green); font-weight: 600; }
#stepsContainer { padding: 12px 16px; display: flex; flex-direction: column; gap: 12px; }
.step-card { background: var(--bg2); border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
.step-header { display: flex; align-items: center; justify-content: space-between;
               padding: 6px 12px; background: var(--bg3); border-bottom: 1px solid var(--border); }
.step-num { font-weight: 600; font-size: 13px; }
.rewards { font-size: 12px; display: flex; gap: 8px; }
.reward-pos  { color: var(--green); }
.reward-neg  { color: var(--red); }
.reward-zero { color: var(--text-dim); }
.agents-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1px;
               background: var(--border); }
.agent-col { background: var(--bg2); padding: 10px 12px; display: flex; flex-direction: column; gap: 8px; }
.agent-label { font-size: 11px; font-weight: 700; padding: 2px 6px; border-radius: 3px;
               background: var(--bg3); display: inline-block; }
.agent-label-0 { color: var(--agent0); border: 1px solid var(--agent0); }
.agent-label-1 { color: var(--agent1); border: 1px solid var(--agent1); }
.agent-label-2 { color: var(--agent2); border: 1px solid var(--agent2); }
.agent-label-3 { color: var(--agent3); border: 1px solid var(--agent3); }
.field { display: flex; flex-direction: column; gap: 4px; }
.field-label { font-size: 10px; font-weight: 600; color: var(--text-dim); text-transform: uppercase;
               letter-spacing: 0.04em; }
.image-field img { max-width: 100%; border-radius: 4px; border: 1px solid var(--border); }
.action-row { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.action-name { font-weight: 700; color: var(--blue); font-size: 13px; }
.reasoning-block { font-size: 11px; white-space: pre; color: var(--text-dim);
                   background: var(--bg3); border-radius: 4px; padding: 6px 8px;
                   overflow-x: auto; max-height: 220px; overflow-y: auto; }
.obs-block { font-size: 11px; white-space: pre-wrap; color: var(--text-dim);
             background: var(--bg3); border-radius: 4px; padding: 6px 8px;
             overflow: hidden; max-height: 80px; }
.obs-block.expanded { max-height: none; }
.expand-btn { font-size: 9px; padding: 1px 5px; border-radius: 3px;
              background: var(--bg3); border: 1px solid var(--border); color: var(--text-dim);
              cursor: pointer; font-family: inherit; }
.expand-btn:hover { border-color: var(--blue); color: var(--blue); }
.nav-btn { background: var(--bg3); border: 1px solid var(--border); color: var(--text);
           border-radius: 4px; padding: 3px 10px; font-size: 14px; cursor: pointer;
           font-family: inherit; transition: border-color 0.1s, color 0.1s; }
.nav-btn:hover:not(:disabled) { border-color: var(--blue); color: var(--blue); }
.nav-btn:disabled { opacity: 0.3; cursor: default; }
.mode-btn { background: var(--bg3); border: 1px solid var(--border); color: var(--text-dim);
            border-radius: 4px; padding: 3px 8px; font-size: 11px; cursor: pointer;
            font-family: inherit; }
.mode-btn.active { border-color: var(--blue); color: var(--blue); }
"""

# ---------------------------------------------------------------------------
# Body HTML (static scaffold)
# ---------------------------------------------------------------------------

_BODY_HTML = """\
<div id="header">
  <span id="title">RL Debug Viewer</span>
  <span id="metaInfo"></span>
</div>
<div id="summaryBar"></div>
<div id="controls">
  <div class="ctrl-group">
    <label><input type="checkbox" id="showObs" checked> Obs</label>
    <label><input type="checkbox" id="showReasoning" checked> Reasoning</label>
    <label><input type="checkbox" id="showImages" checked> Images</label>
  </div>
  <div class="ctrl-group">
    <button class="nav-btn" id="prevBtn" onclick="stepBy(-1)" disabled title="Previous step (←)">&#9664;</button>
    <button class="nav-btn" id="nextBtn" onclick="stepBy(1)" title="Next step (→)">&#9654;</button>
    <input type="range" id="stepSlider" min="0" value="0">
    <span id="stepLabel">Step 0/0</span>
    <button class="mode-btn" id="modeBtn" onclick="toggleMode()" title="Toggle all-steps / single-step view">All steps</button>
  </div>
  <div class="ctrl-group ep-row" id="episodeRow" style="display:none"></div>
</div>
<div id="stepsContainer"></div>
"""

# ---------------------------------------------------------------------------
# JavaScript
# ---------------------------------------------------------------------------

_JS = """\
// Group steps by episode
var EPISODE_STEPS = {};
var EPISODE_LIST = [];
STEPS.forEach(function(s) {
  var ep = s.episode != null ? s.episode : 0;
  if (!EPISODE_STEPS[ep]) { EPISODE_STEPS[ep] = []; EPISODE_LIST.push(ep); }
  EPISODE_STEPS[ep].push(s);
});
EPISODE_LIST.sort(function(a, b) { return a - b; });
var currentEpisode = EPISODE_LIST.length ? EPISODE_LIST[0] : 0;
var filteredSteps = EPISODE_STEPS[currentEpisode] || STEPS;

var currentStep = 0;

var _numAgents = Object.keys(STEPS[0] && STEPS[0].agents || {}).length;
document.getElementById("metaInfo").textContent =
  "agents=" + _numAgents + "  |  episodes=" + EPISODE_LIST.length;

// Episode dropdown
function buildEpisodeRow() {
  var epRow = document.getElementById("episodeRow");
  if (EPISODE_LIST.length <= 1) { epRow.style.display = "none"; return; }
  epRow.style.display = "";
  var options = EPISODE_LIST.map(function(ep) {
    var epSteps = EPISODE_STEPS[ep];
    var lastS = epSteps[epSteps.length - 1];
    var isTruncated  = lastS.truncated && lastS.truncated.some(function(v) { return v; });
    var isTerminated = !isTruncated && lastS.dones && lastS.dones.some(function(v) { return v; });
    var label = "Ep " + ep + (isTruncated ? " [SCORED]" : isTerminated ? " [TIMEOUT]" : "");
    return '<option value="' + ep + '"' + (ep === currentEpisode ? ' selected' : '') + '>' + label + '</option>';
  }).join("");
  epRow.innerHTML = '<span style="color:var(--text-dim);font-size:12px">Episode:</span>' +
    '<select class="ep-select" id="episodeSelect" onchange="setEpisode(parseInt(this.value))">' + options + '</select>';
}
buildEpisodeRow();

// Summary bar
if (SUMMARY) {
  var bar = document.getElementById("summaryBar");
  var numAgents = SUMMARY.num_agents || _numAgents;
  var stats = [
    ["Steps", SUMMARY.num_steps],
    ["Episodes", SUMMARY.num_episodes || EPISODE_LIST.length],
    ["Episode Return", (SUMMARY.episode_return || 0).toFixed(2)],
  ];
  for (var i = 0; i < numAgents; i++) {
    stats.push(["Agent " + i + " Return", (SUMMARY["agent_" + i + "_return"] || 0).toFixed(2)]);
  }
  if (SUMMARY.task) stats.push(["Task", SUMMARY.task]);
  if (SUMMARY.variant) stats.push(["Variant", SUMMARY.variant]);
  bar.innerHTML = stats.map(function(s) {
    return '<div class="stat"><span class="stat-label">' + s[0] + '</span><span class="stat-value">' + s[1] + '</span></div>';
  }).join("");
}

function toggleExpand(btn) {
  var block = btn.parentElement.nextElementSibling;
  block.classList.toggle("expanded");
  btn.textContent = block.classList.contains("expanded") ? "collapse" : "expand";
}

function escHtml(s) {
  if (s == null) return '<span style="color:var(--text-dim)">null</span>';
  var div = document.createElement("div");
  div.textContent = String(s);
  return div.innerHTML;
}

function rewardClass(r) {
  if (r > 0) return "reward-pos";
  if (r < 0) return "reward-neg";
  return "reward-zero";
}

function renderStep(step, idx) {
  var agents = step.agents || {};
  var agentIds = Object.keys(agents).sort();
  var rewards = step.rewards || [];

  var rewardsHtml = rewards.map(function(r, i) {
    return '<span class="' + rewardClass(r) + '">A' + i + ":" + r.toFixed(3) + "</span>";
  }).join(" ");

  var showObs = document.getElementById("showObs").checked;
  var showReasoning = document.getElementById("showReasoning").checked;
  var showImages = document.getElementById("showImages").checked;

  var agentsHtml = agentIds.map(function(aid) {
    var a = agents[aid];
    var labelClass = "agent-label agent-label-" + aid;
    var s = "";

    // Pixel frame
    if (showImages && a.image_base64) {
      s += '<div class="field image-field"><div class="field-label">Frame</div>';
      s += '<div><img src="data:image/png;base64,' + a.image_base64 + '" alt="frame"></div></div>';
    }

    // Action
    s += '<div class="field"><div class="field-label">Action</div>';
    s += '<div class="action-row"><span class="action-name">' + escHtml(a.parsed_action) + '</span></div></div>';

    // Reasoning (probs + entropy)
    if (showReasoning && a.reasoning) {
      s += '<div class="field"><div class="field-label">Reasoning</div>';
      s += '<div class="reasoning-block">' + escHtml(a.reasoning) + "</div></div>";
    }

    // Observation text
    if (showObs && a.obs_long_term) {
      s += '<div class="field obs-field"><div class="field-label">Observation <button class="expand-btn" onclick="toggleExpand(this)">expand</button></div>';
      s += '<div class="obs-block">' + escHtml(a.obs_long_term) + "</div></div>";
    }

    return '<div class="agent-col"><span class="' + labelClass + '">Agent ' + aid + "</span>" + s + "</div>";
  }).join("");

  var outcomeBadge = "";
  if (idx === filteredSteps.length - 1) {
    var _isTruncated  = step.truncated && step.truncated.some(function(v) { return v; });
    var _isTerminated = !_isTruncated && step.dones && step.dones.some(function(v) { return v; });
    if (_isTruncated) outcomeBadge = ' <span class="ep-badge-truncation">SCORED</span>';
    else if (_isTerminated) outcomeBadge = ' <span class="ep-badge-timeout">TIMEOUT</span>';
  }

  return '<div class="step-card" id="step-' + idx + '">' +
    '<div class="step-header"><span class="step-num">Step ' + step.step + outcomeBadge + '</span>' +
    '<span class="rewards">' + rewardsHtml + "</span></div>" +
    '<div class="agents-grid">' + agentsHtml + "</div></div>";
}

var _mode = "all";

function renderAll() {
  var container = document.getElementById("stepsContainer");
  container.innerHTML = filteredSteps.map(function(s, i) { return renderStep(s, i); }).join("");
  document.getElementById("stepSlider").max = filteredSteps.length - 1;
  document.getElementById("stepSlider").value = currentStep;
  document.getElementById("stepLabel").textContent = "Step " + (currentStep + 1) + "/" + filteredSteps.length;
}

function renderSingle(idx) {
  var container = document.getElementById("stepsContainer");
  container.innerHTML = renderStep(filteredSteps[idx], idx);
  document.getElementById("stepLabel").textContent = "Step " + (idx + 1) + "/" + filteredSteps.length;
}

function applyFilters() {
  if (_mode === "all") renderAll();
  else renderSingle(currentStep);
}

document.getElementById("stepSlider").addEventListener("input", function() {
  currentStep = parseInt(this.value);
  if (_mode !== "all") renderSingle(currentStep);
  else {
    var el = document.getElementById("step-" + currentStep);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
  }
});

document.getElementById("showObs").addEventListener("change", applyFilters);
document.getElementById("showReasoning").addEventListener("change", applyFilters);
document.getElementById("showImages").addEventListener("change", applyFilters);

function updateNavButtons() {
  var prev = document.getElementById("prevBtn");
  var next = document.getElementById("nextBtn");
  if (prev) prev.disabled = (currentStep === 0);
  if (next) next.disabled = (currentStep === filteredSteps.length - 1);
}

function stepBy(delta) {
  var newStep = Math.max(0, Math.min(filteredSteps.length - 1, currentStep + delta));
  currentStep = newStep;
  document.getElementById("stepSlider").value = currentStep;
  if (_mode === "all") {
    document.getElementById("stepLabel").textContent = "Step " + (currentStep + 1) + "/" + filteredSteps.length;
    var el = document.getElementById("step-" + currentStep);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
  } else {
    renderSingle(currentStep);
  }
  updateNavButtons();
}

function toggleMode() {
  var btn = document.getElementById("modeBtn");
  if (_mode === "all") {
    _mode = "single";
    btn.textContent = "Single step";
    btn.classList.add("active");
    renderSingle(currentStep);
  } else {
    _mode = "all";
    btn.textContent = "All steps";
    btn.classList.remove("active");
    renderAll();
  }
  updateNavButtons();
}

document.addEventListener("keydown", function(e) {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.key === "ArrowLeft")  { e.preventDefault(); stepBy(-1); }
  if (e.key === "ArrowRight") { e.preventDefault(); stepBy(1); }
});

function setEpisode(ep) {
  currentEpisode = ep;
  filteredSteps = EPISODE_STEPS[ep] || STEPS;
  currentStep = 0;
  buildEpisodeRow();
  applyFilters();
  updateNavButtons();
}

renderAll();
updateNavButtons();
"""

# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _build_html(title: str, steps_json: str, summary_json: str, color_css: str = "") -> str:
    import html as _html

    extra_style = ("\n<style>\n" + color_css + "</style>\n") if color_css else ""
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        "<title>RL Debug: " + _html.escape(title) + "</title>\n"
        "<style>\n" + _CSS + "</style>\n"
        + extra_style +
        "</head>\n"
        "<body>\n" + _BODY_HTML + "<script>\n"
        "var STEPS = " + steps_json + ";\n"
        "var SUMMARY = " + summary_json + ";\n"
        + _JS + "</script>\n"
        "</body>\n"
        "</html>"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_rl_debug_html(
    debug_jsonl_path: str | Path,
    episode_json_path: str | Path | None = None,
    output_path: str | Path | None = None,
    agent_colors: list[str] | None = None,
) -> Path:
    """Read a _debug.jsonl + optional episode .json, write an RL debug HTML file.

    The JSONL schema is the same as smoke_rl_debug.py / smoke_rl_mappo_debug.py:
    one record per step with keys: episode, step, agents, rewards, dones, truncated.

    Per-agent cumulative returns are computed here from step rewards since
    episode.json only stores the total.

    agent_colors: optional list of CSS hex colors, one per agent index. When
    provided, overrides the default agent0/agent1/... palette so all agent
    labels share the correct team color (e.g. both green for G-2v0, both blue
    for B-0v2).
    """
    debug_jsonl_path = Path(debug_jsonl_path)

    if output_path is None:
        name = debug_jsonl_path.name
        if name.endswith("_debug.jsonl"):
            out_name = name[: -len("_debug.jsonl")] + "_debug.html"
        else:
            out_name = debug_jsonl_path.stem + "_debug.html"
        output_path = debug_jsonl_path.parent / out_name
    output_path = Path(output_path)

    steps: list[dict] = []
    with open(debug_jsonl_path, encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                steps.append(json.loads(line))
            except json.JSONDecodeError as exc:
                import logging
                logging.warning("Skipping malformed JSONL line %d in %s: %s", lineno, debug_jsonl_path, exc)

    episode_summary: dict | None = None
    if episode_json_path is not None:
        episode_json_path = Path(episode_json_path)
        if episode_json_path.exists():
            with open(episode_json_path, encoding="utf-8") as fh:
                episode_summary = json.load(fh)

    # Compute per-agent returns from step rewards
    if episode_summary is not None and steps:
        n_agents = episode_summary.get("num_agents", 0) or len(steps[0].get("rewards", []))
        agent_returns = [0.0] * n_agents
        for step in steps:
            for i, r in enumerate(step.get("rewards", [])):
                if i < n_agents:
                    agent_returns[i] += float(r)
        for i, ret in enumerate(agent_returns):
            episode_summary[f"agent_{i}_return"] = ret

    steps_json = json.dumps(steps)
    summary_json = json.dumps(episode_summary) if episode_summary is not None else "null"
    title = os.path.basename(str(debug_jsonl_path)).replace("_debug.jsonl", "")

    color_css = ""
    if agent_colors:
        vars_ = " ".join(f"--agent{i}: {c};" for i, c in enumerate(agent_colors))
        color_css = f":root {{ {vars_} }}"

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(_build_html(title, steps_json, summary_json, color_css=color_css))

    return output_path


__all__ = ["generate_rl_debug_html"]
