"""Interactive runtime for jaxmarl_worker.

JSON line protocol (stdin/stdout):
  in:  {"cmd": "reset", "seed": <int>}
  out: {"type": "ready", "observation_shape": [...], "episode_index": N, "render_payload": {...}}

  in:  {"cmd": "step"}
  out: {"type": "step",  "action": N, "reward": F, "terminated": B, "truncated": B, "obs": [...]}
       or {"type": "episode_done", "total_reward": F, ...}

  in:  {"cmd": "stop"}
  out: {"type": "stopped"}

  in:  {"cmd": "ping"}
  out: {"type": "pong"}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np

LOGGER = logging.getLogger("jaxmarl_worker.runtime")

# ---------------------------------------------------------------------------
# Sport-specific renderers — matching the MosaicMultiGrid visual style
# ---------------------------------------------------------------------------

_CELL = 40   # pixels per grid cell
_W    = 16   # grid width  (including outer walls)
_H    = 11   # grid height (including outer walls)

# Shared agent/ball colours — matches multigrid_sports/rendering/fifa.py palette
_C_TEAM     = [(30, 200, 60), (60, 80, 220)]   # team 0 = green, team 1 = blue
_C_CARRYING = (255, 255, 100)                   # yellow glow when holding ball
_C_WHITE    = (255, 255, 255)
_C_LINE     = (255, 255, 255)                   # bright white field lines

# Sport-specific surface colours — matches multigrid_sports visual style
_SPORT_SURFACE = {
    "soccer": {
        "wall":    (45, 100,  35),              # border wall cells (GRASS_WALL)
        "field_a": (76, 153,  60),              # bright grass stripe A (GRASS_LIGHT)
        "field_b": (58, 128,  45),              # grass stripe B (GRASS_DARK)
        "goal_l":  (30, 200,  60),              # green team goal
        "goal_r":  (60,  80, 220),              # blue team goal
        "ball":    (255,  60,  60),             # red football (BALL_COLOR)
    },
    "bb": {
        "wall":    (100,  70,  40),
        "field_a": (195, 145,  90),             # lighter wood
        "field_b": (175, 130,  80),             # darker wood
        "goal_l":  (30,  200,  60),
        "goal_r":  (60,   80, 220),
        "ball":    (220, 110,  30),             # orange basketball
    },
    "af": {
        "wall":    (40,  25,  10),
        "field_a": (80,  55,  30),              # brown turf A
        "field_b": (65,  43,  22),              # brown turf B
        "goal_l":  (30, 200,  60),
        "goal_r":  (60,  80, 220),
        "ball":    (180,  90,  30),             # brown football
    },
}


def _tri(cx: int, cy: int, r: int, d: int):
    """Filled triangle pointing in direction d (0=R,1=D,2=L,3=U)."""
    if d == 0: return [(cx+r, cy), (cx-r, cy-r), (cx-r, cy+r)]
    if d == 1: return [(cx, cy+r), (cx-r, cy-r), (cx+r, cy-r)]
    if d == 2: return [(cx-r, cy), (cx+r, cy-r), (cx+r, cy+r)]
    return          [(cx, cy-r), (cx-r, cy+r), (cx+r, cy+r)]


def _px(grid_x: int) -> int:
    return grid_x * _CELL

def _pxc(grid_x: int) -> int:
    """Centre pixel of a grid cell."""
    return grid_x * _CELL + _CELL // 2


def _draw_surface(draw, colors: dict) -> None:
    """Vertical grass stripes + thin black grid lines + darkened wall border."""
    # Vertical stripes across the full grid
    for gx in range(_W):
        stripe = colors["field_a"] if gx % 2 == 0 else colors["field_b"]
        draw.rectangle([_px(gx), 0, _px(gx + 1) - 1, _px(_H) - 1], fill=stripe)

    # Darken border/wall cells
    for gx in range(_W):
        for gy in range(_H):
            if gx == 0 or gx == _W - 1 or gy == 0 or gy == _H - 1:
                draw.rectangle(
                    [_px(gx), _px(gy), _px(gx + 1) - 1, _px(gy + 1) - 1],
                    fill=colors["wall"],
                )

    # Thin black grid lines inside playable area
    for gx in range(1, _W):
        draw.line([_px(gx), _px(1), _px(gx), _px(_H - 1)], fill=(0, 0, 0), width=1)
    for gy in range(1, _H):
        draw.line([_px(1), _px(gy), _px(_W - 1), _px(gy)], fill=(0, 0, 0), width=1)


def _draw_soccer_markings(draw, colors: dict) -> None:
    """White pitch lines: center circle, halfway line, penalty boxes, goal boxes."""
    lw = 2
    # Field boundary (inner side of wall)
    fl, fr = _px(1), _px(15)
    ft, fb = _px(1), _px(10)
    draw.rectangle([fl, ft, fr, fb], outline=_C_LINE, width=lw)

    # Halfway line (vertical)
    draw.line([_pxc(8), ft, _pxc(8), fb], fill=_C_LINE, width=lw)

    # Center circle (~2 cell radius)
    cr = int(2.2 * _CELL)
    cx, cy = _pxc(8), _pxc(5)
    draw.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], outline=_C_LINE, width=lw)
    draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=_C_LINE)

    # Penalty boxes (each ~3 cells from end, full height ±3)
    pen_w = int(2.5 * _CELL)
    pen_h = 6 * _CELL
    pen_t = _pxc(5) - pen_h // 2
    # Left penalty box
    draw.rectangle([fl, pen_t, fl + pen_w, pen_t + pen_h], outline=_C_LINE, width=lw)
    # Right penalty box
    draw.rectangle([fr - pen_w, pen_t, fr, pen_t + pen_h], outline=_C_LINE, width=lw)

    # Goal boxes (smaller, ±1 cell from goal row boundaries)
    gb_w = int(1.2 * _CELL)
    gb_h = 3 * _CELL
    gb_t = _pxc(5) - gb_h // 2
    draw.rectangle([fl, gb_t, fl + gb_w, gb_t + gb_h], outline=_C_LINE, width=lw)
    draw.rectangle([fr - gb_w, gb_t, fr, gb_t + gb_h], outline=_C_LINE, width=lw)

    # Goal frames (extend into wall area)
    goal_t = _px(3)
    goal_b = _px(8)
    g_depth = _CELL // 2
    # Left goal (green)
    draw.rectangle([fl - g_depth, goal_t, fl, goal_b], outline=colors["goal_l"], width=3)
    # Right goal (blue)
    draw.rectangle([fr, goal_t, fr + g_depth, goal_b], outline=colors["goal_r"], width=3)


def _draw_basketball_markings(draw, colors: dict) -> None:
    """White court lines: center circle, three-point arcs, paint areas."""
    lw = 2
    fl, fr = _px(1), _px(15)
    ft, fb = _px(1), _px(10)

    # Court boundary
    draw.rectangle([fl, ft, fr, fb], outline=_C_LINE, width=lw)

    # Center line
    draw.line([_pxc(8), ft, _pxc(8), fb], fill=_C_LINE, width=lw)

    # Center circle
    cr = int(1.8 * _CELL)
    cx, cy = _pxc(8), _pxc(5)
    draw.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], outline=_C_LINE, width=lw)

    # Paint/lane rectangles on each side
    paint_w = int(3 * _CELL)
    paint_h = int(5 * _CELL)
    pt = _pxc(5) - paint_h // 2
    draw.rectangle([fl, pt, fl + paint_w, pt + paint_h], outline=_C_LINE, width=lw)
    draw.rectangle([fr - paint_w, pt, fr, pt + paint_h], outline=_C_LINE, width=lw)

    # Free-throw circles at top of paint
    ftr = int(1.5 * _CELL)
    draw.ellipse([fl + paint_w - ftr, cy - ftr, fl + paint_w + ftr, cy + ftr], outline=_C_LINE, width=lw)
    draw.ellipse([fr - paint_w - ftr, cy - ftr, fr - paint_w + ftr, cy + ftr], outline=_C_LINE, width=lw)

    # Three-point arcs (semicircles)
    arc_r = int(3.5 * _CELL)
    draw.arc([fl - arc_r, cy - arc_r, fl + arc_r, cy + arc_r], start=270, end=90, fill=_C_LINE, width=lw)
    draw.arc([fr - arc_r, cy - arc_r, fr + arc_r, cy + arc_r], start=90, end=270, fill=_C_LINE, width=lw)

    # Basket areas (coloured rectangles at each end)
    for gx, col in [(1, colors["goal_l"]), (14, colors["goal_r"])]:
        for gy in [3, 4, 5, 6, 7]:
            draw.rectangle(
                [_px(gx) + 2, _px(gy) + 2, _px(gx + 1) - 2, _px(gy + 1) - 2],
                fill=col,
            )


def _draw_af_markings(draw, colors: dict) -> None:
    """White yard-line markings and end zones for American Football."""
    lw = 2
    fl, fr = _px(1), _px(15)
    ft, fb = _px(1), _px(10)

    # Field boundary
    draw.rectangle([fl, ft, fr, fb], outline=_C_LINE, width=lw)

    # Vertical yard lines evenly across the 14-cell playable width
    for x in range(2, 15):
        draw.line([_px(x), ft, _px(x), fb], fill=_C_LINE, width=1)

    # End zones (coloured cells at goal columns)
    for gy in [3, 4, 5, 6, 7]:
        draw.rectangle(
            [_px(1) + 2, _px(gy) + 2, _px(2) - 2, _px(gy + 1) - 2],
            fill=colors["goal_l"],
        )
        draw.rectangle(
            [_px(14) + 2, _px(gy) + 2, _px(15) - 2, _px(gy + 1) - 2],
            fill=colors["goal_r"],
        )

    # End zone boundary lines (thicker)
    draw.line([_px(2), ft, _px(2), fb], fill=_C_LINE, width=3)
    draw.line([_px(14), ft, _px(14), fb], fill=_C_LINE, width=3)

    # Center line
    draw.line([_pxc(8), ft, _pxc(8), fb], fill=_C_LINE, width=2)


def _draw_agents_and_ball(draw, state, ball_color: tuple) -> None:
    """Draw agents as directional triangles and ball as a circle (sport-agnostic)."""
    agent_pos      = np.array(state.agent_pos)
    agent_dir      = np.array(state.agent_dir)
    agent_team     = np.array(state.agent_team)
    agent_carrying = np.array(state.agent_carrying)
    ball_pos       = np.array(state.ball_pos)
    ball_carried   = np.array(state.ball_carried_by)

    # Loose ball
    for b in range(ball_pos.shape[0]):
        if ball_carried[b] < 0:
            bx, by = int(ball_pos[b, 0]), int(ball_pos[b, 1])
            r = _CELL // 5
            cx, cy = _pxc(bx), _pxc(by)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=ball_color, outline=_C_WHITE)

    # Agents
    for i in range(agent_pos.shape[0]):
        ax, ay  = int(agent_pos[i, 0]), int(agent_pos[i, 1])
        team    = int(agent_team[i])
        color   = _C_TEAM[team] if team < 2 else (200, 200, 0)
        d       = int(agent_dir[i]) % 4
        cx, cy  = _pxc(ax), _pxc(ay)
        tri_r   = _CELL // 2 - 5
        draw.polygon(_tri(cx, cy, tri_r, d), fill=color, outline=_C_WHITE)

        # Ball-on-carrier indicator (small circle in centre of triangle)
        if int(agent_carrying[i]) >= 0:
            br = max(3, _CELL // 7)
            draw.ellipse([cx - br, cy - br, cx + br, cy + br], fill=_C_CARRYING)

    # Score overlay
    scores = np.array(state.scores)
    step_n = int(state.step)
    draw.rectangle([0, 0, 160, 18], fill=(0, 0, 0, 180))
    draw.text((4, 2), f"G:{int(scores[0])}  B:{int(scores[1])}  step:{step_n}", fill=_C_WHITE)


def _draw_fov(img, state, view_size: int = 7):
    """RGBA-composited per-agent field-of-view rectangles (matches render_fifa FOV style)."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return img

    agent_pos  = np.array(state.agent_pos)
    agent_dir  = np.array(state.agent_dir)
    agent_team = np.array(state.agent_team)
    n_agents   = agent_pos.shape[0]
    vs         = view_size

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    odraw   = ImageDraw.Draw(overlay)

    for i in range(n_agents):
        ax, ay    = int(agent_pos[i, 0]), int(agent_pos[i, 1])
        direction = int(agent_dir[i]) % 4
        team      = int(agent_team[i])
        color     = _C_TEAM[team] if team < 2 else (200, 200, 0)

        # View top-left matches get_view_exts() in multigrid_sports obs.py
        if direction == 0:    # RIGHT
            tx, ty = ax, ay - vs // 2
        elif direction == 1:  # DOWN
            tx, ty = ax - vs // 2, ay
        elif direction == 2:  # LEFT
            tx, ty = ax - vs + 1, ay - vs // 2
        else:                 # UP
            tx, ty = ax - vs // 2, ay - vs + 1

        x0 = max(0, tx);           y0 = max(0, ty)
        x1 = min(_W - 1, tx + vs - 1); y1 = min(_H - 1, ty + vs - 1)

        px0, py0 = _px(x0), _px(y0)
        px1, py1 = _px(x1 + 1) - 1, _px(y1 + 1) - 1

        odraw.rectangle([px0, py0, px1, py1], fill=(*color, 35))
        odraw.rectangle([px0, py0, px1, py1], outline=(*color, 160), width=2)

    return Image.alpha_composite(img.convert("RGBA"), overlay)


def _draw_agent_labels(draw, state) -> None:
    """Agent index numbers drawn over each triangle."""
    try:
        from PIL import ImageFont
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
            max(9, _CELL // 3),
        )
    except Exception:
        from PIL import ImageFont
        font = ImageFont.load_default()

    agent_pos = np.array(state.agent_pos)
    for i in range(agent_pos.shape[0]):
        ax, ay = int(agent_pos[i, 0]), int(agent_pos[i, 1])
        cx = _pxc(ax) - _CELL // 8
        cy = _pxc(ay) - _CELL // 5
        draw.text((cx, cy), str(i), fill=_C_WHITE, font=font)


def _render_state(state, sport: str = "soccer", view_size: int = 7) -> np.ndarray:
    """Render any JAX sport state as an RGB array matching the MosaicMultiGrid visual."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return np.zeros((_H * _CELL, _W * _CELL, 3), dtype=np.uint8)

    colors  = _SPORT_SURFACE.get(sport, _SPORT_SURFACE["soccer"])
    W_px, H_px = _W * _CELL, _H * _CELL
    img     = Image.new("RGB", (W_px, H_px), color=colors["wall"])
    draw    = ImageDraw.Draw(img)

    _draw_surface(draw, colors)

    if sport == "soccer":
        _draw_soccer_markings(draw, colors)
    elif sport == "bb":
        _draw_basketball_markings(draw, colors)
    else:
        _draw_af_markings(draw, colors)

    # FOV overlays (RGBA composite)
    img  = _draw_fov(img, state, view_size).convert("RGB")
    draw = ImageDraw.Draw(img)

    _draw_agents_and_ball(draw, state, colors["ball"])
    _draw_agent_labels(draw, state)
    return np.array(img)




def _frame_to_render_payload(rgb: np.ndarray) -> Dict[str, Any]:
    # Emit the same format as xuance_worker so the render_container's
    # _detect_render_mode() sees "rgb" → RenderMode.RGB_ARRAY.
    h, w = rgb.shape[:2]
    return {
        "mode":   "rgb_array",
        "rgb":    rgb.tolist(),
        "width":  w,
        "height": h,
    }


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load_actor_params(path: Path, obs_dim: int, hidden_dim: int = 256, action_dim: int = 8) -> Dict:
    """Reconstruct Flax Actor params from a .npz checkpoint.

    Flax tree_flatten alphabetises keys: bias < kernel, Dense_0 < Dense_1 < Dense_2.
    So layout in the file is: arr_0=D0/bias, arr_1=D0/kernel, ..., arr_4=D2/bias, arr_5=D2/kernel.
    """
    data = np.load(str(path))
    # Sort numerically (not lexicographically) to avoid arr_10 < arr_2 pitfall
    keys = sorted(data.files, key=lambda k: int(k.split("_")[1]))
    if len(keys) < 6:
        raise ValueError(f"Expected ≥6 arrays in checkpoint, got {len(keys)}: {path}")

    def _j(k):
        return jnp.array(data[k])

    params = {
        "params": {
            "Dense_0": {"bias": _j(keys[0]), "kernel": _j(keys[1])},
            "Dense_1": {"bias": _j(keys[2]), "kernel": _j(keys[3])},
            "Dense_2": {"bias": _j(keys[4]), "kernel": _j(keys[5])},
        }
    }

    # Sanity-check shapes
    k0 = params["params"]["Dense_0"]["kernel"]
    if k0.shape[0] != obs_dim:
        raise ValueError(
            f"Checkpoint obs_dim mismatch: kernel shape {k0.shape}, expected first dim={obs_dim}"
        )
    return params


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def _make_env(env_id: str, view_size: int = 7):
    """Construct the JAX env from a full env_id string. Returns (env, sport_key)."""
    from jaxmarl_worker.environments.soccer_jax import VARIANT_MAP as _S_VARIANTS, make_soccer_jax
    if env_id in _S_VARIANTS:
        return make_soccer_jax(env_id, view_size=view_size), "soccer"

    try:
        from jaxmarl_worker.environments.american_football_jax import VARIANT_MAP as _AF_VARIANTS, make_af_jax
        if env_id in _AF_VARIANTS:
            return make_af_jax(env_id, view_size=view_size), "af"
    except Exception:
        pass

    try:
        from jaxmarl_worker.environments.basketball_jax import VARIANT_MAP as _BB_VARIANTS, make_bb_jax
        if env_id in _BB_VARIANTS:
            return make_bb_jax(env_id, view_size=view_size), "bb"
    except Exception:
        pass

    # SocialJax environments — accepts bare "coin_game" or prefixed "socialjax/coin_game"
    _SJ_ENVS = {
        "coin_game", "harvest_common_open", "coop_mining",
        "territory_open", "pd_arena", "mushrooms", "gift", "lb_foraging",
    }
    sj_task = env_id.split("/")[-1]
    if sj_task in _SJ_ENVS:
        from jaxmarl_worker.environments.socialjax.generic import SocialJaxGenericWrapper
        return SocialJaxGenericWrapper(sj_task), "socialjax"

    raise ValueError(f"Unknown env_id for jaxmarl_worker: {env_id!r}")


# ---------------------------------------------------------------------------
# Interactive runtime
# ---------------------------------------------------------------------------

class InteractiveRuntime:
    """JSON line protocol runner for jaxmarl_worker."""

    def __init__(self, env_id: str, policy_path: str, view_size: int = 7):
        self._env_id      = env_id
        self._view_size   = view_size
        self._episode_idx = 0
        self._episode_rew = 0.0
        self._step_idx    = 0

        LOGGER.info("Loading env %s (view_size=%d)", env_id, view_size)
        self._env, self._sport = _make_env(env_id, view_size)
        self._n_agents   = self._env.num_agents
        self._obs_dim    = self._env._obs_dim
        self._action_dim = getattr(self._env, "action_dim", 8)
        self._agents     = [f"agent_{i}" for i in range(self._n_agents)]

        LOGGER.info("Loading actor checkpoint: %s", policy_path)
        actor_params = _load_actor_params(
            Path(policy_path),
            obs_dim=self._obs_dim,
        )

        # Build actor and JIT-compile apply
        import flax.linen as nn
        from flax.linen.initializers import constant, orthogonal

        class Actor(nn.Module):
            action_dim: int
            hidden_dim: int = 256
            @nn.compact
            def __call__(self, x):
                x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
                x = nn.tanh(x)
                x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
                x = nn.tanh(x)
                return nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)

        self._actor        = Actor(action_dim=self._action_dim)
        self._actor_params = actor_params
        self._apply_fn     = jax.jit(self._actor.apply)

        # Pre-warm actor JIT
        dummy = jnp.zeros((1, self._obs_dim))
        _ = self._apply_fn(self._actor_params, dummy)

        # Pre-warm env JIT (reset + step_env) so the first command is fast
        LOGGER.info("Pre-warming env JIT (reset + step) …")
        _wk = jax.random.PRNGKey(0)
        _wk, _rk = jax.random.split(_wk)
        _obs0, _st0 = self._env.reset(_rk)
        _dummy_actions = {f"agent_{i}": jnp.int32(0) for i in range(self._n_agents)}
        _, _, _, _, _ = self._env.step_env(_wk, _st0, _dummy_actions)
        LOGGER.info("JaxMARL runtime ready — n_agents=%d obs_dim=%d action_dim=%d",
                    self._n_agents, self._obs_dim, self._action_dim)

    # -----------------------------------------------------------------------

    def _emit(self, obj: Dict[str, Any]) -> None:
        line = json.dumps(obj, allow_nan=False)
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def _select_actions(self, obs_dict: Dict, key) -> Dict[str, int]:
        """Sample action for each agent from the softmax distribution."""
        actions = {}
        for agent in self._agents:
            obs_arr = jnp.array(obs_dict[agent])[None, :]  # (1, obs_dim)
            logits  = self._apply_fn(self._actor_params, obs_arr)[0]  # (action_dim,)
            key, subkey = jax.random.split(key)
            action  = int(jax.random.categorical(subkey, logits))
            actions[agent] = action
        return actions

    # -----------------------------------------------------------------------

    def _handle_reset(self, cmd: Dict) -> None:
        seed = int(cmd.get("seed", 0))
        key  = jax.random.PRNGKey(seed)

        self._key, rk = jax.random.split(key)
        obs, self._state = self._env.reset(rk)
        self._obs          = obs
        self._episode_rew  = 0.0
        self._step_idx     = 0
        self._episode_idx += 1

        # Render -- non-critical: a failure here skips the frame but keeps the session alive
        render_payload = None
        try:
            if self._sport == "socialjax":
                img = self._env._sj.render(self._state)
                rgb = np.array(img)
                if rgb.dtype != np.uint8:
                    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            else:
                rgb = _render_state(self._state, self._sport, self._view_size)
            render_payload = _frame_to_render_payload(rgb)
        except Exception as _render_exc:
            LOGGER.warning("Render failed for %s (%s): %s", self._env_id, self._sport, _render_exc)

        resp: Dict[str, Any] = {
            "type":              "ready",
            "env_id":            self._env_id,
            "seed":              seed,
            "observation_shape": [self._n_agents, self._obs_dim],
            "episode_index":     self._episode_idx,
            "step_index":        0,
            "episode_reward":    0.0,
        }
        if render_payload is not None:
            resp["render_payload"] = render_payload
        self._emit(resp)

    def _handle_step(self) -> None:
        if self._obs is None:
            self._emit({"type": "error", "message": "Not initialized — send reset first."})
            return

        self._key, step_key = jax.random.split(self._key)
        actions = self._select_actions(self._obs, step_key)

        obs_new, self._state, rewards, dones, _ = self._env.step_env(
            step_key, self._state, actions
        )
        self._obs = obs_new

        reward_scalar = float(rewards.get("agent_0", list(rewards.values())[0]))
        self._episode_rew += reward_scalar
        self._step_idx    += 1

        done = bool(dones.get("__all__", False))
        primary_action = actions.get("agent_0", actions[self._agents[0]])

        # obs for reporting: agent_0's flat obs
        obs_report = np.array(self._obs.get("agent_0", self._obs[self._agents[0]])).tolist()

        # Render -- non-critical: a failure here skips the frame but keeps the session alive
        render_payload = None
        try:
            if self._sport == "socialjax":
                img = self._env._sj.render(self._state)
                rgb = np.array(img)
                if rgb.dtype != np.uint8:
                    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            else:
                rgb = _render_state(self._state, self._sport, self._view_size)
            render_payload = _frame_to_render_payload(rgb)
        except Exception as _render_exc:
            LOGGER.warning("Render failed for %s (%s): %s", self._env_id, self._sport, _render_exc)

        if done:
            resp: Dict[str, Any] = {
                "type":          "episode_done",
                "action":        primary_action,
                "reward":        reward_scalar,
                "total_reward":  self._episode_rew,
                "terminated":    True,
                "truncated":     False,
                "step_index":    self._step_idx,
                "obs":           obs_report,
            }
        else:
            resp = {
                "type":       "step",
                "action":     primary_action,
                "reward":     reward_scalar,
                "terminated": False,
                "truncated":  False,
                "step_index": self._step_idx,
                "obs":        obs_report,
            }

        if render_payload is not None:
            resp["render_payload"] = render_payload

        self._emit(resp)

    def _handle_stop(self) -> None:
        self._emit({"type": "stopped"})
        sys.exit(0)

    # -----------------------------------------------------------------------

    def run(self) -> None:
        """Main command loop — reads JSON lines from stdin."""
        # Reconfigure stdout/stdin for unbuffered line operation on pipes.
        # Python defaults to block-buffering for non-tty streams, which would
        # delay emit() output until the buffer fills. We want each _emit() call
        # to be visible to the parent process immediately.
        import io as _io
        sys.stdout = _io.TextIOWrapper(
            sys.stdout.buffer, line_buffering=False, write_through=True
        )
        sys.stdin = _io.TextIOWrapper(sys.stdin.buffer, line_buffering=True)

        self._obs   = None
        self._state = None
        self._key   = jax.random.PRNGKey(0)

        LOGGER.info("Waiting for commands (jaxmarl_worker interactive runtime)")

        while True:
            raw_line = sys.stdin.readline()
            if not raw_line:  # EOF
                break
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                cmd = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                self._emit({"type": "error", "message": f"JSON parse error: {exc}"})
                continue

            verb = cmd.get("cmd", "")
            try:
                if verb == "reset":
                    self._handle_reset(cmd)
                elif verb == "step":
                    self._handle_step()
                elif verb == "stop":
                    self._handle_stop()
                    return
                elif verb == "ping":
                    self._emit({"type": "pong"})
                else:
                    self._emit({"type": "error", "message": f"Unknown command: {verb!r}"})
            except Exception as exc:
                LOGGER.exception("Error handling cmd=%s", verb)
                self._emit({"type": "error", "message": str(exc)})


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="JaxMARL interactive worker runtime")
    p.add_argument("--env-id",      required=True, help="Gymnasium env ID")
    p.add_argument("--policy-path", required=True, help="Path to .npz checkpoint")
    p.add_argument("--view-size",   type=int, default=7, help="Observation view size")
    p.add_argument("--log-level",   default="INFO", help="Python logging level")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        stream=sys.stderr,
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    runtime = InteractiveRuntime(
        env_id      = args.env_id,
        policy_path = args.policy_path,
        view_size   = args.view_size,
    )
    runtime.run()


if __name__ == "__main__":
    main()
