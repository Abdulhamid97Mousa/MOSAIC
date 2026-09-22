"""CLI entry point for jaxmarl_worker.

Registered as `jaxmarl-worker` console script in pyproject.toml.

Routing:
  --interactive              JSON stdin/stdout operator runtime
  --config <path>            Training via GUI config file (MOSAIC trainer daemon)
  --config <path> --dry-run  Validate config without launching training (exits 0/1)
  --dry-run --emit-summary   Also emit a JSON summary line to stdout
  (default)                  Legacy direct-args MAPPO training
"""

import importlib
import sys

# MOSAIC MultiGrid: sport-family → per-sport subdirectory
# scan files live under algorithms/mosaic_multigrid/<SPORT_DIR>/{mappo,ippo}_scan.py
# (only coma/mat/qmix/commnet-style Phase 2 algos still use this; mappo/ippo/
#  happo/vdppo are sport-agnostic since the 2026-09-18 Hydra Phase 1 refactor)
_SPORT_TO_DIR = {"soccer": "S", "af": "AF", "bb": "BB"}

# mappo/ippo/happo/vdppo: sport is a Hydra env= config group, not a directory.
# Group names differ from the legacy sport keys for soccer only (env/s.yaml).
_ENVIRONMENT_TO_HYDRA_ENV_GROUP = {"soccer": "s", "af": "af", "bb": "bb"}

_VALID_FAMILIES = frozenset({"mosaic_multigrid", "socialjax"})

# SocialJax: (env, algo) → per-family scan-file basename
# scan files live under algorithms/socialjax/<env>/<basename>.py
_SOCIALJAX_FILE_MAP = {
    ("cleanup",     "mappo"):   "mappo_cnn_scan_socialjax",
    ("cleanup",     "ippo"):    "ippo_cnn_scan_socialjax",
    ("coins",       "mappo"):   "mappo_cnn_scan_socialjax",
    ("coins",       "ippo"):    "ippo_cnn_scan_socialjax",
    ("coins",       "mat"):     "mat_cnn_scan_socialjax",
    ("coins",       "commnet"): "commnet_cnn_scan_socialjax",
    ("coop_mining", "mappo"):   "mappo_scan",
    ("coop_mining", "ippo"):    "ippo_scan",
    ("coop_mining", "coma"):    "coma_cnn_4_agents_scan_socialjax",
}


def _resolve_scan_module(algo: str, env_family: str, environment: str | None = None) -> str:
    """Resolve the correct scan-file module path for (algo, env_family, environment).

    Args:
        algo:        Algorithm key, e.g. 'mappo', 'ippo', 'mat', 'commnet', 'coma'
        env_family:  'mosaic_multigrid' or 'socialjax'
        environment: Env within family (e.g. 'soccer' for mosaic_multigrid,
                     'cleanup' for socialjax). For backwards compatibility, if
                     env_family is a legacy sport key ('soccer'/'af'/'bb') and
                     environment is None, we treat env_family as environment
                     under an implicit mosaic_multigrid family.

    Returns:
        Dotted module path, e.g. 'jaxmarl_worker.algorithms.mosaic_multigrid.mappo_scan'
        (mosaic_multigrid mappo/ippo/happo/vdppo are sport-agnostic since the
        2026-09-18 Hydra Phase 1 refactor; sport is env=<family> at the CLI, not
        a subdirectory) or 'jaxmarl_worker.algorithms.socialjax.cleanup.mappo_cnn_scan_socialjax'.

    Raises:
        ValueError: unknown algo, env_family, or (family, env, algo) combination.
    """
    algo_lower = str(algo).lower()

    # Backwards compatibility: old callers passed env_family="soccer"/"af"/"bb"
    if env_family in _SPORT_TO_DIR and environment is None:
        environment = env_family
        env_family = "mosaic_multigrid"

    if env_family not in _VALID_FAMILIES:
        raise ValueError(
            f"Unknown env_family {env_family!r}; expected one of {sorted(_VALID_FAMILIES)}"
        )

    if env_family == "mosaic_multigrid":
        if environment not in _SPORT_TO_DIR:
            raise ValueError(
                f"Unknown mosaic_multigrid environment {environment!r}; "
                f"expected one of {sorted(_SPORT_TO_DIR)}"
            )
        # mappo/ippo/happo/vdppo are sport-agnostic since the 2026-09-18 Hydra
        # Phase 1 refactor — no sport subdirectory, sport is env=<family> instead.
        if algo_lower in ("mappo", "ippo", "happo", "vdppo"):
            return f"jaxmarl_worker.algorithms.mosaic_multigrid.{algo_lower}_scan"
        raise ValueError(
            f"Unknown algo {algo!r} for mosaic_multigrid; "
            f"expected one of 'mappo', 'ippo', 'happo', 'vdppo'"
        )

    # env_family == "socialjax"
    if environment is None:
        raise ValueError("socialjax dispatch requires an environment (cleanup|coins|coop_mining)")
    basename = _SOCIALJAX_FILE_MAP.get((environment, algo_lower))
    if basename is None:
        valid = sorted({(e, a) for (e, a) in _SOCIALJAX_FILE_MAP})
        raise ValueError(
            f"No socialjax scan file for (env={environment!r}, algo={algo!r}). "
            f"Valid (env, algo) pairs: {valid}"
        )
    return f"jaxmarl_worker.algorithms.socialjax.{environment}.{basename}"


def _run_from_config(config_path: str) -> None:
    """Load a MOSAIC trainer config JSON and dispatch to the right algorithm."""
    import json
    import os
    from pathlib import Path

    data = json.loads(Path(config_path).read_text())

    # Support both nested GUI format and flat format
    if "metadata" in data and "worker" in data.get("metadata", {}):
        cfg = data["metadata"]["worker"].get("config", data)
    else:
        cfg = data

    algo        = cfg.get("algo", "mappo").lower()
    env_family  = cfg.get("env_family", "mosaic_multigrid")
    # For back-compat: if env_family looks like a legacy sport, promote it.
    if env_family in _SPORT_TO_DIR:
        environment = env_family
        env_family = "mosaic_multigrid"
    else:
        environment = cfg.get("environment", "soccer")
    variant     = cfg.get("variant", "2v2")
    run_id      = cfg.get("run_id", "jaxmarl_run")
    _mosaic_root = Path(__file__).resolve().parents[2]
    run_dir     = cfg.get("run_dir") or str(_mosaic_root / "var" / "trainer" / "runs" / run_id)
    total_upd   = str(cfg.get("total_updates", 2000))
    n_envs      = str(cfg.get("n_envs", 512))
    n_steps     = str(cfg.get("n_steps", 128))
    n_epochs    = str(cfg.get("n_epochs", 4))
    n_mb        = str(cfg.get("n_minibatches", 4))
    hidden_dim  = str(cfg.get("hidden_dim", 256))
    lr          = str(cfg.get("lr", 3e-4))
    gamma       = str(cfg.get("gamma", 0.99))
    gae_lam     = str(cfg.get("gae_lambda", 0.95))
    clip_eps    = str(cfg.get("clip_eps", 0.2))
    vf_coef     = str(cfg.get("vf_coef", 0.5))
    ent_coef    = str(cfg.get("ent_coef", 0.01))
    seed        = str(cfg.get("seed", 1))
    view_size   = str(cfg.get("view_size", 7))
    log_every   = str(cfg.get("log_every", 50))
    save_every  = str(cfg.get("save_every", 500))
    tensorboard = cfg.get("tensorboard", False)

    # Redirect tensorboard into trainer runs dir
    if tensorboard:
        tb_dir = str(Path(run_dir) / "tensorboard")
        os.environ.setdefault("JAXMARL_TENSORBOARD_DIR", tb_dir)

    module_path = _resolve_scan_module(algo, env_family, environment)

    if env_family == "mosaic_multigrid" and algo.lower() in ("mappo", "ippo", "happo", "vdppo"):
        # Hydra key=value syntax (Phase 1 refactor, 2026-09-18) — replaces the
        # old argparse --flag value pairs, which already included a broken
        # --env-family flag no scan script's argparse ever defined.
        training_reward = str(cfg.get("training_reward", "cooperative-no-opponent")).replace("-", "_")
        env_group = _ENVIRONMENT_TO_HYDRA_ENV_GROUP.get(environment, environment)
        base_args = [
            f"env={env_group}",
            f"env.variant={variant}",
            f"run_dir={run_dir}",
            f"training_reward={training_reward}",
            f"total_updates={total_upd}",
            f"ppo.n_envs={n_envs}",
            f"ppo.n_steps={n_steps}",
            f"ppo.n_epochs={n_epochs}",
            f"ppo.n_minibatches={n_mb}",
            f"ppo.hidden_dim={hidden_dim}",
            f"ppo.lr={lr}",
            f"ppo.gamma={gamma}",
            f"ppo.gae_lambda={gae_lam}",
            f"ppo.clip_eps={clip_eps}",
            f"ppo.vf_coef={vf_coef}",
            f"ppo.ent_coef={ent_coef}",
            f"seed={seed}",
            f"env.view_size={view_size}",
            f"log_every={log_every}",
            f"save_every={save_every}",
            f"tensorboard={'true' if tensorboard else 'false'}",
        ]
    else:
        base_args = [
            "--env-family",    env_family,
            "--variant",       variant,
            "--run-dir",       run_dir,
            "--total-updates", total_upd,
            "--n-envs",        n_envs,
            "--n-steps",       n_steps,
            "--n-epochs",      n_epochs,
            "--n-minibatches", n_mb,
            "--hidden-dim",    hidden_dim,
            "--lr",            lr,
            "--gamma",         gamma,
            "--gae-lambda",    gae_lam,
            "--clip-eps",      clip_eps,
            "--vf-coef",       vf_coef,
            "--ent-coef",      ent_coef,
            "--seed",          seed,
            "--log-every",     log_every,
            "--save-every",    save_every,
            "--view-size",     view_size,
        ]
        if tensorboard:
            base_args.append("--tensorboard")

    old_argv = sys.argv
    try:
        sys.argv = ["jaxmarl_worker.cli"] + base_args
        module = importlib.import_module(module_path)
        if not hasattr(module, "main"):
            raise ImportError(f"Module {module_path} has no main() function")
        module.main()
    finally:
        sys.argv = old_argv


_VALID_ALGOS = {"mappo", "ippo", "mat", "commnet", "coma"}
_VALID_ENV_FAMILIES = {"mosaic_multigrid", "socialjax"}
_VALID_MOSAIC_ENVIRONMENTS = {"soccer", "af", "bb"}
_VALID_SOCIALJAX_ENVIRONMENTS = {"cleanup", "coins", "coop_mining"}
_VALID_VARIANTS = {"G-1v0", "B-0v1", "G-2v0", "G-3v0", "B-0v2", "B-0v3", "1v1", "2v2", "3v3"}


def _dry_run_from_config(config_path: str, emit_summary: bool = False) -> int:
    """Validate a config JSON without launching training.

    Checks: JSON parses, required fields present, algo/env_family/variant valid,
    numeric ranges sane, jaxmarl_worker imports cleanly.

    Returns:
        0 on success, 1 on any error.
    """
    import json
    from pathlib import Path

    errors: list[str] = []
    warnings: list[str] = []

    try:
        raw = Path(config_path).read_text()
        data = json.loads(raw)
    except Exception as exc:
        print(f"[dry-run] FAIL: cannot parse config JSON: {exc}", file=sys.stderr)
        return 1

    # Unwrap nested GUI-format config
    if "metadata" in data and "worker" in data.get("metadata", {}):
        cfg = data["metadata"]["worker"].get("config", data)
    else:
        cfg = data

    for key in ("run_id", "algo", "env_family"):
        if key not in cfg:
            errors.append(f"missing required field: {key}")

    algo = str(cfg.get("algo", "")).lower()
    if algo and algo not in _VALID_ALGOS:
        errors.append(f"invalid algo {algo!r}, expected one of {sorted(_VALID_ALGOS)}")

    env_family = cfg.get("env_family", "")
    # Legacy: if env_family is a sport name, treat it as mosaic_multigrid + environment
    if env_family in _VALID_MOSAIC_ENVIRONMENTS:
        environment = env_family
        env_family = "mosaic_multigrid"
    else:
        environment = cfg.get("environment", "")

    if env_family and env_family not in _VALID_ENV_FAMILIES:
        errors.append(f"invalid env_family {env_family!r}, expected one of {sorted(_VALID_ENV_FAMILIES)}")

    if env_family == "mosaic_multigrid":
        if environment and environment not in _VALID_MOSAIC_ENVIRONMENTS:
            errors.append(f"invalid mosaic_multigrid environment {environment!r}, expected one of {sorted(_VALID_MOSAIC_ENVIRONMENTS)}")
        variant = cfg.get("variant", "")
        if variant and variant not in _VALID_VARIANTS:
            errors.append(f"invalid variant {variant!r}, expected one of {sorted(_VALID_VARIANTS)}")
    elif env_family == "socialjax":
        if environment and environment not in _VALID_SOCIALJAX_ENVIRONMENTS:
            errors.append(f"invalid socialjax environment {environment!r}, expected one of {sorted(_VALID_SOCIALJAX_ENVIRONMENTS)}")
        # No variant check for socialjax

    for key, lo, hi in [
        ("total_updates", 1, 1_000_000),
        ("n_envs",        1, 65536),
        ("n_steps",       1, 4096),
        ("n_epochs",      1, 128),
        ("n_minibatches", 1, 256),
        ("hidden_dim",    8, 8192),
        ("view_size",     3, 31),
    ]:
        v = cfg.get(key)
        if v is not None:
            try:
                iv = int(v)
                if not (lo <= iv <= hi):
                    errors.append(f"{key}={v} out of range [{lo}, {hi}]")
            except (TypeError, ValueError):
                errors.append(f"{key}={v!r} not an integer")

    for key, lo, hi in [
        ("lr",         1e-8, 1.0),
        ("gamma",      0.0,  1.0),
        ("gae_lambda", 0.0,  1.0),
        ("clip_eps",   0.0,  1.0),
        ("vf_coef",    0.0,  100.0),
        ("ent_coef",   0.0,  10.0),
    ]:
        v = cfg.get(key)
        if v is not None:
            try:
                fv = float(v)
                if not (lo <= fv <= hi):
                    errors.append(f"{key}={v} out of range [{lo}, {hi}]")
            except (TypeError, ValueError):
                errors.append(f"{key}={v!r} not a number")

    try:
        import jaxmarl_worker  # noqa: F401
    except Exception as exc:
        errors.append(f"jaxmarl_worker package import failed: {exc}")

    if errors:
        print(f"[dry-run] FAIL: {len(errors)} error(s):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        if emit_summary:
            import json as _json
            print(_json.dumps({"status": "fail", "errors": errors, "warnings": warnings}))
        return 1

    variant_out = cfg.get("variant", "") if env_family == "mosaic_multigrid" else ""
    print(f"[dry-run] OK: config validates ({algo}, {env_family}, {environment}, {variant_out})")
    if warnings:
        print(f"[dry-run] {len(warnings)} warning(s):")
        for w in warnings:
            print(f"  - {w}")

    if emit_summary:
        import json as _json
        summary = {
            "status":      "ok",
            "algo":        algo,
            "env_family":  env_family,
            "environment": environment,
            "variant":     variant_out,
            "run_id":      cfg.get("run_id"),
            "warnings":    warnings,
        }
        print(_json.dumps(summary))

    return 0


def main():
    if "--interactive" in sys.argv:
        sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:] if a != "--interactive"]
        from jaxmarl_worker.runtime import main as runtime_main
        runtime_main()
    elif "--config" in sys.argv:
        idx = sys.argv.index("--config")
        config_path = sys.argv[idx + 1]
        if "--dry-run" in sys.argv:
            emit_summary = "--emit-summary" in sys.argv
            sys.exit(_dry_run_from_config(config_path, emit_summary=emit_summary))
        _run_from_config(config_path)
    else:
        # Legacy direct-args mode: default to Soccer MAPPO.
        # Users can pick a different sport via --config <path> instead.
        # mappo_scan.main is @hydra.main-decorated and reads sys.argv itself;
        # inject env=s (soccer) unless the caller already picked a sport.
        from jaxmarl_worker.algorithms.mosaic_multigrid.mappo_scan import main as mappo_main
        if not any(a == "env=s" or a.startswith("env=") for a in sys.argv[1:]):
            sys.argv = [sys.argv[0], "env=s"] + sys.argv[1:]
        mappo_main()


if __name__ == "__main__":
    main()
