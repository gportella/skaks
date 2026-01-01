# Legacy Script Inventory

This catalog tracks old Python entry points that have been superseded by the unified skaks_opt CLI and are candidates for removal after downstream users migrate.

- [tuning/param_optimize.py](tuning/param_optimize.py) — superseded by the selfplay subcommand, which folds arena orchestration and candidate selection into a maintained module.
- [tuning/texel_fit.py](tuning/texel_fit.py) — functionality now covered by the texel subcommand in skaks_opt.
- [tuning/check_nnue_fit.py](tuning/check_nnue_fit.py) — the new fit workflow exposes the same evaluation and validation loop while reusing shared dataset loaders.
- [tuning/texel_selfplay.py](tuning/texel_selfplay.py) — integrated into the selfplay command with reusable configuration structs.
- [tuning/pyproject.toml](tuning/pyproject.toml) — packaging entry now replaced by the project-level definition and should disappear once callers switch to the root configuration.
- [tuning/skaks_opt/cli.py](tuning/skaks_opt/cli.py) — shim maintained only for backwards compatibility; safe to drop when legacy imports vanish.
