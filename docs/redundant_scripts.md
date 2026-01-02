# Legacy Script Inventory

This catalog tracks legacy Python entry points that have been superseded by the unified skaks_opt CLI.

- Removed in 0.12: `tuning/param_optimize.py`, `tuning/texel_fit.py`, `tuning/texel_selfplay.py`, and `tuning/pyproject.toml`. The bundled CLI now ships the optimizer, Texel fit, and packaging metadata inside `skaks_opt`; use `skaks-opt param-optimize`, `skaks-opt texel`, and `skaks-opt selfplay` instead.
- [tuning/check_nnue_fit.py](tuning/check_nnue_fit.py) — the new fit workflow exposes the same evaluation and validation loop while reusing shared dataset loaders.
