# Optimization recipes

## Phase-weight-only tuning

Use these when you only want to optimize the phase weights (`evaluation.phase_weights_mg` / `evaluation.phase_weights_eg`) already present in `tuning/best_params.yaml`.

```bash
# CP regression on eval_pairs CSV (Optuna)
python -m tuning.skaks_opt.cli \
  --data eval_pairs_pvs_with_results.csv \
  --phase-weights-only \
  --trials 80 \
  --threads 0 \
  --batch-size 512 \
  --pov side \
  --best-out tuning/best_params_phase.yaml

# Self-play optimizer
python -m tuning.param_optimize \
  --start-params tuning/best_params.yaml \
  --phase-weights-only \
  --output tuning/best_params_phase.yaml

# Quick internal arena (baseline vs tuned params)
./build/debug/bin/skaks --arena --params tuning/best_params_phase.yaml --arena-games 200
```

## Arena binding + scaling-factor-only self-play

Use the internal arena binding (`--use-arena-binding`) to avoid spinning up external engine processes. Supply `--arena-pgn` to sample midgame FENs; the binding will fan them out across workers. Combine with `--include-prefix` to tune only a scale factor such as `evaluation.mobility_scaling`.

Requirements: Python bindings built/installed (`pip install ./bindings/python` from repo root) and a PGN file to sample from.

```bash
python -m tuning.param_optimize \
  --start-params tuning/best_params.yaml \
  --include-prefix evaluation.mobility_scaling \
  --use-arena-binding \
  --arena-pgn moves_pgn/LumbrasGigaBase_OTB_2025.pgn \
  --arena-min-ply 8 \
  --arena-max-ply 60 \
  --arena-workers 4 \
  --games 80 \
  --iterations 6 \
  --depth 6 \
  --noise 0.26 \
  --beam-size 2 \
  --output tuning/best_params_scale.yaml
```

Notes:
- Swap `--include-prefix` to any other scalar you want to isolate (repeatable flag).
- Omit `--arena-pgn` to fall back to the start position; keep `--depth` when using the binding (clock controls are not supported there).
- Lower `--games` or `--iterations` for faster feedback; increase `--arena-workers` if you have CPU headroom.
