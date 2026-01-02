# skaks-opt

Parameter tuning for skaks engine using the Python binding `skaks_eval`.

## Install

1) Install the binding (from repo root):
   ```bash
<<<<<<< HEAD
   cd bindings/python && pip install -v .
   ```
2) Install the optimizer package (editable recommended during iteration):
   ```bash
   cd tuning && uv pip install -e .
   ```
3) Optional: self-play helpers (PGN/FEN sampling, Dask fan-out, checkpointing):
   ```bash
   cd tuning && uv pip install -e .[selfplay]
=======
   cd bindings/python
   pip install -v .
   ```
2) Install the unified CLI from the project root:
   ```bash
   cd ../..
   uv pip install -e .
   ```
3) Optional extras for self-play / dataset tooling:
   ```bash
   uv pip install -e .[selfplay]
>>>>>>> nnue_version
   ```

## Dataset format

CSV with columns:
- `fen`: position
- `score` **or** `stockfish_cp`: target centipawn score (positive = White advantage)
- `side_to_move` (optional): `w`/`b`; defaults to FEN side
- `weight` (optional): per-sample weight

Example:
```
fen,stockfish_cp,side_to_move,weight
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1,0,w,1.0
8/8/8/8/8/8/8/4K3 w - - 0 1,0,w,0.5
```

## Run optimization

<<<<<<< HEAD
```bash
cd tuning
skaks-opt --data ../path/to/data.csv \
  --trials 100 \
  --jobs 4 \
  --threads 8 \
  --batch-size 512 \
   --error-penalty 2000 \
   --cp-cap 3000 \
   --pov side \
   --val-split 0.1 \
   --mtpe \
   --quiet \
   --metrics-out metrics.csv \
   --plot-out loss.png \
  --best-out best_params.yaml
```

- `--jobs` controls Optuna parallelism in this process.
- `--threads` controls threads used inside `skaks_eval.eval_fens`.
- `--include-arrays` also tunes the large array parameters (bigger search space).
- `--mtpe` enables multivariate TPE (grouped) which can converge faster on correlated params, especially with multiple `--jobs`.
- `--val-split` holds out a fraction of the rows for validation; training loss is optimized, validation metrics are reported to metrics.csv and the best summary.
- `--cp-cap` clamps targets and predictions to the given centipawn magnitude before computing metrics (useful to soften mate-score outliers).
- `--pov side` (default) matches engine outputs for the exact side to move; use `--pov white` if your targets are normalized to White.
- `--storage sqlite:///optuna.db --study-name skaks-opt` enables distributed tuning; run multiple `skaks-opt` processes pointing to the same storage to scale across machines.
=======
```
skaks-opt texel \
   --data tuning/eval_pairs_eval_scale800.csv \
   --trials 150 \
   --jobs 4 \
   --threads 8 \
   --batch-size 512 \
   --cp-cap 2000 \
   --param-set full \
   --sampler cmaes \
   --pruner median \
   --error-penalty 6.0 \
   --quiet \
   --metrics-out tuning/out/texel_metrics.csv \
   --best-out tuning/out/texel_params.yaml
```

- `--jobs` controls Optuna parallelism within the process.
- `--threads` selects worker threads for `skaks_eval.eval_fens`.
- `--include-arrays` expands tuning to PST arrays and other vectors (slower).
- `--pov side` (default) scores from the mover’s perspective; use `--pov white` for absolute scores.
- `--storage sqlite:///optuna.db --study-name skaks-texel` enables distributed tuning.
>>>>>>> nnue_version

## Outputs

- stdout prints best trial number, loss (weighted MAE), MSE/RMSE, and error count.
- `--quiet` hides Optuna per-trial logging; only the summary prints.
- `--metrics-out` writes per-trial metrics (loss/MAE, MSE, RMSE, errors) as CSV for plotting.
- `--plot-out` saves a loss-vs-trial PNG (matplotlib required).
- `--best-out` writes merged parameters (evaluation + search) as YAML; pass this dict back to `skaks_eval.eval_fens(params=...)`.

## Notes

- The search space defaults cover scalar eval + search parameters. Array parameters are opt-in to keep dimensionality manageable.
- Failed FEN evaluations are penalized by `--error-penalty` per occurrence.
<<<<<<< HEAD
- Random seed is configurable (`--seed`) 
=======
- Random seed is configurable (`--seed`).
>>>>>>> nnue_version
