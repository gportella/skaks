# Skaks Optimization Guide

Concise menu of tuning paths with ready-to-run examples. Adjust paths as needed (built engine, data files, Python venv).

## CP Regression (Texel-style) — `tuning.skaks_opt.cli`
Use offline regression on labeled centipawn data.

- **Typical filtered run (scalars + arrays)**
  ```
  python -m tuning.skaks_opt.cli \
    --data eval_pairs_pvs_filtered.csv \
    --trials 400 \
    --require-quiet \
    --jobs 2 \
    --threads 0 \
    --batch-size 1024 \
    --cp-cap 2000 \
    --error-penalty 2000 \
    --sampler cmaes \
    --pruner median \
    --val-split 0.1 \
    --include-arrays \
    --best-out tuning/out/best_filtered.yaml \
    --metrics-out tuning/out/metrics_filtered.csv \
    --rich
  ```
- **Phase weights only**
  Add `--phase-weights-only` to confine the search to `evaluation.phase_weights_mg/eg`.
- Notes: `--include-arrays` adds king_attack/threat/phase arrays; without it, only scalars tune. Use `--cp-cap` to clip mate scores.

## Self-Play Optimizer — `tuning/param_optimize.py`
Explores params via self-play vs baseline.

- **Beam (fast feedback, shallower search)**
  ```
  python -m tuning.param_optimize \
    --engine build/debug/bin/skaks \
    --baseline-params tuning/best_params.yaml \
    --start-params tuning/best_params.yaml \
    --include-prefix evaluation \
    --games 30 --iterations 2 --repeats 1 \
    --noise 0.2 --strategy beam --beam-size 1 \
    --concurrency 4 --depth 5
  ```
- **CMA (broader search)**
  ```
  python -m tuning.param_optimize \
    --engine build/debug/bin/skaks \
    --baseline-params tuning/best_params.yaml \
    --start-params tuning/best_params.yaml \
    --include-prefix evaluation \
    --games 60 --iterations 8 --repeats 2 \
    --noise 0.2 --strategy cma --cma-popsize 8 \
    --concurrency 4 --depth 5
  ```
- **Phase weights only**: add `--phase-weights-only`.
- Timing knobs: choose exactly one of `--depth`, `--time-per-move`, or `--clock` (plus increments). Lower for faster feedback.

## NNUE Training (small/large)
Train networks from FEN/feature datasets. Use the Python binding entrypoints:

- **Small net** (faster):
  ```
  python -m tuning.nnue.train \
    --data training.fens \
    --net-config configs/nnue_small.yaml \
    --epochs 10 --batch-size 8192 \
    --lr 1e-3 --eval-interval 1 \
    --out checkpoints/nnue_small.pt
  ```
- **Large net** (stronger):
  ```
  python -m tuning.nnue.train \
    --data training.fens \
    --net-config configs/nnue_large.yaml \
    --epochs 10 --batch-size 4096 \
    --lr 7e-4 --eval-interval 1 \
    --out checkpoints/nnue_large.pt
  ```
- Swap configs to switch size; adjust batch/learning rate to fit VRAM.

## HalfKP (Half-King-Piece) NNUE — Python only
Offline halfkp trainer over FEN/CP CSVs.

- **Typical run**
  ```
  python -m tuning.halfkp_train \
    --data eval_pairs_pvs_ply8_80_cp2k.csv \
    --pov side \
    --min-ply 8 --max-ply 80 \
    --clamp-cp 2000 \
    --embed 320 --hidden 192 \
    --batch-size 2048 --epochs 5 \
    --lr 1e-3 --weight-decay 1e-5 \
    --save tuning/out/halfkp.pt
  ```
- **Hyperparam search**: add `--optuna-trials 50 --optuna-epochs 4 --optuna-sample-fraction 0.5`.
- Controls: `--delta` (Huber), `--sample-fraction`, `--max-rows`, `--pov` (side/white). Uses Huber loss with MAE/corr reporting.

## Texel / Sigmoid Fits (alt regressions)
Alternate regression heads (sigmoid/logistic) for eval fitting.

- **Sigmoid fit**:
  ```
  python -m tuning.texel_fit \
    --data eval_pairs_pvs_ply8_80_cp2k.csv \
    --loss sigmoid \
    --cp-cap 2000 \
    --lr 1e-3 \
    --epochs 5 \
    --out tuning/out/texel_sigmoid.yaml
  ```
- **Plain Texel (L2/Huber)**:
  ```
  python -m tuning.texel_fit \
    --data eval_pairs_pvs_ply8_80_cp2k.csv \
    --loss huber \
    --delta 64 \
    --cp-cap 2000 \
    --lr 1e-3 \
    --epochs 5 \
    --out tuning/out/texel_huber.yaml
  ```

## Validation Data Generation — `validation_moves/eval_from_pgn.py`
Harvest labeled positions from PGN with Stockfish and skaks static evals.

```
python -m validation_moves.eval_from_pgn \
  --pgn game.pgn \
  --stockfish /path/to/stockfish \
  --skaks build/debug/bin/skaks \
  --stockfish-depth 12 \
  --sample-stride 10 \
  --min-ply 8 --max-ply 80 \
  --output eval_pairs_pvs.csv \
  --pov side
```
- Use `--min-ply/--max-ply` to trim openings/endgames; raise depth for quality, lower for speed.

## Tips
- Clamp mates (`--cp-cap`) before fitting; filter ply ranges to reduce noise.
- For faster feedback, lower games/depth/time-per-move and repeats; for stability, increase repeats.
- Keep a baseline params YAML and write tuned outputs to distinct files.
