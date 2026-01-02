# Dataset Sampling Pipeline

This workflow builds large Texel-ready datasets from bulk PGN archives and then fits parameters with the existing `texel` CLI.

## 1. Sample PGNs With Stockfish

```bash
python -m skaks_opt.cli dataset-sample \
  --inputs "~/Downloads/Lichess Elite Database/*.pgn" \
  --output-dir tuning/datasets/lichess_elite \
  --stockfish /usr/local/bin/stockfish \
  --depth 15 \
  --stride 4 \
  --min-ply 10 \
  --max-ply 80 \
  --chunk-size 512 \
  --rows-per-shard 100000 \
  --quiet-only \
  --max-total 5000000
```

Key notes:
- Globs, directories, or explicit PGN paths are accepted via `--inputs`.
- Set `--quiet-only` to keep only quiet middlegame positions (requires `skaks_eval`).
- Use `--rows-per-shard` to split each PGN into manageable CSV shards for distributed processing.
- Supply `--stockfish-option Threads=8 Hash=512` if you want to tweak UCI options.

Each shard contains:
`source, game_index, ply, fen, side_to_move, stockfish_cp, result, outcome, winner, weight`.

## 2. Launch Texel Fitting (Local or Dask Worker)

```bash
python -m skaks_opt.cli texel \
  --data tuning/datasets/lichess_elite/*.csv \
  --trials 200 \
  --threads 8 \
  --batch-size 1024 \
  --cp-cap 2000 \
  --param-set phase \
  --best-out tuning/texel_out/best.yaml \
  --metrics-out tuning/texel_out/metrics.csv
```

For Dask clusters, hand each worker a slice of the PGN glob (or run multiple `dataset-sample` jobs with different `--max-total` limits) and store shards in a shared bucket. Workers can then run `texel` independently and write their own `best.yaml` files for downstream sweep/selection.

## 3. Refresh Optimizer Baselines

After fitting, reuse the emitted params in self-play or arena loops:

```bash
python -m skaks_opt.cli param-optimize \
  --engine ./build/debug/skaks \
  --opponent sunfish \
  --external-opponent \
  --baseline-params tuning/texel_out/best.yaml \
  --start-params tuning/texel_out/best.yaml \
  --output tuning/texel_out/best.yaml \
  --games 80 \
  --iterations 10 \
  --depth 4 \
  --opponent-depth-factor 0.2 \
  --opponent-time-per-move 0.2
```

Adjust depths, sampling stride, and trial counts to suit runtime budgets. For very large archives, prefer running many sampling jobs in parallel and concatenating their CSV output before launching Texel tuning.
