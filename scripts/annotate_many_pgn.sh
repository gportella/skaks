#!/usr/bin/env bash
python -m skaks_opt.cli dataset-sample \
  --inputs ~/Downloads/Lichess\ Elite\ Database/*.pgn \
  --output-dir tuning/datasets/lichess_elite \
  --stockfish stockfish \
  --depth 15 \
  --stride 4 \
  --min-ply 10 \
  --max-ply 80 \
  --chunk-size 512 \
  --rows-per-shard 100000 \
  --quiet-only \
  --stockfish-option Threads=8 \
  --stockfish-option Hash=512 \
  "$@"
