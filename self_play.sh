#!/bin/bash

# Self-play script to optimize Skaks params against handicapped Stockfish
# Use Stockfish with reduced time for handicap

python -m tuning.param_optimize \
  --engine skaks \
  --baseline-params besty.yaml \
  --start-params besty.yaml \
  --external-opponent \
  --opponent sunfish \
  --depth 3 \
  --opponent-time-per-move 0.3 \
  --games 20 \
  --iterations 20 \
  --repeats 6 \
  --concurrency 1 \
  --output besty_vs_stockfish.yaml
