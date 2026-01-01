#!/bin/bash

# Self-play script to optimize Skaks params against Sunfish
# Use Stockfish with reduced time for handicap

python -m tuning.param_optimize \
  --engine skaks \
  --baseline-params besty.yaml \
  --start-params besty.yaml \
  --opponent sunfish \
  --depth 4 \
  --games 10 \
  --iterations 25 \
  --repeats 4 \
  --concurrency 4 \
  --output besty_vs_sunfish.yaml \
  --external-opponent 
