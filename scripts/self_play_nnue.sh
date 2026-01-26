#!/bin/bash
skaks-opt param-optimize \
  --engine skaks \
  --baseline-params search_nnue_baseline.yaml \
  --start-params search_nnue_baseline.yaml \
  --include-prefix search_nnue \
  --games 20 \
  --iterations 50 \
  --use-arena-binding \
  --strategy spsa \
  --repeats 4 \
  --arena-workers 4 \
  --spsa-c 2 \
  --spsa-a 0.5 \
  --nodes 10000 \
  --output best_search_nnue.yaml

