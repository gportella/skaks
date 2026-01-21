#!/bin/bash

python -m skaks_opt.cli param-optimize \
  --engine skaks \
  --baseline-params besty.yaml \
  --start-params search_nnue_init.yaml \
  --include-prefix search_nnue \
  --games 20 \
  --iterations 50 \
  --repeats 4 \
  --arena-workers 4 \
  --depth 5 \
  --output besty_search_nnue.yaml
