#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Long-running Optuna (TPE) sweep. Uses fixed-budget points objective.
# Adjust --trials / --rounds / --concurrency for your machine.

python "$ROOT_DIR/optuna_runner.py" \
  --trials 200 \
  --objective points \
  --run-dir "$ROOT_DIR/spsa_runs" \
  --study-name tpe_long \
  --storage "sqlite:///$ROOT_DIR/optuna_tpe.db" \
  -- \
  --baseline-cmd skaks \
  --baseline-name SKX \
  --baseline-tc 1+0.01 \
  --test-cmd skaks \
  --test-name Skaks \
  --test-tc 1+0.01 \
  --openings-file "/Users/guillem/work/repos/skaks_tuning/datasets/Lichess_elite_db/lichess_elite_2015-09.pgn" \
  --openings-format pgn \
  --openings-order random \
  --rounds 800 \
  --repeat \
  --concurrency 8 \
  --recover \
  --autosaveinterval 50
