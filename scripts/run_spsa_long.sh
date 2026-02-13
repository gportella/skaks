#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Long-running SPSA sweep optimized for throughput.
# Tweak SPSA_* / --iterations / --rounds / --concurrency for your machine.
# Tip: increase --concurrency to match (or slightly exceed) physical cores.

export SPSA_C=${SPSA_C:-1.0}
export SPSA_A=${SPSA_A:-0.5}
export SPSA_BIG_A=${SPSA_BIG_A:-100}

python "$ROOT_DIR/spsa_runner.py" \
  --iterations 1400 \
  --objective points \
  --reject-wdl-drop \
  --monotonic-wdl \
  --parallel-evals \
  --random-starts 6 \
  --restart-stagnation 80 \
  --run-dir "$ROOT_DIR/spsa_runs" \
  --only lmr_intercept \
  --only lmr_divisor \
  --only lmr_pv_offset \
  --only null_move_reduction \
  --only null_move_min_depth \
  --only quiescence_delta_margin \
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
  --rounds 200 \
  --repeat \
  --concurrency 8 \
  --recover \
  --autosaveinterval 50 \
  --sprt \
  --sprt-elo1 1.0 \
  --sprt-alpha 0.10 \
  --sprt-beta 0.10
