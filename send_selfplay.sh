#!/bin/bash
#SBATCH -J skaks-param-opt
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -o logs/selfplay-%j.out
#SBATCH -e logs/selfplay-%j.err

set -euo pipefail

module load Clang

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$HOME/work/repos/skaks:$PYTHONPATH"

cd "$HOME/work/repos/skaks"

ENGINE=skaks
BASELINE=tuning/phase_weights_optimized.yaml
START=tuning/phase_weights_optimized.yaml
OUTPUT=tuning/best_params_optimized.yaml

GAMES=64
ITERATIONS=40
REPEATS=3
BEAM_SIZE=16
NOISE=0.08
DEPTH=4

ARENA_WORKERS=1
DASK_SHARDS=4                # low gate so we start ASAP
JOBQUEUE_CONFIG=tuning/slurm_cluster.yaml
JOBQUEUE_JOBS=8              # initial scale target
DASK_ADAPT_MIN=4
DASK_ADAPT_MAX=20

RESET_BASELINE=${RESET_BASELINE:-0}
if [[ "$RESET_BASELINE" == "1" ]]; then
  cp -f "$OUTPUT" "$BASELINE" 2>/dev/null || true
  rm -f "$OUTPUT" "${OUTPUT}.best.json"
fi

skaks-opt param-optimize \
  --engine "$ENGINE" \
  --baseline-params "$BASELINE" \
  --start-params "$START" \
  --output "$OUTPUT" \
  --games "$GAMES" \
  --iterations "$ITERATIONS" \
  --noise "$NOISE" \
  --beam-size "$BEAM_SIZE" \
  --repeats "$REPEATS" \
  --depth "$DEPTH" \
  --strategy beam \
  --use-arena-binding \
  --arena-workers "$ARENA_WORKERS" \
  --dask-shards "$DASK_SHARDS" \
  --baseline-decay 0.004 \
  --force-accept-first 4 \
  --min-score 0.55 \
  --child-output \
  --dask-jobqueue \
  --dask-jobqueue-config "$JOBQUEUE_CONFIG" \
  --dask-jobqueue-jobs "$JOBQUEUE_JOBS" \
  --dask-jobqueue-adapt-min "$DASK_ADAPT_MIN" \
  --dask-jobqueue-adapt-max "$DASK_ADAPT_MAX"