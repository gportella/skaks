#!/usr/bin/env bash
set -euo pipefail

# Usage: batch_eval_launcher.sh samples.csv tuning/default_start.yaml --batch-size 500 --games 20 --depth 2 --concurrency 1 --outdir samples_out --dry-run

SAMPLES_CSV=${1:-}
BASE_YAML=${2:-}
shift 2 || true

if [ -z "$SAMPLES_CSV" ] || [ -z "$BASE_YAML" ]; then
  echo "Usage: $0 samples.csv base_start.yaml [--batch-size N] [--games G] [--depth D] [--concurrency C] [--outdir DIR] [--dry-run]"
  exit 1
fi

# defaults
BATCH_SIZE=500
GAMES=20
DEPTH=2
CONCURRENCY=1
OUTDIR=samples_out
DRY_RUN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --batch-size) BATCH_SIZE=$2; shift 2;;
    --games) GAMES=$2; shift 2;;
    --depth) DEPTH=$2; shift 2;;
    --concurrency) CONCURRENCY=$2; shift 2;;
    --outdir) OUTDIR=$2; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

mkdir -p "$OUTDIR" logs

# split samples into chunks of BATCH_SIZE lines (include header in each)
TMPDIR="logs/samples_chunks"
rm -rf "$TMPDIR"
mkdir -p "$TMPDIR"

HEADER=$(head -n1 "$SAMPLES_CSV")
tail -n +2 "$SAMPLES_CSV" | split -l $BATCH_SIZE - "$TMPDIR/chunk_"

i=0
for f in "$TMPDIR"/chunk_*; do
  i=$((i+1))
  chunk_csv="$TMPDIR/chunk_${i}.csv"
  (echo "$HEADER"; cat "$f") > "$chunk_csv"
  session="skaks-sample-$i"
  log="logs/${session}.log"
  cmd="python scripts/eval_samples_runner.py --samples $chunk_csv --base $BASE_YAML --games $GAMES --depth $DEPTH --concurrency $CONCURRENCY --outdir $OUTDIR"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "+ $cmd"
  else
    tmux kill-session -t "$session" 2>/dev/null || true
    tmux new-session -d -s "$session" "$cmd > $log 2>&1"
    echo "Started $session -> $log"
  fi
done

echo "Launched $i batches (batch-size=$BATCH_SIZE). Output dir: $OUTDIR"
