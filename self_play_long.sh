#!/usr/bin/env bash
# Long-running two-phase training script for Skaks parameter optimization.
# Phase 1: Exploration (wide, faster)
# Phase 2: Refinement (narrower, thorough)

set -euo pipefail
cd "$(dirname "$0")"

# Configurable targets
TARGET_GAMES=${TARGET_GAMES:-1000000} # target total games to play
LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$LOG_DIR"

# Engine / params
ENGINE=${ENGINE:-skaks}
BASELINE_PARAMS=${BASELINE_PARAMS:-texel_ext_all.yaml}
START_PARAMS=${START_PARAMS:-texel_ext_all.yaml}
OPPONENT=${OPPONENT:-sunfish}
OPPONENT_DEPTH_FACTOR=${OPPONENT_DEPTH_FACTOR:-0.30}

# Detect arena binding availability (later checked with a single python call)

# Default concurrency
CPU_COUNT=$(getconf _NPROCESSORS_ONLN || echo 4)
CONCURRENCY=${CONCURRENCY:-$((CPU_COUNT < 8 ? CPU_COUNT : 8))}
# If using an external opponent (often a Python process), cap concurrency
# to avoid frequent crashes; keep it conservative by default.
if [ "${OPPONENT:-}" != "${ENGINE}" ]; then
  # If user provided CONCURRENCY externally, respect it; otherwise cap to 4.
  if [ -z "${CONCURRENCY:-}" ] || [ "$CONCURRENCY" -gt 4 ]; then
    CONCURRENCY=4
  fi
fi

# Phase configs (tweak these values for your machine)
EXPL_ITERATIONS=${EXPL_ITERATIONS:-100}
EXPL_BEAM=${EXPL_BEAM:-12}
EXPL_GAMES=${EXPL_GAMES:-20}
EXPL_REPEATS=${EXPL_REPEATS:-1}
EXPL_DEPTH=${EXPL_DEPTH:-5}
EXPL_NOISE=${EXPL_NOISE:-0.12}

REF_ITERATIONS=${REF_ITERATIONS:-60}
REF_BEAM=${REF_BEAM:-6}
REF_GAMES=${REF_GAMES:-80}
REF_REPEATS=${REF_REPEATS:-3}
REF_DEPTH=${REF_DEPTH:-5}
REF_NOISE=${REF_NOISE:-0.02}

# Relaxation: per-iteration baseline decay to make acceptance gradual
# Small positive value makes it easier to replace a stubborn best.
BASELINE_DECAY=${BASELINE_DECAY:-0.05}

# Acceptance tuning: allow keeping early improvements below default thresholds
MIN_SCORE=${MIN_SCORE:-0.0}
FORCE_ACCEPT_FIRST=${FORCE_ACCEPT_FIRST:-0}

# Helpers
# Use a portable ISO-8601 timestamp compatible with macOS/BSD `date`
timestamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG_DIR/self_play_long.log"; }

# Build common flags
# Common flags (kept as plain strings; we'll construct the command string later)
COMMON_FLAGS="--engine '$ENGINE' --baseline-params '$BASELINE_PARAMS' --start-params '$START_PARAMS' --opponent '$OPPONENT'"
ARENA_FLAGS=""
# Only enable arena binding when running internal self-play; skip for external opponent runs.
if [ "${OPPONENT:-}" = "${ENGINE}" ] || [ -z "${OPPONENT:-}" ]; then
  if python -c 'import importlib,sys
sys.stdout.write("1" if importlib.util.find_spec("skaks_eval") else "0")' 2>/dev/null | grep -q 1; then
    log "Detected skaks_eval: enabling arena binding"
    ARENA_FLAGS+=(--use-arena-binding   --arena-min-ply 8 --arena-max-ply 40 --arena-workers "$CONCURRENCY")
  else
    log "No skaks_eval detected: using batch runner mode"
  fi
else
  log "External opponent configured ($OPPONENT): skipping arena binding"
fi

# Counting helper
TOTAL_PLAYED=0

run_phase() {
  local phase_name=$1
  shift
  local iterations=$1
  shift
  local beam=$1
  shift
  local games=$1
  shift
  local repeats=$1
  shift
  local depth=$1
  shift
  local noise=$1
  shift

  for i in $(seq 1 $iterations); do
    log "Phase=$phase_name iter=$i/$iterations games=$games repeats=$repeats depth=$depth beam=$beam noise=$noise"
    # Build command string (use eval to allow ARENA_FLAGS expansion)
    # If running against an external opponent, pass opponent depth factor so
    # the opponent runs at ~60% of the candidate depth (handicap-like).
    EXTF=""
    if [ "${OPPONENT:-}" != "${ENGINE}" ] && [ -n "${OPPONENT_DEPTH_FACTOR:-}" ]; then
      EXTF=" --opponent-depth-factor ${OPPONENT_DEPTH_FACTOR}"
    fi
    # For long runs prefer tuning only the phase weight arrays (faster, lower dim).
    # Use the convenience alias the optimizer provides so the script focuses on
    # `evaluation.phase_weights_mg` and `evaluation.phase_weights_eg`.
    INCLUDE_FLAGS=" --weights-only"
    log "Running weights-only optimizer for phase=$phase_name (weights-only mode, baseline-decay=${BASELINE_DECAY})"
    CMD="skaks-opt param-optimize --external-opponent $COMMON_FLAGS --depth '$depth' --games '$games' --iterations 1 --repeats '$repeats' --beam-size '$beam' --noise '$noise' --concurrency '$CONCURRENCY' --baseline-decay '$BASELINE_DECAY' --min-score '$MIN_SCORE' --force-accept-first '$FORCE_ACCEPT_FIRST'${EXTF}${INCLUDE_FLAGS} $ARENA_FLAGS --output '$LOG_DIR/${phase_name}_best.yaml'"
    eval "$CMD" 2>&1 | tee -a "$LOG_DIR/${phase_name}_iter_${i}.log"

    # estimate games played this iteration
    TOTAL_PLAYED=$((TOTAL_PLAYED + games * repeats))
    log "Total games (approx): $TOTAL_PLAYED / $TARGET_GAMES"
    if [ "$TOTAL_PLAYED" -ge "$TARGET_GAMES" ]; then
      log "Reached target games: $TOTAL_PLAYED >= $TARGET_GAMES"
      return 0
    fi
  done
}

# Run exploration then refinement until target reached
log "Starting long run target=$TARGET_GAMES"
while [ "$TOTAL_PLAYED" -lt "$TARGET_GAMES" ]; do
  run_phase EXPLORE $EXPL_ITERATIONS $EXPL_BEAM $EXPL_GAMES $EXPL_REPEATS $EXPL_DEPTH $EXPL_NOISE
  if [ "$TOTAL_PLAYED" -ge "$TARGET_GAMES" ]; then break; fi
  # After exploration, set START_PARAMS to the last best file
  START_PARAMS="$LOG_DIR/EXPLORE_best.yaml"
  run_phase REFINE $REF_ITERATIONS $REF_BEAM $REF_GAMES $REF_REPEATS $REF_DEPTH $REF_NOISE
  START_PARAMS="$LOG_DIR/REFINE_best.yaml"
done

log "Long run finished. Total games approx: $TOTAL_PLAYED"
log "Final best params: $START_PARAMS"

exit 0
