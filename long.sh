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
BASELINE_PARAMS=${BASELINE_PARAMS:-../fitting_models/base_params.yaml}
START_PARAMS=${START_PARAMS:-../fitting_models/base_params.yaml}
OPPONENT=${OPPONENT:-skaks}
OPPONENT_DEPTH_FACTOR=${OPPONENT_DEPTH_FACTOR:-1.0}


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
EXPL_GAMES=${EXPL_GAMES:-60}
EXPL_REPEATS=${EXPL_REPEATS:-1}
EXPL_DEPTH=${EXPL_DEPTH:-5}
EXPL_NOISE=${EXPL_NOISE:-0.12}

REF_ITERATIONS=${REF_ITERATIONS:-0}
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
SCORE_CONFIDENCE=${SCORE_CONFIDENCE:-0.95}
REQUIRE_SCORE_CONFIDENCE=${REQUIRE_SCORE_CONFIDENCE:-1}

# Arena sampling (override via env vars to change PGN/min/max ply or disable PGN entirely)
ARENA_PGN=${ARENA_PGN:-moves_pgn/LumbrasGigaBase_OTB_2025.pgn}
ARENA_MIN_PLY=${ARENA_MIN_PLY:-8}
ARENA_MAX_PLY=${ARENA_MAX_PLY:-40}

# Control which phases should run in weights-only mode (default: REFINE only).
WEIGHTS_ONLY_PHASES=${WEIGHTS_ONLY_PHASES:-REFINE}

phase_uses_weights_only() {
  local phase=$1
  case " ${WEIGHTS_ONLY_PHASES} " in
    *" ${phase} "*) return 0 ;;
    *) return 1 ;;
  esac
}

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
    ARENA_FLAGS+=" --use-arena-binding --arena-workers '$CONCURRENCY'"
    if [ -n "${ARENA_PGN:-}" ]; then
      log "Arena PGN source: ${ARENA_PGN} (ply range ${ARENA_MIN_PLY}-${ARENA_MAX_PLY})"
      ARENA_FLAGS+=" --arena-pgn '${ARENA_PGN}' --arena-min-ply ${ARENA_MIN_PLY} --arena-max-ply ${ARENA_MAX_PLY}"
    else
      log "Arena PGN disabled: starting from standard startpos"
    fi
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
    INCLUDE_FLAGS=""
    if phase_uses_weights_only "$phase_name"; then
      # Limit tuning to the phase weight arrays for final polish sweeps.
      INCLUDE_FLAGS=" --weights-only"
    fi
    CONFIDENCE_FLAGS=" --score-confidence '$SCORE_CONFIDENCE'"
    if [ "${REQUIRE_SCORE_CONFIDENCE}" -ne 0 ]; then
      CONFIDENCE_FLAGS+=" --require-score-confidence"
    fi
    EXTERNAL_FLAG=""
    if [ "${OPPONENT:-}" != "${ENGINE}" ] && [ -n "${OPPONENT:-}" ]; then
      EXTERNAL_FLAG=" --external-opponent"
    fi
    if [ -n "$INCLUDE_FLAGS" ]; then
      log "Running weights-only optimizer for phase=$phase_name (baseline-decay=${BASELINE_DECAY})"
    else
      log "Running full-parameter optimizer for phase=$phase_name (baseline-decay=${BASELINE_DECAY})"
    fi
    CMD="skaks-opt param-optimize ${EXTERNAL_FLAG} $COMMON_FLAGS --depth '$depth' --games '$games' --iterations 1 --repeats '$repeats' --beam-size '$beam' --noise '$noise' --concurrency '$CONCURRENCY' --baseline-decay '$BASELINE_DECAY' --min-score '$MIN_SCORE' --force-accept-first '$FORCE_ACCEPT_FIRST'${EXTF}${INCLUDE_FLAGS}${CONFIDENCE_FLAGS} $ARENA_FLAGS --output '$LOG_DIR/${phase_name}_best.yaml'"
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
