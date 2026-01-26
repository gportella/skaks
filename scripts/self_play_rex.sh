#!/usr/bin/env bash
# Long-running two-phase training script for Skaks parameter optimization.
# Phase 1: Exploration (wide, faster)
# Phase 2: Refinement (narrower, thorough)

set -euo pipefail
cd "$(dirname "$0")"

# Phase configs (tweak these values for your machine)
EXPL_ITERATIONS=${EXPL_ITERATIONS:-100}
EXPL_GAMES=${EXPL_GAMES:-300}
EXPL_REPEATS=${EXPL_REPEATS:-4}
EXPL_DEPTH=${EXPL_DEPTH:-4}
EXPL_NOISE=${EXPL_NOISE:-0.30}

# Configurable targets
TARGET_GAMES=${TARGET_GAMES:-1000000} # target total games to play
LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$LOG_DIR"

# Engine / params
ENGINE=${ENGINE:-skaks}
BASELINE_PARAMS=${BASELINE_PARAMS:-../fitting_models/base_params_pst.yaml}
START_PARAMS=${START_PARAMS:-../fitting_models/texel_ext_all.yaml}
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


REF_ITERATIONS=${REF_ITERATIONS:-0}
REF_GAMES=${REF_GAMES:-80}
REF_REPEATS=${REF_REPEATS:-3}
REF_DEPTH=${REF_DEPTH:-5}
REF_NOISE=${REF_NOISE:-0.02}

# Replica exchange controls
REX_CHAINS=${REX_CHAINS:-4}
REX_TEMP_MIN=${REX_TEMP_MIN:-0.25}
REX_TEMP_MAX=${REX_TEMP_MAX:-2.5}
REX_TEMPS=${REX_TEMPS:-}
REX_SWAP_INTERVAL=${REX_SWAP_INTERVAL:-2}
REX_TARGET_ACCEPT=${REX_TARGET_ACCEPT:-0.2}
REX_ADAPT_RATE=${REX_ADAPT_RATE:-0.05}
REX_SEED=${REX_SEED:-}

# Relaxation: per-iteration baseline decay to make acceptance gradual
# Small positive value makes it easier to replace a stubborn best.
BASELINE_DECAY=${BASELINE_DECAY:-0.05}

# Acceptance tuning: allow keeping early improvements below default thresholds
MIN_SCORE=${MIN_SCORE:-0.0}
FORCE_ACCEPT_FIRST=${FORCE_ACCEPT_FIRST:-0}
SCORE_CONFIDENCE=${SCORE_CONFIDENCE:-0.90}
REQUIRE_SCORE_CONFIDENCE=${REQUIRE_SCORE_CONFIDENCE:-1}

# Arena sampling (override via env vars to change PGN/min/max ply or disable PGN entirely)
ARENA_PGN=${ARENA_PGN:-../datasets/lichess_elite_db/lichess_elite_2019-01.pgn}
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
timestamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG_DIR/self_play_long.log"; }

# Build common flags
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
  local games=$1
  shift
  local repeats=$1
  shift
  local depth=$1
  shift
  local noise=$1
  shift

  for i in $(seq 1 $iterations); do
    local strategy_desc="strategy=rex noise=$noise"
    strategy_desc+=" chains=${REX_CHAINS} tmin=${REX_TEMP_MIN} tmax=${REX_TEMP_MAX}"
    if [ -n "${REX_TEMPS:-}" ]; then
      strategy_desc+=" temps=${REX_TEMPS}"
    fi
    strategy_desc+=" swap=${REX_SWAP_INTERVAL} target=${REX_TARGET_ACCEPT} adapt=${REX_ADAPT_RATE}"
    if [ -n "${REX_SEED:-}" ]; then
      strategy_desc+=" seed=${REX_SEED}"
    fi
    log "Phase=$phase_name iter=$i/$iterations games=$games repeats=$repeats depth=$depth $strategy_desc"
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
    STRATEGY_FLAGS=" --strategy rex --noise '$noise'"
    STRATEGY_FLAGS+=" --rex-chains '${REX_CHAINS}'"
    STRATEGY_FLAGS+=" --rex-temp-min '${REX_TEMP_MIN}'"
    STRATEGY_FLAGS+=" --rex-temp-max '${REX_TEMP_MAX}'"
    if [ -n "${REX_TEMPS:-}" ]; then
      STRATEGY_FLAGS+=" --rex-temps '${REX_TEMPS}'"
    fi
    STRATEGY_FLAGS+=" --rex-swap-interval '${REX_SWAP_INTERVAL}'"
    STRATEGY_FLAGS+=" --rex-target-accept '${REX_TARGET_ACCEPT}'"
    STRATEGY_FLAGS+=" --rex-adapt-rate '${REX_ADAPT_RATE}'"
    if [ -n "${REX_SEED:-}" ]; then
      STRATEGY_FLAGS+=" --rex-seed '${REX_SEED}'"
    fi
    if [ -n "$INCLUDE_FLAGS" ]; then
      log "Running weights-only optimizer for phase=$phase_name (baseline-decay=${BASELINE_DECAY}, strategy=rex)"
    else
      log "Running full-parameter optimizer for phase=$phase_name (baseline-decay=${BASELINE_DECAY}, strategy=rex)"
    fi
    CMD="skaks-opt param-optimize ${EXTERNAL_FLAG} $COMMON_FLAGS --skip-pst --param-set full  --depth '$depth' --games '$games' --iterations 1 --repeats '$repeats' --concurrency '$CONCURRENCY' --baseline-decay '$BASELINE_DECAY' --min-score '$MIN_SCORE' --force-accept-first '$FORCE_ACCEPT_FIRST'${EXTF}${INCLUDE_FLAGS}${CONFIDENCE_FLAGS}${STRATEGY_FLAGS} $ARENA_FLAGS --output '$LOG_DIR/${phase_name}_best.yaml'"
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
  run_phase EXPLORE $EXPL_ITERATIONS $EXPL_GAMES $EXPL_REPEATS $EXPL_DEPTH $EXPL_NOISE
  if [ "$TOTAL_PLAYED" -ge "$TARGET_GAMES" ]; then break; fi
  # After exploration, set START_PARAMS to the last best file
  START_PARAMS="$LOG_DIR/EXPLORE_best.yaml"
  run_phase REFINE $REF_ITERATIONS $REF_GAMES $REF_REPEATS $REF_DEPTH $REF_NOISE
  START_PARAMS="$LOG_DIR/REFINE_best.yaml"
done

log "Long run finished. Total games approx: $TOTAL_PLAYED"
log "Final best params: $START_PARAMS"

exit 0
