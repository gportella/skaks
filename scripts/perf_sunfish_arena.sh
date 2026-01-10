#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: perf_sunfish_arena.sh [options]

Options:
  -g, --games N     Number of games per run (default: 20)
  -l, --limit N     Ply limit passed to skaks-opt arena (default: 200)
  --blitz           Run an additional blitz pass (time-per-move mode)
  --blitz-movetime S  Seconds per move for blitz pass (default: 1.0)
  -h, --help        Show this help and exit
EOF
}

GAMES=30
LIMIT=200
DO_BLITZ=0
BLITZ_MOVETIME=1.0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -g|--games)
      GAMES="$2"
      shift 2
      ;;
    -l|--limit)
      LIMIT="$2"
      shift 2
      ;;
    --blitz)
      DO_BLITZ=1
      shift
      ;;
    --blitz-movetime)
      BLITZ_MOVETIME="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

ENGINE_BIN=${SKAKS_BIN:-skaks}
ARENA_BIN=${SKAKS_OPT_BIN:-skaks-opt}
OPPONENT_BIN=${SUNFISH_BIN:-sunfish}
OUTPUT_CSV=${SUNFISH_PERF_CSV:-perf_sunfish_runs.csv}

timestamp=$(date -Iseconds)
engine_version=""

csv_escape() {
  local data=${1//$'\r'/}
  data=${data//$'\n'/\\n}
  data=${data//\"/\"\"}
  printf '%s' "$data"
}

ensure_csv() {
  local dir
  dir=$(dirname "$OUTPUT_CSV")
  mkdir -p "$dir"
  if [[ ! -f "$OUTPUT_CSV" ]]; then
    printf 'timestamp,version,control_type,control_value,white_label,black_label,wins,draws,losses,failures,total\n' \
      >>"$OUTPUT_CSV"
  fi
}

append_result_row() {
  ensure_csv
  local mode="$1" value="$2" wins="$3" draws="$4" losses="$5" fail_n="$6" total="$7"
  local failures="${fail_n}/${total}"
  local version_value=$(csv_escape "$engine_version")
  printf '"%s","%s","%s","%s","skaks","sunfish","%s","%s","%s","%s","%s"\n' \
    "$timestamp" "$version_value" "$mode" "$value" "$wins" "$draws" "$losses" "$failures" "$total" >>"$OUTPUT_CSV"
}

show_progress() {
  local mode="$1"
  local label="$2"
  local total="${3:-$GAMES}"
  local done=0
  local line
  printf '[%s %s] %s/%s' "$mode" "$label" "$done" "$total"
  while IFS= read -r line; do
    if [[ $line =~ ^\[([0-9]+)\/([0-9]+)\] ]]; then
      done="${BASH_REMATCH[1]}"
      total="${BASH_REMATCH[2]}"
      printf '\r[%s %s] %s/%s' "$mode" "$label" "$done" "$total"
    fi
  done
  printf '\r[%s %s] %s/%s done\n' "$mode" "$label" "$done" "$total"
}

extract_version_number() {
  local text="$1" line
  while IFS= read -r line; do
    if [[ $line =~ [Vv]ersion[[:space:]]+([0-9]+(\.[0-9]+)*) ]]; then
      printf '%s' "${BASH_REMATCH[1]}"
      return
    fi
  done <<<"$text"
  printf '%s' "${text%%$'\n'*}"
}

engine_version_output=""
if command -v "$ENGINE_BIN" >/dev/null 2>&1; then
  if engine_version_output=$("$ENGINE_BIN" -vv 2>&1); then
    engine_version=$(extract_version_number "$engine_version_output")
  else
    printf 'WARNING: failed to run %s -vv\n' "$ENGINE_BIN" >&2
  fi
else
  printf -- 'WARNING: engine binary %s not found on PATH.\n' "$ENGINE_BIN" >&2
fi

run_clock_pass() {
  local clock="${1:-}"
  if [[ -z "$clock" ]]; then
    printf 'clock value missing for run_clock_pass\n' >&2
    exit 1
  fi
  printf 'Running %s games @ %ss...\n' "$GAMES" "$clock"
  local cmd=(
    "$ARENA_BIN" arena
    --engine "$ENGINE_BIN"
    --opponent "$OPPONENT_BIN"
    --engine-label skaks
    --opponent-label sunfish
    --games "$GAMES"
    --clock "$clock"
    --opponent-clock "$clock"
    --increment 0.001
    --opponent-increment 0.001
    --limit "$LIMIT"
    --concurrency 2
  )
  local tmp
  tmp=$(mktemp)
  if ! "${cmd[@]}" 2>&1 | tee "$tmp" | show_progress "clock" "${clock}s" "$GAMES"; then
    printf '\nRun failed:\n' >&2
    cat "$tmp" >&2
    rm -f "$tmp"
    exit 1
  fi
  summarize_run "clock" "$clock" "$tmp"
  rm -f "$tmp"
}

run_blitz_pass() {
  local movetime="${1:-}"
  if [[ -z "$movetime" ]]; then
    printf 'movetime value missing for run_blitz_pass\n' >&2
    exit 1
  fi
  printf 'Running blitz %s games @ %ss per move...\n' "$GAMES" "$movetime"
  local cmd=(
    "$ARENA_BIN" arena
    --engine "$ENGINE_BIN"
    --opponent "$OPPONENT_BIN"
    --engine-label skaks
    --opponent-label sunfish
    --games "$GAMES"
    --time-per-move "$movetime"
    --opponent-time-per-move "$movetime"
    --limit "$LIMIT"
    --concurrency 2
  )
  local tmp
  tmp=$(mktemp)
  if ! "${cmd[@]}" 2>&1 | tee "$tmp" | show_progress "blitz" "${movetime}s" "$GAMES"; then
    printf '\nRun failed:\n' >&2
    cat "$tmp" >&2
    rm -f "$tmp"
    exit 1
  fi
  summarize_run "blitz" "$movetime" "$tmp"
  rm -f "$tmp"
}

summarize_run() {
  local mode="$1"
  local value="$2"
  local file="$3"
  local summary_line="" failures_line=""
  summary_line=$(grep '^Summary:' "$file" | tail -1 || true)
  failures_line=$(grep '^Failures:' "$file" | tail -1 || true)
  if [[ -z "$summary_line" ]]; then
    printf 'Summary line missing in %s\n' "$file" >&2
    exit 1
  fi
  local skaks_wins sunfish_wins draws fail_n fail_d tmp_line
  skaks_wins=$(sed -E 's/.*skaks=([0-9]+).*/\1/' <<<"$summary_line")
  sunfish_wins=$(sed -E 's/.*sunfish=([0-9]+).*/\1/' <<<"$summary_line")
  draws=$(sed -E 's/.*draw=([0-9]+).*/\1/' <<<"$summary_line")
  fail_n=0
  fail_d="$GAMES"
  if [[ -n "$failures_line" ]]; then
    tmp_line=${failures_line#Failures: }
    tmp_line=${tmp_line// /}
    if [[ $tmp_line =~ ^([0-9]+)\/([0-9]+)$ ]]; then
      fail_n="${BASH_REMATCH[1]}"
      fail_d="${BASH_REMATCH[2]}"
    fi
  fi
  printf '[%s %s] skaks W=%s D=%s L=%s failures=%s/%s\n' \
    "$mode" "$value" "$skaks_wins" "$draws" "$sunfish_wins" "$fail_n" "$fail_d"
  append_result_row "$mode" "$value" "$skaks_wins" "$draws" \
    "$sunfish_wins" "$fail_n" "$fail_d"
}

run_clock_pass 5
run_clock_pass 10

if [[ "$DO_BLITZ" -eq 1 ]]; then
  run_blitz_pass "$BLITZ_MOVETIME"
fi
