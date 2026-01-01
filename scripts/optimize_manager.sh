#!/usr/bin/env bash
# Manage optimizer runs with tmux + caffeinate
# Usage: optimize_manager.sh <start|stop|status|tail|runonce> [OPTIONS]

set -euo pipefail

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
SESSION_NAME="skaks-optimize"
LOG_DIR="$WORKDIR/logs"
mkdir -p "$LOG_DIR"

DEFAULT_CMD=(python "$WORKDIR/tuning/param_optimize.py" --weights-only --iterations 50 --games 40)

start() {
  # Any extra args passed after 'start' are forwarded to the optimizer
  # (e.g. --start-params path/to/params.yaml)
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session $SESSION_NAME already running"
    exit 0
  fi
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  LOG="$LOG_DIR/optimize_$TIMESTAMP.log"
  EXTRA_ARGS="$*"
  # If caller passed a leading '--' as a separator, remove it so the
  # optimizer doesn't receive a stray '--' argument.
  if [ "${EXTRA_ARGS# }" != "" ]; then
    set -- $EXTRA_ARGS
    if [ "${1-}" = "--" ]; then
      shift
    fi
    EXTRA_ARGS="$*"
  fi
  # Build a shell command string so extra args are passed through cleanly.
  CMD_STR="caffeinate -dims python '$WORKDIR/tuning/param_optimize.py' --weights-only --iterations 50 --games 40 $EXTRA_ARGS"
  tmux new-session -d -s "$SESSION_NAME" bash -lc "cd '$WORKDIR' && $CMD_STR 2>&1 | tee '$LOG'"
  echo "Started optimizer in tmux session $SESSION_NAME; log=$LOG"
}

stop() {
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    tmux kill-session -t "$SESSION_NAME"
    echo "Stopped session $SESSION_NAME"
  else
    echo "No session $SESSION_NAME"
  fi
}

status() {
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session $SESSION_NAME: running"
    tmux list-panes -t "$SESSION_NAME"
  else
    echo "Session $SESSION_NAME: not running"
  fi
}

tail_log() {
  latest=$(ls -1t "$LOG_DIR"/optimize_*.log 2>/dev/null | head -n1 || true)
  if [ -n "$latest" ]; then
    tail -n +1 -f "$latest"
  else
    echo "No optimizer logs found"
  fi
}

runonce() {
  # Run once in foreground (useful for testing); forward extra args
  EXTRA_ARGS="$*"
  if [ "${EXTRA_ARGS# }" != "" ]; then
    set -- $EXTRA_ARGS
    if [ "${1-}" = "--" ]; then
      shift
    fi
    EXTRA_ARGS="$*"
  fi
  CMD_STR="python '$WORKDIR/tuning/param_optimize.py' --weights-only --iterations 50 --games 40 $EXTRA_ARGS"
  (cd "$WORKDIR" && eval $CMD_STR)
}

usage() {
  cat <<EOF
Usage: $0 <start|stop|status|tail|runonce> [-- <optimizer-args>]

Commands:
  start    Start optimizer in tmux with caffeinate (pass optimizer args after 'start')
  stop     Stop the tmux session
  status   Show tmux session status
  tail     Tail latest optimizer log
  runonce  Run optimizer once in foreground (pass optimizer args after 'runonce')
EOF
}

if [ $# -lt 1 ]; then
  usage
  exit 1
fi

case "$1" in
  start) shift; start "$@" ;;
  stop) shift; stop "$@" ;;
  status) shift; status "$@" ;;
  tail) shift; tail_log "$@" ;;
  runonce) shift; runonce "$@" ;;
  *) usage ; exit 1 ;;
esac
