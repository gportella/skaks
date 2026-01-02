#!/usr/bin/env bash
# Simple launchd/tmux wrapper to run the weights-only parameter optimizer
# Usage: scripts/weights_optimize_launchd_wrapper.sh [--tmux-session NAME] [--] [<param_optimize args>]
# By default runs: skaks-opt param-optimize --weights-only --start-params tuning/default_start.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

TMUX_BIN=$(command -v tmux || true)
CAFFEINATE_BIN=$(command -v caffeinate || true)

SESSION_NAME=skaks_weights
LOGFILE="$LOG_DIR/weights_optimize.log"
PIDFILE="$LOG_DIR/weights_optimize.pid"

# Parse simple wrapper flags; remaining args are forwarded to the optimizer
while [[ $# -gt 0 ]]; do
  case ${1:-} in
    start|stop|status|attach)
      ACTION="$1"; shift; break;;
    --tmux-session)
      SESSION_NAME="$2"; shift 2;;
    --)
      shift; break;;
    *)
      break;;
  esac
done

EXTRA_ARGS=("$@")
if [[ ${#EXTRA_ARGS[@]} -eq 0 ]]; then
  EXTRA_ARGS=("--weights-only" "--start-params" "tuning/default_start.yaml")
fi

# If the caller provided a leading '--' separator, drop it so it's not
# forwarded as a literal argument to the optimizer script.
if [[ ${#EXTRA_ARGS[@]} -gt 0 && "${EXTRA_ARGS[0]}" == "--" ]]; then
        EXTRA_ARGS=("${EXTRA_ARGS[@]:1}")
fi

# Use unbuffered python output so logs appear promptly
PY_CMD=("python" "-u" "-m" "skaks_opt.cli" "param-optimize" "${EXTRA_ARGS[@]}")

# Build the shell line that caffeinate will run (redirect to tee)
RUN_SH_LINE="${PY_CMD[@]} 2>&1 | tee -a \"$LOGFILE\""

start_tmux() {
    if ! command -v tmux >/dev/null 2>&1; then
        return 1
    fi
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "tmux session $SESSION_NAME already running"
        return 0
    fi
    # Create a small run script in the log dir to avoid nested quoting issues
    RUNFILE="$LOG_DIR/run_weights.sh"
    # Build the runtime argv array; include caffeinate in front if available
    RUN_ARR=()
    if [[ -n "$CAFFEINATE_BIN" ]]; then
        RUN_ARR+=("$CAFFEINATE_BIN" "-i")
    fi
    RUN_ARR+=("${PY_CMD[@]}")

    # Write a simple exec script with shell-escaped args so no sh -c quoting is needed
    printf '%s\n' '#!/usr/bin/env sh' 'set -e' 'exec ' > "$RUNFILE"
    for idx in "${!RUN_ARR[@]}"; do
        a="${RUN_ARR[$idx]}"
        printf '%s ' "$(printf '%q' "$a")" >> "$RUNFILE"
    done
    printf '2>&1 | tee -a "%s"\n' "$LOGFILE" >> "$RUNFILE"
    chmod +x "$RUNFILE"
    tmux new-session -d -s "$SESSION_NAME" "env TERM=xterm-256color bash -lc '$RUNFILE'"
    sleep 1
    echo "tmux:$SESSION_NAME" > "$PIDFILE"
    echo "Started tmux session $SESSION_NAME; logs -> $LOGFILE"
    return 0
}

start_foreground() {
    echo "tmux not installed; running in foreground, logs -> $LOGFILE"
    if [[ -n "$CAFFEINATE_BIN" ]]; then
        /usr/bin/env $CAFFEINATE_BIN -i sh -c "$RUN_SH_LINE" &
    else
        /usr/bin/env sh -c "$RUN_SH_LINE" &
    fi
    child=$!
    echo "$child" > "$PIDFILE"
    wait $child
}

stop_tmux() {
    if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        tmux kill-session -t "$SESSION_NAME" || true
        rm -f "$PIDFILE" || true
        echo "Stopped tmux session $SESSION_NAME"
    else
        echo "No tmux session $SESSION_NAME found"
    fi
}

status() {
    if [ -f "$PIDFILE" ]; then
        echo "PIDFILE: $(cat "$PIDFILE")"
    fi
    if command -v tmux >/dev/null 2>&1; then
        tmux ls 2>/dev/null || echo "no tmux sessions"
    fi
    ps aux | grep -E "param-optimize|skaks_opt" | grep -v grep || true
}

case ${ACTION:-start} in
start)
    if start_tmux; then
        exit 0
    else
        start_foreground
    fi
    ;;
stop)
    stop_tmux
    ;;
status)
    status
    ;;
attach)
    if command -v tmux >/dev/null 2>&1; then
        tmux attach -t "$SESSION_NAME" || true
    else
        echo "tmux not found"
    fi
    ;;
*)
    echo "Usage: $0 {start|stop|status|attach} [--tmux-session NAME] [-- <optimizer-args>]"
    exit 2
    ;;
esac

