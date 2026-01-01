#!/usr/bin/env bash
set -euo pipefail
# Wrapper to run `self_play_long.sh` under `tmux` and keep a single live session
# Suitable to be invoked from a launchd plist. The script will:
# - create logs dir
# - if `tmux` is present, start a detached session named `skaks_selfplay` that
#   runs caffeinate + the self_play_long.sh pipeline and logs to logs/self_play_long.log
# - if tmux is not available, fall back to running in foreground (so launchd
#   can manage it) while still using caffeinate and tee to keep a single log file.

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

SESSION_NAME=skaks_selfplay
LOGFILE="$LOG_DIR/self_play_long.log"
PIDFILE="$LOG_DIR/self_play_long.pid"

CMD="/usr/bin/caffeinate -i sh -c './self_play_long.sh 2>&1 | tee -a \"$LOGFILE\"'"

start_tmux() {
    if ! command -v tmux >/dev/null 2>&1; then
        return 1
    fi
    # If session exists, do nothing (keeps a single progress display)
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "tmux session $SESSION_NAME already running"
        return 0
    fi
    # Start a new detached tmux session that runs our command. Use exec so the
    # program occupies the tmux pane and its stdout/stderr go to the log via tee.
    tmux new-session -d -s "$SESSION_NAME" "env TERM=xterm-256color bash -lc '$CMD'"
    sleep 1
    # Write the tmux session info as 'pid' (tmux doesn't expose child pid easily)
    echo "tmux:$SESSION_NAME" > "$PIDFILE"
    echo "Started tmux session $SESSION_NAME; logs -> $LOGFILE"
    return 0
}

start_foreground() {
    # Run foreground so launchd can keep it alive if configured. This blocks.
    echo "tmux not installed; running in foreground, logs -> $LOGFILE"
    # Save backgrounded child's pid to PIDFILE
    /usr/bin/caffeinate -i sh -c './self_play_long.sh 2>&1 | tee -a "'$LOGFILE'"' &
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
    ps aux | grep -E "self_play_long.sh|tuning.param_optimize" | grep -v grep || true
}

case ${1:-start} in
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
    echo "Usage: $0 {start|stop|status|attach}"
    exit 2
    ;;
esac
