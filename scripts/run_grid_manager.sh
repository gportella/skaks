#!/usr/bin/env bash
set -euo pipefail

WORKDIR="logs"
SESSION_NAME="skaks-grid-manager"
PYTHON=${PYTHON:-python}
mkdir -p "$WORKDIR"

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 [--dry-run] --start tuning/default_start.yaml [other args passed to grid_run_manager.py]"
  exit 1
fi

LOGFILE="$WORKDIR/grid_manager.log"
CMD="$PYTHON scripts/grid_run_manager.py $*"

echo "Starting grid manager in tmux session $SESSION_NAME; logs: $LOGFILE"

# create a small runfile to ensure quoting handled by tmux
RUNFILE="$WORKDIR/grid_manager_run.sh"
cat > "$RUNFILE" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$(pwd)"
echo "+ $CMD"
exec $CMD
EOF
chmod +x "$RUNFILE"

# start tmux session detached
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
tmux new-session -d -s "$SESSION_NAME" "$RUNFILE" 

echo "Launched. To attach: tmux attach -t $SESSION_NAME"
echo "Logs will be written to $LOGFILE by the manager script (if used)"
