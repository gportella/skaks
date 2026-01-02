Weights optimizer wrapper

Use scripts/weights_optimize_launchd_wrapper.sh to run the weights-only optimizer in a tmux session.

Examples:

- Start in background tmux session named skaks-weights (default):
  scripts/weights_optimize_launchd_wrapper.sh

- Start with a custom tmux session name and pass extra args:
  scripts/weights_optimize_launchd_wrapper.sh --tmux-session myrun -- --max-iters 50 --repeats 3

This wrapper will use `caffeinate` on macOS if available to keep the machine awake.

Grid evaluation helper

- Evaluate a coarse grid of scale factors for the phase-weight arrays with:
  scripts/grid_weights_eval.py --start tuning/default_start.yaml --scales 0.5,0.75,1.0,1.25,1.5 --games 50

  The script writes `grid_results.csv` with wins/losses/draws per scale pair.
