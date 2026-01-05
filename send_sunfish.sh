#!/bin/bash
#SBATCH -J sk-v-sf                   # SLURM job name for this matchup controller
#SBATCH -N 1                          # Controller node count (runs the optimizer loop)
#SBATCH -t 12:00:00                   # Controller walltime budget (adjust to taste)

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$HOME/work/repos/skaks:$PYTHONPATH"

module load Clang                    # Or activate your virtual environment

ENGINE=skaks                         # Skaks engine binary
OPPONENT=sunfish                     # External opponent binary (Sunfish)
OPP_DEPTH_FACTOR=0.4                 # Scale depth for Sunfish when time controls use depth
DEPTH=4                              # Reference search depth for skaks candidates
BASELINE=phase_weights_optimized.yaml   # Baseline Skaks params (also used for opponent baseline eval)
START=phase_weights_optimized.yaml      # Starting candidate params
OUTPUT=tuning/best_params_vs_sunfish.yaml # Output YAML for the best params + .best.json meta
GAMES=20                             # Total games per candidate (split across colors)
ITERATIONS=100                       # Optimization iterations (candidate batches) to run
NOISE=0.05                           # Lognormal perturbation sigma for beam/CMA sampling
BEAM_SIZE=5                          # Beam width (number of parents kept each iteration)
REPEATS=2                            # Repeat evaluations per candidate (averaged for stability)
CONCURRENCY=2                        # Parallel games when using the batch runner (external mode)
JOBQUEUE=--dask-jobqueue             # Enable SLURM-backed Dask worker management
JOBQUEUE_CONFIG=tuning/slurm_cluster.yaml # SLURMCluster(**kwargs) config for worker jobs
JOBQUEUE_JOBS=16                     # Number of Dask worker jobs (arena shards) to request

skaks-opt param-optimize \
  --engine "$ENGINE" \
  --external-opponent \
  --opponent "$OPPONENT" \
  --opponent-depth-factor "$OPP_DEPTH_FACTOR" \
  --baseline-params "$BASELINE" \
  --start-params "$START" \
  --output "$OUTPUT" \
  --games "$GAMES" \
  --iterations "$ITERATIONS" \
  --noise "$NOISE" \
  --beam-size "$BEAM_SIZE" \
  --repeats "$REPEATS" \
  --concurrency "$CONCURRENCY" \
  --depth "$DEPTH" \
  $JOBQUEUE \
  --dask-jobqueue-config "$JOBQUEUE_CONFIG" \
  --dask-jobqueue-jobs "$JOBQUEUE_JOBS" \
  --child-output
