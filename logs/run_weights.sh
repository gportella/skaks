#!/usr/bin/env sh
set -e
exec 
/usr/bin/caffeinate -i python -u tuning/param_optimize.py --weights-only --start-params tuning/default_start.yaml --iterations 1 --repeats 1 --games 2 --concurrency 1 --depth 1 2>&1 | tee -a "/Users/ktxc111/work/repos/skaks/logs/weights_optimize.log"
