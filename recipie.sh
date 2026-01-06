#! /usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# 1) Quiet-only Texel fit with cp cap + wider scale search to soften near-zero
skaks-opt texel \
  --data tuning/datasets/lichess_elite \
  \
  --cp-cap 400 \
  --texel-scale-min 120 \
  --texel-scale-max 800 \
  --error-penalty 12 \
  --param-set phase \
  --include-arrays \
  --base-params tuning/phase_weights_optimized.yaml \
  --trials 400 \
  --jobs 8 \
  --threads 8 \
  --batch-size 1024 \
  --limit 1500000 \
  --best-out tuning/texel_fit_quiet.yaml \
  --metrics-out tuning/texel_metrics.csv \
  --plot-out tuning/texel_loss.png \
  --quiet #--require-quiet \
#--quiet-batch 2048 \

echo -e "\nDONE texel quiet\n"
sleep 2

# 2) Self-play polish: allow eval scalers + phase weights to settle
skaks-opt param-optimize \
  --engine skaks \
  --baseline-params tuning/texel_fit_quiet.yaml \
  --start-params tuning/texel_fit_quiet.yaml \
  --output tuning/phase_weights_optimized.yaml \
  --games 80 \
  --iterations 12 \
  --noise 0.05 \
  --beam-size 4 \
  --repeats 2 \
  --depth 4 \
  --use-arena-binding \
  --arena-workers 2 \
  --baseline-decay 0.003 \
  --force-accept-first 3 \
  --min-score 0.55 \
  --include-prefix evaluation.phase_weights_mg \
  --include-prefix evaluation.phase_weights_eg \
  --include-prefix evaluation.eval_ \
  --child-output

echo -e "\nDONE polish\n"
