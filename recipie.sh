#! /usr/bin/env bash
# 1) Quiet-only Texel fit; writes filtered params for later runs
skaks-opt texel \
  --data tuning/datasets/lichess_elite \
  --require-quiet \
  --base-params tuning/regressed_phase_weights.yaml \
  --trials 250 \
  --jobs 4 \
  --threads 8 \
  --limit 1000000 \
  --quiet \
  --metrics-out tuning/texel_metrics.csv \
  --plot-out tuning/texel_loss.png

echo
echo "DONE filtered"
echo "DONE filtered"
echo ""
sleep 5
# 2) Use the quiet YAML as the optimizer baseline, freeze eval_* scalers, tune phase weights only
skaks-opt param-optimize \
  --engine skaks \
  --baseline-params tuning/texel_fit_quiet.yaml \
  --start-params tuning/texel_fit_quiet.yaml \
  --output tuning/phase_weights_optimized.yaml \
  --games 160 \
  --iterations 10 \
  --noise 0.03 \
  --beam-size 2 \
  --repeats 1 \
  --concurrency 6 \
  --time-per-move 0.20 \
  --use-arena-binding \
  --arena-workers 2 \
  --include-prefix evaluation.phase_weights_mg \
  --include-prefix evaluation.phase_weights_eg \
  --exclude-prefix evaluation.eval_

echo
echo "DONE freeze"
echo "DONE fee-fee-fee-freeze"
echo ""
sleep 5

# 3) Optional follow-up: focus on passed-pawn terms only with the same quiet baseline
skaks-opt param-optimize \
  --engine skaks \
  --baseline-params tuning/phase_weights_optimized.yaml \
  --start-params tuning/phase_weights_optimized.yaml \
  --output tuning/passed_pawn_optimized.yaml \
  --games 160 \
  --iterations 8 \
  --noise 0.02 \
  --beam-size 2 \
  --repeats 1 \
  --concurrency 6 \
  --time-per-move 0.20 \
  --use-arena-binding \
  --arena-workers 2 \
  --include-prefix evaluation.passed_pawn_ \
  --exclude-prefix evaluation.eval_

echo
echo "DONE"
echo "DONE"
echo ""
sleep 5
