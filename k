skaks-opt arena-sweep \
  --candidates "tuning/best_params_*.yaml" \
  --games 240 \
  --depth 4 \
  --engine skaks \
  --opponent sunfish \
  --opponent-depth-factor 0.4 \
  --best-out eval_params/best_vs_sunfish.yaml \
  --best-summary eval_params/best_vs_sunfish.summary.json
