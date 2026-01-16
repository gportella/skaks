skaks-opt texel \
  --data ../datasets/lichess_elite_db_annotated \
  --base-params ../fitting_models/base_params_pst.yaml \
  --trials 100 \
  --param-set pst \
  --cp-cap 600 \
  --require-quiet \
  --max-stockfish-cp 600 \
  --max-qsearch-delta 15 \
  --texel-scale-min 100 \
  --texel-scale-max 700 \
  --error-penalty 12.0 \
  --pov white \
  --sampler tpe \
  --pruner hyperband \
  --jobs 8 \
  --threads 8 \
  --batch-size 1024 \
  --best-out texel_fit_all.yaml \
  --metrics-out texel_metrics.csv \
  --plot-out texel_loss.png \
  --calibration-out plots/texel_calib.png \
  --stockfish-correlation-out plots/stockfish_corr.png \
  --outcome-hist-out plots/outcome_hist.png \
  --progress-style fancy \
  --quiet \
  --progress-every 5

  #--base-params ../self_play/logs/EXPLORE_best.yaml \
