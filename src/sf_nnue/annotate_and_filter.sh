echo "Annotated validation moves"
python validation_moves/add_outcomes_from_pgn.py \
  moves_pgn/LumbrasGigaBase_OTB_2025.pgn eval_pairs_pvs.csv \
  --output eval_pairs_pvs_with_results.csv
echo "Now filtering texel"
python tuning/filter_texel_dataset.py \  !13272
eval_pairs_pvs_with_results.csv \
  tuning/eval_pairs_eval_scale800.csv \
  --max-per-game 512 \
  --max-per-bucket 100000 \
  --min-ply 8 \
  --replace-outcome-with-eval \
  --eval-scale 800 \
  --seed 42
