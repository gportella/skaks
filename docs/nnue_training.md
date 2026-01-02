# NNUE feature export and training notes

This engine now exposes a minimal NNUE-style feature extractor and reference network so you can prototype training in Python and later re-use weights in C++.

## Feature extraction

- Function: `skaks_eval.features_from_fen(fen: str) -> numpy.ndarray[int8]`
- Shape: `(1537,)` with values in {0, 1}
  - Layout: 12 piece kinds × 64 squares × 2 king buckets = 1536 bits, plus one side-to-move bit at the end.
  - King buckets are split: white-king-relative bucket first, black-king-relative bucket second.
- Example:

```python
import numpy as np
from skaks_eval import features_from_fen

fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
feat = features_from_fen(fen)
assert feat.shape == (1537,)
```

## Reference network shape

The C++ helper `chess::NnueNetwork` expects:

- `w1`: flattened row-major `[hidden, input]` (float)
- `b1`: `[hidden]`
- `w2`: `[hidden]`
- `b2`: scalar

The forward pass applies a ReLU hidden layer and a single linear output. Input size is fixed to 1537; hidden size is inferred from `b1.size()`.

## One-shot trainer (Python)

Use `tuning/nnue_train.py` to train and export weights in the engine format (quantized int8/int32 with optional `scale`). It reads CSVs like `eval_pairs_pvs_with_results.csv` (fen + outcome) or `eval_pairs_pvs.csv` (fen + eval_cp).

Example (outcome labels already in [0, 1]):

```bash
python tuning/nnue_train.py --data eval_pairs_pvs_with_results.csv --out nnue_weights.json
```

Example (centipawn evals converted to probability, scaled by 600, POV = side_to_move if present):

```bash
python tuning/nnue_train.py \
  --data eval_pairs_pvs.csv \
  --eval-col eval_cp \
  --cp-scale 600 \
  --pov side \
  --out nnue_weights.json
```

Flags of interest: `--hidden` (default 256), `--epochs`, `--batch-size`, `--lr`, `--val-split`, `--cache-features`, `--scale` (post-int scaling). Device preference: CUDA, else MPS on macOS, else CPU.

Output schema (consumed by engine `--nnue` and `skaks_eval.load_nnue_yaml`):

```json
{
  "nnue": {
    "hidden": 256,
    "w1": [int8...],
    "b1": [int32...],
    "w2": [int8...],
    "b2": int32,
    "scale": 1.0
  }
}
```

## Dataset tips

- If using engine-eval-derived outcomes, keep them in [0, 1] and consider label smoothing to reduce overconfidence.
- De-duplicate positions and enforce minimum ply to avoid trivial book openings (use `tuning/filter_texel_dataset.py`).
- Shuffle data and monitor loss on a held-out split; prefer large batch sizes for stable gradients.

## Next steps

- Add a Python binding to accept serialized weights and call the C++ `NnueNetwork` forward for evaluation parity.
- Wire a loader in the engine that reads on-disk weights and swaps them into the search stack.
