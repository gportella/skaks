import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import skaks_eval as sk


def load_weights(path: Path):
    blob = sk.load_nnue_yaml(str(path))
    w1 = np.array(blob["w1"], dtype=np.float32)
    b1 = np.array(blob["b1"], dtype=np.float32)
    w2 = np.array(blob["w2"], dtype=np.float32)
    b2 = np.array([blob["b2"]], dtype=np.float32)
    hidden = int(blob.get("hidden", w2.shape[0]))
    return w1, b1, w2, b2, hidden


def sample_rows(
    csv_path: Path,
    fen_col: str,
    eval_col: str,
    eval_clip: float,
    max_rows: int,
    stride: int,
    start: int,
) -> List[Tuple[str, float]]:
    rows: List[Tuple[str, float]] = []
    idx = 0
    for df in pd.read_csv(csv_path, chunksize=200_000):
        if fen_col not in df.columns:
            raise ValueError(f"Missing fen column '{fen_col}' in {csv_path}")
        if eval_col not in df.columns:
            raise ValueError(f"Missing eval column '{eval_col}' in {csv_path}")
        for fen, ev in zip(df[fen_col].tolist(), df[eval_col].tolist()):
            if eval_clip > 0:
                ev = max(-eval_clip, min(eval_clip, ev))
            if idx >= start and (idx - start) % stride == 0:
                rows.append((fen, float(ev)))
                if len(rows) >= max_rows:
                    return rows
            idx += 1
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Validate NNUE regression vs CSV evals")
    p.add_argument("--weights", required=True, help="Path to nnue_weights.json/yaml")
    p.add_argument("--csv", required=True, help="CSV with fen/eval columns")
    p.add_argument("--fen-col", default="fen")
    p.add_argument("--eval-col", default="stockfish_cp")
    p.add_argument(
        "--eval-clip",
        type=float,
        default=0.0,
        help="Clamp true evals to +/- this many centipawns (0 = no clip)",
    )
    p.add_argument(
        "--max-rows", type=int, default=100, help="Number of positions to sample"
    )
    p.add_argument(
        "--stride", type=int, default=10_000, help="Stride when sampling rows"
    )
    p.add_argument(
        "--start", type=int, default=0, help="Start index before applying stride"
    )
    p.add_argument(
        "--scale-out",
        type=float,
        default=1.0,
        help="Multiply NNUE output by this (e.g., eval-target-scale to get cp)",
    )
    p.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Optional path to save predicted-vs-true scatter plot",
    )
    args = p.parse_args()

    w1, b1, w2, b2, hidden = load_weights(Path(args.weights))
    print(f"Loaded weights (hidden={hidden}) from {args.weights}")

    rows = sample_rows(
        csv_path=Path(args.csv),
        fen_col=args.fen_col,
        eval_col=args.eval_col,
        eval_clip=args.eval_clip,
        max_rows=args.max_rows,
        stride=args.stride,
        start=args.start,
    )
    if not rows:
        raise SystemExit("No rows sampled; adjust stride/start/max-rows")

    preds = []
    truths = []
    for fen, truth in rows:
        pred = sk.nnue_forward(w1, b1, w2, b2, fen=fen) * args.scale_out
        preds.append(pred)
        truths.append(truth)

    preds_arr = np.array(preds, dtype=np.float64)
    truths_arr = np.array(truths, dtype=np.float64)

    diff = preds_arr - truths_arr
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    ss_res = float(np.sum(diff**2))
    ss_tot = float(np.sum((truths_arr - truths_arr.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    corr = float(np.corrcoef(preds_arr, truths_arr)[0, 1])

    print(f"Samples: {len(rows)}")
    print(f"MAE:   {mae:.3f} cp")
    print(f"RMSE:  {rmse:.3f} cp")
    print(f"R^2:   {r2:.4f}")
    print(f"Corr:  {corr:.4f}")

    for i, (fen, truth, pred) in enumerate(zip(rows, truths_arr, preds_arr)):
        pass  # placeholder to keep line numbers stable; no per-row printing by default

    if args.plot:
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(6, 6))
            plt.scatter(truths_arr, preds_arr, s=10, alpha=0.6, label="samples")
            lo = min(truths_arr.min(), preds_arr.min())
            hi = max(truths_arr.max(), preds_arr.max())
            plt.plot([lo, hi], [lo, hi], "k--", label="ideal")
            plt.xlabel("True eval (cp)")
            plt.ylabel("Pred eval (cp)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(args.plot, dpi=160)
            print(f"Saved plot to {args.plot}")
        except ImportError:
            print("matplotlib not installed; skipping plot")


if __name__ == "__main__":
    main()
