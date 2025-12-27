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
    stm_col: str,
    pov_side: bool,
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
        if pov_side and stm_col not in df.columns:
            raise ValueError(f"pov_side requested but missing stm column '{stm_col}'")
        if pov_side:
            stm = (
                df[stm_col]
                .astype(str)
                .str.lower()
                .str.startswith("w")
                .map({True: 1, False: -1})
            )
            vals = df[eval_col].astype(float) * stm
        else:
            vals = df[eval_col].astype(float)
        for fen, ev in zip(df[fen_col].tolist(), vals.tolist()):
            if idx >= start and (idx - start) % stride == 0:
                rows.append((fen, float(ev)))
                if len(rows) >= max_rows:
                    return rows
            idx += 1
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Quick NNUE fit check vs CSV evals")
    p.add_argument(
        "--weights",
        required=False,
        default=None,
        help="Path to nnue_weights.json/yaml (omit if using --fp32)",
    )
    p.add_argument(
        "--fp32",
        type=str,
        default=None,
        help="Optional fp32 state_dict .pt; overrides weights",
    )
    p.add_argument("--csv", required=True, help="CSV with fen/eval columns")
    p.add_argument("--fen-col", default="fen")
    p.add_argument("--eval-col", default="stockfish_cp")
    p.add_argument(
        "--stm-col", default="side_to_move", help="Column with side to move (w/b)"
    )
    p.add_argument(
        "--pov-side", action="store_true", help="Multiply eval by stm for side POV"
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
        help="Multiply NNUE output by this (e.g., eval-target-scale)",
    )
    p.add_argument(
        "--hidden",
        type=int,
        default=None,
        help="Hidden size override for fp32 model",
    )
    args = p.parse_args()

    if args.fp32:
        import torch

        state = torch.load(args.fp32, map_location="cpu")
        hidden = args.hidden or state["fc1.weight"].shape[0]
        w1 = state["fc1.weight"].cpu().numpy()
        b1 = state["fc1.bias"].cpu().numpy()
        w2 = state["fc2.weight"].cpu().numpy().reshape(hidden)
        b2 = state["fc2.bias"].cpu().numpy()
        print(f"Loaded fp32 state_dict (hidden={hidden}) from {args.fp32}")
    else:
        w1, b1, w2, b2, hidden = load_weights(Path(args.weights))
        print(f"Loaded weights (hidden={hidden}) from {args.weights}")
    print(f"w1 max|abs|={np.abs(w1).max():.3f}, mean|abs|={np.abs(w1).mean():.3f}")
    print(f"w2 max|abs|={np.abs(w2).max():.3f}, mean|abs|={np.abs(w2).mean():.3f}")
    print(f"b1 max|abs|={np.abs(b1).max():.3f}, mean|abs|={np.abs(b1).mean():.3f}")
    print(f"b2={b2[0]:.3f}")

    rows = sample_rows(
        csv_path=Path(args.csv),
        fen_col=args.fen_col,
        eval_col=args.eval_col,
        stm_col=args.stm_col,
        pov_side=args.pov_side,
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


if __name__ == "__main__":
    main()
