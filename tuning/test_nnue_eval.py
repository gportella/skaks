import argparse
from pathlib import Path
from typing import List

import numpy as np
import skaks_eval as sk


def load_weights(path: Path):
    blob = sk.load_nnue_yaml(str(path))
    w1 = np.array(blob["w1"], dtype=np.float32)
    b1 = np.array(blob["b1"], dtype=np.float32)
    w2 = np.array(blob["w2"], dtype=np.float32)
    b2 = np.array([blob["b2"]], dtype=np.float32)
    hidden = int(blob.get("hidden", w2.shape[0]))
    return w1, b1, w2, b2, hidden


def collect_fens(args) -> List[str]:
    fens: List[str] = []
    if args.fen:
        fens.extend(args.fen)
    if args.fen_file:
        for line in Path(args.fen_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fens.append(line)
    if not fens:
        raise SystemExit("Provide at least one FEN via --fen or --fen-file")
    return fens


def main() -> None:
    p = argparse.ArgumentParser(description="Test NNUE weights using skaks_eval")
    p.add_argument("--weights", required=True, help="Path to nnue_weights.json/yaml")
    p.add_argument("--fen", nargs="*", help="FEN strings to evaluate")
    p.add_argument("--fen-file", help="File with one FEN per line")
    p.add_argument(
        "--scale-out",
        type=float,
        default=1.0,
        help="Multiply raw NNUE output by this (e.g., eval-target-scale to get cp)",
    )
    args = p.parse_args()

    w1, b1, w2, b2, hidden = load_weights(Path(args.weights))
    print(f"Loaded weights (hidden={hidden}) from {args.weights}")

    fens = collect_fens(args)

    for fen in fens:
        try:
            val = sk.nnue_forward(w1, b1, w2, b2, fen=fen)
            val_scaled = val * args.scale_out
            print(f"{val:.6f}\t{val_scaled:.6f}\t{fen}")
        except Exception as exc:  # pragma: no cover
            print(f"error\t{fen}\t{exc}")


if __name__ == "__main__":
    main()
