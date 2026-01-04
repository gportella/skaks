#!/usr/bin/env python3
"""Generate scatter plot comparing Stockfish target CPs with Skaks evaluations."""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import subprocess
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt


@dataclass
class Sample:
    fen: str
    stockfish_cp: float
    skaks_cp: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="tests/data/nnue_validation_sample.csv",
        help="CSV file containing validation samples",
    )
    parser.add_argument(
        "--skaks",
        default="build/debug/src/skaks",
        help="Path to the skaks executable (built with static-eval support)",
    )
    parser.add_argument(
        "--nnue",
        default="nn-c0ae49f08b40.nnue",
        help="Path to Stockfish-format NNUE file",
    )
    parser.add_argument(
        "--output",
        default="perf_plots/nnue_correlation.png",
        help="Destination image path for the scatter plot",
    )
    parser.add_argument(
        "--eval-mode",
        default="stockfish",
        choices=["stockfish", "native"],
        help="Evaluation mode to request from skaks",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of rows to evaluate",
    )
    return parser.parse_args()


def read_dataset(path: pathlib.Path) -> List[tuple[str, float]]:
    rows: List[tuple[str, float]] = []
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None or len(header) < 6:
            raise ValueError("Validation dataset missing expected columns")
        for line in reader:
            if not line:
                continue
            fen = line[3].strip()
            target = float(line[5])
            rows.append((fen, target))
    return rows


def evaluate_fen(
    skaks: pathlib.Path, nnue: pathlib.Path, eval_mode: str, fen: str
) -> float:
    cmd = [
        str(skaks),
        "--eval",
        eval_mode,
        "--nnue",
        str(nnue),
        "--static-eval",
        "--fen",
        fen,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    for line in result.stdout.splitlines():
        if line.startswith("static_eval_white "):
            return float(line.split()[1])
    raise RuntimeError(
        f"Failed to parse evaluation output:\n{result.stdout}\n{result.stderr}"
    )


def collect_samples(
    dataset: Sequence[tuple[str, float]],
    skaks: pathlib.Path,
    nnue: pathlib.Path,
    eval_mode: str,
) -> List[Sample]:
    samples: List[Sample] = []
    for idx, (fen, target) in enumerate(dataset, start=1):
        value = evaluate_fen(skaks, nnue, eval_mode, fen)
        samples.append(Sample(fen=fen, stockfish_cp=target, skaks_cp=value))
        print(f"[{idx}/{len(dataset)}] {value:+.1f} cp (target {target:+.1f})")
    return samples


def pearson(xs: Iterable[float], ys: Iterable[float]) -> float:
    x_list = list(xs)
    y_list = list(ys)
    if len(x_list) != len(y_list) or not x_list:
        return math.nan
    n = len(x_list)
    mean_x = sum(x_list) / n
    mean_y = sum(y_list) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_list, y_list))
    denom_x = sum((x - mean_x) ** 2 for x in x_list)
    denom_y = sum((y - mean_y) ** 2 for y in y_list)
    denom = math.sqrt(denom_x * denom_y)
    return num / denom if denom else math.nan


def plot_samples(samples: Sequence[Sample], output_path: pathlib.Path) -> None:
    bounded: List[Sample] = [
        s for s in samples if abs(s.stockfish_cp) <= 1000 and abs(s.skaks_cp) <= 1000
    ]
    if not bounded:
        raise RuntimeError("No samples remain after applying ±1000 cp clamp")

    expected = [s.stockfish_cp for s in bounded]
    actual = [s.skaks_cp for s in bounded]
    corr = pearson(expected, actual)

    plt.figure(figsize=(8, 6))
    plt.scatter(expected, actual, c="#1f77b4", edgecolors="white", linewidths=0.5)
    plt.title(f"Skaks vs Stockfish NNUE (r = {corr:.3f})")
    plt.xlabel("Stockfish centipawns (white perspective)")
    plt.ylabel("Skaks centipawns (white perspective)")
    plt.grid(True, linestyle="--", alpha=0.3)

    # Best-fit line for reference
    mean_x = sum(expected) / len(expected)
    mean_y = sum(actual) / len(actual)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(expected, actual))
    den = sum((x - mean_x) ** 2 for x in expected)
    if den:
        slope = num / den
        intercept = mean_y - slope * mean_x
        xs = [min(expected), max(expected)]
        ys = [slope * x + intercept for x in xs]
        plt.plot(xs, ys, color="#ff7f0e", linestyle="-", linewidth=1, label="Best fit")
        plt.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    print(f"Saved plot to {output_path}")


def main() -> None:
    args = parse_args()
    dataset_path = pathlib.Path(args.dataset)
    skaks_path = pathlib.Path(args.skaks)
    nnue_path = pathlib.Path(args.nnue)
    output_path = pathlib.Path(args.output)

    rows = read_dataset(dataset_path)
    if args.limit is not None:
        rows = rows[: args.limit]
        print(f"Limiting to first {len(rows)} samples")

    samples = collect_samples(rows, skaks_path, nnue_path, args.eval_mode)
    plot_samples(samples, output_path)


if __name__ == "__main__":
    main()
