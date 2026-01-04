#!/usr/bin/env python3
"""Generate a scatter plot of Skaks evaluations vs reference centipawns."""

from __future__ import annotations

import argparse
import math
import pathlib
import random
import sys
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TUNING_DIR = PROJECT_ROOT / "tuning"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(TUNING_DIR) not in sys.path:
    sys.path.insert(0, str(TUNING_DIR))

from eval_correlation import eval_fen, iter_dataset_rows, pearson
from skaks_opt.progress import FancyProgress, RICH_AVAILABLE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to CSV dataset file or directory containing CSV files",
    )
    parser.add_argument(
        "--skaks",
        default="build/debug/src/skaks",
        help="Path to the skaks executable",
    )
    parser.add_argument(
        "--output",
        default="perf_plots/eval_correlation.png",
        help="Path to the scatter plot image to write",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2000,
        help="Maximum number of positions to evaluate after sampling",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.2,
        help="Randomly sample this fraction of rows (0-1]; applied before limit",
    )
    parser.add_argument(
        "--clamp",
        type=float,
        default=1000.0,
        help="Clamp absolute centipawns when plotting and computing correlation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="PRNG seed for sampling",
    )
    parser.add_argument(
        "--title",
        default="Skaks vs reference centipawns",
        help="Plot title to display",
    )
    parser.add_argument(
        "--no-rich",
        action="store_true",
        help="Disable Rich progress even if the library is installed",
    )
    return parser.parse_args()


def sample_rows(
    dataset: pathlib.Path, fraction: float, limit: int, rng: random.Random
) -> Tuple[List[Tuple[str, float]], int]:
    rows: List[Tuple[str, float]] = []
    total_seen = 0
    fraction = float(max(0.0, min(1.0, fraction)))
    if fraction == 0.0:
        return rows, 0

    for fen, target in iter_dataset_rows(dataset):
        total_seen += 1
        if fraction < 1.0 and rng.random() > fraction:
            continue
        rows.append((fen, target))
        if 0 < limit == len(rows):
            break
    return rows, total_seen


def collect_evaluations(
    rows: Sequence[Tuple[str, float]],
    skaks: pathlib.Path,
    use_rich: bool,
    rng: random.Random,
) -> List[Tuple[float, float]]:
    results: List[Tuple[float, float]] = []
    if not rows:
        return results

    if RICH_AVAILABLE and use_rich:
        with FancyProgress(
            total=len(rows), rng=rng, description="Evaluating positions"
        ) as fancy:
            for idx, (fen, target) in enumerate(rows, start=1):
                value = eval_fen(skaks, fen)
                results.append((target, value))
                fancy.update(
                    idx,
                    message=f"[cyan]{idx}/{len(rows)}[/] positions",
                )
    else:
        for idx, (fen, target) in enumerate(rows, start=1):
            value = eval_fen(skaks, fen)
            results.append((target, value))
            if idx == 1 or idx == len(rows) or idx % 100 == 0:
                print(f"[{idx}/{len(rows)}] target={target:+.1f} eval={value:+.1f}")
    return results


def make_plot(
    samples: Sequence[Tuple[float, float]],
    clamp: float,
    title: str,
    output: pathlib.Path,
) -> Tuple[float, float, int]:
    if clamp > 0:
        bounded = [(t, v) for t, v in samples if abs(t) <= clamp and abs(v) <= clamp]
    else:
        bounded = list(samples)

    if not bounded:
        raise RuntimeError("No samples remain after applying clamp")

    expected = [t for t, _ in bounded]
    actual = [v for _, v in bounded]
    corr = pearson(expected, actual)

    plt.figure(figsize=(8, 6))
    plt.scatter(expected, actual, s=12, alpha=0.6, c="#1f77b4", edgecolors="none")
    plt.title(f"{title}\nPearson r = {corr:.3f} (|cp| ≤ {clamp:g})")
    plt.xlabel("Reference centipawns (white perspective)")
    plt.ylabel("Skaks centipawns (white perspective)")
    plt.grid(True, linestyle="--", alpha=0.25)

    # Best fit line
    mean_x = sum(expected) / len(expected)
    mean_y = sum(actual) / len(actual)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(expected, actual))
    den = sum((x - mean_x) ** 2 for x in expected)
    if den:
        slope = num / den
        intercept = mean_y - slope * mean_x
        xs = [min(expected), max(expected)]
        ys = [slope * x + intercept for x in xs]
        plt.plot(xs, ys, color="#ff7f0e", linewidth=1.5, label="Best fit")
        plt.legend()

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()
    return corr, pearson(*zip(*samples)) if samples else math.nan, len(bounded)


def main() -> None:
    args = parse_args()
    dataset = pathlib.Path(args.dataset)
    skaks = pathlib.Path(args.skaks)
    output = pathlib.Path(args.output)
    rng = random.Random(args.seed)

    rows, total_seen = sample_rows(dataset, args.sample_fraction, args.limit, rng)
    if not rows:
        raise SystemExit("No rows available after sampling")

    use_rich = not args.no_rich
    evaluations = collect_evaluations(rows, skaks, use_rich, rng)
    clamped_corr, full_corr, plotted = make_plot(
        evaluations, args.clamp, args.title, output
    )

    print(f"rows_scanned: {total_seen}")
    print(f"samples_evaluated: {len(evaluations)}")
    print(f"samples_plotted: {plotted}")
    print(f"pearson_full: {full_corr:.6f}")
    print(f"pearson_clamped: {clamped_corr:.6f}")
    print(f"saved_plot: {output}")


if __name__ == "__main__":
    main()
