#!/usr/bin/env python3
"""Compute correlation between Skaks static evals and reference centipawns."""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import random
import subprocess
from itertools import cycle
from typing import Iterable, Iterator, List, Sequence, Tuple

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - rich is optional at runtime
    Console = None  # type: ignore
    Progress = None  # type: ignore
    RICH_AVAILABLE = False


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
        "--limit",
        type=int,
        default=200,
        help="Maximum number of rows to evaluate (default: 200)",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=1.0,
        help="Randomly sample this fraction of rows (0-1]; applied before limit",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="PRNG seed for sampling",
    )
    parser.add_argument(
        "--clamp",
        type=float,
        default=1000.0,
        help="Clamp absolute centipawns when computing the secondary metric",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Print progress every N samples (<=0 disables progress output)",
    )
    return parser.parse_args()


def iter_dataset_rows(path: pathlib.Path) -> Iterator[Tuple[str, float]]:
    if path.is_dir():
        csv_files = sorted(path.glob("*.csv"))
        for csv_path in csv_files:
            yield from iter_dataset_rows(csv_path)
        return

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            fen = row["fen"].strip()
            target = float(row["stockfish_cp"])
            yield fen, target


def load_rows(
    dataset: pathlib.Path, limit: int, fraction: float, rng: random.Random
) -> Tuple[List[Tuple[str, float]], int]:
    rows: List[Tuple[str, float]] = []
    fraction = float(max(0.0, min(1.0, fraction)))
    if fraction == 0.0:
        return rows, 0

    total_seen = 0
    for fen, target in iter_dataset_rows(dataset):
        total_seen += 1
        if fraction < 1.0 and rng.random() > fraction:
            continue
        rows.append((fen, target))
        if 0 < limit == len(rows):
            break
    return rows, total_seen


def eval_fen(skaks: pathlib.Path, fen: str) -> float:
    result = subprocess.run(
        [str(skaks), "--static-eval", "--fen", fen],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("static_eval_white "):
            return float(line.split()[1])
    raise RuntimeError(f"Unexpected engine output:\n{result.stdout}\n{result.stderr}")


def pearson(xs: Iterable[float], ys: Iterable[float]) -> float:
    x_list = list(xs)
    y_list = list(ys)
    if len(x_list) != len(y_list) or not x_list:
        return math.nan
    mean_x = sum(x_list) / len(x_list)
    mean_y = sum(y_list) / len(y_list)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_list, y_list))
    den_x = sum((x - mean_x) ** 2 for x in x_list)
    den_y = sum((y - mean_y) ** 2 for y in y_list)
    denom = math.sqrt(den_x * den_y)
    return num / denom if denom else math.nan


def main() -> None:
    args = parse_args()
    dataset_path = pathlib.Path(args.dataset)
    skaks_path = pathlib.Path(args.skaks)
    rng = random.Random(args.seed)

    rows, total_seen = load_rows(dataset_path, args.limit, args.sample_fraction, rng)
    if not rows:
        raise SystemExit("No rows available from dataset")

    evaluations: List[Tuple[float, float]] = []

    def run_with_plain_logging() -> None:
        interval = args.progress_interval
        for index, (fen, target) in enumerate(rows, start=1):
            value = eval_fen(skaks_path, fen)
            evaluations.append((target, value))
            if interval > 0 and (
                index == 1 or index == len(rows) or index % interval == 0
            ):
                print(f"[{index}/{len(rows)}] target={target:+.1f} eval={value:+.1f}")

    if RICH_AVAILABLE:
        console = Console()
        chess_frames = cycle(
            ["♔", "♕", "♖", "♗", "♘", "♙", "♚", "♛", "♜", "♝", "♞", "♟"]
        )
        progress_columns = [
            TextColumn("[bold magenta]{task.fields[chess]}"),
            SpinnerColumn(spinner_name="line", style="cyan"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None, style="green"),
            TextColumn("{task.completed}/{task.total}", style="bold cyan"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ]

        with Progress(*progress_columns, console=console, transient=True) as progress:
            task_id = progress.add_task(
                "Evaluating positions", total=len(rows), chess=next(chess_frames)
            )
            log_interval = (
                max(1, args.progress_interval) if args.progress_interval > 0 else 0
            )
            for index, (fen, target) in enumerate(rows, start=1):
                value = eval_fen(skaks_path, fen)
                evaluations.append((target, value))
                progress.update(
                    task_id,
                    advance=1,
                    chess=next(chess_frames),
                    description=f"[cyan]{index}/{len(rows)}[/] positions",
                )
                if log_interval and (index == len(rows) or index % log_interval == 0):
                    progress.log(
                        f"[white]{index}/{len(rows)}[/] target {target:+.1f} → eval {value:+.1f}"
                    )
    else:
        run_with_plain_logging()

    expected = [t for t, _ in evaluations]
    actual = [v for _, v in evaluations]
    corr_full = pearson(expected, actual)

    clamp = args.clamp
    if clamp > 0:
        filtered = [
            (t, v) for t, v in evaluations if abs(t) <= clamp and abs(v) <= clamp
        ]
    else:
        filtered = evaluations
    corr_clamped = pearson([t for t, _ in filtered], [v for _, v in filtered])

    print("rows_scanned", total_seen)
    print("samples", len(evaluations))
    print("pearson_full", corr_full)
    print("pearson_clamped", corr_clamped)


if __name__ == "__main__":
    main()
