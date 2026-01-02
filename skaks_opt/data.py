from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple
import numpy as np

__all__ = [
    "Example",
    "Dataset",
    "load_csv",
    "split_dataset",
    "filter_quiet",
]


@dataclass(frozen=True)
class Example:
    fen: str
    target_cp: float
    weight: float = 1.0


class Dataset:
    def __init__(
        self, examples: Sequence[Example], side_override: Sequence[int] | None = None
    ):
        self.fens: List[str] = [e.fen for e in examples]
        self.targets = np.array([e.target_cp for e in examples], dtype=np.float64)
        self.weights = np.array([e.weight for e in examples], dtype=np.float64)
        if side_override is not None:
            self.side = np.array(side_override, dtype=np.int8)
        else:
            self.side = np.array(
                [parse_side_to_move(e.fen) for e in examples], dtype=np.int8
            )

    def __len__(self) -> int:
        return len(self.fens)

    def iter_batches(
        self, batch_size: int
    ) -> Iterator[Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]]:
        n = len(self.fens)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            yield (
                self.fens[start:end],
                self.targets[start:end],
                self.weights[start:end],
                self.side[start:end],
            )


def parse_side_to_move(fen: str) -> int:
    parts = fen.split()
    if len(parts) < 2:
        return 1
    return 1 if parts[1] == "w" else -1


def load_csv(path: Path | str, limit: int | None = None) -> Dataset:
    path = Path(path)
    examples: List[Example] = []
    sides: List[int] = []
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "fen" not in reader.fieldnames:
            raise ValueError("CSV must contain fen column")
        has_score = "score" in reader.fieldnames
        has_sf = "stockfish_cp" in reader.fieldnames
        if not has_score and not has_sf:
            raise ValueError("CSV must contain either score or stockfish_cp column")
        for row_idx, row in enumerate(reader):
            if limit is not None and len(examples) >= limit:
                break
            fen = row["fen"].strip()
            if has_score and row.get("score", "") != "":
                score = float(row["score"])
            else:
                score = float(row["stockfish_cp"])
            weight = float(row.get("weight", 1.0) or 1.0)
            examples.append(Example(fen=fen, target_cp=score, weight=weight))

            stm = row.get("side_to_move")
            if stm is not None and stm.lower().startswith("w"):
                sides.append(1)
            elif stm is not None and stm.lower().startswith("b"):
                sides.append(-1)
            else:
                sides.append(parse_side_to_move(fen))
    if not examples:
        raise ValueError(f"no rows loaded from {path}")
    return Dataset(examples, side_override=sides)


def filter_quiet(dataset: Dataset, batch_size: int = 2048) -> Dataset:
    try:
        import skaks_eval as sk
    except Exception:  # pragma: no cover - optional dependency
        warnings.warn("skaks_eval not available; skipping quiet filtering")
        return dataset

    keep_mask = np.zeros(len(dataset.fens), dtype=bool)
    for start in range(0, len(dataset.fens), batch_size):
        end = min(start + batch_size, len(dataset.fens))
        chunk = dataset.fens[start:end]
        flags = sk.is_quiet_batch(chunk)
        for idx, flag in enumerate(flags):
            keep_mask[start + idx] = bool(flag) if flag is not None else False

    if keep_mask.all():
        return dataset
    if not keep_mask.any():
        raise ValueError("quiet filtering removed all positions")

    examples = [
        Example(
            fen=fen,
            target_cp=float(dataset.targets[i]),
            weight=float(dataset.weights[i]),
        )
        for i, fen in enumerate(dataset.fens)
        if keep_mask[i]
    ]
    sides = [int(dataset.side[i]) for i in range(len(dataset.fens)) if keep_mask[i]]
    return Dataset(examples, side_override=sides)


def split_dataset(
    dataset: Dataset, val_split: float, seed: int = 42
) -> Tuple[Dataset, Dataset]:
    if not 0.0 < val_split < 1.0:
        raise ValueError("val_split must be in (0,1)")

    n = len(dataset)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    val_size = max(1, int(n * val_split))
    val_idx = order[:val_size]
    train_idx = order[val_size:]

    def _subset(idxs: np.ndarray) -> Dataset:
        ex = [
            Example(
                fen=dataset.fens[i],
                target_cp=float(dataset.targets[i]),
                weight=float(dataset.weights[i]),
            )
            for i in idxs
        ]
        sides = [int(dataset.side[i]) for i in idxs]
        return Dataset(ex, side_override=sides)

    return _subset(train_idx), _subset(val_idx)
