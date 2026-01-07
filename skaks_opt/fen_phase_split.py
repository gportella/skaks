from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import chess

Row = Tuple[str, float, float]
BUCKET_NAMES = ("opening", "middlegame", "endgame")
PHASE_WEIGHTS = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
}
MAX_PHASE = sum(PHASE_WEIGHTS.values()) * 2  # two sides


def add_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "fen-phase-split",
        help="Split FEN+score data into opening/middlegame/endgame buckets",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="CSV with fen+score/stockfish_cp columns or text lines: score [weight] <fen>",
    )
    parser.add_argument(
        "--output-dir",
        help="Where to write phase files (default: alongside input)",
    )
    parser.add_argument(
        "--output-prefix",
        help="Prefix for output files (default: input stem)",
    )
    parser.add_argument(
        "--opening-threshold",
        type=int,
        default=16,
        help="Phase score >= threshold is considered opening",
    )
    parser.add_argument(
        "--endgame-threshold",
        type=int,
        default=8,
        help="Phase score <= threshold is considered endgame",
    )
    parser.add_argument(
        "--phase-column",
        action="store_true",
        help="Write a single CSV with a phase column instead of per-phase files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional row limit from input",
    )
    parser.add_argument(
        "--weight-default",
        type=float,
        default=1.0,
        help="Weight to apply when a text row omits weight",
    )
    return parser


def _load_csv_rows(path: Path, limit: int | None) -> List[Row]:
    rows: List[Row] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "fen" not in reader.fieldnames:
            raise ValueError("CSV must contain fen column")
        has_score = "score" in reader.fieldnames
        has_sf = "stockfish_cp" in reader.fieldnames
        if not has_score and not has_sf:
            raise ValueError("CSV must contain either score or stockfish_cp column")
        for row in reader:
            if limit is not None and len(rows) >= limit:
                break
            fen = (row.get("fen") or "").strip()
            if not fen:
                continue
            if has_score and (row.get("score") or "").strip() != "":
                score = float(row["score"])
            else:
                score = float(row.get("stockfish_cp", 0.0))
            weight = float(row.get("weight", 1.0) or 1.0)
            rows.append((fen, score, weight))
    if not rows:
        raise ValueError(f"no rows loaded from {path}")
    return rows


def _load_txt_rows(path: Path, limit: int | None, weight_default: float) -> List[Row]:
    rows: List[Row] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 7:
                continue
            fen_start = len(parts) - 6
            try:
                score = float(parts[0])
            except ValueError:
                continue
            weight = weight_default
            if fen_start > 1:
                try:
                    weight = float(parts[1])
                except ValueError:
                    weight = weight_default
            fen = " ".join(parts[fen_start:])
            rows.append((fen, score, weight))
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"no rows loaded from {path}")
    return rows


def _load_rows(path: Path, limit: int | None, weight_default: float) -> List[Row]:
    if path.suffix.lower() == ".csv":
        return _load_csv_rows(path, limit)
    return _load_txt_rows(path, limit, weight_default)


def _phase_score(board: chess.Board) -> int:
    score = 0
    for piece in board.piece_map().values():
        score += PHASE_WEIGHTS.get(piece.piece_type, 0)
    return max(0, min(MAX_PHASE, score))


def _bucket_rows(
    rows: Sequence[Row], opening_threshold: int, endgame_threshold: int
) -> Dict[str, List[Tuple[str, float, float, int]]]:
    buckets: Dict[str, List[Tuple[str, float, float, int]]] = {
        name: [] for name in BUCKET_NAMES
    }
    for fen, score, weight in rows:
        try:
            board = chess.Board(fen)
        except ValueError:
            continue
        phase_val = _phase_score(board)
        if phase_val >= opening_threshold:
            phase = "opening"
        elif phase_val <= endgame_threshold:
            phase = "endgame"
        else:
            phase = "middlegame"
        buckets[phase].append((fen, score, weight, phase_val))
    return buckets


def _write_buckets(
    *,
    buckets: Dict[str, List[Tuple[str, float, float, int]]],
    output_dir: Path,
    prefix: str,
    phase_column: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if phase_column:
        path = output_dir / f"{prefix}_phases.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["fen", "score", "weight", "phase", "phase_score"]
            )
            writer.writeheader()
            for phase_name in BUCKET_NAMES:
                for fen, score, weight, phase_score in buckets[phase_name]:
                    writer.writerow(
                        {
                            "fen": fen,
                            "score": score,
                            "weight": weight,
                            "phase": phase_name,
                            "phase_score": phase_score,
                        }
                    )
        return

    for phase_name in BUCKET_NAMES:
        rows = buckets[phase_name]
        if not rows:
            continue
        path = output_dir / f"{prefix}_{phase_name}.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["fen", "score", "weight", "phase_score"]
            )
            writer.writeheader()
            for fen, score, weight, phase_score in rows:
                writer.writerow(
                    {
                        "fen": fen,
                        "score": score,
                        "weight": weight,
                        "phase_score": phase_score,
                    }
                )


def run_fen_phase_split(args: argparse.Namespace) -> None:
    input_path = Path(args.input).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).resolve() if args.output_dir else input_path.parent
    )
    prefix = args.output_prefix or input_path.stem

    rows = _load_rows(input_path, args.limit, args.weight_default)
    buckets = _bucket_rows(rows, args.opening_threshold, args.endgame_threshold)
    _write_buckets(
        buckets=buckets,
        output_dir=output_dir,
        prefix=prefix,
        phase_column=args.phase_column,
    )


__all__ = ["add_subparser", "run_fen_phase_split"]
