import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple

PieceValues = {
    "p": 1,
    "n": 3,
    "b": 3,
    "r": 5,
    "q": 9,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Filter/stratify Texel CSV to reduce duplicates and balance buckets"
    )
    p.add_argument("input", type=Path, help="Input CSV (fen,outcome,...)")
    p.add_argument("output", type=Path, help="Filtered output CSV")
    p.add_argument("--max-per-game", type=int, default=256, help="Cap samples per game")
    p.add_argument(
        "--ply-bucket-size",
        type=int,
        default=8,
        help="Bucket size in plies for stratification",
    )
    p.add_argument(
        "--max-per-bucket",
        type=int,
        default=50000,
        help="Cap samples per (ply_bucket,phase_bucket,side) after dedup",
    )
    p.add_argument(
        "--total-cap",
        type=int,
        default=None,
        help="Optional total cap after all filtering",
    )
    p.add_argument(
        "--min-ply",
        type=int,
        default=0,
        help="Drop positions with ply < min_ply (filters early opening noise)",
    )
    p.add_argument(
        "--no-stratify",
        action="store_true",
        help="Skip ply/phase bucket capping (keep per-game cap and dedup only)",
    )
    p.add_argument(
        "--eval-scale",
        type=float,
        default=400.0,
        help="Scale for converting centipawns to WDL prob (Texel-style)",
    )
    p.add_argument(
        "--write-eval-outcome",
        action="store_true",
        help="Write outcome_eval column computed from stockfish_cp and side_to_move",
    )
    p.add_argument(
        "--replace-outcome-with-eval",
        action="store_true",
        help="Overwrite outcome column with eval-derived probability",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    return p.parse_args()


def phase_bucket(fen: str) -> int:
    board = fen.split()[0]
    score = 0
    for c in board:
        if c.isalpha():
            score += PieceValues.get(c.lower(), 0)
    # Rough buckets: 0-12 endgame, 13-24 middlegame, >24 opening
    if score <= 12:
        return 2  # endgame-ish
    if score <= 24:
        return 1  # middlegame-ish
    return 0  # opening-ish


def normalize_fen(fen: str) -> str:
    parts = fen.split()
    if len(parts) >= 4:
        # drop halfmove/fullmove counters to merge repeats
        return " ".join(parts[:4])
    return fen


def side_from_row(row: Dict[str, str]) -> int:
    stm = row.get("side_to_move")
    if stm:
        return 1 if stm.lower().startswith("w") else -1
    fen = row.get("fen", "")
    parts = fen.split()
    if len(parts) >= 2:
        return 1 if parts[1] == "w" else -1
    return 1


def add_eval_outcome(
    rows: List[Dict[str, str]], eval_scale: float, replace: bool, write_column: bool
) -> List[Dict[str, str]]:
    import math

    updated: List[Dict[str, str]] = []
    for row in rows:
        try:
            cp = float(row.get("stockfish_cp", ""))
        except Exception:
            cp = None
        if cp is None:
            updated.append(row)
            continue
        side = side_from_row(row)
        cp_side = cp * side
        p = 1.0 / (1.0 + math.pow(10.0, -cp_side / eval_scale))
        if write_column:
            row = dict(row)
            row["outcome_eval"] = f"{p:.6f}"
        if replace:
            row = dict(row)
            row["outcome"] = f"{p:.6f}"
        updated.append(row)
    return updated


def load_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        if not fieldnames or "fen" not in fieldnames:
            raise SystemExit("input CSV must have fen column")
        rows = [row for row in reader]
    return list(fieldnames), rows


def filter_min_ply(rows: List[Dict[str, str]], min_ply: int) -> List[Dict[str, str]]:
    if min_ply <= 0:
        return rows
    kept = []
    for row in rows:
        try:
            ply = int(row.get("ply", 0))
        except Exception:
            ply = 0
        if ply >= min_ply:
            kept.append(row)
    return kept


def cap_per_game(rows: List[Dict[str, str]], max_per_game: int, rng: random.Random):
    buckets: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        gid = row.get("game_index", "")
        buckets.setdefault(gid, []).append(row)
    kept: List[Dict[str, str]] = []
    for gid, items in buckets.items():
        if len(items) > max_per_game:
            kept.extend(rng.sample(items, max_per_game))
        else:
            kept.extend(items)
    return kept


def dedup_fens(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    kept = []
    for row in rows:
        fen = row.get("fen", "")
        key = normalize_fen(fen)
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    return kept


def stratified_cap(
    rows: List[Dict[str, str]],
    ply_bucket_size: int,
    max_per_bucket: int,
    rng: random.Random,
) -> List[Dict[str, str]]:
    buckets: Dict[Tuple[int, int, str], List[Dict[str, str]]] = {}
    for row in rows:
        try:
            ply = int(row.get("ply", 0))
        except Exception:
            ply = 0
        ply_b = ply // ply_bucket_size
        phase_b = phase_bucket(row.get("fen", ""))
        side = row.get("side_to_move", "?")
        key = (ply_b, phase_b, side)
        buckets.setdefault(key, []).append(row)

    kept: List[Dict[str, str]] = []
    for key, items in buckets.items():
        if len(items) > max_per_bucket:
            kept.extend(rng.sample(items, max_per_bucket))
        else:
            kept.extend(items)
    return kept


def write_rows(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    fieldnames, rows = load_rows(args.input)

    rows = filter_min_ply(rows, args.min_ply)
    rows = cap_per_game(rows, args.max_per_game, rng)
    rows = dedup_fens(rows)
    if not args.no_stratify:
        rows = stratified_cap(rows, args.ply_bucket_size, args.max_per_bucket, rng)

    if args.write_eval_outcome or args.replace_outcome_with_eval:
        rows = add_eval_outcome(
            rows,
            eval_scale=args.eval_scale,
            replace=args.replace_outcome_with_eval,
            write_column=args.write_eval_outcome,
        )

    if args.total_cap is not None and len(rows) > args.total_cap:
        rows = rng.sample(rows, args.total_cap)

    write_rows(args.output, fieldnames, rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
