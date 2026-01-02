import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    from validation_moves.eval_from_pgn import _parse_result, parse_pgn_games
except ModuleNotFoundError:
    from eval_from_pgn import _parse_result, parse_pgn_games


def load_game_outcomes(
    pgn_path: Path,
) -> Dict[int, Tuple[str, float | None, str | None]]:
    outcomes: Dict[int, Tuple[str, float | None, str | None]] = {}
    for idx, game in enumerate(parse_pgn_games(pgn_path)):
        result_raw = game.headers.get("Result", "")
        outcomes[idx] = _parse_result(result_raw)
    return outcomes


def add_outcomes(input_csv: Path, output_csv: Path, pgn_path: Path) -> None:
    mapping = load_game_outcomes(pgn_path)

    with input_csv.open("r", newline="", encoding="utf-8") as src:
        with output_csv.open("w", newline="", encoding="utf-8") as dst:
            reader = csv.DictReader(src)
            if not reader.fieldnames or "game_index" not in reader.fieldnames:
                raise SystemExit("input CSV must have a game_index column")

            fieldnames = list(reader.fieldnames)
            for extra in ("result", "outcome", "winner"):
                if extra not in fieldnames:
                    fieldnames.append(extra)

            writer = csv.DictWriter(dst, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                try:
                    game_idx = int(row["game_index"])
                except Exception:
                    game_idx = None

                if game_idx is not None and game_idx in mapping:
                    result, outcome, winner = mapping[game_idx]
                    row["result"] = result

                    stm = (row.get("side_to_move") or "").lower()
                    row_winner = winner or ""
                    row_outcome = outcome
                    if winner in {"w", "b"}:
                        if stm == "w":
                            row_winner = "w" if winner == "w" else "l"
                            row_outcome = 1.0 if winner == "w" else 0.0
                        elif stm == "b":
                            row_winner = "w" if winner == "b" else "l"
                            row_outcome = 1.0 if winner == "b" else 0.0
                    row["winner"] = row_winner
                    row["outcome"] = "" if row_outcome is None else row_outcome
                else:
                    row.setdefault("result", "")
                    row.setdefault("outcome", "")
                    row.setdefault("winner", "")

                writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach PGN game results to an eval_pairs CSV"
    )
    parser.add_argument("pgn", type=Path, help="Source PGN with game results")
    parser.add_argument("input", type=Path, help="Existing eval_pairs CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: <input> with _with_results suffix)",
    )
    args = parser.parse_args()

    if not args.pgn.exists():
        raise SystemExit(f"PGN not found: {args.pgn}")
    if not args.input.exists():
        raise SystemExit(f"input CSV not found: {args.input}")

    out_path = (
        args.output
        if args.output is not None
        else args.input.with_name(args.input.stem + "_with_results" + args.input.suffix)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    add_outcomes(args.input, out_path, args.pgn)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
