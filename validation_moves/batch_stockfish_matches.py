#!/usr/bin/env python
"""Batch runner that stores fight results (including PGNs) into SQLite."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

DEFAULT_GAMES = 100
DEFAULT_MATCH_LIMIT = 500
DEFAULT_DEPTH = 8
DEFAULT_DB_NAME = "validation_matches.sqlite3"
PGN_HEADER = "PGN:"
DEFAULT_SUMMARY_LABELS = ("skaks", "stockfish", "draw", "unknown")
DEFAULT_ELO_START = 1500.0
DEFAULT_ELO_OPPONENT = 2600.0
DEFAULT_ELO_K_FACTOR = 20.0
DEFAULT_ELO_STORE = str(Path(__file__).resolve().with_name(".skaks_elo.json"))


@dataclass
class ExecutedMatch:
    """Holds stdout/stderr and timing information for a single match execution."""

    index: int
    started_at: datetime
    finished_at: datetime
    duration_ms: int
    exit_code: int
    stdout: str
    stderr: str


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be a positive number")
    return parsed


@dataclass
class EloComputation:
    rating_before: float
    rating_after: float
    delta: float
    expected_score: float
    actual_score: float
    games: int
    wins: int
    losses: int
    draws: int
    opponent_rating: float


def load_rating(path: Path, fallback: float) -> float:
    try:
        content = path.read_text(encoding="utf-8")
        data = json.loads(content)
        rating = float(data.get("rating", fallback))
        return rating
    except FileNotFoundError:
        return fallback
    except (json.JSONDecodeError, OSError, ValueError):
        return fallback


def store_rating(path: Path, rating: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"rating": rating}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def compute_elo(
    *,
    rating: float,
    opponent_rating: float,
    wins: int,
    losses: int,
    draws: int,
    k_factor: float,
) -> EloComputation:
    games = wins + losses + draws
    actual_score = wins + 0.5 * draws
    if games == 0:
        return EloComputation(
            rating_before=rating,
            rating_after=rating,
            delta=0.0,
            expected_score=0.0,
            actual_score=actual_score,
            games=0,
            wins=wins,
            losses=losses,
            draws=draws,
            opponent_rating=opponent_rating,
        )

    expected_per_game = 1.0 / (1.0 + 10 ** ((opponent_rating - rating) / 400.0))
    expected_score = expected_per_game * games
    delta = k_factor * (actual_score - expected_score)
    rating_after = rating + delta
    return EloComputation(
        rating_before=rating,
        rating_after=rating_after,
        delta=delta,
        expected_score=expected_score,
        actual_score=actual_score,
        games=games,
        wins=wins,
        losses=losses,
        draws=draws,
        opponent_rating=opponent_rating,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run many fights against a reference engine and store the PGNs in SQLite."
    )
    parser.add_argument(
        "--games",
        type=positive_int,
        default=DEFAULT_GAMES,
        help=f"Number of new games to play (default: {DEFAULT_GAMES})",
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=DEFAULT_MATCH_LIMIT,
        help=f"Half-move limit forwarded to fight script (default: {DEFAULT_MATCH_LIMIT})",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default=None,
        help="Reference engine binary (default: delegated to fight script)",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default=None,
        help="Opponent engine binary (default: delegated to fight script)",
    )
    parser.add_argument(
        "--engine-params",
        type=str,
        help="Path to params file for the reference engine (passed as --engine-params)",
    )
    parser.add_argument(
        "--engine-label",
        type=str,
        help="Display label for the reference engine in summary/Elo (default: engine basename)",
    )
    parser.add_argument(
        "--opponent-params",
        type=str,
        help="Path to params file for the opponent engine (passed as --opponent-params)",
    )
    parser.add_argument(
        "--opponent-label",
        type=str,
        help="Display label for the opponent engine in summary/Elo (default: opponent basename)",
    )
    parser.add_argument(
        "--stockfish",
        action="store_true",
        help="Shortcut for --opponent stockfish",
    )
    parser.add_argument(
        "--database",
        type=str,
        default=DEFAULT_DB_NAME,
        help=f"SQLite database file path (default: {DEFAULT_DB_NAME})",
    )
    parser.add_argument(
        "--concurrency",
        type=positive_int,
        default=None,
        help="Maximum number of concurrent games to run",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume numbering from the existing dataset (appends new games)",
    )
    parser.add_argument(
        "--timeout",
        type=positive_float,
        default=None,
        help="Timeout forwarded to fight script (seconds)",
    )
    timing_group = parser.add_mutually_exclusive_group()
    timing_group.add_argument(
        "--depth",
        type=positive_int,
        help="Depth passed to the reference engine",
    )
    timing_group.add_argument(
        "--time-per-move",
        type=positive_float,
        help="Seconds per move for the reference engine",
    )
    timing_group.add_argument(
        "--clock",
        type=positive_float,
        help="Primary clock time in seconds for the reference engine",
    )
    parser.add_argument(
        "--opponent-time-per-move",
        type=positive_float,
        help="Seconds per move for the opponent engine",
    )
    parser.add_argument(
        "--opponent-clock",
        type=positive_float,
        help="Clock time in seconds for the opponent engine",
    )
    parser.add_argument(
        "--increment",
        type=positive_float,
        help="Increment in seconds for the reference engine (requires --clock)",
    )
    parser.add_argument(
        "--opponent-increment",
        type=positive_float,
        help="Increment in seconds for the opponent engine (requires --clock)",
    )
    parser.add_argument(
        "--moves-to-go",
        type=positive_int,
        help="Approximate moves remaining to next time control",
    )
    parser.add_argument(
        "--no-handicap",
        action="store_true",
        help="Disable opponent handicap when forwarding to fight script",
    )
    parser.add_argument(
        "--handicap-factor",
        type=positive_float,
        default=0.35,
        help="Scaling factor applied to opponent time when handicap is enabled",
    )
    parser.add_argument(
        "--handicap-depth",
        type=non_negative_int,
        default=6,
        help="Depth advantage retained by reference engine when handicap is enabled",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print stdout/stderr for every game (default: only failures)",
    )
    parser.add_argument(
        "--elo-start",
        type=float,
        default=DEFAULT_ELO_START,
        help=(
            "Starting Elo for skaks when no stored rating is present "
            f"(default: {DEFAULT_ELO_START})"
        ),
    )
    parser.add_argument(
        "--elo-opponent",
        type=float,
        default=DEFAULT_ELO_OPPONENT,
        help=f"Assumed opponent Elo rating (default: {DEFAULT_ELO_OPPONENT})",
    )
    parser.add_argument(
        "--elo-k-factor",
        type=positive_float,
        default=DEFAULT_ELO_K_FACTOR,
        help=f"K-factor for Elo updates (default: {DEFAULT_ELO_K_FACTOR})",
    )
    parser.add_argument(
        "--elo-store",
        type=str,
        default=DEFAULT_ELO_STORE,
        help="Path to JSON file for persisting skaks Elo between runs",
    )
    parser.add_argument(
        "--no-elo-store",
        action="store_true",
        help="Skip loading/saving Elo state; compute only for this batch",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        help="Optional path to write a JSON summary (counts, Elo, timings)",
    )

    args = parser.parse_args(argv)

    if args.stockfish and args.opponent:
        parser.error("--stockfish may not be combined with --opponent")
    if args.stockfish:
        args.opponent = "stockfish"

    mode_count = sum(
        flag is not None for flag in (args.depth, args.time_per_move, args.clock)
    )
    if mode_count == 0:
        args.depth = DEFAULT_DEPTH
    if mode_count > 1:
        parser.error("choose exactly one of --depth, --time-per-move, or --clock")

    if args.time_per_move is None and args.opponent_time_per_move is not None:
        parser.error("--opponent-time-per-move requires --time-per-move")
    if args.clock is None and args.opponent_clock is not None:
        parser.error("--opponent-clock requires --clock")
    if args.clock is None and (
        args.increment is not None or args.opponent_increment is not None
    ):
        parser.error("increments require --clock")
    if args.moves_to_go is not None and args.moves_to_go <= 0:
        parser.error("--moves-to-go must be positive")

    if args.no_handicap:
        args.handicap_factor = 1.0
        args.handicap_depth = 0

    args.handicap_enabled = not args.no_handicap

    if args.concurrency is None:
        detected = os.cpu_count() or 1
        args.concurrency = max(1, min(detected, 4))

    return args


def build_fight_arguments(args: argparse.Namespace) -> List[str]:
    base_args: List[str] = ["--limit", str(args.limit)]
    if args.depth is not None:
        base_args.extend(["--depth", str(args.depth)])
    elif args.time_per_move is not None:
        base_args.extend(["--time-per-move", str(args.time_per_move)])
        if args.opponent_time_per_move is not None:
            base_args.extend(
                ["--opponent-time-per-move", str(args.opponent_time_per_move)]
            )
    elif args.clock is not None:
        base_args.extend(["--clock", str(args.clock)])
        if args.opponent_clock is not None:
            base_args.extend(["--opponent-clock", str(args.opponent_clock)])
        if args.increment is not None:
            base_args.extend(["--increment", str(args.increment)])
        if args.opponent_increment is not None:
            base_args.extend(["--opponent-increment", str(args.opponent_increment)])
        if args.moves_to_go is not None:
            base_args.extend(["--moves-to-go", str(args.moves_to_go)])

    if args.engine:
        base_args.extend(["--engine", args.engine])
    if args.engine_params:
        base_args.extend(["--engine-params", args.engine_params])
    if args.opponent:
        base_args.extend(["--opponent", args.opponent])
    if args.opponent_params:
        base_args.extend(["--opponent-params", args.opponent_params])
    if args.no_handicap:
        base_args.append("--no-handicap")
    if args.handicap_factor is not None:
        base_args.extend(["--handicap-factor", str(args.handicap_factor)])
    if args.handicap_depth is not None:
        base_args.extend(["--handicap-depth", str(args.handicap_depth)])
    if args.timeout is not None:
        base_args.extend(["--timeout", str(args.timeout)])

    return base_args


def prepare_database(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=WAL;")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            depth INTEGER,
            limit_plies INTEGER,
            parameters TEXT NOT NULL,
            game_index INTEGER NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT NOT NULL,
            duration_ms INTEGER NOT NULL,
            exit_code INTEGER NOT NULL,
            result TEXT,
            winner TEXT,
            pgn TEXT,
            stdout TEXT,
            stderr TEXT,
            white_engine TEXT,
            black_engine TEXT
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_matches_depth
        ON matches(depth)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_matches_parameters_game
        ON matches(parameters, game_index)
        """
    )
    connection.commit()

    # Backward compatibility: add engine name columns if the table predates them.
    existing_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(matches);")
    }
    if "white_engine" not in existing_columns:
        connection.execute("ALTER TABLE matches ADD COLUMN white_engine TEXT")
    if "black_engine" not in existing_columns:
        connection.execute("ALTER TABLE matches ADD COLUMN black_engine TEXT")
    connection.commit()


def run_single_match(
    index: int,
    executable: str,
    command: Sequence[str],
) -> ExecutedMatch:
    started_at = datetime.now(timezone.utc)
    timer_start = time.perf_counter()
    proc = subprocess.run(
        [executable, *command],
        capture_output=True,
        text=True,
        check=False,
    )
    timer_end = time.perf_counter()
    finished_at = datetime.now(timezone.utc)
    duration_ms = int(round((timer_end - timer_start) * 1000.0))

    return ExecutedMatch(
        index=index,
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=duration_ms,
        exit_code=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def extract_metadata(
    stdout: str,
    stderr: str,
) -> Tuple[Optional[str], Optional[str], Optional[str], bool]:
    result_value: Optional[str] = None
    winner_value: Optional[str] = None
    pgn_value: Optional[str] = None

    lines = stdout.splitlines()
    for idx, line in enumerate(lines):
        lowered = line.lower()
        if lowered.startswith("result:"):
            result_value = line.split(":", 1)[1].strip() or None
        elif lowered.startswith("winner:"):
            winner_value = line.split(":", 1)[1].strip() or None
        elif line.strip() == PGN_HEADER:
            pgn_lines = lines[idx + 1 :]
            pgn_text = "\n".join(pgn_lines).strip()
            pgn_value = pgn_text or None
            break

    combined = (stdout + "\n" + stderr).lower()
    failure_tokens = (
        "illegal",
        "engine terminated unexpectedly",
        "engine error",
        "terminated unexpectedly",
    )
    illegal_flag = any(token in combined for token in failure_tokens)
    return result_value, winner_value, pgn_value, illegal_flag


def determine_winner_label(
    result_value: Optional[str],
    winner_value: Optional[str],
    white_label: str,
    black_label: str,
) -> str:
    def _normalize(name: str) -> str:
        return Path(str(name)).name.lower()

    white_tokens = {_normalize(white_label), white_label.lower()}
    black_tokens = {_normalize(black_label), black_label.lower()}

    if winner_value:
        normalized = winner_value.strip().lower()
        if normalized:
            if normalized.startswith("draw"):
                return "draw"
            if any(token in normalized for token in white_tokens):
                return white_label
            if any(token in normalized for token in black_tokens):
                return black_label
            tokens = normalized.replace("-", " ").split()
            if "white" in tokens:
                return white_label
            if "black" in tokens:
                return black_label
            if normalized == "unknown":
                return "unknown"
    if result_value:
        normalized_result = result_value.strip().lower()
        if normalized_result in {"1-0", "1 - 0"}:
            return white_label
        if normalized_result in {"0-1", "0 - 1"}:
            return black_label
        if normalized_result in {"1/2-1/2", "1/2 - 1/2", "½-½"}:
            return "draw"
        if normalized_result.startswith("draw"):
            return "draw"
    return "unknown"


def record_match(
    connection: sqlite3.Connection,
    *,
    result: ExecutedMatch,
    depth: Optional[int],
    limit_plies: int,
    parameters_blob: str,
    parsed_result: Optional[str],
    parsed_winner: Optional[str],
    pgn_text: Optional[str],
    white_engine: str,
    black_engine: str,
) -> None:
    connection.execute(
        """
        INSERT INTO matches (
            depth,
            limit_plies,
            parameters,
            game_index,
            started_at,
            finished_at,
            duration_ms,
            exit_code,
            result,
            winner,
            pgn,
            stdout,
            stderr,
            white_engine,
            black_engine
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            depth,
            limit_plies,
            parameters_blob,
            result.index,
            result.started_at.isoformat(),
            result.finished_at.isoformat(),
            result.duration_ms,
            result.exit_code,
            parsed_result,
            parsed_winner,
            pgn_text,
            result.stdout,
            result.stderr,
            white_engine,
            black_engine,
        ),
    )
    connection.commit()


def choose_starting_index(
    connection: sqlite3.Connection,
    *,
    parameters_blob: str,
    resume: bool,
) -> int:
    if not resume:
        return 1
    cursor = connection.execute(
        "SELECT MAX(game_index) FROM matches WHERE parameters = ?", (parameters_blob,)
    )
    row = cursor.fetchone()
    if row is None or row[0] is None:
        return 1
    return int(row[0]) + 1


def summarize_counts(summary: Dict[str, int], labels: Sequence[str]) -> str:
    ordered_labels = list(labels)
    extra_labels = [label for label in summary if label not in labels]
    display_labels = ordered_labels + sorted(extra_labels)
    return ", ".join(f"{label}={summary.get(label, 0)}" for label in display_labels)


def run_batch(args: argparse.Namespace) -> int:
    fight_script = Path(__file__).resolve().with_name("fight_against_engine.py")
    if not fight_script.exists():
        print(f"Fight script not found at {fight_script}", file=sys.stderr)
        return 2

    wall_start = time.perf_counter()

    db_path = Path(args.database).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path))
    try:
        prepare_database(connection)

        rating_path = Path(args.elo_store)
        if not rating_path.is_absolute():
            rating_path = Path(__file__).resolve().parent / rating_path
        rating_before = (
            args.elo_start
            if args.no_elo_store
            else load_rating(rating_path, fallback=args.elo_start)
        )

        base_args = build_fight_arguments(args)
        parameters_snapshot = {
            "limit": args.limit,
            "depth": args.depth,
            "time_per_move": args.time_per_move,
            "opponent_time_per_move": args.opponent_time_per_move,
            "clock": args.clock,
            "opponent_clock": args.opponent_clock,
            "increment": args.increment,
            "opponent_increment": args.opponent_increment,
            "moves_to_go": args.moves_to_go,
            "engine": args.engine,
            "engine_params": args.engine_params,
            "opponent": args.opponent,
            "opponent_params": args.opponent_params,
            "handicap_factor": args.handicap_factor,
            "handicap_depth": args.handicap_depth,
            "handicap_enabled": args.handicap_enabled,
            "timeout": args.timeout,
        }
        parameters_blob = json.dumps(
            parameters_snapshot, sort_keys=True, separators=(",", ":")
        )

        start_index = choose_starting_index(
            connection, parameters_blob=parameters_blob, resume=args.resume
        )
        indices = list(range(start_index, start_index + args.games))
        if not indices:
            print("No games scheduled (games=0). Nothing to do.")
            return 0

        white_label = (
            args.engine_label
            if args.engine_label
            else (Path(args.engine).name if args.engine else DEFAULT_SUMMARY_LABELS[0])
        )
        black_label = (
            args.opponent_label
            if args.opponent_label
            else (
                Path(args.opponent).name if args.opponent else DEFAULT_SUMMARY_LABELS[1]
            )
        )
        if white_label == black_label:
            white_label = f"{white_label}_white"
            black_label = f"{black_label}_black"

        summary_labels = (white_label, black_label, "draw", "unknown")
        summary: Dict[str, int] = {label: 0 for label in summary_labels}
        failures = 0
        completed = 0

        print(
            f"Running {len(indices)} games "
            f"(depth={args.depth}, limit={args.limit}, concurrency={args.concurrency})"
        )
        print(f"Database: {db_path}")
        sys.stdout.flush()

        executable = sys.executable
        command = [str(fight_script), *base_args]

        white_engine = args.engine if args.engine else "skaks"
        black_engine = args.opponent if args.opponent else "stockfish"

        futures: Dict[concurrent.futures.Future[ExecutedMatch], int] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as executor:
            for game_index in indices:
                future = executor.submit(
                    run_single_match, game_index, executable, command
                )
                futures[future] = game_index

            try:
                for future in concurrent.futures.as_completed(futures):
                    match = future.result()
                    (
                        parsed_result,
                        parsed_winner,
                        pgn_text,
                        illegal_flag,
                    ) = extract_metadata(match.stdout, match.stderr)
                    if parsed_result is None:
                        parsed_result = "unknown"
                    if parsed_winner is None:
                        parsed_winner = "unknown"
                    winner_label = determine_winner_label(
                        parsed_result, parsed_winner, white_label, black_label
                    )
                    failed = match.exit_code != 0 or illegal_flag
                    if failed:
                        failures += 1
                        summary["unknown"] = summary.get("unknown", 0) + 1
                    else:
                        summary[winner_label] = summary.get(winner_label, 0) + 1
                    record_match(
                        connection,
                        result=match,
                        depth=args.depth,
                        limit_plies=args.limit,
                        parameters_blob=parameters_blob,
                        parsed_result=parsed_result,
                        parsed_winner=parsed_winner,
                        pgn_text=pgn_text,
                        white_engine=white_engine,
                        black_engine=black_engine,
                    )
                    completed += 1

                    status = "FAIL" if failed else "OK"
                    duration_sec = match.duration_ms / 1000.0
                    display_result = parsed_result or "?"
                    display_winner = parsed_winner or winner_label
                    print(
                        f"[{completed}/{len(indices)}] {status:<4} "
                        f"game={match.index} result={display_result:<7} "
                        f"winner={display_winner:<10} time={duration_sec:6.2f}s"
                    )
                    if failed and not args.verbose:
                        stdout_preview = match.stdout.strip().splitlines()
                        stderr_preview = match.stderr.strip().splitlines()
                        if stdout_preview:
                            print("  stdout snippet:")
                            for line in stdout_preview[:5]:
                                print(f"    {line}")
                            if len(stdout_preview) > 5:
                                print("    ...")
                        if stderr_preview:
                            print("  stderr snippet:")
                            for line in stderr_preview[:5]:
                                print(f"    {line}")
                            if len(stderr_preview) > 5:
                                print("    ...")
                    if args.verbose:
                        if match.stdout.strip():
                            print("--- stdout ---")
                            print(match.stdout.rstrip())
                        if match.stderr.strip():
                            print("--- stderr ---")
                            print(match.stderr.rstrip())
                    sys.stdout.flush()
            except KeyboardInterrupt:
                print(
                    "Interrupted by user. Cancelling outstanding games...",
                    file=sys.stderr,
                )
                for future in futures:
                    future.cancel()
                return 130

        print()
        print(f"Stored results in {db_path}")
        print(f"Summary: {summarize_counts(summary, summary_labels)}")
        print(f"Failures: {failures} / {len(indices)}")

        wins = summary.get(white_label, 0)
        losses = summary.get(black_label, 0)
        draws = summary.get("draw", 0)
        elo = compute_elo(
            rating=rating_before,
            opponent_rating=args.elo_opponent,
            wins=wins,
            losses=losses,
            draws=draws,
            k_factor=args.elo_k_factor,
        )
        if elo.games > 0:
            print(
                "Elo: "
                f"start={elo.rating_before:.1f}, "
                f"opp={elo.opponent_rating:.1f}, "
                f"score={elo.actual_score:.1f}/{elo.games}, "
                f"expected={elo.expected_score:.2f}, "
                f"delta={elo.delta:+.1f}, "
                f"new={elo.rating_after:.1f}"
            )
            if not args.no_elo_store:
                store_rating(rating_path, elo.rating_after)
                print(f"Stored Elo rating at {rating_path}")
        else:
            print("Elo: no completed games to rate")

        if args.summary_json:
            summary_payload = {
                "games": len(indices),
                "completed": completed,
                "failures": failures,
                "summary": summary,
                "labels": {
                    "white": white_label,
                    "black": black_label,
                },
                "elo": {
                    "start": elo.rating_before,
                    "opponent": elo.opponent_rating,
                    "delta": elo.delta,
                    "new": elo.rating_after,
                    "expected_score": elo.expected_score,
                    "actual_score": elo.actual_score,
                    "games": elo.games,
                    "wins": elo.wins,
                    "losses": elo.losses,
                    "draws": elo.draws,
                },
                "timing_sec": time.perf_counter() - wall_start,
                "parameters": parameters_snapshot,
            }
            try:
                Path(args.summary_json).expanduser().write_text(
                    json.dumps(summary_payload, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                print(f"Wrote summary JSON to {args.summary_json}")
            except OSError as exc:
                print(f"Failed to write summary JSON: {exc}", file=sys.stderr)

        return 0 if failures == 0 else 1
    finally:
        connection.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    return run_batch(args)


if __name__ == "__main__":
    sys.exit(main())
