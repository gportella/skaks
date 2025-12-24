#!/usr/bin/env python
"""Run repeated Skaks matches to stress move validation."""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def run_single_match(
    fight_script: Path, base_args: List[str]
) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(fight_script)] + base_args
    return subprocess.run(cmd, capture_output=True, text=True)


def extract_outcome(stdout: str) -> str:
    lowered = stdout.lower()
    winner_marker = "winner:"
    idx = lowered.find(winner_marker)
    if idx != -1:
        rest = lowered[idx + len(winner_marker) :].strip()
        winner_line = rest.splitlines()[0].strip() if rest else ""
        if "skaks" in winner_line:
            return "skaks"
        if "stockfish" in winner_line:
            return "stockfish"
        if winner_line.startswith("draw"):
            return "draw"
        if winner_line.startswith("unknown"):
            return "unknown"
    if 'result "1-0"' in lowered or "result: 1-0" in lowered:
        return "skaks"
    if 'result "0-1"' in lowered or "result: 0-1" in lowered:
        return "stockfish"
    if (
        'result "1/2-1/2"' in lowered
        or "result: 1/2-1/2" in lowered
        or "result \u00bd-\u00bd" in lowered
    ):
        return "draw"
    return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch runner that repeatedly invokes fight_against_engine.py"
    )
    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Number of games to play (default: 100)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--depth",
        type=int,
        help="Depth passed to the reference engine (default: 8)",
    )
    group.add_argument(
        "--time-per-move",
        type=float,
        help="Seconds per move for the reference engine",
    )
    group.add_argument(
        "--clock",
        type=float,
        help="Primary clock time in seconds for the reference engine",
    )
    parser.add_argument(
        "--opponent-time-per-move",
        type=float,
        help="Seconds per move for the opponent engine",
    )
    parser.add_argument(
        "--opponent-clock",
        type=float,
        help="Clock time in seconds for the opponent engine",
    )
    parser.add_argument(
        "--increment",
        type=float,
        help="Increment in seconds added to the reference engine after each move",
    )
    parser.add_argument(
        "--opponent-increment",
        type=float,
        help="Increment in seconds added to the opponent engine after each move",
    )
    parser.add_argument(
        "--moves-to-go",
        type=int,
        help="Approximate moves remaining to next time control",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=120,
        help="Half-move limit provided to the fight script (default: 120)",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default=None,
        help="Reference engine binary; forwarded to fight script",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default=None,
        help="Opponent engine binary; forwarded to fight script",
    )
    parser.add_argument(
        "--stockfish",
        action="store_true",
        help="Shortcut for --opponent stockfish",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional timeout override forwarded to fight script",
    )
    parser.add_argument(
        "--no-handicap",
        action="store_true",
        help="Disable opponent handicap when forwarding to fight script",
    )
    parser.add_argument(
        "--handicap-factor",
        type=float,
        default=None,
        help="Override handicap factor forwarded to fight script",
    )
    parser.add_argument(
        "--handicap-depth",
        type=int,
        default=None,
        help="Override handicap depth forwarded to fight script",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full stdout/stderr for every game (default: only failures)",
    )
    args = parser.parse_args()

    if args.depth is None and args.time_per_move is None and args.clock is None:
        args.depth = 8

    if args.stockfish and args.opponent:
        parser.error("--stockfish cannot be combined with --opponent")
    if args.stockfish:
        args.opponent = "stockfish"

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

    if args.handicap_factor is not None and args.handicap_factor <= 0:
        parser.error("--handicap-factor must be positive")
    if args.handicap_depth is not None and args.handicap_depth < 0:
        parser.error("--handicap-depth must be non-negative")

    return args


def main() -> int:
    args = parse_args()
    fight_script = Path(__file__).resolve().with_name("fight_against_engine.py")
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
    if args.opponent:
        base_args.extend(["--opponent", args.opponent])
    if args.no_handicap:
        base_args.append("--no-handicap")
    if args.handicap_factor is not None:
        base_args.extend(["--handicap-factor", str(args.handicap_factor)])
    if args.handicap_depth is not None:
        base_args.extend(["--handicap-depth", str(args.handicap_depth)])
    if args.timeout is not None:
        base_args.extend(["--timeout", str(args.timeout)])

    failures = 0
    outcomes: Dict[str, int] = {"skaks": 0, "stockfish": 0, "draw": 0, "unknown": 0}
    for idx in range(1, args.games + 1):
        result = run_single_match(fight_script, base_args)
        success = result.returncode == 0
        marker = "OK" if success else "FAIL"
        header = f"[{idx}/{args.games}] {marker}"
        outcome_label = extract_outcome(result.stdout)
        illegal_detected = (
            "illegal" in result.stdout.lower() or "illegal" in result.stderr.lower()
        )
        if not success or illegal_detected:
            outcome_label = "unknown"
        outcomes[outcome_label] += 1
        print(f"\r{header} winner={outcome_label:<9}", end="", flush=True)
        if not success or illegal_detected:
            print()
            failures += 1
            if not success:
                print(f"  exit code: {result.returncode}")
            if not args.verbose:
                print("  stdout snippet:")
                print("    " + result.stdout.strip().replace("\n", "\n    "))
                if result.stderr.strip():
                    print("  stderr snippet:")
                    print("    " + result.stderr.strip().replace("\n", "\n    "))
            else:
                print("--- stdout ---")
                print(result.stdout)
                print("--- stderr ---")
                print(result.stderr)

    print()
    if failures == 0:
        print(f"Completed {args.games} games. Failures detected: {failures}.")
    else:
        print(
            f"Completed {args.games} games. Failures detected: {failures}. "
            "Review output above for details."
        )

    total_finished = sum(outcomes.values())
    finished_results = total_finished - outcomes["unknown"]
    if total_finished > 0:
        print("Results summary:")
        if finished_results > 0:
            for label in ("skaks", "stockfish", "draw"):
                count = outcomes[label]
                pct = (count / finished_results) * 100.0
                print(f"  {label:9}: {count:4d} ({pct:5.1f}%)")
        else:
            print("  no completed games to summarize")
        if outcomes["unknown"]:
            count = outcomes["unknown"]
            pct = (count / total_finished) * 100.0
            print(f"  unknown  : {count:4d} ({pct:5.1f}%)")
        if finished_results > 0:
            scored = outcomes["skaks"] + outcomes["draw"] * 0.5
            avg_score = scored / finished_results
            print(f"  skaks average score: {avg_score:.3f} / 1.000")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
