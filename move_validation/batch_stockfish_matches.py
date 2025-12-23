#!/usr/bin/env python
"""Run repeated Skaks vs Stockfish matches to stress move validation."""

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
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Depth passed to each skaks search (default: 4)",
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
        help="Path to skaks binary; forwarded to fight script if provided",
    )
    parser.add_argument(
        "--stockfish",
        action="store_true",
        help="Forward --stockfish flag to fight script",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional timeout override forwarded to fight script",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full stdout/stderr for every game (default: only failures)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fight_script = Path(__file__).resolve().with_name("fight_against_engine.py")
    base_args: List[str] = ["--depth", str(args.depth), "--limit", str(args.limit)]

    if args.engine:
        base_args.extend(["--engine", args.engine])
    if args.stockfish:
        base_args.append("--stockfish")
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
