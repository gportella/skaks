#!/usr/bin/env python
"""Run every EPD puzzle suite with fixed depth/limit settings."""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def find_epd_files(directory: Path) -> List[Path]:
    return sorted(p for p in directory.glob("*.epd") if p.is_file())


def build_command(script: Path, epd: Path, args: argparse.Namespace) -> List[str]:
    command = [
        sys.executable,
        str(script),
        "--puzzles",
        str(epd),
        "--depth",
        str(args.depth),
        "--limit",
        str(args.limit),
        "--progress-interval",
        str(args.progress_interval),
    ]

    if args.stockfish:
        command.append("--stockfish")
    else:
        command.extend(["--engine", args.engine])
        if args.no_nnue:
            command.append("--no-nnue")

    if args.show_failures:
        command.append("--show-failures")

    if args.timeout is not None:
        command.extend(["--timeout", str(args.timeout)])

    return command


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Run all EPD puzzle suites.")
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing EPD files (default: script directory)",
    )
    parser.add_argument(
        "--engine",
        default="skaks",
        help="Engine binary to execute when not using stockfish (default: skaks_basic)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=6,
        help="Search depth passed to run_puzzle_suite.py (default: 6)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum puzzles per suite (default: 50)",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
        help="Progress reporting interval for the underlying runner (default: 10)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional reply timeout in seconds for the engine (default: inherited)",
    )
    parser.add_argument(
        "--stockfish",
        action="store_true",
        help="Use stockfish instead of the configured engine",
    )
    parser.add_argument(
        "--no-nnue",
        action="store_true",
        help="Disable NNUE evaluation when running skaks",
    )
    parser.add_argument(
        "--show-failures",
        action="store_true",
        help="Print failed puzzle details from the underlying runs",
    )

    args = parser.parse_args(argv)

    directory = args.directory.resolve()
    epd_files = find_epd_files(directory)
    if not epd_files:
        print(f"No EPD files found in {directory}", file=sys.stderr)
        return 1

    runner = Path(__file__).resolve().with_name("run_puzzle_suite.py")

    overall_status = 0
    for epd in epd_files:
        print(f"=== {epd.name} ===")
        command = build_command(runner, epd, args)
        result = subprocess.run(command, cwd=directory)
        if result.returncode != 0:
            overall_status = result.returncode
            print(f"Run failed for {epd.name} (exit {result.returncode})", file=sys.stderr)

    return overall_status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
