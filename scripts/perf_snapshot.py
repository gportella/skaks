#!/usr/bin/env python3
"""Capture skaks version info and perf snapshots for a handful of FENs."""

import argparse
import datetime
import os
import shutil
import subprocess
import sys
from pathlib import Path

FEN_CASES = [
    (
        "startpos",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ),
    (
        "balanced_midgame",
        "r2q1rk1/pp1bbppp/2n1pn2/2pp4/3P1B2/2P1PN2/PP1NBPPP/R2Q1RK1 w - - 0 9",
    ),
    (
        "simplified_endgame",
        "8/5k2/3p4/4p3/4P3/3P4/5K2/8 w - - 0 1",
    ),
]

PERF_ITERS = 3
SEARCH_DEPTH = 8
OUTPUT_FILE = Path("perf_runs.log")


def locate_skaks() -> str:
    """Locate the skaks binary, honouring SKAKS_BIN when provided."""
    override = os.environ.get("SKAKS_BIN")
    if override:
        return override

    discovered = shutil.which("skaks")
    if discovered:
        return discovered

    fallback = Path(__file__).resolve().parent / "build/debug/src/skaks"
    if fallback.exists():
        return str(fallback)

    raise SystemExit(
        "Unable to locate skaks binary. Export SKAKS_BIN or ensure it is on PATH."
    )


def run_command(args, env=None):
    """Run a command and surface stderr on failure."""
    try:
        return subprocess.run(
            args, capture_output=True, text=True, check=True, env=env
        )
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(f"Command failed: {' '.join(args)}\n")
        if exc.stdout:
            sys.stderr.write(exc.stdout)
        if exc.stderr:
            sys.stderr.write(exc.stderr)
        raise


def main():
    parser = argparse.ArgumentParser(description="Capture skaks perf snapshots")
    parser.add_argument("--nnue", help="Path to NNUE file to use")
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Search threads to use for perf runs (0 = engine default)",
    )
    args = parser.parse_args()

    binary = locate_skaks()
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    if args.threads < 0:
        raise SystemExit("--threads must be non-negative")

    env = os.environ.copy()
    if args.nnue:
        env.setdefault("SKAKS_NNUE_BIG", args.nnue)
        env.setdefault("SKAKS_NNUE_SMALL", args.nnue)

    lines = []
    lines.append(f"=== perf snapshot {timestamp} ===")
    lines.append(f"binary: {binary}")
    lines.append(f"threads: {args.threads}")

    version_output = run_command([binary, "-vv"], env=env).stdout.strip()
    if args.nnue:
        version_output += f" NNUE: {args.nnue}"
    lines.append("--- version ---")
    lines.extend(version_output.splitlines())

    lines.append("--- perf ---")
    for depth in sorted({1, 2, 3, 4, SEARCH_DEPTH}):
        for name, fen in FEN_CASES:
            cmd = [
                binary,
                "--perf",
                "--perf-iters",
                str(PERF_ITERS),
                "--depth",
                str(depth),
                "--fen",
                fen,
                "--threads",
                str(args.threads),
            ]
            result = run_command(cmd, env=env)
            lines.append(f"[case] {name}")
            lines.extend(result.stdout.strip().splitlines())

    lines.append("")

    with OUTPUT_FILE.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


if __name__ == "__main__":
    main()
