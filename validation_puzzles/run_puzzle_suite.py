#!/usr/bin/env python3
"""Evaluate skaks tactical performance on a CSV puzzle suite."""

import argparse
import csv
import os
import select
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

DEFAULT_PUZZLE_FILE = Path(__file__).resolve().with_name("puzzles.csv")
DEFAULT_ENGINE = "skaks"


@dataclass
class Puzzle:
    identifier: str
    fen: str
    moves: List[str]

    @classmethod
    def from_row(cls, row: dict):
        moves = [
            token.strip().lower() for token in row["Moves"].split() if token.strip()
        ]
        return cls(identifier=row["PuzzleId"], fen=row["FEN"], moves=moves)


class UciEngine:
    def __init__(self, binary: str, timeout: float):
        env = os.environ.copy()
        self._proc = subprocess.Popen(
            [binary],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=False,
            bufsize=0,
            env=env,
        )
        self._timeout = timeout
        self._init_protocol()

    def _init_protocol(self) -> None:
        self._send("uci")
        self._drain_until("uciok")
        self._ready()

    def _send(self, command: str) -> None:
        assert self._proc.stdin is not None
        payload = (command + "\n").encode("utf-8")
        self._proc.stdin.write(payload)
        self._proc.stdin.flush()

    def _ready(self) -> None:
        self._send("isready")
        self._drain_until("readyok")

    def _read_line(self) -> str:
        assert self._proc.stdout is not None
        fd = self._proc.stdout
        ready, _, _ = select.select([fd], [], [], self._timeout)
        if not ready:
            raise TimeoutError("engine response timed out")
        line = fd.readline()
        if line == b"":
            raise RuntimeError("engine terminated unexpectedly")
        text = line.decode("utf-8", errors="replace").strip()
        return text

    def _drain_until(self, token: str) -> None:
        while True:
            line = self._read_line()
            if line == token:
                return

    def bestmove(self, fen: str, depth: int, moves: Optional[List[str]] = None) -> str:
        self._send("ucinewgame")
        position_cmd = f"position fen {fen}"
        if moves:
            position_cmd += " moves " + " ".join(moves)
        self._send(position_cmd)
        self._send(f"go depth {depth}")
        while True:
            line = self._read_line()
            if line.startswith("info "):
                continue
            if line.startswith("bestmove "):
                return line.split()[1].strip().lower()

    def close(self) -> None:
        if self._proc.poll() is None:
            self._send("quit")
            self._proc.wait(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.close()
        except Exception:
            pass


def load_puzzles(path: Path, limit: Optional[int]) -> List[Puzzle]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        puzzles: List[Puzzle] = []
        for row in reader:
            if limit is not None and len(puzzles) >= limit:
                break
            puzzles.append(Puzzle.from_row(row))
    return puzzles


def run_suite(
    puzzles: Iterable[Puzzle], engine: UciEngine, depth: int, progress_interval: int
) -> Tuple[int, int, List[Tuple[Puzzle, str]]]:
    puzzle_list = list(puzzles)
    total = len(puzzle_list)
    solved = 0
    failures: List[Tuple[Puzzle, str]] = []

    for index, puzzle in enumerate(puzzle_list, start=1):
        expected_moves = puzzle.moves
        if not expected_moves:
            failures.append((puzzle, "no-solution-moves"))
            continue

        played_moves: List[str] = []
        puzzle_solved = True

        for ply_idx in range(0, len(expected_moves), 2):
            expected_move = expected_moves[ply_idx]
            predicted: Optional[str]
            try:
                predicted = engine.bestmove(puzzle.fen, depth, played_moves)
            except TimeoutError:
                failures.append((puzzle, f"timeout at ply {ply_idx}"))
                predicted = None
                puzzle_solved = False
                break
            except RuntimeError as exc:
                failures.append((puzzle, f"error:{exc}"))
                print()
                raise

            if predicted != expected_move:
                display = predicted if predicted is not None else "no-move"
                failures.append((
                    puzzle,
                    f"expected {expected_move} at ply {ply_idx}, got {display}",
                ))
                puzzle_solved = False
                break

            played_moves.append(expected_move)

            opponent_idx = ply_idx + 1
            if opponent_idx < len(expected_moves):
                played_moves.append(expected_moves[opponent_idx])

        if puzzle_solved:
            solved += 1

        if progress_interval > 0 and (index % progress_interval == 0 or index == total):
            percent = (index / total) * 100.0 if total else 100.0
            print(f"Progress: {index}/{total} ({percent:.1f}%)", end="\r", flush=True)

    if total:
        print()

    return solved, total, failures


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tactical puzzles against skaks.")
    parser.add_argument(
        "--puzzles",
        type=Path,
        default=DEFAULT_PUZZLE_FILE,
        help="CSV file with puzzle data (default: puzzles.csv next to this script)",
    )
    parser.add_argument(
        "--engine",
        default=DEFAULT_ENGINE,
        help="Engine binary to execute (default: skaks)",
    )
    parser.add_argument(
        "--stockfish",
        action="store_true",
        help="Use stockfish (overrides --engine)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=6,
        help="Search depth in plies for each puzzle (default: 6)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of puzzles (default: all)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for engine replies (default: 30)",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=50,
        help="Report progress every N puzzles (default: 50)",
    )
    parser.add_argument(
        "--show-failures", action="store_true", help="Print details for failed puzzles"
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    puzzles = load_puzzles(args.puzzles, args.limit)
    if not puzzles:
        print("No puzzles loaded.", file=sys.stderr)
        return 1

    engine_path = "stockfish" if args.stockfish else args.engine

    try:
        with UciEngine(engine_path, args.timeout) as engine:
            solved, total, failures = run_suite(
                puzzles, engine, args.depth, args.progress_interval
            )
    except FileNotFoundError:
        print(f"Engine binary not found: {engine_path}", file=sys.stderr)
        return 2
    except TimeoutError as exc:
        print(f"Engine timed out: {exc}", file=sys.stderr)
        return 3
    except RuntimeError as exc:
        print(f"Engine error: {exc}", file=sys.stderr)
        return 4

    percentage = (solved / total) * 100.0
    print(f"Solved {solved}/{total} puzzles at depth {args.depth} ({percentage:.1f}%)")

    if args.show_failures and failures:
        print("Failures:")
        for puzzle, predicted in failures:
            expected = puzzle.moves[0] if puzzle.moves else ""
            print(f"  {puzzle.identifier}: expected {expected}, got {predicted}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
