#!/usr/bin/env python
"""Evaluate skaks tactical performance on a CSV puzzle suite."""

import argparse
import csv
import datetime
import os
import select
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import chess
except ImportError:  # pragma: no cover
    chess = None
    print("chess library not available, cannot parse EPD files.", file=sys.stderr)

DEFAULT_PUZZLE_FILE = Path(__file__).resolve().with_name("puzzles.csv")
DEFAULT_ENGINE = "skaks"
OUTPUT_FILE = Path("puzzle_runs.log")


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


def run_command(args):
    """Run a command and surface stderr on failure."""
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(f"Command failed: {' '.join(args)}\n")
        if exc.stdout:
            sys.stderr.write(exc.stdout)
        if exc.stderr:
            sys.stderr.write(exc.stderr)
        raise


def find_epd_files(directory: Path) -> List[Path]:
    return sorted(p for p in directory.glob("*.epd") if p.is_file())


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
    def __init__(self, binary: str, timeout: float, extra_args: Optional[List[str]] = None):
        self.binary = binary
        self.eval_mode: Optional[str] = None
        self.hash_mb: Optional[int] = None
        env = os.environ.copy()
        argv = [binary]
        if extra_args:
            argv.extend(extra_args)
        self._proc = subprocess.Popen(
            argv,
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
            if line.startswith("info string eval_mode="):
                self.eval_mode = line.split("=", 1)[1].strip().lower() or None
            if line.startswith("option name Hash "):
                parts = line.split()
                if "default" in parts:
                    idx = parts.index("default")
                    if idx + 1 < len(parts):
                        try:
                            self.hash_mb = int(parts[idx + 1])
                        except ValueError:
                            self.hash_mb = None
            if line == token:
                return

    def clear_tt(self) -> None:
        if self.hash_mb is None:
            self._send("ucinewgame")
            self._ready()
            return
        temp_hash = 1 if self.hash_mb != 1 else 2
        self._send(f"setoption name Hash value {temp_hash}")
        self._send(f"setoption name Hash value {self.hash_mb}")
        self._ready()

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


def load_puzzles(
    path: Path, directory: Path, limit: Optional[int]
) -> Dict[str, List[Puzzle]]:
    epd_files = find_epd_files(directory=directory)
    out: Dict[str, List[Puzzle]] = {}
    candidates: List[Path]

    if path.exists():
        candidates = [path]
    else:
        candidates = epd_files

    for candidate in candidates:
        suffix = candidate.suffix.lower()
        if suffix == ".epd":
            puzzles = load_puzzles_epd(candidate, limit)
        elif suffix == ".csv":
            puzzles = load_puzzles_csv(candidate, limit)
        else:
            continue

        if puzzles:
            out[candidate.name] = puzzles

    return out


def load_puzzles_csv(path: Path, limit: Optional[int]) -> List[Puzzle]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        puzzles: List[Puzzle] = []
        for row in reader:
            if limit is not None and len(puzzles) >= limit:
                break
            puzzles.append(Puzzle.from_row(row))
    return puzzles


def parse_epd_line(line: str) -> Optional[Puzzle]:
    stripped = line.split("#", 1)[0].strip()
    if not stripped or chess is None:
        return None

    segments = [segment.strip() for segment in stripped.split(";") if segment.strip()]
    if not segments:
        return None

    epd_fragment = segments[0]

    board = chess.Board()
    try:
        operations = board.set_epd(epd_fragment)
    except ValueError:
        return None

    best_moves = operations.get("bm")
    if isinstance(best_moves, chess.Move):
        iterable_moves = [best_moves]
    elif isinstance(best_moves, Iterable) and not isinstance(best_moves, (str, bytes)):
        iterable_moves = list(best_moves)
    else:
        iterable_moves = []

    move_list = [move.uci().lower() for move in iterable_moves if move is not None]
    if not move_list:
        return None

    identifier = epd_fragment
    for segment in segments[1:]:
        if segment.lower().startswith("id "):
            candidate = segment[3:].strip()
            if (
                candidate.startswith('"')
                and candidate.endswith('"')
                and len(candidate) >= 2
            ):
                candidate = candidate[1:-1]
            identifier = candidate or identifier
            break

    fen = board.fen()
    return Puzzle(identifier=identifier, fen=fen, moves=move_list)


def load_puzzles_epd(path: Path, limit: Optional[int]) -> List[Puzzle]:
    puzzles: List[Puzzle] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parsed = parse_epd_line(line)
            if parsed is None:
                continue
            puzzles.append(parsed)
            if limit is not None and len(puzzles) >= limit:
                break
    return puzzles


def run_suite(
    puzzles: Dict[str, List[Puzzle]],
    directory: Path,
    engine: UciEngine,
    depth: int,
    progress_interval: int,
    clear_tt_per_puzzle: bool,
) -> Tuple[int, int, List[Tuple[Puzzle, str]]]:
    log_enabled = "skaks" in engine.binary
    log_handle = None

    if log_enabled:
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        lines = []
        lines.append(f"=== perf snapshot {timestamp} ===")
        lines.append(f"binary: {engine.binary}")
        if engine.eval_mode:
            lines.append(f"eval_mode: {engine.eval_mode}")
        else:
            lines.append("eval_mode: unknown")

        version_output = run_command([engine.binary, "-vv"]).stdout.strip()
        lines.append("--- version ---")
        lines.extend(version_output.splitlines())

        lines.append("--- perf ---")
        log_handle = OUTPUT_FILE.open("a", encoding="utf-8")
        log_handle.write("\n".join(lines) + "\n")

    try:
        overall_solved = 0
        overall_total = 0
        failures: List[Tuple[Puzzle, str]] = []

        for fname, puzzle_list in puzzles.items():
            file_total = len(puzzle_list)
            file_solved = 0
            last_progress_len = 0

            for index, puzzle in enumerate(puzzle_list, start=1):
                expected_moves = puzzle.moves
                if not expected_moves:
                    failures.append((puzzle, "no-solution-moves"))
                    continue

                played_moves: List[str] = []
                puzzle_solved = True
                if clear_tt_per_puzzle:
                    engine.clear_tt()

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
                        failures.append(
                            (
                                puzzle,
                                f"{expected_move} at ply {ply_idx}, computed {display}",
                            )
                        )
                        puzzle_solved = False
                        break

                    played_moves.append(expected_move)

                    opponent_idx = ply_idx + 1
                    if opponent_idx < len(expected_moves):
                        played_moves.append(expected_moves[opponent_idx])

                if puzzle_solved:
                    file_solved += 1

                if progress_interval > 0 and (
                    index % progress_interval == 0 or index == file_total
                ):
                    percent = (index / file_total) * 100.0 if file_total else 100.0
                    progress_line = f"Progress: {index}/{file_total} ({percent:.1f}%)"
                    padding = max(0, last_progress_len - len(progress_line))
                    sys.stdout.write("\r" + progress_line + " " * padding)
                    sys.stdout.flush()
                    last_progress_len = len(progress_line)
            overall_solved += file_solved
            overall_total += file_total

            if last_progress_len:
                sys.stdout.write("\r" + " " * last_progress_len + "\r")
                sys.stdout.flush()

            if file_total:
                percentage = (file_solved / file_total) * 100.0
            else:
                percentage = 0.0

            summary_line = f"Solved {file_solved}/{file_total} puzzles at depth {depth} ({percentage:.1f}%)"
            print(f"[{fname}] {summary_line}")
            if log_handle:
                log_handle.write(f"[{fname}] {summary_line}\n")

        overall_line = (
            f"Overall solved {overall_solved}/{overall_total} puzzles at depth {depth}"
        )
        if log_handle:
            log_handle.write(overall_line + "\n")
    finally:
        if log_handle:
            log_handle.close()

    return overall_solved, overall_total, failures


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tactical puzzles against skaks.")
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing EPD files (default: script directory)",
    )
    parser.add_argument(
        "--puzzles",
        type=Path,
        default=DEFAULT_PUZZLE_FILE,
        help="CSV or EPD file with puzzle data (default: puzzles.csv next to this script)",
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
        default=10,
        help="Report progress every N puzzles (default: 10)",
    )
    parser.add_argument(
        "--show-failures", action="store_true", help="Print details for failed puzzles"
    )
    parser.add_argument(
        "--clear-tt-per-puzzle",
        action="store_true",
        help="Clear the engine transposition table before each puzzle (default: on)",
    )
    parser.add_argument(
        "--no-clear-tt",
        action="store_false",
        dest="clear_tt_per_puzzle",
        help="Do not clear the engine transposition table between puzzles",
    )
    parser.set_defaults(clear_tt_per_puzzle=True)
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    directory = args.directory.resolve()
    puzzles = load_puzzles(args.puzzles, directory, args.limit)
    if not puzzles:
        print("No puzzles loaded.", file=sys.stderr)
        return 1

    engine_path = "stockfish" if args.stockfish else args.engine

    try:
        extra_args: List[str] = []
        with UciEngine(engine_path, args.timeout, extra_args=extra_args) as engine:
            solved, total, failures = run_suite(
                puzzles,
                directory,
                engine,
                args.depth,
                args.progress_interval,
                args.clear_tt_per_puzzle,
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

    percentage = (solved / total) * 100.0 if total else 0.0
    print(f"Solved {solved}/{total} puzzles at depth {args.depth} ({percentage:.1f}%)")

    if args.show_failures and failures:
        print("Failures:")
        for puzzle, predicted in failures:
            expected = puzzle.moves[0] if puzzle.moves else ""
            print(f"  {puzzle.identifier}: {expected} ==> {predicted}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
