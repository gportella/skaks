#!/usr/bin/env python
"""Evaluate skaks tactical performance on a CSV puzzle suite."""

import argparse
import csv
import os
from dataclasses import dataclass
import select
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import chess

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


def load_puzzles(path: Path, limit: Optional[int]) -> List[Puzzle]:
    suffix = path.suffix.lower()
    if suffix == ".epd":
        return load_puzzles_epd(path, limit)
    return load_puzzles_csv(path, limit)


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
    move_list = [move.uci().lower() for move in (best_moves or []) if move is not None]
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
        if moves and fen == chess.STARTING_FEN:
            self._send("position startpos moves " + " ".join(moves))
        else:
            self._send(f"position fen {fen}")
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


def run_game(depth: int, limit: int) -> int:
    stock_path = "stockfish"
    skaks_path = DEFAULT_ENGINE

    l_depth = depth

    try:
        with (
            UciEngine(stock_path, timeout=10.0) as stockfish,
            UciEngine(skaks_path, timeout=30.0) as skaks,
        ):
            fen = chess.STARTING_FEN
            moves = []
            for turn in range(limit):  # Limit to 20 moves for brevity
                engine = skaks if turn % 2 == 0 else stockfish
                if engine == stockfish:
                    depth = max(
                        1, l_depth - 3
                    )  # Stockfish plays better, need it to last longer
                else:
                    depth = l_depth
                try:
                    best_move = engine.bestmove(fen, depth=depth, moves=moves)
                except Exception as e:
                    print(f"Engine error: {e}", file=sys.stderr)
                    print("Final position FEN:", fen)
                    return 3
                board = chess.Board(fen)
                if best_move == "0000":
                    if board.is_checkmate():
                        print(
                            f"Turn {turn + 1}: {'Stockfish' if turn % 2 == 0 else 'Skaks'} reports checkmate."
                        )
                    elif board.is_stalemate():
                        print(
                            f"Turn {turn + 1}: {'Stockfish' if turn % 2 == 0 else 'Skaks'} reports stalemate."
                        )
                    else:
                        print(
                            f"Turn {turn + 1}: {'Stockfish' if turn % 2 == 0 else 'Skaks'} reports no legal moves."
                        )
                    print("Final position FEN:", fen)
                    return 0

                moves.append(best_move)
                board.push_uci(best_move)
                fen = board.fen()
                print(
                    f"Turn {turn + 1}: {'Stockfish' if turn % 2 == 0 else 'Skaks'} plays {best_move}"
                )
            print("Final position FEN:", fen)
    except FileNotFoundError:
        print("Engine binary not found", file=sys.stderr)
        return 2
    except TimeoutError as exc:
        print(f"Engine timed out: {exc}", file=sys.stderr)
        return 3
    except RuntimeError as exc:
        print(f"Engine error: {exc}", file=sys.stderr)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tactical puzzles against skaks.")
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
        default=3,
        help="Search depth in plies for each puzzle (default: 6)",
    )
    parser.add_argument(
        "--limit", type=int, default=30, help="Limit number of puzzles (default: all)"
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
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    run_game(depth=args.depth, limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
