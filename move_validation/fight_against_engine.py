#!/usr/bin/env python
"""Evaluate skaks tactical performance on a CSV puzzle suite."""

import argparse
import csv
import os
import select
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import chess

DEFAULT_PUZZLE_FILE = Path(__file__).resolve().with_name("puzzles.csv")
DEFAULT_ENGINE = "skaks"


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


def run_game(depth: int, limit: int) -> int:
    stock_path = "stockfish"
    skaks_path = DEFAULT_ENGINE

    try:
        with (
            UciEngine(stock_path, timeout=10.0) as stockfish,
            UciEngine(skaks_path, timeout=10.0) as skaks,
        ):
            fen = chess.STARTING_FEN
            moves = []
            for turn in range(limit):  # Limit to 20 moves for brevity
                engine = stockfish if turn % 2 == 0 else skaks
                best_move = engine.bestmove(fen, depth=depth, moves=moves)
                if not best_move:
                    print("No move found, game over.")
                    break
                moves.append(best_move)
                board = chess.Board(fen)
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
        default=6,
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


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
