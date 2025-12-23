#!/usr/bin/env python
"""Evaluate skaks tactical performance on a CSV puzzle suite."""

import argparse
import csv
import os
import select
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import chess
import chess.pgn

DEFAULT_PUZZLE_FILE = Path(__file__).resolve().with_name("puzzles.csv")
DEFAULT_ENGINE = "skaks"
STOCKFISH_TIME_FACTOR = 0.35


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

    def bestmove(
        self,
        fen: str,
        *,
        depth: Optional[int] = None,
        movetime_ms: Optional[int] = None,
        moves: Optional[List[str]] = None,
    ) -> str:
        if depth is None and movetime_ms is None:
            raise ValueError("either depth or movetime_ms must be provided")
        if depth is not None and movetime_ms is not None:
            raise ValueError("depth and movetime_ms are mutually exclusive")
        self._send("ucinewgame")
        if moves and fen == chess.STARTING_FEN:
            self._send("position startpos moves " + " ".join(moves))
        else:
            self._send(f"position fen {fen}")
        if movetime_ms is not None:
            self._send(f"go movetime {movetime_ms}")
        else:
            assert depth is not None
            self._send(f"go depth {depth}")
        while True:
            line = self._read_line()
            if line.startswith("info "):
                continue
            if line.startswith("bestmove "):
                parts = line.split()
                if len(parts) >= 2:
                    token = parts[1].strip().lower()
                    if token == "(none)":
                        return "0000"
                    return token
                return "0000"

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


def _print_game_pgn(game: chess.pgn.Game) -> None:
    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
    print("PGN:")
    print(game.accept(exporter))


def _to_millis(seconds: float) -> int:
    millis = int(round(seconds * 1000.0))
    return max(millis, 1)


def _scale_stockfish_movetime(
    skaks_ms: Optional[int], stockfish_ms: Optional[int], override: bool
) -> Optional[int]:
    if skaks_ms is None:
        return stockfish_ms
    if override:
        return stockfish_ms
    scaled = int(round(skaks_ms * STOCKFISH_TIME_FACTOR))
    return max(scaled, 1)


def run_game(
    *,
    depth: Optional[int],
    limit: int,
    time_per_move: Optional[float],
    stockfish_time_per_move: Optional[float],
    stockfish_time_overridden: bool,
) -> int:
    stock_path = "stockfish"
    skaks_path = DEFAULT_ENGINE

    white_name = Path(skaks_path).name
    black_name = Path(stock_path).name

    board = chess.Board(chess.STARTING_FEN)
    game = chess.pgn.Game()
    game.headers["Event"] = "Skaks Validation"
    game.headers["Site"] = "Local"
    game.headers["Date"] = datetime.now(timezone.utc).strftime("%Y.%m.%d")
    game.headers["Round"] = "1"
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    game.headers["Result"] = "*"
    game.setup(board.copy(stack=False))
    node = game
    moves: List[str] = []
    final_fen_printed = False
    result_code = 0

    try:
        with (
            UciEngine(stock_path, timeout=30.0) as stockfish,
            UciEngine(skaks_path, timeout=240.0) as skaks,
        ):
            skaks_movetime_ms = (
                _to_millis(time_per_move) if time_per_move is not None else None
            )
            stockfish_initial_ms = (
                _to_millis(stockfish_time_per_move)
                if stockfish_time_per_move is not None
                else skaks_movetime_ms
            )
            stockfish_movetime_ms = _scale_stockfish_movetime(
                skaks_movetime_ms, stockfish_initial_ms, stockfish_time_overridden
            )

            for turn in range(limit):
                engine = skaks if turn % 2 == 0 else stockfish
                if skaks_movetime_ms is None:
                    assert depth is not None
                    chosen_depth = depth if engine == skaks else max(depth - 6, 1)
                    go_kwargs = {"depth": chosen_depth}
                else:
                    if engine == skaks:
                        movetime_ms = skaks_movetime_ms
                    else:
                        movetime_ms = stockfish_movetime_ms
                    if movetime_ms is None:
                        raise RuntimeError("time control not configured for Stockfish")
                    go_kwargs = {"movetime_ms": movetime_ms}
                try:
                    best_move = engine.bestmove(board.fen(), moves=moves, **go_kwargs)
                except Exception as e:
                    print(f"Engine error: {e}", file=sys.stderr)
                    print("Final position FEN:", board.fen())
                    _print_game_pgn(game)
                    return 3
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
                    print("Final position FEN:", board.fen())
                    final_fen_printed = True
                    break

                moves.append(best_move)
                move_obj = chess.Move.from_uci(best_move)
                if move_obj not in board.legal_moves:
                    print(
                        f"Turn {turn + 1}: Illegal move produced: {best_move}",
                        file=sys.stderr,
                    )
                    print("Final position FEN:", board.fen())
                    _print_game_pgn(game)
                    return 3
                node = node.add_variation(move_obj)
                board.push(move_obj)
                print(
                    f"Turn {turn + 1}: {'Stockfish' if turn % 2 == 0 else 'Skaks'} plays {best_move}"
                )
    except FileNotFoundError:
        print("Engine binary not found", file=sys.stderr)
        print("Final position FEN:", board.fen())
        _print_game_pgn(game)
        return 2
    except TimeoutError as exc:
        print(f"Engine timed out: {exc}", file=sys.stderr)
        print("Final position FEN:", board.fen())
        _print_game_pgn(game)
        return 3
    except RuntimeError as exc:
        print(f"Engine error: {exc}", file=sys.stderr)
        print("Final position FEN:", board.fen())
        _print_game_pgn(game)
        return 3

    if not final_fen_printed:
        print("Final position FEN:", board.fen())

    result_str = board.result(claim_draw=True)
    if not result_str:
        result_str = "*"
    game.headers["Result"] = result_str

    winner = "unknown"
    if result_str == "1-0":
        winner = white_name
    elif result_str == "0-1":
        winner = black_name
    elif result_str == "1/2-1/2":
        winner = "draw"

    print(f"Result: {result_str}")
    print(f"Winner: {winner}")

    _print_game_pgn(game)
    return result_code


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


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
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--depth",
        type=int,
        help="Search depth in plies per move (default: 9)",
    )
    group.add_argument(
        "--time-per-move",
        type=_positive_float,
        help="Seconds per move for Skaks (enables time controls)",
    )
    parser.add_argument(
        "--stockfish-time-per-move",
        type=_positive_float,
        help="Seconds per move for Stockfish when using time controls (default: match Skaks)",
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
    args = parser.parse_args(argv)
    stockfish_time_explicit = args.stockfish_time_per_move is not None
    if args.depth is None and args.time_per_move is None:
        args.depth = 9
    if args.time_per_move is not None and args.stockfish_time_per_move is None:
        args.stockfish_time_per_move = args.time_per_move
    args.stockfish_time_overridden = stockfish_time_explicit
    return args


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    run_game(
        depth=args.depth,
        limit=args.limit,
        time_per_move=args.time_per_move,
        stockfish_time_per_move=args.stockfish_time_per_move,
        stockfish_time_overridden=args.stockfish_time_overridden,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
