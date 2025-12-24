#!/usr/bin/env python
"""Evaluate skaks tactical performance on a CSV puzzle suite."""

import argparse
import os
import select
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import chess
import chess.pgn

DEFAULT_PUZZLE_FILE = Path(__file__).resolve().with_name("puzzles.csv")
DEFAULT_ENGINE = "skaks"
STOCKFISH_TIME_FACTOR = 0.35
MIN_CLOCK_MS = 1


@dataclass
class GoClock:
    wtime: Optional[int] = None
    btime: Optional[int] = None
    winc: Optional[int] = None
    binc: Optional[int] = None
    movestogo: Optional[int] = None


@dataclass
class MatchClock:
    white_ms: int
    black_ms: int
    white_increment_ms: int = 0
    black_increment_ms: int = 0
    white_moves_to_go: Optional[int] = None
    black_moves_to_go: Optional[int] = None

    def snapshot(self, stm: chess.Color) -> GoClock:
        if stm == chess.WHITE:
            movestogo = self.white_moves_to_go
        else:
            movestogo = self.black_moves_to_go
        return GoClock(
            wtime=max(self.white_ms, 0),
            btime=max(self.black_ms, 0),
            winc=self.white_increment_ms if self.white_increment_ms else None,
            binc=self.black_increment_ms if self.black_increment_ms else None,
            movestogo=movestogo if movestogo and movestogo > 0 else None,
        )

    def apply_elapsed(self, mover: chess.Color, spent_ms: int) -> None:
        spent = max(spent_ms, 0)
        if mover == chess.WHITE:
            self.white_ms = max(self.white_ms - spent, 0)
            if self.white_increment_ms:
                self.white_ms += self.white_increment_ms
            if self.white_moves_to_go and self.white_moves_to_go > 0:
                self.white_moves_to_go = max(self.white_moves_to_go - 1, 0)
        else:
            self.black_ms = max(self.black_ms - spent, 0)
            if self.black_increment_ms:
                self.black_ms += self.black_increment_ms
            if self.black_moves_to_go and self.black_moves_to_go > 0:
                self.black_moves_to_go = max(self.black_moves_to_go - 1, 0)


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
        clock: Optional[GoClock] = None,
        moves: Optional[List[str]] = None,
    ) -> str:
        provided = sum(1 for flag in (depth, movetime_ms, clock) if flag is not None)
        if provided != 1:
            raise ValueError(
                "exactly one of depth, movetime_ms, or clock must be provided"
            )
        self._send("ucinewgame")
        if moves and fen == chess.STARTING_FEN:
            self._send("position startpos moves " + " ".join(moves))
        else:
            self._send(f"position fen {fen}")
        if movetime_ms is not None:
            self._send(f"go movetime {movetime_ms}")
        elif depth is not None:
            self._send(f"go depth {depth}")
        else:
            assert clock is not None
            parts: List[str] = ["go"]
            if clock.wtime is not None:
                parts.extend(["wtime", str(clock.wtime)])
            if clock.btime is not None:
                parts.extend(["btime", str(clock.btime)])
            if clock.winc is not None:
                parts.extend(["winc", str(clock.winc)])
            if clock.binc is not None:
                parts.extend(["binc", str(clock.binc)])
            if clock.movestogo is not None:
                parts.extend(["movestogo", str(clock.movestogo)])
            self._send(" ".join(parts))
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
    clock_seconds: Optional[float],
    stockfish_clock_seconds: Optional[float],
    increment_seconds: Optional[float],
    stockfish_increment_seconds: Optional[float],
    moves_to_go: Optional[int],
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
            skaks_movetime_ms: Optional[int]
            stockfish_movetime_ms: Optional[int]
            clock: Optional[MatchClock]

            clock = None
            skaks_movetime_ms = None
            stockfish_movetime_ms = None

            if time_per_move is not None:
                skaks_movetime_ms = _to_millis(time_per_move)
                stockfish_initial_ms = (
                    _to_millis(stockfish_time_per_move)
                    if stockfish_time_per_move is not None
                    else skaks_movetime_ms
                )
                stockfish_movetime_ms = _scale_stockfish_movetime(
                    skaks_movetime_ms, stockfish_initial_ms, stockfish_time_overridden
                )
            elif clock_seconds is not None:
                white_base = max(_to_millis(clock_seconds), MIN_CLOCK_MS)
                if stockfish_clock_seconds is not None:
                    black_base = max(_to_millis(stockfish_clock_seconds), MIN_CLOCK_MS)
                else:
                    black_base = max(
                        int(round(white_base * STOCKFISH_TIME_FACTOR)), MIN_CLOCK_MS
                    )

                white_inc = (
                    _to_millis(increment_seconds)
                    if increment_seconds is not None
                    else 0
                )
                if stockfish_increment_seconds is not None:
                    black_inc = _to_millis(stockfish_increment_seconds)
                else:
                    if white_inc:
                        black_inc = max(
                            int(round(white_inc * STOCKFISH_TIME_FACTOR)), MIN_CLOCK_MS
                        )
                    else:
                        black_inc = 0

                clock = MatchClock(
                    white_ms=white_base,
                    black_ms=black_base,
                    white_increment_ms=white_inc,
                    black_increment_ms=black_inc,
                    white_moves_to_go=moves_to_go,
                    black_moves_to_go=moves_to_go,
                )
            else:
                skaks_movetime_ms = None

            for turn in range(limit):
                engine = skaks if turn % 2 == 0 else stockfish
                try:
                    if clock is not None:
                        go_clock = clock.snapshot(board.turn)
                        search_start = time.perf_counter()
                        best_move = engine.bestmove(
                            board.fen(), moves=moves, clock=go_clock
                        )
                        search_end = time.perf_counter()
                    elif skaks_movetime_ms is not None:
                        movetime_ms = (
                            skaks_movetime_ms
                            if engine == skaks
                            else stockfish_movetime_ms
                        )
                        if movetime_ms is None:
                            raise RuntimeError(
                                "time control not configured for Stockfish"
                            )
                        search_start = None
                        search_end = None
                        best_move = engine.bestmove(
                            board.fen(), moves=moves, movetime_ms=movetime_ms
                        )
                    else:
                        assert depth is not None
                        chosen_depth = depth if engine == skaks else max(depth - 6, 1)
                        search_start = None
                        search_end = None
                        best_move = engine.bestmove(
                            board.fen(), moves=moves, depth=chosen_depth
                        )
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
                if (
                    clock is not None
                    and search_start is not None
                    and search_end is not None
                ):
                    elapsed_ms = int(round((search_end - search_start) * 1000.0))
                    mover = chess.BLACK if engine == stockfish else chess.WHITE
                    clock.apply_elapsed(mover, elapsed_ms)
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
    group.add_argument(
        "--clock",
        type=_positive_float,
        help="Primary clock time in seconds for Skaks (enables clock controls)",
    )
    parser.add_argument(
        "--stockfish-time-per-move",
        type=_positive_float,
        help="Seconds per move for Stockfish when using time controls (default: match Skaks)",
    )
    parser.add_argument(
        "--stockfish-clock",
        type=_positive_float,
        help="Clock time in seconds for Stockfish (default: scaled from --clock)",
    )
    parser.add_argument(
        "--increment",
        type=_positive_float,
        help="Increment in seconds added to Skaks after each move",
    )
    parser.add_argument(
        "--stockfish-increment",
        type=_positive_float,
        help="Increment in seconds added to Stockfish after each move",
    )
    parser.add_argument(
        "--moves-to-go",
        type=int,
        help="Approximate moves remaining to next time control",
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
    stockfish_clock_explicit = args.stockfish_clock is not None

    mode_count = sum(
        flag is not None for flag in (args.depth, args.time_per_move, args.clock)
    )
    if mode_count == 0:
        args.depth = 9
    elif mode_count > 1:
        parser.error("choose exactly one of --depth, --time-per-move, or --clock")

    if args.time_per_move is None and args.stockfish_time_per_move is not None:
        parser.error("--stockfish-time-per-move requires --time-per-move")
    if args.clock is None and args.stockfish_clock is not None:
        parser.error("--stockfish-clock requires --clock")
    if args.clock is None and (
        args.increment is not None or args.stockfish_increment is not None
    ):
        parser.error("increments require --clock")
    if args.moves_to_go is not None and args.moves_to_go <= 0:
        parser.error("--moves-to-go must be positive")

    if args.time_per_move is not None and args.stockfish_time_per_move is None:
        args.stockfish_time_per_move = args.time_per_move

    args.stockfish_time_overridden = stockfish_time_explicit
    args.stockfish_clock_overridden = stockfish_clock_explicit
    return args


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    run_game(
        depth=args.depth,
        limit=args.limit,
        time_per_move=args.time_per_move,
        stockfish_time_per_move=args.stockfish_time_per_move,
        clock_seconds=args.clock,
        stockfish_clock_seconds=args.stockfish_clock,
        increment_seconds=args.increment,
        stockfish_increment_seconds=args.stockfish_increment,
        moves_to_go=args.moves_to_go,
        stockfish_time_overridden=args.stockfish_time_overridden,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
