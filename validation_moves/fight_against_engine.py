#!/usr/bin/env python
"""Evaluate skaks tactical performance on a CSV puzzle suite."""

import argparse
import os
import select
import shutil
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
DEFAULT_HANDICAP_FACTOR = 0.35
DEFAULT_HANDICAP_DEPTH = 6
MIN_CLOCK_MS = 1


def _resolve_engine(path_str: str) -> str:
    """Resolve an engine binary path, searching common build locations and PATH."""
    candidate = Path(path_str)
    search = []
    if candidate.is_absolute():
        search.append(candidate)
    else:
        search.append(Path.cwd() / candidate)
        repo_root = Path(__file__).resolve().parent.parent
        search.append(repo_root / candidate)
        search.append(repo_root / "build" / "debug" / "bin" / candidate.name)
        search.append(repo_root / "build" / "release" / "bin" / candidate.name)
    for cand in search:
        if cand.exists():
            return str(cand.resolve())
    which = shutil.which(path_str)
    if which:
        return which
    raise FileNotFoundError(f"Engine binary not found: {path_str}")


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
    def __init__(
        self,
        binary: str,
        timeout: float,
        *,
        add_uci_arg: bool = False,
        extra_args: Optional[list[str]] = None,
    ):
        env = os.environ.copy()
        argv = [binary]
        if add_uci_arg:
            argv.append("--uci")
        if extra_args:
            argv.extend(extra_args)
        self._transcript: list[str] = []
        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
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
            err_msg = "engine terminated unexpectedly"
            code = self._proc.poll()
            if code is not None:
                err_msg += f" (exit {code})"
            if self._proc.stderr is not None:
                try:
                    tail = self._proc.stderr.read()
                    if tail:
                        decoded = tail.decode("utf-8", errors="replace").strip()
                        err_msg += f": {decoded}"
                except Exception:
                    pass
            if self._transcript:
                err_msg += f" | stdout prior: {' | '.join(self._transcript[-10:])}"
            raise RuntimeError(err_msg)
        text = line.decode("utf-8", errors="replace").strip()
        self._transcript.append(text)
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
        if moves:
            if fen == chess.STARTING_FEN:
                self._send("position startpos moves " + " ".join(moves))
            else:
                self._send(f"position fen {fen} moves " + " ".join(moves))
        else:
            if fen == chess.STARTING_FEN:
                self._send("position startpos")
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


def _scale_handicapped_time(base_ms: int, factor: float) -> int:
    scaled = int(round(base_ms * factor))
    return max(scaled, MIN_CLOCK_MS)


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _extra_args(
    params_path: Optional[str], nnue_path: Optional[str]
) -> Optional[list[str]]:
    argv: list[str] = []
    if params_path:
        argv.extend(["--params", params_path])
    if nnue_path:
        argv.extend(["--nnue", nnue_path])
    return argv or None


def _needs_uci_arg(binary: str) -> bool:
    name = Path(binary).name.lower()
    return name.startswith("skaks")


def run_game(
    *,
    depth: Optional[int],
    limit: int,
    time_per_move: Optional[float],
    opponent_time_per_move: Optional[float],
    clock_seconds: Optional[float],
    opponent_clock_seconds: Optional[float],
    increment_seconds: Optional[float],
    opponent_increment_seconds: Optional[float],
    moves_to_go: Optional[int],
    reference_engine: str,
    opponent_engine: str,
    reference_params: Optional[str],
    opponent_params: Optional[str],
    reference_nnue: Optional[str],
    opponent_nnue: Optional[str],
    handicap_factor: float,
    handicap_depth: int,
    opponent_depth_factor: Optional[float],
    handicap_enabled: bool,
) -> int:
    reference_path = _resolve_engine(reference_engine)
    opponent_path = _resolve_engine(opponent_engine)

    white_name = Path(reference_path).name
    black_name = Path(opponent_path).name

    board = chess.Board(chess.STARTING_FEN)
    root_fen = board.fen()
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

    move_limit_hit = False
    try:
        with (
            UciEngine(
                opponent_path,
                timeout=240.0,
                add_uci_arg=_needs_uci_arg(opponent_path),
                extra_args=_extra_args(opponent_params, opponent_nnue),
            ) as opponent,
            UciEngine(
                reference_path,
                timeout=240.0,
                add_uci_arg=_needs_uci_arg(reference_path),
                extra_args=_extra_args(reference_params, reference_nnue),
            ) as reference,
        ):
            clock: Optional[MatchClock] = None
            reference_movetime_ms: Optional[int] = None
            opponent_movetime_ms: Optional[int] = None

            if time_per_move is not None:
                reference_movetime_ms = _to_millis(time_per_move)
                if opponent_time_per_move is not None:
                    opponent_movetime_ms = _to_millis(opponent_time_per_move)
                else:
                    opponent_movetime_ms = reference_movetime_ms
                    if handicap_enabled:
                        opponent_movetime_ms = _scale_handicapped_time(
                            reference_movetime_ms, handicap_factor
                        )
            elif clock_seconds is not None:
                white_base = max(_to_millis(clock_seconds), MIN_CLOCK_MS)
                if opponent_clock_seconds is not None:
                    black_base = max(_to_millis(opponent_clock_seconds), MIN_CLOCK_MS)
                else:
                    black_base = white_base
                    if handicap_enabled:
                        black_base = _scale_handicapped_time(
                            white_base, handicap_factor
                        )

                white_inc = (
                    _to_millis(increment_seconds)
                    if increment_seconds is not None
                    else 0
                )
                if opponent_increment_seconds is not None:
                    black_inc = _to_millis(opponent_increment_seconds)
                else:
                    black_inc = white_inc
                    if handicap_enabled and white_inc:
                        black_inc = _scale_handicapped_time(white_inc, handicap_factor)

                clock = MatchClock(
                    white_ms=white_base,
                    black_ms=black_base,
                    white_increment_ms=white_inc,
                    black_increment_ms=black_inc,
                    white_moves_to_go=moves_to_go,
                    black_moves_to_go=moves_to_go,
                )

            depth_penalty = handicap_depth if handicap_enabled else 0

            for turn in range(limit):
                mover_color = board.turn
                engine = reference if mover_color == chess.WHITE else opponent
                engine_name = white_name if mover_color == chess.WHITE else black_name
                try:
                    if clock is not None:
                        go_clock = clock.snapshot(mover_color)
                        search_start = time.perf_counter()
                        best_move = engine.bestmove(
                            root_fen, moves=moves, clock=go_clock
                        )
                        search_end = time.perf_counter()
                    elif reference_movetime_ms is not None:
                        movetime_ms = (
                            reference_movetime_ms
                            if mover_color == chess.WHITE
                            else opponent_movetime_ms
                        )
                        if movetime_ms is None:
                            raise RuntimeError(
                                "time control not configured for opponent engine"
                            )
                        search_start = None
                        search_end = None
                        best_move = engine.bestmove(
                            root_fen, moves=moves, movetime_ms=movetime_ms
                        )
                    else:
                        assert depth is not None
                        if mover_color == chess.WHITE:
                            chosen_depth = depth
                        else:
                            if opponent_depth_factor is not None:
                                chosen_depth = max(
                                    1, int(round(depth * opponent_depth_factor))
                                )
                            elif depth_penalty != 0:
                                chosen_depth = max(depth - depth_penalty, 1)
                            else:
                                chosen_depth = depth
                        search_start = None
                        search_end = None
                        best_move = engine.bestmove(
                            root_fen, moves=moves, depth=chosen_depth
                        )
                except Exception as e:
                    print(f"Engine error for {engine_name}: {e}", file=sys.stderr)
                    print("Final position FEN:", board.fen())
                    _print_game_pgn(game)
                    return 3
                if best_move == "0000":
                    if board.is_checkmate():
                        print(f"Turn {turn + 1}: {engine_name} reports checkmate.")
                    elif board.is_stalemate():
                        print(f"Turn {turn + 1}: {engine_name} reports stalemate.")
                    else:
                        print(f"Turn {turn + 1}: {engine_name} reports no legal moves.")
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
                print(f"Turn {turn + 1}: {engine_name} plays {best_move}")
                if (
                    clock is not None
                    and search_start is not None
                    and search_end is not None
                ):
                    elapsed_ms = int(round((search_end - search_start) * 1000.0))
                    clock.apply_elapsed(mover_color, elapsed_ms)
            else:
                print(
                    f"Warning: reached move limit ({limit}); terminating early.",
                    file=sys.stderr,
                )
            move_limit_hit = True
    except FileNotFoundError as exc:
        print(f"Engine binary not found: {exc}", file=sys.stderr)
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
    if move_limit_hit and result_str == "*":
        result_str = "1/2-1/2"
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
        help="Reference engine binary to execute (default: skaks)",
    )
    parser.add_argument(
        "--opponent",
        default="stockfish",
        help="Opponent engine binary to execute (default: stockfish)",
    )
    parser.add_argument(
        "--engine-params",
        type=str,
        help="Path to YAML/JSON params for the reference engine (passed as --params)",
    )
    parser.add_argument(
        "--engine-nnue",
        type=str,
        help="Path to NNUE weights for the reference engine (passed as --nnue)",
    )
    parser.add_argument(
        "--opponent-params",
        type=str,
        help="Path to params file for the opponent engine (passed as --params)",
    )
    parser.add_argument(
        "--opponent-nnue",
        type=str,
        help="Path to NNUE weights for the opponent engine (passed as --nnue)",
    )
    parser.add_argument(
        "--stockfish",
        action="store_true",
        help="Shortcut to set the opponent to Stockfish",
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
        "--opponent-time-per-move",
        "--stockfish-time-per-move",
        dest="opponent_time_per_move",
        type=_positive_float,
        help="Seconds per move for the opponent when using time controls",
    )
    parser.add_argument(
        "--opponent-depth-factor",
        type=_positive_float,
        help="Scale factor applied to depth for the opponent (e.g. 0.6 gives opponent depth=round(depth*0.6))",
    )
    parser.add_argument(
        "--opponent-clock",
        "--stockfish-clock",
        dest="opponent_clock",
        type=_positive_float,
        help="Clock time in seconds for the opponent (default: handicapped from --clock)",
    )
    parser.add_argument(
        "--increment",
        type=_positive_float,
        help="Increment in seconds added to Skaks after each move",
    )
    parser.add_argument(
        "--opponent-increment",
        "--stockfish-increment",
        dest="opponent_increment",
        type=_positive_float,
        help="Increment in seconds added to the opponent after each move",
    )
    parser.add_argument(
        "--moves-to-go",
        type=int,
        help="Approximate moves remaining to next time control",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Limit number of plies in a match (default: 500)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for engine replies (default: 30)",
    )
    parser.add_argument(
        "--no-handicap",
        action="store_true",
        help="Disable time/depth handicap for the opponent",
    )
    parser.add_argument(
        "--handicap-factor",
        type=_positive_float,
        default=DEFAULT_HANDICAP_FACTOR,
        help="Scaling factor applied to opponent time when handicap is enabled",
    )
    parser.add_argument(
        "--handicap-depth",
        type=_non_negative_int,
        default=DEFAULT_HANDICAP_DEPTH,
        help="Depth advantage in plies retained by the reference engine when handicap is enabled",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
        help="Report progress every N puzzles (default: 10)",
    )
    args = parser.parse_args(argv)

    if args.stockfish and args.opponent != "stockfish":
        parser.error("--stockfish cannot be combined with --opponent")
    if args.stockfish:
        args.opponent = "stockfish"

    mode_count = sum(
        flag is not None for flag in (args.depth, args.time_per_move, args.clock)
    )
    if mode_count == 0:
        args.depth = 9
    elif mode_count > 1:
        parser.error("choose exactly one of --depth, --time-per-move, or --clock")

    # if args.time_per_move is None and args.opponent_time_per_move is not None:
    #     parser.error("--opponent-time-per-move requires --time-per-move")
    if args.clock is None and args.opponent_clock is not None:
        parser.error("--opponent-clock requires --clock")
    if args.clock is None and (
        args.increment is not None or args.opponent_increment is not None
    ):
        parser.error("increments require --clock")
    if args.moves_to_go is not None and args.moves_to_go <= 0:
        parser.error("--moves-to-go must be positive")

    if args.no_handicap:
        args.handicap_factor = 1.0
        args.handicap_depth = 0

    args.handicap_enabled = not args.no_handicap
    return args


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    return run_game(
        depth=args.depth,
        limit=args.limit,
        time_per_move=args.time_per_move,
        opponent_time_per_move=args.opponent_time_per_move,
        clock_seconds=args.clock,
        opponent_clock_seconds=args.opponent_clock,
        increment_seconds=args.increment,
        opponent_increment_seconds=args.opponent_increment,
        moves_to_go=args.moves_to_go,
        reference_engine=args.engine,
        opponent_engine=args.opponent,
        reference_params=args.engine_params,
        opponent_params=args.opponent_params,
        reference_nnue=args.engine_nnue,
        opponent_nnue=args.opponent_nnue,
        handicap_factor=args.handicap_factor,
        handicap_depth=args.handicap_depth,
        opponent_depth_factor=args.opponent_depth_factor,
        handicap_enabled=args.handicap_enabled,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
