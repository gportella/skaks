import argparse
import csv
import concurrent.futures
import os
import re
import select
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import chess
import chess.pgn


def parse_pgn_games(pgn_path: Path) -> Iterator[chess.pgn.Game]:
    with pgn_path.open("r", encoding="utf-8", errors="ignore") as fh:
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            yield game


def count_sampled_positions(
    pgn_path: Path,
    sample_stride: int,
    max_positions: Optional[int],
    min_ply: Optional[int],
    max_ply: Optional[int],
) -> int:
    """Count how many positions will be sampled given stride and cap."""
    total = 0
    for game in parse_pgn_games(pgn_path):
        board = game.board()
        for ply, move in enumerate(game.mainline_moves()):
            board.push(move)
            if ply % sample_stride != 0:
                continue
            if min_ply is not None and ply + 1 < min_ply:
                continue
            if max_ply is not None and ply + 1 > max_ply:
                continue
            total += 1
            if max_positions is not None and total >= max_positions:
                return total
    return total


def _parse_result(raw: str) -> Tuple[str, Optional[float], Optional[str]]:
    """Normalize a PGN result string and derive outcome/winner.

    Returns (normalized_result, outcome, winner), where outcome is in [0,1]
    with 1.0 = white win, 0.0 = black win, 0.5 = draw. winner is one of
    "w", "b", "d" or None if unknown.
    """

    normalized = raw.strip() if raw else ""
    if normalized in {"1-0", "1/2-1/2", "0-1"}:
        if normalized == "1-0":
            return normalized, 1.0, "w"
        if normalized == "0-1":
            return normalized, 0.0, "b"
        return normalized, 0.5, "d"
    return normalized or "*", None, None


def collect_positions(
    pgn_path: Path,
    sample_stride: int,
    max_positions: Optional[int],
    min_ply: Optional[int],
    max_ply: Optional[int],
) -> List[Tuple[int, int, str, bool, str, Optional[float], Optional[str]]]:
    """Collect sampled positions as (game_idx, ply, fen, white_to_move, result, outcome, winner)."""

    positions: List[
        Tuple[int, int, str, bool, str, Optional[float], Optional[str]]
    ] = []
    for game_idx, game in enumerate(parse_pgn_games(pgn_path)):
        result_raw = game.headers.get("Result", "")
        result_norm, outcome, winner = _parse_result(result_raw)
        board = game.board()
        for ply, move in enumerate(game.mainline_moves()):
            board.push(move)
            if ply % sample_stride != 0:
                continue
            if min_ply is not None and ply + 1 < min_ply:
                continue
            if max_ply is not None and ply + 1 > max_ply:
                continue
            positions.append(
                (
                    game_idx,
                    ply + 1,
                    board.fen(),
                    board.turn == chess.WHITE,
                    result_norm,
                    outcome,
                    winner,
                )
            )
            if max_positions is not None and len(positions) >= max_positions:
                return positions
    return positions


def process_chunk(
    chunk: Sequence[Tuple[int, int, str, bool, str, Optional[float], Optional[str]]],
    stockfish_path: Path,
    skaks_path: Path,
    stockfish_depth: int,
    skaks_params: Optional[Path],
    pov: str,
) -> Tuple[List[Dict[str, object]], List[str]]:
    rows: List[Dict[str, object]] = []
    failures: List[str] = []
    with (
        UciSearchEngine(stockfish_path, depth=stockfish_depth) as sf,
        UciStaticEngine(
            skaks_path,
            eval_command="staticeval",
            params_path=skaks_params,
            add_uci_arg=True,
        ) as sk,
    ):
        for game_idx, ply, fen, white_to_move, result, outcome, winner in chunk:
            sf_white = sf.search_eval_white(fen)
            sk_white = sk.static_eval_white(fen)
            if sf_white is None or sk_white is None:
                if len(failures) < 20:
                    failures.append(fen)
                continue

            if pov == "white":
                sf_cp = sf_white
                sk_cp = sk_white
            else:
                sf_cp = sf_white if white_to_move else -sf_white
                sk_cp = sk_white if white_to_move else -sk_white

            rows.append(
                {
                    "game_index": game_idx,
                    "ply": ply,
                    "side_to_move": "w" if white_to_move else "b",
                    "fen": fen,
                    "stockfish_cp": sf_cp,
                    "skaks_cp": sk_cp,
                    "result": result,
                    "outcome": outcome,
                    "winner": winner,
                }
            )
    return rows, failures


class UciStaticEngine:
    def __init__(
        self,
        binary: Path,
        eval_command: str,
        params_path: Optional[Path] = None,
        add_uci_arg: bool = False,
    ) -> None:
        argv: List[str] = [str(binary)]
        if add_uci_arg:
            argv.append("--uci")
        if params_path is not None:
            argv.extend(["--params", str(params_path)])
        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
        )
        self._eval_command = eval_command
        self._handshake()

    def _send(self, payload: str) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write((payload + "\n").encode("utf-8"))
        self._proc.stdin.flush()

    def _read_line(self, timeout: float = 1.0) -> Optional[str]:
        assert self._proc.stdout is not None
        fd = self._proc.stdout
        ready, _, _ = select.select([fd], [], [], timeout)
        if not ready:
            return ""
        line = fd.readline()
        if line == b"":
            return None
        text = line.decode("utf-8", errors="replace").rstrip("\n")
        return text

    def _drain_until(self, token: str, timeout: float = 60.0) -> None:
        deadline = time.monotonic() + timeout
        seen: List[str] = []
        while True:
            if time.monotonic() > deadline:
                code = self._proc.poll()
                stderr_tail = ""
                if self._proc.stderr is not None:
                    try:
                        stderr_tail = self._proc.stderr.read().strip()
                    except Exception:
                        stderr_tail = ""
                detail = f"timeout waiting for '{token}' during handshake"
                if code is not None:
                    detail += f" (exit {code})"
                if stderr_tail:
                    detail += f" | stderr: {stderr_tail}"
                if seen:
                    detail += f" | seen: {' | '.join(seen[-8:])}"
                raise RuntimeError(detail)
            line = self._read_line(timeout=0.2)
            if line is None:
                code = self._proc.poll()
                stderr_tail = ""
                if self._proc.stderr is not None:
                    try:
                        stderr_tail = self._proc.stderr.read().strip()
                    except Exception:
                        stderr_tail = ""
                detail = (
                    f"engine terminated unexpectedly during handshake (exit {code})"
                )
                if stderr_tail:
                    detail += f": {stderr_tail}"
                raise RuntimeError(detail)
            if line == "":
                continue
            seen.append(line)
            if line.strip() == token:
                return

    def _handshake(self) -> None:
        self._send("uci")
        self._drain_until("uciok")
        self._send("isready")
        self._drain_until("readyok")

    def _parse_eval(self, lines: List[str]) -> Optional[int]:
        for line in lines:
            if "static_eval_white" in line:
                parts = line.split()
                for token in reversed(parts):
                    try:
                        return int(token)
                    except ValueError:
                        continue
        for line in lines:
            if "Total evaluation" in line or "Final evaluation" in line:
                match = re.search(r"([-+]?\d+(?:\.\d+)?)", line)
                if match:
                    try:
                        val = float(match.group(1))
                        return int(round(val * 100.0))
                    except ValueError:
                        continue
        for line in lines:
            if "score cp" in line:
                match = re.search(r"score cp\s+([-+]?\d+)", line)
                if match:
                    return int(match.group(1))
        return None

    def static_eval_white(self, fen: str) -> Optional[int]:
        self._send(f"position fen {fen}")
        self._send(self._eval_command)
        self._send("isready")
        lines: List[str] = []
        deadline = time.monotonic() + 10.0
        while True:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"static eval timed out; last lines: {' | '.join(lines[-5:])}"
                )
            line = self._read_line(timeout=0.2)
            if line is None:
                break
            if line == "":
                continue
            stripped = line.strip()
            if stripped == "readyok":
                break
            lines.append(stripped)
        return self._parse_eval(lines)

    def close(self) -> None:
        try:
            if self._proc.poll() is None:
                self._send("quit")
                self._proc.wait(timeout=2)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class UciSearchEngine:
    def __init__(
        self,
        binary: Path,
        depth: int,
        params_path: Optional[Path] = None,
        add_uci_arg: bool = False,
    ) -> None:
        argv: List[str] = [str(binary)]
        if add_uci_arg:
            argv.append("--uci")
        if params_path is not None:
            argv.extend(["--params", str(params_path)])
        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
        )
        self._depth = depth
        self._handshake()

    def _send(self, payload: str) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write((payload + "\n").encode("utf-8"))
        self._proc.stdin.flush()

    def _read_line(self, timeout: float = 1.0) -> Optional[str]:
        assert self._proc.stdout is not None
        fd = self._proc.stdout
        ready, _, _ = select.select([fd], [], [], timeout)
        if not ready:
            return ""
        line = fd.readline()
        if line == b"":
            return None
        text = line.decode("utf-8", errors="replace").rstrip("\n")
        return text

    def _drain_until(self, token: str, timeout: float = 60.0) -> None:
        deadline = time.monotonic() + timeout
        seen: List[str] = []
        while True:
            if time.monotonic() > deadline:
                code = self._proc.poll()
                detail = f"timeout waiting for '{token}' during handshake"
                if code is not None:
                    detail += f" (exit {code})"
                if seen:
                    detail += f" | seen: {' | '.join(seen[-8:])}"
                raise RuntimeError(detail)
            line = self._read_line(timeout=0.2)
            if line is None:
                code = self._proc.poll()
                detail = (
                    f"engine terminated unexpectedly during handshake (exit {code})"
                )
                if seen:
                    detail += f" | seen: {' | '.join(seen[-8:])}"
                raise RuntimeError(detail)
            if line == "":
                continue
            seen.append(line)
            if line.strip() == token:
                return

    def _handshake(self) -> None:
        self._send("uci")
        self._drain_until("uciok")
        self._send("isready")
        self._drain_until("readyok")

    @staticmethod
    def _parse_score(text: str) -> Optional[int]:
        if "score" not in text:
            return None
        # Typical: "info depth 15 ... score cp 34" or "score mate 3"
        tokens = text.split()
        for idx, tok in enumerate(tokens):
            if tok == "score" and idx + 2 < len(tokens):
                kind = tokens[idx + 1]
                val = tokens[idx + 2]
                if kind == "cp":
                    try:
                        return int(val)
                    except ValueError:
                        return None
                if kind == "mate":
                    try:
                        mate_ply = int(val)
                        return 32000 if mate_ply > 0 else -32000
                    except ValueError:
                        return None
        return None

    def search_eval_white(self, fen: str) -> Optional[int]:
        self._send(f"position fen {fen}")
        self._send(f"go depth {self._depth}")
        last_score: Optional[int] = None
        deadline = time.monotonic() + 60.0
        while True:
            if time.monotonic() > deadline:
                raise RuntimeError(f"search eval timed out; last score={last_score}")
            line = self._read_line(timeout=0.5)
            if line is None:
                break
            if line == "":
                continue
            stripped = line.strip()
            if stripped.startswith("info") and "score" in stripped:
                parsed = self._parse_score(stripped)
                if parsed is not None:
                    last_score = parsed
            if stripped.startswith("bestmove"):
                break
        return last_score

    def close(self) -> None:
        try:
            if self._proc.poll() is None:
                self._send("quit")
                self._proc.wait(timeout=2)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def process_games(
    pgn_path: Path,
    stockfish_path: Path,
    skaks_path: Path,
    sample_stride: int,
    max_positions: Optional[int],
    min_ply: Optional[int],
    max_ply: Optional[int],
    output_path: Path,
    total_expected: Optional[int],
    pov: str,
    skaks_params: Optional[Path],
    stockfish_depth: int,
    workers: int,
    chunk_size: int,
) -> None:
    out_fields = [
        "game_index",
        "ply",
        "side_to_move",
        "fen",
        "stockfish_cp",
        "skaks_cp",
        "result",
        "outcome",
        "winner",
    ]

    positions = collect_positions(
        pgn_path, sample_stride, max_positions, min_ply, max_ply
    )

    total = 0
    failures: List[str] = []
    header_needed = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=out_fields)
        if header_needed:
            writer.writeheader()

        if workers <= 1:
            # Stream sequentially in chunks so we see progress and avoid holding all rows.
            for idx in range(0, len(positions), chunk_size):
                chunk = positions[idx : idx + chunk_size]
                rows, chunk_failures = process_chunk(
                    chunk,
                    stockfish_path,
                    skaks_path,
                    stockfish_depth,
                    skaks_params,
                    pov,
                )
                writer.writerows(rows)
                total += len(rows)
                failures.extend(chunk_failures)
                if total % 100 == 0 or total == 1:
                    expected = total_expected if total_expected is not None else "?"
                    print(f"[progress] {total}/{expected} positions", end="\r", flush=True)
            print()
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
                futures = []
                for idx in range(0, len(positions), chunk_size):
                    chunk = positions[idx : idx + chunk_size]
                    futures.append(
                        ex.submit(
                            process_chunk,
                            chunk,
                            stockfish_path,
                            skaks_path,
                            stockfish_depth,
                            skaks_params,
                            pov,
                        )
                    )
                for fut in concurrent.futures.as_completed(futures):
                    rows, chunk_failures = fut.result()
                    writer.writerows(rows)
                    total += len(rows)
                    failures.extend(chunk_failures)
                    if total % 10 == 0 or total == 1:
                        expected = total_expected if total_expected is not None else "?"
                        print(
                            f"[progress] {total}/{expected} positions",
                            end="\r",
                            flush=True,
                        )

    # Ensure progress prints end with a newline
    if total > 0:
        expected = total_expected if total_expected is not None else "?"
        print(f"[progress] {total}/{expected} positions", flush=True)

    if failures:
        print(f"[warn] skipped {len(failures)} positions with engine errors")
        print("[warn] sample failing FENs:")
        for fen in failures[:20]:
            print(f"  {fen}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract eval pairs from PGN using Stockfish and skaks"
    )
    parser.add_argument(
        "pgn",
        type=Path,
        help="Path to PGN file (e.g., moves_pgn/LumbrasGigaBase_OTB_2025.pgn)",
    )
    parser.add_argument(
        "stockfish",
        type=str,
        nargs="?",
        default="stockfish",
        help="Stockfish binary (path or name in PATH)",
    )
    parser.add_argument(
        "skaks",
        type=str,
        nargs="?",
        default="skaks",
        help="skaks binary (path or name in PATH)",
    )
    parser.add_argument(
        "--sample-stride", type=int, default=4, help="Sample every N plies"
    )
    parser.add_argument(
        "--max-positions", type=int, default=None, help="Cap total positions (optional)"
    )
    parser.add_argument(
        "--min-ply", type=int, default=None, help="Minimum ply to sample (1-based)"
    )
    parser.add_argument(
        "--max-ply", type=int, default=None, help="Maximum ply to sample (1-based)"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("eval_pairs.csv"), help="Output CSV path"
    )
    parser.add_argument(
        "--pov",
        choices=["white", "side"],
        default="side",
        help="Score perspective: side-to-move (default) or white",
    )
    parser.add_argument(
        "--skaks-params",
        type=Path,
        default=None,
        help="Path to skaks params file (passed as --params)",
    )
    parser.add_argument(
        "--stockfish-depth",
        type=int,
        default=15,
        help="Depth for Stockfish search eval (PV score)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes to parallelize evaluation",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=32,
        help="Positions per worker chunk when using parallel mode",
    )
    args = parser.parse_args()

    if not args.pgn.exists():
        raise SystemExit(f"PGN not found: {args.pgn}")

    sf_path = Path(shutil.which(args.stockfish) or args.stockfish)
    sk_path = Path(shutil.which(args.skaks) or args.skaks)

    if not sf_path.exists():
        raise SystemExit(f"Stockfish not found: {args.stockfish}")
    if not sk_path.exists():
        raise SystemExit(f"skaks not found: {args.skaks}")

    total_expected = count_sampled_positions(
        pgn_path=args.pgn,
        sample_stride=args.sample_stride,
        max_positions=args.max_positions,
        min_ply=args.min_ply,
        max_ply=args.max_ply,
    )
    print(f"Planning to sample up to {total_expected} positions")

    os.makedirs(args.output.parent, exist_ok=True)
    process_games(
        pgn_path=args.pgn,
        stockfish_path=sf_path,
        skaks_path=sk_path,
        sample_stride=args.sample_stride,
        max_positions=args.max_positions,
        min_ply=args.min_ply,
        max_ply=args.max_ply,
        output_path=args.output,
        total_expected=total_expected,
        pov=args.pov,
        skaks_params=args.skaks_params,
        stockfish_depth=args.stockfish_depth,
        workers=max(1, args.workers),
        chunk_size=max(1, args.chunk_size),
    )


if __name__ == "__main__":
    main()
