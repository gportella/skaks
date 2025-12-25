import argparse
import csv
import os
import re
import select
import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterator, List, Optional

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
    pgn_path: Path, sample_stride: int, max_positions: Optional[int]
) -> int:
    """Count how many positions will be sampled given stride and cap."""
    total = 0
    for game in parse_pgn_games(pgn_path):
        board = game.board()
        for ply, move in enumerate(game.mainline_moves()):
            board.push(move)
            if ply % sample_stride != 0:
                continue
            total += 1
            if max_positions is not None and total >= max_positions:
                return total
    return total


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


def process_games(
    pgn_path: Path,
    stockfish_path: Path,
    skaks_path: Path,
    sample_stride: int,
    max_positions: Optional[int],
    output_path: Path,
    total_expected: Optional[int],
    pov: str,
    skaks_params: Optional[Path],
) -> None:
    out_fields = [
        "game_index",
        "ply",
        "side_to_move",
        "fen",
        "stockfish_cp",
        "skaks_cp",
    ]

    total = 0
    with (
        UciStaticEngine(stockfish_path, eval_command="eval") as sf,
        UciStaticEngine(
            skaks_path,
            eval_command="staticeval",
            params_path=skaks_params,
            add_uci_arg=True,
        ) as sk,
        output_path.open("a", newline="", encoding="utf-8") as out_f,
    ):
        writer = csv.DictWriter(out_f, fieldnames=out_fields)
        writer.writeheader()

        failures = 0
        failure_samples = []

        for game_idx, game in enumerate(parse_pgn_games(pgn_path)):
            board = game.board()
            for ply, move in enumerate(game.mainline_moves()):
                board.push(move)
                if ply % sample_stride != 0:
                    continue
                if max_positions is not None and total >= max_positions:
                    return

                fen = board.fen()
                sf_white = sf.static_eval_white(fen)
                sk_white = sk.static_eval_white(fen)

                if sf_white is None or sk_white is None:
                    failures += 1
                    if len(failure_samples) < 20:
                        failure_samples.append(fen)
                    continue

                if pov == "white":
                    sf_cp = sf_white
                    sk_cp = sk_white
                else:
                    sf_cp = sf_white if board.turn == chess.WHITE else -sf_white
                    sk_cp = sk_white if board.turn == chess.WHITE else -sk_white
                if sf_cp is None or sk_cp is None:
                    failures += 1
                    if len(failure_samples) < 20:
                        failure_samples.append(board.fen())
                    continue

                writer.writerow(
                    {
                        "game_index": game_idx,
                        "ply": ply + 1,
                        "side_to_move": "w" if board.turn == chess.WHITE else "b",
                        "fen": board.fen(),
                        "stockfish_cp": sf_cp,
                        "skaks_cp": sk_cp,
                    }
                )
                total += 1

                if total % 10 == 0 or total == 1:
                    expected = total_expected if total_expected is not None else "?"
                    print(
                        f"[progress] {total}/{expected} positions (last fen: {fen})",
                        end="\r",
                        flush=True,
                    )

                if total_expected is not None and total >= total_expected:
                    expected = total_expected if total_expected is not None else "?"
                    print(f"[progress] {total}/{expected} positions", flush=True)
                    return

    # Ensure progress prints end with a newline
    if total > 0:
        expected = total_expected if total_expected is not None else "?"
        print(f"[progress] {total}/{expected} positions", flush=True)

    if failures > 0:
        print(f"[warn] skipped {failures} positions with engine errors")
        if failure_samples:
            print("[warn] sample failing FENs:")
            for fen in failure_samples:
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
    )
    print(f"Planning to sample up to {total_expected} positions")

    os.makedirs(args.output.parent, exist_ok=True)
    process_games(
        pgn_path=args.pgn,
        stockfish_path=sf_path,
        skaks_path=sk_path,
        sample_stride=args.sample_stride,
        max_positions=args.max_positions,
        output_path=args.output,
        total_expected=total_expected,
        pov=args.pov,
        skaks_params=args.skaks_params,
    )


if __name__ == "__main__":
    main()
