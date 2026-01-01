"""PGN sampling and annotation utilities."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import chess
import chess.engine
import chess.pgn

try:
    import skaks_eval  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    skaks_eval = None


@dataclass
class Sample:
    game_index: int
    ply: int
    fen: str
    white_to_move: bool
    result: str
    outcome: Optional[float]
    winner: Optional[str]


CSV_FIELDS = (
    "source",
    "game_index",
    "ply",
    "fen",
    "side_to_move",
    "stockfish_cp",
    "result",
    "outcome",
    "winner",
    "weight",
)


def add_subparser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "dataset-sample",
        help="Sample PGNs, annotate with Stockfish, and write Texel-ready CSV shards",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="PGN files, directories, or glob patterns",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for CSV shards",
    )
    parser.add_argument(
        "--stockfish",
        required=True,
        help="Path to Stockfish binary",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=15,
        help="Search depth for Stockfish",
    )
    parser.add_argument(
        "--movetime",
        type=float,
        help="Search time per position (seconds, optional)",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        help="Node budget per position (optional)",
    )
    parser.add_argument(
        "--stockfish-threads",
        type=int,
        default=1,
        help="Threads option passed to Stockfish",
    )
    parser.add_argument(
        "--stockfish-option",
        action="append",
        default=[],
        help="Extra UCI option in key=value form (repeatable)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Sample every Nth ply",
    )
    parser.add_argument(
        "--min-ply",
        type=int,
        default=8,
        help="Minimum ply to sample",
    )
    parser.add_argument(
        "--max-ply",
        type=int,
        help="Maximum ply to sample",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Positions per evaluation chunk",
    )
    parser.add_argument(
        "--rows-per-shard",
        type=int,
        help="Split output every N rows (per input)",
    )
    parser.add_argument(
        "--quiet-only",
        action="store_true",
        help="Filter to quiet positions (requires skaks_eval)",
    )
    parser.add_argument(
        "--quiet-batch",
        type=int,
        default=2048,
        help="Batch size for quiet filtering",
    )
    parser.add_argument(
        "--max-per-file",
        type=int,
        help="Cap sampled positions per PGN",
    )
    parser.add_argument(
        "--max-total",
        type=int,
        help="Cap sampled positions overall",
    )
    parser.add_argument(
        "--pov",
        choices=["side", "white"],
        default="side",
        help="Interpret centipawns relative to side-to-move or white",
    )
    parser.add_argument(
        "--cp-cap",
        type=float,
        help="Clamp Stockfish centipawns to +/- cap",
    )
    parser.add_argument(
        "--weight",
        type=float,
        default=1.0,
        help="Default sample weight",
    )
    parser.add_argument(
        "--keep-missing-result",
        action="store_true",
        help="Retain positions without a known game result",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip writing when shard already exists",
    )
    return parser


def run_dataset(args: argparse.Namespace) -> None:
    inputs = _expand_inputs(args.inputs)
    if not inputs:
        raise SystemExit("No PGNs matched --inputs")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    limit = _build_limit(args)
    option_map = _parse_options(args.stockfish_option)
    print(f"[dataset] sampling {len(inputs)} PGNs -> {output_dir}")
    total_written = 0
    with chess.engine.SimpleEngine.popen_uci(str(Path(args.stockfish))) as engine:
        _configure_engine(engine, args.stockfish_threads, option_map)
        for pgn_path in inputs:
            base_name = pgn_path.stem
            sink = CsvShardWriter(
                output_dir=output_dir,
                base_name=base_name,
                rows_per_shard=args.rows_per_shard,
                skip_existing=args.skip_existing,
            )
            written = 0
            for chunk in _iter_chunks(
                pgn_path,
                stride=args.stride,
                min_ply=args.min_ply,
                max_ply=args.max_ply,
                chunk_size=args.chunk_size,
                max_positions=args.max_per_file,
                keep_missing=args.keep_missing_result,
            ):
                if args.quiet_only:
                    chunk = _filter_quiet(chunk, args.quiet_batch)
                    if not chunk:
                        continue
                rows = _evaluate_chunk(
                    engine=engine,
                    samples=chunk,
                    limit=limit,
                    pov=args.pov,
                    cp_cap=args.cp_cap,
                    source=pgn_path.name,
                    weight=args.weight,
                )
                if not rows:
                    continue
                sink.write_rows(rows)
                written += len(rows)
                total_written += len(rows)
                if args.max_total and total_written >= args.max_total:
                    break
                if args.max_per_file and written >= args.max_per_file:
                    break
            sink.close()
            print(f"[dataset] {pgn_path.name}: wrote {written} rows")
            if args.max_total and total_written >= args.max_total:
                break
    print(f"[dataset] done, total rows={total_written}")


def _expand_inputs(patterns: Sequence[str]) -> List[Path]:
    seen: set[Path] = set()
    results: List[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        candidates: List[Path] = []
        if any(token in pattern for token in "*?[]"):
            candidates.extend(Path.cwd().glob(pattern))
        elif path.is_dir():
            candidates.extend(sorted(path.glob("*.pgn")))
        elif path.exists():
            candidates.append(path)
        else:
            candidates.extend(Path.cwd().glob(pattern))
        for cand in candidates:
            resolved = cand.resolve()
            if resolved.suffix.lower() != ".pgn":
                continue
            if resolved not in seen:
                seen.add(resolved)
                results.append(resolved)
    results.sort()
    return results


def _build_limit(args: argparse.Namespace) -> chess.engine.Limit:
    kwargs = {}
    if args.depth:
        kwargs["depth"] = int(args.depth)
    if args.movetime:
        kwargs["time"] = float(args.movetime)
    if args.nodes:
        kwargs["nodes"] = int(args.nodes)
    if not kwargs:
        raise SystemExit("Provide at least one of --depth/--movetime/--nodes")
    return chess.engine.Limit(**kwargs)


def _parse_options(raw: Sequence[str]) -> dict[str, object]:
    options: dict[str, object] = {}
    for item in raw:
        if "=" not in item:
            key = item.strip()
            if key:
                options[key] = True
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        parsed: object
        if value.lower() in {"true", "false"}:
            parsed = value.lower() == "true"
        else:
            try:
                if "." in value:
                    parsed = float(value)
                else:
                    parsed = int(value)
            except ValueError:
                parsed = value
        options[key] = parsed
    return options


def _configure_engine(
    engine: chess.engine.SimpleEngine,
    threads: int,
    options: dict[str, object],
) -> None:
    config = dict(options)
    if threads > 0:
        config.setdefault("Threads", threads)
    if not config:
        return
    try:
        engine.configure(config)
    except chess.engine.EngineError as exc:  # pragma: no cover - depends on engine
        print(f"[dataset] warning: failed to set options: {exc}", file=sys.stderr)


def _iter_chunks(
    pgn_path: Path,
    *,
    stride: int,
    min_ply: Optional[int],
    max_ply: Optional[int],
    chunk_size: int,
    max_positions: Optional[int],
    keep_missing: bool,
) -> Iterator[List[Sample]]:
    game_idx = 0
    sampled = 0
    chunk: List[Sample] = []
    with pgn_path.open("r", encoding="utf-8", errors="ignore") as fh:
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            result_raw = game.headers.get("Result", "")
            result_norm, outcome, winner = _parse_result(result_raw)
            if outcome is None and not keep_missing:
                game_idx += 1
                continue
            board = game.board()
            for ply, move in enumerate(game.mainline_moves(), start=1):
                board.push(move)
                if stride > 1 and (ply % stride) != 0:
                    continue
                if min_ply is not None and ply < min_ply:
                    continue
                if max_ply is not None and ply > max_ply:
                    continue
                chunk.append(
                    Sample(
                        game_index=count,
                        ply=ply,
                        fen=board.fen(),
                        white_to_move=board.turn == chess.WHITE,
                        result=result_norm,
                        outcome=outcome,
                        winner=winner,
                    )
                )
                sampled += 1
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
                if max_positions is not None and sampled >= max_positions:
                    break
            game_idx += 1
            if max_positions is not None and sampled >= max_positions:
                break
    if chunk:
        yield chunk


_RESULT_MAP = {
    "1-0": (1.0, "w"),
    "0-1": (0.0, "b"),
    "1/2-1/2": (0.5, "d"),
}


def _parse_result(raw: str) -> tuple[str, Optional[float], Optional[str]]:
    norm = raw.strip() if raw else "*"
    if norm in _RESULT_MAP:
        val, winner = _RESULT_MAP[norm]
        return norm, val, winner
    return norm, None, None


def _filter_quiet(samples: List[Sample], batch_size: int) -> List[Sample]:
    if skaks_eval is None:
        raise SystemExit("skaks_eval not available but --quiet-only was set")
    keep: List[Sample] = []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        flags = skaks_eval.is_quiet_batch([item.fen for item in chunk])
        for item, flag in zip(chunk, flags):
            if flag:
                keep.append(item)
    return keep


def _evaluate_chunk(
    *,
    engine: chess.engine.SimpleEngine,
    samples: Sequence[Sample],
    limit: chess.engine.Limit,
    pov: str,
    cp_cap: Optional[float],
    source: str,
    weight: float,
) -> List[dict[str, object]]:
    rows: List[dict[str, object]] = []
    for sample in samples:
        try:
            board = chess.Board(sample.fen)
        except ValueError:
            continue
        try:
            info = engine.analyse(board, limit=limit)
        except chess.engine.EngineTerminatedError:
            raise
        except chess.engine.EngineError:
            continue
        score = info.get("score")
        if score is None:
            continue
        cp = score.white().score(mate_score=100000)
        if cp is None:
            continue
        if pov == "side" and not sample.white_to_move:
            cp = -cp
        cp_val = float(cp)
        if cp_cap is not None:
            cp_val = max(-cp_cap, min(cp_cap, cp_val))
        row = {
            "source": source,
            "game_index": sample.game_index,
            "ply": sample.ply,
            "fen": sample.fen,
            "side_to_move": "w" if sample.white_to_move else "b",
            "stockfish_cp": cp_val,
            "result": sample.result,
            "outcome": sample.outcome if sample.outcome is not None else "",
            "winner": sample.winner or "",
            "weight": weight,
        }
        rows.append(row)
    return rows


class CsvShardWriter:
    def __init__(
        self,
        *,
        output_dir: Path,
        base_name: str,
        rows_per_shard: Optional[int],
        skip_existing: bool,
    ) -> None:
        self._output_dir = output_dir
        self._base_name = base_name
        self._rows_per_shard = rows_per_shard
        self._skip_existing = skip_existing
        self._shard_index = 0
        self._rows_in_shard = 0
        self._file = None
        self._writer: Optional[csv.DictWriter] = None

    def write_rows(self, rows: Sequence[dict[str, object]]) -> None:
        for row in rows:
            if self._writer is None or (
                self._rows_per_shard is not None
                and self._rows_in_shard >= self._rows_per_shard
            ):
                self._open_next_writer()
            assert self._writer is not None
            self._writer.writerow(row)
            self._rows_in_shard += 1

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None

    def _open_next_writer(self) -> None:
        self.close()
        shard_suffix = ""
        if self._rows_per_shard is not None:
            shard_suffix = f"_{self._shard_index:04d}"
        path = self._output_dir / f"{self._base_name}{shard_suffix}.csv"
        if self._skip_existing and path.exists():
            self._shard_index += 1
            self._rows_in_shard = 0
            self._open_next_writer()
            return
        self._file = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        self._rows_in_shard = 0
        self._shard_index += 1


__all__ = ["add_subparser", "run_dataset"]
