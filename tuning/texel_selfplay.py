from __future__ import annotations

import argparse
import concurrent.futures
import csv
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

try:
    import skaks_eval  # type: ignore
except ImportError:  # pragma: no cover
    skaks_eval = None  # type: ignore

import chess
import chess.pgn


def load_start_fens(path: Optional[Path], limit: Optional[int], seed: int) -> List[str]:
    if path is None:
        return [chess.STARTING_FEN]
    fens: List[str] = []
    with path.open("r") as fh:
        for line in fh:
            fen = line.strip()
            if not fen:
                continue
            fens.append(fen)
            if limit is not None and len(fens) >= limit:
                break
    if not fens:
        fens = [chess.STARTING_FEN]
    random.Random(seed).shuffle(fens)
    return fens


def load_start_fens_from_pgn(
    path: Path, limit: Optional[int], seed: int, min_ply: int, max_ply: Optional[int]
) -> List[str]:
    fens: List[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            board = game.board()
            ply = 0
            for move in game.mainline_moves():
                board.push(move)
                ply += 1
                if ply < min_ply:
                    continue
                if max_ply is not None and ply > max_ply:
                    break
                fens.append(board.fen())
                if limit is not None and len(fens) >= limit:
                    break
            if limit is not None and len(fens) >= limit:
                break
    if not fens:
        return [chess.STARTING_FEN]
    rng = random.Random(seed)
    rng.shuffle(fens)
    if limit is not None:
        fens = fens[:limit]
    return fens


def run_selfplay(
    *,
    engine_bin: str,
    params_path: Optional[Path],
    start_fen: str,
    depth: Optional[int],
    movetime_ms: Optional[int],
    max_full_moves: int,
    polyglot: bool,
) -> List[str]:
    if (depth is None) == (movetime_ms is None):
        raise ValueError("provide exactly one of depth or movetime_ms")

    argv = [engine_bin, "--self", "--onlyfen", "--max-moves", str(max_full_moves)]
    if not polyglot:
        argv.append("--no-polyglot")
    if params_path is not None:
        argv.extend(["--params", str(params_path)])
    if start_fen != chess.STARTING_FEN:
        argv.extend(["--fen", start_fen])
    if movetime_ms is not None:
        argv.extend(["--move-time", str(movetime_ms)])
    else:
        argv.extend(["--depth", str(depth)])

    proc = subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "").strip()
        raise RuntimeError(
            f"skaks self-play failed (exit {proc.returncode})"
            + (f": {stderr_tail}" if stderr_tail else "")
        )

    fens: List[str] = []
    for raw in (proc.stdout or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[1] in ("w", "b"):
            fens.append(line)
    if not fens:
        raise RuntimeError("no FENs produced by skaks self-play")
    return fens


def run_selfplay_binding(
    *,
    start_fens: Sequence[str],
    params_path: Optional[Path],
    depth: Optional[int],
    movetime_ms: Optional[int],
    max_plies: int,
    sample_stride: int,
) -> Tuple[List[tuple[str, float, str]], int]:
    if skaks_eval is None:
        raise RuntimeError("skaks_eval bindings not available")

    params_dict = None
    if params_path is not None:
        import yaml

        with params_path.open("r") as fh:
            params_dict = yaml.safe_load(fh)

    depth_val = depth if depth is not None else 0
    movetime_val = movetime_ms if movetime_ms is not None else 0
    if (depth_val == 0) == (movetime_val == 0):
        raise ValueError("exactly one of depth or movetime_ms must be set")

    result = skaks_eval.selfplay(
        start_fens=list(start_fens),
        params=params_dict,
        depth=depth_val,
        movetime_ms=movetime_val,
        max_plies=max_plies,
        sample_stride=sample_stride,
    )

    fens = result["fen"]
    outcomes = result["outcome"]
    sides = result["side_to_move"]
    samples: List[tuple[str, float, str]] = []
    for fen, outcome, stm in zip(fens, outcomes, sides):
        samples.append((str(fen), float(outcome), str(stm)))
    games_played = int(result.get("games_played", 0))
    return samples, games_played


def _side_from_fen(fen: str) -> str:
    parts = fen.split()
    if len(parts) < 2:
        raise ValueError(f"expected 'w' or 'b' for turn part of fen: '{fen}'")
    side = parts[1]
    if side not in ("w", "b"):
        raise ValueError(f"expected 'w' or 'b' for turn part of fen: '{fen}'")
    return side


def reconstruct_samples(
    fen_sequence: Sequence[str], sample_stride: int, max_plies: int
) -> Tuple[List[tuple[str, str]], float, int]:
    if not fen_sequence:
        raise ValueError("no FENs to reconstruct")
    if sample_stride <= 0:
        raise ValueError("sample_stride must be positive")

    # fen_sequence[0] is the starting position; subsequent entries are after each ply
    plies_available = max(0, len(fen_sequence) - 1)
    plies_played = min(plies_available, max_plies)

    samples: List[tuple[str, str]] = []
    for idx in range(1, plies_played + 1):
        fen = fen_sequence[idx]
        if (idx - 1) % sample_stride == 0:
            samples.append((fen, _side_from_fen(fen)))

    final_fen = fen_sequence[plies_played] if plies_played > 0 else fen_sequence[0]
    board = chess.Board(final_fen)
    if board.is_game_over(claim_draw=True):
        result = board.result(claim_draw=True)
    else:
        result = "1/2-1/2"

    if result == "1-0":
        outcome = 1.0
    elif result == "0-1":
        outcome = 0.0
    else:
        outcome = 0.5

    return samples, outcome, plies_played


def play_game_task(
    fen: str,
    *,
    engine_bin: str,
    params_path: Optional[Path],
    depth: Optional[int],
    movetime_ms: Optional[int],
    max_full_moves: int,
    sample_stride: int,
    max_plies: int,
    polyglot: bool,
) -> tuple[List[tuple[str, float, str]], int]:
    fen_sequence = run_selfplay(
        engine_bin=engine_bin,
        params_path=params_path,
        start_fen=fen,
        depth=depth,
        movetime_ms=movetime_ms,
        max_full_moves=max_full_moves,
        polyglot=polyglot,
    )
    samples, outcome, plies_played = reconstruct_samples(
        fen_sequence, sample_stride, max_plies
    )
    rows: List[tuple[str, float, str]] = []
    for row_fen, stm in samples:
        rows.append((row_fen, outcome, stm))
    return rows, plies_played


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate Texel data via skaks self-play"
    )
    parser.add_argument("--engine", type=str, default="skaks", help="skaks binary")
    parser.add_argument(
        "--params",
        type=Path,
        default=Path("tuning/best_params_lots_2.yaml"),
        help="Params YAML for skaks",
    )
    parser.add_argument(
        "--start-fens", type=Path, default=None, help="File with one FEN per line"
    )
    parser.add_argument(
        "--start-pgn",
        type=Path,
        default=None,
        help="PGN file to sample starting positions from (uses random midgame plies)",
    )
    parser.add_argument(
        "--pgn-min-ply",
        type=int,
        default=6,
        help="Minimum ply in PGN game to sample as start (default 6)",
    )
    parser.add_argument(
        "--pgn-max-ply",
        type=int,
        default=60,
        help="Maximum ply in PGN game to sample as start (default 60; 0 means no cap)",
    )
    parser.add_argument(
        "--games", type=int, default=100, help="Number of self-play games"
    )
    parser.add_argument("--max-plies", type=int, default=160, help="Max plies per game")
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Search depth (mutually exclusive with movetime)",
    )
    parser.add_argument(
        "--movetime",
        type=int,
        default=200,
        help="Move time in ms (default 200ms, mutually exclusive with depth)",
    )
    parser.add_argument(
        "--sample-stride", type=int, default=4, help="Sample every N plies"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("texel_selfplay.csv"), help="Output CSV"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="RNG seed for start FEN order"
    )
    parser.add_argument(
        "--polyglot",
        action="store_true",
        help="Enable Polyglot book during self-play (default off to keep stdout clean)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for self-play when using the skaks binary fallback",
    )
    parser.add_argument(
        "--dask-scheduler",
        type=str,
        default=None,
        help="Optional Dask scheduler address (e.g. tcp://host:8786) for cluster execution",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run; appends to existing output and uses progress file",
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=None,
        help="Path to progress file (defaults to <output>.progress)",
    )
    parser.add_argument(
        "--max-duration-min",
        type=float,
        default=None,
        help="Optional max runtime in minutes; stops before cluster timeout",
    )
    args = parser.parse_args(argv)

    if (args.depth is None) == (args.movetime is None):
        print("Provide exactly one of --depth or --movetime", file=sys.stderr)
        return 2

    if args.params is not None and not args.params.exists():
        print(f"Params file not found: {args.params}", file=sys.stderr)
        return 2

    if args.max_plies <= 0:
        print("--max-plies must be positive", file=sys.stderr)
        return 2
    if args.workers <= 0:
        print("--workers must be positive", file=sys.stderr)
        return 2
    if args.resume and args.dask_scheduler:
        print("--resume cannot be combined with --dask-scheduler", file=sys.stderr)
        return 2
    if args.resume and args.workers > 1:
        print(
            "--resume with workers>1 is not supported; set --workers 1", file=sys.stderr
        )
        return 2

    max_full_moves = max(1, math.ceil(args.max_plies / 2))
    if args.start_pgn is not None:
        pgn_cap = None if args.pgn_max_ply == 0 else args.pgn_max_ply
        starts = load_start_fens_from_pgn(
            args.start_pgn,
            limit=args.games,
            seed=args.seed,
            min_ply=args.pgn_min_ply,
            max_ply=pgn_cap,
        )
    else:
        starts = load_start_fens(args.start_fens, limit=args.games, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    progress_path = args.progress_file or args.output.with_suffix(
        args.output.suffix + ".progress"
    )
    games_done = 0
    if args.resume and progress_path.exists():
        try:
            games_done = int(progress_path.read_text().strip() or "0")
        except Exception:
            games_done = 0
    write_mode = "a" if (args.resume and args.output.exists()) else "w"
    if write_mode == "w":
        games_done = 0

    start_time = time.time()
    max_duration_sec = args.max_duration_min * 60.0 if args.max_duration_min else None

    try:
        with args.output.open(write_mode, newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if write_mode == "w":
                writer.writerow(["fen", "outcome", "side_to_move"])

            # Prefer in-process binding if available (disabled when resuming)
            used_binding = False
            if skaks_eval is not None and not args.resume:
                expanded_starts = [starts[i % len(starts)] for i in range(args.games)]
                try:
                    chunk_size = max(1, args.games // 20)
                    binding_samples: List[tuple[str, float, str]] = []
                    games_played_total = 0
                    for start in range(0, args.games, chunk_size):
                        end = min(args.games, start + chunk_size)
                        print(
                            f"[selfplay][binding] {end}/{args.games} games...",
                            end="\r",
                            flush=True,
                        )
                        chunk_starts = expanded_starts[start:end]
                        chunk_samples, chunk_games = run_selfplay_binding(
                            start_fens=chunk_starts,
                            params_path=args.params,
                            depth=args.depth,
                            movetime_ms=args.movetime,
                            max_plies=args.max_plies,
                            sample_stride=args.sample_stride,
                        )
                        binding_samples.extend(chunk_samples)
                        games_played_total += chunk_games

                    for fen, outcome, stm in binding_samples:
                        writer.writerow([fen, outcome, stm])
                    print(
                        f"[selfplay][binding] {games_played_total}/{args.games} games",
                        end="\r",
                        flush=True,
                    )
                    used_binding = True
                except KeyboardInterrupt:
                    print(
                        "\n[warn] interrupted during binding self-play; partial data written",
                        file=sys.stderr,
                    )
                    return 130
                except Exception as exc:
                    print(
                        f"[warn] binding self-play failed; falling back to skaks binary: {exc}",
                        file=sys.stderr,
                    )

            if not used_binding:
                if args.dask_scheduler:
                    try:
                        from dask.distributed import Client
                    except ImportError as exc:  # pragma: no cover
                        print(
                            f"[warn] dask.distributed not installed; cannot use --dask-scheduler: {exc}",
                            file=sys.stderr,
                        )
                        return 2

                    client = Client(args.dask_scheduler)
                    futures = []
                    for game_idx in range(games_done, args.games):
                        fen = starts[game_idx % len(starts)]
                        futures.append(
                            client.submit(
                                play_game_task,
                                fen,
                                engine_bin=args.engine,
                                params_path=args.params,
                                depth=args.depth,
                                movetime_ms=args.movetime,
                                max_full_moves=max_full_moves,
                                sample_stride=args.sample_stride,
                                max_plies=args.max_plies,
                                polyglot=args.polyglot,
                            )
                        )
                    try:
                        for future in futures:
                            rows, plies_played = future.result()
                            for row_fen, outcome, stm in rows:
                                writer.writerow([row_fen, outcome, stm])
                            games_done += 1
                            progress_path.write_text(str(games_done))
                            print(
                                f"[selfplay][dask] {games_done}/{args.games} games (plies {plies_played})",
                                end="\r",
                                flush=True,
                            )
                    except KeyboardInterrupt:
                        print(
                            "\n[warn] interrupted; partial data written",
                            file=sys.stderr,
                        )
                        client.cancel(futures)
                        return 130
                    except Exception as exc:
                        print(
                            f"\n[warn] dask run failed; partial data written: {exc}",
                            file=sys.stderr,
                        )
                        client.cancel(futures)
                        return 1
                elif args.workers > 1:
                    with concurrent.futures.ProcessPoolExecutor(
                        max_workers=args.workers
                    ) as pool:
                        futures = []
                        for game_idx in range(games_done, args.games):
                            fen = starts[game_idx % len(starts)]
                            futures.append(
                                pool.submit(
                                    play_game_task,
                                    fen,
                                    engine_bin=args.engine,
                                    params_path=args.params,
                                    depth=args.depth,
                                    movetime_ms=args.movetime,
                                    max_full_moves=max_full_moves,
                                    sample_stride=args.sample_stride,
                                    max_plies=args.max_plies,
                                    polyglot=args.polyglot,
                                )
                            )
                        try:
                            for future in concurrent.futures.as_completed(futures):
                                rows, plies_played = future.result()
                                for row_fen, outcome, stm in rows:
                                    writer.writerow([row_fen, outcome, stm])
                                games_done += 1
                                progress_path.write_text(str(games_done))
                                print(
                                    f"[selfplay][workers] {games_done}/{args.games} games (plies {plies_played})",
                                    end="\r",
                                    flush=True,
                                )
                        except KeyboardInterrupt:
                            print(
                                "\n[warn] interrupted; partial data written",
                                file=sys.stderr,
                            )
                            pool.shutdown(cancel_futures=True)
                            return 130
                        except Exception as exc:
                            print(
                                f"\n[warn] parallel run failed; partial data written: {exc}",
                                file=sys.stderr,
                            )
                            pool.shutdown(cancel_futures=True)
                            return 1
                else:
                    for game_idx in range(games_done, args.games):
                        if max_duration_sec is not None:
                            elapsed = time.time() - start_time
                            if elapsed >= max_duration_sec:
                                print(
                                    "\n[warn] max-duration reached; stopping early",
                                    file=sys.stderr,
                                )
                                return 0
                        fen = starts[game_idx % len(starts)]
                        print(
                            f"[selfplay] starting game {game_idx + 1}/{args.games} from seed FEN",
                            end="\r",
                            flush=True,
                        )
                        try:
                            rows, plies_played = play_game_task(
                                fen,
                                engine_bin=args.engine,
                                params_path=args.params,
                                depth=args.depth,
                                movetime_ms=args.movetime,
                                max_full_moves=max_full_moves,
                                sample_stride=args.sample_stride,
                                max_plies=args.max_plies,
                                polyglot=args.polyglot,
                            )
                        except KeyboardInterrupt:
                            print(
                                "\n[warn] interrupted; partial data written",
                                file=sys.stderr,
                            )
                            return 130
                        except Exception as exc:
                            print(
                                f"[warn] skipping game {game_idx + 1}: {exc}",
                                file=sys.stderr,
                            )
                            continue

                        for row_fen, outcome, stm in rows:
                            writer.writerow([row_fen, outcome, stm])
                        games_done += 1
                        progress_path.write_text(str(games_done))
                        print(
                            f"[selfplay] {games_done}/{args.games} games (plies {plies_played})",
                            end="\r",
                            flush=True,
                        )
    except KeyboardInterrupt:
        print("\n[warn] interrupted; partial data may be incomplete", file=sys.stderr)
        return 130

    print()
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
