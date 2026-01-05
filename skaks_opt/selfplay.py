from __future__ import annotations


import os
import sys
import math
import multiprocessing as mp
import random
import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from skaks_opt.params import DEFAULT_PARAMS

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    HAS_RICH = True
    console = Console(highlight=False, soft_wrap=False)
except Exception:
    HAS_RICH = False
    console = None
    Panel = None  # type: ignore
    Table = None  # type: ignore


ANSI_COLORS = {
    "cyan": "\033[36m",
    "magenta": "\033[35m",
    "yellow": "\033[33m",
    "green": "\033[32m",
    "bright_white": "\033[97m",
    "bright_blue": "\033[94m",
}
ANSI_RESET = "\033[0m"

SPINNER_FRAMES = ["|", "/", "-", "\\"]
CHESS_SWARM = ["♔", "♕", "♖", "♗", "♘", "♙", "♚", "♛", "♜", "♝", "♞", "♟"]
PALETTE = ["cyan", "magenta", "yellow", "green", "bright_white", "bright_blue"]
DEFAULT_START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


@dataclass
class CandidateResult:
    index: int
    score: float
    data: Dict[str, Any]
    wld: Tuple[int, int, int]
    wall: float
    accepted: bool = False


def _color(text: str, color: str) -> str:
    if HAS_RICH:
        return f"[{color}]{text}[/{color}]"
    if sys.stdout.isatty():
        prefix = ANSI_COLORS.get(color, "")
        return f"{prefix}{text}{ANSI_RESET if prefix else ''}"
    return text


def _live_line(text: str) -> None:
    if HAS_RICH and console:
        console.print(text, end="\r")
        try:
            console.file.flush()
        except Exception:
            pass
    else:
        sys.stdout.write("\r" + text)
        sys.stdout.flush()


def _final_line(text: str) -> None:
    if HAS_RICH and console:
        console.print(text)
    else:
        print(text)


try:
    import skaks_eval
except ImportError:  # pragma: no cover - surfaced when run
    skaks_eval = None


@dataclass
class SelfPlayConfig:
    engine: str
    baseline_params: Optional[Path]
    start_params: Optional[Path]
    output: Optional[Path]
    include_prefix: Optional[List[str]]
    exclude_prefix: Optional[List[str]]
    phase_weights_only: bool
    games: int
    iterations: int
    repeats: int
    noise: float
    strategy: str
    beam_size: int
    depth: Optional[int]
    time_per_move: Optional[float]
    clock: Optional[float]
    concurrency: int
    arena_workers: int
    arena_pgn: Optional[Path]
    arena_min_ply: int
    arena_max_ply: int
    arena_seed: int
    child_output: bool
    cma_popsize: Optional[int]
    cma_sigma: Optional[float]
    seed: int
    dask_scheduler: Optional[str] = None
    dask_local_workers: Optional[int] = None
    dask_local_threads: Optional[int] = None
    start_fens: Optional[List[str]] = None
    dask_shard_hint: Optional[int] = None
    rich_progress: bool = True


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = deepcopy(val)
    return merged


def _normalize_params(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if payload is None:
        return deepcopy(DEFAULT_PARAMS)
    return _deep_merge(DEFAULT_PARAMS, payload)


def _load_params(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _save_params(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=True)


def _collect_numeric_leaves(
    data: Any,
    prefix: str = "",
    include_prefixes: Optional[Sequence[str]] = None,
    exclude_prefixes: Optional[Sequence[str]] = None,
) -> List[Tuple[str, Any]]:
    items: List[Tuple[str, Any]] = []
    if isinstance(data, dict):
        for key, value in data.items():
            next_prefix = f"{prefix}.{key}" if prefix else key
            items.extend(
                _collect_numeric_leaves(
                    value,
                    next_prefix,
                    include_prefixes,
                    exclude_prefixes,
                )
            )
    elif isinstance(data, list):
        for idx, value in enumerate(data):
            next_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            items.extend(
                _collect_numeric_leaves(
                    value,
                    next_prefix,
                    include_prefixes,
                    exclude_prefixes,
                )
            )
    elif isinstance(data, (int, float)):
        allowed = True
        if include_prefixes:
            allowed = any(prefix.startswith(pfx) for pfx in include_prefixes)
        if allowed and exclude_prefixes:
            allowed = not any(prefix.startswith(pfx) for pfx in exclude_prefixes)
        if allowed:
            items.append((prefix, data))
    return items


def _set_by_path(payload: Dict[str, Any], dotted: str, value: Any) -> None:
    current = payload
    tokens: List[Any] = []
    buf = ""
    i = 0
    while i < len(dotted):
        ch = dotted[i]
        if ch == ".":
            if buf:
                tokens.append(buf)
                buf = ""
            i += 1
        elif ch == "[":
            if buf:
                tokens.append(buf)
                buf = ""
            j = dotted.index("]", i)
            tokens.append(int(dotted[i + 1 : j]))
            i = j + 1
            if i < len(dotted) and dotted[i] == ".":
                i += 1
        else:
            buf += ch
            i += 1
    if buf:
        tokens.append(buf)

    for token in tokens[:-1]:
        current = current[token]
    current[tokens[-1]] = value


def perturb_params(
    base: Dict[str, Any],
    noise: float,
    include_prefixes: Optional[List[str]] = None,
    exclude_prefixes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    replica = deepcopy(base)
    leaves = _collect_numeric_leaves(
        replica,
        include_prefixes=include_prefixes,
        exclude_prefixes=exclude_prefixes,
    )
    for path, val in leaves:
        scale = math.exp(random.gauss(0.0, noise))
        if isinstance(val, int):
            new_val = max(1, int(round(val * scale)))
        else:
            new_val = float(val * scale)
        _set_by_path(replica, path, new_val)
    return replica


def _flatten_params(
    data: Dict[str, Any],
    include_prefixes: Optional[List[str]],
    exclude_prefixes: Optional[List[str]],
) -> Tuple[List[str], List[float], List[bool]]:
    leaves = _collect_numeric_leaves(
        data,
        include_prefixes=include_prefixes,
        exclude_prefixes=exclude_prefixes,
    )
    paths = [path for path, _ in leaves]
    values = [float(value) for _, value in leaves]
    is_int = [
        isinstance(value, int) and not isinstance(value, bool) for _, value in leaves
    ]
    return paths, values, is_int


def _vector_to_params(
    template: Dict[str, Any],
    paths: List[str],
    vec: List[float],
    is_int: List[bool],
) -> Dict[str, Any]:
    replica = deepcopy(template)
    for path, val, flag in zip(paths, vec, is_int):
        if flag:
            val = int(round(val))
            if val == 0:
                val = 1
        _set_by_path(replica, path, val)
    return replica


def _normalize_fens(fens: Optional[List[str]], games: int) -> List[str]:
    if not fens:
        base = [DEFAULT_START_FEN]
    else:
        base = []
        for fen in fens:
            fen_stripped = fen.strip().lower()
            if fen_stripped in {"start", "startpos"}:
                base.append(DEFAULT_START_FEN)
            else:
                parts = fen.split()
                if len(parts) != 6:
                    raise ValueError(
                        f"FEN must have 6 fields separated by spaces: {fen}"
                    )
                base.append(fen)
    if games > len(base):
        copies = (games + len(base) - 1) // len(base)
        base = (base * copies)[:games]
    else:
        base = base[:games]
    return base


def _load_fens_from_pgn(
    pgn_path: Path,
    target: int,
    min_ply: int,
    max_ply: int,
    seed: int,
) -> List[str]:
    try:
        import chess.pgn  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("python-chess is required for --arena-pgn") from exc

    rng = random.Random(seed)
    result: List[str] = []
    with pgn_path.open("r", encoding="utf-8") as fh:
        while len(result) < target:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            board = game.board()
            moves = list(game.mainline_moves())
            if not moves:
                continue
            target_ply = rng.randint(min_ply, max_ply) if max_ply > 0 else min_ply
            target_ply = max(1, target_ply)
            applied = 0
            for move in moves:
                board.push(move)
                applied += 1
                if applied >= target_ply:
                    break
            result.append(board.fen())
    if not result:
        raise RuntimeError("Failed to sample start positions from PGN")
    while len(result) < target:
        result.append(rng.choice(result))
    return result


def _run_arena_shard(
    payload: Tuple[
        List[str],
        Dict[str, Any],
        Dict[str, Any],
        int,
        int,
        int,
        float,
        float,
        float,
        int,
    ],
) -> Dict[str, int]:
    (
        start_fens,
        base_params,
        cand_params,
        depth,
        movetime_ms,
        max_plies,
        wtime,
        btime,
        increment,
        moves_to_go,
    ) = payload
    import skaks_eval as _skaks_eval

    arena_result = _skaks_eval.arena(
        start_fens=start_fens,
        base_params=base_params,
        cand_params=cand_params,
        games=len(start_fens),
        depth=depth,
        movetime_ms=movetime_ms,
        max_plies=max_plies,
        wtime=wtime,
        btime=btime,
        increment=increment,
        moves_to_go=moves_to_go,
    )
    return {
        "wins": int(arena_result.get("wins", 0)),
        "losses": int(arena_result.get("losses", 0)),
        "draws": int(arena_result.get("draws", 0)),
    }


def evaluate_candidate(
    *,
    baseline_data: Dict[str, Any],
    candidate_data: Dict[str, Any],
    games: int,
    depth: Optional[int],
    time_per_move: Optional[float],
    clock: Optional[float],
    arena_workers: int,
    start_fens: Optional[List[str]] = None,
    dask_client: Optional[Any] = None,
    shard_hint: Optional[int] = None,
) -> Tuple[float, Tuple[int, int, int], float]:
    if skaks_eval is None:
        raise RuntimeError("skaks_eval not available; install bindings first")

    movetime_ms = int(time_per_move * 1000) if time_per_move is not None else 0
    fen_pool = _normalize_fens(start_fens, games)
    base_params = _normalize_params(baseline_data)
    cand_params = _normalize_params(candidate_data)

    # New: clock controls
    wtime = btime = increment = moves_to_go = 0
    # Strict enforcement: if clock is set, override depth and movetime_ms
    if clock is not None:
        wtime = btime = int(clock * 1000)
        increment = 0
        moves_to_go = 40
        depth_value = 0
        movetime_ms = 0
    else:
        depth_value = depth or 0
        movetime_ms = int(time_per_move * 1000) if time_per_move is not None else 0

    start_time = time.monotonic()

    effective_workers = shard_hint if shard_hint and shard_hint > 0 else arena_workers

    if dask_client is not None and effective_workers != 0:
        shard_count = effective_workers if effective_workers > 0 else len(fen_pool)
        shard_count = max(1, min(shard_count, len(fen_pool)))
        chunk = max(1, (len(fen_pool) + shard_count - 1) // shard_count)
        futures = []
        try:
            for idx in range(0, len(fen_pool), chunk):
                payload = (
                    fen_pool[idx : idx + chunk],
                    base_params,
                    cand_params,
                    depth_value,
                    movetime_ms,
                    160,
                    wtime,
                    btime,
                    increment,
                    moves_to_go,
                )
                futures.append(dask_client.submit(_run_arena_shard, payload))
            shard_results = dask_client.gather(futures)
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            raise
        except Exception:
            for future in futures:
                future.cancel()
            raise
        wins = sum(entry["wins"] for entry in shard_results)
        losses = sum(entry["losses"] for entry in shard_results)
        draws = sum(entry["draws"] for entry in shard_results)
    elif arena_workers <= 1 or len(fen_pool) <= arena_workers:
        arena_result = skaks_eval.arena(
            start_fens=fen_pool,
            base_params=base_params,
            cand_params=cand_params,
            games=games,
            depth=depth_value,
            movetime_ms=movetime_ms,
            max_plies=160,
            wtime=wtime,
            btime=btime,
            increment=increment,
            moves_to_go=moves_to_go,
        )
        wins = int(arena_result.get("wins", 0))
        losses = int(arena_result.get("losses", 0))
        draws = int(arena_result.get("draws", 0))
    else:
        chunk = max(1, (len(fen_pool) + arena_workers - 1) // arena_workers)
        tasks: List[
            Tuple[
                List[str],
                Dict[str, Any],
                Dict[str, Any],
                int,
                int,
                int,
                float,
                float,
                float,
                int,
            ]
        ] = []
        for idx in range(0, len(fen_pool), chunk):
            tasks.append(
                (
                    fen_pool[idx : idx + chunk],
                    base_params,
                    cand_params,
                    depth_value,
                    movetime_ms,
                    160,
                    wtime,
                    btime,
                    increment,
                    moves_to_go,
                )
            )
        with mp.Pool(processes=arena_workers) as pool:
            try:
                shard_results = pool.map(_run_arena_shard, tasks)
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
                raise
        wins = sum(entry["wins"] for entry in shard_results)
        losses = sum(entry["losses"] for entry in shard_results)
        draws = sum(entry["draws"] for entry in shard_results)

    total = wins + losses + draws
    score = (wins + 0.5 * draws) / total if total > 0 else 0.0
    wall = time.monotonic() - start_time
    return score, (wins, losses, draws), wall


def evaluate_candidate_repeats(
    *,
    candidate_data: Dict[str, Any],
    baseline_data: Dict[str, Any],
    games: int,
    depth: Optional[int],
    time_per_move: Optional[float],
    clock: Optional[float],
    arena_workers: int,
    repeats: int,
    start_fens: Optional[List[str]],
    quiet: bool,
    dask_client: Optional[Any] = None,
    shard_hint: Optional[int] = None,
) -> Tuple[float, Tuple[int, int, int], float]:
    total_score = 0.0
    total_wins = total_losses = total_draws = 0
    total_wall = 0.0
    start_time = time.monotonic()

    fen_source = start_fens or []
    slice_size = games

    for rep_idx in range(repeats):
        if fen_source:
            offset = rep_idx * slice_size
            subset = fen_source[offset : offset + slice_size]
            if not subset:
                subset = fen_source[:slice_size]
        else:
            subset = None

        # Strict enforcement: if clock is set, override depth and time_per_move
        if clock is not None:
            depth = 0
            time_per_move = 0
        score, (wins, losses, draws), wall = evaluate_candidate(
            baseline_data=baseline_data,
            candidate_data=candidate_data,
            games=games,
            depth=depth,
            time_per_move=time_per_move,
            clock=clock,
            arena_workers=arena_workers,
            start_fens=subset,
            dask_client=dask_client,
            shard_hint=shard_hint,
        )
        total_score += score
        total_wins += wins
        total_losses += losses
        total_draws += draws
        total_wall += wall

        if not quiet:
            spinner = SPINNER_FRAMES[rep_idx % len(SPINNER_FRAMES)]
            piece = CHESS_SWARM[rep_idx % len(CHESS_SWARM)]
            colour = PALETTE[rep_idx % len(PALETTE)]
            elapsed = time.monotonic() - start_time
            print(
                f"{spinner} {piece} {colour} repeat {rep_idx + 1}/{repeats} t={elapsed:.1f}s",
                end="\r",
                flush=True,
            )

    if not quiet:
        print()

    avg_score = total_score / repeats if repeats > 0 else 0.0
    return avg_score, (total_wins, total_losses, total_draws), total_wall


def score_to_elo(score: float) -> float:
    if score <= 0.0:
        return float("-inf")
    if score >= 1.0:
        return float("inf")
    return 400.0 * math.log10(score / (1.0 - score))


def _normalize_scheduler_address(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    if "://" not in value:
        return f"tcp://{value}"
    return value


def _dask_probe_worker(dask_worker: Any) -> Dict[str, Any]:
    try:
        import skaks_eval as _skaks_eval
    except Exception as exc:  # pragma: no cover - diagnostic surface
        return {"ok": False, "error": f"import failed: {exc}"}

    ok = False
    cp: Optional[int] = None
    error_msg: Optional[str] = None

    if hasattr(_skaks_eval, "eval_fen_single"):
        try:
            probe = _skaks_eval.eval_fen_single(DEFAULT_START_FEN)
        except Exception as exc:
            return {"ok": False, "error": f"eval_fen_single failed: {exc}"}
        ok = bool(probe.get("ok"))
        cp = probe.get("cp")  # type: ignore[assignment]
        error_msg = probe.get("error")
        if not ok and not error_msg:
            error_msg = "eval_fen_single reported failure"
    elif hasattr(_skaks_eval, "eval_fens"):
        try:
            probe = _skaks_eval.eval_fens([DEFAULT_START_FEN], params=None, threads=1)
        except Exception as exc:
            return {"ok": False, "error": f"eval_fens failed: {exc}"}
        cp_data = probe.get("cp")
        if cp_data is not None:
            try:
                cp = int(cp_data[0])  # type: ignore[index]
            except Exception:
                cp = None
        err_list = probe.get("errors") or []
        if err_list:
            first_err = err_list[0]
            if first_err:
                error_msg = str(first_err)
        if error_msg is None:
            ok = True
    else:
        return {"ok": False, "error": "skaks_eval missing eval bindings"}

    if not ok:
        return {"ok": False, "error": error_msg or "probe failed", "worker": getattr(dask_worker, "address", None)}

    payload: Dict[str, Any] = {
        "ok": True,
        "cp": cp,
        "error": error_msg,
        "worker": getattr(dask_worker, "address", None),
    }
    return payload


def _ensure_dask_ready(client: Any, expected: Optional[int]) -> int:
    try:
        from distributed.utils import TimeoutError as _DaskTimeout
    except Exception:  # pragma: no cover - optional dependency
        _DaskTimeout = TimeoutError  # type: ignore[assignment]

    wait_target = expected if expected and expected > 0 else None
    if wait_target:
        try:
            client.wait_for_workers(n_workers=wait_target, timeout=60)
        except _DaskTimeout:
            pass

    info = client.nthreads()
    worker_count = len(info)
    if worker_count == 0:
        raise RuntimeError("Dask connected but no workers registered")

    probe_results: Dict[str, Dict[str, Any]] = client.run(_dask_probe_worker)
    failures = {
        addr: data
        for addr, data in probe_results.items()
        if not data.get("ok", False)
    }
    if failures:
        msgs = ", ".join(
            f"{addr}: {entry.get('error', 'unknown')}" for addr, entry in failures.items()
        )
        raise RuntimeError(f"skaks_eval unavailable on workers: {msgs}")

    return worker_count


@contextmanager
def _dask_client(config: "SelfPlayConfig"):
    host_env = os.environ.get("DASK_SCHEDULER_SERVICE_HOST")
    port_env = os.environ.get("DASK_SCHEDULER_SERVICE_PORT")

    scheduler_addr: Optional[str]
    if config.dask_scheduler:
        scheduler_addr = _normalize_scheduler_address(config.dask_scheduler)
    elif host_env and port_env:
        endpoint = f"{host_env}:{port_env}".strip()
        scheduler_addr = _normalize_scheduler_address(endpoint)
    else:
        scheduler_addr = None

    if not scheduler_addr and not config.dask_local_workers:
        yield None, config.dask_shard_hint
        return

    if config.dask_scheduler and config.dask_local_workers:
        raise RuntimeError("Use either --dask-scheduler or --dask-workers, not both")

    try:
        from distributed import Client, LocalCluster
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "dask.distributed is required for Dask-enabled selfplay"
        ) from exc

    client = None
    cluster = None
    try:
        if scheduler_addr:
            client = Client(scheduler_addr)
            source = (
                config.dask_scheduler
                if config.dask_scheduler
                else f"env {host_env}:{port_env}"
            )
            _final_line(f"[dask] Connected to scheduler at {scheduler_addr} ({source})")
        else:
            cluster = LocalCluster(
                n_workers=config.dask_local_workers,
                threads_per_worker=config.dask_local_threads or 1,
                processes=True,
                dashboard_address=None,
            )
            client = Client(cluster)
            _final_line(
                f"[dask] Started LocalCluster with {config.dask_local_workers} workers"
            )

        shard_hint = config.dask_shard_hint
        if client is not None:
            expected = config.dask_local_workers if cluster else shard_hint
            worker_count = _ensure_dask_ready(client, expected)
            if shard_hint is None or shard_hint <= 0:
                shard_hint = worker_count
        yield client, shard_hint
    finally:
        try:
            if client is not None:
                client.close()
        finally:
            if cluster is not None:
                cluster.close()


def _render_iteration_summary(
    iteration: int, candidates: List[CandidateResult], config: "SelfPlayConfig"
) -> None:
    if not candidates:
        return

    rich_enabled = (
        config.rich_progress
        and HAS_RICH
        and console is not None
        and Table is not None
        and Panel is not None
    )

    if rich_enabled:
        assert Table is not None
        assert Panel is not None
        table = Table(box=None, expand=False)
        table.add_column("#", justify="right")
        table.add_column("Score", justify="right")
        table.add_column("W-L-D", justify="center")
        table.add_column("Elo≈", justify="right")
        table.add_column("Time (s)", justify="right")
        table.add_column("Status", justify="left")

        for idx, result in enumerate(candidates, start=1):
            marker = "★" if idx == 1 else ""
            saved = "✅" if result.accepted else ""
            table.add_row(
                f"{marker}{result.index}",
                f"{result.score:.3f}",
                f"{result.wld[0]}/{result.wld[1]}/{result.wld[2]}",
                f"{score_to_elo(result.score):+.1f}",
                f"{result.wall:.1f}",
                saved,
            )

        console.print(
            Panel(
                table,
                title=f"Iteration {iteration}",
                expand=False,
                border_style="bright_blue",
            )
        )
    else:
        _final_line(f"Iteration {iteration} results:")
        for idx, result in enumerate(candidates, start=1):
            marker = "*" if idx == 1 else "-"
            saved = " [saved]" if result.accepted else ""
            _final_line(
                f"  {marker} cand#{result.index} score={result.score:.3f} "
                f"WLD={result.wld[0]}/{result.wld[1]}/{result.wld[2]} "
                f"Elo~{score_to_elo(result.score):+.1f} time={result.wall:.1f}s{saved}"
            )


def run_selfplay(config: SelfPlayConfig) -> Path:
    if skaks_eval is None:
        raise RuntimeError("skaks_eval not available; install bindings first")

    random.seed(config.seed)

    baseline_path = config.baseline_params.resolve() if config.baseline_params else None
    start_path = config.start_params.resolve() if config.start_params else baseline_path
    if start_path is None:
        raise RuntimeError("Provide --start-params or --baseline-params")

    base_data = _load_params(start_path)
    best_data = base_data
    best_score = -1.0

    include_prefixes = list(config.include_prefix or [])
    if config.phase_weights_only:
        include_prefixes = [
            "evaluation.phase_weights_mg",
            "evaluation.phase_weights_eg",
        ]

    if config.output:
        best_path = config.output.resolve()
    else:
        src = start_path
        best_path = src.with_name(f"{src.stem}_optimized.yaml").resolve()

    _save_params(best_data, best_path)

    baseline_data = (
        _load_params(baseline_path) if baseline_path else deepcopy(base_data)
    )

    target_fens: Optional[List[str]] = None
    if config.arena_pgn:
        target = config.games * max(1, config.repeats)
        target_fens = _load_fens_from_pgn(
            config.arena_pgn,
            target,
            config.arena_min_ply,
            config.arena_max_ply,
            config.arena_seed,
        )
    elif config.start_fens:
        target_fens = config.start_fens

    beam: List[Tuple[float, Dict[str, Any]]] = [(best_score, best_data)]

    rich_enabled = (
        config.rich_progress
        and HAS_RICH
        and console is not None
        and Table is not None
        and Panel is not None
    )

    with _dask_client(config) as (dask_client, shard_hint):
        for iteration in range(1, config.iterations + 1):
            candidate_results: List[CandidateResult] = []

            if config.strategy == "beam":
                parents = [data for (_, data) in beam] or [best_data]
                for idx in range(config.beam_size):
                    parent = parents[idx % len(parents)]
                    cand_data = perturb_params(
                        parent,
                        config.noise,
                        include_prefixes=include_prefixes or None,
                        exclude_prefixes=config.exclude_prefix,
                    )
                    score, wld, wall = evaluate_candidate_repeats(
                        candidate_data=cand_data,
                        baseline_data=baseline_data,
                        games=config.games,
                        depth=config.depth,
                        time_per_move=config.time_per_move,
                        clock=config.clock,
                        arena_workers=config.arena_workers,
                        repeats=config.repeats,
                        start_fens=target_fens,
                        quiet=config.child_output,
                        dask_client=dask_client,
                        shard_hint=shard_hint,
                    )
                    candidate_results.append(
                        CandidateResult(
                            index=idx + 1,
                            score=score,
                            data=cand_data,
                            wld=wld,
                            wall=wall,
                        )
                    )
                    if not rich_enabled:
                        color = PALETTE[idx % len(PALETTE)]
                        _final_line(
                            f"{_color('✓', 'green')} iter={iteration} cand={idx + 1}/{config.beam_size} "
                            f"score={_color(f'{score:.3f}', color)} "
                            f"WLD={wld[0]}/{wld[1]}/{wld[2]} elo~{score_to_elo(score):+.1f} time={wall:.1f}s"
                        )

                candidate_results.sort(key=lambda item: item.score, reverse=True)
                beam = [
                    (result.score, result.data)
                    for result in candidate_results[: config.beam_size]
                ]
                top_result = candidate_results[0]
            else:
                paths, mean_vec, is_int = _flatten_params(
                    best_data,
                    include_prefixes=include_prefixes or None,
                    exclude_prefixes=config.exclude_prefix,
                )
                popsize = (
                    config.cma_popsize
                    if config.cma_popsize is not None
                    else max(4, int(4 + 3 * math.log(len(mean_vec) + 1)))
                )
                sigma = (
                    config.cma_sigma if config.cma_sigma is not None else config.noise
                )
                sigma = max(1e-4, sigma)

                for idx in range(popsize):
                    vec = []
                    for value in mean_vec:
                        scale = sigma * max(abs(value), 1.0)
                        vec.append(value + random.gauss(0.0, scale))
                    cand_data = _vector_to_params(base_data, paths, vec, is_int)
                    score, wld, wall = evaluate_candidate_repeats(
                        candidate_data=cand_data,
                        baseline_data=baseline_data,
                        games=config.games,
                        depth=config.depth,
                        time_per_move=config.time_per_move,
                        clock=config.clock,
                        arena_workers=config.arena_workers,
                        repeats=config.repeats,
                        start_fens=target_fens,
                        quiet=config.child_output,
                        dask_client=dask_client,
                        shard_hint=shard_hint,
                    )
                    candidate_results.append(
                        CandidateResult(
                            index=idx + 1,
                            score=score,
                            data=cand_data,
                            wld=wld,
                            wall=wall,
                        )
                    )
                    if not rich_enabled:
                        color = PALETTE[idx % len(PALETTE)]
                        _final_line(
                            f"{_color('✓', 'green')} iter={iteration} cand={idx + 1}/{popsize} "
                            f"score={_color(f'{score:.3f}', color)} "
                            f"WLD={wld[0]}/{wld[1]}/{wld[2]} elo~{score_to_elo(score):+.1f} time={wall:.1f}s"
                        )

                candidate_results.sort(key=lambda item: item.score, reverse=True)
                top_result = candidate_results[0]
                mu = max(1, popsize // 2)
                weights = [math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)]
                weight_sum = sum(weights)
                weights = [w / weight_sum for w in weights]

                new_mean = [0.0 for _ in mean_vec]
                for rank in range(min(mu, len(candidate_results))):
                    vec_paths, vec_values, _ = _flatten_params(
                        candidate_results[rank].data,
                        include_prefixes=include_prefixes or None,
                        exclude_prefixes=config.exclude_prefix,
                    )
                    if vec_paths != paths:
                        continue
                    for value_idx, value in enumerate(vec_values):
                        new_mean[value_idx] += weights[rank] * value
                mean_vec = new_mean
                best_data = _vector_to_params(base_data, paths, mean_vec, is_int)
                beam = [(top_result.score, best_data)]

            accept_candidate = top_result.score > best_score
            if accept_candidate:
                best_score = top_result.score
                best_data = top_result.data
                baseline_data = deepcopy(best_data)
                top_result.accepted = True
                _save_params(best_data, best_path)
                _final_line(
                    _color(
                        f"⚑ new best score={best_score:.3f} saved to {best_path}",
                        "yellow",
                    )
                )

            _render_iteration_summary(iteration, candidate_results, config)

    _final_line(
        _color(f"🏁 Best score={best_score:.3f}, params at {best_path}", "bright_white")
    )
    return best_path
