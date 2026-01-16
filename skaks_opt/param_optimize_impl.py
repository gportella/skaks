#!/usr/bin/env python3
"""Simple self-play optimizer loop using the internal arena batch runner.

This script perturbs numeric YAML params, runs head-to-head matches against a
baseline (with both color orders), and keeps the best candidate by score.
Machine-readable results are obtained via arena_runner with --summary-json.
"""

import argparse
import contextlib
import itertools
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

from skaks_opt import arena_runner
from skaks_opt.dask_support import dask_client_from_args
from skaks_opt.params import DEFAULT_PARAMS, param_set_prefixes
from skaks_opt.pst import apply_pst_symmetry
from skaks_opt.stats import score_to_elo, summarize_wld

try:
    import skaks_eval
except Exception:
    skaks_eval = None
    # Delay import error until we actually need the binding

try:
    import yaml
except ImportError:  # pragma: no cover - dependency should be present
    print("PyYAML is required: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

try:  # pragma: no cover - purely cosmetic
    from rich.console import Console

    HAS_RICH = True
    console = Console(highlight=False, soft_wrap=False)
except Exception:  # pragma: no cover
    HAS_RICH = False
    console = None

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
CHESS_SWARM = [
    "♔",
    "♕",
    "♖",
    "♗",
    "♘",
    "♙",
    "♚",
    "♛",
    "♜",
    "♝",
    "♞",
    "♟",
]
PALETTE = ["cyan", "magenta", "yellow", "green", "bright_white", "bright_blue"]

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _live_line(text: str) -> None:
    """Render a single updating line (carriage return based)."""
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


def _color(text: str, color: str) -> str:
    if HAS_RICH:
        return f"[{color}]{text}[/{color}]"
    if sys.stdout.isatty():
        prefix = ANSI_COLORS.get(color, "")
        return f"{prefix}{text}{ANSI_RESET if prefix else ''}"
    return text


def _resolve_engine(path_str: str) -> Path:
    """Resolve an engine path, trying common build locations and PATH."""
    candidate = Path(path_str)
    search: List[Path] = []
    if candidate.is_absolute():
        search.append(candidate)
    else:
        search.append(Path.cwd() / candidate)
        search.append(Path(__file__).resolve().parent.parent / candidate)
        search.append(
            Path(__file__).resolve().parent.parent
            / "build"
            / "debug"
            / "bin"
            / candidate.name
        )
        search.append(
            Path(__file__).resolve().parent.parent
            / "build"
            / "release"
            / "bin"
            / candidate.name
        )
    for cand in search:
        if cand.exists():
            return cand.resolve()
    which = shutil.which(path_str)
    if which:
        return Path(which).resolve()
    raise FileNotFoundError(f"Engine binary not found: {path_str}")


def _load_params(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_params(data: Dict[str, Any], path: Path) -> None:
    # Align numeric leaf types so downstream consumers (UCI engine params) keep
    # integers where expected while still preserving floats for the few
    # genuinely real-valued parameters.
    snapshot = json.loads(json.dumps(data))
    coerced = _coerce_numeric_types(DEFAULT_PARAMS, snapshot)
    with path.open("w", encoding="utf-8") as f:
        from skaks_opt.yaml_utils import dump_yaml

        dump_yaml(coerced, f, sort_keys=True)


def _collect_numeric_leaves(
    data: Any,
    prefix: str = "",
    include_prefixes: Optional[List[str]] = None,
    exclude_prefixes: Optional[List[str]] = None,
) -> List[Tuple[str, Any]]:
    items: List[Tuple[str, Any]] = []
    if isinstance(data, dict):
        for k, v in data.items():
            items.extend(
                _collect_numeric_leaves(
                    v,
                    f"{prefix}.{k}" if prefix else k,
                    include_prefixes,
                    exclude_prefixes,
                )
            )
    elif isinstance(data, list):
        for idx, v in enumerate(data):
            items.extend(
                _collect_numeric_leaves(
                    v,
                    f"{prefix}[{idx}]" if prefix else f"[{idx}]",
                    include_prefixes,
                    exclude_prefixes,
                )
            )
    elif isinstance(data, (int, float)):
        if include_prefixes:
            if not any(prefix.startswith(pfx) for pfx in include_prefixes):
                return items
        if exclude_prefixes:
            if any(prefix.startswith(pfx) for pfx in exclude_prefixes):
                return items
        items.append((prefix, data))
    return items


def _set_by_path(data: Any, path: str, value: Any) -> None:
    if "[" in path or "." in path:
        parts: List[Any] = []
        buf = ""
        i = 0
        while i < len(path):
            if path[i] == ".":
                if buf:
                    parts.append(buf)
                    buf = ""
                i += 1
            elif path[i] == "[":
                if buf:
                    parts.append(buf)
                    buf = ""
                j = path.index("]", i)
                parts.append(int(path[i + 1 : j]))
                i = j + 1
                if i < len(path) and path[i] == ".":
                    i += 1
            else:
                buf += path[i]
                i += 1
        if buf:
            parts.append(buf)
    else:
        parts = [path]

    ref = data
    for key in parts[:-1]:
        ref = ref[key]
    ref[parts[-1]] = value


def perturb_params(
    base: Dict[str, Any],
    noise: float,
    include_prefixes: Optional[List[str]] = None,
    exclude_prefixes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Return a copy with multiplicative noise on numeric leaves (filtered)."""
    data = json.loads(json.dumps(base))  # cheap deep copy
    leaves = _collect_numeric_leaves(
        data, include_prefixes=include_prefixes, exclude_prefixes=exclude_prefixes
    )
    for path, val in leaves:
        scale = math.exp(random.gauss(0.0, noise))
        if isinstance(val, int):
            new_val = max(1, int(round(val * scale)))
        else:
            new_val = float(val * scale)
        _set_by_path(data, path, new_val)
    return data


def _coerce_numeric_types(template: Any, payload: Any) -> Any:
    """Best-effort type alignment so ints in the template stay ints."""
    # When no template is available we still want to recurse into the payload so
    # we can round near-integer floats to plain ints before handing them to the
    # native arena binding.
    if isinstance(payload, dict):
        template_dict = template if isinstance(template, dict) else {}
        return {
            key: _coerce_numeric_types(template_dict.get(key), value)
            for key, value in payload.items()
        }

    if isinstance(payload, list):
        template_list = template if isinstance(template, list) else []
        coerced: List[Any] = []
        for idx, value in enumerate(payload):
            tmpl = template_list[idx] if idx < len(template_list) else None
            coerced.append(_coerce_numeric_types(tmpl, value))
        return coerced

    if template is None:
        if isinstance(payload, float) and math.isfinite(payload):
            rounded = float(round(payload))
            if abs(payload - rounded) < 1e-9:
                return int(rounded)
        return payload
    if isinstance(template, bool):
        return bool(payload)
    if isinstance(template, int):
        if isinstance(payload, (int, bool)):
            return int(payload)
        if isinstance(payload, float):
            return int(round(payload))
        return payload
    if isinstance(template, float):
        if isinstance(payload, (int, float)):
            return float(payload)
        return payload
    return payload


def _coerce_params_for_arena(
    payload: Optional[Dict[str, Any]],
    *templates: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if payload is None:
        return None
    result: Any = json.loads(json.dumps(payload))
    for tmpl in templates:
        if isinstance(tmpl, dict) and tmpl:
            result = _coerce_numeric_types(tmpl, result)
    return result


def _prepare_arena_params(
    base_params: Optional[Dict[str, Any]],
    cand_params: Dict[str, Any],
    coerce_template: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    templates = (coerce_template, base_params, DEFAULT_PARAMS)
    coerced_base = _coerce_params_for_arena(base_params, *templates)
    coerced_cand = cast(
        Dict[str, Any],
        _coerce_params_for_arena(cand_params, *templates),
    )
    if coerced_base is not None:
        apply_pst_symmetry(coerced_base)
    apply_pst_symmetry(coerced_cand)
    return coerced_base, coerced_cand


def _flatten_params(
    data: Dict[str, Any],
    include_prefixes: Optional[List[str]],
    exclude_prefixes: Optional[List[str]],
) -> Tuple[List[str], List[float], List[bool]]:
    paths_vals = _collect_numeric_leaves(
        data, include_prefixes=include_prefixes, exclude_prefixes=exclude_prefixes
    )
    paths: List[str] = []
    vals: List[float] = []
    is_int: List[bool] = []
    for path, v in paths_vals:
        paths.append(path)
        vals.append(float(v))
        is_int.append(isinstance(v, int) and not isinstance(v, bool))
    return paths, vals, is_int


def _vector_to_params(
    template: Dict[str, Any], paths: List[str], vec: List[float], is_int: List[bool]
) -> Dict[str, Any]:
    data = json.loads(json.dumps(template))
    for path, val, flag in zip(paths, vec, is_int):
        if flag:
            val = int(round(val))
            if val == 0:
                val = 1  # avoid zeroed integer weights
        _set_by_path(data, path, val)
    return data


def run_batch(
    *,
    engine: Path,
    opponent: Path,
    engine_params: Optional[Path],
    opponent_params: Optional[Path],
    engine_label: str,
    opponent_label: str,
    games: int,
    depth: Optional[int],
    time_per_move: Optional[float],
    clock: Optional[float],
    increment: Optional[float],
    opponent_clock: Optional[float],
    opponent_increment: Optional[float],
    moves_to_go: Optional[int],
    opponent_time_per_move: Optional[float] = None,
    opponent_depth_factor: Optional[float] = None,
    concurrency: int,
    quiet: bool,
    disable_elo_store: bool = False,
) -> Dict[str, Any]:
    summary_path = Path(tempfile.mkstemp(suffix="_summary.json")[1])
    argv: List[str] = [
        "--engine",
        str(engine),
        "--opponent",
        str(opponent),
        "--engine-label",
        engine_label,
        "--opponent-label",
        opponent_label,
        "--games",
        str(games),
        "--concurrency",
        str(concurrency),
        "--summary-json",
        str(summary_path),
        "--no-handicap",
        "--timeout",
        "60",
    ]
    if engine_params:
        argv.extend(["--engine-params", str(engine_params)])
    if opponent_params:
        argv.extend(["--opponent-params", str(opponent_params)])
    if opponent_time_per_move is not None:
        argv.extend(["--opponent-time-per-move", str(opponent_time_per_move)])
    if opponent_depth_factor is not None:
        argv.extend(["--opponent-depth-factor", str(opponent_depth_factor)])
    if depth is not None:
        argv.extend(["--depth", str(depth)])
    elif time_per_move is not None:
        argv.extend(["--time-per-move", str(time_per_move)])
    elif clock is not None:
        argv.extend(["--clock", str(clock)])
    if increment is not None:
        argv.extend(["--increment", str(increment)])
    if opponent_clock is not None:
        argv.extend(["--opponent-clock", str(opponent_clock)])
    if opponent_increment is not None:
        argv.extend(["--opponent-increment", str(opponent_increment)])
    if moves_to_go is not None:
        argv.extend(["--moves-to-go", str(moves_to_go)])
    if not quiet:
        argv.append("--verbose")
    else:
        argv.append("--quiet")
    if disable_elo_store:
        argv.append("--no-elo-store")

    try:
        args = arena_runner.parse_args(argv)
        exit_code = arena_runner.run_batch(args)
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            payload = None
        if exit_code != 0:
            if payload is not None:
                sys.stderr.write(
                    "batch runner returned non-zero exit; using summary results anyway.\n"
                )
                return payload
            raise RuntimeError(
                f"batch runner failed with code {exit_code} and no summary available"
            )
        if payload is None:
            raise RuntimeError("batch runner succeeded but produced no summary")
        return payload
    finally:
        try:
            summary_path.unlink()
        except OSError:
            pass


def _run_external_shard(
    *,
    engine: Path,
    opponent: Path,
    candidate_params_data: Dict[str, Any],
    opponent_params: Optional[Path],
    games: int,
    depth: Optional[int],
    time_per_move: Optional[float],
    clock: Optional[float],
    increment: Optional[float],
    opponent_clock: Optional[float],
    opponent_increment: Optional[float],
    moves_to_go: Optional[int],
    opponent_time_per_move: Optional[float],
    opponent_depth_factor: Optional[float],
    concurrency: int,
    engine_label: str,
    opponent_label: str,
    quiet: bool,
) -> Dict[str, Any]:
    fd, tmp_path = tempfile.mkstemp(suffix="_cand.yaml")
    os.close(fd)
    cand_path = Path(tmp_path)
    try:
        _save_params(candidate_params_data, cand_path)
        return run_batch(
            engine=engine,
            opponent=opponent,
            engine_params=cand_path,
            opponent_params=opponent_params,
            engine_label=engine_label,
            opponent_label=opponent_label,
            games=games,
            depth=depth,
            time_per_move=time_per_move,
            clock=clock,
            increment=increment,
            opponent_clock=opponent_clock,
            opponent_increment=opponent_increment,
            moves_to_go=moves_to_go,
            opponent_time_per_move=opponent_time_per_move,
            opponent_depth_factor=opponent_depth_factor,
            concurrency=concurrency,
            quiet=quiet,
            disable_elo_store=True,
        )
    finally:
        try:
            cand_path.unlink()
        except OSError:
            pass


def _merge_external_payloads(payloads: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    first = next((payload for payload in payloads if payload), None)
    if first is None:
        return {}

    combined_summary: Dict[str, int] = {}
    total_games = 0
    total_completed = 0
    total_failures = 0
    total_timing = 0.0

    for payload in payloads:
        if not payload:
            continue
        total_games += int(payload.get("games", 0))
        total_completed += int(payload.get("completed", 0))
        total_failures += int(payload.get("failures", 0))
        total_timing += float(payload.get("timing_sec", 0.0))
        summary = payload.get("summary", {})
        for key, value in summary.items():
            combined_summary[key] = combined_summary.get(key, 0) + int(value)

    labels = first.get("labels", {}) or {}
    white_label = labels.get("white")
    black_label = labels.get("black")
    wins = int(combined_summary.get(white_label, 0)) if white_label else 0
    losses = int(combined_summary.get(black_label, 0)) if black_label else 0
    draws = int(combined_summary.get("draw", 0))

    elo_seed = first.get("elo", {}) or {}
    rating_path = Path(arena_runner.DEFAULT_ELO_STORE)
    if not rating_path.is_absolute():
        rating_path = Path(arena_runner.__file__).resolve().parent / rating_path
    rating_start = arena_runner.load_rating(
        rating_path, fallback=arena_runner.DEFAULT_ELO_START
    )
    opponent_rating = float(elo_seed.get("opponent", arena_runner.DEFAULT_ELO_OPPONENT))
    elo = arena_runner.compute_elo(
        rating=rating_start,
        opponent_rating=opponent_rating,
        wins=wins,
        losses=losses,
        draws=draws,
        k_factor=arena_runner.DEFAULT_ELO_K_FACTOR,
    )

    if elo.games > 0:
        try:
            arena_runner.store_rating(rating_path, elo.rating_after)
        except Exception:
            pass

    return {
        "games": total_games,
        "completed": total_completed,
        "failures": total_failures,
        "summary": combined_summary,
        "labels": labels,
        "timing_sec": total_timing,
        "parameters": first.get("parameters"),
        "elo": {
            "start": elo.rating_before,
            "opponent": elo.opponent_rating,
            "delta": elo.delta,
            "new": elo.rating_after,
            "expected_score": elo.expected_score,
            "actual_score": elo.actual_score,
            "games": elo.games,
            "wins": elo.wins,
            "losses": elo.losses,
            "draws": elo.draws,
        },
    }


def aggregate_two_sided(
    base_label: str, cand_label: str, s1: Dict[str, Any], s2: Dict[str, Any]
) -> Tuple[int, int, int]:
    summary1 = s1.get("summary", {})
    summary2 = s2.get("summary", {})
    wins = summary1.get(cand_label, 0) + summary2.get(cand_label, 0)
    losses = summary1.get(base_label, 0) + summary2.get(base_label, 0)
    draws = summary1.get("draw", 0) + summary2.get("draw", 0)
    return wins, losses, draws


def _stats_from_meta(meta: Dict[str, Any], confidence: float) -> Dict[str, Any]:
    wins = int(meta.get("wins", 0))
    losses = int(meta.get("losses", 0))
    draws = int(meta.get("draws", 0))
    stats = summarize_wld(wins, losses, draws, confidence=confidence)
    if "score_ci_low" in meta:
        stats["ci_low"] = float(meta.get("score_ci_low", stats["ci_low"]))
    if "score_ci_high" in meta:
        stats["ci_high"] = float(meta.get("score_ci_high", stats["ci_high"]))
    stats["confidence"] = float(meta.get("score_confidence", stats["confidence"]))
    stats["score"] = float(meta.get("score", stats["score"]))
    return stats

def _run_arena_shard(
    payload: Tuple[
        List[str],
        Optional[Dict[str, Any]],
        Dict[str, Any],
        Optional[Dict[str, Any]],
        int,
        int,
        int,
    ],
) -> Dict[str, int]:
    (
        start_fens,
        base_params,
        cand_params,
        coerce_template,
        depth,
        movetime_ms,
        max_plies,
    ) = payload
    try:
        import skaks_eval as _skaks_eval  # local import for multiprocessing
    except Exception as exc:  # pragma: no cover - surfaced in parent process
        raise RuntimeError(f"skaks_eval not available in worker: {exc}")
    coerced_base, coerced_cand = _prepare_arena_params(
        base_params, cand_params, coerce_template
    )
    res = _skaks_eval.arena(
        start_fens=start_fens,
        base_params=coerced_base,
        cand_params=coerced_cand,
        games=len(start_fens),
        depth=depth,
        movetime_ms=movetime_ms,
        max_plies=max_plies,
    )
    return {
        "wins": int(res.get("wins", 0)),
        "losses": int(res.get("losses", 0)),
        "draws": int(res.get("draws", 0)),
    }


def evaluate_candidate(
    *,
    engine: Path,
    baseline_params: Optional[Path],
    candidate_params: Path,
    baseline_data: Optional[Dict[str, Any]],
    candidate_data: Dict[str, Any],
    coerce_template: Optional[Dict[str, Any]] = None,
    games: int,
    depth: Optional[int],
    time_per_move: Optional[float],
    clock: Optional[float],
    increment: Optional[float],
    opponent_clock: Optional[float],
    opponent_increment: Optional[float],
    moves_to_go: Optional[int],
    concurrency: int,
    base_label: str,
    cand_label: str,
    quiet: bool,
    use_arena: bool,
    arena_workers: int,
    start_fens: Optional[List[str]] = None,
    external: bool = False,
    opponent: Optional[Path] = None,
    opponent_params: Optional[Path] = None,
    opponent_time_per_move: Optional[float] = None,
    opponent_depth_factor: Optional[float] = None,
    dask_client: Optional[Any] = None,
    dask_shard_hint: Optional[int] = None,
) -> Tuple[float, Tuple[int, int, int], float]:
    if use_arena:
        if skaks_eval is None:
            raise RuntimeError(
                "skaks_eval not available; install bindings or disable --use-arena-binding"
            )
        start_time = time.monotonic()
        movetime_ms = 0
        if time_per_move is not None:
            movetime_ms = int(time_per_move * 1000)
        elif clock is not None:
            raise ValueError(
                "Arena binding does not support clock controls; use depth or time-per-move"
            )
        fen_pool = list(start_fens) if start_fens else [START_FEN]
        fen_pool = [START_FEN if fen == "start" else fen for fen in fen_pool]
        if games > len(fen_pool):
            copies = (games + len(fen_pool) - 1) // len(fen_pool)
            fen_pool = (fen_pool * copies)[:games]
        else:
            fen_pool = fen_pool[:games]

        shard_target = (
            dask_shard_hint
            if dask_client is not None and dask_shard_hint and dask_shard_hint > 0
            else arena_workers
        )

        if dask_client is not None:
            target = (
                shard_target if shard_target and shard_target > 0 else len(fen_pool)
            )
            chunk = max(1, (len(fen_pool) + target - 1) // target)
            payloads = []
            for i in range(0, len(fen_pool), chunk):
                payloads.append(
                    (
                        fen_pool[i : i + chunk],
                        baseline_data,
                        candidate_data,
                        coerce_template,
                        depth or 0,
                        movetime_ms,
                        160,
                    )
                )
            futures = [
                dask_client.submit(_run_arena_shard, payload) for payload in payloads
            ]
            try:
                shard_results = dask_client.gather(futures)
            finally:
                for fut in futures:
                    try:
                        fut.release()
                    except Exception:
                        pass
            wins = sum(r["wins"] for r in shard_results)
            losses = sum(r["losses"] for r in shard_results)
            draws = sum(r["draws"] for r in shard_results)
        elif arena_workers <= 1:
            coerced_base, coerced_cand = _prepare_arena_params(
                baseline_data, candidate_data, coerce_template
            )
            res = skaks_eval.arena(
                start_fens=fen_pool,
                base_params=coerced_base,
                cand_params=coerced_cand,
                games=games,
                depth=depth or 0,
                movetime_ms=movetime_ms,
                max_plies=160,
            )
            wins = int(res.get("wins", 0))
            losses = int(res.get("losses", 0))
            draws = int(res.get("draws", 0))
        else:
            chunk = max(1, (len(fen_pool) + arena_workers - 1) // arena_workers)
            tasks = []
            for i in range(0, len(fen_pool), chunk):
                tasks.append(
                    (
                        fen_pool[i : i + chunk],
                        baseline_data,
                        candidate_data,
                        coerce_template,
                        depth or 0,
                        movetime_ms,
                        160,
                    )
                )
            with mp.Pool(processes=arena_workers) as pool:
                try:
                    shard_results = pool.map(_run_arena_shard, tasks)
                except KeyboardInterrupt:
                    pool.terminate()
                    pool.join()
                    raise
            wins = sum(r["wins"] for r in shard_results)
            losses = sum(r["losses"] for r in shard_results)
            draws = sum(r["draws"] for r in shard_results)

        total = wins + losses + draws
        score = (wins + 0.5 * draws) / total if total > 0 else 0.0
        wall = time.monotonic() - start_time
        return score, (wins, losses, draws), wall
    else:
        if external:
            if dask_client is not None and games > 0:
                shard_target = (
                    dask_shard_hint if dask_shard_hint and dask_shard_hint > 0 else None
                )
                if shard_target is None:
                    shard_target = concurrency if concurrency > 0 else 1
                shard_target = max(1, min(shard_target, games))
                chunk = max(1, (games + shard_target - 1) // shard_target)
                shard_concurrency = (
                    max(1, concurrency // shard_target) if concurrency > 0 else 1
                )
                futures = []
                for start in range(0, games, chunk):
                    shard_games = min(chunk, games - start)
                    futures.append(
                        dask_client.submit(
                            _run_external_shard,
                            engine=engine,
                            opponent=opponent,
                            candidate_params_data=candidate_data,
                            opponent_params=opponent_params,
                            games=shard_games,
                            depth=depth,
                            time_per_move=time_per_move,
                            clock=clock,
                            increment=increment,
                            opponent_clock=opponent_clock,
                            opponent_increment=opponent_increment,
                            moves_to_go=moves_to_go,
                            opponent_time_per_move=opponent_time_per_move,
                            opponent_depth_factor=opponent_depth_factor,
                            concurrency=shard_concurrency,
                            engine_label=base_label,
                            opponent_label=cand_label,
                            quiet=quiet,
                            pure=False,
                        )
                    )
                try:
                    shard_payloads = dask_client.gather(futures)
                finally:
                    for fut in futures:
                        try:
                            fut.release()
                        except Exception:
                            pass
                s1 = _merge_external_payloads(shard_payloads)
                if not s1:
                    raise RuntimeError("external shards returned no summary payloads")
            else:
                s1 = run_batch(
                    engine=engine,
                    opponent=opponent,
                    engine_params=candidate_params,
                    opponent_params=opponent_params,
                    engine_label=base_label,
                    opponent_label=cand_label,
                    games=games,
                    depth=depth,
                    time_per_move=time_per_move,
                    clock=clock,
                    increment=increment,
                    opponent_clock=opponent_clock,
                    opponent_increment=opponent_increment,
                    moves_to_go=moves_to_go,
                    opponent_time_per_move=opponent_time_per_move,
                    opponent_depth_factor=opponent_depth_factor,
                    concurrency=concurrency,
                    quiet=quiet,
                )
            s2 = {"summary": {}}
        else:
            half_games = max(2, games // 2)
            s1 = run_batch(
                engine=engine,
                opponent=engine,
                engine_params=baseline_params,
                opponent_params=candidate_params,
                engine_label=base_label,
                opponent_label=cand_label,
                games=half_games,
                depth=depth,
                time_per_move=time_per_move,
                clock=clock,
                increment=increment,
                opponent_clock=opponent_clock,
                opponent_increment=opponent_increment,
                moves_to_go=moves_to_go,
                concurrency=concurrency,
                opponent_depth_factor=None,
                quiet=quiet,
            )
            s2 = run_batch(
                engine=engine,
                opponent=engine,
                engine_params=candidate_params,
                opponent_params=baseline_params,
                engine_label=cand_label,
                opponent_label=base_label,
                games=half_games,
                depth=depth,
                time_per_move=time_per_move,
                clock=clock,
                increment=increment,
                opponent_clock=opponent_clock,
                opponent_increment=opponent_increment,
                moves_to_go=moves_to_go,
                concurrency=concurrency,
                opponent_depth_factor=None,
                quiet=quiet,
            )
        if external:
            # When running against an external opponent the batch runner
            # returns a single summary where the `white`/`black` labels
            # correspond to `engine_label` and `opponent_label`. In this
            # mode `engine` is the candidate (Skaks) and `opponent` is the
            # external engine. Compute wins/losses directly from that
            # summary using the provided labels to avoid swapping sides.
            summary = s1.get("summary", {})
            wins = int(summary.get(base_label, 0))
            losses = int(summary.get(cand_label, 0))
            draws = int(summary.get("draw", 0))
            total = wins + losses + draws
            score = (wins + 0.5 * draws) / total if total > 0 else 0.0
            wall = s1.get("timing_sec", 0.0)
            return score, (wins, losses, draws), wall
        else:
            wins, losses, draws = aggregate_two_sided(base_label, cand_label, s1, s2)
            total = wins + losses + draws
            score = (wins + 0.5 * draws) / total if total > 0 else 0.0
            wall = s1.get("timing_sec", 0.0) + s2.get("timing_sec", 0.0)
            return score, (wins, losses, draws), wall


def evaluate_candidate_repeats(
    *,
    repeats: int,
    progress_cb: Optional[Callable[[int, int, float], None]] = None,
    quiet: bool = True,
    use_arena: bool,
    start_fens: Optional[List[str]],
    coerce_template: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[float, Tuple[int, int, int], float]:
    total_score = 0.0
    total_wins = total_losses = total_draws = 0
    total_time = 0.0
    start = time.monotonic()
    for rep_idx in range(repeats):
        rep_fens: Optional[List[str]] = None
        if start_fens:
            # Offset into the pool so each repeat can use a different slice.
            offset = rep_idx * kwargs.get("games", 0)
            if offset < len(start_fens):
                rep_fens = start_fens[offset : offset + kwargs.get("games", 0)]
            if not rep_fens:
                rep_fens = start_fens[: kwargs.get("games", 0)]
        if progress_cb:
            progress_cb(rep_idx, repeats, time.monotonic() - start)
        score, (w, losses, d), wall = evaluate_candidate(
            quiet=quiet,
            use_arena=use_arena,
            start_fens=rep_fens,
            coerce_template=coerce_template,
            **kwargs,
        )
        total_score += score
        total_wins += w
        total_losses += losses
        total_draws += d
        total_time += wall
        if progress_cb:
            progress_cb(rep_idx + 1, repeats, time.monotonic() - start)
    avg_score = total_score / repeats if repeats > 0 else 0.0
    return avg_score, (total_wins, total_losses, total_draws), total_time


def optimize_loop(args: argparse.Namespace) -> None:
    engine = _resolve_engine(args.engine)
    baseline_params = (
        Path(args.baseline_params).resolve() if args.baseline_params else None
    )
    current_params = Path(args.start_params).resolve() if args.start_params else None
    if current_params is None and baseline_params is None:
        raise SystemExit("Provide --start-params or --baseline-params")
    if current_params is None:
        current_params = baseline_params
    assert current_params is not None

    confidence_level = float(getattr(args, "score_confidence", 0.95))
    if not 0.0 < confidence_level < 1.0:
        raise SystemExit("--score-confidence must be between 0 and 1")
    args.score_confidence = confidence_level

    if args.external_opponent:
        external = True
        opponent = _resolve_engine(args.opponent)
        opponent_params = None
        opponent_time = args.opponent_time_per_move
        base_label = "skaks"
        cand_label = args.opponent
        # Use a lower neutral baseline (0.25) so early improvements are
        # easier to accept when starting from un-tuned parameters.
        base_score = 0.25
        baseline_data = None
        baseline_params = None
    else:
        external = False
        opponent = engine
        opponent_params = baseline_params
        opponent_time = None
        base_label = "baseline"
        cand_label = "candidate"
        # Use a lower neutral baseline score of 0.25 for internal self-play
        # so that early candidates that show modest improvement are accepted
        # and the optimizer can climb from a weak starting point.
        base_score = 0.25

    base_data = _load_params(current_params)
    best_data = base_data
    best_score = base_score
    best_ci_low = best_score
    best_ci_high = best_score
    best_stats: Dict[str, Any] = {
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "games": 0,
        "ci_low": best_ci_low,
        "ci_high": best_ci_high,
        "confidence": confidence_level,
    }

    # The arena binding runs in-process and cannot evaluate an external
    # opponent binary. Running with both flags would silently run an
    # in-process arena (skaks vs skaks) while the user expects an
    # external match. Fail fast with a clear message to avoid
    # misleading optimizer results.
    if args.external_opponent and args.use_arena_binding:
        raise SystemExit(
            "--external-opponent cannot be combined with --use-arena-binding;"
            " disable arena binding or remove --external-opponent"
        )

    # If running against an external opponent, compute the baseline's
    # actual score up-front so `best_score` reflects reality instead of the
    # arbitrary default (0.5). This prevents the optimizer from reporting
    # a misleading best score when no candidate outperforms the baseline.
    if args.external_opponent and args.baseline_params:
        try:
            baseline_path = Path(args.baseline_params).resolve()
            baseline_data = _load_params(baseline_path)
            _live_line("Evaluating baseline performance against external opponent...")
            score, (w, losses, d), wall = evaluate_candidate(
                engine=engine,
                baseline_params=None,
                candidate_params=baseline_path,
                baseline_data=None,
                candidate_data=baseline_data,
                coerce_template=baseline_data,
                games=args.games,
                depth=args.depth,
                time_per_move=args.time_per_move,
                clock=args.clock,
                increment=args.increment,
                opponent_clock=args.opponent_clock,
                opponent_increment=args.opponent_increment,
                moves_to_go=args.moves_to_go,
                concurrency=args.concurrency,
                base_label="skaks",
                cand_label=args.opponent,
                quiet=True,
                use_arena=args.use_arena_binding,
                arena_workers=args.arena_workers,
                start_fens=None,
                external=True,
                opponent=_resolve_engine(args.opponent),
                opponent_params=None,
                opponent_time_per_move=args.opponent_time_per_move,
            )
            base_stats = summarize_wld(
                w,
                losses,
                d,
                confidence=confidence_level,
            )
            # If the user provided a `--start-params` (current_params), prefer
            # keeping that as the working best_data. Only use the evaluated
            # baseline score to seed `best_score` if we don't already have a
            # start params matching the baseline. Otherwise, make sure the
            # numeric `best_score` reflects the better of the two so the
            # optimizer doesn't mistakenly treat the baseline as better and
            # overwrite the provided start params with a worse value.
            try:
                baseline_path = baseline_path.resolve()
            except Exception:
                pass
            if current_params is None or (baseline_path == current_params):
                best_score = score
                best_data = baseline_data
                best_ci_low = base_stats["ci_low"]
                best_ci_high = base_stats["ci_high"]
                best_stats = base_stats
            else:
                # Keep the start params but ensure best_score is at least
                # the evaluated baseline score so comparisons are meaningful.
                best_score = max(best_score, score)
            _final_line(f"Baseline score={score:.3f} WLD={w}/{losses}/{d}")
        except Exception:
            # If baseline evaluation fails we keep the default base_score
            pass

    # Support the convenience alias --weights-only
    if getattr(args, "weights_only", False):
        args.phase_weights_only = True

    include_prefixes = args.include_prefix
    exclude_prefixes = list(args.exclude_prefix) if args.exclude_prefix else []
    if getattr(args, "phase_weights_only", False):
        include_prefixes = [
            "evaluation.phase_weights_mg",
            "evaluation.phase_weights_eg",
        ]
    elif getattr(args, "param_set", None):
        if args.param_set != "full":
            include_prefixes = param_set_prefixes(args.param_set)
        else:
            include_prefixes = include_prefixes or param_set_prefixes("full")
    if getattr(args, "skip_pst", False):
        exclude_prefixes.append("evaluation.pst_")

    if args.output:
        best_path = Path(args.output).resolve()
    else:
        src = Path(args.start_params or args.baseline_params or "best_params")
        best_path = src.with_name(f"{src.stem}_optimized.yaml").resolve()

    best_path.parent.mkdir(parents=True, exist_ok=True)
    # Persist an initial best params file if none exists. We also maintain
    # a small JSON meta file alongside the YAML to record the best score so
    # that separate runs and replicates won't be overwritten by worse
    # candidates.
    meta_path = best_path.with_suffix(best_path.suffix + ".best.json")
    # Load on-disk best meta if present to initialize `best_score` and
    # `best_data` from previous runs.
    if best_path.exists() and meta_path.exists():
        try:
            import json as _json

            meta = _json.loads(meta_path.read_text(encoding="utf-8"))
            disk_stats = _stats_from_meta(meta, confidence_level)
            disk_score = float(disk_stats.get("score", best_score))
            if disk_score > best_score:
                # Replace in-memory best with on-disk superior result
                best_score = disk_score
                best_ci_low = float(disk_stats.get("ci_low", best_ci_low))
                best_ci_high = float(disk_stats.get("ci_high", best_ci_high))
                best_stats = dict(disk_stats)
                try:
                    best_data = _load_params(best_path)
                except Exception:
                    pass
        except Exception:
            # If the meta is corrupted ignore and continue with in-memory
            # defaults.
            pass
    else:
        # Ensure initial files exist for tooling that expects them.
        _save_params(best_data, best_path)
        try:
            import json as _json

            meta = {
                "score": float(best_score),
                "timestamp": time.time(),
                "wins": int(best_stats.get("wins", 0)),
                "losses": int(best_stats.get("losses", 0)),
                "draws": int(best_stats.get("draws", 0)),
                "games": int(best_stats.get("games", 0)),
                "score_ci_low": float(best_ci_low),
                "score_ci_high": float(best_ci_high),
                "score_confidence": float(best_stats.get("confidence", confidence_level)),
            }
            meta_path.write_text(_json.dumps(meta), encoding="utf-8")
        except Exception:
            pass

    beam: List[Tuple[float, Dict[str, Any]]] = [(best_score, best_data)]

    spin_cycle = itertools.cycle(SPINNER_FRAMES)
    piece_cycle = itertools.cycle(CHESS_SWARM)
    move_cycle = itertools.cycle(range(16))
    color_cycle = itertools.cycle(PALETTE)
    quiet_child = not args.child_output
    baseline_data = _load_params(baseline_params) if baseline_params else None

    start_fens: Optional[List[str]] = None
    if args.use_arena_binding and args.arena_pgn:
        try:
            import chess.pgn  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise SystemExit(f"python-chess is required for --arena-pgn: {exc}")
        target_fens = args.games * max(1, args.repeats)
        start_fens = []
        rng = random.Random(args.arena_seed)
        with Path(args.arena_pgn).open("r", encoding="utf-8") as fh:
            while len(start_fens) < target_fens:
                game = chess.pgn.read_game(fh)
                if game is None:
                    break
                board = game.board()
                moves = list(game.mainline_moves())
                if not moves:
                    continue
                target = (
                    rng.randint(args.arena_min_ply, args.arena_max_ply)
                    if args.arena_max_ply > 0
                    else args.arena_min_ply
                )
                target = max(1, target)
                applied = 0
                for mv in moves:
                    board.push(mv)
                    applied += 1
                    if applied >= target:
                        break
                start_fens.append(board.fen())
        if start_fens and len(start_fens) < target_fens:
            while len(start_fens) < target_fens:
                start_fens.append(random.choice(start_fens))
        if not start_fens:
            raise SystemExit("Failed to sample start positions from PGN")

    dask_cm = dask_client_from_args(args)
    dask_client, dask_shard_hint = dask_cm.__enter__()
    try:
        for step in range(1, args.iterations + 1):
            candidates: List[
                Tuple[float, Dict[str, Any], Tuple[int, int, int], float]
            ] = []

            if args.strategy == "beam":
                parents = [data for (_, data) in beam] or [best_data]
                for i in range(args.beam_size):
                    parent = parents[i % len(parents)]
                    cand_data = perturb_params(
                        parent,
                        args.noise,
                        include_prefixes=include_prefixes,
                        exclude_prefixes=exclude_prefixes,
                    )
                    cand_path = Path(tempfile.mkstemp(suffix="_cand.yaml")[1])
                    _save_params(cand_data, cand_path)
                    try:
                        color = next(color_cycle)

                        def progress_cb(
                            rep_idx: int, repeats: int, elapsed: float
                        ) -> None:
                            spinner = next(spin_cycle)
                            piece = next(piece_cycle)
                            offset = next(move_cycle)
                            pad_left = " " * offset
                            pad_right = " " * (15 - offset)
                            track = f"{pad_left}{piece}{pad_right}"
                            lead = _color(f"{spinner} {track}", color)
                            display_rep = min(rep_idx + 1, repeats)
                            line = (
                                f"{lead} iter {_color(str(step), color)}/{args.iterations} "
                                f"cand {_color(str(i + 1), color)}/{args.beam_size} "
                                f"repeat {_color(str(display_rep), color)}/{repeats} "
                                f"t={elapsed:.1f}s"
                            )
                            _live_line(line)

                        # Emit an explicit batch start line so logs show when a
                        # candidate's games begin (useful for long-running runs).
                        _final_line(
                            f"[BATCH START] iter={step} cand={i + 1}/{args.beam_size} games={args.games} depth={args.depth} external={external} opponent={getattr(opponent, 'name', str(opponent))}"
                        )
                        sys.stdout.flush()

                        score, (w, losses, d), wall = evaluate_candidate_repeats(
                            repeats=args.repeats,
                            progress_cb=progress_cb,
                            quiet=quiet_child,
                            use_arena=args.use_arena_binding,
                            start_fens=start_fens,
                            coerce_template=baseline_data
                            if baseline_data is not None
                            else parent,
                            engine=engine,
                            baseline_params=baseline_params,
                            candidate_params=cand_path,
                            baseline_data=baseline_data,
                            candidate_data=cand_data,
                            games=args.games,
                            depth=args.depth,
                            time_per_move=args.time_per_move,
                            clock=args.clock,
                            increment=args.increment,
                            opponent_clock=args.opponent_clock,
                            opponent_increment=args.opponent_increment,
                            moves_to_go=args.moves_to_go,
                            concurrency=args.concurrency,
                            arena_workers=args.arena_workers,
                            base_label="base",
                            cand_label="cand",
                            external=external,
                            opponent=opponent,
                            opponent_params=opponent_params,
                            opponent_time_per_move=opponent_time,
                            opponent_depth_factor=args.opponent_depth_factor
                            if hasattr(args, "opponent_depth_factor")
                            else None,
                            dask_client=dask_client,
                            dask_shard_hint=dask_shard_hint,
                        )
                        stats = summarize_wld(
                            w,
                            losses,
                            d,
                            confidence=confidence_level,
                        )
                        ci_pct = int(round(stats.get("confidence", 0.0) * 100))
                        if stats.get("confidence", 0.0) > 0:
                            score_block = (
                                f"{_color(f'{score:.3f}', color)}+/-{stats['margin']:.3f} "
                                f"({ci_pct}% CI {stats['ci_low']:.3f}-{stats['ci_high']:.3f})"
                            )
                        else:
                            score_block = _color(f"{score:.3f}", color)
                        _final_line(
                            f"{_color('✓', 'green')} iter={step} cand={i + 1}/{args.beam_size} "
                            f"score={score_block} "
                            f"WLD={w}/{losses}/{d} elo~{score_to_elo(score):+.1f} time={wall:.1f}s"
                        )
                        candidates.append((score, cand_data, stats, wall))
                    finally:
                        try:
                            cand_path.unlink()
                        except OSError:
                            pass

                candidates.sort(key=lambda x: x[0], reverse=True)
                beam = [
                    (score, data) for score, data, _, _ in candidates[: args.beam_size]
                ]
                top_score, top_data, top_stats, _ = candidates[0]
            else:  # CMA-like strategy
                paths, mean_vec, is_int = _flatten_params(
                    best_data,
                    include_prefixes=include_prefixes,
                    exclude_prefixes=exclude_prefixes,
                )
                popsize = (
                    args.cma_popsize
                    if args.cma_popsize is not None
                    else max(4, int(4 + 3 * math.log(len(mean_vec) + 1)))
                )
                mu = max(1, popsize // 2)
                weights = [math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)]
                weight_sum = sum(weights)
                weights = [w / weight_sum for w in weights]

                sigma = args.cma_sigma if args.cma_sigma is not None else args.noise
                sigma = max(1e-4, sigma)

                template_data = json.loads(json.dumps(best_data))
                for i in range(popsize):
                    vec = []
                    for val in mean_vec:
                        scale = sigma * max(abs(val), 1.0)
                        vec.append(val + random.gauss(0.0, scale))
                    cand_data = _vector_to_params(template_data, paths, vec, is_int)
                    cand_path = Path(tempfile.mkstemp(suffix="_cand.yaml")[1])
                    _save_params(cand_data, cand_path)
                    try:
                        color = next(color_cycle)

                        def progress_cb(
                            rep_idx: int, repeats: int, elapsed: float
                        ) -> None:
                            spinner = next(spin_cycle)
                            piece = next(piece_cycle)
                            offset = next(move_cycle)
                            pad_left = " " * offset
                            pad_right = " " * (15 - offset)
                            track = f"{pad_left}{piece}{pad_right}"
                            lead = _color(f"{spinner} {track}", color)
                            display_rep = min(rep_idx + 1, repeats)
                            line = (
                                f"{lead} iter {_color(str(step), color)}/{args.iterations} "
                                f"cand {_color(str(i + 1), color)}/{popsize} "
                                f"repeat {_color(str(display_rep), color)}/{repeats} "
                                f"t={elapsed:.1f}s"
                            )
                            _live_line(line)

                        _final_line(
                            f"[BATCH START] iter={step} cand={i + 1}/{popsize} games={args.games} depth={args.depth} external={external} opponent={getattr(opponent, 'name', str(opponent))}"
                        )
                        sys.stdout.flush()

                        score, (w, losses, d), wall = evaluate_candidate_repeats(
                            repeats=args.repeats,
                            progress_cb=progress_cb,
                            quiet=quiet_child,
                            use_arena=args.use_arena_binding,
                            start_fens=start_fens,
                            coerce_template=baseline_data
                            if baseline_data is not None
                            else base_data,
                            engine=engine,
                            baseline_params=baseline_params,
                            candidate_params=cand_path,
                            baseline_data=baseline_data,
                            candidate_data=cand_data,
                            games=args.games,
                            depth=args.depth,
                            time_per_move=args.time_per_move,
                            clock=args.clock,
                            increment=args.increment,
                            opponent_clock=args.opponent_clock,
                            opponent_increment=args.opponent_increment,
                            moves_to_go=args.moves_to_go,
                            concurrency=args.concurrency,
                            arena_workers=args.arena_workers,
                            base_label="base",
                            cand_label="cand",
                            external=external,
                            opponent=opponent,
                            opponent_params=opponent_params,
                            opponent_time_per_move=opponent_time,
                            opponent_depth_factor=args.opponent_depth_factor
                            if hasattr(args, "opponent_depth_factor")
                            else None,
                            dask_client=dask_client,
                            dask_shard_hint=dask_shard_hint,
                        )
                        stats = summarize_wld(
                            w,
                            losses,
                            d,
                            confidence=confidence_level,
                        )
                        ci_pct = int(round(stats.get("confidence", 0.0) * 100))
                        if stats.get("confidence", 0.0) > 0:
                            score_block = (
                                f"{_color(f'{score:.3f}', color)}+/-{stats['margin']:.3f} "
                                f"({ci_pct}% CI {stats['ci_low']:.3f}-{stats['ci_high']:.3f})"
                            )
                        else:
                            score_block = _color(f"{score:.3f}", color)
                        _final_line(
                            f"{_color('✓', 'green')} iter={step} cand={i + 1}/{popsize} "
                            f"score={score_block} "
                            f"WLD={w}/{losses}/{d} elo~{score_to_elo(score):+.1f} time={wall:.1f}s"
                        )
                        candidates.append((score, cand_data, stats, wall))
                    finally:
                        try:
                            cand_path.unlink()
                        except OSError:
                            pass

                candidates.sort(key=lambda x: x[0], reverse=True)
                top_score, top_data, top_stats, _ = candidates[0]

                # Recombine top mu candidates to update mean_vec
                new_mean = [0.0 for _ in mean_vec]
                for rank in range(min(mu, len(candidates))):
                    vec_paths, vec_vals, _ = _flatten_params(
                        candidates[rank][1],
                        include_prefixes=include_prefixes,
                        exclude_prefixes=exclude_prefixes,
                    )
                    if vec_paths != paths:
                        continue
                    for idx, val in enumerate(vec_vals):
                        new_mean[idx] += weights[rank] * val
                mean_vec = new_mean
                best_data = _vector_to_params(template_data, paths, mean_vec, is_int)
                beam = [(top_score, best_data)]

            # Only persist when the candidate is strictly better than the
            # current on-disk best. We read the meta file first in case another
            # replicate already wrote an improved best params during this run.
            try:
                import json as _json

                disk_meta = None
                if meta_path.exists():
                    try:
                        disk_meta = _json.loads(meta_path.read_text(encoding="utf-8"))
                    except Exception:
                        disk_meta = None
                disk_score = (
                    float(disk_meta.get("score")) if disk_meta else best_score
                )
                disk_ci_high = (
                    float(disk_meta.get("score_ci_high"))
                    if disk_meta and "score_ci_high" in disk_meta
                    else best_ci_high
                )
            except Exception:
                disk_score = best_score
                disk_ci_high = best_ci_high

            # Apply a per-iteration relaxation to the on-disk score if requested.
            # This makes it gradually easier to overwrite a stubborn best.
            try:
                decay = float(getattr(args, "baseline_decay", 0.0))
            except Exception:
                decay = 0.0
            effective_disk_score = disk_score - (decay * float(step))
            effective_ci_high = max(0.0, min(1.0, disk_ci_high - (decay * float(step))))
            best_ci_high = disk_ci_high

            # Acceptance policy: prefer strictly greater score, but allow
            # forced/seed accepts or raising the floor when applicable.
            accept_candidate = False
            confidence_guard = bool(getattr(args, "require_score_confidence", False))
            confidence_ok = True
            if confidence_guard:
                candidate_ci_low = float(top_stats.get("ci_low", top_score))
                confidence_ok = candidate_ci_low > effective_ci_high
            if top_score > effective_disk_score and confidence_ok:
                accept_candidate = True
            else:
                # Force-accept first N replacements to seed the leaderboard
                force_n = int(getattr(args, "force_accept_first", 0))
                if force_n > 0:
                    if not hasattr(args, "_accepted_replacements"):
                        setattr(args, "_accepted_replacements", 0)
                    if getattr(args, "_accepted_replacements") < force_n:
                        accept_candidate = True
                        setattr(
                            args,
                            "_accepted_replacements",
                            getattr(args, "_accepted_replacements") + 1,
                        )
                # If current on-disk best is below min_score, allow replacing
                # when candidate reaches the min_score floor
                if not accept_candidate:
                    min_floor = float(getattr(args, "min_score", 0.0))
                    try:
                        if (
                            float(top_score) >= min_floor
                            and float(effective_disk_score) < min_floor
                        ):
                            accept_candidate = True
                    except Exception:
                        pass

            if accept_candidate:
                best_score = top_score
                best_data = top_data
                best_ci_low = float(top_stats.get("ci_low", best_score))
                best_ci_high = float(top_stats.get("ci_high", best_score))
                best_stats = dict(top_stats)
                # Atomic write: write to temp then move into place.
                tmp = best_path.with_suffix(best_path.suffix + ".tmp")
                try:
                    _save_params(best_data, tmp)
                    tmp.replace(best_path)
                    meta = {
                        "score": float(best_score),
                        "timestamp": time.time(),
                        "wins": int(top_stats.get("wins", 0)),
                        "losses": int(top_stats.get("losses", 0)),
                        "draws": int(top_stats.get("draws", 0)),
                        "games": int(top_stats.get("games", 0)),
                        "score_ci_low": best_ci_low,
                        "score_ci_high": best_ci_high,
                        "score_confidence": float(
                            top_stats.get(
                                "confidence", confidence_level
                            )
                        ),
                    }
                    meta_path.write_text(_json.dumps(meta), encoding="utf-8")
                    _final_line(
                        f"{_color('⚑', 'yellow')} new best score={best_score:.3f} saved to {best_path}"
                    )
                    # Also emit a greppable log line
                    _final_line(f"[NEW BEST] score={best_score:.3f} path={best_path}")
                    # When a new best is accepted, make it the baseline for
                    # subsequent candidate evaluations in this run (so future
                    # candidates are tested against the best-known params).
                    if not getattr(args, "external_opponent", False):
                        try:
                            baseline_params = best_path
                            baseline_data = best_data
                        except Exception:
                            pass
                finally:
                    try:
                        if tmp.exists():
                            tmp.unlink()
                    except Exception:
                        pass

    finally:
        exc_type, exc_value, exc_tb = sys.exc_info()
        suppressed = dask_cm.__exit__(exc_type, exc_value, exc_tb)
        if suppressed and exc_type is not None:
            return
    # Read on-disk meta if available so final report reflects global best
    try:
        import json as _json

        if meta_path.exists():
            meta = _json.loads(meta_path.read_text(encoding="utf-8"))
            stats = _stats_from_meta(meta, confidence_level)
            disk_score = float(stats.get("score", best_score))
            ci_low = float(stats.get("ci_low", disk_score))
            ci_high = float(stats.get("ci_high", disk_score))
            wins = int(stats.get("wins", 0))
            losses = int(stats.get("losses", 0))
            draws = int(stats.get("draws", 0))
            games = int(stats.get("games", wins + losses + draws))
            conf_pct = float(stats.get("confidence", confidence_level) * 100.0)
            _final_line(
                f"{_color('🏁', 'bright_white')} Best score={disk_score:.3f} "
                f"(CI {ci_low:.3f}-{ci_high:.3f} @ {conf_pct:.1f}%), "
                f"W/L/D={wins}/{losses}/{draws} games={games}, params at {best_path}"
            )
        else:
            _final_line(
                f"{_color('🏁', 'bright_white')} Best score={best_score:.3f}, params at {best_path}"
            )
    except Exception:
        _final_line(
            f"{_color('🏁', 'bright_white')} Best score={best_score:.3f}, params at {best_path}"
        )
    if getattr(args, "baseline_decay", 0.0):
        _final_line(f"(baseline-decay={args.baseline_decay})")
    if args.output:
        out_path = Path(args.output).resolve()
        _save_params(best_data, out_path)
        _final_line(f"{_color('→', 'cyan')} Wrote best params to {out_path}")


def configure_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--engine", default="skaks", help="Path to engine binary")
    parser.add_argument(
        "--opponent", default="stockfish", help="Opponent engine binary"
    )
    parser.add_argument(
        "--external-opponent",
        action="store_true",
        help="Opponent is external engine, not self-play",
    )
    parser.add_argument("--baseline-params", help="YAML params for baseline (opponent)")
    parser.add_argument(
        "--start-params",
        help=(
            "Initial YAML params for candidate (defaults to baseline). "
            "If --output already exists (with its .best.json), that saved best is loaded instead."
        ),
    )
    parser.add_argument(
        "--output",
        help="Where to write best params (default: <start/baseline>_optimized.yaml)",
    )
    parser.add_argument(
        "--child-output",
        action="store_true",
        help="Show child match output (otherwise suppressed for clean animation)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=40,
        help="Total games per candidate (split by color)",
    )
    parser.add_argument(
        "--iterations", type=int, default=5, help="Number of candidate evaluations"
    )
    parser.add_argument(
        "--baseline-decay",
        type=float,
        default=0.0,
        help=(
            "Per-iteration score relaxation applied to the stored best score. "
            "A small positive value (e.g. 0.005) will lower the effective "
            "on-disk best score each iteration, making acceptance of new "
            "candidates gradually easier."
        ),
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.05,
        help="Lognormal stddev for multiplicative noise",
    )
    parser.add_argument("--depth", type=int, help="Depth per move")
    parser.add_argument("--time-per-move", type=float, help="Seconds per move")
    parser.add_argument("--clock", type=float, help="Clock time seconds")
    parser.add_argument(
        "--increment", type=float, help="Increment seconds (requires --clock)"
    )
    parser.add_argument(
        "--opponent-time-per-move", type=float, help="Opponent seconds per move"
    )
    parser.add_argument(
        "--opponent-clock", type=float, help="Opponent clock time seconds"
    )
    parser.add_argument(
        "--opponent-increment",
        type=float,
        help="Opponent increment seconds (requires --opponent-clock)",
    )
    parser.add_argument(
        "--moves-to-go",
        type=int,
        help="Approximate moves remaining to next time control",
    )
    parser.add_argument(
        "--opponent-depth-factor",
        type=float,
        help="Scale factor applied to depth for the opponent when running external opponent (e.g. 0.6)",
    )
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent games")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="How many repeated evaluations per candidate (averaged)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=1,
        help="How many top candidates to keep each iteration",
    )
    parser.add_argument(
        "--use-arena-binding",
        action="store_true",
        help="Use internal arena binding (skaks_eval.arena) instead of batch runner",
    )
    parser.add_argument(
        "--include-prefix",
        action="append",
        help="Only tune params whose dotted path starts with this prefix (repeatable)",
    )
    parser.add_argument(
        "--exclude-prefix",
        action="append",
        help="Skip params whose dotted path starts with this prefix (repeatable)",
    )
    parser.add_argument(
        "--phase-weights-only",
        action="store_true",
        help="Only tune evaluation.phase_weights_mg/eg arrays",
    )
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="Alias for --phase-weights-only (tune only phase weight arrays)",
    )
    parser.add_argument(
        "--param-set",
        choices=["full", "phase", "offense", "defense", "pst"],
        default="full",
        help=(
            "Select parameter subset to optimize: full (all evaluation params), "
            "phase (phase weights only), offense/defense (subset of eval terms), "
            "pst (piece-square tables only)"
        ),
    )
    parser.add_argument(
        "--skip-pst",
        action="store_true",
        help="Exclude PST tables from optimization",
    )
    parser.add_argument(
        "--strategy",
        choices=["beam", "cma"],
        default="beam",
        help="Search strategy: beam (default) or CMA-like",
    )
    parser.add_argument(
        "--arena-pgn",
        help="Optional PGN to sample start positions from (requires python-chess)",
    )
    parser.add_argument(
        "--arena-min-ply",
        type=int,
        default=6,
        help="Minimum ply to sample from PGN (arena binding mode)",
    )
    parser.add_argument(
        "--arena-max-ply",
        type=int,
        default=60,
        help="Maximum ply to sample from PGN (0 = no cap)",
    )
    parser.add_argument(
        "--arena-seed",
        type=int,
        default=0,
        help="RNG seed for PGN sampling (arena binding mode)",
    )
    parser.add_argument(
        "--arena-workers",
        type=int,
        default=1,
        help="Parallel arena shards (arena binding mode)",
    )
    parser.add_argument(
        "--dask-scheduler",
        help="Existing Dask scheduler address (e.g. tcp://scheduler:8786)",
    )
    parser.add_argument(
        "--dask-shards",
        type=int,
        help="Override number of arena shards submitted to Dask",
    )
    parser.add_argument(
        "--dask-workers",
        type=int,
        help="Spawn a local Dask cluster with this many workers",
    )
    parser.add_argument(
        "--dask-threads",
        type=int,
        help="Threads per local Dask worker (default: 1)",
    )
    parser.add_argument(
        "--dask-jobqueue",
        action="store_true",
        help="Launch a SLURMCluster via dask-jobqueue for arena shards",
    )
    parser.add_argument(
        "--dask-jobqueue-config",
        help="Optional YAML/JSON file with kwargs for SLURMCluster(...)",
    )
    parser.add_argument(
        "--dask-jobqueue-jobs",
        type=int,
        help="Number of SLURM jobs/nodes to request when scaling the cluster",
    )
    parser.add_argument(
        "--dask-jobqueue-adapt-min",
        type=int,
        help="Minimum jobs when using adaptive scaling with dask-jobqueue",
    )
    parser.add_argument(
        "--dask-jobqueue-adapt-max",
        type=int,
        help="Maximum jobs when using adaptive scaling with dask-jobqueue",
    )
    parser.add_argument(
        "--cma-popsize",
        type=int,
        help="Population size for CMA-like strategy (default ~4+3*log(n))",
    )
    parser.add_argument(
        "--cma-sigma",
        type=float,
        help="Step scale for CMA-like strategy (default uses --noise)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum score floor to allow accepting a candidate when current best is below this floor",
    )
    parser.add_argument(
        "--force-accept-first",
        type=int,
        default=0,
        help="Force-accept the first N candidate replacements (useful to seed a weak leaderboard)",
    )
    parser.add_argument(
        "--score-confidence",
        type=float,
        default=0.95,
        help="Confidence level used for score/Elo intervals (0-1)",
    )
    parser.add_argument(
        "--require-score-confidence",
        action="store_true",
        help="Require the candidate's lower confidence bound to exceed the stored best upper bound",
    )
    return parser


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = configure_parser(
        argparse.ArgumentParser(
            description=(
                "Self-play parameter optimizer (beam or CMA). "
                "If --output already exists, the saved best params/metadata"
                " are loaded so runs resume from the previous best instead"
                " of the --start-params file."
            )
        )
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        optimize_loop(args)
    except KeyboardInterrupt:
        _final_line("Interrupted. Exiting cleanly.")


if __name__ == "__main__":
    main()
