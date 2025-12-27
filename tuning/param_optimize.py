#!/usr/bin/env python3
"""Simple self-play optimizer loop using the existing batch match runner.

This script perturbs numeric YAML params, runs head-to-head matches against a
baseline (with both color orders), and keeps the best candidate by score.
It relies on validation_moves/batch_stockfish_matches.py with --summary-json
for machine-readable results.
"""

import argparse
import itertools
import json
import math
import multiprocessing as mp
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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

BATCH_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "validation_moves"
    / "batch_stockfish_matches.py"
)

SPINNER_FRAMES = ["|", "/", "-", "\\"]
# Unicode chess glyphs for livelier progress (requested by user)
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
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=True)


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
    include_prefixes: Optional[List[str]],
    exclude_prefixes: Optional[List[str]],
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
    concurrency: int,
    quiet: bool,
) -> Dict[str, Any]:
    summary_path = Path(tempfile.mkstemp(suffix="_summary.json")[1])
    cmd = [
        sys.executable,
        str(BATCH_SCRIPT),
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
    ]
    if engine_params:
        cmd.extend(["--engine-params", str(engine_params)])
    if opponent_params:
        cmd.extend(["--opponent-params", str(opponent_params)])
    if depth is not None:
        cmd.extend(["--depth", str(depth)])
    elif time_per_move is not None:
        cmd.extend(["--time-per-move", str(time_per_move)])
    elif clock is not None:
        cmd.extend(["--clock", str(clock)])
    run_kwargs: Dict[str, Any] = {"check": False, "text": True}
    if quiet:
        run_kwargs["stdout"] = subprocess.DEVNULL
        run_kwargs["stderr"] = subprocess.STDOUT
    try:
        proc = subprocess.run(cmd, **run_kwargs)
        if proc.returncode != 0:
            raise RuntimeError(f"batch runner failed with code {proc.returncode}")
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        return payload
    finally:
        try:
            summary_path.unlink()
        except OSError:
            pass


def aggregate_two_sided(
    base_label: str, cand_label: str, s1: Dict[str, Any], s2: Dict[str, Any]
) -> Tuple[int, int, int]:
    summary1 = s1.get("summary", {})
    summary2 = s2.get("summary", {})
    wins = summary1.get(cand_label, 0) + summary2.get(cand_label, 0)
    losses = summary1.get(base_label, 0) + summary2.get(base_label, 0)
    draws = summary1.get("draw", 0) + summary2.get("draw", 0)
    return wins, losses, draws


def score_to_elo(score: float) -> float:
    if score <= 0.0:
        return -math.inf
    if score >= 1.0:
        return math.inf
    return 400.0 * math.log10(score / (1.0 - score))


def _run_arena_shard(
    payload: Tuple[
        List[str],
        Optional[Dict[str, Any]],
        Dict[str, Any],
        int,
        int,
        int,
    ],
) -> Dict[str, int]:
    start_fens, base_params, cand_params, depth, movetime_ms, max_plies = payload
    try:
        import skaks_eval as _skaks_eval  # local import for multiprocessing
    except Exception as exc:  # pragma: no cover - surfaced in parent process
        raise RuntimeError(f"skaks_eval not available in worker: {exc}")
    res = _skaks_eval.arena(
        start_fens=start_fens,
        base_params=base_params,
        cand_params=cand_params,
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
    games: int,
    depth: Optional[int],
    time_per_move: Optional[float],
    clock: Optional[float],
    concurrency: int,
    base_label: str,
    cand_label: str,
    quiet: bool,
    use_arena: bool,
    arena_workers: int,
    start_fens: Optional[List[str]] = None,
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
        fen_pool = start_fens or ["start"]
        if games > len(fen_pool):
            copies = (games + len(fen_pool) - 1) // len(fen_pool)
            fen_pool = (fen_pool * copies)[:games]
        else:
            fen_pool = fen_pool[:games]

        if arena_workers <= 1:
            res = skaks_eval.arena(
                start_fens=fen_pool,
                base_params=baseline_data,
                cand_params=candidate_data,
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
            concurrency=concurrency,
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
            concurrency=concurrency,
            quiet=quiet,
        )
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
            quiet=quiet, use_arena=use_arena, start_fens=rep_fens, **kwargs
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

    base_data = _load_params(current_params)
    best_data = base_data
    best_score = -1.0

    if args.output:
        best_path = Path(args.output).resolve()
    else:
        src = Path(args.start_params or args.baseline_params or "best_params")
        best_path = src.with_name(f"{src.stem}_optimized.yaml").resolve()

    best_path.parent.mkdir(parents=True, exist_ok=True)
    _save_params(best_data, best_path)

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

    for step in range(1, args.iterations + 1):
        candidates: List[Tuple[float, Dict[str, Any], Tuple[int, int, int], float]] = []

        if args.strategy == "beam":
            parents = [data for (_, data) in beam] or [best_data]
            for i in range(args.beam_size):
                parent = parents[i % len(parents)]
                cand_data = perturb_params(
                    parent,
                    args.noise,
                    include_prefixes=args.include_prefix,
                    exclude_prefixes=args.exclude_prefix,
                )
                cand_path = Path(tempfile.mkstemp(suffix="_cand.yaml")[1])
                _save_params(cand_data, cand_path)
                try:
                    color = next(color_cycle)

                    def progress_cb(rep_idx: int, repeats: int, elapsed: float) -> None:
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

                    score, (w, losses, d), wall = evaluate_candidate_repeats(
                        repeats=args.repeats,
                        progress_cb=progress_cb,
                        quiet=quiet_child,
                        use_arena=args.use_arena_binding,
                        start_fens=start_fens,
                        engine=engine,
                        baseline_params=baseline_params,
                        candidate_params=cand_path,
                        baseline_data=baseline_data,
                        candidate_data=cand_data,
                        games=args.games,
                        depth=args.depth,
                        time_per_move=args.time_per_move,
                        clock=args.clock,
                        concurrency=args.concurrency,
                        arena_workers=args.arena_workers,
                        base_label="base",
                        cand_label="cand",
                    )
                    _final_line(
                        f"{_color('✓', 'green')} iter={step} cand={i + 1}/{args.beam_size} "
                        f"score={_color(f'{score:.3f}', color)} "
                        f"WLD={w}/{losses}/{d} elo~{score_to_elo(score):+.1f} time={wall:.1f}s"
                    )
                    candidates.append((score, cand_data, (w, losses, d), wall))
                finally:
                    try:
                        cand_path.unlink()
                    except OSError:
                        pass

            candidates.sort(key=lambda x: x[0], reverse=True)
            beam = [(score, data) for score, data, _, _ in candidates[: args.beam_size]]
            top_score, top_data, _, _ = candidates[0]
        else:  # CMA-like strategy
            paths, mean_vec, is_int = _flatten_params(
                best_data,
                include_prefixes=args.include_prefix,
                exclude_prefixes=args.exclude_prefix,
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

            for i in range(popsize):
                vec = []
                for val in mean_vec:
                    scale = sigma * max(abs(val), 1.0)
                    vec.append(val + random.gauss(0.0, scale))
                cand_data = _vector_to_params(base_data, paths, vec, is_int)
                cand_path = Path(tempfile.mkstemp(suffix="_cand.yaml")[1])
                _save_params(cand_data, cand_path)
                try:
                    color = next(color_cycle)

                    def progress_cb(rep_idx: int, repeats: int, elapsed: float) -> None:
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

                    score, (w, losses, d), wall = evaluate_candidate_repeats(
                        repeats=args.repeats,
                        progress_cb=progress_cb,
                        quiet=quiet_child,
                        use_arena=args.use_arena_binding,
                        start_fens=start_fens,
                        engine=engine,
                        baseline_params=baseline_params,
                        candidate_params=cand_path,
                        baseline_data=baseline_data,
                        candidate_data=cand_data,
                        games=args.games,
                        depth=args.depth,
                        time_per_move=args.time_per_move,
                        clock=args.clock,
                        concurrency=args.concurrency,
                        arena_workers=args.arena_workers,
                        base_label="base",
                        cand_label="cand",
                    )
                    _final_line(
                        f"{_color('✓', 'green')} iter={step} cand={i + 1}/{popsize} "
                        f"score={_color(f'{score:.3f}', color)} "
                        f"WLD={w}/{losses}/{d} elo~{score_to_elo(score):+.1f} time={wall:.1f}s"
                    )
                    candidates.append((score, cand_data, (w, losses, d), wall))
                finally:
                    try:
                        cand_path.unlink()
                    except OSError:
                        pass

            candidates.sort(key=lambda x: x[0], reverse=True)
            top_score, top_data, _, _ = candidates[0]

            # Recombine top mu candidates to update mean_vec
            new_mean = [0.0 for _ in mean_vec]
            for rank in range(min(mu, len(candidates))):
                vec_paths, vec_vals, _ = _flatten_params(
                    candidates[rank][1],
                    include_prefixes=args.include_prefix,
                    exclude_prefixes=args.exclude_prefix,
                )
                if vec_paths != paths:
                    continue
                for idx, val in enumerate(vec_vals):
                    new_mean[idx] += weights[rank] * val
            mean_vec = new_mean
            best_data = _vector_to_params(base_data, paths, mean_vec, is_int)
            beam = [(top_score, best_data)]

        if top_score > best_score:
            best_score = top_score
            best_data = top_data
            _save_params(best_data, best_path)
            _final_line(
                f"{_color('⚑', 'yellow')} new best score={best_score:.3f} saved to {best_path}"
            )

    _final_line(
        f"{_color('🏁', 'bright_white')} Best score={best_score:.3f}, params at {best_path}"
    )
    if args.output:
        out_path = Path(args.output).resolve()
        _save_params(best_data, out_path)
        _final_line(f"{_color('→', 'cyan')} Wrote best params to {out_path}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-play parameter optimizer (beam + repeats)"
    )
    parser.add_argument("--engine", default="skaks", help="Path to engine binary")
    parser.add_argument("--baseline-params", help="YAML params for baseline (opponent)")
    parser.add_argument(
        "--start-params",
        help="Initial YAML params for candidate (defaults to baseline)",
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
        "--noise",
        type=float,
        default=0.05,
        help="Lognormal stddev for multiplicative noise",
    )
    parser.add_argument("--depth", type=int, help="Depth per move")
    parser.add_argument("--time-per-move", type=float, help="Seconds per move")
    parser.add_argument("--clock", type=float, help="Clock time seconds")
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
        "--cma-popsize",
        type=int,
        help="Population size for CMA-like strategy (default ~4+3*log(n))",
    )
    parser.add_argument(
        "--cma-sigma",
        type=float,
        help="Step scale for CMA-like strategy (default uses --noise)",
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
