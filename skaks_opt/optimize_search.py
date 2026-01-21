"""Optimize search speed and node reduction parameters using Optuna."""

import os
import random
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

import optuna
import yaml
from rich.console import Console

from skaks_opt.params import DEFAULT_PARAMS, default_param_space

PERF_TIMEOUT_SECONDS = int(os.environ.get("SKAKS_PERF_TIMEOUT", "120"))
OPTIMIZE_METRIC = os.environ.get("SKAKS_OPT_METRIC", "nodes").lower()
OPTIMIZE_NOISE = float(os.environ.get("SKAKS_OPT_NOISE", "0"))


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
        raise RuntimeError("python-chess is required for --pgn") from exc

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


def run_perf_with_params(params, depth, pgn, fens, *, section_key: str = "search"):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        from skaks_opt.yaml_utils import dump_yaml

        dump_yaml({section_key: params}, f)
        param_path = f.name
    stats = {}
    if pgn and fens:
        overall_total_nodes = 0
        overall_total_ms = 0
        overall_total_nps = 0
        for fen in fens:
            cmd = [
                "skaks",
                "--perf",
                "-d",
                str(depth),
                "--params",
                param_path,
                "-f",
                fen,
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=PERF_TIMEOUT_SECONDS,
            )
            output = result.stdout
            m = re.search(r"total_nodes=(\d+).*total_ms=(\d+).*total_nps=(\d+)", output)
            if m:
                overall_total_nodes += int(m.group(1))
                overall_total_ms += int(m.group(2))
                overall_total_nps += int(m.group(3))
        stats["avg_nodes"] = overall_total_nodes / (len(fens) * 3)
        stats["avg_ms"] = overall_total_ms / (len(fens) * 3)
        stats["avg_nps"] = overall_total_nps / len(fens)
    else:
        cmd = ["skaks", "--perf", "-d", str(depth), "--params", param_path]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=PERF_TIMEOUT_SECONDS,
        )
        output = result.stdout
        m = re.search(r"avg_nodes=(\d+).*avg_ms=(\d+)", output)
        if m:
            stats["avg_nodes"] = int(m.group(1))
            stats["avg_ms"] = int(m.group(2))
        else:
            stats["avg_nodes"] = float("inf")
            stats["avg_ms"] = float("inf")
        stats["avg_nps"] = 0
    return stats


def search_param_space():
    return [spec for spec in default_param_space() if spec.name.startswith("search.")]


def _build_objective(
    *,
    pgn_path: Optional[str],
    depth: int,
    quiet: bool,
    stored_fens: List[str],
    section_key: str,
):
    def objective(trial: optuna.trial.Trial) -> float:
        base_key = "search_nnue" if section_key == "search_nnue" else "search"
        params = DEFAULT_PARAMS.get(base_key, DEFAULT_PARAMS["search"]).copy()
        for spec in search_param_space():
            if spec.is_float:
                val = trial.suggest_float(
                    spec.name, spec.low, spec.high, step=spec.step
                )
            else:
                val = trial.suggest_int(
                    spec.name, int(spec.low), int(spec.high), step=int(spec.step or 1)
                )
            key = spec.name.split(".", 1)[1]
            params[key] = val
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            from skaks_opt.yaml_utils import dump_yaml

            dump_yaml({section_key: params}, f)
            param_path = f.name
        try:
            if pgn_path:
                fens = stored_fens
                if not fens:
                    if not quiet:
                        print(
                            "[ERROR] No FENs found in PGN. Check the PGN file and extraction logic."
                        )
                    return float("inf")
                avg_ms_list = []
                avg_nodes_list = []
                for fen in fens:
                    cmd = [
                        "skaks",
                        "--perf",
                        "-d",
                        str(depth),
                        "--params",
                        param_path,
                        "-f",
                        fen,
                    ]
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=PERF_TIMEOUT_SECONDS,
                    )
                    output = result.stdout
                    m = re.search(
                        r"total_nodes=(\d+).*total_ms=(\d+).*total_nps=(\d+)", output
                    )
                    if m:
                        avg_nodes_list.append(int(m.group(1)) / 3)
                        avg_ms_list.append(int(m.group(2)) / 3)
                    else:
                        avg_nodes_list.append(float("inf"))
                        avg_ms_list.append(float("inf"))
                avg_ms = sum(avg_ms_list) / len(avg_ms_list)
                avg_nodes = sum(avg_nodes_list) / len(avg_nodes_list)
            else:
                cmd = ["skaks", "--perf", "-d", str(depth), "--params", param_path]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=PERF_TIMEOUT_SECONDS,
                )
                output = result.stdout
                m = re.search(r"avg_nodes=(\d+).*avg_ms=(\d+)", output)
                if m:
                    avg_nodes = int(m.group(1))
                    avg_ms = int(m.group(2))
                else:
                    avg_nodes = float("inf")
                    avg_ms = float("inf")
        except Exception:
            avg_nodes = float("inf")
            avg_ms = float("inf")
        noise = random.uniform(-OPTIMIZE_NOISE, OPTIMIZE_NOISE)
        metric = avg_nodes if OPTIMIZE_METRIC == "nodes" else avg_ms
        return metric + noise

    return objective


def _build_callback(quiet: bool, total_trials: int, metric_label: str):
    def callback(study: optuna.Study, trial: optuna.Trial) -> None:
        if quiet:
            console = Console()
            end = "\r" if trial.number + 1 < total_trials else "\n"
            console.print(
                f"[bold green]♟️ Trial {trial.number + 1}/{total_trials}: best {metric_label} = {study.best_value:.0f}[/bold green]",
                end=end,
            )

    return callback


def run_optimize_search(args) -> None:
    if args.quiet:
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)

    section_key = "search_nnue" if getattr(args, "search_nnue", False) else "search"

    if args.pgn:
        stored_fens = _load_fens_from_pgn(
            Path(args.pgn),
            max(1, args.games),
            args.pgn_min_ply,
            args.pgn_max_ply,
            args.pgn_seed,
        )
    else:
        stored_fens = []

    pruner = optuna.pruners.HyperbandPruner()
    sampler = optuna.samplers.CmaEsSampler()
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="minimize")

    print("Running baseline with default parameters...")
    base_key = "search_nnue" if section_key == "search_nnue" else "search"
    baseline_stats = run_perf_with_params(
        DEFAULT_PARAMS.get(base_key, DEFAULT_PARAMS["search"]),
        args.depth,
        args.pgn,
        stored_fens,
        section_key=section_key,
    )

    objective = _build_objective(
        pgn_path=args.pgn,
        depth=args.depth,
        quiet=args.quiet,
        stored_fens=stored_fens,
        section_key=section_key,
    )
    metric_label = "nodes" if OPTIMIZE_METRIC == "nodes" else "ms"
    callback = _build_callback(args.quiet, args.trials, metric_label)

    study.optimize(objective, n_trials=args.trials, callbacks=[callback])

    best_params = study.best_params
    search_params = {
        k.split(".", 1)[1]: (int(v) if isinstance(v, float) and v.is_integer() else v)
        for k, v in best_params.items()
    }
    output_path = args.output
    with open(output_path, "w", encoding="utf-8") as f:
        from skaks_opt.yaml_utils import dump_yaml

        dump_yaml({section_key: search_params}, f)
    print(f"\nBest parameter set saved to: {output_path}")
    print("\n=== Optimized parameters stats ===")
    optimized_stats = run_perf_with_params(
        search_params, args.depth, args.pgn, stored_fens, section_key=section_key
    )

    print("\n=== Comparison ===")
    print(
        f"Baseline avg_nodes: {baseline_stats['avg_nodes']:.0f}, avg_ms: {baseline_stats['avg_ms']:.0f}, avg_nps: {baseline_stats['avg_nps']:.0f}"
    )
    print(
        f"Optimized avg_nodes: {optimized_stats['avg_nodes']:.0f}, avg_ms: {optimized_stats['avg_ms']:.0f}, avg_nps: {optimized_stats['avg_nps']:.0f}"
    )
    diff_nodes = baseline_stats["avg_nodes"] - optimized_stats["avg_nodes"]
    diff_ms = baseline_stats["avg_ms"] - optimized_stats["avg_ms"]
    diff_nps = optimized_stats["avg_nps"] - baseline_stats["avg_nps"]
    base_nodes = baseline_stats["avg_nodes"]
    base_ms = baseline_stats["avg_ms"]
    base_nps = baseline_stats["avg_nps"]
    nodes_pct = (diff_nodes / base_nodes * 100.0) if base_nodes else 0.0
    ms_pct = (diff_ms / base_ms * 100.0) if base_ms else 0.0
    nps_pct = (diff_nps / base_nps * 100.0) if base_nps else 0.0
    print(
        f"Improvement: nodes {diff_nodes:+.0f} ({nodes_pct:+.1f}%), ms {diff_ms:+.0f} ({ms_pct:+.1f}%), nps {diff_nps:+.0f} ({nps_pct:+.1f}%)"
    )
