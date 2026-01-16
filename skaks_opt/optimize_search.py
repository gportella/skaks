"""Optimize search speed and node reduction parameters using Optuna."""

import random
import re
import subprocess
import tempfile
from typing import List, Optional

import optuna
import yaml
from rich.console import Console

from skaks_opt.params import DEFAULT_PARAMS, default_param_space


def run_perf_with_params(params, depth, pgn, fens):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        from skaks_opt.yaml_utils import dump_yaml

        dump_yaml({"search": params}, f)
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
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            output = result.stdout
            m = re.search(r"total_nodes=(\d+).*total_ms=(\d+).*total_nps=(\d+)", output)
            if m:
                overall_total_nodes += int(m.group(1))
                overall_total_ms += int(m.group(2))
                overall_total_nps += int(m.group(3))
        stats["avg_nodes"] = overall_total_nodes / (len(fens) * 3)
        stats["avg_nodes"] = overall_total_nodes / (len(fens) * 3)
        stats["avg_ms"] = overall_total_ms / (len(fens) * 3)
        stats["avg_nps"] = overall_total_nps / len(fens)
    else:
        cmd = ["skaks", "--perf", "-d", str(depth), "--params", param_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout
        m = re.search(r"avg_nodes=(\d+)", output)
        if m:
            stats["avg_nodes"] = int(m.group(1))
            stats["avg_ms"] = 0  # Placeholder
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
):
    def objective(trial: optuna.trial.Trial) -> float:
        params = DEFAULT_PARAMS["search"].copy()
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

            dump_yaml({"search": params}, f)
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
                        cmd, capture_output=True, text=True, timeout=30
                    )
                    output = result.stdout
                    m = re.search(
                        r"total_nodes=(\d+).*total_ms=(\d+).*total_nps=(\d+)", output
                    )
                    if m:
                        avg_ms_list.append(int(m.group(2)) / 3)
                    else:
                        avg_ms_list.append(float("inf"))
                avg_ms = sum(avg_ms_list) / len(avg_ms_list)
            else:
                cmd = ["skaks", "--perf", "-d", str(depth), "--params", param_path]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                output = result.stdout
                m = re.search(r"avg_nodes=(\d+)", output)
                if m:
                    avg_ms = 0
                else:
                    avg_ms = float("inf")
        except Exception:
            avg_ms = float("inf")
        noise = random.uniform(-5, 5)
        return avg_ms + noise

    return objective


def _build_callback(quiet: bool, total_trials: int):
    def callback(study: optuna.Study, trial: optuna.Trial) -> None:
        if quiet:
            console = Console()
            end = "\r" if trial.number + 1 < total_trials else "\n"
            console.print(
                f"[bold green]♟️ Trial {trial.number + 1}/{total_trials}: best ms = {study.best_value:.0f}[/bold green]",
                end=end,
            )

    return callback


def run_optimize_search(args) -> None:
    if args.quiet:
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)

    if args.pgn:
        stored_fens = [
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/3P1N2/PPP2PPP/RNBQKB1R w KQkq - 0 5",
            "rnbqkbnr/ppp2ppp/3p4/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3",
            "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
        ]
    else:
        stored_fens = []

    pruner = optuna.pruners.HyperbandPruner()
    sampler = optuna.samplers.CmaEsSampler()
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="minimize")

    print("Running baseline with default parameters...")
    baseline_stats = run_perf_with_params(
        DEFAULT_PARAMS["search"], args.depth, args.pgn, stored_fens
    )

    objective = _build_objective(
        pgn_path=args.pgn,
        depth=args.depth,
        quiet=args.quiet,
        stored_fens=stored_fens,
    )
    callback = _build_callback(args.quiet, args.trials)

    study.optimize(objective, n_trials=args.trials, callbacks=[callback])

    best_params = study.best_params
    search_params = {
        k.split(".", 1)[1]: (int(v) if isinstance(v, float) and v.is_integer() else v)
        for k, v in best_params.items()
    }
    output_path = args.output
    with open(output_path, "w", encoding="utf-8") as f:
        from skaks_opt.yaml_utils import dump_yaml

        dump_yaml({"search": search_params}, f)
    print(f"\nBest parameter set saved to: {output_path}")
    print("\n=== Optimized parameters stats ===")
    optimized_stats = run_perf_with_params(
        search_params, args.depth, args.pgn, stored_fens
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
    print(
        f"Improvement: nodes {diff_nodes:+.0f} ({diff_nodes / baseline_stats['avg_nodes'] * 100:+.1f}%), ms {diff_ms:+.0f} ({diff_ms / baseline_stats['avg_ms'] * 100:+.1f}%), nps {diff_nps:+.0f} ({diff_nps / baseline_stats['avg_nps'] * 100:+.1f}%)"
    )
