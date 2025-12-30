"""
Optimize search speed and node reduction parameters using Optuna.
This script focuses on tuning only the 'search' parameters in skaks_opt.
"""

# --- Imports ---
import sys
import optuna
import yaml
import subprocess
import re
import tempfile
import random
from skaks_opt.params import DEFAULT_PARAMS, default_param_space
from rich.console import Console

# --- Globals ---
ALL_PGN_FENS = []


# --- Functions ---
def run_perf_with_params(params, depth, pgn, fens):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({"search": params}, f)
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


def objective(trial):
    import re

    params = DEFAULT_PARAMS["search"].copy()
    for spec in search_param_space():
        if spec.is_float:
            val = trial.suggest_float(spec.name, spec.low, spec.high, step=spec.step)
        else:
            val = trial.suggest_int(
                spec.name, int(spec.low), int(spec.high), step=int(spec.step or 1)
            )
        key = spec.name.split(".", 1)[1]
        params[key] = val
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({"search": params}, f)
        param_path = f.name
    quiet = getattr(sys.modules["__main__"], "quiet", False)
    pgn = getattr(sys.modules["__main__"], "pgn", None)
    depth = getattr(sys.modules["__main__"], "depth", 6)
    try:
        if pgn:
            fens = ALL_PGN_FENS
            if not fens:
                if not quiet:
                    print(
                        "[ERROR] No FENs found in PGN. Check the PGN file and extraction logic."
                    )
                return float("inf")
            avg_nodes_list = []
            avg_ms_list = []
            avg_nps_list = []
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
                m = re.search(
                    r"total_nodes=(\d+).*total_ms=(\d+).*total_nps=(\d+)", output
                )
                if m:
                    avg_nodes_list.append(int(m.group(1)) / 3)  # Since 3 iterations
                    avg_ms_list.append(int(m.group(2)) / 3)
                    avg_nps_list.append(int(m.group(3)))
                else:
                    avg_nodes_list.append(float("inf"))
                    avg_ms_list.append(float("inf"))
                    avg_nps_list.append(0)
            avg_ms = sum(avg_ms_list) / len(avg_ms_list)
        else:
            cmd = ["skaks", "--perf", "-d", str(depth), "--params", param_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            output = result.stdout
            m = re.search(r"avg_nodes=(\d+)", output)
            if m:
                avg_ms = 0  # Placeholder, since no ms parsing for default
            else:
                avg_ms = float("inf")
    except Exception:
        avg_ms = float("inf")
    # Add small random noise to help escape plateaus
    noise = random.uniform(-5, 5)
    return avg_ms + noise


def callback(study, trial):
    quiet = getattr(sys.modules["__main__"], "quiet", False)
    if quiet:
        trials_total = getattr(sys.modules["__main__"], "trials", 10)
        console = Console()
        end = "\r" if trial.number + 1 < trials_total else "\n"
        console.print(
            f"[bold green]♟️ Trial {trial.number + 1}/{trials_total}: best ms = {study.best_value:.0f}[/bold green]",
            end=end,
        )


def main():
    global ALL_PGN_FENS
    pgn = getattr(sys.modules["__main__"], "pgn", None)
    depth = getattr(sys.modules["__main__"], "depth", 6)
    if pgn:
        # Use hardcoded FENs for speed, ignoring the PGN
        ALL_PGN_FENS = [
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # Sicilian
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/3P1N2/PPP2PPP/RNBQKB1R w KQkq - 0 5",  # Ruy Lopez
            "rnbqkbnr/ppp2ppp/3p4/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3",  # French
            "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # Italian
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",  # Caro-Kann
        ]
    else:
        ALL_PGN_FENS = []

    pruner = optuna.pruners.HyperbandPruner()
    sampler = optuna.samplers.CmaEsSampler()
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="minimize")
    trials = getattr(sys.modules["__main__"], "trials", 10)

    # Run baseline with default params
    print("Running baseline with default parameters...")
    baseline_stats = run_perf_with_params(
        DEFAULT_PARAMS["search"], depth, pgn, ALL_PGN_FENS
    )

    study.optimize(objective, n_trials=trials, callbacks=[callback])

    best_params = study.best_params
    search_params = {
        k.split(".", 1)[1]: (int(v) if isinstance(v, float) and v.is_integer() else v)
        for k, v in best_params.items()
    }
    output_path = getattr(sys.modules["__main__"], "output", "best_for_chrono.yaml")
    with open(output_path, "w") as f:
        yaml.dump({"search": search_params}, f)
    print(f"\nBest parameter set saved to: {output_path}")
    print("\n=== Optimized parameters stats ===")
    optimized_stats = run_perf_with_params(search_params, depth, pgn, ALL_PGN_FENS)

    # Compare and show differences
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


def main_entry(*args, **kwargs):
    # Suppress Optuna logs if --quiet is set, but keep progress bar
    quiet = getattr(sys.modules["__main__"], "quiet", False)
    if quiet:
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    main()
