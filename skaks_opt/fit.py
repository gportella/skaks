from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path
from threading import Lock
from typing import List

import optuna

from skaks_opt.data import Dataset, filter_quiet, load_csv, split_dataset
from skaks_opt.evaluator import EvalResult, evaluate_params
from skaks_opt.params import (DEFAULT_PARAMS, apply_param_updates,
                              default_param_space, param_space_for_mode,
                              phase_weight_param_space)

try:  # Optional rich progress support
    from rich.console import Console
    from rich.progress import (BarColumn, Progress, TextColumn,
                               TimeElapsedColumn, TimeRemainingColumn)
except Exception:  # pragma: no cover - optional dependency
    Console = None
    Progress = None

try:  # Optional tqdm support
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

__all__ = ["add_subparser", "run_fit"]


def add_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "fit",
        help="Run supervised parameter fitting against labeled positions",
        description=(
            "Supervised Optuna loop that minimizes centipawn error on labeled "
            "datasets (no gameplay). Useful for bringing eval parameters in line "
            "with Stockfish targets before switching to arena-based tuning."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="CSV with FEN + score columns")
    parser.add_argument("--trials", type=int, default=50, help="Optuna trial count")
    parser.add_argument(
        "--jobs", type=int, default=1, help="Parallel Optuna jobs (threads)"
    )
    parser.add_argument(
        "--threads", type=int, default=0, help="Threads to pass into skaks_eval"
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Eval batch size")
    parser.add_argument("--limit", type=int, help="Optional dataset row limit")
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.0,
        help="Validation split fraction (0.0 disables)",
    )
    parser.add_argument(
        "--include-arrays",
        action="store_true",
        help="Include array-valued parameters in search space",
    )
    parser.add_argument(
        "--phase-weights-only",
        action="store_true",
        help="Limit tuning to phase weight arrays",
    )
    parser.add_argument(
        "--param-set",
        choices=["full", "phase", "offense", "defense"],
        default="full",
        help="Select parameter subset to optimize",
    )
    parser.add_argument(
        "--error-penalty",
        type=float,
        default=2000.0,
        help="Penalty (centipawns) for failed evals",
    )
    parser.add_argument(
        "--cp-cap",
        type=float,
        help="Clamp centipawn targets/preds before scoring",
    )
    parser.add_argument(
        "--pov",
        choices=["side", "white"],
        default="side",
        help="Interpret scores relative to side-to-move or white",
    )
    parser.add_argument(
        "--sampler",
        choices=["cmaes", "tpe", "random"],
        default="cmaes",
        help="Optuna sampler",
    )
    parser.add_argument(
        "--mtpe",
        action="store_true",
        help="Enable multivariate/grouped TPE tuning",
    )
    parser.add_argument(
        "--pruner",
        choices=["none", "median", "successive_halving", "hyperband"],
        default="median",
        help="Optuna pruner",
    )
    parser.add_argument("--storage", help="Optuna storage URL")
    parser.add_argument(
        "--study-name", default="skaks-opt", help="Optuna study identifier"
    )
    parser.add_argument(
        "--timeout", type=int, help="Overall timeout (seconds) for study"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--best-out",
        type=Path,
        help="Write best nested params to YAML",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        help="Write per-trial metrics CSV",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        help="Write loss/RMSE plot (requires matplotlib)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress Optuna logging noise",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars",
    )
    parser.add_argument(
        "--rich",
        dest="rich",
        action="store_true",
        default=True,
        help="Enable rich progress output when available",
    )
    parser.add_argument(
        "--no-rich",
        dest="rich",
        action="store_false",
        help="Disable rich progress even if installed",
    )
    parser.add_argument(
        "--require-quiet",
        action="store_true",
        help="Filter dataset to quiet positions (needs skaks_eval)",
    )
    parser.add_argument(
        "--quiet-batch",
        type=int,
        default=2048,
        help="Batch size for quiet filtering",
    )
    return parser


def _select_param_space(args: argparse.Namespace):
    if args.phase_weights_only:
        return phase_weight_param_space()
    if args.param_set == "phase":
        return phase_weight_param_space()
    if args.param_set in {"offense", "defense", "full"}:
        return param_space_for_mode(args.param_set, include_arrays=args.include_arrays)
    return default_param_space(include_arrays=args.include_arrays)


def _build_sampler(args: argparse.Namespace) -> optuna.samplers.BaseSampler:
    if args.sampler == "tpe":
        return optuna.samplers.TPESampler(
            seed=args.seed,
            multivariate=args.mtpe,
            group=args.mtpe,
            constant_liar=args.mtpe,
            n_startup_trials=30 if args.mtpe else 10,
            warn_independent_sampling=not args.mtpe,
        )
    if args.sampler == "random":
        return optuna.samplers.RandomSampler(seed=args.seed)
    return optuna.samplers.CmaEsSampler(seed=args.seed)


def _build_pruner(args: argparse.Namespace) -> optuna.pruners.BasePruner:
    if args.pruner == "median":
        return optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    if args.pruner == "successive_halving":
        return optuna.pruners.SuccessiveHalvingPruner(
            min_resource=1,
            reduction_factor=3,
            bootstrap_count=0,
        )
    if args.pruner == "hyperband":
        return optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource="auto",
            reduction_factor=3,
            bootstrap_count=0,
        )
    return optuna.pruners.NopPruner()


def run_fit(args: argparse.Namespace) -> None:
    if args.quiet:
        optuna.logging.set_verbosity(optuna.logging.ERROR)
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    if args.mtpe:
        from optuna.exceptions import ExperimentalWarning

        warnings.filterwarnings("ignore", category=ExperimentalWarning)

    dataset = load_csv(args.data, limit=args.limit)
    if args.require_quiet:
        dataset = filter_quiet(dataset, batch_size=args.quiet_batch)
        print(f"Filtered to {len(dataset)} quiet positions")

    train_ds: Dataset = dataset
    val_ds: Dataset | None = None
    if args.val_split and args.val_split > 0.0:
        train_ds, val_ds = split_dataset(dataset, args.val_split, seed=args.seed)

    param_space = _select_param_space(args)
    sampler = _build_sampler(args)
    pruner = _build_pruner(args)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=True,
    )

    progress = None
    progress_lock: Lock | None = None
    rich_progress = None
    rich_task = None
    glyph_bar_width = 40

    chess_glyphs = ["♙", "♘", "♗", "♖", "♕", "♔", "♟", "♞", "♝", "♜", "♛", "♚"]
    color_cycle = ["cyan", "magenta", "green", "yellow", "blue", "white"]
    glyph_pattern = [
        f"[{color_cycle[i % len(color_cycle)]}]{chess_glyphs[i % len(chess_glyphs)]}[/]"
        for i in range(glyph_bar_width)
    ]

    if (
        args.rich
        and not args.no_progress
        and Progress is not None
        and Console is not None
    ):
        console = Console(highlight=False, soft_wrap=False)
        rich_progress = Progress(
            TextColumn("[bold cyan]Tuning"),
            TextColumn("{task.fields[pieces]}", justify="left"),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("[bold]{task.fields[status]}"),
            console=console,
            expand=True,
        )
        rich_task = rich_progress.add_task(
            "tuning", total=args.trials, status="", pieces=""
        )
        rich_progress.start()
    elif tqdm is not None and not args.no_progress:
        progress = tqdm(total=args.trials, desc="Tuning [N]", dynamic_ncols=True)
        progress_lock = Lock()

    def quantized_suggest(trial: optuna.Trial, spec) -> float:
        if spec.is_float:
            return trial.suggest_float(spec.name, spec.low, spec.high, step=spec.step)
        if args.sampler == "cmaes":
            raw = trial.suggest_float(spec.name, spec.low, spec.high)
            stepped = round((raw - spec.low) / spec.step) * spec.step + spec.low
            return int(max(spec.low, min(spec.high, stepped)))
        return int(trial.suggest_int(spec.name, spec.low, spec.high, step=spec.step))

    def _update_progress(trial: optuna.Trial, result: EvalResult, val_res=None) -> None:
        glyph = chess_glyphs[trial.number % len(chess_glyphs)]
        color = color_cycle[trial.number % len(color_cycle)]
        if rich_progress is not None and rich_task is not None:
            progress_frac = (trial.number + 1) / max(1, args.trials)
            filled = min(glyph_bar_width, int(round(progress_frac * glyph_bar_width)))
            bar_text = "".join(glyph_pattern[:filled]).ljust(glyph_bar_width)
            if val_res is not None:
                status = (
                    f"[{color}]loss {result.loss:.1f} mae {result.mae:.1f} "
                    f"val {val_res.mae:.1f}[/]"
                )
            else:
                status = f"[{color}]loss {result.loss:.1f} mae {result.mae:.1f}[/]"
            rich_progress.update(rich_task, advance=1, status=status, pieces=bar_text)
        elif progress is not None:
            payload = {
                f"{glyph} loss": f"{result.loss:.1f}",
                "mae": f"{result.mae:.1f}",
            }
            if val_res is not None:
                payload["val"] = f"{val_res.mae:.1f}"
            if progress_lock is not None:
                with progress_lock:
                    progress.update(1)
                    progress.set_postfix(payload, refresh=False)
            else:
                progress.update(1)
                progress.set_postfix(payload, refresh=False)

    def objective(trial: optuna.Trial) -> float:
        updates = {spec.name: quantized_suggest(trial, spec) for spec in param_space}
        result = evaluate_params(
            param_updates=updates,
            dataset=train_ds,
            batch_size=args.batch_size,
            threads=args.threads,
            error_penalty=args.error_penalty,
            pov=args.pov,
            cp_cap=args.cp_cap,
        )
        trial.set_user_attr("error_count", result.error_count)
        trial.set_user_attr("evaluated", result.evaluated)
        trial.set_user_attr("mae", result.mae)
        trial.set_user_attr("mse", result.mse)
        trial.set_user_attr("rmse", result.rmse)

        val_res: EvalResult | None = None
        if val_ds is not None:
            val_res = evaluate_params(
                param_updates=updates,
                dataset=val_ds,
                batch_size=args.batch_size,
                threads=args.threads,
                error_penalty=args.error_penalty,
                pov=args.pov,
                cp_cap=args.cp_cap,
            )
            trial.set_user_attr("val_mae", val_res.mae)
            trial.set_user_attr("val_mse", val_res.mse)
            trial.set_user_attr("val_rmse", val_res.rmse)
            trial.set_user_attr("val_error_count", val_res.error_count)

        _update_progress(trial, result, val_res)
        return result.loss

    study.optimize(
        objective, n_trials=args.trials, timeout=args.timeout, n_jobs=args.jobs
    )

    if progress is not None:
        progress.close()
    if rich_progress is not None:
        rich_progress.stop()

    best = study.best_trial
    merged = apply_param_updates(DEFAULT_PARAMS, best.params)

    def _round_params(payload):
        if isinstance(payload, dict):
            return {
                key: _round_params(val)
                for key, val in payload.items()
                if key not in {"phase_weights_mg", "phase_weights_eg"}
            } | {
                key: payload[key]
                for key in ("phase_weights_mg", "phase_weights_eg")
                if key in payload
            }
        if isinstance(payload, list):
            return [_round_params(x) for x in payload]
        if isinstance(payload, (int, float)):
            return int(round(payload))
        return payload

    rounded = {
        "evaluation": _round_params(merged.get("evaluation", {})),
        "search": _round_params(merged.get("search", {})),
    }

    summary = {
        "number": best.number,
        "loss": best.value,
        "mae": best.user_attrs.get("mae", best.value),
        "mse": best.user_attrs.get("mse"),
        "rmse": best.user_attrs.get("rmse"),
        "error_count": best.user_attrs.get("error_count", 0),
        "evaluated": best.user_attrs.get("evaluated"),
        "val_mae": best.user_attrs.get("val_mae"),
        "val_mse": best.user_attrs.get("val_mse"),
        "val_rmse": best.user_attrs.get("val_rmse"),
        "val_error_count": best.user_attrs.get("val_error_count"),
    }

    print("=== best trial ===")
    print(json.dumps(summary, indent=2))

    if args.best_out:
        args.best_out.parent.mkdir(parents=True, exist_ok=True)
        with args.best_out.open("w", encoding="utf-8") as fh:
            import yaml

            yaml.safe_dump(rounded, fh)
        print(f"wrote best params to {args.best_out}")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    if args.metrics_out:
        _write_metrics_csv(args.metrics_out, completed)
        print(f"wrote metrics to {args.metrics_out}")

    if args.plot_out:
        _write_plot(args.plot_out, completed)
        print(f"wrote loss plot to {args.plot_out}")


def _write_metrics_csv(path: Path, trials: List[optuna.trial.FrozenTrial]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial",
        "loss",
        "mae",
        "mse",
        "rmse",
        "error_count",
        "evaluated",
        "val_mae",
        "val_mse",
        "val_rmse",
        "val_error_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for trial in sorted(trials, key=lambda tr: tr.number):
            loss = float("nan") if trial.value is None else float(trial.value)
            attrs = trial.user_attrs
            writer.writerow(
                {
                    "trial": trial.number,
                    "loss": loss,
                    "mae": attrs.get("mae", loss),
                    "mse": attrs.get("mse"),
                    "rmse": attrs.get("rmse"),
                    "error_count": attrs.get("error_count", 0),
                    "evaluated": attrs.get("evaluated"),
                    "val_mae": attrs.get("val_mae"),
                    "val_mse": attrs.get("val_mse"),
                    "val_rmse": attrs.get("val_rmse"),
                    "val_error_count": attrs.get("val_error_count"),
                }
            )


def _write_plot(path: Path, trials: List[optuna.trial.FrozenTrial]) -> None:
    if not trials:
        print("no completed trials to plot")
        return

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for --plot-out") from exc

    ordered = sorted(
        [
            (
                t.number,
                t.value,
                t.user_attrs.get("rmse"),
                t.user_attrs.get("val_rmse"),
            )
            for t in trials
            if t.value is not None
        ],
        key=lambda tpl: tpl[0],
    )
    if not ordered:
        print("no numerical trial values to plot")
        return

    xs = [num for num, _, _, _ in ordered]
    ys_loss = [float(val) for _, val, _, _ in ordered]
    best = float("inf")
    best_curve = []
    for val in ys_loss:
        best = min(best, val)
        best_curve.append(best)

    xs_rmse = [num for num, _, rmse, _ in ordered if rmse is not None]
    ys_rmse = [float(rmse) for _, _, rmse, _ in ordered if rmse is not None]
    xs_val_rmse = [num for num, _, _, rmse in ordered if rmse is not None]
    ys_val_rmse = [float(rmse) for _, _, _, rmse in ordered if rmse is not None]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax1.plot(xs, ys_loss, marker="o", linewidth=1.1, label="loss (MAE)")
    ax1.plot(xs, best_curve, linestyle="--", linewidth=1.0, label="best so far")
    ax1.set_ylabel("loss")
    ax1.set_title("skaks-opt: loss and RMSE by trial")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()

    if xs_rmse:
        ax2.plot(xs_rmse, ys_rmse, marker="o", linewidth=1.0, label="train RMSE")
    if xs_val_rmse:
        ax2.plot(xs_val_rmse, ys_val_rmse, marker="o", linewidth=1.0, label="val RMSE")
    ax2.set_xlabel("trial")
    ax2.set_ylabel("rmse")
    ax2.grid(True, linestyle="--", alpha=0.4)
    if xs_rmse or xs_val_rmse:
        ax2.legend()

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
