from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path
from typing import List
from threading import Lock

import optuna
import yaml

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dep
    tqdm = None

try:
    from rich.console import Console
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
    from rich.progress import TimeRemainingColumn
except Exception:  # pragma: no cover - optional dep
    Console = None
    Progress = None

from .data import load_csv, split_dataset, filter_quiet
from .evaluator import evaluate_params
from .params import (
    DEFAULT_PARAMS,
    apply_param_updates,
    default_param_space,
    phase_weight_param_space,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Tune skaks evaluation parameters with Optuna"
    )
    p.add_argument("--data", required=True, help="CSV with fen,score[,weight] columns")
    p.add_argument("--trials", type=int, default=50, help="Number of trials to run")
    p.add_argument("--jobs", type=int, default=1, help="Parallel Optuna jobs (threads)")
    p.add_argument(
        "--threads", type=int, default=0, help="Threads for skaks_eval (0=HW)"
    )
    p.add_argument(
        "--batch-size", type=int, default=512, help="Batch size for eval_fens"
    )
    p.add_argument(
        "--limit", type=int, default=None, help="Optional row limit from dataset"
    )
    p.add_argument(
        "--val-split",
        type=float,
        default=0.0,
        help="Hold-out fraction for validation (0.0 = no split)",
    )
    p.add_argument(
        "--include-arrays", action="store_true", help="Tune array params too"
    )
    p.add_argument(
        "--phase-weights-only",
        action="store_true",
        help="Tune only phase mg/eg weights (ignores other params)",
    )
    p.add_argument(
        "--error-penalty", type=float, default=2000.0, help="Penalty per failed FEN"
    )
    p.add_argument(
        "--cp-cap",
        type=float,
        default=None,
        help="Clamp targets/preds to +/- cp-cap before metrics (helps with mate scores)",
    )
    p.add_argument(
        "--pov",
        choices=["side", "white"],
        default="side",
        help="Interpret scores as side-to-move (default, matches engine outputs) or white POV",
    )
    p.add_argument(
        "--sampler",
        choices=["cmaes", "tpe", "random"],
        default="cmaes",
        help="Sampler to use (default: cmaes)",
    )
    p.add_argument(
        "--mtpe",
        action="store_true",
        help="Use multivariate TPE (grouped) for correlated params; best with multiple jobs",
    )
    p.add_argument(
        "--pruner",
        choices=["none", "median", "successive_halving", "hyperband"],
        default="median",
    )
    p.add_argument("--storage", help="Optuna storage URL for distributed tuning")
    p.add_argument(
        "--study-name", default="skaks-opt", help="Study name (used with storage)"
    )
    p.add_argument(
        "--timeout", type=int, default=None, help="Timeout seconds for all trials"
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--best-out", type=Path, help="Write best params (nested) to YAML")
    p.add_argument(
        "--metrics-out",
        type=Path,
        help="Write per-trial metrics (loss, mae, mse, rmse, errors) to CSV",
    )
    p.add_argument(
        "--plot-out",
        type=Path,
        help="Save loss vs trial plot (png, requires matplotlib)",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress Optuna per-trial info logging (show only summary)",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar (on by default if tqdm is installed)",
    )
    p.add_argument(
        "--rich",
        action="store_true",
        default=True,
        help="Use rich progress bar with chess glyphs (if rich installed)",
    )
    p.add_argument(
        "--require-quiet",
        action="store_true",
        help="Filter dataset to quiet positions (requires skaks_eval)",
    )
    p.add_argument(
        "--quiet-batch",
        type=int,
        default=2048,
        help="Batch size for quiet filtering",
    )
    return p


def main(argv: List[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    # Hide Optuna trial spam by default; re-enable with OPTUNA_LOG_LEVEL if needed.
    if args.quiet:
        optuna.logging.set_verbosity(optuna.logging.ERROR)
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    if args.mtpe:
        # Suppress Optuna experimental warnings for multivariate/group/constant_liar flags.
        from optuna.exceptions import ExperimentalWarning

        warnings.filterwarnings("ignore", category=ExperimentalWarning)

    dataset = load_csv(args.data, limit=args.limit)
    if args.require_quiet:
        dataset = filter_quiet(dataset, batch_size=args.quiet_batch)
        print(f"Filtered to {len(dataset)} quiet positions")
    train_ds = dataset
    val_ds = None
    if args.val_split and args.val_split > 0.0:
        train_ds, val_ds = split_dataset(dataset, args.val_split, seed=args.seed)
    if args.phase_weights_only:
        param_space = phase_weight_param_space()
    else:
        param_space = default_param_space(include_arrays=args.include_arrays)

    if args.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(
            seed=args.seed,
            multivariate=args.mtpe,
            group=args.mtpe,
            constant_liar=args.mtpe,
            n_startup_trials=30 if args.mtpe else 10,
            warn_independent_sampling=not args.mtpe,
        )
    elif args.sampler == "random":
        sampler = optuna.samplers.RandomSampler(seed=args.seed)
    else:
        sampler = optuna.samplers.CmaEsSampler(seed=args.seed)

    if args.pruner == "median":
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    elif args.pruner == "successive_halving":
        pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=1, reduction_factor=3, bootstrap_count=0
        )
    elif args.pruner == "hyperband":
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource="auto", reduction_factor=3, bootstrap_count=0
        )
    else:
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=True,
    )

    # Optional progress bars: prefer rich if requested and available.
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
        and Progress is not None
        and Console is not None
        and not args.no_progress
    ):
        console = Console()
        rich_progress = Progress(
            TextColumn("[bold cyan]Tuning"),
            TextColumn("{task.fields[pieces]}", justify="left"),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("[bold]{task.fields[status]}", justify="left"),
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

    def quantized_suggest(trial: optuna.Trial, spec):
        if spec.is_float:
            return trial.suggest_float(spec.name, spec.low, spec.high, step=spec.step)
        if args.sampler == "cmaes":
            raw = trial.suggest_float(spec.name, spec.low, spec.high)
            stepped = round((raw - spec.low) / spec.step) * spec.step + spec.low
            return int(max(spec.low, min(spec.high, stepped)))
        return int(trial.suggest_int(spec.name, spec.low, spec.high, step=spec.step))

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
        glyph = chess_glyphs[trial.number % len(chess_glyphs)]
        color = color_cycle[trial.number % len(color_cycle)]

        if rich_progress is not None and rich_task is not None:
            progress_frac = (trial.number + 1) / max(1, args.trials)
            filled = min(glyph_bar_width, int(round(progress_frac * glyph_bar_width)))
            bar_text = "".join(glyph_pattern[:filled]).ljust(glyph_bar_width)
            status = f"[{color}]loss {result.loss:.1f} mae {result.mae:.1f}[/]"
            rich_progress.update(rich_task, advance=1, status=status, pieces=bar_text)
        elif progress is not None:
            payload = {
                f"{glyph} loss": f"{result.loss:.1f}",
                "mae": f"{result.mae:.1f}",
            }
            if val_ds is not None:
                payload["val"] = "pending"
            if progress_lock is not None:
                with progress_lock:
                    progress.update(1)
                    progress.set_postfix(payload, refresh=False)
            else:
                progress.update(1)
                progress.set_postfix(payload, refresh=False)
        trial.set_user_attr("error_count", result.error_count)
        trial.set_user_attr("evaluated", result.evaluated)
        trial.set_user_attr("mae", result.mae)
        trial.set_user_attr("mse", result.mse)
        trial.set_user_attr("rmse", result.rmse)

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
            if rich_progress is not None and rich_task is not None:
                progress_frac = (trial.number + 1) / max(1, args.trials)
                filled = min(
                    glyph_bar_width, int(round(progress_frac * glyph_bar_width))
                )
                bar_text = "".join(glyph_pattern[:filled]).ljust(glyph_bar_width)
                status = (
                    f"[{color}]loss {result.loss:.1f} mae {result.mae:.1f} "
                    f"val {val_res.mae:.1f}[/]"
                )
                rich_progress.update(rich_task, status=status, pieces=bar_text)
            elif progress is not None:
                payload = {
                    f"{glyph} loss": f"{result.loss:.1f}",
                    "mae": f"{result.mae:.1f}",
                    "val": f"{val_res.mae:.1f}",
                }
                if progress_lock is not None:
                    with progress_lock:
                        progress.set_postfix(payload, refresh=False)
                else:
                    progress.set_postfix(payload, refresh=False)

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

    print("=== best trial ===")
    print(
        json.dumps(
            {
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
            },
            indent=2,
        )
    )

    if args.best_out:
        args.best_out.parent.mkdir(parents=True, exist_ok=True)
        with args.best_out.open("w") as fh:
            yaml.safe_dump(merged, fh)
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
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for t in sorted(trials, key=lambda tr: tr.number):
            loss = float("nan") if t.value is None else float(t.value)
            ua = t.user_attrs
            writer.writerow(
                {
                    "trial": t.number,
                    "loss": loss,
                    "mae": ua.get("mae", loss),
                    "mse": ua.get("mse"),
                    "rmse": ua.get("rmse"),
                    "error_count": ua.get("error_count", 0),
                    "evaluated": ua.get("evaluated"),
                    "val_mae": ua.get("val_mae"),
                    "val_mse": ua.get("val_mse"),
                    "val_rmse": ua.get("val_rmse"),
                    "val_error_count": ua.get("val_error_count"),
                }
            )


def _write_plot(path: Path, trials: List[optuna.trial.FrozenTrial]) -> None:
    if not trials:
        print("no completed trials to plot")
        return

    try:
        import matplotlib.pyplot as plt
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
        key=lambda p: p[0],
    )
    if not ordered:
        print("no numerical trial values to plot")
        return

    xs = [num for num, _, _, _ in ordered]
    ys_loss = [float(val) for _, val, _, _ in ordered]
    ys_rmse = [float(r) for _, _, r, _ in ordered if r is not None]
    xs_rmse = [num for num, _, r, _ in ordered if r is not None]
    ys_val_rmse = [float(r) for _, _, _, r in ordered if r is not None]
    xs_val_rmse = [num for num, _, _, r in ordered if r is not None]

    best = float("inf")
    best_curve = []
    for val in ys_loss:
        best = min(best, val)
        best_curve.append(best)

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


if __name__ == "__main__":  # pragma: no cover
    main()
