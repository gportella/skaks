from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import optuna
import yaml

from skaks_opt.params import (
    DEFAULT_PARAMS,
    apply_param_updates,
    param_space_for_mode,
)

__all__ = ["add_subparser", "run_texel", "load_texel_csv", "TexelDataset"]


def add_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "texel",
        help="Run Texel logistic regression tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", required=True, help="CSV with fen,outcome[,weight] columns"
    )
    parser.add_argument("--trials", type=int, default=100, help="Number of trials")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel Optuna jobs")
    parser.add_argument("--threads", type=int, default=0, help="Threads for skaks_eval")
    parser.add_argument(
        "--batch-size", type=int, default=512, help="Batch size for eval_fens"
    )
    parser.add_argument("--limit", type=int, help="Optional dataset cap")
    parser.add_argument(
        "--cp-cap",
        type=float,
        help="Clamp centipawn evaluations/targets before loss",
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
        "--pruner",
        choices=["none", "median", "successive_halving", "hyperband"],
        default="median",
        help="Optuna pruner",
    )
    parser.add_argument("--storage", help="Optuna storage URL")
    parser.add_argument("--study-name", default="skaks-texel", help="Optuna study name")
    parser.add_argument("--timeout", type=int, help="Timeout seconds for study")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
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
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print progress every N completed trials",
    )
    parser.add_argument(
        "--progress-style",
        choices=["simple", "fancy"],
        default="simple",
        help="Progress output style",
    )
    parser.add_argument(
        "--progress-color",
        choices=["auto", "ansi", "none"],
        default="auto",
        help="Colorize progress output",
    )
    parser.add_argument("--best-out", type=Path, help="Write best params to YAML")
    parser.add_argument("--metrics-out", type=Path, help="Write per-trial metrics CSV")
    parser.add_argument(
        "--plot-out",
        type=Path,
        help="Write loss plot (requires matplotlib)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce Optuna logging",
    )
    parser.add_argument(
        "--texel-scale-min",
        type=float,
        default=200.0,
        help="Minimum Texel sigmoid scale (cp)",
    )
    parser.add_argument(
        "--texel-scale-max",
        type=float,
        default=1200.0,
        help="Maximum Texel sigmoid scale (cp)",
    )
    parser.add_argument(
        "--error-penalty",
        type=float,
        default=6.0,
        help="Penalty added per failed evaluation",
    )
    parser.add_argument(
        "--param-set",
        choices=["full", "phase", "offense", "defense"],
        default="full",
        help="Select parameter subset to optimize",
    )
    parser.add_argument(
        "--include-arrays",
        action="store_true",
        help="Include array params when available",
    )
    parser.add_argument(
        "--base-params",
        type=Path,
        default=Path("tuning/best_params_lots_2.yaml"),
        help="Initial YAML merged with tuned updates",
    )
    return parser


def _logit_prob(cp: np.ndarray, scale: float) -> np.ndarray:
    return 1.0 / (1.0 + np.power(10.0, -cp / scale))


@dataclass(frozen=True)
class TexelDataset:
    fens: List[str]
    outcomes: np.ndarray
    weights: np.ndarray
    side: np.ndarray

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.fens)

    def iter_batches(self, batch_size: int):
        for start in range(0, len(self.fens), batch_size):
            end = min(start + batch_size, len(self.fens))
            yield (
                self.fens[start:end],
                self.outcomes[start:end],
                self.weights[start:end],
                self.side[start:end],
            )


def _parse_side_to_move(fen: str) -> int:
    parts = fen.split()
    if len(parts) < 2:
        return 1
    return 1 if parts[1] == "w" else -1


def load_texel_csv(path: Path | str, limit: int | None = None) -> TexelDataset:
    path = Path(path)
    fens: List[str] = []
    outcomes: List[float] = []
    weights: List[float] = []
    sides: List[int] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "fen" not in reader.fieldnames:
            raise ValueError("CSV must contain fen column")
        outcome_col = None
        for candidate in ("outcome", "result", "score"):
            if candidate in reader.fieldnames:
                outcome_col = candidate
                break
        if outcome_col is None:
            raise ValueError("CSV must contain an outcome/result column")
        for row in reader:
            if limit is not None and len(fens) >= limit:
                break
            fen = row["fen"].strip()
            try:
                value = float(row[outcome_col])
            except Exception:
                continue
            if not (0.0 <= value <= 1.0):
                continue
            weight = float(row.get("weight", 1.0) or 1.0)
            fens.append(fen)
            outcomes.append(value)
            weights.append(weight)
            stm = row.get("side_to_move")
            if stm is not None and stm.lower().startswith("w"):
                sides.append(1)
            elif stm is not None and stm.lower().startswith("b"):
                sides.append(-1)
            else:
                sides.append(_parse_side_to_move(fen))
    if not fens:
        raise ValueError(f"no rows loaded from {path}")
    return TexelDataset(
        fens=fens,
        outcomes=np.array(outcomes, dtype=np.float64),
        weights=np.array(weights, dtype=np.float64),
        side=np.array(sides, dtype=np.int8),
    )


def filter_quiet(dataset: TexelDataset, batch_size: int) -> TexelDataset:
    try:
        import skaks_eval as sk
    except Exception:  # pragma: no cover - optional dependency
        warnings.warn("skaks_eval not available; skipping quiet filtering")
        return dataset

    keep_mask = np.zeros(len(dataset.fens), dtype=bool)
    for start in range(0, len(dataset.fens), batch_size):
        end = min(start + batch_size, len(dataset.fens))
        chunk = dataset.fens[start:end]
        flags = sk.is_quiet_batch(chunk)
        for idx, flag in enumerate(flags):
            keep_mask[start + idx] = bool(flag) if flag is not None else False

    if keep_mask.all():
        return dataset
    if not keep_mask.any():
        raise ValueError("quiet filtering removed all positions")

    return TexelDataset(
        fens=[fen for fen, keep in zip(dataset.fens, keep_mask) if keep],
        outcomes=dataset.outcomes[keep_mask],
        weights=dataset.weights[keep_mask],
        side=dataset.side[keep_mask],
    )


def quantized_suggest(trial: optuna.Trial, spec, sampler: str):
    if spec.is_float:
        step = spec.step if isinstance(spec.step, float) else None
        return trial.suggest_float(spec.name, spec.low, spec.high, step=step)
    if sampler == "cmaes":
        raw = trial.suggest_float(spec.name, spec.low, spec.high)
        stepped = round((raw - spec.low) / spec.step) * spec.step + spec.low
        return int(max(spec.low, min(spec.high, stepped)))
    return int(trial.suggest_int(spec.name, spec.low, spec.high, step=spec.step))


def texel_loss(
    *,
    params,
    dataset: TexelDataset,
    batch_size: int,
    threads: int,
    cp_cap: float | None,
    pov: str,
    scale: float,
    error_penalty: float,
) -> tuple[float, int]:
    try:
        import skaks_eval as sk
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "skaks_eval is not installed; install bindings first"
        ) from exc

    eps = 1e-9
    total_weight = 0.0
    total_loss = 0.0
    error_count = 0

    for fens, outcomes, weights, side in dataset.iter_batches(batch_size):
        result = sk.eval_fens(fens, params=params, threads=threads)
        cp_raw = np.asarray(result["cp"], dtype=np.float64)
        if cp_cap is not None:
            cp_raw = np.clip(cp_raw, -cp_cap, cp_cap)
        cp = cp_raw * side if pov == "side" else cp_raw
        probs = _logit_prob(cp, scale=scale)
        errs = result["errors"]
        for i, err in enumerate(errs):
            w = float(weights[i])
            if err is not None:
                total_loss += error_penalty * w
                total_weight += w
                error_count += 1
                continue
            p = float(np.clip(probs[i], eps, 1.0 - eps))
            y = float(outcomes[i])
            total_loss += (-y * math.log(p) - (1.0 - y) * math.log(1.0 - p)) * w
            total_weight += w

    mean_loss = total_loss / total_weight if total_weight > 0 else float("inf")
    return mean_loss, error_count


def run_texel(args: argparse.Namespace) -> None:
    if args.quiet:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    if args.sampler == "tpe":
        from optuna.exceptions import ExperimentalWarning

        warnings.filterwarnings("ignore", category=ExperimentalWarning)

    dataset = load_texel_csv(args.data, limit=args.limit)
    if args.require_quiet:
        dataset = filter_quiet(dataset, batch_size=args.quiet_batch)
        print(f"Filtered to {len(dataset)} quiet positions")

    param_space = param_space_for_mode(
        mode=args.param_set, include_arrays=args.include_arrays
    )

    if args.base_params and Path(args.base_params).exists():
        with Path(args.base_params).open("r", encoding="utf-8") as fh:
            base_params = yaml.safe_load(fh)
    else:
        base_params = DEFAULT_PARAMS

    if args.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(
            seed=args.seed,
            multivariate=False,
            n_startup_trials=15,
            warn_independent_sampling=True,
        )
    elif args.sampler == "random":
        sampler = optuna.samplers.RandomSampler(seed=args.seed)
    else:
        sampler = optuna.samplers.CmaEsSampler(seed=args.seed)

    pruner: optuna.pruners.BasePruner
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

    start_time = time.time()
    initial_completed = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    )
    planned_new = args.trials if args.trials is not None else None
    planned_total = initial_completed + planned_new if planned_new is not None else None
    color_enabled = args.progress_color in {"ansi"} or (
        args.progress_color == "auto" and sys.stdout.isatty()
    )

    def objective(trial: optuna.Trial) -> float:
        updates = {
            spec.name: quantized_suggest(trial, spec, args.sampler)
            for spec in param_space
        }
        scale = trial.suggest_float(
            "texel_scale", args.texel_scale_min, args.texel_scale_max
        )
        params = apply_param_updates(base_params, updates)
        loss, err_count = texel_loss(
            params=params,
            dataset=dataset,
            batch_size=args.batch_size,
            threads=args.threads,
            cp_cap=args.cp_cap,
            pov=args.pov,
            scale=scale,
            error_penalty=args.error_penalty,
        )
        trial.set_user_attr("error_count", err_count)
        trial.set_user_attr("texel_scale", scale)
        return loss

    def _progress_callback(
        study_obj: optuna.study.Study, trial: optuna.trial.FrozenTrial
    ) -> None:
        if args.progress_every <= 0:
            return
        if (trial.number + 1) % args.progress_every != 0:
            return
        best = study_obj.best_trial
        best_val = getattr(best, "value", None)
        best_loss = f"{best_val:.4f}" if best_val is not None else "-"
        scale_val = trial.user_attrs.get("texel_scale")
        scale_txt = f"{scale_val:.1f}" if scale_val is not None else "-"
        err_cnt = trial.user_attrs.get("error_count", 0)
        completed_all = sum(
            1 for t in study_obj.trials if t.state == optuna.trial.TrialState.COMPLETE
        )
        session_completed = max(0, completed_all - initial_completed)
        total = planned_new if planned_new is not None else "?"
        elapsed = time.time() - start_time

        green = "\x1b[32m" if color_enabled else ""
        yellow = "\x1b[33m" if color_enabled else ""
        cyan = "\x1b[36m" if color_enabled else ""
        reset = "\x1b[0m" if color_enabled else ""

        improving = False
        try:
            if best_val is not None and trial.value <= best_val + 1e-9:
                improving = True
        except Exception:
            improving = False

        if args.progress_style == "fancy":
            width = 30
            pieces = ["♔", "♕", "♖", "♗", "♘", "♙", "♚", "♛", "♜", "♝", "♞", "♟"]
            pos = (trial.number * 3) % width
            piece = pieces[trial.number % len(pieces)]
            track = "".join(piece if idx == pos else "·" for idx in range(width))
            track_colored = f"{cyan}{track}{reset}" if color_enabled else track
            loss_txt = (
                f"{green if improving else yellow}{trial.value:.4f}{reset}"
                if color_enabled
                else f"{trial.value:.4f}"
            )
            best_txt = f"{green}{best_loss}{reset}" if color_enabled else best_loss
            line = (
                f"\r[progress] {session_completed}/{total} (total {completed_all}) [{track_colored}] "
                f"loss={loss_txt} best={best_txt} "
                f"scale={scale_txt} errors={err_cnt} "
                f"elapsed={elapsed:.1f}s"
            )
        else:
            loss_txt = (
                f"{green if improving else yellow}{trial.value:.4f}{reset}"
                if color_enabled
                else f"{trial.value:.4f}"
            )
            best_txt = f"{green}{best_loss}{reset}" if color_enabled else best_loss
            line = (
                f"\r[progress] {session_completed}/{total} (total {completed_all}) "
                f"loss={loss_txt} best={best_txt} "
                f"scale={scale_txt} errors={err_cnt} "
                f"elapsed={elapsed:.1f}s"
            )

        print(line, end="", flush=True)

    study.optimize(
        objective,
        n_trials=args.trials,
        timeout=args.timeout,
        n_jobs=args.jobs,
        callbacks=[_progress_callback],
    )

    if args.progress_every > 0:
        print()

    best = study.best_trial
    merged = apply_param_updates(base_params, best.params)
    best_record = {
        "number": best.number,
        "loss": best.value,
        "error_count": best.user_attrs.get("error_count", 0),
        "texel_scale": best.user_attrs.get("texel_scale"),
    }
    print("=== best trial ===")
    print(json.dumps(best_record, indent=2))

    if args.best_out:
        args.best_out.parent.mkdir(parents=True, exist_ok=True)
        with args.best_out.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(merged, fh)
        print(f"wrote best params to {args.best_out}")

    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["number", "loss", "texel_scale", "error_count"])
            for trial in study.trials:
                if trial.state != optuna.trial.TrialState.COMPLETE:
                    continue
                writer.writerow(
                    [
                        trial.number,
                        trial.value,
                        trial.user_attrs.get("texel_scale"),
                        trial.user_attrs.get("error_count", 0),
                    ]
                )
        print(f"wrote metrics to {args.metrics_out}")

    if args.plot_out:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError:
            print("matplotlib not installed; skipping plot")
        else:
            xs = []
            ys = []
            for trial in study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    xs.append(trial.number)
                    ys.append(trial.value)
            plt.figure(figsize=(8, 4))
            plt.plot(xs, ys, marker="o", linestyle="-", linewidth=1)
            plt.xlabel("trial")
            plt.ylabel("texel loss")
            plt.title("skaks texel tuning")
            plt.tight_layout()
            args.plot_out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(args.plot_out)
            print(f"wrote loss plot to {args.plot_out}")
