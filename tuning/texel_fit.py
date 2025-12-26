from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import optuna
import yaml

from skaks_opt.params import DEFAULT_PARAMS, apply_param_updates, default_param_space


def _logit_prob(cp: np.ndarray, scale: float) -> np.ndarray:
    # Texel-style sigmoid using base 10 to match Elo scaling
    return 1.0 / (1.0 + np.power(10.0, -cp / scale))


@dataclass(frozen=True)
class TexelDataset:
    fens: List[str]
    outcomes: np.ndarray  # shape (n,), float in [0,1]
    weights: np.ndarray  # shape (n,)
    side: np.ndarray  # shape (n,), +1 for white, -1 for black

    def __len__(self) -> int:
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
    outs: List[float] = []
    wts: List[float] = []
    sides: List[int] = []
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "fen" not in reader.fieldnames:
            raise ValueError("CSV must contain fen column")
        col = None
        for candidate in ("outcome", "result", "score"):
            if candidate in reader.fieldnames:
                col = candidate
                break
        if col is None:
            raise ValueError("CSV must contain an outcome/result column")
        for row in reader:
            if limit is not None and len(fens) >= limit:
                break
            fen = row["fen"].strip()
            try:
                outcome = float(row[col])
            except Exception:
                continue
            if not (0.0 <= outcome <= 1.0):
                continue
            weight = float(row.get("weight", 1.0) or 1.0)
            fens.append(fen)
            outs.append(outcome)
            wts.append(weight)
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
        outcomes=np.array(outs, dtype=np.float64),
        weights=np.array(wts, dtype=np.float64),
        side=np.array(sides, dtype=np.int8),
    )


def quantized_suggest(trial: optuna.Trial, spec, sampler: str) -> int:
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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Texel tuning for skaks eval")
    p.add_argument(
        "--data", required=True, help="CSV with fen,outcome[,weight] columns"
    )
    p.add_argument("--trials", type=int, default=100, help="Number of trials")
    p.add_argument("--jobs", type=int, default=1, help="Parallel Optuna jobs")
    p.add_argument("--threads", type=int, default=0, help="Threads for skaks_eval")
    p.add_argument(
        "--batch-size", type=int, default=512, help="Batch size for eval_fens"
    )
    p.add_argument("--limit", type=int, default=None, help="Optional row cap")
    p.add_argument("--cp-cap", type=float, default=None, help="Clamp evals and targets")
    p.add_argument(
        "--pov",
        choices=["side", "white"],
        default="side",
        help="Interpret scores as side-to-move (default) or white",
    )
    p.add_argument(
        "--sampler",
        choices=["cmaes", "tpe", "random"],
        default="cmaes",
        help="Optuna sampler (default cmaes)",
    )
    p.add_argument(
        "--pruner",
        choices=["none", "median", "successive_halving", "hyperband"],
        default="median",
    )
    p.add_argument("--storage", help="Optuna storage URL")
    p.add_argument("--study-name", default="skaks-texel", help="Study name")
    p.add_argument("--timeout", type=int, default=None, help="Timeout seconds")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--best-out", type=Path, help="Write best params to YAML")
    p.add_argument("--metrics-out", type=Path, help="Write per-trial metrics to CSV")
    p.add_argument("--plot-out", type=Path, help="Loss plot output (png)")
    p.add_argument("--quiet", action="store_true", help="Reduce Optuna logging")
    p.add_argument(
        "--texel-scale-min",
        type=float,
        default=200.0,
        help="Minimum Texel scale (cp)",
    )
    p.add_argument(
        "--texel-scale-max",
        type=float,
        default=1200.0,
        help="Maximum Texel scale (cp)",
    )
    p.add_argument(
        "--error-penalty",
        type=float,
        default=6.0,
        help="Loss added for failed evals (per weighted sample)",
    )
    p.add_argument("--include-arrays", action="store_true", help="Tune array params")
    p.add_argument(
        "--base-params",
        type=Path,
        default=Path("tuning/best_params_lots_2.yaml"),
        help="Starting params YAML to merge updates into",
    )
    return p


def main(argv: List[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.quiet:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    if args.sampler == "tpe":
        from optuna.exceptions import ExperimentalWarning

        warnings.filterwarnings("ignore", category=ExperimentalWarning)

    dataset = load_texel_csv(args.data, limit=args.limit)
    param_space = default_param_space(include_arrays=args.include_arrays)

    if args.base_params and Path(args.base_params).exists():
        with Path(args.base_params).open("r") as fh:
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

    study.optimize(
        objective, n_trials=args.trials, timeout=args.timeout, n_jobs=args.jobs
    )

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
        with args.best_out.open("w") as fh:
            yaml.safe_dump(merged, fh)
        print(f"wrote best params to {args.best_out}")

    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_out.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["number", "loss", "texel_scale", "error_count"])
            for t in study.trials:
                if t.state != optuna.trial.TrialState.COMPLETE:
                    continue
                writer.writerow(
                    [
                        t.number,
                        t.value,
                        t.user_attrs.get("texel_scale"),
                        t.user_attrs.get("error_count", 0),
                    ]
                )
        print(f"wrote metrics to {args.metrics_out}")

    if args.plot_out:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skipping plot")
        else:
            xs = []
            ys = []
            for t in study.trials:
                if t.state == optuna.trial.TrialState.COMPLETE:
                    xs.append(t.number)
                    ys.append(t.value)
            plt.figure(figsize=(8, 4))
            plt.plot(xs, ys, marker="o", linestyle="-", linewidth=1)
            plt.xlabel("trial")
            plt.ylabel("texel loss")
            plt.title("skaks texel tuning")
            plt.tight_layout()
            args.plot_out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(args.plot_out)
            print(f"wrote loss plot to {args.plot_out}")


if __name__ == "__main__":  # pragma: no cover
    main()
