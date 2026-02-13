#!/usr/bin/env python
"""Optuna (TPE) tuner for fastchess_wrapper.py."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional, Sequence

try:
    import optuna
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Optuna is required for this runner. Install it in your Python environment."
    ) from exc

from spsa_runner import (ParamSpec, default_param_specs,
                         ensure_force_concurrency, load_params,
                         parse_objective, parse_points_pct, run_wrapper,
                         strip_flag_args, strip_flags_with_values,
                         write_best_yaml, write_options_file)


def suggest_from_spec(trial: optuna.Trial, spec: ParamSpec) -> float:
    if spec.min_val is None or spec.max_val is None:
        return spec.value

    if spec.param_type == "spin" and not spec.is_float:
        step = int(spec.step) if spec.step and spec.step > 0 else 1
        return float(
            trial.suggest_int(spec.name, int(spec.min_val), int(spec.max_val), step=step)
        )

    step = spec.step if spec.step and spec.step > 0 else None
    return float(
        trial.suggest_float(spec.name, float(spec.min_val), float(spec.max_val), step=step)
    )


def build_trial_specs(base_specs: Sequence[ParamSpec], trial: optuna.Trial) -> List[ParamSpec]:
    specs: List[ParamSpec] = []
    for spec in base_specs:
        value = suggest_from_spec(trial, spec)
        value = spec.normalize(value)
        specs.append(
            ParamSpec(
                spec.name,
                value,
                spec.param_type,
                spec.min_val,
                spec.max_val,
                step=spec.step,
                is_float=spec.is_float,
            )
        )
    return specs


def main() -> int:
    parser = argparse.ArgumentParser(description="Optuna TPE tuner for fastchess_wrapper.py")
    parser.add_argument("--params-file", default="spsa_params.json")
    parser.add_argument("--best-file", default="best_params.yaml")
    parser.add_argument("--only", action="append", help="restrict to parameter name")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--objective", choices=["points", "llr"], default="points")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-dir", default="spsa_runs")
    parser.add_argument("--study-name", default="optuna_tpe")
    parser.add_argument("--storage", default="", help="optuna storage URL (optional)")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--keep-artifacts", action="store_true", help="keep per-trial logs/options")
    parser.add_argument("--wrapper-timeout", type=int, default=0, help="kill a test if it exceeds N seconds (0 disables)")
    parser.add_argument("wrapper_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    base_args = list(args.wrapper_args)
    if base_args and base_args[0] == "--":
        base_args = base_args[1:]

    wrapper_path = os.path.join(os.path.dirname(__file__), "fastchess_wrapper.py")

    specs = load_params(args.params_file, args.only)
    os.makedirs(args.run_dir, exist_ok=True)

    sprt_flags_with_values = [
        "--sprt-elo0",
        "--sprt-elo1",
        "--sprt-alpha",
        "--sprt-beta",
        "--sprt-model",
    ]
    base_for_eval = strip_flag_args(base_args, ["--sprt"])
    base_for_eval = strip_flags_with_values(base_for_eval, sprt_flags_with_values)
    base_for_eval = list(base_for_eval) + ["--no-sprt"]
    base_for_eval = ensure_force_concurrency(list(base_for_eval), parallel_evals=False)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=max(5, args.trials // 10))

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=args.study_name,
        storage=args.storage or None,
        load_if_exists=bool(args.storage),
    )

    best_options_file = os.path.join(args.run_dir, "best.options")

    def objective(trial: optuna.Trial) -> float:
        trial_specs = build_trial_specs(specs, trial)
        tag = f"trial_{trial.number:04d}_{int(time.time())}"
        options_path = os.path.join(args.run_dir, f"{tag}.options")
        log_path = os.path.join(args.run_dir, f"{tag}.log")
        work_dir = os.path.join(args.run_dir, f"{tag}.work")
        write_options_file(options_path, trial_specs)
        trial_args = list(base_for_eval) + ["--test-options-file", options_path]
        out = run_wrapper(
            wrapper_path,
            trial_args,
            log_path,
            False,
            args.keep_artifacts,
            args.wrapper_timeout or None,
            work_dir,
        )
        value = parse_objective(out, args.objective)
        wdl = parse_points_pct(out)
        if wdl is not None:
            trial.set_user_attr("wdl", wdl)
        return value

    def on_trial_end(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if study.best_trial.number != trial.number:
            return
        best_specs = build_trial_specs(specs, trial)
        write_best_yaml(args.best_file, best_specs)
        write_options_file(best_options_file, best_specs)

    study.optimize(objective, n_trials=args.trials, n_jobs=args.n_jobs, callbacks=[on_trial_end])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
