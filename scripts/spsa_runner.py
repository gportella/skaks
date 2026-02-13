#!/usr/bin/env python
"""Run SPSA tuning by calling fastchess_wrapper.py."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from spsa import SPSAState

try:
    from rich.console import Console
    from rich.progress import (BarColumn, Progress, SpinnerColumn, TextColumn,
                               TimeElapsedColumn)
except Exception:  # pragma: no cover
    Console = None
    Progress = None
    BarColumn = None
    SpinnerColumn = None
    TextColumn = None
    TimeElapsedColumn = None


@dataclass
class ParamSpec:
    name: str
    value: float
    param_type: str
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    step: float = 1.0
    is_float: bool = False

    def clamp(self, value: float) -> float:
        if self.min_val is not None:
            value = max(value, self.min_val)
        if self.max_val is not None:
            value = min(value, self.max_val)
        return value

    def snap(self, value: float) -> float:
        if self.step <= 0:
            return value
        base = self.min_val if self.min_val is not None else 0.0
        return round((value - base) / self.step) * self.step + base

    def normalize(self, value: float) -> float:
        value = self.clamp(value)
        value = self.snap(value)
        return self.clamp(value)

    def format_value(self, value: float) -> str:
        value = self.normalize(value)
        if self.param_type == "spin" and not self.is_float:
            return str(int(round(value)))
        if self.param_type == "string" or self.is_float:
            return f"{value:.3f}"
        return str(value)

    def base_name(self) -> str:
        if self.name.startswith("search_nnue."):
            return self.name.split("search_nnue.", 1)[1]
        return self.name


def default_param_specs() -> List[ParamSpec]:
    return [
        ParamSpec("aspiration_window_initial", 80, "spin", 50, 200, step=10),
        ParamSpec("aspiration_window_max", 400, "spin", 400, 800, step=20),
        ParamSpec("quiescence_delta_margin", 330, "spin", 240, 420, step=5),
        ParamSpec("quiescence_max_ply", 8, "spin", 8, 10),
        ParamSpec("quiescence_max_noisy_moves", 21, "spin", 12, 30),
        ParamSpec("quiescence_zero_gain_skip_index", 3, "spin", 0, 6),
        ParamSpec("quiescence_max_quiet_checks", 6, "spin", 0, 12),
        ParamSpec("null_move_reduction", 3, "spin", 2, 5),
        ParamSpec("null_move_min_depth", 6, "spin", 4, 6),
        ParamSpec("lmr_intercept", 0.5, "string", -0.5, 1.5, step=0.05, is_float=True),
        ParamSpec("lmr_divisor", 1.55, "string", 0.6, 2.5, step=0.05, is_float=True),
        ParamSpec("lmr_history_divisor", 11000.0, "string", 2000.0, 20000.0, step=500.0, is_float=True),
        ParamSpec("lmr_pv_offset", 1.25, "string", 0.0, 2.5, step=0.05, is_float=True),
        ParamSpec("futility_margins_0", 0, "spin", 0, 0, step=1),
        ParamSpec("futility_margins_1", 120, "spin", 80, 200, step=5),
        ParamSpec("futility_margins_2", 300, "spin", 200, 500, step=10),
        ParamSpec("futility_margins_3", 800, "spin", 600, 1000, step=20),
    ]


def futility_margin_specs(specs: Sequence[ParamSpec]) -> List[ParamSpec]:
    ordered: List[Tuple[int, ParamSpec]] = []
    for spec in specs:
        base = spec.base_name()
        if base.startswith("futility_margins_"):
            tail = base.split("futility_margins_", 1)[1]
            if tail.isdigit():
                ordered.append((int(tail), spec))
    return [spec for _, spec in sorted(ordered, key=lambda item: item[0])]


def load_params(params_path: str, only: Optional[Iterable[str]]) -> List[ParamSpec]:
    defaults = default_param_specs()
    specs = clone_specs(defaults)
    defaults_by_name = {spec.name: spec for spec in specs}

    if params_path and os.path.exists(params_path):
        with open(params_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        for item in data.get("params", []):
            name = item["name"]
            if name in defaults_by_name:
                spec = defaults_by_name[name]
                spec.value = float(item.get("value", spec.value))
                spec.param_type = item.get("type", spec.param_type)
                spec.min_val = item.get("min", spec.min_val)
                spec.max_val = item.get("max", spec.max_val)
                spec.step = float(item.get("step", spec.step))
                spec.is_float = bool(item.get("is_float", spec.is_float))
            else:
                specs.append(
                    ParamSpec(
                        name=name,
                        value=float(item["value"]),
                        param_type=item.get("type", "spin"),
                        min_val=item.get("min"),
                        max_val=item.get("max"),
                        step=float(item.get("step", 1.0)),
                        is_float=bool(item.get("is_float", False)),
                    )
                )
    if only:
        wanted = {name.strip() for name in only}
        specs = [spec for spec in specs if spec.name in wanted]
    if not specs:
        raise ValueError("No tunable parameters found")
    return specs


def save_params(path: str, specs: Sequence[ParamSpec]) -> None:
    payload = {
        "params": [
            {
                "name": spec.name,
                "value": spec.value,
                "type": spec.param_type,
                "min": spec.min_val,
                "max": spec.max_val,
                "step": spec.step,
                "is_float": spec.is_float,
            }
            for spec in specs
        ]
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_options_file(path: str, specs: Sequence[ParamSpec]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        futility_specs = futility_margin_specs(specs)
        futility_names = {spec.name for spec in futility_specs}
        for spec in specs:
            if spec.name in futility_names:
                continue
            handle.write(f"{spec.base_name()}={spec.format_value(spec.value)}\n")
        if futility_specs:
            values = ",".join(spec.format_value(spec.value) for spec in futility_specs)
            handle.write(f"futility_margins={values}\n")


def parse_objective(text: str, objective: str) -> float:
    if objective == "llr":
        llr_pattern = re.compile(r"LLR:\s*([-0-9.]+)")
        matches = llr_pattern.findall(text)
        if not matches:
            raise ValueError("LLR not found in output")
        return float(matches[-1])

    points_pattern = re.compile(r"Points:\s*([0-9.]+)\s*\(([-0-9.]+)\s*%\)")
    matches = points_pattern.findall(text)
    if not matches:
        raise ValueError("Points not found in output")
    _, pct = matches[-1]
    return float(pct)


def parse_points_pct(text: str) -> Optional[float]:
    points_pattern = re.compile(r"Points:\s*([0-9.]+)\s*\(([-0-9.]+)\s*%\)")
    matches = points_pattern.findall(text)
    if not matches:
        return None
    _, pct = matches[-1]
    return float(pct)


def parse_llr_bounds(text: str) -> Optional[Tuple[float, float, float]]:
    llr_pattern = re.compile(r"LLR:\s*([-0-9.]+).*?\(([-0-9.]+),\s*([-0-9.]+)\)")
    matches = llr_pattern.findall(text)
    if not matches:
        return None
    llr, lower, upper = matches[-1]
    return float(llr), float(lower), float(upper)


def format_yaml_value(spec: ParamSpec) -> str:
    value = spec.normalize(spec.value)
    if spec.base_name().startswith("lmr_") or spec.is_float or spec.param_type == "string":
        formatted = f"{value:.3f}".rstrip("0").rstrip(".")
        if "." not in formatted:
            formatted = f"{formatted}.0"
        return formatted
    return str(int(round(value)))


def write_best_yaml(path: str, specs: Sequence[ParamSpec]) -> None:
    lines: List[str] = ["search_nnue:"]
    futility_specs = futility_margin_specs(specs)
    futility_names = {spec.name for spec in futility_specs}
    for spec in specs:
        if spec.name in futility_names:
            continue
        lines.append(f"  {spec.base_name()}: {format_yaml_value(spec)}")
    if futility_specs:
        values = ", ".join(format_yaml_value(spec) for spec in futility_specs)
        lines.append(f"  futility_margins: [{values}]")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def clone_specs(specs: Sequence[ParamSpec]) -> List[ParamSpec]:
    return [
        ParamSpec(
            name=spec.name,
            value=spec.value,
            param_type=spec.param_type,
            min_val=spec.min_val,
            max_val=spec.max_val,
            step=spec.step,
            is_float=spec.is_float,
        )
        for spec in specs
    ]


def run_wrapper(
    wrapper_path: str,
    args: Sequence[str],
    log_path: str,
    dry_run: bool,
    keep_artifacts: bool,
    timeout_seconds: Optional[int],
    work_dir: Optional[str] = None,
    retry_once: bool = True,
) -> str:
    cmd = [sys.executable, wrapper_path, "--quiet"] + list(args) + ["--output-file", log_path]
    if dry_run:
        return ""
    if work_dir:
        os.makedirs(work_dir, exist_ok=True)
    attempts = 0
    while True:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            cwd=work_dir,
        )
        try:
            _, stderr = proc.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise RuntimeError(f"Timeout after {timeout_seconds}s while running fastchess")
        if proc.returncode == 0:
            with open(log_path, "r", encoding="utf-8") as handle:
                content = handle.read()
            if not keep_artifacts:
                try:
                    os.remove(log_path)
                except OSError:
                    pass
                if work_dir:
                    try:
                        for name in os.listdir(work_dir):
                            os.remove(os.path.join(work_dir, name))
                        os.rmdir(work_dir)
                    except OSError:
                        pass
            return content

        tail = ""
        if log_path and os.path.exists(log_path):
            try:
                with open(log_path, "r", encoding="utf-8") as handle:
                    lines = handle.readlines()
                if lines:
                    tail_lines = lines[-40:]
                    tail = "\n".join(line.rstrip("\n") for line in tail_lines)
            except OSError:
                tail = ""
        err = (stderr or "").strip() or "fastchess wrapper failed"
        if tail:
            err = f"{err}\n\nFastchess output tail:\n{tail}"

        should_retry = False
        if retry_once and attempts == 0 and tail:
            lowered = tail.lower()
            for token in ("not responsive", "stalled", "disconnected", "interrupted"):
                if token in lowered:
                    should_retry = True
                    break
        if should_retry:
            attempts += 1
            continue
        raise RuntimeError(err)


def strip_flag_args(base_args: Sequence[str], flags: Sequence[str]) -> List[str]:
    if not flags:
        return list(base_args)
    flag_set = set(flags)
    return [arg for arg in base_args if arg not in flag_set]


def strip_flags_with_values(args: Sequence[str], flags_with_values: Sequence[str]) -> List[str]:
    if not flags_with_values:
        return list(args)
    flag_set = set(flags_with_values)
    out: List[str] = []
    skip_next = False
    for item in args:
        if skip_next:
            skip_next = False
            continue
        if item in flag_set:
            skip_next = True
            continue
        out.append(item)
    return out


def get_flag_value(args: Sequence[str], flag: str) -> Optional[str]:
    for idx, item in enumerate(args):
        if item == flag and idx + 1 < len(args):
            return args[idx + 1]
    return None


def ensure_force_concurrency(args: List[str], parallel_evals: bool) -> List[str]:
    cpu_count = os.cpu_count() or 1
    conc_value = get_flag_value(args, "--concurrency")
    if conc_value is None:
        return args
    try:
        concurrency = int(conc_value)
    except ValueError:
        return args
    need_force = concurrency > cpu_count
    if parallel_evals:
        need_force = need_force or (concurrency * 2 > cpu_count)
    if need_force and "--force-concurrency" not in args:
        return list(args) + ["--force-concurrency"]
    return args


def randomize_specs(specs: Sequence[ParamSpec], rng: random.Random) -> List[ParamSpec]:
    randomized: List[ParamSpec] = []
    for spec in specs:
        if spec.min_val is not None and spec.max_val is not None:
            value = rng.uniform(spec.min_val, spec.max_val)
            value = spec.normalize(value)
        else:
            value = spec.value
        randomized.append(
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
    return randomized


def evaluate_candidate(
    wrapper_path: str,
    base_args: Sequence[str],
    specs: Sequence[ParamSpec],
    run_dir: str,
    keep_artifacts: bool,
    timeout_seconds: Optional[int],
    tag: str,
    objective: str,
) -> Tuple[float, Optional[Tuple[float, float, float]], Optional[float]]:
    options_path = os.path.join(run_dir, f"{tag}.options")
    log_path = os.path.join(run_dir, f"{tag}.log")
    work_dir = os.path.join(run_dir, f"{tag}.work")
    write_options_file(options_path, specs)
    args = list(base_args) + ["--test-options-file", options_path]
    out = run_wrapper(wrapper_path, args, log_path, False, keep_artifacts, timeout_seconds, work_dir)
    llr_bounds = parse_llr_bounds(out)
    wdl_pct = parse_points_pct(out)
    metric = parse_objective(out, objective)
    if not keep_artifacts:
        for path in (options_path, log_path):
            try:
                os.remove(path)
            except OSError:
                pass
    return metric, llr_bounds, wdl_pct


def with_baseline_options(base_args: Sequence[str], options_file: Optional[str]) -> List[str]:
    args: List[str] = []
    skip = False
    for item in base_args:
        if skip:
            skip = False
            continue
        if item == "--baseline-options-file":
            skip = True
            continue
        args.append(item)
    if options_file:
        args.extend(["--baseline-options-file", options_file])
    return args


def main() -> int:
    parser = argparse.ArgumentParser(description="SPSA driver for fastchess_wrapper.py")
    parser.add_argument("--params-file", default="spsa_params.json")
    parser.add_argument("--best-file", default="best_params.yaml")
    parser.add_argument("--only", action="append", help="restrict to parameter name")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--objective", choices=["points", "llr"], default="points")
    parser.add_argument("--maximize", action="store_true", default=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--state-file", default="spsa_state.json")
    parser.add_argument("--run-dir", default="spsa_runs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-artifacts", action="store_true", help="keep per-iteration logs/options")
    parser.add_argument("--spsa-a", type=float, help="override SPSA a (step size)")
    parser.add_argument("--spsa-c", type=float, help="override SPSA c (perturbation size)")
    parser.add_argument("--spsa-big-a", type=float, help="override SPSA A (stability constant)")
    parser.add_argument("--accept-wdl", type=float, default=0.0, help="accept if WDL%% >= 50 + this threshold")
    parser.add_argument(
        "--random-starts",
        type=int,
        default=0,
        help="number of random candidates to try before SPSA",
    )
    parser.add_argument(
        "--restart-stagnation",
        type=int,
        default=0,
        help="randomize params after N iterations without accept",
    )
    parser.add_argument(
        "--parallel-evals",
        action="store_true",
        help="run + and - evaluations concurrently",
    )
    parser.add_argument(
        "--reject-wdl-drop",
        action="store_true",
        help="reject acceptance when WDL%% is lower than previous iteration",
    )
    parser.add_argument(
        "--monotonic-wdl",
        action="store_true",
        help="revert parameter updates when WDL%% drops vs previous iteration",
    )
    parser.add_argument("wrapper_args", nargs=argparse.REMAINDER)
    parser.add_argument("--wrapper-timeout", type=int, default=0, help="kill a test if it exceeds N seconds (0 disables)")

    args = parser.parse_args()

    base_args = list(args.wrapper_args)
    if base_args and base_args[0] == "--":
        base_args = base_args[1:]

    wrapper_path = os.path.join(os.path.dirname(__file__), "fastchess_wrapper.py")

    specs = load_params(args.params_file, args.only)
    theta = [spec.value for spec in specs]

    if os.path.exists(args.state_file):
        with open(args.state_file, "r", encoding="utf-8") as handle:
            state_data = json.load(handle)
        state = SPSAState(**{k: state_data[k] for k in state_data if k in SPSAState.__dataclass_fields__})
    else:
        state = SPSAState(rng_seed=args.seed)

    env_a = os.getenv("SPSA_A")
    env_c = os.getenv("SPSA_C")
    env_big_a = os.getenv("SPSA_BIG_A")
    if args.spsa_a is not None or env_a is not None:
        state.a = float(args.spsa_a if args.spsa_a is not None else env_a)
    if args.spsa_c is not None or env_c is not None:
        state.c = float(args.spsa_c if args.spsa_c is not None else env_c)
    if args.spsa_big_a is not None or env_big_a is not None:
        state.A = float(args.spsa_big_a if args.spsa_big_a is not None else env_big_a)

    os.makedirs(args.run_dir, exist_ok=True)

    console = Console(log_path=False) if Console is not None else None
    progress = None
    task_id = None
    if Progress is not None:
        progress = Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=None, style="green"),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        )
        progress.start()
        task_id = progress.add_task("SPSA iterations", total=args.iterations)

    prev_metric: Optional[float] = None
    prev_wdl: Optional[float] = None

    best_specs = clone_specs(specs)
    best_options_file = os.path.join(args.run_dir, "best.options")
    write_best_yaml(args.best_file, best_specs)
    write_options_file(best_options_file, best_specs)

    sprt_flags_with_values = [
        "--sprt-elo0",
        "--sprt-elo1",
        "--sprt-alpha",
        "--sprt-beta",
        "--sprt-model",
    ]
    sprt_present = "--sprt" in base_args

    base_with_best = with_baseline_options(base_args, best_options_file)
    base_with_best = ensure_force_concurrency(base_with_best, args.parallel_evals)
    base_for_grad = strip_flag_args(base_with_best, ["--sprt"])
    base_for_grad = strip_flags_with_values(base_for_grad, sprt_flags_with_values)
    base_for_grad = list(base_for_grad) + ["--no-sprt"]

    rng_seed = args.seed if args.seed is not None else state.rng_seed
    rng = random.Random(rng_seed)
    if args.random_starts > 0 and not args.dry_run:
        best_metric = float("-inf")
        best_candidate = clone_specs(specs)
        if progress is not None:
            progress.log(f"Warmup: trying {args.random_starts} random candidates")
        for idx in range(args.random_starts):
            candidate = randomize_specs(specs, rng)
            tag = f"warmup_{idx+1:03d}_{int(time.time())}"
            metric, _, _ = evaluate_candidate(
                wrapper_path,
                base_for_grad,
                candidate,
                args.run_dir,
                args.keep_artifacts,
                args.wrapper_timeout or None,
                tag,
                args.objective,
            )
            if metric > best_metric:
                best_metric = metric
                best_candidate = clone_specs(candidate)
        specs = clone_specs(best_candidate)
        theta = [spec.value for spec in specs]
        save_params(args.params_file, specs)

    no_accept_count = 0

    for _ in range(args.iterations):
        theta_prev = list(theta)
        specs_prev = clone_specs(specs)
        theta_plus, theta_minus, deltas, c_k = state.propose(theta)

        plus_specs: List[ParamSpec] = []
        minus_specs: List[ParamSpec] = []
        bounds: List[Tuple[float, float]] = []
        for spec, delta in zip(specs, deltas):
            step = spec.step if spec.step > 0 else 1.0
            perturb = max(c_k, step) * delta
            p_val = spec.normalize(spec.value + perturb)
            m_val = spec.normalize(spec.value - perturb)
            plus_specs.append(
                ParamSpec(
                    spec.name,
                    p_val,
                    spec.param_type,
                    spec.min_val,
                    spec.max_val,
                    step=spec.step,
                    is_float=spec.is_float,
                )
            )
            minus_specs.append(
                ParamSpec(
                    spec.name,
                    m_val,
                    spec.param_type,
                    spec.min_val,
                    spec.max_val,
                    step=spec.step,
                    is_float=spec.is_float,
                )
            )
            if spec.min_val is not None or spec.max_val is not None:
                bounds.append((spec.min_val or float("-inf"), spec.max_val or float("inf")))
            else:
                bounds.append((float("-inf"), float("inf")))

        iteration_tag = f"iter_{state.iteration:03d}_{int(time.time())}"
        plus_options = os.path.join(args.run_dir, f"{iteration_tag}_plus.options")
        minus_options = os.path.join(args.run_dir, f"{iteration_tag}_minus.options")
        plus_log = os.path.join(args.run_dir, f"{iteration_tag}_plus.log")
        minus_log = os.path.join(args.run_dir, f"{iteration_tag}_minus.log")

        write_options_file(plus_options, plus_specs)
        write_options_file(minus_options, minus_specs)

        base_with_best = with_baseline_options(base_args, best_options_file)
        base_with_best = ensure_force_concurrency(base_with_best, args.parallel_evals)
        base_for_grad = strip_flag_args(base_with_best, ["--sprt"])
        base_for_grad = strip_flags_with_values(base_for_grad, sprt_flags_with_values)
        base_for_grad = list(base_for_grad) + ["--no-sprt"]
        plus_args = list(base_for_grad) + ["--test-options-file", plus_options]
        minus_args = list(base_for_grad) + ["--test-options-file", minus_options]

        if progress is not None:
            progress.log(f"Iteration {state.iteration + 1}: running + evaluation")

        plus_work_dir = os.path.join(args.run_dir, f"{iteration_tag}_plus.work")
        minus_work_dir = os.path.join(args.run_dir, f"{iteration_tag}_minus.work")

        if args.dry_run:
            run_wrapper(wrapper_path, plus_args, plus_log, True, args.keep_artifacts, None, plus_work_dir)
            run_wrapper(wrapper_path, minus_args, minus_log, True, args.keep_artifacts, None, minus_work_dir)
            if progress is not None:
                progress.stop()
            return 0

        timeout_seconds = args.wrapper_timeout or None
        if args.parallel_evals:
            if progress is not None:
                progress.log(f"Iteration {state.iteration + 1}: running +/- evaluations in parallel")
            results = {}
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {
                    pool.submit(
                        run_wrapper,
                        wrapper_path,
                        plus_args,
                        plus_log,
                        False,
                        args.keep_artifacts,
                        timeout_seconds,
                        plus_work_dir,
                    ): "plus",
                    pool.submit(
                        run_wrapper,
                        wrapper_path,
                        minus_args,
                        minus_log,
                        False,
                        args.keep_artifacts,
                        timeout_seconds,
                        minus_work_dir,
                    ): "minus",
                }
                for fut in as_completed(futures):
                    label = futures[fut]
                    try:
                        results[label] = fut.result()
                    except Exception as exc:
                        raise RuntimeError(f"{label} evaluation failed: {exc}") from exc
            plus_out = results["plus"]
            minus_out = results["minus"]
        else:
            plus_out = run_wrapper(
                wrapper_path,
                plus_args,
                plus_log,
                False,
                args.keep_artifacts,
                timeout_seconds,
                plus_work_dir,
            )

            if progress is not None:
                progress.log(f"Iteration {state.iteration + 1}: running - evaluation")
            minus_out = run_wrapper(
                wrapper_path,
                minus_args,
                minus_log,
                False,
                args.keep_artifacts,
                timeout_seconds,
                minus_work_dir,
            )
        if not args.keep_artifacts:
            for path in (plus_options, minus_options):
                try:
                    os.remove(path)
                except OSError:
                    pass

        y_plus = parse_objective(plus_out, args.objective)
        y_minus = parse_objective(minus_out, args.objective)

        theta = state.update(
            theta,
            deltas,
            y_plus,
            y_minus,
            maximize=args.maximize,
            bounds=bounds,
        )

        for spec, new_val in zip(specs, theta):
            spec.value = new_val

        save_params(args.params_file, specs)
        with open(args.state_file, "w", encoding="utf-8") as handle:
            json.dump({
                "a": state.a,
                "c": state.c,
                "A": state.A,
                "alpha": state.alpha,
                "gamma": state.gamma,
                "iteration": state.iteration,
                "rng_seed": state.rng_seed,
            }, handle, indent=2)
            handle.write("\n")

        wdl_plus = parse_points_pct(plus_out)
        wdl_minus = parse_points_pct(minus_out)
        wdl_values = [val for val in (wdl_plus, wdl_minus) if val is not None]
        wdl_pct = sum(wdl_values) / len(wdl_values) if wdl_values else None
        metric = wdl_pct if wdl_pct is not None else (y_plus + y_minus) / 2.0
        delta = None if prev_metric is None else metric - prev_metric
        wdl_delta = None if prev_wdl is None or wdl_pct is None else wdl_pct - prev_wdl
        monotonic_revert = False
        if args.monotonic_wdl and prev_wdl is not None and wdl_pct is not None and wdl_pct < prev_wdl:
            theta = theta_prev
            specs = clone_specs(specs_prev)
            save_params(args.params_file, specs)
            wdl_pct = prev_wdl
            metric = prev_wdl
            delta = 0.0
            wdl_delta = 0.0
            monotonic_revert = True
        prev_metric = metric
        if wdl_pct is not None:
            prev_wdl = wdl_pct

        heartbeat_path = os.path.join(args.run_dir, "last_iteration.json")
        with open(heartbeat_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "iteration": state.iteration,
                    "metric": metric,
                    "delta": delta,
                },
                handle,
                indent=2,
            )
            handle.write("\n")

        status = ""
        if wdl_delta is not None:
            status = f"WDL% Δ={wdl_delta:+.2f} vs prev"
        elif delta is not None:
            status = f"WDL% Δ={delta:+.2f} vs prev"
        if monotonic_revert and status:
            status = f"{status} (reverted)"
        accept_plus = wdl_plus is not None and wdl_plus >= 50.0
        accept_minus = wdl_minus is not None and wdl_minus >= 50.0
        if args.accept_wdl > 0:
            accept_plus = accept_plus and wdl_plus is not None and wdl_plus >= 50.0 + args.accept_wdl
            accept_minus = accept_minus and wdl_minus is not None and wdl_minus >= 50.0 + args.accept_wdl
        if args.reject_wdl_drop and prev_wdl is not None:
            accept_plus = accept_plus and wdl_plus is not None and wdl_plus >= prev_wdl
            accept_minus = accept_minus and wdl_minus is not None and wdl_minus >= prev_wdl
        if monotonic_revert:
            accept_plus = False
            accept_minus = False

        accepted = False
        accepted_label = ""
        if accept_plus or accept_minus:
            chosen_specs = plus_specs
            chosen_metric = y_plus
            if accept_minus and y_minus > chosen_metric:
                chosen_specs = minus_specs
                chosen_metric = y_minus
            accepted_label = "+" if chosen_specs is plus_specs else "-"

            if sprt_present:
                gate_tag = f"gate_{state.iteration:03d}_{int(time.time())}"
                gate_metric, gate_llr, gate_wdl = evaluate_candidate(
                    wrapper_path,
                    base_with_best,
                    chosen_specs,
                    args.run_dir,
                    args.keep_artifacts,
                    args.wrapper_timeout or None,
                    gate_tag,
                    "llr",
                )
                gate_accept = gate_llr is not None and gate_llr[0] >= gate_llr[2]
                if args.accept_wdl > 0:
                    gate_accept = gate_accept and gate_wdl is not None and gate_wdl >= 50.0 + args.accept_wdl
                if gate_wdl is not None:
                    gate_accept = gate_accept and gate_wdl >= 50.0
                if gate_accept:
                    accepted = True
            else:
                accepted = True

        if accepted:
            best_specs = clone_specs(chosen_specs)
            write_best_yaml(args.best_file, best_specs)
            write_options_file(best_options_file, best_specs)
            specs = clone_specs(best_specs)
            theta = [spec.value for spec in specs]
            save_params(args.params_file, specs)
            prev_metric = None
            prev_wdl = None
            no_accept_count = 0
        else:
            no_accept_count += 1

        if args.restart_stagnation > 0 and no_accept_count >= args.restart_stagnation:
            if progress is not None:
                progress.log("Stagnation detected: randomizing parameters")
            specs = randomize_specs(specs, rng)
            theta = [spec.value for spec in specs]
            save_params(args.params_file, specs)
            prev_metric = None
            prev_wdl = None
            no_accept_count = 0

        if progress is not None and task_id is not None:
            desc = f"SPSA iterations | WDL%≈{metric:.2f}"
            progress.update(task_id, advance=1, description=desc)
            if status:
                progress.log(f"Iteration {state.iteration}: {status}")
            if accepted:
                reason = "SPRT LLR" if sprt_present else "WDL%"
                progress.log(f"Accepted new baseline ({accepted_label}) based on {reason}")
        elif console is not None:
            console.print(
                f"Iteration {state.iteration} complete: y+={y_plus:.3f}, y-={y_minus:.3f}, WDL%≈{metric:.2f} {status}",
                style="green" if delta is None or delta >= 0 else "red",
            )
            if accepted:
                reason = "SPRT LLR" if sprt_present else "WDL%"
                console.print(
                    f"Accepted new baseline ({accepted_label}) based on {reason}",
                    style="bold green",
                )
        else:
            print(
                f"Iteration {state.iteration} complete: y+={y_plus:.3f}, y-={y_minus:.3f}, WDL%≈{metric:.2f} {status}"
            )
            if accepted:
                reason = "SPRT LLR" if sprt_present else "WDL%"
                print(f"Accepted new baseline ({accepted_label}) based on {reason}")

    if progress is not None:
        progress.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
