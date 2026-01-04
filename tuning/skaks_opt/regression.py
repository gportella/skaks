from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import random
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .progress import FancyProgress, RICH_AVAILABLE


@dataclass
class EvalSample:
    term_values: List[int]
    mg_ratio: float
    engine_eval: float
    raw_linear: float
    term_names: List[str]


@dataclass
class RegressionRow:
    target_cp: float
    features: np.ndarray
    original_eval: float


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0
        self.sum_xy = 0.0
        self.sum_abs_err = 0.0
        self.sum_sq_err = 0.0

    def add(self, target: float, actual: float) -> None:
        self.count += 1
        self.sum_x += target
        self.sum_y += actual
        self.sum_x2 += target * target
        self.sum_y2 += actual * actual
        self.sum_xy += target * actual
        error = target - actual
        self.sum_abs_err += abs(error)
        self.sum_sq_err += error * error

    def mae(self) -> float:
        if self.count == 0:
            return math.nan
        return self.sum_abs_err / self.count

    def rmse(self) -> float:
        if self.count == 0:
            return math.nan
        return math.sqrt(self.sum_sq_err / self.count)

    def pearson(self) -> float:
        if self.count == 0:
            return math.nan
        num = self.count * self.sum_xy - self.sum_x * self.sum_y
        den_x = self.count * self.sum_x2 - self.sum_x * self.sum_x
        den_y = self.count * self.sum_y2 - self.sum_y * self.sum_y
        denom = math.sqrt(max(den_x, 0.0) * max(den_y, 0.0))
        if denom == 0.0:
            return math.nan
        return num / denom


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit phase weights so Skaks static eval tracks Stockfish centipawns"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        nargs="+",
        help="Path(s) to CSV dataset file or directory containing CSV files",
    )
    parser.add_argument(
        "--skaks",
        default="build/debug/src/skaks",
        help="Path to the skaks executable",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Maximum number of sampled rows (after sampling)",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.1,
        help="Randomly sample this fraction of rows (0-1]; applied before limit",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="PRNG seed for sampling",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=1.0,
        help="Ridge regularization strength (lambda). Use 0 for unregularized least squares",
    )
    parser.add_argument(
        "--write-yaml",
        type=pathlib.Path,
        help="Optional path to write YAML overrides for phase weights",
    )
    parser.add_argument(
        "--pairs-out",
        type=pathlib.Path,
        help="Optional CSV path to store per-sample target, original, and fitted values",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Print progress every N samples when Rich is unavailable",
    )
    parser.add_argument(
        "--no-rich",
        action="store_true",
        help="Disable Rich progress output (falls back to plain logging)",
    )
    return parser


def iter_dataset_rows(path: pathlib.Path) -> Iterable[Tuple[str, float]]:
    if path.is_dir():
        for csv_path in sorted(path.rglob("*.csv")):
            yield from iter_dataset_rows(csv_path)
        return
    import csv

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            fen = row["fen"].strip()
            target = float(row["stockfish_cp"])
            yield fen, target


def sample_dataset(
    paths: Sequence[pathlib.Path],
    fraction: float,
    limit: int,
    rng: random.Random,
) -> Tuple[List[Tuple[str, float]], int]:
    rows: List[Tuple[str, float]] = []
    fraction = float(max(0.0, min(1.0, fraction)))
    total_seen = 0
    if fraction == 0.0 or limit == 0:
        return rows, total_seen
    for root in paths:
        for fen, target in iter_dataset_rows(root):
            total_seen += 1
            if fraction < 1.0 and rng.random() > fraction:
                continue
            rows.append((fen, target))
            if 0 < limit == len(rows):
                return rows, total_seen
    return rows, total_seen


def fetch_eval_breakdown(skaks: pathlib.Path, fen: str) -> EvalSample:
    result = subprocess.run(
        [str(skaks), "--static-eval", "--eval-breakdown", "--fen", fen],
        capture_output=True,
        text=True,
        check=True,
    )

    payload: dict | None = None
    engine_eval = math.nan
    for line in result.stdout.splitlines():
        if line.startswith("eval_terms "):
            payload = json.loads(line.split(" ", 1)[1])
        elif line.startswith("static_eval_white "):
            engine_eval = float(line.split()[1])

    if payload is None:
        raise RuntimeError(
            "Engine did not emit eval_terms payload.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    term_values = [int(v) for v in payload["term_values"]]
    mg_ratio = float(payload["mg_ratio"])
    raw_linear = float(payload.get("raw_linear", engine_eval))
    term_names = [str(name) for name in payload.get("term_names", [])]
    if math.isnan(engine_eval):
        engine_eval = float(payload.get("static_eval_white", 0.0))

    return EvalSample(
        term_values=term_values,
        mg_ratio=mg_ratio,
        engine_eval=engine_eval,
        raw_linear=raw_linear,
        term_names=term_names,
    )


def build_feature_vector(sample: EvalSample) -> np.ndarray:
    term_values = np.asarray(sample.term_values, dtype=np.float64)
    mg_component = term_values * sample.mg_ratio
    base_component = term_values
    return np.concatenate((base_component, mg_component))


def collect_rows(
    skaks: pathlib.Path,
    rows: Sequence[Tuple[str, float]],
    progress_interval: int,
    rng: random.Random,
    use_rich: bool,
) -> Tuple[List[RegressionRow], List[str], RunningStats]:
    collected: List[RegressionRow] = []
    errors: List[str] = []
    stats = RunningStats()

    if RICH_AVAILABLE and use_rich:
        with FancyProgress(
            total=len(rows), rng=rng, description="Evaluating positions"
        ) as fancy:
            for idx, (fen, target) in enumerate(rows, start=1):
                try:
                    eval_sample = fetch_eval_breakdown(skaks, fen)
                except Exception as exc:  # pragma: no cover - defensive
                    errors.append(f"{fen}: {exc}")
                    continue
                features = build_feature_vector(eval_sample)
                stats.add(target, eval_sample.engine_eval)
                collected.append(
                    RegressionRow(
                        target_cp=float(target),
                        features=features,
                        original_eval=eval_sample.engine_eval,
                    )
                )
                status = _format_status(stats)
                fancy.update(
                    idx,
                    message=f"[cyan]{idx}/{len(rows)}[/] positions",
                    status=status,
                )
    else:
        for idx, (fen, target) in enumerate(rows, start=1):
            try:
                eval_sample = fetch_eval_breakdown(skaks, fen)
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(f"{fen}: {exc}")
                continue
            features = build_feature_vector(eval_sample)
            stats.add(target, eval_sample.engine_eval)
            collected.append(
                RegressionRow(
                    target_cp=float(target),
                    features=features,
                    original_eval=eval_sample.engine_eval,
                )
            )
            if progress_interval > 0 and (
                idx == 1 or idx == len(rows) or idx % progress_interval == 0
            ):
                print(
                    f"[{idx}/{len(rows)}] target={target:+.1f} engine={eval_sample.engine_eval:+.1f} {_format_status(stats)}"
                )
    return collected, errors, stats


def _format_status(stats: RunningStats) -> str:
    r = stats.pearson()
    rmse = stats.rmse()
    mae = stats.mae()
    return f"r={r:+.3f} rmse={rmse:.1f} mae={mae:.1f}" if stats.count else ""


def solve_regression(rows: Sequence[RegressionRow], ridge_lambda: float) -> np.ndarray:
    if not rows:
        raise ValueError("No regression rows available")
    feature_matrix = np.vstack([row.features for row in rows])
    targets = np.array([row.target_cp for row in rows], dtype=np.float64)

    if ridge_lambda > 0:
        feat_dim = feature_matrix.shape[1]
        gram = feature_matrix.T @ feature_matrix
        rhs = feature_matrix.T @ targets
        reg = ridge_lambda * np.eye(feat_dim, dtype=np.float64)
        solution = np.linalg.solve(gram + reg, rhs)
    else:
        solution, *_ = np.linalg.lstsq(feature_matrix, targets, rcond=None)
    return solution


def apply_weights(rows: Sequence[RegressionRow], weights: np.ndarray) -> np.ndarray:
    feature_matrix = np.vstack([row.features for row in rows])
    return feature_matrix @ weights


def compute_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    baseline: Iterable[float],
) -> dict[str, float]:
    baseline_arr = np.asarray(baseline, dtype=np.float64)
    residuals = targets - predictions
    rmse = math.sqrt(float(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    pearson_reg = pearson(targets, predictions)
    pearson_base = pearson(targets, baseline_arr)
    return {
        "rmse": rmse,
        "mae": mae,
        "pearson_regressed": pearson_reg,
        "pearson_current": pearson_base,
    }


def pearson(xs: Iterable[float], ys: Iterable[float]) -> float:
    x = np.asarray(list(xs), dtype=np.float64)
    y = np.asarray(list(ys), dtype=np.float64)
    if x.size != y.size or x.size == 0:
        return math.nan
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    num = float(np.sum((x - x_mean) * (y - y_mean)))
    den = math.sqrt(float(np.sum((x - x_mean) ** 2)) * float(np.sum((y - y_mean) ** 2)))
    if den == 0.0:
        return math.nan
    return num / den


def reconstruct_phase_weights(weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    feature_count = weights.shape[0]
    if feature_count % 2 != 0:
        raise ValueError(
            "Unexpected number of feature weights; expected 2 * term_count"
        )
    half = feature_count // 2
    eg_weights = weights[:half]
    delta = weights[half:]
    mg_weights = eg_weights + delta
    return mg_weights, eg_weights


def fit_linear_calibration(
    targets: np.ndarray, raw_predictions: np.ndarray
) -> Tuple[float, float]:
    if raw_predictions.size == 0:
        return 0.0, 1.0
    design = np.column_stack((np.ones(raw_predictions.shape[0]), raw_predictions))
    coeffs, *_ = np.linalg.lstsq(design, targets, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def write_yaml(
    path: pathlib.Path,
    mg_weights: np.ndarray,
    eg_weights: np.ndarray,
    intercept: float,
    slope_factor: float,
    quiet_cap: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mg_list = ", ".join(f"{w:.6f}" for w in mg_weights)
    eg_list = ", ".join(f"{w:.6f}" for w in eg_weights)
    content = (
        "evaluation:\n"
        f"  eval_bias: {intercept:.6f}\n"
        f"  eval_slope: {slope_factor:.6f}\n"
        f"  eval_quiet_cap: {quiet_cap:.6f}\n"
        f"  phase_weights_mg: [{mg_list}]\n"
        f"  phase_weights_eg: [{eg_list}]\n"
    )
    path.write_text(content, encoding="utf-8")


def run_regression(args: argparse.Namespace) -> None:
    dataset_paths = [pathlib.Path(p) for p in args.dataset]
    skaks = pathlib.Path(args.skaks)
    rng = random.Random(args.seed)

    rows, total_seen = sample_dataset(
        dataset_paths, args.sample_fraction, args.limit, rng
    )
    if not rows:
        raise SystemExit("No rows selected; adjust --sample-fraction or --limit")

    collected, errors, baseline_stats = collect_rows(
        skaks, rows, args.progress_interval, rng, use_rich=not args.no_rich
    )
    if not collected:
        raise SystemExit("Engine produced no usable samples")

    targets = np.array([row.target_cp for row in collected], dtype=np.float64)
    baseline = np.array([row.original_eval for row in collected], dtype=np.float64)

    weights = solve_regression(collected, args.ridge)
    raw_predictions = apply_weights(collected, weights)
    intercept, slope_factor = fit_linear_calibration(targets, raw_predictions)
    predictions = intercept + slope_factor * raw_predictions
    metrics = compute_metrics(targets, predictions, baseline)

    mg_weights, eg_weights = reconstruct_phase_weights(weights)

    print(f"rows_scanned: {total_seen}")
    print(f"samples_used: {len(collected)}")
    print(f"ridge_lambda: {args.ridge}")
    print(f"baseline_pearson: {baseline_stats.pearson():.6f}")
    print(f"baseline_rmse: {baseline_stats.rmse():.3f}")
    print(f"baseline_mae: {baseline_stats.mae():.3f}")
    print(f"calibration_intercept: {intercept:.3f}")
    print(f"calibration_slope: {slope_factor:.6f}")
    quiet_cap = (
        float(np.percentile(np.abs(predictions), 95)) if len(predictions) else 800.0
    )
    quiet_cap = float(np.clip(quiet_cap, 200.0, 1200.0))
    print(f"quiet_cap: {quiet_cap:.3f}")
    print(f"rmse: {metrics['rmse']:.3f}")
    print(f"mae: {metrics['mae']:.3f}")
    print(f"pearson_current: {metrics['pearson_current']:.6f}")
    print(f"pearson_regressed: {metrics['pearson_regressed']:.6f}")

    print("phase_weights_mg:")
    for idx, value in enumerate(mg_weights):
        print(f"  [{idx:02d}] {value:.6f}")
    print("phase_weights_eg:")
    for idx, value in enumerate(eg_weights):
        print(f"  [{idx:02d}] {value:.6f}")

    if errors:
        print("warnings:")
        for message in errors:
            print(f"  {message}")

    if args.write_yaml:
        write_yaml(
            args.write_yaml,
            mg_weights,
            eg_weights,
            intercept,
            slope_factor,
            quiet_cap,
        )
        print(f"wrote_yaml: {args.write_yaml}")

    if args.pairs_out:
        _write_pairs_csv(
            args.pairs_out,
            targets,
            baseline,
            predictions,
        )
        print(f"pairs_csv: {args.pairs_out}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_regression(args)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])


def _write_pairs_csv(
    path: pathlib.Path,
    targets: np.ndarray,
    baseline: np.ndarray,
    predictions: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target_cp", "original_eval", "fitted_eval"])
        for t, base, pred in zip(targets, baseline, predictions):
            writer.writerow([f"{t:.6f}", f"{base:.6f}", f"{pred:.6f}"])
