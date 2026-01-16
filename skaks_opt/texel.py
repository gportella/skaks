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
from typing import Any, List, Optional

import numpy as np
import optuna
import yaml

from skaks_opt.params import (DEFAULT_PARAMS, apply_param_updates,
                              param_space_for_mode)
from skaks_opt.pst import apply_pst_symmetry
from skaks_opt.yaml_utils import dump_yaml

__all__ = ["add_subparser", "run_texel", "load_texel_csv", "TexelDataset"]


def _coerce_like_template(template, payload):
    if isinstance(template, dict) and isinstance(payload, dict):
        return {
            key: _coerce_like_template(template.get(key), value)
            for key, value in payload.items()
        }
    if isinstance(template, list) and isinstance(payload, list):
        result = []
        for idx, value in enumerate(payload):
            tmpl = template[idx] if idx < len(template) else None
            result.append(_coerce_like_template(tmpl, value))
        return result
    if isinstance(template, int):
        if isinstance(payload, (int, float)):
            return int(round(payload))
        return payload
    if isinstance(template, float):
        if isinstance(payload, (int, float)):
            return float(payload)
        return payload
    return payload


def _prepare_params_for_eval(
    base_params: dict,
    updates: dict | None = None,
    *,
    pst_symmetry: str | None = "mirror-files",
) -> dict:
    updates = updates or {}
    merged = apply_param_updates(base_params, updates)
    apply_pst_symmetry(merged, mode=pst_symmetry)

    sanitized = {}
    for key, value in merged.items():
        template = DEFAULT_PARAMS.get(key)
        if template is not None:
            sanitized[key] = _coerce_like_template(template, value)
        else:
            sanitized[key] = value
    return sanitized


def add_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "texel",
        help="Run Texel logistic regression tuning",
        description=(
            "Texel fitting pipeline that learns logistic regression weights from "
            "outcome-labeled FENs. Ideal for calibrating evaluation terms via "
            "probabilistic loss before arena-based validation.\n\n"
            "Parameter sets:\n"
            "  - full: all eval scalars (plus phase weights if --include-arrays)\n"
            "  - phase: phase weights only\n"
            "  - offense: attack/initiative subset\n"
            "  - defense: king safety/structure subset\n"
            "  - pst: piece-square table entries only"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        required=True,
        help="CSV file or directory containing fen,outcome[,weight] data",
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
        "--outcome-pov",
        choices=["white", "side"],
        default="white",
        help="Interpret outcome column as white- or side-to-move POV",
    )
    parser.add_argument(
        "--sampler",
        choices=["cmaes", "tpe", "random"],
        default="cmaes",
        help="Optuna sampler",
    )
    parser.add_argument(
        "--loss",
        choices=["cross_entropy", "mse"],
        default="cross_entropy",
        help="Objective function for Texel tuning",
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
        choices=["full", "phase", "offense", "defense", "pst"],
        default="full",
        help="Select parameter subset to optimize (pst = PST table entries)",
    )
    parser.add_argument(
        "--min-ply",
        type=int,
        default=12,
        help="Minimum ply to include (<=0 disables lower bound)",
    )
    parser.add_argument(
        "--max-ply",
        type=int,
        default=80,
        help="Maximum ply to include (<=0 disables upper bound)",
    )
    parser.add_argument(
        "--max-stockfish-cp",
        type=float,
        help="Only keep rows whose |stockfish_cp| is <= this value",
    )
    parser.add_argument(
        "--max-qsearch-delta",
        type=float,
        help="Only keep rows whose |qsearch_delta| is <= this value",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=1.0,
        help="Randomly keep this fraction of rows during loading",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Seed for --sample-fraction",
    )
    parser.add_argument(
        "--include-arrays",
        action="store_true",
        help="Include array params when available",
    )
    parser.add_argument(
        "--ext",
        action="store_true",
        help="Load external datasets using only fen/outcome and ignore filters",
    )
    parser.add_argument(
        "--skip-pst",
        action="store_true",
        help="Exclude PST entries from tuning unless param-set is 'pst'",
    )
    parser.add_argument(
        "--pst-symmetry",
        choices=["mirror-files", "none"],
        default="mirror-files",
        help="Apply PST file symmetry when preparing params",
    )
    parser.add_argument(
        "--base-params",
        type=Path,
        default=Path("tuning/best_params_lots_2.yaml"),
        help="Initial YAML merged with tuned updates",
    )
    parser.add_argument(
        "--calibration-out",
        type=Path,
        help="Write sigmoid calibration plot for the best params",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=40,
        help="Bins for empirical calibration curve",
    )
    parser.add_argument(
        "--stockfish-correlation-out",
        type=Path,
        help="Write Stockfish cp vs outcome correlation plot",
    )
    parser.add_argument(
        "--outcome-hist-out",
        type=Path,
        help="Write weighted histogram of training outcomes (win/draw/loss mix)",
    )
    return parser


def _logit_prob(cp: np.ndarray, scale: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp( -cp / scale))

def _sigmoid_fit_metrics(
    cp_values: np.ndarray, outcomes: np.ndarray, weights: np.ndarray, scale: float
) -> tuple[float, float]:
    eps = 1e-9
    cp = np.asarray(cp_values, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    probs = np.clip(_logit_prob(cp, scale=scale), eps, 1.0 - eps)
    weight_sum = float(np.sum(w))
    if weight_sum <= eps:
        return float("inf"), float("inf")
    loss = float(
        np.sum((-y * np.log(probs) - (1.0 - y) * np.log(1.0 - probs)) * w) / weight_sum
    )
    diff = probs - y
    brier = float(np.sum((diff * diff) * w) / weight_sum)
    return loss, brier

def _fit_sigmoid_scale(
    cp_values: np.ndarray,
    outcomes: np.ndarray,
    weights: np.ndarray,
    scale_min: float | None,
    scale_max: float | None,
    *,
    coarse_steps: int = 60,
    refine_steps: int = 40,
) -> tuple[float, dict[str, float]]:
    min_scale = float(scale_min) if scale_min and scale_min > 0 else 50.0
    max_scale = float(scale_max) if scale_max and scale_max > min_scale else max(min_scale * 5.0, 1000.0)
    max_scale = max(max_scale, min_scale * 1.01)

    best_scale = min_scale
    best_loss = float("inf")
    best_brier = float("inf")

    def evaluate(scale: float) -> None:
        nonlocal best_scale, best_loss, best_brier
        loss, brier = _sigmoid_fit_metrics(cp_values, outcomes, weights, scale)
        if loss < best_loss:
            best_loss = loss
            best_brier = brier
            best_scale = scale

    for scale in np.geomspace(min_scale, max_scale, num=max(5, coarse_steps)):
        evaluate(scale)

    window_low = max(min_scale, best_scale / 2.0)
    window_high = min(max_scale, best_scale * 2.0)
    if window_high <= window_low:
        window_high = max_scale
    for scale in np.linspace(window_low, window_high, num=max(5, refine_steps)):
        evaluate(scale)

    metrics = {
        "loss": best_loss,
        "brier": best_brier,
        "rmse": math.sqrt(max(best_brier, 0.0)) if best_brier < float("inf") else float("inf"),
    }
    return best_scale, metrics

def _to_plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: _to_plain(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return [_to_plain(value) for value in obj.tolist()]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


@dataclass(frozen=True)
class TexelDataset:
    fens: List[str]
    outcomes: np.ndarray
    weights: np.ndarray
    side: np.ndarray
    stockfish_cp: Optional[np.ndarray] = None

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


def load_texel_csv(
    path: Path | str,
    limit: int | None = None,
    min_ply: int | None = None,
    max_ply: int | None = None,
    max_stockfish_cp: float | None = None,
    max_qsearch_delta: float | None = None,
    *,
    ext: bool = False,
    sample_fraction: float = 1.0,
    sample_seed: int = 42,
) -> TexelDataset:
    path = Path(path)
    fens: List[str] = []
    outcomes: List[float] = []
    weights: List[float] = []
    sides: List[int] = []
    stockfish_cps: List[float] = []
    paths: List[Path]
    if path.is_dir():
        paths = sorted(p for p in path.rglob("*.csv") if p.is_file())
    else:
        paths = [path]

    if not paths:
        raise ValueError(f"no CSV files found at {path}")

    remaining = limit
    sample_fraction = float(sample_fraction)
    if not (0.0 < sample_fraction <= 1.0):
        raise ValueError("sample_fraction must be in (0, 1]")
    rng = np.random.default_rng(sample_seed)

    if ext:
        min_ply = None
        max_ply = None
        max_stockfish_cp = None
        max_qsearch_delta = None
    else:
        min_ply = min_ply if (min_ply is not None and min_ply > 0) else None
        max_ply = max_ply if (max_ply is not None and max_ply > 0) else None
        max_stockfish_cp = (
            max_stockfish_cp if (max_stockfish_cp is not None and max_stockfish_cp > 0) else None
        )
        max_qsearch_delta = (
            max_qsearch_delta if (max_qsearch_delta is not None and max_qsearch_delta > 0) else None
        )
    warned_missing_ply = False

    def load_one(csv_path: Path, remaining_limit: int | None) -> int | None:
        nonlocal fens, outcomes, weights, sides
        limit_left = remaining_limit
        with csv_path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames:
                raise ValueError(f"CSV {csv_path} missing fen column")
            field_map = {
                name.strip().lower(): name for name in reader.fieldnames if name
            }
            fen_key = field_map.get("fen")
            if fen_key is None:
                raise ValueError(f"CSV {csv_path} missing fen column")
            outcome_col = None
            for candidate in ("outcome", "result", "score"):
                actual = field_map.get(candidate)
                if actual:
                    outcome_col = actual
                    break
            if outcome_col is None:
                raise ValueError(f"CSV {csv_path} missing outcome/result column")
            for row in reader:
                if limit_left is not None and limit_left <= 0:
                    return 0
                fen = row[fen_key].strip()
                if not fen:
                    continue
                if min_ply is not None or max_ply is not None:
                    ply_raw = row.get("ply")
                    ply_value: int | None = None
                    if ply_raw not in (None, ""):
                        try:
                            ply_value = int(ply_raw)
                        except Exception:
                            ply_value = None
                    if ply_value is None:
                        if not warned_missing_ply:
                            warnings.warn(
                                "ply column missing or non-numeric; ply filtering skipped for those rows"
                            )
                            warned_missing_ply = True
                    else:
                        if min_ply is not None and ply_value < min_ply:
                            continue
                        if max_ply is not None and ply_value > max_ply:
                            continue
                try:
                    value = float(row[outcome_col])
                except Exception:
                    continue
                if not (0.0 <= value <= 1.0):
                    continue
                if sample_fraction < 1.0 and rng.random() > sample_fraction:
                    continue
                cp_val: Optional[float] = None
                cp_raw = row.get("stockfish_cp")
                if cp_raw not in (None, ""):
                    try:
                        cp_val = float(cp_raw)
                    except Exception:
                        cp_val = None
                q_delta_val: Optional[float] = None
                q_delta_raw = row.get("qsearch_delta")
                if q_delta_raw not in (None, ""):
                    try:
                        q_delta_val = float(q_delta_raw)
                    except Exception:
                        q_delta_val = None
                if max_stockfish_cp is not None:
                    if cp_val is None or abs(cp_val) > max_stockfish_cp:
                        continue
                if max_qsearch_delta is not None:
                    if q_delta_val is None or abs(q_delta_val) > max_qsearch_delta:
                        continue
                weight = float(row.get("weight", 1.0) or 1.0)
                fens.append(fen)
                outcomes.append(value)
                weights.append(weight)
                stockfish_cps.append(cp_val if cp_val is not None else float("nan"))
                stm = row.get("side_to_move")
                if stm is not None and stm.lower().startswith("w"):
                    sides.append(1)
                elif stm is not None and stm.lower().startswith("b"):
                    sides.append(-1)
                else:
                    sides.append(_parse_side_to_move(fen))
                if limit_left is not None:
                    limit_left -= 1
        if limit_left is None:
            return None
        return max(limit_left, 0)

    for csv_path in paths:
        remaining = load_one(csv_path, remaining)
        if remaining is not None and remaining <= 0:
            break

    if not fens:
        raise ValueError(f"no rows loaded from {path}")
    stockfish_cp_array: Optional[np.ndarray] = None
    if stockfish_cps:
        stockfish_cp_array = np.array(stockfish_cps, dtype=np.float64)

    return TexelDataset(
        fens=fens,
        outcomes=np.array(outcomes, dtype=np.float64),
        weights=np.array(weights, dtype=np.float64),
        side=np.array(sides, dtype=np.int8),
        stockfish_cp=stockfish_cp_array,
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

    filtered_cps = None
    if dataset.stockfish_cp is not None:
        filtered_cps = dataset.stockfish_cp[keep_mask]

    return TexelDataset(
        fens=[fen for fen, keep in zip(dataset.fens, keep_mask) if keep],
        outcomes=dataset.outcomes[keep_mask],
        weights=dataset.weights[keep_mask],
        side=dataset.side[keep_mask],
        stockfish_cp=filtered_cps,
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
    loss_mode: str = "cross_entropy",
    outcome_pov: str = "white",
) -> tuple[float, int, dict[str, float]]:
    try:
        import skaks_eval as sk
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "skaks_eval is not installed; install bindings first"
        ) from exc

    eps = 1e-9
    total_weight = 0.0
    total_loss = 0.0
    total_brier = 0.0
    prob_weight = 0.0
    error_count = 0

    for fens, outcomes, weights, side in dataset.iter_batches(batch_size):
        result = sk.eval_fens(fens, params=params, threads=threads)
        cp_raw = np.asarray(result["cp"], dtype=np.float64)
        if cp_cap is not None:
            cp_raw = np.clip(cp_raw, -cp_cap, cp_cap)
        cp = cp_raw * side if pov == "side" else cp_raw
        probs = _logit_prob(cp, scale=scale)
        errs = result["errors"]
        if outcome_pov != pov:
            adj_outcomes = np.where(np.asarray(side) >= 0, outcomes, 1.0 - outcomes)
        else:
            adj_outcomes = outcomes
        for i, err in enumerate(errs):
            w = float(weights[i])
            if err is not None:
                total_loss += error_penalty * w
                total_weight += w
                error_count += 1
                continue
            p = float(np.clip(probs[i], eps, 1.0 - eps))
            y = float(adj_outcomes[i])
            total_loss += (-y * math.log(p) - (1.0 - y) * math.log(1.0 - p)) * w
            total_weight += w
            diff = p - y
            total_brier += diff * diff * w
            prob_weight += w

    mean_loss = total_loss / total_weight if total_weight > 0 else float("inf")
    metrics: dict[str, float] = {"cross_entropy": mean_loss}
    if prob_weight > 0:
        brier = total_brier / prob_weight
        metrics["brier"] = brier
        metrics["rmse"] = math.sqrt(brier)
    if loss_mode == "mse":
        objective_value = metrics.get("brier", float("inf"))
    else:
        objective_value = mean_loss
    return objective_value, error_count, metrics


def _collect_eval_samples(
    *,
    params,
    dataset: TexelDataset,
    batch_size: int,
    threads: int,
    cp_cap: float | None,
    pov: str,
    outcome_pov: str = "white",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import skaks_eval as sk
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "skaks_eval is not installed; install bindings first"
        ) from exc

    cp_chunks: List[np.ndarray] = []
    outcome_chunks: List[np.ndarray] = []
    weight_chunks: List[np.ndarray] = []

    eps = 1e-9
    for fens, outcomes, weights, side in dataset.iter_batches(batch_size):
        result = sk.eval_fens(fens, params=params, threads=threads)
        cp_raw = np.asarray(result["cp"], dtype=np.float64)
        if cp_cap is not None:
            cp_raw = np.clip(cp_raw, -cp_cap, cp_cap)
        cp_vals = cp_raw * side if pov == "side" else cp_raw
        errs = result["errors"]
        if outcome_pov != pov:
            adj_outcomes = np.where(np.asarray(side) >= 0, outcomes, 1.0 - outcomes)
        else:
            adj_outcomes = outcomes
        mask = np.array([err is None for err in errs], dtype=bool)
        if not mask.any():
            continue
        cp_chunks.append(cp_vals[mask])
        outcome_chunks.append(np.asarray(adj_outcomes, dtype=np.float64)[mask])
        weight_chunks.append(np.asarray(weights, dtype=np.float64)[mask])

    if not cp_chunks:
        raise RuntimeError("no valid evaluation samples for calibration plot")

    cps = np.concatenate(cp_chunks)
    outs = np.concatenate(outcome_chunks)
    wts = np.concatenate(weight_chunks)
    total_weight = np.maximum(wts, eps)
    return cps, outs, total_weight


def _collect_stockfish_cp_samples(dataset: TexelDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if dataset.stockfish_cp is None:
        raise RuntimeError("dataset has no stockfish_cp column")
    mask = np.isfinite(dataset.stockfish_cp)
    if not mask.any():
        raise RuntimeError("dataset has no finite stockfish cp entries")
    eps = 1e-9
    cps = dataset.stockfish_cp[mask]
    outs = dataset.outcomes[mask]
    wts = np.maximum(dataset.weights[mask], eps)
    return cps, outs, wts


def _write_calibration_plot(
    *,
    path: Path,
    cp_values: np.ndarray,
    outcomes: np.ndarray,
    weights: np.ndarray,
    scale: Optional[float],
    bins: int,
    label: str = "empirical",
    xlabel: str = "stockfish cp (side POV)",
    x_range: tuple[float, float] | None = None,
    show_error: bool = False,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        print("matplotlib not installed; skipping calibration plot")
        return

    if bins <= 0:
        bins = 40
    cp_vals = np.asarray(cp_values, dtype=np.float64)
    y_vals = np.asarray(outcomes, dtype=np.float64)
    w_vals = np.asarray(weights, dtype=np.float64)
    if x_range is not None:
        cp_min, cp_max = x_range
    else:
        cp_min = float(np.min(cp_vals))
        cp_max = float(np.max(cp_vals))
    if cp_min == cp_max:
        cp_min -= 1.0
        cp_max += 1.0
    edges = np.linspace(cp_min, cp_max, bins + 1)
    centers: list[float] = []
    actual: list[float] = []
    errors: list[float] = []
    for start, end in zip(edges[:-1], edges[1:]):
        mask = (cp_vals >= start) & (cp_vals < end)
        if not mask.any():
            continue
        bin_weights = w_vals[mask]
        prob = float(np.average(y_vals[mask], weights=bin_weights))
        centers.append(0.5 * (start + end))
        actual.append(prob)
        if show_error:
            w_sum = float(np.sum(bin_weights))
            w_sq_sum = float(np.sum(bin_weights * bin_weights))
            eff = (w_sum * w_sum) / w_sq_sum if w_sq_sum > 0 else 0.0
            stderr = math.sqrt(max(prob * (1.0 - prob), 0.0) / eff) if eff > 0 else 0.0
            errors.append(stderr)

    plt.figure(figsize=(8, 4))
    if centers:
        if show_error and errors:
            plt.errorbar(
                centers,
                actual,
                yerr=errors,
                fmt="o",
                label=label,
                ms=5,
                capsize=3,
                alpha=0.85,
            )
        else:
            plt.scatter(centers, actual, label=label, s=24, alpha=0.8)
    if scale is not None:
        line_x = np.linspace(cp_min, cp_max, 400)
        line_y = _logit_prob(line_x, scale=scale)
        plt.plot(line_x, line_y, color="orange", label="sigmoid", linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel("observed outcome")
    plt.title("Texel calibration")
    plt.legend()
    plt.xlim(cp_min, cp_max)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    print(f"wrote calibration plot to {path}")


def _write_outcome_histogram(
    *, path: Path, outcomes: np.ndarray, weights: np.ndarray
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        print("matplotlib not installed; skipping outcome histogram")
        return

    bins = np.linspace(0.0, 1.0, 21)
    plt.figure(figsize=(8, 3.5))
    plt.hist(
        outcomes,
        bins=bins,
        weights=weights,
        color="#4c72b0",
        edgecolor="white",
        linewidth=0.8,
    )
    plt.xlabel("outcome probability (0=loss, 0.5=draw, 1=win)")
    plt.ylabel("weighted count")
    plt.title("Training outcome distribution")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    print(f"wrote outcome histogram to {path}")

    total_weight = float(np.sum(weights))
    eps = 1e-9
    loss_w = float(np.sum(weights[outcomes <= 1.0 / 3.0]))
    draw_mask = (outcomes > 1.0 / 3.0) & (outcomes < 2.0 / 3.0)
    draw_w = float(np.sum(weights[draw_mask]))
    win_w = float(np.sum(weights[outcomes >= 2.0 / 3.0]))
    if total_weight > eps:
        loss_pct = 100.0 * loss_w / total_weight
        draw_pct = 100.0 * draw_w / total_weight
        win_pct = 100.0 * win_w / total_weight
        print(
            "dataset outcome mix: "
            f"loss {loss_w:.1f} ({loss_pct:.1f}%) | "
            f"draw {draw_w:.1f} ({draw_pct:.1f}%) | "
            f"win {win_w:.1f} ({win_pct:.1f}%)"
        )


def run_texel(args: argparse.Namespace) -> None:
    if args.quiet:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    if args.sampler == "tpe":
        from optuna.exceptions import ExperimentalWarning

        warnings.filterwarnings("ignore", category=ExperimentalWarning)

    min_ply = args.min_ply if args.min_ply is None or args.min_ply > 0 else None
    max_ply = args.max_ply if args.max_ply is None or args.max_ply > 0 else None
    max_stockfish_cp = (
        args.max_stockfish_cp if args.max_stockfish_cp and args.max_stockfish_cp > 0 else None
    )
    max_qsearch_delta = (
        args.max_qsearch_delta if args.max_qsearch_delta and args.max_qsearch_delta > 0 else None
    )
    dataset = load_texel_csv(
        args.data,
        limit=args.limit,
        min_ply=min_ply,
        max_ply=max_ply,
        max_stockfish_cp=max_stockfish_cp,
        max_qsearch_delta=max_qsearch_delta,
        ext=args.ext,
        sample_fraction=args.sample_fraction,
        sample_seed=args.sample_seed,
    )
    if args.require_quiet:
        dataset = filter_quiet(dataset, batch_size=args.quiet_batch)
        print(f"Filtered to {len(dataset)} quiet positions")

    if args.outcome_hist_out:
        try:
            _write_outcome_histogram(
                path=args.outcome_hist_out,
                outcomes=dataset.outcomes,
                weights=dataset.weights,
            )
        except Exception as exc:
            print(f"failed to write outcome histogram: {exc}")

    cp_plot_span: tuple[float, float] | None = None
    if args.cp_cap is not None and args.cp_cap > 0:
        span_limit = float(args.cp_cap)
        cp_plot_span = (-span_limit, span_limit)

    include_pst = not args.skip_pst or args.param_set == "pst"
    param_space = param_space_for_mode(
        mode=args.param_set,
        include_arrays=args.include_arrays,
        include_pst=include_pst,
    )
    param_space = [spec for spec in param_space if not spec.name.startswith("search.")]

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
        params = _prepare_params_for_eval(
            base_params,
            updates,
            pst_symmetry=args.pst_symmetry,
        )
        params = _to_plain(params)
        loss, err_count, fit_metrics = texel_loss(
            params=params,
            dataset=dataset,
            batch_size=args.batch_size,
            threads=args.threads,
            cp_cap=args.cp_cap,
            pov=args.pov,
            scale=scale,
            error_penalty=args.error_penalty,
            loss_mode=args.loss,
            outcome_pov=args.outcome_pov,
        )
        trial.set_user_attr("error_count", err_count)
        trial.set_user_attr("texel_scale", scale)
        for key, value in fit_metrics.items():
            trial.set_user_attr(key, value)
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
        rmse_val = trial.user_attrs.get("rmse")
        rmse_txt = f"{rmse_val:.4f}" if rmse_val is not None else "-"
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
                f"rmse={rmse_txt} "
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
                f"rmse={rmse_txt} "
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
    best_param_updates = {
        key: value
        for key, value in best.params.items()
        if key.startswith("evaluation.") or key.startswith("search.")
    }
    merged = _prepare_params_for_eval(
        base_params,
        best_param_updates,
        pst_symmetry=args.pst_symmetry,
    )
    merged = _to_plain(merged)
    best_record = {
        "number": best.number,
        "loss": best.value,
        "error_count": best.user_attrs.get("error_count", 0),
    }
    if "brier" in best.user_attrs:
        best_record["brier"] = best.user_attrs.get("brier")
    if "rmse" in best.user_attrs:
        best_record["rmse"] = best.user_attrs.get("rmse")
    if "cross_entropy" in best.user_attrs:
        best_record["cross_entropy"] = best.user_attrs.get("cross_entropy")
    print("=== best trial ===")
    print(json.dumps(best_record, indent=2))

    if args.best_out:
        args.best_out.parent.mkdir(parents=True, exist_ok=True)
        with args.best_out.open("w", encoding="utf-8") as fh:
            dump_yaml(
                {
                    key: _coerce_like_template(DEFAULT_PARAMS.get(key), value)
                    for key, value in merged.items()
                },
                fh,
            )
        print(f"wrote best params to {args.best_out}")

    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "number",
                    "loss",
                    "error_count",
                    "brier",
                    "rmse",
                    "cross_entropy",
                ]
            )
            for trial in study.trials:
                if trial.state != optuna.trial.TrialState.COMPLETE:
                    continue
                writer.writerow(
                    [
                        trial.number,
                        trial.value,
                        trial.user_attrs.get("error_count", 0),
                        trial.user_attrs.get("brier"),
                        trial.user_attrs.get("rmse"),
                        trial.user_attrs.get("cross_entropy"),
                    ]
                )
        print(f"wrote metrics to {args.metrics_out}")

    if args.calibration_out:
        scale_val = best.user_attrs.get("texel_scale")
        if scale_val is None:
            print("texel_scale missing on best trial; skipping calibration plot")
        else:
            try:
                cp_vals, outs, wts = _collect_eval_samples(
                    params=merged,
                    dataset=dataset,
                    batch_size=args.batch_size,
                    threads=args.threads,
                    cp_cap=args.cp_cap,
                    pov=args.pov,
                    outcome_pov=args.outcome_pov,
                )
                xlabel = "skaks cp (side POV)" if args.pov == "side" else "skaks cp (white POV)"
                _write_calibration_plot(
                    path=args.calibration_out,
                    cp_values=cp_vals,
                    outcomes=outs,
                    weights=wts,
                    scale=float(scale_val),
                    bins=args.calibration_bins,
                    xlabel=xlabel,
                    x_range=cp_plot_span,
                    show_error=True,
                )
            except Exception as exc:
                print(f"failed to write calibration plot: {exc}")

    if args.stockfish_correlation_out:
        if dataset.stockfish_cp is None:
            print("dataset missing stockfish_cp values; skipping correlation plot")
        else:
            try:
                cp_vals, outs, wts = _collect_stockfish_cp_samples(dataset)
                finite_mask = np.isfinite(cp_vals)
                if not finite_mask.any():
                    raise RuntimeError("no finite stockfish cp values available")
                sf_cp = cp_vals[finite_mask]
                sf_outs = outs[finite_mask]
                sf_wts = wts[finite_mask]
                sf_min = float(np.min(sf_cp))
                sf_max = float(np.max(sf_cp))
                print(
                    f"stockfish cp span in dataset: {sf_min:.1f} to {sf_max:.1f}"
                )
                sf_scale, sf_metrics = _fit_sigmoid_scale(
                    sf_cp,
                    sf_outs,
                    sf_wts,
                    args.texel_scale_min,
                    args.texel_scale_max,
                )
                print(
                    "stockfish texel-fit: "
                    f"scale={sf_scale:.1f} loss={sf_metrics['loss']:.6f} "
                    f"rmse={sf_metrics['rmse']:.4f}"
                )
                _write_calibration_plot(
                    path=args.stockfish_correlation_out,
                    cp_values=cp_vals,
                    outcomes=outs,
                    weights=wts,
                    scale=sf_scale,
                    bins=args.calibration_bins,
                    label="stockfish cp",
                    x_range=cp_plot_span,
                    show_error=True,
                )
            except Exception as exc:
                print(f"failed to write stockfish correlation plot: {exc}")

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
