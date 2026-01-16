from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import random
import subprocess
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import yaml

from skaks_opt.params import DEFAULT_PARAMS

try:  # pragma: no cover - optional rich dependency
    from rich import box
    from rich.console import Console
    from rich.table import Table

    RICH_TABLE_AVAILABLE = True
except Exception:  # pragma: no cover - rich optional
    Console = None  # type: ignore
    Table = None  # type: ignore
    box = None  # type: ignore
    RICH_TABLE_AVAILABLE = False

_TERM_BAR_COLORS = ["#0ea5e9", "#38bdf8", "#22c55e", "#f59e0b", "#f97316", "#ef4444"]
_TERM_BAR_TRAIL = "#1f2937"
_TERM_BAR_CHAR = "▉"
_TERM_NAMES = [
    "Material",
    "PawnCenter",
    "CenterControl",
    "Attacking",
    "KingSafety",
    "KingMobility",
    "Pins",
    "PstMg",
    "PstEg",
    "PassedPawns",
    "Initiative",
    "Hanging",
    "KingRing",
    "BishopPair",
    "RookFiles",
    "MinorMobility",
    "PawnStructure",
]

__all__ = ["add_subparser", "run_eval_stats"]


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
        if self.count < 2:
            return math.nan
        num = self.count * self.sum_xy - self.sum_x * self.sum_y
        den_x = self.count * self.sum_x2 - self.sum_x * self.sum_x
        den_y = self.count * self.sum_y2 - self.sum_y * self.sum_y
        denom = math.sqrt(max(den_x, 0.0) * max(den_y, 0.0))
        if denom == 0.0:
            return math.nan
        return num / denom


@dataclass
class EvalSample:
    term_values: List[int]
    mg_ratio: float
    raw_linear: float
    engine_eval: float
    term_names: List[str]


def add_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "eval-stats",
        help="Summarize per-term contributions and calibration against Stockfish",
        description=(
            "Collects term-by-term contribution breakdowns from the engine, compares "
            "them to Stockfish centipawns, and surfaces scaling/quiet-cap hints. Use "
            "this to understand how the current eval behaves before retuning it."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        nargs="+",
        help="CSV file(s) or directory with fen,stockfish_cp columns",
    )
    parser.add_argument(
        "--skaks",
        default="skaks",
        help="Path to the skaks executable",
    )
    parser.add_argument(
        "--params",
        type=pathlib.Path,
        help="Optional YAML whose phase weights should be used in contribution math",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Cap sampled positions after filtering",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=1.0,
        help="Randomly down-sample rows before cap",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--target-abs",
        type=float,
        default=40.0,
        help="Target mean absolute contribution (cp) for per-term scaling hints",
    )
    parser.add_argument(
        "--quiet-cap-percentile",
        type=float,
        default=95.0,
        help="Percentile of |fitted cp| used when proposing eval_quiet_cap",
    )
    parser.add_argument(
        "--quiet-cap-min",
        type=float,
        default=200.0,
        help="Lower bound when clipping proposed quiet cap",
    )
    parser.add_argument(
        "--quiet-cap-max",
        type=float,
        default=1200.0,
        help="Upper bound when clipping proposed quiet cap",
    )
    parser.add_argument(
        "--stats-csv",
        type=pathlib.Path,
        help="Optional CSV path for per-term statistics",
    )
    parser.add_argument(
        "--json-out",
        type=pathlib.Path,
        help="Optional JSON dump with raw aggregates",
    )
    parser.add_argument(
        "--cp-cap",
        type=float,
        help="Clamp Stockfish centipawns before analysis",
    )
    parser.add_argument(
        "--require-quiet",
        action="store_true",
        help="Filter to quiet positions using skaks_eval",
    )
    parser.add_argument(
        "--quiet-batch",
        type=int,
        default=2048,
        help="Batch size for quiet filtering",
    )
    parser.add_argument(
        "--table-png",
        type=pathlib.Path,
        help="Optional PNG path for visualizing summary metrics",
    )
    parser.add_argument(
        "--no-table",
        action="store_true",
        help="Skip Rich summary table output",
    )
    parser.add_argument(
        "--apply-scale-yaml",
        type=pathlib.Path,
        help="Write YAML with phase weights scaled toward target |term| magnitudes",
    )
    parser.add_argument(
        "--scale-min",
        type=float,
        default=0.25,
        help="Lower clamp when applying scale hints",
    )
    parser.add_argument(
        "--scale-max",
        type=float,
        default=2.5,
        help="Upper clamp when applying scale hints",
    )
    parser.add_argument(
        "--scale-include",
        action="append",
        help="Limit scaling to specified term name or index (repeatable)",
    )
    parser.add_argument(
        "--scale-exclude",
        action="append",
        help="Exclude specified term name or index from scaling",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=200,
        help="Emit progress every N samples when Rich is unavailable",
    )
    parser.add_argument(
        "--no-rich",
        action="store_true",
        help="Disable Rich progress output",
    )
    return parser


def run_eval_stats(args: argparse.Namespace) -> None:
    dataset_paths = _expand_dataset_inputs(args.dataset)
    rng = random.Random(args.seed)

    rows, total_seen = _sample_dataset(
        dataset_paths,
        args.sample_fraction,
        args.limit,
        rng,
        cp_cap=args.cp_cap,
    )
    if not rows:
        raise SystemExit("No dataset rows selected; adjust sampling options")

    if args.require_quiet:
        rows = _filter_quiet_rows(rows, batch_size=args.quiet_batch)
        if not rows:
            raise SystemExit("Quiet filtering removed all positions")

    collected: List[EvalSample] = []
    targets: List[float] = []
    errors: List[str] = []
    raw_stats = RunningStats()
    engine_stats = RunningStats()

    use_rich = False
    fancy = None
    if not args.no_rich:
        try:  # pragma: no cover - optional dependency
            from tuning.skaks_opt.progress import FancyProgress

            fancy = FancyProgress(total=len(rows), rng=rng, description="Collecting")
            fancy.__enter__()
            use_rich = True
        except Exception:
            fancy = None
            use_rich = False

    try:
        for idx, (fen, target) in enumerate(rows, start=1):
            try:
                sample = _fetch_eval_breakdown(pathlib.Path(args.skaks), fen)
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(f"{fen}: {exc}")
                continue
            targets.append(float(target))
            collected.append(sample)
            raw_stats.add(target, sample.raw_linear)
            engine_stats.add(target, sample.engine_eval)
            if use_rich and fancy is not None:
                status = _format_progress_status(raw_stats, engine_stats)
                fancy.update(
                    idx,
                    message=f"{idx}/{len(rows)} samples",
                    status=status,
                )
            elif args.progress_interval > 0 and (
                idx == 1 or idx == len(rows) or idx % args.progress_interval == 0
            ):
                status = _format_progress_status(raw_stats, engine_stats)
                metric_suffix = f" {status}" if status else ""
                print(
                    f"[{idx}/{len(rows)}] target={target:+.1f} raw={sample.raw_linear:+.1f} eval={sample.engine_eval:+.1f}{metric_suffix}"
                )
    finally:
        if fancy is not None:
            fancy.__exit__(None, None, None)

    if not collected:
        raise SystemExit("No usable samples gathered")

    stats = _compute_statistics(
        collected=collected,
        targets=np.asarray(targets, dtype=np.float64),
        params_path=args.params,
        target_abs=args.target_abs,
        quiet_cap_percentile=args.quiet_cap_percentile,
        quiet_cap_min=args.quiet_cap_min,
        quiet_cap_max=args.quiet_cap_max,
    )

    _print_summary(stats, total_seen=total_seen, errors=errors)

    if args.stats_csv:
        _write_csv(args.stats_csv, stats["per_term"])
        print(f"stats_csv: {args.stats_csv}")

    if args.json_out:
        with args.json_out.open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2)
        print(f"json_out: {args.json_out}")

    png_written = False
    if not args.no_table:
        png_written = _render_metrics_table(stats, args.table_png)
    elif args.table_png:
        png_written = _render_png_table(stats, args.table_png)

    if png_written and args.table_png:
        print(f"table_png: {args.table_png}")

    if args.apply_scale_yaml:
        applied = _write_scaled_params(
            stats=stats,
            params_path=args.params,
            output_path=args.apply_scale_yaml,
            scale_min=args.scale_min,
            scale_max=args.scale_max,
            include_specs=args.scale_include,
            exclude_specs=args.scale_exclude,
        )
        if applied is not None:
            print(f"scaled_yaml: {args.apply_scale_yaml}")
            if applied:
                print("  scaled terms:")
                for idx, name, factor in applied:
                    print(f"    [{idx:02d}] {name} -> ×{factor:.3f}")
            else:
                print("  (no per-term scale hints applied; weights unchanged)")
        else:
            print("scaled_yaml: skipped (no valid phase weights found)")

    if errors:
        print(f"warnings: {len(errors)} failures while sampling (showing first 5)")
        for message in errors[:5]:
            print(f"  {message}")


def _expand_dataset_inputs(raw_paths: Sequence[str]) -> List[pathlib.Path]:
    expanded: List[pathlib.Path] = []
    for raw in raw_paths:
        path = pathlib.Path(raw).expanduser()
        if path.is_dir():
            expanded.extend(sorted(p for p in path.rglob("*.csv") if p.is_file()))
        else:
            expanded.append(path)
    if not expanded:
        raise SystemExit("No CSV files found for the provided --dataset inputs")
    return expanded


def _iter_dataset_rows(path: pathlib.Path) -> Iterable[Tuple[str, float]]:
    if path.is_dir():
        for csv_path in sorted(path.rglob("*.csv")):
            yield from _iter_dataset_rows(csv_path)
        return
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "fen" not in reader.fieldnames or "stockfish_cp" not in reader.fieldnames:
            raise ValueError(f"{path}: missing fen/stockfish_cp columns")
        for row in reader:
            fen = row["fen"].strip()
            if not fen:
                continue
            try:
                target = float(row["stockfish_cp"])
            except Exception:
                continue
            yield fen, target


def _clip_cp(value: float, cp_cap: float | None) -> float:
    if cp_cap is None or cp_cap <= 0:
        return value
    return float(max(-cp_cap, min(cp_cap, value)))


def _sample_dataset(
    paths: Sequence[pathlib.Path],
    fraction: float,
    limit: int,
    rng: random.Random,
    *,
    cp_cap: float | None,
) -> Tuple[List[Tuple[str, float]], int]:
    rows: List[Tuple[str, float]] = []
    fraction = float(max(0.0, min(1.0, fraction)))
    total_seen = 0
    if fraction == 0.0 or limit == 0:
        return rows, total_seen
    for root in paths:
        for fen, target in _iter_dataset_rows(root):
            total_seen += 1
            target = _clip_cp(target, cp_cap)
            if fraction < 1.0 and rng.random() > fraction:
                continue
            rows.append((fen, target))
            if 0 < limit == len(rows):
                return rows, total_seen
    return rows, total_seen


def _filter_quiet_rows(
    rows: Sequence[Tuple[str, float]], *, batch_size: int
) -> List[Tuple[str, float]]:
    try:
        import skaks_eval as sk
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "skaks_eval module not available; rebuild python bindings or drop --require-quiet"
        ) from exc

    kept: List[Tuple[str, float]] = []
    batch_size = max(1, int(batch_size))
    for start in range(0, len(rows), batch_size):
        end = min(start + batch_size, len(rows))
        chunk = rows[start:end]
        fens = [fen for fen, _ in chunk]
        try:
            flags = sk.is_quiet_batch(fens)
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"quiet filtering failed: {exc}") from exc
        for (fen, target), flag in zip(chunk, flags):
            if bool(flag):
                kept.append((fen, target))

    if not kept:
        raise RuntimeError("quiet filtering removed every sampled position")
    return kept


def _fetch_eval_breakdown(skaks: pathlib.Path, fen: str) -> EvalSample:
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
        raw_linear=raw_linear,
        engine_eval=engine_eval,
        term_names=term_names,
    )


def _load_phase_weights(
    params_path: pathlib.Path | None,
) -> Tuple[np.ndarray, np.ndarray]:
    eval_defaults = DEFAULT_PARAMS.get("evaluation", {})
    mg = np.asarray(eval_defaults.get("phase_weights_mg", []), dtype=np.float64)
    eg = np.asarray(eval_defaults.get("phase_weights_eg", []), dtype=np.float64)

    if params_path and params_path.exists():
        with params_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        evaluation = data.get("evaluation", {}) if isinstance(data, dict) else {}
        mg_override = evaluation.get("phase_weights_mg")
        eg_override = evaluation.get("phase_weights_eg")
        if mg_override is not None:
            mg = np.asarray(mg_override, dtype=np.float64)
        if eg_override is not None:
            eg = np.asarray(eg_override, dtype=np.float64)

    if mg.size == 0 or eg.size == 0:
        raise ValueError("Phase weight arrays could not be resolved from defaults/YAML")
    if mg.size != eg.size:
        raise ValueError("Midgame and endgame weight arrays differ in length")
    return mg, eg


def _compute_statistics(
    *,
    collected: Sequence[EvalSample],
    targets: np.ndarray,
    params_path: pathlib.Path | None,
    target_abs: float,
    quiet_cap_percentile: float,
    quiet_cap_min: float,
    quiet_cap_max: float,
) -> dict:
    term_names: List[str]
    first_names = collected[0].term_names if collected[0].term_names else None
    if first_names and len(first_names) == len(collected[0].term_values):
        term_names = list(first_names)
    else:
        term_names = [f"term_{i}" for i in range(len(collected[0].term_values))]

    term_values = np.asarray(
        [sample.term_values for sample in collected], dtype=np.float64
    )
    mg_ratios = np.asarray([sample.mg_ratio for sample in collected], dtype=np.float64)
    raw_linear = np.asarray(
        [sample.raw_linear for sample in collected], dtype=np.float64
    )
    engine_eval = np.asarray(
        [sample.engine_eval for sample in collected], dtype=np.float64
    )

    mg_weights, eg_weights = _load_phase_weights(params_path)
    weights = mg_ratios[:, None] * mg_weights + (1.0 - mg_ratios)[:, None] * eg_weights
    contributions = weights * term_values

    intercept, slope = _fit_linear(targets, raw_linear)
    fitted = intercept + slope * raw_linear
    quiet_cap = float(np.percentile(np.abs(fitted), quiet_cap_percentile))
    quiet_cap = float(np.clip(quiet_cap, quiet_cap_min, quiet_cap_max))

    baseline_rmse = float(np.sqrt(np.mean((targets - engine_eval) ** 2)))
    baseline_mae = float(np.mean(np.abs(targets - engine_eval)))
    baseline_r = float(_pearson(targets, engine_eval))

    raw_rmse = float(np.sqrt(np.mean((targets - raw_linear) ** 2)))
    raw_mae = float(np.mean(np.abs(targets - raw_linear)))
    raw_r = float(_pearson(targets, raw_linear))

    fitted_rmse = float(np.sqrt(np.mean((targets - fitted) ** 2)))
    fitted_mae = float(np.mean(np.abs(targets - fitted)))
    fitted_r = float(_pearson(targets, fitted))

    per_term: List[dict] = []
    abs_target = max(target_abs, 1e-3)
    for idx, name in enumerate(term_names):
        raw_col = term_values[:, idx]
        contrib_col = contributions[:, idx]
        abs_mean = float(np.mean(np.abs(contrib_col)))
        scale_hint = abs_target / abs_mean if abs_mean > 1e-6 else float("inf")
        per_term.append(
            {
                "index": idx,
                "name": name,
                "raw_mean": float(np.mean(raw_col)),
                "raw_std": float(np.std(raw_col)),
                "contrib_mean": float(np.mean(contrib_col)),
                "contrib_std": float(np.std(contrib_col)),
                "contrib_abs_mean": abs_mean,
                "contrib_abs_max": float(np.max(np.abs(contrib_col))),
                "scale_hint": scale_hint,
            }
        )

    per_term.sort(key=lambda item: -item["contrib_abs_mean"])

    return {
        "count": int(len(collected)),
        "baseline_rmse": baseline_rmse,
        "baseline_mae": baseline_mae,
        "baseline_r": baseline_r,
        "raw_rmse": raw_rmse,
        "raw_mae": raw_mae,
        "raw_r": raw_r,
        "fitted_rmse": fitted_rmse,
        "fitted_mae": fitted_mae,
        "fitted_r": fitted_r,
        "intercept": float(intercept),
        "slope": float(slope),
        "quiet_cap": quiet_cap,
        "per_term": per_term,
    }


def _fit_linear(targets: np.ndarray, predictions: np.ndarray) -> Tuple[float, float]:
    if targets.size == 0:
        return 0.0, 1.0
    design = np.column_stack((np.ones(targets.shape[0]), predictions))
    coeffs, *_ = np.linalg.lstsq(design, targets, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def _pearson(xs: np.ndarray, ys: np.ndarray) -> float:
    if xs.size == 0 or ys.size == 0 or xs.size != ys.size:
        return math.nan
    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)
    x_mean = float(np.mean(xs))
    y_mean = float(np.mean(ys))
    num = float(np.sum((xs - x_mean) * (ys - y_mean)))
    den = math.sqrt(
        float(np.sum((xs - x_mean) ** 2)) * float(np.sum((ys - y_mean) ** 2))
    )
    if den == 0.0:
        return math.nan
    return num / den


def _print_summary(stats: dict, *, total_seen: int, errors: Sequence[str]) -> None:
    print(
        f"rows_scanned: {total_seen}\n"
        f"samples_used: {stats['count']}\n"
        f"baseline_rmse: {stats['baseline_rmse']:.2f}\n"
        f"baseline_mae: {stats['baseline_mae']:.2f}\n"
        f"baseline_pearson: {stats['baseline_r']:+.4f}\n"
        f"raw_rmse: {stats['raw_rmse']:.2f}\n"
        f"raw_mae: {stats['raw_mae']:.2f}\n"
        f"raw_pearson: {stats['raw_r']:+.4f}\n"
        f"fitted_rmse: {stats['fitted_rmse']:.2f}\n"
        f"fitted_mae: {stats['fitted_mae']:.2f}\n"
        f"fitted_pearson: {stats['fitted_r']:+.4f}\n"
        f"calibration_intercept: {stats['intercept']:.3f}\n"
        f"calibration_slope: {stats['slope']:.6f}\n"
        f"quiet_cap: {stats['quiet_cap']:.2f}"
    )

    _render_per_term_table(stats)

    if errors:
        print(f"\nerrors: {len(errors)} positions failed to evaluate")


def _write_csv(path: pathlib.Path, rows: Sequence[dict]) -> None:
    fieldnames = [
        "index",
        "name",
        "raw_mean",
        "raw_std",
        "contrib_mean",
        "contrib_std",
        "contrib_abs_mean",
        "contrib_abs_max",
        "scale_hint",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _render_metrics_table(stats: dict, table_png: pathlib.Path | None) -> bool:
    png_written = False
    if not RICH_TABLE_AVAILABLE:
        print("Rich summary table unavailable; install 'rich' to enable it.")
        if table_png:
            png_written = _render_png_table(stats, table_png)
        return png_written

    console = Console(highlight=False, soft_wrap=False)
    console.rule("[bold magenta]♔ Quiet Fit Summary ♚")

    table = Table(
        box=box.DOUBLE_EDGE,
        expand=True,
        show_header=True,
        header_style="bold cyan",
        pad_edge=True,
        title="Skaks Quiet Fit Metrics",
        title_style="bold white",
    )
    table.add_column("Metric ♞", justify="left", style="bold cyan")
    table.add_column("Baseline ♖", justify="right", style="bright_white")
    table.add_column("Raw ♗", justify="right", style="bright_white")
    table.add_column("Fitted ♕", justify="right", style="bold green")

    for label, values, fmt, higher in _metric_specs(stats):
        row = _format_metric_row(label, values, fmt, higher)
        table.add_row(*row)

    table.add_section()
    table.add_row(
        "Samples ♙",
        f"{stats['count']:,}",
        "—",
        "—",
    )
    table.add_row(
        "Quiet Cap (cp)",
        "—",
        "—",
        _format_numeric(stats["quiet_cap"], fmt=".1f", highlight=True),
    )
    table.add_row(
        "Intercept (cp)",
        "—",
        "—",
        _format_numeric(stats["intercept"], fmt=".2f", highlight=True),
    )
    table.add_row(
        "Slope",
        "—",
        "—",
        _format_numeric(stats["slope"], fmt=".6f", highlight=True),
    )

    console.print(table)
    console.print(
        f"[bold green]♔ Key metric:[/] fitted RMSE = {stats['fitted_rmse']:.2f} cp",
        justify="center",
    )

    if table_png:
        png_written = _render_png_table(stats, table_png)

    return png_written


def _render_per_term_table(stats: dict, top_n: int = 12) -> None:
    per_term = stats.get("per_term", [])
    if not per_term:
        return

    top_entries = per_term[:top_n]

    if RICH_TABLE_AVAILABLE:
        console = Console(highlight=False, soft_wrap=False)
        console.print(
            "\n[bold magenta]Per-term contributions[/bold magenta]",
            justify="left",
        )
        console.print(
            "[dim]Bars indicate the mean absolute centipawn impact for each term; longer bars signal bigger sway in the quiet fit. Scale hints show how much to shrink a term to approach the target magnitude.[/dim]"
        )

        table = Table(
            box=box.SIMPLE_HEAVY,
            expand=True,
            show_header=True,
            header_style="bold cyan",
            pad_edge=False,
        )
        table.add_column("#", justify="right", style="bold cyan", width=3)
        table.add_column("Term", justify="left", style="bright_white")
        table.add_column("|mean| (cp)", justify="right", style="bold white")
        table.add_column("Impact", justify="left", style="white")
        table.add_column("raw σ", justify="right", style="white")
        table.add_column("scale→", justify="right", style="white")

        max_abs = max(entry["contrib_abs_mean"] for entry in top_entries)
        for rank, entry in enumerate(top_entries, start=1):
            abs_mean = entry["contrib_abs_mean"]
            bar = _term_bar(abs_mean, max_abs)
            raw_std = entry["raw_std"]
            scale_hint = entry["scale_hint"]
            scale_text = "∞" if math.isinf(scale_hint) else f"×{scale_hint:.3f}"
            table.add_row(
                f"{rank}",
                entry["name"],
                f"{abs_mean:,.2f}",
                bar,
                f"{raw_std:,.2f}",
                scale_text,
            )

        console.print(table)
    else:
        print("\nPer-term contributions (largest |mean| first):")
        print("Install 'rich' for colored bar visualization.")
        header = (
            f"{'#':>2}  {'term':<18}  {'|c|_mean':>9}  {'raw_std':>8}  "
            f"{'c_std':>8}  {'|c|max':>8}  {'scale->':>8}"
        )
        print(header)
        print("-" * len(header))
        for idx, entry in enumerate(top_entries, start=1):
            name = entry["name"][:18]
            display_abs = entry["contrib_abs_mean"]
            scale_hint = entry["scale_hint"]
            scale_text = "inf" if math.isinf(scale_hint) else f"{scale_hint:.3f}"
            print(
                f"{idx:2d}  {name:<18}  {display_abs:9.2f}  {entry['raw_std']:8.2f}  "
                f"{entry['contrib_std']:8.2f}  {entry['contrib_abs_max']:8.2f}  {scale_text:>8}"
            )


def _render_png_table(stats: dict, path: pathlib.Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"warning: unable to render PNG table ({exc})")
        return False

    rows = []
    highlights = []
    for idx, (label, values, fmt, higher) in enumerate(_metric_specs(stats)):
        formatted, winning_index = _plain_metric_row(values, fmt, higher)
        rows.append([label, *formatted])
        highlights.append(winning_index + 1 if winning_index >= 0 else -1)

    extras = [
        ["Samples", f"{stats['count']:,}", "", ""],
        ["Quiet Cap (cp)", "", "", f"{stats['quiet_cap']:.1f}"],
        ["Intercept (cp)", "", "", f"{stats['intercept']:.2f}"],
        ["Slope", "", "", f"{stats['slope']:.6f}"],
    ]

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    fig.patch.set_facecolor("#0f172a")
    ax.axis("off")

    col_labels = ["Metric", "Baseline ♖", "Raw ♗", "Fitted ♕"]
    cell_text = rows + [[""] * len(col_labels)] + extras

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.scale(1.0, 1.4)

    header_color = "#1e293b"
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for col in range(len(col_labels)):
        header_cell = table[(0, col)]
        header_cell.set_facecolor(header_color)
        header_cell.set_edgecolor("#38bdf8")
        header_cell.set_text_props(color="#e2e8f0", weight="bold")

    for row_index, row in enumerate(rows, start=1):
        for col in range(len(col_labels)):
            cell = table[(row_index, col)]
            cell.set_edgecolor("#334155")
            cell.set_facecolor("#0f172a")
            cell.set_text_props(color="#f8fafc")
        best_col = highlights[row_index - 1]
        if best_col >= 0:
            cell = table[(row_index, best_col)]
            cell.set_facecolor("#14532d")
            cell.set_text_props(color="#bbf7d0", weight="bold")

    divider_row = len(rows) + 1
    for col in range(len(col_labels)):
        cell = table[(divider_row, col)]
        cell.visible_edges = ""

    extras_start = divider_row + 1
    for idx, extra in enumerate(extras):
        row_pos = extras_start + idx
        for col in range(len(col_labels)):
            cell = table[(row_pos, col)]
            cell.set_edgecolor("#334155")
            if col == 0:
                cell.set_facecolor("#0f172a")
                cell.set_text_props(color="#f8fafc", weight="bold")
            else:
                cell.set_facecolor("#0f172a")
                cell.set_text_props(color="#cbd5f5")
        # highlight fitted column when available
        fitted_val = extra[3]
        if fitted_val:
            cell = table[(row_pos, 3)]
            cell.set_facecolor("#14532d")
            cell.set_text_props(color="#bbf7d0", weight="bold")

    fig.text(
        0.02,
        0.96,
        "Skaks Quiet Fit Summary",
        color="#e2e8f0",
        fontsize=14,
        weight="bold",
    )
    fig.text(
        0.02,
        0.92,
        f"Samples: {stats['count']:,}",
        color="#94a3b8",
        fontsize=10,
    )
    fig.text(
        0.02,
        0.1,
        f"Key metric (RMSE): {stats['fitted_rmse']:.2f} cp",
        color="#22c55e",
        fontsize=11,
        weight="bold",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def _metric_specs(
    stats: dict,
) -> List[tuple[str, tuple[float, float, float], str, bool]]:
    return [
        (
            "RMSE (↓)",
            (
                stats["baseline_rmse"],
                stats["raw_rmse"],
                stats["fitted_rmse"],
            ),
            ".2f",
            False,
        ),
        (
            "MAE (↓)",
            (
                stats["baseline_mae"],
                stats["raw_mae"],
                stats["fitted_mae"],
            ),
            ".2f",
            False,
        ),
        (
            "Pearson r (↑)",
            (
                stats["baseline_r"],
                stats["raw_r"],
                stats["fitted_r"],
            ),
            "+.3f",
            True,
        ),
    ]


def _format_metric_row(
    label: str,
    values: tuple[float, float, float],
    fmt: str,
    higher_is_better: bool,
) -> List[str]:
    formatted, best_index = _plain_metric_row(values, fmt, higher_is_better)
    row = [label]
    for idx, text in enumerate(formatted):
        highlight = idx == best_index
        row.append(_format_numeric_text(text, highlight))
    return row


def _plain_metric_row(
    values: tuple[float, float, float],
    fmt: str,
    higher_is_better: bool,
) -> tuple[List[str], int]:
    formatted = []
    best_index = -1
    comparable: List[tuple[int, float]] = []
    for idx, value in enumerate(values):
        if _is_nan(value):
            formatted.append("—")
        else:
            formatted.append(format(value, fmt))
            comparable.append((idx, value))
    if comparable:
        key_fn = lambda item: item[1]
        selected = (
            max(comparable, key=key_fn)
            if higher_is_better
            else min(comparable, key=key_fn)
        )
        best_index = selected[0]
    return formatted, best_index


def _format_numeric(value: float, fmt: str = ".2f", highlight: bool = False) -> str:
    if _is_nan(value):
        return "—"
    text = format(value, fmt)
    if highlight and RICH_TABLE_AVAILABLE:
        return f"[bold green]{text}[/]"
    return text


def _format_numeric_text(text: str, highlight: bool) -> str:
    if highlight and RICH_TABLE_AVAILABLE:
        return f"[bold green]{text}[/]"
    return text


def _is_nan(value: float) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _write_scaled_params(
    *,
    stats: dict,
    params_path: pathlib.Path | None,
    output_path: pathlib.Path,
    scale_min: float,
    scale_max: float,
    include_specs: Sequence[str] | None,
    exclude_specs: Sequence[str] | None,
) -> List[tuple[int, str, float]] | None:
    try:
        mg_weights, eg_weights = _load_phase_weights(params_path)
    except Exception:
        return None

    include_indices = _resolve_term_filter(include_specs)
    exclude_indices = _resolve_term_filter(exclude_specs)

    applied: List[tuple[int, str, float]] = []
    scale_min = max(0.0, scale_min)
    scale_max = max(scale_min if scale_min > 0 else 0.0, scale_max)

    for entry in stats.get("per_term", []):
        idx = entry["index"]
        name = entry["name"]
        if not _term_allowed(idx, include_indices, exclude_indices):
            continue
        hint = entry.get("scale_hint")
        if not isinstance(hint, (int, float)) or not math.isfinite(hint) or hint <= 0.0:
            continue
        factor = max(scale_min, min(scale_max, float(hint)))
        if abs(factor - 1.0) < 1e-4:
            continue
        mg_weights[idx] = float(mg_weights[idx] * factor)
        eg_weights[idx] = float(eg_weights[idx] * factor)
        applied.append((idx, name, factor))

    payload = _load_base_payload(params_path)
    eval_section = payload.setdefault("evaluation", {})
    eval_section["phase_weights_mg"] = [float(round(val, 6)) for val in mg_weights]
    eval_section["phase_weights_eg"] = [float(round(val, 6)) for val in eg_weights]
    eval_section["eval_quiet_cap"] = float(stats["quiet_cap"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        from skaks_opt.yaml_utils import dump_yaml

        dump_yaml(payload, handle, sort_keys=False)

    return applied


def _load_base_payload(params_path: pathlib.Path | None) -> dict:
    if params_path and params_path.exists():
        with params_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if isinstance(data, dict):
            return data.copy()
    return {}


def _resolve_term_filter(specs: Sequence[str] | None) -> set[int] | None:
    if not specs:
        return None
    indices: set[int] = set()
    for spec in specs:
        if spec is None:
            continue
        token = spec.strip()
        if not token:
            continue
        try:
            idx = int(token)
        except ValueError:
            idx = _lookup_term_index(token)
        else:
            if idx < 0 or idx >= len(_TERM_NAMES):
                raise ValueError(f"term index out of range: {token}")
        indices.add(idx)
    return indices


def _lookup_term_index(name: str) -> int:
    normalized = name.strip().lower()
    for idx, candidate in enumerate(_TERM_NAMES):
        if candidate.lower() == normalized:
            return idx
    raise ValueError(f"unknown term name: {name}")


def _term_allowed(
    idx: int, include_indices: set[int] | None, exclude_indices: set[int] | None
) -> bool:
    if exclude_indices and idx in exclude_indices:
        return False
    if include_indices is None:
        return True
    return idx in include_indices


def _term_bar(value: float, max_value: float, length: int = 24) -> str:
    if max_value <= 0.0:
        return f"[{_TERM_BAR_TRAIL}]{_TERM_BAR_CHAR * length}[/]"
    ratio = max(0.0, min(1.0, value / max_value))
    filled = min(length, max(0, int(round(ratio * length))))
    if ratio > 0.0 and filled == 0:
        filled = 1
    color_index = min(
        len(_TERM_BAR_COLORS) - 1, int(ratio * (len(_TERM_BAR_COLORS) - 1))
    )
    color = _TERM_BAR_COLORS[color_index]
    filled_str = _TERM_BAR_CHAR * filled
    empty_str = _TERM_BAR_CHAR * (length - filled)
    if empty_str:
        return f"[{color}]{filled_str}[/][{_TERM_BAR_TRAIL}]{empty_str}[/]"
    return f"[{color}]{filled_str}[/]"


def _format_progress_status(raw_stats: RunningStats, engine_stats: RunningStats) -> str:
    parts: List[str] = []
    if raw_stats.count >= 2:
        mae = raw_stats.mae()
        pear = raw_stats.pearson()
        if not math.isnan(mae):
            parts.append(f"raw_mae {mae:.1f}")
        if not math.isnan(pear):
            parts.append(f"raw_r {pear:+.3f}")
    if engine_stats.count >= 2:
        mae = engine_stats.mae()
        pear = engine_stats.pearson()
        if not math.isnan(mae):
            parts.append(f"eng_mae {mae:.1f}")
        if not math.isnan(pear):
            parts.append(f"eng_r {pear:+.3f}")
    return " ".join(parts)
