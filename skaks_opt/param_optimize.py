"""Param optimizer CLI integration layer."""

from __future__ import annotations

import argparse
import importlib.util
import types
from pathlib import Path


_PARAM_OPTIMIZE_SPEC_NAME = "skaks_param_optimize_script"
_MODULE_CACHE: types.ModuleType | None = None


def _load_param_optimize_module() -> types.ModuleType:
    global _MODULE_CACHE
    if _MODULE_CACHE is not None:
        return _MODULE_CACHE
    base_dir = Path(__file__).resolve().parents[1]
    script_path = base_dir / "tuning" / "param_optimize.py"
    if not script_path.exists():
        raise FileNotFoundError(
            f"Expected optimizer script at {script_path}, but it was not found."
        )
    spec = importlib.util.spec_from_file_location(
        _PARAM_OPTIMIZE_SPEC_NAME, script_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MODULE_CACHE = module
    return module


def _configure_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--engine", default="skaks", help="Path to engine binary")
    parser.add_argument(
        "--opponent", default="stockfish", help="Opponent engine binary"
    )
    parser.add_argument(
        "--external-opponent",
        action="store_true",
        help="Opponent is external engine, not self-play",
    )
    parser.add_argument("--baseline-params", help="YAML params for baseline (opponent)")
    parser.add_argument(
        "--start-params",
        help="Initial YAML params for candidate (defaults to baseline)",
    )
    parser.add_argument(
        "--output",
        help="Where to write best params (default: <start/baseline>_optimized.yaml)",
    )
    parser.add_argument(
        "--child-output",
        action="store_true",
        help="Show child match output (otherwise suppressed)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=40,
        help="Total games per candidate (split by color)",
    )
    parser.add_argument(
        "--iterations", type=int, default=5, help="Number of candidate evaluations"
    )
    parser.add_argument(
        "--baseline-decay",
        type=float,
        default=0.0,
        help=(
            "Per-iteration score relaxation applied to the stored best score. "
            "Small positive values make accepting new candidates gradually easier."
        ),
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.05,
        help="Lognormal stddev for multiplicative noise",
    )
    parser.add_argument("--depth", type=int, help="Depth per move")
    parser.add_argument("--time-per-move", type=float, help="Seconds per move")
    parser.add_argument("--clock", type=float, help="Clock time seconds")
    parser.add_argument(
        "--opponent-time-per-move", type=float, help="Opponent seconds per move"
    )
    parser.add_argument(
        "--opponent-depth-factor",
        type=float,
        help="Scale factor applied to opponent depth when external opponent",
    )
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent games")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeated evaluations per candidate (averaged)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=1,
        help="How many top candidates to keep each iteration",
    )
    parser.add_argument(
        "--use-arena-binding",
        action="store_true",
        help="Use internal arena binding (skaks_eval.arena)",
    )
    parser.add_argument(
        "--include-prefix",
        action="append",
        help="Only tune params whose dotted path starts with this prefix",
    )
    parser.add_argument(
        "--exclude-prefix",
        action="append",
        help="Skip params whose dotted path starts with this prefix",
    )
    parser.add_argument(
        "--phase-weights-only",
        action="store_true",
        help="Only tune evaluation.phase_weights arrays",
    )
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="Alias for --phase-weights-only",
    )
    parser.add_argument(
        "--strategy",
        choices=["beam", "cma"],
        default="beam",
        help="Search strategy: beam (default) or CMA-like",
    )
    parser.add_argument(
        "--arena-pgn",
        help="Sample start positions from PGN (requires python-chess)",
    )
    parser.add_argument(
        "--arena-min-ply",
        type=int,
        default=6,
        help="Minimum ply to sample from PGN",
    )
    parser.add_argument(
        "--arena-max-ply",
        type=int,
        default=60,
        help="Maximum ply to sample from PGN (0 = no cap)",
    )
    parser.add_argument(
        "--arena-seed",
        type=int,
        default=0,
        help="RNG seed for PGN sampling",
    )
    parser.add_argument(
        "--arena-workers",
        type=int,
        default=1,
        help="Parallel arena shards when using arena binding",
    )
    parser.add_argument(
        "--cma-popsize",
        type=int,
        help="Population size for CMA-like strategy",
    )
    parser.add_argument(
        "--cma-sigma",
        type=float,
        help="Step scale for CMA-like strategy",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum score floor to accept a candidate",
    )
    parser.add_argument(
        "--force-accept-first",
        type=int,
        default=0,
        help="Force-accept the first N replacements",
    )
    return parser


def add_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "param-optimize",
        help="Run parameter optimizer loop against a baseline or external opponent",
    )
    return _configure_parser(parser)


def run_param_optimize(args: argparse.Namespace) -> None:
    module = _load_param_optimize_module()
    optimize_loop = getattr(module, "optimize_loop", None)
    if optimize_loop is None:
        raise AttributeError("Optimizer script does not expose optimize_loop")
    optimize_loop(args)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Self-play parameter optimizer (beam + repeats)"
    )
    _configure_parser(parser)
    parsed = parser.parse_args(argv)
    run_param_optimize(parsed)


if __name__ == "__main__":
    main()
