"""
Unified CLI entry point for Skaks optimization and fitting.
"""

import argparse
from pathlib import Path
from typing import List, Optional
import sys

from skaks_opt.texel import run_texel
from skaks_opt.fit import run_fit
from skaks_opt.selfplay import SelfPlayConfig, run_selfplay


def main():
    parser = argparse.ArgumentParser(
        description="Skaks unified optimization/fitting CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Optimize search subcommand
    optimize_parser = subparsers.add_parser(
        "optimize-search", help="Optimize search parameters with Optuna"
    )
    optimize_parser.add_argument(
        "--quiet", action="store_true", help="Suppress output (only show best result)"
    )
    optimize_parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of optimization steps (default: %(default)s)",
    )
    optimize_parser.add_argument(
        "--games",
        type=int,
        default=10,
        help="Number of games per perf run (default: %(default)s)",
    )
    optimize_parser.add_argument(
        "--output",
        type=str,
        default="best_for_chrono.yaml",
        help="Output YAML file for best parameter set (default: %(default)s)",
    )
    optimize_parser.add_argument(
        "--pgn",
        type=str,
        help="PGN file to sample initial positions from (optional)",
    )
    optimize_parser.add_argument(
        "--depth",
        type=int,
        default=6,
        help="Search depth for skaks --perf (default: %(default)s)",
    )

    # Texel fitting subcommand
    texel_parser = subparsers.add_parser("texel", help="Run Texel fitting")
    texel_parser.add_argument("--params", required=True, help="YAML parameter file")
    texel_parser.add_argument("--games", type=int, default=100, help="Number of games")

    # Parameter fitting subcommand
    fit_parser = subparsers.add_parser("fit", help="Run parameter fitting")
    fit_parser.add_argument("--params", required=True, help="YAML parameter file")
    fit_parser.add_argument("--games", type=int, default=100, help="Number of games")

    # Self-play optimization subcommand
    selfplay_parser = subparsers.add_parser(
        "selfplay", help="Run self-play optimization"
    )
    selfplay_parser.add_argument("--params", required=True, help="YAML parameter file")
    selfplay_parser.add_argument(
        "--baseline-params",
        help="Baseline YAML parameters (defaults to start params)",
    )
    selfplay_parser.add_argument(
        "--start-params",
        help="Initial YAML parameters (defaults to baseline params)",
    )
    selfplay_parser.add_argument(
        "--output",
        help="Output path for best parameter set (defaults next to start params)",
    )
    selfplay_parser.add_argument(
        "--include-prefix",
        action="append",
        help="Limit perturbations to parameters matching this prefix (repeatable)",
    )
    selfplay_parser.add_argument(
        "--exclude-prefix",
        action="append",
        help="Exclude parameters matching this prefix from perturbations (repeatable)",
    )
    selfplay_parser.add_argument(
        "--phase-weights-only",
        action="store_true",
        help="Restrict tuning to phase weight parameters",
    )
    selfplay_parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Number of games per evaluation (default: %(default)s)",
    )
    selfplay_parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Optimization iterations (default: %(default)s)",
    )
    selfplay_parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Arena repeats per candidate for stability (default: %(default)s)",
    )
    selfplay_parser.add_argument(
        "--noise",
        type=float,
        default=0.15,
        help="Log-normal noise sigma for beam perturbations (default: %(default)s)",
    )
    selfplay_parser.add_argument(
        "--strategy",
        choices=["beam", "cma"],
        default="beam",
        help="Candidate generation strategy (default: %(default)s)",
    )
    selfplay_parser.add_argument(
        "--beam-size",
        type=int,
        default=4,
        help="Number of candidates per iteration for beam search (default: %(default)s)",
    )
    selfplay_parser.add_argument("--engine", required=True, help="Engine binary")
    selfplay_parser.add_argument("--depth", type=int, help="Search depth")
    selfplay_parser.add_argument(
        "--time-per-move", type=float, help="Time per move (sec)"
    )
    selfplay_parser.add_argument("--clock", type=float, help="Clock time (sec)")
    selfplay_parser.add_argument(
        "--arena-workers",
        type=int,
        default=1,
        help="Parallel arena shards (multiprocessing) (default: %(default)s)",
    )
    selfplay_parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Reserved for engine-level concurrency (future use) (default: %(default)s)",
    )
    selfplay_parser.add_argument(
        "--cma-popsize",
        type=int,
        help="Override CMA-ES population size",
    )
    selfplay_parser.add_argument(
        "--cma-sigma",
        type=float,
        help="Override CMA-ES initial sigma",
    )
    selfplay_parser.add_argument(
        "--arena-pgn",
        help="Sample start positions from PGN instead of default start",
    )
    selfplay_parser.add_argument(
        "--arena-min-ply",
        type=int,
        default=6,
        help="Minimum ply to sample from PGN games (default: %(default)s)",
    )
    selfplay_parser.add_argument(
        "--arena-max-ply",
        type=int,
        default=40,
        help="Maximum ply to sample from PGN games (default: %(default)s)",
    )
    selfplay_parser.add_argument(
        "--arena-seed",
        type=int,
        default=1337,
        help="Random seed for PGN sampling (default: %(default)s)",
    )
    selfplay_parser.add_argument(
        "--start-fen",
        action="append",
        help="Explicit start FEN (repeatable)",
    )
    selfplay_parser.add_argument(
        "--child-output",
        action="store_true",
        help="Silence per-repeat progress output",
    )
    selfplay_parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed for optimizer (default: %(default)s)",
    )

    args = parser.parse_args()
    if args.command == "texel":
        run_texel(args.params, args.games)
    elif args.command == "fit":
        run_fit(args.params, args.games)
    elif args.command == "selfplay":

        def _to_path(value: Optional[str]) -> Optional[Path]:
            return Path(value).resolve() if value else None

        include_prefixes: Optional[List[str]] = args.include_prefix
        exclude_prefixes: Optional[List[str]] = args.exclude_prefix

        # If clock is set, force depth and time_per_move to None
        if args.clock is not None:
            depth = None
            time_per_move = None
        elif args.depth is not None:
            depth = args.depth
            time_per_move = args.time_per_move
        elif args.time_per_move is not None:
            depth = None
            time_per_move = args.time_per_move
        else:
            # Default to depth=4 if nothing is set
            depth = 4
            time_per_move = None
        config = SelfPlayConfig(
            engine=args.engine,
            baseline_params=_to_path(args.baseline_params) or _to_path(args.params),
            start_params=_to_path(args.start_params) or _to_path(args.params),
            output=_to_path(args.output) if args.output else None,
            include_prefix=include_prefixes,
            exclude_prefix=exclude_prefixes,
            phase_weights_only=args.phase_weights_only,
            games=args.games,
            iterations=args.iterations,
            repeats=args.repeats,
            noise=args.noise,
            strategy=args.strategy,
            beam_size=args.beam_size,
            depth=depth,
            time_per_move=time_per_move,
            clock=args.clock,
            concurrency=args.concurrency,
            arena_workers=args.arena_workers,
            arena_pgn=_to_path(args.arena_pgn) if args.arena_pgn else None,
            arena_min_ply=args.arena_min_ply,
            arena_max_ply=args.arena_max_ply,
            arena_seed=args.arena_seed,
            child_output=args.child_output,
            cma_popsize=args.cma_popsize,
            cma_sigma=args.cma_sigma,
            seed=args.seed,
            start_fens=args.start_fen,
        )
        run_selfplay(config)
    elif args.command == "optimize-search":
        from skaks_opt import optimize_search

        sys.modules["__main__"].quiet = args.quiet
        sys.modules["__main__"].trials = args.trials
        sys.modules["__main__"].games = args.games
        sys.modules["__main__"].output = args.output
        sys.modules["__main__"].pgn = args.pgn
        sys.modules["__main__"].depth = args.depth
        optimize_search.main_entry()
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
