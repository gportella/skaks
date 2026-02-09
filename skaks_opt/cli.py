"""Unified CLI entry point for Skaks optimization and tooling."""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

from skaks_opt.arena import add_subparser as add_arena_subparser
from skaks_opt.arena import run_arena
from skaks_opt.arena_sweep import add_subparser as add_arena_sweep_subparser
from skaks_opt.arena_sweep import run_sweep as run_arena_sweep
from skaks_opt.dataset import add_subparser as add_dataset_subparser
from skaks_opt.dataset import run_dataset
from skaks_opt.eval_stats import add_subparser as add_eval_stats_subparser
from skaks_opt.eval_stats import run_eval_stats
from skaks_opt.fen_phase_split import \
  add_subparser as add_fen_phase_split_subparser
from skaks_opt.fen_phase_split import run_fen_phase_split
from skaks_opt.param_optimize import \
  add_subparser as add_param_optimize_subparser
from skaks_opt.param_optimize import run_param_optimize
from skaks_opt.perf_pgn import add_subparser as add_perf_pgn_subparser
from skaks_opt.perf_pgn import run_perf_pgn
from skaks_opt.selfplay import SelfPlayConfig, run_selfplay

CATEGORY_PARAMETER_TUNING = "Parameter tuning"
CATEGORY_DATA_ANALYSIS = "Data & analysis"
CATEGORY_HELPERS = "Helper tools"
CATEGORY_ORDER = [
    CATEGORY_PARAMETER_TUNING,
    CATEGORY_DATA_ANALYSIS,
    CATEGORY_HELPERS,
]


class CategorizedHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Help formatter that groups subcommands by topic."""

    def _format_action(self, action: argparse.Action) -> str:
        if (
            isinstance(action, argparse._SubParsersAction)
            and getattr(action, "_category_order", None)
        ):
            return self._format_categorized_subparsers(action)
        return super()._format_action(action)

    def _format_categorized_subparsers(
        self, action: argparse._SubParsersAction
    ) -> str:
        title = getattr(action, "title", None) or "subcommands"
        raw_indent = getattr(self, "_current_indent", "")
        if isinstance(raw_indent, str):
            indent_str = raw_indent
        else:
            try:
                indent_val = int(raw_indent)
            except (TypeError, ValueError):
                indent_val = 0
            indent_str = " " * max(indent_val, 0)

        parts: List[str] = ["\n", f"{indent_str}{title}:\n"]
        description = getattr(action, "description", None)
        if description:
            parts.append(self._format_text(description))

        name_order = list(action._name_parser_map.keys())
        entries: Dict[str, List[tuple[str, str]]] = {
            category: [] for category in getattr(action, "_category_order", [])
        }
        extras: List[tuple[str, str]] = []
        for name in name_order:
            parser = action._name_parser_map[name]
            category = getattr(parser, "_category", None)
            short_help = getattr(parser, "_short_help", "")
            if category in entries:
                entries[category].append((name, short_help))
            else:
                extras.append((name, short_help))

        if extras:
            extras_category = getattr(action, "_fallback_category", "Other")
            entries.setdefault(extras_category, []).extend(extras)

        name_width = 0
        for rows in entries.values():
            for name, _ in rows:
                name_width = max(name_width, len(name))
        max_help_pos = max(self._max_help_position - len(indent_str) - 4, 8)
        name_width = min(name_width, max_help_pos)

        category_indent = indent_str + " " * self._indent_increment
        command_indent = category_indent + " " * self._indent_increment

        for category in getattr(action, "_category_order", []):
            rows = entries.get(category, [])
            if not rows:
                continue
            parts.append(f"{category_indent}{category}:\n")
            for name, help_text in rows:
                padded = name.ljust(name_width)
                parts.append(f"{command_indent}{padded}  {help_text}\n")

        return "".join(parts)


def _register_category(
    subparsers: argparse._SubParsersAction,
    name: str,
    parser: argparse.ArgumentParser,
    category: str,
    help_text: Optional[str] = None,
) -> None:
    parser._category = category  # type: ignore[attr-defined]
    if help_text is None:
        help_text = _extract_help(subparsers, name)
    parser._short_help = help_text or ""  # type: ignore[attr-defined]


def _extract_help(subparsers: argparse._SubParsersAction, name: str) -> str:
    for choice in getattr(subparsers, "_choices_actions", []):
        choice_name = getattr(choice, "dest", None) or getattr(choice, "name", None)
        if choice_name == name:
            return getattr(choice, "help", "") or ""
    return ""


def main():
    parser = argparse.ArgumentParser(
        description="Skaks unified optimization/tooling CLI",
        formatter_class=CategorizedHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        title="Commands", dest="command", metavar="COMMAND", required=True
    )
    subparsers._category_order = CATEGORY_ORDER  # type: ignore[attr-defined]
    subparsers._fallback_category = CATEGORY_HELPERS  # type: ignore[attr-defined]

    # Parameter tuning -----------------------------------------------------
    param_parser = add_param_optimize_subparser(subparsers)
    _register_category(
        subparsers, "param-optimize", param_parser, CATEGORY_PARAMETER_TUNING
    )

    selfplay_help = "Beam/CMA self-play tuner across distributed arenas"
    selfplay_parser = subparsers.add_parser(
        "selfplay",
        help=selfplay_help,
        description=(
            "Modern self-play pipeline that perturbs search parameters, "
            "spawns arenas locally or via Dask, and keeps the best candidates. Use "
            "this when you want an end-to-end tuner with beam search or CMA-ES, "
            "as opposed to the arena-driven `param-optimize` loop."
        ),
    )
    _register_category(
        subparsers, "selfplay", selfplay_parser, CATEGORY_PARAMETER_TUNING, selfplay_help
    )

    optimize_help = "Tune search-only knobs using skaks --perf metrics"
    optimize_parser = subparsers.add_parser(
        "optimize-search",
        help=optimize_help,
        description=(
            "Use Optuna and repeated skaks --perf runs to minimize per-position "
            "time/nodes. This sticks to search.* settings and never plays games, "
            "so it complements arena-driven optimizers."
        ),
    )
    _register_category(
        subparsers,
        "optimize-search",
        optimize_parser,
        CATEGORY_PARAMETER_TUNING,
        optimize_help,
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
        "--pgn-min-ply",
        type=int,
        default=8,
        help="Minimum ply to sample from PGN (default: %(default)s)",
    )
    optimize_parser.add_argument(
        "--pgn-max-ply",
        type=int,
        default=40,
        help="Maximum ply to sample from PGN (default: %(default)s)",
    )
    optimize_parser.add_argument(
        "--pgn-seed",
        type=int,
        default=0,
        help="Random seed for PGN sampling (default: %(default)s)",
    )
    optimize_parser.add_argument(
        "--depth",
        type=int,
        default=6,
        help="Search depth for skaks --perf (default: %(default)s)",
    )
    optimize_parser.add_argument(
        "--search-nnue",
        action="store_true",
        help="Tune the search_nnue parameter block instead of search",
    )

    # Data & analysis ------------------------------------------------------
    dataset_parser = add_dataset_subparser(subparsers)
    _register_category(
        subparsers, "dataset-sample", dataset_parser, CATEGORY_DATA_ANALYSIS
    )

    fen_phase_parser = add_fen_phase_split_subparser(subparsers)
    _register_category(
        subparsers, "fen-phase-split", fen_phase_parser, CATEGORY_DATA_ANALYSIS
    )

    arena_parser = add_arena_subparser(subparsers)
    _register_category(subparsers, "arena", arena_parser, CATEGORY_DATA_ANALYSIS)

    arena_sweep_parser = add_arena_sweep_subparser(subparsers)
    _register_category(
        subparsers, "arena-sweep", arena_sweep_parser, CATEGORY_DATA_ANALYSIS
    )

    eval_stats_parser = add_eval_stats_subparser(subparsers)
    _register_category(
        subparsers, "eval-stats", eval_stats_parser, CATEGORY_DATA_ANALYSIS
    )

    perf_pgn_parser = add_perf_pgn_subparser(subparsers)
    _register_category(
        subparsers, "perf-pgn", perf_pgn_parser, CATEGORY_DATA_ANALYSIS
    )

    # Helper tools ---------------------------------------------------------
    monitor_help = "Track ongoing scan_out results"
    monitor_parser = subparsers.add_parser(
        "monitor",
        help=monitor_help,
        description=(
            "Track the CSV/JSON artifacts produced by scripts/scan_progress.py to "
            "keep an eye on long-running scan_out jobs. This is read-only and "
            "never schedules arenas."
        ),
    )
    _register_category(
        subparsers, "monitor", monitor_parser, CATEGORY_HELPERS, monitor_help
    )
    monitor_parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top samples to display (default: %(default)s)",
    )
    monitor_parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Refresh interval in seconds (default: %(default)s)",
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
        help=(
            "Limit perturbations to parameters matching this prefix (repeatable). "
            "Defaults to search_nnue.* when omitted."
        ),
    )
    selfplay_parser.add_argument(
        "--exclude-prefix",
        action="append",
        help="Exclude parameters matching this prefix from perturbations (repeatable)",
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
        "--dask-scheduler",
        help="Dask scheduler address for distributed arenas (e.g. tcp://host:8786)",
    )
    selfplay_parser.add_argument(
        "--dask-workers",
        type=int,
        help="Spawn a local Dask cluster with this many workers",
    )
    selfplay_parser.add_argument(
        "--dask-threads",
        type=int,
        help="Threads per local Dask worker (default: 1)",
    )
    selfplay_parser.add_argument(
        "--dask-shards",
        type=int,
        help="Override number of arena shards submitted to Dask",
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
        "--no-rich-progress",
        action="store_true",
        help="Disable rich table summaries (plain text output)",
    )
    selfplay_parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed for optimizer (default: %(default)s)",
    )

    args = parser.parse_args()
    if args.command == "arena":
        try:
            exit_code = run_arena(args)
        except ValueError as exc:
            print(f"arena: {exc}", file=sys.stderr)
            sys.exit(2)
        if exit_code:
            sys.exit(exit_code)
    elif args.command == "arena-sweep":
        run_arena_sweep(args)
    elif args.command == "dataset-sample":
        run_dataset(args)
    elif args.command == "fen-phase-split":
        run_fen_phase_split(args)
    elif args.command == "eval-stats":
        run_eval_stats(args)
    elif args.command == "perf-pgn":
        run_perf_pgn(args)
    elif args.command == "param-optimize":
        run_param_optimize(args)
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
            dask_scheduler=args.dask_scheduler,
            dask_local_workers=args.dask_workers,
            dask_local_threads=args.dask_threads,
            dask_shard_hint=args.dask_shards,
            rich_progress=not args.no_rich_progress,
        )
        run_selfplay(config)
    elif args.command == "optimize-search":
        from skaks_opt.optimize_search import run_optimize_search

        run_optimize_search(args)
    elif args.command == "monitor":
        from scripts import scan_progress

        scan_progress.run(top=args.top, interval=args.interval)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
