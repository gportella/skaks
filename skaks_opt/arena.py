from __future__ import annotations

import argparse
import os

from skaks_opt.arena_runner import (DEFAULT_DEPTH, DEFAULT_ELO_K_FACTOR,
                                    DEFAULT_ELO_OPPONENT, DEFAULT_ELO_START,
                                    DEFAULT_ELO_STORE, DEFAULT_GAMES,
                                    DEFAULT_MATCH_LIMIT, RECOMMENDED_DB_NAME,
                                    non_negative_int, parse_uci_option,
                                    positive_float, positive_int)
from skaks_opt.arena_runner import run_batch as _run_batch

__all__ = ["add_subparser", "run_arena"]


def _configure_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--games",
        type=positive_int,
        default=DEFAULT_GAMES,
        help=f"Number of new games to play (default: {DEFAULT_GAMES})",
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=DEFAULT_MATCH_LIMIT,
        help=f"Half-move limit forwarded to fight script (default: {DEFAULT_MATCH_LIMIT})",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default=None,
        help="Reference engine binary (default: delegated to fight script)",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default=None,
        help="Opponent engine binary (default: delegated to fight script)",
    )
    parser.add_argument(
        "--engine-params",
        type=str,
        help="Path to params file for the reference engine (passed as --engine-params)",
    )
    parser.add_argument(
        "--engine-nnue",
        type=str,
        help="Path to NNUE weights for the reference engine (passed as --engine-nnue)",
    )
    parser.add_argument(
        "--engine-eval",
        type=str,
        help="Evaluation mode override for the reference engine (forwarded as --eval)",
    )
    parser.add_argument(
        "--engine-uci-option",
        action="append",
        type=parse_uci_option,
        default=[],
        help="UCI option for the reference engine (repeatable, format name=value)",
    )
    parser.add_argument(
        "--eval",
        dest="engine_eval",
        type=str,
        help="Alias of --engine-eval",
    )
    parser.add_argument(
        "--engine-label",
        type=str,
        help="Display label for the reference engine in summary/Elo (default: engine basename)",
    )
    parser.add_argument(
        "--threads",
        type=non_negative_int,
        default=4,
        help="Search threads for the reference engine (0 = auto)",
    )
    parser.add_argument(
        "--opponent-params",
        type=str,
        help="Path to params file for the opponent engine (passed as --opponent-params)",
    )
    parser.add_argument(
        "--opponent-nnue",
        type=str,
        help="Path to NNUE weights for the opponent engine (passed as --opponent-nnue)",
    )
    parser.add_argument(
        "--opponent-eval",
        type=str,
        help="Evaluation mode override for the opponent engine (forwarded as --eval)",
    )
    parser.add_argument(
        "--opponent-uci-option",
        action="append",
        type=parse_uci_option,
        default=[],
        help="UCI option for the opponent engine (repeatable, format name=value)",
    )
    parser.add_argument(
        "--opponent-depth-factor",
        type=positive_float,
        help="Scale factor applied to depth for the opponent (e.g. 0.6)",
    )
    parser.add_argument(
        "--opponent-label",
        type=str,
        help="Display label for the opponent engine in summary/Elo (default: opponent basename)",
    )
    parser.add_argument(
        "--opponent-threads",
        type=non_negative_int,
        default=None,
        help="Search threads for the opponent engine (0 = auto)",
    )
    parser.add_argument(
        "--stockfish",
        action="store_true",
        help="Shortcut for --opponent stockfish",
    )
    parser.add_argument(
        "--database",
        type=str,
        help=(
            "SQLite database file path (recommended: "
            f"{RECOMMENDED_DB_NAME}); omit to disable persistence"
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=positive_int,
        default=None,
        help="Maximum number of concurrent games to run",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume numbering from the existing dataset (appends new games)",
    )
    parser.add_argument(
        "--timeout",
        type=positive_float,
        default=None,
        help="Timeout forwarded to fight script (seconds)",
    )
    timing_group = parser.add_mutually_exclusive_group()
    timing_group.add_argument(
        "--depth",
        type=positive_int,
        help="Depth passed to the reference engine",
    )
    timing_group.add_argument(
        "--time-per-move",
        type=positive_float,
        help="Seconds per move for the reference engine",
    )
    timing_group.add_argument(
        "--clock",
        type=positive_float,
        help="Primary clock time in seconds for the reference engine",
    )
    parser.add_argument(
        "--opponent-time-per-move",
        type=positive_float,
        help="Seconds per move for the opponent engine",
    )
    parser.add_argument(
        "--opponent-clock",
        type=positive_float,
        help="Clock time in seconds for the opponent engine",
    )
    parser.add_argument(
        "--increment",
        type=positive_float,
        help="Increment in seconds for the reference engine (requires --clock)",
    )
    parser.add_argument(
        "--opponent-increment",
        type=positive_float,
        help="Increment in seconds for the opponent engine (requires --clock)",
    )
    parser.add_argument(
        "--moves-to-go",
        type=positive_int,
        help="Approximate moves remaining to next time control",
    )
    parser.add_argument(
        "--no-handicap",
        action="store_true",
        help="Disable opponent handicap when forwarding to fight script",
    )
    parser.add_argument(
        "--handicap-factor",
        type=positive_float,
        default=0.65,
        help="Scaling factor applied to opponent time when handicap is enabled",
    )
    parser.add_argument(
        "--handicap-depth",
        type=non_negative_int,
        default=3,
        help="Depth advantage retained by reference engine when handicap is enabled",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print stdout/stderr for every game (default: only failures)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-game output (still writes summary JSON if requested)",
    )
    parser.add_argument(
        "--elo-start",
        type=float,
        default=DEFAULT_ELO_START,
        help=(
            "Starting Elo for skaks when no stored rating is present "
            f"(default: {DEFAULT_ELO_START})"
        ),
    )
    parser.add_argument(
        "--elo-opponent",
        type=float,
        default=DEFAULT_ELO_OPPONENT,
        help=f"Assumed opponent Elo rating (default: {DEFAULT_ELO_OPPONENT})",
    )
    parser.add_argument(
        "--elo-k-factor",
        type=positive_float,
        default=DEFAULT_ELO_K_FACTOR,
        help=f"K-factor for Elo updates (default: {DEFAULT_ELO_K_FACTOR})",
    )
    parser.add_argument(
        "--elo-store",
        type=str,
        default=DEFAULT_ELO_STORE,
        help="Path to JSON file for persisting skaks Elo between runs",
    )
    parser.add_argument(
        "--no-elo-store",
        action="store_true",
        help="Skip loading/saving Elo state; compute only for this batch",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        help="Optional path to write a JSON summary (counts, Elo, timings)",
    )
    return parser


def add_subparser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "arena",
        help="Run engine-vs-engine matches with optional SQLite logging",
        description=(
            "Launch head-to-head matches between a reference engine and an opponent, "
            "recording JSON summaries, Elo estimates, and optional SQLite history. "
            "This is the low-level building block that sweep and tuning commands reuse."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    return _configure_parser(parser)


def _normalize_args(args: argparse.Namespace) -> None:
    if args.stockfish and args.opponent:
        raise ValueError("--stockfish may not be combined with --opponent")
    if args.stockfish:
        args.opponent = "stockfish"
    if args.resume and not args.database:
        raise ValueError("--resume requires --database")

    mode_count = sum(
        flag is not None for flag in (args.depth, args.time_per_move, args.clock)
    )
    if mode_count == 0:
        args.depth = DEFAULT_DEPTH
    if mode_count > 1:
        raise ValueError("choose exactly one of --depth, --time-per-move, or --clock")

    if args.clock is None and args.opponent_clock is not None:
        raise ValueError("--opponent-clock requires --clock")
    if args.clock is None and (
        args.increment is not None or args.opponent_increment is not None
    ):
        raise ValueError("increments require --clock")
    if args.moves_to_go is not None and args.moves_to_go <= 0:
        raise ValueError("--moves-to-go must be positive")

    if args.no_handicap:
        args.handicap_factor = 1.0
        args.handicap_depth = 0

    args.handicap_enabled = not args.no_handicap

    if args.concurrency is None:
        detected = os.cpu_count() or 1
        args.concurrency = max(1, min(detected, 4))


def run_arena(args: argparse.Namespace) -> int:
    _normalize_args(args)
    exit_code = _run_batch(args)
    return 0 if exit_code is None else exit_code
