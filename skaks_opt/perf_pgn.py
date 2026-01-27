"""PGN-backed perf profiling using the skaks_eval Python bindings."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional


def add_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "perf-pgn",
        help="Run perf profiling from PGN positions",
        description=(
            "Sample positions from a PGN file and run skaks_eval arena_perf to "
            "collect nodes/time stats. This is intended for quick perf sweeps "
            "while tuning search parameters."
        ),
    )
    parser.add_argument("--pgn", required=True, help="Path to PGN file")
    parser.add_argument(
        "--max-positions",
        type=int,
        default=24,
        help="Maximum number of start positions to sample (default: %(default)s)",
    )
    parser.add_argument(
        "--min-ply",
        type=int,
        default=8,
        help="Minimum ply to sample from each game (default: %(default)s)",
    )
    parser.add_argument(
        "--max-ply",
        type=int,
        default=40,
        help="Maximum ply to sample from each game (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for PGN sampling (default: %(default)s)",
    )
    parser.add_argument(
        "--params",
        type=str,
        help="YAML parameter file (search/search_nnue overrides)",
    )
    parser.add_argument(
        "--baseline-params",
        type=str,
        help="Optional baseline YAML (defaults to --params or engine defaults)",
    )
    parser.add_argument(
        "--search-nnue",
        action="store_true",
        help="Force use of search_nnue block when params are provided",
    )
    parser.add_argument(
        "--use-native",
        action="store_true",
        help="Force native (HCE) evaluation and search params",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=0,
        help="Number of games to run (0 = once per sampled position)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Search depth per move (default: %(default)s)",
    )
    parser.add_argument(
        "--movetime-ms",
        type=int,
        default=0,
        help="Fixed move time in milliseconds (default: %(default)s)",
    )
    parser.add_argument(
        "--node-limit",
        type=int,
        default=0,
        help="Node limit per move (default: %(default)s)",
    )
    parser.add_argument(
        "--max-plies",
        type=int,
        default=160,
        help="Max plies per game (default: %(default)s)",
    )
    return parser


def _load_yaml(path: Path) -> Dict:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"PyYAML is required to load params: {exc}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _force_search_nnue(payload: Dict) -> Dict:
    if "search_nnue" not in payload and "search" in payload:
        payload = dict(payload)
        payload["search_nnue"] = payload.get("search", {})
    return payload


def _sample_pgn_positions(
    pgn_path: Path,
    max_positions: int,
    min_ply: int,
    max_ply: int,
    seed: int,
) -> List[str]:
    try:
        import chess.pgn  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"python-chess is required for PGN sampling: {exc}")

    rng = random.Random(seed)
    fens: List[str] = []
    with pgn_path.open("r", encoding="utf-8") as handle:
        while len(fens) < max_positions:
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            board = game.board()
            moves = list(game.mainline_moves())
            if not moves:
                continue
            target = min_ply
            if max_ply > min_ply:
                target = rng.randint(min_ply, max_ply)
            target = max(1, target)
            for idx, mv in enumerate(moves, start=1):
                board.push(mv)
                if idx >= target:
                    break
            fens.append(board.fen())
    return fens


def run_perf_pgn(args: argparse.Namespace) -> int:
    from skaks_eval import arena_perf

    pgn_path = Path(args.pgn).expanduser().resolve()
    if not pgn_path.exists():
        raise SystemExit(f"PGN not found: {pgn_path}")

    if args.max_positions <= 0:
        raise SystemExit("--max-positions must be positive")

    fens = _sample_pgn_positions(
        pgn_path,
        max_positions=args.max_positions,
        min_ply=max(1, args.min_ply),
        max_ply=max(args.min_ply, args.max_ply),
        seed=args.seed,
    )
    if not fens:
        raise SystemExit("No positions sampled from PGN")

    base_params: Optional[Dict] = None
    cand_params: Optional[Dict] = None

    if args.params:
        params = _load_yaml(Path(args.params).expanduser().resolve())
        if args.search_nnue:
            params = _force_search_nnue(params)
        if args.use_native:
            params = dict(params)
            params["use_search_nnue"] = False
        cand_params = params

    if args.baseline_params:
        base_params = _load_yaml(Path(args.baseline_params).expanduser().resolve())
        if args.search_nnue:
            base_params = _force_search_nnue(base_params)
        if args.use_native:
            base_params = dict(base_params)
            base_params["use_search_nnue"] = False
    elif cand_params is not None:
        base_params = cand_params
    elif args.use_native:
        base_params = {"use_search_nnue": False}
        cand_params = {"use_search_nnue": False}

    games = args.games if args.games > 0 else len(fens)

    result = arena_perf(
        fens,
        base_params,
        cand_params,
        games,
        args.depth,
        args.movetime_ms,
        args.max_plies,
        args.node_limit,
        0,
        0,
        0,
        40,
    )

    print("[perf-pgn] positions=", len(fens), "games=", games)
    print(
        "[perf-pgn] score={score:.3f} W-L-D={wins}-{losses}-{draws}"
        " nodes={total_nodes} ms={total_ms} nps={nps}"
        " avg_nodes/ply={avg_nodes_per_ply} avg_ms/ply={avg_ms_per_ply}".format(
            **result
        )
    )
    return 0
