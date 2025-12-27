#!/usr/bin/env python3
"""Run internal arena using start positions sampled from a PGN.

Uses the skaks_eval.arena binding (baseline = built-in defaults, candidate = provided params).
Positions are sampled at random plies within a given range for each game.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List

import chess.pgn
import yaml

import skaks_eval


def load_start_fens_from_pgn(
    pgn_path: Path,
    games: int,
    min_ply: int,
    max_ply: int,
    seed: int,
) -> List[str]:
    rng = random.Random(seed)
    fens: List[str] = []
    with pgn_path.open("r", encoding="utf-8") as fh:
        while len(fens) < games:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            board = game.board()
            moves = list(game.mainline_moves())
            if not moves:
                continue
            target_ply = rng.randint(min_ply, max_ply) if max_ply > 0 else min_ply
            target_ply = max(1, target_ply)
            applied = 0
            for mv in moves:
                board.push(mv)
                applied += 1
                if applied >= target_ply:
                    break
            fens.append(board.fen())
    if len(fens) < games and fens:
        # Re-sample with replacement if we ran out of games in the PGN
        while len(fens) < games:
            fens.append(rng.choice(fens))
    return fens


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Arena vs default using PGN starts")
    ap.add_argument("pgn", type=Path, help="PGN file to sample starts from")
    ap.add_argument("params", type=Path, help="YAML params for candidate")
    ap.add_argument("--games", type=int, default=40, help="Number of games")
    ap.add_argument("--depth", type=int, default=4, help="Search depth (plies)")
    ap.add_argument(
        "--movetime-ms",
        type=int,
        default=0,
        help="Per-move time in ms (use 0 to stick with depth)",
    )
    ap.add_argument("--max-plies", type=int, default=160, help="Ply cap per game")
    ap.add_argument("--min-ply", type=int, default=6, help="Minimum ply to sample")
    ap.add_argument(
        "--max-ply", type=int, default=60, help="Maximum ply to sample (0 = no cap)"
    )
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for sampling")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.pgn.exists():
        raise SystemExit(f"PGN not found: {args.pgn}")
    if not args.params.exists():
        raise SystemExit(f"Params not found: {args.params}")
    start_fens = load_start_fens_from_pgn(
        args.pgn,
        games=args.games,
        min_ply=args.min_ply,
        max_ply=args.max_ply,
        seed=args.seed,
    )
    if not start_fens:
        raise SystemExit("No positions sampled from PGN")
    cand = yaml.safe_load(args.params.read_text(encoding="utf-8"))
    res = skaks_eval.arena(
        start_fens=start_fens,
        cand_params=cand,
        games=args.games,
        depth=args.depth,
        movetime_ms=args.movetime_ms,
        max_plies=args.max_plies,
    )
    print(json.dumps(res))


if __name__ == "__main__":
    main()
