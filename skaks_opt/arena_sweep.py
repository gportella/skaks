from __future__ import annotations

import argparse
import glob
import json
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from skaks_opt.arena import _configure_parser as _configure_arena_parser
from skaks_opt.arena import _normalize_args

try:  # Optional rich progress support
    from rich.console import Console
    from rich.progress import (BarColumn, Progress, TextColumn,
                               TimeElapsedColumn, TimeRemainingColumn)

    HAS_RICH = True
except Exception:  # pragma: no cover - rich is optional
    Console = None
    Progress = None
    HAS_RICH = False


@dataclass
class CandidateResult:
    path: Path
    score: float
    wins: int
    losses: int
    draws: int
    summary_path: Path


def _expand_candidates(patterns: Sequence[str]) -> List[Path]:
    resolved: List[Path] = []
    for pattern in patterns:
        candidate = Path(pattern).expanduser()
        if candidate.is_file():
            resolved.append(candidate)
            continue
        if candidate.is_dir():
            resolved.extend(sorted(candidate.glob("**/*.yaml")))
            continue
        globbed = sorted(
            Path(match) for match in glob.glob(str(candidate), recursive=True)
        )
        if globbed:
            resolved.extend(globbed)
            continue
        raise FileNotFoundError(f"No candidates match: {pattern}")
    unique: List[Path] = []
    seen = set()
    for item in resolved:
        resolved_item = item.resolve()
        if resolved_item in seen:
            continue
        seen.add(resolved_item)
        unique.append(resolved_item)
    return unique


def _format_score(wins: int, losses: int, draws: int) -> float:
    total = wins + losses + draws
    if total == 0:
        return 0.0
    return (wins + 0.5 * draws) / total


def _build_arena_command(
    *,
    args: argparse.Namespace,
    candidate: Path,
    summary_path: Path,
    games_override: Optional[int] = None,
) -> List[str]:
    cmd: List[str] = [sys.executable, "-m", "skaks_opt.cli", "arena"]
    games = games_override if games_override is not None else args.games
    cmd.extend(["--games", str(games)])
    cmd.extend(["--limit", str(args.limit)])
    if args.depth is not None:
        cmd.extend(["--depth", str(args.depth)])
    if args.time_per_move is not None:
        cmd.extend(["--time-per-move", str(args.time_per_move)])
    if args.clock is not None:
        cmd.extend(["--clock", str(args.clock)])
    if args.increment is not None:
        cmd.extend(["--increment", str(args.increment)])
    if args.opponent_increment is not None:
        cmd.extend(["--opponent-increment", str(args.opponent_increment)])
    if args.moves_to_go is not None:
        cmd.extend(["--moves-to-go", str(args.moves_to_go)])
    if args.opponent_time_per_move is not None:
        cmd.extend(["--opponent-time-per-move", str(args.opponent_time_per_move)])
    if args.opponent_clock is not None:
        cmd.extend(["--opponent-clock", str(args.opponent_clock)])
    if args.opponent_depth_factor is not None:
        cmd.extend(["--opponent-depth-factor", str(args.opponent_depth_factor)])
    if args.engine:
        cmd.extend(["--engine", args.engine])
    cmd.extend(["--engine-params", str(candidate)])
    if args.engine_nnue:
        cmd.extend(["--engine-nnue", args.engine_nnue])
    if args.engine_eval:
        cmd.extend(["--engine-eval", args.engine_eval])
    if args.engine_label:
        cmd.extend(["--engine-label", args.engine_label])
    if args.opponent:
        cmd.extend(["--opponent", args.opponent])
    if args.opponent_params:
        cmd.extend(["--opponent-params", args.opponent_params])
    if args.opponent_nnue:
        cmd.extend(["--opponent-nnue", args.opponent_nnue])
    if args.opponent_eval:
        cmd.extend(["--opponent-eval", args.opponent_eval])
    if args.opponent_label:
        cmd.extend(["--opponent-label", args.opponent_label])
    if args.stockfish:
        cmd.append("--stockfish")
    if args.database:
        cmd.extend(["--database", args.database])
    if args.concurrency is not None:
        cmd.extend(["--concurrency", str(args.concurrency)])
    if args.resume:
        cmd.append("--resume")
    if args.timeout is not None:
        cmd.extend(["--timeout", str(args.timeout)])
    if args.no_handicap:
        cmd.append("--no-handicap")
    if args.handicap_factor is not None:
        cmd.extend(["--handicap-factor", str(args.handicap_factor)])
    if args.handicap_depth is not None:
        cmd.extend(["--handicap-depth", str(args.handicap_depth)])
    if args.verbose:
        cmd.append("--verbose")
    if args.elo_start is not None:
        cmd.extend(["--elo-start", str(args.elo_start)])
    if args.elo_opponent is not None:
        cmd.extend(["--elo-opponent", str(args.elo_opponent)])
    if args.elo_k_factor is not None:
        cmd.extend(["--elo-k-factor", str(args.elo_k_factor)])
    if args.elo_store:
        cmd.extend(["--elo-store", args.elo_store])
    if args.no_elo_store:
        cmd.append("--no-elo-store")
    cmd.extend(["--summary-json", str(summary_path)])
    return cmd


def _format_best(best: Optional[CandidateResult]) -> str:
    if not best:
        return "best: -"
    total = best.wins + best.losses + best.draws
    return (
        f"best: {best.score:.3f} (W{best.wins} D{best.draws} L{best.losses}) "
        f"{best.path.name} [{total}g]"
    )


def _copy_best(candidate_path: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate_path, destination)


def _resolve_summary_destination(
    args: argparse.Namespace,
    candidate: Path,
    index: int,
    *,
    iteration: Optional[int] = None,
    aggregate: bool = False,
) -> Path:
    if args.summaries_dir:
        target_dir = Path(args.summaries_dir).expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
    else:
        target_dir = Path(args._sweep_tmpdir)

    if aggregate:
        name = f"arena_{index:04d}_aggregate.json"
    elif iteration is not None:
        name = f"arena_{index:04d}_run{iteration:02d}.json"
    else:
        name = f"arena_{index:04d}.json"
    return target_dir / name


def _compute_candidate_score(summary_path: Path) -> tuple[CandidateResult, dict]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    params = payload.get("parameters", {})
    engine_params = params.get("engine_params")
    labels = payload.get("labels", {})
    summary = payload.get("summary", {})
    wins = int(summary.get(labels.get("white", "skaks"), 0))
    losses = int(summary.get(labels.get("black", "sunfish"), 0))
    draws = int(summary.get("draw", 0))
    score = _format_score(wins, losses, draws)
    path = Path(engine_params) if engine_params else summary_path.with_suffix(".yaml")
    return CandidateResult(
        path=path,
        score=score,
        wins=wins,
        losses=losses,
        draws=draws,
        summary_path=summary_path,
    ), payload


def _write_aggregate_summary(
    template: Optional[dict],
    summary_path: Path,
    *,
    wins: int,
    losses: int,
    draws: int,
    runs: int,
    sources: Sequence[Path],
    min_games: int,
) -> None:
    labels = (template or {}).get("labels", {})
    white_label = labels.get("white", "skaks")
    black_label = labels.get("black", "sunfish")

    summary = {
        white_label: wins,
        black_label: losses,
        "draw": draws,
        "unknown": 0,
    }

    payload = {
        "games": wins + losses + draws,
        "completed": wins + losses + draws,
        "failures": 0,
        "summary": summary,
        "labels": labels or {"white": white_label, "black": black_label},
        "parameters": (template or {}).get("parameters", {}),
        "aggregate": {
            "runs": runs,
            "sources": [str(path) for path in sources],
            "min_games": min_games,
        },
    }

    maybe_elo = (template or {}).get("elo")
    if isinstance(maybe_elo, dict):
        start = maybe_elo.get("start")
        opponent = maybe_elo.get("opponent")
        if start is not None and opponent is not None:
            actual_score = wins + 0.5 * draws
            payload["elo"] = {
                "start": start,
                "opponent": opponent,
                "actual_score": actual_score,
                "games": wins + losses + draws,
            }

    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def _score_better(lhs: CandidateResult, rhs: Optional[CandidateResult]) -> bool:
    if rhs is None:
        return True
    if math.isclose(lhs.score, rhs.score, rel_tol=1e-9, abs_tol=1e-9):
        if lhs.wins != rhs.wins:
            return lhs.wins > rhs.wins
        if lhs.losses != rhs.losses:
            return lhs.losses < rhs.losses
        return lhs.draws > rhs.draws
    return lhs.score > rhs.score


def run_sweep(args: argparse.Namespace) -> None:
    _normalize_args(args)
    candidates = _expand_candidates(args.candidates)
    if not candidates:
        raise SystemExit("No parameter files to evaluate")

    with tempfile.TemporaryDirectory(prefix="arena_sweep_") as tmpdir:
        args._sweep_tmpdir = tmpdir
        best: Optional[CandidateResult] = None
        best_summary_export: Optional[Path] = None
        console: Optional[Console] = Console() if HAS_RICH else None
        progress: Optional[Progress] = None
        task_id: Optional[int] = None
        if HAS_RICH:
            progress = Progress(
                TextColumn("[bold]Arena sweep"),
                BarColumn(bar_width=None),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                TextColumn("{task.fields[status]}", justify="left"),
                TextColumn("{task.fields[best]}", justify="left"),
                console=console,
                transient=False,
            )
            progress.start()
            task_id = progress.add_task(
                "arena", total=len(candidates), status="", best=_format_best(best)
            )

        def _update(status: str) -> None:
            if progress is not None and task_id is not None:
                progress.update(task_id, status=status, best=_format_best(best))
            else:
                print(status)

        for idx, candidate in enumerate(candidates, start=1):
            target_games = max(
                args.min_games_per_candidate,
                args.games if args.games and args.games > 0 else 0,
            )
            if target_games == 0:
                target_games = args.min_games_per_candidate

            games_per_pass = (
                args.games if args.games and args.games > 0 else target_games
            )
            games_per_pass = max(1, games_per_pass)

            wins_total = losses_total = draws_total = 0
            run_count = 0
            candidate_failed = False
            payload_template: Optional[dict] = None
            run_summaries: List[Path] = []

            while wins_total + losses_total + draws_total < target_games:
                remaining = target_games - (wins_total + losses_total + draws_total)
                games_this_pass = min(games_per_pass, remaining)
                summary_path = _resolve_summary_destination(
                    args, candidate, idx, iteration=run_count
                )
                cmd = _build_arena_command(
                    args=args,
                    candidate=candidate,
                    summary_path=summary_path,
                    games_override=games_this_pass,
                )
                _update(
                    f"[{idx}/{len(candidates)}] {candidate.name}: run {run_count + 1} of "
                    f"{math.ceil(target_games / games_per_pass)} (games={games_this_pass})"
                )
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if result.returncode != 0:
                    candidate_failed = True
                    if not args.keep_failed:
                        raise RuntimeError(
                            f"Arena run failed for {candidate}: {result.stderr.strip()}"
                        )
                    _update(
                        f"[{idx}/{len(candidates)}] {candidate.name}: failed (rc={result.returncode})"
                    )
                    break
                try:
                    current, payload = _compute_candidate_score(summary_path)
                except (OSError, json.JSONDecodeError) as exc:
                    candidate_failed = True
                    if not args.keep_failed:
                        raise RuntimeError(
                            f"Failed to parse summary for {candidate}: {exc}"
                        )
                    _update(
                        f"[{idx}/{len(candidates)}] {candidate.name}: summary parse failed"
                    )
                    break

                if payload_template is None:
                    payload_template = payload

                wins_total += current.wins
                losses_total += current.losses
                draws_total += current.draws
                run_count += 1
                run_summaries.append(summary_path)

                partial_score = _format_score(wins_total, losses_total, draws_total)
                if progress is not None and task_id is not None:
                    progress.update(
                        task_id,
                        status=(
                            f"[{idx}/{len(candidates)}] {candidate.name}: "
                            f"runs={run_count} score={partial_score:.3f} "
                            f"W{wins_total} D{draws_total} L{losses_total}"
                        ),
                        best=_format_best(best),
                    )
                else:
                    print(
                        f"[{idx}/{len(candidates)}] {candidate.name}: runs={run_count} "
                        f"score={partial_score:.3f} W{wins_total} D{draws_total} L{losses_total}"
                    )

            if candidate_failed:
                continue

            aggregate_summary_path = _resolve_summary_destination(
                args, candidate, idx, aggregate=True
            )
            _write_aggregate_summary(
                payload_template,
                aggregate_summary_path,
                wins=wins_total,
                losses=losses_total,
                draws=draws_total,
                runs=run_count,
                sources=run_summaries,
                min_games=args.min_games_per_candidate,
            )
            current = CandidateResult(
                path=candidate,
                score=_format_score(wins_total, losses_total, draws_total),
                wins=wins_total,
                losses=losses_total,
                draws=draws_total,
                summary_path=aggregate_summary_path,
            )

            if progress is not None and task_id is not None:
                progress.update(
                    task_id,
                    advance=1,
                    status=(
                        f"[{idx}/{len(candidates)}] {candidate.name}: "
                        f"score={current.score:.3f} W{current.wins} D{current.draws} L{current.losses}"
                    ),
                    best=_format_best(best),
                )
            else:
                print(
                    f"[{idx}/{len(candidates)}] {candidate.name}: score={current.score:.3f} "
                    f"W{current.wins} D{current.draws} L{current.losses}"
                )

            if _score_better(current, best):
                best = current
                if args.best_out:
                    best_out_path = Path(args.best_out).expanduser().resolve()
                    _copy_best(candidate, best_out_path)
                destination_summary: Optional[Path] = None
                if args.best_summary:
                    destination_summary = Path(args.best_summary).expanduser().resolve()
                elif args.best_out:
                    best_out_path = Path(args.best_out).expanduser().resolve()
                    destination_summary = best_out_path.with_suffix(
                        best_out_path.suffix + ".summary.json"
                    )
                if destination_summary:
                    destination_summary.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(aggregate_summary_path, destination_summary)
                    best_summary_export = destination_summary
                if progress is not None and task_id is not None:
                    progress.update(task_id, best=_format_best(best))
                else:
                    print(f"Updated best: {candidate} -> score {current.score:.3f}")
        if progress is not None:
            progress.stop()
    if not best:
        raise SystemExit("No successful arena runs recorded")
    print("\nFinal best candidate:")
    if best_summary_export:
        summary_note = str(best_summary_export)
    else:
        summary_note = "not persisted (use --best-summary to store)"
    print(
        f"  file: {best.path}"
        f"\n  score: {best.score:.4f}"
        f"\n  record: {best.wins} wins / {best.draws} draws / {best.losses} losses"
        f"\n  summary: {summary_note}"
    )


def add_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "arena-sweep",
        help="Evaluate parameter files via arena matches and keep the best",
        description=(
            "Batch runner that points the `arena` subcommand at every YAML candidate "
            "(files, directories, or globs), aggregates their match summaries, and "
            "copies out the strongest performer. Use this when you already have "
            "params to compare rather than when you need to generate new ones."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _configure_arena_parser(parser)
    parser.add_argument(
        "--candidates",
        nargs="+",
        required=True,
        help="Parameter files or glob patterns to evaluate (YAML)",
    )
    parser.add_argument(
        "--best-out",
        type=str,
        help="Write the best parameter YAML to this path (updated on improvement)",
    )
    parser.add_argument(
        "--best-summary",
        type=str,
        help="Write the summary JSON for the best candidate to this path",
    )
    parser.add_argument(
        "--summaries-dir",
        type=str,
        help="Directory to store per-candidate summary JSON files",
    )
    parser.add_argument(
        "--keep-failed",
        action="store_true",
        help="Skip candidates whose arena run fails instead of aborting",
    )
    parser.add_argument(
        "--min-games-per-candidate",
        type=int,
        default=5,
        help="Repeat arenas until each candidate accumulates at least this many games",
    )
    return parser
