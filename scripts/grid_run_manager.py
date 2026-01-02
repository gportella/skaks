#!/usr/bin/env python3
"""
Grid run manager: run a coarse grid scan, pick top-N candidates and run a finer scan around them.

This script wraps the existing `scripts/grid_weights_eval.py` and makes a resumeable
coarse->fine flow. It supports `--dry-run` which prints the commands instead of
executing them (useful for quick verification).

Output: writes CSV files for each phase and a final aggregated CSV.
"""

import argparse
import csv
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path


def parse_scales(scales_str):
    return [float(s) for s in scales_str.split(",") if s.strip()]


def run_cmd(cmd, dry_run=False, capture_output=False):
    print("+ ", cmd)
    if dry_run:
        return 0, "(dry-run)"
    proc = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True)
    out = proc.stdout if capture_output and proc.stdout is not None else None
    return proc.returncode, out


def read_grid_csv(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                wins = int(r.get("wins", 0))
                losses = int(r.get("losses", 0))
                draws = int(r.get("draws", 0))
            except Exception:
                continue
            total = wins + losses + draws
            if total == 0:
                score = 0.0
            else:
                score = (wins + 0.5 * draws) / total
            rows.append(
                {
                    "mg_scale": float(r.get("mg_scale", 0)),
                    "eg_scale": float(r.get("eg_scale", 0)),
                    "wins": wins,
                    "losses": losses,
                    "draws": draws,
                    "score": score,
                }
            )
    return rows


def build_grid_cmd(start_yaml, scales, games, depth, concurrency, out_csv):
    scales_csv = ",".join([str(s) for s in scales])
    cmd = f"python scripts/grid_weights_eval.py --start {shlex.quote(start_yaml)} --scales {shlex.quote(scales_csv)} --games {games} --depth {depth} --concurrency {concurrency} --out {shlex.quote(out_csv)}"
    return cmd


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Coarse->Fine grid manager for weight scaling"
    )
    parser.add_argument("--start", required=True, help="Start YAML file")
    parser.add_argument(
        "--coarse-scales", default="0.8,1.0,1.2", help="Comma list for coarse scales"
    )
    parser.add_argument("--coarse-games", type=int, default=80)
    parser.add_argument("--coarse-depth", type=int, default=2)
    parser.add_argument("--coarse-concurrency", type=int, default=1)
    parser.add_argument(
        "--fine-window",
        type=float,
        default=0.1,
        help="Relative window around top scale (e.g., 0.1 means +/-10%%)",
    )
    parser.add_argument(
        "--fine-steps", type=int, default=5, help="Number of fine steps per axis"
    )
    parser.add_argument(
        "--reval-top",
        type=int,
        default=3,
        help="Number of top candidates to re-evaluate with fine grid",
    )
    parser.add_argument(
        "--reval-games",
        type=int,
        default=200,
        help="Games for re-evaluation of fine grid",
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--out", default="grid_manager_results.csv")
    parser.add_argument("--workdir", default="logs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    os.makedirs(args.workdir, exist_ok=True)
    coarse_scales = parse_scales(args.coarse_scales)

    coarse_out = Path(args.out).with_name(Path(args.out).stem + "_coarse.csv")
    # Run coarse grid
    coarse_cmd = build_grid_cmd(
        args.start,
        coarse_scales,
        args.coarse_games,
        args.coarse_depth,
        args.coarse_concurrency,
        str(coarse_out),
    )
    rc, _ = run_cmd(coarse_cmd, dry_run=args.dry_run)
    if rc != 0 and not args.dry_run:
        print(f"coarse grid failed with rc={rc}")
        sys.exit(rc)

    # Parse results and pick top candidates
    coarse_rows = read_grid_csv(str(coarse_out))
    if not coarse_rows:
        print("no coarse results to analyze (empty CSV).")
        if args.dry_run:
            print("(dry-run) skipping fine stage")
            return 0
        sys.exit(1)

    coarse_rows.sort(key=lambda r: r["score"], reverse=True)
    top = coarse_rows[: args.reval_top]
    print(f"Top {len(top)} candidates:")
    for i, r in enumerate(top, 1):
        print(
            f"{i}. mg={r['mg_scale']} eg={r['eg_scale']} score={r['score']:.3f} W/L/D={r['wins']}/{r['losses']}/{r['draws']}"
        )

    # For each top candidate, run a fine grid
    fine_results = []
    for idx, r in enumerate(top, 1):
        mg_center = r["mg_scale"]
        eg_center = r["eg_scale"]
        window = args.fine_window
        steps = args.fine_steps
        mg_scales = [
            mg_center * (1 + (i / (steps - 1) - 0.5) * 2 * window) for i in range(steps)
        ]
        eg_scales = [
            eg_center * (1 + (i / (steps - 1) - 0.5) * 2 * window) for i in range(steps)
        ]

        # build combined scale pairs into a single scales list (cartesian product)
        combined = []
        for mg in mg_scales:
            for eg in eg_scales:
                combined.append((mg, eg))

        # For grid_weights_eval.py we pass only mg_scale list and eg_scale list together as a single list
        # But its simple KISS interface expects a single comma list which it interprets as mg/eg pairs by enumerating
        # We'll encode pairs by using mg,eg,mg,eg,... sequence and let the script interpret pairs by reading two values per entry.
        # To keep compatibility, we will instead call grid_weights_eval.py multiple times per mg,eg pair separately.
        # Simpler: pass mg_scales as the scales list and
        # rely on grid_weights_eval to interpret them. To avoid complicated assumptions, we'll run a simple nested loop and call grid_weights_eval for each mg,eg pair separately.

        for mg_val, eg_val in combined:
            scales_str = f"{mg_val},{eg_val}"
            fine_out = Path(args.out).with_name(
                f"{Path(args.out).stem}_fine_{idx}_{mg_val:.3f}_{eg_val:.3f}.csv"
            )
            cmd = f"python scripts/grid_weights_eval.py --start {shlex.quote(args.start)} --scales {shlex.quote(scales_str)} --games {args.reval_games} --depth {args.coarse_depth} --concurrency {args.concurrency} --out {shlex.quote(str(fine_out))}"
            rc, _ = run_cmd(cmd, dry_run=args.dry_run)
            if rc != 0 and not args.dry_run:
                print(f"fine run failed rc={rc} for mg={mg_val} eg={eg_val}")
            # read fine_out and append to fine_results
            rows = read_grid_csv(str(fine_out))
            fine_results.extend(rows)

    # Aggregate results to final out CSV
    final_out = Path(args.out)
    fields = ["mg_scale", "eg_scale", "wins", "losses", "draws", "score"]
    with open(final_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in fine_results:
            writer.writerow({k: r[k] for k in fields})

    print(f"Wrote aggregated results to {final_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
