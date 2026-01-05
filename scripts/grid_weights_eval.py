#!/usr/bin/env python3
"""Evaluate a grid of scale factors for phase weight arrays.

Generates parameter files by scaling `evaluation.phase_weights_mg` and
`evaluation.phase_weights_eg` from a start YAML and evaluates each
candidate with the existing `validation_moves/batch_stockfish_matches.py`.
Writes results to a CSV for easy inspection.
"""

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

import yaml


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data, path: Path):
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=True)


def build_grid(scales: List[float]):
    for mg_scale in scales:
        for eg_scale in scales:
            yield mg_scale, eg_scale


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="Start YAML with phase weights")
    p.add_argument(
        "--scales", default="0.5,0.75,1.0,1.25,1.5", help="Comma list of scales"
    )
    p.add_argument("--games", type=int, default=50)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--out", default="grid_results.csv")
    p.add_argument("--opponent", default="stockfish")
    p.add_argument("--engine", default="skaks")
    p.add_argument(
        "--opponent-depth-factor",
        type=float,
        help=(
            "Scale factor applied to opponent depth (defaults to 0.3 when "
            "opponent is sunfish and depth mode is used)."
        ),
    )
    args = p.parse_args()

    start = Path(args.start)
    data = load_yaml(start)
    mg = data["evaluation"].get("phase_weights_mg")
    eg = data["evaluation"].get("phase_weights_eg")
    if not mg or not eg:
        print("start yaml missing phase_weights_mg/eg", file=sys.stderr)
        sys.exit(2)

    scales = [float(s) for s in args.scales.split(",") if s.strip()]

    opponent_depth_factor = args.opponent_depth_factor
    if (
        opponent_depth_factor is None
        and args.opponent
        and args.opponent.lower() == "sunfish"
        and args.depth is not None
    ):
        opponent_depth_factor = 0.3

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    def write_row(payload):
        header = [
            "mg_scale",
            "eg_scale",
            "returncode",
            "wins",
            "losses",
            "draws",
            "white_label",
            "black_label",
        ]
        new_file = not out_path.exists()
        with out_path.open("a", newline="", encoding="utf-8") as csvf:
            writer = csv.writer(csvf)
            if new_file:
                writer.writerow(header)
            writer.writerow(payload)

    def is_better(lhs, rhs):
        lhs_rank = (lhs["wins"], -lhs["losses"], lhs["draws"])
        rhs_rank = (rhs["wins"], -rhs["losses"], rhs["draws"])
        return lhs_rank > rhs_rank

    best_entry = None

    for mg_scale, eg_scale in build_grid(scales):
        nd = dict(data)
        nd_ev = dict(nd.get("evaluation", {}))
        nd_ev["phase_weights_mg"] = [float(v) * mg_scale for v in mg]
        nd_ev["phase_weights_eg"] = [float(v) * eg_scale for v in eg]
        nd["evaluation"] = nd_ev

        tmp = Path(tempfile.mkstemp(suffix="_grid.yaml")[1])
        save_yaml(nd, tmp)

        cmd = [
            sys.executable,
            str(
                Path(__file__).resolve().parent.parent
                / "validation_moves"
                / "batch_stockfish_matches.py"
            ),
            "--engine",
            args.engine,
            "--opponent",
            args.opponent,
            "--games",
            str(args.games),
            "--depth",
            str(args.depth),
            "--concurrency",
            str(args.concurrency),
            "--summary-json",
            str(tmp.with_suffix(".json")),
            "--engine-params",
            str(tmp),
        ]
        if opponent_depth_factor is not None:
            cmd.extend(["--opponent-depth-factor", str(opponent_depth_factor)])

        print("Running", mg_scale, eg_scale)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        summary = None
        try:
            summary = json.loads(tmp.with_suffix(".json").read_text(encoding="utf-8"))
        except Exception:
            pass

        if proc.returncode != 0:
            sys.stderr.write(
                f"[grid] evaluation failed mg={mg_scale} eg={eg_scale} exit={proc.returncode}\n"
            )
            if proc.stdout:
                sys.stderr.write(proc.stdout + "\n")
            if proc.stderr:
                sys.stderr.write(proc.stderr + "\n")
            raise RuntimeError(
                "grid evaluation aborted due to failed batch_stockfish_matches run"
            )

        labels = (summary or {}).get("labels", {})
        white_label = labels.get("white", "skaks")
        black_label = labels.get("black", "opponent")
        summary_counts = (summary or {}).get("summary", {})
        wins = summary_counts.get(white_label, 0)
        losses = summary_counts.get(black_label, 0)
        draws = summary_counts.get("draw", 0)
        entry = {
            "mg_scale": mg_scale,
            "eg_scale": eg_scale,
            "returncode": proc.returncode,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "white_label": white_label,
            "black_label": black_label,
        }
        write_row(
            [
                entry["mg_scale"],
                entry["eg_scale"],
                entry["returncode"],
                entry["wins"],
                entry["losses"],
                entry["draws"],
                entry["white_label"],
                entry["black_label"],
            ]
        )
        if best_entry is None or is_better(entry, best_entry):
            best_entry = entry
        try:
            tmp.unlink()
        except Exception:
            pass

    if best_entry is not None:
        total_games = best_entry["wins"] + best_entry["losses"] + best_entry["draws"]
        win_rate = best_entry["wins"] / total_games if total_games else 0.0
        print(
            "Best candidate: "
            f"mg_scale={best_entry['mg_scale']} "
            f"eg_scale={best_entry['eg_scale']} "
            f"wins={best_entry['wins']} "
            f"losses={best_entry['losses']} "
            f"draws={best_entry['draws']} "
            f"win_rate={win_rate:.2%}"
        )

    print("Wrote", args.out)


if __name__ == "__main__":
    main()
