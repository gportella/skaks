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
    args = p.parse_args()

    start = Path(args.start)
    data = load_yaml(start)
    mg = data["evaluation"].get("phase_weights_mg")
    eg = data["evaluation"].get("phase_weights_eg")
    if not mg or not eg:
        print("start yaml missing phase_weights_mg/eg", file=sys.stderr)
        sys.exit(2)

    scales = [float(s) for s in args.scales.split(",") if s.strip()]

    results = []
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

        print("Running", mg_scale, eg_scale)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        summary = None
        try:
            summary = json.loads(tmp.with_suffix(".json").read_text(encoding="utf-8"))
        except Exception:
            pass

        results.append(
            {
                "mg_scale": mg_scale,
                "eg_scale": eg_scale,
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "summary": summary,
            }
        )
        try:
            tmp.unlink()
        except Exception:
            pass

    with open(args.out, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(
            ["mg_scale", "eg_scale", "returncode", "wins", "losses", "draws"]
        )
        for r in results:
            s = r["summary"] or {}
            wins = s.get("summary", {}).get("skaks", 0)
            losses = s.get("summary", {}).get("stockfish", 0)
            draws = s.get("summary", {}).get("draw", 0)
            writer.writerow(
                [r["mg_scale"], r["eg_scale"], r["returncode"], wins, losses, draws]
            )

    print("Wrote", args.out)


if __name__ == "__main__":
    main()
