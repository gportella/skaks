#!/usr/bin/env python3
"""
Run evaluations for parameter samples produced by `sample_params.py`.

This script reads a CSV of samples (columns: parameter names) and for each sample
creates a temporary start YAML by scaling `phase_weights_mg` and `phase_weights_eg` in the base
start YAML, then invokes `scripts/grid_weights_eval.py` with the single sample as scales.

Supports batching and `--dry-run`.
"""

import argparse
import csv
import copy
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
import yaml


def read_samples(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: float(v) for k, v in r.items()})
    return rows


def _collect_group_scales(row, prefix):
    keys = [f"{prefix}_g{i}" for i in range(7)]
    values = [row.get(k) for k in keys]
    if any(value is not None for value in values):
        return [float(value) if value is not None else 1.0 for value in values]
    return None


def make_temp_start(base_yaml, mg_scale, eg_scale):
    # read text and strip possible markdown fences (```yaml ... ```)
    text = open(base_yaml, "r").read()
    stripped = text.lstrip()
    if stripped.startswith("```"):
        lines = text.splitlines()
        # drop first fence line
        if lines and lines[0].startswith("```"):
            # drop final fence if present
            if lines[-1].startswith("```"):
                text = "\n".join(lines[1:-1])
            else:
                text = "\n".join(lines[1:])

    base = yaml.safe_load(text)
    # look for phase_weights in top-level or under `evaluation` key
    mg = None
    eg = None
    if isinstance(base, dict):
        mg = base.get("phase_weights_mg")
        eg = base.get("phase_weights_eg")
        if mg is None or eg is None:
            ev = (
                base.get("evaluation")
                if isinstance(base.get("evaluation"), dict)
                else {}
            )
            mg = mg or ev.get("phase_weights_mg")
            eg = eg or ev.get("phase_weights_eg")

    if mg is None or eg is None:
        raise SystemExit("base yaml does not contain phase_weights_mg/eg")

    new = copy.deepcopy(base)

    # place scaled arrays where they originally were
    def expand_group_scales(group_scales, length=15):
        # group_scales: list of 7 scalars -> map to 15 elements
        # map first 6 groups to 2 elements each, last group to remaining (3)
        if group_scales is None:
            return None
        gs = [float(x) for x in group_scales]
        if len(gs) == length:
            return gs
        if len(gs) == 7:
            out = []
            for i in range(6):
                out.extend([gs[i], gs[i]])
            out.extend([gs[6], gs[6], gs[6]])
            return out[:length]
        raise SystemExit("unsupported group length for expansion")

    mg_per = (
        expand_group_scales(mg_scale) if isinstance(mg_scale, (list, tuple)) else None
    )
    eg_per = (
        expand_group_scales(eg_scale) if isinstance(eg_scale, (list, tuple)) else None
    )

    if "phase_weights_mg" in new:
        mg_list = [float(x) for x in mg]
        eg_list = [float(x) for x in eg]
        if mg_per is not None:
            new["phase_weights_mg"] = [a * b for a, b in zip(mg_list, mg_per)]
        else:
            new["phase_weights_mg"] = [float(x) * mg_scale for x in mg]
        if eg_per is not None:
            new["phase_weights_eg"] = [a * b for a, b in zip(eg_list, eg_per)]
        else:
            new["phase_weights_eg"] = [float(x) * eg_scale for x in eg]
    else:
        # nested under evaluation
        if "evaluation" not in new or not isinstance(new["evaluation"], dict):
            new["evaluation"] = {}
        mg_list = [float(x) for x in mg]
        eg_list = [float(x) for x in eg]
        if mg_per is not None:
            new["evaluation"]["phase_weights_mg"] = [
                a * b for a, b in zip(mg_list, mg_per)
            ]
        else:
            new["evaluation"]["phase_weights_mg"] = [float(x) * mg_scale for x in mg]
        if eg_per is not None:
            new["evaluation"]["phase_weights_eg"] = [
                a * b for a, b in zip(eg_list, eg_per)
            ]
        else:
            new["evaluation"]["phase_weights_eg"] = [float(x) * eg_scale for x in eg]

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", prefix="sample_")
    tmp_name = tmp.name
    tmp.close()
    # open in text mode for YAML emitter
    with open(tmp_name, "w", encoding="utf-8") as fh:
        yaml.safe_dump(new, fh)
    return tmp_name


def run_cmd(cmd, dry_run=False):
    print("+", cmd)
    if dry_run:
        return 0
    return subprocess.run(cmd, shell=True).returncode


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run evaluations for parameter samples"
    )
    parser.add_argument("--samples", required=True, help="CSV samples file")
    parser.add_argument("--base", required=True, help="Base start YAML")
    parser.add_argument("--games", type=int, default=80)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--outdir", default="samples_out")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-run", type=int, default=0, help="Max samples to run (0=all)"
    )
    args = parser.parse_args(argv)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    samples = read_samples(args.samples)
    total = len(samples)
    print(f"Loaded {total} samples")
    if args.max_run > 0:
        samples = samples[: args.max_run]

    for i, s in enumerate(samples, 1):
        mg = (
            _collect_group_scales(s, "mg")
            or s.get("mg_scale")
            or s.get("phase_weights_mg")
            or s.get("mg")
        )
        eg = (
            _collect_group_scales(s, "eg")
            or s.get("eg_scale")
            or s.get("phase_weights_eg")
            or s.get("eg")
        )
        if mg is None or eg is None:
            print(f"Skipping sample {i} missing mg/eg: {s}")
            continue
        tmp_yaml = make_temp_start(args.base, mg, eg)
        out_csv = Path(args.outdir) / f"sample_{i:06d}.csv"
        if isinstance(mg, (list, tuple)) or isinstance(eg, (list, tuple)):
            cmd = (
                "python scripts/grid_weights_eval.py "
                f"--start {shlex.quote(tmp_yaml)} "
                f"--games {args.games} --depth {args.depth} "
                f"--concurrency {args.concurrency} --out {shlex.quote(str(out_csv))}"
            )
        else:
            cmd = (
                "python scripts/grid_weights_eval.py "
                f"--start {shlex.quote(tmp_yaml)} --scales {mg},{eg} "
                f"--games {args.games} --depth {args.depth} "
                f"--concurrency {args.concurrency} --out {shlex.quote(str(out_csv))}"
            )
        rc = run_cmd(cmd, dry_run=args.dry_run)
        if rc != 0 and not args.dry_run:
            print(f"Sample {i} failed with rc={rc}")
        # keep tmp_yaml for reproducibility

    print("Done")


if __name__ == "__main__":
    main()
