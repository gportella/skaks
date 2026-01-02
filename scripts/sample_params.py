#!/usr/bin/env python3
"""
Sample parameter combinations using Latin Hypercube Sampling (LHS).

This produces N samples uniformly distributed across the ranges for each parameter.
The script writes a CSV with columns for each parameter.

Usage:
  python scripts/sample_params.py --params params.json --n 10000 --out samples.csv

`params.json` format:
{
  "mg_scale": [0.6, 1.4],
  "eg_scale": [0.6, 1.4]
}
"""

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path


def lhs_sample(ranges, n, seed=None):
    # ranges: dict name->(min,max)
    # n samples -> return list of dicts
    rng = random.Random(seed)
    dims = list(ranges.keys())
    # create n intervals per dim
    samples = [dict() for _ in range(n)]
    for dim in dims:
        lo, hi = ranges[dim]
        # create n stratified points
        points = [(i + rng.random()) / n for i in range(n)]
        rng.shuffle(points)
        for i, p in enumerate(points):
            samples[i][dim] = lo + p * (hi - lo)
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Latin Hypercube sampler for parameter ranges"
    )
    parser.add_argument(
        "--params", required=True, help="JSON file with parameter ranges"
    )
    parser.add_argument("--n", type=int, required=True, help="Number of samples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="samples.csv")
    args = parser.parse_args()

    params = json.load(open(args.params))
    ranges = {}
    for k, v in params.items():
        if not (isinstance(v, list) and len(v) == 2):
            raise SystemExit(f"invalid range for {k}: {v}")
        ranges[k] = (float(v[0]), float(v[1]))

    samples = lhs_sample(ranges, args.n, seed=args.seed)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ranges.keys()))
        writer.writeheader()
        for s in samples:
            writer.writerow(s)
    print(f"Wrote {args.n} samples to {outp}")


if __name__ == "__main__":
    main()
