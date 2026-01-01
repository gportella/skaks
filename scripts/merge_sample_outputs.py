#!/usr/bin/env python3
import argparse
import csv
import glob
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--indir", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

files = sorted(Path(args.indir).glob("sample_*.csv"))
if not files:
    print("no sample files found")
    raise SystemExit(1)

fieldnames = ["mg_scale", "eg_scale", "wins", "losses", "draws", "score"]
with open(args.out, "w", newline="") as out:
    w = csv.DictWriter(out, fieldnames=fieldnames)
    w.writeheader()
    for f in files:
        with open(f, newline="") as inf:
            r = csv.DictReader(inf)
            for row in r:
                w.writerow({k: row.get(k, "") for k in fieldnames})

print("merged", len(files), "files to", args.out)
