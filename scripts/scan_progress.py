#!/usr/bin/env python3
"""
Simple progress and top-K viewer for scan_out.

Usage: python scripts/scan_progress.py --top 5 --interval 10
"""

import time
import argparse
import glob
import csv
import os


def snapshot(top=5):
    files = sorted(glob.glob("scan_out/sample_*.csv"))
    rows = []
    for p in files:
        try:
            with open(p, newline="") as f:
                r = csv.DictReader(f)
                rec = next(r)
                w = int(rec.get("wins", 0))
                l = int(rec.get("losses", 0))
                d = int(rec.get("draws", 0))
                t = w + l + d
                if t == 0:
                    continue
                score = (w + 0.5 * d) / t
                rows.append((score, w, l, d, t, p))
        except Exception:
            continue
    rows.sort(reverse=True, key=lambda x: x[0])
    return rows


def run(top: int = 5, interval: int = 10) -> None:
    """Continuously print scan progress until interrupted."""
    try:
        while True:
            rows = snapshot(top)
            os.system("clear")
            print("Scan progress: found", len(rows), "completed samples")
            print(f"Top {top} so far:")
            print(" score  wins loss draw total file")
            for s, w, l, d, t, p in rows[:top]:
                print(f"{s:6.3f} {w:5d} {l:5d} {d:5d} {t:6d} {p}")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--interval", type=int, default=10)
    args = p.parse_args(argv)

    run(top=args.top, interval=args.interval)


if __name__ == "__main__":
    main()
