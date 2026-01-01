#!/usr/bin/env python
"""
Memory-efficient plotting of skaks_cp vs stockfish_cp from large CSV files.

This script reads one or more CSVs with a header containing at least
`stockfish_cp` and `skaks_cp` columns, accumulates a 2D histogram in
streaming fashion (fixed memory), and writes a PNG heatmap. It also
supports reservoir sampling to produce an optional scatter plot of a
representative subset of points.

Usage examples:
  python3 scripts/plot_cp_vs_cp.py eval_pairs_pvs_with_results.csv -o out.png
  python3 scripts/plot_cp_vs_cp.py file1.csv file2.csv -o heat.png --bins 800 --clip 1500 --log

Designed to handle millions of rows.
"""

import argparse
import csv
import math
import random
import sys
from typing import List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot skaks_cp vs stockfish_cp heatmap from large CSVs"
    )
    p.add_argument("files", nargs="+", help="One or more CSV files")
    p.add_argument("-o", "--output", required=True, help="Output PNG path")
    p.add_argument(
        "--bins", type=int, default=800, help="Number of bins per axis (default 800)"
    )
    p.add_argument(
        "--clip",
        type=float,
        default=2000.0,
        help="Clip CP values to range [-clip, clip] (default 2000)",
    )
    p.add_argument("--log", action="store_true", help="Use log color scale")
    p.add_argument(
        "--cmap", default="magma", help="Matplotlib colormap (default magma)"
    )
    p.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="Rows to aggregate per chunk (default 200k)",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=0,
        help="If >0, reservoir-sample this many points and also save scatter to <output>.scatter.png",
    )
    p.add_argument(
        "--xcol", default="skaks_cp", help="Column name for x axis (default skaks_cp)"
    )
    p.add_argument(
        "--ycol",
        default="stockfish_cp",
        help="Column name for y axis (default stockfish_cp)",
    )
    p.add_argument("--title", default=None, help="Plot title")
    return p.parse_args()


def reservoir_sample(
    reservoir: List[Tuple[float, float]], item: Tuple[float, float], seen: int, k: int
) -> None:
    """Classic reservoir sampling: keep k items uniformly from stream."""
    if len(reservoir) < k:
        reservoir.append(item)
    else:
        s = random.randrange(seen)
        if s < k:
            reservoir[s] = item


def process_files(
    file_paths: List[str],
    xcol: str,
    ycol: str,
    bins: int,
    clip: float,
    chunksize: int,
    sample_k: int,
):
    xmin, xmax = -clip, clip
    ymin, ymax = -clip, clip
    hist = np.zeros((bins, bins), dtype=np.int64)
    reservoir: List[Tuple[float, float]] = []
    seen = 0

    # Precompute bin edges
    xedges = np.linspace(xmin, xmax, bins + 1)
    yedges = np.linspace(ymin, ymax, bins + 1)

    for fp in file_paths:
        with open(fp, "r", newline="") as fh:
            reader = csv.reader(fh)
            try:
                header = next(reader)
            except StopIteration:
                continue
            # map headers to indices
            hdr = [h.strip() for h in header]
            try:
                xi = hdr.index(xcol)
                yi = hdr.index(ycol)
            except ValueError:
                print(
                    f"ERROR: cannot find columns {xcol} or {ycol} in {fp}",
                    file=sys.stderr,
                )
                print("Available columns:", ",".join(hdr), file=sys.stderr)
                raise

            buf_x: List[float] = []
            buf_y: List[float] = []
            for row in reader:
                # defensive: skip short/malformed rows
                if len(row) <= max(xi, yi):
                    continue
                try:
                    xv = float(row[xi])
                    yv = float(row[yi])
                except Exception:
                    continue
                # clip
                if math.isfinite(xv) and math.isfinite(yv):
                    if xv < xmin:
                        xv = xmin
                    elif xv > xmax:
                        xv = xmax
                    if yv < ymin:
                        yv = ymin
                    elif yv > ymax:
                        yv = ymax
                    buf_x.append(xv)
                    buf_y.append(yv)
                    seen += 1
                    if sample_k > 0:
                        reservoir_sample(reservoir, (xv, yv), seen, sample_k)

                if len(buf_x) >= chunksize:
                    h, _, _ = np.histogram2d(buf_x, buf_y, bins=[xedges, yedges])
                    hist += h.astype(np.int64)
                    buf_x.clear()
                    buf_y.clear()

            # remaining
            if buf_x:
                h, _, _ = np.histogram2d(buf_x, buf_y, bins=[xedges, yedges])
                hist += h.astype(np.int64)
                buf_x.clear()
                buf_y.clear()

    return hist, xedges, yedges, reservoir


def plot_hist(
    hist: np.ndarray, xedges: np.ndarray, yedges: np.ndarray, outpath: str, args
):
    fig, ax = plt.subplots(figsize=(10, 8))
    # extent is left, right, bottom, top in value coordinates
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    norm = None
    if args.log:
        norm = LogNorm(vmin=1, vmax=hist.max() if hist.max() > 1 else 1)

    im = ax.imshow(
        hist.T, origin="lower", extent=extent, aspect="auto", cmap=args.cmap, norm=norm
    )
    ax.set_xlabel(args.xcol)
    ax.set_ylabel(args.ycol)
    if args.title:
        ax.set_title(args.title)
    else:
        ax.set_title(f"{args.xcol} vs {args.ycol} (bins={args.bins})")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("counts")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_scatter_sample(reservoir: List[Tuple[float, float]], outpath: str, args):
    if not reservoir:
        return
    xs = [p[0] for p in reservoir]
    ys = [p[1] for p in reservoir]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(xs, ys, s=1, alpha=0.6)
    ax.set_xlabel(args.xcol)
    ax.set_ylabel(args.ycol)
    ax.set_title(f"Sample scatter ({len(xs)} points)")
    fig.tight_layout()
    out_scatter = outpath + ".scatter.png"
    fig.savefig(out_scatter, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    hist, xedges, yedges, reservoir = process_files(
        args.files,
        args.xcol,
        args.ycol,
        args.bins,
        args.clip,
        args.chunksize,
        args.sample,
    )
    plot_hist(hist, xedges, yedges, args.output, args)
    if args.sample > 0:
        plot_scatter_sample(reservoir, args.output, args)
    print(f"Wrote {args.output} (bins={args.bins})")


if __name__ == "__main__":
    main()
