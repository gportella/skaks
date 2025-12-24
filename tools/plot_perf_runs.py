#!/usr/bin/env python

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

SNAPSHOT_RE = re.compile(r"=== perf snapshot (.+) ===")
VERSION_RE = re.compile(r"skaks version\s+(.+)")
CASE_RE = re.compile(r"\[case\]\s+(.+)")
DEPTH_RE = re.compile(r"\[perf\]\s+depth=(\d+)\s+iterations=(\d+)")
TOTAL_RE = re.compile(
    r"\[perf\]\s+total_nodes=(\d+)\s+total_ms=(\d+)\s+total_nps=(\d+)\s+avg_nodes=(\d+)\s+avg_ms=(\d+)"
)


@dataclass
class PerfRow:
    timestamp: datetime
    version: str
    case: str
    depth: int
    total_nodes: int
    total_ms: int
    avg_nodes: int
    avg_ms: int


def parse_log(path: Path) -> list[PerfRow]:
    rows: list[PerfRow] = []
    snapshot_time: datetime | None = None
    version: str | None = None
    current_case: str | None = None
    current_depth: int | None = None
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            snap = SNAPSHOT_RE.match(line)
            if snap:
                snapshot_time = datetime.fromisoformat(snap.group(1))
                version = None
                continue
            version_match = VERSION_RE.match(line)
            if version_match:
                version = version_match.group(1).strip()
                continue
            case_match = CASE_RE.match(line)
            if case_match:
                current_case = case_match.group(1).strip()
                current_depth = None
                continue
            depth_match = DEPTH_RE.match(line)
            if depth_match:
                current_depth = int(depth_match.group(1))
                continue
            total_match = TOTAL_RE.match(line)
            if (
                total_match
                and snapshot_time
                and current_case
                and current_depth is not None
            ):
                rows.append(
                    PerfRow(
                        timestamp=snapshot_time,
                        version=version or "unknown",
                        case=current_case,
                        depth=current_depth,
                        total_nodes=int(total_match.group(1)),
                        total_ms=int(total_match.group(2)),
                        avg_nodes=int(total_match.group(4)),
                        avg_ms=int(total_match.group(5)),
                    )
                )
    return rows


def group_by_case(rows: Iterable[PerfRow]) -> dict[str, list[PerfRow]]:
    by_case: dict[str, list[PerfRow]] = defaultdict(list)
    for row in rows:
        by_case[row.case].append(row)
    return by_case


def _version_sort_key(version: str) -> tuple[object, ...]:
    parts = re.split(r"([0-9]+)", version)
    key: list[object] = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return tuple(key)


def plot_metric(
    rows: Iterable[PerfRow],
    metric: str,
    ylabel: str,
    out_path: Path | None,
    show: bool = False,
) -> None:
    case_map = group_by_case(rows)
    if not case_map:
        print("no perf data found", file=sys.stderr)
        return
    ordered_versions = sorted({row.version for row in rows}, key=_version_sort_key)
    version_index = {version: idx for idx, version in enumerate(ordered_versions)}
    snapshot_entries = sorted(
        {(row.version, row.timestamp) for row in rows},
        key=lambda pair: (version_index.get(pair[0], sys.maxsize), pair[1]),
    )
    snapshot_index = {
        entry: idx for idx, entry in enumerate(snapshot_entries)
    }
    version_totals = Counter(version for version, _ in snapshot_entries)
    cases = sorted(case_map)
    fig, axes = plt.subplots(len(cases), 1, sharex=True, figsize=(11, 3 * len(cases)))
    if len(cases) == 1:
        axes = [axes]
    for ax, case in zip(axes, cases):
        case_rows = case_map[case]
        depth_groups: dict[int, list[PerfRow]] = defaultdict(list)
        for row in case_rows:
            depth_groups[row.depth].append(row)
        for depth, items in sorted(depth_groups.items()):
            items.sort(key=lambda r: snapshot_index[(r.version, r.timestamp)])
            xs = [snapshot_index[(r.version, r.timestamp)] for r in items]
            ys_raw = [getattr(r, metric) for r in items]
            ys = [max(value, 1) for value in ys_raw]
            ax.plot(xs, ys, marker="o", label=f"depth={depth}")
            for x, y, r in zip(xs, ys, items):
                ax.annotate(
                    f"v{r.version}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize="x-small",
                )
        ax.set_yscale("log", nonpositive="clip")
        ax.set_ylim(bottom=1)
        tick_positions = [snapshot_index[entry] for entry in snapshot_entries]
        occurrence_counts: Counter[str] = Counter()
        tick_labels: list[str] = []
        for version, _ in snapshot_entries:
            occurrence_counts[version] += 1
            if version_totals[version] > 1:
                tick_labels.append(f"{version} ({occurrence_counts[version]})")
            else:
                tick_labels.append(version)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=40, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(case)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
        ax.legend(loc="best", fontsize="small")
    axes[-1].set_xlabel("engine version order")
    fig.tight_layout()
    if show:
        return
    if out_path is None:
        plt.close(fig)
        return
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot skaks perf log snapshots")
    parser.add_argument(
        "--log", dest="log_path", default="perf_runs.log", help="log file path"
    )
    parser.add_argument("--outdir", default="perf_plots", help="output directory")
    parser.add_argument("--ext", default="png", help="image file extension")
    parser.add_argument(
        "--show", action="store_true", help="display plots instead of saving"
    )
    args = parser.parse_args()

    log_path = Path(args.log_path)
    if not log_path.exists():
        print(f"log not found: {log_path}", file=sys.stderr)
        return 1
    rows = parse_log(log_path)
    if not rows:
        print("log did not contain perf summaries", file=sys.stderr)
        return 1

    if args.show:
        plot_metric(rows, "avg_nodes", "avg nodes", None, show=True)
        plot_metric(rows, "avg_ms", "avg ms", None, show=True)
        plt.show()
        return 0

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    nodes_path = outdir / f"avg_nodes.{args.ext}"
    ms_path = outdir / f"avg_ms.{args.ext}"
    plot_metric(rows, "avg_nodes", "avg nodes", nodes_path)
    plot_metric(rows, "avg_ms", "avg ms", ms_path)
    print(f"saved {nodes_path}")
    print(f"saved {ms_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
