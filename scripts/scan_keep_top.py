#!/usr/bin/env python3
"""
Sample N parameter combinations (LHS), evaluate each against Sunfish (via existing grid evaluator),
keep top-K best scoring samples. Runs with limited concurrency (default 5). Defaults to dry-run.

Usage (dry-run):
  python scripts/scan_keep_top.py --n 10000 --keep 5 --concurrency 5 --dry-run

When not dry-run, the script will run evaluations and maintain an in-memory top-K leaderboard
and write the final top-K as CSV.
"""

import argparse
import csv
import math
import os
import shlex
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import random
import heapq
import json
import yaml

try:  # Allow import as a package module or standalone script
    if __package__:
        from .sample_params import lhs_sample  # type: ignore
    else:  # pragma: no cover - fallback when executed directly
        raise ImportError
except (ImportError, AttributeError):
    from sample_params import lhs_sample  # type: ignore


def run_eval(
    base_yaml,
    mg,
    eg,
    games,
    depth,
    concurrency,
    out_csv,
    opponent="sunfish",
    dry_run=False,
):
    # For dry-run we avoid creating temp YAML files to prevent filling /tmp.
    tmp_yaml = None
    if dry_run:
        tmp_yaml = base_yaml
    else:
        tmp_yaml = make_temp_start(base_yaml, mg, eg)

    # If mg/eg are sequences (per-element scalars), pass them via a temp YAML instead
    if isinstance(mg, (list, tuple)) or isinstance(eg, (list, tuple)):
        cmd = f"python scripts/grid_weights_eval.py --start {shlex.quote(tmp_yaml)} --games {games} --depth {depth} --concurrency {concurrency} --opponent {shlex.quote(opponent)} --out {shlex.quote(out_csv)}"
    else:
        cmd = f"python scripts/grid_weights_eval.py --start {shlex.quote(tmp_yaml)} --scales {mg},{eg} --games {games} --depth {depth} --concurrency {concurrency} --opponent {shlex.quote(opponent)} --out {shlex.quote(out_csv)}"
    print("+", cmd)
    if dry_run:
        return 0, ""
    rc = subprocess.run(cmd, shell=True).returncode
    # cleanup temp YAML created for this sample
    try:
        if tmp_yaml and tmp_yaml != base_yaml and os.path.exists(tmp_yaml):
            os.unlink(tmp_yaml)
    except Exception:
        pass
    return rc, out_csv


def read_score_from_csv(path):
    # read first row and compute score (wins + 0.5 draws) / total
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                return 0.0
            r = rows[0]
            w = int(r.get("wins", 0))
            l = int(r.get("losses", 0))
            d = int(r.get("draws", 0))
            total = w + l + d
            if total == 0:
                return 0.0
            return (w + 0.5 * d) / total
    except Exception:
        return 0.0


def make_temp_start(base_yaml, mg_scale, eg_scale):
    # reuse logic from eval_samples_runner: produce a temp YAML with scaled phase weights
    text = open(base_yaml, "r").read()
    stripped = text.lstrip()
    if stripped.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            if lines[-1].startswith("```"):
                text = "\n".join(lines[1:-1])
            else:
                text = "\n".join(lines[1:])
    base = yaml.safe_load(text)
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
        raise SystemExit("base yaml missing phase_weights arrays")
    new = dict(base)

    def expand_group_scales(group_scales, length=15):
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
            new["phase_weights_mg"] = [float(x) * float(mg_scale) for x in mg]
        if eg_per is not None:
            new["phase_weights_eg"] = [a * b for a, b in zip(eg_list, eg_per)]
        else:
            new["phase_weights_eg"] = [float(x) * float(eg_scale) for x in eg]
    else:
        if "evaluation" not in new or not isinstance(new["evaluation"], dict):
            new["evaluation"] = {}
        mg_list = [float(x) for x in mg]
        eg_list = [float(x) for x in eg]
        if mg_per is not None:
            new["evaluation"]["phase_weights_mg"] = [
                a * b for a, b in zip(mg_list, mg_per)
            ]
        else:
            new["evaluation"]["phase_weights_mg"] = [
                float(x) * float(mg_scale) for x in mg
            ]
        if eg_per is not None:
            new["evaluation"]["phase_weights_eg"] = [
                a * b for a, b in zip(eg_list, eg_per)
            ]
        else:
            new["evaluation"]["phase_weights_eg"] = [
                float(x) * float(eg_scale) for x in eg
            ]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", prefix="scan_")
    tmp_name = tmp.name
    tmp.close()
    with open(tmp_name, "w", encoding="utf-8") as fh:
        yaml.safe_dump(new, fh)
    return tmp_name


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--keep", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--opponent", default="sunfish", help="Opponent engine name")
    parser.add_argument("--base", default="tuning/default_start.yaml")
    parser.add_argument("--params", default="tuning/sample_params.json")
    parser.add_argument("--out", default="scan_topk.csv")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum score floor to consider promoting samples (0.0-1.0)",
    )
    parser.add_argument(
        "--force-accept-first",
        type=int,
        default=0,
        help="Always accept the first N completed samples into the top-K (useful to seed leaderboard)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    params = json.load(open(args.params))
    ranges = {k: (float(v[0]), float(v[1])) for k, v in params.items()}
    samples = lhs_sample(ranges, args.n, seed=args.seed)

    out_dir = Path("scan_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    # min-heap of (score, idx, mg, eg, out_csv)
    topk = []

    # ThreadPool to run limited concurrency
    # stream tasks: submit up to concurrency tasks, then as each completes submit the next
    it = iter(enumerate(samples, 1))
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {}
        # submit initial batch
        for _ in range(args.concurrency):
            try:
                i, s = next(it)
            except StopIteration:
                break

            # build group lists if present
            def collect_groups(prefix):
                keys = [f"{prefix}_g{i}" for i in range(7)]
                vals = [s.get(k) for k in keys]
                if any(v is not None for v in vals):
                    return [float(v) for v in vals]
                return None

            mg = collect_groups("mg") or s.get("mg_scale") or s.get("mg")
            eg = collect_groups("eg") or s.get("eg_scale") or s.get("eg")
            out_csv = out_dir / f"sample_{i:06d}.csv"
            fut = ex.submit(
                run_eval,
                args.base,
                mg,
                eg,
                args.games,
                args.depth,
                1,
                str(out_csv),
                args.opponent,
                args.dry_run,
            )
            futures[fut] = (i, mg, eg, out_csv)
            # write meta for reproducibility for submitted sample
            try:
                import json as _json

                meta_path = out_dir / f"sample_{i:06d}.meta.json"
                with open(meta_path, "w", encoding="utf-8") as mf:
                    _json.dump(s, mf, indent=2)
            except Exception:
                pass

        # as tasks complete, submit one more until iterator exhausted
        seen = 0
        while futures:
            done_future = next(as_completed(list(futures)), None)
            if done_future is None:
                break
            i, mg, eg, out_csv = futures.pop(done_future)
            rc, out = done_future.result()
            seen += 1
            if rc != 0:
                print(f"sample {i} failed rc={rc}")
            else:
                if args.dry_run:
                    score = random.random()
                else:
                    score = read_score_from_csv(out)

                # Decide whether to insert/replace in topk.
                # Always fill until we have args.keep samples.
                if len(topk) < args.keep:
                    heapq.heappush(topk, (score, i, mg, eg, str(out)))
                else:
                    curr_min = topk[0][0]
                    replaced = False
                    if score > curr_min:
                        heapq.heapreplace(topk, (score, i, mg, eg, str(out)))
                        replaced = True
                    else:
                        # Allow seeding/improvement even if score not above current min
                        if args.force_accept_first and seen <= args.force_accept_first:
                            heapq.heapreplace(topk, (score, i, mg, eg, str(out)))
                            replaced = True
                        elif score >= args.min_score and curr_min < args.min_score:
                            # Raise the floor: ensure entries in topk are at least min_score
                            heapq.heapreplace(topk, (score, i, mg, eg, str(out)))
                            replaced = True
                    if replaced:
                        pass

            # submit next sample if available
            try:
                i, s = next(it)
            except StopIteration:
                continue
            mg = collect_groups("mg") or s.get("mg_scale") or s.get("mg")
            eg = collect_groups("eg") or s.get("eg_scale") or s.get("eg")
            out_csv = out_dir / f"sample_{i:06d}.csv"
            fut = ex.submit(
                run_eval,
                args.base,
                mg,
                eg,
                args.games,
                args.depth,
                1,
                str(out_csv),
                args.opponent,
                args.dry_run,
            )
            futures[fut] = (i, mg, eg, out_csv)
            # write meta for reproducibility for submitted sample
            try:
                import json as _json

                meta_path = out_dir / f"sample_{i:06d}.meta.json"
                with open(meta_path, "w", encoding="utf-8") as mf:
                    _json.dump(s, mf, indent=2)
            except Exception:
                pass
    # write out top-k sorted descending
    top_sorted = sorted(topk, key=lambda x: x[0], reverse=True)
    fields = ["score", "index", "mg", "eg", "out_csv"]
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for row in top_sorted:
            writer.writerow(row)

    print(f"Wrote top-{args.keep} to {args.out}")


if __name__ == "__main__":
    main()
