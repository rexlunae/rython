#!/usr/bin/env python3
"""Render results.json / bench.json into the markdown tables used in REPORT.md."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ORDER = {"PASS": 0, "DIVERGE": 1, "RUN_FAIL": 2, "BUILD_FAIL": 3, "CONVERT_FAIL": 4}


def worst(statuses):
    return sorted(statuses, key=lambda s: ORDER.get(s, 9))[-1] if statuses else "?"


def status_table(results):
    lines = ["| case | default (scalar) | scalar | rayon | simd | auto |",
             "|---|---|---|---|---|---|"]
    cols = ["default-auto", "cpu-scalar", "cpu-rayon", "cpu-simd", "cpu-auto"]
    for rec in results:
        row = [rec["case"]]
        for c in cols:
            row.append(rec["variants"].get(c, {}).get("status", "-"))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def tally(results):
    counts = {}
    for rec in results:
        s = worst([v.get("status", "?") for v in rec["variants"].values()])
        counts[s] = counts.get(s, 0) + 1
    return counts


def backend_agreement(results):
    """Do all backends produce identical stdout for the same case?"""
    disagree = []
    for rec in results:
        outs = {k: (v.get("rc"), v.get("stdout"))
                for k, v in rec["variants"].items() if "stdout" in v}
        if len(set(outs.values())) > 1:
            disagree.append((rec["case"], outs))
    return disagree


def bench_table(bench):
    lines = ["| program | kernel | n | CPython (s/op) | scalar | simd | rayon | "
             "scalar vs CPython | rayon vs CPython |", "|---|---|---|---|---|---|---|---|---|"]
    for prog, entry in sorted(bench.items()):
        if prog.startswith("_"):
            continue
        cpy = entry.get("cpython") or {}
        keys = sorted(cpy, key=lambda k: (k.split("|")[0], int(k.split("|")[1])))
        for k in keys:
            kernel, n = k.split("|")
            c = cpy[k]["seconds"]
            row = [prog.replace("bench_", ""), kernel, n, f"{c:.3e}"]
            speeds = {}
            for be in ("scalar", "simd", "rayon"):
                v = entry["rython"].get(be, {})
                if "error" in v or k not in v:
                    row.append("-")
                    speeds[be] = None
                else:
                    speeds[be] = v[k]["seconds"]
                    row.append(f"{speeds[be]:.3e}")
            for be in ("scalar", "rayon"):
                row.append(f"{c / speeds[be]:.2f}x" if speeds.get(be) else "-")
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/tmp/npeval2/results.json")
    ap.add_argument("--bench", default="/tmp/npbench/bench.json")
    args = ap.parse_args()

    rp = Path(args.results)
    if rp.exists():
        results = json.loads(rp.read_text())
        print("## Correctness\n")
        print(status_table(results))
        print("\ntally:", tally(results))
        d = backend_agreement(results)
        print(f"\ncases where backends disagree with each other: {len(d)}")
        for case, outs in d:
            print(" ", case)
            for k, v in outs.items():
                print("   ", k, repr(v[1][:120]))

    bp = Path(args.bench)
    if bp.exists():
        bench = json.loads(bp.read_text())
        print("\n## Speed\n")
        print(bench_table(bench))
        if "_startup" in bench:
            s = bench["_startup"]
            print(f"\nstartup: cpython {s['cpython_seconds']*1000:.1f} ms, "
                  f"rython {s['rython_seconds']*1000:.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
