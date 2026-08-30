#!/usr/bin/env python3
"""Diff two sweep results: baseline vs candidate.

Prints, per package, the error-count delta and the per-error-code delta,
then the "expected X, found Y" pairs that changed — the same
transmutation signal (clearing an access error exposing a type mismatch)
the round notes hand-rolled from full logs.

Usage:
    python3 summarize.py results/run-BASELINE.json results/run-CANDIDATE.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(path: str) -> dict:
    data = json.loads(Path(path).read_text())
    assert "packages" in data, f"{path} is not a sweep result (missing packages)"
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    args = ap.parse_args()

    base = load(args.baseline)
    cand = load(args.candidate)
    print(f"baseline  : {args.baseline}  ({base.get('rypip_commit', '?')})")
    print(f"candidate : {args.candidate}  ({cand.get('rypip_commit', '?')})")
    print()

    grand_delta = 0
    for name in sorted(set(base["packages"]) | set(cand["packages"])):
        b = base["packages"].get(name, {})
        c = cand["packages"].get(name, {})
        bt = b.get("total")
        ct = c.get("total")
        if bt is None or ct is None:
            state = "convert-failed" if ct is None else "new"
            print(f"== {name}: {state} (base total={bt}, cand total={ct})")
            continue
        delta = ct - bt
        grand_delta += delta
        print(f"== {name}: {bt} -> {ct} ({delta:+d})")
        bh, ch = b.get("histogram", {}), c.get("histogram", {})
        for code in sorted(set(bh) | set(ch), key=lambda k: -(ch.get(k, 0) - bh.get(k, 0))):
            d = ch.get(code, 0) - bh.get(code, 0)
            if d:
                print(f"    {code}: {bh.get(code, 0)} -> {ch.get(code, 0)} ({d:+d})")
        bp, cp = b.get("e0308_pairs", {}), c.get("e0308_pairs", {})
        changed = {k for k in set(bp) | set(cp) if bp.get(k) != cp.get(k)}
        if changed:
            print("    E0308 pairs that changed:")
            for k in sorted(changed, key=lambda k: -abs(cp.get(k, 0) - bp.get(k, 0)))[:12]:
                d = cp.get(k, 0) - bp.get(k, 0)
                print(f"      expected {k}: {bp.get(k, 0)} -> {cp.get(k, 0)} ({d:+d})")
        print()
    print(f"grand total delta: {grand_delta:+d}")


if __name__ == "__main__":
    main()
