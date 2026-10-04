#!/usr/bin/env python3
"""Diff sweep measurements without treating missing/failed measurements as improvements."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(path: str) -> dict:
    data = json.loads(Path(path).read_text())
    if not isinstance(data.get("packages"), dict):
        raise ValueError(f"{path} is not a sweep result (missing packages)")
    return data


def measurement_problem(result: dict) -> str | None:
    if not result:
        return "package absent"
    if result.get("error"):
        return result["error"]
    if result.get("convert_status") != 0 or result.get("total") is None:
        return result.get("status", "conversion failed or unmeasured")
    if "status" in result:
        if result["status"] not in {"built", "build-failed"}:
            return result["status"]
        if result["status"] == "built" and result.get("build_status") == 0 and result["total"] == 0 and result.get("uncoded_total") == 0:
            return None
        if result["status"] == "build-failed" and result.get("build_status") == 101 and result["total"] + result.get("uncoded_total", 0) > 0:
            return None
        return "inconsistent build status and diagnostic counts"
    # Older records cannot account for uncoded errors, but a nonzero coded
    # count is still useful. A failed build with zero coded errors is NOT
    # distinguishable from an infrastructure failure in that schema.
    if result.get("build_status") == 0 and result["total"] == 0:
        return None
    if result.get("build_status") == 101 and result["total"] > 0:
        return None
    return "legacy record has no verifiable build measurement"


def summarize(base: dict, cand: dict) -> bool:
    """Return whether the entire requested comparison is measurable."""
    if not base["packages"] or not cand["packages"]:
        print("grand total delta: unavailable (empty corpus)")
        return False
    complete = True
    for label, data in (("baseline", base), ("candidate", cand)):
        if data.get("measurement_error"):
            print(f"{label}: UNMEASURED — {data['measurement_error']}")
            print("grand total delta: unavailable")
            return False
        if data.get("measurement_complete") is False:
            complete = False
            print(f"{label}: incomplete measurement")
    grand_delta, uncoded_delta = 0, 0
    uncoded_known = True
    for name in sorted(set(base["packages"]) | set(cand["packages"])):
        b, c = base["packages"].get(name, {}), cand["packages"].get(name, {})
        problems = [f"{label}: {problem}" for label, r in (("base", b), ("candidate", c))
                    if (problem := measurement_problem(r))]
        if b.get("requirement") != c.get("requirement"):
            problems.append("package pin changed")
        if problems:
            print(f"== {name}: NOT COMPARABLE ({'; '.join(problems)})")
            complete = False
            continue
        bt, ct = b["total"], c["total"]
        delta = ct - bt
        grand_delta += delta
        bs = b.get("status", "built" if b["build_status"] == 0 else "build-failed")
        cs = c.get("status", "built" if c["build_status"] == 0 else "build-failed")
        print(f"== {name}: {bt} -> {ct} ({delta:+d}) coded errors; {bs} -> {cs}")
        bu, cu = b.get("uncoded_total"), c.get("uncoded_total")
        if bu is None or cu is None:
            uncoded_known = False
            print(f"    uncoded errors: {bu if bu is not None else 'unknown (legacy)'} -> {cu if cu is not None else 'unknown (legacy)'}")
        else:
            uncoded_delta += cu - bu
            print(f"    uncoded errors: {bu} -> {cu} ({cu - bu:+d}); all diagnostics: {bt + bu} -> {ct + cu}")
        bh, ch = b.get("histogram", {}), c.get("histogram", {})
        for code in sorted(set(bh) | set(ch), key=lambda k: (-(ch.get(k, 0) - bh.get(k, 0)), k)):
            d = ch.get(code, 0) - bh.get(code, 0)
            if d:
                print(f"    {code}: {bh.get(code, 0)} -> {ch.get(code, 0)} ({d:+d})")
        bp, cp = b.get("e0308_pairs", {}), c.get("e0308_pairs", {})
        changed = {k for k in set(bp) | set(cp) if bp.get(k) != cp.get(k)}
        if changed:
            print("    E0308 pairs that changed:")
            for k in sorted(changed, key=lambda k: (-abs(cp.get(k, 0) - bp.get(k, 0)), k))[:12]:
                d = cp.get(k, 0) - bp.get(k, 0)
                print(f"      expected {k}: {bp.get(k, 0)} -> {cp.get(k, 0)} ({d:+d})")
        print()
    if complete:
        print(f"grand total delta (coded errors only): {grand_delta:+d}")
        if uncoded_known:
            print(f"grand total delta (all diagnostics): {grand_delta + uncoded_delta:+d}")
        else:
            print("grand total delta (all diagnostics): unavailable (legacy uncoded counts unknown)")
    else:
        print(f"comparable-package subtotal (coded errors only): {grand_delta:+d}")
        print("grand total delta: unavailable (incomplete comparison)")
    return complete


def rustc_host(record: dict) -> str:
    """The `host:` line of the record's rustc banner, or '?' when absent."""
    for line in (record.get("rustc") or "").splitlines():
        if line.strip().startswith("host:"):
            return line.split(":", 1)[1].strip()
    return "?"


def effective_target(record: dict) -> str:
    """The target the generated crate was actually built for.

    An empty `build_target` means "no CARGO_BUILD_TARGET", which builds for
    the rustc HOST — so a run that sets the variable to its own native triple
    and a run that leaves it unset are the SAME build and must not be
    reported as a mismatch. Normalising empty to the host makes that true.
    Returns "?" when neither fact is available.
    """
    target = record.get("build_target")
    if target:
        return target
    return rustc_host(record)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    args = ap.parse_args()
    base, cand = load(args.baseline), load(args.candidate)
    print(f"baseline  : {args.baseline}  ({base.get('rypip_commit', '?')})")
    print(f"candidate : {args.candidate}  ({cand.get('rypip_commit', '?')})\n")

    # The counts are host-dependent (see README: same commit and rustc
    # measure 3006 on aarch64-apple-darwin and 2575 on x86_64-linux, of which
    # requests alone is 428). A delta across two hosts is not a delta, so say
    # so loudly rather than printing a number that invites a claim.
    hb, hc = rustc_host(base), rustc_host(cand)
    mismatched = []
    if "?" not in (hb, hc) and hb != hc:
        mismatched.append(f"host: baseline {hb}, candidate {hc}")
    # A cross-CARGO_BUILD_TARGET pair counts too: the rustc banner reports the
    # HOST, so cross-compiling keeps the host identical while the generated
    # crate is built for a different target. `effective_target` normalises the
    # unset case to the host, so only a REAL difference is flagged.
    tb, tc = effective_target(base), effective_target(cand)
    if "?" not in (tb, tc) and tb != tc:
        mismatched.append(f"build target: baseline {tb}, candidate {tc}")
    # Cargo also takes the target from .cargo/config.toml, and
    # `cargo config get` is nightly-only, so run_sweep.py fingerprints that
    # file instead. Different configs cannot be compared even when the
    # environment variable agrees.
    cb, cc = base.get("cargo_config_sha256"), cand.get("cargo_config_sha256")
    if cb is not None and cc is not None and cb != cc:
        mismatched.append("cargo config (.cargo/config.toml) differs between the runs")
    if mismatched:
        print("!! NOT COMPARABLE — these records do not measure the same build:")
        for m in mismatched:
            print(f"!!   {m}")
        print("!! The counts are host-dependent; this delta does not mean")
        print("!! anything. Re-run BOTH records on the same machine and target.\n")
    return 0 if summarize(base, cand) else 1


if __name__ == "__main__":
    sys.exit(main())
