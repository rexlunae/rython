#!/usr/bin/env python3
"""The idiom corpus: small idiomatic programs that must convert, compile,
run, and print exactly what CPython prints.

The corpus sweep (eval/sweep/) counts rustc errors on real packages. That
metric cannot see two things: whether ordinary Python -- the shapes a
programmer writes, not urllib3's -- translates at all, and whether a crate
that compiles is silently wrong. This harness measures both: a program
passes only if its generated crate builds AND its stdout diffs clean
against the pinned CPython output. Every program is written to make its
state observable (a printed total after a mutation, not just the mutation's
return value), so a copy-instead-of-alias divergence shows up in the diff.

Usage:
    cargo build -p python-ast -p rypip          # stale rypip = stale measurement
    python3 eval/idioms/run_idioms.py            # run everything, print a table
    python3 eval/idioms/run_idioms.py --only inventory tree
    python3 eval/idioms/run_idioms.py --check-baseline   # CI: fail on regression
    python3 eval/idioms/run_idioms.py --update-baseline  # after a round makes more pass

The baseline (baseline.json) is a ratchet: it lists the programs that pass
today. --check-baseline exits non-zero if any of them stops passing, and
merely reports programs that newly pass (bump the baseline in the PR that
makes them pass). Nothing here fails because a program has never passed;
that is the frontier, not a regression.

Expected outputs are pinned files (programs/NAME.expected) captured from
python3. The runner re-derives them from python3 when it is available and
refuses to measure against a stale pin, so a program edit cannot silently
change what "correct" means.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PROGRAMS = HERE / "programs"
RESULTS = HERE / "results"
BASELINE = HERE / "baseline.json"
DEFAULT_WORKDIR = Path("/tmp/rython-idioms")

ERROR_HEADER = re.compile(r"^error(?:\[(E\d+)\])?: (.*)$", re.M)


def run(cmd: list[str], cwd: Path | None = None, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=cwd, timeout=timeout, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def git_head() -> str:
    p = run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO)
    return p.stdout.strip() if p.returncode == 0 else "unknown"


def error_histogram(log: str) -> dict[str, int]:
    hist: dict[str, int] = {}
    for m in ERROR_HEADER.finditer(log):
        code = m.group(1) or "error"
        if m.group(2).startswith("could not compile"):
            continue
        hist[code] = hist.get(code, 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: -kv[1]))


def verify_expected(program: Path, expected: Path) -> str | None:
    """Re-run the program under python3 and compare with the pin. Returns
    an error string if the pin is stale (or python3 fails), else None."""
    python = shutil.which("python3")
    if python is None:
        return None  # no oracle available; trust the pin
    p = run([python, str(program)], timeout=60)
    if p.returncode != 0:
        return f"python3 exited {p.returncode}: {p.stderr.strip()[-200:]}"
    if p.stdout != expected.read_text():
        return "pinned .expected differs from a live python3 run (regenerate the pin)"
    return None


def run_one(program: Path, rypip: Path, workdir: Path, keep: bool) -> dict:
    name = program.stem
    expected = program.with_suffix(".expected")
    result: dict = {"program": name}
    if not expected.is_file():
        result["status"] = "no-expected"
        return result
    stale = verify_expected(program, expected)
    if stale:
        result["status"] = "expected-stale"
        result["detail"] = stale
        return result

    crate = workdir / f"crate-{name}"
    if crate.exists():
        shutil.rmtree(crate)
    convert = run([str(rypip), "convert", "--out", str(crate), str(program)])
    if convert.returncode != 0:
        result["status"] = "convert-failed"
        result["detail"] = convert.stderr.strip()[-400:]
        return result
    result["warnings"] = sum(1 for line in convert.stderr.splitlines() if "warning" in line)

    build = run(["cargo", "build", "--quiet"], cwd=crate, timeout=1800)
    log = build.stderr
    (crate / "build.log").write_text(log)
    if build.returncode != 0:
        hist = error_histogram(log)
        result["status"] = "build-failed"
        result["errors"] = sum(hist.values())
        result["histogram"] = hist
        return result

    binary = crate / "target" / "debug" / name
    if not binary.is_file():
        candidates = [p for p in (crate / "target" / "debug").iterdir()
                      if p.is_file() and p.stat().st_mode & 0o111 and p.suffix == ""]
        if len(candidates) != 1:
            result["status"] = "no-binary"
            result["detail"] = f"expected target/debug/{name}; found {[c.name for c in candidates]}"
            return result
        binary = candidates[0]
    ran = run([str(binary)], timeout=60)
    if ran.returncode != 0:
        result["status"] = "run-failed"
        result["detail"] = f"exit {ran.returncode}: {ran.stderr.strip()[-300:]}"
        return result
    if ran.stdout != expected.read_text():
        result["status"] = "output-mismatch"
        result["detail"] = first_difference(expected.read_text(), ran.stdout)
        return result
    result["status"] = "pass"
    if not keep:
        shutil.rmtree(crate, ignore_errors=True)
    return result


def first_difference(want: str, got: str) -> str:
    w, g = want.splitlines(), got.splitlines()
    for i, (a, b) in enumerate(zip(w, g)):
        if a != b:
            return f"line {i + 1}: expected {a!r}, got {b!r}"
    if len(w) != len(g):
        return f"expected {len(w)} lines, got {len(g)}"
    return "trailing whitespace or newline differs"


def load_baseline() -> list[str]:
    if not BASELINE.is_file():
        return []
    return sorted(json.loads(BASELINE.read_text()).get("passing", []))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rypip", default=str(REPO / "target" / "debug" / "rypip"))
    ap.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    ap.add_argument("--out", default=None, help="results JSON path (default results/run-<head>.json)")
    ap.add_argument("--only", nargs="*", default=None, help="program names to run")
    ap.add_argument("--keep", action="store_true", help="keep every generated crate, not just failing ones")
    ap.add_argument("--check-baseline", action="store_true",
                    help="exit 1 if a program listed in baseline.json no longer passes")
    ap.add_argument("--update-baseline", action="store_true",
                    help="rewrite baseline.json with the programs that pass now")
    args = ap.parse_args()

    rypip = Path(args.rypip)
    if not rypip.is_file():
        print(f"rypip not found at {rypip}; run `cargo build -p python-ast -p rypip` first", file=sys.stderr)
        return 2
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    programs = sorted(PROGRAMS.glob("*.py"))
    if args.only:
        wanted = set(args.only)
        programs = [p for p in programs if p.stem in wanted]
        missing = wanted - {p.stem for p in programs}
        if missing:
            print(f"unknown programs: {sorted(missing)}", file=sys.stderr)
            return 2

    results = {}
    for program in programs:
        r = run_one(program, rypip, workdir, args.keep)
        results[program.stem] = r
        extra = ""
        if r["status"] == "build-failed":
            extra = f"{r['errors']} errors " + " ".join(f"{k}x{v}" for k, v in list(r["histogram"].items())[:4])
        elif "detail" in r:
            extra = r["detail"].splitlines()[-1][:100] if r["detail"] else ""
        print(f"{program.stem:<14} {r['status']:<16} {extra}")

    passing = sorted(n for n, r in results.items() if r["status"] == "pass")
    stale = sorted(n for n, r in results.items() if r["status"] == "expected-stale")
    print(f"\n{len(passing)}/{len(results)} pass" + (f"; STALE PINS: {stale}" if stale else ""))

    payload = {
        "rypip_commit": git_head(),
        "passed": len(passing),
        "total": len(results),
        "passing": passing,
        "programs": results,
    }
    out = Path(args.out) if args.out else RESULTS / f"run-{payload['rypip_commit']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")

    rc = 0
    if stale:
        rc = 2
    if args.update_baseline:
        BASELINE.write_text(json.dumps({"passing": passing}, indent=2) + "\n")
        print(f"baseline updated: {passing}")
    elif args.check_baseline:
        baseline = load_baseline()
        checked = [n for n in baseline if n in results]
        regressed = [n for n in checked if results[n]["status"] != "pass"]
        new = [n for n in passing if n not in baseline]
        if regressed:
            print(f"REGRESSION: baseline programs no longer pass: {regressed}")
            rc = 1
        if new:
            print(f"newly passing (add to baseline.json in this PR): {new}")
        if not regressed:
            print(f"baseline holds: {len(checked)} program(s)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
