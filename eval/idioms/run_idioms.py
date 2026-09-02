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
from datetime import datetime, timezone
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
# The interpreter the pins were captured under. Another minor version is
# not refused -- the live re-derivation already catches any output drift
# loudly -- but it is reported, so a "stale pin" verdict can be read right.
PINNED_PYTHON = (3, 11)


def run(cmd: list[str], cwd: Path | None = None, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=cwd, timeout=timeout, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def git_head() -> str:
    p = run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO)
    return p.stdout.strip() if p.returncode == 0 else "unknown"


def converter_commit() -> str:
    """The last commit that touched the converter's source -- what
    target/debug/rypip was built from, PROVIDED it was rebuilt (the README
    trap). This, not HEAD, names a result file: a branch that only edits
    eval/ measures the same converter as its merge-base."""
    p = run(["git", "log", "-1", "--format=%h", "--",
             "crates/python-ast", "crates/rypip", "crates/stdpython"], cwd=REPO)
    return p.stdout.strip() if p.returncode == 0 and p.stdout.strip() else "unknown"


def oracle() -> tuple[str | None, str | None]:
    """(path, description) of the CPython oracle, or (None, reason). The
    corpus pins CPython's behavior; another implementation on PATH would
    silently redefine what "correct" means, so it is refused."""
    python = shutil.which("python3")
    if python is None:
        return None, "no python3 on PATH"
    p = run([python, "-c",
             "import platform, sys; print(platform.python_implementation(), "
             "sys.version_info[0], sys.version_info[1], sys.version_info[2])"], timeout=30)
    if p.returncode != 0:
        return None, f"python3 failed to report its version: {p.stderr.strip()[-100:]}"
    impl, major, minor, micro = p.stdout.split()
    if impl != "CPython":
        return None, f"python3 is {impl}, not CPython; the corpus pins CPython behavior"
    desc = f"CPython {major}.{minor}.{micro}"
    if (int(major), int(minor)) != PINNED_PYTHON:
        desc += f" (pins captured under {PINNED_PYTHON[0]}.{PINNED_PYTHON[1]})"
    return python, desc


def converter_sources_newest() -> float:
    """mtime of the newest tracked file under the converter crates."""
    p = run(["git", "ls-files", "-z", "--",
             "crates/python-ast", "crates/rypip", "crates/stdpython"], cwd=REPO)
    newest = 0.0
    for rel in p.stdout.split("\0"):
        if rel:
            try:
                newest = max(newest, (REPO / rel).stat().st_mtime)
            except FileNotFoundError:
                pass
    return newest


def stale_binary(rypip: Path) -> str | None:
    """A result file is named by the converter's source commit, which is
    only true of a binary built AFTER the last source change. Refuse to
    measure with one that was not -- the trap the sweep README records as
    having cost rounds -- rather than infer provenance and hope."""
    built = rypip.stat().st_mtime
    newest = converter_sources_newest()
    if newest > built:
        fmt = lambda t: datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds")
        return (f"{rypip} was built at {fmt(built)} but converter source changed at "
                f"{fmt(newest)}: rebuild it (cargo build -p python-ast -p rypip) or pass --allow-stale")
    return None


def error_histogram(log: str) -> dict[str, int]:
    hist: dict[str, int] = {}
    for m in ERROR_HEADER.finditer(log):
        code = m.group(1) or "error"
        if m.group(2).startswith("could not compile"):
            continue
        hist[code] = hist.get(code, 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: -kv[1]))


def verify_expected(program: Path, expected: Path, python: str | None) -> tuple[str | None, str]:
    """Re-run the program under the CPython oracle and compare with the
    pin. Returns (error, expected_stderr): error is set if the pin is stale
    or the oracle fails; expected_stderr is what the generated binary must
    also emit (empty when no oracle is available -- a passing program is
    silent on stderr, so a warning or panic text there is a divergence)."""
    if python is None:
        return None, ""  # no oracle available; trust the pin
    p = run([python, str(program)], timeout=60)
    if p.returncode != 0:
        return f"python3 exited {p.returncode}: {p.stderr.strip()[-200:]}", ""
    if p.stdout != expected.read_text():
        return "pinned .expected differs from a live python3 run (regenerate the pin)", ""
    return None, p.stderr


def run_one(program: Path, rypip: Path, workdir: Path, keep: bool, python: str | None) -> dict:
    """One program's measurement. A hang anywhere in its pipeline is that
    program's `timeout` status, never an abort of the remaining corpus."""
    try:
        return measure(program, rypip, workdir, keep, python)
    except subprocess.TimeoutExpired as e:
        cmd = e.cmd[0] if isinstance(e.cmd, list) else str(e.cmd)
        return {"program": program.stem, "status": "timeout",
                "detail": f"{Path(cmd).name} exceeded {e.timeout:.0f}s"}


def measure(program: Path, rypip: Path, workdir: Path, keep: bool, python: str | None) -> dict:
    name = program.stem
    expected = program.with_suffix(".expected")
    result: dict = {"program": name}
    if not expected.is_file():
        result["status"] = "no-expected"
        return result
    stale, expected_err = verify_expected(program, expected, python)
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
    if ran.stderr != expected_err:
        result["status"] = "output-mismatch"
        result["detail"] = f"stderr differs: expected {expected_err!r}, got {ran.stderr[-200:]!r}"
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
    ap.add_argument("--only", nargs="+", default=None, metavar="NAME", help="program names to run")
    ap.add_argument("--keep", action="store_true", help="keep every generated crate, not just failing ones")
    ap.add_argument("--check-baseline", action="store_true",
                    help="exit 1 if a program listed in baseline.json no longer passes")
    ap.add_argument("--update-baseline", action="store_true",
                    help="rewrite baseline.json with the programs that pass now")
    ap.add_argument("--force-baseline", action="store_true",
                    help="with --update-baseline: drop programs that regressed or were deleted (the ratchet refuses otherwise)")
    ap.add_argument("--allow-stale", action="store_true",
                    help="measure with a rypip older than the converter source (the result file will be misnamed)")
    args = ap.parse_args()

    rypip = Path(args.rypip)
    if not rypip.is_file():
        print(f"rypip not found at {rypip}; run `cargo build -p python-ast -p rypip` first", file=sys.stderr)
        return 2
    stale = stale_binary(rypip)
    if stale and not args.allow_stale:
        print(f"stale converter: {stale}", file=sys.stderr)
        return 2
    python, oracle_desc = oracle()
    if python is None and oracle_desc != "no python3 on PATH":
        print(f"oracle refused: {oracle_desc}", file=sys.stderr)
        return 2
    print(f"oracle: {oracle_desc if python else 'none (trusting pins)'}")
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
        r = run_one(program, rypip, workdir, args.keep, python)
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
        "converter_commit": converter_commit(),
        "repo_head": git_head(),
        "rypip_path": str(rypip.relative_to(REPO)) if rypip.is_relative_to(REPO) else str(rypip),
        "oracle": oracle_desc if python else None,
        "passed": len(passing),
        "total": len(results),
        "passing": passing,
        "programs": results,
    }
    out = Path(args.out) if args.out else RESULTS / f"run-{payload['converter_commit']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")

    rc = 0
    if stale:
        rc = 2
    baseline = load_baseline()
    if args.update_baseline:
        # Under --only, programs that were not run keep their entry; only
        # the selected ones are re-decided. Without --only every program
        # ran, so an absent entry is a deleted program.
        kept = [n for n in baseline if n not in results] if args.only else []
        lost = [n for n in baseline if n not in kept and n not in passing]
        if lost and not args.force_baseline:
            # The ratchet only ever tightens on its own: a transient failure
            # must not silently remove a guarantee. Dropping one is a
            # deliberate act, taken with --force-baseline.
            print(f"REFUSED: updating would drop baseline programs that no longer pass "
                  f"or no longer exist: {lost} (pass --force-baseline to drop them)")
            return 1
        merged = sorted(set(kept) | set(passing))
        BASELINE.write_text(json.dumps({"passing": merged}, indent=2) + "\n")
        print(f"baseline updated: {merged}" + (f" (dropped {lost})" if lost else ""))
    elif args.check_baseline:
        if args.only:
            skipped = [n for n in baseline if n not in results]
            if skipped:
                print(f"not run (--only), not checked: {skipped}")
            to_check = [n for n in baseline if n in results]
        else:
            to_check = baseline  # everything ran: a missing entry is a deleted program
        regressed = [n for n in to_check
                     if n not in results or results[n]["status"] != "pass"]
        missing = [n for n in regressed if n not in results]
        new = [n for n in passing if n not in baseline]
        if regressed:
            print(f"REGRESSION: baseline programs no longer pass: {regressed}"
                  + (f" (missing from programs/: {missing})" if missing else ""))
            rc = 1
        if new:
            print(f"newly passing (add to baseline.json in this PR): {new}")
        if not regressed:
            print(f"baseline holds: {len(to_check)} program(s)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
