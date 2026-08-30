#!/usr/bin/env python3
"""Error-count sweep for the issue #137 frontier.

For every pinned package in packages.json this:

  1. writes a probe package (pyproject.toml with the pinned requirement,
     plus `import <name>` in __init__.py);
  2. converts it with rypip (the freshly built debug binary by default —
     a stale binary silently measures old codegen, a trap this harness
     exists to prevent);
  3. builds the generated crate with cargo and captures the log;
  4. records the rustc error histogram: total, by error code, and the
     "expected X, found Y" pair breakdown for E0308.

Output is one JSON file per run under results/, so summarize.py can diff
two runs (baseline vs candidate) into a delta report.

Usage:
    python3 run_sweep.py [--out results/run-YYYYMMDD-HHMMSS.json]
                         [--rypip /path/to/target/debug/rypip]
                         [--workdir /tmp/rython-sweep]
                         [--package urllib3]      # subset, repeatable
                         [--jobs 4]              # parallel crate builds

Traps documented here so no future session rediscovers them:
  * rypip must be freshly rebuilt (`cargo build -p python-ast -p rypip`)
    before a sweep — the PATH-installed binary may be stale.
  * The build log redirect must be `> log 2>&1` (stderr first). The
    `2>&1 > file` order sends stderr to the terminal and you count
    nothing.
  * The error-count line (`could not compile ... due to N previous
    errors`) is not an error site; the histogram counts `^error[` lines.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RYPIP = ROOT / "target" / "debug" / "rypip"
DEFAULT_WORKDIR = Path("/tmp/rython-sweep")
RESULTS = Path(__file__).resolve().parent / "results"

ERROR_HEADER = re.compile(r"^error\[(E[0-9]{4})\]:", re.MULTILINE)
E0308_PAIR = re.compile(r"expected `([^`]+)`, found `([^`]+)`")


def run(cmd, cwd=None, timeout=1800):
    p = subprocess.run(cmd, cwd=cwd, timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode not in (0, 101):
        # 101 = rustc "could not compile"; the log is the measurement.
        if p.returncode not in (1,):
            pass
    return p


def write_probe(pkg_root: Path, name: str, requirement: str) -> None:
    pkg_root.mkdir(parents=True, exist_ok=True)
    (pkg_root / "pyproject.toml").write_text(
        f'[project]\nname = "probe"\nversion = "0.1.0"\n'
        f'dependencies = ["{requirement}"]\n'
    )
    pkg_dir = pkg_root / name.replace("-", "_")
    pkg_dir.mkdir(exist_ok=True)
    (pkg_dir / "__init__.py").write_text(f"import {name.replace('-', '_')}\n")


def parse_errors(log: str) -> dict:
    histogram = {}
    pairs = {}
    for m in ERROR_HEADER.finditer(log):
        code = m.group(1)
        histogram[code] = histogram.get(code, 0) + 1
    # E0308 "expected X, found Y" breakdown (the dominant frontier class).
    for m in E0308_PAIR.finditer(log):
        key = f"{m.group(1)} | {m.group(2)}"
        pairs[key] = pairs.get(key, 0) + 1
    return {
        "total": sum(histogram.values()),
        "histogram": dict(sorted(histogram.items(), key=lambda kv: -kv[1])),
        "e0308_pairs": dict(sorted(pairs.items(), key=lambda kv: -kv[1])),
    }


def sweep_one(spec: dict, rypip: Path, workdir: Path) -> dict:
    name = spec["name"]
    crate = workdir / f"crate-{name}"
    probe = workdir / f"probe-{name}"
    write_probe(probe, name, spec["requirement"])
    convert = run([str(rypip), "convert", "--out", str(crate), str(probe)])
    if convert.returncode != 0:
        return {
            "package": name,
            "requirement": spec["requirement"],
            "convert_status": convert.returncode,
            "convert_stderr": convert.stderr[-800:],
            "total": None,
        }
    log_path = workdir / f"{name}-build.log"
    with open(log_path, "w") as f:
        p = subprocess.run(
            ["cargo", "build"],
            cwd=crate,
            stdout=f,
            stderr=subprocess.STDOUT,
            timeout=1800,
        )
    build_status = p.returncode
    stats = parse_errors(log_path.read_text(errors="replace"))
    stats["package"] = name
    stats["requirement"] = spec["requirement"]
    stats["convert_status"] = 0
    stats["build_status"] = build_status
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="results/run-*.json path")
    ap.add_argument("--rypip", type=Path, default=DEFAULT_RYPIP)
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--package", action="append", default=None)
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()

    specs = json.loads((Path(__file__).resolve().parent / "packages.json").read_text())["packages"]
    if args.package:
        specs = [s for s in specs if s["name"] in args.package]
    if not specs:
        sys.exit("no packages selected")

    if not args.rypip.exists():
        sys.exit(f"rypip not found at {args.rypip} — build it first "
                 f"(`cargo build -p python-ast -p rypip`)")
    args.workdir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    results = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futures = {ex.submit(sweep_one, s, args.rypip, args.workdir): s["name"] for s in specs}
        for fut, name in futures.items():
            try:
                results[name] = fut.result()
            except Exception as e:  # noqa: BLE001 — a sweep must not die on one package
                results[name] = {"package": name, "error": str(e)}

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = args.out or str(RESULTS / f"run-{stamp}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rypip_commit": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip(),
        "rypip_path": str(args.rypip),
        "elapsed_seconds": round(time.time() - started, 1),
        "packages": results,
    }
    Path(out).write_text(json.dumps(payload, indent=2) + "\n")
    for name, r in results.items():
        total = r.get("total")
        print(f"{name:12} {specs and ''}{'' if total is None else total} errors"
              f"{' (convert failed)' if total is None else ''}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
