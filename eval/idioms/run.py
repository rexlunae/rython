#!/usr/bin/env python3
"""Idiom-corpus runner (issue #137, Directive 3).

Every program in this directory is a shape a Python programmer writes
(30-100 lines). A program PASSES only if it
  1. converts (rythonc exits 0),
  2. the generated crate compiles, and
  3. the built binary's stdout diffs byte-for-byte against the
     CPython-captured <name>.expected.txt.

The corpus exists because the rustc-error sweep can be gamed by
urllib3-shaped code: this corpus is written BEFORE the fix that claims
to address it, and it asserts STATE (the expected stdout includes
values that only hold if mutations reached the stored objects), not
just the happy-path output.

Usage:
    python3 eval/idioms/run.py                 # run every program
    python3 eval/idioms/run.py inventory       # run one program
Prints one line per program and a final pass summary.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
RYTHONC = ROOT / "target" / "debug" / "rythonc"
STDPYTHON = ROOT / "crates" / "stdpython"

CARGO_TOML = """\
[package]
name = "{crate}"
version = "0.1.0"
edition = "2021"

[dependencies]
stdpython = {{ path = "{stdpython}" }}
"""


def run_one(py: pathlib.Path) -> tuple[str, str]:
    """Run one program. Returns (status, detail).

    status: pass | fail-convert | fail-compile | fail-run | fail-diff
    """
    expected = py.with_suffix(".expected.txt")
    want = expected.read_text() if expected.exists() else ""
    with tempfile.TemporaryDirectory(prefix="rython-idiom-") as td:
        td = pathlib.Path(td)
        conv = subprocess.run(
            [str(RYTHONC), str(py)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if conv.returncode != 0:
            detail = (conv.stderr or conv.stdout).strip().splitlines()
            return "fail-convert", detail[0] if detail else "conversion error"
        src = td / "src"
        src.mkdir()
        # The converted program is a script: its `main()` is the entry,
        # so the generated code builds as the crate's main.rs.
        (src / "main.rs").write_text(conv.stdout)
        (td / "Cargo.toml").write_text(
            CARGO_TOML.format(crate=py.stem.replace("-", "_"), stdpython=STDPYTHON)
        )
        build = subprocess.run(
            ["cargo", "build", "--release", "-q"],
            cwd=td,
            capture_output=True,
            text=True,
            timeout=900,
        )
        if build.returncode != 0:
            errors = [
                l
                for l in build.stderr.splitlines()
                if l.startswith("error[E") or l.startswith("error:")
            ]
            return "fail-compile", f"{len(errors)} rustc errors"
        run = subprocess.run(
            [str(td / "target" / "release" / py.stem.replace("-", "_"))],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if run.returncode != 0:
            return "fail-run", f"exit {run.returncode}"
        if run.stdout != want:
            return "fail-diff", "stdout differs from CPython"
    return "pass", ""


def collect(names: list[str] | None = None) -> dict:
    """Run every program (or the named ones) and return the per-program
    status map plus counts — the shape run_sweep.py embeds next to the
    rustc-error histogram."""
    programs = sorted(HERE.glob("*.py"))
    if names:
        wanted = set(names)
        programs = [p for p in programs if p.stem in wanted]
    statuses: dict[str, str] = {}
    details: dict[str, str] = {}
    for py in programs:
        status, detail = run_one(py)
        statuses[py.stem] = status
        if detail:
            details[py.stem] = detail
    counts: dict[str, int] = {}
    for s in statuses.values():
        counts[s] = counts.get(s, 0) + 1
    return {
        "total": len(programs),
        "pass": counts.get("pass", 0),
        "statuses": statuses,
        "details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", help="program stem(s); default: all")
    args = parser.parse_args()

    result = collect(args.names)
    programs = sorted(p.stem for p in HERE.glob("*.py"))
    if args.names:
        programs = [p for p in programs if p in args.names]
    for name in programs:
        detail = result["details"].get(name, "")
        print(f"{name:14s} {result['statuses'].get(name, 'missing')}"
              + (f"  ({detail})" if detail else ""))
    print(f"idioms: {result['pass']}/{result['total']} pass")
    if result["pass"] == result["total"]:
        return 0
    non_pass = {k: v for k, v in result.items() if k not in ("total", "pass", "statuses", "details")}
    print(" ".join(f"{k}={v}" for k, v in sorted(non_pass.items())), file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
