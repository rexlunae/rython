#!/usr/bin/env python3
"""Rank the shared compiler causes behind a sweep record.

Answers "which fix moves the frontier?" rather than "which package is
biggest": it merges the per-package E-code histograms into one corpus
histogram, splits each code by the *shape* of its messages, and names the
attribute/method a missing-operand fix would have to cover.

The frontier is dominated by one boundary — attribute access on a
statically-`PyValue` receiver — so the ranking counts those separately
from the code totals they hide inside. Codes stay ranked by corpus total;
the PyValue split is a cross-cutting view, not a replacement for the
histogram.

Needs the workdir the sweep ran in (for the Cargo JSONL events): the JSON
record carries histograms and E0308 pairs but not message text.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

MEASURED = {"built", "build-failed"}

# `no method named `m` found for enum `T` in the current scope`
METHOD = re.compile(
    r"no method named `([^`]+)` found for (?:enum|struct|mutable reference|tuple|type) `([^`]+)`")
# `no field `f` on type `T``
FIELD = re.compile(r"no field `([^`]+)` on (?:type|enum|struct) `([^`]+)`")
# `the trait bound `X` is not satisfied`
BOUND = re.compile(r"the trait bound `([^`]+)` is not satisfied")


def load_record(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data.get("packages"), dict):
        raise SystemExit(f"{path} is not a sweep result (missing packages)")
    return data


class MissingLog:
    """Sentinel for a measured package whose Cargo log is not in the workdir."""

    def __init__(self, package: str):
        self.package = package


def read_events(workdir: Path, packages: list[str]):
    """Yield error-level compiler messages for the GENERATED CRATE only.

    Mirrors `run_sweep.parse_build`: a diagnostic whose target is outside
    the crate (a dependency) belongs to `dependency_errors` in the record,
    never in its histogram, so counting it here would inflate every shape
    against a denominator that excluded it.
    """
    for name in packages:
        crate = workdir / f"crate-{name}"
        path = workdir / f"{name}-cargo.jsonl"
        if not path.is_file():
            yield MissingLog(name)  # the caller refuses a partial ranking
            continue
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("reason") != "compiler-message":
                continue
            message = event.get("message") or {}
            if message.get("level") != "error":
                continue
            target = (event.get("target") or {}).get("src_path")
            if target is not None and not Path(target).is_absolute():
                target = str(crate / target)
            if target is None or not Path(target).resolve().is_relative_to(crate.resolve()):
                continue
            yield message


def is_plain_pyvalue(receiver: str) -> bool:
    """True only for a bare PyValue receiver, not a wrapper mentioning one.

    `Option<PyValue>` and `Vec<PyValue>` fail at the WRAPPER: attribute
    access there is a different defect from an instance value that cannot
    carry attributes, so counting them as PyValue overstates the fix's
    reach. Reference sugar (`&mut PyValue`) is still a PyValue receiver.
    """
    bare = receiver.strip().replace("&mut ", "").replace("&", "").strip()
    return bare in {"PyValue", "stdpython::PyValue"}


def code_of(message: dict):
    return (message.get("code") or {}).get("code")


def site_file(message: dict):
    for span in message.get("spans") or []:
        if span.get("is_primary") and span.get("file_name"):
            return span["file_name"]
    return None
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("record", help="sweep results/run-*.json")
    ap.add_argument("--workdir", type=Path, default=Path("/tmp/rython-sweep"),
                    help="workdir the sweep ran in (holds <pkg>-cargo.jsonl)")
    ap.add_argument("--top", type=int, default=15, help="rows per section")
    args = ap.parse_args()

    record = load_record(Path(args.record))
    packages = {n: r for n, r in record["packages"].items() if r.get("status") in MEASURED}

    corpus = collections.Counter()
    for result in packages.values():
        for code, count in (result.get("histogram") or {}).items():
            corpus[code] += count
    total = sum(corpus.values())
    commit = str(record.get("rypip_commit", "?"))[:7]

    print(f"record : {args.record}")
    print(f"commit : {commit}   measured packages: {len(packages)}")
    print(f"total  : {total} E-coded errors\n")

    if not corpus:
        print("== corpus histogram: every measured package has zero E-coded errors ==")
        print("   Nothing to rank. Check that this record measures a frontier;")
        print("   a clean build and an empty histogram are not the same frontier.")
        return 1

    print("== corpus histogram (shared causes, per-package totals merged) ==")
    for code, count in corpus.most_common(args.top):
        print(f"{code:10} {count:6}  {100 * count / total:5.1f}%")
    print()

    methods = collections.Counter()
    fields = collections.Counter()
    shapes = collections.Counter()
    code_total = collections.Counter()
    code_pyvalue = collections.Counter()
    files = collections.Counter()
    missing = []

    for message in read_events(args.workdir, sorted(packages)):
        if isinstance(message, MissingLog):
            missing.append(message.package)
            continue
        code = code_of(message)
        text = message.get("message", "")
        if code:
            code_total[code] += 1
        if code == "E0599":
            found = METHOD.search(text)
            if found:
                on_pyvalue = is_plain_pyvalue(found.group(2))
                shapes["E0599 no method on PyValue" if on_pyvalue
                       else "E0599 no method (other receiver)"] += 1
                if on_pyvalue:
                    methods[found.group(1)] += 1
                    code_pyvalue[code] += 1
                    if site_file(message):
                        files[site_file(message)] += 1
        elif code == "E0609":
            found = FIELD.search(text)
            if found:
                on_pyvalue = is_plain_pyvalue(found.group(2))
                shapes["E0609 no field on PyValue" if on_pyvalue
                       else "E0609 no field (other receiver)"] += 1
                if on_pyvalue:
                    fields[found.group(1)] += 1
                    code_pyvalue[code] += 1
                    if site_file(message):
                        files[site_file(message)] += 1
        elif code == "E0277":
            found = BOUND.search(text)
            if found and "PyValue" in found.group(1):
                code_pyvalue[code] += 1

    if missing:
        # A partial log set would report shapes against a denominator that
        # covers every package, so the percentages would be quietly wrong.
        print("== NOT COMPARABLE: missing Cargo log(s) for measured packages ==")
        for name in missing:
            print(f"   {name}: no {name}-cargo.jsonl in {args.workdir}")
        print("   Refusing to rank a partial log set against the full corpus.")
        return 1

    if not shapes:
        print("== no E0599/E0609 events found in the generated crates ==")
        print(f"(looked for <package>-cargo.jsonl in {args.workdir})")
        return 1

    print("== E0599/E0609 message shapes ==")
    for shape, count in shapes.most_common():
        print(f"{count:6}  {shape}")
    # Everything else the crates reported, so the shapes never look like the
    # whole corpus. Ranked by code in the histogram above.
    print(f"{sum(code_total.values()) - sum(shapes.values()):6}  "
          "(outside E0599/E0609 — ranked by code above)")
    print()

    boundary = sum(methods.values()) + sum(fields.values())
    print(f"== the PyValue attribute boundary: {boundary} sites "
          f"({100 * boundary / total:.1f}% of {total}) ==")
    print("   A value held in a statically-PyValue slot loses its attributes:")
    print("   PyValue has no instance variant, so field/method access cannot")
    print("   resolve and the generated crate stops compiling at that line.")
    print()
    print(f"   top missing methods ({len(methods)} distinct):")
    for name, count in methods.most_common(args.top):
        print(f"     {count:5}  .{name}()")
    print(f"   top missing fields ({len(fields)} distinct):")
    for name, count in fields.most_common(args.top):
        print(f"     {count:5}  .{name}")
    print()

    print("== how much of each code is the PyValue boundary ==")
    for code, count in code_total.most_common(args.top):
        hit = code_pyvalue.get(code, 0)
        if hit:
            print(f"{code:10} {hit:5} of {count:5}  ({100 * hit / count:.0f}%)")
    print()

    print(f"== most-affected generated files ({len(files)} distinct) ==")
    for name, count in files.most_common(args.top):
        print(f"{count:5}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())