#!/usr/bin/env python3
"""Measure generated-crate build failures on the pinned issue #137 corpus.

Rebuild python-ast and rypip first. Coded Rust errors remain the historical
`total` metric; uncoded errors and build status are recorded separately.
Conversion/infrastructure failures are unmeasured, never zero-error builds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import signal
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
SOURCE_PATHS = ["crates", "Cargo.toml", "Cargo.lock", "rust-toolchain.toml", ".cargo"]
E0308_PAIR = re.compile(r"expected `([^`]+)`, found `([^`]+)`")
MEASURED = {"built", "build-failed"}


def run(cmd, cwd=None, timeout=30):
    return subprocess.run(cmd, cwd=cwd, timeout=timeout, check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def source_snapshot(root: Path = ROOT) -> dict:
    def git(*args):
        return run(["git", *args], cwd=root).stdout
    names = git("ls-files", "-z", "--cached", "--others", "--exclude-standard",
                "--", *SOURCE_PATHS).split("\0")
    digest = hashlib.sha256()
    for name in sorted(set(names) - {""}):
        path = root / name
        digest.update(name.encode() + b"\0")
        if path.is_file():
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        else:
            digest.update(b"missing")
    return {
        "source_commit": git("log", "-1", "--format=%H", "--", *SOURCE_PATHS).strip(),
        "source_sha256": digest.hexdigest(),
        "source_dirty": bool(git("status", "--porcelain", "--untracked-files=all",
                                 "--", *SOURCE_PATHS).strip()),
    }


def binary_digest(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def cargo_config_digest(root: Path = ROOT) -> str:
    """sha256 of the repo's Cargo config, or "" when there is none.

    Cargo takes its build target from `CARGO_BUILD_TARGET` **and** from
    `build.target` in `.cargo/config.toml`, and `cargo config get` is
    nightly-only — so the environment variable alone records "" for a run
    that is in fact cross-compiling. Fingerprinting the config file catches
    that without parsing it: two records whose configs differ are flagged,
    and the flag does not claim to know which target was selected.
    """
    for name in ("config.toml", "config"):
        path = root / ".cargo" / name
        if path.is_file():
            with path.open("rb") as f:
                return hashlib.file_digest(f, "sha256").hexdigest()
    return ""


def check_binary_inputs(rypip: Path, root: Path = ROOT) -> None:
    """Use Cargo's input list, not runtime/tests that aren't linked into rypip.

    This is a build freshness sanity check, not cryptographic attestation.
    Cargo emits the adjacent .d file for a locally built executable.
    """
    depfile = rypip.with_suffix(".d")
    try:
        rule = depfile.read_text().splitlines()[0]
        _, inputs = rule.split(": ", 1)
        paths = [Path(p).resolve() for p in shlex.split(inputs)]
        expected = (root / "crates/rypip/src/main.rs").resolve()
        if expected not in paths:
            raise ValueError("binary was not built from this checkout")
        built = rypip.stat().st_mtime_ns
        if any(not p.is_file() or p.stat().st_mtime_ns > built for p in paths):
            raise ValueError("stale rypip: a compiler input is missing or newer than the binary")
    except (OSError, IndexError, ValueError) as exc:
        raise ValueError(f"{exc}; rebuild here with cargo build -p python-ast -p rypip and supply its executable with the adjacent .d file") from exc


def run_logged(cmd, log: Path, *, cwd=None, timeout=1800, env=None):
    """Kill the whole subprocess group on timeout, including cargo's rustc children."""
    with log.open("w") as out:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=out, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
        try:
            return proc.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            raise


def write_probe(pkg_root: Path, name: str, requirement: str, pins=()) -> None:
    """Write a one-dependency project that imports the package under test.

    `pins` are corpus requirements listed AHEAD of `requirement`: rypip
    resolves each listed dependency's tree in order and the first version
    of an import name wins, so a pin fixes the version a transitive
    dependency resolves to. Without it, rypip picks the newest PyPI release
    satisfying the dependent's specifier (or whichever one happens to be
    cached), and the measurement drifts with PyPI and the cache.
    """
    pkg_root.mkdir(parents=True, exist_ok=True)
    deps = ", ".join(f'"{r}"' for r in [*pins, requirement])
    (pkg_root / "pyproject.toml").write_text(
        f'[project]\nname = "probe"\nversion = "0.1.0"\n'
        f'dependencies = [{deps}]\n'
    )
    pkg_dir = pkg_root / name.replace("-", "_")
    pkg_dir.mkdir(exist_ok=True)
    (pkg_dir / "__init__.py").write_text(f"import {name.replace('-', '_')}\n")


def parse_build(lines, crate: Path, build_status: int, rendered_log) -> dict:
    """Count compiler diagnostics for this crate, not Cargo summaries or dependencies.

    Cargo JSON supplies the diagnostic code, level, and compilation target.
    A complete compiler failure is a measurement; a missing build-finished
    event, dependency failure, or other process failure is not.
    """
    histogram, pairs, uncoded = {}, {}, {}
    dependency_errors = 0
    finished = []
    crate = crate.resolve()
    for line in lines:
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            rendered_log.write(line)
            continue
        if not isinstance(event, dict):
            rendered_log.write(line)
            continue
        if event.get("reason") == "build-finished":
            finished.append(event.get("success"))
        if event.get("reason") != "compiler-message":
            continue
        message = event.get("message", {})
        rendered = message.get("rendered") or message.get("message", "") + "\n"
        rendered_log.write(rendered)
        if message.get("level") != "error":
            continue
        src = event.get("target", {}).get("src_path")
        target = Path(src) if src else None
        if target is not None and not target.is_absolute():
            target = crate / target
        if target is None or not target.resolve().is_relative_to(crate):
            dependency_errors += 1
            continue
        code = (message.get("code") or {}).get("code")
        # Denied lints also have a JSON `code`, but it is a lint name,
        # not a historical E-code. Keep those in the uncoded count.
        if code and re.fullmatch(r"E[0-9]{4}", code):
            histogram[code] = histogram.get(code, 0) + 1
        else:
            text = message.get("message", "unknown compiler error")
            if code:
                text = f"{code}: {text}"
            uncoded[text] = uncoded.get(text, 0) + 1
        if code == "E0308":
            for m in E0308_PAIR.finditer(rendered):
                key = f"{m.group(1)} | {m.group(2)}"
                pairs[key] = pairs.get(key, 0) + 1
    coded_total = sum(histogram.values())
    uncoded_total = sum(uncoded.values())
    diagnostics = coded_total + uncoded_total
    reason = None
    if dependency_errors:
        reason = f"{dependency_errors} compiler error(s) in dependencies"
    elif build_status == 0 and finished == [True] and diagnostics == 0:
        status = "built"
    elif build_status == 101 and finished == [False] and diagnostics > 0:
        status = "build-failed"
    else:
        reason = "Cargo did not finish a successful build or a diagnosed generated-crate compile failure"
    if reason:
        status = "unmeasured"
    return {
        "status": status,
        "build_status": build_status,
        "total": coded_total if status in MEASURED else None,
        "uncoded_total": uncoded_total if status in MEASURED else None,
        "diagnostic_total": diagnostics if status in MEASURED else None,
        "histogram": dict(sorted(histogram.items(), key=lambda kv: (-kv[1], kv[0]))),
        "uncoded_errors": dict(sorted(uncoded.items(), key=lambda kv: (-kv[1], kv[0]))),
        "e0308_pairs": dict(sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0]))),
        "dependency_errors": dependency_errors,
        **({"error": reason} if reason else {}),
    }


def dependency_pins(spec: dict, corpus: list) -> list:
    """The corpus requirements `spec["pin_dependencies"]` names, in order.

    A name outside the corpus is an error, not a silently unpinned dependency.
    """
    by_name = {s["name"]: s["requirement"] for s in corpus}
    names = spec.get("pin_dependencies", [])
    unknown = [n for n in names if n not in by_name or n == spec["name"]]
    if unknown:
        raise ValueError(f"{spec['name']}: pin_dependencies names {', '.join(unknown)}, "
                         "which must be other packages in packages.json")
    return [by_name[n] for n in names]


def sweep_one(spec: dict, rypip: Path, workdir: Path, timeout=1800, pins=()) -> dict:
    name = spec["name"]
    crate = workdir / f"crate-{name}"
    probe = workdir / f"probe-{name}"
    result = {"package": name, "requirement": spec["requirement"], "total": None}
    if pins:
        result["dependency_pins"] = list(pins)
    write_probe(probe, name, spec["requirement"], pins)
    phase = "convert"
    try:
        convert_log = workdir / f"{name}-convert.log"
        convert_status = run_logged(
            [str(rypip), "convert", "--out", str(crate), str(probe)],
            convert_log, timeout=timeout,
        )
        result["convert_status"] = convert_status
        if convert_status != 0:
            return {**result, "status": "convert-failed",
                    "error": f"conversion exited {convert_status}; see {convert_log.name}"}
        phase = "build"
        raw_log = workdir / f"{name}-cargo.jsonl"
        env = os.environ.copy()
        for key in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_TARGET_DIR"):
            env.pop(key, None)
        env["CARGO_TERM_COLOR"] = "never"
        build_status = run_logged(
            ["cargo", "build", "--message-format=json"], raw_log,
            cwd=crate, timeout=timeout, env=env,
        )
        with raw_log.open(errors="replace") as lines, (workdir / f"{name}-build.log").open("w") as rendered:
            result.update(parse_build(lines, crate, build_status, rendered))
        return result
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {**result, "status": "unmeasured", "phase": phase, "error": str(exc)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="results/run-*.json path")
    ap.add_argument("--rypip", type=Path, default=DEFAULT_RYPIP)
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--package", action="append", default=None)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=1800, help="seconds per conversion/build")
    ap.add_argument("--with-idioms", action="store_true")
    args = ap.parse_args()
    if args.jobs < 1 or args.timeout <= 0:
        ap.error("--jobs and --timeout must be positive")
    specs = json.loads((Path(__file__).resolve().parent / "packages.json").read_text())["packages"]
    try:
        pins = {s["name"]: dependency_pins(s, specs) for s in specs}
    except ValueError as exc:
        ap.error(str(exc))
    if args.package:
        unknown = set(args.package) - {s["name"] for s in specs}
        if unknown:
            ap.error(f"unknown packages: {', '.join(sorted(unknown))}")
        specs = [s for s in specs if s["name"] in args.package]
    if not specs:
        ap.error("no packages selected")
    rypip = args.rypip.resolve()
    if not rypip.is_file():
        ap.error(f"rypip not found at {rypip}; build python-ast and rypip first")
    snapshot = source_snapshot()
    repo_head = run(["git", "rev-parse", "HEAD"], cwd=ROOT).stdout.strip()
    rustc = run(["rustc", "-Vv"]).stdout.strip()
    try:
        check_binary_inputs(rypip)
    except ValueError as exc:
        ap.error(str(exc))
    fingerprint = binary_digest(rypip)
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futures = {ex.submit(sweep_one, s, rypip, workdir, args.timeout, pins[s["name"]]): s for s in specs}
        for fut, spec in futures.items():
            try:
                results[spec["name"]] = fut.result()
            except Exception as exc:  # retain the other packages' measurements
                results[spec["name"]] = {**spec, "status": "unmeasured", "total": None, "error": str(exc)}
    payload = {
        "schema_version": 2,
        "rypip_commit": snapshot["source_commit"],
        "repo_head": repo_head,
        "provenance": snapshot,
        "rypip_path": str(rypip),
        "rypip_sha256": fingerprint,
        "rustc": rustc,
        "build_target": os.environ.get("CARGO_BUILD_TARGET", ""),
        "cargo_config_sha256": cargo_config_digest(),
        "python": platform.python_version(),
        "packages": results,
    }
    incomplete = any(r["status"] not in MEASURED for r in results.values())
    if args.with_idioms:
        idioms_json = workdir / "idioms.json"
        idioms_json.unlink(missing_ok=True)
        p = subprocess.run(
            [sys.executable, str(ROOT / "eval" / "idioms" / "run_idioms.py"),
             "--rypip", str(rypip), "--workdir", str(workdir / "idioms"), "--out", str(idioms_json)],
            check=False,
        )
        if p.returncode != 0 or not idioms_json.is_file():
            payload["idioms"] = {"error": f"idiom measurement failed (exit {p.returncode})"}
            incomplete = True
        else:
            idioms = json.loads(idioms_json.read_text())
            payload["idioms"] = {k: idioms[k] for k in ("passed", "total", "passing")}
    try:
        changed = source_snapshot()["source_sha256"] != snapshot["source_sha256"] or binary_digest(rypip) != fingerprint
        if changed:
            payload["measurement_error"] = "converter sources or binary changed during this run"
    except (OSError, subprocess.SubprocessError) as exc:
        payload["measurement_error"] = f"could not verify final source/binary snapshot: {exc}"
    if payload.get("measurement_error"):
        incomplete = True
    payload["measurement_complete"] = not incomplete
    payload["elapsed_seconds"] = round(time.time() - started, 1)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else RESULTS / f"run-{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    for name, result in results.items():
        counts = (f"{result['total']} coded + {result['uncoded_total']} uncoded errors"
                  if result["status"] in MEASURED else result.get("error", "no measurement"))
        print(f"{name:20} {result['status']:15} {counts}")
    if "idioms" in payload:
        print(f"idioms: {payload['idioms']}")
    if payload.get("measurement_error"):
        print(payload["measurement_error"], file=sys.stderr)
    print(f"wrote {out} ({'INCOMPLETE' if incomplete else 'measurement complete; inspect build statuses'})")
    return 1 if incomplete else 0


if __name__ == "__main__":
    sys.exit(main())
