#!/usr/bin/env python3
"""Speed harness: the same numpy program under CPython and as a rython
binary, once per numpy backend.

Each bench program times its own kernels with time.perf_counter and prints
`kernel<TAB>n<TAB>reps<TAB>seconds_per_rep<TAB>checksum`, so the numbers
exclude interpreter/binary startup. Startup is measured separately.

Usage:
    python3 run_bench.py --workdir /tmp/npbench [--repeat 3]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = Path(__file__).resolve().parent / "bench"
RYPIP = ROOT / "target" / "release" / "rypip"
STDPYTHON = ROOT / "crates" / "stdpython"

FEATURES = ["numpy-rayon", "numpy-simd"]
# Backends are pinned by rewriting the program to call np.set_backend as
# main()'s first statement: the documented RYPY_NUMPY_BACKEND environment
# variable is never read by the runtime, so it cannot select an engine.
BACKENDS = ["scalar", "simd", "rayon"]


def inject_backend(src: str, backend: str) -> str:
    return src.replace("def main() -> None:\n",
                       f'def main() -> None:\n    np.set_backend("{backend}")\n', 1)


def run(cmd, cwd=None, env=None, timeout=3600):
    p = subprocess.run(cmd, cwd=cwd, env=env, timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr


def parse(out: str) -> dict:
    """kernel/n -> seconds_per_rep (plus the checksum, for a sanity diff)."""
    rows = {}
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 5 or parts[0] == "kernel":
            continue
        kernel, n, reps, secs, check = parts
        rows[(kernel, int(n))] = {"seconds": float(secs), "reps": int(reps),
                                  "checksum": check}
    return rows


def best(runs: list[dict]) -> dict:
    """Element-wise best (minimum) across repeats — the least-noise estimate."""
    keys = set()
    for r in runs:
        keys |= set(r)
    out = {}
    for k in keys:
        cands = [r[k] for r in runs if k in r]
        b = min(cands, key=lambda c: c["seconds"])
        out[k] = b
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="/tmp/npbench")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--programs", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    results = {}

    programs = sorted(BENCH.glob("bench_*.py"))
    if args.programs:
        programs = [p for p in programs
                    if any(p.name.startswith(q) or p.stem == q for q in args.programs)]

    for prog in programs:
        name = prog.stem
        print(f"=== {name}", flush=True)
        entry = {"cpython": None, "rython": {}}

        runs = []
        for _ in range(args.repeat):
            rc, out, err = run([sys.executable, str(prog)])
            if rc != 0:
                print(f"  cpython FAILED: {err[-500:]}")
                break
            runs.append(parse(out))
        if runs:
            entry["cpython"] = {f"{k[0]}|{k[1]}": v for k, v in best(runs).items()}
            print(f"  cpython: {len(entry['cpython'])} kernels")

        env = dict(os.environ)
        env["CARGO_TARGET_DIR"] = str(work / "target")

        for backend in BACKENDS:
            src = work / f"{name}.{backend}.py"
            src.write_text(inject_backend(prog.read_text(), backend))
            crate = work / f"{name}.{backend}"
            if crate.exists():
                shutil.rmtree(crate)
            rc, out, err = run([str(RYPIP), "convert", str(src), "--out", str(crate),
                                "--stdpython", str(STDPYTHON)])
            if rc != 0:
                entry["rython"][backend] = {"error": "CONVERT_FAIL: " + err[-1500:]}
                print(f"  {backend}: CONVERT_FAIL")
                continue

            toml = crate / "Cargo.toml"
            feats = ", ".join(f'"{f}"' for f in FEATURES)
            toml.write_text(re.sub(
                r'^stdpython = \{ path = "([^"]+)" \}$',
                lambda m: f'stdpython = {{ path = "{m.group(1)}", features = [{feats}] }}',
                toml.read_text(), flags=re.M))

            rc, out, err = run(["cargo", "build", "--release"], cwd=crate, env=env)
            if rc != 0:
                entry["rython"][backend] = {"error": "BUILD_FAIL: " + err[-2500:]}
                print(f"  {backend}: BUILD_FAIL")
                continue

            pkg = re.search(r'^name = "([^"]+)"', toml.read_text(), re.M).group(1)
            binary = Path(env["CARGO_TARGET_DIR"]) / "release" / pkg

            runs = []
            failed = None
            for _ in range(args.repeat):
                rc, out, err = run([str(binary)])
                if rc != 0:
                    failed = err[-500:]
                    break
                runs.append(parse(out))
            if failed:
                entry["rython"][backend] = {"error": failed}
                print(f"  {backend}: RUN_FAIL")
                continue
            entry["rython"][backend] = {f"{k[0]}|{k[1]}": v
                                        for k, v in best(runs).items()}
            print(f"  {backend}: {len(entry['rython'][backend])} kernels")

        results[name] = entry

    # Process startup: an empty numpy program vs an empty rython binary.
    startup = work / "startup"
    startup_py = work / "startup.py"
    startup_py.write_text("import numpy as np\n\n\ndef main() -> None:\n"
                          "    print(np.sum(np.zeros(1)))\n\n\n"
                          'if __name__ == "__main__":\n    main()\n')
    if startup.exists():
        shutil.rmtree(startup)
    rc, out, err = run([str(RYPIP), "convert", str(startup_py), "--out", str(startup),
                        "--stdpython", str(STDPYTHON)])
    if rc == 0:
        env = dict(os.environ)
        env["CARGO_TARGET_DIR"] = str(work / "target")
        run(["cargo", "build", "--release"], cwd=startup, env=env)
        pkg = re.search(r'^name = "([^"]+)"',
                        (startup / "Cargo.toml").read_text(), re.M).group(1)
        binary = Path(env["CARGO_TARGET_DIR"]) / "release" / pkg

        def wall(cmd, env=None):
            ts = []
            for _ in range(5):
                t0 = time.perf_counter()
                run(cmd, env=env)
                ts.append(time.perf_counter() - t0)
            return min(ts)
        results["_startup"] = {
            "cpython_seconds": wall([sys.executable, str(startup_py)]),
            "rython_seconds": wall([str(binary)], env=env),
        }
        print(f"=== startup: cpython {results['_startup']['cpython_seconds']:.4f}s "
              f"rython {results['_startup']['rython_seconds']:.4f}s")

    out_path = Path(args.out) if args.out else work / "bench.json"
    out_path.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
