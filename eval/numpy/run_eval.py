#!/usr/bin/env python3
"""Correctness harness: diff rython's numpy against CPython's, per backend.

For every case in cases/ this:

  1. runs it under CPython + real numpy and records stdout/stderr/exit code;
  2. converts it with `rypip convert`, builds the generated crate, and runs
     the native binary;
  3. classifies the result: PASS (byte-identical stdout and exit code),
     DIVERGE (ran, different output), CONVERT_FAIL (loud conversion error),
     BUILD_FAIL (generated Rust does not compile), or RUN_FAIL.

Backends are selected by rewriting the case source to call
`np.set_backend("<name>")` as the first statement of main() — that is the
only selection path that is actually wired: the documented
`RYPY_NUMPY_BACKEND` environment variable is never read by the runtime
(engine.rs documents it, nothing consumes it), so setting it silently
leaves the engine on `auto`.

CONVERT_FAIL is rython's documented contract for anything outside the
subset; DIVERGE and BUILD_FAIL are the interesting failures.

Usage:
    python3 run_eval.py --workdir /tmp/npeval [--cases 01 07 ...]
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
CASES = Path(__file__).resolve().parent / "cases"
RYPIP = ROOT / "target" / "release" / "rypip"
STDPYTHON = ROOT / "crates" / "stdpython"

# variant name -> (stdpython features, injected np.set_backend argument)
VARIANTS = {
    # The default build: only the always-on sequential engine, engine left
    # on `auto` (which resolves to scalar here).
    "default-auto": ([], None),
    # Everything that runs on this machine's CPU, pinned per run.
    "cpu-scalar": (["numpy-rayon", "numpy-simd"], "scalar"),
    "cpu-rayon": (["numpy-rayon", "numpy-simd"], "rayon"),
    "cpu-simd": (["numpy-rayon", "numpy-simd"], "simd"),
    # rayon+simd compiled in, engine left on auto (resolves to rayon).
    "cpu-auto": (["numpy-rayon", "numpy-simd"], None),
}


def run(cmd, cwd=None, env=None, timeout=1800):
    p = subprocess.run(cmd, cwd=cwd, env=env, timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr


def inject_backend(src: str, backend: str) -> str:
    """Pin the engine by adding set_backend as main()'s first statement."""
    return src.replace("def main() -> None:\n",
                       f'def main() -> None:\n    np.set_backend("{backend}")\n', 1)


def set_features(crate: Path, features: list[str]) -> None:
    if not features:
        return
    toml = crate / "Cargo.toml"
    feats = ", ".join(f'"{f}"' for f in features)
    toml.write_text(re.sub(
        r'^stdpython = \{ path = "([^"]+)" \}$',
        lambda m: f'stdpython = {{ path = "{m.group(1)}", features = [{feats}] }}',
        toml.read_text(), flags=re.M))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="/tmp/npeval")
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    variants = {k: v for k, v in VARIANTS.items()
                if not args.variants or k in args.variants}

    case_files = sorted(CASES.glob("*.py"))
    if args.cases:
        case_files = [c for c in case_files
                      if any(c.name.startswith(p) for p in args.cases)]

    results = []
    for case in case_files:
        name = case.stem
        rec = {"case": name, "variants": {}}
        print(f"=== {name}", flush=True)

        t0 = time.perf_counter()
        rc, out, err = run([sys.executable, str(case)])
        rec["cpython"] = {"rc": rc, "stdout": out, "stderr": err,
                          "seconds": time.perf_counter() - t0}
        if rc != 0:
            print(f"    cpython rc={rc} (the case itself is invalid numpy)")

        for vname, (features, backend) in variants.items():
            src = work / f"{name}.{vname}.py"
            text = case.read_text()
            src.write_text(inject_backend(text, backend) if backend else text)

            crate = work / f"{name}.{vname}"
            if crate.exists():
                shutil.rmtree(crate)
            rc, cout, cerr = run([str(RYPIP), "convert", str(src), "--out", str(crate),
                                  "--stdpython", str(STDPYTHON)])
            v = {}
            if rc != 0:
                v["status"] = "CONVERT_FAIL"
                v["convert_stderr"] = cerr[-3000:]
                rec["variants"][vname] = v
                print(f"    [{vname}] CONVERT_FAIL")
                continue
            set_features(crate, features)

            env = dict(os.environ)
            env["CARGO_TARGET_DIR"] = str(work / f"target-{'-'.join(features) or 'default'}")
            t0 = time.perf_counter()
            rc, bout, berr = run(["cargo", "build", "--release"], cwd=crate, env=env)
            v["build_seconds"] = time.perf_counter() - t0
            if rc != 0:
                v["status"] = "BUILD_FAIL"
                v["build_stderr"] = berr[-6000:]
                rec["variants"][vname] = v
                print(f"    [{vname}] BUILD_FAIL")
                continue

            pkg = re.search(r'^name = "([^"]+)"',
                            (crate / "Cargo.toml").read_text(), re.M).group(1)
            binary = Path(env["CARGO_TARGET_DIR"]) / "release" / pkg
            t0 = time.perf_counter()
            rrc, rout, rerr = run([str(binary)])
            v.update({"rc": rrc, "stdout": rout, "stderr": rerr,
                      "seconds": time.perf_counter() - t0})
            v["status"] = ("PASS" if rrc == rec["cpython"]["rc"]
                           and rout == rec["cpython"]["stdout"]
                           else ("RUN_FAIL" if rrc != 0 else "DIVERGE"))
            rec["variants"][vname] = v
            print(f"    [{vname}] {v['status']}")

        rec["status"] = ("PASS" if all(v.get("status") == "PASS"
                                       for v in rec["variants"].values())
                         else "FAIL")
        results.append(rec)

    out_path = Path(args.out) if args.out else work / "results.json"
    out_path.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out_path}")
    npass = sum(1 for r in results if r["status"] == "PASS")
    print(f"{npass}/{len(results)} cases byte-identical to CPython on every backend")
    return 0


if __name__ == "__main__":
    sys.exit(main())
