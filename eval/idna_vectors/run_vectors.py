#!/usr/bin/env python3
"""idna's IdnaTestV2.txt conformance vectors, run under rython (issue #334).

idna's own UTS-46 test module loads its 6000+ vectors through unittest's
`load_tests` protocol. This runs the same vectors through an in-subset
port of its `runTest` body (vectors_template.py) instead:

  1. fetches the pinned idna sdist from PyPI (checked against its sha256)
     — the sdist, not the wheel, because the vectors and the upstream
     skip list live in its tests/;
  2. assembles a project: idna's package plus `idna/vectors.py`, the
     template with the upstream `_SKIP_TESTS` list spliced in, and
     IdnaTestV2.txt beside it;
  3. converts it with rypip, builds the crate, and runs the binary and
     `python -m idna.vectors` in the project directory;
  4. compares stdout and exit status byte for byte.

Usage:
    python3 eval/idna_vectors/run_vectors.py [--rypip target/debug/rypip]
        [--workdir /tmp/rython-idna-vectors] [--python python3]

Exit status 0 when the outputs are identical. Needs network access for
the first fetch; absent from CI, like the sweep.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
VERSION = "3.10"
SDIST_SHA256 = "12f65c9b470abda6dc35cf8e63cc574b1c52b11df2c86030af0ac09b01b13ea9"
MARKER = "@@SKIP@@"


def fetch_sdist(workdir: Path) -> Path:
    tarball = workdir / f"idna-{VERSION}.tar.gz"
    if not tarball.exists():
        meta = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/idna/{VERSION}/json"))
        url = next(u["url"] for u in meta["urls"] if u["packagetype"] == "sdist")
        tarball.write_bytes(urllib.request.urlopen(url).read())
    digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
    if digest != SDIST_SHA256:
        sys.exit(f"idna-{VERSION}.tar.gz: sha256 {digest}, expected {SDIST_SHA256}")
    src = workdir / f"idna-{VERSION}"
    if not src.exists():
        with tarfile.open(tarball) as tar:
            tar.extractall(workdir, filter="data")
    return src


def assemble(src: Path, project: Path) -> None:
    if project.exists():
        shutil.rmtree(project)
    project.mkdir(parents=True)
    shutil.copytree(src / "idna", project / "idna",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(src / "tests" / "IdnaTestV2.txt", project / "IdnaTestV2.txt")
    (project / "pyproject.toml").write_text(
        f'[project]\nname = "idna"\nversion = "{VERSION}"\n')
    upstream = (src / "tests" / "test_idna_uts46.py").read_text()
    skip = re.search(r"_SKIP_TESTS = \[.*?\n\]\n", upstream, re.S)
    if skip is None:
        sys.exit("tests/test_idna_uts46.py: no _SKIP_TESTS list")
    template = (HERE / "vectors_template.py").read_text()
    (project / "idna" / "vectors.py").write_text(
        template.replace(MARKER, skip.group(0).rstrip() + "\n"))


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rypip", default=str(REPO / "target" / "debug" / "rypip"))
    ap.add_argument("--workdir", default="/tmp/rython-idna-vectors")
    ap.add_argument("--python", default="python3", help="the CPython oracle")
    args = ap.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    project = workdir / "project"
    assemble(fetch_sdist(workdir), project)

    env = dict(os.environ, PYTHONHASHSEED="0")
    oracle = run([args.python, "-m", "idna.vectors"], project, env)
    crate = workdir / "crate"
    if crate.exists():
        shutil.rmtree(crate)
    convert = run([args.rypip, "convert", ".", "--out", str(crate)], project)
    if convert.returncode != 0:
        sys.exit(f"convert failed:\n{convert.stderr[-2000:]}")
    # The default rustc flags, as the idiom corpus builds (a CI-style
    # RUSTFLAGS=-D warnings would fail the generated crate's warnings).
    build_env = {k: v for k, v in os.environ.items() if "RUSTFLAGS" not in k}
    build = run(["cargo", "build", "--quiet"], crate, build_env)
    (crate / "build.log").write_text(build.stderr)
    if build.returncode != 0:
        errors = len(re.findall(r"^error\[E", build.stderr, re.M))
        sys.exit(f"build failed: {errors} rustc errors (see {crate / 'build.log'})")
    rython = run([str(crate / "target" / "debug" / "idna")], project, env)

    print(f"CPython ({args.python}): {oracle.stdout.strip().splitlines()[-1]}  exit {oracle.returncode}")
    print(f"rython:  {rython.stdout.strip().splitlines()[-1] if rython.stdout.strip() else rython.stderr[-300:]}  exit {rython.returncode}")
    if (oracle.stdout, oracle.returncode) == (rython.stdout, rython.returncode):
        print("identical")
        return
    sys.exit("outputs differ")


if __name__ == "__main__":
    main()
