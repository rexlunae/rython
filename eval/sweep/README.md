# The issue #137 sweep

Error-count measurement for the frontier tracked on
[issue #137](https://github.com/rexlunae/rython/issues/137): convert pinned
real-world packages with rypip, build the generated crates, and tally the
rustc errors. Modeled on `eval/numpy/` (which has the full harness the sweep
lacked); the difference is that numpy's harness measures *correctness* per
case, while this measures the *error histogram* on full packages.

## The corpus

`packages.json` pins every package by exact version. rypip resolves the
dependency from PyPI into its own cache (`~/.cache/rypip`), so a sweep is
reproducible across containers — the version never depends on what happens
to be installed in site-packages (the historical trap: a fresh container
with urllib3 2.6.3 installed measured 1,638 where the rounds measure 2.0.7
= ~1,100).

## Usage

```sh
# Always rebuild the binaries first — a stale rypip silently measures old
# codegen (a trap the rounds hit twice).
cargo build -p python-ast -p rypip

# Baseline (e.g. main) and candidate (e.g. a feature branch) runs:
python3 eval/sweep/run_sweep.py --out results/run-main.json
python3 eval/sweep/run_sweep.py --out results/run-branch.json

# Diff them:
python3 eval/sweep/summarize.py results/run-main.json results/run-branch.json
```

`run_sweep.py` writes one JSON per run: per package, the error-code
histogram and the `expected X, found Y` pair breakdown for E0308 (the
dominant frontier class) — the same transmutation signal the rounds used to
hand-grep for.

## Traps (each cost a round or more before being recorded here)

- **Stale binary.** `rypip` on PATH may be an old install; always rebuild
  (`cargo build -p python-ast -p rypip`) and pass the fresh
  `target/debug/rypip`.
- **Redirect order.** The build log must be captured as `> log 2>&1`
  (stderr first). `2>&1 > file` sends stderr to the terminal and the file
  ends up empty — the "missing error" illusion.
- **The count line is not a site.** `error: could not compile ... due to N
  previous errors` is a summary, not an error site; count `^error[` lines.
- **A green codegen suite proves nothing about generated-crate behaviour.**
  The single-module codegen tests and the multi-module rypip conversion
  diverge on real packages; the sweep is the ground truth.
- **Parallel conversion runs share `~/.cache/rypip` and the workdir.**
  Use separate `--workdir` values for concurrent sweeps.

## CI ratchet (recommended, not yet wired)

The corpus is absent from CI. The cheap gate: rerun the urllib3 sweep in
CI and fail when the count *increases* (any regression — including the
transmutation class — is a count increase or a pair-shape change).
