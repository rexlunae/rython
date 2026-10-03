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

## Build outcomes and comparison safety

Schema version 2 uses Cargo's JSON diagnostics and records a `status` for
each package:

| Status | Meaning | `total` |
|---|---|---|
| `built` | Cargo completed successfully with no error diagnostics | 0 |
| `build-failed` | Cargo completed a diagnosed compilation failure in the generated crate | Historical E-code count, possibly 0 |
| `convert-failed` | rypip exited unsuccessfully | null |
| `unmeasured` | Timeout, missing tool/dependency, dependency compilation failure, or incomplete Cargo output | null |

`total` and `histogram` still count only `E####` diagnostics. `uncoded_total`
and `uncoded_errors` additionally count `compile_error!` refusals, denied
lints (whose JSON codes are names rather than E-numbers), and other errors
without an E-code. `diagnostic_total` is their sum. Cargo's summary lines
are not diagnostic sites, and warnings are not counted as errors.

A completed sweep exits **0 even when generated crates fail to compile**:
this is a failure-measurement tool. Any conversion or infrastructure failure
makes the measurement incomplete and exits **1**, after writing the results
for the other packages. Preflight argument/stale-binary errors exit **2**.
Check each package's `status` before claiming that it builds. `--timeout`
sets the conversion/build time limit in seconds (default 1800); a timeout
kills that command's process group, including compiler children.

`summarize.py` shows build states and both kinds of error count. If a package
is absent, its pin changed, or either measurement failed, it exits nonzero
and prints only a comparable-package subtotal, **not a whole-corpus delta**.
Legacy results remain usable for coded-error comparisons when their build
statuses support the measurement. Their uncoded counts are **unknown**, not
zero. A failed legacy build with zero coded errors is ambiguous and cannot
be compared as an improvement.

The workdir retains `<package>-convert.log` (including conversion warnings),
`<package>-cargo.jsonl` (Cargo events and stderr), and a readable
`<package>-build.log`. Generated builds discard inherited `RUSTFLAGS`,
`CARGO_ENCODED_RUSTFLAGS`, and `CARGO_TARGET_DIR` so warning policy and shared
target directories do not distort the measurement. Use separate workdirs
for separate agents.

## Provenance and limits

Python 3.11 or newer is required. The runner records the last source-changing
commit as `rypip_commit`, separately from `repo_head`, plus source content
hash/dirty state, the binary's SHA-256, and Rust/Python versions. The source
snapshot includes tracked and untracked files under `crates`, Cargo manifests
and lockfile, the Rust toolchain file, and `.cargo`. The timestamp preflight uses Cargo’s adjacent `rypip.d` input list: it
requires that this checkout’s `rypip/src/main.rs` is listed and rejects
missing or newer compiler inputs. Runtime-only changes do not falsely require
relinking rypip. Supply the locally built executable with its `.d` file. This
is a freshness sanity check, not cryptographic build attestation; always
rebuild here and supply that binary. Source/binary changes during a run
invalidate the overall comparison while preserving its recorded diagnostics.

This slice measures **generated-crate compilation**, not conversion coverage
or runtime correctness. A successful conversion may still omit unsupported
modules or operations. Module/source coverage manifests, transitive dependency
pinning, and executed-assertion accounting remain follow-ups under #137; a
`built` result alone must not be presented as Python compatibility.

The accounting tests are fast and run in CI without downloading the corpus:

```sh
python3 -m unittest discover -s eval/sweep -p 'test_*.py'
```

## Recorded baseline at `28efebd`

[`run-28efebd-accounting.json`](results/run-28efebd-accounting.json) measures
converter commit `28efebd` (main, after #429) with schema 2:

| Package | E-coded errors | Other error diagnostics | Build |
|---|---:|---:|---|
| urllib3 | 605 | 54 | failed |
| certifi | 0 | 0 | built |
| idna | 0 | 0 | built |
| charset_normalizer | 41 | 3 | failed |
| requests | 2127 | 205 | failed |
| **Total** | **2773** | **262** | **2/5 built** |

Against the [`24d1987`](results/run-main-24d1987.json) record (a legacy schema
with unknown uncoded counts, so `summarize.py` reports a coded-only delta):
urllib3 650 -> 605, charset_normalizer 42 -> 41, requests 2249 -> 2127, certifi
and idna unchanged at 0 — **-168 coded errors** overall. `idna` builds at this
commit; it last failed with 44 errors in earlier records. This is a
measurement of existing compiler code, not an improvement made by the
accounting change itself. Its matching [idiom
record](../idioms/results/run-28efebd.json) is **48/49 passing** with
`--check-baseline` reporting `baseline holds: 49 program(s)`; the single
failure is `schedule`, an intentional conversion refusal. Neither
measurement establishes coverage of omitted modules or unexecuted
operations.

Recorded baselines are per-commit, not per-corpus: a record is only a valid
comparison point when its provenance matches the candidate's checkout and
binary. `summarize.py` will refuse a whole-corpus delta rather than compare
runs it cannot account for.
