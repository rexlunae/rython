# The idiom corpus

Small idiomatic Python programs that must **convert, build, run, and print
exactly what CPython prints**. This is the complement of the corpus sweep in
[`eval/sweep/`](../sweep/): the sweep counts rustc errors on real PyPI
packages, which is the right frontier metric for issue #137 but is blind to
two things —

1. **Generality.** The sweep's shapes are urllib3's shapes. A round that
   pins the `return`-guard form of `if x is None:` and never sees a `raise`
   guard reports progress while ordinary code still fails. These programs
   are the shapes a Python programmer writes.
2. **Silent divergence.** A crate that compiles and prints the wrong thing
   is invisible to an error count. Every program here is written so its
   *state* is observable in the output — a total printed after a mutation,
   not just the mutation's return value — so a copy-instead-of-alias
   lowering shows up as a diff, not a pass.

## Usage

```sh
cargo build -p python-ast -p rypip          # a stale rypip measures stale codegen
python3 eval/idioms/run_idioms.py            # everything; prints a status table
python3 eval/idioms/run_idioms.py --only inventory tree
python3 eval/idioms/run_idioms.py --check-baseline    # what CI runs
python3 eval/idioms/run_idioms.py --update-baseline   # in the PR that improves a program
python3 eval/sweep/run_sweep.py --with-idioms         # embed the pass count in a sweep run
```

A result file is named by the **converter's source commit** (the last
commit touching `crates/python-ast`, `crates/rypip`, or `crates/stdpython`),
which is what `target/debug/rypip` was built from — and the runner
**refuses a binary older than the newest converter source file** (pass
`--allow-stale` to override), so the name is trustworthy because the run
could not otherwise have happened. The payload also records `repo_head`
and the oracle's version; it carries no machine-local paths or timestamps,
so a committed result file changes only when a measurement does. A branch that only edits `eval/`
therefore measures, and is named by, the same converter as its merge-base.

Generated crates are built with the default rustc flags whatever the
caller's `RUSTFLAGS` says (CI's `-D warnings` would count every warning as
an error and the ratchet would compare counts taken under different rules).
Per-program status is one of `convert-failed`, `build-failed` (with the
rustc error histogram), `run-failed`, `timeout` (a hang in any stage is that
program's result, not the corpus's), `output-mismatch` (with the first
differing line, on either stream), or `pass`. Each run writes `results/run-<commit>.json`;
failing crates are left in the workdir (`/tmp/rython-idioms/crate-<name>`)
with a `build.log` for diagnosis.

## The ratchet

`baseline.json` records every program's **stage** (`convert-failed` <
`build-failed` < `run-failed` < `output-mismatch` < `pass`) and, for a build
failure, its rustc **error count**. `--check-baseline` exits non-zero when a
program is worse than recorded: a lower stage, or more errors at the same
build-failed stage. Programs that have never passed are the frontier, not a
regression, so CI stays green while the corpus is mostly red — but the
frontier cannot retreat silently either: a 3-error program becoming an
11-error one, or a building program becoming a conversion refusal, fails
CI until the PR that does it records the new state with `--update-baseline
--force-baseline`, which makes the trade visible where it is reviewed.
Improvements are reported, never failed; bump the baseline in the same PR
— that is the claim the PR is making, recorded where CI can hold it.

A result that is not a measurement of the generated code — a missing or
stale pin, a missing binary, a cargo failure before rustc ran (dependency
resolution, a manifest error: no `could not compile` in the log) — is
**unmeasured**: `--check-baseline` fails on it and `--update-baseline`
refuses to record it, forced or not. Otherwise a harness failure would read
as an improvement over any recorded error count.

## Adding a program

- 30–100 lines, one file, `if __name__ == "__main__": main()`, no imports
  beyond the stdlib, deterministic output (no dict-order tricks beyond
  insertion order, no timing, no randomness).
- Write it **before** the fix it is meant to exercise, in the shape a
  programmer would naturally write — not the shape that happens to lower.
- Make state observable: after every mutation the program relies on, print
  something that would differ if the mutation were lost.
- Capture the pin from CPython — `python3 programs/NAME.py > programs/NAME.expected`
  — and commit both. The runner re-derives the pin from `python3` on every
  run and refuses to measure against a stale one, so an edit to the program
  cannot silently redefine "correct".

## What's here

| program | shapes |
|---|---|
| `inventory` | classes with `Optional` fields, inheritance + `super()`, a dict of objects, exceptions, get-then-mutate (aliasing) |
| `text_stats` | string methods, dict counting, sort by compound key, formatted columns |
| `shapes` | polymorphic dispatch through a base-typed list, `isinstance`, `__repr__`, `max`/`sorted` with key |
| `bank` | custom exception hierarchy, `try/except/else/finally`, mutation through a registry |
| `matrix` | nested lists, comprehensions over ranges, `zip(*)`, tuple unpacking, in-place nested mutation |
| `tree` | recursive class with `Optional` children, a generator with `yield from`, recursion |
| `tokenizer` | `while` scanner with slicing and char predicates, list of tuples, stack evaluator |
| `pipeline` | closures, functions as values, `map`/`filter`/`lambda`, a dispatch table, `*args`/keyword args |
| `records` | `@property` with setter, `@classmethod`/`@staticmethod`, `__eq__`/`__lt__`, `__str__` |
| `schedule` | dict of lists, tuple sorting, `list.remove`/`index`, `del` of a key, sets |
