# Agent instructions for the Rython repository

Instructions for AI coding agents (and welcome context for humans)
working **on** this repository, or **with** it to port Python code.

## If your task is porting Python code to Rust

Follow [`docs/porting-guide.md`](docs/porting-guide.md). Summary of the
contract you're working under:

- Rython converts a **subset** of Python (defined in
  [`docs/spec.md`](docs/spec.md)). Conversion errors are the workflow,
  not failures: refactor the *Python* into the subset (keeping it
  running under CPython at every step), reconvert, and only edit the
  generated Rust after the output diffs clean against CPython.
- Verify with `PYTHONHASHSEED=0`, seeded randomness, and byte-level
  diffs of stdout/stderr/exit codes.
- A silent output difference with no conversion error is a rython bug —
  file it with a minimal reproducer; never absorb it into the port.

## If your task is changing this repository

### The prime directive

**Correct or loud — never silently different.** Every construct either
tracks CPython's observable behavior (pinned byte-for-byte by
transcript tests wherever verified, with known differences on the
ledger in `docs/spec.md` §12), fails conversion with an error naming
the construct and the fix, or raises a typed exception (or, where
unrepresentable, panics) at the exact point of divergence. Read [`docs/goals-and-design.md`](docs/goals-and-design.md)
before adding features; its checklist ("How to evaluate a proposed
feature") is the review bar. Never "improve" a behavior away from
CPython's, even when CPython's looks wrong — including exception
message text, ordering, and float formatting.

### Layout

| Crate | Role |
|---|---|
| `crates/python-ast` | Parser (CPython `ast` via PyO3) + Python→Rust codegen |
| `crates/stdpython` | Runtime: builtins, types, stdlib, exceptions; `core ⊂ alloc ⊂ std` feature tiers |
| `crates/rypip` | Package converter/builder/installer; pyo3/no_std/kernel targets; FFI manifests |
| `crates/rythonc` | Single-file CLI compiler |
| `crates/python-mod` | `python_module!` proc-macros |
| `crates/rykernel-shim` | C-free runtime for kernel modules (unpublished, path-resolved) |

### Working rules

- **Review consumption is a merge precondition** (issue #137's R1): every
  review-bot thread ends in either a responsive commit with a pin test, or
  a written waiver citing a `docs/spec.md` §12 entry — then the thread is
  resolved. Never merge with an unresolved thread; a shipped
  wrong-semantics finding (the #260 bool-fold reversal) is the cost of
  treating review as advisory.
- **Fetch the remote head before proposing or starting anything**
  (issue #137's process rule 1): with multiple agents landing PRs, a stale
  local `origin/main` is the default state. `git fetch origin` + check
  `origin/main` first, and re-measure against the real remote head before
  claiming a delta or building on a base.
- **Every behavior claim needs a CPython-verified test.** End-to-end
  tests (`crates/rypip/tests/convert_tests.rs`) pin expected output as
  literals captured from real `python3` runs — mark them
  `// Verified against python3.` and actually run python3 to produce
  them, never write the expectation from memory. Runtime pins go in
  `crates/stdpython/tests/python_semantics.rs` with the Python
  expression in a comment; codegen-shape and loud-error tests go in
  `crates/python-ast/tests/codegen_semantics.rs` (`compile_err` for
  rejections).
- **New unsupported edges must be loud.** When your feature has a case
  you can't reproduce exactly, emit a conversion error naming the
  construct, location, and rewrite (see the message style in existing
  errors: "…is not supported yet…; rython refuses to silently ignore
  it"). Add a `compile_err` test for it.
- **Generated code must stay readable** and warning-clean under the
  default lints: run the mutability/hoisting analyses rather than
  sprinkling `#[allow]`.
- **Parse strings into enums at the boundary; never scatter string
  lists.** AST identifiers arrive as strings, so ONE `match` on the
  string is unavoidable — but it happens exactly once, in a typed
  enum's `from_name` (see `ThreadingType` in
  `python-ast/src/ast/tree/threading_types.rs`); every other consumer
  (classifiers, lowerings, registries) works with the enum. Duplicated
  stringly-typed name lists that can drift out of sync are a defect,
  not a style choice. When adding a compiler-known runtime surface,
  give its name set an enum first.
- **Respect the tiers.** Anything OS-touching is `std`-gated in
  stdpython; alloc-tier additions must build with
  `--no-default-features --features alloc` (CI cross-checks
  `thumbv7m-none-eabi`). Conversion-time guards for missing tiers
  belong in the converter, not as rustc errors in generated crates.
- **Check the ledger.** Known divergences live in `docs/spec.md` §12
  and issue #82; if your change fixes one, update both. If you discover
  a new silent divergence, file it immediately — that class outranks
  whatever you were doing.
- Run `cargo test --workspace --all-targets` (plus the alloc-tier
  stdpython tests when touching the runtime) before pushing. CI runs
  exactly those.
- Docs to keep in sync when the surface changes: root `README.md`
  (compatibility list), `docs/spec.md`, and the relevant crate README.
  Some crate READMEs predate the current design — when in doubt, the
  root README and `docs/` are authoritative.

### Documentation map

- [`docs/goals-and-design.md`](docs/goals-and-design.md) — goals,
  non-goals, design principles, feature-evaluation checklist
- [`docs/spec.md`](docs/spec.md) — the language specification and the
  deviation ledger
- [`docs/cpython-vs-rython.md`](docs/cpython-vs-rython.md) — model
  comparison and tradeoffs
- [`docs/porting-guide.md`](docs/porting-guide.md) — the porting
  workflow
- [`docs/context-awareness.md`](docs/context-awareness.md) — type
  inference/coercion internals

### The corpus sweep (issue #137)

- The frontier metric is the **rustc error count on pinned real packages**,
  measured with `eval/sweep/` (modeled on `eval/numpy/`): `run_sweep.py`
  converts the pinned corpus (`eval/sweep/packages.json` — urllib3 2.0.7
  by default, plus certifi and idna), builds each generated crate, and
  writes a JSON error histogram. `summarize.py` diffs two runs into the
  per-code and per-`expected X, found Y` delta. The corpus is **absent
  from CI** — a round must run the sweep itself and report the delta.
- The **idiom corpus** (`eval/idioms/`) is the sweep's complement: small
  idiomatic programs (the shapes a Python programmer writes, not urllib3's)
  that must convert, build, and print exactly what CPython prints.
  `run_idioms.py` reports per-program status; `baseline.json` records every
  program's stage and, for a build failure, its rustc error count, and CI
  enforces it as a ratchet (`--check-baseline` fails when any program gets
  worse: a lower stage, or more errors — the frontier is allowed to be red
  but not to retreat). A round that claims a shape fixed adds the program that
  exercises it **before** the fix, written so state is observable in the
  output (a printed total after a mutation), and bumps the baseline in the
  same PR. Report the idiom pass count next to the sweep delta.
- **Traps** (each cost a round before being recorded):
  - Rebuild the binaries first (`cargo build -p python-ast -p rypip`) and
    use `target/debug/rypip` — a stale `rypip` on PATH silently measures
    old codegen.
  - Capture build logs as `> log 2>&1` (stderr first); the reverse order
    sends stderr to the terminal and leaves an empty file.
  - Count `^error[E` lines; the `could not compile ... due to N previous
    errors` line is a summary, not a site.
  - A green single-module codegen suite proves nothing about
    multi-module conversion; the sweep is the ground truth.
  - Parallel rounds sharing one base produce deltas that do not compose —
    rebase-and-remeasure before claiming a delta.
