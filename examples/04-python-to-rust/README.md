# From Python to Rust, step by step

This example takes an ordinary Python program and turns it into a Rust
crate you own. Everything is checked in so you can read the before and
after side by side:

- [`wordstats.py`](wordstats.py) — the input: classes, inheritance with
  `super()`, an overridden method, dicts, f-strings, and an unannotated
  helper whose types are inferred.
- [`generated/`](generated) — the actual crate `rypip convert` produced
  from it (unedited except for one path, noted below).

## 1. Start from working Python

```bash
python3 wordstats.py
```

```
words: 12 (9 distinct)
top: the x3
longest: quick
tweet-sized: True
```

Keeping the program running under CPython is the workflow: rython
converts a subset of Python, and anything outside it is a **loud
conversion error naming the construct** — you refactor the Python (still
running it under CPython), reconvert, and repeat. Nothing silently
changes behavior. See [`docs/porting-guide.md`](../../docs/porting-guide.md).

## 2. Convert

```bash
cargo run -p rypip -- convert wordstats.py --out generated
```

This wrote `generated/Cargo.toml` and `generated/src/` — a standalone
crate depending only on the `stdpython` runtime. (The checked-in copy's
only edit: `Cargo.toml`'s stdpython path is rewritten from the absolute
path rypip recorded to the repo-relative `../../../crates/stdpython`.)

## 3. Build and compare

```bash
cd generated
cargo build
./target/debug/wordstats
```

The output is **byte-identical** to the CPython run above — that
equivalence is the project's contract, enforced by transcript tests
(`crates/rypip/tests/convert_tests.rs`) that diff generated-binary
output against pinned `python3` runs.

## What to look at in the generated code

All in [`generated/src/wordstats.rs`](generated/src/wordstats.rs):

- **Classes → structs.** `class Tally:` becomes `pub struct Tally` with
  typed fields (`label: String`, `total: i64`) and an inherent `impl`;
  `Tally("words")` becomes `Tally::new("words")?`.
- **Inheritance → embedding + traits.** `class WordTally(Tally):`
  embeds a `Tally` and gets the base's methods through `TallyTrait`,
  whose accessor methods (`label()`, `total_mut()`, ...) route to the
  embedded struct. Overridden methods dispatch to the override even
  when called from base-class code, and `super().summary()` runs the
  base body with the derived `self` — CPython's MRO semantics, kept at
  runtime.
- **Inferred generics.** The unannotated `def within(value, low, high)`
  comes out as a generic function with exactly the bounds its body
  implies:

  ```rust
  pub fn within<A, B, C>(value: A, low: B, high: C) -> Result<bool, PyException>
  where
      A: PyLe<C, Output = bool>,
      A: Clone,
      B: PyLe<A, Output = bool>,
  ```

- **Exceptions → Result.** Every fallible function returns
  `Result<T, PyException>`; the `__main__` block becomes `fn main()`
  that prints the exception and exits 1, exactly like the interpreter.
- **Readable output.** The crate is rustfmt-formatted, warning-clean,
  and marked "Edit freely" — it is a starting point for a port, not an
  opaque artifact.

## Variations

```bash
# Build + run in one step (release profile), or install to ~/.cargo/bin:
cargo run -p rypip -- build wordstats.py
cargo run -p rypip -- install wordstats.py     # see ../05-rypip-install

# Wrap the crate in PyO3 bindings so Python can import the fast version:
cargo run -p rypip -- convert wordstats.py --out wordstats-py --pyo3

# Single file, no crate scaffolding:
cargo run -p rythonc -- wordstats.py -o wordstats.rs --pretty
```
