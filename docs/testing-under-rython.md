# Running Python test suites under rython (issue #334)

This document records what actually works today for running a Python
library's test suite through rython, what does not, the exact code sites
that decide it, and a phased plan. It supersedes the *status* (not the
analysis) in [`cpython-suite-status.md`](cpython-suite-status.md), which is
still the per-file record for the 17 CPython stdlib test files.

Prime directive and feature bar: [`goals-and-design.md`](goals-and-design.md).
Deviation ledger: [`spec.md`](spec.md) §12.

## Short version

- **`unittest` suites work today, for a bounded subset.** A `unittest`
  module converts, builds, and runs; assertions, fixtures and gates are
  lowered; exit status matches CPython (0 on pass, 1 on failure).
- **One common construct is broken:** `with self.assertRaises(...)` whose
  body ends in a `raise` statement converts silently and then fails to
  build. Reproducer and root cause below.
- **`pytest` is not supported at all** — not partially, not with warnings.
  A pytest module converts "successfully", then fails to build on a
  phantom `use crate::pytest;`. Zero occurrences of the string `pytest`
  exist in `crates/`.
- **Callables-as-data is implemented**, which is what the issue's original
  text says is missing. The issue's own text is stale on this point.

## What the issue says, and what is stale

Issue #334 was filed against the warning:

```
warning: tests/test_intranges.py: call to `self.assertEqual(...)` is dropped:
the member is neither a method nor a field of `IntrangeTests`
(the member is an unmodeled callable value — the callable-as-value divergence)
```

The issue's suggested direction — "a callable representation unblocks test
frameworks, argparse coercion, and dispatch tables together" — **has
landed** (PR #345 and successors). Three parts of the issue are now stale:

1. **Callables are values.** `stdpython::PyCallable<A, R>`
   (`crates/stdpython/src/callable.rs:36`) is the runtime type behind a
   `Callable[...]` annotation, a lambda, a nested `def`, and a module
   function used as a value. A call through one is `f.call((x,))`, returning
   the same `Result` every generated function returns, so an exception
   raised inside a callable propagates and is catchable at the caller.
   `PyCallable` is reference-counted over a `Fn + Send + Sync`, which is
   what lets a module-level registry live in a Rust `static`.

2. **Errors on assertions are no longer dropped.** `self.assertEqual` and
   friends are lowered to runtime helpers, not dropped:
   `crates/python-ast/src/ast/tree/call.rs:10903-10960` recognizes twelve
   assertion methods and calls `lower_unittest_assert`
   (`crates/python-ast/src/ast/tree/call.rs:447`), which renders
   `unittest::assert_eq(&a, &b, msg)?` against the runtime in
   `crates/stdpython/src/stdlib/unittest.rs:53`.

3. **A test runner is emitted.** `unittest.main()` in the
   `if __name__ == "__main__"` block lowers to a real runner
   (`crates/python-ast/src/ast/tree/module.rs:6711` `emit_test_runner`,
   dispatched from `module.rs:1452`).

`crates/stdpython/src/stdlib/unittest.rs:544` still holds the fallback
`unittest::main()`, which raises a loud `NotImplementedError` — the
correct posture for a module with no direct `unittest.TestCase` subclass
(`emit_test_runner` returns `None` there, by design; see its doc comment).

## What works today (verified)

The following program converts with **zero warnings**, builds clean, and its
exit status matches CPython:

```python
import unittest


class AlphaTests(unittest.TestCase):
    def test_one(self) -> None:
        self.assertEqual(1 + 1, 2)

    def test_two(self) -> None:
        self.assertIn("b", ["a", "b"])


if __name__ == "__main__":
    unittest.main()
```

`rypip convert` → `cargo build` → run: exit 0, matching `python3` exit 0.
Adding a failing test (`self.assertEqual(1, 2)`) makes both exit 1.

Verified against CPython (3.14) and rython `rypip` built from `a055a18`:

| Construct | Converts | Builds | Runs | Matches CPython |
|---|---|---|---|---|
| `assertEqual` (scalar, computed) | yes | yes | yes | pass/fail outcome yes |
| `assertIn` (list) | yes | yes | yes | pass/fail outcome yes |
| `assertTrue` | yes | yes | yes | pass/fail outcome yes |
| `setUp` fixture | yes | yes | yes | yes |
| `with self.assertRaises(E): f()` | yes | yes | yes | yes |
| `with self.assertRaises(E): raise E(...)` | yes | **no** | — | **broken** |
| `@unittest.skip` / `skipIf` / `skipUnless` | yes | yes | yes | yes (gates honoured) |
| `@expectedFailure` | yes | yes | yes | yes |
| pytest-style module | "yes" | **no** | — | **no support** |

### The runner's output is not CPython's

This is an observable divergence that no test pins, and it should be
recorded rather than glossed.

CPython, all tests passing:

```
..
----------------------------------------------------------------------
Ran 2 tests in 0.001s

OK
```

rython, same program: **prints nothing**, exit 0.

CPython, one failure:

```
...F
======================================================================
FAIL: test_fail (__main__.MoreTests.test_fail)
...
Ran 4 tests in 0.002s

FAILED (failures=1)
```

rython, same program:

```
FAIL: AssertionError: 1 != 2
AssertionError: 1 test(s) failed
```

The **pass/fail outcome and exit status agree**, which is the property that
matters for CI gating. The *report format* does not: no per-test dots, no
`Ran N tests` line, no `FAILED (failures=1)` summary, and notably **no
output at all on a clean run** — a suite that ran zero tests and a suite
that ran two passing tests are indistinguishable from the outside.

`docs/cpython-suite-status.md` describes the runner as "counting failures
and … raising `AssertionError: <n> test(s) failed`" — accurate. It does not
claim CPython's format, and this document makes that gap explicit. It is
**not** currently on the §12 ledger; it should be, since a program whose
*output* is part of its contract (goal 2, `P2`) diverges here.

## The defect: `raise` as the last statement of `assertRaises`

### Reproducer

```python
import unittest


class CMTests(unittest.TestCase):
    def test_cm_raise(self) -> None:
        with self.assertRaises(ValueError):
            raise ValueError("boom")


if __name__ == "__main__":
    unittest.main()
```

- `python3`: passes, exit 0 (`Ran 1 test`, `OK`).
- `rypip convert`: succeeds, **no warning**.
- `cargo build` on the generated crate: **fails**, exit 101.

```
error[E0308]: mismatched types
  --> src/test_cm_raise.rs:51:34
   |
51 |     pub fn test_raises(&self) -> Result<(), PyException> {
   |            -----------           ^^^^^^^^^^^^^^^^^^^^^^^
   |            |                     expected `Result<(), PyException>`, found `()`
   = note: expected enum `Result<(), PyException>`
              found unit type `()`
```

plus a `warning: unreachable expression` on the `Ok(())` that follows.

### Root cause

`crates/python-ast/src/ast/tree/with_stmt.rs:232` lowers the context-manager
form by wrapping the body in a closure and appending `Ok(())`:

```rust
unittest::assert_raises(#exc_lit, || -> Result<(), PyException> {
    #(#body_tokens;)*
    Ok(())
})?;
```

When the body's last statement is an unconditional `raise`, the lowering
emits `return Err(...);` and then the appended `Ok(())` is dead code. The
closure therefore has two problems: unreachable code (a warning) and,
because the `return` makes the tail expression unreachable in a way the type
checker reports, an inferred `()` tail (the error).

The guard above that lowering, `body_escapes_a_closure`
(`with_stmt.rs:53`), only rejects bodies containing `return`/`break`/
`continue`. A `raise` is not in that set, so the shape is routed into the
closure anyway.

### Why the test suite does not catch it

`crates/python-ast/tests/codegen_semantics.rs:6275`
(`unittest_assert_raises_with_statement_lowers`) asserts only on the
**generated token text**:

```rust
assert!(out.contains("assert_raises (\"ValueError\"") /* … */);
```

Its body is `int("x")` — a *call*, not a `raise` — so it exercises the
shape that happens to work. No test in the suite compiles the generated
crate for this construct, so the failure is invisible to
`cargo test --workspace`.

This is worth stating plainly: **a codegen-shape assertion is not a
correctness test.** AGENTS.md's "every behavior claim needs a
CPython-verified test" is satisfied by an end-to-end pin that builds and
runs; a `contains()` check on tokens is necessary but not sufficient, and
this defect is the evidence.

## `pytest` has no support

A pytest-style module:

```python
import pytest


def test_add():
    assert 1 + 1 == 2
```

`rypip convert` reports success and exits 0. The generated module then
fails to build:

```
error[E0432]: unresolved import `crate::pytest`
   | use crate::pytest;
   |     no `pytest` in the root
```

and **no runner is emitted** — there is no `main.rs`, and the module's
`test_*` functions are never called. The converter treats `pytest` as a
crate-sibling import (the documented `--no-deps` contract,
`crates/python-ast/src/ast/tree/import.rs`) rather than as an unsupported
framework.

This violates the "new unsupported edges must be loud" rule: the program
converts without a warning and produces a crate that does not build. It
should be a conversion error naming the construct.

Confirmed by search: `pytest` appears **0 times** in `crates/`. The
`StdModule` enum (`crates/python-ast/src/ast/tree/std_module.rs:106`) has
`unittest` and no `pytest`.

## The real remaining gap for `unittest`: bound methods through unmodeled bases

The issue's narrowing comment (2026-09-08) says the wall is
"`self.<method>` lookup through an unmodeled base class". That is still true
for any framework whose helpers are **not** in the hardcoded assertion list.
The assertion methods work because they are special-cased; a `self.<anything
else>` inherited from an external base still hits the callable-as-value drop
at `crates/python-ast/src/ast/tree/call.rs:10985-11010`.

So the honest statement is: **the twelve assertion methods and the
`setUp`/`tearDown`/gate protocol are modeled; the general `TestCase` surface
is not.** `assertIs`, `assertRaisesRegex`, `assertMultiLineEqual`, and
`assertLogs` are named as still-open in `docs/cpython-suite-status.md`.

## Phased plan

Ordered by cost-to-value, each phase independently landable under the
prime directive.

### Phase 1 — fix the `raise`-in-`assertRaises` defect (small, do first)

A one-shape bug with a known localization and a CPython-verified
reproducer. Either:

- add `raise` to the statements `body_escapes_a_closure` refuses, so the
  shape falls through to the plain with-lowering and its existing loud
  behaviour; or
- better, thread the raise correctly: a `raise` inside the closure is the
  *expected* outcome for `assertRaises`, and `assert_raises` already
  handles `Err(e) if e.matches(expected)`. The lowering needs the closure to
  *not* append `Ok(())` when the body cannot fall through.

The second is correct and keeps the shape supported; the first is loud but
loses a legal program. Either way, the pin must be **end-to-end** (convert,
build, run, diff exit status against `python3`), not a token check.

### Phase 2 — make `pytest` loud, and decide whether to support it

Distinguish "unmodeled framework" from "crate sibling import". A bare
`import pytest` / `from pytest import …` with no crate sibling should be a
conversion error naming the construct and the rewrite (port the tests to
`unittest` under rython, or run them under CPython).

Supporting pytest properly is a substantially larger project: fixtures,
`conftest.py`, parametrization, `assert` rewriting. Note that rython does
**not** need `assert` rewriting — it has no `assert` in the generated
language to enhance — but it does need fixtures and collection. Whether that
is in scope is a product decision, not an implementation one; this document
does not recommend it.

### Phase 3 — match CPython's runner output

Emit the per-test status characters (`.`/`F`/`E`/`s`), the
`Ran N tests in T.TTTs` line, the `OK`/`FAILED (...)` summary, and — most
importantly — *something* on a clean run. This makes suite output
comparable byte-for-byte, which is what P2 asks for, and removes the
"zero tests ran" ambiguity.

Record the current divergence on the §12 ledger until this lands.

### Phase 4 — the vector-file path (framework-free coverage)

The issue proposes this and it is still the cheapest route to *behavioral*
coverage: idna ships `IdnaTestV2.txt` and its library modules
(`intranges.py`, `codec.py`, `uts46data.py`) already convert and build with
0 errors (idna is one of the two packages at 0 on the corpus sweep —
see [`frontier.md`](frontier.md)). A small driver that reads the vectors and
asserts against the library's converted functions needs no framework at all.
This is real coverage with no framework work.

### Phase 5 — broaden the framework surface

Only after the above: the remaining assertion methods, transitive
`TestCase` subclasses, `load_tests`. `docs/cpython-suite-status.md` has the
per-file wall for all 17 CPython stdlib test files and is the better
running record for this phase.

## Status of issue #334

**Open, and correctly so — but its premise is stale.** The stated cause
(callables as values) is implemented. What blocks the subject is a
different, narrower list:

1. the `raise`-in-`assertRaises` defect (Phase 1);
2. pytest converting silently into a non-building crate (Phase 2);
3. the runner's non-CPython output format (Phase 3);
4. the general `TestCase` surface beyond twelve hardcoded assertions.

It cannot be closed today, but it should be **rewritten**: the callable-value
framing is done, and keeping it makes the issue read as more unstarted than
it is.
