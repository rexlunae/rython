# Running Python test suites under rython (issues #334 and #482)

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
  lowered; exit status matches CPython (0 on pass, 1 on failure). A
  `skipIf`/`skipUnless` condition that reads a module name rebound after
  its decorated class is refused, because CPython snapshots it at
  decoration time and the current runner evaluates it at test time (#521).
- **`with self.assertRaises(...): raise ...` works** since #457. The
  former build failure and its cause are recorded below as history.
- **`with self.assertRaises(E) as cm:` works** since #523: the matching
  exception is suppressed, and `cm.exception` can be read after the block.
  Uses of the context object other than `.exception` are refused. `E` must
  be statically known; dynamic attributes such as
  `self.failureException` are refused with a source-located error. Aliased
  imports of known stdlib exception classes (for example
  `import socket as s; s.timeout`) resolve to their canonical class.
- **There is no pytest runner or automatic collection.** A module with
  only `test_*` functions remains a library. An unresolved `import pytest`
  warns through the external-module channel (#456); attempts to use
  `pytest.main`, including imported aliases and runner values, fail
  conversion with a rewrite (#482).
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
correct posture for a module with no `unittest.TestCase` subclass
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

Test methods may create fields directly (`self.total = 3` followed by
`self.assertEqual(self.total, 3)`), including when the class has no
`__init__` or `setUp` (issue #480). The runner follows the MRO for
inherited tests, fixtures, default arguments and gates (issue #511),
constructing a fresh instance per test. A field read must follow an
assignment on every path, or an initialization on every successful
`setUp` exit. A different test's writes and calls to helpers establish no
initialization facts. A helper method may read what every successful
`setUp` exit assigned, unless `setUp` reaches it (directly or through
other helpers) — such a helper can run before those stores, so it must
initialize its own reads. `@staticmethod` and `@classmethod` members have
no instance receiver and are not checked. A class data attribute that an
instance field shadows (`value = 0` with `self.value = 5` in `setUp`) is
accepted; reads of it follow the same rule as any other field. Uncertain
reads, receiver escapes/captures, attribute deletion, a method shadowed by
a class value or an instance field, and custom constructors/attribute
hooks fail conversion with a line and rewrite, so construction defaults
never stand in for missing Python attributes. These restrictions apply to
a TestCase that has instance fields: one with none has no default to
expose. Manual construction of field-bearing cases (including
`t = T(); t.setUp(); t.test_a()`, which CPython runs) and explicit
lifecycle calls/captures are refused; use `unittest.main()` to preserve
fixture order. Setting a runner flag by hand (`__unittest_skip__`,
`__unittest_skip_why__` or `__unittest_expecting_failure__`, as a class
attribute or as `test_b.__unittest_skip__ = True`) is refused for every
TestCase; use the decorators. Native pins cover fixtures and errors against CPython
3.12 and 3.14.

The original unittest measurements used CPython 3.14 and rython `a055a18`;
the #457, #480 and #482 pins below update the affected rows:

| Construct | Converts | Builds | Runs | Matches CPython |
|---|---|---|---|---|
| `assertEqual` (scalar, computed) | yes | yes | yes | pass/fail outcome yes |
| `assertIn` (list) | yes | yes | yes | pass/fail outcome yes |
| `assertTrue` | yes | yes | yes | pass/fail outcome yes |
| `setUp` fixture | yes | yes | yes | yes |
| `with self.assertRaises(E): f()` | yes | yes | yes | yes |
| `with self.assertRaises(E): raise E(...)` | yes | yes | yes | yes (#457) |
| `with self.assertRaises(E) as cm: raise E(...); cm.exception` | yes | yes | yes | yes (#523) |
| `@unittest.skip` / `skipIf` / `skipUnless` | yes | yes | yes | yes for stable conditions; post-definition module rebinding is refused (#521) |
| `@expectedFailure` | yes | yes | yes | yes |
| Function-only `test_*` module | yes | yes (library) | no binary | no implicit collection, as with `python file.py` |
| External `pytest.main(...)` / imported alias | **conversion error** | — | — | no runner; run under CPython or port to unittest |

### The runner's output (Phase 3, done)

`unittest.main()` prints CPython's report on stderr, byte-for-byte except
for two things: the elapsed time (which varies in CPython too), and the
traceback frames. CPython, one passing and one failing test:

```
F.
======================================================================
FAIL: test_a (__main__.T.test_a)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/tmp/ut/f.py", line 6, in test_a
    self.assertEqual(1 + 1, 3)
    ~~~~~~~~~~~~~~~~^^^^^^^^^^
AssertionError: 2 != 3

----------------------------------------------------------------------
Ran 2 tests in 0.000s

FAILED (failures=1)
```

rython prints the same, without the three frame lines: it has no frames,
so a block keeps the header and the final exception line (the model limit
a thread's unhandled exception already has; `docs/spec.md` §12.3). The
runtime half is `stdpython::unittest::TextTestResult`: status characters
`.` `F` `E` `s` `x` `u`; `setUp`/`tearDown` exceptions reported against
the test as `TestCase.run` reports them; an `AssertionError` is a FAIL and
anything else an ERROR; a test's docstring's first line under its name;
the exit status 1 for an unsuccessful run, 5 when no test ran, 0
otherwise. `unittest.main(exit=False)` prints the report and returns to
the caller; any other argument (`verbosity=`, `argv=`, ...) is refused at
conversion rather than ignored. Each test's case drops before the next test
and before the exit, so a non-daemon thread it holds is joined, never cut
off. A clean run now reports `Ran N tests` and `OK`, so zero tests
and two passing tests are no longer indistinguishable. Pinned end to end
by `unittest_main_prints_cpythons_report` (convert_tests).

## Historical defect: `raise` as the last statement of `assertRaises`

Fixed by #457; the native regression is
`unittest_assert_raises_with_a_raise_builds_and_runs`. The original
reproducer and analysis follow.

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
warning: unreachable expression
  --> src/test_cm_raise.rs:32:13
error[E0308]: mismatched types
  --> src/test_cm_raise.rs:29:36
error: could not compile `test_cm_raise` (lib) due to 1 previous error; 1 warning emitted
```

Note where the error lands: on `src/test_cm_raise.rs:29:36`, the method's
signature, **not** inside the closure. The warning at `32:13` is the
closure's dead `Ok(())` — the visible symptom, not the cause.

### Root cause

The error is **not** inside the closure. `return Err(...); Ok(())` type
checks fine: the `Ok(())` is merely unreachable (a warning), and the
closure's return type is fixed by its annotation. The E0308 is on the
**enclosing method**, which is missing its own trailing `Ok(())`.

Two sites combine:

1. `crates/python-ast/src/ast/tree/with_stmt.rs:232` wraps the body in a
   closure:

   ```rust
   unittest::assert_raises(#exc_lit, || -> Result<(), PyException> {
       #(#body_tokens;)*
       Ok(())
   })?;
   ```

   Its guard, `body_escapes_a_closure` (`with_stmt.rs:53`), rejects bodies
   containing `return`/`break`/`continue` — a `raise` is not in that set,
   so the shape is routed into the closure.

2. `crates/python-ast/src/ast/tree/function_def.rs:4011` appends the
   method's trailing `Ok(())` only when `!guarantees_return(&self.body)`.
   `guarantees_return` (`function_def.rs:5943`) **unwraps a `with`
   statement and recurses into its body** (`function_def.rs:5949`), where a
   `raise` reports `true` (`function_def.rs:5955`, "`raise` lowers to
   `return Err(...)`, which terminates the path").

   For `with self.assertRaises(E): raise E(...)`, that reasoning is sound
   for the *closure* but wrong for the *method*: the `with` statement does
   not terminate the method, so the method falls off the end and needs its
   tail. The generated Rust is therefore:

   ```rust
   pub fn test_cm_raise(&self) -> Result<(), PyException> {
       unittest::assert_raises("ValueError", || -> Result<(), PyException> {
           return Err(PyException::new("ValueError", format!("{}", "boom")));
           Ok(())
       })?;
   }   // <- no Ok(()) tail: E0308 "expected Result<(), PyException>, found ()"
   ```

   Compare the working shape (a call in the body), which emits the tail
   `Ok(())` after the `assert_raises(...)?;`:

   ```rust
   unittest::assert_raises("ValueError", || -> Result<(), PyException> {
       might_fail(-1)?;
       Ok(())
   })?;
   Ok(())
   ```

The defect is therefore in `guarantees_return`'s `With` arm — or in that
call site's use of it — not in the closure's appended `Ok(())`. Removing
the closure's `Ok(())` alone would leave the E0308 in place (and drop the
`unreachable_code` warning, which is the only visible symptom at the
closure). A correct fix makes `guarantees_return` treat a `with` whose body
merely *raises* as not guaranteeing the enclosing return, or gives the
assertRaises lowering its own tail accounting.

### Why the old test suite did not catch it

`crates/python-ast/tests/codegen_semantics.rs:6275`
(`unittest_assert_raises_with_statement_lowers`) asserts only on the
**generated token text**:

```rust
assert!(out.contains("assert_raises (\"ValueError\"") /* … */);
```

Its body is `int("x")` — a *call*, not a `raise` — so it exercises the
shape that happens to work. Before #457, no test compiled the generated
crate for this construct,
so the failure was invisible to `cargo test --workspace`. The native
regression test added by #457 covers it now.

This is worth stating plainly: **a codegen-shape assertion is not a
correctness test.** AGENTS.md's "every behavior claim needs a
CPython-verified test" is satisfied by an end-to-end pin that builds and
runs; a `contains()` check on tokens is necessary but not sufficient, and
this defect is the evidence.

## pytest: libraries stay libraries; runner requests are refused

Consider issue #482's complete example:

```python
def test_add() -> None:
    assert 1 + 1 == 2


def test_sub() -> None:
    assert 2 - 1 == 0
```

`python3 app.py` exits 0 with empty stdout and stderr: defining the
functions does not call them. Rython converts it to a library containing
both functions, with no `src/main.rs`. `cargo build` succeeds; `cargo run`
exits 101 with `a bin target must be available`. Neither tool reports a
successful test run. Explicitly calling the functions still executes their
assertions; an ordinary function named `test_*` is never rejected or
implicitly collected.

Adding `import pytest` does not select an entry point. Since #456, an
unresolved import is dropped with an `external-module divergence` warning;
it no longer emits a phantom `use crate::pytest;`. That warning does not
provide pytest functionality.

A request to run pytest is different:

```python
import pytest as pt

if __name__ == "__main__":
    raise SystemExit(pt.main(["-q", "test_math.py"]))
```

There is no faithful pytest runner in the native runtime. Conversion
refuses `pytest.main(...)`, `pt.main(...)`, `from pytest import main`
followed by `main(...)`, and imported `as` spellings. References to the
runner as a value (`runner = pytest.main`) are refused too, as are calls
through crate re-exports of the imported runner. The public
`pytest.console_main()` and `pytest.cmdline.main()` entrypoints have the
same boundary. Using the pytest module or its cmdline namespace as a value
(compared, passed, stored or returned) is refused before a placeholder
could hide a subsequent runner call; use `import pytest as pt` for a
module alias. Member reads such as `pytest.mark.slow` and
`pytest.raises(...)` keep the external-module warning. The diagnostic includes
the Python location and asks the user to run `python -m pytest` under
CPython or port the suite to the supported unittest subset with an
explicit `unittest.main()` entry. This is a conversion error even with
`-W allow`. Parameters shadowing the import and real crate siblings named
`pytest` retain their own behavior.

Before the #482 guard, execution context mattered: a single-file runner
could emit unresolved Rust calls, whereas a multi-module package could
drop the call through the external-module warning path and produce a
binary with no test output and exit 0. That explicit execution request was
the silent gap; the function-only library was not.

The CPython oracle is reproducible in an isolated environment (the
measurement host's global packages are not a blocker):

```sh
python3 -m venv /tmp/rython-pytest-oracle
/tmp/rython-pytest-oracle/bin/python -m pip install pytest==8.4.2
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONHASHSEED=0 \
  /tmp/rython-pytest-oracle/bin/python -m pytest -q test_math.py
```

With the two functions above, pytest 8.4.2 reports one failure and one pass
and exits 1. Calling `pytest.main(...)` directly returns its status; a
script discarding that value still exits 0, but prints the test report.
`raise SystemExit(pytest.main(...))` propagates status 1. Native transcript
controls pin the library and explicit-function-call behavior against
CPython 3.12 and 3.14; conversion-error tests pin the runner boundary.

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

### Phase 1 — fix the `raise`-in-`assertRaises` defect (done, #457)

The following describes the former defect;
`unittest_assert_raises_with_a_raise_builds_and_runs` now pins the fix.

A one-shape bug with a known localization and a CPython-verified
reproducer. The fix is in the **enclosing-return** accounting, not in the
closure (see "Root cause" above):

- teach `guarantees_return` (`function_def.rs:5943`) that a `with` whose
  body only *raises* does not terminate the enclosing function — the
  `With` arm at `function_def.rs:5949` currently inherits the body's
  answer, which is right for `return` and wrong for `raise`; or
- special-case the assertRaises lowering so the method still gets its
  trailing `Ok(())`.

Removing the closure's appended `Ok(())` does **not** fix it — the E0308 is
on the method, not the closure. Forcing the shape through
`body_escapes_a_closure` (a loud refusal) would also work and is the
cheapest option if the accounting fix is judged too risky, but it loses a
legal program that CPython runs.

The pin must be **end-to-end** (convert, build, run, diff exit status
against `python3`), not a token check — the existing token pin
(`codegen_semantics.rs:6275`) is exactly why this shipped.

### Phase 2 — make explicit pytest execution loud (done, #482)

Keep imports and function definitions as module/library code. Refuse the
unsupported runner rather than silently drop a requested test run. See
the pytest section above for the guarded spellings and the CPython oracle.

Supporting pytest itself remains a separate, substantially larger project:
collection, fixtures, `conftest.py`, parametrization, plugins, and assertion
rewriting all need reference behavior. Installing pytest in an isolated
environment supplies an oracle; it does not supply a native runner.

### Phase 3 — match CPython's runner output (done)

Landed: see "The runner's output" above. The one remaining difference is
the traceback frames, recorded on the §12 ledger.

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

## Status of issues #334 and #482

Issue #334 closed after #476 landed the unittest report; #457 also fixed
the `raise`-in-`assertRaises` build failure. The general `TestCase` surface
still has limits described above and in `cpython-suite-status.md`.

Issue #482 carried over the pytest question. Its function-only example is
normal library behavior, not evidence of a successful native test run.
The actionable gap was an explicit pytest runner request being accepted
and dropped in a package; the conversion guard makes that request loud.
A full pytest runner remains unimplemented.
