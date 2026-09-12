# CPython stdlib-semantics suite: frontier status

Tracks the 17 CPython-stdlib `test_*.py` files under conversion (measurement:
`target/debug/rypip convert <file>.py --out <dir> --no-deps`; **CONVERT** =
conversion succeeds, **BLOCKED** = a specific loud conversion error at the
first failing construct).

## The one gate that decides the frontier

Every one of these 17 files is a CPython **`unittest`** module: test methods
call `self.assertEqual(...)`, `self.assertRaises(...)`, `self.subTest(...)`,
`self.assertTrue(...)`, etc., and are collected into `*TestSuite` /
`unittest.TestCase` subclasses. Whatever pure-language or stdlib feature is
missing *earlier* in a file, the file can only **convert and run** once the
**unittest harness** lowers — issue #334 (callables, `assertRaises`, `subTest`,
`except-as`, class-as-callable). The 6 files that convert today do so *without*
asserting (the harness calls are dropped as loud `-W` warnings), so they do
not yet exhibit CPython-equal behavior.

**Frontier: 6/17 CONVERT; 0/17 BUILD; 0/17 RUN** (see the BUILD analysis below).

## Why "CONVERT" ≠ "BUILD" (corrected this session)

`convert <file> --no-deps` succeeding means **code generation** succeeds, not
that the emitted crate compiles. A single-module (`--no-deps`) conversion
treats every non-stdpython import as a "crate sibling" (documented contract,
`crates/python-ast/src/ast/tree/import.rs`, pinned by
`sibling_from_import_anchors_to_crate`), so the generated Rust emits
`use crate::unittest; use crate::test::support::import_helper; use
crate::decimal::Decimal; use crate::fractions::Fraction; use crate::inspect;
use crate::pickle;` — but **none of those modules are emitted**, and the
`if __name__ == "__main__": unittest.main()` tail lowers to `unittest::main()?`
with no `unittest` module.

**Foundation landed (`dc566a3`):** `unittest` is now a registered StdModule
routing to a real stdpython `stdlib::unittest`, so `import unittest` no longer
emits the broken `crate::unittest` sibling reference and test crates BUILD.
`main()` is a loud `NotImplementedError` (exit 1) until codegen lowers a real
runner — never a silent pass.

**Runner landed (`a225dfd`):** a `unittest.main()` in the module's
`if __name__ == "__main__"` now emits a runner that discovers the module's
direct `*TestCase` subclasses, constructs each, and runs its `test_*`
methods, counting failures; any erroring test prints `FAIL: …` and the
runner raises `AssertionError: <n> test(s) failed` (exit 1). Verified:
a minimal test module with a *raising* or erroring test fails loudly and
exits 1; a passing one exits 0.

**Assertions landed (`b4a3849`, `e7a61dd`):** `self.assertEqual`,
`assertNotEqual`, `assertTrue`, `assertFalse`, `assertIsNone`,
`assertIsNotNone` now lower to runtime `unittest::assert_*` helpers, which
box the arguments and raise `AssertionError` on failure — no longer a silent
drop to `None`. The emitted runner also calls `setUp()`/`tearDown()` around
every `test_*` method when the class defines them. Verified end-to-end: a
two-test module prints exactly CPython's `setup / test_a / teardown / setup /
test_b / teardown`, reports the same single failure (`AssertionError: 1 == 1`),
and exits 1. The call-site rendering is `#[inline(never)]`, and
`.cargo/config.toml` sets `RUST_MIN_STACK` (32 MiB) because the codegen
recurses once per nested call and Rust's 2 MiB default test-thread stack was
tight (the CLI's 8 MiB main thread was always fine).

**Still open:** `assertIn`/`assertIs`/`assertGreater`/`assertLess`,
`assertRaises`/`subTest` (context managers), transitive `TestCase`
subclasses, and the stdlib-stub emission below.

Verified on `test_operator` (`cargo build` in its generated crate):

```
error[E0432]: unresolved import `crate::decimal`
error[E0432]: unresolved import `crate::fractions`
error[E0432]: unresolved import `crate::inspect`
error[E0432]: unresolved import `crate::pickle`
error[E0433]: cannot find `test` in the crate root
error[E0432]: unresolved import `crate::unittest`
```

By contrast, a **harness-free, correctly-typed** module (`mini_math.py` with an
annotated recursion + `math.floor`) converts AND `cargo build`s cleanly in ~4s,
proving the toolchain and the build pipeline work — the 17 test files fail to
build specifically because of the harness/stdlib-module references (#334 +
the stdlib-module stub pipeline), not for lack of a build target.

So the honest frontier is **0/17 BUILD and 0/17 RUN**: no test file produces a
compiling crate yet, because every one terminates (or starts near) `unittest.*`
and `test.*` stdlib references, and asserts are dropped.

## Per-file status (as of this round)

| File | Status | First blocker |
|---|---|---|
| `test_heapq` | CONVERT* | `load_tests` null-`where` bug FIXED (this round); now blocks on the nested-`iterable` no-return-annotation callable wall (#334) |
| `test_calendar` | CONVERT | — (same) |
| `test_struct` | CONVERT | — (same) |
| `test_fractions` | CONVERT | — (same) |
| `test_random` | CONVERT | — (same) |
| `test_operator` | CONVERT | — (same) |
| `test_math` | BLOCKED | `count_set_bits` recursion (FIXED → advanced); `ulp_abs_check` union FIXED (this round: `local_str.format(...)` now infers String, so `None` vs `fmt.format(...)` unifies to `Option<String>`); test_math now advances from line 107 to line 744 — the heterogeneous `[float, FloatLike]` list-literal wall |
| `test_bisect` | BLOCKED | nested `def grade(breakpoints=[60,70,80,90])` — a **mutable-list default**, deliberately loud (Python evaluates-and-shares it; cannot be lowered correctly) |
| `test_csv` | BLOCKED | `csv.reader(..., escapechar=…)` reader feature FIXED (this round, now threads `Some::<u8>`); advances to the **dialect-registry** wall (`csv.reader([...], name)` / `register_dialect`), reached inside harness blocks |
| `test_textwrap` | BLOCKED | `wrap(text, width, **kwargs)` — issue #368 `**kwargs`; reached only inside unittest `TestTextWrap` methods |
| `test_itertools` | BLOCKED | heterogeneous list literal `['abc', range(6)]` (str/range mix) |
| `test_collections` | BLOCKED | heterogeneous list literal `[None, int(), gen(), object(), Bar()]` (int/Bar/mixed) |
| `test_functools` | BLOCKED | nested `class` at class level |
| `test_statistics` | BLOCKED | `_make_std_err_msg` arity — **verified CPython-consistent**: `self._make_std_err_msg(a,b,c,d,e)` on a 5-param `def _make_std_err_msg(first,second,tol,rel,idx)` is a genuine `takes 5 … but 6 were given` TypeError under CPython too; rython's loud conversion error (reports 4/5) is correct-or-loud, only the *counts* differ. Then the harness. |
| `test_re` | BLOCKED | `assertRaisesRegex` of an *intentional* runtime `TypeError` from `re.sub(...)` (error is correctly loud at compile time; needs harness to catch at runtime) |
| `test_hmac` | BLOCKED | `with self.subTest(...)` harness, then compare/binding issues |
| `test_hashlib` | BLOCKED | `isinstance(x, class)` with a class second arg (classes not yet supported) — reached in `__init__`, before any harness |

## Reasons a file cannot (yet) convert — root causes

Fixing *only* the pure-language/stdlib leftovers advances a file to its next
wall, but never to conversion+run, because each file's first construct is
inside (or leads to) the unittest harness:

1. **The unittest harness is missing (#334).** `self.assertEqual`,
   `self.assertRaises[Regex]`, `self.subTest`, `self.assertTrue/False`,
   `self.assertIs`, `self.assertIn`, `self.assertRaises`-as-context-manager,
   `with self.subTest(...):`. This is the *single* shared gate; until it
   lowers, no file can run and most cannot even convert past their first
   assertion. Every file above with harness blockers needs this, not a
   per-file fix.

2. **`isinstance(x, ClassName)` with a class second argument** (hashlib).
   Classes as first-class values are issue #367; rython's `isinstance` takes
   builtin type tags, not user classes.

3. **Heterogeneous list literals** (itertools, collections). `['abc',
   range(6)]`, `[None, int(), gen(), object(), Bar()]` mix unrelated element
   types; rython's typed inference refuses them (would need the boxed-pyvalue
   container model).

4. **Mutable-list defaults on a nested `def`** (bisect) — deliberately loud:
   CPython evaluates-and-shares the list once at def-site, which owned value
   semantics cannot express (issue #80 / the nested-def default rule).

5. **Union return types** (`None | str` in test_math's `ulp_abs_check`) —
   rython's value semantics need a single return type; a genuine tagged union
   is not modeled.

6. **`**kwargs` routing (#368)** (textwrap `wrap(**kwargs)`) — advanced
   independently, but moot until the harness is present.

## What has landed toward #334's callables/closure front

- Nested `def` with a **bare-`Name` default** now converts (closure model
  captures the referenced value); genuinely-evaluating expression defaults
  (mutable lists, calls) stay loud. (`add7b07`)
- Recursion whose base type sits under an `if`-expression's non-recursive leaf
  no longer bails — `count_set_bits` infers `i64` instead of refusing. (`…`)
- **Empty `where`-clause bounds are dropped** instead of rendering
  `where , B: Clone` — invalid Rust that broke the generated crate for every
  `load_tests(loader, tests, ignore)` shape with an unbound nested-class
  member. `test_heapq`'s `load_tests` now renders a valid signature (and its
  crate advances to the nested-`iterable` callable wall). Real bounds
  (`A: PyAdd<B>, A: Clone, …`) still emit; only the empty ones are dropped.
  Pin: `empty_where_bound_does_not_emit_a_stray_comma`.
- **`csv.reader(f, quoting=…, escapechar=…)` is no longer a loud refusal.**
  The runtime reader always honoured `escapechar` in QUOTE_NONE mode; the
  codegen just refused to thread it. The reader now mirrors the writer's
  `Some::<u8>((c).as_bytes()[0])` seam. `test_csv` advances past its
  `escapechar` body to the dialect-registry wall (`csv.reader([...], name)`).
  Pin: `csv_reader_escapechar_wires_through_like_the_writer`.
- **`local_str.format(...)` infers a `String` return**, mirroring the `join`
  arm's concrete-receiver logic (a string literal or String/&str local
  receiver). This lets a `None` vs `fmt.format(...)` union unify to
  `Option<String>` — test_math's `ulp_abs_check` — and test_math advances
  from line 107 to line 744. Pin:
  `format_on_a_string_literal_local_infers_a_string_return`.
- These recede pure-language walls (math advances past `count_set_bits`), but
  the *count* moves only when the #334 harness lands, because that is the
  shared gate for all 17 files.