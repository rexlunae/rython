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

**Frontier: 6/17 CONVERT; 0/17 RUN.**

## Per-file status (as of this round)

| File | Status | First blocker |
|---|---|---|
| `test_heapq` | CONVERT | — (asserts dropped as warnings; needs #334 to assert/run) |
| `test_calendar` | CONVERT | — (same) |
| `test_struct` | CONVERT | — (same) |
| `test_fractions` | CONVERT | — (same) |
| `test_random` | CONVERT | — (same) |
| `test_operator` | CONVERT | — (same) |
| `test_math` | BLOCKED | `count_set_bits` recursion (FIXED this round → advances to `ulp_abs_check` union `None|String` return); then the `unittest` harness |
| `test_bisect` | BLOCKED | nested `def grade(breakpoints=[60,70,80,90])` — a **mutable-list default**, deliberately loud (Python evaluates-and-shares it; cannot be lowered correctly) |
| `test_csv` | BLOCKED | `csv.reader(..., escapechar=…)` reader feature, reached only inside `with self.subTest(...)` / `TemporaryFile` harness blocks |
| `test_textwrap` | BLOCKED | `wrap(text, width, **kwargs)` — issue #368 `**kwargs`; reached only inside unittest `TestTextWrap` methods |
| `test_itertools` | BLOCKED | heterogeneous list literal `['abc', range(6)]` (str/range mix) |
| `test_collections` | BLOCKED | heterogeneous list literal `[None, int(), gen(), object(), Bar()]` (int/Bar/mixed) |
| `test_functools` | BLOCKED | nested `class` at class level |
| `test_statistics` | BLOCKED | undecorated class-method binding + harness (`self.assertRaises`, `_make_std_err_msg` class call) |
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
- These recede pure-language walls (math advances past `count_set_bits`), but
  the *count* moves only when the #334 harness lands, because that is the
  shared gate for all 17 files.