# The Rython Language Specification

**Status**: living specification, tracking the implementation in this
repository. Statements about what *converts* and what is *rejected*
describe the current toolchain; the conformance rules in §1 are
normative and stable.

Companion documents: [`goals-and-design.md`](goals-and-design.md) (why
the boundaries are where they are), [`cpython-vs-rython.md`](cpython-vs-rython.md)
(model-level comparison), [`porting-guide.md`](porting-guide.md) (how to
get a program through the boundary), [`context-awareness.md`](context-awareness.md)
(design notes for the inference layer).

---

## 1. Conformance

### 1.1 Definitions

- **Conversion time** — when the Rython toolchain (`rypip`, `rythonc`,
  `python_module!`) parses Python source and generates Rust.
- **Runtime** — when the generated, compiled program executes.
- **Observable behavior** — the bytes a program writes to stdout/stderr,
  its exit code, and the files it produces.
- **Loud error** — a conversion-time failure with a message naming the
  unsupported construct (and, where possible, the source location and a
  suggested rewrite), or a runtime `PyException`/panic at the point of
  divergence. The opposite of a loud error is *silent divergence*:
  output that differs from CPython's without any signal.

### 1.2 The conformance rule

Rython is a **subset of Python**: every Rython program is a syntactically
valid Python 3 program, and the toolchain parses source with CPython's
own `ast` module. There is no Rython-only syntax.

For every Python construct, a conforming implementation does exactly one
of the following:

1. **Converts it**, with CPython's observable behavior as the intended
   semantics (under `PYTHONHASHSEED=0` where hashing is observable).
2. **Rejects it at conversion time** with a loud error.
3. **Converts it to a loud runtime failure**: a typed `PyException`
   raised where CPython would raise one, or — for conditions the model
   cannot represent as catchable values — a panic at the point of
   divergence.

Silent divergence is nonconformant. Byte-for-byte agreement with the
reference is the *target* for converted constructs and is enforced by
transcript tests wherever a behavior is pinned (§1.3) — it is not a
blanket claim about the implementation. The places where the current
toolchain knowingly differs — deliberate model tradeoffs, model limits,
and unfixed defects — are enumerated in §12; the project's direction is
to buy those differences down over time (§14), never to reclassify them
as acceptable.

### 1.3 The oracle

"Matches CPython" is defined by transcript: the end-to-end test suite
(`crates/rypip/tests/convert_tests.rs`) compiles converted programs and
diffs their output line for line against expected output captured from
real `python3` runs and pinned into the tests. Runtime-level pins live in
`crates/stdpython/tests/python_semantics.rs`.

---

## 2. Source language and accepted syntax

Input is Python 3 source. The parser accepts whatever CPython's `ast`
accepts; the *language* is then defined by which AST constructs the
lowering accepts.

**Statement kinds with defined lowerings**: expression statements,
assignment, annotated assignment, augmented assignment, `if`/`elif`/`else`,
`for`/`else`, `while`/`else`, `break`, `continue`, `pass`, `def`
(including `async def` at module level), `return`, `class`, `import`,
`from … import …`, `raise`, `try`/`except`/`else`/`finally`, `assert`,
`with`.

**Statement kinds rejected at conversion time** (loud error naming the
statement and its line/column): `match`, `del`, `global`, `nonlocal`,
and any other statement kind without a lowering. The error text is of
the form ``the `X` statement is not yet supported by rython``, with the
parser adding: *"Rewrite it using supported constructs, or file an
issue."*

Docstrings are extracted into Rust doc comments and otherwise dropped,
as in CPython.

---

## 3. Types

### 3.1 The static model

Rython is statically typed. There are no runtime type tags and no boxed
"any" value in generated code. Types come from three sources, in order:

1. **Annotations** — trusted, and effectively required on function
   parameters and (where the body cannot prove one) on returns.
2. **Literals and operations** — locals are inferred bottom-up through a
   small type lattice (`TypeInfo`; see
   [`context-awareness.md`](context-awareness.md)).
3. **Use-site pinning** — empty container literals take their element
   types from later uses (§3.4).

**One name, one type.** A variable binds to exactly one type for its
whole (function-wide) scope. A container literal mixing incompatible
element types is a conversion-time loud error — with one unification
exception: a literal mixing `int` and `float` elements unifies to
`float`, and string literals unify with computed strings as `String`.
Rebinding a name to a different type is not detected at conversion time
today: it fails as a rustc type error against the single hoisted
declaration — loud, but at the wrong layer (§12.1).

### 3.2 Type mapping

| Python type | Rust type | Notes |
|---|---|---|
| `int` | `i64` | No arbitrary precision; overflow is not detected as a Python-level error (§12.2) |
| `float` | `f64` | |
| `bool` | `bool` | |
| `str` | `String` (fields, returns, elements); `impl Into<String>` as a parameter | String *literals* are `&'static str` internally and are coerced to `String` where an owned string is expected |
| `bytes` | `Vec<u8>` | Literals lower to Rust byte strings |
| `list[T]` | `Vec<T>` | |
| `dict[K, V]` | `PyDict<K, V>` | `PyDict` is an insertion-ordered `IndexMap` alias — Python 3.7+ dict ordering is preserved |
| `set[T]` | `std::collections::HashSet<T>` | |
| `frozenset[T]` | `std::collections::HashSet<T>` (as an annotation) | The `frozenset(iterable)` *call* produces the distinct runtime type `FrozenSet<T>`; empty `frozenset()` is a loud error |
| `tuple` (literal) | Rust tuple `(A, B, …)` | There is no `tuple[…]` annotation mapping; tuples exist structurally |
| `None` / `Optional[T]` / `T \| None` | `Option<T>` | See §3.5 |
| `np.ndarray`, `np.float64`, `np.int32`, … | `numpy::NdArray`, `f64`, `i32`, … | Provided by the runtime's `numpy` module |

Dict keys normalize `&str → String`, so `{"a": 1}` is `PyDict<String, i64>`
and matches a `dict[str, int]` annotation.

**Bare container annotations are rejected.** `list`, `dict`, `set`,
`tuple`, `Optional` without a subscript are loud errors ("use a
subscripted annotation like `list[float]`"). Other unrecognized
annotations split by shape: a non-name annotation the mapper can't
handle is a loud error, but a plain-*name* annotation it doesn't know
is rendered verbatim as a Rust type — that is what makes user-defined
classes work as annotations, and it means a typo'd name (`x: itn`)
surfaces in rustc rather than at conversion time (§12.1).

**Unannotated parameters** lower to `impl Into<PyObject>`, where
`PyObject` is a PyO3 alias no ordinary Rython value converts into. Such
a function type-checks as a definition but is uncallable — in practice,
**parameter annotations are mandatory**.

### 3.3 Coercions

Conversions Rust needs but Python doesn't write are inserted by the
context-aware layer — lossless, with one documented exception (the
`int → f64` unification):

- `len(x)` and other `usize` producers coerce to `i64` at index and
  `range()` positions via `try_into().unwrap()` (overflow panics loudly
  rather than wrapping).
- String literals gain `.to_string()` where an owned `String` is
  expected; `String`s gain `.as_str()` where a `&str` is expected.
- `int` unifies to `f64` in mixed numeric literal contexts. This is
  lossy above 2⁵³ and accepted anyway, as the only way to compile a
  mixed numeric list at all.
- A non-`Copy` value read more than once in its function is cloned when
  passed as an argument to a user-defined function with a known
  signature — the *reuse-clone rule*. Method-call and subscript
  receivers are never cloned, so `xs.pop(); xs.pop()` mutates one
  vector. (Other move-prone positions, such as container elements, are
  not yet covered by the rule.)

### 3.4 Empty containers and type pinning

`xs = []` / `d = {}` have no inferable type by themselves. The
assignment converts only if either:

- the target is annotated (`xs: list[float] = []`), or
- a later use in the same scope pins the type: `append`/`push`/
  `extend`/`insert` for lists; a subscript store (`d[k] = v`) or a
  `get(k)` call for dicts.

The literal then renders as `Vec::<T>::new()` / `PyDict::<K, V>::from([])`.
An empty literal nothing pins is a loud error naming the variable and
suggesting both fixes.

### 3.5 `None` and optionals

A name that is `None` on any assignment path, or annotated
`Optional[T]`, becomes `Option<T>`: non-`None` stores are wrapped in
`Some(…)`, and values that already produce an `Option` (`dict.get`, an
optional-returning call) pass through unwrapped. Places where CPython
would put `None` *inside* a typed container that Rython cannot represent
(a non-participating regex group in `re.split`, `groupdict` of an
unmatched group) fail loudly instead of inventing a value.

---

## 4. Expressions

### 4.1 Arithmetic and comparison

- `+ - *` on `int` are plain `i64` ops (overflow: §12.2); on `float`,
  IEEE `f64` ops.
- `/` is true division producing `float`. **Known divergence**: `/` by
  zero does not raise — it lowers to an infallible IEEE division, so
  `x / 0` silently yields `inf`/`nan` where CPython raises
  `ZeroDivisionError` (issue #107; ledgered in §12.3).
- `//` and `%` follow Python — `//` floors toward negative infinity,
  `%` takes the divisor's sign — and a zero divisor raises
  `ZeroDivisionError` as a catchable `PyException` (the operators
  return `Result` and propagate with `?`). `divmod` is floor-based and
  fallible in the same way.
- `**` follows Python's promotion rules (int base with non-negative int
  exponent stays int); three-argument `pow` supports the modular
  inverse.
- Chained comparisons (`a < b < c`) evaluate each operand exactly once,
  via bound temporaries.
- `is`/`is not` against `None` test the `Option`; on other operands they
  currently lower to `==`/`!=` (a known deviation, §12.3).

### 4.2 Truthiness and boolean operators

Conditions implement Python truthiness through a `Truthy` trait:
`and`/`or`/`not` recurse, comparisons pass through, and any other value
is tested with `is_truthy()` (empty string/container and zero are
false). This applies to *condition position*. The value-producing form
(`x = a or b`) is lowered separately: Rust's short-circuiting
`&&`/`||` over the raw operands (so it is correct for boolean operands
and fails in rustc for others), plus a special case giving `a or None`
`Option` semantics — Python's return-the-operand behavior for arbitrary
truthy values is not modeled.

### 4.3 f-strings and `str.format`

A whole f-string lowers to a single `format!` call, with the critical
pin that a bare `{x}` renders through Python's `str()` (so `1.0` prints
`1.0`, `True` prints `True` — not Rust's `Display`). `!r`/`!a` apply
Python `repr` first, then pad. Integer radix and sign formatting go
through a helper reproducing Python's sign+magnitude form
(`format(-255, 'x')` is `-ff`, not two's complement).

`str.format` works on **literal templates only**; every argument is
evaluated exactly once in Python's order, whether used or not. Errors
mirror CPython's (mixing automatic and manual numbering, missing
keywords, `Single '{' encountered in format string`).

Loud errors: non-literal templates, `format(**kwargs)`, format specs
Rust cannot reproduce (`,` grouping, `=` alignment, space sign, `e`/`g`
presentations, nested spec interpolation `f"{x:{w}}"`, `!r` combined
with a numeric presentation type), and attribute/index access inside a
replacement field (`{a.b}`, `{a[0]}`).

### 4.4 Comprehensions and generator expressions

List, set, and dict comprehensions lower to eager nested loops building
a `Vec`, `HashSet`, or `PyDict`; conditions become `continue` guards;
bodies may raise (they run in `Result` contexts). Generator expressions
are **materialized eagerly** and then iterated — laziness is not modeled
(deviation, §12.3).

### 4.5 Lambdas

Lambdas lower to Rust closures. A lambda body that can raise cannot
propagate a `PyException` through the closure boundary; it panics
loudly instead of being catchable (deviation, §12.2).

### 4.6 Indexing and slicing

Indexing goes through checked helpers that raise
`IndexError`/`KeyError`. `IndexError` messages match CPython
(`"list index out of range"`); `KeyError` messages currently render the
key with Rust's `Debug` quoting (`KeyError: "name"`) where CPython uses
repr quoting (`KeyError: 'name'`) — a message-shape divergence on the
ledger (§12.3). Negative indices and slice *reads* follow Python
semantics. Slice **assignment** (`x[a:b] = …`) and augmented assignment
to a slice are loud errors.

---

## 5. Statements

### 5.1 Assignment and scoping

Python names are function-scoped; the generated Rust **hoists** every
assigned name to a single declaration at the top of its scope, and
assignments become stores. Mutability (`let` vs `let mut`) is computed
by a definite-initialization analysis that mirrors rustc's rules, so
generated code carries neither `unused_mut` warnings nor missing-`mut`
errors; mutation through methods (`append`, `pop`, file writes, …) and
through subscript/attribute stores marks the chain's base variable.

- **Chained assignment** `a = b = <container literal>` is a loud error:
  CPython binds both names to one object, and the value-semantics
  lowering cannot preserve that aliasing (issue #104).
- **Aliasing in general** (`b = a` then mutating either) is not modeled;
  see §12.3 and issue #79.
- Names first assigned inside a `try` body (which lowers to a closure)
  are pre-initialized with `Default::default()` to satisfy rustc's
  capture rules; behavior is unchanged on the paths Python defines.

### 5.2 Control flow

`if`/`elif`/`else`, `for`, and `while` lower to their Rust
counterparts. Loop `else` runs when the loop wasn't left by `break`,
implemented with a broke-flag only when the body actually contains a
direct `break`. A loop variable read after the loop is hoisted so
Python's scope-leak of the induction variable is preserved. Tuple
targets destructure.

### 5.3 `with`

`with` binds the context expression and relies on Rust's `Drop` at end
of scope; `__enter__`/`__exit__` are not called (deviation, §12.3). For
the supported file objects this reproduces `with open(...) as f:`
behavior (close on exit).

### 5.4 Imports

Runtime stdlib modules are already in scope in generated code (each
module emits `use stdpython::*;`), so `import math` lowers to nothing.
Consequences, all enforced loudly:

- `import math as m` (aliasing a runtime module) is a conversion error;
  `numpy` is the exception (it is a real path, so aliasing works).
- `from typing import …` and `from functools import partial/lru_cache/cache`
  lower to nothing.
- Importing a module that neither the runtime nor an FFI manifest
  provides is **not** caught at conversion time: it lowers to a bare
  `use name;` and fails as a rustc resolution error in the generated
  crate (§12.1). Under `--no-std`, importing a std-tier module *is* a
  conversion error naming the tier (§11.3).
- `from rython import rust` is the FFI declaration surface (§11.1);
  `import rython` by itself is rejected with a pointer to the correct
  spelling.
- Imports of user modules within the converted package resolve to the
  generated Rust modules.

---

## 6. Functions

### 6.1 Signatures

```python
def scale(values: list[float], factor: float) -> list[float]: ...
```

lowers to

```rust
pub fn scale(values: Vec<f64>, factor: f64) -> Result<Vec<f64>, PyException>
```

- **Every generated function returns `Result<T, PyException>`**; `T` is
  `()` when nothing (or nothing provable) is returned. Calls to user
  functions take `?`, so exceptions propagate exactly like Python's.
- The declared return annotation is honored only when every path
  provably returns; a body that can fall through returns `()` — with a
  conversion *warning* recording the ignored annotation.
- Parameters are owned values; `str` parameters are `impl Into<String>`
  converted on entry.
- Visibility derives from Python naming: `_name` is private,
  `__dunder__` is `pub(crate)`, everything else is `pub`.

### 6.2 Defaults and keyword arguments

Rust has neither, so both are resolved **at each call site** at
conversion time:

- Positional arguments fill left to right; keywords map by name;
  defaults fill the gaps. Mismatches CPython would raise `TypeError`
  for — an unexpected keyword, multiple values for one argument, a
  missing required argument — are conversion-time errors with the
  corresponding message. (One gap: surplus positional arguments on a
  keyword-free call skip the mapping pass and surface as a rustc arity
  error instead, §12.1.)
- When keyword reordering would change evaluation order, arguments are
  first bound to temporaries in Python source order.
- Defaults must be **constants**, checked at each call site that
  actually omits the argument (a bad default on a function whose calls
  all pass the argument explicitly is not flagged). Mutable
  container-literal defaults (`[]`, `{}`, `{…}`) are rejected with an
  explanation of CPython's shared single-evaluation semantics;
  non-constant defaults — including `set()`, which is a call — are
  rejected because call-site inlining would re-evaluate them.
- Keyword arguments require the callee's signature: keywords on an
  unknown callee are a loud error. (Keyword `replace()` on the datetime
  family is special-cased in the runtime.)

`*args`/`**kwargs` are effectively unsupported: signatures containing
them generate uncallable parameter types, and every concrete use site
(calls with `**kwargs`, keyword calls against such signatures,
`__init__`/methods/`lru_cache` with them) is rejected loudly. Starred
unpacking `f(*xs)` is likewise a loud error.

### 6.3 Decorators

The only supported decorators are `functools.lru_cache` (all spellings:
bare, called, `maxsize=n`, `maxsize=None`) and `functools.cache`. Any
other decorator — including multiple decorators — is a loud error:
*"rython refuses to silently ignore it."*

`lru_cache` lowering reproduces CPython's LRU discipline exactly
(touch-on-hit, front eviction, recursion through the cache) using a
global mutex-guarded cache. Constraints, all loud: functions only (no
methods), plain positional parameters, key parameters annotated
`int`/`bool`/`str` (Rust cannot hash floats Python-compatibly), not
available under `--no-std`.

### 6.4 `functools.partial`

Supported over statically-known functions: lowers to a move closure
binding the leading arguments. Keyword arguments, dynamic targets, and
over-binding are loud errors.

---

## 7. Classes

A Python class lowers to a plain Rust struct with an inherent impl:

```rust
#[derive(Clone, Default)]
pub struct Point { pub x: f64, pub y: f64 }
impl Point {
    pub fn new(x: f64, y: f64) -> Result<Self, PyException> { … }  // synthesized
    pub fn magnitude(&self) -> Result<f64, PyException> { … }
}
```

- **Fields** are inferred from `self.attr = …` stores in `__init__`
  (recursing through control flow); types come from annotated
  parameters, literals, or construction of another known class.
  Conflicting or uninferable field types (`self.x = None`) are loud
  errors.
- **Construction**: `Point(1.0, 2.0)` lowers to `Point::new(…)?`. The
  synthesized `new` defaults the struct then runs `__init__`. A
  user-defined method named `new` is a loud error.
- **`self`** becomes the method receiver — `&mut self` exactly when the
  method mutates through `self`, directly or transitively through its
  own calls (the analysis follows the call graph, including composed
  fields).
- **Rejected, loudly**: inheritance from anything but `object`, class
  attributes and any class-level statement besides methods/docstring/
  `pass`, nested classes, `async` methods, and `*args`/`**kwargs` in
  `__init__`. (For other methods, the `*args`/`**kwargs` rejection
  fires at each call site; a definition that is never called slips
  through to rustc.)
- **Dunder protocols are not modeled** — `__init__` is the only dunder
  with semantics. Defining other dunder-named methods (`__repr__`,
  `__eq__`, `__len__`, …) is *accepted*: they lower as ordinary
  `pub(crate)` methods with no protocol wiring, so nothing calls them
  implicitly. Protocol *uses* — printing an object, `==`, `len()`,
  operator overloading, `super()`, multiple inheritance — are out of
  the current subset; uses that reach codegen fail in rustc rather
  than at conversion time (§12.1).

---

## 8. Exceptions

### 8.1 Representation and matching

A raised exception is a value:

```rust
pub struct PyException { pub message: String, pub exception_type: String }
```

Matching is **by exact type-name string**, with `Exception` and
`BaseException` matching everything. The class *hierarchy in between is
not modeled*: `except LookupError` does not catch `IndexError`
(deviation, §12.3). `except (A, B)` ORs the names; a dotted name
matches on its final attribute; a bare `except:` catches all. (A bare
`except:` anywhere but last is a `SyntaxError` in CPython's parser —
which rython uses — so the not-last case never reaches conversion.)
`except E as e` binds a copy of the exception; `str(e)` is the message,
`repr(e)` is `Type('message')`. An uncaught exception exits with
status 1, printing `Type: message` to stderr on the wrapped entry
paths — §9 notes one entry path that currently prints Rust's `Debug`
form instead.

### 8.2 Raising

`raise E("msg")` lowers to `return Err(PyException::new("E", …))`;
whether a name is an exception class is decided by a known-name list
plus the `*Error`/`*Exception`/`*Warning` suffixes; other values are
assumed to already be exceptions. Bare `raise` re-raises the in-scope
exception and is a loud error outside a handler. `raise X from Y` folds
the cause into the message text — `__cause__` is not modeled
(deviation, §12.3). `assert` raises `AssertionError`.

### 8.3 `try` / `except` / `else` / `finally`

The `try` body runs in an immediately-invoked closure returning
`Result`; handler matching is a `match` on the error. `return`, `break`,
and `continue` inside the body are carried out of the closure through a
`PyFlow` enum and **replayed after `finally`**, reproducing Python's
ordering. With a `finally` present, handler and `else` bodies get their
own closures so their `return`/`raise` still runs the `finally` first.
The `else` body's exceptions are not caught by its own `try`. One loud
limitation: `break`/`continue` inside a handler or `else` when a
`finally` is present is a conversion-time error.

### 8.4 Runtime errors: catchable vs. panic

Raised as catchable `PyException`s, with CPython's message text except
where noted: `ZeroDivisionError` from `//`, `%`, and `divmod` — but
**not** from `/`, which silently yields `inf` on a zero divisor (issue
#107, §12.3); `IndexError`/`KeyError` from indexing (`KeyError`'s key
quoting diverges, §4.6); `ValueError` from `int()` (its message omits
CPython's "with base 10"), `chr()`, and `math` domain errors;
`FileNotFoundError`/`PermissionError`/… from `open` (messages embed
Rust's `io::Error` text rather than CPython's `[Errno N]` form);
`EOFError` from `input()`.

**Panics** (loud, not catchable), because the condition is not
representable in the model: `i64` overflow, sorting a `NaN`,
`hash(nan)` (identity-based in CPython), arithmetic on `None`, and an
exception escaping a lambda (§4.5).

---

## 9. Modules and programs

- Each Python module becomes a Rust module; packages become nested
  modules (`__init__.py` → `mod.rs`, root `__init__.py` → `lib.rs`).
- Module-level code: a name assigned exactly once from a literal
  becomes `pub static NAME: T = value;`. All other module-level
  executable code moves into a generated
  `fn __module_init__() -> Result<(), PyException>`, which the entry
  point runs first. Consequently, **non-constant module globals are not
  visible inside functions** — the reference fails to compile (loud,
  but with a rustc error; see §12.1).
- The entry point is `__main__.py`, or the module containing an
  `if __name__ == "__main__":` block. The block becomes `fn main()`.
  When the block does more than call `main()` (or module-init code
  exists), a generated wrapper runs `__module_init__` and the body,
  printing `Type: message` to stderr and exiting 1 on an uncaught
  exception. When the block is just `main()` with no module-level
  code, the user's `main` — which returns `Result` — becomes the Rust
  entry point directly, so an uncaught exception still exits 1 but
  prints the error's `Debug` form rather than `Type: message` (a
  cosmetic inconsistency, §12.3). A dedicated `__main__.py` is
  bin-only; any other entry module appears in both the library and the
  binary.
- Packages without an entry point convert to library crates and cannot
  be `rypip install`ed (loud error naming the fix).

---

## 10. The standard library

### 10.1 Builtins

Available without import (implemented in `stdpython`, re-exported into
every generated module): `print` (with `sep`/`end`; `file=` is a loud
error), `len`, `range` (lazy), `open`, `input`, `sorted`/`sort` (stable;
`key=` evaluated once per element; `reverse=` is a stable descending
sort), `min`/`max` (Python's NaN-fold semantics), `sum`, `abs`, `round`
(half-to-even; `round(x, n)` decimal-correct), `pow`, `divmod`,
`enumerate`, `zip`, `map`/`filter`, `all`/`any`, `repr`, `hash`
(CPython's algorithms under `PYTHONHASHSEED=0`, including siphash13 for
strings over the internal representation), `ord`/`chr`, `isinstance`
(on statically-known types only — otherwise a loud error), and the
`bool`/`int`/`float`/`str`/`list`/`dict`/`frozenset` conversions.
(`set(xs)` and `tuple(xs)` conversion *calls* are not implemented —
they lower unresolved and fail in rustc, §12.1; set and tuple
*literals* work.)

String, list, dict, and set methods cover the CPython surface for the
supported types, pinned to CPython edge cases (code-point `len`,
Unicode-correct `capitalize`/`title` titlecasing, Python's whitespace
and `splitlines` boundary sets, `str`/`repr` quoting and escapes).
Deliberate hole: **sets have no `repr`** — printing a set would expose
unordered iteration, so it is a compile error rather than
nondeterministic output.

### 10.2 Modules

Implemented (std tier): `math`, `random` (seeded output bit-identical
to CPython's MT19937), `os`/`os.path`, `sys`, `json`, `re`
(regex-crate backed: flags, named groups, `findall` tuples up to 3
groups; backreferences/lookarounds are a loud `re.error`),
`datetime`/`time` (incl. `strptime`, keyword `replace()`), `itertools`
(lazy iterators), `functools` (`reduce`, `partial`, `lru_cache`),
`heapq`, `copy`, `textwrap`, `hashlib`, `csv` (default excel dialect),
`collections` (`Counter`, `deque`, `defaultdict`, `OrderedDict`,
`ChainMap`), `pathlib`, `glob`, `subprocess`, `tempfile`, `argparse`
(conversion-time; §10.3), `string`, `io.StringIO`, `numpy` (a sizable
subset with pluggable execution backends).

Available on the `alloc` (no-OS) tier: `string`, `json`, `collections`,
`itertools`, `functools`, `heapq`, `copy`, `textwrap`, `hashlib`,
`csv`. Everything OS-touching is std-only and is a loud conversion
error under `--no-std`.

Known stdlib divergences from CPython that are verified but not yet
fixed are tracked in issue #82; they are defects, not spec.

### 10.3 Conversion-time `argparse`

The parser specification must be literal: the toolchain evaluates
`ArgumentParser(...)`/`add_argument(...)`/`parse_args()` **at conversion
time**, deletes those statements, and emits a typed namespace struct
plus a runtime parse whose usage line, help layout, error messages,
exit codes, and streams are byte-identical to CPython's. Supported:
`str`/`int`/`float` positionals, `--long` options with `default=`,
`action="store_true"`, `help=`, `prog=`, `description=`. Loud errors:
short options, `nargs`, `choices`, subcommands, dynamic specs, a
value-taking option without `default=`.

### 10.4 File objects

Text modes (`r`/`w`/`a`) and `io.StringIO` behind one surface,
including iteration, `with … as f:`, and CPython's
`"I/O operation on closed file"` error. Not supported (loud): binary
modes, `BytesIO`, `seek`/`tell`, file-based `json.dump`/`load`.

---

## 11. Interop and targets

### 11.1 Binding Rust from Python

Two equivalent surfaces share one lowering (so they cannot diverge):

**Declaration form** — `rust.bind` / `rust.c_bind`:

```python
from rython import rust
crc = rust.bind("crc32c", "crc32c",
                args=[("data", "&[u8]"), ("seed", "u32")],
                returns="u32", version="2")
```

Module level only, one target name, exactly one of `path=`/`version=`,
types from a fixed Rust-type list. Calls lower to direct Rust calls
with the documented conversion matrix (ints cast, `&str`/`&[u8]` via
`as_ref`, returns widened to `i64`/`f64`/`String`); `c_bind` wraps the
call in `unsafe`. rypip injects the crate dependency into the generated
`Cargo.toml`.

**Import form** — `rython.toml` at the package root:

```toml
[rust-modules]
crc32c = { path = "../crc32c" }   # or version = "…"; optional crate = "real-name"
```

Then plain `import crc32c` binds the crate. Signatures come from a
`.pyi` stub next to the manifest (annotations are string literals
naming Rust types) or are inferred from the crate's public functions
via `syn`. Imported names not in the manifest fall through to normal
import handling; importing a name the binding doesn't provide is a loud
error listing what is available. Bound imports lower to nothing — the
crate is a dependency, not a module.

### 11.2 Exporting to CPython (`--pyo3`)

`rypip convert --pyo3` adds a `python` cargo feature, a `cdylib` crate
type, and a generated `#[pymodule]` wrapping every public top-level
function whose signature is expressible in concrete types (every
parameter annotated with a mappable type —
`int`/`float`/`str`/`bool`/`bytes`/containers/`Optional`; no defaults,
no `*args`/`**kwargs`, no keyword-only or positional-only params;
underscore-prefixed names skipped). `PyException` maps onto
real CPython exception classes. If nothing is bindable, conversion
fails loudly listing the skipped functions. Incompatible with
`--no-std` and the kernel target.

### 11.3 `no_std` (`--no-std`)

Generates a `#![no_std]` library on stdpython's `alloc` tier. Loud
conversion errors (never deferred to rustc): `print`/`input`/`open`,
std-tier imports (`os`, `sys`, `math`, `random`, `datetime`, `re`,
`argparse`, …), and `__main__` blocks. The runtime ladder is
`core ⊂ alloc ⊂ std`; a strictly-core tier is not implemented and
fails loudly.

### 11.4 Kernel modules (`--kernel-module`, optionally `--rust-for-linux`)

A deliberately tiny sub-language, separate from the general transpiler.
Only `def module_init()` / `def module_exit()` (zero parameters) are
entry points; bodies may contain `printk("…")`/`printk(f"…")`,
assignments of integer literals or of allowlisted `rykernel_shim` call
results, bare allowlisted shim calls
(`from rykernel_shim import ktime_get_real_seconds`), docstrings
(dropped, as in CPython), `pass`, and `return` — anything else is a
loud error. Floating point is rejected *anywhere in the module*
(literals, `float()`, annotations, FP-using stdlib imports), with a
message explaining the kernel's FPU state — in the plain kernel
target; the device-manifest sub-mode currently bypasses this scan
(issue #108).
Module metadata comes from `__module_license__` (defaults to `"GPL"`),
`__module_author__`, `__module_description__`, `__module_version__`,
`__module_name__`; a misc-device sub-mode is driven by
`__device_name__`/`__bufsz__`/`__magic__`/`__device_mode__` dunders.
The raw-FFI target generates a C-free build pipeline (Makefile +
`-Zbuild-std`, kmalloc-backed allocator, `.modinfo` sections);
`--rust-for-linux` (valid only alongside `--kernel-module`) generates a
`module!`-macro crate for CONFIG_RUST kernels instead. `printk`
f-strings may interpolate only integer locals and literals; `!s`/`!r`
and format specs are loud errors.

`--driver` is a related but separate mode: it generates the
*userspace* companion to a kernel device, converted through the full
transpiler into an ordinary std crate (plus `libc`). It cannot be
combined with `--kernel-module`, `--no-std`, or `--pyo3`, and none of
this section's kernel restrictions apply to it.

---

## 12. Deviations from CPython

This section is the honest ledger §1.2 requires. Three categories.

### 12.1 Loud at the wrong layer

Conformant in outcome (nothing silent) but the diagnostic quality is
below the bar the project sets — the failure surfaces as a rustc error
in generated code rather than a conversion-time message:

- `eval`, `exec`, `compile`, `globals`, `locals`, `getattr`, `setattr`
  lower to unresolved calls and fail in rustc; so do the unimplemented
  `set(xs)` and `tuple(xs)` conversion calls.
- Importing a module neither the runtime nor an FFI manifest provides
  lowers to a bare `use name;` and fails at resolution in rustc.
- Reading a non-constant module global from inside a function fails in
  rustc (the global lives in `__module_init__`'s scope).
- Rebinding a local name to a different type fails as a rustc type
  error against the hoisted declaration.
- A typo'd or unsupported plain-name annotation (`x: itn`) renders
  verbatim (the mechanism that admits user classes) and fails in rustc.
- Dunder-protocol *uses* (printing an object, `==`, `len()`, operator
  overloading) fail in rustc — the method definitions themselves are
  accepted but never wired to a protocol (§7).
- Surplus positional arguments on a keyword-free call to a user
  function skip the argument-mapping pass and fail as a rustc arity
  error; a `*args`/`**kwargs` method that is defined but never called
  likewise surfaces only when rustc sees a use.
- Most aliasing shapes (`b = a` then mutate) fail in rustc's move
  checker (issue #79 proposes conversion-time detection).

### 12.2 Loud, by panic instead of exception

The condition is real at runtime but not currently representable as a
catchable `PyException`:

- `i64` overflow (CPython would grow the int). Note: in release builds
  unchecked arithmetic may wrap rather than panic — treat any
  overflow-adjacent arithmetic as out of contract until the opt-in
  bigint tier exists.
- Sorting a `NaN`; `hash(nan)`.
- Arithmetic on `None`.
- An exception escaping a lambda body.

The intended model is the one `ZeroDivisionError` already follows —
fallible operations return `Result<T, PyException>` and propagate with
`?` to the matching handler — and the direction (§14) is to move each
of these cases into that model wherever CPython defines an exception
for it (arithmetic on `None` as a catchable `TypeError`, exceptions
propagating through lambdas), and to eliminate the overflow panic
entirely via the bigint tier.

### 12.3 Known silent divergences (defects and model limits)

Each is either an open defect or a documented model limit; none is
accepted as permanent spec:

| Divergence | Status |
|---|---|
| True division by zero (`x / 0`, `1.0 / 0.0`) silently yields `inf`/`nan` instead of raising `ZeroDivisionError` (`//`, `%`, `divmod` raise correctly) | Defect, issue #107 |
| Exception message shapes: `KeyError` renders keys with Rust `Debug` quoting; `int()`'s message omits "with base 10"; `open` errors embed Rust `io::Error` text instead of `[Errno N]` | Defect class, same family as issue #82 |
| An uncaught exception on the direct-`main` entry path prints Rust's `Debug` form instead of `Type: message` (exit code 1 either way) | Defect (cosmetic) |
| The kernel device-manifest sub-mode skips the module-wide floating-point scan | Defect, issue #108 |
| Generator expressions are materialized eagerly (side-effect timing differs from lazy CPython) | Model limit until generator lowering lands |
| `with` does not call `__enter__`/`__exit__` (Drop approximates cleanup) | Model limit; correct for the supported file objects |
| `is`/`is not` on non-`None` operands lower to `==`/`!=` | Model limit (no identity model) |
| `raise X from Y` folds the cause into the message; no `__cause__` | Model limit |
| Exception matching ignores the hierarchy between `BaseException`/`Exception` and leaf names (`except LookupError` misses `IndexError`) | Defect class; flat matching is exact-name only |
| Argument-render-then-mutate shapes (`print(xs, xs.pop(), xs)`) render the first argument before the mutation | Recorded in issue #79 |
| Release-mode integer overflow may wrap (debug panics) | Bounded by §12.2's contract |
| Verified stdlib divergences (json/defaultdict ordering, `math.remainder`, `strftime` edge cases, `glob` paths, `pathlib` edges, `string.Template`, …) | Tracked as defects in issue #82 |

---

## 13. Conformance testing

- `crates/rypip/tests/convert_tests.rs` — end-to-end: write Python,
  convert, `cargo build`, run the binary, compare stdout line-for-line
  against pinned CPython transcripts (comments mark
  `// Verified against python3.`). Also the loud-error matrix for
  no_std, kernel, and pyo3 modes; the FFI loud-error matrices live
  alongside it in `rust_bind_tests.rs` and `rust_module_tests.rs`.
- `crates/python-ast/tests/codegen_semantics.rs` — generated-Rust
  shape assertions and `compile_err` loud-error assertions.
- `crates/stdpython/tests/python_semantics.rs` — runtime pins with the
  Python expression quoted in a comment; includes hash values, seeded
  `random`, digests, and formatting edge cases.
- `crates/python-ast/tests/error_propagation.rs` — unsupported input
  must surface as structured `Err`, never a panic.

A change is conformant when: the construct's tests pin CPython's
observable behavior, unsupported edges have `compile_err`-style tests,
and no test asserts behavior that diverges from CPython silently.

---

## 14. Planned evolution (non-normative)

Directions the language intends to grow, stated so the boundaries in
this spec read as current edges rather than permanent walls. Rationale
and ordering live in [`goals-and-design.md`](goals-and-design.md);
everything here lands under §1.2's rules when it lands.

- **Opt-in bigint tier**: a feature flag backing `int` with an
  arbitrary-precision crate. Retires the overflow panic (§12.2);
  default stays `i64`.
- **Opt-in dynamic typing at the edges**: a boxed value type, behind a
  flag or explicit annotation, for heterogeneous collections and
  similar compatibility wins. Static typing stays the default.
- **Panics → catchable exceptions**: extend the `Result<T, PyException>`
  model of §8 to the §12.2 cases with defined CPython behavior.
- **Class model**: single inheritance, then multiple; common dunder
  protocols; generalized decorators (unblocking `dataclasses`,
  `property`, user wrappers).
- **Generators/`yield`** as iterator-struct lowering; lazy generator
  expressions with them.
- **Aliasing**: conversion-time detection of alias-and-mutate shapes,
  possibly an opt-in shared-mutability lowering (issue #79).
- **I/O and stdlib**: binary file modes, `io.BytesIO`, file-based
  `json`; continued module expansion against the issue #82 register.
