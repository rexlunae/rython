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
statement and its line/column): `match`, `del`, `nonlocal`,
and any other statement kind without a lowering. (`global` is accepted;
see §5.1 for what its writes support.) The error text is of
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
| `bytearray` | `Vec<u8>` | Bytes-like, same lowering as `bytes` |
| `list[T]` | `Vec<T>` | |
| `dict[K, V]` | `PyDict<K, V>` | `PyDict` is an insertion-ordered `IndexMap` alias — Python 3.7+ dict ordering is preserved |
| `set[T]` | `std::collections::HashSet<T>` | |
| `frozenset[T]` | `std::collections::HashSet<T>` (as an annotation) | The `frozenset(iterable)` *call* produces the distinct runtime type `FrozenSet<T>`; empty `frozenset()` is a loud error |
| `tuple` (literal) | Rust tuple `(A, B, …)` | There is no `tuple[…]` annotation mapping; tuples exist structurally |
| `None` / `Optional[T]` / `T \| None` | `Option<T>` | See §3.5 |
| `Any` / `typing.Any` / `object` | `stdpython::PyValue` | The boxed heterogeneous value — both the bare `Any` name and the `typing.Any` spelling (urllib3's `dict[str, typing.Any]` returns), so a method annotated `-> dict[str, typing.Any]` types as `PyDict<String, PyValue>` instead of collapsing to unit (round 44) |
| `bytes \| bytearray` | `Vec<u8>` | Members mapping to the same Rust type collapse to it |
| `str \| bytes` (and `\| bytearray`) | `stdpython::StrOrBytes` | Heterogeneous pair; narrowed by `is_str()`/`is_bytes()`; `str()`/`print()` render bytes in their `b'...'` repr form |
| any other all-boxable union (`str \| int`, `bool \| str \| None`, …) | `stdpython::PyValue` | The boxed heterogeneous value (issue #121): members keep concrete types, `isinstance` narrows at runtime; `str()`/`repr()`/`print()` render Python-faithfully. Operators on a boxed value are not modeled — they fail the build loudly rather than guessing. A union containing None is NOT an Option slot — the box absorbs None, so `None`-defaulted parameters of such a type (`cert_reqs: int \| str \| None`, `retries: Retry \| bool \| int \| None` — urllib3) store plain values through `PyValue::from`, never a `Some(...)` wrap (rounds 40/42). A class-instance member has no boxed repr — storing one stays loudly unboxable (`PyValue: From<Retry>` fails) |
| `np.ndarray`, `np.float64`, `np.int32`, … | `numpy::NdArray`, `f64`, `i32`, … | Provided by the runtime's `numpy` module |
| `socket.socket` | `socket::Socket` | The runtime socket handle — `wait.py`'s `sock: socket.socket` parameters compile as real `Socket` values, not boxed PyValues |
| `threading.Thread/Lock/RLock/Event/Semaphore` | `threading::*` | The runtime threading handles (`ready: threading.Event` — a real shared handle) |
| `type[X]` / `Type[X]` | `Option<()>` | A CLASS value: rython cannot hold classes as values (the callables-as-data divergence); the tolerated opaque marker |
| `typing.Tuple/Dict/List/Set/FrozenSet/Optional/Literal/…` | like the bare containers | The typing-module spellings map identically to the bare `tuple[...]`/`dict[...]`/… (one resolver, one answer) |

All of the above resolve through ONE annotation authority (`resolve_alias_typeinfo` over the syntax core `annotation_type_info`; the old token-level resolver `python_annotation_to_rust_type` is a thin `TypeInfo::to_rust_type()` wrapper, issue #137's review of rounds 38–47). `set[T]`/`frozenset[T]` are `HashSet` everywhere — the generated structs are the arbiter (urllib3's PoolKey fields are `Option<HashSet<(String, String)>>`), and 1-tuples render `(T,)` with the trailing comma.

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
  expected; `String`s gain `.as_str()` where a `&str` is expected. The
  ownership is applied at the STORE/PUSH site when the slot's type is
  known: a literal stored into a `str`-annotated NAME (`method = "GET"`
  where the parameter is `str` — urllib3's urlopen), appended or
  inserted into a `list[str]` (`lines.append("\r\n")`), destructured
  into a String-typed tuple slot (`(body, content_type) = (None,
  "application/x-www-form-urlencoded")` where `content_type` is String
  from a `tuple[bytes, str]` return), or used as a String-keyed dict
  index (`d["b"] = v`) all own the literal at that site (round 46). A
  literal-only local (typed `&'static str`) keeps the bare store.
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
optional-returning call) pass through unwrapped. The `typing.Optional[T]`
spelling resolves the same way — a `typing.NamedTuple` field annotated
`typing.Optional[str]` (`Url` — urllib3) is an `Option<String>` field,
not a boxed PyValue (round 47; the alias-aware resolver previously
lumped the `typing.Optional` Subscript into the boxed-union tolerance).
A TUPLE-target store of all-`None` literals (`auth, host, port = None,
None, None` — urllib3's parse_url) marks each name an Option binding,
mirroring the single-name rule, so a later Option-returning store passes
through the name unwrapped instead of nesting `Some(Some(…))` (round
47). The same `Some(…)`
wrap applies to a plain value stored into an `Option`-typed FIELD
(`self._start_connect = time.monotonic()` where the field is
`float | None` — urllib3): Python's `int | None` slot absorbs a plain
`int`. A LOCAL assigned from an `Option`-typed parameter
(`release_this_conn = release_conn` where the param is `bool | None` —
urllib3's urlopen) is likewise an Option binding: its later plain
stores wrap in `Some(…)` (round 45; the `T | None` parameter annotation
resolves through the local-type map). Places where CPython
would put `None` *inside* a typed container that Rython cannot represent
(a non-participating regex group in `re.split`, `groupdict` of an
unmatched group) fail loudly instead of inventing a value.

Augmented assignment to an `Option` target (`self.chunk_left -= n`,
`options |= x`) operates on the INNER value — `-=` unwraps, subtracts
through the runtime `py_sub`, and re-wraps; `|=` ORs the inner value.
A `None` target at that point is CPython's `TypeError: unsupported
operand type(s) for -=: 'NoneType' and 'int'` — a loud §12.2 panic
with the message (guarded code never hits it). An `Option`-typed RHS
of `-` (`self.chunk_left - amt` where `amt: int | None`) likewise
unwraps with the loud panic.

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
(`x = a or b`) lowers with Python's return-the-operand semantics when
the operands' types unify: `a and b = if truthy(a) { b } else { a }`,
`a or b = if truthy(a) { a } else { b }`, and when exactly one operand
is `Option<T>` with the other `T`, the `T` arm wraps in `Some` (the
`ca_certs and expanduser(ca_certs)` shape — `str | None` and `str`).
The `Option` arm also fires when the other operand's type is UNKNOWN (a
call whose return is unresolved but renders the inner type) or a string
LITERAL (which is owned at the wrap — `Some(("http").to_string())` for
`scheme or "http"`), and a BoolOp with an Option operand yields an
Option for store purposes (round 43). The truthy arm of an `Option and
call(option)` fold passes the UNWRAPPED inner to the call
(`ca_certs and os.path.expanduser(ca_certs)` — round 48): Python's
`expanduser` receives the string, never `None`. A SELF-FIELD `Option<T>
or <concrete T>` fold (`self.path or "/"` — urllib3's Url, whose
`-> str` property needs the plain value) UNWRAPS to the inner `T` and
defaults to the concrete operand — Python's result is never `None`
(round 48); a NAME-typed Option operand (`scheme or "http"`) keeps the
Option-producing fold. An ununifiable mix (`bool and
str`, two different types) falls back to Rust's `&&`/`||`, which fails
loudly in rustc (§12.1) rather than silently returning a bool where
Python returns a value. `a or None` gets Option semantics via the same
unification.

A subscript STORE into a boxed dict (`dict[str, Any]` →
`PyDict<String, PyValue>`) absorbs an `Option` value the way the box
absorbs `None`: `ctx["scheme"] = scheme or "http"` where `scheme` is
`str | None` lowers to an explicit `match` — `Some(v) => PyValue::from(
v)`, `None => PyValue::None_` — matching CPython's `dict[str, Any]`
storing the string or `None` (round 46; an explicit match rather than a
`From<Option<T>>` blanket, whose multiple candidates would make an
UNTYPED value like `PyValue::from(resolve(None)?)` ambiguous at build
time). A LOCAL assigned from a DICT-RETURNING self-method call
(`request_context = self._merge_pool_kwargs(pool_kwargs)` — urllib3's
PoolManager) types from the callee's `-> dict[str, typing.Any]` return
annotation, so those stores own their string keys and box their values
(round 46; the class-aware seeding types ONLY dict-returning
self-method calls — the broad round-44 version cascaded on
conn-style locals).

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
semantics. Slice **assignment** `xs[a:b] = R` and **range delete**
`del xs[a:b]` on lists replace/remove the range in place with CPython's
exact bound rules (issue #153): a different-length RHS inserts or
removes elements, an inverted range is an insertion point, negatives
count from the end, out-of-range bounds clamp. Stepped forms
(`xs[a:b:s] = …`, `del xs[a:b:s]`) and augmented assignment to a slice
are loud errors.

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
- **Aliasing in general** (`b = a` then mutating either) is not modeled
  for plain-struct classes and containers; see §12.3 and issue #79. A
  SHARED class (below) is the exception: its references alias.
- **Shared instances (the aliasing representation, issue #137).** A
  class whose instances are stored in a container anywhere in the crate
  (a `list[C]`, `dict[K, C]`, `set[C]`, … slot, as the typed annotation
  authority resolves it — an alias `Items = list[C]`, a class's inferred
  field table, an un-annotated store the inferrer types — count; the
  boxed generics `Sequence[C]`, `Iterable[C]` hold no struct and do not)
  AND mutated after construction anywhere in the crate (a method, an
  `async def` included, that stores into or `del`s a `self` field, a
  container-mutating call on one, a field of the class stored through a
  non-`self` receiver — or the same on any base class, since a mutator
  is inherited) is SHARED: its values are
  `stdpython::PyRef<C>` (`Rc<RefCell<C>>`), so the container slot, a
  local fetched from it (`item = self.find(name)`), a loop variable, and
  a parameter are ONE object, as every CPython reference is. Reads go
  through `borrow()`, stores and mutating method calls through
  `borrow_mut()` (`item.qty -= qty`, `acct.deposit(5)`, `first.balance =
  1` all reach the stored object); a hierarchy family (a root and its
  subtree) shares one representation, the root's sum type holding the
  references. Every other class stays a plain struct (cloning an
  immutable object, or one no container holds, is unobservable). The
  `shared.rs` analysis is the one authority; a shared class's method
  that lets `self` escape (stores or returns `self`) is loud in rustc
  (the struct is not the reference). On shared references `is` is
  identity of the one object, `==` is CPython's default (identity) —
  a shared class that defines `__eq__` (own or inherited) cannot route
  `==` through it (the emitted `__eq__` takes the boxed value), so `==`
  on its references panics at runtime and the conversion warns. An
  instance's truth (`bool(x)`, also through `Option` and the sum type)
  is `__bool__`, else `__len__() != 0`, else True, the dunder resolved
  on the MRO — on a shared instance it runs on the one object. `is` on
  shared references, optional ones included, is identity of the one
  object. The registry is keyed by the class name the type side carries,
  so a name two modules of the crate both define is excluded from sharing
  with a conversion warning (as the hierarchy index excludes it); a store
  through a non-`self` receiver — a field store, or a mutating call on a
  field's container — counts for the class the receiver's scope names, and
  for every class with that field when it names none (a side
  effect in `__bool__` is what every reference sees). One boundary: a
  mutating method on a shared object holds the object for its duration,
  so a read of the SAME object through another reference inside that
  call (`a.merge(a)`, where `merge` mutates `self` and reads its
  argument) is a loud runtime error naming the aliasing, not a silent
  copy; every other alias shape is exact.
- Names first assigned inside a `try` body (which lowers to a closure)
  are pre-initialized with `Default::default()` to satisfy rustc's
  capture rules; behavior is unchanged on the paths Python defines.
- **`global` (issue #115).** `global name` declares module scope; reads
  resolve to the module statics. A module-level name written through
  `global` lowers to a MUTABLE static when it has exactly one
  module-level store and no function binds the name as a plain local
  (parameters included). The initializer decides the static's shape:
  an int/float/bool literal → `static name: Mutex<T>`; `None` →
  `Mutex<stdpython::PyValue>` (scalar/string stores boxed via
  `PyValue::from`); a string literal → `LazyLock<Mutex<String>>`; any
  other single-store expression → `LazyLock<Mutex<T>>` with the
  inferred type, or the boxed PyValue when none infers — with a touch
  in `__module_init__` so a computed initializer's side effects still
  run at import time, in order (a fallible initializer panics on
  failure, §12.2). Reads render `py_global_read(&name)` (`&*name` for
  the LazyLock forms), writes in owning scopes `py_global_write` —
  each takes the lock briefly, so compound assignment is CPython's
  non-atomic LOAD/op/STORE, and `+` on a boxed global dispatches at
  runtime (PyValue arithmetic; a member mismatch panics CPython's
  TypeError, §12.2). One boxed-global shape is typed instead: a
  None-initialized global whose `global`-writing functions store
  exactly one LOCAL class construction among `None` stores (the
  lazy-singleton idiom — `HISTORY_RECORDER = HistoryRecorder()`,
  botocore's history.py) lowers to `Mutex<Option<Class>>`; value reads
  unwrap the instance (a read while None is a loud runtime panic,
  §12.2), `is None` compares read the Option, and the getter's return
  type is the class. Storing any other container or class instance
  into a boxed global is a loud error (issue #189); multiple/
  conditional module stores, shadowing locals, and no_std keep the
  documented divergence: the write is dropped and reported through the
  `-W` channel.

### 5.2 Control flow

`if`/`elif`/`else`, `for`, and `while` lower to their Rust
counterparts. Loop `else` runs when the loop wasn't left by `break`,
implemented with a broke-flag only when the body actually contains a
direct `break`. A loop variable read after the loop is hoisted so
Python's scope-leak of the induction variable is preserved. Tuple
targets destructure.

### 5.3 `with`

`with` binds the context expression and relies on Rust's `Drop` at end
of scope; the general `__enter__`/`__exit__` protocol is not called
(deviation, §12.3). For the supported file objects this reproduces
`with open(...) as f:` behavior (close on exit). One real
implementation: `with lock:` over a threading `Lock`/`RLock`/
`Semaphore` (a name assigned from the constructor, or the constructor
inline) lowers to the runtime's RAII guard — acquire at entry, release
when the guard drops, exception-safe through `?` unwinding — exactly
Python's with-lock discipline. `with lock as x:` (CPython binds
`__enter__`'s `True`) is a loud error.

### 5.4 Imports

Runtime stdlib modules are already in scope in generated code (each
module emits `use stdpython::*;`), so `import math` lowers to nothing.
Consequences, all enforced loudly:

- `import math as m` (aliasing a runtime module) is a conversion error;
  `numpy` is the exception (it is a real path, so aliasing works).
- `from typing import …` and `from functools import
  partial/lru_cache/cache/singledispatch` lower to nothing.
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
- **Import guards are decided statically.** A module-level
  `try: import X … except ImportError:` (bare, or a tuple of
  ImportError/AttributeError) folds at conversion time, because
  rython's imports either always succeed or always fail. When every
  import in the try body is *unresolvable* (external to the crate,
  the runtime, and the vendored python-modules), the handler branch
  IS the module's body — its `brotli = None` / `HAS_ZSTD = False`
  fallbacks then make `if brotli is not None:`-style gates fold too
  (statically-decided names). When every import *resolves*, the try
  body splices in place and the dead handler never emits; a name
  bound by a resolvable `import ssl` is statically truthy and never
  None, so `if not ssl:` fallback classes fold away. Both decisions
  need the whole crate in view (a single-module conversion assumes
  unknown absolute imports are siblings). Stdlib exception aliases
  in such bodies (`BaseSSLError = ssl.SSLError`) register as
  aliases: `raise`/`except` sites canonicalize — including through a
  sibling's `from .connection import BaseSSLError` — and the store
  itself emits nothing (classes-as-values divergence). A
  `getattr(<stdlib module>, "NAME", default)` with a literal name is
  the same static decision: the runtime item when it exists
  (promoted to a `pub use` alias at module level), else the default.

An isinstance-DISPATCHED call whose axis argument is a boxed or
statically-unknown value, when no dynamic router could be planned (an
unannotated non-axis parameter, or an underivable morph return type),
DROPS loudly with a warning naming the rewrite instead of failing the
whole module (round 54 — requests' `_validate_header_part(header, name,
0)`, the last requests conversion blocker; the package now converts).
The dropped dispatch is the documented dynamic-dispatch divergence: the
isinstance checks cannot run on an unknown value.

An UNANNOTATED parameter that is REASSIGNED inside the function
(`hooks = hooks or {}`, `hook_data = _hook_data` — requests'
`dispatch_hook`) cannot keep one inferred generic type: it lowers as
the boxed PyValue with a definition warning naming the rewrite
(`annotate the parameter to keep a concrete type`) instead of failing
the module (round 53 — unblocks charset_normalizer and lets requests
progress past hooks.py). Downstream uses of the boxed value stay loud
(boxed-receiver drops, E0599).

An `except <builtin>:` clause whose class name is a source literal
(`except ValueError:`, `except socket.timeout:` — the dotted spelling
canonicalizes) lowers to a DISCRIMINANT comparison: `PyException`
carries the raised type's `BuiltinException` variant computed once at
construction, and the handler tests it against the clause's variant and
its precomputed ancestor slice — no string walk per clause (round 52).
Builtin ALIASES (`except EnvironmentError:` — a variant of OSError) and
user-defined classes keep the string `matches` walk (the alias has no
variant of its own; user classes are an open set). The semantics are
byte-identical to the string walk — the interpreter-derived MRO table
drives both, and a runtime pin verifies the two agree.

A from-imported STDPYTHON item is a plain call or a class construction
per the runtime's class registry: `OrderedDict(...)` lowers to
`OrderedDict::new(...)`, while FUNCTION items — `urlparse(url)`,
`quote(s)`, `re.compile(...)`, `warnings.warn(...)`, `json.dumps(...)` —
lower as direct runtime calls with `?` (round 55). Previously every
stdpython from-import was treated as a class, producing
`urlparse::new(...)` (E0433: a function used as a module path) at every
requests/urllib3 call site. The same registry applies through a
RE-EXPORT chain (requests' compat re-exports urllib.parse's functions).
An ALIASED import (`from re import compile as re_compile`) dispatches on
the canonical name but renders the BOUND name — only the alias is in
scope. `from json import dumps` routes through `dumps_pyvalue` (the
runtime converts the boxed value to the JSON model); `warnings.warn`
resolves through the signed runtime signature for both the qualified and
the from-imported spelling.

A `bool and <boxed value>` fold (`redirect and
response.get_redirect_location()` — urllib3's poolmanager, where the
call returns a boxed PyValue) keeps the VALUE operand when the bool is
truthy and boxes the bool otherwise — Python returns the second operand,
not a boolean (round 55). The previous `&&` fallback typed the result as
bool, which then poisoned every downstream use (`urljoin(url,
redirect_location)` failed on a `&bool` arg once urljoin became a real
runtime call). Only a DEFINITELY boxed operand takes this path; an
unknown operand (`bom_or_sig_available and should_strip_sig_or_bom(...)`
— charset_normalizer, both bool) stays `&&`.

The urllib.parse runtime surface (round 55) — `urlparse`/`urlsplit`
(returning a `ParseResult` with the six components plus
hostname/port/username/password/geturl), `urlunparse` (six string-like
components), `urljoin`, `urlencode(query, doseq=)`, `quote`/`quote_plus`,
`unquote`/`unquote_plus`, `urldefrag` — is pinned against CPython in the
runtime semantics tests. `urlencode`'s iterable-of-pairs form accepts a
boxed tuple/list; the boxed-model list-as-tuple divergence applies
(§12.3). Non-serializable values passed to `json.dumps` convert to JSON
Null rather than raising TypeError (the §12 loud-fallback divergence).

Round 57's corrective fixes on the round-55 urllib.parse surface
(the retrospective's R6 findings, each pinned against python3): the
`scheme` is now LOWERCASED (`urlparse("HTTP://e/").scheme` == "http"),
`urlsplit` no longer deletes `;params` from the path (urlsplit does NOT
split params — the path keeps `/p;q`, params stays empty), `hostname`
strips IPv6 BRACKETS (`[::1]` -> `::1`), and `username`/`password` split
the userinfo at the LAST `@` (`user@name:pass@host` -> username
"user@name"). The `urllib.parse` tests are un-gated from the `http-ureq`
feature so they run in the default workspace suite.

Round 56: the class-as-value model's value positions now cover the
BUILTIN classes. A bare `str`/`bytes`/`int`/`float`/`bool`/`list`/
`dict`/`tuple`/`set`/`frozenset`/`object`/`type`/`bytearray` name in a
tuple, dict key, or argument position (`basestring = (str, bytes)`,
`HEADER_VALIDATORS = {bytes: ..., str: ...}` — requests' compat/
_internal_utils) lowers to its name string, exactly like a user class
(round 33) — one predicate (`is_builtin_class_name`) for the value
renderer, the builtin-call dispatch, and the import handling. A
module-level BUILTIN-CLASS SELF-alias (`str = str`, `bytes = bytes` —
requests' py2 shims) is a no-op drop, so a sibling's `from .compat
import str` emits no runtime item and drops loudly; the name still means
the builtin, and calls through it dispatch to the builtin arms
(`str(x, encoding)` → the codec decode). Python TUPLE values box as
Tuple members through new `From<(T,)>`..`From<(T,..,T)>` impls (the
boxed model had no tuple path at all — idna's 800 `PyValue: From<(A,
B)>` errors). The loop-target reference walk now reads `del d[k]`
targets, so `for key in none_keys: del d[key]` keeps its name (it was
declared unused and lowered to `_` while the body's `py_pop(key)` still
referenced it). And the `range` class as a parameter annotation
(`offsets: range` — charset_normalizer) maps to the runtime `PyRange`,
the same type `range(...)` calls infer.

Round 57 (idna's data tables): a list literal whose elements are tuples
of DIFFERENT arities (`[(0, "3"), (65, "M", "a"), ...]` — idna's `_seg`
tables) boxes each element as PyValue — the element-type fold was
order-dependent, so a trailing short tuple re-absorbed the heterogeneous
result and every other row mismatched the inferred element type. The
`Union[A, B]` SUBSCRIPT annotation (idna's `List[Union[Tuple[int, str],
Tuple[int, str, str]]]`) now resolves like the `A | B` spelling — the
boxed union — and a function whose return annotation is a
`List[Union[...]]`-style boxed-element list threads the element type
into its RETURNING list literals, which box each element. And a
module-level TUPLE-UNPACK (`_STATUS_VALID, _STATUS_MAPPED, ... =
b"VMDI"` — idna's core.py) promotes each name functions read to a
static that extracts the value at its position (a module-init local is
invisible to function bodies — E0425). FUNCTION-LOCAL sibling imports
are deliberately NOT promoted (idna 3.10's `from .uts46data import
uts46data` inside a method: promoting the huge table cascaded through
the consumers' inference, 87 -> 179 rustc errors — measured and
reverted).

---

Round 58 (the retrospective's R2 start — the Option-adaptation
family): an OPTION-typed FIELD READ flowing into an Option slot no
longer double-wraps. `ca_cert_dir=self.ca_cert_dir` (the field is
`str | None`, so the accessor already returns the Option) rendered
`Some(self.ca_cert_dir())` — `Option<Option<String>>`; the store twin
(`destination_scheme = parsed_url.scheme` with parsed_url a Url whose
scheme field is Option) did the same. The ctx-aware
`expr_yields_option_ctx` predicate resolves the receiver's class — self
fields, typed params, factory-assigned locals (`u = parse_url(url)` —
now resolving IMPORTED factories too), and method-call receivers
(`self.proxy().host`, resolving the accessor's field class) — and
passes the Option through unwrapped. Pinned in both the argument and
store positions.

Round 59 (R2 continued): the Option-field STORE with a reused name no
longer bypasses the Some-wrap. `self._last_printable_char = character`
(field `str | None`, character read again later — charset_normalizer's
_count_suspicious) rendered `(character).clone()` — a bare String into
the Option field (E0308) — because the reused-name clone arm preceded
the Option-wrap arm. The Option arm now runs first and clones INTO the
Some (`Some((character).clone())`). Pinned.

Round 60: a literal SET (`{"utf_16", "utf_32"}`) builds as
`HashSet<&str>`; membership against an owned String operand
(`encoding_iana in {...}` — charset_normalizer) now resolves through a
`PyContains<String> for HashSet<&str>` impl (the generic `PyContains<T>
for HashSet<T>` already covered the &str spellings). Pinned.

Round 61: a field assigned from a CLASS CONSTRUCTION in one branch
(`self.headers = HTTPHeaderDict(headers)` — urllib3's HTTPResponse)
resolves through the constructed class even when another branch assigns
an unresolvable external param — `field_class` prefers the constructed
store. That unblocks the Mapping `.get(k, default)` path for such
receivers: the __getitem__+KeyError lowering, whose Ok arm wraps in Some
when the default is None (`headers.get(name, default=None)`), the
Option-typed returns box through the Some/None match (never
`PyValue::from(Option)`), and the boxed-dict store boxes Option field
reads the same way. Literal lists and sets (`Vec<&str>`,
`HashSet<&str>`) gain the str/String/PyValue membership spellings.
Pinned in each position.

Round 62 (the boolean-fold Option family): the `and`/`or` operand fold
now takes the Option arm where the operands' types unify through
containers and where a NAME's Option-ness is invisible to `infer_type`.
`conn or self._new_conn()` (urllib3's _get_conn — a local seeded
`conn = None` whose recorded None assignment infers PyObject) fell to
the `||` approximation (E0308 ×9); `fold_operand_type` now consults
`optional_names` for Name operands. `headers or {}` / `proxy_headers or
{}` (a `Mapping[str, str] | None` parameter OR'd with an empty-dict
literal, whose element types infer unknown) fell to `||` because
`inner_matches` only unified scalars — it now applies the same `unify()`
relation the rest of the codebase uses, and `infer_field_type`'s BoolOp
arm types the STORED FIELD from the fold's own operand analysis (an
Option operand keeps the Option; a PyObject-containing result falls back
to Bool — `PyDict<_, _>` is E0121 in a field signature). The field
inference resolves Name operands through the caller's explicit
`name_types` map so the field type is context-independent (the __init__
parameter types are invisible to module-level `options`). Sweep −20
(urllib3 1053→1033; the `bool | Option<_>` pair 7→1). Pinned in each
shape. `assert_hostname or server_hostname` (a boxed `bool | str | None`
OR'd with `str | None` into `impl Into<String>`) stays on the loud `||`
approximation — the truthy arm is a boxed value that cannot convert to
String without a silent guess, and CPython itself raises AttributeError
on the falsy-None path (`None.strip("[]")`).

Round 63: a SUBSCRIPT STORE through an Option-typed receiver
(`headers["k"] = v` where headers is `Mapping[str, str] | None` —
urllib3's RequestMethods; `request_context["blocksize"] =
_DEFAULT_BLOCKSIZE` where request_context is `dict[str, Any] | None` —
poolmanager) emitted `(#receiver).py_set_index(...)` on the raw Option
(E0599 ×4). The store now unwraps the Option receiver the same way the
read/call paths do (`as_mut().unwrap_or_else(panic)`, with CPython's
`TypeError: 'NoneType' object does not support item assignment` — the
receiver is guaranteed non-None after the `if x is None:` fill), and the
receiver's DICT TYPE is read THROUGH the Option so the String-keyed
index owning fires, the stored member of a `PyDict<String, PyValue>`
boxes in PyValue::from, and a str literal into a `PyDict<String,
String>` owns itself. The copy() family was attempted in the same
window but REVERTED: `x.copy()` on dict receivers (7 sites, −4 E0599)
unmasked a +12 cascade in poolmanager's _default_key_normalizer — the
copy's success types the `context` local, surfacing the boxed-PyValue
method gaps (py_index → PyValue → .lower(), PyValue::from(Option)) that
the never-type had masked. The copy arm returns when that boxed family
lands; the E0599s stay loud (§12.1). Sweep −4 (urllib3 1033→1029).
Pinned in both the plain and boxed-valued shapes.

Round 64 (the boxed-str-method family's first piece): `context["scheme"]
.lower()` where context is `dict[str, Any]` (urllib3's poolmanager — 8
sites) emits `(#recv).lower()` through the blanket `PyStrOps for T:
AsRef<str>`, which PyValue does not satisfy (E0599 "trait bounds not
satisfied"). A new `PyBoxedStrOps` trait dispatches on the runtime
member — Str → the operation; anything else → CPython's AttributeError
panic (§12.2) — and the attr-call path routes lower/upper/strip there
when the receiver POSITIVELY infers PyValue (PyObject stays on the
plain method, loud in rustc if the member is boxed). `infer_type` of a
SUBSCRIPT now reads the element type through an Option-wrapped base
(`request_context["scheme"]` on `dict[str, Any] | None`), which the
dispatch keys off; the round-63 store fix composes (the lowered value
boxes back into the `PyDict<String, PyValue>` member). Sweep −6
(urllib3 1029→1023). Pinned in codegen and runtime.

Round 65 (the unbound builtin-str method): `str.title(header)` —
urllib3's SKIPPABLE_HEADERS titlecasing — and `map(str.lower,
headers.keys())` — its request() content-type check — treated the
builtin `str` class as a value and read the method off the runtime
str() fn item (E0609/E0599). Python's `str.m(s)` is `s.m()`, so the
direct call lowers to the bound method on the argument (only the
zero-arg-beyond-receiver str methods qualify; `str.join(sep, xs)` is
the two-argument bound form), and `map(str.m, xs)` lowers the function
argument to a closure applying the bound method. Sweep −4 (urllib3
1023→1019). Pinned in both shapes.

Round 66 (the Option-dict method-call family): three members.
`for key in ("headers", "_proxy_headers", "_socks_options")` (urllib3's
poolmanager — 4 sites) iterated a Rust tuple, which is not
IntoIterator (E0277): an all-constant tuple iterates as an array, with
string literals OWING themselves so the loop target feeds String-keyed
dict calls (`request_context.pop(key, None)`). The call path's
`string_keyed_dict` flag and the `in`/`not in` membership arms read the
receiver's dict type THROUGH an Option (`request_context.pop("scheme")`,
`key in request_context` on `dict[str, Any] | None`), and the
membership test unwraps an Option comparator with a loud §12.2 panic
(CPython's `TypeError: argument of type 'NoneType' is not iterable`) —
the call path's Option receiver for the in-test. Sweep −12 (urllib3
1019→1011, charset_normalizer 289→285). Pinned in each shape.

Round 67 (super() factory locals): `r = super().make()` — an override
that assigns the BASE's method result and later reads a member of it —
left `r`'s class unresolved: the factory-local receiver resolution
(`x = self.make()` and imported factories) recognized only a bare
`self` callee, so a field read on the super-factory local emitted the
bare name. When the result class's member lives on the result class's
OWN embedded base (a base class's field read through a derived
instance), the bare read is an E0615 method-not-a-field. The
super-callee now resolves the method through the ENCLOSING class's base
chain (the override's own class does not define it) and takes its
return class, so the embedded-base chain rewrite fires. A DIRECT
imported-factory CALL as the receiver (`parse_url(url).netloc` — a
property of the return class, read in place) resolves the same way, and
the return class's name resolves against the DEFINING module's symbol
table (the annotation names classes in that module — the rule the
conservative receiver path already followed). Sweep −9 (urllib3
1011→1002: E0615 22→10, plus the boxed-union pair shifts the retyping
unmasked −6); the direct-call property routing traded the remaining
E0615s for the honest `Option<String>`-value store gaps (+3). Pinned
cross-module in both shapes.

Round 68 (the other-object Option-field local): `destination_scheme =
parsed_url.scheme` — a `str | None` field of a factory-local object —
then passed to a `str | None` parameter: the class-aware local seeding
typed only `self.<field>` reads, so the local stayed untyped and the
Option-slot ARGUMENT adaptation wrapped the already-Option local in
`Some(...)` (Option<Option<String>>, E0308). The seeding now also types
a local from an Option-typed field of ANY object whose class resolves
(through the same factory-local receiver resolution the read side
uses), so the argument passes through unwrapped. Sweep −3 (urllib3
1002→999). Pinned cross-module. The DIRECT-call receiver resolution also
grew to cover CLASS CONSTRUCTIONS — local or imported (`Url(...).url` —
a property of the constructed class read in place), routing the property
read to the getter. Sweep −2 more (urllib3 999→997; E0615 10→8).

Round 69: a PROPERTY read whose property is defined on a BASE class
(`self.host` from a derived method, where the base declares the
@property) emitted the bare name — the property check looked at the
derived class's own methods only, so the getter METHOD was an E0615
method-not-a-field. `has_property_getter` now walks the base chain, and
the read routes to the getter call. Sweep flat: the E0615s traded for
the honest mixed-Option-local gaps (+6 E0308 — a local annotated `str`
that later receives `str | None` stores; the store-type analysis does
not yet widen annotated names). Pinned.

Round 70 (the widened-local family that round 69 exposed): a local
annotated `str` that later receives `str | None` stores (`server_hostname:
str = self.host` then `server_hostname = self._tunnel_host` — the Python
value becomes None-able; the annotation was a hint, not a constraint).
The class-aware seeding now RECURSES into nested bodies (the Option
stores sit inside `if` blocks), walks the base chain for the field (the
store may be a base's field), and WIDENS a plain-typed name when an
Option-valued field store lands. The store path wraps a plain value into
the widened local (`Some(self.host()?)`, a str literal owning itself),
`expr_yields_option_ctx` recognizes a self-field ACCESSOR CALL (the
getter of an Option field) and reads through the base chain, and the
Option-slot ARGUMENT adaptation consults name_types directly (an
annotation in local_types would otherwise shadow the widened type and
re-wrap). Sweep −8 (urllib3 997→989; the round-69 exposure −6, plus
double-wrap fixes −2). Pinned.

Round 71 (the base-chain walk across modules): the class base chain used
a symbol-table-only walk that could not follow IMPORTED bases, so a
chain crossing a module boundary (`PoolManager(RequestMethods)` with
the field stored in the imported base) stopped at the derived class —
the field-walk, the property check, and the Option-ness resolution all
missed the ancestor's fields, and a generic-trait read emitted the bare
name (E0615 method-not-a-field / E0609). A new options-aware
`base_chain_with_options` resolves imported bases through the module
definitions (sharing one walk with the plain chain so the shapes cannot
drift), and the read path uses it. Sweep −1 (urllib3 989→988: E0615
2→0, E0609 −2; the corrected reads exposed honest arg-adaptation gaps
+3). Pinned cross-module.

Round 72 (the compiled-regex family): `_TARGET_RE = re.compile(...)`
module statics lowered as boxed PyValue — `PyValue::from(regex::Regex)`
(E0277, since the boxed union has no regex member) — and `.match(x)` /
`.search(x)` / `.fullmatch(x)` on them emitted `.r#match(x)` on the
boxed value (E0599). A `re.compile` static now types as the runtime's
compiled `Regex` — a `LazyLock<...::re::Regex>` whose type is a runtime
WRAPPER around the regex crate's engine (the wrapper retains a second,
whole-text-anchored engine, `\A(?:pattern)\z`, compiled at construction;
it is `Clone` so generated `(*_TARGET_RE).clone()` keeps the wrapper,
and `Deref` to the engine so the free functions keep working) — and the
method calls dispatch through a new `PyRegexOps` trait (root-exported so
generated crates can see it): `py_match` anchors at the START of the
text (the engine's `captures_at(text, 0)` filtered to a match starting
at 0), `py_search` finds the first match anywhere, and `py_fullmatch`
matches against the pre-anchored whole-text engine — CONSTRAINING the
engine rather than filtering its first unanchored result, so
`re.fullmatch("a|ab", "ab")` matches "ab" and `re.fullmatch("a*?", "aaa")`
consumes the whole text (a post-hoc filter wrongly rejected both; the
free `re.fullmatch` already anchored, the compiled-pattern path did
not — Devin review on #278). The `PyMatchOps for Option<PyMatch>`
surface provides `.groups()`. Sweep −11 (urllib3 988→978, idna 64→63).
Pinned in codegen and runtime (CPython-verified: alternation, lazy
quantifier, IGNORECASE, and capture-group preservation through the
anchored engine).

Round 73 (module and item names through typed enums): module names were
compared as raw string literals at ~50 sites across a dozen files, and
the datetime item set was duplicated in three places. All StdModule-
backed module checks now go through `StdModule::from_name` (including
the `module_name_shadowed` literals, which take `StdModule::X.name()`);
the datetime item set lives in a new `DatetimeType` enum consumed by the
import registries, the constructor lowering, and the static typing; the
annotation modules (`typing`, `typing_extensions`, `contextlib`, `abc`,
`dataclasses`) have no runtime backing and stay OUT of `StdModule` (its
`from_name` gates import lowering under the runtime crate), so they get
their own `AnnotationModule` enum + `is_typing()` predicate. Submodule
path segments (`numpy.linalg`, `collections.abc`) remain path structure,
not module-name-set membership. Sweep-neutral (978/63/0/285/25).

Round 74 (the Option-narrowing cluster, unmasked by round 73): the
compiled-regex fixes of round 73 let urllib3's url.py compile one step
further, exposing the Option-value-in-string-position family. Five
pieces close it: (1) a truthiness-narrowed `Option<String>` ARGUMENT to
a compiled pattern's match/search/fullmatch unwraps with the loud
NoneType panic (urllib3's `_normalize_host`); (2) `m.span(i)` — the
group-indexed span, Python's optional argument — routes to a new
`span_group`; (3) an Option-typed SLICE receiver (`host[start:end]`
after `if host:`) unwraps with the loud "not subscriptable" TypeError
panic; (4) a `-> T | None` function wraps its PLAIN returns in `Some`
and lowers `return None` to the None member, passing already-Option
values (a `T | None` property read, a `.get(key, None)` call, a local
assigned an Option-returning call — all recognized by the extended
`expr_yields_option_ctx`) through unwrapped; (5) an `if v is not None:`-
narrowed Option-typed name READS by unwrapping — the PyValue `as_str()`
path is for isinstance-narrowed boxed values only. Devin review on #279
tightened two of the edges: `m.span(i)` returns Python's exact `(-1, -1)`
for a non-participating group (not a panic), and the Option-arg unwrap
before a compiled pattern's match raises Python's exact `TypeError`
("expected string or bytes-like object, got 'NoneType'"), not the
AttributeError spelling. Sweep −16 (urllib3 978→962). Pinned in codegen
and end-to-end (CPython-verified).

Round 75 (the double-Option family): charset_normalizer's 285-error
residual was dominated by Option<Option<T>> nests. Two roots: (1) a
store from an IMPORTED function whose return annotation is `T | None`
(`character_range = unicode_range(chunk)` — cd.py, where utils'
callee returns `str | None`) Some-wrapped the already-Option result —
`expr_yields_option` now resolves imported callees through the module
defs and checks the defining FunctionDef's return annotation (the
store's pass-through guard sees the Option and stops wrapping); (2) an
@lru_cache function with an Optional key parameter
(`lg_inclusion: Optional[str] = None`) typed the cache key as
Option<Option<T>> — the key typing called python_annotation_to_rust_type
(which already returns the full Option) and wrapped it again. Both
fixed; the lru_cache hit/miss paths now type-check for Option keys.
Sweep −47 (charset_normalizer 285→238, everything else flat). Pinned in
codegen (imported-Option store, lru_cache Option key).

Round 76 (the str.is* family): charset_normalizer's biggest residual
was 48 E0599s — str.isupper/isalpha/isdigit/isspace/islower/
isprintable (plus isdecimal/isalnum/istitle) had NO runtime
counterpart. The PyStrOps trait gains the family with Python's exact
semantics (verified against python3: isupper = at least one cased
character and no lowercase, isalpha = non-empty all alphabetic,
isspace = non-empty all White_Space, isprintable = no Other-category
characters with the empty string printable, istitle = cased runs form
titlecase words). Two documented §12 approximations: isdigit/isdecimal
are ASCII-exact (Rust's std exposes no Unicode digit property, so the
'²'-class superscripts Python accepts are False here), and isprintable
treats format characters (Cf) as printable. Devin review on #281
tightened the Unicode edges: isspace now includes the four separator
controls U+001C..U+001F (CPython's White_Space includes them; Rust's
is_whitespace excludes Cc), isprintable is exact through the regex
engine's Unicode tables (`[\p{Cf}\p{Cn}\p{Zl}\p{Zp}\p{Zs}--[ ]]` —
non-ASCII spaces, line separators, format characters, and unassigned
code points are all False, ASCII space/tab True), and isalpha/isalnum
use the LETTER/NUMBER categories (`^\p{L}+$` / `^[\p{L}\p{N}]+$`) —
U+0345 (a combining mark with the Alphabetic property) is False like
CPython. The regex-backed implementations are gated on re-regex (the
default); the alloc tier keeps the approximation. Sweep −48
(charset_normalizer 238→190, everything else flat). Pinned in the
runtime against the CPython truth table.

Round 77 (is-None early-exit narrowing): charset_normalizer's
`String: From<Option<String>>` family (11 sites) — an imported
`str | None`-returning callee's local (`character_range =
unicode_range(chunk)`) stayed untyped in the hoisting analysis, so it
never entered optional_names and the is-not-None narrowing never fired.
Two fixes: (1) `call_return_typeinfo` resolves imported callees through
`module_defs_key` (the same root-relative normalization the import
lowering uses), so the local is typed Option<String> from the start;
(2) an `if X is None:` guard whose body ALWAYS exits
(continue/break/return/raise) narrows the FOLLOWING statements — they
are reachable only when X is not None (`if character_range is None:
continue` in encoding_unicode_range) — with the membership-comparison
unwrapping guarded against already-narrowed receivers. Sweep −3
(charset_normalizer 190→187, everything else flat). Pinned in codegen
(the is-None early-exit guard narrows the following reads).

Round 78 (string-literal ownership in Vec contexts): charset's
`String | &str` family — a `-> list[str]` function returning a list
LITERAL (`return ["Latin Based"]` — unicode_range_languages) and a
module `Vec<String>` static with a list-literal init
(`UNICODE_SECONDARY_RANGE_KEYWORD = ["Supplement", ...]`) emitted
`Vec<&'static str>` elements that mismatched the `Vec<String>` type.
The return statement's forced-list-element mechanism (round 57, the
boxed-element case) now also fires for `Vec<String>` returns, and the
promoted-static emission re-renders a list-literal init with the String
element type when the static's type is `Vec<String>`. Sweep −3
(charset_normalizer 187→182, everything else flat). Pinned in codegen.

Round 79 (virtual dispatch through abstract stubs): a call to a
`raise NotImplementedError()` stub method whose MISSING arguments all
have defaults (`self.read(len(b))` in BaseHTTPResponse.readinto, where
HTTPResponse overrides read) was DROPPED to a boxed None — the stub
arity exceeded the call's, so the abstract-protocol guard fired even
though the call is mappable to the full-arity invocation, which
dispatches VIRTUALLY to the derived override. The guard now drops only
stubs whose missing params are truly REQUIRED (unmappable —
botocore's extra `parsed`); a defaultable stub call lowers to
`(self).read(Some(len(b)), None, false)` and the derived override runs.
Also: a REUSED Name value in a slice-assign (`b[:len(temp)] = temp;
return len(temp)`) now clones — the assign MOVED the value and the
later read was an E0382 use-after-move. Sweep −3 (urllib3 962→961,
everything else flat). Pinned in codegen (the virtual stub dispatch and
the slice-assign clone).

Round 80 (loud dropped-call returns; generic From<PyValue> recovery):
two families in urllib3/idna. (1) A call on a BOXED receiver in return
position — `self._obj.method(...)` where `_obj` is a `PyValue` — was
lowered to `Ok(PyValue::None_)` in a TYPED function, silently returning
None for the call's real value. The return lowering now panics loudly
(`rython: the value ... cannot be returned as the function's typed
result (external-module / boxed-global divergence)`) at the exact point
of divergence, sharing the call-side drop predicate
(`boxed_receiver_method_dropped`: protocol methods, module receivers
like `copy`, and positively-boxed receivers all keep their prior
behavior). (2) The reverse generic impls (`From<PyValue> for
String/Vec<u8>/i64/f64/bool`) let a boxed value flow back into a typed
slot or `impl Into<T>` parameter via the accessors; a wrong member is a
loud `TypeError`-style panic (`value_member_panic`), matching Python's
fail-at-use. Sweep −38 (urllib3 961→931, idna 63→55, everything else
flat). Pinned in the runtime (boxed values convert back to typed
members) and in codegen (the loud dropped-call return).

Round 81 (generics at the typed-slot boundaries): the round-80 reverse
`From<PyValue>` impls were only half the story — the codegen still left
boxed values as E0308 at every typed slot that did not route through
`render_typed`. Four boundaries now convert, all with the loud
wrong-member panic (Python fails at use, rython at the conversion):
(1) `coerce_tokens` gains the reverse arms — `PyValue → i64/f64/bool/
String/Vec<u8>` via `.into()` and `Option<PyValue> → Option<T>` via
`.map(Into::into)` (the Option arm must precede the generic `T →
Option` arm, whose `from_ty != PyValue` guard would otherwise eat it);
(2) a CONCRETE typed return (`-> bytes`/`-> i64`) whose value is a boxed
local (`return decompressed` where a dropped call stored `PyValue::None_`,
and `return returned_chunk` where the `bytes | None` local holds
`Option<PyValue>`) converts at the return site (`.into()` / `.map(..).
expect(..)`); (3) call arguments into `X | None` parameters whose X is a
concrete member (`cert_reqs: int | None` receiving
`resolve_cert_reqs(...)` — a boxed callee return) wrap in `Some` and
convert the inner (`Some({ let __rython_v = ...; (__rython_v).into() })`),
recognizing boxed CALLS through the resolved return type and boxed
ATTRIBUTE reads through the same positive-evidence predicate the read-side
drop uses; (4) a REUSED boxed name stored into a name/field target clones
(`context = ssl_context` then `elif ssl_context is None` — urllib3's
_ssl_wrap_socket_and_match_hostname) — the Arc copy is Python reference
semantics. Also: an `and`-CHAIN in condition position narrows an
Option-typed name for the LATER operands (`if conn and
is_connection_dropped(conn)` — the second conjunct reads the unwrapped
value), with the compare/attribute Option-unwraps taught not to
double-fire on the narrowed read. Sweep −25 (urllib3 931→911, idna
55→51, charset 182→181, requests/certifi flat). Pinned in codegen
(four round-81 pins: and-chain narrowing, boxed-value return, boxed
argument into a concrete optional slot, reused boxed name store).

Round 82 (external-module return annotations box; inherited-field
stores use the base-most owner): three families in urllib3. (1) A
function whose return annotation is an EXTERNAL-module class
(`-> ssl.SSLSocket`, `-> logging.StreamHandler`) previously resolved to
NOTHING — the function silently typed `()` while its body returned a
value, so every caller of the return mismatched (the `() | PyValue`
family, 11 sites). The return-annotation path now falls back to the
symbols-aware authority (`resolve_alias_typeinfo`), which resolves the
external import to the boxed PyValue (the external-object divergence);
the body's value converts back via the round-81 `.into()`. (2) A raise
of a name imported from an EXTERNAL module (`raise ResponseNotReady()`
— http.client) dropped to the boxed None (the external-module call
divergence), silently replacing a raised exception with a returned
value (`PyException | PyValue` on every handler); the string-tagged
exception model now constructs the tagged PyException from the class
name. (3) Field STORES on an INHERITED field (`self.is_verified =
sock_and_verified.is_verified` in HTTPSConnection, where the field
lives on base HTTPConnection) consulted the DERIVED class's own
`infer_fields` — which joined the boxed member read to PyValue — and
wrapped in `PyValue::from` against the struct's real `bool` field
(`bool | PyValue`). The store-side field-type helpers
(`attr_field_is_pyvalue`/`_is_option`/`_concrete_type`) now resolve the
BASE-MOST class in the chain that assigns the field (its struct is the
ground truth) and convert boxed values into concrete inherited fields
via `.into()`. Sweep −25 (urllib3 911→887, charset 181→180, requests/idna/
certifi flat). Pinned in codegen (external-module return annotation
boxes; boxed value into a concrete inherited field converts).

Round 83 (a `bytes | None` FIELD widens honestly, and Option values
convert into concrete slots — the generics directive): urllib3's
DeflateDecoder stores `self._data = b""` in `__init__` and later
`self._data = None` in the error paths (`# type: ignore[assignment]`
in the source — the author KNOWS it is a `bytes | None`). The field
previously stayed `Vec<u8>` with the None stores dropping to the boxed
PyValue (`Vec<u8> | Option<_>` ×19 and `Vec<u8> | PyValue` families).
Two changes land the honest model. (1) A None STORE joining a concrete
field widens it to `Option<T>` — both in `__init__` (the
declare-then-fill conflict arm) and in any method's store list — the
same join the round-23 Option fields used, now covering the
method-store half (`_data` becomes `Option<Vec<u8>>`, `Vec<u8> |
Option<_>` 19→7). The reads then lower through three Option-aware
conversions, all loud on the empty case (Python fails at use on a None
value, rython at the conversion, §12.2): (a) a generic `Option<T> →
concrete` arm in `coerce_tokens` unwraps with the loud panic, so
`self.decompress(self._data)` (a `bytes` param) converts at the call
site; (b) `render_typed` answers `Option<expected>` for a read the
ctx-aware `expr_yields_option_ctx` proves Option-typed even when
`infer_type` reports PyObject ("no answer" for attribute reads) — the
coercion fires where the class table is the authority; (c) the
augmented-assignment `+=` with an Option target operates on the INNER
value and stores back wrapped (`self._data += data` → py_add on the
unwrapped member, `Some` rewrap), with CPython's exact TypeError text
(`unsupported operand type(s) for +=: 'NoneType' and 'bytes'`), the
`-=` arm's twin. A `Option<X> → Option<Y>` (different inner) arm maps
the inner conversion with None passing through — the `T → Option` arm's
recursion must not turn the empty case into a panic. The `.get`-with-
Option-default synthesis (`getheader(name, default: str | None)` —
http.client's HTTPResponse model) renders the fallback as the Option
itself: the round-83 unwrap would break the Ok-arm match
(`Option<String> | String` ×3 — caught by the re-sweep). Sweep −19
(urllib3 887→868, everything else flat — 1143→1124). Pinned in codegen
(Option-field aug-add inner; Option field value into a concrete slot;
mapping-get Option default stays Option).

Round 84 (a None-stored local into a BOXED-class parameter unwraps with
Python's None passing through): urllib3's `urlopen` binds `conn = None`
then `conn = self._get_conn(...)`, and passes the local to
`self._prepare_proxy(conn)` / `self._put_conn(conn)` — methods whose
parameter annotation is a TYPE_CHECKING-imported Protocol stub
(`BaseHTTPConnection`, `BaseHTTPSConnection` — imported only for
typing). The syntax-only annotation mapping cannot see a class name, so
the argument rendered RAW: the `Option<PyValue>` binding went straight
into the `PyValue`-typed parameter (the `PyValue | Option<PyValue>`
family, ×18 — the frontier's top residual). The argument-side expected
type now falls back to the symbols-aware authority
(`resolve_alias_typeinfo` — the same resolution the parameter's Rust
type used), so an OPTION-typed argument coerces `Option<PyValue> →
PyValue` via `unwrap_or(PyValue::None_)` — Python's None IS the boxed
None, no panic needed (the round-83 `Option→concrete` panic is for
concrete members; the boxed slot absorbs the empty case). The coercion
is gated three ways so it cannot misfire: only OPTION-typed arguments
(a plain class-instance argument keeps its loud raw mismatch — the
`err: _TYPE_TIMEOUT` sites would otherwise box through `PyValue::from`,
which has no From for a class); only unannotated names (an ANNOTATED
name's PyValue-ness is authoritative — `body: _TYPE_BODY | None` and
`chunks: Iterable[bytes] | None` put their None INSIDE the box, so the
fabricated Option-unwrap would be a wrong `unwrap_or` on a plain
PyValue); and the `X | None`-form parameter branch (a union that boxes)
takes the same unwrap for its present Option-typed arguments. Sweep −18
(urllib3 868→853 with `PyValue | Option<PyValue>` 30→12; charset
180→177 with `CharsetMatch | Option<CharsetMatch>` 3→0 — the same
None-stored-local family; idna/certifi/requests flat — 1124→1106).
Pinned in codegen (a None-stored local into a boxed-class parameter
unwraps).

Round 85 (the return-type directive: a function that can return exactly
`T | None` returns `Option<T>` — the caller decides what to do with the
None, and a caller that does not handle it in a concrete context gets a
LOUD error, never a mangled behavior): an unannotated function whose
returns unify to ONE concrete type T plus a None path (a `return None`,
a bare `return`, or a fall-through — the type-ignore "bug" pattern)
infers `Option<T>` instead of the boxed PyValue. Both inference paths
change: the generic collector's None-mixing arm (`return None` +
`return x` — `for x in p: return x` falls through to Option<B>, the
loop element; `if c: return 1` falls through to Option<i64>) and the
signature chain's `unified_return_type` (the annotated-parameter shape —
`return "yes"` / `return None` under `flag: bool` is Option<String>).
The return site follows the signature: `fn_return_is_option` now derives
from the same symbols-aware authority the signature uses (the syntax-
only `annotation_type_info` cannot see a quoted class name —
`Optional["CharsetMatch"]` — so the flag and the signature agree), plain
returns Some-wrap (a string literal owns itself), `return None` lowers
to the None member, and a fall-through tail is `Ok(None)` — an
Option-returning function's fall-through is Python's None. The caller
side propagates: `call_return_typeinfo` and `expr_yields_option` consult
the inferred return for an unannotated callee, so a store
(`v = pick(flag)`) types the local as an Option and `if v is None:`
narrows it — while a caller that passes the Option into a concrete slot
unhandled keeps the loud rustc mismatch (the directive's "throw an
error rather than mangle" — Python's likely-bug pattern). A plain
`return None` in a unit-returning function lowers to the unit value (the
NAME-None shape no longer falls through to a raw `None` token that
types as an Option against `Result<(), _>`). Sweep +7 (urllib3 853→856,
charset 177→181, idna/certifi/requests flat — 1106→1113): the
directive's honest cost — the newly-Option returns surface loud errors
where callers have not yet been taught to handle the None. Pinned in
codegen (loop-element fall-through returns Option; a partial literal
return becomes an Option; a caller of an inferred Option function
narrows and unwraps).

Round 86 (the argument-side expected-type fallback reaches the GENERAL
call path): the round-84 fallback (`resolve_alias_typeinfo` for an
annotation the syntax-only mapping cannot see, so an OPTION-typed
argument coerces into the boxed slot) lived only in the mapped-call fill
(`map_call_arguments_inner`) — a plain call whose arguments matched the
signature with no keywords/defaults rendered through the general
argument loop and never coerced (`g(resolve_default_timeout(timeout))`
where `_TYPE_TIMEOUT = Union[float, str, None]` lowers to the boxed
PyValue: the callee's `-> float | None` result went in raw). The
fallback is now a shared helper covering BOTH paths, with three gates:
a NARROWED name's read already unwraps (the `if conn and
is_connection_dropped(conn)` chain — wrapping the member again would
match on a non-Option); an `Option<Class>` inner cannot box into a
PyValue slot (no `From<Class>` — the raw mismatch stays loud instead of
shifting to an E0277 — but a CONCRETE slot coerces any inner via the
round-83 match-unwrap, charset's `fallback_specified: CharsetMatch`);
and the argument can be a CALL whose callee returns an Option — a
classmethod (`Timeout::resolve_default_timeout`) or a `self` property
accessor (`self.proxy()`), resolved through the class's method table.
Sweep −1 (urllib3 856→855, everything else flat — 1113→1112). Pinned in
codegen (an Option-typed callee result into a boxed-union parameter
coerces via the Some/None match).

Round 87 (the property-read Option local): a local assigned from a
PROPERTY read on a class-resolved receiver (`read_timeout =
timeout_obj.read_timeout` — urllib3's `_make_request`, where
`timeout_obj = self._get_timeout(timeout)` is a `-> Timeout` self-method
call) is recorded as an Option binding in the class-aware walk, seeded
from the getter's ACTUAL return annotation (`float | None` → Option<f64>,
not the unknown marker — the argument fallback's boxable gate must pass).
The walk's class-seeding arm types the factory local (`timeout_obj`) so
the property arm can resolve its receiver; the property-yields-Option
check (`expr_yields_option_ctx`) uses the READ-flavored receiver
resolution (`receiver_class_for_read` — the conservative `receiver_class`
hard-returns None on the attribute-callee Assign shape), so the store
passes the Option through instead of double-wrapping
`Some(timeout_obj.read_timeout()?)` into `Option<Option<f64>>`. The
Option<f64> local then coerces into a `_TYPE_TIMEOUT` (boxed PyValue)
parameter via the Some/None match. Three companion fixes fall out of the
same family: an integer-literal comparator against a Float-typed operand
promotes to the float (`read_timeout == 0` → `py_eq(&((0) as f64))` —
Rust std has no int/float cross-PartialEq, and Python promotes the int);
an ANNOTATED return wins over the body's inferred type (`-> float | None`
with a `return 0.5` body stays `Option<f64>` — the annotation is the
contract, and the return-site Some-wrap already agreed); and dict-literal
string VALUES normalize to owned String exactly like keys (`headers_ =
{"Accept": "*/*"}` in a `-> Mapping[str, str]` function renders
`IndexMap<String, String>`, never `IndexMap<String, &str>`), with
string-LIST elements owning the same way (`ks = ["Retry-After"]; return
ks` in a `-> list[str]` function builds `Vec<String>`); the runtime
gains the owned-form boxed-operand impls the container change surfaces
(`PyContains<PyValue> for Vec<String>`, `PyIndex<String> for PyValue`).
Sweep −15 (urllib3 855→845, charset 181→176, idna/certifi/requests flat —
1112→1097). Pinned in codegen (a property-read local on a factory local
coerces into a boxed-union parameter; a float Option compares with an
int literal as a float; a dict literal's string value is owned like its
key; an annotated Option return keeps the Option against a plain literal
body).

Round 88 (the unmodeled-base `super().__init__` and the ownership-clone
family): a `super().__init__(args)` call against a class with NO
structural base (urllib3's `_HTTPConnection` inherits the external
`http.client.HTTPConnection`) previously fell to the generic
"base implementation unmodeled — call the class's own method" fallback,
rendering a SELF-RECURSIVE `self.__init__(raw args)` call against the
class's own 8-parameter signature with 5 args (E0061). A non-empty
`super().__init__(...)` with an unresolvable base is now the same
documented divergence as the unmodeled-method path: a definition warning
and a no-op (the external constructor would set up the socket rython
cannot model). Separately, a REUSED class-typed local's ownership clone
(`timeout_obj` passed to two calls — urllib3's urlopen) previously
rendered `(x).clone()`, which resolves to the class's OWN `clone` method
when it defines one (urllib3's Timeout does) — a REAL semantic call where
Python just re-reads the variable. The reuse-clone now renders
`Clone::clone(&x)` — the trait-qualified std Clone, immune to inherent
shadowing and never naming the concrete type (a TYPE_CHECKING-only class
stub stays valid). And `dict.update(other)` — the stdpython PyDictOps
method takes the other dict by value, so an OPTION-typed argument
(`headers.update(self.proxy_headers)` — a `Mapping[str, str] | None`
field) coerces via the round-83 Option→concrete match: Python's
update(None) is a TypeError, and the loud panic is the honest model.
Sweep −6 (urllib3 845→839, everything else flat — 1097→1091). Pinned in
codegen (a reused class local clones via the qualified std Clone; a dict
update with an Option dict argument unwraps loudly).

Round 89 (Option values behind self-fields): the compare Option-match
trigger and the Option-slot pass-through both keyed on `infer_type`,
which cannot see through SELF-FIELD accessors. A `self.length_remaining !=
0` comparison (the field is `int | None` — urllib3's _handle_chunk)
rendered `py_ne(&(0))` on the raw Option (E0308 — the runtime
PartialEq impls only compare Option with Option); the trigger now also
consults the FIELD TABLE for a `self.<field>` LHS, unwrapping to the
inner comparison with Python's None-equality answers (`None != 0` is
True, ordered compares are the loud §12.2 panic). Likewise an OPTION-
typed FIELD READ on a class-resolved receiver (`self.proxy.host` — Url's
`host`/`port`/`scheme` fields are `T | None`, urllib3's ProxyManager
`super().connection_from_host(self.proxy.host, ...)`) now counts as
yielding the Option in `expr_yields_option_ctx` (the property arm only
covered accessor METHODS; the field table covers plain fields on any
receiver), so an Option-slot argument passes the read through instead of
double-wrapping `Some(self.proxy().host)` into `Option<Option<String>>`.
Sweep −10 (urllib3 839→830, charset 176→175, idna/certifi/requests flat —
1091→1081). Pinned in codegen (a self Option field compares via the
Option match; an Option field read passes through an Option slot without
double-wrapping).

Round 90 (factory-local field stores and the Option pass-through clone):
`field_class` — the class resolver for `self.<field>` receivers — only
named the field's class when the __init__ store came from an annotated
PARAMETER; a LOCAL store (`proxy = parse_url(...); self.proxy = proxy` —
urllib3's ProxyManager.__init__) returned None, so `self.proxy.host`
never resolved its receiver's class and the round-89 Option-field reads
still double-wrapped. The Name arm now resolves a local through its
single factory-CALL assignment's return annotation (same-module and
imported factories). And the Option-slot pass-through (`lower_optional_value`)
rendered an Option-typed FIELD read bare, MOVING it out of the shared
receiver (`headers = self.headers` where the field is Option<IndexMap> —
urllib3's _request_methods; `self.proxy.host` in the ProxyManager super
call) — E0507. The pass-through now CLONES attribute reads
(`(self.headers).clone()` — the Python object is shared by reference, so
the clone is the faithful copy).
Sweep −4 (urllib3 830→826 — E0308 −3, E0507 −1; everything else flat —
1081→1077). Pinned in codegen (a factory-local field store resolves the
receiver class; Option field reads clone out of the shared receiver).

Round 91 (base-chain Option field comparisons): the compare
Option-match trigger handled the bare `self.<field>` shape, but a BASE-
class field read through the embedded base struct (`self.__rython_base.
_tunnel_scheme == "https"` — urllib3's HTTPSConnection tunnel checks,
and the trait-base accessor twin `HTTPSConnectionTrait::base(self).
_tunnel_scheme()`) has an Attribute receiver chain, not a bare `self` —
the raw Option compared with the `&str` literal (E0308, the runtime
PartialEq only compares Option with Option). The trigger now walks a
`self.<chain>.<field>` receiver chain to confirm it roots at `self`, then
looks the field up in EVERY class of the enclosing class's base chain,
unwrapping to the inner comparison with Python's None-equality answers.
Sweep −3 (urllib3 826→823, everything else flat — 1077→1074). Pinned in
codegen (a base-chain Option field compares via the Option match).

Round 92 (plain-LHS Option comparators and the inferrer-unknown
reuse-clone): the Option-COMPARATOR unwrap (`x < y` where y is
`T | None`) lived only inside the LHS-Option branch of the compare
lowering, so a PLAIN LHS compared the raw Option (`len(self._decoded_buffer)
< amt` — urllib3's _read — E0277 on the missing PyLt<Option<i64>>
bound); the unwrap now applies to the py_* six ops regardless of the
LHS, with CPython's ordered-compare TypeError text naming the LHS type.
Fixing the comparisons EXPOSED six latent moved-value errors: a local
bound from a SELF-METHOD call whose return the inferrer cannot see
(`data = self._raw_read(amt)` — the response read family) infers as the
unknown marker, and the reuse-clone gate deliberately skipped
PyObject-named names — so the first `_decode(data)` moved the Vec<u8>
and every later read borrowed a moved value (E0382, previously masked
by the comparison E0277s). `.clone()` compiles on every generated type
(Copy types clone via Copy; classes derive Clone), so the gate now
clones whenever the name is not statically Copy, unknown type included.
Sweep −7 (urllib3 823→822 — E0277 −3, E0599 −1; charset 175→169 —
E0277 −5, E0382 −1; idna/certifi/requests flat — 1074→1067). Pinned in
codegen (a plain LHS with an Option comparator unwraps the comparator).

Round 93 (type-alias parameter annotations): the parameter-typing loop
recorded a BARE-NAME annotation that is neither a builtin scalar nor a
container as a CLASS (`value: _TYPE_FIELD_VALUE` where
`_TYPE_FIELD_VALUE = Union[str, bytes]` — urllib3's fields — recorded
Class("_TYPE_FIELD_VALUE")) — disagreeing with the parameter's actual
boxed PyValue Rust type, so a store into the local
(`value = "%s*=%s" % (name, value)`) went in raw (E0308) and every
method call on the local dispatched to the boxed value's unmodeled
methods (E0599 — the `_TYPE_FIELD_VALUE` / `_TYPE_TIMEOUT` /
`_TYPE_BODY` family in fields, response, connection, util). The bare-name
arm now resolves the annotation through the same symbols-aware authority
the parameter lowering uses (a module alias resolves to its value type;
a real class still records the class).
Sweep −13 (urllib3 822→809 — E0599 −22, E0308 +2, E0609 +6, E0382 +1;
everything else flat — 1067→1054). Pinned in codegen (a type-alias
annotated parameter stores into its boxed local).

Round 94 (cross-module boxed-param resolution and the qualified
typing.cast): the boxed-union argument fill (`headers: ValidHTTPHeaderSource
| None` → the boxed PyValue param) resolved the annotation's alias in the
CALLER's symbols — for a CONSTRUCTION of an imported class
(`HTTPHeaderDict(headers)` from _request_methods.py, whose
`ValidHTTPHeaderSource` is defined in _collections.py) the alias is
absent there, so the branch never fired and the OPTION-typed argument
went in raw (`PyValue | Option<IndexMap>` — E0308). The resolution now
uses the DEFINING module's symbols (the same `default_symbols` the
dropped-default constants already use). And `typing.cast(T, value)` — the
MODULE-QUALIFIED form (`typing.cast(ProxyConfig, self.proxy_config)` —
urllib3's _connect_tls_proxy) previously fell to the external-module
drop (`PyValue::None_`); it now lowers to its VALUE argument, exactly
like the imported `cast` name (a runtime identity).
Sweep −8 (urllib3 809→801 — E0308 −8; everything else flat —
1054→1046). Pinned in codegen (a module-qualified typing cast is a
runtime identity).

Round 95 (Option values through `typing.cast` locals): a cast-assigned
local (`proxy_config = typing.cast(ProxyConfig, self.proxy_config)` —
urllib3's _connect_tls_proxy, where the field is `ProxyConfig | None`)
previously stayed unknown — the field reads on it never unwrapped the
Option (E0609). The class-aware walk now looks through the identity
cast and seeds the local with the field's REAL Option type (not the
unknown placeholder, so the inner class resolves); the Option-slot
store passes the value through (a cast yields what its value yields)
and the self-field-read clone looks through the cast (an Option field
— any inner — clones out of the shared receiver, the same rule the
Option-slot pass-through already applied); and the field-read Option
detection resolves a receiver that is an OPTION-CLASS-typed local by
looking the inner class's field table up directly.
Sweep −8 (urllib3 801→793 — E0609 −9, E0507 +1; everything else flat —
1046→1038). Pinned in codegen (a cast-assigned Option field local
unwraps and clones on read).

Round 96 (boxed statics promoted from scalar initializers): a module
binding whose static type resolves but whose initializer is a PLAIN
value (`_FAILEDTELL: Final[_TYPE_FAILEDTELL] = _TYPE_FAILEDTELL.token` —
an Enum sentinel member, an i64 associated const — urllib3's
util/request and util/timeout) went through the inferred-type static
promotion path, which emitted the initializer RAW against a
`LazyLock<PyValue>` (E0308 — the boxed value was only wrapped on the
unknown-type path). A static whose resolved type is EXACTLY the boxed
PyValue now wraps its initializer in `PyValue::from` (a `PyDict<String,
PyValue>` typed static keeps its literal unwrapped — the wrap check is
token-exact, not a substring).
Sweep −2 (urllib3 793→791 — E0308 −2; everything else flat —
1038→1036). Pinned in codegen (a boxed static promoted from a scalar
initializer wraps).

Round 97 (the value-adaptation authority, first step — idiom corpus
acceptance): the idiom corpus (eval/idioms/) exposed that the
local-type analysis never typed a local assigned from an OPTION-
returning SELF-METHOD call (`item = self.find(name)` — a `-> Optional[Item]`
finder), so the `is None` early-exit-guard narrowing never fired and
every later field read hit the raw `Option<Item>` (the corpus's four
`no field qty on Option<Item>` errors). The class-aware walk now seeds
an Option-of-CLASS self-method local as the Option binding (name_types
AND optional_names — the narrow shape the guard narrowing consumes),
and `expr_yields_option_ctx` treats a self-method call whose return
annotation is an Option as yielding the Option (so the Option-slot
store passes it through instead of nesting `Some(Option<Item>)`) — the
raise-guard narrowing itself was already recognized; no raise-specific
branch was added. Separately, a comprehension `if` filter lowered its
condition RAW (`if !(w.strip())` — the unary `!` applied to a String,
E0600); it now routes through `condition_to_rust`, the SAME truthiness
authority the if-statement uses (Directive 5 — one path, not two).
Sweep −4 (urllib3 793→789, charset 169→167 — E0308 −3, E0600 −1;
everything else flat — 1036→1032). Idiom corpus: 13 → 9 errors on
inventory. Pinned in codegen (a comprehension filter uses the if-
statement truthiness authority).

Round 98 (the reuse-clone reaches attribute reads; sum over a
comprehension): three corpus shapes from inventory. `sum(item.qty for
item in self.items.values())` — the generator collector ends its Vec
with `.into_iter()`, so sum received an IntoIter with no PySum impl
(E0277 ×2) — the runtime implements PySum for the numeric IntoIter
forms (alloc-tier clean). `self.items[item.name] = item` — the dict
takes the value BY OWNED VALUE while the key reads a field of the SAME
object: the value clones when its root name is read again (the key read
would borrow a moved value, E0382 ×2), with CPython's value-then-key
evaluation order preserved (verified against python3). And the
reuse-clone generally covers ATTRIBUTE reads on a reused non-self
receiver (render_reused/render_typed_reused walked only plain names) —
a clone cannot wrap an `.into()` adaptation (the Into target is
unconstrained inside the clone, E0282), so that one adaptation skips
the clone while the round-92 boxing keeps it.
Sweep −1 (urllib3 789→788 — E0382 −1; everything else flat —
1032→1031). Idiom corpus: 9 → 5 errors on inventory (the remaining
five: a method call on an un-narrowed Option result, a moved `name`
read in a raise format, a `?` mismatch, `Result: PyDisplay` inside an
f-string, and `sorted` over (str, Item) — Item has no Ord). Pinned in
codegen (a dict store of a reused instance clones value and key; sum
over a generator comprehension lowers to the runtime sum).

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
- **Mixed returns box** (issue #133): when an unannotated function's
  return paths have no single concrete type — `return 1` / `return None`,
  a value return with a possible fall-through (Python returns `None`
  there), or a parameter returned as-is alongside a comparison result —
  `T` is the boxed `stdpython::PyValue`, every return site wraps in
  `PyValue::from` (`None` becomes `PyValue::None_`), and generic
  signatures carry the `PyValue: From<…>` bounds the wrapping needs. A
  returned list literal whose elements the list lowering boxes (§3.2's
  all-boxable union) gets the agreeing `Vec<stdpython::PyValue>`. This is
  the §3.2 boxed-union divergence applied to returns; consistent returns
  keep their concrete type.
- The declared return annotation is honored only when every path
  provably returns; a body that can fall through returns `()` — with a
  conversion *warning* recording the ignored annotation.
- A VALUE-PINNED unannotated parameter — reassigned from a call result
  (`path = os.path.expandvars(path)`, issue #161) — is the boxed
  `stdpython::PyValue`: it takes `impl Into<stdpython::PyValue>` (so
  call sites pass plain values exactly like Python; bytes literals box
  through `From<&[u8; N]>`), a prologue boxes it on entry, and stores
  into it wrap in `PyValue::from`. It carries no type variable and no
  bounds.
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

`*args`/`**kwargs` on module functions lower to the boxed heterogeneous
containers (issue #120): `*args` is `Vec<stdpython::PyValue>` and
`**kwargs` is `PyDict<String, stdpython::PyValue>`. Call sites with a
known callee pack the extras boxed (`PyValue::from` per value; a call
with none still passes the empty container), `f(*args)` forwards the
vector, and the body reads them like any list/dict (len, indexing,
iteration, membership — elements are `PyValue`, narrowed by isinstance
where a concrete type is needed; arithmetic directly on an un-narrowed
element fails in rustc). A keyword argument that matches no parameter of
a callee without `**kwargs` is a loud error, as in Python. Methods and
`__init__` with variadic parameters keep their existing per-site
handling; a `*t` spread to a NON-variadic callee still fills missing
positional parameters with the spread value (the spread-argument
divergence).

### 6.3 Decorators

The supported decorators are `functools.lru_cache` (all spellings:
bare, called, `maxsize=n`, `maxsize=None`), `functools.cache`,
`functools.singledispatch` with its `@<generic>.register(T)`
specializations (§6.5), `classmethod`, `staticmethod`, `property` (and
its `.setter`/`.getter`/`.deleter` spellings), and `dataclass`. Any
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

### 6.5 `functools.singledispatch`

A `@functools.singledispatch` generic and its `@<generic>.register(T)`
specializations are one FAMILY of definitions bound to one name. rython
fuses the family into the single `isinstance`-dispatching function that
expresses the same dispatch, and the monomorphizing specialization pass
(§10.1's `isinstance` model, `ast::tree::specialize`) lowers that into
one Rust function per registered type plus the `_any` residual, with a dynamic
router where every morph derives a return type. Inside a morph the
dispatch parameter is CONCRETE, so a `register(str)` body gets a real
`String` — ordinary `str` methods, not method calls on a boxed value.

Constraints, all loud: only `@<generic>.register(<type>)` with a single
bare type name is read (the annotation-typed and two-argument forms are
not); the generic and every specialization must live in one module; and
each specialization must share the generic's positional signature.
Dispatch is first-match in REGISTRATION order rather than CPython's MRO
walk (§12.3).

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
  Conflicting field types are loud errors; a `self.x = None` store and
  an attribute read off a call result (`urlparse(...).scheme`) box to
  PyValue (the external-object divergence).
- **Class-level computed constants** (`_encode_url_methods =
  frozenset([...])`) lower to a MODULE-level LazyLock static under a
  class-mangled name (`Class_name` — associated statics are not legal
  Rust) typed from the value when inferable, plus an associated accessor
  `Class::name()` that reads clone it through — so `self.name`,
  `cls.name`, and cross-module `Class.name` reads all work (issue #137).
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
- **Dunder protocols are mostly not modeled** — `__init__` has
  semantics, and the MAPPING trio is wired: a user class's own
  `__getitem__` receives `x[k]`, its `__setitem__` receives
  `x[k] = v`, and its `__contains__` receives `k in x` — the class's
  methods ARE Python's behavior, including its exceptions and any
  case-insensitivity. The routing fires only for a WELL-TYPED dunder (a
  concrete first-argument annotation; an `Any`-typed dunder cannot
  coerce the call's arguments either, so it keeps the loud py_index
  path). A class subclassing the `MutableMapping` ABC also gains the
  mixin's `.get(key[, default])` via a synthesis over `__getitem__`
  that catches `KeyError` only — gated on the ABC, so a plain
  `__getitem__`-only class does not silently gain a method CPython
  raises `AttributeError` for. Other dunder-named methods (`__repr__`,
  `__eq__`, `__len__`, …) are *accepted*: they lower as ordinary
  `pub(crate)` methods with no protocol wiring, so nothing calls them
  implicitly. Protocol *uses* — printing an object, `==`, `len()`,
  operator overloading, `super()`, multiple inheritance — are out of
  the current subset; uses that reach codegen fail in rustc rather
  than at conversion time (§12.1).
- **Inheritance and the hierarchy sum type.** A class with a base (or
  used as one) lowers with the trait machinery: its methods live on a
  `{Name}Trait`, the derived struct embeds its base (`__rython_base`),
  and every ancestor trait is implemented for it, so `super()` and
  inherited calls resolve. Rust structs have no subtyping, so a slot
  declared with a class that other classes derive from — a
  **polymorphic root**: a parameter `item: Item`, a field `dict[str,
  Item]`, a local `list[Shape]`, a return `-> Shape` — is the
  generated **sum type** `Any{Name}` with one variant per class of the
  root's subtree (the root included), computed once over the whole
  crate. A value of any class in the subtree flows into the slot through
  `From<Class> for Any{Name}` (a nested root's own sum type converts
  variant by variant); every method of the root's MRO and every field
  accessor dispatches by `match` to the variant's own implementation, so
  an override stored through a base-typed container runs — CPython's
  dynamic dispatch, decided by a `match`. `isinstance(x, T)` on a
  root-typed value is a runtime variant test (`x.__rython_is_T()`), true
  for an ancestor and false outside the subtree — one registry test that
  every target form consults: a class name (through its aliases), each
  element of a tuple of classes (their OR; an ancestor among them is true
  outright), and `type(self)` (the enclosing class); inside the guarded
  branch the name reads as the sum type's view of `T`, and a store or a
  mutating call through it — or through a field chain rooted at it
  (`s.tags.append(..)`, `s.center.bump()`) — takes the mutable view
  (`x.__rython_as_T()`, an owned clone), so `T`'s own fields and methods
  resolve. A leaf class IS its struct, and `isinstance` on it folds
  exactly through the class tree. Every class implements `PyDisplay`
  (`__str__`, else `__repr__`, else `<module.Class object>`) and
  `PyRepr` (`__repr__`, else the default form), so instances and
  containers of them print. Multiple inheritance beyond the first base
  is still dropped (loud where a call needs it).

---

## 8. Exceptions

### 8.1 Representation and matching

A raised exception is a value:

```rust
pub struct PyException { pub message: String, pub exception_type: String }
```

Matching walks **CPython's built-in exception hierarchy**: the clause
matches when its name is the raised type or one of its ancestors, so
`except LookupError:` catches `IndexError`/`KeyError`, `except OSError:`
catches the whole file-exception subtree, and — like CPython —
`except Exception:` does NOT catch `SystemExit`, `KeyboardInterrupt` or
`GeneratorExit` (they hang off `BaseException` directly). The tree is
the interpreter's own data: python-ast dumps every builtin
`BaseException` subclass's real `__mro__` (plus the stdlib-module
exceptions the runtime models — `urllib.error`, `socket`, `ssl`,
`codeop`) through PyO3 — the same path that produces parse trees — and
the checked-in table in stdpython carries it (regenerated and verified
by python-ast's `exception_tree_is_current` test, so the runtime can
never silently diverge from the reference interpreter). The
`EnvironmentError`/`IOError` aliases resolve to `OSError` for matching
(they are the same class object), and the stdlib exception aliases
(`socket.timeout` IS `TimeoutError`, the `socket.error`/`gaierror`/
`herror` family IS `OSError` — CPython aliases the class objects)
canonicalize at conversion time on both the raise and the except side,
however they were imported or renamed (issue #137). An exception type
outside the built-in tree is caught only by `Exception`,
`BaseException`, or its exact name. `except (A, B)` ORs the names; a
dotted name matches on its final attribute; a bare `except:` catches
all. An `except` whose type is a *runtime value* — a field or name
holding the boxed exception-name list (`except
self._retryable_exceptions:` — botocore's retryhandler, where the list
arrives as a `tuple` of class-name strings or `None`) matches through
`PyException::matches_value`: a Str member matches by name, a Tuple
matches when any member does, and any other value raises CPython's
`TypeError` ("catching classes that do not inherit from BaseException
is not allowed") exactly when CPython evaluates the clause. (A bare
`except:` anywhere but last is a `SyntaxError` in CPython's
parser — which rython uses — so the not-last case never reaches
conversion.) `except E as e` binds a copy of the exception; `str(e)` is
the message, `repr(e)` is `Type('message')`. An uncaught exception exits
with status 1, printing `Type: message` to stderr on the wrapped entry
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
- The module attribute protocol (PEP 562) is not supported: a
  module-level `__getattr__` or `__dir__` definition is a loud
  conversion error naming the dunder and the fix (issue #119). Module
  attributes resolve statically, so the dynamic fallback could never
  run; lowering it as an ordinary function would misstate the module's
  behavior.

---

## 10. The standard library

### 10.1 Builtins

Available without import (implemented in `stdpython`, re-exported into
every generated module): `print` (with `sep`/`end`; `file=` is a loud
error), `len`, `range` (lazy), `open`, `input`, `sorted`/`sort` (stable;
`key=` evaluated once per element; `reverse=` is a stable descending
sort), `min`/`max` (Python's NaN-fold semantics), `sum` (the
associated-Output `PySum` trait — int, float, and bool lists, bool
counting the Trues; on a generic parameter the return projects
`<T as PySum>::Output`, and a sum stored into an already-typed slot
pins the Output, issue #133), `abs`, `round`
(half-to-even; `round(x, n)` decimal-correct), `pow`, `divmod`,
`enumerate`, `zip`, `map`/`filter`, `all`/`any`, `repr`, `hash`
(CPython's algorithms under `PYTHONHASHSEED=0`, including siphash13 for
strings over the internal representation), `ord`/`chr`, `isinstance`
(decided at conversion time on a statically-known LEAF class, walking
the class inheritance tree — `isinstance(dog, Animal)` folds true for
`dog: Dog` — with constant branches pruned; a RUNTIME variant test on a
value of a polymorphic root, whose slot holds any class of the subtree
(§7); a module function whose unannotated
parameter(s) are isinstance-dispatched in plain `if` tests
monomorphizes — one specialized Rust function per input type, and with
SEVERAL tested parameters one per combination in their cartesian
product (`f_str_int`, `f_str_any`, ..., capped at 32 morphs) — plus
the generic residual, with call sites dispatching each argument
independently by static type (an int-tested parameter also gets a bool
morph of its own — bool ⊂ int in Python — so a bool argument takes the
int arm while `str(x)` still renders True/False); a dynamic router is
also emitted under the original name — one argument enum per tested
parameter (`FArg`, or `FArg1`/`FArg2`/... numbered by position), each
with one variant per morph plus `Other(PyValue)` and `From<T>` per
variant, taken as `impl Into<Enum>` parameters and tuple-matched — so
plain values pass through unchanged, untested parameters pass through
positionally, and a boxed `PyValue` argument routes at runtime in
Python's first-true-test order — an argument with NO statically-known
type (a value-pinned parameter reassigned through an untyped call,
issue #161's `path = os.path.expandvars(path)`) dispatches the same
way, and a residual morph whose fall-through decodes
(`path.decode(enc, errors)`) types as `str` and bounds `PyDecode`
(implemented by `Vec<u8>` and the boxed PyValue; a non-bytes boxed
value raises CPython's AttributeError, `errors="replace"` follows
CPython for utf-8, other non-strict errors values decode strictly —
the documented decode divergence); morphs whose return types differ route
through an output enum (`FOut`, one variant per distinct return type,
`From<T>` per member, and `From<FOut> for PyValue` when every member
boxes, so a runtime-dispatched result lands as Python's union value);
other
inferred-generic shapes lower to false with the class-as-value
divergence warning), and the
`bool`/`int`/`float`/`str`/`list`/`dict`/`frozenset` conversions.
(`set(xs)` works for the string shape — the set of its characters,
urllib3's `_UNRESERVED_CHARS = set("...")` — and a boxed set-of-strings
boxes as a Tuple of Str members; other `set(xs)` shapes and `tuple(xs)`
conversion *calls* are not implemented — they lower unresolved and fail
in rustc, §12.1; set and tuple *literals* work.) `iter(callable, sentinel)` (issue #155) is supported
as a for-loop iterable — `for x in iter(f, sentinel):` desugars to a
loop calling `f()` until the result equals the sentinel (bound once,
before the loop); anywhere else the two-argument form is a loud error.

String, list, dict, and set methods cover the CPython surface for the
supported types, pinned to CPython edge cases (code-point `len`,
Unicode-correct `capitalize`/`title` titlecasing, Python's whitespace
and `splitlines` boundary sets, `str`/`repr` quoting and escapes).
Old-style `%`-formatting works on `str` and `bytes` (round 34): the
full conversion set (`%s %r %a %d %i %u %o %x %X %e %E %f %g %G %c %b`
and `%%`), flags/width/precision incl. `*`, the `%(name)s` mapping form
with a dict RHS, and CPython's exact TypeErrors/ValueErrors for bad
codes, wrong value kinds, and argument-count mismatches — pinned to
CPython transcripts. `str(x)`/`print(x)`/f-string `{x}` on a class
INSTANCE route through the class's `__str__` (falling back to
`__repr__`, then the default object repr; §12.3 notes the dropped
address). A hierarchy trait whose DEFAULT body formats `self` in an
exception message (`raise ClosedPoolError(self)` — urllib3) declares
`Self: PyDisplay` on the trait — every implementor carries the
generated impl, so the bound is always satisfiable (round 41).
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
(conversion-time; §10.3), `string`, `io` (`StringIO`/`BytesIO`),
`threading` (§10.5), `socket` (§10.5), `numpy` (a sizable subset with
pluggable execution backends). `urllib.request` (§10.5) rides the
feature-gate convention below.

Available on the `alloc` (no-OS) tier: `string`, `json`, `collections`,
`itertools`, `functools`, `heapq`, `copy`, `textwrap`, `hashlib`,
`csv`, and `io`'s in-memory buffers (`StringIO`/`BytesIO` — the no_std
profile's file I/O; `open()` and disk files stay std-only). Everything
OS-touching is std-only and is a loud conversion error under
`--no-std`.

#### 10.2.1 Feature-gated platform surfaces

Platform-heavy functionality is wrapped, not reimplemented: where an
existing Rust crate provides the behavior (HTTP clients are the first
case), stdpython depends on it behind an opt-in cargo feature and the
default build stays dependency-light. Conventions:

- **Feature naming**: `<module>` for one natural backing crate,
  `<module>-<backend>` where several exist — extending the established
  precedents `async-tokio` (asyncio on tokio) and the numpy backends
  (`numpy-rayon`, …). The alloc/no_std tier is never affected.
- **Tooling contract**: rypip detects the import and enables the named
  feature on the generated crate's stdpython dependency (the asyncio/
  `async-tokio` mechanism); under `--no-std` the import is a loud
  conversion error naming the tier, like every std surface.

The first consumer is `urllib.request`: stdpython's `http-ureq`
feature wraps the ureq crate (with rustls, so `https://` works) and
`import urllib.request` in a converted package puts
`features = ["http-ureq"]` on the generated stdpython dependency.

The second is `ssl`: stdpython's `ssl-rustls` feature wraps the
rustls crate (client-side TLS). It stays in stdpython's own `default`
— TLS is load-bearing for the top converted packages (urllib3,
requests) — but generated crates do not ride those defaults (below),
so `import ssl` is what turns it on. The alloc/no_std tiers never
see it.

The third is `re`: stdpython's `re-regex` feature wraps the regex
crate, likewise in `default` and likewise requested by `import re`.

The fourth, `pyo3-interop`, is not import-driven: it carries the
`From<PyException> for pyo3::PyErr` surfacing and the `PyAny`/
`PyObject` aliases, which only rypip's `--pyo3` extension-module mode
names, so that flag is what enables it.

Because the surfaces are opt-in, the generated manifest names the
tier and the surfaces explicitly — `default-features = false,
features = ["std", …]` — rather than inheriting stdpython's defaults.
A converted package that imports none of them and needs no bindings
compiles neither rustls, nor the regex engine, nor pyo3 and its
proc-macro chain: **54 dependency crates drop to 23, and the
dependency build's CPU cost falls by about two thirds** (52.4s to
16.6s on a converted `print("hi")`), leaving the std tier only three
crates above the alloc tier. Getting a detection predicate too narrow
is loud in the prime directive's sense — the generated crate names a
module that was not compiled in, and the build fails — never a silent
loss of the surface.

Known stdlib divergences from CPython that are verified but not yet
fixed are tracked in issue #82; they are defects, not spec.

### 10.3 Conversion-time `argparse`

The parser specification must be literal: the toolchain evaluates
`ArgumentParser(...)`/`add_argument(...)`/`parse_args()` **at conversion
time**, deletes those statements, and emits a typed namespace struct
plus a runtime parse whose usage line, help layout, error messages,
exit codes, and streams are byte-identical to CPython's (3.11 help
format). The rewrite runs in function bodies AND at module level
(certifi's `__main__.py` builds its parser at top level — issue #118);
a module-level namespace lives in `__module_init__`, so only later
module-level statements can read it (a function read is loud in
rustc). Supported: `str`/`int`/`float` positionals, `--long` options
with `default=`, `-short, --long` alias pairs (exact and
attached-value forms at runtime; an unknown option-like token is an
"unrecognized arguments" error, never a positional),
`action="store_true"`, `help=`, `prog=`, `description=`. Loud errors:
a short option without a long alias, `nargs`, `choices`, subcommands,
dynamic specs, a value-taking option without `default=`. Not
reproduced: short-flag bundling (`-cv`) and `--opt=value` on shorts.

### 10.4 File objects

Text modes (`r`/`w`/`a`) and `io.StringIO` behind one surface,
including iteration, `with … as f:`, and CPython's
`"I/O operation on closed file"` error. `io.BytesIO` is the binary
sibling (its own type: `read`/`write`/`getvalue`/`close` over bytes,
with StringIO's overwrite-at-cursor discipline). Both buffers are pure
alloc and exist on the no_std tier. Not supported (loud): binary DISK
modes, `seek`/`tell`, file-based `json.dump`/`load`.

### 10.5 Threading and networking

**`threading`** (std tier): `Thread(target=, args=, daemon=)` —
the target must be a plain function name and args a tuple/list literal
(callables are not values; the lowering resolves the target at
conversion time, the `functools.partial` model), `start()`, `join()`,
`is_alive()`, `Lock`/`RLock` (`acquire`/`release`/`locked`, CPython's
RuntimeError messages, catchable), `Event`
(`is_set`/`set`/`clear`/`wait`), `Semaphore`, `current_thread().name`
("MainThread" / "Thread-N (target)"), `active_count()`. Thread objects
and locks are HANDLES with Python's reference semantics — cloning
shares — and thread args follow ordinary argument semantics (shared
handles share; containers copy, the §12.3 value-semantics divergence).
An unhandled exception in a thread prints CPython's
"Exception in thread NAME:" header and the exception line (no
traceback frames). `start()`/`join()` misuse panics with CPython's
RuntimeError text (§12.2 family).

**`socket`** (std tier): `socket.socket(AF_INET|AF_INET6,
SOCK_STREAM|SOCK_DGRAM)`, `bind`, `listen`, `accept`, `connect`,
`send`/`sendall`/`recv`, `sendto`/`recvfrom`, `settimeout`,
`getsockname`/`getpeername`, `close`, `gethostname()`. Errors raise
the real CPython hierarchy (`ConnectionRefusedError` IS-A
`ConnectionError` IS-A `OSError`; timeouts raise
`TimeoutError('timed out')`) with CPython's `[Errno N] text` message
shape. Not modeled (loud rustc error): `setsockopt`, `makefile`, the
address families beyond AF_INET/AF_INET6.

**`ssl`** (std tier, `ssl-rustls` feature — ON by default; or
`ssl-openssl`; §10.2.1): TLS with a pluggable backend, exactly one
enabled. `SSLContext(protocol)` / `create_default_context()` with
CPython's PROTOCOL_TLS_CLIENT defaults (CERT_REQUIRED +
check_hostname); `load_default_certs`, `load_verify_locations(cafile)`
(PEM, loud on an empty bundle), `set_alpn_protocols`, `wrap_socket(sock,
server_hostname=...)` → `SSLSocket` with `send`/`sendall`/`recv`
(ragged EOF reads as `b""`), `version()`,
`selected_alpn_protocol()`, `close()` (close_notify). The module
constants match python3 (CERT_*/PROTOCOL_*/OP_NO_*/VERIFY_X509_*/
SSL_ERROR_*, `TLSVersion` as a nested constants module), and the ssl
exception family (`SSLError` IS-A `OSError`; `SSLCertVerificationError`
also IS-A `ValueError`, `CertificateError` its alias) is wired into
the runtime hierarchy.

The default backend is **rustls** (ring provider, webpki/Mozilla
roots): client-side TLS. Divergences, all deliberate:
`OPENSSL_VERSION` reports `"rustls …"` (never an "OpenSSL" string, so
version-sniffing code takes its generic path) with
`OPENSSL_VERSION_NUMBER = 0` and a 3-tuple all-zero
`OPENSSL_VERSION_INFO` (§12.3); `set_ciphers` and `keylog_filename`
are stored-only no-ops (rustls's policy governs); the OP_*/VERIFY_*
bits are stored and readable, but rustls's own policy decides the
handshake, with `minimum_version`/`maximum_version` and the
OP_NO_TLSv1_2/1_3 bits clamping the negotiated range; `CERT_NONE`
installs a no-verification path exactly like CPython's unverified
context. Not modeled (loud rustc error): `MemoryBIO`/`wrap_bio`
(TLS-in-TLS), server-side sockets, client certificates.

The **`ssl-openssl`** backend links the SYSTEM OpenSSL/LibreSSL (the
`openssl` crate) and implements the full CPython surface with real
OpenSSL semantics: `CERT_OPTIONAL` half-verification (`CERT_REQUIRED`
verifies and requires a peer cert, `CERT_OPTIONAL` verifies one when
present), real `set_ciphers()` (an unknown cipher string raises
`SSLError`), client certificates via `load_cert_chain(certfile,
keyfile)`, server-side contexts (`PROTOCOL_TLS_SERVER` + accepting
`wrap_socket`), and `OPENSSL_VERSION*` reporting the linked library's
real version (so urllib3's OpenSSL sniffing takes its OpenSSL path).
The wire protocol is standard TLS 1.2/1.3; the system CA store backs
`load_default_certs`. Like the rustls backend it is std-tier; the two
features are mutually exclusive.

**`urllib.request`** (std tier, `http-ureq` feature; §10.2.1):
`urlopen(url)` for http/https with redirects, returning a response
with `.status`, `read()` (bytes), `getcode()`, `geturl()`,
`getheader(name)`, `close()`. An error status raises `HTTPError`
("HTTP Error 404: Not Found"); transport failures raise `URLError`
in CPython's `<urlopen error …>` shape (reason wording is
backend-derived, §12.3). `URLError`/`HTTPError`/`gaierror` are wired
into the exception hierarchy (`except OSError:` catches them).

### 10.6 numpy

`import numpy as np` maps onto the runtime's `numpy` module. Arrays are
VALUES (copies), not views: indexing, slicing and reshaping copy the
elements they touch.

**Accepted surface.** Creation: `array`, `asarray`, `zeros`, `ones`,
`full`, `empty`, `arange`, `linspace`, `eye`, `identity`, `dtype`.
Shape: `reshape`, `ravel`, `transpose`, `concatenate`, `vstack`,
`hstack`. Selection: `clip`, `where`, `sort`, `argsort`. Reductions:
`sum`, `prod`, `mean`, `max`, `min`, `std`, `var`, `all`, `any`,
`argmax`, `argmin`. Linear algebra: `dot`, `matmul`, `vdot`, and
`np.linalg.{inv,det,solve}`. The ufuncs (`add`, `subtract`, `multiply`,
`divide`, `floor_divide`, `mod`, `power`, `maximum`, `minimum`, `sqrt`,
`exp`, `log`, `log2`, `log10`, the trig and hyperbolic family, `floor`,
`ceil`, `abs`, `negative`, `square`, `sign`, `reciprocal`, `expm1`,
`log1p`, `isfinite`, `isinf`, `isnan`, `logical_not`) and the
comparisons (`equal`, `not_equal`, `less`, `less_equal`, `greater`,
`greater_equal`, `bitwise_and`, `bitwise_or`, `bitwise_xor`). The dtype
casts (`np.float64(x)`, `np.int64(x)`, …) and `dtype=` on the
constructors, over `float64`, `float32`, `int64`, `int32`, `bool_`.
Array attributes `shape` (a tuple), `ndim`, `size`, `dtype`, `T`, and
the methods `sum`, `prod`, `mean`, `max`, `min`, `std`, `var`, `all`,
`any`, `argmax`, `argmin`, `reshape`, `ravel`, `copy`, `astype`.
Indexing, slicing, boolean-mask indexing, iteration, in-place `+=`/`*=`,
and broadcasting all follow numpy.

**Printing** reproduces numpy's own formatter: `precision=8`,
column-aligned cells, `linewidth=75` wrapping, `threshold=1000`
summarization with `edgeitems=3`, exponential mode past the dtype's
cutoff, and numpy's `repr` with its `shape=`/`dtype=` extras.

**Outside the subset**, and a loud conversion error naming the
construct: `axis=` on reductions, the positional `axis` of
`np.concatenate`/`np.std`/`np.var`, `np.linspace(..., endpoint=False)`,
`np.full(..., dtype=…)`, every numpy submodule other than `linalg`
(including `np.random`), and any function not listed above.

**Execution backends.** Every elementwise kernel dispatches through one
engine chosen once per process: `scalar` (always built), `rayon`
(`numpy-rayon`), `simd` (`numpy-simd`, currently an alias of `scalar`),
`cuda` (`numpy-cuda`) and `vulkan` (`numpy-vulkan`) — the last two
compile but ship no kernels yet. Selection is `np.set_backend("...")`,
the `RYPY_NUMPY_BACKEND` environment variable, or `rythonc
--numpy-backend`; `auto` picks the best engine compiled in. Requesting
an engine the binary lacks is a loud `RuntimeError`, never a silent
fallback. Every backend produces identical results — the parity is
pinned by tests.

The divergences that remain are ledgered in §12.2 and §12.3.

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
type, a generated `build.rs` requesting pyo3's extension-module link
args (without them a macOS linker rejects the cdylib's undefined
`_Py_*` symbols), and a generated `#[pymodule]` wrapping every public top-level
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
`argparse`, `threading`, `socket`, `urllib`, …), and `__main__`
blocks. `import io` works: the in-memory `StringIO`/`BytesIO` buffers
are the profile's file I/O (§10.4). The runtime ladder is
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
`-Zbuild-std`, kmalloc-backed allocator, `.modinfo` sections — the
`.modinfo` placement is gated with `cfg_attr(target_os = "linux", …)`
so a module authored on any host still type-checks there);
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


- Without the `re-regex` feature (the light no-re build), the str.is*
  Unicode classification methods fall back to Rust std's char
  properties: isalpha/isalnum use the Alphabetic property (which
  includes combining marks like U+0345 that CPython's Letter-category
  rule rejects), and isprintable misses format/unassigned categories.
  The DEFAULT build (re-regex on) is exact through the regex engine's
  Unicode tables; the light build's approximation is documented here.
## 12. Deviations from CPython

This section is the honest ledger §1.2 requires. Three categories.

### 12.1 Loud at the wrong layer

Conformant in outcome (nothing silent) but the diagnostic quality is
below the bar the project sets — the failure surfaces as a rustc error
in generated code rather than a conversion-time message:

- `eval`, `exec`, `compile`, `globals`, `locals`, `getattr`, `setattr`
  lower to unresolved calls and fail in rustc; so do the unimplemented
  `set(xs)` (beyond the string shape) and `tuple(xs)` conversion calls.
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

- `m.groups()` returns `Vec<String>` and FAILS LOUDLY (a ValueError-
  typed panic) when a capture group did not participate: Python yields
  `None` for that member (a tuple), which a typed `Vec<String>` cannot
  hold — the same divergence `m.group(i)` documents. The tuple-
  DESTRUCTURE form (`path, query = m.groups()`) hits the same wall: the
  None-able members need an Option-typed lowering, which is not wired
  yet (round 74 defers it to the Option-slot widening work). `m.span(i)`
  is exact: an absent group's span is Python's `(-1, -1)`, not an error.

### 12.2 Loud, by panic instead of exception

The condition is real at runtime but not currently representable as a
catchable `PyException`:

- `i64` overflow (CPython would grow the int). Note: in release builds
  unchecked arithmetic may wrap rather than panic — treat any
  overflow-adjacent arithmetic as out of contract until the opt-in
  bigint tier exists.
- Sorting a `NaN`; `hash(nan)`.
- Arithmetic on `None` (including aug-assign `-=`/`|=` on an `Option`
  target whose value is `None`, and a `-` whose RHS is `None` — the
  Option-unwrap panics carry CPython's TypeError text).
- An ORDERED comparison (`<`/`<=`/`>`/`>=`) on an `Option` whose value
  is `None` (`amt < self.chunk_left` where either is `int | None` —
  urllib3) — CPython's `'<' not supported between instances of
  'NoneType' and 'int'` TypeError, but a panic with the exact text (the
  `is not None` guard in real code prevents it). Equality (`==`/`!=`)
  with None is NOT a panic: Python answers `False`/`True`, and the
  Option LHS comparison unwraps the inner value accordingly (round 43).
- An exception escaping a lambda body.
- `in` on a boxed value whose member is not a container (`1 in boxed_int`),
  or a non-str probe on the boxed str member — CPython 3.11's TypeError
  text, but a panic (the boxed-value iteration precedent).
- Access THROUGH an `Option`-typed receiver (`self.timeout.
  connect_timeout()` where the field is `Timeout | None`) unwraps the
  Option — CPython's AttributeError on a None receiver, but a panic with
  CPython's message. Guarded access (`if x is not None:`) is unaffected:
  the unwrap only fires if the guard lied.
- numpy shape mismatches through the OPERATOR spelling (`a + b` on
  arrays of different shapes). The function spelling `np.add(a, b)`
  raises a catchable `ValueError`; the operator traits have no fallible
  form yet, so that spelling panics with the same message.
- Returning an EXTERNAL-MODULE value from a typed (non-`PyValue`)
  function — `return files("certifi").joinpath("cacert.pem").
  read_text("ascii")` (certifi's `contents()`, an importlib.resources
  chain) or `return _CACERT_PATH` where the global is a boxed `None`-
  initialized mutable static (certifi's `where()`): the value genuinely
  cannot exist in the generated crate (the call chain was dropped at
  conversion time with a `-W` warning), so the return is a panic at the
  exact point of divergence, never a plausible-looking placeholder
  (round 51; certifi 2025.1.31 measures **0** rustc errors).
- Module-level `if sys.version_info >= (3, N):` gates are decided at
  conversion time (rython targets 3.11.0) and the taken branch's
  statements are spliced into the module body — a version-gated `def`
  is a module item, and the dead branches are dropped with a `-W`
  warning (round 51; certifi's core.py).

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
| Argument-render-then-mutate shapes (`print(xs, xs.pop(), xs)`) render the first argument before the mutation | Recorded in issue #79 |
| A read of a module member the generated module has no item for (`util.ssl_.PROTOCOL_TLS` — an external ssl constant) lowers to the boxed `None` with a warning (dynamic-module-member divergence) | Model limit; module members are static path items |
| A call through a sibling-module member that is not a module-level function/class (`probe.acquire_and_get`, a bound-method alias) is dropped with the callable-as-value warning | Model limit; callables cannot be runtime values |
| Release-mode integer overflow may wrap (debug panics) | Bounded by §12.2's contract |
| A non-daemon thread never joined is joined when its LAST handle drops (at latest, end of `main`) — CPython joins at interpreter exit, so a fire-and-forget thread can block a scope exit earlier than CPython would | Model limit; the common create/start/join shape is identical |
| A thread's unhandled exception prints CPython's header and final exception line but no traceback frames | Model limit (no frames) — same family as §8's messages |
| TCP `socket.bind()` binds AND starts listening (std::net has no half-bound TCP socket); a connection can be accepted by the OS before `listen()` runs, and binding a client socket before `connect()` is a loud error | Model limit of the std::net backend |
| `URLError`'s reason text inside CPython's `<urlopen error …>` shape is the HTTP backend's wording, not CPython's | Model limit of the wrapped-crate convention (§10.2.1) |
| `HTTPError` carries CPython's message but not the error response's body/headers (exceptions are string-tagged values) | Model limit; §8.1 representation |
| numpy reductions on integer and bool arrays return `float` (`np.sum(np.array([1, 2, 3]))` prints `6.0`, not `6`) — `NdArray`'s dtype is a runtime value, so one static return type must serve every dtype | Model limit; `np.all`/`np.any`/`np.argmax`/`np.argmin` are single-typed in numpy too and match exactly |
| `np.linalg.det`/`inv`/`solve` differ from LAPACK in the last bits (rython's decomposition is not LAPACK's) | Model limit; results agree to floating-point method differences, not bit-for-bit |
| `np.dot` returns an ARRAY for the 1-D x 1-D case unless both operands are provably 1-D at conversion time, where numpy returns a scalar; the printed form is identical either way, so only arithmetic on the result differs | Model limit of one static type per expression (issue #206) |
| numpy `RuntimeWarning`s (integer divide by zero, invalid value) are not emitted; the VALUES match numpy exactly | Model limit; no `warnings` machinery on the numpy path |
| Verified stdlib divergences (json/defaultdict ordering, `math.remainder`, `strftime` edge cases, `glob` paths, `pathlib` edges, `string.Template`, …) | Tracked as defects in issue #82 |
| `functools.singledispatch` picks the FIRST registered type the argument matches, not CPython's MRO walk; a registration on a base class followed by one on its subclass resolves to the base | Model limit (issue #181); disjoint concrete registrations — what real code writes — agree exactly |
| A CLASS NAME in value position lowers to its NAME STRING (`[ChecksumError]` → `vec!["ChecksumError".to_string()]`; `pool_classes_by_scheme` → `PyDict<String, String>`) — the class object's only runtime-relevant data, since exceptions are string-tagged; identity comparisons of class values compare names, and a dynamic `except <boxed value>:` matches the strings | Model limit (issue #137 round 33); the class's runtime attributes and hierarchy beyond exact-name matching are unmodeled, and a call THROUGH an indirect class value (`pool_cls(...)` read from the dict) fails in rustc (§12.1) |
| A `type(x).__name__` on a non-`self` receiver lowers through the boxed value's runtime type name, and on an inferred generic parameter is dropped as the boxed None | Model limit; `type(self).__name__` is exact |
| The default object repr of a class instance (`str(x)` without `__str__`/`__repr__`) prints `<module.ClassName object>` — CPython appends `at 0x…`, a nondeterministic address (CPython's own output varies run to run) that rython cannot model | Model limit (round 34); a `__str__`/`__repr__` that raises aborts loudly (the §12.2 raise-in-infallible family), and a class whose `__str__` uses `type(self).__name__` shows the defining class's name |
| Old-style `%`-formatting of a DICT as a positional value (`"%s" % {...}`) raises "not enough arguments" instead of CPython's dict repr — the mapping form (`%(name)s`) is exact | Model limit (round 34); positional-vs-mapping mixing follows CPython's rejection |

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
- **I/O and stdlib**: binary DISK file modes, `seek`/`tell`,
  file-based `json`; continued module expansion against the issue #82
  register (threading's `Condition`/`Barrier`/`queue.Queue`, socket
  `setsockopt`, urllib POST/`Request` objects are the known next
  edges of §10.5).
