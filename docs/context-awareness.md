# Context-Aware Codegen: type inference and coercion in rython

Status: design + implementation notes for the open-issues roundup
(branch `fix/open-issues-roundup`).

## Problem

Python code through rython compiles to Rust, and Rust is statically typed
while Python is not. Several open issues are different faces of the same
gap:

- **#100** `range(len(x))` — `len()` returns `usize`, `range()` takes `i64`
- **#76** `d = {"a": 1}; d[k("a")] += 10` — literal keys are `&str`,
  computed keys are `String`
- **#77** `e = {}` — no way to infer the element type of an empty literal
- **#102** `forward(x); forward(x)` — params are owned values, so reuse
  moves the variable

The fix the issues (and the maintainer's comment on #76) ask for: a layer
that *knows the type context of an expression* and inserts the manual
conversion or explicit type where Rust needs it.

## Design: `type_ctx.rs`

A new module in `crates/python-ast/src/ast/tree/type_ctx.rs` with four
pieces:

### 1. `TypeInfo` — a small lattice of the Rust types codegen produces

```
Int (i64), Float (f64), Bool, StrRef (&str), String, Usize,
Bytes, Vec(T), Dict(K, V), Tuple([..]), Option(T), Range, NdArray,
PyObject (opaque/unknown)
```

`is_copy()` distinguishes move-prone types (String, Vec, Dict, NdArray,
Option, Tuple of non-Copy) for the reuse-clone rule.

### 2. `infer_type(expr, options, symbols)` — bottom-up syntactic inference

- Constants: int/float/bool/str literal kinds.
- `Name`: the existing `local_types` annotation map, then the symbol
  table's `Assign` value, then `PyObject`.
- Calls: `len()` → `Usize` (the key for #100), `range()` → `Range`,
  `str()`/f-strings → `String`, `dict.get()` → `Option`, numpy fns →
  `NdArray`, user functions → `PyObject` (callee signature lookup is
  future work).
- BinOps/comparisons/unary: numeric/string/bool propagation.
- `List`/`Dict`/`Tuple` literals: element types from their contents.
- `Subscript`: element type of the receiver's container type.

### 3. `coerce_tokens(tokens, from, to)` — the conversion matrix

| from \ to | String    | &str    | i64        |
|-----------|-----------|---------|------------|
| StrRef    | `.to_string()` | —   | —          |
| String    | —         | `.as_str()` | —      |
| Usize     | —         | —       | `.try_into().unwrap()` |
| Int       | —         | —       | — (or `as f64` in numeric-unification contexts) |

Rules: never coerce silently when the conversion is lossy (#77-style loud
errors for genuinely incompatible mixes); `usize→i64` uses `try_into()`
so overflow panics loudly rather than wrapping.

### 4. Per-function analysis (stored in `PythonOptions` as `Rc` fields)

- `use_counts: Rc<HashMap<String, usize>>` — read-use counts per name.
  When a `Name` with count > 1 is rendered in a move-prone position (call
  argument, container element), emit `x.clone()` unless the type is Copy
  (#102). Method-call receivers and subscript receivers are NOT wrapped,
  so `xs.pop(); xs.pop()` keeps mutating the same vector.
- `name_types: Rc<HashMap<String, TypeInfo>>` — inferred type of each
  local, from annotations first, then assignments.
- `empty_pinned: Rc<HashMap<String, TypeInfo>>` — names bound to empty
  `[]`/`{}` whose element/key types were pinned by later use (`append`,
  `push`, `extend`, `insert`, `[k] = v`, `[k]`, `for x in name`,
  `x in name`, `pop`, `get`). Rendering the empty literal then emits
  `Vec::<T>::new()` / `PyDict::<K,V>::from([])` (#77). If nothing pins
  the type, the assignment is a loud conversion-time error naming the
  variable and line. Dict keys normalize to `String` (never `&str`) so
  literal-keyed dicts match `dict[str, V]` annotations; codegen owns
  literal keys at `d[k] = v` stores and get/pop/setdefault/`in` call
  sites via `render_typed(expected=String)`.

### Choke points wired up

- `expression.rs` List arm: unify element types (`["a", k()]` →
  all `String` via `.to_string()` on the literals; `[1, 2.0]` → all
  `f64`; `["a", 1]` → loud error).
- `dict.rs`: same unification for keys and values.
- `call.rs`: `range(...)` args coerced to `i64`; generic call args get
  the reuse-clone rule; numpy args already clone.
- `subscript.rs` / `aug_assign.rs`: index coerced to the receiver's key
  type (`Vec(T)` → `i64`, `PyDict(K, V)` → `K`).
- `assign.rs`: empty-literal targets consult `empty_pinned`.
- `stdpython`: `range`/`range_start_stop`/`range_start_stop_step` become
  generic over `TryInto<i64>` as a belt-and-braces API fix for #100.

## What this deliberately does NOT do

- No full Hindley-Milner inference: the lattice is syntactic and the
  expected types come from the immediate context, not a unification pass.
- No `Rc<RefCell<...>>` value-semantics model (#79): aliasing is out of
  scope here; the reuse-clone rule fixes the *compile error* half of #102
  and #79 without changing the value model.
- Callee-side borrow analysis (emit `&[T]` params for read-only params) is
  left as a future optimization; the caller-side clone is sufficient to
  make the reported programs compile.
