# Porting Python to Rust with Rython

Operating instructions for translating a Python project into Rust using
the Rython toolchain. This guide is written so an AI coding agent can
follow it end to end; it works just as well for humans.

Prerequisites: skim [`spec.md`](spec.md) for what converts, and
[`cpython-vs-rython.md`](cpython-vs-rython.md) for what changes when it
does. The one-line mental model: **the converter refuses to guess** —
your job is to move the Python program into the supported subset while
keeping it a working Python program, then let the toolchain do the
translation.

---

## 0. The core loop

```
┌─▶ 1. run the program under CPython, capture golden output
│   2. rypip convert
│   3. conversion error?  → refactor the *Python*, re-verify under CPython, goto 2
│   4. cargo build error? → usually an aliasing/type shape; refactor the Python, goto 2
│   5. run the binary, diff its output against the golden output
└── 6. differences? → treat as a rython bug (file it) or a §12-listed deviation
    7. clean diff → the port is done; the Rust crate is now the source of truth
```

Iron rules for the loop:

- **Refactor the Python, not the generated Rust**, until step 7. The
  generated crate is disposable on every iteration; your durable work
  product during the loop is a Python program that (a) still passes its
  own tests under CPython and (b) converts cleanly. Only after the diff
  is clean do you take ownership of the Rust and edit it directly.
- **Keep CPython as the oracle at every step.** Every refactor must
  leave behavior identical under `python3`. If you can't verify a
  refactor under CPython, you can't verify the port.
- **Never suppress a loud error by deleting the feature it points at**
  unless you have confirmed the program doesn't need it. The error text
  usually names the supported rewrite; prefer that.
- **A silent difference in step 6 is a bug, not your problem to absorb.**
  Rython's contract is "correct or loud" — if the binary's output
  differs from CPython's with no conversion error, that is a
  highest-severity rython defect. File it with a minimal reproducer
  (unless it is one of the known deviations in `spec.md` §12).

### Capturing golden output deterministically

```bash
PYTHONHASHSEED=0 python3 -m your_package args... > golden.txt 2> golden.err
```

- Set `PYTHONHASHSEED=0` — rython pins `hash()` to that seed.
- Seed `random` explicitly in the program if it uses randomness (seeded
  sequences are bit-identical between CPython and rython).
- Pin time-dependent output (inject timestamps as parameters) — there is
  no way to diff wall-clock output.

---

## 1. Pre-flight audit

Before converting anything, inventory the codebase for constructs that
will need refactoring. Grep for these; each maps to a rewrite in §2.

**Hard blockers (no supported rewrite — redesign required):**

| Pattern | Why |
|---|---|
| `eval(`, `exec(`, `compile(`, dynamic `__import__` | No runtime dynamism, permanently |
| `getattr(`/`setattr(` for dynamic dispatch | No reflection; use explicit dispatch |
| Metaclasses, `__init_subclass__`, class decorators | No dynamic class machinery |
| Monkey-patching (assigning to modules/classes at runtime) | Nothing to patch at runtime |
| Arbitrary-precision integer reliance (values beyond ±2⁶³) | `int` is `i64`; overflow is a panic |
| `threading`, `multiprocessing`, `asyncio` as APIs | Not in the subset; add concurrency in Rust after porting |

**Refactorable (mechanical or semi-mechanical rewrites, §2):**

- `yield` / generator functions; generator expressions relied on for
  *laziness* (they convert, but eagerly)
- `*args` / `**kwargs`; starred call unpacking `f(*xs)`
- Decorators other than `functools.lru_cache`/`cache` (including
  `@property`, `@dataclass`, `@staticmethod`, `@classmethod`)
- Inheritance, `super()`, dunder methods (`__repr__`, `__eq__`,
  `__len__`, operator overloading, …)
- Heterogeneous collections; variables rebound to different types
- Unannotated function signatures
- Aliasing: `b = a` on a container then mutating; functions that mutate
  their arguments as their contract
- `match`, `del`, `global`, `nonlocal`
- Mutable default arguments
- Binary file I/O, `seek`/`tell`, `json.dump`/`load` on file objects
- `argparse` beyond literal specs (short options, `nargs`, `choices`,
  subcommands)
- Sets whose contents get printed (set `repr` is deliberately absent)

Estimate honestly: a codebase saturated with the first table is not a
porting candidate — recommend the `--pyo3` incremental strategy (§4)
for its hot core instead, or advise against porting.

---

## 2. Refactoring Python into the subset

Every rewrite below keeps the program valid, behavior-identical Python.
Apply them before or between conversion attempts, re-running the
program's own tests under CPython each time.

### 2.1 Annotate everything at function boundaries

Parameters without annotations generate uncallable Rust. Annotate every
parameter and return: `int`, `float`, `str`, `bool`, `bytes`,
`list[T]`, `dict[K, V]`, `set[T]`, `Optional[T]`. Bare `list`/`dict`
annotations are rejected — always subscript. Annotate empty-container
assignments the pinning pass can't resolve (`xs: list[float] = []`).

### 2.2 One name, one type

- Split variables that hold different types over their lifetime into
  differently-named variables.
- Replace heterogeneous lists with: a tuple (fixed shape), a class
  (named fields), or parallel typed lists.
- Replace "int or None" sentinel patterns with `Optional[int]` and
  explicit `is None` checks (those convert to `Option`).

### 2.3 Dynamic calling conventions → explicit ones

- `*args` aggregation: pass a `list[T]` explicitly.
- `**kwargs` pass-through: pass an explicit parameter list or a typed
  class.
- Starred unpacking `f(*pair)`: `f(pair[0], pair[1])`.
- Mutable defaults `def f(xs=[])`: use `Optional[list[int]] = None` and
  materialize inside — the classic Python fix is also the rython fix.

### 2.4 Generators → lists or explicit state

- A generator consumed once and fully: return a list.
- A generator for laziness/memory: restructure into a class holding
  explicit iteration state with a `next_batch()`-style method, or
  process in chunks. (Generator *expressions* inside `sum(...)`/loops
  convert, but eagerly — acceptable when side-effect timing and memory
  don't matter; verify.)

### 2.5 Classes: flatten the object model

- Inheritance → composition: embed the "base" as a field and delegate
  explicitly; or duplicate small bases. An ABC used only as an
  interface → keep one concrete class per implementation and dispatch
  explicitly at the call site.
- `@property` → explicit `get_x()` methods (call sites change; do it
  under CPython first).
- `@staticmethod`/`@classmethod` → module-level functions.
- `@dataclass` → write the `__init__` by hand.
- Dunders → named methods: `__repr__` → `describe()`, `__eq__` →
  explicit comparison of fields, `__len__` → `.size()`. Only `__init__`
  has meaning to the converter.
- Class attributes (constants at class level) → module-level constants.

### 2.6 Aliasing and mutation

The value-semantics model (spec §5.1, issue #79) means shared mutable
containers don't convert. Rewrites, in order of preference:

1. **Return instead of mutate**: `def add_item(xs: list[int], x: int) -> list[int]`
   returning the new list, with call sites reassigning (`xs = add_item(xs, x)`).
2. **Mutate through one owner**: keep exactly one name for a container;
   pass indices or copies elsewhere.
3. **Method on a class**: `self.items.append(...)` inside a method is
   fine — `&mut self` handles it.

Never leave `b = a; b.append(...)` shapes in place: some fail loudly in
rustc (fine but confusing), and pathological forms are silent (#79).

### 2.7 Miscellaneous

- `match` → `if`/`elif` chains. `del d[k]` → `d.pop(k)`.
  `global` for mutation → pass state explicitly or use a class.
- Text I/O only; do `seek`-dependent logic by reading fully first.
  `json.dump(obj, f)` → `f.write(json.dumps(obj))`.
- `argparse`: reduce to literal specs — `str`/`int`/`float` positionals,
  `--long` options with `default=`, `store_true`, `help=`, `prog=`,
  `description=`. Give every value-taking option a `default=`.
- Don't print sets; print `sorted(s)` instead (also better Python).
- `lru_cache` keys must be `int`/`bool`/`str`-annotated parameters.
- Avoid `is` on non-`None` operands (it converts as `==` — a listed
  deviation; make equality explicit in the Python).
- Check integer ranges: anything that can exceed ±2⁶³ must be
  redesigned (e.g. use `math` floats, or split the quantity).

---

## 3. Converting

```bash
# Whole package or project (pyproject.toml, flat or src/ layout, or single file):
cargo run -p rypip -- convert path/to/project --out ported-crate

# Then:
cd ported-crate && cargo build && cargo run -- <args>
```

- A module with `if __name__ == "__main__":` (or `__main__.py`) becomes
  `fn main`; without one you get a library crate.
- `--stdpython <path>` / `RYPIP_STDPYTHON_PATH` locate the runtime crate
  when converting outside the rython workspace.
- `-W deny` makes lossy-conversion warnings (dropped parameter
  defaults, ignored fall-through return annotations) fail the
  conversion — recommended for agents: it turns every "mostly fine"
  into an explicit decision.
- Read every conversion error fully. The messages are written to name
  the supported rewrite (e.g. the empty-container error suggests both
  the annotation and the pinning fix). Fix them in the Python, one
  class of error at a time, re-running CPython tests between rounds.

### Error → rewrite quick reference

| Error mentions… | Do |
|---|---|
| `not yet supported by rython` (statement) | §2.7 rewrite for that statement |
| `generators (yield)` | §2.4 |
| `starred unpacking` / `**kwargs` | §2.3 |
| `decorator … refuses to silently ignore` | §2.5 |
| `uses inheritance` / `at class level` | §2.5 |
| `empty container literal has no inferable element type` | annotate (§2.1) |
| `mixes incompatible element types` | §2.2 |
| `has a mutable default` | §2.3 |
| `chained assignment to a container literal` | assign once, copy explicitly |
| `keyword arguments require the callee's signature` | call positionally, or define the callee in-module |
| `unsupported annotation` / `no element/key type` | §2.1 |
| `requires stdpython's std tier` / `requires OS I/O` (no_std) | remove the OS dependency or drop `--no-std` |
| rustc `E0382 borrow of moved value` in the generated crate | aliasing shape — §2.6 |
| rustc `cannot find value` for a module-level name | non-constant module global used inside a function — pass it as a parameter or make it a literal constant |

---

## 4. Choosing a strategy and target

| Situation | Strategy |
|---|---|
| Whole program is subset-clean (or refactorable) | Full port: `rypip convert`, verify, own the crate |
| Large dynamic program with a hot, typed core | Incremental: convert the core with `--pyo3`, `cargo build --features python`, import the extension from the remaining CPython code; move the boundary over time |
| Rust project that wants Python-syntax modules | `python_module!(name)` proc-macro; the `.py` lives in `src/` and compiles into the crate |
| Embedded / wasm, no OS | `rypip convert --no-std`; only the alloc-tier stdlib (`json`, `collections`, `itertools`, `functools`, `heapq`, `textwrap`, `hashlib`, `csv`, `string`, `copy`); no `print`/`open`/`__main__` |
| Linux kernel module | `--kernel-module` (or `--rust-for-linux` for CONFIG_RUST trees): a tiny sub-language — see spec §11.4 before writing any Python |
| Distributable CLI | `rypip install path/to/project` (needs an entry point) |

Constraints to remember: `--pyo3` excludes `--no-std` and kernel
targets; PyO3 export skips functions with defaults, `*args`/`**kwargs`,
or underscore-prefixed names, and fails loudly if nothing is bindable.

### Calling existing Rust from the Python side

When part of the answer is "this piece already exists as a Rust crate",
bind it instead of porting around it:

- `rython.toml` next to the package: `[rust-modules]` maps an import
  name to a crate (`path=` or `version=`); provide a `.pyi` stub with
  string-literal Rust type annotations, or let signatures be inferred
  from the crate's public functions. Then plain `import name` works —
  **under CPython too**, if you also provide a real Python
  implementation or shim, which keeps the oracle intact.
- Or inline: `from rython import rust` + `rust.bind(...)` declarations
  (spec §11.1). Note this form is rython-only Python — it parses under
  CPython but fails at runtime, so prefer `rython.toml` + stub when you
  need the dual-run property.

---

## 5. Verifying the port

1. **Golden diff**: run binary and CPython on the same inputs
   (`PYTHONHASHSEED=0`, seeded randomness), `diff` stdout and stderr,
   compare exit codes. Cover the error paths too — exception messages
   and `argparse` usage errors are part of the pinned surface.
2. **Port the test suite**: the project's own tests are Python programs;
   convert the pure-logic ones and run them as Rust, or at minimum
   re-express their assertions against the binary's CLI.
3. **Property spot-checks** where output is large: hash the outputs of
   both implementations across an input corpus.
4. **Panics audit**: grep inputs/domains for the panic conditions
   (overflow-adjacent arithmetic, NaN sorting, `hash` on floats that
   could be NaN) — these are contract edges, decide explicitly that
   they're unreachable or guard them in the Python.
5. Only then refactor the Rust. From this point the crate is the source
   of truth; keep the golden corpus as regression tests on the Rust
   side.

## 6. Reporting boundary bugs

When step 5 finds a divergence that isn't in spec §12's ledger, file it
against rython with: the minimal Python reproducer, CPython's output,
the binary's output, and the toolchain version. Silent divergences are
the project's highest-severity class — reporting them is part of the
porting workflow, not a detour.
