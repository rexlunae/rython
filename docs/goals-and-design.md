# Rython: Goals and Design

This document formalizes what Rython is trying to be, what it deliberately
is not, and the design principles that every feature — and every rejected
feature — is measured against.

Companion documents:

- [`spec.md`](spec.md) — the Rython language specification: exactly what
  subset of Python is accepted and what Rust it lowers to.
- [`cpython-vs-rython.md`](cpython-vs-rython.md) — the observable
  differences and engineering tradeoffs between CPython and Rython.
- [`porting-guide.md`](porting-guide.md) — a practical guide (written for
  AI agents, useful to humans) for translating Python projects to Rust
  with the Rython toolchain.
- [`context-awareness.md`](context-awareness.md) — design notes for the
  type-inference and coercion layer inside the code generator.

## What Rython is

**Rython is a statically compiled subset of Python.** It is not a new
syntax: every Rython program is a valid Python program, parsed by Python's
own grammar. The Rython toolchain translates that program into Rust source
code that depends on one runtime crate (`stdpython`), and `cargo` compiles
it into a native binary or library. There is no interpreter, no bytecode,
no GIL, and no Python installation at runtime.

The name covers both the language subset and the toolchain that enforces
it:

| Piece | Role |
|---|---|
| `python-ast` | Parses Python source and transpiles it to Rust |
| `stdpython` | The Python-semantics runtime: builtins, types, stdlib, exceptions |
| `rypip` | pip-analog: converts/builds/installs Python packages as Rust crates |
| `rythonc` | Single-file compiler: `input.py → output.rs` |
| `python-mod` | Proc-macros that compile `.py` files into a Rust crate at build time |
| `rykernel-shim` | Support crate for the Linux kernel-module target |

## Goals

### 1. Migration: a permanent, ownable port

`rypip convert` exists to move a project *out* of Python. The output is a
standalone Rust crate meant to be read, refactored, and owned from then
on — not an opaque compilation artifact that must be regenerated from the
Python source forever. This drives a hard requirement on the code
generator: **the generated Rust must be readable.** Each Python module
becomes a Rust module, functions keep their names and signatures,
docstrings become doc comments, and the translation prefers plain Rust
constructs (`struct`, `Vec`, `for` loops, `Result`) over an interpreter-ish
object soup.

### 2. Embedding: Python syntax inside Rust projects

The `python_module!` proc-macro compiles `.py` files directly into a Rust
crate at build time. A Rust project can keep algorithm-heavy or
domain-expert code in Python syntax while everything compiles to native
Rust — callable from Rust like any other module, with zero runtime
bridging. The reverse direction exists too: `rypip convert --pyo3` wraps a
converted crate in PyO3 bindings so existing CPython code can import the
fast version during an incremental migration.

### 3. Correctness: a compiler that refuses to guess

Conversion is itself a correctness tool. Running a Python program through
Rython either produces a program with CPython's exact observable behavior,
or it fails with an error saying precisely what is not supported. This is
the project's prime directive (see *Design principles* below), and it is
what distinguishes Rython from "Python-flavored" compilers that accept
everything and quietly change the semantics of the parts they can't do.

### 4. Reach: places CPython cannot go

Because the output is ordinary Rust with a tiered runtime, Rython programs
can target environments Python never could:

- **`no_std` / embedded / wasm** — `rypip convert --no-std` produces a
  `#![no_std]` crate on `stdpython`'s `alloc` tier: heap-backed Python
  semantics (strings, lists, dicts, json, itertools, …) with no OS
  dependency, checkable on bare-metal targets like `thumbv7m-none-eabi`.
- **Linux kernel modules** — the kernel target (including
  `--rust-for-linux`) compiles Python source into a loadable kernel
  module, with a `kmalloc`-backed allocator, `printk`-based printing, a
  loud rejection of floating point, and generated Kbuild plumbing.

These targets are not a gimmick: they are the proof that the language
boundary is real. A subset with loudly-enforced edges can be retargeted;
"all of Python" cannot.

### 5. Performance as a consequence, not a headline

Converted programs are ahead-of-time compiled, statically typed,
monomorphized by rustc, and free of the interpreter loop and the GIL.
Rython does not chase benchmark numbers with unsafe tricks; the speed
comes from the model. Where CPython's semantics force a cost (insertion-
ordered dicts, `i64` overflow checks lowered as panics, exact float
formatting), Rython pays it, because correctness outranks speed.

## Non-goals

Naming these is as load-bearing as the goals; each one has been
deliberately rejected, not merely postponed.

- **Full Python compatibility.** Rython will never run arbitrary PyPI
  code. `eval`/`exec`, metaclasses, monkey-patching, dynamic imports,
  `globals()` mutation — the reflective machinery that makes Python
  Python-the-dynamic-language — is out of scope permanently, because it
  requires an interpreter at runtime, and "no interpreter at runtime" is
  the point.
- **Being a faster Python runtime.** Rython is not a JIT, not a tracing
  optimizer, and not a drop-in `python3` replacement. Projects that need
  all of Python, faster, should look at PyPy or CPython's own JIT work.
- **Approximate semantics.** There is no "close enough" mode. A construct
  whose CPython behavior cannot be reproduced exactly is a conversion
  error, never a best-effort translation. This includes cases where the
  approximation would almost always be right (e.g. chained assignment to
  a container literal, issue #104): *almost always* is precisely the kind
  of bug that escapes review.
- **Silent bignum fallback.** `int` is `i64`. A future opt-in bigint tier
  is on the roadmap, but transparently switching representations based on
  value magnitude — with its pervasive performance and layout
  consequences — is not.
- **Hiding the Rust.** The generated crate is the deliverable. Rython
  does not wrap the output in a launcher, a bundle format, or a build
  system of its own; it emits a crate and gets out of cargo's way.

## Design principles

### P1. Correct or loud — never silently different

The prime directive. For every construct, exactly one of the following
holds:

1. It converts, and the resulting program's *observable behavior* matches
   CPython's.
2. Conversion fails with an error naming the construct, the reason, and
   (where applicable) the variable and line.
3. It converts into code that raises a typed, catchable Python exception
   at runtime, exactly where CPython would raise one — or, for the few
   cases that are only detectable at runtime and not representable
   (`i64` overflow, sorting `NaN`), into a loud panic.

What is never acceptable: output that differs from CPython's without an
error. When a bug of that shape is found, it is treated as the highest-
severity class of defect (see issue #79's "where it *is* silent" section
for the register kept of such cases).

A corollary: **an ugly loud error beats a beautiful wrong answer.** The
chained-assignment guard (#104) rejects `a = b = []` outright rather than
emit clones that would silently break Python's aliasing — even though the
clone version would compile and usually "work".

### P2. Pin to CPython, byte for byte

"Matches CPython" is defined aggressively: not "the same number", but the
same *bytes on stdout*. `str(1e16)` is `1e+16`; `repr` of floats uses
shortest-roundtrip formatting; `hash()` matches CPython under
`PYTHONHASHSEED=0`; sort is stable; dicts iterate in insertion order;
exception messages match CPython's text. The end-to-end test suite runs
the generated binary and `python3` on the same program and diffs the
output line for line. Divergences that are discovered but not yet fixed
are tracked publicly as bugs (issue #82), not reclassified as acceptable.

Pinning to bytes is what makes the loud-boundary promise *testable*: a
weaker promise ("semantically equivalent") degrades into judgment calls.

### P3. The boundary is enforced, not documented

Unsupported constructs are rejected by the compiler, not merely listed in
a README. A decorator the translator doesn't understand is a conversion
error — never silently ignored, even though ignoring it would let more
programs "convert". Every entry in the supported-features list is backed
by tests; every known gap either fails conversion or raises at runtime.

### P4. Static types from annotations plus local inference

Rython does not do whole-program type inference, and it does not require
annotations everywhere. The model is:

- **Annotations are trusted and required at boundaries** — function
  parameters and returns drive signature generation.
- **Locals are inferred bottom-up** from literals, calls, and operators,
  with a small context-aware coercion layer that inserts the conversions
  Rust needs (`usize → i64` at `range(len(x))`, `&str → String` for
  computed dict keys, `.clone()` for reused move-prone values). See
  [`context-awareness.md`](context-awareness.md).
- **One name, one type.** Reassigning a variable to a different type, or
  a heterogeneous list, is a conversion error — not because it couldn't
  be modeled (a boxed enum could), but because the output would stop
  being the readable Rust that goal 1 demands.

### P5. Value semantics with a guarded aliasing boundary

Generated Rust uses value semantics for containers. Python's reference
semantics (`b = a` aliasing the same list) is *not* modeled — and the gap
is guarded rather than papered over: most aliasing shapes fail to compile
(Rust's move checker makes them loud), the known-silent shapes are
tracked as bugs, and the fix under consideration (conversion-time
aliasing detection, or an opt-in `Rc<RefCell<…>>` model) is issue #79.
The alternative — making every container an `Rc<RefCell<…>>` by
default — was rejected because it would tax every program and destroy
the readability of the output for the vast majority of code that never
aliases.

### P6. A tiered runtime: `core ⊂ alloc ⊂ std`

`stdpython` is layered as a feature ladder. The `std` tier has the full
surface. The `alloc` tier keeps everything that doesn't need an OS and
works on embedded targets. Constructs that need a missing tier fail *at
conversion time*, with a Python-level message — not later as inscrutable
rustc errors inside a generated crate. The kernel target builds on the
same discipline with its own stricter rules (no floats, `printk`
printing, `kmalloc` allocation).

### P7. Interop is explicit and compile-time

Crossing the language boundary never happens implicitly:

- Python code binds external Rust crates through declared bindings
  (`rust.bind`, and `rython.toml`'s `[rust-modules]` table mapping
  imports to crates) resolved at conversion time.
- Rust code binds Python modules through `python_module!` at build time.
- CPython binds converted crates through generated PyO3 wrappers behind
  an explicit cargo feature.

There is no runtime FFI discovery, no dynamic loading, no implicit
marshalling. Every boundary crossing is visible in the source and checked
by the compiler.

### P8. Grow the subset deliberately

Features are added when they can be added *whole* — with CPython-pinned
tests, loud edges, and readable output — and not before. The roadmap
(generalized decorators, generators as iterator-struct lowering,
inheritance and dunder protocols, an opt-in bigint tier, binary file
modes) is ordered by what unblocks real programs, but no feature ships in
a half-state where some uses silently misbehave. "Supported" is a binary
property of a construct, and the test suite is its definition.

## How to evaluate a proposed feature

A checklist distilled from the principles, for contributors (human or
agent) weighing an addition:

1. Can its CPython behavior be reproduced exactly — including error
   messages, ordering, and formatting? If only mostly: can the
   non-reproducible cases be detected and rejected loudly (at conversion
   time if possible, as a typed exception if not)?
2. Does the generated Rust remain something a Rust programmer would
   accept in review?
3. Which runtime tier does it belong to, and does it fail loudly on
   tiers that can't support it?
4. Does it keep working under the kernel and `no_std` targets' rules, or
   is it correctly rejected there?
5. Is there an end-to-end test that diffs against `python3`?

If the answer to (1) is no on both counts, the feature is not added — it
is documented as a boundary instead.
