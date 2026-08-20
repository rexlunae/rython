# CPython vs. Rython: differences and tradeoffs

Rython pins its observable behavior to CPython's — a converted program
prints the same bytes CPython would print — but it gets there with a
completely different execution model, and the differences that remain are
deliberate engineering tradeoffs, each with a loud edge. This document
lays out both: what is different by construction, and what each
difference buys and costs.

For the normative statement of what converts and what doesn't, see
[`spec.md`](spec.md). For why the boundaries are drawn this way, see
[`goals-and-design.md`](goals-and-design.md).

## Execution model

| | CPython | Rython |
|---|---|---|
| Execution | Bytecode interpreter | Native code, ahead-of-time compiled by rustc |
| Type checking | At runtime, per operation | At conversion + compile time |
| Memory management | Reference counting + cycle GC | Ownership/RAII; no GC, no refcount traffic |
| Concurrency ceiling | GIL serializes Python bytecode | No GIL; ordinary Rust threads possible in the output |
| Startup | Interpreter + imports at every launch | Native binary startup |
| Distribution | Requires a Python installation | Self-contained binary or library |
| Dynamism | `eval`/`exec`, monkey-patching, reflection | None at runtime; everything resolved at conversion time |
| Errors | All at runtime | Unsupported constructs at conversion time; Python-level errors as typed exceptions at runtime |

The rest of this document walks through the consequences.

## What is identical by contract

Within the supported subset, Rython reproduces CPython's observable
behavior **byte for byte**, and the end-to-end test suite enforces it by
diffing generated-binary output against `python3` line for line:

- **Float formatting**: `str`/`repr` use CPython's shortest-roundtrip
  algorithm; `str(1e16)` is `1e+16`, not `10000000000000000`.
- **`hash()`**: matches CPython with `PYTHONHASHSEED=0`.
- **Dict/set ordering**: dicts iterate in insertion order, like CPython.
- **Sort stability** and comparison behavior.
- **Exception types and messages**: `except ValueError` catches exactly
  what CPython's would, and the message text matches.
- **Stdlib output shapes**: `argparse` help/error text, `csv` quoting,
  `datetime.strptime`, `random` with a given seed (MT19937,
  operation-for-operation), `heapq`'s exact list layout, and so on.

Where a not-yet-fixed divergence is found in the stdlib, it is tracked as
a public bug (issue #82), not shrugged off. The contract is the point:
"works" means "diffs clean against CPython", not "looks right".

## The deliberate differences

Each of these is a tradeoff taken knowingly. The pattern to notice: in
every case, the failure mode is *loud* — a conversion error, a typed
exception, or a panic — never a silently different answer.

### Integers: `i64`, not arbitrary precision

CPython's `int` grows without bound; Rython's `int` is a 64-bit signed
integer, and overflow **panics** rather than wrapping or growing.

- **Bought**: machine-speed arithmetic, `Copy` semantics, no allocation
  for numbers, viability in `no_std` and kernel contexts.
- **Cost**: programs that rely on bignums (cryptography, combinatorics,
  hashes rolled by hand) don't port. An opt-in bigint tier is on the
  roadmap; a *silent* fallback is a non-goal.
- **Edge**: the panic is the loud boundary. Wrapping would have been a
  silent divergence; checking and growing would have been a semantic and
  performance fork from the declared model.

### Static types: one name, one type

CPython lets any name rebind to any type and any list hold any mix.
Rython requires each variable to keep one type, infers locals from
literals and operations, and takes function signatures from annotations.
Heterogeneous lists and cross-type rebinding are conversion errors.

- **Bought**: readable generated Rust with real types (`Vec<i64>`, not a
  boxed any-type), monomorphized performance, rustc as a second checker.
- **Cost**: idiomatic-but-dynamic Python patterns (a variable that's an
  `int` then a `str`, a list of mixed records) must be refactored before
  converting. In practice this is the largest source of porting work —
  see the [porting guide](porting-guide.md).

### Values, not references: the aliasing boundary

In CPython, containers are references — `b = a` makes two names for one
list, and mutation through either shows through both. Rython's generated
Rust uses value semantics: assignment moves (or clones, for reused
values). Python's aliasing is **not modeled**.

- **Bought**: generated code that reads like Rust, no `Rc<RefCell<…>>`
  tax on every container operation in the 95% of code that never aliases.
- **Cost**: alias-and-mutate patterns don't convert. Rust's move checker
  makes most such shapes fail to compile (loud), and the shapes that
  could slip through silently are guarded at conversion time (chained
  assignment to a container literal is refused outright, issue #104) or
  tracked as the highest-severity class of bug (issue #79).
- The long-term options — conversion-time aliasing detection, or an
  opt-in shared-mutability lowering — are an open architectural decision,
  recorded in issue #79.

### Exceptions: same surface, different plumbing

Python semantics — `try`/`except`/`else`/`finally`, typed handlers,
matching by exception class — are reproduced, including `ZeroDivisionError`
raised from `//` and `%` at the exact point CPython raises it. Underneath,
there is no stack unwinding through an interpreter: fallible operations
return `Result<T, PyException>` and `?` propagates to the nearest handler.

- **Bought**: zero-cost when no exception is raised, and exception flow
  visible in the generated code.
- **Cost**: conditions Rust cannot represent as catchable values remain
  panics (integer overflow, sorting `NaN`) — these are documented
  boundaries, not exceptions you can catch.

### No runtime dynamism

`eval`, `exec`, `compile`, dynamic imports, `globals()`/`locals()` as
writable dicts, monkey-patching, metaclasses: none of these exist in
Rython, permanently. Everything a program does is resolved at conversion
time. Some of CPython's *conversion-time-decidable* dynamism is kept by
evaluating it during conversion instead: `argparse` parser construction
from literal specs, `functools.partial` over statically known functions,
f-string and `str.format` templates that are literals.

- **Bought**: the entire "no interpreter at runtime" premise.
- **Cost**: plugin registries, config-driven class loading, and
  decorator-heavy metaprogramming must be restructured (usually into
  explicit dispatch) before porting.

### Concurrency: no GIL, but also no `threading` yet

CPython's GIL caps Python-level parallelism; Rython's output has no GIL
and no interpreter, so nothing structural prevents real parallelism.
However, Python's `threading`/`multiprocessing`/`asyncio` surfaces are
not currently part of the supported subset — today, concurrency is
something you add *in Rust* after converting (the generated crate is
yours to extend), not something you port from Python source.

### Performance profile

Broad strokes, since the point of Rython is not benchmarks:

- **Faster by model**: no interpreter dispatch, no per-object heap boxing
  for ints/floats/bools, monomorphized generics, no refcount traffic,
  native startup. Numeric and string-processing code typically lands in
  the range of hand-written Rust using equivalent data structures.
- **Deliberately unoptimized**: Rython pays for CPython pinning where it
  must — insertion-ordered dicts instead of the fastest hash map, exact
  CPython float formatting, checked division for `ZeroDivisionError`,
  clones inserted for reused values where Rust's moves would otherwise
  bite. Correctness outranks speed, always.
- **Not free**: rustc compile times replace CPython's zero-compile edit
  loop. The development cycle is Rust's, not Python's.

### Deployment reach

This is a difference in kind, not degree. CPython needs an OS, a Python
installation, and dynamic memory; Rython's output needs whatever tier of
the runtime it uses:

- **`std` tier** — full surface, ordinary OS targets.
- **`alloc` tier** (`--no-std`) — strings, collections, json, itertools,
  functools, hashlib, csv and more with *no OS dependency*: embedded and
  wasm targets. OS-touching constructs (`print`, `open`, `os`, `sys`,
  `datetime`, `random`, …) are conversion-time errors under this tier.
- **Kernel target** — Python source compiled into a loadable Linux
  kernel module: `printk`-backed printing, `kmalloc`-backed allocation,
  floating point rejected loudly, Kbuild files generated.

### Interop

- **CPython → native** is dynamic: `ctypes`, C extensions, runtime
  loading. **Rython → Rust** is static: `rust.bind` and `rython.toml`'s
  `[rust-modules]` table bind Python imports to Rust crates at conversion
  time, type-checked by rustc in the generated crate.
- The bridge back to CPython exists as an explicit build artifact:
  `rypip convert --pyo3` produces a crate that CPython can import as an
  extension module — the supported path for incremental migration, where
  the converted core speeds up an otherwise-unconverted Python program.

## Choosing between them

Use **CPython** when the program leans on what Rython excludes: bignums,
runtime dynamism, heavy aliasing, async, the long tail of PyPI, or when
the edit-run loop matters more than the shipped artifact.

Use **Rython** when the program is (or can be made) annotation-typed and
subset-clean, and you want one of: a permanent Rust port, a native
self-contained binary, Python-syntax modules inside a Rust codebase, a
fast extension module for CPython, or Python source running where Python
cannot go — embedded, wasm, or the kernel.

Use **both, during migration**: convert the hot, subset-clean core with
`--pyo3`, keep the dynamic shell in CPython, and move the boundary over
time. The [porting guide](porting-guide.md) describes that workflow in
detail.
