# CPython vs. Rython: differences and tradeoffs

Rython tracks CPython as its reference implementation: within the
supported subset, the goal is that a converted program prints the same
bytes CPython would print, enforced by transcript tests wherever a
behavior is pinned. But Rython gets there with a completely different
execution model, and it has consciously traded away parts of Python's
semantics — so it does not claim blanket byte-for-byte equivalence.
The differences that remain are deliberate engineering tradeoffs, each
with a loud edge, each recorded, and each a candidate for buying down
over time. This document lays out both: what is different by
construction, and what each difference buys and costs.

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

## What is pinned to the reference

Where a behavior is supported and verified, it is pinned **byte for
byte**: the end-to-end test suite diffs generated-binary output line for
line against transcripts captured from real `python3` runs and checked
in as the oracle. The pinned surface includes:

- **Float formatting**: `str`/`repr` use CPython's shortest-roundtrip
  algorithm; `str(1e16)` is `1e+16`, not `10000000000000000`.
- **`hash()`**: matches CPython with `PYTHONHASHSEED=0`.
- **Dict ordering**: dicts iterate in insertion order, like CPython.
  (Sets deliberately have no `repr` at all — printing one is a compile
  error — so unordered set iteration is never observable output.)
- **Sort stability** and comparison behavior.
- **Exception raising and messages**: exceptions are raised where
  CPython raises them, with pinned message text across the verified
  surface. (Handler *matching* is exact-name plus
  `Exception`/`BaseException` catch-alls — the hierarchy in between is
  not modeled; see the ledger, [spec §12.3](spec.md).)
- **Stdlib output shapes**: `argparse` help/error text, `csv` quoting,
  `datetime.strptime`, `random` with a given seed (MT19937,
  operation-for-operation), `heapq`'s exact list layout, and so on.

Where a not-yet-fixed divergence is found in the stdlib, it is tracked
as a public bug (issue #82), not shrugged off; the deliberate,
model-level differences are recorded in the spec's deviation ledger
([spec §12](spec.md)). Both lists exist to be bought down. The pinned
surface is the point: for it, "works" means "diffs clean against
CPython", not "looks right".

## The deliberate differences

Each of these is a tradeoff taken knowingly. The pattern to notice: in
every case, the failure mode is *loud* — a conversion error, a typed
exception, or a panic — never a silently different answer.

### Integers: `i64`, not arbitrary precision

CPython's `int` grows without bound; Rython's `int` is a 64-bit signed
integer that does not grow. Overflow **panics in debug builds**; in
release builds (which `rypip build`/`install` produce) Rust's unchecked
arithmetic currently wraps silently — a ledgered gap
([spec §12.2](spec.md)), and part of why overflow-adjacent arithmetic
is out of contract until the bigint tier lands.

- **Bought**: machine-speed arithmetic, `Copy` semantics, no allocation
  for numbers, viability in `no_std` and kernel contexts.
- **Cost**: programs that rely on bignums (cryptography, combinatorics,
  hashes rolled by hand) don't port today. The planned fix is an
  **opt-in feature flag** backing `int` with an arbitrary-precision
  crate — which also removes the overflow panic, since bigints don't
  overflow. A *silent*, magnitude-triggered fallback remains a non-goal.
- **Edge**: the debug-build panic is the loud boundary, and the
  release-mode wrap is the known exception to it. Checking-and-growing
  would have been a semantic and performance fork from the declared
  model; the bigint flag is the sanctioned way to buy the Python
  behavior.

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
  see the [porting guide](porting-guide.md). A planned relief valve: an
  **opt-in boxed value type** for the spots where limited dynamism buys
  real compatibility (heterogeneous collections first), behind a flag or
  explicit annotation, with static typing staying the default.

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

Python semantics — `try`/`except`/`else`/`finally`, typed handlers
matched by exception name (`Exception` catches everything; the
hierarchy in between is not yet modeled) — are reproduced, including
`ZeroDivisionError` raised from `//` and `%` at the exact point CPython
raises it (true division `/` does not raise yet — issue #107).
Underneath,
there is no stack unwinding through an interpreter: fallible operations
return `Result<T, PyException>` and `?` propagates to the nearest handler.

- **Bought**: zero-cost when no exception is raised, and exception flow
  visible in the generated code.
- **Cost**: a few conditions currently remain panics rather than
  catchable exceptions (integer overflow, sorting `NaN`, arithmetic on
  `None`, an exception escaping a lambda) — documented boundaries, not
  exceptions you can catch. The direction is to shrink this list into
  the `Result` model wherever CPython defines an exception for the
  case, and the planned bigint tier removes the overflow panic
  altogether ([spec §14](spec.md)).

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
