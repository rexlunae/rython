# The Rython JIT: on-demand compilation (issue #166)

**Status: design. Nothing in this document is implemented yet.** It records
the maintainer's decision that a JIT is a project goal ([`goals-and-design.md`](goals-and-design.md),
Goal 6), the approach chosen for it, and the milestones that build it. The
earlier evaluation, [`jit-eval.md`](jit-eval.md), is superseded by this
document but kept for its reasoning: the preconditions it set out for any
JIT became the requirements below.

Contract: [`AGENTS.md`](../AGENTS.md) (the prime directive: *correct or
loud — never silently different*) and the feature checklist in
[`goals-and-design.md`](goals-and-design.md).

## The decision

Issue #166 asks for a JIT that can be "the basis of eval and other
implementations in stdpython" and that lets users "run Python compiled from
the command line the same way you would with CPython".

The maintainer chose **on-demand Rust compilation followed by in-process
loading**:

1. parse the Python source and convert it to a Rust crate, with the
   converter the project already ships;
2. compile that crate with `rustc`/`cargo` into a shared library;
3. load the library into the running process and call into it.

Compiled results are cached by content, so a source that has been seen
before costs a `dlopen`, not a build.

This is a JIT in the sense of *compiling when the program asks for it*,
not in the sense of a tracing or speculating optimizer. It does not make
Python code faster than the AOT pipeline would; it removes the separate
build step and makes `eval`/`exec`-shaped use cases possible.

## Why this approach

`jit-eval.md` rejected a JIT on four grounds. This design answers each one
directly, and the answers are the reason it was chosen over a bytecode
interpreter or a second code generator.

**One execution model, one conformance surface.** The code that runs is the
code the AOT pipeline would have produced for the same source: the same
`python-ast` lowering, the same `stdpython` runtime, the same loud
refusals. There is no interpreter loop, no bytecode, and no second
implementation of Python's semantics to keep in step. Every transcript the
project already pins through the AOT path is evidence for the JIT path too,
and the differential gate below turns that into a checked-in measurement.

**The artifact is unchanged.** Converted output still contains no
interpreter and needs no Python installation. The JIT is a *tool* that runs
the conversion at runtime; it is not something converted programs carry
(until, and unless, a program explicitly opts in; see milestone M5).

**The subset is unchanged.** The JIT compiles exactly the language subset in
[`spec.md`](spec.md). Metaclasses, monkey-patching, dynamic imports and
`globals()` mutation stay out of scope, because the converter cannot
express them, not because a JIT could not.

**It is std-tier only, and says so.** A JIT needs a filesystem, a Rust
toolchain and a loader, so it exists only on the `std` tier. The `no_std`
and kernel targets keep rejecting the constructs that need it at conversion
time (checklist questions 3 and 4).

### The feature checklist, answered

1. *Can its CPython behavior be reproduced exactly?* Yes, by construction:
   it is the AOT behavior. Anything the converter refuses, the JIT refuses
   with the same message.
2. *Does the generated Rust remain reviewable?* Unchanged; the JIT consumes
   the generator's output and adds only a thin, generated `extern "C"`
   boundary module.
3. *Which runtime tier?* `std`.
4. *Kernel / `no_std`?* Correctly rejected there.
5. *End-to-end test diffing against `python3`?* Yes: the differential gate
   (below) and per-milestone transcripts.
6. *Static core, typed dynamic seam?* The static lowering is preserved; the
   dynamic seam is the explicit, typed boundary between the host and
   compiled code (below), never an untyped bag.

## Architecture

```
 Python source ──► convert ──► generated crate ──► cargo build ──► .so/.dylib
   (string)       python-ast   (+ extern "C"       (shared target     │
                  via pyo3      boundary shims)      dir, cached)      ▼
                                                              dlopen, call
                                  content-hash cache ◄──────────  (host process)
```

### Components

- **`rython-jit`** (new crate, `std` only, host side). The public API:
  `Jit::new(config)`, `Jit::compile(name, source) -> Module`, and typed
  calls on a `Module`. It depends on the converter and on `libloading`. It
  is the only place that knows about caching, building and loading.
- **A `cdylib` mode in the converter** (new `ConvertOptions` flag; it does
  not exist today). The crate is emitted with `crate-type = ["cdylib"]`,
  plus one generated `extern "C"` shim per public function. Everything
  else about the emitted crate is what `rypip convert` emits now.
- **`rypip jit`** (new subcommand), a thin CLI over `rython-jit`.

Some of what `rypip run` has is reused: the host-triple detection
(`host_triple()`), the quiet `cargo build --release` invocation, and the
work-directory lock (`lock_work_dir`), which serializes concurrent builds of
one cache entry. Artifact discovery is **not** reusable as it stands:
`cargo_build_executable` refuses a crate without a binary target
(`require_binary`) and selects only `bin` artifacts. The JIT needs the
library's path, so the artifact-message parsing (the
`--message-format=json-render-diagnostics` stream) is factored out of it
into a helper parameterized by target kind, and `cargo_build_executable`
becomes its `bin` caller; the JIT calls it for `cdylib`. That refactor is
part of M1.

### The parser needs a Python installation

`python-ast` parses through CPython's own `ast` module via PyO3 with
`auto-initialize`. A process that wants to JIT therefore needs a Python
installation (libpython) at runtime, in addition to a Rust toolchain. This
is inherent to the chosen approach and is stated here so nobody is
surprised: it applies to the **JIT tool**, not to converted output. A
Rust-native parser would remove the requirement, but it would also make the
parser a second implementation of Python's grammar; that trade is recorded
under open questions rather than taken silently.

### The cache

Entries live under `$XDG_CACHE_HOME/rython/jit/<key>/` (default
`~/.cache`), directory mode `0700`. The key is a hash of everything that
can change the compiled artifact:

- the source bytes and module name;
- the converter version and the `stdpython` version and feature set;
- `rustc -vV` and the host triple;
- the conversion options.

All entries build in one shared `CARGO_TARGET_DIR`, so `stdpython` and its
dependencies compile once per (version, features, toolchain) and later
entries pay only for their own code. A miss builds; a hit loads. There is no
eviction in the first milestone; `rypip jit --clear-cache` is the escape
hatch. Cold-start latency is dominated by the first `stdpython` build and is
**not** a number this document promises; milestone M1 measures it and
records the result here.

### The boundary

Each compiled library statically links its own copy of `stdpython` and of
Rust's `std`. Rust's ABI is not stable across that boundary, and the two
copies do not share statics. So **no Rust-layout type crosses it**. Only a
C-compatible representation does:

- scalars: `i64`, `f64`, `bool`, `None`;
- `str` and `bytes` as pointer plus length (UTF-8 for `str`), owned by the
  callee and released through an exported `ry_free`;
- (M2) `list`, `tuple` and `dict` as arrays of tagged values.

A boundary type the host cannot marshal is a loud `JitError::Marshal` naming
the type, never a lossy conversion.

**Exceptions.** Every shim wraps its body in `catch_unwind` and returns a
status. A raised Python exception comes back as `(type name, message,
args_repr)`, and the host reconstructs a `PyException` from it. Unwinding
never crosses the boundary. A `SystemExit` surfaces as
`JitError::Exit(code)` for the host to act on, rather than the callee
calling `process::exit`, which would skip the host's cleanup.

**Standard streams.** Because each library has its own `std`, each also has
its own stdout buffer. Left alone, output from the host and from compiled
code would interleave differently from CPython. Every shim therefore flushes
the library's stdout before returning, and the host flushes its own before
each call. This is a requirement with a test of its own, not an assumption
(see the differential gate).

**Libraries are never unloaded.** A Rust `cdylib` with thread-locals and
statics is not safe to `dlclose`, so loaded modules live for the life of the
process.

## Correct or loud

The JIT adds failure modes, and each has one loud outcome:

| Failure | Outcome |
|---|---|
| The converter refuses a construct | `JitError::Conversion`, carrying the converter's message verbatim (construct, location, rewrite) |
| The generated crate does not build | `JitError::Build`, carrying rustc's diagnostics. This is a converter bug by the project's own standard; the right response is a minimal reproducer, not a fallback |
| A value cannot cross the boundary | `JitError::Marshal`, naming the type |
| Python or a toolchain is missing | `JitError::Environment`, naming what is missing and how it was looked for |
| A cache entry is corrupt or from another toolchain | rebuilt, never loaded on trust: the key covers the toolchain, and a load failure invalidates the entry loudly |

There is **no fallback path**. A source the JIT cannot compile is an error;
it is never interpreted, approximated or run by another route.

### The differential gate

The honest test of "identical by construction" is to measure it. A new
corpus runner, `eval/jit/run_jit.py`, runs every program in the idiom
corpus (`eval/idioms/`) through both paths and requires byte-identical
stdout, stderr and exit status. The corpus programs are whole programs with
an entry point, not functions, so the gate runs them through
`rypip jit run` (below), which executes a program's entry in-process; it
cannot use `jit call`. Its results are checked in as a baseline
and enforced in CI the way `eval/idioms/baseline.json` is: a program may
not get worse, and the frontier is allowed to be red but not to retreat.
The corpus includes programs that interleave host and compiled output, to
pin the stdout requirement above.

## Surfaces

**Library** (`rython-jit`): compile a module from a string or a file and
call its public functions with typed arguments. This complements
`python_module!`, which does the same thing at *build* time: with the JIT,
a running Rust program can load Python it did not know about when it was
compiled.

**CLI** (`rypip jit`):

```
rypip jit run  FILE [ARG ...]            # whole program, in-process; exit status is the program's
rypip jit call FILE FUNCTION [ARG ...]   # ARGs are Python literals; prints repr(result)
rypip jit eval EXPRESSION [NAME=VALUE ...]   # from M3
rypip jit --clear-cache
```

`rypip jit run` is the whole-program path. In `cdylib` mode the converter
emits a second shim, `ry_entry(argc, argv) -> status`, that runs what the
binary's `main` runs (module initialization, then the `__main__` body), with
`sys.argv` set from the arguments. It differs from the binary's `main` in
one respect: it must not call `process::exit`, which would end the host. An
uncaught exception prints the same traceback header and final line the
binary prints and returns status 1, and a `SystemExit` returns its code, so
the host reports both exactly as the binary's exit would. `rypip run` stays
as it is: the exec-based, whole-program path that runs the program in its
own process.

**`eval`/`exec`** (M3). There is no ambient namespace to reach into: the
converter has none, and it rejects fabricated interpreter namespaces. So
`jit.eval("a + b", [("a", 1), ("b", 2)])` takes its names explicitly. It is
compiled as a function of those names, typed from the values passed, and
cached on (source, names, types). `exec` of statements returns the names it
assigned. Reflective access to the caller's frame (`globals()`, `locals()`)
remains a refusal.

## Milestones

Each is independently mergeable, gated by tests that diff against
`python3`.

- **M0, this document.** Goal 6 added, the two non-goals narrowed,
  `jit-eval.md` marked superseded. Docs only.
- **M1, compile, cache, load, call, run.** `rython-jit` with scalar and
  `str` arguments and results, exception mapping, the content-hash cache,
  the artifact-discovery refactor, the `ry_entry` shim, and `rypip jit call`
  and `rypip jit run`. *Useful on its own:* a Rust program can call a Python
  function it loads at runtime, and the CLI gives repeated runs of an
  unchanged file a cache hit. Gate: transcript tests against `python3` for
  each marshalled type, each exception path and a whole-program run
  (stdout, stderr, exit status), and a stdout-interleaving test. Measure and
  record cold and warm latency here.
- **M2, containers.** `list`, `tuple`, `dict`, `bytes` and nested values
  across the boundary. Gate: round-trip tests against `python3`.
- **M3, `eval` and `exec`.** The explicit-namespace API and `rypip jit
  eval`. Gate: transcripts against `python3` for the supported subset, and
  `compile_err`-style tests for every refused reflective form.
- **M4, the differential gate in CI.** `eval/jit/` with its checked-in
  baseline and a CI job, driving `rypip jit run` over the idiom corpus. (It
  needs M1's entry path, can be developed alongside it, and is enforced from
  here; it is listed on its own because it is the milestone that makes the
  claim checkable.)
- **M5, opt-in `eval` in converted programs.** `rypip convert --jit` links
  `rython-jit` into the output and lowers `eval`/`exec` to the M3 API. Such
  a program needs Python and a Rust toolchain at runtime, and says so in its
  generated `Cargo.toml` and docs. Default off. Refused on `no_std` and
  kernel targets at conversion time.
- **M6, latency and robustness.** A warm shared target directory,
  background compilation, cache eviction, and Windows support. Optionally,
  linking `stdpython` dynamically so the host and compiled code share one
  runtime and the stdout/state duplication goes away.

## Out of scope

- A bytecode or tree-walking interpreter, and any "tier" that executes
  Python without compiling it.
- Speculative or tracing optimization, and deoptimization. The JIT does not
  make Python faster than the AOT pipeline; if that becomes a goal it needs
  its own design and its own evidence.
- A REPL. It could be built on M3 later.
- `no_std`, kernel and embedded targets: AOT only.
- Anything outside the supported subset: the JIT compiles the same language
  as `rypip convert`, and refuses the same things.

## Risks

- **Latency.** A cold miss runs `cargo`. For a REPL-like use that may be too
  slow; that is why M6 exists and why M1 measures instead of promising.
- **Toolchain and Python at runtime.** The JIT is unusable on a machine
  without both. It is a development and tooling feature, not a deployment
  feature, until a Rust-native parser exists.
- **Duplicated runtime state.** Two copies of `stdpython` in one process
  disagree about any process-wide state they each keep. The C-only boundary
  and the flush rule contain this; M6's shared-runtime option removes it.
- **Arbitrary code execution.** Compiling and loading source is executing
  it. There is no sandbox, and none is planned. Untrusted source must not be
  given to the JIT, exactly as it must not be given to `eval`.
- **Cache integrity.** A world-writable cache is a code-injection vector.
  The directory is `0700`, and an entry is trusted only on a key match.

## Open questions for the maintainer

1. **Where should the line be for converted programs?** M5 makes `eval` in a
   converted program possible at the cost of the "no Python at runtime"
   property for that program. Is an explicit, default-off opt-in acceptable,
   or should the JIT stay a tool and never be linked into output?
2. **Parser.** Is requiring a Python installation at JIT time acceptable, or
   is a Rust-native parser (a second grammar to keep exact) worth its cost?
3. **Crate placement.** A new `rython-jit` crate, or a module of `rypip`?
   This document assumes a new crate so `rypip` stays free of the loader.
4. **Windows.** Is Unix-first, as `rypip run` is today, acceptable for M1?
5. **Cache policy.** Is "no eviction until M6" acceptable, and is
   `~/.cache` the right default location?
