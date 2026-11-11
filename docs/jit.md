# The Rython JIT: on-demand compilation (issue #166)

**Status: M1 is implemented** (the `rython-jit` crate, the converter's
cdylib mode and `rypip jit run` / `rypip jit call`; see [M1 as
built](#m1-as-built) for what shipped, its measured latency and where it
deviates from the design below). M2–M6 are design. This document records
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
  (As built, the converter reaches it through a trait rather than a crate
  dependency: see [Deviations](#deviations-from-the-design).)
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
binary's `main` runs (module initialization, then the `__main__` body). It
differs from the binary's `main` in two respects.

*It must not call `process::exit`*, which would end the host. An
uncaught exception prints the same traceback header and final line the
binary prints and returns status 1, and a `SystemExit` returns its code, so
the host reports both exactly as the binary's exit would.

*Its arguments cannot come from its parameters alone.* `sys.argv` is a
`LazyLock` over the real process arguments (`stdlib/sys.rs`), and
`argv_at` reads them independently, so inside a host process both would
report the host's arguments. M1 therefore adds a runtime seam in
`stdpython`: one function that yields the program's arguments, consulted by
both `argv` and `argv_at` (and by the argparse default `prog`, which goes
through `argv_at`), with an override that `ry_entry` installs, in the
library's own copy of `stdpython`, before the program runs. `argv[0]` is the
program path as the user typed it, as `rypip run` already does. Because
`argv` initializes once, a loaded library's entry can run **once** per
process; a second `ry_entry` call on the same library returns
`JitError::EntryAlreadyRun` instead of running with stale arguments. A
resettable `argv` is a later change, since generated code reads the static
directly.

`rypip run` stays as it is: the exec-based, whole-program path that runs the
program in its own process.

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
- **M1, compile, cache, load, call, run. Done** ([M1 as built](#m1-as-built)).
  `rython-jit` with scalar and
  `str` arguments and results, exception mapping, the content-hash cache,
  the artifact-discovery refactor, the `ry_entry` shim and the `sys.argv` seam in `stdpython`, and `rypip jit call`
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

## M1 as built

M1 shipped with the open questions below answered by this document's own
assumptions, pending the maintainer: a new crate (`crates/rython-jit`,
`std` only); Unix first (elsewhere `Jit::new` is a loud
`JitError::Environment`, as `rypip run` refuses); the cache at
`$XDG_CACHE_HOME/rython/jit` (default `~/.cache/rython/jit`), mode `0700`,
no eviction (`rypip jit --clear-cache` only); and Python installed at JIT
time, since the parser is CPython's.

### Surfaces

- **Library** (`rython-jit`): `Jit::new(JitConfig)`,
  `Jit::compile(name, source)` / `Jit::compile_file(name, path)` returning a
  `Module`; `Module::call(function, &[Value]) -> Value`,
  `Module::call_as::<R>` (`R` one of `()`, `bool`, `i64`, `f64`, `String`,
  `Value`), `Module::run_entry(&argv) -> exit status`,
  `Module::set_argv(&argv)`, and introspection (`functions()` with M1
  signatures, `warnings()`, `cache_outcome()`). `Value` is `None`, `Bool`,
  `Int(i64)`, `Float(f64)` or `Str(String)`; its `repr()` is the runtime's
  own `repr`. `rypip::jit::new_jit(options, cache_dir)` is the ready-made
  JIT with rypip's converter.
- **Command line**: `rypip jit run FILE [ARG ...]`, `rypip jit call FILE
  FUNCTION [ARG ...]` (ARGs read by CPython's `ast.literal_eval`; prints
  `repr(result)`), `rypip jit --clear-cache`. `jit call` is defined
  against `python3 -c 'import M; print(repr(M.FUNCTION(ARG, ...)))'`, so
  it installs `sys.argv == ['-c']` before the call, and an uncaught
  exception of a class the module defines is named module-qualified
  (`calls.Overdrawn: ...`), as CPython's traceback names it. `jit run`
  takes one `.py` file: a directory, or a file beside a `rython.toml`
  (whose `[python-modules]` / `[rust-modules]` `rypip run` would honor),
  is refused rather than compiled differently.

### The converter's cdylib mode

`ConvertOptions::cdylib: Option<CdylibTarget>` (single-module packages)
emits what `rypip convert` emits for the module, minus the binary
(`autobins = false`, `[lib] crate-type = ["cdylib"]`), plus one generated
module, `src/__rython_jit.rs`:

- the boundary vocabulary (`RyValue`, `RyStr`, `RyOutcome`, the tags and
  statuses), emitted verbatim from `crates/rython-jit/src/abi_types.rs` —
  the same file the host compiles, so the two sides cannot drift;
- `ry_abi_version`, `ry_manifest` (a static text manifest: the module,
  whether it has a program entry, whether calls may import it, each shimmed
  function's M1 signature, each unshimmed public function with the reason,
  and the module's classes) and `ry_free`;
- one `ryfn_<n>` per public module-level `def` whose parameters are
  annotated `int` / `float` / `bool` / `str` / `None` and whose result the
  converter resolves (`FunctionDef::resolved_return_type`, the authority
  the PyO3 bindings use) to one of those types. Defaults, `*args`,
  `**kwargs`, keyword- or positional-only parameters, decorators,
  generators and `async def` get no shim; each is listed with its reason.
  A shim checks its arguments' tags, imports the module (its
  `__module_init__`, once, as `import` runs the body), calls the function
  under `catch_unwind`, flushes the library's stdout, and returns a status.
  A result whose type the prediction got wrong still compiles (method
  resolution falls back to an "opaque" conversion) and is a loud
  `JitError::Marshal` naming the Rust type, never a library that fails to
  build;
- `ry_entry(argv, argc, out)`, emitted when the module is a program with a
  synchronous entry: it installs `argv` as this library's `sys.argv`, runs
  `__rython_main__` — the program body the binary's `fn main` runs (the
  lowering now emits every synchronous entry as `__rython_main__` plus a
  `fn main` that reports its error), reports an uncaught exception through
  `stdpython::report_uncaught` (the binary's `exit_uncaught` minus the
  process exit) and returns the exit status;
- `ry_set_argv(argv, argc, out)`: the `sys.argv` function calls see.

### The boundary as built

- **Values**: tagged `RyValue`s; strings are UTF-8 pointer + length,
  borrowed from the host for arguments and allocated by the library for
  results (released through `ry_free`). The host checks arity and each
  argument's type against the manifest before anything runs, and never
  converts: an `int` passed for a `float` parameter (or a `bool` for an
  `int`) is `JitError::Marshal`, because CPython would run the function on
  the `int` itself.
- **Exceptions**: a raised exception crosses as netstring-encoded (type
  name, message, argument reprs, user-class ancestors) and is rebuilt as a
  `stdpython::PyException`, so `type_name()`, `message`, `repr` and
  `matches("Exception")` answer as in the library. Instance attributes
  other than those do not cross in M1. `SystemExit` comes back as
  `JitError::Exit { code, message }` — the message being the line CPython
  prints for a non-integer code — and the library never calls
  `process::exit` for it. A panic is caught and becomes `JitError::Panic`
  after the library's panic hook has reported it (the CLI exits 101, a
  panicking binary's status).
- **Standard streams**: the host flushes its stdout before every call into
  a library (`call`, `run_entry`), and every shim flushes the library's
  stdout before returning, so partial lines keep program order (pinned by
  `crates/rypip/tests/jit_interleave.rs` against a python3 transcript).
  stderr is unbuffered on both sides.
- **`sys.argv`**: the stdpython seam (`stdlib/sys.rs`): `sys.argv` and
  `argv_at` (and so argparse's default `prog`) read one source, the
  process's arguments unless a host installed others with
  `install_program_args` before the first read. The first read fixes the
  source; a later install is refused (`InstallArgsError`), never ignored.
- **One role per loaded library**: CPython would run `__main__` and an
  imported module as two separate modules, but a library holds one copy of
  its module state. So a library serves either function calls (the module
  imported) or its program entry (run as `__main__`), not both; the other
  is `JitError::Unsupported` naming why. The entry runs once per process
  (`JitError::EntryAlreadyRun`). A module whose final top-level statement
  is a bare `main()` call (issue #431) — which CPython runs on import, but
  which the library compiles as the entry — refuses calls rather than
  importing without running it.
- **Libraries are never unloaded**; a process-wide registry maps each
  artifact path to its one loaded copy.

### The cache as built

```text
$XDG_CACHE_HOME/rython/jit/       0700, owned by the user (else refused)
  target/                         the one CARGO_TARGET_DIR (release, --target <host>)
  target.lock                     held from a build through its artifact copy
  entries/<key digest>/
    crate/                        the generated crate
    lib-<time>-<pid>-<n>.so       the loaded artifact, never rewritten in place
    entry                         written last: artifact, size, warnings, full key text
  entries/<key digest>.lock       exclusive for a build, shared for a load
```

The key is the full text of: the boundary version; the module name and the
source's length and SHA-256; the compiler's fingerprint — rypip's version,
a build-time hash of the converter's sources (rypip, python-ast and the
boundary text, so a rebuilt converter in a checkout never reuses a library
the old one made), the parser's CPython version (`Py_GetVersion`, read
without starting the interpreter), the stdpython dependency (a registry
version, or a checkout's path plus every source file's size and mtime, as
cargo's own freshness check reads them; the feature set follows from the
source's imports) and every conversion option; `rustc -vV` (run in the cache root,
so a rustup directory override resolves as it does for the build; it names
the host triple); the build-affecting environment (`RUSTFLAGS`,
`CARGO_ENCODED_RUSTFLAGS`, `CARGO_BUILD_RUSTFLAGS`, `RUSTC_WRAPPER`,
`RUSTC_WORKSPACE_WRAPPER`, the `CARGO_PROFILE_RELEASE_*` knobs); and the
release profile. An entry is loaded only when its record's key text matches
exactly, its artifact has the recorded size and SHA-256, and its generated
Cargo.lock has the recorded SHA-256. The compiler build identity also
includes the workspace lockfile and compiler/runtime manifests. Each entry
pins the dependency versions Cargo resolved on its first build; a newly
published registry release does not update that entry. Use `--clear-cache`
to resolve dependencies again. A changed or missing lockfile, a mismatch, or a library
that will not load or reports another boundary version, is reported on
stderr ("cannot be trusted ... rebuilding it") and rebuilt — never loaded
on trust. A hit runs neither the converter nor cargo (pinned with a
counting compiler); the converter's lossy-conversion warnings are stored
with the entry and reported on every load, as `rypip run` reports them on
every run.

All cache ancestors must be directories controlled by this user or root,
without group/world write access (root-owned sticky `/tmp` is allowed).
Symlink ancestors are refused. Cache clearing takes an exclusive lock next
to the root, while readers and builders hold it shared through loading;
clearing waits for active work and never deletes that lock file.

### Deviations from the design

- **The converter is injected, not depended on.** The design has
  `rython-jit` depend on the converter and `rypip jit` be a CLI over
  `rython-jit`. The converter lives in `rypip`, so those two dependencies
  form a cycle cargo refuses. As built, `rython-jit` defines a `Compiler`
  trait (`fingerprint`, `convert`, `build`) and `rypip::jit::RypipCompiler`
  implements it with the converter's cdylib mode and the factored artifact
  helper; `rypip` depends on `rython-jit`. A Rust program using the JIT
  depends on both (`rypip::jit::new_jit`). Moving the converter out of
  `rypip` into its own crate would remove the indirection; that is a
  larger restructuring than M1 and is left to the maintainer.
- **The work-directory lock is rython-jit's own**, with `lock_work_dir`'s
  semantics (an advisory lock on a sibling `<entry>.lock`), for the same
  cycle reason; it adds a shared mode for loads.
- **Additional error variants.** Besides the design's `Conversion`,
  `Build`, `Marshal`, `Environment`, `EntryAlreadyRun`, `Exit` and
  `Python`, `JitError` has `Panic` (a caught panic: the design requires
  every shim to catch unwinds, so a caught one needs an outcome) and
  `Unsupported` (an operation M1 refuses rather than approximates: calling
  a class or a function with defaults, mixing a library's roles, calls
  into a #431 module). `Exit` carries the `SystemExit` message too, since
  `sys.exit("text")` prints it.
- **`ry_set_argv`** is an addition: without it `jit call`'s functions would
  see rypip's own arguments, where `python3 -c` has `['-c']`.
- **The entry body is a named function in every conversion**, not only
  the JIT's: the binary's `fn main` now calls `__rython_main__` and
  reports its error, so `ry_entry` runs the binary's body by construction
  instead of a second lowering of it.

### Known limits (all loud)

- Single-module sources only; packages, `rython.toml` manifests and
  `from rython import rust` path dependencies are refused (the key could
  not cover them). Cargo configuration *files* (`~/.cargo/config.toml` and
  the like) are not part of the key: a change there that alters the build
  needs `--clear-cache`.
- An `async` program entry is refused (its runtime entry is not part of the
  boundary); the module's synchronous functions still call.
- A non-UTF-8 program argument is refused before `jit run` starts (the
  binary fails only when the argument is read).
- Runtime paths that end the process themselves — argparse's usage, help
  and version exits, `unittest.main()`, `os._exit` — end the host process
  too. Under `rypip jit run` / `jit call` that is exactly the binary's (and
  CPython's) observable outcome; a Rust host calling such a function loses
  control rather than receiving `JitError::Exit`.
- Output a library writes after its shim returned (a thread it started)
  is not flushed by the shim.

### Measured latency

Measured on 2026-11-11 on the development machine (32 logical CPUs,
`CARGO_BUILD_JOBS=2`, `CARGO_NET_OFFLINE=true`, rustc 1.98.1,
release builds, debug `rypip`), for `rypip jit call calls.py add 2 40`
(wall clock, median of 5 where repeated):

| Case | Latency |
|---|---|
| Cold: empty cache, so `stdpython` (std tier) and its dependencies build | 23.958 s (one sample) |
| Cold entry, warm target directory: a source not seen before | 0.785 s |
| Warm: cache hit (`rustc -vV`, key, `dlopen`, call) | 51.0 ms |

Raw samples and the input source are in
[`eval/jit/m1-latency-2026-11-11.json`](../eval/jit/m1-latency-2026-11-11.json).
Registry downloads were already cached; other workspace test builds were
active, so these are development-host observations rather than isolated
performance benchmarks.

The cold number is dominated by the shared target directory's first build
and is paid once per (stdpython version, features, toolchain); a new source
then costs its own crate's compile, and an unchanged one a `dlopen`. M6
remains the place to shorten the first two.

### Tests (the M1 gate)

`crates/rypip/tests/jit_tests.rs` (sharing one cache and target directory
under cargo's per-target scratch directory) and
`crates/rypip/tests/jit_interleave.rs`: `jit call` per marshalled type and
`None` against python3; a builtin exception, a module's own exception
class, `SystemExit` with a code and with a message; typed library calls and
the rebuilt `PyException` (type, message, `repr`, ancestry); seven
whole programs through `jit run` (argv, an uncaught exception, classes and
dicts, module-level code ending in `sys.exit("...")` with a partial line,
argparse with arguments and its usage error, a #431 bare `main()`) against
python3's stdout, stderr minus frames and exit status; host/library stdout
interleaving; a cache hit that runs neither converter nor cargo; an
untrusted entry rebuilt; entry-once and role refusals; the converter's
refusal passed through verbatim; marshal refusals naming the type; and the
cache directory's permissions. The argv seam has its own pins in
`crates/stdpython/tests/argv_seam.rs` and `stdlib/sys.rs`.

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

M1 proceeded on this document's assumptions for questions 2–5 (Python
required at JIT time; a new crate; Unix first; no eviction under
`~/.cache`), each of which the maintainer can still redirect. Question 3
has a consequence the design did not foresee: a separate crate cannot
depend on `rypip`'s converter while `rypip` depends on it, so the
converter is injected through a trait (see [Deviations](#deviations-from-the-design));
a converter crate split out of `rypip` would let `rython-jit` depend on it
directly.
