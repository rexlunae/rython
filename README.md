# Rython 🐍→🦀

A Rust workspace for compiling Python to Rust.

Rython parses real Python source, lowers it to idiomatic Rust against a
Python-semantics runtime crate, and builds it with cargo. The result is a
native binary or library with no interpreter, no GIL, and no Python
installation at runtime.

## What these tools are for

**1. Translating Python into Rust to migrate projects.**
`rypip convert` turns a Python file, package, or `pyproject.toml` project
into a standalone Rust crate you own from then on: each module becomes a
Rust module, the `__main__` block becomes `fn main`, and the generated code
depends only on the `stdpython` runtime. The output is a starting point for
a permanent port — readable Rust you can refactor, not an opaque bundle.

**2. Using Python inside Rust projects.**
The `python-mod` proc-macros (`python_module!`, `python_module_nostd!`)
compile `.py` files into your crate at build time, so a Rust project can
keep algorithm-heavy or domain code in Python syntax while everything
compiles to native Rust — callable from Rust like any other module, with
zero runtime bridging. `rypip convert --pyo3` goes the other direction:
it wraps the converted crate in PyO3 bindings so existing Python code can
import the fast version.

**3. Better, faster, more correct Python.**
Converted programs are ahead-of-time compiled, statically typed, and free
of the interpreter and GIL — and the conversion itself is a correctness
tool. The translator refuses to guess: anything it cannot reproduce with
CPython's exact observable behavior is a **loud error at conversion time**
(or a typed, catchable exception at runtime), never silently different
output. Supported surface is pinned against CPython byte-for-byte —
`str(1e16)` is `1e+16`, `hash()` matches `PYTHONHASHSEED=0`, float repr,
sort stability, and exception messages all match — and the end-to-end test
suite diffs generated-binary output against `python3` line for line.

## Crates

| Crate | Description |
|---|---|
| [`python-ast`](crates/python-ast) | Core library — parses Python AST and transpiles to Rust |
| [`python-mod`](crates/python-mod) | Proc-macro for including Python modules in Rust source |
| [`rypip`](crates/rypip) | pip-like tool — converts/builds/installs Python packages as Rust crates |
| [`rythonc`](crates/rythonc) | CLI compiler — `rythonc input.py → output.rs` |
| [`stdpython`](crates/stdpython) | Python standard library runtime implemented in Rust |

## Usage

```bash
# Build all crates
cargo build

# Convert a Python package into a Rust crate
cargo run -p rypip -- convert path/to/package --out my-crate

# Build and install a Python package as a native binary
cargo run -p rypip -- install path/to/package

# Convert for embedded/wasm targets (core+alloc only, no OS)
cargo run -p rypip -- convert path/to/package --out my-crate --no-std

# Compile a single Python file to Rust source
cargo run -p rythonc -- input.py -o output.rs

# Run tests for a specific crate
cargo test -p python-ast
```

`stdpython` is tiered for `no_std` use: the default `std` tier has the full
surface; the `alloc` tier (`--no-default-features --features alloc`) keeps
everything that doesn't need an OS (strings, collections, json, itertools,
functools, heapq, textwrap, hashlib, csv, …) and works on embedded targets.

## Compatibility

Rython is a **subset of Python**, and the boundary is enforced loudly: a
program either converts and behaves like CPython, or conversion fails with
a message saying exactly what isn't supported. Nothing silently diverges.

What works today (all CPython-verified): functions with type annotations,
classes (struct-based), control flow including `try`/`except`/`finally` and
loop `else`, comprehensions, f-strings and `str.format` on literal
templates, keyword arguments and defaults for user functions (plus
keyword `replace()` on the datetime family), the core builtins (`print`,
`len`, `range`, `open`, `sorted`, `min`/`max`, `enumerate`,
`map`/`filter`, `zip`, `sum`, `pow`, `repr`, `hash`, `isinstance` on
annotated locals, …), string/list/dict/set methods, file objects (disk
handles and `io.StringIO` behind one surface, including `with open(...)
as f:`), `functools.partial` over statically-known functions,
`@functools.lru_cache`/`@cache` with CPython's exact LRU discipline,
conversion-time `argparse` (typed namespace, byte-identical help and
error output), and a growing stdlib: `math`, `random`, `os`, `sys`,
`json`, `re` (incl. flags, named groups, findall tuples),
`datetime`/`time` (incl. `strptime`), `itertools`, `functools.reduce`,
`heapq`, `copy`, `textwrap`, `hashlib`, `csv` (reader and writer),
`collections`, `pathlib`, `glob`, `subprocess`, `tempfile`, and more.

Known gaps:

- **Numbers**: `int` is `i64` — arbitrary-precision integers are not
  supported; overflow panics rather than growing.
- **Dynamic typing**: variables keep one type; heterogeneous lists and
  reassigning a name to a different type don't convert.
- **Language features**: generators/`yield`, `async`/`await`,
  `eval`/`exec`, `*args`/`**kwargs`, multiple inheritance and dunder
  protocols are not supported yet. Decorators other than
  `functools.lru_cache`/`cache` are a loud conversion error (never
  silently ignored).
- **`lru_cache` keys** must be `int`/`bool`/`str`-annotated parameters
  — floats are not hashable in Rust, so Python's float-key caching is
  refused rather than approximated.
- **File objects** cover text modes (`r`/`w`/`a`) and `io.StringIO`;
  binary modes, `BytesIO`, `seek`/`tell`, and file-based `json`
  (`dump`/`load`) are not supported yet.
- **`argparse`** supports literal specs only (the parser is evaluated at
  conversion time): `str`/`int`/`float` positionals, `--long` options
  with `default=`, `store_true`, `help=`, `prog=`, `description=`.
  `nargs`, `choices`, subcommands, and short options are loud errors.
- **`csv.writer`** implements the default excel dialect; other dialects
  and `QUOTE_ALL`-style options are not supported yet.
- **`re`** is backed by the `regex` crate: backreferences and lookarounds
  are a loud `re.error`; `findall` supports up to 3 capture groups.
- **Typed-lowering edges**: places where Python produces `None` inside a
  typed container (a non-participating regex group in `split`,
  `groupdict` of an unmatched group) fail loudly instead; sorting
  `NaN` panics, since Python's `NaN` ordering is not reproducible.

## Roadmap

The previously tracked stdlib/feature backlog ([#19](https://github.com/rexlunae/rython/issues/19),
[#65](https://github.com/rexlunae/rython/issues/65)–[#69](https://github.com/rexlunae/rython/issues/69))
is complete. What's ahead:

- Generalized decorator support (unblocks `dataclasses`, `property`,
  user-defined wrappers)
- Generators and `yield` (likely as iterator-struct lowering)
- Richer class model: inheritance, common dunder protocols
- An arbitrary-precision integer strategy (opt-in bigint tier)
- Broader `isinstance`/type-narrowing via real type inference
- Binary file modes and `io.BytesIO`; file-based `json`
- Continued stdlib expansion, always CPython-pinned with loud boundaries

## Publishing

```bash
# Publish all crates in dependency order
cargo publish -p python-ast
cargo publish -p stdpython
cargo publish -p python-mod
cargo publish -p rythonc
cargo publish -p rypip
```

## License

Apache-2.0
