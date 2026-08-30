# rython_stdpython

The default Python runtime library for the Rython compiler ecosystem. This crate provides a comprehensive, Python-compatible standard library implementation in Rust that serves as the runtime foundation for Python code compiled to Rust via the Rython toolchain.

## Overview

`rython_stdpython` is a complete Python runtime environment written in Rust that enables compiled Python code to access all Python built-ins, types, and standard operations without requiring any imports. It provides both `std` and `no_std` variants, making it suitable for everything from desktop applications to embedded systems.

## Key Features

- **Complete Python Built-ins**: 40+ built-in functions implemented (`print`, `len`, `range`, `enumerate`, `zip`, `min`, `max`, `sum`, `all`, `any`, etc.)
- **Full Type System**: Python-compatible implementations of `str`, `list`, `dict`, `tuple`, `set`, `int`, `float`, `bool` with all their methods
- **Generic Trait System**: Flexible, zero-cost abstractions that work with any type implementing the appropriate trait
- **Exception Handling**: Complete Python exception hierarchy (`ValueError`, `TypeError`, `IndexError`, etc.)
- **Both std and no_std**: Supports standard library environments and embedded/no_std targets
- **Memory Safe**: All operations maintain Rust's memory safety guarantees
- **Performance Optimized**: Native Rust implementations provide superior performance to interpreted Python

## Architecture

This library uses a generic trait-based design that mirrors Python's built-in behavior:

### Conversion Traits
- `PyAbs`: Generic absolute value (`abs(-5i64)`, `abs(-3.14f64)`)
- `PyBool`: Generic boolean conversion (`bool(42)`, `bool("")`)
- `PyInt`: Generic integer conversion (`int("123")`, `int(3.14)`)
- `PyFloat`: Generic float conversion (`float("3.14")`, `float(42)`)
- `PyToString`: Generic string conversion (`str(123)`, `str(true)`)
- `PySum`: Generic summation (`sum(&[1,2,3])`, `sum(&pylist)`)

### Runtime Traits
- `Len`: Universal length calculation
- `Truthy`: Python-style truthiness evaluation

## What's Implemented

### Python Built-in Functions (40+)
✅ **Math/Logic**: `abs()`, `min()`, `max()`, `sum()`, `all()`, `any()`, `round()`, `divmod()`, `pow()`  
✅ **Iteration**: `enumerate()`, `zip()`, `range()`, `len()`, `sorted()`, `reversed()`, `map()`, `filter()`  
✅ **Type Conversion**: `bool()`, `int()`, `float()`, `str()`, `list()`, `dict()`, `tuple()`, `set()`, `frozenset()`, `slice()`  
✅ **Object Introspection**: `type()`, `isinstance()`, `hasattr()`, `getattr()`, `setattr()`, `delattr()`, `id()`, `hash()`  
✅ **Character/Unicode**: `ord()`, `chr()`, `ascii()` (repr with non-printable-ASCII escaping)  
✅ **Numeric Formatting**: `hex()`, `oct()`, `bin()`  
✅ **I/O**: `print()` with full parameter support, `input()`, `open()` (std mode only)

### Python Built-in Types with Complete Method Sets

#### PyStr (String Type)
✅ **Core Methods**: `split()`, `join()`, `strip()`, `lower()`, `upper()`, `replace()`  
✅ **Search Methods**: `find()`, `count()`, `startswith()`, `endswith()`  
✅ **Formatting**: `format()` (basic implementation)  

#### PyList (List Type)  
✅ **Modification**: `append()`, `extend()`, `insert()`, `remove()`, `pop()`, `clear()`  
✅ **Search/Sort**: `index()`, `count()`, `sort()`, `reverse()`  
✅ **Utilities**: `copy()`, indexing with `get()`/`set()`  

#### PyDictionary (Dictionary Type)
✅ **Access**: `get()`, `get_or_default()`, `contains_key()`  
✅ **Modification**: `set()`, `pop()`, `clear()`, `update()`  
✅ **Iteration**: `keys()`, `values()`, `items()`  

#### PyTuple (Tuple Type)
✅ **Immutable sequence**: Index access, slicing support  

#### PySet (Set Type)
✅ **Modification**: `add()`, `remove()`, `discard()`, `clear()`  
✅ **Set Operations**: `union()`, `intersection()`, `difference()`  
✅ **Membership**: `contains()`  

### Standard Library Modules
`argparse`, `asyncio` (tokio-backed), `collections`, `copy`, `csv`,
`datetime`, `functools`, `glob`, `hashlib`, `heapq`, `io` (StringIO
and BytesIO, on every tier), `itertools`, `json`, `math`,
`os`/`os.path`, `pathlib`, `random`, `re`, `socket` (TCP/UDP on
std::net), `string`, `subprocess`, `sys`, `sysconfig`, `tempfile`,
`textwrap`, `threading` (Thread/Lock/RLock/Event/Semaphore on
std::thread), `time`, `unicodedata`-style codec handling,
`urllib.request` (ureq-backed, behind the `http-ureq` feature),
`venv`, `warnings`

### Threading
✅ **Thread management**: `threading.Thread(target=, args=, daemon=)`, `start()`, `join()`, `is_alive()`, `current_thread().name`, `active_count()` — CPython's thread naming and RuntimeError messages  
✅ **Locks & synchronization**: `Lock`, `RLock` (reentrant, owner-checked), `Event`, `Semaphore`, all shareable handles with `with lock:` acquire/release guards  
✅ **Exception reporting**: an unhandled exception in a thread prints CPython's `Exception in thread NAME:` header (no traceback frames — rython has no frames)

### Networking
✅ **Sockets**: `socket.socket(AF_INET/AF_INET6, SOCK_STREAM/SOCK_DGRAM)`, `bind`/`listen`/`accept`, `connect`, `send`/`sendall`/`recv`, `sendto`/`recvfrom`, `settimeout` (TimeoutError "timed out"), `getsockname`/`getpeername`, `close`, `gethostname()` — errors raise the real CPython hierarchy (`ConnectionRefusedError` IS-A `ConnectionError` IS-A `OSError`) with `[Errno N]` messages  
✅ **HTTP client**: `urllib.request.urlopen()` for http/https behind the opt-in `http-ureq` feature (see below) — `.status`, `.version`, `.reason`, `read()`, `getcode()`, `getheader()`, HTTPError/URLError wired into the exception tree

### File I/O in no_std mode
✅ **In-memory buffers on the alloc tier**: `io.StringIO` and `io.BytesIO` (Python's cursor semantics, byte/char-exact write counts, the closed-file ValueError) build with `--no-default-features --features alloc` — a target with no OS has no disk, so the in-memory file surface IS its file I/O. Disk files (`open()`, directory handling via `os`/`pathlib`/`glob`) stay std-only.

### String formatting
✅ **Old-style `%`-formatting** on `str` and `bytes` (round 34): the full
conversion set (`%s %r %a %d %i %u %o %x %X %e %E %f %g %G %c %b` and
`%%`), flags/width/precision incl. `*`, the `%(name)s` mapping form with
a dict RHS, and CPython's exact TypeErrors/ValueErrors — pinned to
CPython transcripts. `str(x)`/`print(x)`/f-string `{x}` on a class
instance route through its `__str__`/`__repr__`, else the default
object repr.

### Exception System
✅ **Complete built-in exception tree**: every CPython built-in exception
name is modeled, and `except` matching walks the real hierarchy —
`except LookupError:` catches `IndexError`/`KeyError`, `except OSError:`
catches `FileNotFoundError` and friends, while `except Exception:`
correctly does NOT catch `SystemExit`/`KeyboardInterrupt`/`GeneratorExit`
(the tree is the interpreter's own data: python-ast dumps every builtin
exception's real `__mro__` through PyO3 and the checked-in table is
verified against the live interpreter by the `exception_tree_is_current`
test). A dynamic `except <boxed value>:` (`except
self._retryable_exceptions:`) matches the runtime value with
`matches_value` — Str members match by name, tuples match any member,
and a non-catchable value raises CPython's TypeError.

## What's Not Implemented

❌ **Complex Built-ins**: `exec()`, `eval()`, `compile()`, `globals()`, `locals()` — dynamic code execution and frame introspection have no static-Rust equivalent  
❌ **Frame-introspection surfaces**: `dir()`, `vars()`, `callable()`, first-class `iter()`/`next()` objects — handled at conversion time by the compiler instead of at runtime  
❌ **Disk I/O** (no_std mode): real files and directories need an OS — the alloc tier's file I/O is the in-memory `StringIO`/`BytesIO` surface above  
❌ **Networking edges**: `setsockopt`, `socket.makefile`, address families beyond AF_INET/AF_INET6, urllib POST bodies and `Request` objects  
❌ **Threading edges**: `Condition`, `Barrier`, `queue.Queue`, lock/wait timeouts  
❌ **Dynamic Import System**: `__import__`, importlib-style loading

## Feature-Gated Platform Surfaces

Platform-heavy functionality is **not** hand-reimplemented: where an
existing Rust crate provides the behavior, stdpython wraps it behind an
opt-in cargo **feature** instead of growing a mandatory dependency.

- **Naming**: one feature per surface — `<module>` when there is one
  natural backing crate, `<module>-<backend>` when several exist. This
  extends the existing precedents: `async-tokio` (asyncio on tokio) and
  the numpy backend features (`numpy-rayon`, `numpy-simd`, …).
- **Default posture**: the default build stays dependency-light, and the
  alloc/no_std tier is never affected.
- **Tooling contract**: rypip detects the import and enables the named
  feature on the generated crate's stdpython dependency automatically
  (the `async-tokio` mechanism); under `--no-std` the import is a loud
  conversion error naming the tier.
- HTTP clients are the first consumer: `urllib.request` wraps the ureq
  crate (with rustls, so `https://` works) behind the **`http-ureq`**
  feature — `import urllib.request` in a converted package puts
  `features = ["http-ureq"]` on the generated stdpython dependency.
- TLS and regular expressions follow: `ssl` sits behind **`ssl-rustls`**
  and `re` behind **`re-regex`**. Both stay in this crate's own
  `default` (so building or testing stdpython itself is unchanged), but
  generated crates request the tier and its surfaces explicitly —
  `default-features = false, features = ["std", …]` — so a converted
  package that imports neither compiles neither rustls nor the regex
  engine. That is 54 dependency crates down to 35, and roughly 40% less
  CPU in the dependency build.

## Usage

### Standard Library Mode (default)
```toml
[dependencies]
rython_stdpython = "1.0"
```

### No-Std Mode (embedded systems)

The crate is layered as a `core ⊂ alloc ⊂ std` feature ladder; no_std is
reached by turning the default `std` feature off:

```toml
[dependencies]
rython_stdpython = { version = "1.0", default-features = false, features = ["alloc"] }
```

The `alloc` tier keeps heap-backed Python semantics (strings, lists,
dicts/sets, exceptions, json, the in-memory io.StringIO/io.BytesIO
buffers) with no OS dependency; everything that touches the OS (disk
I/O, os/os.path, datetime, subprocess, tempfile, glob, pathlib,
random, threading, socket, math's float intrinsics) is `std`-only. A
strictly-core tier without an allocator is not implemented yet and
fails loudly at compile time.

### Example Usage

```rust
use rython_stdpython::*;

fn main() {
    // Generic functions work with any compatible type
    let nums = vec![1, 2, 3, 4, 5];
    let total = sum(&nums[..]);  // Generic summation
    
    // Python-like type conversions
    let s = str(total);  // "15"
    let b = bool(&nums); // true (non-empty)
    
    // Full Python collections
    let mut list = PyList::from_vec(nums);
    list.append(6);
    
    // All Python built-ins available
    print(&format!("Total: {}", total));
    assert_eq!(len(&list), 6);
}
```

## Integration with Rython

This crate serves as the runtime foundation for the entire Rython ecosystem:

- **rythonc**: The Python-to-Rust compiler generates code that calls these runtime functions
- **python-ast-rs**: AST code generation targets these built-in implementations  
- **python-mod-rs**: Embedded Python modules depend on these built-ins

When Python code is compiled to Rust, it naturally maps to function calls in this library:

```python
# Python code
my_list = [1, 2, 3]
total = sum(my_list)
print(str(total))
```

```rust
// Generated Rust code
let mut my_list = PyList::from_vec(vec![1, 2, 3]);
let total = sum(&my_list);  // Uses PySum trait
print(str(total));          // Uses PyToString trait
```

## Building and Testing

```bash
# Standard library version
cargo build
cargo test

# No-std version (no_std + alloc)
cargo build --no-default-features --features alloc
cargo test --no-default-features --features alloc

# Prove it on a bare-metal target
rustup target add thumbv7m-none-eabi
cargo check --no-default-features --features alloc --target thumbv7m-none-eabi

# Run specific test modules
cargo test test_python_functions  # Built-in function tests
cargo test test_pystr             # String type tests
cargo test test_pylist            # List type tests
```

## License

This project is part of the Rython compiler ecosystem.
