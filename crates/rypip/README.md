# rypip

A pip-like tool for the rython toolchain. rypip builds Python packages as
native Rust binaries and converts Python packages into Rust crates.

## Commands

```sh
# Build a Python package as a native binary and install it where cargo
# installs binaries (~/.cargo/bin, or --root <dir>):
rypip install path/to/package

# Convert a Python package into a Rust crate:
rypip convert path/to/package --out my-crate

# Convert with PyO3 bindings so the crate can also be imported from Python:
rypip convert path/to/package --out my-crate --pyo3

# Convert into a #![no_std] library crate (embedded/wasm targets):
rypip convert path/to/package --out my-crate --no-std

# Convert and compile without installing:
rypip build path/to/package
```

## Package discovery

`rypip` accepts a single `.py` file, a package directory (containing
`__init__.py`), or a project directory with packaging metadata. Package
name, version, layout, and dependencies are resolved **the way Python
itself resolves them**:

- **`pyproject.toml`** — PEP 621 `[project]` (`name`, `version`,
  `dependencies`) plus `[tool.setuptools]` (`packages`, `py-modules`,
  `package-dir`, `[tool.setuptools.packages.find]` with `where`).
- **`setup.cfg`** — `[metadata]` and `[options]` (packages, py_modules,
  install_requires, `[options.packages.find]`).
- **`setup.py`** — executed through a `python3` shim (pip-style) that
  records the `setup(...)` call without running setuptools; a static
  parser falls back when no interpreter is available.

`find_packages()` / `[tool.setuptools.packages.find]` discover packages
recursively (skipping hidden and underscore-prefixed directories), and both
flat and `src/` layouts are recognized. Projects without any packaging
metadata fall back to the historical layout heuristics.

A module containing an `if __name__ == "__main__":` block (or a
`__main__.py`) becomes the binary entry point; packages without one convert
to library crates and cannot be `install`ed.

## Generated crates

Each Python module becomes a Rust module; subpackages become nested modules.
The crate depends on the `stdpython` runtime (locate it with `--stdpython`,
`RYPIP_STDPYTHON_PATH`, or the copy alongside this tool's source tree).

With `--no-std`, the generated crate is a `#![no_std]` library on
stdpython's `alloc` tier (`default-features = false, features = ["alloc"]`):
no OS dependency, suitable for embedded/wasm targets. Python constructs
that need the OS — `print`/`input`/`open`, imports of
`os`/`sys`/`datetime`/`random`/`math`/…, and `__main__` blocks — fail the
conversion loudly rather than surfacing later as build errors in the
generated crate; `json`, `string`, `collections`, and `itertools` stay
available.

With `--pyo3`, the crate gains a `python` cargo feature, a `cdylib` target,
and a generated `#[pymodule]` exposing every top-level function whose
signature is expressible in concrete Rust types (parameters annotated with
`int`/`float`/`str`/`bool`/`bytes`, returns annotated or inferable). Build
the extension with `cargo build --features python`.

## Python library dependencies

Dependencies declared in the packaging metadata — `[project] dependencies`
or `install_requires` — are resolved **pip-style from PyPI** when they are
not already vendored: rypip queries the PyPI JSON API, picks the newest
version satisfying the PEP 440 specifiers, downloads the pure-Python wheel
(or sdist), extracts it into a cache (`$RYPIP_CACHE_DIR` or
`~/.cache/rypip`), and transpiles it into the generated crate beside the
package's own modules.

- Explicit `rython.toml` `[python-modules]` entries always win over a
  fetched dependency (pin a local copy by vendoring it).
- `--no-deps` skips resolution entirely; `RYPIP_OFFLINE=1` fails loudly
  instead of fetching (a vendored or already-cached dependency still
  resolves offline).
- The dependency's source must fit rython's typed subset, like any other
  vendored library — a failed conversion names the module and construct.

`rython.toml` next to the package can also declare vendored Python
libraries with `[python-modules]`. Each entry maps an import name to a
`.py` file or a package directory:

```toml
[python-modules]
pylev = { path = "vendor/wf.py" }
textlib = { path = "vendor/textlib" }   # a package dir with __init__.py
```

The library is transpiled into the generated crate as a sibling module, so
both import spellings work as direct calls:

```python
import pylev
from pylev import wf_levenshtein as wf

print(wf("kitten", "sitting"))      # -> wf(...)?  (exception-propagating)
print(pylev.wfi_levenshtein("a", "b"))  # -> crate::pylev::wfi_levenshtein(...)?
```

Relative imports inside a vendored package resolve against the package path
(`from .core import double` in `textlib/__init__.py` becomes
`pub use crate::textlib::core::double;`, so `textlib.double` is a
re-exported attribute exactly as in Python). Kernel modules (`--kernel-module`)
compile a single entry file with no module tree and reject `[python-modules]`
loudly.

The library source must be written in rython's typed subset — annotate
parameters (`: str`, `: int`, ...) and returns (`-> int`, ...); unannotated
functions default to `Result<(), PyException>`. Modules targeting the
`from __future__` era's Python-2 shims (e.g. `PY2: range = xrange`) must
have those lines removed. Neither the dependency's own imports nor its
functions' bodies may rely on constructs the transpiler does not support —
a failed conversion names the module and the construct.

