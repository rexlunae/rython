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

# Convert and compile without installing:
rypip build path/to/package
```

## Package discovery

`rypip` accepts a single `.py` file, a package directory (containing
`__init__.py`), or a project directory with a `pyproject.toml` — `[project]
name` and `version` are used for the generated crate, and both flat and
`src/` layouts are recognized.

A module containing an `if __name__ == "__main__":` block (or a
`__main__.py`) becomes the binary entry point; packages without one convert
to library crates and cannot be `install`ed.

## Generated crates

Each Python module becomes a Rust module; subpackages become nested modules.
The crate depends on the `stdpython` runtime (locate it with `--stdpython`,
`RYPIP_STDPYTHON_PATH`, or the copy alongside this tool's source tree).

With `--pyo3`, the crate gains a `python` cargo feature, a `cdylib` target,
and a generated `#[pymodule]` exposing every top-level function whose
signature is expressible in concrete Rust types (parameters annotated with
`int`/`float`/`str`/`bool`/`bytes`, returns annotated or inferable). Build
the extension with `cargo build --features python`.
