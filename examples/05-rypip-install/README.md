# Installing a Python package as a compiled Rust binary

[`linetool/`](linetool) is an ordinary PEP 621 Python package — a
`pyproject.toml` plus a package directory with `__init__.py` (a `Stats`
class) and `__main__.py` (an argparse CLI). It runs under CPython the
usual way:

```bash
cd linetool
printf 'the quick brown fox\njumps over the lazy dog\n' > /tmp/sample.txt
python3 -m linetool /tmp/sample.txt
#   2 9 44 /tmp/sample.txt
```

`rypip install` is the pip-shaped path from that package to a native
executable on your `$PATH`:

```bash
cargo run -p rypip -- install .
#   ...
#   Installed package `linetool v0.1.0` (executable `linetool`)

linetool /tmp/sample.txt
#   2 9 44 /tmp/sample.txt
linetool --help          # byte-identical to python3 -m linetool --help
```

One command did all of this:

1. **Discover** — read `pyproject.toml` (name, version, packages;
   `setup.cfg`/`setup.py` layouts work too), collect the package
   modules, and resolve declared PyPI dependencies (vendorable pure
   subset; `--no-deps` skips).
2. **Convert** — each Python module becomes a Rust module
   (`linetool/__init__.py` → `lib.rs`, `__main__.py` → the binary), the
   conversion-time argparse builds python-identical `--help`/error
   output, and `with open(...)` maps onto the runtime's file objects
   with the CPython `OSError` hierarchy.
3. **Build** — `cargo build --release` against the `stdpython` runtime.
4. **Install** — the binary lands where cargo installs binaries
   (`~/.cargo/bin`, or `--root DIR` to choose; add it to `PATH`).

The result starts with no interpreter, no venv, and no GIL — a
`FileNotFoundError` still exits 1 with the error named, and the CLI
surface stays what argparse defined.

Related commands:

```bash
cargo run -p rypip -- build .                 # convert + release build, no install
cargo run -p rypip -- convert . --out crate/  # just generate the crate (see ../04-python-to-rust)
```
