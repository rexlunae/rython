# Rython 🐍→🦀

A Rust workspace for compiling Python to Rust via PyO3.

## Crates

| Crate | Description |
|---|---|
| [`python-ast`](crates/python-ast) | Core library — parses Python AST and transpiles to Rust |
| [`python-mod`](crates/python-mod) | Proc-macro for including Python modules in Rust source |
| [`rythonc`](crates/rythonc) | CLI compiler — `rythonc input.py → output.rs` |
| [`stdpython`](crates/stdpython) | Python standard library runtime implemented in Rust |

## Usage

```bash
# Build all crates
cargo build

# Compile a Python file
cargo run -p rythonc -- input.py -o output.rs

# Run tests for a specific crate
cargo test -p python-ast
```

## Publishing

```bash
# Publish all crates in dependency order
cargo publish -p python-ast
cargo publish -p stdpython
cargo publish -p python-mod
cargo publish -p rythonc
```

## License

Apache-2.0
