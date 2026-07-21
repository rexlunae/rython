# python-mod

[![Crates.io](https://img.shields.io/crates/v/python-mod.svg)](https://crates.io/crates/python-mod)
[![Documentation](https://docs.rs/python-mod/badge.svg)](https://docs.rs/python-mod)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

A Rust procedural macro library for embedding Python code (Rython) directly into Rust projects. This crate allows you to write Python functions and have them compiled to Rust at build time, providing seamless integration between Python-like syntax and Rust performance.

## Overview

**Rython** is a limited subset of Python that can be compiled to Rust. This crate provides procedural macros that:

- Parse Python source files at compile time
- Generate corresponding Rust code using the `python-ast` crate
- Create Rust modules that can be used like native Rust code
- Support both single-file modules (`.py`) and package-style modules (`__init__.py`)

> **Note**: Rython is currently a very limited subset of Python. This will hopefully expand over time as the ecosystem develops.

## Features

- 🔄 **Compile-time Python-to-Rust translation**
- 📦 **Support for both single files and package modules**
- 🛠️ **Two macro variants**: `python_module!` (with std) and `python_module_nostd!` (without std)
- 📝 **Rust code preamble support** for importing native Rust dependencies
- ⚡ **Zero runtime overhead** - everything happens at compile time
- 🧪 **Comprehensive test suite** with 23+ tests covering various scenarios

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
python-mod = "1.0.0"
```

**Important**: This crate requires nightly Rust due to the use of experimental procedural macro features.

```bash
rustup install nightly
rustup default nightly
```

## Quick Start

### 1. Create a Python Module

Create a file `src/math_ops.py`:

```python
# Simple math operations
def add(a, b):
    return a + b

def multiply(x, y):
    return x * y

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Constants
PI = 3.14159
```

### 2. Import the Module in Rust

In your `src/main.rs` or `src/lib.rs`:

```rust
use python_mod::python_module;

// Import the Python module
python_module!(math_ops);

fn main() {
    // Use Python functions as if they were native Rust
    let result = math_ops::add(5, 3);
    println!("5 + 3 = {}", result);
    
    let product = math_ops::multiply(4, 6);
    println!("4 * 6 = {}", product);
    
    let fib = math_ops::fibonacci(7);
    println!("fibonacci(7) = {}", fib);
    
    let pi = math_ops::PI;
    println!("π ≈ {}", pi);
}
```

### 3. Build and Run

```bash
cargo build
cargo run
```

## Usage Examples

### Basic Module Import

```rust
use python_mod::python_module;

// Import a single Python file: src/utils.py
python_module!(utils);

fn example() {
    let result = utils::some_function(42);
}
```

### Package Module Import

For a package-style module, create `src/my_package/__init__.py`:

```python
def package_function():
    return "Hello from package!"

def calculate_square(x):
    return x * x
```

Then import it:

```rust
python_module!(my_package);

fn example() {
    let message = my_package::package_function();
    let square = my_package::calculate_square(5);
}
```

### Module with Rust Preamble

```rust
python_module!{advanced_module
    // Rust code that will be included at the top of the generated module
    use std::collections::HashMap;
    use std::result::Result;
    
    // This code becomes part of the generated module
    type MyResult<T> = Result<T, String>;
}

fn example() {
    // Now you can use both the Rust preamble and Python functions
    let result = advanced_module::python_function();
}
```

### Module Without Standard Library

```rust
use python_mod::python_module_nostd;

// Import without including standard Python libraries
python_module_nostd!(minimal_module);

fn example() {
    let result = minimal_module::simple_function();
}
```

## Supported Python Features

Currently, Rython supports a limited subset of Python syntax:

### ✅ Supported
- Function definitions (`def`)
- Return statements
- Basic arithmetic operations (`+`, `-`, `*`, `/`)
- Variable assignments
- Constants
- Basic control flow (`if`, `else`)
- Recursive functions
- String operations
- Integer and float literals

### ⚠️ Limited/Experimental
- List operations
- Dictionary operations
- Loop constructs

### ❌ Not Yet Supported
- Classes and objects
- Imports
- Exception handling
- Advanced Python features
- Most of the Python standard library

## File Structure

The macro looks for Python files in the following locations:

1. **Single file module**: `src/{module_name}.py`
2. **Package module**: `src/{module_name}/__init__.py`

Example project structure:
```
my_project/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── math_utils.py          # Single file module
│   ├── string_ops.py          # Another single file module
│   └── advanced/              # Package module
│       └── __init__.py
```

Usage:
```rust
python_module!(math_utils);    // Imports src/math_utils.py
python_module!(string_ops);    // Imports src/string_ops.py  
python_module!(advanced);      // Imports src/advanced/__init__.py
```

## API Reference

### `python_module!(module_name)`

Imports a Python module with standard library support.

**Parameters:**
- `module_name`: The name of the Python module to import

**Example:**
```rust
python_module!(my_module);
```

### `python_module_nostd!(module_name)`

Imports a Python module without standard library support.

**Parameters:**
- `module_name`: The name of the Python module to import

**Example:**
```rust
python_module_nostd!(my_module);
```

### `python_module!{module_name /* Rust preamble */}`

Imports a Python module with custom Rust code preamble.

**Parameters:**
- `module_name`: The name of the Python module to import
- Rust code block: Code to include at the top of the generated module

**Example:**
```rust
python_module!{my_module
    use std::collections::HashMap;
    type MyError = String;
}
```

## Development

### Building
```bash
cargo build
```

### Testing
The crate includes a comprehensive test suite with 23+ tests:

```bash
# Run all tests
cargo test

# Run specific test categories
cargo test --test lib_tests           # Core functionality tests
cargo test --test documentation_tests # Documentation validation
cargo test --test edge_case_tests     # Edge cases and boundaries
```

### Contributing

1. Ensure you're using nightly Rust
2. Run the full test suite: `cargo test`
3. Add tests for new functionality
4. Update documentation as needed

## Architecture

The crate works by:

1. **Parse-time**: The procedural macro reads Python source files
2. **AST Generation**: Uses `python-ast` to parse Python into a Rust AST
3. **Code Generation**: Converts the Python AST to Rust `TokenStream`
4. **Module Creation**: Wraps generated code in a Rust module

```
Python Source → python-ast → Rust AST → TokenStream → Rust Module
```

## Limitations

- **Nightly Rust Required**: Uses experimental proc-macro features
- **Limited Python Subset**: Only basic Python constructs are supported
- **Compile-time Only**: Python code is translated at build time, not runtime
- **No Python Runtime**: Generated code is pure Rust, no Python interpreter needed

## Roadmap

Future improvements may include:

- [ ] Expanded Python syntax support
- [ ] Better error messages and debugging
- [ ] Support for Python classes
- [ ] Integration with more Python constructs
- [ ] Performance optimizations
- [ ] Better tooling and IDE support

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Dependencies

- `quote` - For Rust code generation
- `proc-macro2` - Advanced procedural macro functionality  
- `proc-macro-error` - Better error handling in macros
- `python-ast` - Python AST parsing and Rust code generation
- `path_macro` - Path manipulation utilities

## Related Projects

- [PyO3](https://github.com/PyO3/pyo3) - Python bindings for Rust
- [python-ast](https://crates.io/crates/python-ast) - Python AST parsing for Rust
- [Rython](https://github.com/rexlunae/rython) - The Python subset compiler

---

**Note**: This project is in active development. The Python subset supported by Rython is currently limited, but is expected to grow over time. Contributions and feedback are welcome!