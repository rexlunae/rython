# Test Suite for python-mod

This directory contains comprehensive tests for the `python-mod` crate.

## Test Structure

### `lib_tests.rs`
- Core library functionality tests
- File system operations and path handling
- Module loading logic validation
- String formatting and error message testing

### `documentation_tests.rs`
- Documentation example validation
- Macro syntax verification
- Usage pattern testing
- Rython/PyO3 integration documentation

### `edge_case_tests.rs`
- Boundary condition testing
- File system edge cases
- Python syntax validation
- Directory structure verification
- Module naming convention tests

## Test Python Modules

The following Python modules are used for testing:

- `src/simple_math.py` - Basic arithmetic functions and constants
- `src/string_ops.py` - String manipulation functions  
- `src/empty_module.py` - Empty module for edge case testing
- `src/minimal_test.py` - Minimal function for basic testing
- `src/package_test/__init__.py` - Package-style module testing
- `src/simple_math_with_preamble.py` - Module for preamble testing

## Running Tests

```bash
# Run all tests
cargo test

# Run specific test file
cargo test --test lib_tests
cargo test --test documentation_tests
cargo test --test edge_case_tests

# Run tests with output
cargo test -- --nocapture

# Run tests in verbose mode
cargo test --verbose
```

## Test Coverage

The test suite covers:

1. **Macro Compilation**: Verifies macros compile without errors
2. **File System Operations**: Tests module loading and path resolution
3. **Documentation Examples**: Validates README and code examples
4. **Edge Cases**: Tests boundary conditions and error scenarios
5. **Python Syntax**: Validates Python file syntax and structure
6. **Module Structure**: Tests both single-file and package modules

## Limitations

Due to the current state of the `python-ast` crate and missing `stdpython` dependency, the tests focus on:

- Static analysis and validation
- File system operations  
- Macro compilation verification
- Documentation and example validation

Future test improvements could include:
- Runtime function execution tests (requires `stdpython`)
- Integration tests with actual Python code execution
- Performance benchmarks
- Memory usage testing