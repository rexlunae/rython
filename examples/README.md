# Rython examples

Worked, verified examples of the rython toolchain. Each directory is
self-contained with its own README; every program here also runs under
CPython (that's the workflow — see the
[porting guide](../docs/porting-guide.md)), and each README states
exactly what was verified.

Build the tools once from the repo root (`cargo build -p rypip -p
rythonc`), or use `cargo run -p rypip -- ...` as the READMEs do.

| Example | What it shows |
|---|---|
| [01-rust-with-python-module](01-rust-with-python-module) | A Rust binary whose geometry module is written in Python — classes, inheritance, `super()`, dynamic dispatch — compiled into the crate at build time by `python_module!`. Three of its functions carry no annotations at all: their inferred generic signatures serve floats, ints, strings, and lists from the Rust side. |
| [02-gpu-numpy](02-gpu-numpy) | Array compute through rython's numpy subset with a selectable execution engine: scalar today, rayon/simd/cuda/vulkan as cargo-feature backends, chosen by `np.set_backend` / `--numpy-backend`, with loud errors for engines not compiled in. |
| [03-kernel-module](03-kernel-module) | Linux kernel drivers **maintained as Python**: `make` regenerates and builds the `.ko` from the Python on every edit (via the raw-FFI pipeline and the `rykernel-shim` compatibility layer — entry points, printk f-strings, `.modinfo`, a generated misc device), with load/unload targets; the same file's classes/inheritance become the userspace driver half. |
| [04-python-to-rust](04-python-to-rust) | The conversion walkthrough: a Python program, the `rypip convert` step, and the generated Rust crate checked in to read — with byte-identical output between CPython and the binary, and three fully unannotated helpers showing what parameter type inference generates. |
| [05-rypip-install](05-rypip-install) | `rypip install`: a `pyproject.toml` package installed as a native release binary on your PATH, with python-identical argparse `--help`. |

Two ground rules these examples inherit from the project:

- **Correct or loud.** A construct either behaves exactly like CPython
  or fails conversion/build with an error naming it. Where an example
  brushes against a boundary of the accepted subset
  ([spec](../docs/spec.md)), its README says so explicitly.
- **Verify against CPython.** Every example's expected output was
  produced by actually running both the Python and the generated
  binary; where the two can differ (numpy reduction rounding in 02),
  the README shows both.
