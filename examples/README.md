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
| [01-rust-with-python-module](01-rust-with-python-module) | A Rust binary whose geometry module is written in Python — classes, inheritance, `super()`, dynamic dispatch, inferred parameter types — compiled into the crate at build time by `python_module!`. |
| [02-gpu-numpy](02-gpu-numpy) | Array compute through rython's numpy subset with a selectable execution engine: scalar today, rayon/simd/cuda/vulkan as cargo-feature backends, chosen by `np.set_backend` / `--numpy-backend`, with loud errors for engines not compiled in. |
| [03-kernel-module](03-kernel-module) | Linux kernel modules from Python via the raw-FFI pipeline and the `rykernel-shim` compatibility layer: entry points, printk f-strings, `.modinfo`, a generated misc device — and the same file's classes/inheritance compiled into the userspace driver half. |
| [04-python-to-rust](04-python-to-rust) | The conversion walkthrough: a Python program, the `rypip convert` step, and the generated Rust crate checked in to read — with byte-identical output between CPython and the binary. |
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
