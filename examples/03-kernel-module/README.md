# Linux kernel drivers you write — and maintain — in Python

The Python files in this directory ARE the drivers. You edit them, and
`make` rebuilds the kernel modules from them every time; the generated
Rust under `build/` is a disposable intermediate, like an `.o` file.
There is no "port to Rust and maintain the Rust" step — the Python stays
the source of truth.

- [`hello.py`](hello.py) — the classic hello-world module: kernel entry
  points, `printk` with f-strings, module metadata, and a kernel symbol
  called through **rykernel-shim**, the Rust compatibility layer.
- [`driver.py`](driver.py) — a register-file device in one file: its
  manifest becomes the kernel half (a misc device at `/dev/rython0`),
  and its classes — inheritance, `super()`, inferred parameter types —
  become the userspace half that drives the device node.
- [`Makefile`](Makefile) — the whole workflow: build, load, unload,
  logs.

No C anywhere, and no rust-for-linux tree required: `rykernel-shim`
declares the exported kernel symbols a module may use (kmalloc-backed
allocator, `misc_register` and the full `file_operations` layout,
user-copy helpers, a panic handler).

## The workflow

```bash
# One-time toolchain setup (the .ko build rebuilds core/alloc for the
# freestanding kernel target):
sudo apt install linux-headers-$(uname -r)      # your distro's package
rustup toolchain install nightly --profile minimal
rustup +nightly target add x86_64-unknown-none
rustup +nightly component add rust-src

# Edit hello.py or driver.py, then:
make                    # regenerate + build hello.ko and rython.ko
make load               # insmod both (sudo)
make dmesg              #   hello_rython: loaded at unix time 1756…
                        #   rython: loaded — /dev/rython0 ready (magic 0x52594854)

make tool               # build the userspace driver from the same driver.py
./build/rython-tool/target/release/driver       # WRITE 2 2a / READ 2 / DUMP ...

make unload             # rmmod both
```

Iterating is just: edit the Python → `make` → `make unload load`. Run
`python3 driver.py` first for a CPython smoke test of the protocol
logic — the same behavior the generated crate re-checks as Rust unit
tests (`make check` runs them).

On a machine without kernel headers (CI, a container), `make check`
still verifies everything rython-side: it regenerates both crates from
the Python, cargo-checks the kernel crates, and runs the userspace
driver's generated test suite.

## What `make` actually does

1. `rypip convert <file>.py --out build/<name> --kernel-module` — lowers
   the Python to a `#![no_std]` Rust crate: `module_init`/`module_exit`
   become `extern "C"` entry points, `__module_*__` metadata becomes
   `.modinfo` ELF entries (license defaults to GPL — modpost requires
   one), `printk(f"...")` compiles to the kernel's `_printk` with `%ld`
   conversions, and `from rykernel_shim import ktime_get_real_seconds`
   resolves at conversion time against the shim's allowlist — an
   unknown name is a loud conversion error, not a link failure at
   insmod time. A device manifest (`__device_name__`, `__bufsz__`,
   `__magic__`, ioctl numbers) additionally generates a misc byte-ring
   device with full `file_operations`.
2. The generated crate's own Makefile builds the `.ko`:
   `cargo +nightly -Zbuild-std` (freestanding staticlib, no PIC) →
   `ld -r --gc-sections` (one relocatable object) → Kbuild (modpost,
   BTF, final link).

**The kernel-context subset is deliberately small.** Code that runs in
kernel context is limited to entry points, integer locals, printk, and
shim calls — and floating point is a loud conversion error (the kernel
runs with the FPU in a lazy-save state). Anything fancier belongs on
the other side of the device boundary, which is what `driver.py`
demonstrates: the kernel half stays a small, auditable byte ring; the
full Python language surface (classes, inheritance with `super()`,
dicts, f-strings, unannotated inferred-generic helpers like
`within(value, low, high)`) runs in the userspace driver, compiled by
the full transpiler — and that logic is Python you keep maintaining as
Python.

## Notes

- Verified in this repo: `make check` end to end, plus the `.ko`
  pipeline through `cargo -Zbuild-std` and the `ld -r` link (the final
  Kbuild/modpost step needs the target machine's kernel headers).
- `rypip convert --kernel-module --rust-for-linux` instead emits a
  `module!`-macro crate implementing `kernel::Module`, for in-tree
  builds with the rust-for-linux toolchain.
- See [`crates/rykernel-shim`](../../crates/rykernel-shim) for what the
  compatibility layer provides.
