# Linux kernel modules written in Python

Two Python files, three build targets:

- [`hello.py`](hello.py) — the classic hello-world module: kernel entry
  points, `printk` with f-strings, module metadata, and a kernel symbol
  called through **rykernel-shim**, the Rust compatibility layer.
- [`driver.py`](driver.py) — a tiny register-file device. The **same
  file** builds the kernel half (`--kernel-module`: a misc character
  device at `/dev/rython0`) and the userspace half (`--driver`: the
  protocol logic — classes, inheritance, `super()`, inferred parameter
  types — wrapped in generated syscall glue).

No C anywhere, and no rust-for-linux tree required: `rykernel-shim`
declares the exported kernel symbols a module may use (kmalloc-backed
allocator, `misc_register` and the full `file_operations` layout,
user-copy helpers, a panic handler), and the generated crate is ordinary
`#![no_std]` Rust you can read and edit.

## hello.py — entry points and the shim

```bash
cargo run -p rypip -- convert hello.py --out hello-kmod --kernel-module
cd hello-kmod
make            # needs linux-headers + nightly rust (see the generated Makefile)
sudo insmod hello.ko
sudo dmesg | tail -2
#   hello_rython: loaded at unix time 1756…
sudo rmmod hello
```

What the conversion does:

- `module_init()` / `module_exit()` become `#[no_mangle] extern "C"
  init_module`/`cleanup_module`.
- `__module_license__`, `__module_author__`, ... become `.modinfo` ELF
  entries (license defaults to GPL — modpost requires one).
- `printk(f"... {now}\n")` lowers to the kernel's `_printk` with `%ld`
  conversions — the f-string is compiled, not formatted at runtime.
- `from rykernel_shim import ktime_get_real_seconds` resolves at
  conversion time against the shim's allowlist and lowers to a direct
  call. Importing a name that isn't an exported kernel resource is a
  loud conversion error, not a link failure at insmod time.

**The kernel-context subset is deliberately small.** Code that runs in
kernel context is limited to entry points, integer locals, printk, and
shim calls — and floating point is a loud conversion error (the kernel
runs with the FPU in a lazy-save state; emitting FP code could corrupt
userspace registers). Anything fancier belongs on the other side of the
device boundary, which is exactly what `driver.py` demonstrates: the
kernel half stays a small, auditable byte ring; the full Python
language surface (classes, inheritance, dicts, f-strings) runs in the
userspace driver, compiled by the full transpiler.

## driver.py — a device in two halves

```bash
# Kernel half: the device manifest (__device_name__, __bufsz__, __magic__,
# __ioc_reset__, ...) generates src/device.rs - a misc byte-ring device
# with full file_operations - plus entry points that register it.
cargo run -p rypip -- convert driver.py --out rython-kmod --kernel-module
(cd rython-kmod && make && sudo insmod rython.ko)

# Userspace half: the classes below compile to the driver crate's lib.rs;
# generated glue (main.rs) does open/read/write/ioctl against /dev/rython0.
cargo run -p rypip -- convert driver.py --out rython-driver --driver
cd rython-driver
cargo test          # unit tests for the compiled Python logic (protocol + CRC-8)
cargo run           # interactive: WRITE 2 2a / READ 2 / DUMP / STATS / RESET
```

The Python follows ordinary Python practice, and it all survives the
trip to Rust:

- `RegisterBank` (base class) owns bounds-checked storage; `Device`
  layers the wire protocol on top by **inheritance**, overrides
  `clear()` and extends it with `super().clear()`.
- `def within(value, low, high)` has **no annotations** — rython infers
  a generic, comparison-bounded Rust signature from use.
- The protocol handler is plain string-splitting, dicts, and f-strings.
- It runs under CPython first: `python3 driver.py` exercises the same
  assertions the generated crate re-runs as Rust unit tests
  (`cargo test` in the driver crate: protocol round-trip, bounds
  errors, DUMP/STATS/RESET, CRC-8 against a reference implementation).

## Notes

- The generated Makefile documents its own pipeline: `cargo build
  --target x86_64-unknown-none` → `ld -r --gc-sections` → Kbuild
  (modpost, BTF, final link). Building the `.ko` needs the running
  kernel's headers and a nightly toolchain for `-Zbuild-std`; the
  crates themselves `cargo check` on any host.
- `--kernel-module --rust-for-linux` instead emits a `module!`-macro
  crate implementing `kernel::Module`, for in-tree builds with the
  rust-for-linux toolchain.
- See [`crates/rykernel-shim`](../../crates/rykernel-shim) for what the
  compatibility layer provides.
