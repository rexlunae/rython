# rython-jit

On-demand compilation of Python through the rython converter (issue #166;
the design and what milestone M1 shipped are in
[`docs/jit.md`](../../docs/jit.md)).

`Jit::compile` converts a module's source with the same converter
`rypip convert` uses (in its cdylib mode), builds it with cargo into a
shared library, caches the library by content under
`$XDG_CACHE_HOME/rython/jit` (default `~/.cache/rython/jit`, mode 0700),
and loads it into the running process. `Module::call` calls one of its
public functions; `Module::run_entry` runs it as a program. Nothing is
interpreted: a source the converter refuses is `JitError::Conversion`
with the converter's message, and every other failure is a typed, loud
`JitError`.

```rust
use rython_jit::Value;

let jit = rypip::jit::new_jit(rypip::convert::ConvertOptions::default(), None)?;
let module = jit.compile("geometry", "def area(w: float, h: float) -> float:\n    return w * h\n")?;
let area: f64 = module.call_as("area", &[Value::Float(2.0), Value::Float(3.5)])?;
assert_eq!(area, 7.0);
```

The converter lives in `rypip`, which depends on this crate for
`rypip jit`, so it comes in through the `Compiler` trait;
`rypip::jit::new_jit` wires up rypip's.

M1 limits: Unix hosts; one module per compile; `int` (64-bit), `float`,
`bool`, `str` and `None` across the boundary; the Rust toolchain and the
CPython the parser uses are needed at JIT time (never by converted
output).
