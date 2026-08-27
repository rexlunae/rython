# GPU-ready array compute through Python

[`simulate.py`](simulate.py) is a numpy program — midpoint integration of
a quarter circle and a semi-implicit Euler simulation of 1000 harmonic
oscillators. Converted with rython it becomes a native binary in which
every array operation funnels through one **execution engine**, and the
engine — CPU or GPU — is chosen without touching the program's logic.

## The backend model

`stdpython`'s numpy runtime dispatches every elementwise kernel and
reduction through an engine selected once per process:

| Backend  | cargo feature  | What it is                                  |
|----------|----------------|---------------------------------------------|
| `scalar` | (always built) | Sequential CPU engine                       |
| `rayon`  | `numpy-rayon`  | Multithreaded CPU (rayon)                   |
| `simd`   | `numpy-simd`   | Runtime-detected AVX2/AVX-512/SSE2/NEON     |
| `cuda`   | `numpy-cuda`   | NVIDIA GPUs (CUDA driver at runtime)        |
| `vulkan` | `numpy-vulkan` | Any Vulkan-capable GPU                      |

Selection is loud, never a silent fallback:

- `np.set_backend("cuda")` in the program pins the engine (that's what
  `--gpu` does in `simulate.py`);
- `rythonc --numpy-backend cuda input.py` pins it at startup without
  editing the program;
- `"auto"` (the default) picks the best engine compiled into the binary.

Requesting an engine the binary wasn't built with raises
`RuntimeError: numpy backend `cuda` was requested ... but stdpython was
built without its feature (`numpy-cuda`)` — rython's usual contract:
correct or loud, never silently different.

> **Status**: `scalar` is complete and is what `auto` resolves to in a
> default build. `rayon` is complete too — its kernels match the scalar
> ones element for element, and it falls back to the sequential loop
> below a size floor, since thread dispatch costs more than the work on
> small arrays. `simd` is currently an alias of `scalar`: the scalar
> kernels already auto-vectorize, and no hand-written intrinsics have
> been needed. `cuda` and `vulkan` are the wired-in dispatch targets —
> the feature gates, `np.set_backend` surface and loud-failure paths
> shown here are how a program opts in — but no GPU kernels ship yet, so
> selecting one is a loud `RuntimeError`, never a silent fallback to the
> CPU.

## Run it

```bash
# The program is plain Python first - check it under CPython + numpy:
python3 simulate.py --samples 10000

# Convert to a Rust crate and build the native binary:
cargo run -p rypip -- convert simulate.py --out /tmp/simulate-rs
cargo build --manifest-path /tmp/simulate-rs/Cargo.toml --release

# Same program, no interpreter:
/tmp/simulate-rs/target/release/simulate --samples 10000

# Ask for the GPU engine - loud RuntimeError on a CPU-only build:
/tmp/simulate-rs/target/release/simulate --gpu
```

Expected output with `--samples 10000`, from both:

```
pi ~= 3.14147731828716
oscillator <x^2> after 500 steps: 0.02779515904224993
```

The two agree byte for byte, reductions included: `np.sum`/`np.mean`
replicate numpy's `npy_pairwise_sum` exactly rather than accumulating
sequentially, so even the last bits match. argparse handling, `--help`
text and error output match byte for byte too.

## Notes on the numpy subset

- Arrays are values (copies), not views; `np.ndarray` annotations map to
  the runtime's `NdArray`.
- Elementwise arithmetic can use the ufuncs (`np.add`, `np.multiply`,
  `np.sqrt`, ...) — comparisons must (`np.greater(a, 3)`), since `a > 3`
  returns a bool *array*, which Rust's bool-typed comparison operators
  can't express.
- Reductions (`np.sum`, `np.mean`, ...) return `f64`.
- The full accepted surface is listed in
  [`crates/stdpython`](../../crates/stdpython/src/stdlib/numpy/mod.rs)
  and [`docs/spec.md`](../../docs/spec.md); anything outside it is a
  conversion-time error naming the construct.
