# Evaluating rython's numpy against CPython

An end-to-end evaluation of the `numpy` subset in `crates/stdpython`: does
a converted program print what CPython + real numpy prints, and how fast is
it, on every execution backend that runs on this machine.

Everything below is reproducible with the harnesses in this directory
(`README.md` has the commands). Raw results are the two JSON files the
harnesses write.

## Environment

| | |
|---|---|
| CPU | 4 × Intel Xeon @ 2.10 GHz |
| rustc | 1.98.0 |
| Python / numpy | 3.11.15 / 2.4.6 |
| rython | `768fb53` |

> The pinned `stable` toolchain in the container was 1.94.1, on which the
> workspace does not build: `python-ast` uses `if let` match guards
> (`call.rs:3808`, `class_def.rs:2298`), stabilized later. `rustup update
> stable` to 1.98.0 fixed it. CI tracks `stable`, so this only bites a
> pinned-older toolchain — but the workspace has no `rust-version` floor
> to say so.

## Headline

- **18 of 58 cases** are byte-identical to CPython. 21 diverge, 12 fail to
  compile as generated Rust, 6 panic at runtime, 1 is a clean
  conversion-time rejection.
- **Nearly every divergence traces to one area: float array printing.**
  The kernels and reductions are in good shape — `np.sum` reproduces
  numpy's pairwise summation bit-for-bit — but `str(array)` for float
  dtypes ignores numpy's `precision=8`, mis-places the sign, omits column
  padding, never wraps or summarizes, and *panics* on any array that needs
  exponent notation.
- **All five backend configurations agree exactly.** 150 case-runs of the
  first 40 cases across `default/auto`, `scalar`, `rayon`, `simd` and
  `auto`: byte-identical stdout and exit codes, no exceptions.
- **`RYPY_NUMPY_BACKEND` is documented in three places and read by
  nothing.** Setting it is a silent no-op; `np.set_backend()` is the only
  working selector, and it does fail loudly and correctly.
- **Speed: rython is currently slower than numpy on array work** —
  geometric mean 0.12x–0.41x per benchmark program — with two identified
  root causes (a full buffer copy per reduction, and materializing a
  broadcast scalar into a full-size array). It wins decisively on process
  startup (1.5 ms vs 84.5 ms) and on sub-10k-element reductions, where
  CPython's per-call dispatch dominates.

## Method

58 case programs (`cases/`) exercise creation, dtypes, printing, ufuncs,
comparisons, reductions, shape ops, sorting, linalg, indexing, slicing,
operators, special values, error behaviour and the ndarray attribute /
method surface. Each is run under CPython, then converted with `rypip
convert`, built, and run; stdout and exit code are diffed byte-for-byte.

All five backend configurations were run for cases 01–40; cases 41–58
were added later and run on the default configuration only, after B3
below had established that the backends agree.

Backends are pinned by rewriting the program to call `np.set_backend(...)`
as `main()`'s first statement and adding the matching stdpython features
to the generated `Cargo.toml`. Five configurations were run per case:

| configuration | stdpython features | engine |
|---|---|---|
| `default-auto` | (none) | `auto` → scalar |
| `cpu-scalar` | `numpy-rayon`, `numpy-simd` | `scalar` |
| `cpu-rayon` | `numpy-rayon`, `numpy-simd` | `rayon` |
| `cpu-simd` | `numpy-rayon`, `numpy-simd` | `simd` |
| `cpu-auto` | `numpy-rayon`, `numpy-simd` | `auto` → rayon |

`cuda` and `vulkan` were also built (the features compile) and exercised;
they ship no kernels, so selecting one is a loud `RuntimeError` by design.

Benchmarks (`bench/`) time their own kernels with `time.perf_counter`, so
the numbers exclude interpreter and process startup, which is measured
separately. Each kernel is a dependency chain or accumulates a scalar, so
no iteration can be hoisted or eliminated; best-of-3 runs are reported.

## Correctness results

| status | cases | meaning |
|---|---|---|
| PASS | 18 | byte-identical stdout and exit code |
| DIVERGE | 21 | ran, printed something different |
| BUILD_FAIL | 12 | generated Rust does not compile |
| RUN_FAIL | 6 | panicked at runtime |
| CONVERT_FAIL | 1 | loud conversion error (the documented contract) |

PASS compares stdout and exit code; one case (`42_index_error`) passes on
both while differing in the exception message on stderr (D2 below).

Per-case, all five backend configurations gave the same status; the full
grid is in `summarize.py`'s output.

### A. Silent wrong output

The project's prime directive is "correct or loud — never silently
different". These are the cases that are silently different.

**A1 — float array printing ignores `precision=8`.** numpy formats array
floats with `dragon4_positional(precision=8, ...)`; rython prints the full
`repr`.

```
np.divide(np.array([1.0]), np.array([3.0]))
CPython:  [0.33333333]
rython:   [0.3333333333333333]
```
`cases/36_precision.py`. This alone accounts for most of the 16 DIVERGE
cases.

**A2 — cells are not padded to a common width.** numpy right-pads the
fractional part so columns line up; `np_float_cell` pads only the integer
part (`ndarray.rs:578`).

```
np.linspace(0.0, 1.0, 5)
CPython:  [0.   0.25 0.5  0.75 1.  ]
rython:   [0. 0.25 0.5 0.75 1.]
```
`cases/01_creation_basic.py`.

**A3 — the sign is emitted outside the padding.** `format!("{}{:>width$}.",
sign, int_part, ...)` puts the `-` *before* the left pad instead of inside
it (`ndarray.rs:578`, `ndarray.rs:583`).

```
np.array([-1.0, 2.0])
CPython:  [-1.  2.]
rython:   [-  1.   2.]
```
`cases/37_negative_pad.py`. Integer arrays are correct — `np_int_cell`
formats the whole signed value.

**A4 — `inf` / `nan` cells get a spurious `.`.** `float_layout` skips
non-finite values, so `np_float_cell` falls into the "no decimal point"
branch and appends one.

```
np.divide(np.array([1.0]), np.zeros(1))
CPython:  [inf]
rython:   [inf.]
```
`cases/35_inf_nan_print.py`.

**A5 — no line wrapping.** numpy wraps rows at `linewidth=75`; rython
prints one long line. `cases/31_linewrap.py`.

**A6 — no summarization.** Above `threshold=1000` numpy prints
`[   0    1    2 ...  998  999 1000]`; rython prints all 1001 elements.
`cases/32_summarize.py`.

**A7 — `a.shape` prints a list, not a tuple.** `NdArray::shape` is a public
`Vec<usize>` field, so `a.shape` compiles and prints through `Vec`'s
display.

```
np.array([1.0, 2.0, 3.0]).shape
CPython:  (3,)
rython:   [3]
```
`cases/56_attrs_shape.py`. `.ndim` and `.size` are correct.

**A8 — negative slice bounds are ignored.** `PySlice::py_slice` clamps a
negative bound to `0` (`ndarray.rs:921`, `ndarray.rs:925`) instead of adding `len`, so
Python's negative-index convention is lost. Positive bounds and steps
(including `a[::-1]`) are correct.

```
a = np.arange(10)
a[-3:]    CPython [7 8 9]          rython [0 1 2 3 4 5 6 7 8 9]
a[:-3]    CPython [0 1 2 3 4 5 6]  rython []
a[-5:-2]  CPython [5 6 7]          rython []
```
`cases/58_slice_negative.py`. This is the most dangerous finding here: it
silently returns a *different array*, not just different formatting.

**A9 — `np.std(a, 1)` means something else.** numpy's second positional
parameter is `axis`; rython's is `ddof`. Same call, different answer, no
diagnostic.

```
a = np.array([[1.0, 2.0], [3.0, 4.0]]); np.std(a, 1)
CPython:  [0.5 0.5]      (per-row std)
rython:   1.2909944487358056   (whole-array std with ddof=1)
```
`cases/26_std_ddof_positional.py`.

**A10 — reductions on int/bool arrays return float.** `np.sum(np.array([1,
2, 3]))` prints `6.0`, not `6`. This one *is* deliberate and explained in
`reduce.rs`'s module docs ("the one deliberate numeric divergence"), and
in `examples/02-gpu-numpy/README.md` — but it is not in the `docs/spec.md`
§12 ledger, which the repo's own AGENTS.md calls the place to check.
`cases/11_reductions_int.py`, `cases/19_dtype_promotion.py`.

**A11 — `np.linalg.det`/`inv` differ in the last digits.** rython's
implementation is not LAPACK's, so results differ at the ULP level:
`np.linalg.det(np.array([[1.0, 2.0], [3.0, 4.0]]))` is
`-2.0000000000000004` under numpy and `-2.0` under rython. Reasonable as a
divergence, but undocumented, and visible whenever a determinant is
printed as a scalar. `cases/14_linalg.py`, `cases/28_matmul_bigger.py`.

### B. Backends

**B1 — `RYPY_NUMPY_BACKEND` is never read.** `engine.rs`'s module docs list
it as selection path 2, `rythonc --help` says `--numpy-backend` "overrides
RYPY_NUMPY_BACKEND", and `python_options.rs:362` names it — but nothing in
the workspace calls `env::var` for it. Setting it silently leaves the
engine on `auto`:

```
RYPY_NUMPY_BACKEND=cuda   ./prog   # runs fine on the CPU, exit 0
RYPY_NUMPY_BACKEND=bogus  ./prog   # runs fine, exit 0
```

Both should be loud. This is the same class as A1–A9 — a documented
control that silently does nothing — and it is why the first pass of this
evaluation produced a meaningless "all backends agree" result.

**B2 — `np.set_backend()` behaves exactly as designed.** Verified across
both feature sets:

| request | build | result |
|---|---|---|
| `scalar` | any | runs |
| `rayon`, `simd` | `numpy-rayon`,`numpy-simd` | runs |
| `rayon`, `simd` | cuda+vulkan build | `RuntimeError: … built without its feature (numpy-rayon)`, exit 101 |
| `cuda`, `vulkan` | cpu build | `RuntimeError: … built without its feature (numpy-cuda)`, exit 101 |
| `cuda`, `vulkan` | cuda+vulkan build | `RuntimeError: … not implemented in this build (no cuda kernels ship yet)`, exit 101 |
| `bogus` | any | `RuntimeError: unknown numpy backend 'bogus' (expected one of: …)`, exit 1 |

No silent fallbacks anywhere. `auto` never selects an unimplemented GPU
engine.

**B3 — all backends produce identical results.** Across 150 case-runs
(30 cases that produced output × 5 configurations) stdout and exit codes
are byte-identical. The only stderr difference is the PID embedded in
panic messages. `simd` is documented as an alias of `scalar`, and measures
as one.

### C. Loud, but at the wrong layer

These fail as rustc errors in the generated crate rather than as
conversion-time messages — the class `docs/spec.md` §12.1 already tracks
for other features. Nothing is silently wrong, but the diagnostic points
at generated Rust instead of the Python line.

**C1 — every `dtype=` keyword generates invalid Rust.** `np_dtype_tokens`
(`call.rs:722`) interpolates the variant name as a `&str`, so `quote!`
emits a string literal where an identifier belongs:

```rust
numpy::zeros((3).clone(), numpy::Dtype::"Int64")   // expected identifier
```

This breaks `np.zeros(n, dtype=...)`, `np.ones(...)`, `np.empty(...)` for
*every* dtype, including `float64`. It is a one-token fix
(`format_ident!`), and it is why six cases here (`03`, `05`, `23`, `24`,
`25`, `33`) cannot be evaluated at all — the whole non-default-dtype
surface is currently unreachable from Python.

**C2 — arithmetic operators move their operand.** The ufunc call path
clones arguments (`np_render`), the operator path does not:

```python
print(b / a)
print(a * 2.0)     # error[E0382]: borrow of moved value: `a`
```
`cases/17_operators.py`. Two uses work (`cases/40_operator_reuse.py`
passes); the third moves.

**C3 — list arguments move their elements.** `np.concatenate([p, q],
axis=0)` renders `vec![p, q]` without cloning, so a second concatenate of
the same arrays is `E0382`. `cases/38_concat_axis_kw.py`.

**C4 — a local assigned from an `np.ndarray`-returning function is typed
`PyValue`.** `v = build(5)` where `build(n: int) -> np.ndarray` emits
`v = PyValue::from(build(5)?)` → `E0277`, then `E0308` at every use.
`cases/21_functions_typed.py`. This is the documented "return annotation
not propagated into `name = call(...)`" gap, and it bites numpy code hard:
the natural way to factor array code is helpers that return arrays. Both
benchmark programs here had to inline their array-building helpers.

**C5 — `a.astype(np.int64)`** → `E0425: cannot find value 'int64' in module
'np'` (`np.int64` is only handled as a call or a `dtype=` value).

**C6 — `a.T` → `E0609`**, `print(a.dtype)` → `E0277: Dtype: PyDisplay`.
`cases/46_attributes.py`.

**C7 — `np.dot` returns an array for the 1-D × 1-D case.** numpy returns a
scalar. `print(np.dot(a, b))` matches, but `acc + np.dot(a, b)` is a type
error; `np.vdot` returns `f64` and works.

**C8 — `np.random.*` converts, then fails at build** with `E0433: cannot
find 'random' in 'np'`. The conversion-time surface check does not cover
submodules other than `linalg`. `cases/52_random.py`.

The good news: `a.sum()`, `a.mean()`, `a.max()`, `a.min()`, `a.reshape()`,
`a.ravel()`, `a.copy()`, boolean-mask indexing, in-place `+=`/`*=`,
iteration and broadcasting all work and match CPython
(`cases/47_methods_core.py`, `49`, `51`, `53`, `54` — all PASS).

### D. Exceptions

**D1 — numpy errors arrive as Rust panics, not typed exceptions.** Shape
mismatch, singular matrix and empty-array reductions all `panic!` (exit
101 plus a Rust backtrace note) where CPython raises and exits 1:

| case | CPython | rython |
|---|---|---|
| `np.add` shape mismatch | `ValueError: operands could not be broadcast together with shapes (2,) (3,)`, exit 1 | panic at `ufunc.rs:234`, exit 101 |
| `np.linalg.inv` singular | `LinAlgError: Singular matrix`, exit 1 | panic at `linalg.rs:235`, exit 101 |
| `np.max` of empty | `ValueError: zero-size array to reduction …`, exit 1 | panic at `reduce.rs:116`, exit 101 |

The panic *text* carries the right exception type and a reasonable
message, so this is the documented "panics where unrepresentable" model
rather than a silent difference — but the exit code differs and the error
is not catchable. `cases/41`, `43`, `44`.

**D2 — `IndexError` message text differs.** rython raises a real,
exit-1 `IndexError` (good) with the message `index out of bounds`; numpy
says `index 5 is out of bounds for axis 0 with size 2`. AGENTS.md is
explicit that message text counts. `cases/42_index_error.py`.

**D3 — numpy's `RuntimeWarning`s are not emitted.** Integer division by
zero yields the same values as numpy (`[0 0 0]` — `cases/45` PASSes on
stdout) but numpy also writes a `RuntimeWarning` to stderr.

### E. Clean conversion-time rejections (working as designed)

These are loud, precise and name the fix — the contract working:

- `np.sum(a, axis=…)` and friends
- `np.concatenate(arrays, 1)` — the positional `axis` numpy allows
  (`axis=1` as a keyword converts fine)
- `np.linspace(..., endpoint=False)`, `np.full(..., dtype=…)`
- anything outside the subset (`np.cumsum`, `np.median`, …), with the
  supported list in the message

One nit: those messages end with "See the numpy README for details", and
there is no numpy README in the repo.

## Speed results

Times are seconds per operation, best of 3, kernels timed in-process.
`scalar vs CPython` > 1 means rython is faster.

Geometric mean over each program's kernels:

| program | scalar | rayon | range (scalar) |
|---|---|---|---|
| elementwise | 0.23x | 0.10x | 0.04x – 1.18x |
| scalar_operand | 0.14x | 0.12x | 0.03x – 0.69x |
| reduce | 0.12x | 0.12x | 0.003x – 11.9x |
| sort | 0.41x | 0.42x | 0.18x – 5.3x |
| linalg | 0.36x | 0.35x | 0.02x – 3.6x |
| sim (mixed workload) | 0.23x | 0.09x | 0.16x – 0.28x |

Process startup, `import numpy` + one trivial op: **CPython 84.5 ms,
rython 1.5 ms (56x)**. For short-lived array programs this dominates
everything else.

### Where rython wins

Small arrays, where CPython pays ~2 µs of dispatch per numpy call:

| kernel | n | CPython | rython scalar | ratio |
|---|---|---|---|---|
| `std` | 1 000 | 1.08e-05 | 9.07e-07 | **11.9x** |
| `mean` | 1 000 | 3.57e-06 | 4.39e-07 | **8.1x** |
| `sum` | 1 000 | 2.53e-06 | 4.51e-07 | **5.6x** |
| `linalg.det` | 16×16 | 4.59e-06 | 1.28e-06 | **3.6x** |
| `linalg.inv` | 16×16 | 1.29e-05 | 4.31e-06 | **3.0x** |
| `linalg.solve` | 16×16 | 9.95e-06 | 5.86e-06 | **1.7x** |
| `sqrt` | 1 000 | 1.26e-06 | 1.07e-06 | **1.2x** |

### Where numpy wins, and why

| kernel | n | CPython | rython scalar | ratio |
|---|---|---|---|---|
| `sum` | 10 000 000 | 3.56e-03 | 1.10e-01 | 0.03x |
| `max` | 10 000 000 | 2.94e-03 | 1.50e-01 | 0.02x |
| `add` (array+array) | 10 000 000 | 4.92e-02 | 2.44e-01 | 0.20x |
| `add` (array+scalar) | 10 000 000 | 1.56e-02 | 2.53e-01 | 0.06x |
| `matmul` | 256×256 | 3.31e-04 | 1.71e-02 | 0.02x |
| `sort` | 2 000 000 | 1.75e-02 | 9.72e-02 | 0.18x |

Three root causes account for essentially all of it:

**S1 — every reduction copies the whole array first.** `reduce.rs`'s
`vals()` calls `NdArray::as_f64()`, which is `v.clone()` even for a
`Float64` array (`ndarray.rs:168`). `np.sum` is therefore an allocation
plus a full memory pass before the pairwise sum runs. It shows up as a
cliff exactly where the array stops fitting in cache: `sum` is 5.6x
*faster* than numpy at n=1 000, 0.68x at n=10 000, and 0.05x from
n=100 000 upward. A borrowed-slice fast path for the no-conversion case
would remove it.

**S2 — a scalar operand is materialized into a full-size array.**
`ufunc::binary` turns `np.add(a, 1.0)` into a 0-d array, then
`broadcast_to` builds *two* n-element vectors — a `Vec<usize>` of source
indices and the broadcast `Vec<f64>` — before the kernel runs
(`ufunc.rs:81`, `ufunc.rs:313`). Measured: array+scalar is 2.4x slower
than array+array at n=100 000 (7.18e-04 vs 3.12e-04) even though it does
strictly less work. numpy, which reads the scalar straight out of the
loop, is 2.7x *faster* on array+scalar than on array+array at that size.

**S3 — `broadcast_to` clones even when the shapes already match**
(`ufunc.rs:50`). An array+array op copies both inputs and allocates the
output: three full-size buffers for one add.

`matmul` at 256×256 is a separate story — numpy calls into BLAS, rython
runs a triple loop; 0.02x is the expected gap and not a defect.

### Backend comparison

| kernel | n | scalar | simd | rayon |
|---|---|---|---|---|
| `add` | 1 000 | 9.18e-07 | 9.54e-07 | 2.93e-05 |
| `add` | 100 000 | 3.02e-04 | 3.15e-04 | 3.59e-04 |
| `add` | 10 000 000 | 2.44e-01 | 2.41e-01 | 2.13e-01 |
| `sqrt` | 10 000 000 | 1.43e-01 | 1.42e-01 | 1.17e-01 |
| `oscillators` (1 000 elems × 2 000 steps) | — | 2.93e-02 | 2.92e-02 | 4.66e-01 |

- **`simd` is `scalar`**, within noise, everywhere — exactly as
  `simd.rs` documents.
- **`rayon` is not a win at any size measured here, and is a large loss
  on small arrays**: 32x slower on a 1 000-element `add`, and 16x slower
  on the oscillator simulation (1 000-element arrays in a tight 2 000-step
  loop), where the per-call thread-pool dispatch swamps the work. At
  10 000 000 elements it recovers ~12% — the kernels are memory-bandwidth
  bound (see S1–S3), so extra threads have little to add.
- This matters because **`auto` prefers `rayon`** whenever `numpy-rayon`
  is compiled in. `engine.rs:167` justifies the ranking with "multithreaded
  rayon beats the single-threaded simd/scalar loops at every benchmarked
  size" — that is not what this machine measures. `rayon_eng.rs` calls
  `par_iter()` with no minimum chunk length, so there is no size floor
  below which it stays sequential. Adding one (rayon's `with_min_len`)
  would make `auto` safe.

## Documentation and coverage observations

- **`examples/02-gpu-numpy/README.md` is stale in rython's favour.** It
  says the pi estimate differs in the last digits because "real numpy
  reduces with pairwise summation, rython's scalar engine accumulates
  sequentially", and prints two different numbers. `reduce.rs` now
  replicates `npy_pairwise_sum` exactly, and the shipped example is
  byte-identical today:
  ```
  CPython:  pi ~= 3.14147731828716
  rython:   pi ~= 3.14147731828716
  ```
  (`--help` output matches byte-for-byte too, and `--gpu` fails loudly as
  documented.)
- **`ndarray.rs` has no unit tests.** The numpy module has 22 tests, all
  in `mod.rs` and `rayon_eng.rs`, covering arange, division, promotion,
  predicates and reductions — genuinely good ones, pinned against numpy.
  The formatting code, which is where every divergence in section A lives,
  has none.
- **`docs/spec.md` §12 does not ledger any numpy divergence**, including
  the deliberate one (A10). §10's one-line mention ("`numpy` (a sizable
  subset …)") is all the spec says about it.

## Suggested priority

1. **A8 (negative slice bounds)** — silently returns the wrong data.
2. **C1 (`dtype=` generates invalid Rust)** — one-token fix, unblocks the
   entire dtype surface.
3. **A1–A6 (float formatting) and the `exp mode implies e` panic** — one
   area, one rewrite of `np_float_cell`/`float_layout` against numpy's
   `FloatingFormat`, and it converts most of the DIVERGE column to PASS.
   Needs the unit tests `ndarray.rs` currently lacks.
4. **B1 (`RYPY_NUMPY_BACKEND`)** — either implement it or delete the three
   places that promise it.
5. **S1/S2 (reduction copy, scalar broadcast)** — the two changes that
   would move array throughput most, and neither affects semantics.
6. **A9 (`std(a, 1)` positional)** — reject the positional form loudly
   rather than reinterpreting it.
7. **C4 (ndarray-returning helpers)** — the ergonomic blocker for real
   numpy code.
