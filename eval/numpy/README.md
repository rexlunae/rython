# numpy evaluation harness

Two harnesses that compare rython's numpy subset against CPython + real
numpy, and measure the speed difference across rython's execution
backends.

## What is here

| Path | What it is |
|---|---|
| `cases/` | Self-contained Python programs, each printing deterministic output |
| `rython_only/` | Programs using rython-only surface (`np.set_backend`), not diffable against CPython |
| `bench/` | Benchmark programs that time their own kernels with `time.perf_counter` |
| `run_eval.py` | Correctness harness: CPython vs the converted binary, per backend |
| `run_bench.py` | Speed harness: same program, CPython vs each backend |
| `summarize.py` | Renders the two result JSONs into the tables in `REPORT.md` |
| `REPORT.md` | The evaluation write-up |

## Running it

```bash
cargo build --release -p rypip           # the converter the harnesses drive
python3 -m pip install numpy             # the reference implementation

python3 eval/numpy/run_eval.py  --workdir /tmp/npeval
python3 eval/numpy/run_bench.py --workdir /tmp/npbench --repeat 3
python3 eval/numpy/summarize.py --results /tmp/npeval/results.json \
                                --bench   /tmp/npbench/bench.json
```

`run_eval.py` classifies every case as PASS (byte-identical stdout and
exit code), DIVERGE (ran, different output), CONVERT_FAIL (loud conversion
error — rython's documented contract for anything outside the subset),
BUILD_FAIL (the generated Rust does not compile) or RUN_FAIL.

## Selecting a backend

The harnesses pin the engine by rewriting the program to call
`np.set_backend("<name>")` as `main()`'s first statement, and by adding the
matching stdpython features to the generated crate's `Cargo.toml`.

`RYPY_NUMPY_BACKEND` also works (it did not when this harness was written
— see finding B1 in `REPORT.md` and issue #198); the harness keeps using
`np.set_backend` because it pins the engine in the program itself, which
is what the generated binaries do.

## Notes

- The generated crates depend on stdpython by path, so the harnesses edit
  the generated `Cargo.toml` to add `numpy-rayon` / `numpy-simd`;
  `rypip convert` has no flag for that.
- One shared `CARGO_TARGET_DIR` per feature set keeps stdpython compiled
  once instead of once per case.
- The benchmark programs time their own kernels, so the reported numbers
  exclude interpreter and process startup; startup is measured separately.
