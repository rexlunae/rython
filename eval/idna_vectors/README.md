# idna's IdnaTestV2 vectors under rython

`run_vectors.py` runs idna 3.10's UTS-46 conformance vectors
(`tests/IdnaTestV2.txt`, 6225 vector lines) through the converted idna
package and compares the result with CPython's, byte for byte. This is
issue #334's vector-file path: behavioral coverage of a real library,
beyond the converted code merely building.

```sh
cargo build -p python-ast -p rypip
python3 eval/idna_vectors/run_vectors.py              # oracle: python3
python3 eval/idna_vectors/run_vectors.py --python python3.14
```

The harness (`vectors_template.py`) is an in-subset port of the upstream
test module's `runTest` body. Each vector is checked against `decode()`,
`encode()` and transitional `encode()`, and the upstream `_SKIP_TESTS`
list is spliced in from the sdist. The upstream module drives the
vectors through unittest's `load_tests` protocol; the port does not.

Every vector passes, and 485 are skipped by the upstream rule: an
`Unknown ...` error means the Unicode database is older than the
vector. The output is identical to CPython 3.11's and to CPython 3.14's.
CPython 3.14's `unidata_version` is 16.0.0, as rython's `unicodedata` is
(`docs/spec.md` §12.3).

It needs network access for the first fetch (checked against the sdist's
sha256), so it is absent from CI, like the sweep.
