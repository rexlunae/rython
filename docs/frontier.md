# The frontier: where the corpus still fails to build (issue #335)

This document records the measured state of the rustc-error frontier on the
pinned package corpus, identifies the one structural blocker that dominates
it, and states the candidate designs with their tradeoffs. It is a status
record and a scoping document, not a plan of record: nothing here is
committed work.

Read [`goals-and-design.md`](goals-and-design.md) for the prime directive
and the feature-evaluation checklist every proposal below is measured
against, and §12 of [`spec.md`](spec.md) for the deviation ledger.

## What is measured, and how

The frontier metric is the **rustc error count on pinned real packages**,
produced by `eval/sweep/` (see the "corpus sweep" section of
[`AGENTS.md`](../AGENTS.md)). `run_sweep.py` converts each package in
`eval/sweep/packages.json`, builds the generated crate, and writes a JSON
error histogram; `rank_causes.py` ranks the shared compiler causes across
packages; `summarize.py` diffs two records.

**Baseline used throughout this document:** `run-main-base.json` (schema 2,
`measurement_complete: true`), measured at commit `a055a18` on
`x86_64-unknown-linux-gnu`, rustc 1.98.1, CPython 3.14.4. It is *not*
committed to the repository — it lived in the sweep workdir of the round
that produced it. Treat the filename as a pointer to that measurement, not
as a tracked artifact.

> **Host dependence.** Counts are not portable across machines (issue #442).
> The same commit measured on `aarch64-apple-darwin` gave 3006 diagnostics
> against 2572 on `x86_64-unknown-linux-gnu`. Any delta must compare two
> records from the *same* host; `summarize.py` enforces this with a
> `NOT COMPARABLE` banner. Every number below is from the single host named
> above.

### The baseline, per package

| Package | Coded errors | Uncoded (build-failed) | Diagnostics | Status |
|---|---:|---:|---:|---|
| urllib3 2.0.7 | 595 | 54 | 649 | build-failed |
| certifi | 0 | 0 | 0 | **built** |
| idna | 0 | 0 | 0 | **built** |
| charset_normalizer | 38 | 3 | 41 | build-failed |
| requests 2.32.3 | 1716 | 151 | 1867 | build-failed |
| **total** | **2349** | **208** | **2557** | |

"Uncoded" diagnostics are rython's own `rustc: …` refusal messages emitted as
`compile_error!` into the generated crate — the loud edges, not
miscompilations. They are counted separately because they are a different
class of defect (the converter declined to model a construct) from an
E-code (the converter emitted Rust that does not typecheck).

### Corpus histogram (top causes)

Output of `python3 eval/sweep/rank_causes.py <record>`:

```
E0308         886   37.7%
E0277         469   20.0%
E0609         397   16.9%
E0599         369   15.7%
E0061          77    3.3%
E0382          25    1.1%
E0618          18    0.8%
E0605          16    0.7%
E0615          15    0.6%
E0369          14    0.6%
```

Two packages build with **zero** errors (certifi, idna), which is the
honest counterweight to the headline: the toolchain does produce clean
crates for real packages. requests alone is 1716 of the 2349 coded errors
(73%).

### Top E0308 pairs

From each package's `e0308_pairs`:

| Package | Pair | Count |
|---|---|---:|
| requests | `String \| PyValue` | 38 |
| requests | `PyValue \| String` | 34 |
| requests | `&String \| &&str` | 24 |
| requests | `String \| &str` | 22 |
| requests | `() \| PyValue` | 20 |
| requests | `PyValue \| Option<_>` | 17 |
| urllib3 | `Socket \| PyValue` | 12 |
| urllib3 | `Vec<u8> \| PyValue` | 11 |
| urllib3 | `PyValue \| Option<ProxyConfig>` | 8 |
| urllib3 | `String \| PyValue` | 7 |

The family is consistent: **a `PyValue` where a typed slot expects a
concrete type, or the reverse.** The issue text (#335) names this family
correctly; only its counts are stale (it records 734 urllib3 errors against
the current 595).

## The structural blocker: the `PyValue` attribute boundary

`rank_causes.py` splits the two "member lookup" codes by receiver type. The
result is the single most concentrated cause on the corpus:

```
== the PyValue attribute boundary: 409 sites (17.4% of 2349) ==
   279  E0609 no field on PyValue
   130  E0599 no method on PyValue
```

The 409 is **E0599 + E0609 only**: `rank_causes.py:246` computes
`boundary = sum(methods.values()) + sum(fields.values())`, and only the
`no field`/`no method` shapes increment those counters. It is 70% of E0609
(279 of 397).

E0277 is a **separate** accounting: `rank_causes.py:194-197` counts a
trait-bound error into `code_pyvalue` when the bound names `PyValue`, which
gives the "301 of 469 (64%)" line the ranking prints under "how much of
each code is the PyValue boundary". That 301 is **not** a subset of the
409 — the two counts come from different arms of the same `match` and
overlap only in cause, not in membership. Do not add them.

The mechanism behind both: a value that inference files into a `PyValue`
slot keeps its *runtime* type but loses its *static* one. `PyValue` is a
closed enum without an instance variant and without an attribute accessor,
so `x.attr` / `x.method()` on such a value has nothing to resolve to, and
the generated Rust does not compile at that line. The adjacent E0277 family
is the same loss seen from the trait side: a `PyValue` where a bound
expects a concrete type.

### The variants that exist today

`crates/stdpython/src/lib.rs:3019-3050` — twelve variants, none of them an
instance:

`Int`, `Float`, `Bool`, `Str`, `Bytes`, `Tuple`, `Dict`, `OrderedDict`,
`Complex`, `Range`, `Function`, `None_`.

`Function` is worth noting for issue #334: callables *are* representable
boxed (`PyValue::Function(PyCallable<BoxedArgs, PyValue>)`, called through
`call_boxed`). Instances are not.

### The top missing members

```
methods (25 distinct):        fields (41 distinct):
  53  .close()                  24  .domain
   9  .copy()                   19  .is_closed
   8  .decompressobj()          18  .value
   6  .read()                   18  .timeout
   6  .is_retry()               16  .name
   6  .matches()                15  .path
   6  .readline()               12  .headers
```

### `.close()` is a trap, and it is not taken

`.close()` ranks first by a wide margin (53 of 130 method sites) and looks
like a free slice. It is not, and this is the most important negative
result on this frontier.

`r.close()` in `requests/auth.py:268` is a `Response`. It reaches a
`PyValue` only because inference never pinned the receiver — there is no
`Response` variant to close. Since `PyValue` has no closeable variant, the
only two implementations available are:

1. a **silent no-op** — leaks the connection, diverging from CPython; or
2. an **`AttributeError`** — but CPython's `Response.close()` succeeds, so
   this diverges too.

Both are the "silently different" outcome the prime directive forbids (P1 in
[`goals-and-design.md`](goals-and-design.md)). The durable fix is the
structural one; the trait method is not it. This reasoning is recorded on
issue #137 (round 102 comment) and is why a count-ranked slice is not a
valid slice.

## Candidate designs

### A. An instance variant on `PyValue`

Add a variant carrying a reference to a class instance, e.g.
`PyValue::Instance(PyValueInstance)`, where the payload holds the runtime
class identity plus a `&dyn`-style accessor surface for fields and methods.

**For.** Fixes the whole family in one place: every one of the 409 sites
becomes resolvable without teaching inference to pin receivers. It is the
only design that also serves *heterogeneous* containers of instances and
any shape where the static type genuinely is not knowable.

**Against.** This is the boxed-dynamism default that P9 of
[`goals-and-design.md`](goals-and-design.md) tells the project to reach for
**last**, and only when the dynamic surface is genuinely open-ended. An
instance variant makes attribute access a runtime lookup for every boxed
value, which:

- costs every program that never needs it (P5 rejects exactly this tradeoff
  for aliasing, on the same ground);
- makes the failure mode *silent* rather than loud: a typo'd attribute
  becomes a runtime `AttributeError` instead of a compile error, weakening
  the static guarantee that is goal 1's whole value proposition;
- puts the generated crate's readability at risk (goal 1) by routing an
  ordinary field read through a dynamic accessor.

The §12 ledger row already records the *container* surface as modeled by
member (dict `get`/`keys`/`items`/`pop`/…, and slicing), with everything
else a `MODEL LIMIT` that raises CPython's `TypeError` naming the real type.
An instance variant would have to extend that discipline, not replace it.

### B. Receiver pinning in inference

Teach the inference layer to pin the receiver's class statically, so
`r.close()` resolves to a typed `Response` and never becomes a `PyValue`.

**For.** Keeps the static lowering (P9), keeps attribute access a compile
error when the attribute does not exist, keeps the generated Rust readable,
and costs nothing at runtime. It is also *already the established direction*
— §12's `issue #335` rows describe call-site typing for unannotated
parameters and method-slot typing that resolves a parameter to a class from
every call site in the crate, to a fixpoint. Pinning a receiver is the same
machinery applied one step earlier.

**Against.** It cannot be complete. Any genuinely dynamic origin (a value
read from an untyped dict, a heterogeneous list, an external module's
return) still has no static class, so the site survives as a `PyValue` and
still fails to build. It may also need per-site evidence that in practice
does not converge (the method-slot typing already has documented escape
hatches for spreads, unbound calls, decorated methods and dunders).

### C. A narrow, instance-shaped union instead of a general instance variant

Thread *only* the specific instance types a crate emits into the boxed
type as a generated sum type — the approach #319 took for hierarchy roots
(`Any<Root>` with variants for exactly the classes the crate emits).

**For.** It is P9-compliant in shape: the dynamic part is a *narrow, typed
union* rather than an untypeable bag, and the variant set is closed at
conversion time. It is precedented in this codebase.

**Against.** Variant count grows with the crate (requests emits many
classes), so the enum is large; and it only helps where a value's possible
classes are enumerable at conversion time — which is a strictly smaller set
than design A covers, possibly not overlapping much with the actual 409
sites.

## Recommendation

**Pursue B first, measured per slice, and treat A as the fallback for
exactly the sites B provably cannot reach.**

Rationale, against the evaluation checklist in
[`goals-and-design.md`](goals-and-design.md):

- B keeps every guarantee the project exists for (static lowering, readable
  output, a compile error for a nonexistent attribute) and is the only
  candidate that *improves* the loudness story rather than diluting it.
- B is already the direction the §12 ledger documents, so it composes with
  work in flight instead of opening a second front.
- A's cost is paid by every program for a benefit a minority needs — the
  same tradeoff P5 explicitly rejected for aliasing.

This recommendation is a scoping judgment, **not a measured result**. The
honest open question is the split: how many of the 409 sites are reachable
by receiver pinning and how many are irreducibly dynamic. That split is not
known, and no round has measured it. **Unverified.**

A defensible first slice would therefore be: instrument the boxed-member
sites to classify the receiver's origin (pinned-by-inference vs. dynamic
origin), land the number as a recorded measurement, and only then choose
between B and A on evidence. Anything else is choosing a design by
argument.

## What would need code changes (not done here)

Per this document's scope, nothing below was attempted:

1. The 409 boxed-member sites and their origins — unmeasured split (above).
2. `PyValue`'s variant set (`crates/stdpython/src/lib.rs:3019`).
3. The `§12` ledger row's `477 sites at aca31b3` figure is **stale**; the
   current measurement is **409** at `a055a18`. The row needs updating when
   the next structural slice lands.
4. Issue #335's own body records 734 urllib3 errors and
   `eval/sweep/results/run-current3.json` as its reference. Both are stale:
   the current count is **595 coded + 54 uncoded**, and no
   `run-current3.json` exists in the results directory.

## Status of issue #335

**Open, and correctly so.** Its *diagnosis* (boxed-`PyValue` coercions
dominate) is confirmed by the current measurement; its *numbers* and its
*reference record* are stale. It should not be closed: the family is the
largest single cause on the corpus, and the structural fix has not landed.
What this document adds is a verified current baseline and a named design
choice to make before the next slice.
