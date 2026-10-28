# Evaluating a JIT for rython (issue #166)

This document evaluates issue #166 ("feat: A JIT compiler") against the
project's stated contract. It is an evaluation, not a design commitment:
its conclusion is a recommendation to **not** build a JIT now, and to split
the issue's two asks — a usable `python program.py` command-line experience,
and an actual just-in-time compiler — which have very different costs.

Contract: [`goals-and-design.md`](goals-and-design.md) (goals, non-goals,
and the "How to evaluate a proposed feature" checklist);
[`AGENTS.md`](../AGENTS.md) (the prime directive).

## What the issue asks for

> Add a Just-In-Time compiler to the suite of tools. This can be used as the
> basis of eval and other implementations in stdpython, and also allow users
> to run Python compiled from the command line the same way you would with
> CPython.

Two separate things:

1. **A command-line UX** — run a `.py` file like `python file.py`.
2. **A JIT** — compile and execute Python without a separate ahead-of-time
   compile step, with `eval`/`exec` as a motivation.

The issue's owner comment (2026-08-28) already reached this split and
proposed `rypip run` for (1). That proposal **has since landed**, which
changes the state of this issue materially. See "What has changed" below.

## The contract problem

A JIT as usually meant — interpret or compile Python at runtime, bypassing
codegen — is in direct tension with how this project defines itself.

### Non-goals say no, explicitly

[`goals-and-design.md`](goals-and-design.md), under **Non-goals**:

> **Full Python compatibility.** Rython will never run arbitrary PyPI code.
> `eval`/`exec`, metaclasses, monkey-patching, dynamic imports, `globals()`
> mutation — the reflective machinery that makes Python
> Python-the-dynamic-language — is out of scope permanently, because it
> requires an interpreter at runtime, and "no interpreter at runtime" is the
> point.

> **Being a faster Python runtime.** Rython is not a JIT, not a tracing
> optimizer, and not a drop-in `python3` replacement. Projects that need all
> of Python, faster, should look at PyPy or CPython's own JIT work.

This is stated as a **non-goal**, not a postponed feature: "each one has been
deliberately rejected, not merely postponed."

And "What Rython is":

> There is no interpreter, no bytecode, no GIL, and no Python installation
> at runtime.

A JIT that exists to make `eval`/`exec` work is, in this project's terms,
an interpreter at runtime. That is not a scoping detail to be argued
around; it is the property the design is built on, and it is load-bearing
for goals 1 (ownable port — an interpreter in the output makes the output
not-ownable) and 4 (reach — an interpreter cannot run on `thumbv7m` or in a
kernel module).

### The verification-matrix argument

Every semantic rython pins is verified through the codegen path:
`crates/rypip/tests/convert_tests.rs` diffs generated-binary output against
transcripts captured from real `python3` runs, `crates/python-ast/tests/
codegen_semantics.rs` pins generated-Rust shape, and §12 of
[`spec.md`](spec.md) is the ledger of known differences.

A second execution model means a second conformance surface for the same
subset. Either it is verified to the same standard — doubling the matrix
below §1.2's conformance requirement for the one property the project
refuses to compromise — or it is not, and the project has two code paths
with different fidelity, which is exactly the "silently different" failure
P1 forbids. There is no third option that is cheaper.

### Is there *any* P9-compliant JIT?

The most defensible reading, consistent with P9 (static core, dynamic seam
threaded as a value), would be a **narrow** form: ahead-of-time compile
everything, and add a runtime *specialization* step to already-compiled,
statically-typed code — e.g. inline caching over a boxed `PyValue`
dispatch. That does not add an interpreter (the program is still compiled),
and it does not add a second semantic model (it optimizes code the same
codegen produced). It would be measured by performance, not semantics.

But that is not what issue #166 asks for (`eval`, and a CPython-like
command line), and the project has a stated non-goal against chasing
performance ("Performance as a consequence, not a headline"). It is
mentioned here only so the boundary is unambiguous: **optimizing the
generated code is not ruled out; executing Python without compiling it
is.**

## What has changed since the issue was filed

The command-line ask is **already satisfied**:

`rypip run` exists — `target/debug/rypip --help` lists:

```
run  Convert, build and run a Python program, like `python program.py`:
     the arguments after the package path (or after `--`) are the program's,
     its output is yours, and its exit status is rypip's.
     Unix only for now (sys.argv[0] is set through the exec)
```

Verified: `rypip run test_allpass.py` on a `unittest` module exits 0,
matching `python3`. The command composes the existing convert→build→exec
pipeline, so it adds **zero new semantics** — it cannot diverge, because it
is the same pipeline every transcript test already covers.

The issue text's remaining motivation — "the basis of `eval` and other
implementations in stdpython" — is the part that a JIT would be *for*, and
that part is exactly what the non-goals reject.

There is **no JIT or bytecode groundwork in the tree.** A search of
`crates/` for `jit`/`bytecode`/`interpreter` returns only conversion-time
uses of the Python interpreter: `crates/python-ast/src/exception_tree.rs`
dumps the builtin exception hierarchy from the live interpreter at
conversion time, and `crates/python-ast/src/rust_bind.rs` notes the output
crate has "no runtime, no interpreter, no PyO3". Nothing executes Python at
runtime. **Unverified:** whether any work exists on branches outside
`origin/main`; only `main` was searched.

## Verdict against the feature checklist

Using the six questions in "How to evaluate a proposed feature":

1. **Can its CPython behavior be reproduced exactly?** A general JIT, no —
   not without becoming an interpreter, which the non-goals reject. This
   alone stops the feature under the document's own closing rule: "If the
   answer to (1) is no on both counts, the feature is not added — it is
   documented as a boundary instead."
2. **Does the generated Rust remain reviewable?** Not applicable to a JIT
   (it emits no Rust), and it would displace the artifact goal 1 exists
   for.
3. **Which runtime tier?** A JIT requires the `std` tier (it needs
   allocation and code pages). It cannot exist on `alloc`, on
   `thumbv7m-none-eabi`, or in a kernel module — so it could never be the
   only execution model, which means (4) below.
4. **Does it keep working under the kernel and `no_std` targets?** No. A
   JIT is structurally unavailable there, so rython would permanently carry
   two execution models with different capabilities. That is a large
   ongoing cost for a feature serving no stated goal.
5. **Is there an end-to-end test diffing against `python3`?** It would need
   its own entire mirror of the suite (§13 of [`spec.md`](spec.md)).
6. **Does it keep a static core and thread the dynamic part as a typed
   seam (P9)?** A JIT does the opposite: it replaces the static lowering
   with runtime compilation.

**Four of six are disqualifying, and the first is the one the document
treats as decisive.**

## Recommendation

1. **Do not build a JIT.** The non-goals reject it explicitly, the
   verification cost is a doubled conformance matrix, and it cannot serve
   the embedded/kernel targets that justify the subset's existence.

2. **Close issue #166's command-line half as done.** `rypip run` delivers
   "run Python from the command line the same way you would with CPython"
   with no new semantics. The issue should be edited to record this, or
   split so the remaining part is only the JIT.

3. **Keep the issue open, narrowly, as a boundary statement** if the
   maintainer wants a place to point at — with the non-goal quoted, so the
   next round does not re-litigate it from scratch. The owner comment of
   2026-08-28 already reaches this conclusion; this document adds the
   verification-matrix argument and the current `rypip run` evidence.

4. **If a use case ever appears**, the minimum honest milestone is not "a
   JIT" but this: *one construct, executed two ways, proven byte-identical
   on the pinned corpus.* Concretely, a first milestone must prove:

   - **Semantics identical to the AOT path**, measured by running the
     entire idiom corpus (`eval/idioms/`, 59 programs) and the corpus sweep
     through **both** paths and diffing stdout/stderr/exit status with
     zero differences — not "nearly identical", not a tolerance window;
   - the measurement is **checked in** and **gated** the way
     `eval/idioms/baseline.json` is gated (`--check-baseline` fails on any
     regression) so a divergence cannot be introduced silently;
   - it is **`std`-tier only**, documented as such, and the `no_std` /
     kernel targets still reject the construct at conversion time;
   - it is **opt-in**, never the default lowering.

   That milestone is the *precondition* for discussing a JIT at all, and it
   is already a substantial project. Establishing it is the honest test of
   whether the project wants a JIT; building the JIT first and discovering
   the divergence later is the failure mode P1 exists to prevent.

## Explicitly out of scope

- `eval`/`exec` of arbitrary strings (non-goal: interpreter at runtime).
- Metaclasses, monkey-patching, dynamic imports, `globals()` mutation
  (same non-goal).
- REPL support.
- Any `no_std` / kernel / embedded execution model other than
  ahead-of-time compilation.
- Replacing codegen as the default path, or making a JIT the artifact the
  project ships.

## Status of issue #166

**Open, and it should probably not be closed by a code change — but its
command-line half is already delivered and the issue should say so.**

- The *ask* "run Python from the command line like CPython" is **done**
  (`rypip run`, verified).
- The *ask* "a JIT … as the basis of `eval`" is **rejected by the project's
  own non-goals**, not deferred by lack of effort.

It cannot honestly be closed as "already addressed" in full: no JIT exists,
and the non-goal is a design decision a maintainer may revisit. It also
cannot be closed as "not planned" by an agent — that is the maintainer's
call. This document exists so that call can be made once, on the record,
with the evidence in hand.
