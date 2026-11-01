# Automatic Python integer promotion: implementation design

Status: the first scalar slice is implemented. Oversized integer literal
operands automatically lower to `stdpython::PyInteger`, with no user flag.
Ordinary integer-only expressions and annotated `int` slots still use `i64`.
This is the start of the migration, not a claim that all Python integers are
unbounded throughout the compiler.

## Supported scalar slice

Literal values are extracted as exact decimal integers, including hexadecimal
source literals and the positive operand in `-9223372036854775808`. Local
assignments retain the unbounded representation after a smaller value is
assigned. Adding an unbounded value to an unannotated integer local widens that
local too. Alias assignments preserve immutable integer value semantics.

The supported operators are `+`, `-`, `*`, `//`, `%`, unary `+`, unary `-`,
`~` and `not`. Binary arithmetic accepts an unbounded integer and another
unbounded or ordinary integer. Results use small storage when possible and
grow again on overflow. `//` and `%` follow Python's floor/divisor-sign rules
and raise catchable `ZeroDivisionError` (the existing message-version choice
is recorded in spec §12).

Single numeric equality and ordering comparisons against integers, floats
and bools preserve the integer's exact value, including comparisons above
2**53 and with NaN/infinities. `print`, `str`, `repr`, `hash`, `bool`, `int`,
`abs` and `float` accept the scalar. `float` rounds to nearest with ties to
even and raises `OverflowError` when the result would overflow. Unformatted
f-string interpolations use the exact decimal representation. Decimal
conversion enforces CPython's default 4300-digit limit and raises a catchable
`ValueError`; print evaluates all arguments first and preserves partial
output if a later conversion fails. This slice requires that default limit
in the conversion interpreter and rejects `sys` limit-query/change APIs.

Unsupported operations and boundaries are conversion errors: true division,
powers, shifts, binary bit operations, mixed float arithmetic, containers,
subscripts, attributes/methods, formatted interpolation specifications,
operand-selecting boolean/conditional expressions, function arguments and
returns, annotated/class storage, alloc-only module-level initialization,
print keywords/non-scalar companions, and kernel/PyO3/driver targets. A string
produced from the integer can cross an ordinary function boundary. Local scalar
operations work in the alloc-only runtime as well as std.

Supported scalar results can cross boundaries appropriate to their types:
a string result can enter a container, act as a dictionary key or subscript
receiver, or be iterated. These checks apply to the value entering each
boundary, rather than intermediate bigint operands.
PyO3 and driver target checks run when a literal is emitted, so inactive
branches and shadowed definitions do not reject a conversion.

## Representation and remaining integration

Python integers are unbounded. Promoting only oversized literals is not
sufficient: an ordinary input or a sequence of small operations can overflow
later. The representation is a canonical small `i64` or a large
`num_bigint::BigInt`, chosen by value, with checked small arithmetic and
promotion on overflow. Values that shrink back into range use the small form.
Both representations are internal to one type, so assignment, function
calls and containers do not change type when a value grows.

The runtime type belongs to the alloc tier; bigint storage requires allocation.
It must remain usable without `std`. The strictly core tier already fails
loudly. Kernel and FFI paths must be audited for their own allocation and ABI
constraints before the converter emits this type there.

Integration must cover all integer-producing and integer-consuming paths:

1. Parse literals without passing through `f64`; preserve their exact decimal
   value and emit a runtime integer. Implement the full set of operators,
   including Python floor division, modulo, shifts, powers and zero errors.
2. Replace typed integer storage and boxed integer values together. Update
   ownership, globals, lists, dictionaries, functions and annotations; a
   partial replacement would lose values at a boundary or fail in rustc.
3. Implement Python-compatible comparison and hashing across integers, bools,
   floats and complex numbers, including exact comparisons above 2**53. Do not
   use a lossy float conversion to compare or hash a large integer.
4. Audit indexes, lengths, ranges, repetition, OS calls, NumPy and FFI. Each
   fixed-width boundary needs Python's own overflow or clipping behavior; an
   unchecked cast is not acceptable. Where that behavior is not implemented,
   refuse conversion with the construct and rewrite, or raise the correct
   exception at the point of divergence.
5. Verify CPython transcripts for literals, arithmetic crossing both bounds,
   negative division and modulo, shifts, mixed numeric operations, containers,
   indexing, errors and values passing through functions and boxed storage.
   Test both debug and release, plus the alloc tier and generated crates.

No existing `i64` overflow ledger entry is removed until all affected
generated paths are covered. In particular, `9223372036854775807 + 1` still
uses the old bounded path when neither operand has been introduced through
an unbounded literal or local. A JSON integer, `int(text)`, an integer-returning
stdlib call and boxed `PyValue::Int` likewise remain migration work. These
producers must change together with their storage and consumers before the
compiler can promise automatic promotion for every Python integer.
