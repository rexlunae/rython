"""math.fsum compensated summation (#369).

The promise of math.fsum is error-free summation: every rounding error is
preserved in a partial and folded back in, so error-free compensation
applies even across multiple partials (half-even), a small magnitude
survives a large one, and special values (inf/nan) are handled exactly.
Each case prints its value so a naive sum would show up as a diff.
"""

import math


def main() -> None:
    print(repr(math.fsum([0.1, 0.2, 0.3])))
    print(repr(math.fsum([])))
    # A large term must not swallow a small one.
    print(repr(math.fsum([1e100, -1e100, 1e-100])))
    print(repr(math.fsum([1e100, 1e-100, -1e100, 1e-100])))
    # Half-even rounding across partials (commutative).
    print(repr(math.fsum([1e16, 1.0, 1e-16])))
    print(repr(math.fsum([1e-16, 1.0, 1e16])))
    # An infinity short-circuits to that infinity.
    print(repr(math.fsum([1.0, math.inf])))
    # An intermediate overflow is an OverflowError, not inf.
    try:
        math.fsum([1.0, 1e308, 1e308])
    except OverflowError:
        print("overflow")
    # A NaN anywhere dominates an infinity.
    print(math.isnan(math.fsum([math.inf, math.nan])))


if __name__ == "__main__":
    main()
