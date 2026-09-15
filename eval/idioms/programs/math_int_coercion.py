"""math.* with a COMPUTED int argument coerces to float (#369).

CPython's math functions coerce an int argument to f64 (`math.sqrt(2**52)`
is 67108864.0). Previously a COMPUTED i64 (`m = 2 ** 53`) rendered as the
bare `m` into the runtime's `T: Into<f64>` bound and failed to build — std
has no `From<i64> for f64` — while a small LITERAL (27) was coerced by type
inference. Each line below passes a computed (or inferred) int.
"""

import math


def main() -> None:
    m = 2 ** 53
    n = 2 ** 12
    print(math.ulp(m))
    print(math.cbrt(n))
    print(math.sqrt(m))
    print(math.hypot(m, 3.0))
    print(math.nextafter(m, 5.0))
    print(math.floor(2 ** 20))
    print(math.fabs(0 - 5))
    print(math.log10(10000))


if __name__ == "__main__":
    main()
