"""math.ceil/floor/trunc over an integer are EXACT (#369).

CPython's integer rounding returns the argument unchanged, so
`math.floor(2**53 + 1)` is `9007199254740993` — never a lossy float
round-trip. rython routes an i64 argument to the runtime `<fn>_i64`
overload (the argument, unchanged), while a float / FloatLike argument
still goes through f64.
"""

import math
from math import floor, trunc


def main() -> None:
    m = 2 ** 53 + 1
    print(math.floor(m))
    print(math.ceil(m))
    print(math.trunc(m))
    print(floor(m))
    print(trunc(m))
    print(math.floor(3.7))
    print(math.ceil(2.1))
    print(math.trunc(-1.9))


if __name__ == "__main__":
    main()
