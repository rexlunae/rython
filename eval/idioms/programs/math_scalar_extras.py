"""math.isqrt/cbrt/hypot/nextafter scalar extras (#369).

A batch of test_math-missing math functions available across the oracle's
Python versions: the exact integer square root, the cube root (exact on a
perfect cube), the Euclidean norm, and the next representable float.
(math.fma is Python 3.13+ — absent from the 3.11 oracle — so it lives only
in the runtime/codegen pins, not this transcript.) Each prints its value,
and isqrt's negative-n ValueError prints too.
"""

import math


def main() -> None:
    print(math.isqrt(16))
    print(math.isqrt(15))
    print(math.isqrt(0))
    print(math.cbrt(27.0))
    print(math.cbrt(-8.0))
    print(math.hypot(3.0, 4.0))
    print(math.hypot(0.0, 0.0))
    print(repr(math.nextafter(1.0, 2.0)))
    try:
        math.isqrt(-1)
    except ValueError as e:
        print(e)


if __name__ == "__main__":
    main()
