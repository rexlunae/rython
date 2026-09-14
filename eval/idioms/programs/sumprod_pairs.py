"""math.sumprod dot product, type-preserving + exact (#369).

sumprod is element-wise multiply + sum; the RESULT type follows the input:
all-int stays int, any float becomes float, an empty pair is the int
additive identity. The float path is EXACT (extended-precision): a
cancellation keeps the surviving term. Each case prints its value so a
wrong sum or type shows up as a diff.
"""

import math


def main() -> None:
    print(math.sumprod([10, 20, 30], [1, 2, 3]))
    print(math.sumprod([1, 2], [3, 4]))
    print(math.sumprod([], []))
    print(math.sumprod([1.5, 2.5], [3.5, 4.5]))
    # Mixed int/float pair: the int list coerces to float.
    print(math.sumprod([-1], [1.]))
    # Exactness under cancellation: the 1.0 term survives.
    print(repr(math.sumprod([1e16, 1.0, -1e16], [1.0, 1.0, 1.0])))
    # Unequal lengths.
    try:
        math.sumprod([1, 2], [1])
    except ValueError as e:
        print(e)


if __name__ == "__main__":
    main()
