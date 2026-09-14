"""math.sumprod dot product, type-preserving (#369).

sumprod is element-wise multiply + sum; the RESULT type follows the input:
all-int stays int, any float becomes float, an empty pair is the additive
identity. Each case prints its value (and a repr where the type matters)
so a wrong type or sum shows up as a diff.
"""

import math


def main() -> None:
    print(math.sumprod([10, 20, 30], [1, 2, 3]))
    print(math.sumprod([1, 2], [3, 4]))
    print(math.sumprod([], []))
    print(math.sumprod([1.5, 2.5], [3.5, 4.5]))
    # Mixed int/float pair: the int list coerces to float.
    print(math.sumprod([-1], [1.]))
    print(repr(math.sumprod([2.5, 1], [2, 3])))


if __name__ == "__main__":
    main()
