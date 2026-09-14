"""math.comb and math.perm integer combinatorics (#369).

The binomial coefficient and the permutations count, with CPython's exact
edges: k > n is 0, a NEGATIVE k or n is ValueError, a 1-arg perm defaults
k to n, and a large coefficient that fits i64 but would overflow an
uncancelled intermediate (comb(62, 28)) is computed exactly. Each prints
its value.
"""

import math


def main() -> None:
    print(math.comb(5, 2))
    print(math.comb(10, 3))
    print(math.comb(5, 0))
    print(math.comb(0, 0))
    print(math.comb(5, 8))  # k > n -> 0
    print(math.comb(62, 28))  # large but fits i64
    print(math.perm(5, 2))
    print(math.perm(5))  # k defaults to n
    print(math.perm(0, 0))
    try:
        math.comb(-1, 2)
    except ValueError as e:
        print(e)
    try:
        math.comb(5, -1)
    except ValueError as e:
        print(e)


if __name__ == "__main__":
    main()
