"""math.comb and math.perm integer combinatorics (#369).

The binomial coefficient and the permutations count, with CPython's
edges: k > n or k < 0 is 0 (for comb), a 1-arg perm defaults k to n, and a
negative n is ValueError. Each prints its value.
"""

import math


def main() -> None:
    print(math.comb(5, 2))
    print(math.comb(10, 3))
    print(math.comb(5, 0))
    print(math.comb(0, 0))
    print(math.comb(5, 8))  # k > n -> 0
    print(math.perm(5, 2))
    print(math.perm(5))  # k defaults to n
    print(math.perm(0, 0))
    try:
        math.comb(-1, 2)
    except ValueError as e:
        print(e)


if __name__ == "__main__":
    main()
