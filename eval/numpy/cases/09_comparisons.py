"""Comparison ufuncs produce bool arrays; bitwise/logical on them."""
import numpy as np


def main() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    print(np.greater(a, 2.0))
    print(np.greater_equal(a, 2.0))
    print(np.less(a, 2.0))
    print(np.less_equal(a, 2.0))
    print(np.equal(a, 2.0))
    print(np.not_equal(a, 2.0))
    m = np.greater(a, 1.0)
    n = np.less(a, 4.0)
    print(np.bitwise_and(m, n))
    print(np.bitwise_or(m, n))
    print(np.bitwise_xor(m, n))
    print(np.logical_not(m))
    print(np.all(m))
    print(np.any(np.greater(a, 100.0)))


if __name__ == "__main__":
    main()
