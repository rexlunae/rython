"""Reductions over integer and bool arrays (result type matters)."""
import numpy as np


def main() -> None:
    a = np.array([1, 2, 3, 4])
    print(np.sum(a))
    print(np.prod(a))
    print(np.mean(a))
    print(np.max(a))
    print(np.min(a))
    print(np.argmax(a))
    print(np.argmin(a))
    b = np.arange(6)
    print(np.sum(b))
    print(np.max(b))
    m = np.array([True, False, True])
    print(np.sum(m))
    print(np.all(m))
    print(np.any(m))


if __name__ == "__main__":
    main()
