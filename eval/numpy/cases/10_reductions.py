"""Reductions over float arrays."""
import numpy as np


def main() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    print(np.sum(a))
    print(np.prod(a))
    print(np.mean(a))
    print(np.max(a))
    print(np.min(a))
    print(np.std(a))
    print(np.var(a))
    print(np.std(a, ddof=1))
    print(np.var(a, ddof=1))
    print(np.argmax(a))
    print(np.argmin(a))
    b = np.array([[1.0, 2.0], [3.0, 4.0]])
    print(np.sum(b))
    print(np.mean(b))


if __name__ == "__main__":
    main()
