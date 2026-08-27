"""Array creation: array, zeros, ones, full, arange, linspace, eye, identity."""
import numpy as np


def main() -> None:
    print(np.array([1.0, 2.0, 3.0]))
    print(np.array([1, 2, 3]))
    print(np.zeros(4))
    print(np.ones(3))
    print(np.full(3, 2.5))
    print(np.arange(5))
    print(np.arange(2, 10, 3))
    print(np.linspace(0.0, 1.0, 5))
    print(np.eye(3))
    print(np.identity(2))


if __name__ == "__main__":
    main()
