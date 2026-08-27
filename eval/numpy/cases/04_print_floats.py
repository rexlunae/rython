"""numpy's float formatting: widths, exponents, negatives, specials."""
import numpy as np


def main() -> None:
    print(np.array([1.5, 22.25, 333.125]))
    print(np.array([-1.0, 2.0, -3.5]))
    print(np.array([1e-9, 1.0, 1e9]))
    print(np.array([0.1, 0.2, 0.30000000000000004]))
    print(np.array([1e16, 2e16]))
    print(np.array([0.0, -0.0]))
    print(np.array([1.0, 2.0, 3.0]))
    print(np.array([100000.0, 2.0]))
    print(np.array([1.0e-5, 1.0]))
    print(np.linspace(0.0, 1.0, 11))


if __name__ == "__main__":
    main()
