"""Python arithmetic operators on arrays (a + b, a * 2, ...)."""
import numpy as np


def main() -> None:
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([10.0, 20.0, 30.0])
    print(a + b)
    print(b - a)
    print(a * b)
    print(b / a)
    print(a * 2.0)
    print(a + 1.0)
    print(-a)


if __name__ == "__main__":
    main()
