"""np.array from mixed and bool literal lists."""
import numpy as np


def main() -> None:
    print(np.array([1, 2.0]))
    print(np.array([True, False]))
    print(np.array([[1.0], [2.0]]))
    print(np.arange(0.0, 1.0, 0.25))
    print(np.pi)


if __name__ == "__main__":
    main()
