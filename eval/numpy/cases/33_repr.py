"""repr() of arrays."""
import numpy as np


def main() -> None:
    print(repr(np.array([1.0, 2.0, 3.0])))
    print(repr(np.array([1, 2, 3])))
    print(repr(np.zeros((2, 2))))
    print(repr(np.ones(3, dtype=np.bool_)))


if __name__ == "__main__":
    main()
