"""numpy prints array floats at precision=8, not full repr."""
import numpy as np


def main() -> None:
    print(np.divide(np.array([1.0]), np.array([3.0])))
    print(np.array([2.718281828459045]))
    print(np.array([0.1, 0.123456789012345]))


if __name__ == "__main__":
    main()
