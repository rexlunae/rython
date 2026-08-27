"""The `+` operator on arrays, with the operands used again afterwards."""
import numpy as np


def main() -> None:
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    print(a + b)
    print(a)
    print(b)


if __name__ == "__main__":
    main()
