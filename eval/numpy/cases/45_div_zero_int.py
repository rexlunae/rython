"""Integer division by zero: numpy warns and yields 0, it does not raise."""
import numpy as np


def main() -> None:
    a = np.array([1, 2, 3])
    z = np.array([0, 0, 0])
    print(np.floor_divide(a, z))
    print(np.mod(a, z))


if __name__ == "__main__":
    main()
