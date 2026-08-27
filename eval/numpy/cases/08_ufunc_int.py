"""Integer-dtype elementwise ops: floor division, mod, negative operands."""
import numpy as np


def main() -> None:
    a = np.array([7, -7, 8, -8])
    b = np.array([3, 3, -3, -3])
    print(np.add(a, b))
    print(np.subtract(a, b))
    print(np.multiply(a, b))
    print(np.floor_divide(a, b))
    print(np.mod(a, b))
    print(np.power(np.array([2, 3, 4]), 3))
    print(np.negative(a))
    print(np.abs(a))
    print(np.maximum(a, b))
    print(np.minimum(a, b))


if __name__ == "__main__":
    main()
