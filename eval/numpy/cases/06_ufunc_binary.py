"""Binary ufuncs, array-array and array-scalar."""
import numpy as np


def main() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([10.0, 20.0, 30.0, 40.0])
    print(np.add(a, b))
    print(np.subtract(b, a))
    print(np.multiply(a, b))
    print(np.divide(b, a))
    print(np.power(a, 2.0))
    print(np.maximum(a, 2.5))
    print(np.minimum(a, 2.5))
    print(np.add(a, 1.0))
    print(np.multiply(a, -1.0))
    print(np.subtract(10.0, a))
    print(np.divide(1.0, a))


if __name__ == "__main__":
    main()
