"""Mixed int/float operands and the resulting dtype."""
import numpy as np


def main() -> None:
    i = np.array([1, 2, 3])
    f = np.array([0.5, 0.5, 0.5])
    print(np.add(i, f))
    print(np.multiply(i, 2.0))
    print(np.divide(i, 2))
    print(np.add(i, 1))
    print(np.subtract(f, 1))
    print(np.mean(i))
    print(np.sum(np.array([1, 2, 3])))


if __name__ == "__main__":
    main()
