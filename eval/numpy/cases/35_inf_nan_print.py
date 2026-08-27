"""inf / nan cells in a float array."""
import numpy as np


def main() -> None:
    print(np.divide(np.array([1.0]), np.zeros(1)))
    print(np.divide(np.array([0.0]), np.zeros(1)))
    print(np.divide(np.array([-1.0]), np.zeros(1)))


if __name__ == "__main__":
    main()
