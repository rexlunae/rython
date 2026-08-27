"""Inverting a singular matrix: numpy raises LinAlgError."""
import numpy as np


def main() -> None:
    m = np.array([[1.0, 2.0], [2.0, 4.0]])
    print(np.linalg.inv(m))


if __name__ == "__main__":
    main()
