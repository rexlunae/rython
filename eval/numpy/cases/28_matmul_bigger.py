"""A larger matmul and a solve, to compare numerics."""
import numpy as np


def main() -> None:
    n = 6
    a = np.reshape(np.linspace(1.0, 36.0, 36), (n, n))
    b = np.transpose(a)
    print(np.matmul(a, b))
    m = np.add(np.eye(4), np.reshape(np.linspace(0.1, 1.6, 16), (4, 4)))
    print(np.linalg.det(m))
    print(np.linalg.inv(m))
    print(np.linalg.solve(m, np.ones(4)))


if __name__ == "__main__":
    main()
