"""dot / matmul / vdot and np.linalg."""
import numpy as np


def main() -> None:
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    print(np.dot(a, b))
    print(np.vdot(a, b))
    m = np.array([[1.0, 2.0], [3.0, 4.0]])
    n = np.array([[5.0, 6.0], [7.0, 8.0]])
    print(np.matmul(m, n))
    print(np.dot(m, n))
    print(np.matmul(m, np.array([1.0, 1.0])))
    print(np.linalg.det(m))
    print(np.linalg.inv(m))
    print(np.linalg.solve(m, np.array([5.0, 11.0])))
    i3 = np.eye(3)
    print(np.linalg.det(i3))
    print(np.linalg.inv(i3))


if __name__ == "__main__":
    main()
