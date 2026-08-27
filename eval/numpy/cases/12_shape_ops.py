"""reshape / ravel / transpose / concatenate / vstack / hstack."""
import numpy as np


def main() -> None:
    a = np.arange(6)
    print(np.reshape(a, (2, 3)))
    print(np.reshape(a, (3, 2)))
    print(np.ravel(np.reshape(a, (2, 3))))
    print(np.transpose(np.reshape(a, (2, 3))))
    x = np.array([1.0, 2.0])
    y = np.array([3.0, 4.0])
    print(np.concatenate([x, y], 0))
    print(np.vstack([x, y]))
    print(np.hstack([x, y]))
    p = np.reshape(np.arange(4), (2, 2))
    q = np.reshape(np.arange(4, 8), (2, 2))
    print(np.concatenate([p, q], 0))
    print(np.concatenate([p, q], 1))
    print(np.vstack([p, q]))
    print(np.hstack([p, q]))


if __name__ == "__main__":
    main()
