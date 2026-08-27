"""ndarray methods: sum, mean, max, reshape, astype, copy, ravel."""
import numpy as np


def main() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    print(a.sum())
    print(a.mean())
    print(a.max())
    print(a.min())
    print(a.reshape((2, 2)))
    print(a.ravel())
    print(a.copy())
    print(a.astype(np.int64))


if __name__ == "__main__":
    main()
