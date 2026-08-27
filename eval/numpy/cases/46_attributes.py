"""ndarray attributes: shape, ndim, size, dtype, T."""
import numpy as np


def main() -> None:
    a = np.array([1.0, 2.0, 3.0])
    print(a.shape)
    print(a.ndim)
    print(a.size)
    print(a.dtype)
    m = np.reshape(np.arange(6), (2, 3))
    print(m.shape)
    print(m.ndim)
    print(m.size)
    print(m.T)


if __name__ == "__main__":
    main()
