"""shape / ndim / size attributes (these compile — do they agree?)."""
import numpy as np


def main() -> None:
    a = np.array([1.0, 2.0, 3.0])
    print(a.shape)
    print(a.ndim)
    print(a.size)
    m = np.reshape(np.arange(6), (2, 3))
    print(m.shape)
    print(m.ndim)
    print(m.size)


if __name__ == "__main__":
    main()
