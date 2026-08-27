"""Indexing and len()."""
import numpy as np


def main() -> None:
    a = np.array([10.0, 20.0, 30.0, 40.0])
    print(a[0])
    print(a[3])
    print(a[-1])
    print(len(a))
    m = np.reshape(np.arange(6), (2, 3))
    print(m[0])
    print(m[1])
    print(len(m))


if __name__ == "__main__":
    main()
