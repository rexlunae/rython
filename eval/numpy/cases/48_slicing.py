"""Slicing a 1-D array."""
import numpy as np


def main() -> None:
    a = np.arange(10)
    print(a[0:3])
    print(a[2:])
    print(a[:4])
    print(a[::2])
    print(a[-3:])


if __name__ == "__main__":
    main()
