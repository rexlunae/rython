"""Slices with negative bounds and steps."""
import numpy as np


def main() -> None:
    a = np.arange(10)
    print(a[::2])
    print(a[1::2])
    print(a[-3:])
    print(a[:-3])
    print(a[-5:-2])
    print(a[::-1])


if __name__ == "__main__":
    main()
