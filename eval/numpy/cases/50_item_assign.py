"""Assigning into an array element."""
import numpy as np


def main() -> None:
    a = np.zeros(4)
    a[0] = 5.0
    a[2] = -1.5
    print(a)


if __name__ == "__main__":
    main()
