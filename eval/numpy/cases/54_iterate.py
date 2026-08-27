"""Iterating over an array."""
import numpy as np


def main() -> None:
    for x in np.array([1.0, 2.0, 3.0]):
        print(x)
    for row in np.reshape(np.arange(4), (2, 2)):
        print(row)


if __name__ == "__main__":
    main()
