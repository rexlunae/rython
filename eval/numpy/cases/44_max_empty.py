"""Reducing an empty array with max: numpy raises ValueError."""
import numpy as np


def main() -> None:
    print(np.max(np.zeros(0)))


if __name__ == "__main__":
    main()
