"""Broadcast/shape error: numpy raises ValueError."""
import numpy as np


def main() -> None:
    a = np.array([1.0, 2.0])
    b = np.array([1.0, 2.0, 3.0])
    print(np.add(a, b))


if __name__ == "__main__":
    main()
