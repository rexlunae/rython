"""Out-of-bounds index: numpy raises IndexError."""
import numpy as np


def main() -> None:
    a = np.array([1.0, 2.0])
    print(a[5])


if __name__ == "__main__":
    main()
