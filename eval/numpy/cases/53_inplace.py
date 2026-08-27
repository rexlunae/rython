"""In-place operators on arrays."""
import numpy as np


def main() -> None:
    a = np.array([1.0, 2.0, 3.0])
    a += 1.0
    print(a)
    a *= 2.0
    print(a)


if __name__ == "__main__":
    main()
