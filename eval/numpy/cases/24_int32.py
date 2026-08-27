"""int32 arithmetic and printing."""
import numpy as np


def main() -> None:
    a = np.ones(4, dtype=np.int32)
    print(a)
    print(np.multiply(a, 3))
    print(np.add(a, a))
    print(np.sum(a))
    b = np.zeros((2, 2), dtype=np.int32)
    print(np.add(b, 5))


if __name__ == "__main__":
    main()
