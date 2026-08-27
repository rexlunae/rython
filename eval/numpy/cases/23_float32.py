"""float32 arithmetic and printing."""
import numpy as np


def main() -> None:
    a = np.ones(4, dtype=np.float32)
    print(a)
    print(np.multiply(a, 0.1))
    print(np.divide(a, 3.0))
    print(np.sum(np.divide(a, 3.0)))
    b = np.zeros(3, dtype=np.float32)
    print(np.add(b, 1.5))


if __name__ == "__main__":
    main()
