"""A large reduction — pairwise vs sequential summation shows up here."""
import numpy as np


def main() -> None:
    a = np.linspace(0.0, 1.0, 100001)
    print(np.sum(a))
    print(np.mean(a))
    b = np.full(1000000, 0.1)
    print(np.sum(b))
    print(np.mean(b))
    c = np.arange(1000000)
    print(np.sum(c))


if __name__ == "__main__":
    main()
