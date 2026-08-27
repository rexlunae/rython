"""Bool arrays: creation, logical ops, reductions, printing."""
import numpy as np


def main() -> None:
    t = np.ones(3, dtype=np.bool_)
    f = np.zeros(3, dtype=np.bool_)
    print(np.bitwise_and(t, f))
    print(np.bitwise_or(t, f))
    print(np.bitwise_xor(t, f))
    print(np.logical_not(t))
    print(np.all(t))
    print(np.all(f))
    print(np.any(t))
    print(np.any(f))
    print(np.sum(t))


if __name__ == "__main__":
    main()
