"""Empty and single-element arrays."""
import numpy as np


def main() -> None:
    e = np.zeros(0)
    print(e)
    print(len(e))
    print(np.sum(e))
    one = np.array([42.0])
    print(one)
    print(np.sum(one))
    print(np.mean(one))
    print(np.std(one))
    print(np.arange(0))
    print(np.linspace(1.0, 1.0, 1))


if __name__ == "__main__":
    main()
