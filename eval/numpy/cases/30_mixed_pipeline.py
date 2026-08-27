"""A longer pipeline mixing creation, ufuncs, reshape and reductions."""
import numpy as np


def main() -> None:
    a = np.reshape(np.linspace(0.0, 1.0, 12), (3, 4))
    b = np.add(np.multiply(a, 2.0), 1.0)
    print(b)
    print(np.transpose(b))
    print(np.sum(b))
    print(np.mean(b))
    print(np.max(b))
    print(np.min(b))
    c = np.ravel(b)
    print(np.sort(c))
    print(np.argsort(c))
    print(np.clip(c, 1.2, 2.5))
    print(np.where(np.greater(c, 2.0), c, np.zeros(12)))
    print(np.std(c))
    print(np.var(c))


if __name__ == "__main__":
    main()
