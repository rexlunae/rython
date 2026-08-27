"""np.set_backend on a supported engine, then ordinary work."""
import numpy as np


def main() -> None:
    np.set_backend("scalar")
    a = np.linspace(0.0, 1.0, 9)
    print(np.sum(np.multiply(a, a)))
    print(np.add(a, 1.0))


if __name__ == "__main__":
    main()
