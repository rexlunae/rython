"""Broadcasting a row across a matrix."""
import numpy as np


def main() -> None:
    m = np.reshape(np.linspace(0.0, 3.0, 4), (2, 2))
    r = np.array([10.0, 20.0])
    print(np.add(m, r))
    print(np.multiply(m, r))


if __name__ == "__main__":
    main()
