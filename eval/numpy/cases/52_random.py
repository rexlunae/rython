"""np.random — accepted at conversion; numpy needs a seed to be reproducible."""
import numpy as np


def main() -> None:
    np.random.seed(0)
    print(np.random.rand(3))


if __name__ == "__main__":
    main()
