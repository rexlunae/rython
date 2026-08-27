"""Arrays whose values need numpy's exponential format."""
import numpy as np


def main() -> None:
    print(np.array([1e-9, 1.0]))


if __name__ == "__main__":
    main()
