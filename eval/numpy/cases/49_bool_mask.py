"""Boolean-mask indexing."""
import numpy as np


def main() -> None:
    a = np.array([1.0, 5.0, 2.0, 8.0])
    m = np.greater(a, 2.0)
    print(a[m])


if __name__ == "__main__":
    main()
