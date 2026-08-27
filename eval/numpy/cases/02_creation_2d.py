"""2-D and 3-D creation and printing."""
import numpy as np


def main() -> None:
    print(np.zeros((2, 3)))
    print(np.ones((3, 2)))
    print(np.array([[1.0, 2.0], [3.0, 4.0]]))
    print(np.array([[1, 2, 3], [4, 5, 6]]))
    print(np.full((2, 2), 7.0))
    print(np.reshape(np.arange(12), (2, 2, 3)))


if __name__ == "__main__":
    main()
