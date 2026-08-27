"""sort / argsort, including ties and negatives."""
import numpy as np


def main() -> None:
    a = np.array([3.0, 1.0, 2.0, 1.0])
    print(np.sort(a))
    print(np.argsort(a))
    b = np.array([5, -2, 7, 0, -2])
    print(np.sort(b))
    print(np.argsort(b))
    print(np.sort(np.array([1.0])))
    print(np.argsort(np.array([1.0])))


if __name__ == "__main__":
    main()
