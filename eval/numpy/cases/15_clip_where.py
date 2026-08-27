"""clip and where."""
import numpy as np


def main() -> None:
    a = np.array([-2.0, 0.5, 3.0, 7.0])
    print(np.clip(a, 0.0, 5.0))
    print(np.clip(a, 0.0, 1.0))
    b = np.array([1, 5, 9])
    print(np.clip(b, 2, 6))
    cond = np.greater(a, 1.0)
    print(np.where(cond, a, np.zeros(4)))
    print(np.where(cond, np.ones(4), np.full(4, -1.0)))


if __name__ == "__main__":
    main()
