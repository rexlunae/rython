"""int and bool array formatting (column alignment)."""
import numpy as np


def main() -> None:
    print(np.array([1, 22, 333]))
    print(np.array([-1, 2, -30]))
    print(np.arange(10))
    print(np.array([[1, 200], [30, 4]]))
    print(np.ones(3, dtype=np.bool_))
    print(np.zeros(3, dtype=np.bool_))
    print(np.greater(np.array([1.0, 5.0, 3.0]), 2.0))


if __name__ == "__main__":
    main()
