"""A user function annotated to return np.ndarray."""
import numpy as np


def double(a: np.ndarray) -> np.ndarray:
    return np.multiply(a, 2.0)


def main() -> None:
    print(double(np.array([1.0, 2.0])))


if __name__ == "__main__":
    main()
