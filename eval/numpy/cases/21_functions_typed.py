"""np.ndarray annotations on user functions, arrays through call boundaries."""
import numpy as np


def scale(a: np.ndarray, k: float) -> np.ndarray:
    return np.multiply(a, k)


def norm2(a: np.ndarray) -> float:
    return np.sum(np.multiply(a, a))


def build(n: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n)


def main() -> None:
    v = build(5)
    print(v)
    print(scale(v, 3.0))
    print(norm2(v))
    print(norm2(scale(v, 2.0)))


if __name__ == "__main__":
    main()
