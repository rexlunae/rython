"""Unary ufuncs / math kernels."""
import numpy as np


def main() -> None:
    a = np.array([0.25, 1.0, 4.0, 9.0])
    print(np.sqrt(a))
    print(np.exp(np.array([0.0, 1.0, 2.0])))
    print(np.log(a))
    print(np.log2(np.array([1.0, 2.0, 8.0])))
    print(np.log10(np.array([1.0, 10.0, 100.0])))
    print(np.sin(np.array([0.0, 0.5, 1.0])))
    print(np.cos(np.array([0.0, 0.5, 1.0])))
    print(np.tan(np.array([0.0, 0.5, 1.0])))
    print(np.tanh(np.array([0.0, 0.5, 1.0])))
    print(np.floor(np.array([1.7, -1.7, 2.0])))
    print(np.ceil(np.array([1.2, -1.2, 2.0])))
    print(np.abs(np.array([-1.0, 2.0, -3.5])))
    print(np.negative(a))
    print(np.square(np.array([1.0, 2.0, 3.0])))
    print(np.sign(np.array([-2.0, 0.0, 3.0])))
    print(np.reciprocal(np.array([1.0, 2.0, 4.0])))
    print(np.expm1(np.array([0.0, 1e-8, 1.0])))
    print(np.log1p(np.array([0.0, 1e-8, 1.0])))


if __name__ == "__main__":
    main()
