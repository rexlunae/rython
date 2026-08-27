"""inf / nan production and the float predicates."""
import numpy as np


def main() -> None:
    a = np.array([1.0, 0.0, -1.0])
    d = np.divide(a, np.zeros(3))
    print(d)
    print(np.isnan(d))
    print(np.isinf(d))
    print(np.isfinite(d))
    print(np.sqrt(np.array([-1.0, 4.0])))
    print(np.log(np.array([0.0, 1.0])))
    print(np.isnan(np.array([1.0, 2.0])))


if __name__ == "__main__":
    main()
