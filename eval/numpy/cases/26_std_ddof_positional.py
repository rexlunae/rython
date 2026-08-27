"""np.std/np.var second positional argument.

In numpy the second positional parameter of std/var is `axis`, not `ddof`.
"""
import numpy as np


def main() -> None:
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    print(np.std(a, 1))
    print(np.var(a, 1))


if __name__ == "__main__":
    main()
