"""Arrays inside loops — value semantics and repeated reuse."""
import numpy as np


def main() -> None:
    acc = np.zeros(4)
    i = 0
    while i < 5:
        acc = np.add(acc, np.full(4, 1.5))
        i = i + 1
    print(acc)

    x = np.arange(4)
    for k in range(3):
        x = np.add(x, 1)
    print(x)

    a = np.array([1.0, 2.0, 3.0])
    s = np.sum(a)
    m = np.mean(a)
    print(s)
    print(m)
    print(np.sum(a))


if __name__ == "__main__":
    main()
