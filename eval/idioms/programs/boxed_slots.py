"""Methods whose parameters carry no annotations: ranges, lists and empty
lists passed in, handed on to typed helpers, iterated (issue #335)."""

from typing import List, Tuple


def total(xs: List[int]) -> int:
    return sum(xs)


def has(x: int, xs: Tuple[int, ...]) -> bool:
    return x in xs


class Checker:
    def check(self, ints, others):
        t = total(ints)
        for i in ints:
            assert has(i, tuple(ints))
        for o in others:
            assert not has(o, tuple(ints))
        return t

    def describe(self, seq):
        return str(seq) + " len=" + str(len(seq))


def main() -> None:
    c = Checker()
    print(c.check(range(3, 6), [1, 9]))
    print(c.check([7], [8]))
    print(c.check([], range(2)))
    print(c.describe(range(0, 10, 2)))
    print(c.describe(range(4)))


if __name__ == "__main__":
    main()
