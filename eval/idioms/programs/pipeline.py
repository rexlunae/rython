"""Closures, functions as values, map/filter/lambda, a dispatch table of
callables, *args and keyword arguments."""
from typing import Callable


def make_adder(n: int) -> Callable[[int], int]:
    def add(x: int) -> int:
        return x + n
    return add


def compose(f: Callable[[int], int], g: Callable[[int], int]) -> Callable[[int], int]:
    return lambda x: f(g(x))


def apply_all(fs: list[Callable[[int], int]], x: int) -> list[int]:
    return [f(x) for f in fs]


def total(*nums: int, scale: int = 1) -> int:
    return sum(nums) * scale


def main() -> None:
    add5 = make_adder(5)
    double = lambda x: x * 2
    both = compose(add5, double)
    print(add5(1), double(4), both(3), compose(double, add5)(3))
    ops: dict[str, Callable[[int], int]] = {"add5": add5, "double": double, "both": both}
    for name in sorted(ops):
        print(name, ops[name](10))
    nums = list(range(1, 11))
    evens = list(filter(lambda n: n % 2 == 0, nums))
    squares = list(map(lambda n: n * n, evens))
    print(evens, squares, sum(squares))
    print(apply_all([add5, double, both], 7))
    print(total(1, 2, 3), total(1, 2, 3, scale=10), total())
    counter = {"n": 0}

    def bump() -> int:
        counter["n"] += 1
        return counter["n"]

    bump(); bump()
    print(bump(), counter["n"])
    print(sorted(["bb", "a", "ccc"], key=len, reverse=True))


if __name__ == "__main__":
    main()
