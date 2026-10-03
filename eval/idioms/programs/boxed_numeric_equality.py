"""The numeric tower (1 == 1.0 == True == 1+0j) inside boxed containers."""

from collections import OrderedDict
from typing import Any


def same(a: Any, b: Any) -> bool:
    return a == b


def differ(a: Any, b: Any) -> bool:
    return a != b


def has(items: Any, x: Any) -> bool:
    return x in items


def as_ordered(value: Any) -> Any:
    return OrderedDict(value)


def main() -> None:
    a = {"n": 1, "s": "x"}
    b = {"n": 1.0, "s": "x"}
    print(a == b)
    print(a != b)

    print(same((1, "x"), (1.0, "x")))
    print(same({"n": 1, "s": "x"}, {"n": True, "s": "x"}))
    print(differ({"n": 1, "s": "x"}, {"n": 2.0, "s": "x"}))
    print(differ({"n": 1, "s": "x"}, {"n": 1.5, "s": "x"}))
    print(same((1, (2, "a")), (1.0, (2.0, "a"))))
    print(same([1, "x", 2.5], [1.0, "x", 2.5]))
    print(same({"k": (1, 2), "t": "v"}, {"k": (1.0, 2.0), "t": "v"}))
    print(same({"k": {"j": 1}, "t": "v"}, {"k": {"j": True}, "t": "v"}))

    print(same(as_ordered({"n": 1, "s": "x"}), as_ordered({"n": 1.0, "s": "x"})))
    print(same(as_ordered({"n": 1, "s": "x"}), {"s": "x", "n": 1.0}))
    print(same(as_ordered({"n": 1, "s": "x"}), as_ordered({"s": "x", "n": 1})))

    items = [1, "x", 2.0, True]
    print(has(items, 1.0))
    print(has(items, 2))
    print(has(items, 3))
    print(has((1, "x"), 1.0))
    print(has([(2,), "y"], (2.0,)))
    print(has(range(5), 2.0))


if __name__ == "__main__":
    main()
