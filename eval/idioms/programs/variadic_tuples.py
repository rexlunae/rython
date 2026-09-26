"""Variable-length tuples (issue #399): an interval table keyed by id,
a `*args` collector, and `tuple(xs)` snapshots — each printed, so a tuple
that printed as a list would show."""
from typing import Dict, List, Tuple


def snapshot(xs: List[int]) -> Tuple[int, ...]:
    return tuple(xs)


def span(lo: int, hi: int) -> Tuple[int, ...]:
    return (lo, hi)


def collect(*parts: int) -> Tuple[int, ...]:
    print("collected", parts)
    return parts


def main() -> None:
    table = {1: (0, 10), 2: (20,), 3: (30, 31, 32)}
    total = 0
    for key in sorted(table):
        entry = table[key]
        total += sum(entry)
        print(key, entry, len(entry))
    print("total", total)
    history: List[Tuple[int, ...]] = []
    xs: List[int] = []
    for i in range(3):
        xs.append(i * i)
        history.append(snapshot(xs))
    print(history)
    print(span(4, 8), collect(1, 2, 3), collect())
    names: Dict[str, Tuple[str, ...]] = {"ok": ("200", "204"), "moved": ("301",)}
    print(names["ok"], names["moved"])


if __name__ == "__main__":
    main()
