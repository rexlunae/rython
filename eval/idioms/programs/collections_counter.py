"""Counting with collections.Counter.

Issue #137's charset_normalizer frontier: `Counter(layer)` is how coherence
detection tallies a decoded sequence, and the round-100 sweep could not
convert that package at all. The printed total after each mutation makes
the state observable, so a lowering that silently changed a count (or
answered `dict.get`'s None where `__missing__` owes a 0) shows up here.
"""

from collections import Counter


def tally(words: list[str]) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter(words)
    return counts.most_common()


def top_only(words: list[str]) -> list[tuple[str, int]]:
    return Counter(words).most_common(1)


def main() -> None:
    c = Counter("aab")
    print(c.get("a"), c.get("z"), c.get("z", 99))
    print(c.most_common(), c.most_common(1))

    # A MAPPING's values are the counts, not one per key.
    d = Counter({"a": 3, "b": 1})
    print(d.get("a"), d.get("q"))

    e = Counter()
    print(e.get("x"), e.get("x", 5))

    # Subscripting is __missing__, so it answers 0 where get answers None.
    print(c["a"], c["z"])

    print(tally(["x", "y", "x"]))
    print(top_only(["x", "y", "x"]))


if __name__ == "__main__":
    main()
