"""Grouping and counting with collections.defaultdict."""

from collections import defaultdict


def count_words(words: list[str]) -> dict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
    for w in words:
        counts[w] += 1
    return dict(counts)


def main() -> None:
    dd = defaultdict(int)
    dd["a"] += 2
    print(dd["a"])
    print(len(dd))

    # A READ of a missing key inserts the default.
    print(dd["missing"])
    print(len(dd))
    print(list(dd))
    print("missing" in dd, "other" in dd)
    print(dd.get("other"))
    print(len(dd))

    groups = defaultdict(list)
    for word in ["apple", "avocado", "banana", "blueberry", "cherry"]:
        groups[word[0]].append(word)
    for letter in sorted(groups):
        print(letter, groups[letter])
    print(list(groups.keys()))
    print(list(groups.values()))
    for k, v in groups.items():
        print(k, len(v))

    print(sorted(count_words(["x", "y", "x", "z", "x"]).items()))
    print(dd)

    total = 0
    for v in dd.values():
        total += v
    print(total)


if __name__ == "__main__":
    main()
