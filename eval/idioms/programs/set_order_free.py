"""Sets consumed only in ways whose result cannot depend on iteration
order (#444): sorted copies, membership, len, min/max, sum of ints,
any/all, read-only set comprehensions, set algebra."""


def tally(words: list[str]) -> set[str]:
    seen: set[str] = set()
    for w in words:
        seen.add(w.lower())
    return seen


def main() -> None:
    seen = tally(["Pear", "apple", "fig", "APPLE", "kiwi", "pear"])
    print(len(seen), sorted(seen))
    print(min(seen), max(seen), "fig" in seen, "plum" in seen)
    lengths = {len(w) for w in seen}
    print(sorted(lengths), sum(lengths))
    print(any(w.startswith("k") for w in seen), all(len(w) > 2 for w in seen))
    initials = sorted(list({w[0] for w in seen if w}))
    print(initials)
    both = seen & {"fig", "plum", "kiwi"}
    print(sorted(both), len(seen | {"plum"}), sorted(seen - both))
    for w in sorted(seen):
        print(w.upper())


if __name__ == "__main__":
    main()
