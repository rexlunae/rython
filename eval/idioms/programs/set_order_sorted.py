"""Sets read only through order-free consumers that build today (#444):
a sorted copy, membership, len, min/max, any/all, and a read-only
comprehension handed straight to sorted."""


def tally(words: list[str]) -> set[str]:
    seen: set[str] = set()
    for w in words:
        seen.add(w.lower())
    return seen


def main() -> None:
    seen = tally(["Pear", "apple", "fig", "APPLE", "kiwi", "pear"])
    print(len(seen), sorted(seen))
    print(min(seen), max(seen), "fig" in seen, "plum" in seen)
    print(any(w.startswith("k") for w in seen), all(len(w) > 2 for w in seen))
    print(sorted(w.upper() for w in seen))
    for w in sorted(seen):
        print(w, len(w))


if __name__ == "__main__":
    main()
